"""
NBA Rookies Dashboard: análisis visual de rookies de la NBA y predictor de Hall of Fame.

Ejecutar en local:
    python app.py            -> http://127.0.0.1:8050
Producción (Render, Railway...):
    gunicorn app:server
"""
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# 1. DATOS
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "nba_rookies.csv"

data = pd.read_csv(DATA_PATH)
data = data.dropna(subset=['Name'])

data['3P%'] = data['3P%'].replace('-', '0')
data['3P%'] = data['3P%'].astype(float)
data['Year Drafted'] = data['Year Drafted'].astype(int)

data['Is_HOF'] = data['Hall of Fame Class'].notna().astype(int)
data['Hall of Fame Class'] = data['Hall of Fame Class'].fillna('No')

# ---------------------------------------------------------------------------
# 2. MODELO DE PREDICCIÓN
# ---------------------------------------------------------------------------
# Solo entrenamos con jugadores drafteados hasta 1992: los más recientes pueden
# no haber tenido tiempo de entrar en el Hall of Fame y serían "0 falsos".
data_entrenamiento = data[data['Year Drafted'] <= 1992]

columnas_modelo = ['MIN', 'PTS', 'REB', 'AST']
X = data_entrenamiento[columnas_modelo]
y = data_entrenamiento['Is_HOF']

# class_weight='balanced' compensa que solo ~4% de los jugadores sean HOF
modelo_lr = LogisticRegression(class_weight='balanced', random_state=42)
modelo_lr.fit(X, y)

# ---------------------------------------------------------------------------
# 3. APP
# ---------------------------------------------------------------------------
fuente_google = "https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap"
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY, fuente_google],
           title="NBA Rookies Dashboard")
server = app.server  # para gunicorn

nombres_jugadores = data['Name'].unique().tolist()

# --- DISEÑO DEL TÍTULO ---
custom_header = html.Div([
    html.Div([
        html.H1("Análisis de Rookies NBA",
                style={'fontFamily': 'Montserrat, sans-serif', 'fontWeight': '800', 'color': '#2c3e50'},
                className="mb-2"),
        html.Img(
            src="https://upload.wikimedia.org/wikipedia/en/0/03/National_Basketball_Association_logo.svg",
            style={'height': '60px', 'marginLeft': '15px'}
        )
    ], style={'display': 'flex', 'alignItems': 'center'}),

    html.P("Explorando el rendimiento de los novatos y su camino al Hall of Fame",
           style={'fontFamily': 'Montserrat, sans-serif', 'fontWeight': '400', 'color': '#5a6b7c', 'fontSize': '1.2rem'},
           className="mb-0"),
], style={
    'backgroundColor': '#eef2f5',
    'padding': '35px',
    'borderRadius': '12px',
    'marginBottom': '30px',
    'boxShadow': '0 4px 10px rgba(0,0,0,0.05)'
})

# --- PESTAÑA 1: ANÁLISIS VISUAL ---
tab1_content = dbc.Card(dbc.CardBody([

    dbc.Row([
        # --- GRÁFICO 1: EVOLUCIÓN DE TIPOS DE TIRO ---
        dbc.Col([
            html.H5("Evolución de cada Tipo de Tiro", style={'fontFamily': 'Montserrat', 'color': '#2c3e50'},
                    className="fw-bold text-center mb-3"),
            dcc.Dropdown(
                id='dropdown-tiros',
                options=[
                    {'label': 'Triples Intentados (3PA)', 'value': '3PA'},
                    {'label': 'Tiros de Campo Totales (FGA)', 'value': 'FGA'},
                    {'label': 'Tiros Libres Intentados (FTA)', 'value': 'FTA'}
                ],
                value='3PA',
                clearable=False,
                style={'fontFamily': 'Montserrat', 'marginBottom': '10px'}
            ),
            dcc.Graph(id='grafico-evolucion-tiros', style={'height': '300px'})
        ], md=6),

        # --- GRÁFICO 2: HALL OF FAME ---
        dbc.Col([
            html.H5("¿Qué hace a un Hall of Fame?", style={'fontFamily': 'Montserrat', 'color': '#2c3e50'},
                    className="fw-bold text-center mb-3"),
            dcc.Dropdown(
                id='dropdown-hof',
                options=[
                    {'label': 'Minutos Jugados (MIN)', 'value': 'MIN'},
                    {'label': 'Puntos (PTS)', 'value': 'PTS'},
                    {'label': 'Rebotes (REB)', 'value': 'REB'},
                    {'label': 'Asistencias (AST)', 'value': 'AST'},
                    {'label': 'Robos (STL)', 'value': 'STL'},
                    {'label': 'Tapones (BLK)', 'value': 'BLK'},
                    {'label': 'Pérdidas (TOV)', 'value': 'TOV'},
                    {'label': 'Eficiencia (EFF)', 'value': 'EFF'},
                    {'label': 'Tiros de Campo Int. (FGA)', 'value': 'FGA'},
                    {'label': 'Tiros Libres Int. (FTA)', 'value': 'FTA'},
                    {'label': 'Triples Intentados (3PA)', 'value': '3PA'},
                ],
                value='PTS',
                clearable=False,
                style={'fontFamily': 'Montserrat', 'marginBottom': '10px'}
            ),
            dcc.Graph(id='grafico-hof', style={'height': '300px'})
        ], md=6),
    ], className="mb-4"),

    # --- GRÁFICO 3: COMPARADOR ---
    dbc.Row([
        dbc.Col([
            html.H5("Comparador", style={'fontFamily': 'Montserrat', 'color': '#2c3e50'}, className="fw-bold mb-4"),
            html.P("Selecciona dos jugadores para solapar sus perfiles estadísticos y compararlos directamente.",
                   style={'color': '#7f8c8d'}),

            html.Label("🔵 Jugador 1:", style={'fontWeight': 'bold', 'color': '#3498db'}),
            dcc.Dropdown(
                id='dropdown-jugador1',
                options=[{'label': nombre, 'value': nombre} for nombre in nombres_jugadores],
                value="Mark Jackson",
                clearable=False,
                style={'marginBottom': '20px'}
            ),

            html.Label("🟠 Jugador 2:", style={'fontWeight': 'bold', 'color': '#e67e22'}),
            dcc.Dropdown(
                id='dropdown-jugador2',
                options=[{'label': nombre, 'value': nombre} for nombre in nombres_jugadores],
                value="Manute Bol",
                clearable=False
            ),
        ], md=4, style={'backgroundColor': '#f8f9fa', 'padding': '25px', 'borderRadius': '10px'}),

        dbc.Col([
            dcc.Graph(id='grafico-radar', style={'height': '400px'})
        ], md=8)
    ], className="align-items-center")
]), className="mt-3 border-0 shadow-sm")

# --- PESTAÑA 2: MODELO DE PREDICCIÓN ---
tab2_content = dbc.Card(dbc.CardBody([

    # --- GRÁFICO 4: BUSCADOR ---
    html.H4("Buscador de Estrellas", style={'fontFamily': 'Montserrat', 'color': '#2c3e50'},
            className="fw-bold text-center mb-2"),
    html.P("Cruza dos estadísticas para ver dónde se agrupan las leyendas. "
           "Pasa el ratón por los puntos dorados para ver quiénes son.",
           className="text-center mb-4", style={'color': '#5a6b7c'}),

    dbc.Row([
        dbc.Col([
            html.Label("Eje Horizontal (X):", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            dcc.Dropdown(id='scatter-x', options=[{'label': c, 'value': c} for c in ['MIN', 'PTS', 'REB', 'AST', 'TOV', 'EFF']],
                         value='MIN', clearable=False)
        ], md=6),
        dbc.Col([
            html.Label("Eje Vertical (Y):", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            dcc.Dropdown(id='scatter-y', options=[{'label': c, 'value': c} for c in ['MIN', 'PTS', 'REB', 'AST', 'TOV', 'EFF']],
                         value='PTS', clearable=False)
        ], md=6),
    ], className="mb-3"),

    dcc.Graph(id='grafico-scatter', style={'height': '400px'}),

    html.Hr(style={'borderColor': '#eef2f5', 'borderWidth': '2px', 'margin': '40px 0'}),

    # --- PREDICTOR ---
    html.H4("Predictor de Hall of Fame", style={'fontFamily': 'Montserrat', 'color': '#2c3e50'},
            className="fw-bold text-center mb-3"),
    html.P("Introduce las estadísticas proyectadas para la temporada de rookie del jugador. "
           "El modelo de Regresión logística calculará su probabilidad de entrar al Hall of Fame "
           "basado en datos históricos.",
           className="text-center mb-5", style={'fontFamily': 'Montserrat', 'color': '#5a6b7c'}),

    dbc.Row([
        dbc.Col([html.Label("Minutos (MIN):", style={'fontFamily': 'Montserrat', 'fontWeight': '600'}),
                 dbc.Input(id="input-min", type="number", value=15.0, step=0.1)], md=2),
        dbc.Col([html.Label("Puntos (PTS):", style={'fontFamily': 'Montserrat', 'fontWeight': '600'}),
                 dbc.Input(id="input-pts", type="number", value=8.0, step=0.1)], md=2),
        dbc.Col([html.Label("Rebotes (REB):", style={'fontFamily': 'Montserrat', 'fontWeight': '600'}),
                 dbc.Input(id="input-reb", type="number", value=3.0, step=0.1)], md=2),
        dbc.Col([html.Label("Asist. (AST):", style={'fontFamily': 'Montserrat', 'fontWeight': '600'}),
                 dbc.Input(id="input-ast", type="number", value=2.0, step=0.1)], md=2),
        dbc.Col([html.Br(),
                 dbc.Button("Predecir Futuro", id="btn-predecir", color="primary", className="w-100",
                            style={'fontFamily': 'Montserrat', 'fontWeight': '600'})], md=2),
    ], className="mb-4"),

    html.Div(id="resultado-prediccion", className="mt-4")

]), className="mt-3 border-0 shadow-sm", style={'padding': '30px'})

# --- LAYOUT ---
app.layout = dbc.Container([
    custom_header,
    dbc.Tabs([
        dbc.Tab(tab1_content, label="Análisis Visual", tab_id="tab-1",
                label_style={'fontFamily': 'Montserrat', 'fontWeight': '600'}),
        dbc.Tab(tab2_content, label="Modelo de Predicción", tab_id="tab-2",
                label_style={'fontFamily': 'Montserrat', 'fontWeight': '600'}),
    ], id="tabs", active_tab="tab-1")
], fluid=True, className="p-4")


# ---------------------------------------------------------------------------
# 4. CALLBACKS
# ---------------------------------------------------------------------------
@app.callback(
    Output('grafico-evolucion-tiros', 'figure'),
    Input('dropdown-tiros', 'value')
)
def actualizar_grafico_tiros(metrica_seleccionada):
    df_agrupado = data.groupby('Year Drafted')[metrica_seleccionada].mean().reset_index()

    fig = px.line(
        df_agrupado,
        x='Year Drafted',
        y=metrica_seleccionada,
        markers=True,
        labels={'Year Drafted': 'Año del Draft', metrica_seleccionada: 'Media por Partido'}
    )
    fig.update_traces(
        line_color='#2c3e50',
        line_width=3,
        marker=dict(color='#3498db', size=8, line=dict(color='white', width=1.5))
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family="Montserrat",
        xaxis=dict(showgrid=True, gridcolor='#eef2f5'),
        yaxis=dict(showgrid=True, gridcolor='#eef2f5')
    )
    return fig


@app.callback(
    Output('grafico-hof', 'figure'),
    Input('dropdown-hof', 'value')
)
def actualizar_grafico_hof(metrica):
    data_plot = data.copy()
    data_plot['Leyenda'] = data_plot['Is_HOF'].map({1: 'Salón de la Fama', 0: 'Resto de Jugadores'})

    fig = px.box(
        data_plot,
        x='Leyenda',
        y=metrica,
        color='Leyenda',
        color_discrete_map={'Salón de la Fama': '#2c3e50', 'Resto de Jugadores': '#aab7c4'},
        points="outliers"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_family="Montserrat", xaxis=dict(title='', showgrid=False),
        yaxis=dict(title='Media por Partido', showgrid=True, gridcolor='#eef2f5'),
        showlegend=False
    )
    return fig


@app.callback(
    Output('grafico-radar', 'figure'),
    [Input('dropdown-jugador1', 'value'),
     Input('dropdown-jugador2', 'value')]
)
def actualizar_radar(jugador1, jugador2):
    categorias = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    fig = go.Figure()

    # Si hay nombres duplicados cogemos la primera aparición
    datos_j1 = data[data['Name'] == jugador1].iloc[0]
    datos_j2 = data[data['Name'] == jugador2].iloc[0]

    # Repetimos el primer valor al final para cerrar el polígono
    valores_j1 = [datos_j1[cat] for cat in categorias] + [datos_j1[categorias[0]]]
    valores_j2 = [datos_j2[cat] for cat in categorias] + [datos_j2[categorias[0]]]
    categorias_cerradas = categorias + [categorias[0]]

    fig.add_trace(go.Scatterpolar(
        r=valores_j1, theta=categorias_cerradas, fill='toself', name=jugador1,
        line_color='#3498db', fillcolor='rgba(52, 152, 219, 0.4)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=valores_j2, theta=categorias_cerradas, fill='toself', name=jugador2,
        line_color='#e67e22', fillcolor='rgba(230, 126, 34, 0.4)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, showline=False, gridcolor='#eef2f5')),
        showlegend=True,
        margin=dict(l=40, r=40, t=20, b=20),
        font_family="Montserrat",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


@app.callback(
    Output('grafico-scatter', 'figure'),
    [Input('scatter-x', 'value'), Input('scatter-y', 'value')]
)
def actualizar_scatter(eje_x, eje_y):
    data_plot = data.copy()
    data_plot['Leyenda'] = data_plot['Is_HOF'].map({1: 'Salón de la Fama', 0: 'Resto de Jugadores'})
    # Los HOF se dibujan los últimos para que queden por encima
    data_plot = data_plot.sort_values(by='Is_HOF')

    fig = px.scatter(
        data_plot, x=eje_x, y=eje_y, color='Leyenda',
        hover_name='Name',
        color_discrete_map={'Salón de la Fama': '#f1c40f', 'Resto de Jugadores': '#aab7c4'},
        opacity=0.8
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=1, color='white')))
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_family="Montserrat",
        xaxis=dict(showgrid=True, gridcolor='#eef2f5'), yaxis=dict(showgrid=True, gridcolor='#eef2f5'),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


@app.callback(
    Output('resultado-prediccion', 'children'),
    Input('btn-predecir', 'n_clicks'),
    [State('input-min', 'value'),
     State('input-pts', 'value'),
     State('input-reb', 'value'),
     State('input-ast', 'value')]
)
def realizar_prediccion(n_clicks, min_val, pts_val, reb_val, ast_val):
    if n_clicks is None:
        return html.Div()

    if None in [min_val, pts_val, reb_val, ast_val]:
        return html.Div("Por favor, rellena todos los campos.", className="alert alert-warning text-center")

    datos_jugador = pd.DataFrame([[min_val, pts_val, reb_val, ast_val]], columns=columnas_modelo)
    prob_hof = modelo_lr.predict_proba(datos_jugador)[0][1] * 100

    if prob_hof >= 65:
        estilo = {'backgroundColor': '#fff9e6', 'color': '#856404', 'border': '2px solid #ffeeba',
                  'padding': '20px', 'borderRadius': '10px'}
        mensaje = "¡Perfil de HOF!"
    else:
        estilo = {'backgroundColor': '#eef2f5', 'color': '#383d41', 'border': '2px solid #d6d8db',
                  'padding': '20px', 'borderRadius': '10px'}
        mensaje = "Jugador de rotación/titular sólido"

    return html.Div([
        html.H3(mensaje, style={'fontFamily': 'Montserrat', 'fontWeight': '800'}, className="text-center mb-2"),
        html.H4(f"Probabilidad de Hall of Fame: {prob_hof:.1f}%", style={'fontFamily': 'Montserrat'},
                className="text-center")
    ], style=estilo)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)
