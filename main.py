# Agrega al inicio del archivo
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configura CORS si es necesario
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
import pandas as pd
import json
from bokeh.embed import json_item
from bokeh.plotting import figure
from bokeh.transform import cumsum
import numpy as np
from plotnine import ggplot, aes, geom_bar, coord_flip, theme_minimal, labs
import plotly.express as px
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = FastAPI()

# Configuración de MongoDB
MONGO_URI = "mongodb+srv://ojjkd27:wsF5W5p6kq0enBAs@cluster0.dd37x83.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["iotdb"]
users_col = db["users"]
sensordatas_col = db["sensordatas"]
entornos_col = db["entornos"]

# Configuración de templates y archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Función para convertir gráficas matplotlib a base64
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 1. Cantidad de sensores por tipo (global y por usuario) - Seaborn
@app.get("/plot1")
async def get_plot1():
    # Obtener y expandir sensores
    sensores_expandido = pd.DataFrame([
        {"usuario": s["usuario"], "tipoSensor": sensor["tipoSensor"]}
        for s in sensordatas_col.find()
        for sensor in s["sensores"]
    ])

    # Conteo global
    conteo_global = sensores_expandido["tipoSensor"].value_counts().reset_index()
    conteo_global.columns = ["tipoSensor", "cantidad"]

    plt.figure(figsize=(12, 6))
    sns.barplot(data=conteo_global, x="tipoSensor", y="cantidad", palette="muted")
    plt.title("Cantidad de sensores por tipo (Global)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convertir a base64 para HTML
    plot_base64 = fig_to_base64(plt)
    plt.close()
    
    return {"image": plot_base64}

# 2. Sensores asignados vs. no asignados a entornos - Bokeh
@app.get("/plot2")
async def get_plot2():
    # Sensor IDs usados en entornos
    ids_usados = set(sensor["idSensor"] for e in entornos_col.find() for sensor in e["sensores"])

    # Todos los sensores en la base
    todos_los_sensores = [sensor["idSensor"] for s in sensordatas_col.find() for sensor in s["sensores"]]

    asignados = sum(1 for sid in todos_los_sensores if sid in ids_usados)
    no_asignados = len(todos_los_sensores) - asignados

    # Crear dataset
    data = pd.Series({"Asignados": asignados, "No asignados": no_asignados})
    data = data.reset_index(name='value').rename(columns={'index':'categoria'})
    data["angle"] = data["value"]/data["value"].sum() * 2*np.pi
    data["color"] = ["#3182bd", "#6baed6"]  # Colores azules para las categorías

    # Crear figura Bokeh
    p = figure(height=350, title="Sensores asignados vs no asignados", toolbar_location=None,
               tools="hover", tooltips="@categoria: @value", x_range=(-1, 1))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum("angle", include_zero=True),
            end_angle=cumsum("angle"),
            line_color="white", fill_color="color", legend_field="categoria", source=data)

    p.axis.visible = False
    p.grid.visible = False
    
    return json.loads(json.dumps(json_item(p)))

# 3. Usuarios con mayor número de entornos - ggplot (plotnine)
@app.get("/plot3")
async def get_plot3():
    # Obtener nombres de usuarios desde la colección
    usuarios_dict = {str(u["_id"]): u["nombre"] for u in users_col.find()}

    # Cargar los entornos en DataFrame
    entornos_df = pd.DataFrame(list(entornos_col.find()))

    # Asegurar que la columna 'usuario' sea string para el mapeo
    entornos_df["usuario"] = entornos_df["usuario"].astype(str)

    # Contar entornos por usuario
    entornos_count = entornos_df["usuario"].value_counts().reset_index()
    entornos_count.columns = ["usuario", "cantidad"]

    # Mapear los nombres
    entornos_count["nombre_usuario"] = entornos_count["usuario"].map(usuarios_dict)

    # Solo mostrar los 15 con más entornos
    top15 = entornos_count.head(15)

    # Crear gráfico plotnine
    plot = (ggplot(top15, aes(x="reorder(nombre_usuario, cantidad)", y="cantidad")) +
            geom_bar(stat="identity", fill="#69b3a2") +
            coord_flip() +
            theme_minimal() +
            labs(title="Top 15 usuarios con más entornos", x="Usuario", y="Cantidad de entornos"))
    
    # Guardar temporalmente y convertir a base64
    plot.save("temp_plot.png", dpi=100)
    with open("temp_plot.png", "rb") as f:
        plot_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    return {"image": plot_base64}

# 4. Promedio de sensores por entorno - Plotly
@app.get("/plot4")
async def get_plot4():
    entornos_df = pd.DataFrame(list(entornos_col.find()))
    entornos_df["num_sensores"] = entornos_df["sensores"].apply(len)
    entornos_usuario = entornos_df.groupby("usuario")["num_sensores"].mean().reset_index()
    entornos_usuario.columns = ["usuario", "prom_sensores"]

    fig = px.box(entornos_usuario, y="prom_sensores", title="Promedio de sensores por entorno (por usuario)")
    fig.update_traces(marker_color="indianred")
    
    return fig.to_json()

# 5. Proporción de tipos de sensores por entorno - Altair
@app.get("/plot5")
async def get_plot5():
    # Expandir sensores por entorno
    datos = []
    for e in entornos_col.find():
        for s in e["sensores"]:
            datos.append({
                "entorno": e["nombre"],
                "tipoSensor": s["tipoSensor"]
            })

    df = pd.DataFrame(datos)

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('entorno:N', sort='-y', title="Entorno"),
        y=alt.Y('count()', title="Cantidad"),
        color=alt.Color('tipoSensor:N', title="Tipo de sensor"),
        tooltip=["entorno:N", "tipoSensor:N", "count()"]
    ).properties(
        width=700,
        height=400,
        title="Proporción de tipos de sensores por entorno"
    ).interactive()
    
    return chart.to_dict()