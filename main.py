from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
import os
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
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from pathlib import Path
from dotenv import load_dotenv
import logging

# Configuración inicial para Matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI
plt.switch_backend('Agg')

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración de la aplicación FastAPI
app = FastAPI(
    title="Visualización de Datos IoT",
    description="API para visualizar datos de sensores IoT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "HEAD"],
    allow_headers=["*"],
)

# Configuración de directorios estáticos
static_dir = Path("static")
if not static_dir.exists():
    static_dir.mkdir()
    logger.info("Directorio 'static' creado")

templates_dir = Path("templates")
if not templates_dir.exists():
    templates_dir.mkdir()
    logger.info("Directorio 'templates' creado")

# Configuración de templates y archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Conexión segura a MongoDB
def get_mongo_connection():
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        error_msg = "La variable de entorno MONGO_URI no está configurada"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        client = MongoClient(
            MONGO_URI,
            connectTimeoutMS=5000,
            serverSelectionTimeoutMS=5000,
            retryWrites=True,
            retryReads=True
        )
        # Test de conexión
        client.admin.command('ping')
        logger.info("Conexión a MongoDB establecida correctamente")
        return client
    except Exception as e:
        error_msg = f"Error de conexión a MongoDB: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Obtener conexión y colecciones
try:
    mongo_client = get_mongo_connection()
    db = mongo_client["iotdb"]
    users_col = db["users"]
    sensordatas_col = db["sensordatas"]
    entornos_col = db["entornos"]
except Exception as e:
    logger.error(f"Error inicializando MongoDB: {str(e)}")
    raise

# Función para convertir gráficas matplotlib a base64
def fig_to_base64(fig):
    try:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error convirtiendo figura a base64: {str(e)}")
        raise

# Health check endpoint
@app.get("/health", include_in_schema=False)
async def health_check():
    try:
        # Verificar conexión a MongoDB
        mongo_client.admin.command('ping')
        return JSONResponse(
            content={"status": "ok", "database": "connected"},
            status_code=200
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            content={"status": "error", "database": "disconnected"},
            status_code=503
        )

# Ruta principal
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.head("/", include_in_schema=False)
async def head_root():
    return Response(status_code=200)

# 1. Cantidad de sensores por tipo (global)
@app.get("/plot1", summary="Cantidad de sensores por tipo (global)")
async def get_plot1():
    try:
        # Obtener y expandir sensores con proyección para mejorar rendimiento
        sensores_expandido = pd.DataFrame([
            {"tipoSensor": sensor["tipoSensor"]}
            for s in sensordatas_col.find({}, {"sensores.tipoSensor": 1})
            for sensor in s["sensores"]
        ])

        if sensores_expandido.empty:
            error_msg = "No se encontraron datos de sensores"
            logger.warning(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)

        conteo_global = sensores_expandido["tipoSensor"].value_counts().reset_index()
        conteo_global.columns = ["tipoSensor", "cantidad"]

        # Ordenar y construir gráfica interactiva con Plotly (ya presente en el proyecto)
        conteo_global = conteo_global.sort_values("cantidad", ascending=False)

        fig = px.bar(
            conteo_global,
            x="tipoSensor",
            y="cantidad",
            title="Cantidad de sensores por tipo (Global)",
            labels={"tipoSensor": "Tipo de sensor", "cantidad": "Cantidad"},
            color="cantidad",
            color_continuous_scale="Blues",
        )
        fig.update_layout(xaxis_tickangle=-45, yaxis_title="Cantidad", xaxis_title="Tipo de sensor")

        return json.loads(fig.to_json())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /plot1: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al generar la gráfica")

# 2. Sensores asignados vs. no asignados a entornos
@app.get("/plot2", summary="Distribución de sensores asignados")
async def get_plot2():
    try:
        # Optimización: Solo obtener los campos necesarios
        sensores_entornos = set()
        for e in entornos_col.find({}, {"sensores.idSensor": 1}):
            for sensor in e["sensores"]:
                sensores_entornos.add(sensor["idSensor"])

        total_sensores = sum(1 for _ in sensordatas_col.aggregate([
            {"$unwind": "$sensores"},
            {"$project": {"sensores.idSensor": 1}},
            {"$group": {"_id": None, "count": {"$sum": 1}}}
        ]))

        if total_sensores == 0:
            error_msg = "No se encontraron sensores"
            logger.warning(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)

        asignados = len(sensores_entornos)
        no_asignados = total_sensores - asignados

        data = pd.Series({"Asignados": asignados, "No asignados": no_asignados})
        data = data.reset_index(name='value').rename(columns={'index':'categoria'})
        data["angle"] = data["value"]/data["value"].sum() * 2*np.pi
        data["color"] = ["#3182bd", "#6baed6"]

        p = figure(height=350, title="Sensores asignados vs no asignados", 
                  toolbar_location=None, tools="hover", 
                  tooltips="@categoria: @value", x_range=(-1, 1))

        p.wedge(x=0, y=1, radius=0.4,
                start_angle=cumsum("angle", include_zero=True),
                end_angle=cumsum("angle"),
                line_color="white", fill_color="color", 
                legend_field="categoria", source=data)

        p.axis.visible = False
        p.grid.visible = False
        
        return json.loads(json.dumps(json_item(p)))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /plot2: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al generar la gráfica")

# 3. Usuarios con mayor número de entornos
@app.get("/plot3", summary="Top usuarios con más entornos")
async def get_plot3():
    try:
        usuarios_dict = {str(u["_id"]): u["nombre"] for u in users_col.find({}, {"nombre": 1})}
        
        pipeline = [
            {"$group": {"_id": "$usuario", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 15}
        ]
        
        top_usuarios = list(entornos_col.aggregate(pipeline))
        if not top_usuarios:
            error_msg = "No se encontraron entornos"
            logger.warning(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)

        top15 = pd.DataFrame([{
            "usuario": str(u["_id"]),
            "cantidad": u["count"],
            "nombre_usuario": usuarios_dict.get(str(u["_id"]), "Desconocido")
        } for u in top_usuarios])

        # Gráfica interactiva con Altair (ya presente en el proyecto)
        chart = (
            alt.Chart(top15)
            .mark_bar()
            .encode(
                x=alt.X("cantidad:Q", title="Cantidad de entornos"),
                y=alt.Y("nombre_usuario:N", sort='-x', title="Usuario"),
                tooltip=["nombre_usuario:N", "cantidad:Q"],
                color=alt.value("#69b3a2"),
            )
            .properties(
                width=700,
                height=400,
                title="Top 15 usuarios con más entornos",
            )
            .interactive()
        )

        return chart.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /plot3: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al generar la gráfica")

# 4. Promedio de sensores por entorno
@app.get("/plot4", summary="Distribución de sensores por entorno")
async def get_plot4():
    try:
        pipeline = [
            {"$project": {"num_sensores": {"$size": "$sensores"}, "usuario": 1}},
            {"$group": {"_id": "$usuario", "prom_sensores": {"$avg": "$num_sensores"}}}
        ]
        
        result = list(entornos_col.aggregate(pipeline))
        if not result:
            error_msg = "No se encontraron entornos"
            logger.warning(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)

        entornos_usuario = pd.DataFrame([{
            "usuario": str(r["_id"]),
            "prom_sensores": r["prom_sensores"]
        } for r in result])

        fig = px.box(entornos_usuario, y="prom_sensores", 
                    title="Promedio de sensores por entorno (por usuario)")
        fig.update_traces(marker_color="indianred")
        fig.update_layout(yaxis_title="Promedio de sensores")
        
        return json.loads(fig.to_json())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /plot4: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al generar la gráfica")

# 5. Proporción de tipos de sensores por entorno
@app.get("/plot5", summary="Tipos de sensores por entorno")
async def get_plot5():
    try:
        pipeline = [
            {"$unwind": "$sensores"},
            {"$project": {"entorno": "$nombre", "tipoSensor": "$sensores.tipoSensor"}},
            {"$group": {"_id": {"entorno": "$entorno", "tipo": "$tipoSensor"}, "count": {"$sum": 1}}}
        ]
        
        datos = list(entornos_col.aggregate(pipeline))
        if not datos:
            error_msg = "No se encontraron datos"
            logger.warning(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)

        df = pd.DataFrame([{
            "entorno": d["_id"]["entorno"],
            "tipoSensor": d["_id"]["tipo"],
            "count": d["count"]
        } for d in datos])

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('entorno:N', sort='-y', title="Entorno"),
            y=alt.Y('count:Q', title="Cantidad"),
            color=alt.Color('tipoSensor:N', title="Tipo de sensor"),
            tooltip=["entorno:N", "tipoSensor:N", "count:Q"]
        ).properties(
            width=700,
            height=400,
            title="Proporción de tipos de sensores por entorno"
        ).interactive()
        
        return chart.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /plot5: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno al generar la gráfica")

# Middleware para manejo de errores
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Para ejecución local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
        timeout_keep_alive=30
    )