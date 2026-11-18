"""
GEODETEC - Sistema inteligente de análisis territorial simulado
================================================================

Proyecto académico de Sistemas Inteligentes.

Todos los datos son SINTÉTICOS: se generan territorios ficticios en forma de
cuadrícula (filas x columnas). Ninguna celda representa coordenadas, rutas,
lugares ni hechos reales. Las categorías de prioridad son etiquetas de un
escenario experimental, no probabilidades reales.

Flujo general:
    1. Simulación   -> territorio de ENTRENAMIENTO (80 regiones, semilla A)
                       territorio de PRUEBA (20 regiones, semilla B y
                       distribución ligeramente desplazada)
                       territorio de EXPLORACIÓN (mapa 20x20, semilla C)
    2. Persistencia -> todo se guarda en SQLite (geodetec.db)
    3. Aprendizaje  -> RandomForestClassifier aprende la relación
                       variables -> prioridad (baja / media / alta)
    4. Evaluación   -> accuracy, precision, recall, F1, matriz de confusión,
                       validación cruzada e importancia de variables
    5. Localización -> el modelo entrenado clasifica cada celda del mapa de
                       exploración y genera un mapa de calor de prioridad
    6. Búsqueda     -> dada la última ubicación conocida (ficticia) de una
                       persona, combina la probabilidad de "prioridad alta"
                       con la cercanía para delimitar una zona prioritaria

Uso:
    python geodetec.py            -> abre la interfaz gráfica (Tkinter)
    python geodetec.py --consola  -> ejecuta todo sin interfaz y guarda las
                                     gráficas como PNG en la carpeta salidas/
"""

import os
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
DB_PATH = "geodetec.db"

SEMILLA_TRAIN = 2024
SEMILLA_TEST = 7
SEMILLA_MAPA = 99

# Tamaño de cada territorio (filas x columnas)
DIM_TRAIN = (10, 8)    # 80 regiones
DIM_TEST = (5, 4)      # 20 regiones
DIM_MAPA = (20, 20)    # 400 celdas para el mapa de calor

VARIABLES = [
    "terrain_index",
    "vegetation_index",
    "population_density",
    "accessibility_index",
    "historical_record_count",
    "report_frequency",
]

CLASES = ["Baja", "Media", "Alta"]          # 0, 1, 2
COLORES_CLASES = ["#2ecc71", "#f1c40f", "#e74c3c"]

# Umbrales de la regla conceptual (iguales para todos los conjuntos)
UMBRAL_MEDIA = 0.42
UMBRAL_ALTA = 0.55

# Perfiles de simulación: el de prueba está ligeramente desplazado
PERFIL_TRAIN = {
    "veg_shift": 0.00,
    "pop_factor": 1.00,
    "hist_factor": 1.00,
    "report_factor": 1.00,
    "ruido_variables": 0.03,
    "ruido_etiqueta": 0.025,
}
PERFIL_TEST = {
    "veg_shift": 0.06,       # un poco más de vegetación
    "pop_factor": 0.85,      # algo menos de población
    "hist_factor": 1.15,     # algunos registros históricos más
    "report_factor": 0.90,   # algunos reportes menos
    "ruido_variables": 0.05, # más ruido
    "ruido_etiqueta": 0.035,
}
PERFIL_MAPA = dict(PERFIL_TRAIN)


# ---------------------------------------------------------------------------
# 1. SIMULACIÓN DE DATOS
# ---------------------------------------------------------------------------
def campo_suave(rng, filas, cols, n_focos=4):
    """Genera un campo espacial suave en [0, 1] sumando 'focos' gaussianos.

    Así las regiones vecinas se parecen entre sí, como ocurriría en un
    territorio, en lugar de ser valores totalmente aleatorios.
    """
    yy, xx = np.mgrid[0:filas, 0:cols]
    campo = np.zeros((filas, cols))
    escala = max(filas, cols)
    for _ in range(n_focos):
        cy, cx = rng.uniform(0, filas), rng.uniform(0, cols)
        sigma = rng.uniform(0.18, 0.40) * escala
        amp = rng.uniform(0.5, 1.0)
        campo += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    campo = (campo - campo.min()) / (campo.max() - campo.min() + 1e-9)
    return campo


def puntaje_latente(df, rng, ruido):
    """Regla conceptual (oculta para el modelo) que define la prioridad.

    El modelo NUNCA ve esta fórmula: solo ve las variables y la etiqueta
    resultante, y debe aprender el patrón a partir de los ejemplos.
    """
    hist = np.clip(df["historical_record_count"] / 15, 0, 1)
    rep = np.clip(df["report_frequency"] / 10, 0, 1)
    veg = df["vegetation_index"]
    pob = 1 - np.clip(df["population_density"] / 500, 0, 1)
    # accesibilidad intermedia: ni totalmente aislada ni muy transitada
    acc = 1 - np.abs(df["accessibility_index"] - 0.5) * 2
    ter = df["terrain_index"]
    s = (0.30 * hist + 0.20 * rep + 0.20 * veg + 0.15 * pob
         + 0.10 * acc + 0.05 * ter)
    return s + rng.normal(0, ruido, len(df))


def etiquetar(puntaje):
    return np.where(puntaje >= UMBRAL_ALTA, 2,
                    np.where(puntaje >= UMBRAL_MEDIA, 1, 0))


def generar_territorio(dim, semilla, perfil, nombre):
    """Crea un territorio ficticio de dim=(filas, cols) regiones."""
    rng = np.random.default_rng(semilla)
    filas, cols = dim
    r = perfil["ruido_variables"]

    def ruido():
        return rng.normal(0, r, (filas, cols))

    terreno = np.clip(campo_suave(rng, filas, cols) + ruido(), 0, 1)
    vegetacion = np.clip(campo_suave(rng, filas, cols) + perfil["veg_shift"] + ruido(), 0, 1)
    poblacion = np.clip((20 + 480 * campo_suave(rng, filas, cols)) * perfil["pop_factor"]
                        + ruido() * 100, 0, None)
    accesibilidad = np.clip(campo_suave(rng, filas, cols) + ruido(), 0, 1)

    base_hist = 0.6 * vegetacion + 0.4 * campo_suave(rng, filas, cols)
    historicos = rng.poisson((1 + 13 * base_hist) * perfil["hist_factor"])
    base_rep = 0.5 * base_hist + 0.5 * campo_suave(rng, filas, cols)
    reportes = rng.poisson((0.5 + 9 * base_rep) * perfil["report_factor"])

    ff, cc = np.mgrid[0:filas, 0:cols]
    df = pd.DataFrame({
        "region_id": [f"{nombre[:2].upper()}-{i:03d}" for i in range(filas * cols)],
        "fila": ff.ravel(),
        "columna": cc.ravel(),
        "terrain_index": terreno.ravel().round(4),
        "vegetation_index": vegetacion.ravel().round(4),
        "population_density": poblacion.ravel().round(2),
        "accessibility_index": accesibilidad.ravel().round(4),
        "historical_record_count": historicos.ravel(),
        "report_frequency": reportes.ravel(),
    })
    df["prioridad"] = etiquetar(puntaje_latente(df, rng, perfil["ruido_etiqueta"]))
    df["conjunto"] = nombre
    return df


def generar_datos():
    train = generar_territorio(DIM_TRAIN, SEMILLA_TRAIN, PERFIL_TRAIN, "entrenamiento")
    test = generar_territorio(DIM_TEST, SEMILLA_TEST, PERFIL_TEST, "prueba")
    mapa = generar_territorio(DIM_MAPA, SEMILLA_MAPA, PERFIL_MAPA, "mapa")
    return train, test, mapa


# ---------------------------------------------------------------------------
# 2. PERSISTENCIA EN SQLITE
# ---------------------------------------------------------------------------
def guardar_datos(train, test, mapa, db=DB_PATH):
    with sqlite3.connect(db) as con:
        train.to_sql("regiones_entrenamiento", con, if_exists="replace", index=False)
        test.to_sql("regiones_prueba", con, if_exists="replace", index=False)
        mapa.to_sql("regiones_mapa", con, if_exists="replace", index=False)


def cargar_tabla(tabla, db=DB_PATH):
    with sqlite3.connect(db) as con:
        return pd.read_sql(f"SELECT * FROM {tabla}", con)


def guardar_metricas(m, db=DB_PATH):
    fila = pd.DataFrame([{
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "accuracy": m["accuracy"],
        "precision_macro": m["precision"],
        "recall_macro": m["recall"],
        "f1_macro": m["f1"],
        "cv_media": m["cv_media"],
        "cv_desv": m["cv_desv"],
    }])
    with sqlite3.connect(db) as con:
        fila.to_sql("historial_metricas", con, if_exists="append", index=False)


def guardar_predicciones_mapa(mapa_pred, db=DB_PATH):
    cols = ["region_id", "fila", "columna", "prediccion",
            "prob_baja", "prob_media", "prob_alta"]
    with sqlite3.connect(db) as con:
        mapa_pred[cols].to_sql("predicciones_mapa", con, if_exists="replace", index=False)


# ---------------------------------------------------------------------------
# 3. APRENDIZAJE (Random Forest)
# ---------------------------------------------------------------------------
def entrenar_modelo(train):
    X, y = train[VARIABLES], train["prioridad"]
    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    # Validación cruzada estratificada SOLO con los datos de entrenamiento
    k = min(5, int(y.value_counts().min()))
    k = max(k, 2)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")
    modelo.fit(X, y)
    return modelo, cv_scores


# ---------------------------------------------------------------------------
# 4. EVALUACIÓN
# ---------------------------------------------------------------------------
def evaluar_modelo(modelo, test, cv_scores):
    X, y = test[VARIABLES], test["prioridad"]
    y_pred = modelo.predict(X)
    etiquetas = [0, 1, 2]
    m = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, labels=etiquetas, average="macro", zero_division=0),
        "recall": recall_score(y, y_pred, labels=etiquetas, average="macro", zero_division=0),
        "f1": f1_score(y, y_pred, labels=etiquetas, average="macro", zero_division=0),
        "matriz": confusion_matrix(y, y_pred, labels=etiquetas),
        "reporte": classification_report(y, y_pred, labels=etiquetas,
                                         target_names=CLASES, zero_division=0),
        "cv_media": float(np.mean(cv_scores)),
        "cv_desv": float(np.std(cv_scores)),
        "y_pred": y_pred,
    }
    return m


def texto_metricas(m, train, test):
    dist_tr = train["prioridad"].value_counts().reindex([0, 1, 2], fill_value=0)
    dist_te = test["prioridad"].value_counts().reindex([0, 1, 2], fill_value=0)
    t = "GEODETEC - Resultados del modelo\n"
    t += "=" * 44 + "\n\n"
    t += f"Registros de entrenamiento: {len(train)}\n"
    t += f"Registros de prueba:        {len(test)}\n\n"
    t += "Distribución de clases (Baja / Media / Alta)\n"
    t += f"  Entrenamiento: {dist_tr[0]} / {dist_tr[1]} / {dist_tr[2]}\n"
    t += f"  Prueba:        {dist_te[0]} / {dist_te[1]} / {dist_te[2]}\n\n"
    t += f"Validación cruzada (train): {m['cv_media']:.3f} ± {m['cv_desv']:.3f}\n\n"
    t += "Métricas en el conjunto de PRUEBA (promedio macro)\n"
    t += f"  Accuracy : {m['accuracy']:.3f}\n"
    t += f"  Precision: {m['precision']:.3f}\n"
    t += f"  Recall   : {m['recall']:.3f}\n"
    t += f"  F1-score : {m['f1']:.3f}\n\n"
    t += "Reporte por clase\n" + m["reporte"] + "\n"
    t += ("Nota: con 20 registros de prueba cada error cambia la accuracy\n"
          "en 5 puntos porcentuales; interpretar con cautela.\n")
    return t


# ---------------------------------------------------------------------------
# 5. LOCALIZACIÓN: aplicar el modelo al mapa de exploración
# ---------------------------------------------------------------------------
def predecir_mapa(modelo, mapa):
    proba = modelo.predict_proba(mapa[VARIABLES])
    # asegurar columnas en orden 0,1,2 aunque falte alguna clase
    prob = np.zeros((len(mapa), 3))
    for j, c in enumerate(modelo.classes_):
        prob[:, int(c)] = proba[:, j]
    out = mapa.copy()
    out["prediccion"] = prob.argmax(axis=1)
    out["prob_baja"], out["prob_media"], out["prob_alta"] = prob[:, 0], prob[:, 1], prob[:, 2]
    return out


def a_matriz(df, columna):
    filas, cols = df["fila"].max() + 1, df["columna"].max() + 1
    M = np.zeros((filas, cols))
    M[df["fila"], df["columna"]] = df[columna]
    return M


# ---------------------------------------------------------------------------
# 6. CAPA DE BÚSQUEDA (persona ficticia)
# ---------------------------------------------------------------------------
def zona_busqueda(mapa_pred, fila0, col0, sigma=4.0, peso_modelo=0.6, porcentaje=0.10):
    """Combina la probabilidad de prioridad alta con la cercanía a la
    última ubicación conocida (ficticia) y devuelve la matriz de prioridad
    de búsqueda y la máscara de las celdas más prioritarias."""
    P = a_matriz(mapa_pred, "prob_alta")
    yy, xx = np.mgrid[0:P.shape[0], 0:P.shape[1]]
    cercania = np.exp(-((yy - fila0) ** 2 + (xx - col0) ** 2) / (2 * sigma ** 2))
    prioridad = peso_modelo * P + (1 - peso_modelo) * cercania
    corte = np.quantile(prioridad, 1 - porcentaje)
    mascara = prioridad >= corte
    # rellenar huecos interiores: la zona de búsqueda debe ser continua
    from scipy.ndimage import binary_fill_holes
    mascara = binary_fill_holes(mascara)
    return prioridad, mascara


# ---------------------------------------------------------------------------
# 7. GRÁFICAS (devuelven objetos Figure: sirven en Tkinter y en consola)
# ---------------------------------------------------------------------------
def fig_matriz_confusion(m):
    fig = Figure(figsize=(5.5, 4.5))
    ax = fig.add_subplot(111)
    M = m["matriz"]
    im = ax.imshow(M, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, M[i, j], ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "black", fontsize=13)
    ax.set_xticks(range(3), CLASES)
    ax.set_yticks(range(3), CLASES)
    ax.set_xlabel("Predicción del modelo")
    ax.set_ylabel("Etiqueta simulada (real)")
    ax.set_title("Matriz de confusión – conjunto de prueba")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def fig_importancia(modelo):
    imp = pd.Series(modelo.feature_importances_, index=VARIABLES).sort_values()
    fig = Figure(figsize=(6.5, 4))
    ax = fig.add_subplot(111)
    ax.barh(imp.index, imp.values, color="#3498db")
    for i, v in enumerate(imp.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Importancia (reducción media de impureza)")
    ax.set_title("Importancia de variables – Random Forest")
    ax.set_xlim(0, imp.max() * 1.2)
    fig.tight_layout()
    return fig


def fig_distribuciones(train, test):
    fig = Figure(figsize=(10, 6))
    for k, var in enumerate(VARIABLES, 1):
        ax = fig.add_subplot(2, 3, k)
        ax.hist(train[var], bins=12, alpha=0.6, density=True, label="Entrenamiento", color="#3498db")
        ax.hist(test[var], bins=8, alpha=0.6, density=True, label="Prueba", color="#16a085")
        ax.set_title(var, fontsize=9)
        ax.tick_params(labelsize=7)
    fig.axes[0].legend(fontsize=7)
    fig.suptitle("Distribuciones: entrenamiento vs prueba (datos simulados distintos)")
    fig.tight_layout()
    return fig


def fig_mapa(mapa_pred):
    cmap = ListedColormap(COLORES_CLASES)
    fig = Figure(figsize=(13, 4.5))
    ax1 = fig.add_subplot(131)
    ax1.imshow(a_matriz(mapa_pred, "prioridad"), cmap=cmap, vmin=0, vmax=2)
    ax1.set_title("Prioridad simulada (oculta al modelo)")
    ax2 = fig.add_subplot(132)
    ax2.imshow(a_matriz(mapa_pred, "prediccion"), cmap=cmap, vmin=0, vmax=2)
    ax2.set_title("Prioridad predicha por el modelo")
    ax3 = fig.add_subplot(133)
    im = ax3.imshow(a_matriz(mapa_pred, "prob_alta"), cmap="inferno", vmin=0, vmax=1)
    ax3.set_title("Mapa de calor: P(prioridad alta)")
    fig.colorbar(im, ax=ax3, fraction=0.046)
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("columna")
        ax.set_ylabel("fila")
    import matplotlib.patches as mpatches
    fig.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in zip(COLORES_CLASES, CLASES)],
               loc="lower center", ncol=3)
    acierto = (mapa_pred["prioridad"] == mapa_pred["prediccion"]).mean()
    fig.suptitle(f"Territorio de exploración ficticio 20x20 – coincidencia {acierto:.1%}")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    return fig


def fig_busqueda(mapa_pred, fila0, col0):
    prioridad, mascara = zona_busqueda(mapa_pred, fila0, col0)
    fig = Figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(prioridad, cmap="YlOrRd")
    # Contorno exacto de la zona: se dibujan los bordes entre celdas
    # dentro y fuera de la máscara
    filas, cols = mascara.shape
    for i in range(filas):
        for j in range(cols):
            if not mascara[i, j]:
                continue
            if i == 0 or not mascara[i - 1, j]:
                ax.plot([j - .5, j + .5], [i - .5, i - .5], color="blue", lw=2.5)
            if i == filas - 1 or not mascara[i + 1, j]:
                ax.plot([j - .5, j + .5], [i + .5, i + .5], color="blue", lw=2.5)
            if j == 0 or not mascara[i, j - 1]:
                ax.plot([j - .5, j - .5], [i - .5, i + .5], color="blue", lw=2.5)
            if j == cols - 1 or not mascara[i, j + 1]:
                ax.plot([j + .5, j + .5], [i - .5, i + .5], color="blue", lw=2.5)
    ax.plot(col0, fila0, marker="*", color="cyan", markersize=18, linestyle="none",
            markeredgecolor="black", label="Última ubicación conocida (ficticia)")
    ax.plot([], [], color="blue", lw=2.5, label="Zona prioritaria")
    ax.set_title("Zona de búsqueda prioritaria (10% de celdas)")
    ax.set_xlabel("columna")
    ax.set_ylabel("fila")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Índice de prioridad de búsqueda")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. ORQUESTADOR (estado compartido entre consola y GUI)
# ---------------------------------------------------------------------------
class Geodetec:
    def __init__(self):
        self.train = self.test = self.mapa = None
        self.modelo = None
        self.cv_scores = None
        self.metricas = None
        self.mapa_pred = None

    def paso_generar(self):
        self.train, self.test, self.mapa = generar_datos()
        guardar_datos(self.train, self.test, self.mapa)
        # se leen de nuevo desde SQLite para usar la base como fuente
        self.train = cargar_tabla("regiones_entrenamiento")
        self.test = cargar_tabla("regiones_prueba")
        self.mapa = cargar_tabla("regiones_mapa")

    def paso_entrenar(self):
        self.modelo, self.cv_scores = entrenar_modelo(self.train)

    def paso_evaluar(self):
        self.metricas = evaluar_modelo(self.modelo, self.test, self.cv_scores)
        guardar_metricas(self.metricas)

    def paso_mapa(self):
        self.mapa_pred = predecir_mapa(self.modelo, self.mapa)
        guardar_predicciones_mapa(self.mapa_pred)

    def ejecutar_todo(self):
        self.paso_generar()
        self.paso_entrenar()
        self.paso_evaluar()
        self.paso_mapa()


# ---------------------------------------------------------------------------
# 9. MODO CONSOLA
# ---------------------------------------------------------------------------
def modo_consola(carpeta="salidas"):
    os.makedirs(carpeta, exist_ok=True)
    g = Geodetec()
    g.ejecutar_todo()
    print(texto_metricas(g.metricas, g.train, g.test))
    figuras = {
        "matriz_confusion.png": fig_matriz_confusion(g.metricas),
        "importancia_variables.png": fig_importancia(g.modelo),
        "distribuciones.png": fig_distribuciones(g.train, g.test),
        "mapa_prioridad.png": fig_mapa(g.mapa_pred),
        "zona_busqueda.png": fig_busqueda(g.mapa_pred, 12, 6),
    }
    for nombre, fig in figuras.items():
        fig.savefig(os.path.join(carpeta, nombre), dpi=130)
    print(f"Gráficas guardadas en ./{carpeta}/ y datos en {DB_PATH}")
    return g


# ---------------------------------------------------------------------------
# 10. INTERFAZ GRÁFICA (Tkinter)
# ---------------------------------------------------------------------------
def modo_gui():
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, simpledialog
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    g = Geodetec()

    def ventana_figura(titulo, fig):
        v = tk.Toplevel(root)
        v.title(titulo)
        canvas = FigureCanvasTkAgg(fig, master=v)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def requiere(*attrs):
        for a in attrs:
            if getattr(g, a) is None:
                messagebox.showwarning("GEODETEC", "Primero completa los pasos anteriores.")
                return False
        return True

    def accion(func):
        def envoltura():
            try:
                func()
            except Exception as e:
                messagebox.showerror("Error", str(e))
        return envoltura

    def actualizar_estado(msg):
        estado.config(text=msg)

    @accion
    def generar():
        g.paso_generar()
        actualizar_estado(f"Datos simulados: {len(g.train)} train, {len(g.test)} test, "
                          f"{len(g.mapa)} celdas de mapa (guardados en {DB_PATH})")

    @accion
    def entrenar():
        if not requiere("train"):
            return
        g.paso_entrenar()
        g.paso_evaluar()
        g.paso_mapa()
        actualizar_estado(f"Modelo entrenado. Accuracy en prueba: {g.metricas['accuracy']:.2%}")

    @accion
    def metricas():
        if not requiere("metricas"):
            return
        v = tk.Toplevel(root)
        v.title("Métricas del modelo")
        v.geometry("560x520")
        txt = scrolledtext.ScrolledText(v, wrap=tk.WORD, font=("Courier", 10))
        txt.pack(expand=True, fill="both", padx=10, pady=10)
        txt.insert(tk.END, texto_metricas(g.metricas, g.train, g.test))
        txt.config(state=tk.DISABLED)

    @accion
    def matriz():
        if requiere("metricas"):
            ventana_figura("Matriz de confusión", fig_matriz_confusion(g.metricas))

    @accion
    def importancia():
        if requiere("modelo"):
            ventana_figura("Importancia de variables", fig_importancia(g.modelo))

    @accion
    def distribuciones():
        if requiere("train"):
            ventana_figura("Distribuciones train vs test", fig_distribuciones(g.train, g.test))

    @accion
    def mapa():
        if requiere("mapa_pred"):
            ventana_figura("Mapa de prioridad", fig_mapa(g.mapa_pred))

    @accion
    def busqueda():
        if not requiere("mapa_pred"):
            return
        f = simpledialog.askinteger("Zona de búsqueda", "Fila de la última ubicación (0-19):",
                                    minvalue=0, maxvalue=DIM_MAPA[0] - 1, parent=root)
        c = simpledialog.askinteger("Zona de búsqueda", "Columna de la última ubicación (0-19):",
                                    minvalue=0, maxvalue=DIM_MAPA[1] - 1, parent=root)
        if f is not None and c is not None:
            ventana_figura("Zona de búsqueda prioritaria", fig_busqueda(g.mapa_pred, f, c))

    root = tk.Tk()
    root.title("GEODETEC – Análisis territorial simulado")
    root.geometry("640x620")
    root.config(bg="#f0f0f0")

    tk.Label(root, text="GEODETEC", font=("Helvetica", 28, "bold"),
             fg="#1e3d59", bg="#f0f0f0").pack(pady=(18, 4))
    tk.Label(root, text="Sistema inteligente de análisis territorial simulado\n"
                        "Random Forest sobre regiones ficticias (datos 100% sintéticos)",
             font=("Helvetica", 11), bg="#f0f0f0").pack(pady=(0, 12))

    botones = [
        ("1. Generar datos simulados", generar, "#1e3d59"),
        ("2. Entrenar y evaluar modelo", entrenar, "#4CAF50"),
        ("Ver métricas", metricas, "#34495e"),
        ("Matriz de confusión", matriz, "#34495e"),
        ("Importancia de variables", importancia, "#34495e"),
        ("Distribuciones train vs test", distribuciones, "#34495e"),
        ("Mapa de prioridad (localización)", mapa, "#c0392b"),
        ("Zona de búsqueda prioritaria", busqueda, "#c0392b"),
    ]
    for texto, cmd, color in botones:
        tk.Button(root, text=texto, command=cmd, font=("Helvetica", 12), bg=color, fg="white",
                  width=32, pady=4).pack(pady=4)

    estado = tk.Label(root, text="Listo. Empieza generando los datos.", bg="#f0f0f0",
                      fg="#555", font=("Helvetica", 9), wraplength=600)
    estado.pack(pady=12)

    root.mainloop()


if __name__ == "__main__":
    if "--consola" in sys.argv:
        modo_consola()
    else:
        modo_gui()
