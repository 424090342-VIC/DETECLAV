import tkinter as tk
from tkinter import messagebox
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

EXCEL_PATH = "simulacion_trans.xlsx"

from tkinter import scrolledtext


def mostrar_analisis_anomalias_combinado(df):
    try:
        cuentas = list(set(df['Cuenta Origen']).union(set(df['Cuenta Destino'])))

        resumen = []
        for cuenta in cuentas:
            enviadas = df[df['Cuenta Origen'] == cuenta].shape[0]
            recibidas = df[df['Cuenta Destino'] == cuenta].shape[0]
            resumen.append([enviadas, recibidas])

        X = np.array(resumen)

        # DBSCAN
        dbscan = DBSCAN(eps=1.5, min_samples=2)
        dbscan_clusters = dbscan.fit_predict(X)

        # KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_clusters = kmeans.fit_predict(X)

        # Preparar texto de resultados
        texto = "🔍 Análisis combinado de anomalías\n\n"

        # DBSCAN: cuentas en ruido
        ruido = [cuentas[i] for i, c in enumerate(dbscan_clusters) if c == -1]
        if ruido:
            texto += "Cuentas ANÓMALAS detectadas por DBSCAN (ruido):\n"
            texto += ", ".join(str(cuenta) for cuenta in ruido) + "\n\n"
        else:
            texto += "No se detectaron cuentas anómalas por DBSCAN.\n\n"

        # KMeans: cluster más pequeño como anomalías
        from collections import Counter
        counts = Counter(kmeans_clusters)
        cluster_min = min(counts, key=counts.get)
        anomalas_kmeans = [cuentas[i] for i, c in enumerate(kmeans_clusters) if c == cluster_min]

        texto += f"Cuentas en cluster pequeño (posible anomalía) según KMeans (cluster {cluster_min}):\n"
        texto += ", ".join(str(cuenta) for cuenta in anomalas_kmeans) + "\n\n"


        # Mostrar ventana con scroll
        ventana_analisis = tk.Toplevel()
        ventana_analisis.title("Análisis de Anomalías Combinado")
        ventana_analisis.geometry("550x400")

        texto_area = scrolledtext.ScrolledText(ventana_analisis, wrap=tk.WORD, font=("Helvetica", 11))
        texto_area.pack(expand=True, fill='both', padx=10, pady=10)
        texto_area.insert(tk.END, texto)
        texto_area.config(state=tk.DISABLED)

    except Exception as e:
        messagebox.showerror("Error Análisis combinado", f"No se pudo realizar el análisis combinado:\n{e}")


from sklearn.cluster import KMeans
import numpy as np

from sklearn.cluster import DBSCAN



def ejecutar_hits(grafo):
    try:
        hubs, authorities = nx.hits(grafo, max_iter=1000, normalized=True)
        
        top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:5]
        top_auth = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:5]

        mensaje = "🔎 HITS – Top cuentas:\n\n"
        mensaje += "🏦 Autoridades (receptores importantes):\n"
        for cuenta, score in top_auth:
            mensaje += f" - Cuenta {cuenta}: {score:.4f}\n"

        mensaje += "\n📤 Hubs (emisores importantes):\n"
        for cuenta, score in top_hubs:
            mensaje += f" - Cuenta {cuenta}: {score:.4f}\n"

        messagebox.showinfo("HITS (Hubs & Authorities)", mensaje)

    except Exception as e:
        messagebox.showerror("Error HITS", f"No se pudo ejecutar HITS:\n{e}")


def ejecutar_pagerank(grafo):
    try:
        pr = nx.pagerank(grafo, alpha=0.85)
        top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
        mensaje = "Top 5 cuentas más influyentes (PageRank):\n\n"
        for cuenta, score in top:
            mensaje += f"Cuenta {cuenta}: {score:.4f}\n"
        messagebox.showinfo("PageRank", mensaje)
    except Exception as e:
        messagebox.showerror("Error PageRank", f"No se pudo ejecutar PageRank:\n{e}")

def detectar_ciclos(grafo):
    ciclos = list(nx.simple_cycles(grafo))
    if ciclos:
        mensaje = f"Se detectaron {len(ciclos)} ciclo(s) en las transacciones:\n\n"
        # Mostrar solo primeros 5 ciclos para no saturar
        for i, ciclo in enumerate(ciclos[:5], 1):
            mensaje += f"Ciclo {i}: {' -> '.join(map(str, ciclo))} -> {ciclo[0]}\n"
        if len(ciclos) > 5:
            mensaje += "\n... y más ciclos no mostrados."
        messagebox.showwarning("Ciclos detectados", mensaje)
    else:
        messagebox.showinfo("Ciclos detectados", "No se encontraron ciclos en las transacciones.")

def construir_grafo_y_mostrar():
    try:
        df = pd.read_excel(EXCEL_PATH)

        G = nx.DiGraph()

        tipos_transaccion = df['Tipo de Transaccion'].unique()
        cmap = plt.get_cmap('tab10', len(tipos_transaccion))
        tipo_color_map = {tipo: mcolors.rgb2hex(cmap(i)) for i, tipo in enumerate(tipos_transaccion)}

        for _, fila in df.iterrows():
            origen = fila['Cuenta Origen']
            destino = fila['Cuenta Destino']
            tipo = fila['Tipo de Transaccion']
            categoria = fila['CategoriaNaturaleza de Transaccion']
            hora = fila['Hora de la Transaccion']

            G.add_node(origen)
            G.add_node(destino)
            G.add_edge(origen, destino, tipo=tipo, categoria=categoria, hora=hora)

        detectar_ciclos(G)
        ejecutar_pagerank(G)
        ejecutar_hits(G)
        mostrar_analisis_anomalias_combinado(df)



        # Determinar roles de los nodos
        nodos_colores = {}
        for nodo in G.nodes():
            in_deg = G.in_degree(nodo)
            out_deg = G.out_degree(nodo)
            if in_deg > 0 and out_deg > 0:
                nodos_colores[nodo] = "#9b59b6"  
            elif out_deg > 0:
                nodos_colores[nodo] = "#3498db"  
            else:
                nodos_colores[nodo] = "#2ecc71"  

        plt.figure(figsize=(12, 9))
        pos = nx.spring_layout(G, k=0.4, seed=42)

        # Dibujar nodos con colores
        nx.draw_networkx_nodes(G, pos, node_color=[nodos_colores[n] for n in G.nodes()],
                               node_size=600, alpha=0.9)

        # Dibujar aristas con color por tipo de transacción
        for tipo in tipos_transaccion:
            edges = [(u, v) for u, v, d in G.edges(data=True) if d['tipo'] == tipo]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=tipo_color_map[tipo], width=2)

        # Etiquetas
        nx.draw_networkx_labels(G, pos, font_size=8)

        # Leyenda de aristas (tipos de transacción)
        patches_aristas = [mpatches.Patch(color=color, label=tipo) for tipo, color in tipo_color_map.items()]
        legend1 = plt.legend(handles=patches_aristas, title="Tipo de Transacción", loc="upper left", bbox_to_anchor=(1.02, 1))

        # Leyenda de nodos (roles)
        circle_emisor = mpatches.Circle((0, 0), radius=0.15, color="#3498db", label="Solo Emisor")
        circle_receptor = mpatches.Circle((0, 0), radius=0.15, color="#2ecc71", label="Solo Receptor")
        circle_mixto = mpatches.Circle((0, 0), radius=0.15, color="#9b59b6", label="Mixto")

        legend2 = plt.legend(
            handles=[circle_emisor, circle_receptor, circle_mixto],
            title="Tipo de Cuenta",
            loc="upper left",
            bbox_to_anchor=(1.02, 0.6)
        )

        plt.gca().add_artist(legend1)  # Añade primero la leyenda de aristas

        plt.title("Grafo de Transacciones – DETECLAV", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


    except Exception as e:
        messagebox.showerror("Error", f"No se pudo construir el grafo:\n{e}")

# Interfaz
def iniciar_analisis():
    construir_grafo_y_mostrar()

ventana = tk.Tk()
ventana.title("DETECLAV – Detección de Lavado de Dinero")
ventana.geometry("600x400")
ventana.config(bg="#f0f0f0")

titulo = tk.Label(ventana, text="DETECLAV", font=("Helvetica", 28, "bold"), fg="#1e3d59", bg="#f0f0f0")
titulo.pack(pady=20)

descripcion = tk.Label(
    ventana,
    text="Este programa analiza transacciones bancarias\npara detectar patrones asociados al lavado de dinero.",
    font=("Helvetica", 12),
    bg="#f0f0f0"
)
descripcion.pack(pady=10)

boton_iniciar = tk.Button(
    ventana,
    text="Iniciar análisis",
    font=("Helvetica", 14),
    bg="#4CAF50",
    fg="white",
    padx=20,
    pady=10,
    command=iniciar_analisis
)
boton_iniciar.pack(pady=30)

ventana.mainloop()
