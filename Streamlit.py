# =================================================================
# == INSTITUTO TECNOLOGICO Y DE ESTUDIOS SUPERIORES DE OCCIDENTE ==
# == ITESO, UNIVERSIDAD JESUITA DE GUADALAJARA                   ==
# ==                                                             ==
# == LABORATORIO DE DESARROLLO DE SOLUCIONES TECNOLÓGICAS        ==
# == Análisis Multimedia basado en Inteligencia Artificial para  ==
# == Soluciones Operativas Empresariales                         ==
# ==                                                             ==
# == TEMA 1                                                      ==
# == Ejemplo de Implementación en Streamlit                      ==
# =================================================================


#----- Importación de Librerías -----------------------------------
import streamlit as st                                                      # elaboración del dashboard
import numpy as np                                                          # operaciones numéricas
import plotly.graph_objects as go                                           # gráficos de bajo nivel
import plotly.express as px                                                 # gráficos rápidos (mostrar imágenes)
from scipy import signal as sp_signal                                       # generación de señales y filtrado digital
from scipy.fft import fft, fftfreq                                          # transformada rápida de Fourier
from PIL import Image                                                       # carga/reescalado de imágenes
from skimage import filters, color, exposure, feature                       # operaciones de procesamiento de imágenes


# -----------------------------------------------------------------
# Configuración general de la página
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Laboratorio de Desarrollo de Soluciones Tecnológicas",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded")


# -----------------------------------------------------------------
# Configuración CSS (Cascading Style Sheet) - Estilo del Dashboard
# -----------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { background-color: #FAFAFA; }
        h1, h2, h3 { font-weight: 600; color: #1F2933; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #F0F2F6;
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;}
        section[data-testid="stSidebar"] {
            background-color: #F4F6F8;
            border-right: 1px solid #E0E0E0;}
        .footer-curso {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #9DB6DA;
            border-top: 1px solid #E0E0E0;
            padding: 6px 24px;
            font-size: 0.88rem;
            color: #1A365D;
            text-align: center;
            z-index: 999;}
        .block-container { padding-bottom: 3.5rem;}
    </style>
    """, unsafe_allow_html=True)


def layout_base(fig, x_title="", y_title="", height=320):
    """Aplica un formato consistente (tema, tamaño, márgenes, leyenda)
    a cualquier figura de Plotly, para no repetir código en cada gráfica."""
    fig.update_layout(
        template="plotly",
        height=height,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified")
    return fig


# -----------------------------------------------------------------
# Encabezado del Dashboard
# -----------------------------------------------------------------
#----- Lectura y Renderizado del Logotipo -------------------------
Logo = Image.open("./Imagenes/Logo.png").convert("RGB")
st.image(Logo, width=700)

#----- Renderizado del Texto --------------------------------------
st.title("💻 Laboratorio de Desarrollo de Soluciones Tecnológicas ⚙️")
st.subheader(":blue[Implementación de Código de Análisis en Streamlit]")
st.caption("🌎️ **Ingenierías ITESO - Departamento de Electrónica, Sistemas e Informática (DESI)**")

tab_senales, tab_imagenes, tab_info = st.tabs(
    ["📈 Análisis de Señales", "🖼️ Análisis de Imágenes", "ℹ️ Acerca del Dashboard"])


# ===================================================================
# TABULACIÓN NÚMERO 1 — ANÁLISIS DE SEÑALES
# ===================================================================
with tab_senales:
    # Dos columnas: controles a la izquierda (más angosta), gráficas a la
    # derecha (más ancha). El "gap" agrega separación entre ambas.
    col_ctrl, col_plot = st.columns([1, 2.3], gap="large")

    with col_ctrl:
        st.subheader("Parámetros de la Señal")

        tipo_senal = st.selectbox(
            "Tipo de Señal Base",
            ["Senoidal", "Cosenoidal", "Cuadrada", "Diente de Sierra", "Chirp (barrido)"])

        col_a, col_b = st.columns(2)
        with col_a:
            frecuencia = st.slider("Frecuencia (Hz)", 1, 100, 5)
        with col_b:
            amplitud = st.slider("Amplitud", 0.1, 5.0, 1.0, step=0.1)

        # Frecuencia de muestreo: cuántas muestras por segundo se toman
        # de la señal continua. Debe ser al menos el doble de la
        # frecuencia máxima de la señal (criterio de Nyquist) para
        # poder reconstruirla sin distorsión (aliasing).
        fs = st.select_slider(
            "Frecuencia de Muestreo (Hz)",
            options=[8, 16, 32, 64, 128, 256, 512, 1024],
            value=256)
        duracion = st.slider("Duración (s)", 0.5, 5.0, 1.0, step=0.5)
        st.divider()

        st.subheader("Ruido y Filtrado")

        ruido_std = st.slider("Nivel de Ruido Gaussiano", 0.0, 1.0, 0.0, step=0.05)

        aplicar_filtro = st.checkbox("Aplicar Filtro Pasa-Bajas (Butterworth)")
        if aplicar_filtro:
            fc = st.slider("Frecuencia de Corte (Hz)", 1, int(fs / 2) - 1, 10)
            orden = st.slider("Orden del Filtro", 1, 8, 4)

    # --- Generación de la señal ---
    # Vector de tiempo: "duracion*fs" muestras equiespaciadas entre 0 y
    # "duracion" segundos. endpoint=False evita repetir el primer punto
    # del siguiente ciclo (importante para señales periódicas).
    t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)

    if tipo_senal == "Senoidal":
        x = amplitud * np.sin(2 * np.pi * frecuencia * t)
    elif tipo_senal == "Cosenoidal":
        x = amplitud * np.cos(2 * np.pi * frecuencia * t)
    elif tipo_senal == "Cuadrada":
        x = amplitud * sp_signal.square(2 * np.pi * frecuencia * t)
    elif tipo_senal == "Diente de Sierra":
        x = amplitud * sp_signal.sawtooth(2 * np.pi * frecuencia * t)
    else:  # Chirp
        x = amplitud * sp_signal.chirp(t, f0=1, f1=frecuencia, t1=duracion, method="linear")

    # Ruido aditivo gaussiano (blanco): simula interferencia/ruido de
    # instrumentación. Se usa una semilla fija (rng) para que el ruido
    # no cambie en cada recarga de la app, facilitando la comparación.
    if ruido_std > 0:
        rng = np.random.default_rng(42)
        x = x + rng.normal(0, ruido_std, size=x.shape)

    x_filtrada = None
    if aplicar_filtro:
        # Filtro Butterworth pasa-bajas: deja pasar frecuencias por
        # debajo de "fc" y atenúa las superiores. "sos" (second-order
        # sections) es la forma numéricamente más estable de aplicar
        # el filtro. sosfiltfilt aplica el filtro dos veces (adelante
        # y atrás) para eliminar el desfase que introduciría un solo pase.
        sos = sp_signal.butter(orden, fc, btype="low", fs=fs, output="sos")
        x_filtrada = sp_signal.sosfiltfilt(sos, x)

    with col_plot:
        st.subheader("Dominio del Tiempo")
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=t, y=x, mode="lines", name="Señal Original",
            line=dict(color="#94A3B8", width=1.5)))
        if x_filtrada is not None:
            fig_t.add_trace(go.Scatter(
                x=t, y=x_filtrada, mode="lines", name="Señal Filtrada",
                line=dict(color="#25B3EB", width=2)))
        fig_t = layout_base(fig_t, "Tiempo (s)", "Amplitud")
        st.plotly_chart(fig_t, use_container_width=True, theme=None)

        st.subheader("Dominio de la Frecuencia (FFT)")
        # Si hay señal filtrada, analizamos esa; si no, la original.
        senal_para_fft = x_filtrada if x_filtrada is not None else x
        N = len(senal_para_fft)
        # FFT: descompone la señal en sus componentes de frecuencia.
        X = fft(senal_para_fft)
        # fftfreq calcula qué frecuencia (Hz) corresponde a cada índice
        # del resultado de la FFT, dado el número de muestras y fs.
        freqs = fftfreq(N, 1 / fs)
        # La FFT de una señal real es simétrica; solo interesa la
        # primera mitad del espectro (frecuencias positivas).
        mitad = N // 2

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(
            # Se normaliza la magnitud por (2/N) para que la amplitud
            # del espectro coincida con la amplitud real de la señal
            # en el dominio del tiempo (factor 2 porque descartamos la
            # mitad negativa del espectro).
            x=freqs[:mitad], y=(2.0 / N) * np.abs(X[:mitad]),
            mode="lines", fill="tozeroy",
            line=dict(color="#16A34A", width=1.8)))
        fig_f = layout_base(fig_f, "Frecuencia (Hz)", "|X(f)|")
        st.plotly_chart(fig_f, use_container_width=True, theme=None)

        # Métricas para reforzar los conceptos de muestreo.
        m1, m2, m3 = st.columns(3)
        m1.metric("📈 Frecuencia de Muestreo:", f"{fs} Hz")
        m2.metric("📊 Frecuencia de Nyquist:", f"{fs/2:.0f} Hz")
        m3.metric("#️⃣ Número de Muestras:", f"{N}")


# ===================================================================
# TABULACIÓN NÚMERO 2 — ANÁLISIS DE IMÁGENES
# ===================================================================
with tab_imagenes:
    col_ctrl_img, col_plot_img = st.columns([1, 2.3], gap="large")

    with col_ctrl_img:
        st.subheader("Imagen de Entrada")
        archivo = st.file_uploader("Sube una Imagen", type=["png", "jpg", "jpeg", "bmp"])
        usar_ejemplo = st.checkbox("Usar imagen de ejemplo", value=archivo is None)

        st.divider()
        st.subheader("Operación a Aplicar")
        operacion = st.selectbox(
            "Selecciona el procesamiento",
            ["Escala de Grises",
             "Desenfoque (Gaussiano)",
             "Detección de Bordes (Sobel)",
             "Detección de Bordes (Canny)",
             "Ecualización de Histograma",
             "Umbralización"])

        if operacion == "Desenfoque (Gaussiano)":
            sigma_blur = st.slider("Sigma del Desenfoque", 0.5, 10.0, 2.0, step=0.5)
        if operacion == "Detección de Bordes (Canny)":
            sigma_canny = st.slider("Sigma (Canny)", 0.5, 5.0, 1.5, step=0.5)

    # --- Carga de imagen ---
    if archivo is not None and not usar_ejemplo:
        img = np.array(Image.open(archivo).convert("RGB"))
    elif usar_ejemplo:
        img = Image.open("./Imagenes/Lily.jpg").convert("RGB")
    else:
        img = None

    with col_plot_img:
        if img is None:
            st.info("Sube una imagen o activa la casilla de imagen de ejemplo para comenzar.")
        else:
            gris = color.rgb2gray(img)
            if operacion == "Escala de Grises":
                resultado = gris
            elif operacion == "Desenfoque (Gaussiano)":
                # Suaviza la imagen convolucionándola con una campana de
                # Gauss; sigma controla qué tan "ancho" es el suavizado.
                resultado = filters.gaussian(gris, sigma=sigma_blur)
            elif operacion == "Detección de Bordes (Sobel)":
                # Calcula el gradiente de intensidad en cada píxel;
                # los bordes son zonas de cambio brusco de intensidad.
                resultado = filters.sobel(gris)
            elif operacion == "Detección de Bordes (Canny)":
                # Detector de bordes más elaborado: suaviza, calcula
                # gradiente, adelgaza bordes y aplica umbral por
                # histéresis. Devuelve una imagen binaria (borde/no borde).
                resultado = feature.canny(gris, sigma=sigma_canny).astype(float)
            elif operacion == "Ecualización de Histograma":
                # Redistribuye las intensidades para que el histograma
                # sea lo más uniforme posible, mejorando el contraste
                # global de la imagen.
                resultado = exposure.equalize_hist(gris)
            else:  # Umbralización
                # Calcula automáticamente el umbral que mejor separa la
                # imagen en dos clases (fondo/objeto) y binariza con él.
                umbral = filters.threshold_otsu(gris)
                resultado = (gris > umbral).astype(float)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Imagen Original:**")
                st.image(img, width=500)
            with c2:
                st.markdown(f"**Resultado - {operacion}:**")
                st.image(resultado, width=500)
            st.divider()

            c3, c4 = st.columns(2)
            with c3:
                st.markdown("**Imagen Original:**")
                fig_orig = px.imshow(img)
                fig_orig.update_layout(
                    template="plotly", height=500,
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False)
                fig_orig.update_xaxes(visible=False, constrain="domain")
                fig_orig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
                st.plotly_chart(fig_orig, use_container_width=True)
            with c4:
                st.markdown(f"**Resultado - {operacion}:**")
                fig_res = px.imshow(resultado, color_continuous_scale="gray", zmin=0, zmax=1)
                fig_res.update_traces(zsmooth="best")
                fig_res.update_layout(
                    template="plotly", height=500,
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False)
                fig_res.update_xaxes(visible=False, constrain="domain")
                fig_res.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
                st.plotly_chart(fig_res, use_container_width=True)
            st.divider()

            st.subheader("Histograma de Intensidades")
            # El histograma siempre se calcula sobre la imagen en gris
            # original (no sobre "resultado"), para servir como
            # referencia constante sin importar la operación elegida.
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=gris.ravel(), nbinsx=256, marker_color="#64748B"))
            fig_hist = layout_base(fig_hist, "Intensidad", "Frecuencia", height=250)
            st.plotly_chart(fig_hist, use_container_width=True, theme=None)


# ===================================================================
# TABULACIÓN NÚMERO 3 — ACERCA DEL DASHBOARD
# ===================================================================
with tab_info:
    st.subheader("Sobre este Dashboard:")
    Banner = Image.open("./Imagenes/Banner.png").convert("RGB")
    st.image(Banner, width=500)
    st.markdown(
        """
        Este Dashboard muestra el uso de la librería **Streamlit** para el
        procesamiento de señales e imágenes y forma parte de los temas de
        introducción del curso "***Laboratorio de Desarrollo de Soluciones
        Tecnológicas***" que aborda el tema "*Análisis Multimedia basado en 
        Inteligencia Artificial para Soluciones Operativas Empresariales*" 
        para alumnos de las Ingenierías del *Instituto Tecnológico y de
        Estudios Superiores de Occidente (ITESO)*.
        
        **Interactividad:** 
        Todos los gráficos usan la librería **Plotly**, 
        debido a ello se puede hacer zoom a los gráficos, desplazamiento (*pan*), 
        pasar el cursor para ver valores exactos (*hoover*), así como realizar la
        descarga de cada gráfica en formato PNG desde su propia barra de herramientas.

        **Contenido:**
        - Generación de señales (senoidal, cosenoidal, cuadrada, diente de sierra, chirp).
        - Adición de ruido gaussiano.
        - Filtrado digital (Butterworth pasa-bajas).
        - Análisis espectral mediante Transformada Rápida de Fourier (FFT).
        - Operaciones clásicas de procesamiento de imágenes (detección de bordes, desenfoque,
          ecualización, umbralización).
          
        ***Nota:*** Los temas de cada uno de los contenidos se revisarán a detalle en
        posteriores sesiones del curso, por el momento sólamente son de carácter ilustrativo.

        **Librerías utilizadas:** `streamlit`, `numpy`, `scipy`, `plotly`,
        `pillow`, `scikit-image`.
        
        :blue[**Desarrollado por: Dr. Iván Esteban Villalón Turrubiates.
        Departamento de Electrónica, Sistemas e Informática (DESI).
        *villalon@iteso.mx***]
        """)


# ===================================================================
# PIE DE PÁGINA FIJO CON DATOS DEL CURSO
# ===================================================================
st.markdown(f"""
    <div class="footer-curso">
        💻 Dr. Iván Esteban Villalón Turrubiates · Laboratorio de
        Desarrollo de Soluciones Tecnológicas · ITESO ⚙️
    </div>""", unsafe_allow_html=True)


# ===================================================================
# .: FINAL DEL CÓDIGO :.
# ===================================================================
