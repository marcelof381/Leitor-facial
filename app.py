"""
Sistema de reconhecimento facial AO VIVO com Streamlit + OpenCV (YuNet + SFace).

- Aba "Ao vivo": vídeo da webcam com caixa, nome e status em tempo real.
- Aba "Cadastrar": foto (câmera ou arquivo) + nome.
- Aba "Cadastros": lista e remove pessoas.

Não depende de dlib/face_recognition. Os modelos ONNX são baixados na primeira execução.

Rodar:  streamlit run app.py
"""
import json
import threading
import time
import urllib.request
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "imagens"
DB_PATH = DATA_DIR / "rostos.json"

DETECTOR_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
MODEL_URLS = {
    DETECTOR_PATH: "https://github.com/opencv/opencv_zoo/raw/main/models/"
                   "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    RECOGNIZER_PATH: "https://github.com/opencv/opencv_zoo/raw/main/models/"
                     "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

COSINE_THRESHOLD_PADRAO = 0.363  # valor recomendado pelo autor do SFace
ANALISAR_A_CADA_N_FRAMES = 2     # reduz carga de CPU; os frames intermediários reusam a última análise
RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

for d in (MODELS_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Modelos (funções sem dependência do Streamlit: são usadas em thread de vídeo)
# ----------------------------------------------------------------------------
def baixar_modelos():
    for caminho, url in MODEL_URLS.items():
        if not caminho.exists() or caminho.stat().st_size < 100_000:
            urllib.request.urlretrieve(url, caminho)


def criar_modelos():
    detector = cv2.FaceDetectorYN.create(
        str(DETECTOR_PATH), "", (320, 320), score_threshold=0.8, nms_threshold=0.3
    )
    recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_PATH), "")
    return detector, recognizer


@st.cache_resource
def garantir_modelos():
    baixar_modelos()
    return criar_modelos()  # instância usada nas abas de cadastro (thread principal)


# ----------------------------------------------------------------------------
# Banco de dados (JSON simples: nome -> lista de embeddings)
# ----------------------------------------------------------------------------
def carregar_db() -> dict:
    if DB_PATH.exists():
        return json.loads(DB_PATH.read_text(encoding="utf-8"))
    return {}


def salvar_db(db: dict):
    DB_PATH.write_text(json.dumps(db, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------
# Utilidades de visão computacional (cadastro)
# ----------------------------------------------------------------------------
def decodificar(arquivo) -> np.ndarray:
    dados = np.frombuffer(arquivo.getvalue(), np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def detectar_rostos(img: np.ndarray) -> np.ndarray:
    detector, _ = garantir_modelos()
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, rostos = detector.detect(img)
    return rostos if rostos is not None else np.empty((0, 15))


def extrair_embedding(img: np.ndarray, rosto: np.ndarray) -> np.ndarray:
    _, recognizer = garantir_modelos()
    return recognizer.feature(recognizer.alignCrop(img, rosto)).flatten()


def similaridade(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def identificar(embedding: np.ndarray, db: dict, limiar: float):
    melhor_nome, melhor_score = None, -1.0
    for nome, embeddings in db.items():
        for emb in embeddings:
            s = similaridade(embedding, np.array(emb, dtype=np.float32))
            if s > melhor_score:
                melhor_nome, melhor_score = nome, s
    if melhor_score >= limiar:
        return melhor_nome, melhor_score
    return None, melhor_score


# ----------------------------------------------------------------------------
# Desenho (PIL, para suportar acentos nos rótulos)
# ----------------------------------------------------------------------------
def _carregar_fonte(tamanho=20):
    for nome in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            continue
    return ImageFont.load_default()


FONTE = _carregar_fonte()
VERDE, VERMELHO = (0, 170, 0), (210, 0, 0)


def desenhar_resultados(img_bgr: np.ndarray, resultados) -> np.ndarray:
    """resultados: lista de ((x, y, w, h), nome | None, score)."""
    if not resultados:
        return img_bgr
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for (x, y, w, h), nome, _score in resultados:
        cor = VERDE if nome else VERMELHO
        texto = f"{nome} - AUTORIZADO" if nome else "NÃO RECONHECIDA"
        d.rectangle([x, y, x + w, y + h], outline=cor, width=3)
        l, t, r, b = d.textbbox((0, 0), texto, font=FONTE)
        tw, th = r - l, b - t
        topo = max(y - th - 12, 0)
        d.rectangle([x, topo, x + tw + 12, topo + th + 10], fill=cor)
        d.text((x + 6, topo + 3), texto, font=FONTE, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------
# Processador de vídeo ao vivo
# ----------------------------------------------------------------------------
class ReconhecedorAoVivo(VideoProcessorBase):
    def __init__(self):
        self.detector, self.recognizer = criar_modelos()
        self.lock = threading.Lock()
        self.nomes: list[str] = []
        self.matriz: np.ndarray | None = None  # embeddings normalizados (N x 128)
        self.limiar = COSINE_THRESHOLD_PADRAO
        self.resultado: list = []              # [(nome | None, score)] para a interface
        self._ultimos: list = []               # [((x,y,w,h), nome, score)] para desenho
        self._contador = 0

    # chamado pela thread principal do Streamlit
    def atualizar(self, db: dict, limiar: float):
        nomes, vetores = [], []
        for nome, embs in db.items():
            for e in embs:
                v = np.array(e, dtype=np.float32)
                vetores.append(v / np.linalg.norm(v))
                nomes.append(nome)
        matriz = np.stack(vetores) if vetores else None
        with self.lock:
            self.nomes, self.matriz, self.limiar = nomes, matriz, limiar

    def obter_resultado(self):
        with self.lock:
            return list(self.resultado)

    def _analisar(self, img):
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, rostos = self.detector.detect(img)
        saida = []
        if rostos is None:
            return saida
        with self.lock:
            nomes, matriz, limiar = self.nomes, self.matriz, self.limiar
        for rosto in rostos:
            emb = self.recognizer.feature(self.recognizer.alignCrop(img, rosto)).flatten()
            emb = emb / np.linalg.norm(emb)
            nome, score = None, 0.0
            if matriz is not None:
                scores = matriz @ emb
                i = int(np.argmax(scores))
                score = float(scores[i])
                if score >= limiar:
                    nome = nomes[i]
            saida.append((rosto[:4].astype(int), nome, score))
        return saida

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._contador += 1
        if self._contador % ANALISAR_A_CADA_N_FRAMES == 1:
            self._ultimos = self._analisar(img)
            with self.lock:
                self.resultado = [(n, s) for (_, n, s) in self._ultimos]
        return av.VideoFrame.from_ndarray(desenhar_resultados(img, self._ultimos), format="bgr24")


def mostrar_status(placeholder, resultado):
    with placeholder.container():
        if not resultado:
            st.info("🔎 Procurando rosto...")
            return
        for i, (nome, score) in enumerate(resultado, start=1):
            prefixo = f"Rosto #{i}: " if len(resultado) > 1 else ""
            if nome:
                st.success(f"{prefixo}**{nome}** — ✅ AUTORIZADO (similaridade {score:.2f})")
            else:
                st.error(f"{prefixo}⛔ PESSOA NÃO RECONHECIDA (melhor similaridade {max(score, 0):.2f})")


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Reconhecimento Facial", page_icon="🧑", layout="centered")
st.title("🧑 Reconhecimento Facial")

with st.spinner("Carregando modelos (a primeira vez baixa os arquivos)..."):
    garantir_modelos()

db = carregar_db()

with st.sidebar:
    st.header("Configurações")
    limiar = st.slider(
        "Limiar de similaridade", 0.20, 0.70, COSINE_THRESHOLD_PADRAO, 0.01,
        help="Quanto maior, mais rígido (menos falsos positivos, mais falsos negativos).",
    )
    st.metric("Pessoas cadastradas", len(db))

aba_vivo, aba_cadastrar, aba_gerenciar = st.tabs(["🎥 Ao vivo", "➕ Cadastrar", "🗂️ Cadastros"])

# ------------------------------- CADASTRAR ----------------------------------
with aba_cadastrar:
    st.subheader("Cadastrar novo rosto")
    nome_novo = st.text_input("Nome da pessoa", key="nome_novo").strip()
    modo = st.radio("Fonte da imagem", ["Câmera", "Enviar arquivo"], horizontal=True)
    if modo == "Câmera":
        arquivo = st.camera_input("Tire a foto", key="cam_cadastro")
        st.caption("Se a câmera não abrir, pare o vídeo da aba **Ao vivo** ou use **Enviar arquivo**.")
    else:
        arquivo = st.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png"])

    if arquivo is not None:
        img = decodificar(arquivo)
        rostos = detectar_rostos(img)

        if len(rostos) == 0:
            st.warning("Nenhum rosto detectado. Tente com melhor iluminação e de frente.")
        elif len(rostos) > 1:
            st.warning(f"{len(rostos)} rostos detectados. Use uma foto com apenas uma pessoa.")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            rosto = rostos[0]
            preview = desenhar_resultados(img, [(rosto[:4].astype(int), "Rosto detectado", 1.0)])
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))

            if st.button("Salvar cadastro", type="primary", disabled=not nome_novo):
                emb = extrair_embedding(img, rosto)
                db = carregar_db()

                existente, score = identificar(emb, db, limiar)
                if existente and existente.lower() != nome_novo.lower():
                    st.warning(f"Atenção: esse rosto já parece estar cadastrado como "
                               f"**{existente}** (similaridade {score:.2f}). Cadastro salvo mesmo assim.")

                db.setdefault(nome_novo, []).append(emb.tolist())
                salvar_db(db)

                nome_arquivo = "".join(c if c.isalnum() else "_" for c in nome_novo)
                cv2.imwrite(str(IMAGES_DIR / f"{nome_arquivo}_{int(time.time())}.jpg"), img)

                st.success(f"✅ {nome_novo} cadastrado(a)! ({len(db[nome_novo])} foto(s) no total)")
                st.caption("Dica: cadastre 2–3 fotos por pessoa (ângulos/iluminações diferentes) "
                           "para melhorar a precisão.")
            elif not nome_novo:
                st.info("Digite o nome da pessoa para habilitar o salvamento.")

# ------------------------------- GERENCIAR ----------------------------------
with aba_gerenciar:
    st.subheader("Pessoas cadastradas")
    db_atual = carregar_db()
    if not db_atual:
        st.info("Nenhum cadastro ainda.")
    for nome in sorted(db_atual):
        col1, col2 = st.columns([4, 1])
        col1.write(f"**{nome}** — {len(db_atual[nome])} foto(s)")
        if col2.button("Remover", key=f"del_{nome}"):
            del db_atual[nome]
            salvar_db(db_atual)
            st.rerun()

# ------------------------------- AO VIVO ------------------------------------
# IMPORTANTE: esta aba fica por último no script porque contém um laço de
# atualização contínua; ele não pode bloquear a execução das outras abas.
with aba_vivo:
    st.subheader("Reconhecimento ao vivo")
    db_vivo = carregar_db()
    if not db_vivo:
        st.info("Nenhum rosto cadastrado ainda. Use a aba **Cadastrar** primeiro.")

    ctx = webrtc_streamer(
        key="reconhecimento-ao-vivo",
        video_processor_factory=ReconhecedorAoVivo,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )
    status = st.empty()

    if ctx.video_processor:
        ctx.video_processor.atualizar(db_vivo, limiar)
        while ctx.state.playing:
            mostrar_status(status, ctx.video_processor.obter_resultado())
            time.sleep(0.2)
    else:
        status.info("Clique em **START** e permita o acesso à câmera.")