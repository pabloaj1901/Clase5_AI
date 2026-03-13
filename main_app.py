import streamlit as st
import time
import numpy as np
import PyPDF2
import io
import base64
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Playground · EAFIT MCD",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark sci-fi theme */
.stApp {
    background: #0a0e1a;
    color: #e0e6f0;
}

.stSidebar {
    background: #0f1525 !important;
    border-right: 1px solid #1e2d4a;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
}

.metric-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}

.col-header {
    text-align: center;
    padding: 10px;
    border-radius: 6px;
    font-weight: 800;
    font-size: 1rem;
    margin-bottom: 10px;
    letter-spacing: 1px;
}

.col-llm    { background: #1a1033; border: 1px solid #4a2080; color: #c084fc; }
.col-rag    { background: #0f2a1a; border: 1px solid #1a6640; color: #4ade80; }
.col-opt    { background: #1a1f00; border: 1px solid #5a6000; color: #facc15; }

.answer-box {
    background: #0d1117;
    border-radius: 8px;
    padding: 16px;
    min-height: 200px;
    font-size: 0.9rem;
    line-height: 1.6;
    border: 1px solid #1e2d3a;
    white-space: pre-wrap;
}

.stButton > button {
    background: linear-gradient(135deg, #1e40af, #7c3aed);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
    width: 100%;
    padding: 12px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #9333ea);
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    margin: 2px;
}
.badge-time { background: #1e3a5f; color: #7dd3fc; }
.badge-sim  { background: #1a3a1a; color: #86efac; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='color:#7dd3fc; font-size:2.2rem; margin-bottom:0;'>🧪 RAG Playground</h1>
<p style='color:#64748b; font-family: Space Mono, monospace; font-size:0.85rem; margin-top:4px;'>
EAFIT · Maestría Ciencia de los Datos · Taller 03 — RAG vs. LLM
</p>
<hr style='border-color:#1e2d4a; margin: 16px 0;'>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Hiperparámetros")

    groq_api_key = st.text_input("🔑 Groq API Key", type="password",
                                  placeholder="gsk_...")

    st.markdown("---")
    model_name = st.selectbox(
        "🤖 Modelo LLM",
        ["llama3-70b-8192", "mixtral-8x7b-32768", "llama3-8b-8192"],
        help="Llama 3 70B o Mixtral 8x7B"
    )

    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.2, 0.05)
    chunk_size  = st.slider("📦 Chunk Size (tokens)", 20, 2000, 500, 50)
    chunk_overlap = st.slider("🔗 Chunk Overlap", 0, 200, 50, 10)
    top_k       = st.slider("🔍 Top-K fragmentos", 1, 10, 3)

    st.markdown("---")
    st.markdown("### 📎 Documento")
    uploaded_file = st.file_uploader(
        "Sube PDF o Imagen (JPG/PNG)",
        type=["pdf", "jpg", "jpeg", "png"]
    )

    st.markdown("---")
    system_prompt_no_se = st.checkbox(
        '🛑 Forzar "No sé" si no hay contexto',
        value=False,
        help="Inyecta un system prompt que obliga al modelo a decir 'No sé' cuando la respuesta no está en el RAG."
    )

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_image(file_bytes: bytes, api_key: str) -> str:
    client = Groq(api_key=api_key)
    b64 = base64.b64encode(file_bytes).decode()
    resp = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",
                 "text": "Extrae TODO el texto de esta imagen con la mayor fidelidad posible. Solo devuelve el texto, sin comentarios."}
            ]
        }],
        max_tokens=4096,
    )
    return resp.choices[0].message.content

def build_vector_store(text: str, chunk_size: int, overlap: int, embeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return None, []
    store = FAISS.from_texts(chunks, embeddings)
    return store, chunks

def cosine_sim_score(query: str, chunks: list[str], embeddings) -> float:
    if not chunks:
        return 0.0
    q_emb  = embeddings.embed_query(query)
    c_embs = embeddings.embed_documents(chunks)
    sims   = cosine_similarity([q_emb], c_embs)[0]
    return float(np.max(sims))

MAX_CONTEXT_CHARS = 6000

def truncate_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    return text[:max_chars] + "\n\n[... contexto truncado ...]" if len(text) > max_chars else text

def call_llm(client, model, temperature, messages) -> tuple[str, float]:
    safe_messages = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str) and len(content) > 12000:
            content = content[:12000] + "\n[truncado]"
        safe_messages.append({"role": m["role"], "content": content})
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=safe_messages,
        temperature=temperature,
        max_tokens=512,
    )
    return resp.choices[0].message.content, round(time.time() - t0, 2)

def retrieve_context(store, query: str, k: int) -> tuple[str, list[str]]:
    docs = store.similarity_search(query, k=k)
    chunks = [d.page_content for d in docs]
    context = truncate_context("\n\n---\n\n".join(chunks))
    return context, chunks

# ─────────────────────────────────────────────
#  DOCUMENT PROCESSING
# ─────────────────────────────────────────────

doc_text = ""
vector_store_default  = None
vector_store_optimized = None
embeddings = None

if uploaded_file:
    if not groq_api_key:
        st.warning("⚠️ Ingresa tu Groq API Key en el sidebar para continuar.")
    else:
        with st.spinner("📖 Procesando documento…"):
            file_bytes = uploaded_file.read()
            ext = uploaded_file.name.split(".")[-1].lower()

            if ext == "pdf":
                doc_text = extract_text_from_pdf(file_bytes)
            else:
                doc_text = extract_text_from_image(file_bytes, groq_api_key)

            embeddings = load_embeddings()

            # Default RAG: chunk_size=500, overlap=50, k=3
            vector_store_default, _ = build_vector_store(doc_text, 500, 50, embeddings)
            # Optimized RAG: uses sidebar params
            vector_store_optimized, _ = build_vector_store(doc_text, chunk_size, chunk_overlap, embeddings)

        st.success(f"✅ Documento cargado — {len(doc_text)} caracteres extraídos")
        with st.expander("📄 Ver texto extraído"):
            st.text(doc_text[:3000] + ("…" if len(doc_text) > 3000 else ""))

# ─────────────────────────────────────────────
#  QUERY INPUT
# ─────────────────────────────────────────────
st.markdown("### 💬 Pregunta")
query = st.text_input("", placeholder="¿Qué quieres saber sobre el documento?")
run   = st.button("🚀 Comparar respuestas", disabled=(not query or not doc_text or not groq_api_key))

# ─────────────────────────────────────────────
#  MAIN EXPERIMENT
# ─────────────────────────────────────────────
if run:
    client = Groq(api_key=groq_api_key)

    sys_no_se = (
        "IMPORTANTE: Si la respuesta a la pregunta del usuario NO se encuentra "
        "en el contexto proporcionado, responde ÚNICAMENTE con: 'No sé, la información "
        "no está disponible en el documento.' No inventes ni supongas nada."
    ) if system_prompt_no_se else None

    # ── Column layout ──
    c1, c2, c3 = st.columns(3)

    # ── 1. LLM Simple ──
    with c1:
        st.markdown('<div class="col-header col-llm">⚡ LLM Simple (Zero-shot)</div>', unsafe_allow_html=True)
        try:
            with st.spinner("Generando…"):
                msgs_simple = []
                if sys_no_se:
                    msgs_simple.append({"role": "system", "content": sys_no_se})
                msgs_simple.append({"role": "user", "content": query[:800]})
                ans1, t1 = call_llm(client, model_name, temperature, msgs_simple)
            st.markdown(f'<div class="answer-box">{ans1}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="badge badge-time">⏱ {t1}s</span>'
                f'<span class="badge badge-sim">~ Sin RAG</span>',
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"❌ Error LLM Simple: {str(e)[:300]}")

    # ── 2. RAG Default ──
    with c2:
        st.markdown('<div class="col-header col-rag">🧩 RAG Estándar (default)</div>', unsafe_allow_html=True)
        if vector_store_default:
            try:
                with st.spinner("Recuperando…"):
                    ctx2, chunks2 = retrieve_context(vector_store_default, query, 3)
                    sim2 = cosine_sim_score(query, chunks2, embeddings)
                    sys2_content = "Usa SOLO el contexto proporcionado para responder."
                    if sys_no_se:
                        sys2_content = sys_no_se
                    msgs2 = [
                        {"role": "system", "content": sys2_content},
                        {"role": "user", "content": f"Contexto:\n{ctx2}\n\nPregunta: {query[:800]}"}
                    ]
                    ans2, t2 = call_llm(client, model_name, temperature, msgs2)
                st.markdown(f'<div class="answer-box">{ans2}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="badge badge-time">⏱ {t2}s</span>'
                    f'<span class="badge badge-sim">coseno: {sim2:.3f}</span>',
                    unsafe_allow_html=True
                )
                with st.expander("📎 Fragmentos recuperados (default)"):
                    for i, ch in enumerate(chunks2):
                        st.markdown(f"**Chunk {i+1}:** {ch[:300]}…")
            except Exception as e:
                st.error(f"❌ Error RAG Default: {str(e)[:300]}")
        else:
            st.warning("No hay vector store disponible.")

    # ── 3. RAG Optimizado ──
    with c3:
        st.markdown('<div class="col-header col-opt">🎯 RAG Optimizado (sidebar)</div>', unsafe_allow_html=True)
        if vector_store_optimized:
            try:
                with st.spinner("Recuperando…"):
                    ctx3, chunks3 = retrieve_context(vector_store_optimized, query, top_k)
                    sim3 = cosine_sim_score(query, chunks3, embeddings)
                    sys3_content = "Usa SOLO el contexto proporcionado para responder con precisión."
                    if sys_no_se:
                        sys3_content = sys_no_se
                    msgs3 = [
                        {"role": "system", "content": sys3_content},
                        {"role": "user", "content": f"Contexto:\n{ctx3}\n\nPregunta: {query[:800]}"}
                    ]
                    ans3, t3 = call_llm(client, model_name, temperature, msgs3)
                st.markdown(f'<div class="answer-box">{ans3}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="badge badge-time">⏱ {t3}s</span>'
                    f'<span class="badge badge-sim">coseno: {sim3:.3f}</span>',
                    unsafe_allow_html=True
                )
                with st.expander("📎 Fragmentos recuperados (optimizado)"):
                    for i, ch in enumerate(chunks3):
                        st.markdown(f"**Chunk {i+1}:** {ch[:300]}…")
            except Exception as e:
                st.error(f"❌ Error RAG Optimizado: {str(e)[:300]}")
        else:
            st.warning("No hay vector store disponible.")

# ─────────────────────────────────────────────
#  FASE 4 — ANÁLISIS CONCEPTUAL
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📚 Fase 4 — Análisis de Conceptos")

with st.expander("1. 🤥 Alucinación: ¿Cuándo inventa el LLM?"):
    st.markdown("""
Un **LLM zero-shot** genera texto basado únicamente en su distribución aprendida durante preentrenamiento.
Cuando se le pregunta sobre un documento que **nunca ha visto**, el modelo completa la secuencia con
información *plausible pero falsa*, ya que no tiene acceso al contenido real.

**Ejemplo típico:** si el documento menciona métricas específicas (un porcentaje, un nombre propio, 
una fecha), el LLM simple puede inventar valores similares pero incorrectos. El RAG mitiga esto 
porque el fragmento recuperado actúa como **ancla factual** en el prompt.
    """)

with st.expander("2. 🛡️ Inyección de Contexto: El System Prompt 'No sé'"):
    st.markdown("""
Al agregar un **System Prompt restrictivo**, obligamos al modelo a operar solo dentro del contexto 
recuperado. Esto tiene dos efectos:

- **Sin RAG:** el modelo responde "No sé" incluso cuando podría inferir algo razonable → útil para 
  aplicaciones críticas donde la precisión es primordial.
- **Con RAG:** el modelo responde solo si el fragmento recuperado contiene la respuesta, y admite 
  ignorancia en caso contrario.

Esta técnica se activa con el checkbox **"Forzar No sé"** en el sidebar. 
Cambia el *comportamiento epistémico* del sistema sin cambiar los parámetros del modelo.
    """)

with st.expander("3. ⚖️ Fine-Tuning vs RAG: ¿Por qué RAG gana aquí?"):
    st.markdown("""
| Criterio | Fine-Tuning | RAG |
|---|---|---|
| **Costo** | Alto (GPU horas, datos etiquetados) | Bajo (embedding + FAISS) |
| **Velocidad de actualización** | Lento (requiere reentrenamiento) | Instantáneo (solo reindexar) |
| **Conocimiento nuevo** | Estático post-entrenamiento | Dinámico, se actualiza con el doc |
| **Riesgo de alucinación** | Alto si datos son escasos | Bajo con retrieval correcto |
| **Interpretabilidad** | Baja (pesos del modelo) | Alta (se puede ver el chunk recuperado) |

Para este taller, el documento es nuevo para el modelo. Fine-tuning requeriría **cientos de pares 
pregunta-respuesta** sobre ese documento y días de entrenamiento. RAG lo resuelve en segundos.
    """)

with st.expander("4. 🤖 Transformer vs No-Transformer en Embeddings"):
    st.markdown("""
Los embeddings generados por `sentence-transformers/all-MiniLM-L6-v2` **sí dependen de la 
arquitectura Transformer**:

- El modelo usa **atención multi-cabezal** para capturar relaciones contextuales entre tokens.
- A diferencia de Word2Vec o GloVe (no-Transformer), produce embeddings **contextuales**: 
  la misma palabra tiene representaciones distintas según su contexto.
- La capa final del encoder produce un vector de dimensión fija (384d en MiniLM) mediante 
  **mean pooling** sobre los token embeddings.

Modelos no-Transformer como TF-IDF o BM25 también pueden usarse para retrieval, pero capturan 
solo frecuencia léxica, no semántica. Los Transformers capturan **similitud semántica**, 
lo que los hace superiores para RAG.
    """)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e2d4a; margin-top:40px;'>
<p style='text-align:center; color:#334155; font-family: Space Mono, monospace; font-size:0.75rem;'>
EAFIT · Maestría en Ciencia de los Datos · Prof. Jorge Iván Padilla Buriticá · Taller 03
</p>
""", unsafe_allow_html=True)
