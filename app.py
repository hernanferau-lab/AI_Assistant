import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuracion de entorno ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI-Powered Legal Assistant", page_icon="⚖️")

if not openai_api_key:
    st.error(
        "🔐 No se ha configurado OPENAI_API_KEY. "
        "Agregala en los Secrets de Streamlit Cloud."
    )
    st.stop()

st.title("⚖️ AI-Powered Legal Assistant")
st.markdown(
    "**Prototipo de apoyo al analisis de denuncias de acoso laboral "
    "(protocolo simulado, basado en la Ley Karin 21.643).**"
)
st.info(
    "Demo / proof of concept. Usa un unico documento simulado y corto, sin datos reales. "
    "No reemplaza asesoria profesional ni legal."
)

# --- Limites de uso del demo publico (protegen el costo de la API) ---
MAX_PREGUNTAS_POR_SESION = 10
MAX_CARACTERES = 300
if "preguntas_usadas" not in st.session_state:
    st.session_state.preguntas_usadas = 0

# --- Construccion del pipeline RAG (se cachea: se arma una sola vez) ---
@st.cache_resource
def build_qa_chain():
    # 1. Cargar el documento fuente
    loader = TextLoader("protocolo.txt", encoding="utf-8")
    documents = loader.load()

    # 2. CHUNKING: partir el documento en fragmentos manejables.
    #    Esto es lo que faltaba antes (se indexaba el doc completo como un solo bloque).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # tamano aprox. de cada fragmento (en caracteres)
        chunk_overlap=80,   # solape entre fragmentos para no cortar ideas a la mitad
    )
    chunks = splitter.split_documents(documents)

    # 3. EMBEDDINGS + INDICE VECTORIAL: convertir cada fragmento en vectores
    #    y guardarlos en FAISS para poder buscar por significado, no por palabra exacta.
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. RETRIEVER: trae solo los 3 fragmentos mas relevantes para cada pregunta.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 5. GROUNDING: el prompt obliga al modelo a responder SOLO con el contexto
    #    recuperado. Si no esta en el documento, lo dice. Reduce las "alucinaciones".
    prompt = PromptTemplate(
        template=(
            "Eres un asistente que responde EXCLUSIVAMENTE con la informacion del "
            "contexto entregado. Si la respuesta no esta en el contexto, di "
            "claramente que el documento no lo especifica. No inventes. "
            "Responde en el mismo idioma de la pregunta.\n\n"
            "Contexto:\n{context}\n\n"
            "Pregunta: {question}\n\n"
            "Respuesta:"
        ),
        input_variables=["context", "question"],
    )

    llm = ChatOpenAI(api_key=openai_api_key, temperature=0, model="gpt-4o-mini")

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,   # devuelve tambien las fuentes usadas
        chain_type_kwargs={"prompt": prompt},
    )


try:
    qa_chain = build_qa_chain()
except Exception:
    st.warning(
        "El demo no esta disponible en este momento (limite del servicio de IA). "
        "Puedes revisar el codigo y la arquitectura en GitHub."
    )
    st.stop()

# --- Interfaz ---
pregunta = st.text_input("¿En que te puedo ayudar hoy?", max_chars=MAX_CARACTERES)

if pregunta and st.session_state.preguntas_usadas >= MAX_PREGUNTAS_POR_SESION:
    st.warning("Llegaste al limite de preguntas de este demo. Recarga la pagina mas tarde.")
    st.stop()

if pregunta:
    st.session_state.preguntas_usadas += 1
    try:
        with st.spinner("Buscando en el documento..."):
            resultado = qa_chain.invoke({"query": pregunta})
    except Exception:
        st.error("No se pudo generar una respuesta en este momento. Intenta de nuevo en unos minutos.")
        st.stop()

    st.markdown("### Respuesta")
    st.write(resultado["result"])

    # SOURCE REFERENCING: mostrar de que fragmentos salio la respuesta.
    # Esto da trazabilidad y confianza: el usuario puede verificar.
    with st.expander("📄 Fragmentos del documento usados para responder"):
        for i, doc in enumerate(resultado["source_documents"], start=1):
            st.markdown(f"**Fragmento {i}:**")
            st.write(doc.page_content)
