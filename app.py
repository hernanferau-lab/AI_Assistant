import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Cargar variables de entorno
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verifica que la clave esté presente
if not openai_api_key:
    st.error("🔐 No se ha configurado la clave OPENAI_API_KEY. Agrega tu clave en Secrets de Streamlit Cloud.")
    st.stop()

# Título de la App
st.title("⚖️  AI-Powered Legal Assistant")
st.markdown("**Prototype designed to improve legal case analysis of workplace harasment**")

# Carga y embebe el documento
@st.cache_resource
def load_qa_chain():
    loader = TextLoader("protocolo.txt")
    documents = loader.load()
    texts = [doc.page_content for doc in documents]
    embedding = OpenAIEmbeddings(api_key=openai_api_key)
    db = FAISS.from_texts(texts, embedding)
    retriever = db.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(api_key=openai_api_key, temperature=0),
        retriever=retriever
    )

qa_chain = load_qa_chain()

# UI de interacción
pregunta = st.text_input("¿How can I help you today?")
if pregunta:
    respuesta = qa_chain.run(pregunta)
    st.write("Answer:", respuesta)
