import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Environment ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI-Powered Legal Assistant (Demo)", page_icon="⚖️")

if not openai_api_key:
    st.error("🔐 OPENAI_API_KEY is not set. Add it to the Streamlit Cloud secrets.")
    st.stop()

st.title("⚖️ AI-Powered Legal Assistant")
st.markdown(
    "**Demo of a decision-support assistant for workplace harassment complaints, "
    "based on a simulated procedure inspired by Chile's Ley Karin (Law 21.643).**"
)
st.info(
    "Demo / proof of concept. It answers only from one short, simulated document and holds "
    "no real data. Not a replacement for professional or legal advice."
)
st.caption(
    "The source document is in Spanish. Ask in English or Spanish and the assistant answers "
    "in the language of your question. Try: *How long does the employer have to start an "
    "investigation?* · *What happens when the investigation ends?* · *Is confidentiality required?*"
)

# --- Public demo limits (protect the API budget) ---
MAX_QUESTIONS_PER_SESSION = 10
MAX_CHARACTERS = 300
if "questions_used" not in st.session_state:
    st.session_state.questions_used = 0


# --- RAG pipeline (cached: built once per app start) ---
@st.cache_resource
def build_qa_chain():
    # 1. Load the source document
    loader = TextLoader("protocolo.txt", encoding="utf-8")
    documents = loader.load()

    # 2. CHUNKING: split the document into overlapping fragments
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # approximate fragment size, in characters
        chunk_overlap=80,   # overlap so ideas are not cut in half
    )
    chunks = splitter.split_documents(documents)

    # 3. EMBEDDINGS + VECTOR INDEX: store each fragment as a vector in FAISS,
    #    so search works by meaning, not by exact words.
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. RETRIEVER: return the 3 most relevant fragments for each question
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 5. GROUNDING: the prompt only allows answers from the retrieved context.
    #    If the answer is not in the document, the assistant says so.
    prompt = PromptTemplate(
        template=(
            "You are an assistant that answers ONLY with information from the context provided. "
            "If the answer is not in the context, say clearly that the document does not specify it. "
            "Do not make anything up. Answer in the same language as the question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )

    llm = ChatOpenAI(api_key=openai_api_key, temperature=0, model="gpt-4o-mini")

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,   # also return the fragments used
        chain_type_kwargs={"prompt": prompt},
    )


try:
    qa_chain = build_qa_chain()
except Exception:
    st.warning(
        "The demo is not available right now (AI service limit). "
        "You can still review the code and architecture on GitHub."
    )
    st.stop()

# --- Interface ---
question = st.text_input("How can I help you today?", max_chars=MAX_CHARACTERS)

if question and st.session_state.questions_used >= MAX_QUESTIONS_PER_SESSION:
    st.warning("You have reached the question limit for this demo. Please come back later.")
    st.stop()

if question:
    st.session_state.questions_used += 1
    try:
        with st.spinner("Searching the document..."):
            result = qa_chain.invoke({"query": question})
    except Exception:
        st.error("An answer could not be generated right now. Please try again in a few minutes.")
        st.stop()

    st.markdown("### Answer")
    st.write(result["result"])

    # SOURCE REFERENCING: show which fragments the answer came from (traceability)
    with st.expander("📄 Document fragments used for this answer"):
        for i, doc in enumerate(result["source_documents"], start=1):
            st.markdown(f"**Fragment {i}:**")
            st.write(doc.page_content)
