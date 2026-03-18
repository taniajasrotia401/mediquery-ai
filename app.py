
import os
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

st.set_page_config(page_title="MediQuery AI", page_icon="🏥", layout="wide")

st.title("🏥 MediQuery AI")
st.caption("RAG-powered Medical Document Intelligence Assistant")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("MediQuery AI uses RAG to answer questions from 5000+ real medical transcriptions.")
    st.markdown("**Stack:** LangChain · FAISS · Llama 3 · HuggingFace")

@st.cache_resource
def build_rag_pipeline(api_key):
    # Load data
    df = pd.read_csv("mtsamples.csv")
    df_clean = df.dropna(subset=["transcription"]).reset_index(drop=True)
    df_sample = df_clean.sample(500, random_state=42).reset_index(drop=True)

    # Build documents
    documents = []
    for _, row in df_sample.iterrows():
        content = f"""
Medical Specialty: {row["medical_specialty"]}
Report Name: {row["sample_name"]}
Description: {row["description"]}
Transcription: {row["transcription"]}
Keywords: {row["keywords"]}
        """.strip()
        documents.append(Document(
            page_content=content,
            metadata={
                "specialty": row["medical_specialty"],
                "report_name": row["sample_name"]
            }
        ))

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Embed
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=api_key)

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
You are MediQuery AI, an intelligent medical document assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I couldn't find that in the provided medical records."
Always mention which medical specialty the information comes from.

Context from medical records:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n---\n\n".join([
            f"[{doc.metadata['specialty']} | {doc.metadata['report_name']}]\n{doc.page_content}"
            for doc in docs
        ])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

# Main UI
if not groq_api_key:
    st.warning("👈 Please enter your Groq API key in the sidebar to get started.")
else:
    with st.spinner("🔄 Building RAG pipeline... (first time takes ~2 mins)"):
        rag_chain, retriever = build_rag_pipeline(groq_api_key)
    st.success("✅ MediQuery AI is ready!")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📚 View Sources"):
                    for s in msg["sources"]:
                        st.markdown(s)

    # Chat input
    if question := st.chat_input("Ask anything about medical records..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get answer + sources
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching medical records..."):
                answer = rag_chain.invoke(question)
                relevant_docs = retriever.invoke(question)
                sources = list(set([
                    f"📄 **{doc.metadata['specialty']}** → {doc.metadata['report_name'].strip()}"
                    for doc in relevant_docs
                ]))

            st.markdown(answer)
            with st.expander("📚 View Sources"):
                for s in sources:
                    st.markdown(s)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
