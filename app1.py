import os
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


st.set_page_config(page_title="Medicure", page_icon="📑", layout="wide")

st.sidebar.title("📊 MediCure")
st.sidebar.write("Upload a **PDF or CSV**, and ask questions about its content.")
st.sidebar.markdown("---")


tab1, tab2 = st.tabs(["📄 PDF Analysis", "📊 CSV Analysis"])

vector_store = None


with tab1:
    st.header("📑 Upload a Healthcare PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_pdf:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_pdf.read())

     
        loader = PyMuPDFLoader("temp.pdf")
        documents = loader.load()

        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

      
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embeddings)


with tab2:
    st.header("📊 Upload a CSV for Analysis")
    uploaded_csv = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write("### Preview of Uploaded CSV:")
        st.dataframe(df.head())

        
        csv_text = df.to_string()

   
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.create_documents([csv_text])
        vector_store = FAISS.from_documents(docs, embeddings)


if vector_store:
    llm = ChatGroq(api_key="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", model_name="mixtral-8x7b-32768")

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are an AI assistant. Answer the question based on the context:\nContext: {context}\nQuestion: {question}"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt}
    )

    st.markdown("---")
    st.subheader("🔍 Ask a Question")
    query = st.text_input("Type your question here:")

    if query:
        response = qa_chain.run(query)
        st.success("### 🤖 AI Response:")
        st.write(response)
