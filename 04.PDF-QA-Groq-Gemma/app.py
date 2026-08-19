import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

## Load the GROQ and Google API from the .env file

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

llm = ChatGroq(groq_api_key = groq_api_key, model_name ="openai/gpt-oss-20b")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Questions: {input}

"""
)
PDF_DIR = "./us_cencus"

def vector_embeddings():
    if "vectors" not in st.session_state:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
        loader = PyPDFDirectoryLoader(PDF_DIR) #Data Ingestion
        docs = loader.load() ##loading document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        final_documents = text_splitter.split_documents(docs)
        st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)


st.subheader("Upload PDF documents")
uploaded_files = st.file_uploader(
    "Upload one or more PDF files to add them to the document store",
    type="pdf",
    accept_multiple_files=True,
)

if uploaded_files:
    # Streamlit re-delivers the same uploaded_files on every rerun (e.g. when
    # the user submits a question afterwards), not just on the upload event.
    # Only write to disk / reset the vector store when the set of files
    # actually changed, otherwise a later rerun wipes out st.session_state.vectors.
    new_names = {f.name for f in uploaded_files}
    if new_names != st.session_state.get("uploaded_names"):
        os.makedirs(PDF_DIR, exist_ok=True)
        saved_names = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(PDF_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_names.append(uploaded_file.name)
        st.session_state.uploaded_names = new_names
        st.session_state.pop("vectors", None)
        st.success(f"Saved {len(saved_names)} file(s) to {PDF_DIR}: {', '.join(saved_names)}")

prompt1 = st.text_input("Ask the document")

if st.button("Creating Vector Store"):
    if not os.path.isdir(PDF_DIR) or not any(f.lower().endswith(".pdf") for f in os.listdir(PDF_DIR)):
        st.error(f"No PDF files found in {PDF_DIR}. Upload at least one PDF above first.")
    else:
        vector_embeddings()
        st.write("Vector Store DB is Ready")

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({'input' : prompt1})
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("--------------------------------")