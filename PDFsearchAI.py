import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Page Config
st.set_page_config(page_title="Chat with PDFs", page_icon="📄")
st.title("📄 Chat with Multiple PDFs")

# Sidebar for API Key and Upload
with st.sidebar:
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    process_btn = st.button("Process Documents")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Session State to store vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if process_btn and uploaded_files and api_key:
    with st.spinner("Processing PDFs..."):
        documents = []
        # 1. Load PDFs
        for file in uploaded_files:
            # Save temp file to load with PyPDFLoader
            with open(f"temp_{file.name}", "wb") as f:
                f.write(file.getbuffer())
            loader = PyPDFLoader(f"temp_{file.name}")
            documents.extend(loader.load())
            os.remove(f"temp_{file.name}") # Clean up

        # 2. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # 3. Create Embeddings & Vector Store
        embeddings = OpenAIEmbeddings()
        st.session_state.vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        st.success("Documents processed successfully!")

# Chat Interface
if st.session_state.vector_store:
    query = st.chat_input("Ask a question about your PDFs")
    
    if query:
        # 4. Setup Retrieval Chain
        retriever = st.session_state.vector_store.as_retriever()
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # 5. Generate Response
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": query})
            st.write(response["answer"])
            
            # Optional: Show sources
            with st.expander("View Sources"):
                for doc in response["context"]:
                    st.write(doc.page_content[:200] + "...")
else:
    st.info("Please upload PDFs and process them in the sidebar to start chatting.")
    