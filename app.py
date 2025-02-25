import streamlit as st 
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os 

st.title("Hey Ashish, Chat with any website")
st.write("Ask questions.......")

api_key=st.text_input("Please enter your google api key...", type="password")

if api_key:
    os.environ["google_api_key"]=api_key

@st.cache_data
def get_url(url):
    loader=WebBaseLoader(url)
    documents=loader.load()
    text="\n\n".join([doc.page_content for doc in documents])
    return text

@st.cache_data
def get_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=1000)
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(text_chunk):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunk, embeddings)
    return vector_store

st.sidebar.header("Please enter url")
st.sidebar.write("Please ask questions related to url data")

url=st.sidebar.text_input("please enter your url here..")

if api_key:
    st.sidebar.header("Creating data...")

def get_convertional_chain():
    # Define the prompt template
    prompt_template = PromptTemplate(
        template='''
        You are an AI assistant helping to answer questions based on the provided context.

        Context:
        {context}

        Question:
        {question}

        Provide a clear and concise answer based only on the given context.
        ''',
        input_variables=["context", "question"]
    )
    # Initialize the model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    # Load the QA chain
    return load_qa_chain(model, chain_type="stuff", prompt=prompt_template)

st.sidebar.header("Ask your question")
user_questions=st.sidebar.text_area("Enter your question here")

if st.sidebar.button("Get Answer"):
    if url and api_key and user_questions:
        try:
            with st.spinner("Searching for answers..."):
                text = get_url(url)
                chunk = get_chunks(text)
                vector_store = get_vector_store(chunk)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
                docs = db.similarity_search(user_questions)
                chain = get_convertional_chain()
                response = chain({"input_documents":docs,"questions":user_questions},return_only_outputs=True)
            st.success("Answer...")
            st.subheader("Your questions...")
            st.write(user_questions)
        except Exception as issue:
            st.error(issue)