import os

import streamlit as st
from PyPDF2 import PdfReader
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

OPENAI_API_KEY = "YOUR_API_KEY"

st.header("My first Custom Chatbot")

with st.sidebar:
    st.title("Upload your Documents")
    file = st.file_uploader("Upload your documents here to ask any question related to documents")

#extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text += page.extract_text()
        #st.write(text)


    #breaking into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150

    )
    chunks = text_splitter.split_text(text)

    #st.write(chunks)
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    #genrating embeddings
    embeddings = OpenAIEmbeddings()

    #creating vector store
    vector_stores = FAISS.from_texts(chunks, embeddings)
    #embessing(OpenAI)
    #intitalizing vector space
    #storing chunks and embedding

    #get the question
    user_question = st.text_input("Please enter your question here")
    if user_question:
        match = vector_stores.similarity_search(user_question)
        #st.write(match)

        #define llm
        llm = ChatOpenAI(
            max_tokens=1000,
            temperature=0,
            model="gpt-3.5-turbo"
        )

        #chain-> take the question, get relevant document,pass itto the LLM and genrate the output
        chain = load_qa_chain(llm,chain_type="stuff")
        response = chain.run(input_documents = match,question=user_question)
        st.write(response)





