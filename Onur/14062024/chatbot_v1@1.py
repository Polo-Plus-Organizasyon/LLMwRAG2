import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from warnings import filterwarnings
import time

def main():
    load_dotenv()
    filterwarnings('ignore')
    st.set_page_config(page_title="Robark",page_icon="🤖")
    st.header("🦙 Eğitim Robotu")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "vectorestore" not in st.session_state:
        st.session_state.vectorestore = None
        
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message['avatar']):
            st.markdown(message["content"])

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf'],accept_multiple_files=True)
        process = st.button("Process")
    if process:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Dosyalar yükleniyor...")
            progress_bar.progress(20)
            files_text = get_files_text(uploaded_files)
            
            status_text.text("Metin parçalanıyor...")
            progress_bar.progress(40)
            text_chunks = get_text_chunks(files_text)
            
            status_text.text("Vektör deposu oluşturuluyor...")
            progress_bar.progress(60)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.vectorestore = vectorstore.as_retriever()
            
            status_text.text("Konuşma zinciri oluşturuluyor...")
            progress_bar.progress(80)
            st.session_state.conversation = get_conversation_chain()
            
            status_text.text("Tamamlandı")
            progress_bar.progress(100)
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

            st.session_state.processComplete = True

    if  st.session_state.processComplete == True:
        user_question = st.chat_input("Ask your question from the PDF?")
        if user_question:
            user_input(user_question)


def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
    return text


def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    doc_splits = text_splitter.split_text(text)
    return doc_splits


def get_vectorstore(text_chunks):
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}
    embeddings = GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs
    )
    try:
        vectorstore = Chroma(collection_name="rag-chroma", embedding_function=embeddings, persist_directory='db')
    except:
        vectorstore = Chroma.from_texts(texts=text_chunks, collection_name="rag-chroma", embedding=embeddings, persist_directory='db')
    return vectorstore

def get_conversation_chain():
    llm = ChatOllama(model= 'llama3',temperature=0.3)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You're a teacher. When you are asked questions, you are expected to answer them in a way that is faithful to the given context. When answering the questions, you have to answer only from the Turkish point of view. Your answers should be of sufficient length, neither too long nor too short. When answering questions, you cannot say exactly the same as the context. Your answer should not contain text other than Turkish, but may contain concepts.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {content} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "content"],
    )
    rag_chain = prompt | llm | StrOutputParser()
    
    return rag_chain

def user_input(user_question):
    with st.chat_message("user", avatar='👨🏻'):
        st.markdown(user_question)

    st.session_state.chat_history.append({"role": "user",
                                    "avatar": '👨🏻',
                                    "content": user_question})

    response = st.session_state.conversation.invoke({'question': user_question, 'content':st.session_state.vectorestore})

    with st.chat_message("assistant", avatar='🤖'):
        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant",
                            "avatar": '🤖',
                            "content": response})



if __name__ == '__main__':
    main()