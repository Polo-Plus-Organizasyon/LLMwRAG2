import streamlit as st
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import timeit
import re
import asyncio
import time

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1500, chunk_overlap=200
    )

model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
embeddings = GPT4AllEmbeddings(
    model_name=model_name,
    gpt4all_kwargs=gpt4all_kwargs
)

def process_text(retriever_output):
    combined_text = " ".join([d.page_content for d in retriever_output])
    combined_text = re.sub(r'\.{2,}', '', combined_text)
    combined_text = re.sub(r'\n', '', combined_text)
    return combined_text

def get_conversation_chain():
    llm = ChatOllama(model= 'gemma2',temperature=0.5,keep_alive=-1,repeat_last_n = 64,top_k = 50,top_p = 0.9)
    # llm = ChatOllama(model= 'gemma2',temperature=0.5, keep_alive=-1,repeat_last_n = 64,repeat_penalty = 1.2,top_k = 50,top_p = 0.9)
    """llm = Ollama(
    base_url="http://localhost:11434",
    model="gemma2",
    callback_manager=CallbackManager(
        [StreamingStdOutCallbackHandler()],
    ),
    keep_alive=-1)"""
    prompt = PromptTemplate(
        template="""
        You are a teacher who answers questions.\
        Use only the information provided in the context to answers questions using your own words in Turkish.\
        Do not include any additional information.\
        Answer using simple words, following Turkish spelling and grammar rules.\
        If the context does not contain sufficient information to answer the question, respond with "İçerikte yeterli bilgi bulunmuyor."\
        #-----------------
        Question: {question}
        #-----------------
        Context: {content}
        """,
        input_variables=["question", "content"],
    )

    rag_chain = prompt | llm | StrOutputParser()
    
    return rag_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Robark", page_icon="./favicon.webp")
    # st.header("🦙 Eğitim Robotu")
    header = \
    """
        <div style="background-color:#CDE8E5;padding:1.5px">
        <h1 style="color:white;text-align:center;">🦙 Eğitim Robotu</h1>
        </div><br>
    """
    st.markdown(header, unsafe_allow_html=True)
    hide_default_format = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    .stButton button {
        width: 100%;
        height: 50px;
    }
    </style>
    """
    st.markdown(hide_default_format, unsafe_allow_html=True)
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "response_time" not in st.session_state:
        st.session_state.response_time = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message['avatar']):
            st.markdown(message["content"])
    
    with st.sidebar:
        uploaded_files = st.file_uploader("PDF dosyalarını yükleyin", type="pdf", accept_multiple_files=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            process = st.button("Yükle")
        with col2:
            refresh = st.button("Yenile")
            if refresh:
                st.session_state.chat_history = []
                st.session_state.response_time = None
                st.rerun()

        if st.session_state.response_time:
            st.sidebar.markdown(f"**Cevaplama süresi:** {st.session_state.response_time:.2f} saniye")

    if process and uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        documents = []
        
        status_text.text("Dosyalar yükleniyor...")
        progress_bar.progress(20)
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            pdf_loader = PyMuPDFLoader(file_path=temp_file_path)
            docs = pdf_loader.load()
            documents.extend(docs)
            
        status_text.text("Metin parçalanıyor...")
        progress_bar.progress(40)
        doc_splits = text_splitter.split_documents(docs)

        status_text.text("Vektör deposu oluşturuluyor...")
        progress_bar.progress(60)
        try:
            vectorstore = Chroma(collection_name="rag-chroma", embedding_function=embeddings, persist_directory='db')
        except:
            vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=embeddings, persist_directory='db')
        retriever = vectorstore.as_retriever()
        st.session_state.vectorstore = retriever
        
        status_text.text("Konuşma zinciri oluşturuluyor...")
        progress_bar.progress(80)
        st.session_state.conversation = get_conversation_chain()
        
        status_text.text("Tamamlandı")
        progress_bar.progress(100)
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        st.session_state.processComplete = True

    if st.session_state.processComplete:
        user_question = st.chat_input("PDF'den sorunuz nedir?")
        if user_question:
            start_time = timeit.default_timer()
            asyncio.run(user_input(user_question))
            end_time = timeit.default_timer()
            st.session_state.response_time = end_time - start_time
            st.rerun()

async def user_input(user_question):
    with st.chat_message("user", avatar='👨🏻'):
        st.markdown(user_question)

    st.session_state.chat_history.append({"role": "user",
                                        "avatar": '👨🏻',
                                        "content": user_question})

    truncated_history = st.session_state.chat_history[-10:]
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in truncated_history])
    
    loop = asyncio.get_event_loop()
    response_stream = await loop.run_in_executor(None, st.session_state.conversation.stream, {'question': user_question, 'content': process_text(retriever_output=st.session_state.vectorstore.invoke(user_question))})

    response_container = st.empty()
    response_text = ""
    
    for chunk in response_stream:
        response_text += chunk
        response_container.markdown(response_text)

    st.session_state.chat_history.append({"role": "assistant",
                                        "avatar": '🤖',
                                        "content": response_text})

if __name__ == '__main__':
    main()


