import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.output_parsers import StrOutputParser
from warnings import filterwarnings
import timeit
import time
from multiprocessing import Pool, cpu_count
import asyncio
import re

def main():
    load_dotenv()
    filterwarnings('ignore')
    st.set_page_config(page_title="Robark", page_icon="🤖")
    st.header("🦙 Eğitim Robotu")
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

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "response_time" not in st.session_state:
        st.session_state.response_time = None

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message['avatar']):
            st.markdown(message["content"])

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
        
        col1, col2 = st.columns([1, 1])
        with col1:
            process = st.button("Process")
        with col2:
            refresh = st.button("Refresh")
            if refresh:
                st.session_state.chat_history = []
                st.session_state.response_time = None
                st.rerun()

        if st.session_state.response_time:
            st.sidebar.markdown(f"**Cevaplama süresi:** {st.session_state.response_time:.2f} saniye")

    if process and st.session_state.uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Dosyalar yükleniyor...")
        progress_bar.progress(20)
        files_text = get_files_text(st.session_state.uploaded_files)

        status_text.text("Metin parçalanıyor...")
        progress_bar.progress(40)
        text_chunks = get_text_chunks(files_text)

        status_text.text("Vektör deposu oluşturuluyor...")
        progress_bar.progress(60)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.vectorstore = vectorstore.as_retriever()

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
        user_question = st.chat_input("Ask your question from the PDF?")
        if user_question:
            start_time = timeit.default_timer()
            asyncio.run(user_input(user_question))
            end_time = timeit.default_timer()
            st.session_state.response_time = end_time - start_time
            st.rerun()

def process_text(retriever_output):
    combined_text = " ".join([d.page_content for d in retriever_output])
    combined_text = re.sub(r'\.{2,}', '', combined_text)
    combined_text = re.sub(r'\n', '', combined_text)
    return combined_text

def get_files_text(uploaded_files):
    with Pool(cpu_count()) as pool:
        texts = pool.map(get_pdf_text, uploaded_files)
    return " ".join(texts)

def get_pdf_text(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        text = chr(12).join([page.get_text() for page in doc])
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200
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
    llm = ChatOllama(model= 'llama3',temperature=0.5, keep_alive=-1, top_k=45, top_p=0.85, template="")
    prompt = PromptTemplate(
        template="""
        Türkçe cevaplar vermesi ve içerikten dışarı çıkmaması için aşağıdaki kuralları kullan:
        ### Kurallar ###
        - Türkçe cevapla.
        - Yalnızca verilen içeriğe dayanarak yanıt ver.
        - İçerik dışı bilgileri hariç tut.
        - Verilen içerikte soruyla ilgili bilgi yoksa 'İçerikte yeterli bilgi bulunamadı.' yanıtı ver.
        - Sohbet geçmişini de incele.
        - İçeriğe bağlı kalarak kendi cümlelerin ile yanıt ver.
        - Sorudan uzaklaşmadan ve içeriğe bağlı kalarak soruyu yanıtla.
        
        ### İçerik ###
        İçerik: "\n{content}\n"
        
        ### Sohbet Geçmişi ###
        Sohbet Geçmişi: "\n{chat_history}\n"
        
        ### Soru ###
        Soru: "\n{question}\n"
        
        ### Cevap ###
        Cevap:
        """,
        input_variables=["chat_history", "question", "content"],
    )

    rag_chain = prompt | llm | StrOutputParser()
    
    return rag_chain

async def user_input(user_question):
    with st.chat_message("user", avatar='👨🏻'):
        st.markdown(user_question)

    st.session_state.chat_history.append({"role": "user",
                                        "avatar": '👨🏻',
                                        "content": user_question})

    # Limit chat history to the last 10 messages to reduce context length
    truncated_history = st.session_state.chat_history[-10:]
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in truncated_history])
    
    loop = asyncio.get_event_loop()
    response_stream = await loop.run_in_executor(None, st.session_state.conversation.stream, {'chat_history': chat_history, 'question': user_question, 'content': process_text(st.session_state.vectorstore.invoke(user_question))})

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
