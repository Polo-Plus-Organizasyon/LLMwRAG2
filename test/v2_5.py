import streamlit as st
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv
import timeit
import re
import asyncio
import time
from pprint import pprint

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
    llm = ChatOllama(model= 'llama3',temperature=0.5, keep_alive=-1)
    prompt = PromptTemplate(
        template="""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        ### Talimatlar ###
        Sen bir öğretmensin ve sadece verilen içeriğe bağlı kalarak cevaplar vereceksin. İçerikteki cümleleri birebir aynısını kullanmayacaksın ama benzer şekilde cevaplayacaksın. Sohbet geçmişini takip edecek ve öğrencinin sorularına Türkçe, yazım kurallarına uygun ve anlaşılır cevaplar vereceksin. Cevapların, öğrencinin anlayabileceği sadelikte ve yeteri uzunlukta olmalı.
        \n
        ### İçerik ###
        İçerik: {content}
        \n
        ### Sohbet Geçmişi ###
        Sohbet Geçmişi: {chat_history}
        \n
        ### Soru ###
        Soru:{question}<|eot_id|><|start_header_id|>user<|end_header_id|
        \n
        ### Cevap ###
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["chat_history", "question", "content"],
    )

    rag_chain = prompt | llm | StrOutputParser()
    
    return rag_chain

def retrieval_grader():
    llm = ChatOllama(model='llama3', format="json", temperature=0, keep_alive=-1)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )

    retrieval_grader = prompt | llm | JsonOutputParser()
    return retrieval_grader

def hallucination_grader():
    llm = ChatOllama(model='llama3', format="json", temperature=0, keep_alive=-1)
    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )

    hallucination_grader = prompt | llm | JsonOutputParser()
    return hallucination_grader

def answer_grader():
    llm = ChatOllama(model='llama3', format="json", temperature=0, keep_alive=-1)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | llm | JsonOutputParser()
    return answer_grader

def question_router():
    llm = ChatOllama(model='llama3', format="json", temperature=0, keep_alive=-1)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
        prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
        or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
        no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    question_router = prompt | llm | JsonOutputParser()
    return question_router


class GraphState(TypedDict):
    question: str
    answer: str
    web_search: str
    chat_history: str
    content: str

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    content = state["content"]
    return {"content": content, "question": question}

def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state['chat_history']
    generation = get_conversation_chain().invoke({'chat_history': chat_history, "content": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": documents}
        )
        grade = score["score"]

        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)

        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state):
    web_search_tool = TavilySearchResults(k=3)
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    return {"documents": documents.join(web_results), "question": question}

def route_question(state):

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router().invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):

    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]

    if web_search == "Yes":
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader().invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader().invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

def workflow_app():
    workflow = StateGraph(GraphState)
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )
    app = workflow.compile()
    return app

def main():
    load_dotenv()
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
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "response_time" not in st.session_state:
        st.session_state.response_time = None
    
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
    documents = st.session_state.vectorstore.invoke(user_question)
    
    inputs = {"question": user_question, "chat_history": chat_history, "documents": documents}
    for output in workflow_app().stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])

    st.session_state.chat_history.append({"role": "assistant",
                                        "avatar": '🤖',
                                        "content": value["generation"]})
    
if __name__ == '__main__':
    main()


