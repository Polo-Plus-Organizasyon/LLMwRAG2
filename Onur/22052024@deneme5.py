from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

data = DirectoryLoader('../pdf', glob="*.pdf", loader_cls=PyMuPDFLoader, use_multithreading=True).load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(data)

try:
    vectorstore = Chroma(collection_name="rag-chroma", embedding_function=GPT4AllEmbeddings(), persist_directory='db')
except:
    vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=GPT4AllEmbeddings(), persist_directory='db')

retriever = vectorstore.as_retriever()

question = 'mitoz bölünme'

prompt = PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    ### Instruction:
    You're helpful Turk teacher, who answers questions based upon provided content in a clear way and easy to understand way only in Turkish.If there is no content, or the content is irrelevant to answering the question, simply reply that you can't answer..
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    
    ## Content:
    {context}

    ## Question:
    {question}
    
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

llm = ChatOllama(model='llama3', temperature=0)

rag_chain = prompt | llm | StrOutputParser()

docs = retriever.invoke(question)
generation = rag_chain.invoke({"context": docs, "question": question})

print(generation)