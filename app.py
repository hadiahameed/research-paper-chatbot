

#----------------------------------Importing Packages----------------------------------
# A vector store takes care of storing embedded data and performing vector search for you.
#Options are Chroma, FAISS (Facebook AI Similarity Search) or Lance
#Chroma DB is for storing and retrieving vector embeddings and runs on local machine.
from langchain_community.vectorstores import Chroma

#Embeddings create a vector representation of a piece of text.
#sentence_transformers package models are originating from Sentence-BERT
from langchain_community.embeddings import SentenceTransformerEmbeddings

#ChatOpenAI is the primary class used for chatting with OpenAI models.
#This represents LangChain's interface for interacting with OpenAI's API.
from langchain_community.chat_models import ChatOpenAI

#Does question answering over documents you pass in, and cites it sources.
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

#Use document loaders to load data from a source as Document's. A Document is a piece of text and associated metadata.
#For example, there are document loaders for loading a simple .txt file, for loading the text contents of any web page,
#or even for loading a transcript of a YouTube video.
from langchain_community.document_loaders import DirectoryLoader

#Split a long document into smaller chunks that can fit into your model's context window.
#Recursively splits text. Splitting text recursively serves the purpose of trying to keep related pieces of text next to
#each other. This is the recommended way to start splitting text.
from langchain.text_splitter import RecursiveCharacterTextSplitter

import openai
import streamlit as sm
import os
import shutil
import hashlib
from glob import glob
from datetime import datetime


#---------------------------------- Code----------------------------------
sm.title("I am LabBot - your personal research assistant!")

openai.api_key = '<YOUR KEY HERE>' #get your own key from OpenAI
os.environ['OPENAI_API_KEY'] = openai.api_key
tmp_directory = 'temp'

new_uploads = "Upload new research papers."
uploaded_papers = "Uploaded papers."

chatbot_started = False

#Function for uploading new .txt files
def doc_upload(directory: str):
    documents = []
    loader = DirectoryLoader(path=directory, show_progress=True)
    documents = loader.load()
    return documents

#Splitting documents into small chunks
def doc_split(documents, chunk_size=1500, chunk_overlap=30):
    split_text = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = split_text.split_documents(documents)

    return docs

@sm.cache_resource #only loads if there is a change in update
#Wrapper around OpenAI
def startup_event(last_update: str):

    print(f"{last_update}")
    directory = 'temp/'
    documents = doc_upload(directory)
    docs = doc_split(documents)

    #----------------------------------Chroma db----------------------------------
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") #small LLM
    persist_directory = "chroma_db"

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectordb.persist()
    db = Chroma.from_documents(docs, embeddings) #on disk

    #----------------------------------GPT Model----------------------------------
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)
    chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=True)

    return db, chain

#---------------------------------- Querying the GPT Model with RAG---------------
def get_answer(query: str, db, chain):
    matching_docs_score = db.similarity_search_with_score(query)

    matching_docs = [doc for doc, score in matching_docs_score]

    ##LLM will create a prompt for ChatGPT using these docs to build context and add our query to it.
    answer = chain.run(input_documents=matching_docs, question=query)


    sources = [{
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": score
    } for doc, score in matching_docs_score]

    return {"answer": answer, "sources": sources}


#UI of the ChatBot
def start_chatbot():
    db, chain = startup_event(last_db_updated)
    if "openai_model" not in sm.session_state:
        sm.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in sm.session_state:
        sm.session_state.messages = []

    for message in sm.session_state.messages:
        with sm.chat_message(message["role"]):
            sm.markdown(message["content"])

    if prompt := sm.chat_input("What is up?"):
        sm.session_state.messages.append({"role": "user", "content": prompt})
        with sm.chat_message("user"):
            sm.markdown(prompt)

        with sm.chat_message("assistant"):
            message_placeholder = sm.empty()
            full_response = get_answer(sm.session_state.messages[-1]["content"], db, chain)
            answer = full_response['answer']
            message_placeholder.markdown(answer)
            sm.session_state.messages.append({"role": "assistant", "content": answer})

content_type = sm.sidebar.radio("Which knowledge base do you want to use?",
                                [uploaded_papers, new_uploads])
if content_type == new_uploads:
    uploaded_files = sm.sidebar.file_uploader("Choose a txt file", type="txt", accept_multiple_files=True)
    uploaded_file_names = [file.name for file in uploaded_files]

    if uploaded_files is not None and len(uploaded_files):
        if os.path.exists(tmp_directory):
            shutil.rmtree(tmp_directory)

        os.makedirs(tmp_directory)

        if len(uploaded_files):
            last_db_updated = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


        for file in uploaded_files:
            with open(f"{tmp_directory}/{file.name}", 'wb') as tmp:
                tmp.write(file.getvalue())
                tmp.seek(0)
curr_dir = [path.split(os.path.sep)[-1] for path in glob(tmp_directory + '/*')]

if content_type == uploaded_papers:
    sm.sidebar.write("Current Knowledge Base")
    if len(curr_dir):
        sm.sidebar.write(curr_dir)
    else:
        sm.sidebar.write('**No KB Uploaded**')

last_db_updated = hashlib.md5(','.join(curr_dir).encode()).hexdigest()

if curr_dir and len(curr_dir):
    start_chatbot()
else:
    sm.header('No KB Loaded, use left menu to start.')

