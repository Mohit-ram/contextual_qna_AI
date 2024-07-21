
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

#Step1: AI Model
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
groq_api_key = os.getenv('GROQ_API_KEY')

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

#session initalisation
st.title("Contextual Q&A with Chat History ")
st.write("Upload Pdf's and ask about them")
st.write("AI can only respond regarding the context and amount of context provide.")
session_id=st.text_input("Session ID",value="default_session")
if 'session_store' not in st.session_state:
        st.session_state.session_store={}

#Step2: RAG Impementtaion
    #data loadig
uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
if uploaded_files:
    documents =[]
    for uploaded_file in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf, 'wb') as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

        pdf_read=PyPDFLoader(temppdf).load()
        documents.extend(pdf_read)
    #data Chunking and vector stores
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever() 

#Step3:Prompts 
    # History aware prompt
    system_with_history_message=(
                "Given a chat history and the latest user question"
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
    system_with_history_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_with_history_message),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
    history_aware_retriever=create_history_aware_retriever(llm,retriever,system_with_history_prompt)
    # actual QNA prompt
    system_actual_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise."
                    "\n\n"
                    "{context}"
                )
    qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_actual_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
    
    #Step4: Runnable chains
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
    
    def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.session_store:
                st.session_state.session_store[session_id]=ChatMessageHistory()
            return st.session_state.session_store[session_id]
    
    conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
    user_input = st.text_input("Your question:")
    if user_input:
        session_history=get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id":session_id}
            },
        )
        st.write(st.session_state.session_store)            
        st.write("Assistant:", response['answer'])
        st.write("Chat History:", session_history.messages)
