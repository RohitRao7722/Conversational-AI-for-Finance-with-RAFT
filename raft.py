# 



import streamlit as st
import os
import time
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

os.environ['GOOGLE_API_KEY'] = "AIzaSyA9uyD1CbVTqQ7RxLjRRX0IgkuRalSCDAg"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",api_key="AIzaSyA9uyD1CbVTqQ7RxLjRRX0IgkuRalSCDAg")
groq_api_key=os.getenv('GROQ_API_KEY')
st.title("Conversational AI for Finance with RAFT: Feedback-Driven Document Retrieval and Analysis")
st.write("Upload a PDF document")

api_key = st.text_input("Enter your Groq API key:", type="password")

collection_name = "finance_db"

url="http://localhost:6333/dashboard"


if api_key:
    llm= ChatGroq(model_name="llama3-groq-70b-8192-tool-use-preview", api_key=groq_api_key)
    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}
        
    if 'feedback' not in st.session_state:
        st.session_state.feedback = {}
        
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Qdrant.from_documents(
            splits,
            embeddings,
            url=url,
            prefer_grpc=False,
            collection_name=collection_name
        )


        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, which might reference the context in the chat history, "
            "formulate a standalone question that can be understood without the chat history. Do not answer the question, "
            "just reformulate it if needed."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you don't know. Keep the answer concise."
            "\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                }
            )
            st.success(f"Assistant: {response['answer']}")
            st.write("Chat History", session_history.messages)

            # Request Feedback
            feedback = st.radio("Was this answer helpful?", ("Yes", "No"))
            if feedback:
                st.session_state.feedback[session_id] = feedback
                st.write(f"Feedback: {feedback}")
            
            # Simulate RAFT: Adjusting Retrieval based on Feedback
            if feedback == "No":
                st.warning("Improving the model based on your feedback...")
                # Simulate model adjustment based on feedback (e.g., adjust retrieval or LLM tuning)
                # In a full RAFT implementation, you'd refine the model using this feedback.

else:
    st.warning("Please enter the Groq API key")