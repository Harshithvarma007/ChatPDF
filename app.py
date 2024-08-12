import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Retrieve login credentials from .env
VALID_USERNAME = os.getenv("USER_NAME")
VALID_PASSWORD = os.getenv("PASSWORD")

def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits the extracted text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates a vector store from text chunks and saves it locally."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def generate_response(context, question):
    """Generates a response to a question based on the provided context."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in the provided context, just say, "The answer is not in the provided context." Do not provide the wrong answer.
    and make the answer more visually attractive, structured and precise.
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    prompt = prompt_template.format(context=context, question=question)
    
    # Initialize the model with the correct name
    model = genai.GenerativeModel('gemini-1.0-pro')
    
    # Start a chat session and send the message
    chat_session = model.start_chat(
        history=[]
    )
    
    response = chat_session.send_message(prompt)
    return response.text

def user_input(user_question):
    """Handles user input and generates a response based on the vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    
    # Extracting text from the retrieved documents
    context = "\n".join([doc.page_content for doc in docs])
    
    # Generate response using the custom function
    response = generate_response(context, user_question)
    
    st.write("Reply:", response)

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat PDF")
    
    # Login Section
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.success("Login successful!")
            
            st.header("Chat with PDF")
            user_question = st.text_input("Ask a Question from the PDF Files")

            if user_question:
                user_input(user_question)

            with st.sidebar:
                st.title("Menu:")
                pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
                if st.button("Submit & Process"):
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")
        else:
            st.error("Invalid username or password")

if __name__ == "__main__":
    main()
