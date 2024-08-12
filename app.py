import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Google Gemini API key from environment variables
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Retrieve login credentials from Streamlit secrets
VALID_USERNAME = st.secrets["USER_NAME"]
VALID_PASSWORD = st.secrets["PASSWORD"]

def get_pdf_text(pdf_path):
    """Extracts text from a PDF document."""
    text = ""
    pdf_reader = PdfReader(pdf_path)
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
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.getenv("GEMINI_API_KEY"))
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    
    # Extracting text from the retrieved documents
    context = "\n".join([doc.page_content for doc in docs])
    
    # Generate response using the custom function
    response = generate_response(context, user_question)
    
    return response

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat PDF", layout="wide")

    # Use session state to manage login status and selected page
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0

    if not st.session_state.logged_in:
        st.header("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state.logged_in = True
                st.session_state.image_index = 0  # Initialize image index
                st.success("Login successful!")
                st.experimental_rerun()  # Refresh to show the main page
            else:
                st.error("Invalid username or password")
        return  # Exit early to not show the main app while not logged in

    # Main Page Content
    st.header("ChatPDF")

    # Query Section
    st.write("### Ask a Question")
    user_question = st.text_input("Ask a Question")

    if user_question:
        # Get response using the user_input function
        response = user_input(user_question)
        st.write("## Reply:")
        st.write(response)

    # Display images with circular navigation
    images = ["image1.png", "image2.png"]
    image_index = st.session_state.image_index

    # Display text above the image
    st.write(f"### Image {image_index + 1} of {len(images)}")

    # Display image with smaller size
    st.image(images[image_index], width=400)  # Adjust width as needed

    # Navigation buttons for images
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state.image_index = (image_index - 1) % len(images)
            st.experimental_rerun()  # Refresh to show the previous image
    with col2:
        if st.button("Next"):
            st.session_state.image_index = (image_index + 1) % len(images)
            st.experimental_rerun()  # Refresh to show the next image

if __name__ == "__main__":
    main()
