import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()

# Get the path to Google Application Credentials
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if credentials_path is None:
    # Use a hardcoded path if the environment variable is not set
    credentials_path = "saathi-439108-2866ecb350dc.json"

print("GOOGLE_APPLICATION_CREDENTIALS:", credentials_path)  # Debug print

if not credentials_path:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set or is None.")
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

genai.configure(api_key=os.getenv("AIzaSyCpzURfYqs9TbCw7yncdMt09dsj0bsvkW0"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Generate concise notes based on the following content:\n{context}\n
    Notes:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def generate_notes_from_chunks(text_chunks):
    chain = get_conversational_chain()
    all_notes = []

    for chunk in text_chunks:
        # Wrap the chunk in a Document object
        document = Document(page_content=chunk, metadata={})  # You can add relevant metadata if needed
        response = chain({"input_documents": [document]})  # Pass the Document object
        all_notes.append(response["output_text"])

    return "\n".join(all_notes)


def main():
    st.set_page_config("Chat PDF")
    st.header("Generate Notes from PDF by Pyneir 💁")


    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    st.write(pdf_docs)

    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)

                # Generate notes from the text chunks
            notes = generate_notes_from_chunks(text_chunks)
            st.success("Done")
            st.write("Generated Notes:")
            st.write(notes)


if __name__ == "__main__":
    main()
