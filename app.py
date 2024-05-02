import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def data_loader(data_file, uploaded_file):
    if uploaded_file.name.endswith(".csv"):     
        loader = CSVLoader(file_path=data_file, encoding='utf-8', csv_args={'delimiter':','})
    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(file_path=data_file)
    else:
        st.warning("Unsupported File")
    data = loader.load()
    return data

def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_documents(data)
    return chunks

def data_embadding(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(chunks, embeddings)
    return db

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def user_input(user_question, db):
    
    doc = db.similarity_search(user_question,k=3)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":doc, "question": user_question},
        return_only_outputs=True)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat With Document")
    
    st.header("Chat With Documnet")

    uploaded_file = st.file_uploader("Upload Document-Support",type=((["csv","txt"])))
        
    #if st.button("Submit & Process"):
    if uploaded_file is not None:
        if uploaded_file.name.endswith((".csv", ".txt")):
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                    
            data = data_loader(tmp_file_path,uploaded_file)    
            chunks = text_splitter(data)
            db = data_embadding(chunks)
            st.success("Done")
                
            user_question = st.text_input("Ask a Question from the Document")
            if user_question:
                user_input(user_question,db)
        else:
            st.warning("Support Only .csv,.txt Files")
    else:
        st.text("Kindly upload the Document")
            
if __name__ == "__main__":
    main()