import os
import fitz
import streamlit as st
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsAllowInvalidCertificates=True,
    server_api=ServerApi('1')
)
db = client["pdf_database"]
collection = db["pdf_texts"]

st.set_page_config(page_title="📄 Invoice Analyzer", layout="centered")
st.title("📊 Upload & Analyze Invoice with Gemini")

uploaded_file = st.file_uploader("Upload Invoice (PDF only)", type=["pdf"])

def extract_text_from_pdf(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "".join([page.get_text() for page in doc]).strip()
    except Exception as e:
        return f"PDF extraction error: {e}"

prompt = PromptTemplate(
    input_variables=["invoice_text"],
    template="""
You are an expert invoice analyzer. Analyze the following invoice text and extract the following details in structured JSON format:
- Vendor Name
- Invoice Number
- Invoice Date
- Due Date
- Total Amount
- Taxes
- Line Items (name, quantity, unit price, total)
If any information is missing, write "Not Available".

Invoice Text:
\"\"\"
{invoice_text}
\"\"\"
Return only valid JSON.
"""
)

chain = prompt | llm

if uploaded_file:
    pdf_bytes = uploaded_file.read()
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(pdf_bytes)

    if text:
        st.text_area("Extracted Invoice Text", text, height=300)

        if st.button("Analyze with Gemini"):
            with st.spinner("Analyzing..."):
                try:
                    response = chain.invoke({"invoice_text": text}).content
                    st.subheader("Gemini Analysis Result")
                    st.code(response, language="json")
                    collection.insert_one({
                        "filename": uploaded_file.name,
                        "text": text,
                        "gemini_output": response
                    })
                    st.success("Stored in MongoDB")
                except Exception as e:
                    st.error(f"Gemini error: {str(e)}")
    else:
        st.error("Could not extract text from the uploaded PDF.")
