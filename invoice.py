import os
import fitz  # for PDF
import streamlit as st
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from PIL import Image
import pytesseract
import io
import cv2
import numpy as np

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# Connect to MongoDB
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsAllowInvalidCertificates=True,
    server_api=ServerApi('1')
)
db = client["pdf_database"]
collection = db["pdf_texts"]

# Streamlit UI setup
st.set_page_config(page_title="📄 Invoice Analyzer", layout="centered")
st.title("📊 Upload & Analyze Multiple Invoices (PDF/Image)")

uploaded_files = st.file_uploader("Upload Invoices (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

# --- Extraction Functions ---

def extract_text_from_pdf(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "".join([page.get_text() for page in doc]).strip()
    except Exception as e:
        return f"PDF extraction error: {e}"

def extract_text_from_image(image_bytes):
    try:
        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream).convert("RGB")
        np_img = np.array(image)

        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        st.image(thresh, caption="🖼️ Enhanced Image for OCR", use_container_width=True)

        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        return text.strip()
    except Exception as e:
        return f"Image OCR error: {e}"

# --- Prompt for Gemini ---
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
\"\"\"{invoice_text}\"\"\"
Return only valid JSON.
"""
)

# --- Chain ---
chain = prompt | llm

# --- File Processing ---
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(f"📁 File: {uploaded_file.name}")
        file_type = uploaded_file.type
        file_bytes = uploaded_file.read()

        with st.spinner("Extracting text..."):
            if file_type == "application/pdf":
                text = extract_text_from_pdf(file_bytes)
            elif file_type.startswith("image/"):
                text = extract_text_from_image(file_bytes)
            else:
                st.error("Unsupported file type.")
                continue

        if text:
            st.text_area("📝 Extracted Invoice Text", text, height=300)

            if st.button(f"Analyze with Gemini: {uploaded_file.name}"):
                with st.spinner("Analyzing..."):
                    try:
                        response = chain.invoke({"invoice_text": text}).content
                        st.subheader("🧠 Gemini Analysis Result")
                        st.code(response, language="json")

                        # Save to DB
                        collection.insert_one({
                            "filename": uploaded_file.name,
                            "text": text,
                            "gemini_output": response
                        })

                        st.success("✅ Stored in MongoDB")
                    except Exception as e:
                        st.error(f"Gemini error: {str(e)}")
        else:
            st.error("❌ Could not extract text from this file.")
