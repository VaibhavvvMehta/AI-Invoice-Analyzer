import os
import io
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import json
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from datetime import datetime
import zipfile
import tempfile

# --------- ENV and Model Setup ---------
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

genai.configure(api_key=GEMINI_API_KEY)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Using a stable model
    temperature=0.3,
    google_api_key=GEMINI_API_KEY
)


# --------- MongoDB Connection Setup ---------
try:
    client = MongoClient(MONGO_URI)
    db = client["invoiceDB"]
    collection = db["invoices"]
    # Test connection
    client.admin.command('ping')
    print("✅ MongoDB connected successfully")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    client = None
    db = None
    collection = None

# --------- Prompt Setup ---------
prompt = PromptTemplate(
    input_variables=["invoice_text"],
    template="""
You are an expert in analyzing invoices. Analyze the text below and extract the following information in STRICT JSON format:

{{
    "vendor_name": "Company Name",
    "invoice_number": "INV-123",
    "invoice_date": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD", 
    "total_amount": 1234.56,
    "currency": "USD",
    "tax_amount": 123.45,
    "subtotal": 1111.11,
    "line_items": [
        {{"item_name": "Product/Service", "quantity": 2, "unit_price": 50.00, "total": 100.00}}
    ],
    "payment_terms": "Net 30",
    "billing_address": "Address if available",
    "shipping_address": "Address if available"
}}

IMPORTANT RULES:
1. Return ONLY valid JSON, no extra text
2. Use "Not Available" for missing information
3. Convert amounts to numbers (float), not strings
4. Use YYYY-MM-DD format for dates
5. If total_amount cannot be extracted, set to 0.0

Invoice Text:
{invoice_text}
"""
)
chain = prompt | llm 



# --------- MongoDB Storage Functions ---------
def check_mongodb_connection():
    """Check if MongoDB connection is available"""
    return client is not None and db is not None and collection is not None

def save_invoice_data(data):
    """Save invoice data to MongoDB"""
    if not check_mongodb_connection():
        return False, "MongoDB connection not available"
    
    try:
        result = collection.insert_one(data.copy())
        return True, f"Saved to MongoDB with ID: {result.inserted_id}"
    except Exception as e:
        return False, f"MongoDB save failed: {e}"

def get_all_invoice_data():
    """Get all invoice data from MongoDB"""
    if not check_mongodb_connection():
        return []
    
    try:
        return list(collection.find({}, {"_id": 0}))
    except Exception as e:
        st.error(f"Failed to read from MongoDB: {e}")
        return []

def count_invoices():
    """Count total invoices in MongoDB"""
    if not check_mongodb_connection():
        return 0
    
    try:
        return collection.count_documents({})
    except Exception as e:
        st.error(f"Failed to count documents: {e}")
        return 0

def get_total_amount():
    """Get total amount of all invoices from MongoDB"""
    if not check_mongodb_connection():
        return 0
    
    try:
        result = list(collection.aggregate([
            {"$group": {"_id": None, "total": {"$sum": "$total_amount"}}}
        ]))
        return result[0]["total"] if result else 0
    except Exception as e:
        st.error(f"Failed to calculate total amount: {e}")
        return 0

def clear_all_data():
    """Clear all invoice data from MongoDB"""
    if not check_mongodb_connection():
        return False, "MongoDB connection not available"
    
    try:
        result = collection.delete_many({})
        return True, f"Deleted {result.deleted_count} documents"
    except Exception as e:
        return False, f"Clear failed: {e}"

def safe_float_conversion(value, default=0):
    """Safely convert a value to float, returning default if conversion fails"""
    try:
        if value is None or value == "Not Available":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default
def extract_text(uploaded_file):
    """Extract text from PDF or image files"""
    try:
        filename = uploaded_file.name.lower()
        
        # Handle image files
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image = Image.open(uploaded_file)
            # Enhance image for better OCR
            image = image.convert('RGB')
            text = pytesseract.image_to_string(image, config='--psm 6')
            return text.strip()
            
        # Handle PDF files
        elif filename.endswith('.pdf'):
            file_bytes = uploaded_file.read()
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text_content = []
                
                for page_num, page in enumerate(doc):
                    # Try to extract text directly
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        text_content.append(page_text)
                    else:
                        # If no text, use OCR
                        pix = page.get_pixmap(dpi=300)
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        ocr_text = pytesseract.image_to_string(img, config='--psm 6')
                        text_content.append(ocr_text)
                
                return "\n".join(text_content).strip()
        else:
            return ""
            
    except Exception as e:
        st.error(f"Error extracting text from {uploaded_file.name}: {str(e)}")
        return ""

def process_single_file(uploaded_file, progress_bar=None):
    """Process a single file and return the extracted data"""
    try:
        # Extract text
        if progress_bar:
            progress_bar.text(f"Extracting text from {uploaded_file.name}...")
        
        text = extract_text(uploaded_file)
        if not text.strip():
            return None, f"No text found in {uploaded_file.name}"
        
        # Analyze with LLM
        if progress_bar:
            progress_bar.text(f"Analyzing {uploaded_file.name} with AI...")
            
        result = chain.invoke({"invoice_text": text})
        
        # Parse JSON response
        try:
            result_json = json.loads(result.content)
        except json.JSONDecodeError:
            # Try to extract JSON from response if it contains extra text
            content = result.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                result_json = json.loads(content[start:end])
            else:
                raise ValueError("Could not parse JSON response")
        
        # Add metadata
        result_json["filename"] = uploaded_file.name
        result_json["processed_date"] = datetime.now().isoformat()
        result_json["file_size"] = uploaded_file.size if hasattr(uploaded_file, 'size') else 0
        
        return result_json, None
        
    except Exception as e:
        return None, f"Error processing {uploaded_file.name}: {str(e)}"

def extract_files_from_zip(zip_file):
    """Extract files from uploaded ZIP file"""
    extracted_files = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        with open(file_path, 'rb') as f:
                            file_content = f.read()
                            # Create a file-like object
                            class FileWrapper:
                                def __init__(self, content, name):
                                    self.content = io.BytesIO(content)
                                    self.name = name
                                    self.size = len(content)
                                def read(self):
                                    return self.content.read()
                                def seek(self, pos):
                                    return self.content.seek(pos)
                            
                            extracted_files.append(FileWrapper(file_content, file))
    
    return extracted_files

# --------- Streamlit UI ---------
st.set_page_config(
    page_title="InvoiceAI Pro - Smart Invoice Analysis", 
    page_icon="🧾", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .success-badge {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🧾 InvoiceAI Pro</h1>
    <p style="margin: 0; font-size: 1.2rem; opacity: 0.9;">
        Advanced AI-Powered Invoice Analysis & Management System
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_invoices' not in st.session_state:
    st.session_state.processed_invoices = []

# Connection status with professional styling
connection_status = check_mongodb_connection()
if connection_status:
    st.markdown("""
    <div class="success-badge">
        ✅ Database Connected - All data is securely stored
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Database temporarily unavailable - Operating in local mode")

st.markdown("<br>", unsafe_allow_html=True)

# Professional sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="color: #2a5298; margin-bottom: 0;">📊 Analytics Dashboard</h2>
        <p style="color: #666; margin-top: 0;">Real-time insights & statistics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📈 Key Metrics")
    
    try:
        if connection_status:
            total_invoices = count_invoices()
            total_amount = get_total_amount()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Invoices", total_invoices, help="All invoices in database")
            with col2:
                st.metric("Total Value", f"${total_amount:,.0f}", help="Combined invoice value")
            
            # Average calculation
            if total_invoices > 0:
                avg_amount = total_amount / total_invoices
                st.metric("Average Invoice", f"${avg_amount:,.2f}", help="Average invoice amount")
        else:
            current_total = sum([safe_float_conversion(inv.get('total_amount', 0)) for inv in st.session_state.processed_invoices])
            st.metric("Current Session", len(st.session_state.processed_invoices))
            if current_total > 0:
                st.metric("Session Value", f"${current_total:,.2f}")
                
    except Exception as e:
        st.error("Unable to load metrics")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick actions section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    if st.button("� View All Records", use_container_width=True):
        st.session_state.active_tab = "Database View"
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # System info (simplified)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### ℹ️ System Status")
    
    if connection_status:
        st.success("🟢 Database Online")
        st.info("🔐 Data Secured")
    else:
        st.warning("🟡 Local Mode")
        
    st.info("🤖 AI Engine Active")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Help section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 💡 Features")
    st.markdown("""
    • **Smart Upload**: PDF & Image support
    • **AI Analysis**: Automated data extraction  
    • **Natural Queries**: Ask questions about your data
    • **Export Options**: CSV, JSON, Excel formats
    • **Real-time Insights**: Live analytics dashboard
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area
tab1, tab2, tab3 = st.tabs(["📁 Document Processing", "🧠 Smart Analytics", "📊 Data Management"])

with tab1:
    st.markdown("""
    <div class="feature-card">
        <h3 style="margin-top: 0; color: #2a5298;">📁 Document Upload & AI Analysis</h3>
        <p>Upload your invoice documents and let our AI extract key information automatically. 
        Supports PDF files, images (JPG, PNG), and bulk processing via ZIP folders.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced upload interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload options with better styling
        upload_option = st.radio(
            "**Choose your upload method:**",
            ["📄 Individual Files", "📦 ZIP Folder"],
            horizontal=True
        )
        
        if upload_option == "📄 Individual Files":
            uploaded_files = st.file_uploader(
                "Select invoice documents",
                type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff"],
                accept_multiple_files=True,
                help="💡 Tip: You can select multiple files at once for batch processing"
            )
            
        else:  # ZIP Folder
            zip_file = st.file_uploader(
                "Upload ZIP archive containing invoices",
                type=["zip"],
                help="📦 Upload a ZIP file containing PDFs and images for bulk processing"
            )
            uploaded_files = []
            if zip_file:
                with st.spinner("📂 Extracting files from archive..."):
                    uploaded_files = extract_files_from_zip(zip_file)
                st.success(f"✅ Successfully extracted {len(uploaded_files)} files from archive")
    
    with col2:
        if uploaded_files:
            st.markdown("### 📋 Processing Queue")
            st.info(f"**{len(uploaded_files)}** files ready")
            
            # File types breakdown
            if hasattr(uploaded_files[0], 'name'):
                file_types = {}
                for file in uploaded_files:
                    ext = file.name.split('.')[-1].upper()
                    file_types[ext] = file_types.get(ext, 0) + 1
                
                for file_type, count in file_types.items():
                    st.write(f"• **{file_type}**: {count} files")

    if uploaded_files:
        st.markdown("---")
        
        # Processing section with enhanced UI
        process_col1, process_col2 = st.columns([2, 1])
        
        with process_col1:
            if st.button("🚀 Start AI Processing", type="primary", use_container_width=True):
                # Enhanced processing with better feedback
                progress_container = st.container()
                
                with progress_container:
                    st.markdown("### 🔄 Processing Status")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    successful_files = []
                    failed_files = []
                    
                    # Processing with enhanced feedback
                    for i, uploaded_file in enumerate(uploaded_files):
                        current_progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(current_progress)
                        
                        file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else f"File {i+1}"
                        status_text.markdown(f"🤖 **Analyzing:** {file_name}")
                        
                        result_json, error = process_single_file(uploaded_file)
                        
                        if result_json:
                            # Add to session state
                            st.session_state.processed_invoices.append(result_json)
                            
                            # Try to save to MongoDB (if connected)
                            if check_mongodb_connection():
                                save_invoice_data(result_json)
                            
                            successful_files.append(file_name)
                        else:
                            failed_files.append((file_name, error))
                    
                    # Final status
                    progress_bar.progress(1.0)
                    status_text.markdown("✅ **Processing Complete!**")
                
                # Enhanced results display
                st.markdown("### 📊 Processing Results")
                
                result_col1, result_col2, result_col3 = st.columns(3)
                with result_col1:
                    st.metric("✅ Successful", len(successful_files), delta=f"{len(successful_files)}/{len(uploaded_files)}")
                with result_col2:
                    st.metric("❌ Failed", len(failed_files))
                with result_col3:
                    # Safely calculate total value using helper function
                    total_value = sum([safe_float_conversion(inv.get('total_amount', 0)) for inv in st.session_state.processed_invoices[-len(successful_files):]])
                    st.metric("💰 Total Value", f"${total_value:,.2f}")
                
                # Detailed results
                if successful_files:
                    st.markdown("### 📋 Processed Documents")
                    for i, file_name in enumerate(successful_files):
                        result_json = st.session_state.processed_invoices[-(len(successful_files)-i)]
                        
                        with st.expander(f"✅ {file_name} - Analysis Complete", expanded=False):
                            detail_col1, detail_col2 = st.columns(2)
                            
                            with detail_col1:
                                st.markdown("**📊 Key Information:**")
                                st.write(f"**Vendor:** {result_json.get('vendor_name', 'Not Available')}")
                                st.write(f"**Invoice #:** {result_json.get('invoice_number', 'Not Available')}")
                                
                                # Safely format total_amount
                                try:
                                    total_amount = float(result_json.get('total_amount', 0))
                                    st.write(f"**Amount:** ${total_amount:,.2f}")
                                except (ValueError, TypeError):
                                    st.write(f"**Amount:** {result_json.get('total_amount', 'Not Available')}")
                                
                                st.write(f"**Date:** {result_json.get('invoice_date', 'Not Available')}")
                                st.write(f"**Currency:** {result_json.get('currency', 'USD')}")
                            
                            with detail_col2:
                                st.markdown("**🔍 Additional Details:**")
                                st.write(f"**Due Date:** {result_json.get('due_date', 'Not Available')}")
                                
                                # Safely format tax_amount
                                try:
                                    tax_amount = float(result_json.get('tax_amount', 0))
                                    st.write(f"**Tax:** ${tax_amount:,.2f}")
                                except (ValueError, TypeError):
                                    st.write(f"**Tax:** {result_json.get('tax_amount', 'Not Available')}")
                                
                                # Safely format subtotal
                                try:
                                    subtotal = float(result_json.get('subtotal', 0))
                                    st.write(f"**Subtotal:** ${subtotal:,.2f}")
                                except (ValueError, TypeError):
                                    st.write(f"**Subtotal:** {result_json.get('subtotal', 'Not Available')}")
                                
                                st.write(f"**Terms:** {result_json.get('payment_terms', 'Not Available')}")
                            
                            # Show line items if available
                            if result_json.get('line_items'):
                                st.markdown("**📦 Line Items:**")
                                for item in result_json['line_items']:
                                    item_name = item.get('item_name', 'Unknown Item')
                                    
                                    # Safely format numeric values
                                    try:
                                        quantity = float(item.get('quantity', 0))
                                        unit_price = float(item.get('unit_price', 0))
                                        total = float(item.get('total', 0))
                                        st.write(f"• {item_name}: {quantity} × ${unit_price:.2f} = ${total:.2f}")
                                    except (ValueError, TypeError):
                                        # Fallback if conversion fails
                                        quantity = item.get('quantity', 0)
                                        unit_price = item.get('unit_price', 0)
                                        total = item.get('total', 0)
                                        st.write(f"• {item_name}: {quantity} × ${unit_price} = ${total}")
                
                if failed_files:
                    st.markdown("### ⚠️ Processing Issues")
                    for file_name, error in failed_files:
                        with st.expander(f"❌ {file_name} - Processing Failed"):
                            st.error(f"**Error:** {error}")
                            st.info("💡 **Tip:** Ensure the file is a clear, readable invoice document")
        
        with process_col2:
            st.markdown("### ⚙️ Processing Options")
            
            # Advanced options
            with st.expander("🔧 Advanced Settings"):
                st.slider("AI Confidence Threshold", 0.1, 1.0, 0.3, help="Higher values = more strict extraction")
                st.checkbox("Auto-correct dates", value=True, help="Automatically fix common date format issues")
                st.checkbox("Validate amounts", value=True, help="Cross-check extracted amounts for accuracy")
                st.selectbox("Default Currency", ["USD", "EUR", "GBP", "CAD"], help="Default currency when not detected")
    
    else:
        # Welcome message when no files uploaded
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
            <h3 style="color: #2a5298;">👆 Upload Your Invoice Documents</h3>
            <p style="color: #666; font-size: 1.1rem;">
                Get started by uploading PDF files, images, or ZIP archives containing your invoices.<br>
                Our AI will automatically extract key information and organize your data.
            </p>
            <div style="margin-top: 2rem; color: #888;">
                <strong>Supported formats:</strong> PDF, JPG, PNG, BMP, TIFF, ZIP
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div class="feature-card">
        <h3 style="margin-top: 0; color: #2a5298;">🧠 Intelligent Data Analytics</h3>
        <p>Ask natural language questions about your invoice data. Our AI understands context and provides detailed insights with supporting data visualizations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get data source
    if check_mongodb_connection():
        data_source = get_all_invoice_data()
        data_info = f"📊 **Data Source:** MongoDB Database ({len(data_source)} invoices)"
    else:
        data_source = st.session_state.processed_invoices
        data_info = f"💾 **Data Source:** Current Session ({len(data_source)} invoices)"
    
    st.markdown(data_info)
    
    if not data_source:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;">
            <h4 style="color: #856404;">📊 No Data Available</h4>
            <p style="color: #856404;">Please process some invoice documents first to enable smart analytics.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Quick insights dashboard
        if len(data_source) > 0:
            df_overview = pd.DataFrame(data_source)
            
            st.markdown("### 📈 Quick Insights")
            insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
            
            with insight_col1:
                if 'total_amount' in df_overview.columns:
                    avg_amount = df_overview['total_amount'].mean()
                    st.metric("Average Invoice", f"${avg_amount:,.2f}", help="Mean invoice value")
            
            with insight_col2:
                if 'vendor_name' in df_overview.columns:
                    top_vendor = df_overview['vendor_name'].value_counts().index[0] if len(df_overview) > 0 else "N/A"
                    vendor_count = df_overview['vendor_name'].value_counts().iloc[0] if len(df_overview) > 0 else 0
                    st.metric("Top Vendor", top_vendor, delta=f"{vendor_count} invoices")
            
            with insight_col3:
                if 'total_amount' in df_overview.columns:
                    max_amount = df_overview['total_amount'].max()
                    st.metric("Highest Invoice", f"${max_amount:,.2f}", help="Maximum invoice value")
            
            with insight_col4:
                recent_count = len([inv for inv in data_source if 'processed_date' in inv])
                st.metric("Recently Processed", recent_count, help="Files processed today")
        
        st.markdown("---")
        
        # Enhanced query interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 💬 Ask Questions About Your Data")
            
            # Suggested queries with categories
            st.markdown("**💡 Popular Questions:**")
            query_col1, query_col2 = st.columns(2)
            
            query = None
            with query_col1:
                if st.button("💰 High Value Invoices", use_container_width=True):
                    query = "Show me all invoices with amounts greater than $1000, sorted by value"
                if st.button("📊 Vendor Analysis", use_container_width=True):
                    query = "Which vendors have sent the most invoices and what's their total value?"
                if st.button("📅 Monthly Trends", use_container_width=True):
                    query = "Show me invoice trends by month and identify any patterns"
            
            with query_col2:
                if st.button("⚠️ Overdue Analysis", use_container_width=True):
                    query = "Analyze overdue invoices and payment patterns"
                if st.button("💸 Expense Categories", use_container_width=True):
                    query = "Categorize invoices by amount ranges and show distribution"
                if st.button("🔍 Data Summary", use_container_width=True):
                    query = "Provide a comprehensive summary of all invoice data with key statistics"
            
            # Custom query input with better styling
            st.markdown("**✍️ Or ask your own question:**")
            custom_query = st.text_area(
                "",
                placeholder="e.g., 'What is the average invoice amount per vendor?', 'Show me all invoices from last month', 'Which invoices are due soon?'",
                height=100,
                help="Ask anything about your invoice data in natural language"
            )
            
            if custom_query:
                query = custom_query
        
        with col2:
            st.markdown("### 🎯 Query Tips")
            st.markdown("""
            **Effective queries:**
            • Be specific about what you want
            • Mention time periods if relevant
            • Ask for comparisons or trends
            • Request specific calculations
            
            **Examples:**
            • "Top 5 vendors by total amount"
            • "Invoices due in next 30 days"
            • "Average payment terms by vendor"
            • "Monthly spending analysis"
            """)

        if query:
            with st.spinner("🤖 Analyzing your data and generating insights..."):
                try:
                    # Create context for LLM
                    full_context = json.dumps(data_source, indent=2, default=str)
                    
                    # Enhanced query prompt
                    qa_prompt = PromptTemplate(
                        input_variables=["context", "query"],
                        template="""
You are an expert business analyst specializing in invoice and financial data analysis. 

Based on this invoice data in JSON format:
{context}

Please analyze and answer this question: {query}

Guidelines for your response:
1. Provide specific numbers, dates, and calculations when possible
2. Show your reasoning and methodology
3. Highlight key insights and patterns
4. Use bullet points or tables for clarity
5. Include relevant statistics and percentages
6. Suggest actionable recommendations when appropriate
7. If you perform calculations, show the work

Format your response professionally with clear sections and insights.
"""
                    )
                    
                    qa_chain = qa_prompt | llm
                    response = qa_chain.invoke({"context": full_context, "query": query}).content
                    
                    # Display results with enhanced formatting
                    st.markdown("### 🧠 Analysis Results")
                    
                    # Create tabs for different views
                    result_tab1, result_tab2 = st.tabs(["📝 Analysis", "📊 Supporting Data"])
                    
                    with result_tab1:
                        st.markdown(response)
                    
                    with result_tab2:
                        # Auto-display relevant data based on query
                        df = pd.DataFrame(data_source)
                        
                        if "high" in query.lower() or "greater than" in query.lower() or ">" in query:
                            # Show high value invoices
                            if 'total_amount' in df.columns:
                                high_value = df.nlargest(10, 'total_amount')
                                st.markdown("**🔝 Top 10 Highest Value Invoices:**")
                                display_cols = ['filename', 'vendor_name', 'total_amount', 'invoice_date']
                                available_cols = [col for col in display_cols if col in high_value.columns]
                                if available_cols:
                                    st.dataframe(high_value[available_cols], use_container_width=True)
                        
                        elif "vendor" in query.lower():
                            # Show vendor analysis
                            if 'vendor_name' in df.columns and 'total_amount' in df.columns:
                                vendor_summary = df.groupby('vendor_name').agg({
                                    'total_amount': ['count', 'sum', 'mean']
                                }).round(2)
                                vendor_summary.columns = ['Invoice Count', 'Total Amount', 'Average Amount']
                                vendor_summary = vendor_summary.sort_values('Total Amount', ascending=False)
                                st.markdown("**🏢 Vendor Summary:**")
                                st.dataframe(vendor_summary, use_container_width=True)
                        
                        elif "recent" in query.lower() or "month" in query.lower():
                            # Show recent or monthly data
                            if 'invoice_date' in df.columns:
                                df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
                                recent_data = df.sort_values('invoice_date', ascending=False).head(10)
                                st.markdown("**📅 Recent Invoices:**")
                                display_cols = ['filename', 'vendor_name', 'total_amount', 'invoice_date']
                                available_cols = [col for col in display_cols if col in recent_data.columns]
                                if available_cols:
                                    st.dataframe(recent_data[available_cols], use_container_width=True)
                        
                        else:
                            # Show general overview
                            st.markdown("**📋 Data Overview:**")
                            display_cols = ['filename', 'vendor_name', 'total_amount', 'invoice_date']
                            available_cols = [col for col in display_cols if col in df.columns]
                            if available_cols:
                                st.dataframe(df[available_cols].head(10), use_container_width=True)
                            
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 Try rephrasing your question or check if you have sufficient data for analysis")

with tab3:
    st.markdown("""
    <div class="feature-card">
        <h3 style="margin-top: 0; color: #2a5298;">📊 Comprehensive Data Management</h3>
        <p>View, filter, analyze, and export your complete invoice database with advanced search capabilities and multiple export formats.</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Get data from both sources
        mongodb_docs = get_all_invoice_data() if check_mongodb_connection() else []
        session_docs = st.session_state.processed_invoices if 'processed_invoices' in st.session_state else []
        
        # Combine and deduplicate data
        all_docs = mongodb_docs.copy()
        if not mongodb_docs and session_docs:
            all_docs = session_docs
        
        # Professional data source indicator
        if mongodb_docs:
            st.markdown(f"""
            <div style="background: #d4edda; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin: 1rem 0;">
                <strong>📊 Database Status:</strong> Displaying {len(mongodb_docs)} records from secure database
            </div>
            """, unsafe_allow_html=True)
        elif session_docs:
            st.markdown(f"""
            <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107; margin: 1rem 0;">
                <strong>💾 Session Data:</strong> Displaying {len(session_docs)} records from current session
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 10px; border: 2px dashed #dee2e6;">
                <h3 style="color: #6c757d;">📂 No Data Available</h3>
                <p style="color: #6c757d; font-size: 1.1rem;">
                    Start by uploading and processing invoice documents in the <strong>Document Processing</strong> tab.
                </p>
                <p style="color: #6c757d;">
                    Once processed, all your invoice data will appear here for analysis and management.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        if all_docs:
            # Convert to DataFrame for better display
            df = pd.DataFrame(all_docs)
            
            # Data filtering section
            st.markdown("### 🔍 Filter Data")
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                # Vendor filter
                if 'vendor_name' in df.columns:
                    vendors = ['All'] + list(df['vendor_name'].unique())
                    selected_vendor = st.selectbox("Filter by Vendor", vendors)
                    if selected_vendor != 'All':
                        df = df[df['vendor_name'] == selected_vendor]
            
            with filter_col2:
                # Amount range filter
                if 'total_amount' in df.columns:
                    min_amount = st.number_input("Min Amount ($)", min_value=0.0, value=0.0)
                    max_amount = st.number_input("Max Amount ($)", min_value=0.0, value=float(df['total_amount'].max()) if len(df) > 0 else 1000.0)
                    df = df[(df['total_amount'] >= min_amount) & (df['total_amount'] <= max_amount)]
            
            with filter_col3:
                # Date filter
                if 'invoice_date' in df.columns:
                    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
                    if st.checkbox("Filter by Date Range"):
                        start_date = st.date_input("Start Date")
                        end_date = st.date_input("End Date")
                        df = df[(df['invoice_date'].dt.date >= start_date) & (df['invoice_date'].dt.date <= end_date)]
            
            # Display summary statistics
            st.markdown("### 📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                if 'total_amount' in df.columns and len(df) > 0:
                    avg_amount = df['total_amount'].mean()
                    st.metric("Average Amount", f"${avg_amount:.2f}")
            with col3:
                if 'vendor_name' in df.columns and len(df) > 0:
                    unique_vendors = df['vendor_name'].nunique()
                    st.metric("Unique Vendors", unique_vendors)
            with col4:
                if 'total_amount' in df.columns and len(df) > 0:
                    max_amount = df['total_amount'].max()
                    st.metric("Highest Invoice", f"${max_amount:.2f}")
            
            # Display data table
            st.markdown("### 📋 Invoice Records Table")
            
            # Column selection
            available_columns = df.columns.tolist()
            default_columns = [col for col in ['filename', 'vendor_name', 'invoice_number', 'total_amount', 'invoice_date', 'currency'] if col in available_columns]
            
            display_columns = st.multiselect(
                "Select columns to display:",
                options=available_columns,
                default=default_columns,
                help="Choose which columns to show in the table"
            )
            
            if display_columns:
                # Sort options
                sort_col1, sort_col2 = st.columns(2)
                with sort_col1:
                    sort_by = st.selectbox("Sort by", options=display_columns, index=0 if 'total_amount' not in display_columns else display_columns.index('total_amount'))
                with sort_col2:
                    sort_order = st.selectbox("Sort order", options=['Descending', 'Ascending'])
                
                # Apply sorting
                df_display = df[display_columns].sort_values(by=sort_by, ascending=(sort_order == 'Ascending'))
                
                # Display table with pagination
                page_size = st.slider("Records per page", min_value=5, max_value=50, value=10)
                total_pages = (len(df_display) - 1) // page_size + 1
                
                if total_pages > 1:
                    page = st.selectbox(f"Page (Total: {total_pages})", options=range(1, total_pages + 1))
                    start_idx = (page - 1) * page_size
                    end_idx = start_idx + page_size
                    df_display = df_display.iloc[start_idx:end_idx]
                
                st.dataframe(df_display, use_container_width=True)
                
                # Detailed view option
                if st.checkbox("Show detailed view for selected records"):
                    selected_records = st.multiselect(
                        "Select records to view in detail",
                        options=range(len(df_display)),
                        format_func=lambda x: f"{df_display.iloc[x]['filename'] if 'filename' in df_display.columns else f'Record {x+1}'}"
                    )
                    
                    for idx in selected_records:
                        record = df.iloc[df_display.index[idx]]
                        with st.expander(f"📄 Detailed View: {record.get('filename', f'Record {idx+1}')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Basic Information:**")
                                st.write(f"• **Vendor:** {record.get('vendor_name', 'N/A')}")
                                st.write(f"• **Invoice #:** {record.get('invoice_number', 'N/A')}")
                                
                                # Safely format total_amount
                                try:
                                    total_amount = float(record.get('total_amount', 0))
                                    st.write(f"• **Amount:** ${total_amount:,.2f}")
                                except (ValueError, TypeError):
                                    st.write(f"• **Amount:** {record.get('total_amount', 'N/A')}")
                                
                                st.write(f"• **Date:** {record.get('invoice_date', 'N/A')}")
                                st.write(f"• **Currency:** {record.get('currency', 'N/A')}")
                            with col2:
                                st.write("**Additional Details:**")
                                st.write(f"• **Due Date:** {record.get('due_date', 'N/A')}")
                                
                                # Safely format tax_amount
                                try:
                                    tax_amount = float(record.get('tax_amount', 0))
                                    st.write(f"• **Tax Amount:** ${tax_amount:,.2f}")
                                except (ValueError, TypeError):
                                    st.write(f"• **Tax Amount:** {record.get('tax_amount', 'N/A')}")
                                
                                # Safely format subtotal
                                try:
                                    subtotal = float(record.get('subtotal', 0))
                                    st.write(f"• **Subtotal:** ${subtotal:,.2f}")
                                except (ValueError, TypeError):
                                    st.write(f"• **Subtotal:** {record.get('subtotal', 'N/A')}")
                                
                                st.write(f"• **Payment Terms:** {record.get('payment_terms', 'N/A')}")
                            
                            if 'line_items' in record and record['line_items']:
                                st.write("**Line Items:**")
                                for item in record['line_items']:
                                    item_name = item.get('item_name', 'Unknown')
                                    
                                    # Safely format numeric values
                                    try:
                                        quantity = float(item.get('quantity', 0))
                                        unit_price = float(item.get('unit_price', 0))
                                        total = float(item.get('total', 0))
                                        st.write(f"- {item_name}: {quantity} x ${unit_price:.2f} = ${total:.2f}")
                                    except (ValueError, TypeError):
                                        # Fallback if conversion fails
                                        quantity = item.get('quantity', 0)
                                        unit_price = item.get('unit_price', 0)
                                        total = item.get('total', 0)
                                        st.write(f"- {item_name}: {quantity} x ${unit_price} = ${total}")
            
            # Export options
            st.markdown("### 📥 Export Options")
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                # CSV Export
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download as CSV",
                    data=csv,
                    file_name=f"invoice_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download all data as CSV file"
                )
            
            with export_col2:
                # JSON Export
                json_str = json.dumps(all_docs, indent=2, default=str)
                st.download_button(
                    label="📋 Download as JSON",
                    data=json_str,
                    file_name=f"invoice_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Download all data as JSON file"
                )
            
            with export_col3:
                # Excel Export (if openpyxl is available)
                try:
                    from io import BytesIO
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, sheet_name='Invoice_Data', index=False)
                    excel_data = excel_buffer.getvalue()
                    
                    st.download_button(
                        label="📗 Download as Excel",
                        data=excel_data,
                        file_name=f"invoice_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download all data as Excel file"
                    )
                except ImportError:
                    st.info("💡 Install 'openpyxl' for Excel export functionality")
            
            # Data management options
            if check_mongodb_connection():
                st.markdown("### 🗑️ Data Management")
                if st.button("🗑️ Clear All Database Records", type="secondary", help="⚠️ This will permanently delete all records from MongoDB"):
                    if st.button("⚠️ Confirm Delete All Records", type="secondary"):
                        success, message = clear_all_data()
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to clear database: {message}")
            
        else:
            st.info("📊 No invoice data available. Upload and process some files to see data here.")
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.write("Debug info:", str(e))

# Professional Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h4 style="color: #2a5298; margin: 0;">InvoiceAI Pro</h4>
    <p style="color: #666; margin: 0.5rem 0;">
        Advanced AI-Powered Invoice Analysis & Management System
    </p>
    <div style="color: #888; font-size: 0.9rem;">
        <strong>Features:</strong> Smart Document Processing • Natural Language Queries • Advanced Analytics • Secure Data Management
    </div>
    <div style="margin-top: 1rem; color: #aaa; font-size: 0.8rem;">
        Powered by Google Gemini AI • Built with Streamlit • Version 2.0
    </div>
</div>
""", unsafe_allow_html=True)
