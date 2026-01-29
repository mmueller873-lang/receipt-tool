import streamlit as st
import base64
import json
import concurrent.futures
import pandas as pd
from openai import OpenAI, RateLimitError, APITimeoutError
from datetime import datetime
import io
import re
import os

# NEW IMPORTS
from PIL import Image
import PyPDF2
from bs4 import BeautifulSoup

# Try to register HEIF support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False

# ============================================================
# CONFIGURATION & COMMERCIAL STYLING
# ============================================================
st.set_page_config(page_title="Receipt Tool", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 0px;
    }
    .hero-caption {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #34495e;
        border-left: 5px solid #0066cc;
        padding-left: 10px;
        margin-bottom: 15px;
        margin-top: 20px;
    }
    [data-testid="stVegaLiteChart"] { pointer-events: none !important; }
    
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem !important; }
        div.stButton > button { height: 2.5em !important; font-size: 1rem !important; }
        div[data-testid="stDownloadButton"] > button { height: 3em !important; }
    }
    
    div.stButton > button:first-child {
        background-color: #0066cc; color: white; font-weight: 600; border: none; border-radius: 6px; height: 3em; width: 100%; transition: 0.2s;
    }
    div.stButton > button:hover { background-color: #0056b3; transform: translateY(-1px); }
    
    div[data-testid="stDownloadButton"] > button:first-child {
        background-color: #28a745; color: white; font-weight: 700; border: none; border-radius: 6px; height: 3.5em; width: 100%; font-size: 1.1em;
    }
    div.stButton > button:nth-child(2) {
        background-color: #f39c12; color: white; font-weight: 600; border: none; border-radius: 6px; height: 3em; width: 100%;
    }
    [data-testid="stDataFrame"] { border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="hero-title">🧾 Receipt Tool</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-caption">Professional Audit & Categorization</p>', unsafe_allow_html=True)

col_header_left, col_header_right = st.columns([3, 1])
with col_header_right:
    placeholder_dl = st.empty()

api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

# ============================================================
# SESSION STATE
# ============================================================
if 'master_results_df' not in st.session_state:
    st.session_state.master_results_df = None
if 'processing_errors' not in st.session_state:
    st.session_state.processing_errors = []
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'show_uploader' not in st.session_state:
    st.session_state.show_uploader = True

# ============================================================
# PRE-PROCESSING LOGIC (NEW)
# ============================================================

def preprocess_file(file):
    """
    Handles HEIC, PDF, HTML. 
    Returns: {'type': 'image'|'text', 'data': bytes, 'name': str}
    """
    file_bytes = file.read()
    filename = file.name.lower()
    
    # 1. HEIC to JPG
    if filename.endswith('.heic'):
        if not HEIF_SUPPORTED:
            return None, "HEIC not supported on this server"
        try:
            image = Image.open(io.BytesIO(file_bytes))
            rgb_im = image.convert('RGB')
            output = io.BytesIO()
            rgb_im.save(output, format="JPEG")
            return {'type': 'image', 'data': output.getvalue(), 'name': file.name}
        except Exception as e:
            return None, f"HEIC Conversion Error: {str(e)}"

    # 2. PDF Text Extraction
    elif filename.endswith('.pdf'):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            if not text.strip():
                return None, "PDF is a scan (Image-based) - not supported yet."
            return {'type': 'text', 'data': text, 'name': file.name}
        except Exception as e:
            return None, f"PDF Error: {str(e)}"

    # 3. HTML Text Extraction
    elif filename.endswith('.html'):
        try:
            soup = BeautifulSoup(file_bytes, 'html.parser')
            text = soup.get_text(separator="\n")
            return {'type': 'text', 'data': text, 'name': file.name}
        except Exception as e:
            return None, f"HTML Error: {str(e)}"

    # 4. Standard Image (JPG/PNG)
    else:
        return {'type': 'image', 'data': file_bytes, 'name': file.name}

# ============================================================
# AI LOGIC
# ============================================================

def extract_json_safely(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    if "```json" in content:
        content = content.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

def process_single_file(file, key, extract_items_flag):
    try:
        # A. PRE-PROCESS
        preproc = preprocess_file(file)
        if not preproc:
            return None, preproc[1] # Return error
        
        # B. DYNAMIC PROMPT
        if extract_items_flag:
            prompt = """
            Analyze the receipt/data. Extract:
            - Date (YYYY-MM-DD)
            - Vendor
            - Total (number)
            - Category
            - Items (List of objects with 'name', 'qty', 'price')
            JSON: {"date": "...", "vendor": "...", "total": 0.00, "category": "...", "items": [{"name": "...", "qty": 1, "price": 0.00}]}
            """
        else:
            prompt = """
            Extract from image: Date (YYYY-MM-DD), Vendor, Total (number), Category, Description.
            JSON: {"date": "...", "vendor": "...", "total": 0.00, "category": "...", "description": "..."}
            """

        # C. API CALL
        client = OpenAI(api_key=key)
        
        def make_request():
            messages = []
            
            # If Text-based (PDF/HTML)
            if preproc['type'] == 'text':
                messages.append({"role": "user", "content": f"Data:\n{preproc['data'][:10000]}\n\n{prompt}"})
            
            # If Image-based (JPG/HEIC)
            elif preproc['type'] == 'image':
                base64_image = base64.b64encode(preproc['data']).decode('utf-8')
                messages.append({
                    "role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}}
                    ]
                })
            
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

        try:
            response = make_request()
        except (RateLimitError, APITimeoutError, Exception) as e:
            import time
            time.sleep(1)
            response = make_request()
            
        content = response.choices[0].message.content
        data = extract_json_safely(content)
        
        if data:
            data['filename'] = file.name
            return data, None
        else:
            return None, "AI returned invalid text"

    except RateLimitError:
        return None, "API Limit Reached"
    except APITimeoutError:
        return None, "Connection Timed Out"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ============================================================
# ACTIONS BAR
# ============================================================

if st.session_state.master_results_df is not None and not st.session_state.show_uploader:
    st.markdown("<hr>", unsafe_allow_html=True)
    col_act_1, col_act_2, col_act_3, col_act_4 = st.columns([1,1,1,1])
    
    with col_act_1:
        if st.button("➕ Add More Receipts"):
            st.session_state.show_uploader = True
            st.session_state.uploader_key += 1
            st.rerun()
            
    with col_act_4:
        if st.button("🗑️ Start New / Clear Data"):
            st.session_state.master_results_df = None
            st.session_state.processing_errors = []
            st.session_state.show_uploader = True
            st.session_state.uploader_key += 1
            st.rerun()

# ============================================================
# UPLOADER & SETTINGS
# ============================================================

uploaded_files = None

if st.session_state.show_uploader:
    with st.container():
        # Deep Mode Toggle
        extract_items = st.checkbox("🔍 Extract Line Items (Deep Mode)", value=False, help="Extract individual items (Qty/Price) to a separate Excel sheet.")
        
        # Updated types list
        uploaded_files = st.file_uploader(
            "Drag & Drop receipts (JPG, PNG, HEIC, PDF, HTML)", 
            type=['jpg', 'jpeg', 'png', 'heic', 'pdf', 'html'],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}"
        )

# ============================================================
# PROCESSING LOGIC
# ============================================================

if uploaded_files:
    st.info(f"Batch of {len(uploaded_files)} files ready to process.")
    
    if st.button("🚀 Process Batch"):
        results = []
        errors = []
        
        bar = st.progress(0, text="Initializing engine...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_file = {
                executor.submit(process_single_file, file, MY_API_KEY, extract_items): file 
                for file in uploaded_files
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    data, error = future.result()
                    if data:
                        results.append(data)
                    if error:
                        errors.append({"file": file.name, "error": error})
                except Exception as exc:
                    errors.append({"file": file.name, "error": "Unexpected Crash"})
                
                completed += 1
                bar.progress(completed / len(uploaded_files), text=f"Processed {completed}/{len(uploaded_files)}...")

        bar.empty()
        
        # Append to Master Data
        current_batch_df = pd.DataFrame(results)
        
        if st.session_state.master_results_df is not None:
            st.session_state.master_results_df = pd.concat([st.session_state.master_results_df, current_batch_df], ignore_index=True)
        else:
            st.session_state.master_results_df = current_batch_df
            
        st.session_state.processing_errors.extend(errors)
        st.session_state.show_uploader = False
        st.rerun()

# ============================================================
# MAIN RENDER (Persistent)
# ============================================================

if st.session_state.master_results_df is not None:
    df = st.session_state.master_results_df
    
    # Normalize Data
    df['total'] = pd.to_numeric(df['total'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['display_date'] = df['date'].dt.strftime('%Y-%m-%d')
    df = df.sort_values(by=['date', 'total'], ascending=[False, False])

    # 1. DOWNLOAD BUTTON
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Sheet 1: Summary
        excel_cols = ['filename', 'date', 'vendor', 'total', 'category', 'description']
        # If no items extracted, we might just have description, or items list in json
        df.to_excel(writer, index=False, columns=excel_cols, sheet_name='Summary')
        
        # Sheet 2: Line Items (IF EXISTS)
        if 'items' in df.columns:
            items_list = []
            for index, row in df.iterrows():
                if isinstance(row['items'], list):
                    for item in row['items']:
                        items_list.append({
                            'Parent_File': row['filename'],
                            'Item_Name': item.get('name'),
                            'Qty': item.get('qty'),
                            'Price': item.get('price')
                        })
            
            if items_list:
                items_df = pd.DataFrame(items_list)
                items_df.to_excel(writer, index=False, sheet_name='Line Items')
        
        workbook = writer.book
        # Summary Sheet Formatting
        worksheet = writer.sheets['Summary']
        currency_fmt = workbook.add_format({'num_format': '$#,##0.00', 'font_name': 'Arial'})
        date_fmt = workbook.add_format({'num_format': 'yyyy-mm-dd', 'font_name': 'Arial'})
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#2E86C1', 'font_color': 'white', 'font_name': 'Arial'})
        
        worksheet.set_column('A:A', 35) # Filename
        worksheet.set_column('B:B', 15) # Date
        worksheet.set_column('C:C', 30) # Vendor
        worksheet.set_column('D:D', 12) # Total
        worksheet.set_column('E:E', 15) # Category
        worksheet.set_column('F:F', 50) # Description
        
        for col_num, value in enumerate(excel_cols):
            worksheet.write(0, col_num, value.capitalize(), header_fmt)
        
        for row_num in range(1, len(df) + 1):
            for col_idx, col_name in enumerate(excel_cols):
                cell_data = df.iloc[row_num-1][col_name]
                if col_name == 'total' and pd.notna(cell_data):
                    worksheet.write(row_num, col_idx, cell_data, currency_fmt)
                elif col_name == 'date':
                    if pd.notna(df.iloc[row_num-1]['date']):
                        worksheet.write_datetime(row_num, col_idx, df.iloc[row_num-1]['date'], date_fmt)
                    else:
                        worksheet.write(row_num, col_idx, cell_data)
                else:
                    if row_num % 2 == 0:
                        cell_fmt = workbook.add_format({'bg_color': '#EBF5FB', 'font_name': 'Arial'})
                    else:
                        cell_fmt = workbook.add_format({'bg_color': 'white', 'font_name': 'Arial'})
                    worksheet.write(row_num, col_idx, str(cell_data), cell_fmt)
        
        worksheet.autofilter(0, 0, len(df), len(excel_cols) - 1)
        worksheet.freeze_panes(1, 0)

    output.seek(0)
    
    with placeholder_dl.container():
        st.download_button(
            label="📥 Download Excel",
            data=output,
            file_name=f"Receipt_Export_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # 2. METRICS ROW
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Items", len(df))
    with m2:
        st.metric("Total Volume", f"${df['total'].sum():,.2f}")
    with m3:
        avg_trans = df['total'].mean()
        st.metric("Avg. Transaction", f"${avg_trans:,.2f}")

    # 3. CHARTS ROW
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<h3 class="section-header">💵 Spending by Category</h3>', unsafe_allow_html=True)
        st.bar_chart(df.groupby('category')['total'].sum(), use_container_width=False)
    with c2:
        st.markdown('<h3 class="section-header">🧾 Count by Category</h3>', unsafe_allow_html=True)
        st.bar_chart(df['category'].value_counts(), use_container_width=False)

    # 4. DATA TABLE
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("Extraction Complete.")
    
    # Show Description if Deep Mode
    if 'items' in df.columns:
        st.caption("Note: Line Items exported to 'Line Items' sheet in Excel.")
    
    st.dataframe(df[['display_date', 'vendor', 'total', 'category', 'description']], 
                 use_container_width=True, 
                 hide_index=True)

    # 5. ERROR LOG
    if st.session_state.processing_errors:
        with st.expander(f"⚠️ View Errors ({len(st.session_state.processing_errors)})"):
            unique_errors = [dict(t) for t in {tuple(sorted(d.items())) for d in st.session_state.processing_errors}]
            for err in unique_errors:
                st.text(f"📄 {err['file']}: {err['error']}")
