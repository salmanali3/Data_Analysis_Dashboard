import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import base64
import json
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import pdfplumber 
    PDF_SUPPORT = True
except:
    PDF_SUPPORT = False

try:
    import pytesseract
    from PIL import Image
    OCR_SUPPORT = True
except:
    OCR_SUPPORT = False

st.set_page_config(
    page_title="Complete Data Science Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
    }
    .step-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title"> Where Data Learns Intelligence</div>', unsafe_allow_html=True)
st.markdown("""
**Upload ANY file format**  
**Auto convert to CSV**  
**Full data cleaning in progress**  
**Exporting ML-ready dataset**
""")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'processing_steps' not in st.session_state:
    st.session_state.processing_steps = []
if 'file_type' not in st.session_state:
    st.session_state.file_type = None


def get_table_download_link(df, filename="processed_data.csv"):
    """Generate download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
    return href

def add_processing_step(step):
    """Add step to processing history"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.processing_steps.append(f"[{timestamp}] {step}")


def convert_excel_to_csv(uploaded_file):
    """Convert Excel to DataFrame"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=None)
        if isinstance(df, dict):
            
            sheet_names = list(df.keys())
            selected_sheet = st.selectbox("Select sheet:", sheet_names)
            return df[selected_sheet]
        return df
    except Exception as e:
        st.error(f"Error reading Excel: {str(e)}")
        return None

def convert_json_to_csv(uploaded_file):
    """Convert JSON to DataFrame"""
    try:
        content = uploaded_file.read().decode('utf-8')
        data = json.loads(content)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
        
            if all(isinstance(v, dict) for v in data.values()):
                df = pd.DataFrame.from_dict(data, orient='index')
            else:
                df = pd.DataFrame([data])
        else:
            st.error("Unsupported JSON structure")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading JSON: {str(e)}")
        return None

def convert_text_to_csv(uploaded_file, delimiter=None):
    """Convert text file to DataFrame"""
    try:
        content = uploaded_file.read().decode('utf-8')
        
        
        if delimiter is None:
            for delim in [',', ';', '\t', '|']:
                if delim in content[:1000]:
                    delimiter = delim
                    break
            if delimiter is None:
                delimiter = ','  
        
    
        df = pd.read_csv(StringIO(content), delimiter=delimiter)
        return df
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return None

def convert_pdf_to_csv(uploaded_file):
    """Extract tables from PDF"""
    if not PDF_SUPPORT:
        st.error("PDF extraction requires pdfplumber. Install: pip install pdfplumber")
        return None
    
    try:
        import pdfplumber
        tables = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table:
                        df_page = pd.DataFrame(table[1:], columns=table[0])
                        tables.append(df_page)
        
        if tables:
            
            df = pd.concat(tables, ignore_index=True)
            return df
        else:
            st.warning("No tables found in PDF")
            return None
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return None

def convert_image_to_csv(uploaded_file):
    """Extract text from image using OCR"""
    if not OCR_SUPPORT:
        st.error("OCR requires pytesseract and PIL. Install: pip install pytesseract pillow")
        return None
    
    try:
        from PIL import Image
        import pytesseract
        
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        
        
        lines = text.split('\n')
        data = []
        for line in lines:
            if line.strip():
                
                for delim in [',', ';', '\t', '|', '  ']:
                    if delim in line:
                        data.append([cell.strip() for cell in line.split(delim)])
                        break
                else:
                    data.append([line.strip()])
        
        if data:
            df = pd.DataFrame(data)
            return df
        else:
            st.warning("No structured data found in image")
            return None
    except Exception as e:
        st.error(f"Error with OCR: {str(e)}")
        return None

def convert_sql_to_csv(sql_query, db_file=None):
    """Convert SQL query or database to DataFrame"""
    try:
        if db_file is not None:
            
            conn = sqlite3.connect(db_file)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
        else:
        
            st.info("SQL database connection not implemented in demo. Using sample data.")
            df = pd.DataFrame({
                'id': range(1, 11),
                'name': [f'User_{i}' for i in range(1, 11)],
                'value': np.random.randn(10)
            })
        return df
    except Exception as e:
        st.error(f"Error with SQL: {str(e)}")
        return None


def auto_clean_data(df):
    """Automatic data cleaning pipeline"""
    original_shape = df.shape
    cleaning_report = []
    
    
    df_clean = df.copy()
    
    
    df_clean = df_clean.dropna(how='all')
    df_clean = df_clean.loc[:, df_clean.notna().any()]
    

    for col in df_clean.columns:
        
        try:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
        except:
            pass
        
        
        try:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='ignore')
        except:
            pass
    
    
    missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
    for col in missing_cols:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
            cleaning_report.append(f"Filled missing values in '{col}' with median")
        else:
        
            if not df_clean[col].mode().empty:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                cleaning_report.append(f"Filled missing values in '{col}' with mode")
            else:
                df_clean[col].fillna('Unknown', inplace=True)
                cleaning_report.append(f"Filled missing values in '{col}' with 'Unknown'")
    

    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        cleaning_report.append(f"Removed {duplicates} duplicate rows")
    

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        if outliers > 0 and outliers < len(df_clean) * 0.1:  # Only if <10% outliers
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            cleaning_report.append(f"Removed {outliers} outliers from '{col}'")
    
    final_shape = df_clean.shape
    rows_removed = original_shape[0] - final_shape[0]
    cols_removed = original_shape[1] - final_shape[1]
    
    cleaning_report.append(f"Original shape: {original_shape}")
    cleaning_report.append(f"Final shape: {final_shape}")
    cleaning_report.append(f"Rows removed: {rows_removed}, Columns removed: {cols_removed}")
    
    return df_clean, cleaning_report

def feature_engineering(df):
    """Automatic feature engineering"""
    df_fe = df.copy()
    feature_report = []
    
    date_cols = df_fe.select_dtypes(include=['datetime64']).columns
    for col in date_cols:
        df_fe[f'{col}_year'] = df_fe[col].dt.year
        df_fe[f'{col}_month'] = df_fe[col].dt.month
        df_fe[f'{col}_day'] = df_fe[col].dt.day
        df_fe[f'{col}_dayofweek'] = df_fe[col].dt.dayofweek
        feature_report.append(f"Created date features from '{col}'")
    
    categorical_cols = df_fe.select_dtypes(include=['object']).columns
    for col in categorical_cols[:3]: 
        if df_fe[col].nunique() <= 10: 
            dummies = pd.get_dummies(df_fe[col], prefix=col, drop_first=True)
            df_fe = pd.concat([df_fe, dummies], axis=1)
            feature_report.append(f"One-hot encoded '{col}' ({df_fe[col].nunique()} categories)")
    
    numeric_cols = df_fe.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        df_fe[f'{col1}_x_{col2}'] = df_fe[col1] * df_fe[col2]
        df_fe[f'{col1}_div_{col2}'] = df_fe[col1] / (df_fe[col2] + 1e-10)  # Avoid division by zero
        feature_report.append(f"Created interaction features between '{col1}' and '{col2}'")
    

    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        df_fe[f'{col}_squared'] = df_fe[col] ** 2
        df_fe[f'{col}_sqrt'] = np.sqrt(np.abs(df_fe[col]))
        feature_report.append(f"Created polynomial features for '{col}'")
    
    return df_fe, feature_report

def prepare_ml_data(df, target_column=None):
    """Prepare data for machine learning"""
    df_ml = df.copy()
    ml_report = []
    

    if target_column and target_column in df_ml.columns:
        X = df_ml.drop(columns=[target_column])
        y = df_ml[target_column]
        ml_report.append(f"Separated target column: '{target_column}'")
    else:
        X = df_ml
        y = None
        ml_report.append("No target column specified - unsupervised learning mode")
    

    categorical_cols = X.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        
        for col in categorical_cols:
            if X[col].nunique() <= 50: 
                X[col] = le.fit_transform(X[col].astype(str))
                ml_report.append(f"Label encoded '{col}'")
            else:
            
                X = X.drop(columns=[col])
                ml_report.append(f"Dropped '{col}' (too many categories: {X[col].nunique()})")
    
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        ml_report.append(f"Scaled {len(numeric_cols)} numeric columns")
    else:
        X_scaled = X
    
    
    if X_scaled.isnull().any().any():
        X_scaled = X_scaled.fillna(X_scaled.median())
        ml_report.append("Filled remaining missing values with median")
    
    
    if y is not None:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) < 10 else None
        )
        ml_report.append(f"Created train-test split (80-20)")
        ml_report.append(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, ml_report
    else:
        return X_scaled, ml_report



st.sidebar.title("Data Upload Center")
st.sidebar.markdown("---")

file_type = st.sidebar.selectbox(
    "Select File Type:",
    ["CSV", "Excel (.xlsx, .xls)", "JSON", "Text (.txt)", "PDF", "Image", "SQL", "Sample Data"]
)

df = None

if file_type == "CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.file_type = "CSV"
        add_processing_step(f"Uploaded CSV file: {uploaded_file.name}")

elif file_type == "Excel (.xlsx, .xls)":
    uploaded_file = st.sidebar.file_uploader("Upload Excel", type=['xlsx', 'xls'])
    if uploaded_file:
        df = convert_excel_to_csv(uploaded_file)
        st.session_state.file_type = "Excel"
        add_processing_step(f"Converted Excel to CSV: {uploaded_file.name}")

elif file_type == "JSON":
    uploaded_file = st.sidebar.file_uploader("Upload JSON", type=['json'])
    if uploaded_file:
        df = convert_json_to_csv(uploaded_file)
        st.session_state.file_type = "JSON"
        add_processing_step(f"Converted JSON to CSV: {uploaded_file.name}")

elif file_type == "Text (.txt)":
    uploaded_file = st.sidebar.file_uploader("Upload Text", type=['txt'])
    if uploaded_file:
        delimiter = st.sidebar.selectbox("Select delimiter:", [",", ";", "\t", "|", "Space"])
        delim_map = {"Space": " "}
        df = convert_text_to_csv(uploaded_file, delim_map.get(delimiter, delimiter))
        st.session_state.file_type = "Text"
        add_processing_step(f"Converted Text to CSV: {uploaded_file.name}")

elif file_type == "PDF" and PDF_SUPPORT:
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=['pdf'])
    if uploaded_file:
        df = convert_pdf_to_csv(uploaded_file)
        st.session_state.file_type = "PDF"
        if df is not None:
            add_processing_step(f"Extracted table from PDF: {uploaded_file.name}")

elif file_type == "Image" and OCR_SUPPORT:
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        df = convert_image_to_csv(uploaded_file)
        st.session_state.file_type = "Image"
        if df is not None:
            add_processing_step(f"Extracted text from image: {uploaded_file.name}")

elif file_type == "SQL":
    sql_option = st.sidebar.radio("SQL Source:", ["Upload SQLite DB", "Enter SQL Query"])
    
    if sql_option == "Upload SQLite DB":
        uploaded_file = st.sidebar.file_uploader("Upload SQLite DB", type=['db', 'sqlite'])
        if uploaded_file:
            query = st.sidebar.text_area("SQL Query:", "SELECT * FROM table_name")
            if st.sidebar.button("Execute Query"):
                df = convert_sql_to_csv(query, uploaded_file)
                st.session_state.file_type = "SQL"
                add_processing_step(f"Executed SQL query on database")
    
    else:
        query = st.sidebar.text_area("Enter SQL Query:", "SELECT 1 as id, 'test' as name")
        if st.sidebar.button("Execute Query"):
            df = convert_sql_to_csv(query)
            st.session_state.file_type = "SQL"
            add_processing_step(f"Executed SQL query")

elif file_type == "Sample Data":
    sample_option = st.sidebar.selectbox(
        "Choose Sample Dataset:",
        ["Iris", "Titanic", "Boston Housing", "Diabetes", "Random Data"]
    )
    
    if st.sidebar.button("Load Sample Data"):
        if sample_option == "Iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif sample_option == "Titanic":
            df = pd.read_csv("https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv")
        elif sample_option == "Boston Housing":
            from sklearn.datasets import load_boston
            data = load_boston()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        elif sample_option == "Diabetes":
            from sklearn.datasets import load_diabetes
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        else:
            
            np.random.seed(42)
            df = pd.DataFrame({
                'feature1': np.random.randn(100),
                'feature2': np.random.rand(100),
                'category': np.random.choice(['A', 'B', 'C'], 100),
                'target': np.random.randint(0, 2, 100)
            })
        
        st.session_state.file_type = "Sample"
        add_processing_step(f"Loaded sample dataset: {sample_option}")


if df is not None:
    st.session_state.original_df = df.copy()
    st.session_state.df = df.copy()



st.sidebar.markdown("---")
st.sidebar.info("Upload any file type, it will be automatically converted to CSV format for analysis.")

# Main content
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Show file info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", numeric_cols)
    with col4:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)
    
    # Tabs for different operations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Data Preview", 
        "Auto Clean", 
        "Feature Engineering", 
        "ML Preparation", 
        "Visualize", 
        "Export"
    ])
    
    # TAB 1: Data Preview
    with tab1:
        st.subheader("Data Preview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Data preview
            preview_option = st.radio("Preview:", ["First 10 rows", "Last 10 rows", "Random 10 rows"])
            if preview_option == "First 10 rows":
                st.dataframe(df.head(10), use_container_width=True)
            elif preview_option == "Last 10 rows":
                st.dataframe(df.tail(10), use_container_width=True)
            else:
                st.dataframe(df.sample(10), use_container_width=True)
        
        with col2:
            # Data info
            st.subheader("Data Information")
            buffer = StringIO()
            df.info(buf=buffer)
            info_text = buffer.getvalue()
            st.text_area("Data Info:", info_text, height=300)
    
    # TAB 2: Auto Clean
    with tab2:
        st.subheader("Automatic Data Cleaning")
        
        st.markdown("""
        <div class="step-box">
        <h4>Auto-Cleaning Pipeline:</h4>
        1. Remove empty rows/columns<br>
        2. Fix data types<br>
        3. Handle missing values<br>
        4. Remove duplicates<br>
        5. Handle outliers
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Run Auto-Cleaning", type="primary"):
            with st.spinner("Cleaning data..."):
                df_clean, report = auto_clean_data(df)
                
                # Update session state
                st.session_state.df = df_clean
                add_processing_step("Applied auto-cleaning pipeline")
                
                # Show results
                st.success("Data cleaning completed!")
                
                # Show report
                st.subheader("Cleaning Report:")
                for item in report:
                    st.write(f"• {item}")
                
                # Show comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Before Cleaning:**")
                    st.write(f"Shape: {df.shape}")
                    st.write(f"Missing: {df.isnull().sum().sum()}")
                    st.write(f"Duplicates: {df.duplicated().sum()}")
                
                with col2:
                    st.write("**After Cleaning:**")
                    st.write(f"Shape: {df_clean.shape}")
                    st.write(f"Missing: {df_clean.isnull().sum().sum()}")
                    st.write(f"Duplicates: {df_clean.duplicated().sum()}")
    
    # TAB 3: Feature Engineering
    with tab3:
        st.subheader("Feature Engineering")
        
        st.markdown("""
        <div class="step-box">
        <h4>Auto-Feature Engineering:</h4>
        1. Date feature extraction<br>
        2. Categorical encoding<br>
        3. Interaction features<br>
        4. Polynomial features
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate Features", type="primary"):
            with st.spinner("Creating features..."):
                df_features, report = feature_engineering(df)
                
                # Update session state
                st.session_state.df = df_features
                add_processing_step("Applied feature engineering")
                
                # Show results
                st.success(f"Created {len(df_features.columns) - len(df.columns)} new features!")
                
                # Show report
                st.subheader("Feature Engineering Report:")
                for item in report:
                    st.write(f"• {item}")
                
                # Show new columns
                new_cols = set(df_features.columns) - set(df.columns)
                if new_cols:
                    st.write("**New Features Created:**")
                    st.write(list(new_cols))
    
    # TAB 4: ML Preparation
    with tab4:
        st.subheader("Machine Learning Preparation")
        
        # Select target column
        target_col = st.selectbox(
            "Select Target Column (for supervised learning):",
            ["None"] + list(df.columns)
        )
        
        st.markdown("""
        <div class="step-box">
        <h4>ML Preparation Pipeline:</h4>
        1. Separate features & target<br>
        2. Handle categorical data<br>
        3. Scale features<br>
        4. Train-test split
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Prepare for ML", type="primary"):
            with st.spinner("Preparing data for ML..."):
                if target_col != "None":
                    X_train, X_test, y_train, y_test, report = prepare_ml_data(df, target_col)
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    add_processing_step(f"Prepared data for ML with target: {target_col}")
                    
                    # Show results
                    st.success("Data ready for machine learning!")
                    
                    # Show report
                    st.subheader("ML Preparation Report:")
                    for item in report:
                        st.write(f"• {item}")
                    
                    # Show data shapes
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("X_train shape", str(X_train.shape))
                    with col2:
                        st.metric("X_test shape", str(X_test.shape))
                    with col3:
                        st.metric("y_train shape", str(y_train.shape if hasattr(y_train, 'shape') else len(y_train)))
                    with col4:
                        st.metric("y_test shape", str(y_test.shape if hasattr(y_test, 'shape') else len(y_test)))
                    
                    # Show sample of prepared data
                    st.subheader("Sample of Prepared Features (X_train):")
                    st.dataframe(X_train.head(), use_container_width=True)
                
                else:
                    X, report = prepare_ml_data(df)
                    st.session_state.X_unsupervised = X
                    add_processing_step("Prepared data for unsupervised ML")
                    
                    st.success("Data ready for unsupervised learning!")
                    
                    st.subheader("ML Preparation Report:")
                    for item in report:
                        st.write(f"• {item}")
                    
                    st.metric("Processed Features shape", str(X.shape))
                    
                    st.subheader("Sample of Prepared Data:")
                    st.dataframe(X.head(), use_container_width=True)
    
    # TAB 5: Visualize
    with tab5:
        st.subheader("Data Visualization")
        
        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["Distribution Plot", "Correlation Heatmap", "Scatter Plot", "Box Plot", "Count Plot"]
        )
        
        if viz_type == "Distribution Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = st.selectbox("Select column:", numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {col}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        elif viz_type == "Correlation Heatmap":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Correlation Heatmap')
                st.pyplot(fig)
        
        elif viz_type == "Scatter Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis:", numeric_cols)
                y_col = st.selectbox("Y-axis:", numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df[x_col], df[y_col], alpha=0.6)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'{x_col} vs {y_col}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        elif viz_type == "Box Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = st.selectbox("Select column:", numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.boxplot(df[col].dropna())
                ax.set_ylabel(col)
                ax.set_title(f'Box Plot of {col}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        elif viz_type == "Count Plot":
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                col = st.selectbox("Select categorical column:", categorical_cols)
                fig, ax = plt.subplots(figsize=(10, 6))
                df[col].value_counts().head(10).plot(kind='bar', ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.set_title(f'Top 10 Categories in {col}')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
    
    # TAB 6: Export
    with tab6:
        st.subheader("Export Processed Data")
        
        export_format = st.radio(
            "Select Export Format:",
            ["CSV", "Excel", "JSON", "Pickle (for Python)"]
        )
        
        # Processing history
        st.subheader("Processing History")
        if st.session_state.processing_steps:
            for step in st.session_state.processing_steps:
                st.write(f"• {step}")
        else:
            st.write("No processing steps yet.")
        
        # Export buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if export_format == "CSV":
                st.markdown(get_table_download_link(df, "processed_data.csv"), unsafe_allow_html=True)
            elif export_format == "Excel":
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='ProcessedData')
                b64 = base64.b64encode(output.getvalue()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="processed_data.xlsx"> Download Excel File</a>'
                st.markdown(href, unsafe_allow_html=True)
            elif export_format == "JSON":
                json_str = df.to_json(orient='records', indent=2)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="processed_data.json"> Download JSON File</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:  # Pickle
                import pickle
                pickle_bytes = pickle.dumps(df)
                b64 = base64.b64encode(pickle_bytes).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="processed_data.pkl"> Download Pickle File</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # Export ML data if prepared
            if 'X_train' in st.session_state:
                st.success("ML Data Available for Export!")
                
                ml_data = {
                    'X_train': st.session_state.X_train,
                    'X_test': st.session_state.X_test,
                    'y_train': st.session_state.y_train,
                    'y_test': st.session_state.y_test
                }
                
                import pickle
                ml_pickle = pickle.dumps(ml_data)
                b64_ml = base64.b64encode(ml_pickle).decode()
                href_ml = f'<a href="data:application/octet-stream;base64,{b64_ml}" download="ml_ready_data.pkl"> Download ML Ready Data</a>'
                st.markdown(href_ml, unsafe_allow_html=True)
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset to Original Data"):
        if st.session_state.original_df is not None:
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.processing_steps = []
            st.rerun()

else:
    # Welcome screen
    st.markdown("""
    <div class="feature-card">
    <h3>Where Raw Data Learns Intelligence with AI</h3>
    
    <h4><strong>SUPPORTED FILE TYPES:</strong></h4>
    • <strong>CSV</strong>    Comma Separated Values<br>
    • <strong>Excel</strong>  .xlsx, .xls files<br>
    • <strong>JSON</strong>   JavaScript Object Notation<br>
    • <strong>Text</strong>   .txt files with any delimiter<br>
    • <strong>PDF</strong>    Extract tables from PDFs<br>
    • <strong>Images</strong> Extract text using OCR<br>
    • <strong>SQL</strong>    SQLite databases or queries<br>
    • <strong>Sample Data</strong>  Builtin datasets for testing
    </div>
    
    <div class="feature-card">
    <h4><strong>COMPLETE DATA PIPELINE:</strong></h4>
    1. <strong>Auto Conversion</strong>  Any file  CSV<br>
    2. <strong>Smart Cleaning</strong>  Missing values, duplicates, outliers<br>
    3. <strong>Feature Engineering</strong>  Automatic feature creation<br>
    4. <strong>ML Preparation</strong>  Ready for Scikit-learn/TensorFlow<br>
    5. <strong>Visualization</strong>  Interactive plots<br>
    6. <strong>Export</strong>  Multiple format support
    </div>
    
    <div class="warning-box">
    <h4><strong>HOW TO USE:</strong></h4>
    1. Select file type from sidebar<br>
    2. Upload your file<br>
    3. Navigate through tabs for different operations<br>
    4. Download processed data when ready<br>
    5. Your data is now ML ready!
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>Where Raw Data Learns Intelligence with AI</strong>
             \n
             Upload  Clean  Engineer  Visualize  Export
    Supports all file formats | Auto ML preparation | One-click data cleaning
</div>
""", unsafe_allow_html=True)