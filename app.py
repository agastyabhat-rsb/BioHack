import time
import streamlit as st
import numpy as np
import joblib
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress RDKit terminal noise
RDLogger.DisableLog('rdApp.*')

# --- UI Configuration ---
st.set_page_config(
    page_title="BioHackAR | ADMET Prediction",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS Injection (Aesthetic Override) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1535 100%);
        background-size: 200% 200%;
    }
    
    h1 {
        color: #00ffff;
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;
        letter-spacing: 4px;
        text-align: center;
        font-weight: 900;
        font-size: 72px !important;
        margin-bottom: 10px;
        line-height: 1.2;
    }
    
    h2 {
        color: #00d4ff;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 32px !important;
        letter-spacing: 1px;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    
    h3 {
        color: #00d4ff;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 26px !important;
        letter-spacing: 1px;
    }
    
    p, label, .stMarkdown, div[data-testid="stMarkdownContainer"] p {
        color: #b8c5d6 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 20px !important;
        line-height: 1.8 !important;
        font-weight: 400 !important;
    }
    
    .info-text {
        color: #e0e6ed !important;
        font-size: 22px !important;
        line-height: 1.9 !important;
    }
    
    .stTextInput > div > div > input {
        background-color: rgba(30, 36, 66, 0.6);
        color: #00ffff !important;
        border: 2px solid #00ffff;
        border-radius: 8px;
        padding: 16px 20px;
        font-family: 'Courier New', monospace;
        font-size: 18px !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
        transition: all 0.3s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00ffff;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.6);
        outline: none;
    }
    
    .stTextInput label {
        color: #e0e6ed !important;
        font-size: 20px !important;
        font-weight: 500 !important;
        margin-bottom: 12px !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #00ffff 0%, #0080ff 100%);
        color: #0a0e27;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        border: none;
        border-radius: 8px;
        padding: 16px 48px;
        font-size: 20px !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
        transition: all 0.3s;
        cursor: pointer;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #00ffff 0%, #00a8ff 100%);
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.8);
        transform: translateY(-2px);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 36px !important;
        color: #00ffff !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 20px !important;
        color: #b8c5d6 !important;
        font-weight: 500 !important;
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(30, 36, 66, 0.9) 0%, rgba(20, 25, 50, 0.9) 100%);
        border: 2px solid #00ffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.4);
    }
    
    .stAlert {
        background-color: rgba(30, 36, 66, 0.8);
        border-left: 5px solid #00ffff;
        color: #e0e6ed !important;
        font-size: 20px !important;
        border-radius: 8px;
        padding: 20px !important;
    }
    
    hr {
        border: none;
        border-top: 2px solid #00ffff;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        margin: 40px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Backend Engine Loading ---
@st.cache_resource
def load_models():
    return joblib.load('admet_models.pkl')

try:
    models = load_models()
except FileNotFoundError:
    st.error("Model file 'admet_models.pkl' not found. Ensure it is committed and pushed.")
    st.stop()
def get_smiles_from_name(query, max_retries=3):
    """Dynamic API resolution with retries and a fallback API."""
    # 1. Sanitize: Remove all spaces for RDKit parsing
    clean_query = "".join(query.strip().split())
    
    # 2. RDKit Test (Post-Sanitization)
    mol = Chem.MolFromSmiles(clean_query)
    if mol is not None:
        return clean_query
        
    # 3. Primary API: PubChem with Retry Logic
    pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query.strip()}/property/IsomericSMILES,CanonicalSMILES/JSON"
    headers = {"User-Agent": "BioHackAR_App/3.0 (agastyabhat-rsb)"}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(pubchem_url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                props = data.get('PropertyTable', {}).get('Properties', [{}])[0]
                return props.get('IsomericSMILES') or props.get('CanonicalSMILES')
                
            elif response.status_code == 404:
                st.warning(f"PubChem Database Miss: Could not find a chemical named '{query.strip()}'")
                return None
                
            elif response.status_code in [502, 503, 504]:
                # Server is temporarily overloaded. Wait and try again.
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Sleeps for 1s, then 2s...
                    continue
                else:
                    st.warning("PubChem API is currently overloaded. Trying fallback database...")
                    break # Exhausted retries, move to fallback
                    
            else:
                st.warning(f"NIH API Error: Status {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            st.warning("NIH PubChem Server timed out. Trying fallback database...")
            break
        except Exception as e:
            st.error(f"Network Exception: {str(e)}")
            return None

    # 4. Fallback API: NCI/CADD Chemical Identifier Resolver
    try:
        fallback_url = f"https://cactus.nci.nih.gov/chemical/structure/{query.strip()}/smiles"
        fallback_response = requests.get(fallback_url, timeout=5)
        
        if fallback_response.status_code == 200:
            return fallback_response.text.strip()
        else:
            st.error(f"Resolution Failed: Neither PubChem nor NCI APIs could resolve '{query}'.")
            return None
            
    except Exception as e:
        st.error(f"Fallback API Error: {str(e)}")
        return None

def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.array(fp, dtype=np.float32)
    
    physical_descriptors = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ], dtype=np.float32)
    
    return np.concatenate((fp_array, physical_descriptors)).reshape(1, -1)

# --- UI Front-End ---
# --- Updated UI Front-End Section ---
st.markdown("<h2>Molecular Input</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 20px; margin-bottom: 25px;'>Enter a Chemical Name (e.g., Aspirin, Ibuprofen, Benadryl) OR a SMILES String:</p>", unsafe_allow_html=True)

# Wrap input and button in a form
with st.form(key="mol_form"):
    user_query = st.text_input(
        "Chemical Input",
        placeholder="Paracetamol",
        label_visibility="collapsed"
    )
    
    # st.form_submit_button ensures the app waits for the click
    predict_btn = st.form_submit_button("Analyze Molecule")

if predict_btn and user_query:
    with st.spinner("Processing molecular structure..."):
        
        smiles = get_smiles_from_name(user_query)
        
        if smiles:
            if smiles != user_query:
                st.info(f"Resolved Name to SMILES: `{smiles}`")
                
            features = extract_features(smiles)
            
            if features is None:
                st.error("Invalid chemical graph. RDKit could not parse the structure.")
            else:
                st.markdown("---")
                st.markdown("<h2>ADMET Prediction Results</h2>", unsafe_allow_html=True)
                
                classifications = {k: v for k, v in models.items() if k != 'Lipophilicity'}
                regression = {k: v for k, v in models.items() if k == 'Lipophilicity'}
                
                metric_desc = {
                    'hia': "Human Intestinal Absorption",
                    'bbb': "Blood-Brain Barrier Penetration",
                    'cyp3a4_sub': "CYP3A4 Substrate (Metabolism)",
                    'carcin': "Carcinogenicity (Toxicity)",
                    'herg': "hERG Toxicity (Cardiac Risk)"
                }
                
                st.markdown("### Classification Targets")
                c_cols = st.columns(3)
                for idx, (target, model) in enumerate(classifications.items()):
                    prob = model.predict_proba(features)[0][1]
                    with c_cols[idx % 3]:
                        st.metric(label=target.upper(), value=f"{prob:.2%}")
                        st.caption(metric_desc.get(target.lower(), "Target parameter analyzed."))
                
                if regression:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### Physical Chemistry Targets")
                    pred = regression['Lipophilicity'].predict(features)[0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Lipophilicity (LogD)", value=f"{pred:.3f}")
                
                st.success("Analysis complete. Predictions generated successfully.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 25px; color: #6a7f9f; font-family: Inter, sans-serif;'>
    <p style='font-size: 18px; margin: 0;'>Powered by AI-driven molecular modeling | BioHackAR © 2026</p>
</div>
""", unsafe_allow_html=True)