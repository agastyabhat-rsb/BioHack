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
    page_title="BioHackAR | ADMET Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS Injection (Aesthetic Override) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1535 100%);
        background-size: 200% 200%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    h1 {
        color: #00ffff;
        font-family: 'Orbitron', monospace;
        text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;
        letter-spacing: 3px;
        text-align: center;
        font-weight: 900;
        margin-bottom: 30px;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 5px #00ffff, 0 0 10px #00ffff; }
        to { text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff; }
    }
    
    h2, h3 {
        color: #00d4ff;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    p, label, .stMarkdown {
        color: #b8c5d6;
        font-family: 'Rajdhani', sans-serif;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input {
        background-color: rgba(30, 36, 66, 0.8);
        color: #00ffff;
        border: 2px solid #00ffff;
        border-radius: 10px;
        padding: 12px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
        transition: all 0.3s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #ff00ff;
        box-shadow: 0 0 25px rgba(255, 0, 255, 0.5);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #00ffff 0%, #0080ff 100%);
        color: #0a0e27;
        font-weight: bold;
        font-family: 'Orbitron', monospace;
        border: none;
        border-radius: 10px;
        padding: 12px 40px;
        font-size: 16px;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
        transition: all 0.3s;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff00ff 0%, #00ffff 100%);
        box-shadow: 0 0 30px rgba(255, 0, 255, 0.8), 0 0 40px rgba(0, 255, 255, 0.5);
        transform: scale(1.05);
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(30, 36, 66, 0.9) 0%, rgba(20, 25, 50, 0.9) 100%);
        border: 2px solid #00ffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.4);
        transition: all 0.3s;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 30px rgba(0, 255, 255, 0.6);
    }
    
    .stSpinner > div {
        border-color: #00ffff;
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
def get_smiles_from_name(query):
    """Pure dynamic API resolution leveraging RDKit validation."""
    query = query.strip()
    
    # 1. The Native RDKit Test
    # If RDKit can build a molecule graph from the string, it is already a valid SMILES.
    mol = Chem.MolFromSmiles(query)
    if mol is not None:
        return query
        
    # 2. The Pure API Fallback
    # If RDKit fails, assume it is a common name and ping the NIH database dynamically.
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/property/IsomericSMILES,CanonicalSMILES/JSON"
        headers = {"User-Agent": "BioHackAR_App/2.0 (agastyabhat-rsb)"}
        
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            props = data.get('PropertyTable', {}).get('Properties', [{}])[0]
            # Prioritize Isomeric (3D stereochemistry), fallback to Canonical
            return props.get('IsomericSMILES') or props.get('CanonicalSMILES')
            
        elif response.status_code == 404:
            st.warning(f"PubChem Database Miss: Could not find a chemical named '{query}'")
            return None
        else:
            st.warning(f"NIH API Error: Status {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("NIH PubChem Server timed out. The API is currently unreachable.")
        return None
    except Exception as e:
        st.error(f"Network Exception: {str(e)}")
        return None
# --- UI Front-End ---
st.markdown("""
<div style='text-align: center; padding: 20px; margin-bottom: 30px;'>
    <h1 style='margin: 0; font-size: 42px;'>🧬 BioHackAR</h1>
    <p style='color: #00d4ff; font-size: 20px; font-weight: 500;'>Advanced ADMET Property Prediction Engine</p>
</div>
""", unsafe_allow_html=True)

# 1. The Educational Context
st.markdown("""
### What is this tool?
Before a new pharmaceutical drug can be approved, scientists must ensure it behaves safely in the human body. This validation process is called **ADMET** (Absorption, Distribution, Metabolism, Excretion, and Toxicity). 

Instead of spending years and millions of dollars testing chemical compounds in a physical laboratory, this AI architecture analyzes the geometric spatial graph of a molecule and mathematically predicts its biological interactions instantly.
""")
st.markdown("---")

# 2. The Smart Input Handler
user_query = st.text_input("Enter a Chemical Name (e.g., Aspirin, Ibuprofen, Benadryl) OR a SMILES String:", "Aspirin")

if st.button("🚀 Analyze Molecule"):
    with st.spinner("🔄 Interfacing with PubChem and mapping molecular topology..."):
        
        # 3. Dynamic Resolution
        smiles = get_smiles_from_name(user_query)
        
        if not smiles:
            st.error(f"Could not resolve '{user_query}' to a valid chemical structure. Please check the spelling or provide a direct SMILES string.")
        else:
            if smiles != user_query:
                st.info(f"🔍 **Resolved Name to SMILES:** `{smiles}`")
                
            features = extract_features(smiles)
            
            if features is None:
                st.error("Invalid chemical graph. RDKit could not parse the structure.")
            else:
                st.success("✅ Topology mapped. Inference complete.")
                st.markdown("---")
                
                # 4. Semantic Data Split with Contextual Interpretations
                classifications = {k: v for k, v in models.items() if k != 'Lipophilicity'}
                regression = {k: v for k, v in models.items() if k == 'Lipophilicity'}
                
                metric_desc = {
                    'hia': "Human Intestinal Absorption: The probability the drug successfully enters the bloodstream through the gut.",
                    'bbb': "Blood-Brain Barrier: The probability the drug can penetrate the brain (critical for neurological drugs, bad for others).",
                    'cyp3a4_sub': "CYP3A4 Substrate: The probability it is broken down by the primary liver enzyme.",
                    'carcin': "Carcinogenicity: The probability the compound induces cellular mutation (cancer risk).",
                    'herg': "hERG Toxicity: The probability of blocking cardiac channels, causing fatal heart arrhythmias."
                }
                
                st.markdown("### 📊 Classification Targets")
                c_cols = st.columns(3)
                for idx, (target, model) in enumerate(classifications.items()):
                    prob = model.predict_proba(features)[0][1]
                    with c_cols[idx % 3]:
                        st.metric(label=target.upper(), value=f"{prob:.2%}")
                        st.caption(metric_desc.get(target.lower(), "Target parameter analyzed."))
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if regression:
                    st.markdown("### 📈 Physical Chemistry Targets")
                    pred = regression['Lipophilicity'].predict(features)[0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Lipophilicity (LogD)", value=f"{pred:.3f}")
                        st.caption("Measures how well the drug dissolves in fats vs. water. Optimal oral drugs generally score between 1.0 and 3.0.")