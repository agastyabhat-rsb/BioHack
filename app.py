import streamlit as st
import numpy as np
import joblib
import requests
import difflib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress RDKit terminal noise
RDLogger.DisableLog('rdApp.*')

# --- Page Config ---
st.set_page_config(
    page_title="BioHackAR | ADMET Prediction",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS Injection ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: #0a0e27;
    }
    
    /* Hero Section with Chemical Image Background */
    .hero-section {
        padding: 100px 20px;
        text-align: center;
        background-image: url("https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Ibuprofen_structure.svg/1024px-Ibuprofen_structure.svg.png");
        background-size: contain;
        background-position: center;
        background-repeat: no-repeat;
        background-blend-mode: screen;
        background-color: rgba(10, 14, 39, 0.92); /* Dark overlay to keep text readable */
        border-bottom: 2px solid #00ffff;
        margin-bottom: 50px;
        border-radius: 15px;
    }
    
    .hero-title {
        font-family: 'Orbitron', monospace !important;
        font-size: 110px !important;
        font-weight: 900 !important;
        color: #ffffff !important;
        text-shadow: 0 0 20px #00ffff, 0 0 40px #00ffff, 0 0 80px #00ffff !important;
        margin-bottom: 20px;
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif !important;
        font-size: 32px !important;
        color: #00d4ff !important;
        font-weight: 500 !important;
        letter-spacing: 2px;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Orbitron', monospace !important;
        font-size: 50px !important;
        color: #00ffff !important;
        text-align: center !important;
        margin-top: 80px !important;
        margin-bottom: 40px !important;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
    }
    
    /* Form & Input Styling */
    .stTextInput > div > div > input {
        background-color: rgba(30, 36, 66, 0.8);
        color: #00ffff !important;
        border: 2px solid #00ffff;
        border-radius: 10px;
        padding: 20px;
        font-family: 'Courier New', monospace;
        font-size: 22px !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
    }
    
    .stFormSubmitButton > button {
        background: linear-gradient(90deg, #00ffff 0%, #0080ff 100%);
        color: #0a0e27 !important;
        font-weight: 900 !important;
        font-family: 'Inter', sans-serif;
        border: none;
        border-radius: 10px;
        padding: 15px 40px;
        font-size: 24px !important;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.6);
        width: 100%;
        margin-top: 20px;
    }
    
    .stMetric {
        background: rgba(20, 25, 50, 0.9);
        border: 2px solid #00ffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
    }
    /* Force Native Metric Colors */
    [data-testid="stMetricValue"] {
        font-size: 38px !important;
        color: #00ffff !important;
        font-weight: 900 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 18px !important;
        color: #b8c5d6 !important;
        font-weight: 600 !important;
        letter-spacing: 1px;
    }
    
    /* Caption explanations */
    [data-testid="stCaptionContainer"] {
        color: #6a7f9f !important;
        font-size: 14px !important;
        line-height: 1.4 !important;
        margin-top: -10px !important;
        margin-bottom: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Backend Engine Loading ---
@st.cache_resource
def load_models():
    try:
        return joblib.load('admet_models.pkl')
    except FileNotFoundError:
        return None

models = load_models()

# --- The Offline Dictionary & Resolution Engine ---
def get_smiles_from_name(query):
    query = query.strip()
    
    # 1. ATTEMPT SMILES PARSE
    clean_smiles = "".join(query.split())
    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is not None:
        return clean_smiles
        
    # 2. OFFLINE FUZZY DICTIONARY (Bypasses NIH Firewall)
    local_db = {
        "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "benadryl": "CN(C)CCOC(C1=CC=CC=C1)C2=CC=CC=C2",
        "paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
        "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    }
    
    clean_query = query.lower()
    matches = difflib.get_close_matches(clean_query, local_db.keys(), n=1, cutoff=0.7)
    if matches:
        return local_db[matches[0]]
        
    # 3. API FALLBACK
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/property/IsomericSMILES,CanonicalSMILES/JSON"
        headers = {"User-Agent": "BioHackAR_App/4.0"}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            props = data.get('PropertyTable', {}).get('Properties', [{}])[0]
            return props.get('IsomericSMILES') or props.get('CanonicalSMILES')
    except:
        return None
    return None

def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.array(fp, dtype=np.float32)
    physical_descriptors = np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
    ], dtype=np.float32)
    return np.concatenate((fp_array, physical_descriptors)).reshape(1, -1)

# ==================== SECTION 1: HERO ====================
st.markdown("""
<div class='hero-section'>
    <h1 class='hero-title'>BioHackAR</h1>
    <p class='hero-subtitle'>AI-Powered ADMET Property Prediction System</p>
</div>
""", unsafe_allow_html=True)

# ==================== SECTION 2: INFORMATION ====================
st.markdown("<h2 class='section-header'>General Information</h2>", unsafe_allow_html=True)

# Native Streamlit rendering (bulletproof across all browsers)
st.info("**What is a SMILES String?**\nA SMILES string (Simplified Molecular Input Line Entry System) is a compact, text-based shorthand format used to describe 2D chemical structures. Instead of storing complex 3D coordination files, a molecule is represented as a single line of standard text characters (e.g., **CCO** for Ethanol). This format serves as the foundational input for cheminformatics software, allowing computers to instantly map atomic bonds, rings, and branches, and convert them into mathematical inputs for machine learning models.")

st.info("""**What is ADMET?**
ADMET stands for Absorption, Distribution, Metabolism, Excretion, and Toxicity. It is a core framework in pharmacology that evaluates how a living organism interacts with a foreign chemical compound.
* **Absorption:** How the drug gets into the body (e.g., crossing the intestinal wall).
* **Distribution:** Where the drug goes in the body (e.g., staying in the blood or crossing into the brain).
* **Metabolism:** How the body chemically breaks the drug down.
* **Excretion:** How the body eliminates the waste products.
* **Toxicity:** Whether the drug causes harmful side effects to organs, tissues, or DNA.""")

st.info("""**Why It Is Necessary to Analyze These Endpoints**
Analyzing ADMET endpoints simultaneously is critical because biological efficacy is completely useless without safety and deliverability. Historically, roughly 40% of drug candidates failed during clinical trials simply because they had poor ADMET profiles.

**1. Preventing Blind Spots (The Multi-Failure Risk)**
A molecule cannot be evaluated on a single metric alone. If you design a compound that perfectly cures a disease in a petri dish, it will still fail as a medicine if it has poor Absorption, bad Distribution, or high Toxicity.

**2. Saving Immense Time and Capital**
Synthesizing chemicals in a physical laboratory and testing them on animals costs thousands of dollars per compound and takes weeks of work. Running an automated machine learning screening pipeline allows you to evaluate millions of virtual SMILES strings in seconds.

**3. Guiding Structural Refinement**
By mapping multiple endpoints together (like Lipophilicity alongside Bioavailability), the data tells you how to fix a failing molecule. If your pipeline flags a compound as having poor absorption due to high water-solubility, a medicinal chemist can use that specific feedback to modify the SMILES string.""")

# ==================== SECTION 3: ANALYZER ====================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header' style='color: #00ffff;'>Molecular Analyzer</h2>", unsafe_allow_html=True)

if models is None:
    st.error("CRITICAL ERROR: 'admet_models.pkl' missing from server. Model inference disabled.")
else:
    with st.form(key="mol_form"):
        smiles_input = st.text_input(
            "Enter a Chemical Name (e.g., Aspirin, Ibuprofen, Benadryl) OR a SMILES String:",
            placeholder="Aspirin",
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_btn = st.form_submit_button("ANALYZE MOLECULE")

    if predict_btn and smiles_input:
        with st.spinner("Processing molecular structure..."):
            st.markdown("<br>", unsafe_allow_html=True)
            
            smiles = get_smiles_from_name(smiles_input)
            
            if smiles:
                if smiles != "".join(smiles_input.split()):
                    st.success(f"🔍 Resolved Input to SMILES: `{smiles}`")
                    
                features = extract_features(smiles)
                
                if features is None:
                    st.error("Invalid chemical graph. RDKit could not parse the structure.")
                else:
                    classifications = {k: v for k, v in models.items() if k != 'Lipophilicity'}
                    regression = {k: v for k, v in models.items() if k == 'Lipophilicity'}
                    
                    metric_desc = {
                        'hia': "Probability of absorption through the human intestine.",
                        'bbb': "Probability of penetrating the blood-brain barrier.",
                        'cyp3a4_sub': "Probability of metabolization by the CYP3A4 liver enzyme.",
                        'carcin': "Probability of inducing cellular mutation or cancer.",
                        'herg': "Probability of blocking cardiac channels (arrhythmia risk)."
                    }
                    
                    # Fix: Use pure HTML to avoid Streamlit Markdown parser collisions
                    st.markdown("<h3 style='color: #00d4ff; font-family: Inter, sans-serif; margin-bottom: 20px;'>Classification Targets</h3>", unsafe_allow_html=True)
                    
                    c_cols = st.columns(3)
                    for idx, (target, model) in enumerate(classifications.items()):
                        prob = model.predict_proba(features)[0][1]
                        with c_cols[idx % 3]:
                            st.metric(label=target.upper(), value=f"{prob:.2%}")
                            st.caption(metric_desc.get(target.lower(), "Target parameter analyzed."))
                    
                    if regression:
                        st.markdown("<br><h3 style='color: #00d4ff; font-family: Inter, sans-serif; margin-bottom: 20px;'>Physical Chemistry Targets</h3>", unsafe_allow_html=True)
                        pred = regression['Lipophilicity'].predict(features)[0]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(label="Lipophilicity (LogD)", value=f"{pred:.3f}")
                            st.caption("Fat vs. water solubility. Optimal oral range is typically 1.0 to 3.0.")
            else:
                st.error("Could not parse name. Ensure spelling is correct.")

st.markdown("<br><br><div style='text-align: center; color: #6a7f9f;'>Powered by AI-driven molecular modeling | BioHackAR © 2026</div>", unsafe_allow_html=True)