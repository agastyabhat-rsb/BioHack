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

# Page config
st.set_page_config(
    page_title="BioHackAR | ADMET Prediction",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Complete CSS with 3-section design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Remove default padding */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }
    
    /* Main app background */
    .stApp {
        background: #0a0e27;
    }
    
    /* Section 1: Hero with chemical background */
    .hero-section {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1535 100%);
        background-image: 
            url("data:image/svg+xml,%3Csvg width='100' height='100' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='20' cy='20' r='3' fill='%2300ffff' opacity='0.1'/%3E%3Ccircle cx='60' cy='40' r='2' fill='%2300ffff' opacity='0.15'/%3E%3Ccircle cx='80' cy='70' r='2.5' fill='%2300d4ff' opacity='0.1'/%3E%3Cline x1='20' y1='20' x2='60' y2='40' stroke='%2300ffff' opacity='0.08' stroke-width='1'/%3E%3Cline x1='60' y1='40' x2='80' y2='70' stroke='%2300ffff' opacity='0.08' stroke-width='1'/%3E%3C/svg%3E"),
            url("data:image/svg+xml,%3Csvg width='120' height='120' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='30' cy='80' r='2' fill='%2300d4ff' opacity='0.12'/%3E%3Ccircle cx='90' cy='30' r='3' fill='%2300ffff' opacity='0.1'/%3E%3Ccircle cx='70' cy='90' r='2' fill='%2300ffff' opacity='0.15'/%3E%3Cline x1='30' y1='80' x2='70' y2='90' stroke='%2300d4ff' opacity='0.08' stroke-width='1'/%3E%3C/svg%3E");
        background-size: 400px 400px, 500px 500px;
        background-position: 10% 20%, 80% 60%;
        background-repeat: no-repeat;
        position: relative;
        padding: 60px 40px;
        margin: 0 -2rem;
    }
    
    .hero-title {
        font-family: 'Orbitron', monospace;
        font-size: 96px;
        font-weight: 900;
        color: #00ffff;
        text-align: center;
        text-shadow: 
            0 0 20px rgba(0, 255, 255, 0.8),
            0 0 40px rgba(0, 255, 255, 0.5),
            0 0 60px rgba(0, 255, 255, 0.3);
        letter-spacing: 8px;
        margin: 0;
        animation: heroGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes heroGlow {
        from { 
            text-shadow: 
                0 0 20px rgba(0, 255, 255, 0.6),
                0 0 40px rgba(0, 255, 255, 0.4),
                0 0 60px rgba(0, 255, 255, 0.2);
        }
        to { 
            text-shadow: 
                0 0 30px rgba(0, 255, 255, 1),
                0 0 60px rgba(0, 255, 255, 0.7),
                0 0 90px rgba(0, 255, 255, 0.4);
        }
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 28px;
        color: #00d4ff;
        text-align: center;
        margin-top: 30px;
        font-weight: 500;
        letter-spacing: 2px;
    }
    
    /* Section 2: Information section */
    .info-section {
        min-height: 100vh;
        background: linear-gradient(180deg, #0f1535 0%, #1a1f3a 50%, #0a0e27 100%);
        padding: 80px 60px;
        margin: 0 -2rem;
        border-top: 3px solid rgba(0, 255, 255, 0.3);
        border-bottom: 3px solid rgba(0, 255, 255, 0.3);
        position: relative;
    }
    
    .info-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            repeating-linear-gradient(90deg, rgba(0, 255, 255, 0.03) 0px, transparent 1px, transparent 60px, rgba(0, 255, 255, 0.03) 61px),
            repeating-linear-gradient(0deg, rgba(0, 255, 255, 0.03) 0px, transparent 1px, transparent 60px, rgba(0, 255, 255, 0.03) 61px);
        pointer-events: none;
    }
    
    .section-title {
        font-family: 'Orbitron', monospace;
        font-size: 48px;
        font-weight: 700;
        color: #00ffff;
        margin-bottom: 20px;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.6);
        letter-spacing: 3px;
    }
    
    .section-heading {
        font-family: 'Inter', sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: #00d4ff;
        margin-top: 50px;
        margin-bottom: 20px;
        letter-spacing: 1px;
    }
    
    .info-text {
        font-family: 'Inter', sans-serif;
        font-size: 20px;
        line-height: 1.9;
        color: #c8d5e6;
        margin-bottom: 30px;
    }
    
    .info-text strong {
        color: #00ffff;
        font-weight: 600;
    }
    
    .info-card {
        background: rgba(30, 36, 66, 0.4);
        border-left: 4px solid #00ffff;
        border-radius: 12px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .admet-list {
        margin-left: 20px;
        margin-top: 20px;
    }
    
    .admet-list li {
        font-size: 20px;
        line-height: 2;
        color: #c8d5e6;
        margin-bottom: 12px;
    }
    
    .admet-list strong {
        color: #00ffff;
        font-weight: 600;
    }
    
    /* Section 3: Analyzer section */
    .analyzer-section {
        min-height: 100vh;
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        padding: 80px 60px 60px 60px;
        margin: 0 -2rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: rgba(30, 36, 66, 0.6);
        color: #00ffff !important;
        border: 2px solid #00ffff;
        border-radius: 10px;
        padding: 18px 24px;
        font-family: 'Courier New', monospace;
        font-size: 20px !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        transition: all 0.3s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00ffff;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.6);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(0, 212, 255, 0.5);
    }
    
    /* Button styling */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(90deg, #00ffff 0%, #0080ff 100%);
        color: #0a0e27;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        border: none;
        border-radius: 10px;
        padding: 18px 60px;
        font-size: 22px !important;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.6);
        transition: all 0.3s;
        cursor: pointer;
        letter-spacing: 2px;
        text-transform: uppercase;
        width: 100%;
    }
    
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: linear-gradient(90deg, #00ffff 0%, #00a8ff 100%);
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.9);
        transform: translateY(-3px);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 42px !important;
        color: #00ffff !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 22px !important;
        color: #c8d5e6 !important;
        font-weight: 600 !important;
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(30, 36, 66, 0.8) 0%, rgba(20, 25, 50, 0.8) 100%);
        border: 2px solid #00ffff;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.4);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 40px 20px;
        color: #6a7f9f;
        font-family: 'Inter', sans-serif;
        font-size: 18px;
        margin-top: 60px;
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

def get_smiles_from_name(query):
    # 1. ATTEMPT SMILES PARSE (Strictly stripped of spaces)
    clean_smiles = "".join(query.split())
    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is not None:
        return clean_smiles
        
    # 2. API LOOKUP (Name remains untouched with spaces)
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query.strip()}/property/IsomericSMILES,CanonicalSMILES/JSON"
        headers = {"User-Agent": "BioHackAR_App/3.0 (agastyabhat-rsb)"}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            props = data.get('PropertyTable', {}).get('Properties', [{}])[0]
            return props.get('IsomericSMILES') or props.get('CanonicalSMILES')
    except Exception as e:
        st.error(f"Resolution failed: {e}")
        return None
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

# ==================== SECTION 1: HERO ====================
st.markdown("""
<div class='hero-section'>
    <div>
        <h1 class='hero-title'>BioHackAR</h1>
        <p class='hero-subtitle'>AI-Powered ADMET Property Prediction System</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== SECTION 2: INFORMATION ====================
st.markdown("""
<div class='info-section'>
    <h2 class='section-title'>General Information</h2>
    
    <div class='info-card'>
        <h3 class='section-heading'>What is a SMILES String?</h3>
        <p class='info-text'>
            A SMILES string (Simplified Molecular Input Line Entry System) is a compact, text-based shorthand format used to describe 2D chemical structures. Instead of storing complex 3D coordination files, a molecule is represented as a single line of standard text characters (e.g., <strong>CCO</strong> for Ethanol). This format serves as the foundational input for cheminformatics software, allowing computers to instantly map atomic bonds, rings, and branches, and convert them into mathematical inputs for machine learning models.
        </p>
    </div>
    
    <div class='info-card'>
        <h3 class='section-heading'>What is ADMET?</h3>
        <p class='info-text'>
            ADMET stands for <strong>Absorption, Distribution, Metabolism, Excretion, and Toxicity</strong>. It is a core framework in pharmacology that evaluates how a living organism interacts with a foreign chemical compound.
        </p>
        <ul class='admet-list'>
            <li><strong>Absorption:</strong> How the drug gets into the body (e.g., crossing the intestinal wall).</li>
            <li><strong>Distribution:</strong> Where the drug goes in the body (e.g., staying in the blood or crossing into the brain).</li>
            <li><strong>Metabolism:</strong> How the body chemically breaks the drug down (typically via liver enzymes).</li>
            <li><strong>Excretion:</strong> How the body eliminates the waste products (typically via the kidneys or bile).</li>
            <li><strong>Toxicity:</strong> Whether the drug causes harmful side effects to organs, tissues, or DNA.</li>
        </ul>
    </div>
    
    <div class='info-card'>
        <h3 class='section-heading'>Why It Is Necessary to Analyze These Endpoints</h3>
        <p class='info-text'>
            Analyzing ADMET endpoints simultaneously is critical because biological efficacy is completely useless without safety and deliverability. Historically, roughly <strong>40% of drug candidates failed during clinical trials</strong> simply because they had poor ADMET profiles.
        </p>
        <p class='info-text'>
            Evaluating these endpoints early provides several distinct advantages:
        </p>
        
        <h4 class='section-heading' style='font-size: 24px; margin-top: 30px;'>1. Preventing Blind Spots (The Multi-Failure Risk)</h4>
        <p class='info-text'>
            A molecule cannot be evaluated on a single metric alone. If you design a compound that perfectly cures a disease in a petri dish, it will still fail as a medicine if:
        </p>
        <ul class='admet-list'>
            <li>It has poor <strong>Absorption</strong>, meaning it cannot be absorbed as an oral pill.</li>
            <li>It has bad <strong>Distribution</strong>, getting trapped in fat tissue instead of reaching the target organ.</li>
            <li>It has high <strong>Toxicity</strong>, causing fatal side effects like heart failure or tumors.</li>
        </ul>
        
        <h4 class='section-heading' style='font-size: 24px; margin-top: 30px;'>2. Saving Immense Time and Capital</h4>
        <p class='info-text'>
            Synthesizing chemicals in a physical laboratory and testing them on animals costs thousands of dollars per compound and takes weeks of work. Running an automated machine learning screening pipeline allows you to evaluate millions of virtual SMILES strings in seconds. You can eliminate thousands of hazardous or unabsorbable molecules before a scientist ever touches a single glass beaker.
        </p>
        
        <h4 class='section-heading' style='font-size: 24px; margin-top: 30px;'>3. Guiding Structural Refinement</h4>
        <p class='info-text'>
            By mapping multiple endpoints together (like Lipophilicity alongside Bioavailability), the data tells you how to fix a failing molecule. If your pipeline flags a compound as having poor absorption due to high water-solubility, a medicinal chemist can use that specific feedback to modify the SMILES string—adding a small carbon chain to make it more lipid-soluble so it can successfully cross biological membranes.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== SECTION 3: ANALYZER ====================
st.markdown("<div class='analyzer-section'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title' style='text-align: center; margin-bottom: 60px;'>Molecular Analyzer</h2>", unsafe_allow_html=True)

# The restored st.form event loop constraint
with st.form(key="mol_form"):
    smiles_input = st.text_input(
        "Enter a Chemical Name (e.g., Aspirin, Ibuprofen, Benadryl) OR a SMILES String:",
        placeholder="Paracetamol",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_btn = st.form_submit_button("Analyze Molecule")

# Results Engine
if predict_btn and smiles_input:
    with st.spinner("Processing molecular structure..."):
        st.markdown("<div style='margin-top: 60px;'></div>", unsafe_allow_html=True)
        
        smiles = get_smiles_from_name(smiles_input)
        
        if smiles:
            if smiles != "".join(smiles_input.split()):
                st.info(f"Resolved Name to SMILES: `{smiles}`")
                
            features = extract_features(smiles)
            
            if features is None:
                st.error("Invalid chemical graph. RDKit could not parse the structure.")
            else:
                st.markdown("<h3 class='section-heading' style='text-align: center; color: #00ffff;'>ADMET Prediction Results</h3>", unsafe_allow_html=True)
                st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
                
                classifications = {k: v for k, v in models.items() if k != 'Lipophilicity'}
                regression = {k: v for k, v in models.items() if k == 'Lipophilicity'}
                
                c_cols = st.columns(3)
                for idx, (target, model) in enumerate(classifications.items()):
                    prob = model.predict_proba(features)[0][1]
                    with c_cols[idx % 3]:
                        st.metric(label=target.upper(), value=f"{prob:.2%}")
                
                if regression:
                    st.markdown("<br>", unsafe_allow_html=True)
                    pred = regression['Lipophilicity'].predict(features)[0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="Lipophilicity (LogD)", value=f"{pred:.3f}")
                
                st.success("Analysis complete. Predictions generated successfully.")
        else:
            st.error("Could not parse name or SMILES. Ensure spelling is correct or PubChem API is online.")

# Footer
st.markdown("""
<div class='footer'>
    <p>Powered by AI-driven molecular modeling | BioHackAR © 2026</p>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)