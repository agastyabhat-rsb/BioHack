import streamlit as st
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress RDKit terminal noise
RDLogger.DisableLog('rdApp.*')

# --- UI Configuration ---
st.set_page_config(page_title="ADMET Predictor", layout="centered")
st.title("Molecular ADMET Pipeline")
st.markdown("Predict pharmacokinetic properties from SMILES strings using hybrid XGBoost architecture.")

# --- Load Frozen Models ---
@st.cache_resource
def load_models():
    return joblib.load('admet_models.pkl')

try:
    models = load_models()
except FileNotFoundError:
    st.error("Model file 'admet_models.pkl' not found. Run train_xgb.py first.")
    st.stop()

# --- Feature Engineering Engine ---
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

# --- Web App Interface ---
user_smiles = st.text_input("Enter SMILES String:", "CC(=O)OC1=CC=CC=C1C(=O)O") # Default: Aspirin

if st.button("Predict ADMET Properties"):
    with st.spinner("Analyzing molecular topology..."):
        features = extract_features(user_smiles)
        
        if features is None:
            st.error("Invalid SMILES string. RDKit could not parse the chemical graph.")
        else:
            st.success("Analysis Complete")
            
            # Display metrics in a clean grid
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Targets")
                for target, model in models.items():
                    if target != 'Lipophilicity':
                        prob = model.predict_proba(features)[0][1]
                        st.metric(label=target.upper(), value=f"{prob:.2%}")
            
            with col2:
                st.subheader("Regression Target")
                if 'Lipophilicity' in models:
                    pred = models['Lipophilicity'].predict(features)[0]
                    st.metric(label="Lipophilicity (LogD)", value=f"{pred:.3f}")