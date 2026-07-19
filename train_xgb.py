import pandas as pd
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors # ADD THIS TO YOUR IMPORTS AT THE TOP
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, r2_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

trained_models = {} # Define this near the top of your script
# 1. Pipeline Definitions
classification_tasks = ['hia', 'bbb', 'cyp3a4_sub', 'carcin', 'herg']
regression_task = 'Lipophilicity'

print("[+] Loading consolidated matrix...")
df = pd.read_csv('multi_task_admet_v2.csv')



# 2. Strict Bemis-Murcko Scaffold Sorting & Hybrid Feature Extraction
print("[+] Computing molecular scaffolds and hybrid physical descriptors...")
scaffolds = defaultdict(list)
valid_indices = []
features = []

for idx, smiles in enumerate(df['Drug']):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Part A: The Structural Micro-Topology (Binary)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_array = np.array(fp, dtype=np.float32)
        
        # Part B: The Macro-Thermodynamics (Continuous Floats)
        mol_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        
        physical_descriptors = np.array([mol_weight, logp, tpsa, h_donors, h_acceptors], dtype=np.float32)
        
        # Fuse them into a single 2053-dimensional hybrid vector
        hybrid_vector = np.concatenate((fp_array, physical_descriptors))
        
        features.append(hybrid_vector)
        valid_indices.append(idx)
        
        # Extract core molecular framework for anti-leakage splitting
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds[scaffold].append(idx)

X = np.array(features)
df_clean = df.iloc[valid_indices].reset_index(drop=True)

# Helper function to execute a leakage-free scaffold split per task
def scaffold_split_data(task_name, valid_df, X_matrix, test_size=0.2):
    task_clean = valid_df[[task_name]].dropna()
    indices = task_clean.index.tolist()
    
    # Group available indices by their respective scaffold buckets
    task_scaffolds = defaultdict(list)
    for idx in indices:
        mol_smiles = valid_df['Drug'].iloc[idx]
        mol = Chem.MolFromSmiles(mol_smiles)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        task_scaffolds[scaf].append(idx)
        
    # Sort scaffolds by size to distribute large clusters systematically
    sorted_scaffolds = sorted(task_scaffolds.values(), key=len, reverse=True)
    
    train_idx, val_idx = [], []
    val_cutoff = len(indices) * test_size
    
    for cluster in sorted_scaffolds:
        if len(val_idx) + len(cluster) <= val_cutoff:
            val_idx.extend(cluster)
        else:
            train_idx.extend(cluster)
            
    return X_matrix[train_idx], X_matrix[val_idx], valid_df[task_name].iloc[train_idx].values, valid_df[task_name].iloc[val_idx].values

# 3. Processing Core Models
print("\n" + "="*85)
print(f"{'ENDPOINT':<15} | {'VALID ROC-AUC':<15} | {'VALID AUPRC':<15} | {'METRIC 2':<15} | {'STATUS':<15}")
print("="*85)

# Execute Classifiers
for task in classification_tasks:
    try:
        X_train, X_valid, y_train, y_valid = scaffold_split_data(task, df_clean, X)
        
        if len(y_train) == 0 or len(y_valid) == 0:
            continue
            
        imbalance = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
        
        clf = xgb.XGBClassifier(
            n_estimators=250, learning_rate=0.05, max_depth=5,
            scale_pos_weight=imbalance, eval_metric='auc', early_stopping_rounds=15, random_state=42
        )
        clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        trained_models[task] = clf 
        preds = clf.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, preds)
        auprc = average_precision_score(y_valid, preds)
        print(f"{task:<15} | {auc:<15.4f} | {auprc:<15.4f} | {'N/A':<15} | Operational")
    except Exception as e:
        print(f"{task:<15} | Error during compilation.")

# Execute Regressor
if regression_task in df_clean.columns:
    try:
        X_train_r, X_valid_r, y_train_r, y_valid_r = scaffold_split_data(regression_task, df_clean, X)
        
        reg = xgb.XGBRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=5,
            eval_metric='rmse', early_stopping_rounds=15, random_state=42
        )
        reg.fit(X_train_r, y_train_r, eval_set=[(X_valid_r, y_valid_r)], verbose=False)
        trained_models['Lipophilicity'] = reg
        
        preds_r = reg.predict(X_valid_r)
        rmse = np.sqrt(mean_squared_error(y_valid_r, preds_r))
        r2 = r2_score(y_valid_r, preds_r)
        print(f"{regression_task:<15} | {'N/A':<15} | {'N/A':<15} | RMSE: {rmse:.3f}   | R²: {r2:.3f}")
    except Exception as e:
        print(f"{regression_task:<15} | Error during regression tracking.")
print("="*85)


import joblib
print("[+] Serializing XGBoost models to disk...")
joblib.dump(trained_models, 'admet_models.pkl')
print("[+] Export complete. Model binary saved as 'admet_models.pkl'.")