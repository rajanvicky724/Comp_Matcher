import streamlit as st
import pandas as pd
import numpy as np
import math
import io

# Set page to wide mode for better data viewing
st.set_page_config(page_title="Comp Matcher Pro", layout="wide")

# ---------- HELPER FUNCTIONS ----------

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in miles between two lat/lon points"""
    try:
        # Vectorized calculation for Pandas Series if passed, or single floats
        # This handles both cases automatically
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 3956  # Earth radius in miles
    except:
        return 999999

def get_prefix_6(val):
    if pd.isna(val): return ""
    clean = str(val).lower().replace(" ", "").replace(".", "").replace("-", "").replace(",", "").replace("/", "")
    return clean[:6]

def check_uniqueness(subject, candidate, chosen_comps):
    """Ensure candidate is not a duplicate of subject or already chosen comps"""
    def norm(x): return str(x).strip().lower()
    
    # 1. Check Account Number
    s_acc = norm(subject.get("Property Account No", ""))
    c_acc = norm(candidate.get("Property Account No", ""))
    if s_acc == c_acc: return False
    
    # Check against already chosen comps
    for chosen in chosen_comps:
        if norm(chosen.get("Property Account No", "")) == c_acc: return False

    # (You can add your Name/Address fuzzy checks here if needed, keeping it simple for speed)
    return True

# ---------- CORE LOGIC ----------

def process_matching(subject_df, source_df, rules):
    results = []
    
    # Progress bar logic
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_subjects = len(subject_df)
    
    # Pre-calculate source coordinates for faster distance calc
    if rules['use_distance']:
        src_lats = np.radians(source_df['lat'])
        src_lons = np.radians(source_df['lon'])

    for i, (_, srow) in enumerate(subject_df.iterrows()):
        # Update UI every 5 items to keep it responsive
        if i % 5 == 0:
            progress_bar.progress((i + 1) / total_subjects)
            status_text.text(f"Processing Subject {i+1} of {total_subjects}...")
            
        # --- FILTERS ---
        # 1. Filter Source by VPU (Must be lower than Subject?)
        # Only keep candidates with LOWER VPU if that rule is implied, 
        # or just strictly check the tolerance gap.
        
        # Start with all source data
        candidates = source_df.copy()
        
        # Rule: Comp VPU must be valid
        candidates = candidates.dropna(subset=['VPU'])
        
        # Rule: Comp VPU must be LOWER than Subject VPU (Standard Appeal Strategy)
        if rules['comp_vpu_lower']:
            candidates = candidates[candidates['VPU'] < srow['VPU']]
        
        # Rule: VPU Tolerance (e.g., within 50%)
        # Calculate percent difference
        candidates['vpu_diff_pct'] = (srow['VPU'] - candidates['VPU']) / srow['VPU']
        if rules['vpu_tolerance'] > 0:
            candidates = candidates[candidates['vpu_diff_pct'] <= (rules['vpu_tolerance'] / 100.0)]
            
        # Rule: Market Value Tolerance
        if rules['mv_tolerance'] > 0:
            pct_diff = abs(candidates['Total Market value-2023'] - srow['Total Market value-2023']) / srow['Total Market value-2023']
            candidates = candidates[pct_diff <= (rules['mv_tolerance'] / 100.0)]
            
        # Rule: GBA Tolerance
        if rules['gba_tolerance'] > 0:
            pct_diff = abs(candidates['GBA'] - srow['GBA']) / srow['GBA']
            candidates = candidates[pct_diff <= (rules['gba_tolerance'] / 100.0)]

        # --- DISTANCE CALCULATION (Vectorized for Speed) ---
        if rules['use_distance'] and pd.notna(srow['lat']) and pd.notna(srow['lon']):
            # Vectorized Haversine
            slat_rad = np.radians(srow['lat'])
            slon_rad = np.radians(srow['lon'])
            
            dlon = src_lons - slon_rad
            dlat = src_lats - slat_rad
            a = np.sin(dlat/2)**2 + np.cos(slat_rad) * np.cos(src_lats) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            candidates['calc_dist'] = c * 3956
            
            # Filter by Max Distance
            candidates = candidates[candidates['calc_dist'] <= rules['max_distance']]
        else:
            candidates['calc_dist'] = 999.0

        # --- RANKING / MATCH TYPE ---
        # Assign Match Type
        candidates['Match_Type'] = "Generic"
        candidates.loc[candidates['calc_dist'] <= 15, 'Match_Type'] = "Distance Match"
        
        # (Optional: Add ZIP/City logic back if needed, but Distance usually covers it)
        
        # Sort Candidates
        # 1. Best VPU Gap (Higher is better for appeals) -> Ascending False
        # 2. Closest Distance -> Ascending True
        candidates = candidates.sort_values(by=['vpu_diff_pct', 'calc_dist'], ascending=[False, True])
        
        # Select Top N unique comps
        chosen_comps = []
        for _, crow in candidates.iterrows():
            if len(chosen_comps) >= rules['max_comps']: break
            
            if check_uniqueness(srow, crow, chosen_comps):
                crow_dict = crow.to_dict()
                chosen_comps.append(crow_dict)

        # --- BUILD OUTPUT ROW ---
        out_row = srow.to_dict()
        # Rename Subject keys to differentiate? (Optional, based on your output format)
        # Here we just add Comp columns
        
        for k, comp in enumerate(chosen_comps):
            prefix = f"Comp{k+1}_"
            for col, val in comp.items():
                if col in rules['output_columns']: # Only save specific columns to keep file small
                    out_row[f"{prefix}{col}"] = val
            
            out_row[f"{prefix}Dist"] = comp.get('calc_dist', 999)
            out_row[f"{prefix}MatchMethod"] = comp.get('Match_Type', "N/A")

        results.append(out_row)

    progress_bar.progress(1.0)
    status_text.text("✅ Processing Complete!")
    return pd.DataFrame(results)


# ---------- UI LAYOUT ----------

st.title("🏡 Property Tax Comp Matcher")

# --- SIDEBAR: RULES CONFIGURATION ---
st.sidebar.header("⚙️ Matching Rules")

# 1. Max Comps
rule_max_comps = st.sidebar.number_input("Max Comps per Subject", min_value=1, max_value=20, value=10)

# 2. Distance Rule
st.sidebar.subheader("📍 Location Rules")
use_distance = st.sidebar.checkbox("Filter by Distance (Lat/Lon)", value=True)
max_dist_val = 25.0
if use_distance:
    max_dist_val = st.sidebar.slider("Max Distance (Miles)", 1.0, 100.0, 15.0)

# 3. VPU Rules
st.sidebar.subheader("💰 Valuation Rules")
comp_vpu_lower = st.sidebar.checkbox("Comp VPU MUST be lower than Subject", value=True)
vpu_tol = st.sidebar.number_input("VPU Tolerance % (0 to disable)", value=50)

# 4. Property Specs Rules
st.sidebar.subheader("building Specs Rules")
mv_tol = st.sidebar.number_input("Market Value Tolerance % (0 to disable)", value=50)
gba_tol = st.sidebar.number_input("GBA Tolerance % (0 to disable)", value=50)

# Pack rules into a dictionary
RULES = {
    'max_comps': rule_max_comps,
    'use_distance': use_distance,
    'max_distance': max_dist_val,
    'comp_vpu_lower': comp_vpu_lower,
    'vpu_tolerance': vpu_tol,
    'mv_tolerance': mv_tol,
    'gba_tolerance': gba_tol,
    # Define which columns you want in the final Excel for the comps
    'output_columns': ["Property Account No", "Property Address", "VPU", "Total Market value-2023", "GBA", "Owner Name/ LLC Name"]
}

# --- FILE UPLOAD SECTION ---
col1, col2 = st.columns(2)
with col1:
    subj_file = st.file_uploader("📂 Upload SUBJECT File (Excel)", type=["xlsx"])
with col2:
    src_file = st.file_uploader("📂 Upload SOURCE/COMPS File (Excel)", type=["xlsx"])

if subj_file and src_file:
    if st.button("🚀 Start Matching Process"):
        try:
            # Load Data
            df_subj = pd.read_excel(subj_file)
            df_src = pd.read_excel(src_file)
            
            st.write(f"Loaded {len(df_subj)} Subjects and {len(df_src)} Potential Comps.")
            
            # --- DATA CLEANING (Essential!) ---
            # Ensure VPU/Lat/Lon are numeric
            cols_to_numeric = ['VPU', 'Total Market value-2023', 'GBA', 'lat', 'lon']
            for col in cols_to_numeric:
                if col in df_subj.columns:
                    df_subj[col] = pd.to_numeric(df_subj[col], errors='coerce')
                if col in df_src.columns:
                    df_src[col] = pd.to_numeric(df_src[col], errors='coerce')

            # Run Processing
            result_df = process_matching(df_subj, df_src, RULES)
            
            # Show Preview
            st.subheader("Results Preview")
            st.dataframe(result_df.head())
            
            # Download Button
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 Download Results (Excel)",
                data=buffer,
                file_name="Automated_Comp_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
