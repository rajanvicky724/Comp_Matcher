import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Comp Matcher Ultimate", layout="wide")

# ---------- HELPER FUNCTIONS ----------

def haversine(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 3956 
    except:
        return 999999

def norm_class(v):
    try: return int(float(v))
    except: return np.nan

def class_ok(subj_c, comp_c):
    """Hotel Class Logic: Allows slight variance (+/- 1 star typically)"""
    if pd.isna(subj_c) or pd.isna(comp_c): return False
    subj_c = int(subj_c)
    comp_c = int(comp_c)
    if subj_c == 8: return comp_c == 8
    if comp_c == 8: return False
    # Example logic: Allow +/- 1 or specific grouping
    if subj_c == 7: return comp_c in (6, 7)
    if subj_c == 6: return comp_c in (5, 6, 7)
    return (comp_c >= subj_c - 1) and (comp_c <= subj_c + 2)

def check_uniqueness(subject, candidate, chosen_comps, rules):
    """Ensure candidate is unique based on Account No and Project Name"""
    def norm(x): return str(x).strip().lower()
    
    # 1. Basic Account Number Check
    if norm(subject.get("Property Account No", "")) == norm(candidate.get("Property Account No", "")): return False
    
    # Check against already chosen comps
    for chosen in chosen_comps:
        if norm(chosen.get("Property Account No", "")) == norm(candidate.get("Property Account No", "")): return False
        
        # 2. HOTEL RULE: Unique Project Name (Diversity)
        if rules['is_hotel'] and 'Project Name' in candidate and 'Project Name' in chosen:
            p1 = norm(chosen.get('Project Name', 'a'))
            p2 = norm(candidate.get('Project Name', 'b'))
            if p1 and p2 and p1 == p2:
                return False
    return True

# ---------- CORE LOGIC ----------

def process_matching(subject_df, source_df, rules):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(subject_df)
    
    # Pre-calculate source coordinates
    if 'lat' in source_df.columns and 'lon' in source_df.columns:
        src_lats = np.radians(source_df['lat'])
        src_lons = np.radians(source_df['lon'])
    
    for i, (_, srow) in enumerate(subject_df.iterrows()):
        if i % 5 == 0: 
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Processing {i+1}/{total}...")

        candidates = source_df.copy()

        # --- 1. HOTEL SPECIFIC RULES ---
        if rules['is_hotel']:
            # Hotel Class Rule
            if rules['use_hotel_class'] and 'Class_Num' in srow and 'Class_Num' in candidates.columns:
                # Use apply to check row-by-row or filter? For speed, we filter.
                # Since class_ok logic is complex, we might iterate or simple filter.
                # Simplified filter for speed: Must be within +/- 1
                s_cls = srow['Class_Num']
                if pd.notna(s_cls):
                    # Filter candidates based on custom class logic
                    candidates = candidates[candidates['Class_Num'].apply(lambda c: class_ok(s_cls, c))]
            
            # Unit Metric: VPR (Value Per Room)
            val_col = 'VPR'
            size_col = 'Rooms'
        else:
            # Standard Property Rules
            # Unit Metric: VPU (Value Per Unit) or GBA
            val_col = 'VPU'
            size_col = 'GBA'

        # --- 2. VALUATION RULES ---
        # Ensure metric exists
        if val_col in candidates.columns and val_col in srow:
            candidates = candidates.dropna(subset=[val_col])
            s_val = srow[val_col]
            
            if pd.notna(s_val):
                # Rule: Comp Value Lower?
                if rules['comp_val_lower']:
                    candidates = candidates[candidates[val_col] < s_val]

                # Rule: Value Tolerance
                candidates['val_diff_pct'] = (s_val - candidates[val_col]) / s_val
                if rules['val_tolerance'] > 0:
                     # Allow +/- tolerance? Or just gap? Logic implies "Gap" for appeals.
                     # Using absolute diff for general match, or specific direction?
                     # Your original code used: (sVPR - cVPR) / sVPR <= 0.50 check (which is directional)
                     candidates = candidates[candidates['val_diff_pct'] <= (rules['val_tolerance']/100)]
        
        # Rule: Size Tolerance (Rooms or GBA)
        if size_col in candidates.columns and size_col in srow:
             s_size = srow[size_col]
             if pd.notna(s_size) and rules['size_tolerance'] > 0:
                 candidates['size_diff'] = abs(candidates[size_col] - s_size) / s_size
                 candidates = candidates[candidates['size_diff'] <= (rules['size_tolerance']/100)]

        # --- 3. DISTANCE CALCULATION ---
        if pd.notna(srow.get('lat')) and pd.notna(srow.get('lon')) and 'lat' in candidates.columns:
            slat_rad = np.radians(srow['lat'])
            slon_rad = np.radians(srow['lon'])
            dlon = src_lons - slon_rad
            dlat = src_lats - slat_rad
            a = np.sin(dlat/2)**2 + np.cos(slat_rad) * np.cos(src_lats) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            candidates['calc_dist'] = c * 3956
        else:
            candidates['calc_dist'] = 999.0

        # --- 4. LOCATION FILTERING ---
        if rules['filter_by_dist']:
             candidates = candidates[candidates['calc_dist'] <= rules['max_distance']]
        
        # --- 5. WATERFALL MATCH TYPING ---
        candidates['Match_Type'] = "Generic"
        candidates['priority_score'] = 4
        
        # City Match
        s_city = str(srow.get('Property City', '')).strip().lower()
        if s_city:
            mask_city = candidates['Property City'].astype(str).str.strip().str.lower() == s_city
            candidates.loc[mask_city, 'Match_Type'] = "Same City"
            candidates.loc[mask_city, 'priority_score'] = 3
            
        # Zip Match
        s_zip = str(srow.get('Property Zip Code', '')).strip()
        if s_zip:
            mask_zip = candidates['Property Zip Code'].astype(str).str.strip() == s_zip
            candidates.loc[mask_zip, 'Match_Type'] = "Same ZIP"
            candidates.loc[mask_zip, 'priority_score'] = 2
            
        # Distance Match
        mask_dist = candidates['calc_dist'] <= rules['max_distance']
        candidates.loc[mask_dist, 'Match_Type'] = "Distance Match"
        candidates.loc[mask_dist, 'priority_score'] = 1

        # --- 6. SORTING & SELECTION ---
        # Sort by: Priority (Loc) -> Value Gap (High) -> Distance (Low)
        # Note: If 'val_diff_pct' is missing, fill with 0
        if 'val_diff_pct' in candidates.columns:
            candidates = candidates.sort_values(
                by=['priority_score', 'val_diff_pct', 'calc_dist'], 
                ascending=[True, False, True]
            )
        else:
            candidates = candidates.sort_values(by=['priority_score', 'calc_dist'], ascending=[True, True])
        
        chosen_comps = []
        for _, crow in candidates.iterrows():
            if len(chosen_comps) >= rules['max_comps']: break
            if check_uniqueness(srow, crow.to_dict(), chosen_comps, rules):
                chosen_comps.append(crow.to_dict())

        # --- 7. OUTPUT ---
        out_row = srow.to_dict()
        for k, comp in enumerate(chosen_comps):
            prefix = f"Comp{k+1}_"
            for col, val in comp.items():
                if col in rules['output_columns']:
                    out_row[f"{prefix}{col}"] = val
            out_row[f"{prefix}Dist"] = comp.get('calc_dist', 999)
            out_row[f"{prefix}MatchMethod"] = comp.get('Match_Type', "N/A")
            
        results.append(out_row)

    progress_bar.progress(1.0)
    return pd.DataFrame(results)

# ---------- UI LAYOUT ----------

st.title("🏢 Property Tax Comp Matcher v3")

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Configuration")

# 1. Property Type
prop_type = st.sidebar.radio("Property Type", ["Hotel", "Standard/Other"])
is_hotel = (prop_type == "Hotel")

# 2. Location Rules
st.sidebar.subheader("📍 Location Rules")
filter_dist = st.sidebar.checkbox("Filter by Distance?", value=True)
max_dist = st.sidebar.number_input("Max Distance (Miles)", value=15.0)

# 3. Valuation Rules
st.sidebar.subheader("💰 Valuation Rules")
comp_val_lower = st.sidebar.checkbox("Comp Value MUST be lower?", value=True, help="Standard appeal logic: find cheaper comps.")
val_tol = st.sidebar.number_input("Value Tolerance % (Gap)", value=50.0)

# 4. Property Specific Rules
st.sidebar.subheader("🏗️ Property Specs")
if is_hotel:
    use_class = st.sidebar.checkbox("Enforce Hotel Class Match?", value=True)
    size_tol = st.sidebar.number_input("Room Count Tolerance %", value=50.0)
    metric_label = "VPR (Value Per Room)"
else:
    use_class = False
    size_tol = st.sidebar.number_input("GBA Tolerance %", value=50.0)
    metric_label = "VPU (Value Per Unit)"

max_comps = st.sidebar.number_input("Max Comps to Find", value=3)

# Define Output Columns
if is_hotel:
    out_cols = ["Property Account No", "Hotel Name", "Rooms", "VPR", "Hotel Class", "Property Address", "Owner Name/ LLC Name", "Project Name"]
else:
    out_cols = ["Property Account No", "Property Address", "VPU", "GBA", "Total Market value-2023", "Owner Name/ LLC Name"]

RULES = {
    'is_hotel': is_hotel,
    'use_hotel_class': use_class,
    'filter_by_dist': filter_dist,
    'max_distance': max_dist,
    'comp_val_lower': comp_val_lower,
    'val_tolerance': val_tol,
    'size_tolerance': size_tol,
    'max_comps': max_comps,
    'output_columns': out_cols
}

st.info(f"Mode: **{prop_type}**. Matching using **{metric_label}**.")

# --- UPLOAD ---
c1, c2 = st.columns(2)
subj_file = c1.file_uploader("Upload SUBJECTS", type=["xlsx"])
src_file = c2.file_uploader("Upload DATA SOURCE", type=["xlsx"])

if subj_file and src_file:
    if st.button("🚀 Start Matching"):
        with st.spinner("Crunching numbers..."):
            try:
                df_subj = pd.read_excel(subj_file)
                df_src = pd.read_excel(src_file)
                
                # PRE-PROCESSING (Standardize Names)
                # Map user column names if needed, or assume standard names
                # Hotel Mode specific renaming
                if is_hotel:
                    # Try to rename 'Hotel class values' to 'Class_Num' if it exists
                    if 'Hotel class values' in df_subj.columns: df_subj['Class_Num'] = df_subj['Hotel class values'].apply(norm_class)
                    if 'Hotel class values' in df_src.columns: df_src['Class_Num'] = df_src['Hotel class values'].apply(norm_class)
                
                # Numeric Conversions
                cols_num = ['VPR', 'VPU', 'Rooms', 'GBA', 'lat', 'lon', 'Market Value-2023', 'Total Market value-2023']
                for df in [df_subj, df_src]:
                    for c in cols_num:
                        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
                    # Fix longitude
                    if 'lon' in df.columns: df['lon'] = df['lon'].apply(lambda x: -abs(x) if pd.notna(x) else x)

                result = process_matching(df_subj, df_src, RULES)
                
                st.success("Done!")
                st.dataframe(result.head())
                
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    result.to_excel(writer, index=False)
                    
                st.download_button("📥 Download Results", buffer, "Comp_Results.xlsx")
                
            except Exception as e:
                st.error(f"Error: {e}")
