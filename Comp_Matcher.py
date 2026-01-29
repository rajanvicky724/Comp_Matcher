import streamlit as st
import pandas as pd
import numpy as np
import io

# ---------- PAGE CONFIG & STYLING ----------
st.set_page_config(page_title="Comp Matcher Pro", layout="wide", page_icon="🏢")

def set_custom_style():
    st.markdown("""
        <style>
        /* Main Background - Clean White */
        .stApp { background-color: #f8f9fa; color: #333333; }
        
        /* Sidebar - Professional Grey */
        section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e5e7eb; }
        
        /* Headers */
        h1, h2, h3 { color: #111827 !important; font-family: 'Segoe UI', sans-serif; }

        /* Upload Cards */
        .stFileUploader {
            background-color: #ffffff; border: 1px dashed #cbd5e1;
            border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }

        /* Primary Button */
        div.stButton > button {
            width: 100%; background-color: #2563eb; color: white;
            border-radius: 6px; padding: 12px 24px; font-weight: 600; border: none;
            transition: all 0.2s;
        }
        div.stButton > button:hover { background-color: #1d4ed8; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2); }
        
        /* Hide Default Menu */
        #MainMenu {visibility: hidden;} footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

set_custom_style()

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
    if pd.isna(subj_c) or pd.isna(comp_c): return False
    subj_c = int(subj_c)
    comp_c = int(comp_c)
    if subj_c == 8: return comp_c == 8
    if comp_c == 8: return False
    if subj_c == 7: return comp_c in (6, 7)
    if subj_c == 6: return comp_c in (5, 6, 7)
    return (comp_c >= subj_c - 1) and (comp_c <= subj_c + 2)

def check_uniqueness(subject, candidate, chosen_comps, rules):
    def norm(x): return str(x).strip().lower()
    if norm(subject.get("Property Account No", "")) == norm(candidate.get("Property Account No", "")): return False
    for chosen in chosen_comps:
        if norm(chosen.get("Property Account No", "")) == norm(candidate.get("Property Account No", "")): return False
        if rules['is_hotel'] and 'Project Name' in candidate and 'Project Name' in chosen:
            p1 = norm(chosen.get('Project Name', 'a'))
            p2 = norm(candidate.get('Project Name', 'b'))
            if p1 and p2 and p1 == p2: return False
    return True

# ---------- CORE LOGIC ----------

def process_matching(subject_df, source_df, rules):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(subject_df)
    
    if 'lat' in source_df.columns and 'lon' in source_df.columns:
        src_lats = np.radians(source_df['lat'])
        src_lons = np.radians(source_df['lon'])
    
    for i, (_, srow) in enumerate(subject_df.iterrows()):
        if i % 5 == 0: 
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Processing {i+1}/{total}...")

        candidates = source_df.copy()

        if rules['is_hotel']:
            if rules['use_hotel_class'] and 'Class_Num' in srow and 'Class_Num' in candidates.columns:
                s_cls = srow['Class_Num']
                if pd.notna(s_cls):
                    candidates = candidates[candidates['Class_Num'].apply(lambda c: class_ok(s_cls, c))]
            val_col, size_col = 'VPR', 'Rooms'
        else:
            val_col, size_col = 'VPU', 'GBA'

        if val_col in candidates.columns and val_col in srow:
            candidates = candidates.dropna(subset=[val_col])
            s_val = srow[val_col]
            if pd.notna(s_val):
                if rules['comp_val_lower']: candidates = candidates[candidates[val_col] < s_val]
                candidates['val_diff_pct'] = (s_val - candidates[val_col]) / s_val
                if rules['val_tolerance'] > 0:
                     candidates = candidates[candidates['val_diff_pct'] <= (rules['val_tolerance']/100)]
        
        if size_col in candidates.columns and size_col in srow:
             s_size = srow[size_col]
             if pd.notna(s_size) and rules['size_tolerance'] > 0:
                 candidates['size_diff'] = abs(candidates[size_col] - s_size) / s_size
                 candidates = candidates[candidates['size_diff'] <= (rules['size_tolerance']/100)]

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

        if rules['filter_by_dist']: candidates = candidates[candidates['calc_dist'] <= rules['max_distance']]
        
        candidates['Match_Type'] = "Generic"
        candidates['priority_score'] = 4
        
        s_city = str(srow.get('Property City', '')).strip().lower()
        if s_city:
            mask_city = candidates['Property City'].astype(str).str.strip().str.lower() == s_city
            candidates.loc[mask_city, 'Match_Type'] = "Same City"
            candidates.loc[mask_city, 'priority_score'] = 3
            
        s_zip = str(srow.get('Property Zip Code', '')).strip()
        if s_zip:
            mask_zip = candidates['Property Zip Code'].astype(str).str.strip() == s_zip
            candidates.loc[mask_zip, 'Match_Type'] = "Same ZIP"
            candidates.loc[mask_zip, 'priority_score'] = 2
            
        mask_dist = candidates['calc_dist'] <= rules['max_distance']
        candidates.loc[mask_dist, 'Match_Type'] = "Distance Match"
        candidates.loc[mask_dist, 'priority_score'] = 1

        if 'val_diff_pct' in candidates.columns:
            candidates = candidates.sort_values(by=['priority_score', 'val_diff_pct', 'calc_dist'], ascending=[True, False, True])
        else:
            candidates = candidates.sort_values(by=['priority_score', 'calc_dist'], ascending=[True, True])
        
        chosen_comps = []
        for _, crow in candidates.iterrows():
            if len(chosen_comps) >= rules['max_comps']: break
            if check_uniqueness(srow, crow.to_dict(), chosen_comps, rules):
                chosen_comps.append(crow.to_dict())

        out_row = srow.to_dict()
        for k, comp in enumerate(chosen_comps):
            prefix = f"Comp{k+1}_"
            for col, val in comp.items():
                if col in rules['output_columns']: out_row[f"{prefix}{col}"] = val
            out_row[f"{prefix}Dist"] = comp.get('calc_dist', 999)
            out_row[f"{prefix}MatchMethod"] = comp.get('Match_Type', "N/A")
        results.append(out_row)

    progress_bar.progress(1.0)
    return pd.DataFrame(results)

# ---------- UI LAYOUT ----------

st.title("🏢 Property Tax Comp Matcher v4")

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Configuration")
prop_type = st.sidebar.radio("Property Type", ["Hotel", "Standard/Other"])
is_hotel = (prop_type == "Hotel")

st.sidebar.subheader("📍 Location Rules")
filter_dist = st.sidebar.checkbox("Filter by Distance?", value=True)
max_dist = st.sidebar.number_input("Max Distance (Miles)", value=15.0)

st.sidebar.subheader("💰 Valuation Rules")
comp_val_lower = st.sidebar.checkbox("Comp Value MUST be lower?", value=True)
val_tol = st.sidebar.number_input("Value Tolerance % (Gap)", value=50.0)

st.sidebar.subheader("🏗️ Property Specs")
if is_hotel:
    use_class = st.sidebar.checkbox("Enforce Hotel Class Match?", value=True)
    size_tol = st.sidebar.number_input("Room Count Tolerance %", value=50.0)
else:
    use_class = False
    size_tol = st.sidebar.number_input("GBA Tolerance %", value=50.0)

max_comps = st.sidebar.number_input("Max Comps to Find", value=3)

if is_hotel:
    out_cols = ["Property Account No", "Hotel Name", "Rooms", "VPR", "Hotel Class", "Property Address", "Owner Name/ LLC Name", "Project Name"]
else:
    out_cols = ["Property Account No", "Property Address", "VPU", "GBA", "Total Market value-2023", "Owner Name/ LLC Name"]

RULES = {
    'is_hotel': is_hotel, 'use_hotel_class': use_class, 'filter_by_dist': filter_dist,
    'max_distance': max_dist, 'comp_val_lower': comp_val_lower, 'val_tolerance': val_tol,
    'size_tolerance': size_tol, 'max_comps': max_comps, 'output_columns': out_cols
}

# --- GIF PLACEHOLDER (Top Center) ---
# Create an empty container that we can update
hero_container = st.empty()

# --- INITIAL IDLE STATE ---
# Only show this if we are NOT running yet (Session state handles this normally, 
# but for simple scripts, we default to showing Idle until button click updates it)

# Idle GIF: "Waiting / Analytical"
# Use Markdown to display centering and the image
hero_container.markdown(
    """
    <div style="text-align: center;">
        <img src="https://media.giphy.com/media/l0HlHJGHe3yAMhdQY/giphy.gif" width="300">
        <p style="color:gray;"><i>Ready to analyze your property data...</i></p>
    </div>
    """, 
    unsafe_allow_html=True
)

# --- UPLOAD SECTION ---
st.markdown("---")
c1, c2 = st.columns(2)
subj_file = c1.file_uploader("Upload SUBJECTS", type=["xlsx"])
src_file = c2.file_uploader("Upload DATA SOURCE", type=["xlsx"])

if subj_file and src_file:
    # Button triggers the switch
    if st.button("🚀 Start Matching Process"):
        
        # 1. SWITCH GIF TO "ACTIVE" STATE
        hero_container.markdown(
            """
            <div style="text-align: center;">
                <img src="https://media.giphy.com/media/2IudUHdI075HL02Pkk/giphy.gif" width="400">
                <p style="color:#2563eb; font-weight:bold;"><i>Processing Data... Please Wait!</i></p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # 2. Run Process
        try:
            df_subj = pd.read_excel(subj_file)
            df_src = pd.read_excel(src_file)
            
            # Basic Cleaning
            if is_hotel:
                if 'Hotel class values' in df_subj.columns: df_subj['Class_Num'] = df_subj['Hotel class values'].apply(norm_class)
                if 'Hotel class values' in df_src.columns: df_src['Class_Num'] = df_src['Hotel class values'].apply(norm_class)
            
            cols_num = ['VPR', 'VPU', 'Rooms', 'GBA', 'lat', 'lon', 'Market Value-2023', 'Total Market value-2023']
            for df in [df_subj, df_src]:
                for c in cols_num:
                    if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
                if 'lon' in df.columns: df['lon'] = df['lon'].apply(lambda x: -abs(x) if pd.notna(x) else x)

            # Actual Logic
            result = process_matching(df_subj, df_src, RULES)
            
            # 3. SWITCH GIF TO "DONE" STATE
            hero_container.markdown(
                """
                <div style="text-align: center;">
                    <img src="https://media.giphy.com/media/tf9JJY57TR196/giphy.gif" width="300">
                    <p style="color:green; font-weight:bold;"><i>✅ Job Done! Download below.</i></p>
                </div>
                """, 
                unsafe_allow_html=True
            )

            st.success("Analysis Complete!")
            st.dataframe(result.head())
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                result.to_excel(writer, index=False)
                
            st.download_button("📥 Download Results", buffer, "Comp_Results_v4.xlsx")
            
        except Exception as e:
            st.error(f"Error: {e}")
            # Reset GIF on error
            hero_container.markdown("❌ Error occurred.")
