import streamlit as st
import pandas as pd
import math
import io
import time

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def haversine(lat1, lon1, lat2, lon2):
    """Calculates distance in miles between two lat/lon points."""
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a)) 
        return c * 3956  # Radius of earth in miles
    except:
        return 999999

def clean_property_id(val):
    """Cleans property account numbers."""
    return str(val).replace('.0', '').strip()

def get_prefix_6(val):
    if pd.isna(val): return ""
    clean = str(val).lower().replace(" ", "").replace(".", "").replace("-", "").replace(",", "").replace("/", "")
    return clean[:6]

def unique_ok(subject, candidate, chosen_comps):
    """Prevents duplicate owners or identical addresses."""
    def norm(x): return str(x).strip().lower()
    pairs = [(subject, candidate)] + [(c, candidate) for c in chosen_comps]
    for a, b in pairs:
        # Check Account No
        if norm(a.get("Property Account No", "")) == norm(b.get("Property Account No", "")): return False
        # Check Owner Name (first 6 chars)
        if len(get_prefix_6(a.get("Owner Name/ LLC Name", ""))) >= 4 and get_prefix_6(a.get("Owner Name/ LLC Name", "")) == get_prefix_6(b.get("Owner Name/ LLC Name", "")): return False
        # Check Address (first 6 chars)
        if len(get_prefix_6(a.get("Property Address", ""))) >= 4 and get_prefix_6(a.get("Property Address", "")) == get_prefix_6(b.get("Property Address", "")): return False
    return True

# ==========================================
# 2. CORE MATCHING LOGIC
# ==========================================

def find_comps(srow, src_df, config):
    sVPU = srow["VPU"]
    smv = srow["Total Market value-2023"]
    srm = srow["GBA"]
    slat, slon = srow.get("lat"), srow.get("lon")
    sClass = str(srow.get("Class", "")).lower().strip() # For Hotels

    if pd.isna(sVPU) or sVPU == 0: return []

    candidates = []

    # Unpack config
    max_dist = config['max_dist']
    comp_must_be_lower = config['comp_lower']
    val_tol = config['val_tol'] / 100.0
    gba_tol = config['gba_tol'] / 100.0
    prop_type = config['prop_type']

    for _, crow in src_df.iterrows():
        cVPU = crow["VPU"]
        cmv = crow["Total Market value-2023"]
        crm = crow["GBA"]
        cClass = str(crow.get("Class", "")).lower().strip()
        
        # --- RULE 1: Property Type (Hotel vs Standard) ---
        if prop_type == "Hotel":
            # If Hotel, classes must roughly match (e.g. contains "economy")
            if sClass and cClass and (sClass not in cClass and cClass not in sClass):
                continue
        # If Standard, we ignore Class entirely.

        # --- RULE 2: Comp Value Lower? ---
        if pd.isna(cVPU): continue
        if comp_must_be_lower and cVPU > sVPU: continue
        
        # --- RULE 3: Value Tolerance ---
        gap_pct = (sVPU - cVPU) / sVPU
        if abs(gap_pct) > val_tol: continue

        # --- RULE 4: GBA Tolerance ---
        if srm > 0:
            if abs(crm - srm) / srm > gba_tol: continue

        # --- RULE 5: Distance Calculation ---
        clat, clon = crow.get("lat"), crow.get("lon")
        dist_miles = 999
        if pd.notna(slat) and pd.notna(slon) and pd.notna(clat) and pd.notna(clon):
            dist_miles = haversine(slat, slon, clat, clon)
        
        # --- RULE 6: Match Logic & Priority ---
        # Priority 1: Within Radius
        # Priority 2: Same Zip
        # Priority 3: Same City
        # Priority 4: Everything else (only if Distance Filter is OFF)
        
        match_type = None
        priority = 99
        
        is_radius_match = dist_miles <= max_dist
        is_zip_match = str(srow["Property Zip Code"]) == str(crow["Property Zip Code"])
        is_city_match = str(srow["Property City"]).strip().lower() == str(crow["Property City"]).strip().lower()

        if is_radius_match:
            match_type = f"Within {max_dist} Miles"
            priority = 1
        elif is_zip_match:
            match_type = "Same ZIP"
            priority = 2
        elif is_city_match:
            match_type = "Same City"
            priority = 3
        else:
            # If we enforce distance, skip this.
            # If we DO NOT enforce distance, we allow it as a low priority match.
            if config['use_dist_filter']:
                continue 
            else:
                match_type = "Fallback (Location mismatch)"
                priority = 4

        candidates.append({
            "row": crow,
            "priority": priority,
            "dist": dist_miles,
            "vpu_gap": float(sVPU - cVPU),
            "match_type": match_type
        })

    # --- SORTING ---
    # 1. Match Priority (Radius > Zip > City > Fallback)
    # 2. Biggest Price Gap (Descending)
    # 3. Distance (Closer is better)
    candidates.sort(key=lambda x: (x['priority'], -x['vpu_gap'], x['dist']))

    final_comps = []
    chosen_rows = []
    
    for cand in candidates:
        if unique_ok(srow, cand["row"], chosen_rows):
            crow = cand["row"].copy()
            crow["Match_Method"] = cand["match_type"]
            crow["Distance_Calc"] = cand["dist"] if cand["dist"] != 999 else "N/A"
            crow["VPU_Diff"] = cand["vpu_gap"]
            final_comps.append(crow)
            chosen_rows.append(crow)
        
        if len(final_comps) == config['max_comps']: break
            
    return final_comps

# ==========================================
# 3. MAIN APP UI
# ==========================================

st.set_page_config(page_title="Comp Matcher", layout="wide")

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Configuration")

st.sidebar.markdown("### 🏨 Property Type")
prop_type = st.sidebar.radio("Select Type:", ["Standard/Residential", "Hotel"])

st.sidebar.markdown("### 📍 Location Rules")
use_dist = st.sidebar.checkbox("Strict Distance Filter?", value=True, help="If unchecked, will fallback to ZIP/City matches even if far away.")
max_dist = st.sidebar.number_input("Max Radius (Miles)", value=15.0, step=1.0)

st.sidebar.markdown("### 💰 Valuation Rules")
comp_lower = st.sidebar.checkbox("Comp Value MUST be lower?", value=True)
val_tol = st.sidebar.number_input("Value Tolerance % (Gap)", value=50.0, step=5.0)

st.sidebar.markdown("### 🏗️ Property Specs")
gba_tol = st.sidebar.number_input("GBA Tolerance %", value=50.0, step=5.0)
max_comps_count = st.sidebar.number_input("Max Comps to Find", value=10, step=1, min_value=1, max_value=20)

config = {
    'prop_type': prop_type.split("/")[0], # "Standard" or "Hotel"
    'use_dist_filter': use_dist,
    'max_dist': max_dist,
    'comp_lower': comp_lower,
    'val_tol': val_tol,
    'gba_tol': gba_tol,
    'max_comps': max_comps_count
}

# --- HEADER ---
st.title("🏢 Property Tax Comp Matcher")
st.markdown(f"Mode: **{config['prop_type']}** | Priority: **Gap & Location**")

# --- FILE UPLOADS ---
col1, col2 = st.columns(2)
with col1:
    st.info("Step 1: Upload Subject List")
    subj_file = st.file_uploader("Choose Subject Excel", type=["xlsx"], key="subj")

with col2:
    st.info("Step 2: Upload Data Source")
    src_file = st.file_uploader("Choose Data Source Excel", type=["xlsx"], key="src")

# --- PROCESSING ---
if subj_file and src_file:
    if st.button("🚀 Run Matching Process", type="primary"):
        
        st.write("---")
        status_text = st.empty()
        prog_bar = st.progress(0)
        
        with st.spinner("Processing..."):
            try:
                # Load Data
                df_subj = pd.read_excel(subj_file)
                df_src = pd.read_excel(src_file)

                # Pre-processing
                df_subj.columns = df_subj.columns.str.strip()
                df_src.columns = df_src.columns.str.strip()

                for df in [df_subj, df_src]:
                    if "Property Account No" in df.columns:
                        df["Property Account No"] = df["Property Account No"].apply(clean_property_id)
                    elif "Concat" in df.columns:
                        df["Property Account No"] = df["Concat"].astype(str).str.extract(r'(\d+)', expand=False)
                    
                    cols_to_num = ["Property Zip Code", "GBA", "VPU", "Total Market value-2023", "lat", "lon"]
                    for c in cols_to_num:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    
                    if "lon" in df.columns:
                         df["lon"] = df["lon"].apply(lambda x: -abs(x) if pd.notna(x) else x)

                df_subj = df_subj.dropna(subset=["Property Zip Code", "VPU"])
                df_src = df_src.dropna(subset=["Property Zip Code", "VPU"])

                # Matching
                OUTPUT_COLS = [
                    "Property Account No","GBA","VPU","Property Address","Property City",
                    "Property County", "Property State","Property Zip Code","Assessed Value-2023",
                    "Total Market value-2023","Owner Name/ LLC Name","Owner Street Address",
                    "Owner City","Owner State","Owner ZIP"
                ]

                results = []
                total_subj = len(df_subj)

                for i, (_, srow) in enumerate(df_subj.iterrows()):
                    comps = find_comps(srow, df_src, config)
                    
                    row = {}
                    for c in OUTPUT_COLS: row[f"Subject_{c}"] = srow.get(c, "")
                    
                    for k in range(config['max_comps']):
                        prefix = f"Comp{k+1}"
                        if k < len(comps):
                            crow = comps[k]
                            for c in OUTPUT_COLS: row[f"{prefix}_{c}"] = crow.get(c, "")
                            row[f"{prefix}_Match_Method"] = crow.get("Match_Method", "")
                            d = crow.get("Distance_Calc", "")
                            row[f"{prefix}_Distance"] = f"{d:.2f}" if isinstance(d, float) else d
                            gap = crow.get("VPU_Diff", "")
                            row[f"{prefix}_VPU_Gap"] = f"{gap:.2f}" if isinstance(gap, float) else gap
                        else:
                            for c in OUTPUT_COLS: row[f"{prefix}_{c}"] = ""
                            row[f"{prefix}_Match_Method"] = ""
                            row[f"{prefix}_Distance"] = ""
                            row[f"{prefix}_VPU_Gap"] = ""
                    
                    results.append(row)
                    prog_bar.progress((i + 1) / total_subj)

                # Export
                df_final = pd.DataFrame(results)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df_final.to_excel(writer, index=False)
                
                # --- SUCCESS ANIMATION ---
                st.balloons()
                # Permanent, reliable GIF (Confetti Checkmark)
                st.markdown(
                    """
                    <div style="display: flex; justify-content: center;">
                        <img src="https://media.giphy.com/media/26u4lOMA8JKSnL9Uk/giphy.gif" width="300" />
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                st.success(f"✅ Job Done! Processed {total_subj} subjects.")
                
                st.download_button(
                    label="📥 Download Results (Excel)",
                    data=buffer.getvalue(),
                    file_name="Automated_Comps_Results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
