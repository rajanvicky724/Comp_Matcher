import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import math
import io

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
        return c * 3956  # miles
    except Exception:
        return 999999


def norm_class(v):
    try:
        return int(float(v))
    except Exception:
        return np.nan


def tolerance_ok(subj_val, comp_val, pct=0.50):
    if pd.isna(subj_val) or pd.isna(comp_val) or subj_val == 0:
        return False
    return abs(comp_val - subj_val) / subj_val <= pct


def get_prefix_6(val):
    if pd.isna(val):
        return ""
    clean = (
        str(val)
        .lower()
        .replace(" ", "")
        .replace(".", "")
        .replace("-", "")
        .replace(",", "")
        .replace("/", "")
    )
    return clean[:6]


def unique_ok(subject, candidate, chosen_comps, is_hotel):
    """Prevent duplicates based on several keys."""
    def norm(x): return str(x).strip().lower()
    pairs = [(subject, candidate)] + [(c, candidate) for c in chosen_comps]
    for a, b in pairs:
        if norm(a.get("Property Account No", "")) == norm(b.get("Property Account No", "")):
            return False
        if len(get_prefix_6(a.get("Owner Name/ LLC Name", ""))) >= 4 and \
           get_prefix_6(a.get("Owner Name/ LLC Name", "")) == get_prefix_6(b.get("Owner Name/ LLC Name", "")):
            return False
        if is_hotel:
            if len(get_prefix_6(a.get("Hotel Name", ""))) >= 4 and \
               get_prefix_6(a.get("Hotel Name", "")) == get_prefix_6(b.get("Hotel Name", "")):
                return False
            if len(get_prefix_6(a.get("Owner Street Address", ""))) >= 4 and \
               get_prefix_6(a.get("Owner Street Address", "")) == get_prefix_6(b.get("Owner Street Address", "")):
                return False
        if len(get_prefix_6(a.get("Property Address", ""))) >= 4 and \
           get_prefix_6(a.get("Property Address", "")) == get_prefix_6(b.get("Property Address", "")):
            return False
    return True


# ---------- CLASS RULES ----------

def class_ok_hotel(subj_c, comp_c):
    subj_c = int(subj_c)
    comp_c = int(comp_c)
    if subj_c == 8:
        return comp_c == 8
    if comp_c == 8:
        return False
    if subj_c == 7:
        return comp_c in (6, 7)
    if subj_c == 6:
        return comp_c in (5, 6, 7)
    return (comp_c >= subj_c - 1) and (comp_c <= subj_c + 2)


def class_ok_other(subj_c, comp_c):
    try:
        subj_c = int(subj_c)
        comp_c = int(comp_c)
    except Exception:
        return False
    return abs(comp_c - subj_c) <= 2


# ==========================================
# 2. CORE MATCHING LOGIC
# ==========================================

def find_comps(
    srow,
    src_df,
    *,
    is_hotel,
    use_hotel_class_rule,
    max_radius_miles,
    max_gap_pct_main,
    max_gap_pct_value,
    max_gap_pct_size,
    max_comps,
    use_strict_distance,
    use_county_match,
    sort_mode,
):
    """
    sort_mode: 'Distance Priority' or 'VPR/VPU Gap (lower comp value)'.
    """

    if is_hotel:
        metric_field = "VPR"
        size_field = "Rooms"
        value_field = "Market Value-2023"
    else:
        metric_field = "VPU"
        size_field = "GBA"
        value_field = "Total Market value-2023"

    subj_class = srow.get("Class_Num")
    subj_metric = srow.get(metric_field)
    subj_value = srow.get(value_field)
    subj_size = srow.get(size_field)
    slat, slon = srow.get("lat"), srow.get("lon")

    if pd.isna(subj_metric):
        return []

    candidates = []

    for _, crow in src_df.iterrows():
        comp_class = crow.get("Class_Num")

        if is_hotel and use_hotel_class_rule:
            if not class_ok_hotel(subj_class, comp_class):
                continue
        else:
            if pd.notna(subj_class) and pd.notna(comp_class):
                if not class_ok_other(subj_class, comp_class):
                    continue

        comp_metric = crow.get(metric_field)
        comp_value = crow.get(value_field)
        comp_size = crow.get(size_field)

        if pd.isna(comp_metric) or comp_metric > subj_metric:
            continue

        if not tolerance_ok(subj_metric, comp_metric, max_gap_pct_main):
            continue
        if not tolerance_ok(subj_value, comp_value, max_gap_pct_value):
            continue
        if not tolerance_ok(subj_size, comp_size, max_gap_pct_size):
            continue

        clat, clon = crow.get("lat"), crow.get("lon")
        dist_miles = 999
        if pd.notna(slat) and pd.notna(slon) and pd.notna(clat) and pd.notna(clon):
            dist_miles = haversine(slat, slon, clat, clon)

        match_type = None
        priority = 99

        is_radius = dist_miles <= max_radius_miles
        is_zip = str(srow.get("Property Zip Code")) == str(crow.get("Property Zip Code"))
        is_city = str(srow.get("Property City", "")).strip().lower() == \
                  str(crow.get("Property City", "")).strip().lower()
        is_county = str(srow.get("Property County", "")).strip().lower() == \
                    str(crow.get("Property County", "")).strip().lower()

        if use_strict_distance:
            if is_radius:
                match_type = f"Within {max_radius_miles} Miles"
                priority = 1
            elif is_zip:
                match_type = "Same ZIP"
                priority = 2
            elif is_city:
                match_type = "Same City"
                priority = 3
            elif use_county_match and is_county:
                match_type = "Same County"
                priority = 4
            else:
                continue
        else:
            if is_zip:
                match_type = "Same ZIP"
                priority = 1
            elif is_city:
                match_type = "Same City"
                priority = 2
            elif use_county_match and is_county:
                match_type = "Same County"
                priority = 3
            else:
                continue

        metric_gap = float(subj_metric - comp_metric)

        candidates.append(
            (crow, priority, dist_miles, metric_gap, match_type)
        )

    if sort_mode == "Distance Priority":
        candidates.sort(key=lambda x: (x[1], x[2], -x[3]))
    else:
        candidates.sort(key=lambda x: (x[1], -x[3], x[2]))

    final_comps = []
    chosen_rows = []

    for cand in candidates:
        crow, priority, dist_miles, metric_gap, match_type = cand
        if unique_ok(srow, crow, chosen_rows, is_hotel=is_hotel):
            ccopy = crow.copy()
            ccopy["Match_Method"] = match_type
            ccopy["Distance_Calc"] = dist_miles if dist_miles != 999 else "N/A"
            ccopy[f"{metric_field}_Diff"] = metric_gap
            final_comps.append(ccopy)
            chosen_rows.append(ccopy)
        if len(final_comps) == max_comps:
            break

    return final_comps


OUTPUT_COLS_HOTEL = [
    "Property Account No", "Hotel Name", "Rooms", "VPR", "Property Address",
    "Property City", "Property County", "Property State", "Property Zip Code",
    "Assessed Value-2023", "Market Value-2023", "Hotel Class",
    "Owner Name/ LLC Name", "Owner Street Address", "Owner City",
    "Owner State", "Owner ZIP", "Contact Person", "Designation"
]

OUTPUT_COLS_OTHER = [
    "Property Account No", "GBA", "VPU", "Property Address",
    "Property City", "Property County", "Property State", "Property Zip Code",
    "Assessed Value-2023", "Total Market value-2023",
    "Owner Name/ LLC Name", "Owner Street Address", "Owner City",
    "Owner State", "Owner ZIP"
]


def get_val(row, col):
    if col == "Hotel Class":
        return row.get("Hotel class values", "")
    if col == "Property County":
        return row.get("Property County", row.get("County", ""))
    return row.get(col, "")


# ==========================================
# 3. STREAMLIT APP
# ==========================================

st.set_page_config(page_title="Comp Matcher", layout="wide")
st.markdown(
    """
    <style>
    .page-watermark {
        position: fixed;
        bottom: 60px;
        right: 10px;
        color: rgba(0, 0, 0, 0.15);
        font-size: 24px;
        font-weight: 600;
        font-family: "Segoe UI", sans-serif;
        z-index: 1000;
        pointer-events: none;
    }
    </style>
    <div class="page-watermark">Vignesh</div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2980b9);
        color: white;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-family: "Segoe UI", sans-serif;
    }
    .main-header h1 {
        font-size: 26px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
      <h1>🏙️ Property Tax / Hotel Comp Matcher</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- SIDEBAR CONFIG ----------

st.sidebar.header("⚙️ Configuration")

prop_type = st.sidebar.radio(
    "Property Type",
    ["Hotel Property", "Other Property"],
    help="Hotel Property uses VPR & Rooms; Other Property uses VPU & GBA."
)

is_hotel = prop_type == "Hotel Property"

use_hotel_class_rule = False
if is_hotel:
    use_hotel_class_rule = st.sidebar.checkbox(
        "Use Hotel Class Rule",
        value=True,
        help="Uncheck to ignore Hotel Class matching logic."
    )

st.sidebar.markdown("### 📍 Location Rules")
use_strict_distance = st.sidebar.checkbox(
    "Strict Distance Filter?",
    value=True,
    help="If checked: use Radius → ZIP → City → County. If unchecked: ignore Radius, use ZIP → City → County."
)
max_radius = st.sidebar.number_input(
    "Max Radius (Miles)",
    value=15.0,
    step=1.0,
    min_value=0.0
)
use_county_match = st.sidebar.checkbox(
    "Use County Match (after City)",
    value=True,
    help="If checked, Same County is used after City in the priority order."
)

sort_mode = st.sidebar.radio(
    "Choose Comp Based On",
    ["Distance Priority", "VPR/VPU Gap (lower comp value)"],
    help=(
        "Distance Priority: Radius/ZIP/City/County first, then VPR/VPU gap. "
        "Gap: prefers comps whose VPR/VPU is much lower than the subject."
    ),
)

st.sidebar.markdown("### 💰 Main Metric Rules")
if is_hotel:
    st.sidebar.write("Main Metric: **VPR**")
else:
    st.sidebar.write("Main Metric: **VPU**")

max_gap_pct_main = st.sidebar.number_input(
    "Max Main Metric Gap % (subject vs comp)",
    value=50.0,
    step=5.0,
    min_value=0.0,
    max_value=100.0
) / 100.0

st.sidebar.markdown("### 📈 Value & Size Rules")
max_gap_pct_value = st.sidebar.number_input(
    "Max Market/Total Value Gap %",
    value=50.0,
    step=5.0,
    min_value=0.0,
    max_value=100.0
) / 100.0

max_gap_pct_size = st.sidebar.number_input(
    "Max Size Gap % (Rooms or GBA)",
    value=50.0,
    step=5.0,
    min_value=0.0,
    max_value=100.0
) / 100.0

max_comps = st.sidebar.number_input(
    "Max Comps per Subject",
    value=3,
    step=1,
    min_value=1,
    max_value=20
)

st.sidebar.markdown("### 💸 Overpaid Analysis")
use_overpaid = st.sidebar.checkbox(
    "Calculate Overpaid Amount?",
    value=False,
    help="If checked, calculates an overpaid estimate from the selected comps.",
)

overpaid_base_dim = None
if use_overpaid:
    overpaid_base_dim = st.sidebar.radio(
        "Use Rooms / Units / GBA?",
        ["Rooms", "Units", "GBA"],
        index=0 if is_hotel else 1,
        help="Hotel: usually Rooms; Apartments: Units; Other properties: GBA.",
    )

    overpaid_pct = st.sidebar.number_input(
        "Overpaid Percentage (%)",
        value=10.0,
        step=1.0,
        min_value=0.0,
        max_value=100.0,
        help="Percentage used in overpaid formula.",
    ) / 100.0
else:
    overpaid_pct = 0.0
    overpaid_base_dim = None

# ---------- FILE UPLOADS ----------

st.markdown("### Step 1: Upload Files")

col1, col2 = st.columns(2)

with col1:
    st.info("Upload Subject Excel")
    subj_file = st.file_uploader("Subject File (.xlsx)", type=["xlsx"], key="subj_file")

with col2:
    st.info("Upload Data Source Excel")
    src_file = st.file_uploader("Data Source File (.xlsx)", type=["xlsx"], key="src_file")

# ---------- PROCESS ----------

if subj_file is not None and src_file is not None:
    if st.button("🚀 Run Matching", type="primary"):
        with st.spinner("Processing..."):
            try:
                subj = pd.read_excel(subj_file)
                src = pd.read_excel(src_file)

                subj.columns = subj.columns.str.strip()
                src.columns = src.columns.str.strip()

                for df in (subj, src):
                    if "Property Account No" in df.columns:
                        df["Property Account No"] = (
                            df["Property Account No"]
                            .astype(str)
                            .str.replace(r"\.0$", "", regex=True)
                            .str.strip()
                        )
                    elif "Concat" in df.columns:
                        df["Property Account No"] = (
                            df["Concat"].astype(str).str.extract(r"(\d+)", expand=False)
                        )

                    if "Hotel class values" in df.columns:
                        df["Class_Num"] = df["Hotel class values"].apply(norm_class)
                    elif "Class" in df.columns:
                        df["Class_Num"] = df["Class"].apply(norm_class)
                    else:
                        df["Class_Num"] = np.nan

                    for c in ["Property Zip Code", "Rooms", "Units", "GBA", "VPR", "VPU",
                              "Market Value-2023", "Total Market value-2023", "lat", "lon"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")

                    if "lon" in df.columns:
                        df["lon"] = df["lon"].apply(
                            lambda x: -abs(x) if pd.notna(x) else x
                        )

                if is_hotel:
                    required_cols = ["Property Zip Code", "Class_Num", "VPR"]
                else:
                    required_cols = ["Property Zip Code", "VPU"]

                st.subheader("Diagnostics / Hints")

                missing_subj_cols = [c for c in required_cols if c not in subj.columns]
                missing_src_cols = [c for c in required_cols if c not in src.columns]

                if missing_subj_cols:
                    st.error(f"Subject file is missing required columns: {missing_subj_cols}")
                if missing_src_cols:
                    st.error(f"Data Source file is missing required columns: {missing_src_cols}")

                if missing_subj_cols or missing_src_cols:
                    st.stop()

                before_subj = len(subj)
                before_src = len(src)

                st.write("### Null / invalid counts in required columns (Subject)")
                for c in required_cols:
                    if c in subj.columns:
                        st.write(f"- {c}: {subj[c].isna().sum()} nulls")

                st.write("### Null / invalid counts in required columns (Source)")
                for c in required_cols:
                    if c in src.columns:
                        st.write(f"- {c}: {src[c].isna().sum()} nulls")

                subj_valid = subj.dropna(subset=[c for c in required_cols if c in subj.columns])
                src_valid = src.dropna(subset=[c for c in required_cols if c in src.columns])

                st.write(f"Subject rows before filter: {before_subj}, after filter: {len(subj_valid)}")
                st.write(f"Source rows before filter: {before_src}, after filter: {len(src_valid)}")

                if len(subj_valid) == 0:
                    st.error(
                        "All subject rows were dropped because at least one required column "
                        "is null or invalid on every row. Check the null counts above and fix "
                        "those columns in Excel."
                    )

                subj = subj_valid
                src = src_valid

                if len(subj) == 0 or len(src) == 0:
                    st.stop()

                if is_hotel:
                    OUTPUT_COLS = OUTPUT_COLS_HOTEL
                    metric_field = "VPR"
                else:
                    OUTPUT_COLS = OUTPUT_COLS_OTHER
                    metric_field = "VPU"

                results = []
                total_subj = len(subj)
                prog_bar = st.progress(0)

                for i, (_, srow) in enumerate(subj.iterrows()):
                    comps = find_comps(
                        srow,
                        src,
                        is_hotel=is_hotel,
                        use_hotel_class_rule=use_hotel_class_rule,
                        max_radius_miles=max_radius,
                        max_gap_pct_main=max_gap_pct_main,
                        max_gap_pct_value=max_gap_pct_value,
                        max_gap_pct_size=max_gap_pct_size,
                        max_comps=max_comps,
                        use_strict_distance=use_strict_distance,
                        use_county_match=use_county_match,
                        sort_mode=sort_mode,
                    )

                    row = {}
                    for c in OUTPUT_COLS:
                        row[f"Subject_{c}"] = get_val(srow, c)

                    for k in range(max_comps):
                        prefix = f"Comp{k+1}"
                        if k < len(comps):
                            crow = comps[k]
                            for c in OUTPUT_COLS:
                                row[f"{prefix}_{c}"] = get_val(crow, c)
                            row[f"{prefix}_Match_Method"] = crow.get("Match_Method", "N/A")
                            d = crow.get("Distance_Calc", "N/A")
                            row[f"{prefix}_Distance_Miles"] = (
                                f"{d:.2f}" if isinstance(d, (int, float)) else d
                            )
                            diff = crow.get(f"{metric_field}_Diff", "")
                            row[f"{prefix}_{metric_field}_Gap"] = (
                                f"{diff:.2f}" if isinstance(diff, (int, float)) else diff
                            )
                        else:
                            for c in OUTPUT_COLS:
                                row[f"{prefix}_{c}"] = ""
                            row[f"{prefix}_Match_Method"] = ""
                            row[f"{prefix}_Distance_Miles"] = ""
                            row[f"{prefix}_{metric_field}_Gap"] = ""

                    # --- Overpaid calculation (optional) ---
                    if use_overpaid:
                        comp_metrics = []
                        for k2 in range(max_comps):
                            p2 = f"Comp{k2+1}"
                            col_name = f"{p2}_{metric_field}"
                            val = row.get(col_name, None)
                            if val not in (None, "", "N/A"):
                                try:
                                    comp_metrics.append(float(val))
                                except Exception:
                                    pass

                        if len(comp_metrics) > 0:
                            median_metric = float(pd.Series(comp_metrics).median())

                            if overpaid_base_dim:
                                if overpaid_base_dim == "Rooms":
                                    subj_dim = srow.get("Rooms", 0)
                                elif overpaid_base_dim == "Units":
                                    subj_dim = srow.get("Units", 0)
                                else:  # GBA
                                    subj_dim = srow.get("GBA", 0)
                            else:
                                subj_dim = 0

                            try:
                                subj_dim = float(subj_dim)
                            except Exception:
                                subj_dim = 0.0

                            step2_val = median_metric * subj_dim
                            step3_val = step2_val * overpaid_pct

                            if is_hotel:
                                subj_mv = srow.get("Market Value-2023", 0)
                            else:
                                subj_mv = srow.get("Total Market value-2023", 0)
                            try:
                                subj_mv = float(subj_mv)
                            except Exception:
                                subj_mv = 0.0
                            step4_val = subj_mv * overpaid_pct

                            overpaid_val = step4_val - step3_val
                        else:
                            overpaid_val = ""

                        row["Subject_Overpaid_Value"] = overpaid_val
                    else:
                        row["Subject_Overpaid_Value"] = ""

                    results.append(row)
                    prog_bar.progress((i + 1) / total_subj)

                df_final = pd.DataFrame(results)

                st.success(f"✅ Done! Processed {total_subj} subjects.")
                st.dataframe(df_final.head())

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_final.to_excel(writer, index=False)

                st.download_button(
                    label="📥 Download Results (Excel)",
                    data=buffer.getvalue(),
                    file_name="Automated_Comps_Results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please upload both Subject and Data Source Excel files to begin.")

# ---------- BUY ME A COFFEE FLOATING BUTTON ----------
import streamlit.components.v1 as components

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; margin-top: 10px;">
        <a href="https://www.buymeacoffee.com/vigneshna" target="_blank"
           style="
             background-color:#FF813F;
             color:white;
             padding:10px 18px;
             border-radius:6px;
             text-decoration:none;
             font-weight:600;
             font-family: 'Segoe UI', sans-serif;
           ">
           ☕ Buy me a coffee
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)











