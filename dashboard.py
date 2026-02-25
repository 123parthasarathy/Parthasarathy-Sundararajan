import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
import glob

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SRMIST Placement Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GOOGLE COLOR PALETTE & DESIGN CONSTANTS
# ============================================================
GOOGLE_BLUE = "#4285F4"
GOOGLE_GREEN = "#34A853"
GOOGLE_YELLOW = "#FBBC04"
GOOGLE_RED = "#EA4335"
GOOGLE_PURPLE = "#A142F4"
GOOGLE_CYAN = "#24C1E0"
GOOGLE_ORANGE = "#FA7B17"
GOOGLE_PINK = "#F538A0"

# Year-specific colors (Google palette)
YEAR_COLORS = {
    "2023": GOOGLE_BLUE,
    "2024": GOOGLE_GREEN,
    "2025": GOOGLE_YELLOW,
    "2026": GOOGLE_RED,
    "2027": GOOGLE_PURPLE,
    "2028": GOOGLE_CYAN,
    "2029": GOOGLE_ORANGE,
    "2030": GOOGLE_PINK,
}

# Category colors
CAT_COLORS = {
    "Marquee (>=20 LPA)": "#ec4899",
    "Super Dream (10-20 LPA)": "#60a5fa",
    "Dream (5-10 LPA)": "#34d399",
    "Others (<5 LPA)": "#fb923c",
    "Internship": GOOGLE_PURPLE,
}

# Multi-color palette for branches/departments
BRANCH_COLORS = [GOOGLE_BLUE, GOOGLE_GREEN, GOOGLE_YELLOW, GOOGLE_RED,
                 GOOGLE_PURPLE, GOOGLE_CYAN, GOOGLE_ORANGE, GOOGLE_PINK]

# Common chart layout
CHART_FONT = dict(family="Google Sans, Segoe UI, Arial, sans-serif", size=13, color="#202124")
TITLE_FONT = dict(family="Google Sans, Segoe UI, Arial, sans-serif", size=20, color="#202124")
CHART_BG = "#FFFFFF"
PLOT_BG = "#FAFAFA"
GRID_COLOR = "rgba(0,0,0,0.06)"

# ============================================================
# CUSTOM CSS - COLORFUL GOOGLE-STYLE THEME
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4285F4, #EA4335, #FBBC04, #34A853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.8rem 0 0.2rem 0;
        font-family: 'Google Sans', 'Segoe UI', sans-serif;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5f6368;
        text-align: center;
        margin-bottom: 1.5rem;
        font-family: 'Google Sans', 'Segoe UI', sans-serif;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
        border: 2px solid #e8eaed;
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    div[data-testid="stMetric"] label {
        color: #5f6368 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #202124 !important;
        font-weight: 700 !important;
    }
    .batch-card-blue { background: linear-gradient(135deg, #4285F4, #5B9CF6); color: white; padding: 1rem; border-radius: 14px; text-align: center; }
    .batch-card-green { background: linear-gradient(135deg, #34A853, #57BB6D); color: white; padding: 1rem; border-radius: 14px; text-align: center; }
    .batch-card-yellow { background: linear-gradient(135deg, #FBBC04, #FDD663); color: #202124; padding: 1rem; border-radius: 14px; text-align: center; }
    .batch-card-red { background: linear-gradient(135deg, #EA4335, #F07068); color: white; padding: 1rem; border-radius: 14px; text-align: center; }
    .batch-card-purple { background: linear-gradient(135deg, #A142F4, #B86FF6); color: white; padding: 1rem; border-radius: 14px; text-align: center; }
    .card-value { font-size: 2.2rem; font-weight: 700; }
    .card-label { font-size: 0.85rem; opacity: 0.9; margin-top: 4px; }
    .card-title { font-size: 0.95rem; font-weight: 600; margin-bottom: 6px; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e8f0fe 100%);
    }
    h2, h3 { color: #202124 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BRANCH_MAP = {
    "CS": "CSE", "AIML": "CSE", "IOT": "CSE", "CC": "CSE", "CSBS": "CSE",
    "BDA": "CSE", "GT": "CSE", "CSE - BDA": "CSE", "CSE with BDA": "CSE",
    "CSE - CS": "CSE", "CSE-CS": "CSE", "CSE - AIML": "CSE", "CSE-AIML": "CSE",
    "CSE - IOT": "CSE", "CSE-IOT": "CSE", "CSE - CC": "CSE",
    "BME": "BIO MED", "BIO TECH": "BIO TECH", "Bio Tech": "BIO TECH",
    "BIO MED": "BIO MED", "BIO TECH ": "BIO TECH",
    "EKE": "ECE",
}

CATEGORY_MAP = {
    "M": "Marquee (>=20 LPA)",
    "SD": "Super Dream (10-20 LPA)",
    "SD / D1": "Super Dream (10-20 LPA)",
    "D": "Dream (5-10 LPA)",
    "D1": "Dream (5-10 LPA)",
    "D2": "Dream (5-10 LPA)",
    "D1/D": "Dream (5-10 LPA)",
    "Core": "Others (<5 LPA)",
    "C": "Others (<5 LPA)",
    "I": "Internship",
}


def find_batch_files():
    """Auto-detect all SRMIST_*BATCH*.xlsx files."""
    pattern = os.path.join(BASE_DIR, "SRMIST_*BATCH*.xlsx")
    files = glob.glob(pattern)
    pattern2 = os.path.join(BASE_DIR, "srmist_*batch*.xlsx")
    files += glob.glob(pattern2)
    files = list(set(files))
    batch_files = {}
    for f in files:
        match = re.search(r"(\d{4})", os.path.basename(f))
        if match:
            batch_files[int(match.group(1))] = f
    return batch_files


def parse_ctc(val):
    """Parse CTC values - handles '10 LPA', '10', numeric, etc."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip().upper().replace("LPA", "").replace("L", "").strip()
    try:
        return float(val)
    except ValueError:
        return None


def parse_dept_2023(dept_str):
    """Parse 2023-style dept like 'B.TECH ( CSE )' -> 'CSE'."""
    if pd.isna(dept_str):
        return "OTHER"
    dept = str(dept_str).strip()
    match = re.search(r"\(\s*([^)]+)\s*\)", dept)
    if match:
        return match.group(1).strip().upper()
    return dept.upper()


def clean_student_df(df, year):
    """Clean and normalize student data from any batch year."""
    df.columns = df.columns.str.strip()

    # ---- 2023 format: S.NO., REGISTER NUMBER, STUDENT NAME, DEPT, COMPANY, PACKAGE ----
    if "PACKAGE" in df.columns and "DEPT" in df.columns:
        df = df.rename(columns={
            "S.NO.": "SNo", "REGISTER NUMBER": "RegNo", "STUDENT NAME": "Name",
            "DEPT": "Branch_Raw", "COMPANY": "Company", "PACKAGE": "CTC_Raw",
        })
        if "Name" in df.columns:
            df = df.dropna(subset=["Name"])
        df["CTC"] = df["CTC_Raw"].apply(parse_ctc)
        df["Branch"] = df["Branch_Raw"].apply(parse_dept_2023)
        df["Category_Full"] = "Others (<5 LPA)"  # 2023 has no category column
        # Assign category based on CTC
        df.loc[df["CTC"] >= 20, "Category_Full"] = "Marquee (>=20 LPA)"
        df.loc[(df["CTC"] >= 10) & (df["CTC"] < 20), "Category_Full"] = "Super Dream (10-20 LPA)"
        df.loc[(df["CTC"] >= 5) & (df["CTC"] < 10), "Category_Full"] = "Dream (5-10 LPA)"

    # ---- 2024/2025/2026 format ----
    else:
        col_mapping = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("s.no", "s.no.", "sno"):
                col_mapping[col] = "SNo"
            elif "register" in cl:
                col_mapping[col] = "RegNo"
            elif "student name" in cl or cl == "name":
                col_mapping[col] = "Name"
            elif cl in ("progrm", "program"):
                col_mapping[col] = "Program"
            elif cl in ("branch", "dept"):
                col_mapping[col] = "Branch"
            elif "offer" in cl and "category" not in cl and "compan" not in cl:
                col_mapping[col] = "Company"
            elif "offered" in cl:
                col_mapping[col] = "Company"
            elif "ctc" in cl:
                col_mapping[col] = "CTC"
            elif "stipend" in cl:
                col_mapping[col] = "Stipend"
            elif "category" in cl:
                col_mapping[col] = "Category"
            elif "through" in cl or "source" in cl:
                col_mapping[col] = "Source"
        df = df.rename(columns=col_mapping)

        if "Name" in df.columns:
            df = df.dropna(subset=["Name"])
        if "CTC" in df.columns:
            df["CTC"] = df["CTC"].apply(parse_ctc)
        if "Branch" in df.columns:
            df["Branch"] = df["Branch"].astype(str).str.strip().str.replace("\xa0", "").str.upper()
        if "Category" in df.columns:
            df["Category"] = df["Category"].astype(str).str.strip()
            df["Category_Full"] = df["Category"].map(CATEGORY_MAP).fillna("Others (<5 LPA)")
        else:
            df["Category_Full"] = "Others (<5 LPA)"
            if "CTC" in df.columns:
                df.loc[df["CTC"] >= 20, "Category_Full"] = "Marquee (>=20 LPA)"
                df.loc[(df["CTC"] >= 10) & (df["CTC"] < 20), "Category_Full"] = "Super Dream (10-20 LPA)"
                df.loc[(df["CTC"] >= 5) & (df["CTC"] < 10), "Category_Full"] = "Dream (5-10 LPA)"

    # Normalize branch names
    if "Branch" in df.columns:
        df["Branch"] = df["Branch"].replace(BRANCH_MAP)
        # Remove any remaining "B.TECH" prefix artifacts
        df["Branch"] = df["Branch"].str.replace(r"^B\.?TECH\.?\s*", "", regex=True).str.strip()
        df["Branch"] = df["Branch"].replace(BRANCH_MAP)
    if "RegNo" not in df.columns:
        df["RegNo"] = range(len(df))

    df["Year"] = year
    return df


@st.cache_data
def load_all_data():
    """Load student data from all detected batch files."""
    batch_files = find_batch_files()
    all_students = []
    extra_sheets = {}

    for year, filepath in sorted(batch_files.items()):
        try:
            xl = pd.ExcelFile(filepath)
            sheets = xl.sheet_names

            # Find student sheet
            df = None
            for sname in sheets:
                sl = sname.lower()
                if "student" in sl and ("list" in sl or "placed" in sl):
                    header_row = 0
                    # Check if first row is a title row
                    test = pd.read_excel(filepath, sheet_name=sname, header=None, nrows=3)
                    for r in range(min(3, len(test))):
                        row_vals = [str(v).lower() for v in test.iloc[r] if pd.notna(v)]
                        if any(kw in " ".join(row_vals) for kw in ["register", "student name", "s.no", "name"]):
                            header_row = r
                            break
                    df = pd.read_excel(filepath, sheet_name=sname, header=header_row)
                    break

            if df is not None:
                df = clean_student_df(df, year)
                all_students.append(df)

            # Extra sheets
            extra = {}
            for sname in sheets:
                sl = sname.lower()
                if ("company" in sl or "compaine" in sl) and ("count" in sl or "placed" in sl or "wise" in sl):
                    extra["company_count"] = pd.read_excel(filepath, sheet_name=sname, header=1)
                elif "snapshot" in sl or "snapshort" in sl:
                    extra["snapshot"] = pd.read_excel(filepath, sheet_name=sname, header=1)
                elif "visited" in sl:
                    extra["visited"] = pd.read_excel(filepath, sheet_name=sname, header=0)
                elif "day 1" in sl or "day1" in sl:
                    extra["day1"] = pd.read_excel(filepath, sheet_name=sname, header=2)
            extra_sheets[year] = extra

        except Exception as e:
            st.warning(f"Error loading {os.path.basename(filepath)}: {e}")

    combined = pd.concat(all_students, ignore_index=True) if all_students else pd.DataFrame()
    return combined, batch_files, extra_sheets


df_all, batch_files, extra_sheets = load_all_data()
all_years = sorted(batch_files.keys())

# ============================================================
# COMMON CHART TEMPLATE
# ============================================================
def base_layout(**kwargs):
    """Return a consistent plotly layout matching Google style."""
    layout = dict(
        template="plotly_white",
        font=CHART_FONT,
        title_font=TITLE_FONT,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=60, r=40, t=110, b=60),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05,
            xanchor="center", x=0.5, font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(y=0.97),
    )
    layout.update(kwargs)
    return layout


def get_color(year):
    return YEAR_COLORS.get(str(year), "#6b7280")


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## 🔍 Filters")

st.sidebar.markdown("### 📁 Loaded Batch Files")
card_styles = ["batch-card-blue", "batch-card-green", "batch-card-yellow", "batch-card-red", "batch-card-purple"]
for i, (y, f) in enumerate(sorted(batch_files.items())):
    style = card_styles[i % len(card_styles)]
    count = len(df_all[df_all["Year"] == y]) if not df_all.empty else 0
    st.sidebar.markdown(f"✅ **{y} Batch** — {count:,} records")

st.sidebar.markdown("---")
st.sidebar.info("📌 **Add new year:** Place `SRMIST_2027 BATCH.xlsx` in the folder and refresh!")
st.sidebar.markdown("---")

# Year filter
selected_years = st.sidebar.multiselect(
    "📅 Batch Year", [str(y) for y in all_years], default=[str(y) for y in all_years])

# Branch filter
if not df_all.empty:
    all_branches = sorted(df_all["Branch"].dropna().unique().tolist())
    all_branches = [b for b in all_branches if b not in ("nan", "NAN", "")]
else:
    all_branches = []
selected_branches = st.sidebar.multiselect("🏫 Branch", all_branches, default=all_branches)

# Category filter
if not df_all.empty:
    all_categories = sorted(df_all["Category_Full"].dropna().unique().tolist())
else:
    all_categories = []
selected_categories = st.sidebar.multiselect("📊 Offer Category", all_categories, default=all_categories)

# CTC range
if not df_all.empty and not df_all["CTC"].isna().all():
    min_ctc = float(df_all["CTC"].min())
    max_ctc = float(df_all["CTC"].max())
else:
    min_ctc, max_ctc = 0.0, 100.0
ctc_range = st.sidebar.slider("💰 CTC Range (LPA)", min_value=min_ctc, max_value=max_ctc, value=(min_ctc, max_ctc))

company_search = st.sidebar.text_input("🔎 Search Company", "")

# Apply filters
if not df_all.empty:
    filtered = df_all[
        (df_all["Year"].astype(str).isin(selected_years))
        & (df_all["Branch"].isin(selected_branches))
        & (df_all["Category_Full"].isin(selected_categories))
        & (df_all["CTC"] >= ctc_range[0])
        & (df_all["CTC"] <= ctc_range[1])
    ].copy()
    if company_search:
        filtered = filtered[filtered["Company"].str.contains(company_search, case=False, na=False)]
else:
    filtered = pd.DataFrame()

# ============================================================
# HEADER
# ============================================================
year_range = f"{min(all_years)} - {max(all_years)}" if all_years else "N/A"
st.markdown('<div class="main-header">SRMIST Placement Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Comprehensive Placement Statistics ({year_range}) &bull; Live from Excel Data &bull; Auto-updates on refresh</div>', unsafe_allow_html=True)

# ============================================================
# BATCH SUMMARY CARDS
# ============================================================
card_cols = st.columns(len(all_years))
card_css = ["batch-card-blue", "batch-card-green", "batch-card-yellow", "batch-card-red", "batch-card-purple"]

for i, y in enumerate(all_years):
    year_df = df_all[df_all["Year"] == y] if not df_all.empty else pd.DataFrame()
    n_companies = year_df["Company"].nunique() if not year_df.empty else 0
    n_students = year_df["RegNo"].nunique() if not year_df.empty else 0
    css = card_css[i % len(card_css)]
    card_cols[i].markdown(f"""
    <div class="{css}">
        <div class="card-title">{y} Batch</div>
        <div class="card-value">{n_companies}</div>
        <div class="card-label">Companies &bull; {n_students:,} Students</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# KPI METRICS
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)

if not filtered.empty:
    total_offers = len(filtered)
    unique_students = filtered["RegNo"].nunique()
    unique_companies = filtered["Company"].nunique()
    avg_ctc_val = filtered["CTC"].mean()
    max_ctc_val = filtered["CTC"].max()
else:
    total_offers = unique_students = unique_companies = 0
    avg_ctc_val = max_ctc_val = None

col1.metric("Total Offers", f"{total_offers:,}")
col2.metric("Unique Students", f"{unique_students:,}")
col3.metric("Companies", f"{unique_companies:,}")
col4.metric("Avg CTC (LPA)", f"{avg_ctc_val:.2f}" if pd.notna(avg_ctc_val) else "N/A")
col5.metric("Max CTC (LPA)", f"{max_ctc_val:.1f}" if pd.notna(max_ctc_val) else "N/A")

st.markdown("---")

# ============================================================
# HELPER
# ============================================================
def get_year_stats(y):
    if df_all.empty:
        return {"total_placed": 0, "marquee": 0, "super_dream": 0, "dream": 0, "others": 0}
    ydf = df_all[df_all["Year"] == y]
    cats = ydf["Category_Full"].value_counts().to_dict()
    return {
        "total_placed": ydf["RegNo"].nunique(),
        "marquee": cats.get("Marquee (>=20 LPA)", 0),
        "super_dream": cats.get("Super Dream (10-20 LPA)", 0),
        "dream": cats.get("Dream (5-10 LPA)", 0),
        "others": cats.get("Others (<5 LPA)", 0),
    }

# ============================================================
# ROW 1: Year-wise Total Placements + Stacked Category
# ============================================================
st.subheader(f"Year-wise Placement Trends ({year_range})")
r1c1, r1c2 = st.columns(2)

with r1c1:
    year_labels = [str(y) for y in all_years]
    placements = [get_year_stats(y)["total_placed"] for y in all_years]
    colors_y = [get_color(y) for y in all_years]

    fig = go.Figure(data=[go.Bar(
        x=year_labels, y=placements, marker_color=colors_y,
        text=placements, textposition="outside",
        textfont=dict(size=16, color="#202124", family="Google Sans, sans-serif"),
        marker=dict(line=dict(width=0), cornerradius=6),
        width=0.55,
    )])
    fig.update_layout(**base_layout(
        title="Total Students Placed per Year",
        height=450,
        yaxis=dict(range=[0, max(placements) * 1.2] if placements else [0, 100], gridcolor=GRID_COLOR),
    ))
    st.plotly_chart(fig, width="stretch")

with r1c2:
    fig = go.Figure()
    cat_keys = ["marquee", "super_dream", "dream", "others"]
    cat_labels = ["Marquee (>=20 LPA)", "Super Dream (10-20 LPA)", "Dream (5-10 LPA)", "Others (<5 LPA)"]
    # Colors matching sample: Pink, Cornflower Blue, Emerald Green, Orange
    cat_c = ["#ec4899", "#60a5fa", "#34d399", "#fb923c"]

    for i, (key, label) in enumerate(zip(cat_keys, cat_labels)):
        vals = [get_year_stats(y)[key] for y in all_years]
        display_text = [str(v) if v > 50 else "" for v in vals]
        text_color = "white" if i < 2 else ("#202124" if i == 3 else "white")
        fig.add_trace(go.Bar(
            name=label, x=year_labels, y=vals, marker_color=cat_c[i],
            text=display_text, textposition="inside",
            textfont=dict(size=13, color=text_color),
            marker=dict(line=dict(width=0)),
        ))
    fig.update_layout(**base_layout(
        title=dict(text="Package Category Distribution by Year", y=0.98),
        barmode="stack", height=480,
        margin=dict(l=60, r=40, t=100, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.06,
                    xanchor="center", x=0.5, font=dict(size=10)),
    ))
    st.plotly_chart(fig, width="stretch")

# ============================================================
# ROW 2: Branch-wise Placements + Avg CTC
# ============================================================
if not filtered.empty:
    st.subheader("Branch-wise Placement Analysis")
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        bc = filtered.groupby("Branch")["RegNo"].nunique().sort_values(ascending=True).reset_index()
        bc.columns = ["Branch", "Students"]
        colors_b = [BRANCH_COLORS[i % len(BRANCH_COLORS)] for i in range(len(bc))]

        fig = go.Figure(data=[go.Bar(
            x=bc["Students"], y=bc["Branch"], orientation="h",
            marker_color=colors_b,
            text=bc["Students"], textposition="outside",
            textfont=dict(size=13, color="#202124"),
            marker=dict(line=dict(width=0), cornerradius=5),
        )])
        fig.update_layout(**base_layout(
            title="Students Placed by Branch", height=450,
            xaxis=dict(gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=13, color="#202124")),
        ))
        st.plotly_chart(fig, width="stretch")

    with r2c2:
        ac = filtered.groupby("Branch")["CTC"].mean().sort_values(ascending=True).reset_index()
        ac.columns = ["Branch", "Avg CTC"]
        ac["Avg CTC"] = ac["Avg CTC"].round(2)
        colors_a = [BRANCH_COLORS[i % len(BRANCH_COLORS)] for i in range(len(ac))]

        fig = go.Figure(data=[go.Bar(
            x=ac["Avg CTC"], y=ac["Branch"], orientation="h",
            marker_color=colors_a,
            text=[f"{v:.1f} LPA" for v in ac["Avg CTC"]], textposition="outside",
            textfont=dict(size=13, color="#202124"),
            marker=dict(line=dict(width=0), cornerradius=5),
        )])
        fig.update_layout(**base_layout(
            title="Average Package by Department", height=450,
            xaxis=dict(gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=13, color="#202124")),
        ))
        st.plotly_chart(fig, width="stretch")

# ============================================================
# ROW 3: CTC Distribution + Category Pie
# ============================================================
if not filtered.empty:
    st.subheader("CTC & Offer Category Analysis")
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        # CTC Range buckets (like sample graph)
        bins = [0, 3, 5, 6, 8, 10, 15, 20, 30, float("inf")]
        labels_b = ["0-3 LPA", "3-5 LPA", "5-6 LPA", "6-8 LPA", "8-10 LPA",
                     "10-15 LPA", "15-20 LPA", "20-30 LPA", "30+ LPA"]
        # Colors matching sample: gray, yellow-green, green, teal, dark green, blue, rose, pink, red
        range_colors = ["#d4d4d8", "#a3e635", "#22c55e", "#14b8a6", "#059669",
                        "#3b82f6", "#fb7185", "#f472b6", "#e11d48"]

        filtered_ctc = filtered["CTC"].dropna()
        range_counts = pd.cut(filtered_ctc, bins=bins, labels=labels_b, right=False).value_counts().reindex(labels_b).fillna(0)

        fig = go.Figure(data=[go.Bar(
            x=range_counts.index, y=range_counts.values,
            marker_color=range_colors,
            text=range_counts.values.astype(int), textposition="outside",
            textfont=dict(size=12, color="#202124"),
            marker=dict(line=dict(width=0), cornerradius=4),
        )])
        fig.update_layout(**base_layout(
            title="Package Range Distribution", height=450,
            xaxis=dict(tickangle=0, tickfont=dict(size=10)),
            yaxis=dict(gridcolor=GRID_COLOR),
        ))
        st.plotly_chart(fig, width="stretch")

    with r3c2:
        cat_counts = filtered["Category_Full"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        cat_color_map = [CAT_COLORS.get(c, "#ccc") for c in cat_counts["Category"]]

        total = cat_counts["Count"].sum()
        # For small slices: show only percent inside; for large: value+percent
        custom_text = []
        for _, row in cat_counts.iterrows():
            pct = row["Count"] / total * 100
            if pct < 5:
                custom_text.append(f"{pct:.1f}%")
            else:
                custom_text.append(f"{row['Count']}<br>({pct:.1f}%)")

        fig = go.Figure(data=[go.Pie(
            labels=cat_counts["Category"], values=cat_counts["Count"],
            marker=dict(colors=cat_color_map, line=dict(color="white", width=3)),
            hole=0.45,
            text=custom_text,
            textinfo="text",
            textposition="inside",
            textfont=dict(size=13, color="white"),
            insidetextorientation="horizontal",
            pull=[0.05 if v < total * 0.05 else 0 for v in cat_counts["Count"]],
        )])
        fig.update_layout(**base_layout(
            title="CTC Distribution by Category", height=450,
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, font=dict(size=11)),
        ))
        st.plotly_chart(fig, width="stretch")

# ============================================================
# ROW 4: Top Companies
# ============================================================
if not filtered.empty:
    st.subheader("Top Recruiting Companies")
    r4c1, r4c2 = st.columns(2)

    with r4c1:
        top = filtered.groupby("Company").agg(
            Hires=("RegNo", "nunique"), Avg_CTC=("CTC", "mean"),
        ).sort_values("Hires", ascending=False).head(15).reset_index()
        top["Avg_CTC"] = top["Avg_CTC"].round(1)

        fig = go.Figure(data=[go.Bar(
            x=top["Company"], y=top["Hires"],
            marker_color=GOOGLE_BLUE,
            text=top["Hires"], textposition="outside",
            textfont=dict(size=11, color="#202124"),
            marker=dict(line=dict(width=0), cornerradius=4),
        )])
        fig.update_layout(**base_layout(
            title="Top 15 Companies by Hires", height=480,
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        ))
        st.plotly_chart(fig, width="stretch")

    with r4c2:
        top_ctc = filtered.groupby("Company").agg(
            Max_CTC=("CTC", "max"), Hires=("RegNo", "nunique"),
        ).sort_values("Max_CTC", ascending=False).head(15).reset_index()

        fig = go.Figure(data=[go.Bar(
            x=top_ctc["Company"], y=top_ctc["Max_CTC"],
            marker_color=GOOGLE_RED,
            text=[f"{v:.1f}" for v in top_ctc["Max_CTC"]], textposition="outside",
            textfont=dict(size=11, color="#202124"),
            marker=dict(line=dict(width=0), cornerradius=4),
        )])
        fig.update_layout(**base_layout(
            title="Top 15 Companies by Highest CTC (LPA)", height=480,
            xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        ))
        st.plotly_chart(fig, width="stretch")

# ============================================================
# ROW 5: Department-wise Year Comparison (Grouped Bar)
# ============================================================
st.subheader("Department-wise Placements Across Years")

dept_order = ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"]
fig = go.Figure()

max_dept_val = 0
for y in all_years:
    if str(y) not in selected_years:
        continue
    ydf = df_all[df_all["Year"] == y]
    dept_data = ydf.groupby("Branch")["RegNo"].nunique().to_dict()
    vals = [dept_data.get(d, 0) for d in dept_order]
    max_dept_val = max(max_dept_val, max(vals) if vals else 0)
    fig.add_trace(go.Bar(
        name=str(y), x=dept_order, y=vals,
        marker_color=get_color(y),
        text=vals, textposition="outside",
        textfont=dict(size=9, color="#202124"),
        marker=dict(line=dict(width=0), cornerradius=4),
        width=0.18,
    ))

fig.update_layout(**base_layout(
    title=f"Department-wise Placements ({year_range})",
    barmode="group", height=500,
    xaxis=dict(tickfont=dict(size=14, color="#202124")),
    yaxis=dict(range=[0, max_dept_val * 1.18], gridcolor=GRID_COLOR),
    bargap=0.3, bargroupgap=0.05,
    uniformtext=dict(minsize=8, mode="hide"),
))
st.plotly_chart(fig, width="stretch")

# ============================================================
# ROW 6: Package Category Distribution (Grouped Bar)
# ============================================================
st.subheader("Package Category Comparison Across Years")

cat_order = ["Marquee (>=20 LPA)", "Super Dream (10-20 LPA)", "Dream (5-10 LPA)", "Others (<5 LPA)"]
cat_short = ["Marquee (≥20L)", "Super Dream (10-20L)", "Dream (5-10L)", "Others (<5L)"]
fig = go.Figure()

max_cat_val = 0
for y in all_years:
    if str(y) not in selected_years:
        continue
    stats = get_year_stats(y)
    vals = [stats["marquee"], stats["super_dream"], stats["dream"], stats["others"]]
    max_cat_val = max(max_cat_val, max(vals) if vals else 0)
    fig.add_trace(go.Bar(
        name=str(y), x=cat_short, y=vals,
        marker_color=get_color(y),
        text=vals, textposition="outside",
        textfont=dict(size=9, color="#202124"),
        marker=dict(line=dict(width=0), cornerradius=4),
        width=0.18,
    ))

fig.update_layout(**base_layout(
    title=f"Package Category Distribution ({year_range})",
    barmode="group", height=500,
    xaxis=dict(tickfont=dict(size=11, color="#202124")),
    yaxis=dict(range=[0, max_cat_val * 1.18], gridcolor=GRID_COLOR),
    bargap=0.3, bargroupgap=0.05,
    uniformtext=dict(minsize=8, mode="hide"),
))
st.plotly_chart(fig, width="stretch")

# ============================================================
# ROW 7: Marquee Companies (Dual Bar)
# ============================================================
if not filtered.empty:
    marquee_df = filtered[filtered["CTC"] >= 20].copy()
    if not marquee_df.empty:
        st.subheader("Companies Offering Marquee Packages (≥20 LPA)")
        mc = marquee_df.groupby("Company").agg(
            Max_CTC=("CTC", "max"), Students=("RegNo", "nunique"),
        ).sort_values("Max_CTC", ascending=False).head(15).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            name="CTC (LPA)", x=mc["Company"], y=mc["Max_CTC"],
            marker_color=GOOGLE_PINK, text=mc["Max_CTC"], textposition="outside",
            textfont=dict(size=10, color=GOOGLE_PINK),
            marker=dict(line=dict(width=0), cornerradius=4), opacity=0.9,
        ), secondary_y=False)
        fig.add_trace(go.Bar(
            name="Students", x=mc["Company"], y=mc["Students"],
            marker_color=GOOGLE_BLUE, text=mc["Students"], textposition="outside",
            textfont=dict(size=10, color=GOOGLE_BLUE),
            marker=dict(line=dict(width=0), cornerradius=4), opacity=0.9,
        ), secondary_y=True)
        fig.update_layout(**base_layout(
            title="Companies Offering Marquee Packages (≥20 LPA)",
            barmode="group", height=480, xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        ))
        fig.update_yaxes(title_text="CTC (LPA)", title_font=dict(color=GOOGLE_PINK), secondary_y=False)
        fig.update_yaxes(title_text="Students", title_font=dict(color=GOOGLE_BLUE), secondary_y=True)
        st.plotly_chart(fig, width="stretch")

# ============================================================
# ROW 8: Premium Placements Trend (Line with Fill)
# ============================================================
st.subheader(f"Premium Placements Trend ({year_range})")

marquee_vals = [get_year_stats(y)["marquee"] for y in all_years]
sd_vals = [get_year_stats(y)["super_dream"] for y in all_years]
total_premium = [m + s for m, s in zip(marquee_vals, sd_vals)]
year_labels = [str(y) for y in all_years]

fig = go.Figure()
# Fill areas
fig.add_trace(go.Scatter(
    x=year_labels, y=sd_vals, fill="tozeroy", fillcolor="rgba(66,133,244,0.12)",
    line=dict(color=GOOGLE_BLUE, width=3), mode="lines+markers+text",
    name="Super Dream (10-20 LPA)", marker=dict(size=10),
    text=sd_vals, textposition="top left", textfont=dict(size=12, color=GOOGLE_BLUE),
))
fig.add_trace(go.Scatter(
    x=year_labels, y=marquee_vals, fill="tozeroy", fillcolor="rgba(245,56,160,0.12)",
    line=dict(color=GOOGLE_PINK, width=3), mode="lines+markers+text",
    name="Marquee (≥20 LPA)", marker=dict(size=10),
    text=marquee_vals, textposition="bottom right", textfont=dict(size=12, color=GOOGLE_PINK),
))
fig.add_trace(go.Scatter(
    x=year_labels, y=total_premium,
    line=dict(color=GOOGLE_PURPLE, width=3, dash="dash"), mode="lines+markers+text",
    name="Total Premium", marker=dict(size=10),
    text=total_premium, textposition="top right", textfont=dict(size=12, color=GOOGLE_PURPLE),
))
fig.update_layout(**base_layout(
    title="Premium Placements Trend (Marquee + Super Dream)",
    height=450, yaxis=dict(gridcolor=GRID_COLOR),
))
st.plotly_chart(fig, width="stretch")

# ============================================================
# ROW 9: Year-wise Stacked Category (Stacked Bar)
# ============================================================
st.subheader("Year-wise Placement Distribution by Category")

fig = go.Figure()
cat_keys = ["marquee", "super_dream", "dream", "others"]
cat_labels_full = ["Marquee (≥20 LPA)", "Super Dream (10-20 LPA)", "Dream (5-10 LPA)", "Others (<5 LPA)"]
cat_stacked_c = ["#ec4899", "#60a5fa", "#34d399", "#fb923c"]

for i, (key, label) in enumerate(zip(cat_keys, cat_labels_full)):
    vals = [get_year_stats(y)[key] for y in all_years]
    # Hide text for very small segments to avoid overlap
    display_text = [str(v) if v > 50 else "" for v in vals]
    text_color = "white" if i < 2 else ("#202124" if i == 3 else "white")
    fig.add_trace(go.Bar(
        name=label, x=year_labels, y=vals, marker_color=cat_stacked_c[i],
        text=display_text, textposition="inside",
        textfont=dict(size=13, color=text_color),
        marker=dict(line=dict(width=0)),
        width=0.5,
    ))

fig.update_layout(**base_layout(
    title="Year-wise Placement Distribution by Category",
    barmode="stack", height=480,
    xaxis=dict(tickfont=dict(size=16, color="#202124")),
))
st.plotly_chart(fig, width="stretch")

# ============================================================
# ROW 10: Companies Visited + Day 1 Funnel
# ============================================================
latest_year = max(extra_sheets.keys()) if extra_sheets else None
if latest_year and extra_sheets.get(latest_year):
    extras = extra_sheets[latest_year]

    if "visited" in extras or "day1" in extras:
        st.subheader(f"Companies Visited & Day 1 Status ({latest_year} Batch)")
        r10c1, r10c2 = st.columns(2)

        if "visited" in extras:
            dv = extras["visited"]
            dv.columns = dv.columns.str.strip()
            if "Status" in dv.columns:
                dv["Status"] = dv["Status"].astype(str).str.strip()
                dv["Status"] = dv["Status"].replace({
                    "Close": "Closed", "Closed For Applications": "Closed",
                    "Closed For Applicatio": "Closed", "nan": "Unknown",
                })
                with r10c1:
                    # Horizontal bar chart instead of pie to avoid overlapping
                    sc = dv["Status"].value_counts().sort_values(ascending=True).reset_index()
                    sc.columns = ["Status", "Count"]
                    status_colors = [GOOGLE_BLUE, GOOGLE_GREEN, GOOGLE_RED,
                                     GOOGLE_YELLOW, GOOGLE_PURPLE, GOOGLE_CYAN, GOOGLE_ORANGE, "#ccc"]
                    fig = go.Figure(data=[go.Bar(
                        x=sc["Count"], y=sc["Status"], orientation="h",
                        marker_color=status_colors[:len(sc)],
                        text=sc["Count"], textposition="outside",
                        textfont=dict(size=12, color="#202124"),
                        marker=dict(line=dict(width=0), cornerradius=5),
                    )])
                    fig.update_layout(**base_layout(
                        title=f"Company Visit Status ({len(dv)} Companies)", height=450,
                        yaxis=dict(tickfont=dict(size=11)),
                        xaxis=dict(gridcolor=GRID_COLOR),
                    ))
                    st.plotly_chart(fig, width="stretch")

        if "day1" in extras:
            d1 = extras["day1"]
            d1.columns = d1.columns.str.strip()
            d1_clean = d1.dropna(subset=[d1.columns[0]])
            if not d1_clean.empty:
                with r10c2:
                    def safe_int(v):
                        try:
                            return int(float(str(v).replace("*", "").replace("#", "").strip()))
                        except (ValueError, TypeError):
                            return 0

                    # Only top 5 companies to avoid overlap
                    fig = go.Figure()
                    d1_colors = [GOOGLE_BLUE, GOOGLE_GREEN, GOOGLE_YELLOW, GOOGLE_RED, GOOGLE_PURPLE]
                    stages = ["Applied", "Shortlisted", "Tech Int.", "Final Int.", "Selected"]

                    for idx, (_, row) in enumerate(d1_clean.head(5).iterrows()):
                        company = str(row.iloc[0]).strip()
                        raw_vals = [v for v in row.iloc[1:6] if pd.notna(v)]
                        vals = [safe_int(v) for v in raw_vals]
                        vals = [v for v in vals if v > 0]
                        if vals:
                            fig.add_trace(go.Bar(
                                name=company,
                                x=stages[:len(vals)],
                                y=vals, textposition="none",
                                marker_color=d1_colors[idx % len(d1_colors)],
                                marker=dict(cornerradius=3),
                            ))
                    fig.update_layout(**base_layout(
                        title=dict(text="Day 1 - Selection Funnel (Top 5)", y=0.98),
                        barmode="group", height=480,
                        margin=dict(l=60, r=40, t=110, b=60),
                        legend=dict(orientation="h", yanchor="bottom", y=1.08,
                                    xanchor="center", x=0.5, font=dict(size=9)),
                    ))
                    st.plotly_chart(fig, width="stretch")

# ============================================================
# ROW 11: Data Table
# ============================================================
if not filtered.empty:
    st.subheader("📋 Detailed Placement Data")
    st.markdown(f"Showing **{len(filtered):,}** records based on current filters")

    display_cols = ["Year", "Name", "Branch", "Company", "CTC", "Category_Full", "Stipend", "Source"]
    available_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[available_cols].sort_values(["Year", "CTC"], ascending=[False, False]).reset_index(drop=True),
        width="stretch", height=450,
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    f"<div style='text-align:center; color:#5f6368; font-size:0.9rem; font-family:Google Sans,sans-serif;'>"
    f"<b>SRMIST Placement Dashboard</b> &bull; {len(batch_files)} batch file(s) loaded &bull; "
    f"Drop a new <code>SRMIST_YYYY BATCH.xlsx</code> and refresh! &bull; Built with Streamlit + Plotly"
    f"</div>",
    unsafe_allow_html=True,
)
