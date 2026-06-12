"""
Nhóm 13 – Design Inspiration Recommendation System
Streamlit App
"""

import os, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Design Inspiration – Nhóm 13",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts & base ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hide Streamlit chrome ──────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Page background ────────────────────────────────────────── */
.stApp { background: #f7f8fc; }

/* ── Hero banner ────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 6px 0; }
.hero p  { font-size: 1rem; color: #a8b4d8; margin: 0; }

/* ── Label badges ───────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0 4px 4px 0;
}
.badge-style    { background: #e0e7ff; color: #3730a3; }
.badge-material { background: #dcfce7; color: #166534; }
.badge-sim      { background: #fef3c7; color: #92400e; }

/* ── Result cards ───────────────────────────────────────────── */
.result-card {
    background: white;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 20px;
}
.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.12);
}
.card-body { padding: 16px 18px; }
.card-title { font-size: 0.9rem; font-weight: 600; margin-bottom: 6px; color: #1e293b; }
.card-desc  { font-size: 0.82rem; color: #64748b; line-height: 1.5; margin-bottom: 10px; }

/* ── Color swatches ─────────────────────────────────────────── */
.swatch-row { display: flex; gap: 6px; margin-top: 6px; }
.swatch {
    width: 28px; height: 28px;
    border-radius: 6px;
    border: 2px solid rgba(255,255,255,0.6);
    box-shadow: 0 1px 4px rgba(0,0,0,0.15);
    display: inline-block;
}

/* ── Prediction panel ───────────────────────────────────────── */
.pred-box {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}
.pred-box h4 { font-size: 0.78rem; text-transform: uppercase;
               letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 10px; }
.pred-value  { font-size: 1.1rem; font-weight: 700; color: #1e293b; }

/* ── Section heading ────────────────────────────────────────── */
.section-title {
    font-size: 1.15rem; font-weight: 700; color: #1e293b;
    margin: 8px 0 18px 0; border-left: 4px solid #6366f1;
    padding-left: 12px;
}

/* ── Sidebar ────────────────────────────────────────────────── */
[data-testid="stSidebar"] { background: #ffffff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
STYLE_COLS    = ["style_tags_boho", "style_tags_industrial",
                 "style_tags_minimalist", "style_tags_scandinavian"]
MATERIAL_COLS = ["material_type_fabric", "material_type_glass",
                 "material_type_leather", "material_type_marble",
                 "material_type_metal",  "material_type_stone",
                 "material_type_tile",   "material_type_wood"]
STYLE_NAMES    = [c.replace("style_tags_", "").capitalize()    for c in STYLE_COLS]
MATERIAL_NAMES = [c.replace("material_type_", "").capitalize() for c in MATERIAL_COLS]

COLOR_PALETTE_TABLE = {
    ("Boho",         "Fabric"): ["#C19A6B", "#D2691E", "#8B4513"],
    ("Minimalist",   "Wood"):   ["#F5F5DC", "#D2B48C", "#FFFFFF"],
    ("Industrial",   "Metal"):  ["#708090", "#2F4F4F", "#36454F"],
    ("Scandinavian", "Wood"):   ["#DEB887", "#F5DEB3", "#FFFAF0"],
}
STYLE_DEFAULTS = {
    "Boho":         ["#C19A6B", "#D2691E", "#8B4513"],
    "Industrial":   ["#708090", "#2F4F4F", "#36454F"],
    "Minimalist":   ["#F5F5F5", "#E8E8E8", "#D3D3D3"],
    "Scandinavian": ["#DEB887", "#F5DEB3", "#FFFAF0"],
}

# Unsplash Source (free, no key required)
STYLE_UNSPLASH = {
    "Boho":         "bohemian+interior",
    "Industrial":   "industrial+loft+interior",
    "Minimalist":   "minimalist+interior+design",
    "Scandinavian": "scandinavian+interior+design",
}
MATERIAL_UNSPLASH = {
    "Wood":    "wood+furniture",
    "Metal":   "metal+industrial",
    "Fabric":  "fabric+textile+interior",
    "Marble":  "marble+interior",
    "Glass":   "glass+interior",
    "Leather": "leather+furniture",
    "Stone":   "stone+interior",
    "Tile":    "tile+floor+interior",
}

import base64
import glob
import random

def image_url(style: str, material: str, seed: int = 1) -> str:
    """Lấy ảnh thực tế từ bộ dữ liệu Pinterest cục bộ và encode sang Base64 để hiển thị."""
    style_lower = style.lower().strip()
    
    # Trỏ thẳng vào thư mục dataset Pinterest chứa ảnh thực
    base_dir = os.path.join(
        "Interior Design Material & Style Dataset", 
        "archive (11)", 
        "Pinterest Interior Design Images and Metadata"
    )
    
    # Tìm tất cả file ảnh thuộc style này (bất kể phòng nào)
    search_pattern = os.path.join(base_dir, "*", style_lower, "*.jpg")
    img_files = glob.glob(search_pattern)
    
    if not img_files:
        return "" # Fallback nếu không có ảnh
        
    # Cố định random seed bằng Image_id để 1 kết quả luôn ra 1 ảnh giống nhau
    random.seed(seed)
    chosen_img = random.choice(img_files)
    
    try:
        with open(chosen_img, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_palette(style: str, material: str):
    return COLOR_PALETTE_TABLE.get(
        (style, material),
        STYLE_DEFAULTS.get(style, ["#808080", "#A9A9A9", "#DCDCDC"])
    )

def swatch_html(colors):
    swatches = "".join(
        f'<span class="swatch" style="background:{c}" title="{c}"></span>'
        for c in colors
    )
    return f'<div class="swatch-row">{swatches}</div>'


# ─────────────────────────────────────────────────────────────────
# MODEL / DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────────
MODEL_DIR = "models"
# DATA_PATH nằm cùng cấp thư mục với file code này nên chỉ cần gọi tên file
DATA_PATH = "dataset_cleaned_final.csv"
@st.cache_resource(show_spinner="⚙️ Đang tải models...")
def load_models():
    # Sử dụng os.path.join với MODEL_DIR cho tất cả các model để đồng bộ
    style_clf    = joblib.load(os.path.join(MODEL_DIR, "style_model.pkl"))
    material_clf = joblib.load(os.path.join(MODEL_DIR, "material_model.pkl"))
    tfidf_s      = joblib.load(os.path.join(MODEL_DIR, "tfidf_style.pkl"))
    tfidf_m      = joblib.load(os.path.join(MODEL_DIR, "tfidf_material.pkl"))
    color_reg    = joblib.load(os.path.join(MODEL_DIR, "color_model.pkl"))
    return style_clf, material_clf, tfidf_s, tfidf_m, color_reg

@st.cache_resource(show_spinner="📦 Đang tải dữ liệu & embeddings…")
def load_data_and_embeddings():
    from sentence_transformers import SentenceTransformer
    df = pd.read_csv(DATA_PATH)
    df["style_label"]    = (df[STYLE_COLS].fillna(0).idxmax(axis=1)
                            .str.replace("style_tags_", "").str.capitalize())
    df["material_label"] = (df[MATERIAL_COLS].fillna(0).idxmax(axis=1)
                            .str.replace("material_type_", "").str.capitalize())

    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df["clean_design_description"].fillna("").tolist()
    embeddings = st_model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=64)
    return df, st_model, embeddings

# ─────────────────────────────────────────────────────────────────
# RECOMMEND FUNCTION
# ─────────────────────────────────────────────────────────────────
def recommend(query, top_n, style_filter, material_filter,
              df, st_model, embeddings, _cache={}):
    key = query
    if key not in _cache:
        _cache[key] = st_model.encode([query], normalize_embeddings=True)
    q_emb = _cache[key]

    mask = pd.Series([True] * len(df), index=df.index)
    if style_filter and style_filter != "Tất cả":
        mask &= df["style_label"].str.lower() == style_filter.lower()
    if material_filter and material_filter != "Tất cả":
        mask &= df["material_label"].str.lower() == material_filter.lower()

    idx = df.index[mask].tolist()
    if len(idx) < top_n:
        idx = df.index.tolist()

    sims = cosine_similarity(q_emb, embeddings[idx])[0]
    sub  = df.loc[idx].copy()
    sub["similarity"] = sims
    return sub.nlargest(top_n, "similarity").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Cài đặt")
    st.markdown("---")

    top_n = st.slider("Số kết quả trả về", min_value=1, max_value=10, value=4, step=1)

    st.markdown("### 🔍 Lọc nâng cao")
    style_opts    = ["Tất cả"] + STYLE_NAMES
    material_opts = ["Tất cả"] + MATERIAL_NAMES
    style_filter    = st.selectbox("Phong cách",  style_opts,    index=0)
    material_filter = st.selectbox("Vật liệu",    material_opts, index=0)

    st.markdown("---")
    st.markdown("""
**🎓 Nhóm 13 – ML for Design**  
ĐHCN – ĐHQGHN

**Pipeline:**  
`TF-IDF → RandomForest` (Style + Material)  
`SentenceTransformer → Cosine Similarity`
    """)

# ─────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏠 Design Inspiration Recommender</h1>
  <p>Nhập mô tả không gian bạn mong muốn — hệ thống sẽ phân loại phong cách, vật liệu và gợi ý các thiết kế tương tự.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────
try:
    style_clf, material_clf, tfidf_s, tfidf_m, color_reg = load_models()
    df, st_model, embeddings = load_data_and_embeddings()
except Exception as e:
    st.error(f"❌ Không thể tải models: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────
# SEARCH INPUT
# ─────────────────────────────────────────────────────────────────
col_in, col_btn = st.columns([5, 1])
with col_in:
    query = st.text_input(
        label="Nhập yêu cầu thiết kế",
        placeholder="Ví dụ: Phòng khách phong cách tối giản với gỗ sáng màu và ánh sáng tự nhiên…",
        label_visibility="collapsed",
    )
with col_btn:
    search_btn = st.button("🔍 Tìm kiếm", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────
# EXAMPLE QUERIES
# ─────────────────────────────────────────────────────────────────
EXAMPLES = [
    "Phòng ngủ Boho với nhiều cây xanh và vải thủ công",
    "Không gian công nghiệp với kim loại và bê tông",
    "Phòng khách Scandinavian ấm cúng, gỗ sáng",
    "Bếp tối giản, gọn gàng, màu trung tính",
]
st.markdown("**💡 Gợi ý nhanh:**")
ec = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    with ec[i]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            query = ex
            search_btn = True

# ─────────────────────────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────────────────────────
if search_btn and query.strip():
    cleaned = clean_text(query)

    # ── ML Predictions ────────────────────────────────────────────
    X_s = tfidf_s.transform([cleaned])
    X_m = tfidf_m.transform([cleaned])

    style_pred    = style_clf.predict(X_s)[0]
    material_pred = material_clf.predict(X_m)[0]

    style_proba    = style_clf.predict_proba(X_s)[0]
    material_proba = material_clf.predict_proba(X_m)[0]

    palette = get_palette(style_pred, material_pred)

    # ── Prediction Summary ─────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Kết quả phân loại</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        style_conf = style_proba.max() * 100
        st.markdown(f"""
        <div class="pred-box">
            <h4>🎨 Phong cách (Style)</h4>
            <div class="pred-value">{style_pred}</div>
            <div style="margin-top:6px; font-size:0.8rem; color:#64748b;">
                Độ tin cậy: <b>{style_conf:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        mat_conf = material_proba.max() * 100
        st.markdown(f"""
        <div class="pred-box">
            <h4>🪵 Vật liệu (Material)</h4>
            <div class="pred-value">{material_pred}</div>
            <div style="margin-top:6px; font-size:0.8rem; color:#64748b;">
                Độ tin cậy: <b>{mat_conf:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        sw = "".join(
            f'<span class="swatch" style="background:{c};width:36px;height:36px;border-radius:8px;" title="{c}"></span>'
            for c in palette
        )
        hex_labels = " &nbsp; ".join(
            f'<span style="font-size:0.72rem;color:#64748b;">{c}</span>'
            for c in palette
        )
        st.markdown(f"""
        <div class="pred-box">
            <h4>🎨 Bảng màu gợi ý</h4>
            <div class="swatch-row" style="gap:8px;">{sw}</div>
            <div style="margin-top:8px;">{hex_labels}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Confidence bars ────────────────────────────────────────────
    with st.expander("📈 Xem phân phối xác suất mô hình"):
        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            st.markdown("**Style probabilities**")
            classes_s = style_clf.classes_
            for label, prob in sorted(zip(classes_s, style_proba),
                                      key=lambda x: -x[1]):
                st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
        with exp_c2:
            st.markdown("**Material probabilities**")
            classes_m = material_clf.classes_
            for label, prob in sorted(zip(classes_m, material_proba),
                                      key=lambda x: -x[1]):
                st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")

    # ── Recommendations ────────────────────────────────────────────
    st.markdown('<div class="section-title">🏡 Thiết kế gợi ý</div>',
                unsafe_allow_html=True)

    with st.spinner("Đang tìm kiếm thiết kế phù hợp…"):
        results = recommend(
            query, top_n,
            style_filter, material_filter,
            df, st_model, embeddings
        )

    # Show as grid: 2 per row
    n_cols = 2
    rows   = [results.iloc[i:i+n_cols] for i in range(0, len(results), n_cols)]

    for row_df in rows:
        cols = st.columns(n_cols)
        for col, (_, row) in zip(cols, row_df.iterrows()):
            sim_pct = row["similarity"] * 100
            
            # Lấy ảnh base64
            img_url = image_url(row["style_label"], row["material_label"], seed=int(row["Image_id"]))
            
            # THÊM DÒNG NÀY: Xử lý luôn URL thực tế (base64 hoặc link dự phòng từ picsum)
            actual_img_url = img_url if img_url else f'https://picsum.photos/seed/{int(row["Image_id"])}/800/500'
            
            colors  = [row["color1"], row["color2"], row["color3"]]
            # validate hex
            colors  = [c if re.match(r'^#[0-9A-Fa-f]{6}$', str(c).strip()) else "#cccccc" for c in colors]

            with col:
                st.markdown(f"""
                <div class="result-card">
                    <a href="{actual_img_url}" target="_blank" title="Click để xem ảnh kích thước gốc">
                        <img src="{actual_img_url}" style="width:100%;height:200px;object-fit:cover;cursor:pointer;transition:0.2s;" onmouseover="this.style.opacity=0.85" onmouseout="this.style.opacity=1">
                    </a>
                    <div class="card-body">
                        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;">
                            <span class="badge badge-style">🎨 {row["style_label"]}</span>
                            <span class="badge badge-material">🪵 {row["material_label"]}</span>
                            <span class="badge badge-sim">⚡ {sim_pct:.1f}%</span>
                        </div>
                        <div class="card-desc">{row["clean_design_description"]}</div>
                        <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:4px;">Bảng màu thiết kế:</div>
                        {swatch_html(colors)}
                        <div style="font-size:0.72rem;color:#94a3b8;margin-top:6px;">
                            ID #{int(row["Image_id"])} &nbsp;·&nbsp; Similarity {sim_pct:.1f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

elif search_btn and not query.strip():
    st.warning("⚠️ Vui lòng nhập mô tả thiết kế trước khi tìm kiếm.")

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding: 60px 0; color:#94a3b8;">
        <div style="font-size: 3rem; margin-bottom: 12px;">🏠</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: #64748b;">
            Nhập mô tả thiết kế hoặc chọn gợi ý nhanh để bắt đầu
        </div>
        <div style="font-size: 0.9rem; margin-top: 8px;">
            Hệ thống sẽ phân loại <b>phong cách</b>, <b>vật liệu</b> và tìm kiếm <b>thiết kế tương tự</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
