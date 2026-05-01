# ================================================================
# MindCare — Streamlit Dashboard (Production)
# CC26-PSU148 | Coding Camp 2026 powered by DBS Foundation
# Deploy: https://streamlit.io/cloud
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Konfigurasi Halaman ─────────────────────────────────────────
st.set_page_config(
    page_title="MindCare Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Konstanta Warna ─────────────────────────────────────────────
C_MAIN    = "#2196F3"
C_COMPARE = "#FF9800"
C_POS     = "#4CAF50"
C_NEG     = "#F44336"
PALETTE   = [C_MAIN, C_COMPARE, C_POS, C_NEG, "#9C27B0"]
sns.set_theme(style="whitegrid", font_scale=1.0)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1976D2, #42A5F5);
        padding: 16px 20px; border-radius: 12px;
        color: white; text-align: center; margin-bottom: 8px;
    }
    .metric-card h3 { margin: 0; font-size: 28px; }
    .metric-card p  { margin: 0; font-size: 13px; opacity: 0.9; }
    .insight-box {
        background: #E3F2FD; border-left: 4px solid #2196F3;
        padding: 12px 16px; border-radius: 4px; margin: 8px 0;
    }
    .warning-box {
        background: #FFF3E0; border-left: 4px solid #FF9800;
        padding: 12px 16px; border-radius: 4px; margin: 8px 0;
    }
    .success-box {
        background: #E8F5E9; border-left: 4px solid #4CAF50;
        padding: 12px 16px; border-radius: 4px; margin: 8px 0;
    }
    [data-testid="stSidebar"] { background-color: #F0F4FF; }
</style>
""", unsafe_allow_html=True)

# ── Load Model Bundle ───────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        bundle = joblib.load("model_bundle.pkl")
        return bundle
    except Exception as e:
        return None

@st.cache_data
def load_data():
    try:
        return pd.read_csv("data_cleaned.csv")
    except:
        return None

bundle = load_model()
df_default = load_data()

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 MindCare")
    st.markdown("*Mental Health Analytics*")
    st.markdown("---")
    st.markdown("### 🧭 Navigasi")
    page = st.radio("", [
        "📊 Overview & Data",
        "🔍 Eksplorasi Data",
        "📈 Visualisasi Stres",
        "🤖 Prediksi Aktivitas",
        "📋 Insight & Rekomendasi"
    ], label_visibility="collapsed")
    st.markdown("---")
    if bundle:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model tidak ditemukan")
    st.markdown("**v2.0** | CC26-PSU148")
    st.markdown("DBS Coding Camp 2026")

# ================================================================
# HALAMAN 1: OVERVIEW & DATA
# ================================================================
if page == "📊 Overview & Data":
    st.title("🧠 MindCare — Mental Health Analytics Dashboard")
    st.markdown("**Deteksi tingkat stres dan rekomendasi aktivitas berbasis AI**")
    st.markdown("---")

    # Upload data atau pakai default
    uploaded = st.file_uploader(
        "📂 Upload Dataset (opsional — gunakan data_cleaned.csv jika tidak upload)",
        type=["csv"]
    )

    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state["df"] = df
        st.success(f"✅ Dataset berhasil diupload: **{df.shape[0]:,} baris × {df.shape[1]} kolom**")
    elif "df" in st.session_state:
        df = st.session_state["df"]
    elif df_default is not None:
        df = df_default
        st.session_state["df"] = df
        st.info(f"ℹ️ Menggunakan **data_cleaned.csv** bawaan: {df.shape[0]:,} baris × {df.shape[1]} kolom")
    else:
        st.error("❌ Tidak ada data. Upload dataset atau pastikan data_cleaned.csv ada di folder yang sama.")
        st.stop()

    # Metrics
    st.markdown("### 📊 Ringkasan Dataset")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <p>Total Responden</p><h3>{df.shape[0]:,}</h3></div>""", unsafe_allow_html=True)
    with col2:
        if "stress_level_1_5" in df.columns:
            avg = df["stress_level_1_5"].mean()
            st.markdown(f"""<div class="metric-card">
                <p>Rata-rata Stress Level</p><h3>{avg:.2f}/5</h3></div>""", unsafe_allow_html=True)
    with col3:
        if "aktivitas_dipilih" in df.columns:
            top = df["aktivitas_dipilih"].mode()[0]
            st.markdown(f"""<div class="metric-card">
                <p>Aktivitas Terpopuler</p><h3>{top.title()}</h3></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <p>Jumlah Fitur</p><h3>{df.shape[1]}</h3></div>""", unsafe_allow_html=True)

    st.markdown("### 🔎 Preview Data (10 Baris Pertama)")
    st.dataframe(df.head(10), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📋 Info Kolom")
        col_info = pd.DataFrame({
            "Kolom": df.columns,
            "Tipe": df.dtypes.values.astype(str),
            "Non-Null": df.count().values,
            "Missing": df.isnull().sum().values,
            "Unik": df.nunique().values
        })
        st.dataframe(col_info, use_container_width=True)
    with col_b:
        st.markdown("### 📈 Statistik Deskriptif")
        st.dataframe(df.describe().round(2), use_container_width=True)

# ================================================================
# HALAMAN 2: EKSPLORASI DATA
# ================================================================
elif page == "🔍 Eksplorasi Data":
    st.title("🔍 Eksplorasi Data")

    if "df" not in st.session_state:
        st.warning("⚠️ Kembali ke halaman Overview dan upload/load dataset terlebih dahulu.")
        st.stop()
    df = st.session_state["df"]

    tab1, tab2, tab3 = st.tabs(["📊 Distribusi Kolom", "🔗 Korelasi", "📉 Perbandingan Grup"])

    with tab1:
        col1, col2 = st.columns(2)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        with col1:
            sel_num = st.selectbox("Pilih Kolom Numerik", num_cols, key="exp_num")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df[sel_num].dropna(), bins=30, color=C_MAIN, edgecolor="white", alpha=0.85)
            ax.axvline(df[sel_num].mean(), color=C_NEG, linestyle="--",
                       label=f"Mean = {df[sel_num].mean():.2f}")
            ax.axvline(df[sel_num].median(), color=C_COMPARE, linestyle="--",
                       label=f"Median = {df[sel_num].median():.2f}")
            ax.set_title(f"Distribusi: {sel_num}", fontweight="bold")
            ax.set_xlabel(sel_num); ax.set_ylabel("Frekuensi"); ax.legend()
            st.pyplot(fig); plt.close()

        with col2:
            if cat_cols:
                sel_cat = st.selectbox("Pilih Kolom Kategorik", cat_cols, key="exp_cat")
                vc = df[sel_cat].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(vc.index, vc.values,
                        color=PALETTE[:len(vc)], edgecolor="white", alpha=0.85)
                ax.set_title(f"Frekuensi: {sel_cat}", fontweight="bold")
                ax.set_xlabel("Jumlah")
                for i, v in enumerate(vc.values):
                    ax.text(v + 10, i, f"{v:,}", va="center", fontsize=9)
                st.pyplot(fig); plt.close()

    with tab2:
        st.markdown("#### Heatmap Korelasi Antar Fitur Numerik")
        default_sel = [c for c in ["stress_level_1_5","anxiety_score","depression_score",
                                    "self_esteem_score","kualitas_tidur_1_5","durasi_tidur_jam",
                                    "aktivitas_fisik_mnt","waktu_luang_mnt"] if c in num_cols]
        sel_corr = st.multiselect("Pilih kolom:", num_cols, default=default_sel)
        if len(sel_corr) >= 2:
            corr_mat = df[sel_corr].corr()
            fig, ax = plt.subplots(figsize=(10, 7))
            mask = np.zeros_like(corr_mat, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f",
                        cmap="RdBu_r", center=0, ax=ax,
                        linewidths=0.5, annot_kws={"size": 9})
            ax.set_title("Heatmap Korelasi", fontweight="bold")
            st.pyplot(fig); plt.close()
        else:
            st.info("Pilih minimal 2 kolom untuk heatmap korelasi.")

    with tab3:
        if "stress_level_1_5" in df.columns and "aktivitas_dipilih" in df.columns:
            group_col = st.selectbox("Kelompokkan berdasarkan:",
                                     ["stress_level_1_5","aktivitas_dipilih","pekerjaan","jenis_kelamin"],
                                     key="grp_col")
            metric_col = st.selectbox("Tampilkan metrik:",
                                      [c for c in num_cols if c != group_col], key="grp_met")
            grp = df.groupby(group_col)[metric_col].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(grp.index.astype(str), grp.values,
                   color=PALETTE[:len(grp)], edgecolor="white", alpha=0.85)
            for xi, yi in enumerate(grp.values):
                ax.text(xi, yi + grp.values.max()*0.01,
                        f"{yi:.2f}", ha="center", fontsize=9, fontweight="bold")
            ax.set_title(f"Rata-rata {metric_col} per {group_col}", fontweight="bold")
            ax.set_xlabel(group_col); ax.set_ylabel(f"Rata-rata {metric_col}")
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig); plt.close()

# ================================================================
# HALAMAN 3: VISUALISASI STRES
# ================================================================
elif page == "📈 Visualisasi Stres":
    st.title("📈 Visualisasi Tingkat Stres")

    if "df" not in st.session_state:
        st.warning("⚠️ Kembali ke halaman Overview terlebih dahulu.")
        st.stop()
    df = st.session_state["df"]

    if "stress_level_1_5" not in df.columns:
        st.error("Kolom 'stress_level_1_5' tidak ditemukan di dataset.")
        st.stop()

    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distribusi Stress Level")
        vc_stress = df["stress_level_1_5"].value_counts().sort_index()
        color_map = {1:C_POS, 2:"#8BC34A", 3:C_COMPARE, 4:"#FF5722", 5:C_NEG}
        colors_bar = [color_map.get(int(k), C_MAIN) for k in vc_stress.index]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(vc_stress.index.astype(str), vc_stress.values,
                      color=colors_bar, edgecolor="white", alpha=0.9)
        for bar, val in zip(bars, vc_stress.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f"{val:,}\n({val/len(df)*100:.1f}%)",
                    ha="center", fontsize=9, fontweight="bold")
        ax.set_title("Distribusi Stress Level (1=Rendah → 5=Tinggi)", fontweight="bold")
        ax.set_xlabel("Stress Level"); ax.set_ylabel("Jumlah Responden")
        ax.set_ylim(0, vc_stress.max() * 1.2)
        st.pyplot(fig); plt.close()

    with col2:
        if "aktivitas_dipilih" in df.columns:
            st.markdown("#### Proporsi Aktivitas Dipilih")
            vc_act = df["aktivitas_dipilih"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            wedges, texts, autotexts = ax.pie(
                vc_act.values, labels=vc_act.index,
                autopct="%1.1f%%", colors=PALETTE[:len(vc_act)],
                startangle=90, pctdistance=0.75,
                wedgeprops=dict(edgecolor="white", linewidth=2))
            for at in autotexts:
                at.set_fontsize(11); at.set_fontweight("bold")
            ax.set_title("Proporsi Aktivitas yang Dipilih", fontweight="bold")
            st.pyplot(fig); plt.close()

    # Row 2: Stacked bar
    if "aktivitas_dipilih" in df.columns:
        st.markdown("#### Aktivitas vs Stress Level")
        cross = pd.crosstab(df["stress_level_1_5"], df["aktivitas_dipilih"], normalize="index") * 100
        act_colors = {}
        acts = df["aktivitas_dipilih"].unique()
        for act, clr in zip(acts, PALETTE):
            act_colors[act] = clr
        fig, ax = plt.subplots(figsize=(12, 5))
        cross.plot(kind="bar", stacked=True, ax=ax, color=act_colors,
                   edgecolor="white", alpha=0.85)
        ax.set_title("Proporsi Aktivitas Dipilih per Stress Level (%)", fontweight="bold")
        ax.set_xlabel("Stress Level (1=Rendah → 5=Tinggi)")
        ax.set_ylabel("Proporsi (%)")
        ax.set_xticklabels(cross.index.astype(str), rotation=0)
        ax.legend(title="Aktivitas", bbox_to_anchor=(1.01, 1))
        st.pyplot(fig); plt.close()

        st.markdown("""<div class="insight-box">
        📌 <b>Insight:</b> Journaling cenderung mendominasi pada stress level tinggi (4–5),
        sedangkan membaca lebih banyak dipilih pada stress level rendah (1–2). Olahraga terdistribusi
        lebih merata. Model KNN MindCare belajar dari pola ini untuk menghasilkan rekomendasi tepat sasaran.
        </div>""", unsafe_allow_html=True)

    # Row 3: Psikologis per stress level
    st.markdown("#### Rata-rata Skor Psikologis per Stress Level")
    psych_cols = [c for c in ["anxiety_score","depression_score","self_esteem_score"]
                  if c in df.columns]
    if psych_cols:
        fig, axes = plt.subplots(1, len(psych_cols), figsize=(6*len(psych_cols), 4))
        if len(psych_cols) == 1: axes = [axes]
        for ax, col, color in zip(axes, psych_cols, [C_MAIN, C_NEG, C_POS]):
            grp = df.groupby("stress_level_1_5")[col].mean()
            ax.bar(grp.index.astype(str), grp.values, color=color,
                   edgecolor="white", alpha=0.85)
            ax.plot(range(len(grp)), grp.values, "o-", color="black",
                    linewidth=1.5, markersize=6)
            for xi, yi in enumerate(grp.values):
                ax.text(xi, yi + grp.max()*0.02, f"{yi:.1f}",
                        ha="center", fontsize=9, fontweight="bold")
            ax.set_title(f"Rata-rata {col}", fontweight="bold")
            ax.set_xlabel("Stress Level"); ax.set_ylabel(col)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ================================================================
# HALAMAN 4: PREDIKSI AKTIVITAS
# ================================================================
elif page == "🤖 Prediksi Aktivitas":
    st.title("🤖 Prediksi Rekomendasi Aktivitas")
    st.markdown("Isi kuesioner di bawah untuk mendapatkan rekomendasi aktivitas yang dipersonalisasi.")

    if bundle is None:
        st.error("❌ Model tidak berhasil dimuat. Pastikan file `model_bundle.pkl` ada di folder yang sama.")
        st.stop()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧠 Kondisi Psikologis**")
        stress_level     = st.slider("Stress Level (1=Rendah, 5=Tinggi)", 1, 5, 3)
        anxiety_score    = st.slider("Anxiety Score (0–21)", 0, 21, 10,
                                      help="0=Tidak cemas, 21=Sangat cemas")
        depression_score = st.slider("Depression Score (0–27)", 0, 27, 12,
                                      help="0=Tidak depresi, 27=Sangat depresi")
        self_esteem      = st.slider("Self-Esteem Score (0–30)", 0, 30, 18,
                                      help="0=Sangat rendah, 30=Sangat tinggi")
        study_load       = st.slider("Beban Studi/Kerja (1–5)", 1, 5, 3)
        peer_pressure    = st.slider("Tekanan Sosial (1–5)", 1, 5, 2)
        social_support   = st.slider("Dukungan Sosial (1–5)", 1, 5, 3)
        future_concern   = st.slider("Kekhawatiran Masa Depan (1–5)", 1, 5, 3)

    with col2:
        st.markdown("**😴 Gaya Hidup & Tidur**")
        kualitas_tidur   = st.slider("Kualitas Tidur (1–5)", 1, 5, 3,
                                      help="1=Sangat buruk, 5=Sangat baik")
        durasi_tidur     = st.slider("Durasi Tidur (jam/malam)", 3.0, 12.0, 7.0, 0.5)
        aktivitas_fisik  = st.slider("Aktivitas Fisik (menit/hari)", 0, 300, 60)
        waktu_luang      = st.slider("Waktu Luang (menit/hari)", 0, 480, 120)

    with col3:
        st.markdown("**🎯 Preferensi & Profil**")
        pref_olahraga = st.selectbox("Suka Olahraga?", ["Ya", "Tidak"])
        pref_baca     = st.selectbox("Suka Membaca?",  ["Ya", "Tidak"])
        pref_jurnal   = st.selectbox("Suka Journaling?", ["Ya", "Tidak"])
        komitmen      = st.slider("Komitmen (hari/minggu)", 1, 7, 3)
        umur          = st.number_input("Usia", 10, 80, 22)
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Female", "Male"])
        penyebab      = st.selectbox("Penyebab Stres Utama",
                                     ["Akademik","Pekerjaan","Sosial","Keuangan","Lainnya"])
        tujuan        = st.selectbox("Tujuan Utama",
                                     ["Mengurangi stres","Tidur lebih baik",
                                      "Meningkatkan energi","Lebih produktif"])

    st.markdown("---")

    if st.button("🎯 Dapatkan Rekomendasi", type="primary", use_container_width=True):
        # Compute engineered features
        p_ol = 1 if pref_olahraga == "Ya" else 0
        p_bc = 1 if pref_baca == "Ya" else 0
        p_jn = 1 if pref_jurnal == "Ya" else 0
        jk_enc = 1 if jenis_kelamin == "Male" else 0
        penyebab_enc = {"Akademik":0,"Keuangan":1,"Lainnya":2,"Pekerjaan":3,"Sosial":4}.get(penyebab, 0)
        tujuan_enc   = {"Lebih produktif":0,"Meningkatkan energi":1,
                        "Mengurangi stres":2,"Tidur lebih baik":3}.get(tujuan, 2)

        psikologis_score = ((anxiety_score/21)+(depression_score/27)+(1-self_esteem/30))/3*100
        gaya_hidup_score = ((kualitas_tidur/5)+(min(aktivitas_fisik,120)/120))/2*100
        activity_score   = ((p_ol+p_bc+p_jn)/3+(komitmen/7))/2*100
        sleep_ratio      = min(durasi_tidur/8.0, 1.0)
        stress_anx       = stress_level * anxiety_score
        wellbeing        = gaya_hidup_score - psikologis_score
        sp_ratio         = social_support / (peer_pressure + 1)

        feat_vector = [
            umur, jk_enc, 5, stress_level, penyebab_enc,
            2, kualitas_tidur, durasi_tidur, waktu_luang,
            aktivitas_fisik, p_ol, p_bc, p_jn, komitmen, tujuan_enc,
            anxiety_score, depression_score, self_esteem,
            study_load, peer_pressure, social_support, future_concern,
            psikologis_score, gaya_hidup_score, activity_score,
            sleep_ratio, stress_anx, wellbeing, sp_ratio
        ]

        try:
            model  = bundle['model']
            scaler = bundle['scaler']
            le_tgt = bundle['le_target']

            X_input = scaler.transform([feat_vector])
            pred    = model.predict(X_input)[0]
            proba   = model.predict_proba(X_input)[0]
            rekomendasi = le_tgt.inverse_transform([pred])[0]
            confidence  = proba[pred] * 100
        except Exception as e:
            st.error(f"Prediksi gagal: {e}")
            st.stop()

        # Tampilkan hasil
        st.markdown("---")
        emoji_map = {"journaling":"📓", "membaca":"📚", "olahraga":"🏃"}
        color_map  = {"journaling":C_COMPARE, "membaca":C_MAIN, "olahraga":C_POS}

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color_map.get(rekomendasi,C_MAIN)},{color_map.get(rekomendasi,C_MAIN)}99);
             padding:20px 28px;border-radius:16px;color:white;text-align:center;margin-bottom:16px;">
            <p style="margin:0;font-size:14px;opacity:0.9;">Rekomendasi Aktivitas untuk Anda</p>
            <h2 style="margin:8px 0;">{emoji_map.get(rekomendasi,'🎯')} {rekomendasi.upper()}</h2>
            <p style="margin:0;font-size:13px;opacity:0.85;">Keyakinan Model: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Durasi rekomendasi
        if waktu_luang < 60:   durasi_rec = 15
        elif waktu_luang < 120: durasi_rec = 30
        else:                   durasi_rec = 45

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("⏱️ Durasi Sesi",      f"{durasi_rec} menit")
        col_r2.metric("🧠 Psikologis Score", f"{psikologis_score:.1f}/100",
                       delta="Tinggi → perlu perhatian" if psikologis_score > 60 else None,
                       delta_color="inverse")
        col_r3.metric("💪 Gaya Hidup Score", f"{gaya_hidup_score:.1f}/100")
        col_r4.metric("⚖️ Wellbeing Index",  f"{wellbeing:+.1f}")

        # Probabilitas tiap kelas
        st.markdown("#### 📊 Distribusi Probabilitas Prediksi")
        classes = le_tgt.classes_
        fig, ax = plt.subplots(figsize=(8, 3))
        bar_colors = [color_map.get(c, C_MAIN) for c in classes]
        bars = ax.barh(classes, proba*100, color=bar_colors, edgecolor="white", alpha=0.85)
        for bar, val in zip(bars, proba*100):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontweight="bold")
        ax.set_xlabel("Probabilitas (%)")
        ax.set_title("Keyakinan Model per Kelas Aktivitas", fontweight="bold")
        ax.set_xlim(0, 110)
        st.pyplot(fig); plt.close()

        # Tips pelaksanaan
        tips = {
            "journaling": [
                "🖊️ Tulis 3 hal yang Anda syukuri hari ini",
                "💭 Ekspresikan emosi tanpa sensor — tidak perlu sempurna",
                "🎯 Tetapkan 1 intention untuk esok hari",
                "📅 Lakukan rutin di waktu yang sama setiap hari",
            ],
            "membaca": [
                "📖 Pilih buku ringan sesuai mood — novel, self-help, atau non-fiksi",
                "☕ Ciptakan suasana nyaman: minuman hangat dan tempat tenang",
                "🔕 Matikan notifikasi HP selama sesi membaca",
                "⏱️ Mulai dari 15 menit, tingkatkan bertahap",
            ],
            "olahraga": [
                "🚶 Mulai dengan jalan kaki cepat 15–20 menit",
                "🧘 Lanjutkan dengan peregangan ringan 5–10 menit",
                "💧 Minum air putih yang cukup sebelum dan sesudah",
                "🎵 Buat playlist musik favorit untuk menemani olahraga",
            ]
        }
        st.markdown("#### 💡 Tips Pelaksanaan")
        for tip in tips.get(rekomendasi, []):
            st.markdown(f"- {tip}")

        # Warning jika stres berat
        if stress_level >= 4 and anxiety_score >= 15:
            st.markdown("""<div class="warning-box">
            ⚠️ <b>Perhatian:</b> Tingkat stres dan kecemasan Anda tergolong tinggi.
            Aktivitas ini hanya sebagai pertolongan pertama. Jika gejala berlanjut lebih dari
            2 minggu, pertimbangkan untuk berkonsultasi dengan profesional kesehatan mental.
            </div>""", unsafe_allow_html=True)

# ================================================================
# HALAMAN 5: INSIGHT & REKOMENDASI
# ================================================================
elif page == "📋 Insight & Rekomendasi":
    st.title("📋 Insight & Rekomendasi Bisnis")

    st.markdown("""<div class="insight-box">
    <b>📊 Ringkasan Temuan Utama dari Analisis Data MindCare</b><br>
    Berdasarkan EDA dan modeling terhadap 10.841 responden dari 3 sumber dataset.
    Model: Random Forest (Test Accuracy ~83%, F1 Macro ~82%).
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    insights = [
        ("🧠 Faktor Psikologis (BQ1)",
         "Anxiety adalah prediktor stres terkuat. Pengguna dengan anxiety ≥ 15 memiliki kemungkinan 3× lebih tinggi berada di stress level 4–5.",
         "Tambahkan modul pengukuran anxiety (GAD-7) sebagai skrining pertama di aplikasi."),
        ("😴 Pola Tidur (BQ2)",
         "Kualitas tidur berbanding terbalik dengan stress level. Pengguna stres tinggi rata-rata tidur dengan kualitas 2.1/5 vs 3.8/5 pada stres rendah.",
         "Implementasikan modul pengingat tidur dan tracker kualitas tidur di fitur daily check-in."),
        ("📋 Rekomendasi Aktivitas (BQ3)",
         "Journaling mendominasi pada stress level 4–5 (~40% proporsi). Membaca lebih populer pada level 1–2.",
         "Model KNN memanfaatkan pola ini. Pertimbangkan menambahkan guided journaling prompts."),
        ("🎯 Durasi Sesi (BQ5)",
         "Pengguna dengan waktu luang >90 menit menerima rekomendasi 15+ menit lebih panjang.",
         "Sistem harus mendeteksi ketersediaan waktu dan menyesuaikan durasi sesi secara otomatis."),
        ("🤖 Performa Model (A/B Test)",
         "Random Forest (Test F1 ~82%) lebih baik dari KNN (~72%). Feature engineering meningkatkan performa model.",
         "Deploy Random Forest untuk production. Retrain setiap 1.000 pengguna baru."),
    ]

    for title, insight_text, action_text in insights:
        with st.expander(f"**{title}**"):
            st.markdown(f"""<div class="insight-box">
            📌 <b>Insight:</b> {insight_text}</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="warning-box">
            🎯 <b>Action Item:</b> {action_text}</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Roadmap Deployment")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""**Phase 1 — MVP (Bln 1–2)**
- Deploy model ke FastAPI
- Integrasi kuesioner 27 pertanyaan
- Dashboard basic
        """)
    with col2:
        st.markdown("""**Phase 2 — Enhancement (Bln 3–4)**
- Daily check-in (5 pertanyaan)
- Face expression analysis
- Notifikasi berkala
        """)
    with col3:
        st.markdown("""**Phase 3 — Advanced (Bln 5–6)**
- Auto-retraining model
- Collaborative filtering komunitas
- Laporan progres mingguan
        """)

    st.markdown("---")
    st.markdown("""<div class="success-box">
    ✅ <b>Disclaimer:</b> MindCare hanya sebagai alat bantu deteksi dini dan self-assessment.
    Tidak menggantikan diagnosis atau konsultasi dengan profesional kesehatan mental (psikolog/psikiater).
    </div>""", unsafe_allow_html=True)
    st.caption("MindCare v2.0 | CC26-PSU148 | Coding Camp 2026 powered by DBS Foundation")
