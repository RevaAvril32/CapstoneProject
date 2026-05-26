import streamlit as st
import pandas as pd
import plotly.express as px
import re
from pathlib import Path

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

st.set_page_config(
    page_title="MindMate AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f7f9fc; }

.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.15);
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    opacity: 0.95;
}

.card {
    background-color: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}

.metric-card {
    background-color: white;
    padding: 22px;
    border-radius: 18px;
    border-left: 6px solid #667eea;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 12px;
}

.metric-title {
    font-size: 15px;
    color: #6b7280;
    margin-bottom: 5px;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: #111827;
}

.insight {
    background-color: #eef2ff;
    padding: 20px;
    border-radius: 16px;
    border-left: 6px solid #6366f1;
    margin-bottom: 15px;
}

.warning {
    background-color: #fff7ed;
    padding: 20px;
    border-radius: 16px;
    border-left: 6px solid #f97316;
    margin-bottom: 15px;
}

.success {
    background-color: #ecfdf5;
    padding: 20px;
    border-radius: 16px;
    border-left: 6px solid #10b981;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

def find_csv_file():
    current_dir = Path(__file__).parent

    preferred_file = current_dir / "clean_final_dataset.csv"
    if preferred_file.exists():
        return preferred_file

    csv_files = list(current_dir.glob("*.csv"))
    if len(csv_files) > 0:
        return csv_files[0]

    return None

def load_data():
    csv_path = find_csv_file()

    if csv_path is None:
        st.error("File CSV belum ditemukan di folder project.")
        st.info(
            "Letakkan file CSV kamu di folder yang sama dengan file dashboard_no_upload.py. "
            "Nama yang paling disarankan: clean_final_dataset.csv"
        )
        st.stop()

    st.sidebar.success(f"Dataset terbaca: {csv_path.name}")
    return pd.read_csv(csv_path)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def standardize_columns(df):
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {
        "emotion": "intent",
        "label": "intent",
        "labels": "response_text",
        "empathetic_dialogues": "input_text",
        "dialogue": "input_text",
        "text": "input_text",
        "question": "input_text",
        "answer": "response_text",
        "response": "response_text"
    }

    return df.rename(columns=rename_map)

def metric_card(title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def get_outlier_bounds(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound

st.sidebar.title("🧠 MindMate AI")
st.sidebar.caption("Dashboard Analisis Dataset Chatbot")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "🏠 Home",
        "📊 Overview Dataset",
        "📈 Visualisasi",
        "🧹 Data Quality",
        "🔍 Text Explorer",
        "💡 Insight Otomatis",
        "⬇️ Download Data"
    ]
)

try:
    df_raw = load_data()
except Exception as e:
    st.error("Dataset gagal dibaca. Pastikan file CSV tidak rusak dan formatnya benar.")
    st.exception(e)
    st.stop()

df_raw = standardize_columns(df_raw)

required_columns = ["input_text", "response_text", "intent"]
missing_columns = [col for col in required_columns if col not in df_raw.columns]

if missing_columns:
    st.error(f"Kolom wajib belum ditemukan: {missing_columns}")
    st.write("Kolom yang tersedia pada dataset kamu:")
    st.write(df_raw.columns.tolist())
    st.info(
        "Kolom yang dibutuhkan: input_text, response_text, intent. "
        "Kalau dataset mentah, bisa juga memakai: emotion, empathetic_dialogues, labels."
    )
    st.stop()

total_before_cleaning = len(df_raw)

df = df_raw.copy()
df = df.drop_duplicates()
df = df.dropna(subset=["input_text", "response_text", "intent"])

df["input_text"] = df["input_text"].astype(str)
df["response_text"] = df["response_text"].astype(str)
df["intent"] = df["intent"].astype(str)

df["clean_input_text"] = df["input_text"].apply(clean_text)
df["input_length"] = df["input_text"].apply(len)
df["response_length"] = df["response_text"].apply(len)
df["word_count"] = df["input_text"].apply(lambda x: len(str(x).split()))

total_after_cleaning = len(df)
removed_data = total_before_cleaning - total_after_cleaning

lower_bound, upper_bound = get_outlier_bounds(df, "input_length")
df["is_outlier"] = (df["input_length"] < lower_bound) | (df["input_length"] > upper_bound)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filter Data")

intent_options = sorted(df["intent"].unique())

selected_intent = st.sidebar.multiselect(
    "Pilih Intent/Emotion",
    options=intent_options,
    default=intent_options
)

min_length = int(df["input_length"].min())
max_length = int(df["input_length"].max())

selected_length = st.sidebar.slider(
    "Rentang Panjang Input",
    min_value=min_length,
    max_value=max_length,
    value=(min_length, max_length)
)

filtered_df = df[
    (df["intent"].isin(selected_intent)) &
    (df["input_length"] >= selected_length[0]) &
    (df["input_length"] <= selected_length[1])
]

if filtered_df.empty:
    st.warning("Data kosong setelah difilter. Coba ubah filter di sidebar.")
    st.stop()

if menu == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>🧠 MindMate AI Dashboard</h1>
        <p>Dashboard langsung membaca dataset CSV dari folder project tanpa perlu upload file.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        <div class="card">
            <h3>🎯 Tujuan Dashboard</h3>
            <p>
            Dashboard ini membantu melihat distribusi emotion atau intent, mengecek kualitas data,
            mendeteksi outlier, dan memberikan insight awal untuk pengembangan MindMate AI Chatbot.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3>📌 Fitur Utama</h3>
            <ul>
                <li>Dataset langsung terbaca dari folder project</li>
                <li>Ringkasan dataset otomatis</li>
                <li>Visualisasi distribusi emotion atau intent</li>
                <li>Analisis panjang input dan response text</li>
                <li>Deteksi missing value, duplicate, dan outlier</li>
                <li>Text explorer berdasarkan keyword</li>
                <li>Insight otomatis untuk laporan</li>
                <li>Download dataset hasil filter</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("📋 Preview Dataset")
        st.dataframe(filtered_df.head(10), use_container_width=True)

elif menu == "📊 Overview Dataset":
    st.title("📊 Overview Dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Data", f"{len(filtered_df):,}")

    with col2:
        metric_card("Jumlah Intent", filtered_df["intent"].nunique())

    with col3:
        metric_card("Rata-rata Input", round(filtered_df["input_length"].mean(), 2))

    with col4:
        metric_card("Rata-rata Response", round(filtered_df["response_length"].mean(), 2))

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        metric_card("Jumlah Outlier", int(filtered_df["is_outlier"].sum()))

    with col6:
        metric_card("Rata-rata Word Count", round(filtered_df["word_count"].mean(), 2))

    with col7:
        metric_card("Data Terhapus Cleaning", removed_data)

    with col8:
        metric_card("Jumlah Kolom", len(filtered_df.columns))

    st.subheader("👀 Preview Data")
    st.dataframe(
        filtered_df[[
            "input_text",
            "response_text",
            "intent",
            "input_length",
            "response_length",
            "word_count"
        ]],
        use_container_width=True
    )

    st.subheader("🧾 Informasi Tipe Data")
    dtype_df = pd.DataFrame({
        "Kolom": filtered_df.dtypes.index,
        "Tipe Data": filtered_df.dtypes.astype(str).values
    })
    st.dataframe(dtype_df, use_container_width=True)

elif menu == "📈 Visualisasi":
    st.title("📈 Visualisasi Dataset")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distribusi Intent",
        "Pie Chart",
        "Panjang Teks",
        "Rata-rata per Intent",
        "Word Cloud"
    ])

    intent_count = filtered_df["intent"].value_counts().reset_index()
    intent_count.columns = ["intent", "jumlah"]

    with tab1:
        st.subheader("📊 Distribusi Intent/Emotion")
        fig_bar = px.bar(
            intent_count,
            x="intent",
            y="jumlah",
            color="intent",
            title="Jumlah Data Berdasarkan Intent/Emotion"
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader("🧩 Persentase Intent/Emotion")
        fig_pie = px.pie(
            intent_count,
            names="intent",
            values="jumlah",
            hole=0.45,
            title="Persentase Data Berdasarkan Intent/Emotion"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        st.subheader("📝 Distribusi Panjang Teks")
        col1, col2 = st.columns(2)

        with col1:
            fig_hist_input = px.histogram(
                filtered_df,
                x="input_length",
                nbins=50,
                title="Distribusi Panjang Input Text"
            )
            st.plotly_chart(fig_hist_input, use_container_width=True)

        with col2:
            fig_hist_response = px.histogram(
                filtered_df,
                x="response_length",
                nbins=50,
                title="Distribusi Panjang Response Text"
            )
            st.plotly_chart(fig_hist_response, use_container_width=True)

    with tab4:
        st.subheader("📌 Rata-rata Panjang Teks per Intent")
        avg_length = (
            filtered_df
            .groupby("intent")[["input_length", "response_length", "word_count"]]
            .mean()
            .reset_index()
            .sort_values(by="input_length", ascending=False)
        )

        fig_avg = px.bar(
            avg_length,
            x="intent",
            y="input_length",
            color="intent",
            title="Rata-rata Panjang Input Text Berdasarkan Intent"
        )
        fig_avg.update_layout(showlegend=False)
        st.plotly_chart(fig_avg, use_container_width=True)
        st.dataframe(avg_length, use_container_width=True)

    with tab5:
        st.subheader("☁️ Word Cloud Input Text")

        if WORDCLOUD_AVAILABLE:
            text_data = " ".join(filtered_df["clean_input_text"].astype(str).tolist())

            if text_data.strip():
                wordcloud = WordCloud(
                    width=1000,
                    height=500,
                    background_color="white",
                    colormap="viridis"
                ).generate(text_data)

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.warning("Tidak ada teks untuk dibuat Word Cloud.")
        else:
            st.warning("Library wordcloud belum terinstall. Jalankan: pip install wordcloud matplotlib")

elif menu == "🧹 Data Quality":
    st.title("🧹 Data Quality Check")

    missing_df = filtered_df.isnull().sum().reset_index()
    missing_df.columns = ["Kolom", "Jumlah Missing Value"]

    duplicate_count = filtered_df.duplicated().sum()
    outlier_count = filtered_df["is_outlier"].sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Missing Value", int(missing_df["Jumlah Missing Value"].sum()))

    with col2:
        metric_card("Duplicate Data", int(duplicate_count))

    with col3:
        metric_card("Outlier Input Length", int(outlier_count))

    st.subheader("🔍 Missing Value per Kolom")
    st.dataframe(missing_df, use_container_width=True)

    st.subheader("📦 Boxplot Panjang Input")
    fig_box = px.box(
        filtered_df,
        y="input_length",
        points="outliers",
        title="Boxplot Panjang Input Text"
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown(
        f"""
        <div class="warning">
            <b>Batas Outlier Menggunakan IQR:</b><br>
            Lower Bound: <b>{round(lower_bound, 2)}</b><br>
            Upper Bound: <b>{round(upper_bound, 2)}</b><br><br>
            Data dianggap outlier jika panjang input berada di bawah lower bound atau di atas upper bound.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("⚠️ Data Outlier")
    outlier_df = filtered_df[filtered_df["is_outlier"] == True]

    if len(outlier_df) > 0:
        st.dataframe(
            outlier_df[[
                "input_text",
                "response_text",
                "intent",
                "input_length",
                "word_count"
            ]],
            use_container_width=True
        )
    else:
        st.success("Tidak ada outlier pada data yang sedang difilter.")

elif menu == "🔍 Text Explorer":
    st.title("🔍 Text Explorer")
    st.write("Cari kata tertentu dalam dataset, misalnya: anxiety, sad, stress, happy, lonely.")

    keyword = st.text_input("Masukkan keyword:")

    if keyword:
        search_df = filtered_df[
            filtered_df["input_text"].str.contains(keyword, case=False, na=False) |
            filtered_df["response_text"].str.contains(keyword, case=False, na=False)
        ]

        st.success(f"Ditemukan {len(search_df)} data yang mengandung keyword: '{keyword}'")

        st.dataframe(
            search_df[[
                "input_text",
                "response_text",
                "intent",
                "input_length",
                "word_count"
            ]],
            use_container_width=True
        )
    else:
        st.info("Masukkan keyword terlebih dahulu.")

    st.subheader("📋 Contoh Data Berdasarkan Intent")
    selected_single_intent = st.selectbox(
        "Pilih satu intent:",
        options=intent_options
    )

    sample_df = filtered_df[filtered_df["intent"] == selected_single_intent].head(20)

    st.dataframe(
        sample_df[[
            "input_text",
            "response_text",
            "intent"
        ]],
        use_container_width=True
    )

elif menu == "💡 Insight Otomatis":
    st.title("💡 Insight Otomatis")

    total_data = len(filtered_df)
    total_intent = filtered_df["intent"].nunique()

    most_intent = filtered_df["intent"].value_counts().idxmax()
    most_intent_count = filtered_df["intent"].value_counts().max()

    least_intent = filtered_df["intent"].value_counts().idxmin()
    least_intent_count = filtered_df["intent"].value_counts().min()

    avg_input = round(filtered_df["input_length"].mean(), 2)
    avg_response = round(filtered_df["response_length"].mean(), 2)
    outlier_count = int(filtered_df["is_outlier"].sum())

    st.markdown(
        f"""
        <div class="insight">
            <h3>📌 Insight 1: Gambaran Umum Dataset</h3>
            <p>
            Dataset memiliki <b>{total_data}</b> data dengan <b>{total_intent}</b> kategori intent/emotion.
            Hal ini menunjukkan bahwa dataset memiliki beberapa kategori emosi untuk mendukung pengembangan chatbot.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="insight">
            <h3>📊 Insight 2: Distribusi Intent/Emotion</h3>
            <p>
            Kategori terbanyak adalah <b>{most_intent}</b> sebanyak <b>{most_intent_count}</b> data.
            Kategori paling sedikit adalah <b>{least_intent}</b> sebanyak <b>{least_intent_count}</b> data.
            Perbedaan jumlah data antar kategori perlu diperhatikan agar model tidak bias.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="insight">
            <h3>📝 Insight 3: Panjang Teks</h3>
            <p>
            Rata-rata panjang input text adalah <b>{avg_input}</b> karakter, sedangkan rata-rata response text adalah
            <b>{avg_response}</b> karakter. Informasi ini berguna untuk menentukan tokenisasi, padding, dan batas
            maksimum panjang sequence.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if outlier_count > 0:
        st.markdown(
            f"""
            <div class="warning">
                <h3>⚠️ Insight 4: Outlier</h3>
                <p>
                Terdapat <b>{outlier_count}</b> data outlier berdasarkan panjang input text.
                Namun, outlier pada dataset ini masih dapat dianggap <b>aman</b> selama isi teksnya masih relevan
                dengan konteks MindMate AI. Pada data chatbot, outlier bisa terjadi secara wajar karena ada pengguna
                yang menulis curhatan sangat panjang atau kalimat yang lebih detail dibandingkan pengguna lain.
                Jadi, outlier tidak perlu langsung dihapus, kecuali jika teksnya kosong, tidak relevan, duplikat,
                atau mengandung noise.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="success">
                <h3>✅ Insight 4: Outlier</h3>
                <p>Tidak ditemukan outlier pada data yang sedang difilter.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="insight">
            <h3>🧹 Insight 5: Kualitas Data</h3>
            <p>
            Setelah proses cleaning, terdapat <b>{removed_data}</b> data yang dihapus karena duplikat
            atau missing value pada kolom penting. Hal ini menunjukkan bahwa proses pembersihan data
            penting dilakukan agar data yang digunakan untuk analisis dan training model menjadi lebih rapi.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="insight">
            <h3>⚖️ Insight 6: Keseimbangan Kategori</h3>
            <p>
            Perbedaan jumlah data antara kategori <b>{most_intent}</b> dan <b>{least_intent}</b> perlu diperhatikan.
            Jika salah satu kategori terlalu dominan, model dapat lebih sering memprediksi kategori tersebut.
            Oleh karena itu, distribusi data antar intent/emotion sebaiknya dibuat lebih seimbang sebelum proses training.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="insight">
            <h3>🤖 Insight 7: Kesiapan Dataset untuk Model AI</h3>
            <p>
            Dataset sudah dapat digunakan sebagai dasar pengembangan MindMate AI karena memiliki kolom input, response,
            dan intent/emotion. Namun, sebelum digunakan pada model, data tetap perlu melalui tahap preprocessing seperti
            case folding, tokenisasi, padding, encoding label, serta pembagian data train dan test.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="success">
            <h3>✅ Kesimpulan Insight</h3>
            <p>
            Secara umum, dataset sudah cukup informatif untuk mendukung pengembangan chatbot MindMate AI.
            Outlier yang muncul berdasarkan panjang teks masih tergolong aman selama isi datanya relevan
            dan tidak mengandung noise. Analisis dashboard menunjukkan bahwa aspek yang tetap perlu diperhatikan
            adalah kualitas teks, keseimbangan kategori, dan pengecekan outlier agar performa model menjadi lebih stabil.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif menu == "⬇️ Download Data":
    st.title("⬇️ Download Data")
    st.write("Unduh dataset hasil filter atau dataset bersih lengkap dalam format CSV.")

    st.subheader("📋 Data Hasil Filter")
    st.dataframe(filtered_df, use_container_width=True)

    filtered_csv = filtered_df.to_csv(index=False).encode("utf-8")
    clean_csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Dataset Hasil Filter",
        data=filtered_csv,
        file_name="filtered_mental_health_dataset.csv",
        mime="text/csv"
    )

    st.download_button(
        label="⬇️ Download Dataset Bersih Lengkap",
        data=clean_csv,
        file_name="clean_mental_health_dataset.csv",
        mime="text/csv"
    )
