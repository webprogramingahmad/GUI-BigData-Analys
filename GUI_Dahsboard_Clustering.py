import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(layout="wide", page_title="Analisis Clustering Penderita Mata")

st.title("📊 Analisis Clustering Penderita Mata Minus")

# --- Upload File Excel
uploaded_file = st.file_uploader("Unggah file Excel Data Penderita Mata", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # --- Praproses
    df = df.dropna(subset=['Usia', 'Jenis Kelamin', 'Lensa Minus', 'Alamat'])
    df['Jenis Kelamin'] = df['Jenis Kelamin'].astype(str).str.strip().str.upper()
    df['Profesi'] = df['Profesi'].astype(str).str.strip().str.title()
    df['Alamat'] = df['Alamat'].astype(str).str.strip()
    df['Jenis Kelamin Asli'] = df['Jenis Kelamin']
    df['Jenis Kelamin'] = df['Jenis Kelamin'].map({'L': 1, 'P': 0})
    df['Profesi Encoded'] = LabelEncoder().fit_transform(df['Profesi'])
    df['Kota'] = df['Alamat'].apply(lambda a: a.split(',')[-1].strip().title())
    df['Kota Encoded'] = LabelEncoder().fit_transform(df['Kota'])

    def extract_avg_minus(value):
        value = str(value)
        matches = re.findall(r'R-?[\d,\.]+|L-?[\d,\.]+', value, flags=re.IGNORECASE)
        numbers = []
        for m in matches:
            num_str = re.findall(r'-?[\d,\.]+', m)[0].replace(',', '.')
            try:
                num = float(num_str)
                numbers.append(-abs(num))
            except:
                continue
        return sum(numbers) / len(numbers) if numbers else None

    df['Minus Rata-rata'] = df['Lensa Minus'].apply(extract_avg_minus)

    def is_normal_eye(minus, plus, cly):
        def check_zero(value):
            if isinstance(value, str):
                value = value.replace(",", ".").replace(" ", "")
                return bool(re.match(r'R[-+]?0(?:\.0*)?L[-+]?0(?:\.0*)?$', value))
            return False
        return check_zero(minus) and check_zero(plus) and check_zero(cly)

    df['Mata Normal'] = df.apply(lambda row: is_normal_eye(row.get('Lensa Minus'), row.get('Lensa Plush'), row.get('Lensa Cly')), axis=1)
    df['Punya_Minus'] = df['Lensa Minus'].apply(lambda x: not re.fullmatch(r'R-?0(?:\.0*)?L-?0(?:\.0*)?', str(x).replace(",", ".").replace(" ", "").upper()))
    df['Punya_Plus'] = df['Lensa Plush'].apply(lambda x: not re.fullmatch(r'R\+?0(?:\.0*)?L\+?0(?:\.0*)?', str(x).replace(",", ".").replace(" ", "").upper()))
    df['Punya_Cly'] = df['Lensa Cly'].apply(lambda x: not re.fullmatch(r'R-?0(?:\.0*)?L-?0(?:\.0*)?', str(x).replace(",", ".").replace(" ", "").upper()))

    # === Visualisasi 1: Status Mata dan Jenis Kelainan ===
    st.subheader("📊 Visualisasi Kondisi Mata")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 📈 Distribusi Status Mata")
        jumlah_normal = df['Mata Normal'].sum()
        jumlah_tidak_normal = len(df) - jumlah_normal
        data_status = pd.DataFrame({'Status Mata': ['Normal', 'Tidak Normal'], 'Jumlah': [jumlah_normal, jumlah_tidak_normal]})
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        sns.barplot(data=data_status, x='Status Mata', y='Jumlah', hue='Status Mata', palette='pastel', legend=False, ax=ax1)
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', label_type='edge', padding=3)
        st.pyplot(fig1)

    with col2:
        st.markdown("##### 📊 Distribusi Jenis Kelainan Mata")
        jumlah_minus = df['Punya_Minus'].sum()
        jumlah_plus = df['Punya_Plus'].sum()
        jumlah_cly = df['Punya_Cly'].sum()
        data_kelainan = pd.DataFrame({
            'Jenis Kelainan': ['Minus', 'Plus', 'Silinder'],
            'Jumlah': [jumlah_minus, jumlah_plus, jumlah_cly]
        })
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.barplot(data=data_kelainan, x='Jenis Kelainan', y='Jumlah', hue='Jenis Kelainan', palette='Set2', legend=False, ax=ax2)
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%d', label_type='edge', padding=3)
        st.pyplot(fig2)

    # === Clustering dan Visualisasi ===
    df_cluster = df[['Usia', 'Jenis Kelamin', 'Minus Rata-rata', 'Kota Encoded', 'Profesi Encoded',
                     'Jenis Kelamin Asli', 'Kota', 'Profesi']].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster[['Usia', 'Jenis Kelamin', 'Minus Rata-rata', 'Kota Encoded', 'Profesi Encoded']])

    st.subheader("📊 Visualisasi Clustering")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("##### 🔍 Elbow Method")
        wcss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.plot(range(1, 11), wcss, marker='o')
        ax3.set_title("Elbow Method")
        ax3.set_xlabel("Jumlah Cluster")
        ax3.set_ylabel("WCSS")
        ax3.tick_params(labelsize=8)
        st.pyplot(fig3)

    with col4:
        st.markdown("##### 📌 PCA Clustering")
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        df_cluster['PCA1'] = components[:, 0]
        df_cluster['PCA2'] = components[:, 1]

        fig4, ax4 = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df_cluster, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', ax=ax4)
        ax4.set_title("Clustering Berdasarkan PCA")
        ax4.tick_params(labelsize=8)
        ax4.legend(title='Cluster', fontsize=6, title_fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
        st.pyplot(fig4)

    # === Visualisasi Tambahan: Jenis Kelamin, Kota, Profesi ===
    st.subheader("📊 Distribusi Data dalam Cluster")
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("##### 👤 Jenis Kelamin")
        fig5, ax5 = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df_cluster, x='PCA1', y='PCA2', hue='Jenis Kelamin', style='Cluster', palette='Set1', ax=ax5)
        ax5.set_title("Distribusi Jenis Kelamin")
        ax5.tick_params(labelsize=8)
        ax5.legend(title='Jenis Kelamin', fontsize=6, title_fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
        st.pyplot(fig5)

    with col6:
        st.markdown("##### 🏙️ Kota")
        fig6, ax6 = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df_cluster, x='PCA1', y='PCA2', hue='Kota', style='Cluster', palette='tab10', ax=ax6)
        ax6.set_title("Distribusi Kota")
        ax6.tick_params(labelsize=8)
        ax6.legend(title='Kota', fontsize=6, title_fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
        st.pyplot(fig6)

    col7, col8 = st.columns(2)

    with col7:
        st.markdown("##### 💼 Profesi")
        fig7, ax7 = plt.subplots(figsize=(5, 4))
        sns.scatterplot(data=df_cluster, x='PCA1', y='PCA2', hue='Profesi', style='Cluster', palette='tab10', ax=ax7)
        ax7.set_title("Distribusi Profesi")
        ax7.tick_params(labelsize=8)
        ax7.legend(title='Profesi', fontsize=6, title_fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
        st.pyplot(fig7)

    # === Ringkasan Cluster Dalam Satu Tabel Geser Horizontal ===
    st.subheader("📋 Ringkasan Gabungan per Cluster (Geser untuk melihat semua)")

    # Rata-rata usia & minus
    mean_table = df_cluster.groupby('Cluster')[['Usia', 'Minus Rata-rata']].mean().round(2)
    mean_table.columns = ['Rata-rata Usia', 'Rata-rata Minus']

    # Distribusi kategori
    jk_table = df_cluster.groupby(['Cluster', 'Jenis Kelamin Asli']).size().unstack(fill_value=0)
    jk_table.columns = [f'JK: {col}' for col in jk_table.columns]

    profesi_table = df_cluster.groupby(['Cluster', 'Profesi']).size().unstack(fill_value=0)
    profesi_table.columns = [f'Profesi: {col}' for col in profesi_table.columns]

    kota_table = df_cluster.groupby(['Cluster', 'Kota']).size().unstack(fill_value=0)
    kota_table.columns = [f'Kota: {col}' for col in kota_table.columns]

    # Gabungkan semua
    ringkasan_lengkap = pd.concat([mean_table, jk_table, profesi_table, kota_table], axis=1)

    # Tampilkan dengan scroll horizontal
    st.dataframe(ringkasan_lengkap)
