import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
import plotly.express as px

st.set_page_config(page_title="Câncer de Mama — ML simplificado", layout="wide", initial_sidebar_state="expanded")
st.title("🏥 Classificação de Câncer de Mama (Simplificado)")
st.caption("Aplicativo didático com o dataset clássico do scikit-learn.")

@st.cache_data(show_spinner=False)
def load_data():
    ds = datasets.load_breast_cancer(as_frame=True)
    X = ds.frame.drop(columns=[c for c in ['target'] if c in ds.frame.columns], errors='ignore')
    y = pd.Series(ds.target, name='target')
    feature_names = list(X.columns)
    target_names = list(ds.target_names)
    return X, y, feature_names, target_names

X, y, feature_names, target_names = load_data()

st.sidebar.header("⚙️ Configurações")
test_size = st.sidebar.slider("Proporção de teste", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
C = st.sidebar.slider("C — Força do modelo (LR)", 0.01, 10.0, 1.0)
scale = st.sidebar.checkbox("Padronizar variáveis (recomendado)", value=True)
show_unsup = st.sidebar.checkbox("Mostrar exploração não supervisionada (PCA + K-Means)", value=False)

eda_tab, train_tab, extra_tab = st.tabs(["🔬 EDA essencial", "🧠 Treinar & Avaliar", "🌀 Não supervisionado (opcional)"])

with eda_tab:
    st.subheader("Visão geral")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Amostras", X.shape[0])
    c2.metric("Atributos", X.shape[1])
    counts = y.value_counts().rename(index={0: "malignant", 1: "benign"})
    # mapeia para ordem desejada
    c3.metric("Benigno", int(counts.get("benign", 0)))
    c4.metric("Maligno", int(counts.get("malignant", 0)))

    st.subheader("Primeiras linhas")
    st.dataframe(X.head(), width='stretch')

    st.subheader("Estatísticas")
    st.dataframe(X.describe().T, width='stretch')

    with st.expander("Distribuição do alvo"):
        counts = y.value_counts().rename(index={0: "Maligno", 1: "Benigno"})
        counts_df = counts.reset_index()
        counts_df.columns = ['Classe', 'count']
        fig = px.bar(counts_df, x='Classe', y='count',
                     labels={'Classe': 'Classe', 'count': 'Contagem'},
                     title='Distribuição das classes')
        st.plotly_chart(fig, width='stretch')
