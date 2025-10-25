import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets

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

# Sidebar
st.sidebar.header("⚙️ Configurações")
test_size = st.sidebar.slider("Proporção de teste", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
C = st.sidebar.slider("C — Força do modelo (LR)", 0.01, 10.0, 1.0)
scale = st.sidebar.checkbox("Padronizar variáveis (recomendado)", value=True)
show_unsup = st.sidebar.checkbox("Mostrar exploração não supervisionada (PCA + K-Means)", value=False)
