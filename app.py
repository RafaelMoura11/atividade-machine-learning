import streamlit as st
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Câncer de Mama — ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1) Dados
@st.cache_data(show_spinner=False)
def load_data():
    ds = datasets.load_breast_cancer(as_frame=True)
    X = ds.frame.drop(columns=[c for c in ['target'] if c in ds.frame.columns], errors='ignore')
    y = pd.Series(ds.target, name='target')
    feature_names = list(X.columns)
    target_names = list(ds.target_names)
    return X, y, feature_names, target_names

X, y, feature_names, target_names = load_data()

st.title("🏥 Classificação de Câncer de Mama")
st.caption("Aplicativo didático com o dataset clássico do scikit-learn.")

# Sidebar
st.sidebar.header("⚙️ Configurações")
test_size = st.sidebar.slider("Proporção de teste", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
C = st.sidebar.slider("C — Força do modelo (LR)", 0.01, 10.0, 1.0)
scale = st.sidebar.checkbox("Padronizar variáveis (recomendado)", value=True)
show_unsup = st.sidebar.checkbox("Mostrar exploração não supervisionada (PCA + K-Means)", value=False)

# Abas
eda_tab, train_tab, extra_tab = st.tabs(["🔬 EDA essencial", "🧠 Treinar & Avaliar", "🌀 Não supervisionado (opcional)"])

# EDA
with eda_tab:
    st.subheader("Visão geral")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Amostras", X.shape[0])
    c2.metric("Atributos", X.shape[1])
    counts = y.value_counts().rename(index={0: target_names[0], 1: target_names[1]})
    c3.metric("Benigno", int(counts.get(target_names[1], 0)))
    c4.metric("Maligno", int(counts.get(target_names[0], 0)))

    st.subheader("Primeiras linhas")
    st.dataframe(X.head(), width='stretch')

    st.subheader("Estatísticas")
    st.dataframe(X.describe().T, width='stretch')

    with st.expander("Distribuição do alvo"):
        counts_df = counts.reset_index()
        counts_df.columns = ['Classe','count']
        fig = px.bar(counts_df, x='Classe', y='count', labels={'Classe': 'Classe', 'count': 'Contagem'}, title='Distribuição das classes')
        st.plotly_chart(fig, width='stretch')

# Supervisionado
with train_tab:
    st.subheader("Pipeline simples")
    st.write("Usamos **Logistic Regression**. Se 'Padronizar' estiver ativo, aplicamos `StandardScaler` antes do modelo.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", LogisticRegression(C=C, max_iter=1000, class_weight='balanced')))
    pipe = Pipeline(steps)

    if st.button("Treinar modelo", type="primary"):
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba),
        }

        st.subheader("📈 Métricas")
        cols = st.columns(len(metrics))
        for i, (k, v) in enumerate(metrics.items()):
            cols[i].metric(k, f"{v:.4f}")

        st.subheader("🔢 Matriz de confusão")
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                           x=[target_names[0], target_names[1]],
                           y=[target_names[0], target_names[1]],
                           title="Matriz de Confusão")
        fig_cm.update_layout(xaxis_title="Predito", yaxis_title="Real")
        st.plotly_chart(fig_cm, width='stretch')

        st.subheader("📉 Curva ROC")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Aleatório", line=dict(dash="dash")))
        fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, width='stretch')

        st.subheader("🔍 O que mais pesou na decisão? (coeficientes)")
        try:
            clf = pipe.named_steps["clf"]
            coefs = np.abs(clf.coef_[0])
            imp = pd.DataFrame({"feature": feature_names, "importance": coefs}).sort_values("importance", ascending=False)
            fig_imp = px.bar(imp.head(15), x="importance", y="feature", orientation="h", title="Top 15 coeficientes (magnitude)")
            st.plotly_chart(fig_imp, width='stretch')
            with st.expander("Ver tabela completa de importâncias"):
                st.dataframe(imp, width='stretch')
        except Exception as e:
            st.info(f"Não foi possível calcular a importância: {e}")

# Não supervisionado
with extra_tab:
    st.subheader("Exploração não supervisionada (opcional)")
    st.caption("Isto **não usa** as respostas/labels. Serve para visualizar padrões.")
    if show_unsup:
        k = st.slider("Número de clusters (k)", 2, 6, 2)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(X_scaled)

        km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        clusters = km.fit_predict(X_scaled)

        try:
            sil = silhouette_score(X_scaled, clusters)
            st.metric("Silhouette score", f"{sil:.4f}")
        except Exception:
            pass

        df_pca = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1], "Cluster": clusters.astype(str)})
        fig_pca = px.scatter(df_pca, x="PC1", y="PC2", color="Cluster", title="PCA 2D + K-Means")
        st.plotly_chart(fig_pca, width='stretch')
    else:
        st.info("Ative a opção na barra lateral para ver PCA + K-Means.")

st.markdown('---')
st.caption("© 2025 — App para estudo de ML em saúde (câncer de mama).")
