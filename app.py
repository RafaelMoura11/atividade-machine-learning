import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Câncer de Mama — ML simplificado",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 Classificação de Câncer de Mama (Simplificado)")
st.caption("Aplicativo didático com o dataset clássico do scikit-learn.")
