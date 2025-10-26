# 🏥 Machine Learning Aplicado à Saúde — Câncer de Mama (Streamlit)

Aplicação didática em **Streamlit** que demonstra **aprendizado supervisionado** (classificação com Logistic Regression) e **não supervisionado** (PCA + K-Means) usando o dataset clássico **Breast Cancer (scikit-learn)**.

> **Objetivo:** mostrar de forma clara como coletar, tratar, modelar e interpretar dados na área da saúde — com interface simples, métricas, gráficos e explicações.

---

## 🔗 Links
- **Repositório GitHub:** [https://github.com/RafaelMoura11/atividade-machine-learning](https://github.com/RafaelMoura11/atividade-machine-learning)]

---

## 📦 Requisitos
- Python 3.9+
- Bibliotecas: `streamlit`, `scikit-learn`, `pandas`, `numpy`, `plotly`

> Há um `requirements.txt` no repositório. Instale tudo com:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Como rodar localmente
```bash
streamlit run app.py
```
O navegador abrirá em `http://localhost:8501`.

---

## 🗂️ Estrutura do projeto (sugerida)
```
.
├── app.py
├── requirements.txt
└── README.md
```

---

## 🧠 Dataset utilizado
- **Fonte:** `sklearn.datasets.load_breast_cancer()`
- **Tamanho:** 569 amostras, 30 atributos numéricos
- **Alvo (`target`):** 0 = **maligno**, 1 = **benigno**

> As features descrevem **características das células** observadas em exames (ex.: `mean radius`, `mean texture`, `mean area` etc.).

---

## 🧭 Como usar o app
A interface possui **três abas**:

### 1) 🔬 EDA essencial
- **Métricas iniciais:** nº de amostras, nº de atributos, contagem de benignos/malignos.
- **Primeiras linhas** e **estatísticas** (`count`, `mean`, `std`, quartis).
- **Distribuição do alvo** (gráfico de barras).

> **Objetivo:** conhecer os dados e entender escala, variação e equilíbrio entre classes.

### 2) 🧠 Treinar & Avaliar (aprendizado **supervisionado**)
- **Modelo:** Logistic Regression (com opção de **padronizar** via `StandardScaler`).
- **Configurações na sidebar:** `test_size`, `random_state`, `C` (força do modelo) e `padronização`.
- **Métricas exibidas:** `Accuracy`, `Precision`, `Recall`, `F1-score`, `ROC AUC`.
- **Gráficos:** Matriz de confusão, Curva ROC.
- **Interpretabilidade:** “Importância” das variáveis via **magnitude dos coeficientes**.

> **Leitura rápida:**  
> - **Precision:** acerto entre as vezes que o modelo previu “positivo”.  
> - **Recall (sensibilidade):** entre todos os positivos reais, quantos o modelo encontrou.  
> - **F1:** equilíbrio entre Precision e Recall.  
> - **ROC AUC:** separação global entre classes (1.0 = perfeito; 0.5 = aleatório).  
> - **Saúde:** prioriza-se **Recall** (evitar falsos negativos).

### 3) 🌀 Exploração não supervisionada (PCA + K-Means)
- **PCA (2D):** redução de 30 → 2 dimensões para visualização.
- **K-Means:** ajuste do nº de clusters (**k**).
- **Qualidade do agrupamento:** `Silhouette score` (próximo de 1 é melhor).

> **Observação:** aqui **não usamos o alvo**; o algoritmo só descobre **padrões naturais** nos dados.

---

## ⚙️ Parâmetros importantes
- **Padronizar variáveis:** recomendado para dar escala comparável às features (impacta LR, PCA e K-Means).
- **C (Logistic Regression):** controla complexidade (quanto maior, modelo mais “ajustado” aos dados).
- **Random state:** torna os resultados reprodutíveis.
- **k (K-Means):** nº de grupos. Para este dataset, **k=2** costuma refletir benigno/maligno melhor.

---

## 📊 Interpretação das métricas (resumo)
- **Accuracy:** acertos no geral.  
- **Precision:** se o modelo disse “positivo”, quão certo ele estava?  
- **Recall:** entre todos os positivos reais, quantos o modelo achou? (**prioridade em saúde**)  
- **F1-score:** equilíbrio entre Precision e Recall.  
- **ROC AUC:** separação global entre classes (quanto maior, melhor).

**Matriz de confusão:** mostra acertos/erros por classe (falsos positivos x falsos negativos).  
**Coeficientes (LR):** mostram quais variáveis mais influenciam a decisão (maior magnitude = maior peso).


