import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from statsmodels.tsa.stattools import adfuller

# Configuração da página
st.set_page_config(page_title="Simulador AR(1) e Raiz Unitária", layout="wide")

st.title("Simulador de Processos AR(1) e Estacionariedade")
st.markdown("""
Esta ferramenta permite simular um processo Autoregressivo de Ordem 1: 
**$Y_t = \\alpha + \\beta t + \\phi Y_{t-1} + \\epsilon_t$**

Ajuste os parâmetros na barra lateral para observar o comportamento da série e o resultado do Teste Dickey-Fuller Aumentado (ADF).
""")

# Barra Lateral - Controles
st.sidebar.header("Parâmetros do Modelo")
phi = st.sidebar.slider("Coeficiente Autoregressivo (Φ)", min_value=0.0, max_value=1.05, value=0.80, step=0.01)
drift = st.sidebar.slider("Constante / Drift (α)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
trend = st.sidebar.slider("Tendência Determinística (β)", min_value=-0.5, max_value=0.5, value=0.0, step=0.01)
n_obs = st.sidebar.slider("Tamanho da Amostra (T)", min_value=50, max_value=500, value=200, step=50)

# Controle de aleatoriedade para gerar novos choques
if 'seed' not in st.session_state:
    st.session_state.seed = 42

if st.sidebar.button("Gerar Novos Choques Aleatórios"):
    st.session_state.seed = np.random.randint(0, 10000)

# Motor da Simulação
np.random.seed(st.session_state.seed)
erros = np.random.normal(0, 1, n_obs) # Ruído branco com desvio padrão 1
y = np.zeros(n_obs)
y[0] = drift + erros[0]

for t in range(1, n_obs):
    y[t] = drift + trend * t + phi * y[t-1] + erros[t]

df = pd.DataFrame({'Tempo': range(n_obs), 'Y': y})

# Gráfico Interativo com Plotly
fig = px.line(df, x='Tempo', y='Y', title=f"Série Simulada (Φ = {phi:.2f})")
fig.update_layout(xaxis_title="Tempo (t)", yaxis_title="Y_t", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

# Diagnóstico Teórico
with col1:
    st.subheader("Classificação Teórica")
    if phi > 1.0:
        st.error("🚨 **Não-Estacionário Explosivo** (Φ > 1)")
        st.write("A variância cresce exponencialmente com o tempo. Choques passados ganham força no futuro.")
    elif phi == 1.0:
        if trend != 0:
            st.warning("📈 **Passeio Aleatório com Tendência** (Raiz Unitária + Drift Determinístico)")
        elif drift != 0:
            st.warning("🚶 **Passeio Aleatório com Drift** (Raiz Unitária)")
        else:
            st.warning("🎲 **Passeio Aleatório Puro** (Raiz Unitária)")
        st.write("A variância cresce linearmente. Choques passados têm impacto permanente (memória infinita).")
    else:
        if trend != 0:
            st.info("📉 **Estacionário em Torno de uma Tendência** (Trend-Stationary)")
        else:
            st.success("✅ **Estacionário** (Covariance Stationary)")
        st.write("O processo reverte à sua média (ou tendência). Choques passados se dissipam ao longo do tempo.")

# Teste Empírico (ADF)
with col2:
    st.subheader("Teste Empírico ADF")
    st.write("**H0:** A série possui Raiz Unitária (Não é estacionária)")
    
    # Define a especificação do teste com base na presença visual de tendência
    reg_param = 'ct' if trend != 0 else 'c'
    
    try:
        resultado_adf = adfuller(y, regression=reg_param)
        estatistica_t = resultado_adf[0]
        p_valor = resultado_adf[1]
        valor_critico_5 = resultado_adf[4]['5%']
        
        st.metric("Estatística t", f"{estatistica_t:.4f}")
        st.write(f"**Valor-p:** {p_valor:.4f}")
        st.write(f"**Valor Crítico (5%):** {valor_critico_5:.4f}")
        
        if p_valor <= 0.05:
            st.success("Rejeita-se H0. As evidências empíricas indicam que a série é **Estacionária**.")
        else:
            st.error("Não se rejeita H0. As evidências empíricas indicam a presença de **Raiz Unitária**.")
            
    except Exception as e:
        st.write("Não foi possível calcular o teste para esta configuração extrema.")
