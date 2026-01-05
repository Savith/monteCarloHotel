import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from PIL import Image
import os
import json
import base64

# --- Configuração da Página ---
st.set_page_config(page_title="Hotel Revenue Monte Carlo", layout="wide")
st.title("Monte Carlo Hotel Revenue Simulator - Podere Sasso")

# --- Carregamento de Imagem Seguro ---
try:
    image_path = "podere.png"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="Podere di Sasso - Siena", use_container_width=True)
    else:
        st.info("Nota: 'podere.png' não encontrado. O simulador continuará sem a imagem.")
except Exception as e:
    st.warning(f"Erro ao carregar imagem: {e}")

# --- Abas ---
tab_inputs, tab_results, tab_export = st.tabs(["Configurações (Inputs)", "Resultados da Simulação", "Exportar"])

with tab_inputs:
    st.header("1. Parâmetros do Negócio")
    col1, col2 = st.columns(2)
    with col1:
        scenario_name = st.text_input("Nome do Cenário", value="Plano de Negócios v1")
        n_rooms = st.number_input("Número de Quartos", min_value=1, value=11)
        n_simulations = st.slider("Número de Simulações", 1000, 50000, 10000)
        base_adr = st.number_input("ADR Alvo (Alta Temporada €)", value=250)

    with col2:
        inflation_rate = st.slider("Inflação Anual (%)", 0.0, 10.0, 2.5)
        guest_spending = st.number_input("Gasto Extra Médio por Quarto (€)", value=50)
        adr_std_dev = st.slider("Volatilidade do Preço (Std Dev %)", 0.0, 0.3, 0.10)

    st.subheader("2. Ocupação e Sazonalidade")
    c3, c4, c5 = st.columns(3)
    occ_min = c3.slider("Ocupação Mínima", 0.0, 1.0, 0.30)
    occ_mode = c4.slider("Ocupação Provável", 0.0, 1.0, 0.60)
    occ_max = c5.slider("Ocupação Máxima", 0.0, 1.0, 0.85)

    # Coeficientes de Sazonalidade (Baseados na Toscana)
    default_seasonality = [0.34, 0.40, 0.60, 0.85, 1.00, 1.50, 2.00, 2.30, 1.40, 0.90, 0.50, 0.40]
    months_labels = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    
    st.write("Ajuste de Sazonalidade Mensal (Multiplicador):")
    cols_mo = st.columns(6)
    season_factors = {}
    for i, m_label in enumerate(months_labels):
        with cols_mo[i % 6]:
            season_factors[i+1] = st.number_input(f"{m_label}", value=default_seasonality[i], step=0.1)

# --- Lógica da Simulação (CORRIGIDA) ---
days_in_month = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
results_monthly = {}
max_possible_rev = 0

for month, coef in season_factors.items():
    days = days_in_month[month]
    
    # 1. Ajuste de ADR por Sazonalidade (Corrigido para não inflar demais)
    # O base_adr é o teto, coef/max(coef) garante que os outros meses sejam menores
    target_adr = base_adr * (coef / max(default_seasonality))
    adr_sim = np.random.normal(target_adr, target_adr * adr_std_dev, n_simulations)
    adr_sim = np.clip(adr_sim, 80, base_adr * 1.2) # Trava de realidade

    # 2. Ocupação correlacionada com a temporada
    occ_sim = np.random.triangular(occ_min, occ_mode, occ_max, n_simulations)
    occ_seasonal = np.clip(occ_sim * (coef / max(default_seasonality)) * 1.2, 0.1, 0.98)

    # 3. Cálculo da Receita
    rev_per_night = (adr_sim + guest_spending) * (occ_seasonal * n_rooms)
    results_monthly[month] = rev_per_night * days

yearly_revenue_sim = np.array(list(results_monthly.values())).sum(axis=0)

# --- Exibição de Resultados ---
with tab_results:
    st.header("Análise de Receita Anual")
    m1, m2, m3 = st.columns(3)
    m1.metric("Média (P50)", f"€ {np.mean(yearly_revenue_sim):,.0f}")
    m2.metric("Conservador (P10)", f"€ {np.percentile(yearly_revenue_sim, 10):,.0f}")
    m3.metric("Otimista (P90)", f"€ {np.percentile(yearly_revenue_sim, 90):,.0f}")

    # Gráfico de Distribuição Mensal
    st.subheader("Distribuição da Receita por Mês")
    monthly_df = pd.DataFrame({months_labels[m-1]: results_monthly[m] for m in range(1, 13)})
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=monthly_df, ax=ax, palette="vlag")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab_export:
    st.header("Exportar Dados")
    csv = monthly_df.describe().to_csv().encode('utf-8')
    st.download_button("Baixar Resumo Estatístico (CSV)", csv, "resumo_hotel.csv", "text/csv")
