...

# --- Nova Aba: Reformas ---
with tabs[0]:
    st.header("Simulação de Reformas")
    st.markdown("Defina setores diferentes da reforma com custos médios e incertezas. Ideal para estimar reformas de quartos, áreas comuns, etc.")

    room_types = ["Quarto Standard", "Quarto Luxo", "Banheiros/Ambientes"]
    reformas = []
    distros = {"Normal": "Distribuição normal: simétrica, com média e desvio padrão.",
               "Triangular": "Triangular: define mínimo, mais provável e máximo.",
               "Uniforme": "Uniforme: qualquer valor entre mínimo e máximo é igualmente provável."}

    n_simulations_reforma = st.slider("Nº de Simulações (Reforma)", 1000, 100000, 10000)

    for i, nome in enumerate(room_types):
        with st.expander(f"{nome}"):
            metragem = st.number_input(f"Metragem {nome} (m²)", min_value=1.0, value=30.0, key=f"metragem_{i}")
            tipo_dist = st.selectbox(f"Distribuição do Custo {nome}", options=["Normal", "Triangular", "Uniforme"], key=f"dist_{i}")

            if tipo_dist == "Normal":
                media = st.number_input(f"Custo Médio €/m² - {nome}", value=500.0, key=f"media_{i}")
                std = st.number_input(f"Desvio Padrão €/m² - {nome}", value=50.0, key=f"std_{i}")
                custo_sim = np.random.normal(media, std, n_simulations_reforma)
            elif tipo_dist == "Triangular":
                min_val = st.number_input(f"Valor Mínimo €/m² - {nome}", value=400.0, key=f"min_{i}")
                mode_val = st.number_input(f"Valor Mais Provável €/m² - {nome}", value=500.0, key=f"mode_{i}")
                max_val = st.number_input(f"Valor Máximo €/m² - {nome}", value=600.0, key=f"max_{i}")
                custo_sim = np.random.triangular(min_val, mode_val, max_val, n_simulations_reforma)
            elif tipo_dist == "Uniforme":
                min_val = st.number_input(f"Valor Mínimo €/m² - {nome}", value=400.0, key=f"umin_{i}")
                max_val = st.number_input(f"Valor Máximo €/m² - {nome}", value=600.0, key=f"umax_{i}")
                custo_sim = np.random.uniform(min_val, max_val, n_simulations_reforma)

            total_sim = custo_sim * metragem
            reformas.append(total_sim)

    total_reforma_sim = np.sum(reformas, axis=0)

    st.subheader("Resultados da Reforma (Simulação Total)")
    st.metric("Média Total da Reforma", f"€{np.mean(total_reforma_sim):,.2f}")
    st.metric("P5", f"€{np.percentile(total_reforma_sim, 5):,.2f}")
    st.metric("P95", f"€{np.percentile(total_reforma_sim, 95):,.2f}")
    st.metric("Intervalo de Confiança 90%", f"€{np.percentile(total_reforma_sim, 5):,.2f} - €{np.percentile(total_reforma_sim, 95):,.2f}")

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.histplot(total_reforma_sim, bins=50, kde=True, ax=ax3, color='lightgray')
    ax3.axvline(np.mean(total_reforma_sim), color='red', linestyle='--', label='Média')
    ax3.axvline(np.percentile(total_reforma_sim, 5), color='orange', linestyle='--', label='P5')
    ax3.axvline(np.percentile(total_reforma_sim, 95), color='green', linestyle='--', label='P95')
    ax3.set_title("Distribuição do Custo Total da Reforma")
    ax3.set_xlabel("Custo (€)")
    ax3.legend()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    if st.button("Exportar Reforma para Excel"):
        df_reformas = pd.DataFrame({"Simulação Reforma Total (€)": total_reforma_sim})
        df_reformas.to_excel("reformas_simuladas.xlsx", index=False)
        st.success("Exportado como reformas_simuladas.xlsx")

    st.markdown("**Distribuições Explicadas:**")
    for k, v in distros.items():
        st.markdown(f"**{k}**: {v}")

    # Armazena no cenário salvo
    st.session_state["reforma_total_sim"] = total_reforma_sim.tolist()
