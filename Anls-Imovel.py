import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from io import BytesIO

st.set_page_config(layout="wide")
st.title("📈 Análise de Regressão Linear Interativa")

st.write("""
Faça uma análise de regressão linear simples com seus próprios dados.
Você pode inserir manualmente os dados ou carregar um arquivo `.csv`.
""")

# Escolha do modo de entrada
modo = st.radio("Escolha o modo de entrada dos dados:", ["📄 Upload de CSV", "✍️ Entrada Manual"])

if modo == "📄 Upload de CSV":
    arquivo = st.file_uploader("Faça o upload de um arquivo .csv com colunas numéricas", type="csv")
    
    if arquivo is not None:
        df = pd.read_csv(arquivo)
        st.subheader("Pré-visualização dos Dados")
        st.dataframe(df.head())

        # Filtrar colunas numéricas
        colunas_numericas = df.select_dtypes(include=np.number).columns.tolist()

        if len(colunas_numericas) < 2:
            st.error("O arquivo precisa ter pelo menos duas colunas numéricas.")
        else:
            col_x = st.selectbox("Escolha a variável independente (X - Ex: Valor Total):", colunas_numericas)
            col_y = st.selectbox("Escolha a variável dependente (Y - Ex: Preço por m²):", [col for col in colunas_numericas if col != col_x])

            X = df[[col_x]]
            y = df[col_y]

elif modo == "✍️ Entrada Manual":
    st.subheader("Entrada de Dados Manual")
    num_linhas = st.slider("Quantos imóveis você deseja inserir?", min_value=2, max_value=20, value=5)

    valores_total = []
    precos_m2 = []

    for i in range(num_linhas):
        col1, col2 = st.columns(2)
        with col1:
            valor = st.number_input(f"Valor Total do Imóvel #{i+1} (R$)", key=f"x_{i}")
        with col2:
            preco = st.number_input(f"Preço por m² do Imóvel #{i+1} (R$)", key=f"y_{i}")
        valores_total.append(valor)
        precos_m2.append(preco)

    df = pd.DataFrame({
        "Valor_Total_do_Imovel": valores_total,
        "Preco_por_m2": precos_m2
    })
    df = df[(df["Valor_Total_do_Imovel"] > 0) & (df["Preco_por_m2"] > 0)]
    if df.shape[0] >= 2:
        X = df[["Valor_Total_do_Imovel"]]
        y = df["Preco_por_m2"]
    else:
        st.warning("Insira pelo menos 2 pares de valores maiores que zero para continuar.")
        X = None
        y = None

# Realizar análise se os dados forem válidos
if 'X' in locals() and X is not None and len(X) >= 2:
    st.subheader("🔍 Resultados da Regressão Linear")

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    coef = modelo.coef_[0]
    intercepto = modelo.intercept_
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    valor_medio_m2 = y.mean()

    st.markdown(f"**Equação da reta:** `Preço_m2 = {coef:.4f} × Valor_Total + {intercepto:.2f}`")
    st.markdown(f"- **R²:** {r2:.4f}")
    st.markdown(f"- **MSE:** {mse:.2f}")
    st.markdown(f"- **Valor médio do metro quadrado:** R$ {valor_medio_m2:,.2f}")

    # Exportar resultados
    resultados = pd.DataFrame({
        "Coeficiente": [coef],
        "Intercepto": [intercepto],
        "R²": [r2],
        "MSE": [mse],
        "Valor médio do m²": [valor_medio_m2]
    })

        buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        resultados.to_excel(writer, sheet_name='Resultados', index=False)
        df.to_excel(writer, sheet_name='Dados', index=False)
        writer.save()

       st.download_button(
        "📥 Baixar Resultados (.xlsx)",
        data=buffer.getvalue(),
        file_name="regressao_resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Gráfico com Plotly
    st.subheader("📊 Gráfico de Dispersão com Regressão")
    df_grafico = X.copy()
    df_grafico["Preço_real_m2"] = y
    df_grafico["Preço_previsto_m2"] = y_pred

    fig = px.scatter(df_grafico, x=df_grafico.columns[0], y="Preço_real_m2", labels={"Preço_real_m2": "Preço por m² (real)"}, title="Regressão Linear")
    fig.add_scatter(x=df_grafico[df_grafico.columns[0]], y=df_grafico["Preço_previsto_m2"], mode="lines", name="Regressão Linear", line=dict(color="red"))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write("💻 Desenvolvido por Patricia Gutierrez")

