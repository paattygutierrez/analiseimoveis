import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(layout="wide")

st.title("Análise de Regressão Linear: Valor Total do Imóvel vs Preço por m²")

st.write("""
Digite manualmente até 20 pares de dados para realizar uma análise de regressão linear.
Cada linha representa um imóvel, com seu valor total e preço por metro quadrado.
""")

# Número de linhas para entrada manual (até 20)
num_linhas = st.slider("Quantos imóveis você deseja inserir?", min_value=2, max_value=20, value=5)

valores_totais = []
precos_m2 = []

st.subheader("Entrada de Dados Manual")

for i in range(num_linhas):
    col1, col2 = st.columns(2)
    with col1:
        valor = st.number_input(f"Valor_Total do imóvel #{i+1} (R$)", key=f"valor_{i}")
    with col2:
        preco = st.number_input(f"Preço por m² do imóvel #{i+1} (R$)", key=f"preco_{i}")
    
    valores_totais.append(valor)
    precos_m2.append(preco)

# Converter para DataFrame
df = pd.DataFrame({
    "Valor_Total": valores_totais,
    "Preco_m2": precos_m2
})

# Remover linhas com zeros (opcional)
df = df[(df["Valor_Total"] > 0) & (df["Preco_m2"] > 0)]

if df.shape[0] < 2:
    st.warning("Insira pelo menos 2 imóveis com valores válidos (maiores que zero) para continuar.")
else:
    st.subheader("Dados Inseridos")
    st.dataframe(df)

    # Separar X e y
    X = df[["Valor_Total"]]
    y = df["Preco_m2"]

    # Criar e treinar o modelo
    modelo = LinearRegression()
    modelo.fit(X, y)

    # Prever
    y_pred = modelo.predict(X)

    # Avaliar o modelo
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    coef = modelo.coef_[0]
    intercepto = modelo.intercept_

    st.subheader("Resultados da Regressão Linear")
    st.write(f"**Equação da reta:** $Preço\\_m2 = {coef:.6f} \\times Valor\\_Total + {intercepto:.2f}$")
    st.write(f"**R² (Coeficiente de Determinação):** ${r2:.4f}$")
    st.write(f"**MSE (Erro Quadrático Médio):** ${mse:.2f}$")

    # Gráfico
    st.subheader("Gráfico de Regressão Linear")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', label="Dados reais")
    ax.plot(X, y_pred, color='red', label="Regressão Linear")
    ax.set_xlabel("Valor Total do Imóvel (R$)")
    ax.set_ylabel("Preço por m² (R$)")
    ax.set_title("Regressão Linear: Valor Total vs Preço por m²")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.write("Desenvolvido por Patricia Gutierrez")
