import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(layout="wide") # Opcional: para usar a largura total da tela

st.title("Análise de Regressão Linear: Valor Total do Imóvel vs Preço por m²")

st.write("""
Este aplicativo permite que você faça upload de um arquivo CSV contendo dados de imóveis para realizar uma análise de regressão linear.
O modelo tentará prever o 'Preço por m²' com base no 'Valor_Total' do imóvel.

**Formato do arquivo CSV esperado:**
O arquivo CSV deve conter pelo menos duas colunas:
- `Valor_Total`: O valor total do imóvel (numérico).
- `Preco_m2`: O preço por metro quadrado do imóvel (numérico).
""")

# Widget para upload de arquivo
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Verificar se as colunas necessárias existem
        if "Valor_Total" not in df.columns or "Preco_m2" not in df.columns:
            st.error("O arquivo CSV deve conter as colunas 'Valor_Total' e 'Preco_m2'.")
        else:
            # Converter colunas para numérico, tratando erros
            df["Valor_Total"] = pd.to_numeric(df["Valor_Total"], errors='coerce')
            df["Preco_m2"] = pd.to_numeric(df["Preco_m2"], errors='coerce')

            # Remover linhas com valores NaN após a conversão
            df.dropna(subset=["Valor_Total", "Preco_m2"], inplace=True)

            if df.empty:
                st.warning("Não há dados válidos nas colunas 'Valor_Total' e 'Preco_m2' após a limpeza.")
            else:
                st.subheader("Prévia dos dados carregados:")
                st.dataframe(df.head())

                # Separar X e Y
                X = df[["Valor_Total"]]
                y = df["Preco_m2"]

                # Verificar se há dados suficientes para a regressão
                if len(X) < 2:
                    st.warning("São necessários pelo menos 2 pontos de dados para realizar a regressão linear.")
                else:
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

                    st.subheader("Resultados da Regressão Linear:")
                    st.write(f"**Equação da reta:** $Preço\_m2 = {coef:.6f} \\times Valor\_Total + {intercepto:.2f}$")
                    st.write(f"**R² (Coeficiente de Determinação):** ${r2:.4f}$")
                    st.write(f"**MSE (Erro Quadrático Médio):** ${mse:.2f}$")

                    # Gráfico
                    st.subheader("Gráfico de Regressão Linear")
                    fig, ax = plt.subplots(figsize=(10,6))
                    ax.scatter(X, y, color='blue', label="Dados reais")
                    ax.plot(X, y_pred, color='red', label="Regressão Linear")
                    ax.set_xlabel("Valor Total do Imóvel (R$)")
                    ax.set_ylabel("Preço por m² (R$)")
                    ax.set_title("Regressão Linear: Valor Total vs Preço por m²")
                    ax.legend()
                    ax.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
else:
    st.info("Por favor, faça upload de um arquivo CSV para começar.")
