import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

st.title("Página 1")
st.write("Esta é a Página 1.")
# Função para carregar os dados
@st.cache_data
def carregar_dados(caminho_csv):
    try:
        df = pd.read_csv(caminho_csv)
        return df
    except FileNotFoundError:
        st.error("Erro: Arquivo CSV não encontrado. Verifique o caminho.")
        return None

# Definir o caminho para os arquivos CSV
caminho_csv = 'datasets/df_rides.csv'
df = carregar_dados(caminho_csv)

if df is not None:
    # Verificar se as colunas necessárias estão presentes
    required_columns = ['ride_date', 'time_start', 'user_birthdate', 'station_start', 'user_gender']
    if all(col in df.columns for col in required_columns):
        # Certificar-se de que as colunas de data estão no formato correto
        df['ride_date'] = pd.to_datetime(df['ride_date'], errors='coerce')
        df['time_start'] = pd.to_datetime(df['time_start'], errors='coerce')

        # Remover linhas com datas inválidas
        df.dropna(subset=['ride_date', 'time_start'], inplace=True)

        # Criar a coluna 'time_start_15min' arredondando para o intervalo de 15 minutos
        df['time_start_15min'] = df['time_start'].dt.floor('15T')

        # Criar variáveis adicionais para o modelo
        df['hour_of_day'] = df['time_start'].dt.hour
        df['day_of_week'] = df['time_start'].dt.weekday
        df['user_age'] = (pd.to_datetime('today') - pd.to_datetime(df['user_birthdate'], errors='coerce')).dt.days // 365

        # Substituir valores ausentes em 'user_age' por uma média
        df['user_age'].fillna(df['user_age'].mean(), inplace=True)

        # Criar o alvo (bike_count) agregando os dados por intervalo de 15 minutos
        bike_counts = df.groupby('time_start_15min').size().reset_index(name='bike_count')
        df = pd.merge(df, bike_counts, on='time_start_15min', how='left')

        # Garantir que bike_count não tenha valores ausentes
        df['bike_count'] = df['bike_count'].fillna(0)

        # Seleção de variáveis para o modelo
        df_rides_model = df[['user_gender', 'user_age', 'station_start', 'hour_of_day', 'day_of_week']]
        df_rides_model = pd.get_dummies(df_rides_model, drop_first=True)

        X = df_rides_model
        y = df['bike_count']

        # Normalização
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Verificar se há valores ausentes ou inválidos em X_scaled
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            st.error("Erro: Os dados contêm valores ausentes ou inválidos após a normalização.")
            st.stop()

        # Divisão dos dados para treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Criar e treinar o modelo KNN
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)

        # Previsões
        y_pred = knn.predict(X_test)

        # Avaliação de métricas
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Cálculo da acurácia
        margem_erro = 1  # Definir margem de erro
        acertos = np.sum(np.abs(y_test - y_pred) <= margem_erro)
        total = len(y_test)
        accuracy = acertos / total

        # Cálculo da Curva ROC (transformando valores contínuos em binários para demonstração)
        y_test_bin = (y_test > y_test.median()).astype(int)
        y_pred_bin = (y_pred > y_test.median()).astype(int)
        fpr, tpr, _ = roc_curve(y_test_bin, y_pred)
        roc_auc = roc_auc_score(y_test_bin, y_pred)

        # Exibir informações no Streamlit
        st.title('Análise do Modelo KNN')
        st.subheader('Métricas de Avaliação')
        st.write(f'R²: {r2:.2f}')
        st.write(f'MSE (Mean Squared Error): {mse:.2f}')
        st.write(f'RMSE (Root Mean Squared Error): {rmse:.2f}')
        st.write(f'Acurácia: {accuracy:.2%} (margem de erro de {margem_erro})')

        # Gráfico da Curva ROC
        st.subheader('Curva ROC')
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_title('Curva ROC')
        ax.set_xlabel('Taxa de Falsos Positivos')
        ax.set_ylabel('Taxa de Verdadeiros Positivos')
        ax.legend(loc='lower right')
        st.pyplot(fig)

        # Gráfico de Dispersão (Real vs. Previsto)
        st.subheader('Gráfico de Dispersão (Real vs. Previsto)')
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='green', alpha=0.6)
        ax.set_title('Real vs. Previsto')
        ax.set_xlabel('Valor Real')
        ax.set_ylabel('Valor Previsto')
        st.pyplot(fig)
    else:
        st.error("Erro: Colunas necessárias não estão presentes no arquivo CSV.")
else:
    st.error("Erro ao carregar os dados.")
