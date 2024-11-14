import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Carregando o CSV da pasta 'datasets'
df_rides = pd.read_csv('datasets/df_rides.csv')

# Convertendo as colunas de data e hora para o formato datetime
df_rides['ride_date'] = pd.to_datetime(df_rides['ride_date'])
df_rides['time_start'] = pd.to_datetime(df_rides['time_start'])
df_rides['time_end'] = pd.to_datetime(df_rides['time_end'])  # Corrigido aqui

# Criando a coluna de intervalo de 15 minutos
df_rides['time_start_15min'] = df_rides['time_start'].dt.floor('15T')

# Contando a quantidade de bicicletas por intervalo de 15 minutos
bike_counts = df_rides.groupby('time_start_15min').size().reset_index(name='bike_count')

# Merge para associar 'bike_count' com o DataFrame original
df_rides = pd.merge(df_rides, bike_counts, on='time_start_15min', how='left')

# Extraindo características adicionais
df_rides['hour_of_day'] = df_rides['time_start'].dt.hour
df_rides['day_of_week'] = df_rides['time_start'].dt.weekday
df_rides['user_age'] = (pd.to_datetime('today') - pd.to_datetime(df_rides['user_birthdate'])).dt.days // 365

# Seleção das colunas que serão usadas como features
df_rides_model = df_rides[['user_gender', 'user_age', 'station_start', 'station_end', 'hour_of_day', 'day_of_week']]

# Convertendo variáveis categóricas em variáveis numéricas
df_rides_model = pd.get_dummies(df_rides_model, drop_first=True)

# Dividindo os dados em variáveis independentes (X) e dependente (y)
X = df_rides_model
y = df_rides['bike_count']

# Normalizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinando o modelo KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Streamlit - Interface do Usuário
st.title('Previsão de Bicicletas por Estação e Hora')

# Selecionar a estação de início e fim
stations = df_rides['station_start'].unique()
selected_station_start = st.selectbox('Escolha a estação de início', stations)

# Selecionar a hora de início
hour_of_day = st.slider('Escolha a hora de início', min_value=0, max_value=23, value=12)

# Selecionar o dia da semana
day_of_week = st.selectbox('Escolha o dia da semana', ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'])

# Convertendo o dia da semana para número (0-6)
day_of_week_dict = {'Segunda': 0, 'Terça': 1, 'Quarta': 2, 'Quinta': 3, 'Sexta': 4, 'Sábado': 5, 'Domingo': 6}
day_of_week_num = day_of_week_dict[day_of_week]

# Calcular a idade do usuário (vamos simular uma idade média para este exemplo)
user_age = 30  # Você pode ajustar isso conforme necessário

# Preprocessando os dados de entrada
# Criar um DataFrame para o novo input de dados
input_data = pd.DataFrame({
    'user_gender': [1],  # Assumindo '1' para feminino, ou 0 para masculino
    'user_age': [user_age],
    'station_start': [selected_station_start],
    'station_end': [0],  # Supondo que a estação de chegada não seja relevante aqui
    'hour_of_day': [hour_of_day],
    'day_of_week': [day_of_week_num]
})

# Convertendo variáveis categóricas (como a estação de início) em variáveis dummy
input_data = pd.get_dummies(input_data, drop_first=True)

# Ajustando o número de colunas para corresponder ao treinamento
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Normalizando os dados de entrada
input_data_scaled = scaler.transform(input_data)

# Fazendo a previsão
bike_pred = knn.predict(input_data_scaled)

# Dividindo o valor da previsão por 490 e ignorando os valores após a vírgula
bike_pred_normalized = int(bike_pred[0] / 490)  # Convertendo para inteiro (truncando a parte decimal)

# Exibindo a previsão no Streamlit
st.write(f'Predição de bicicletas para o intervalo selecionado: {bike_pred_normalized} bicicletas')
