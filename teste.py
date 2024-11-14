import streamlit as st
import pandas as pd
import pydeck as pdk
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

# Função para carregar os dados com cache
@st.cache_data
def carregar_dados(caminho_csv_rides, caminho_csv_stations):
    try:
        df_rides = pd.read_csv(caminho_csv_rides)
        df_stations = pd.read_csv(caminho_csv_stations)
        return df_rides, df_stations
    except FileNotFoundError:
        st.error("Erro: Arquivos CSV não encontrados. Verifique o caminho dos arquivos.")
        return None, None

# Definir o caminho relativo para os arquivos CSV
caminho_csv_rides = os.path.join('datasets', 'df_rides.csv')
caminho_csv_stations = os.path.join('datasets', 'df_stations.csv')

# Carregar os DataFrames
df_rides, df_stations = carregar_dados(caminho_csv_rides, caminho_csv_stations)

if df_rides is not None and df_stations is not None:
    # Verificar as colunas e dados
    st.write("Colunas de df_rides:", df_rides.columns)
    st.write("Colunas de df_stations:", df_stations.columns)

    # Prepara a coluna de hora (extraímos a hora da data da viagem)
    df_rides['hour'] = pd.to_datetime(df_rides['ride_date']).dt.hour

    # Contar o número de viagens por estação de partida (station_start) e hora
    station_counts = df_rides.groupby(['station_start', 'hour']).size().reset_index(name='bikes_out')

    # Realizar o merge com a tabela df_stations para adicionar informações de lat, lon
    merged_data = pd.merge(station_counts, df_stations, how='left', left_on='station_start', right_on='station')

    # Adicionar um título à página
    st.title("Previsão de Bicicletas Disponíveis por Estação e Hora")

    # Adicionar a barra lateral para seleção de múltiplas estações
    stations_selected = st.sidebar.multiselect(
        "Selecione as estações para visualizar:",
        options=merged_data['station_start'].unique(),
        default=[]
    )

    # Filtrar dados com base na seleção
    filtered_data = merged_data if not stations_selected else merged_data[merged_data['station_start'].isin(stations_selected)]

    # Previsão de bicicletas disponíveis
    st.sidebar.header("Previsão de Bicicletas Disponíveis")

    # Seleção de estação e horário
    station_for_forecast = st.sidebar.selectbox(
        "Escolha uma estação:",
        options=stations_selected if stations_selected else ["Selecione uma estação..."],
        disabled=not stations_selected
    )

    # Removendo o valor atual e permitindo que o usuário escolha uma hora
    time_input = st.sidebar.time_input("Escolha o horário para previsão:")

    # Treinar o modelo para prever bicicletas disponíveis
    label_encoder = LabelEncoder()
    merged_data['station_start_encoded'] = label_encoder.fit_transform(merged_data['station_start'])

    X = merged_data[['hour', 'station_start_encoded']]  # Usando hora e estação codificada
    y = merged_data['bikes_out']  # Número de bicicletas que saíram

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)

    # Função de previsão
    def forecast_bikes(station_for_forecast, time_input):
        if station_for_forecast:
            hour = time_input.hour
            minute = time_input.minute

            # Codificar a estação selecionada
            station_encoded = label_encoder.transform([station_for_forecast])[0]

            # Fazer a previsão com base na hora e estação
            predicted_bikes_out = model.predict([[hour, station_encoded]])

            # Formatar a mensagem para incluir hora e minuto
            bikes_available_message = f"Estimativa de bicicletas disponíveis em **{station_for_forecast}** às **{hour:02d}:{minute:02d}** é de **{int(predicted_bikes_out[0])}** bicicletas retiradas."
        else:
            bikes_available_message = "Estação não encontrada para a previsão."

        return bikes_available_message

    # Exibir o mapa com as coordenadas das estações filtradas
    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=filtered_data['lat'].mean(),
                longitude=filtered_data['lon'].mean(),
                zoom=12,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=filtered_data,
                    get_position="[lon, lat]",
                    radius=200,
                    elevation_scale=4,
                    elevation_range=[0, 1000],
                    pickable=True,
                    extruded=True,
                    get_fill_color="[bikes_out * 2, 30, 0]",
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=filtered_data,
                    get_position="[lon, lat]",
                    get_color="[200, 30, 0, 160]",
                    get_radius=200,
                ),
            ],
        )
    )

    # Chamar a função de previsão de bicicletas
    bikes_available_message = forecast_bikes(station_for_forecast, time_input)

    # Exibir a mensagem abaixo do mapa
    if bikes_available_message:
        st.write(bikes_available_message)
    else:
        st.error("Os dados não foram carregados corretamente.")
