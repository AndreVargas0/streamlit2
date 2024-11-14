import streamlit as st
import pandas as pd
import pydeck as pdk
import os
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

# Função para carregar os dados com cache
@st.cache_data
def carregar_dados(caminho_csv, caminho_estacoes):
    try:
        df = pd.read_csv(caminho_csv)
        df_stations = pd.read_csv(caminho_estacoes)
        return df, df_stations
    except FileNotFoundError:
        st.error("Erro: Arquivo CSV não encontrado. Verifique o caminho dos arquivos.")
        return None, None

# Definir o caminho relativo para os arquivos CSV
caminho_csv = os.path.join('datasets', 'df_rides.csv')
caminho_estacoes = os.path.join('datasets', 'df_stations.csv')

# Carregar os DataFrames
df, df_stations = carregar_dados(caminho_csv, caminho_estacoes)

if df is not None and df_stations is not None:
    # Layout e CSS personalizados
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #446f57;
        }
        .resultado-previsao {
            font-size: 24px;
            color: #ffcc00;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Exibe o logo na barra lateral
    imagemlogo = os.path.abspath('ciclolazer.jpg')
    st.sidebar.image(imagemlogo, width=300)

    # Configurar colunas de data e hora
    df['ride_date'] = pd.to_datetime(df['ride_date'])
    df['time_start'] = pd.to_datetime(df['time_start'])
    df['time_start_15min'] = df['time_start'].dt.floor('15T')  # Intervalos de 15 minutos

    # Criar recursos adicionais
    df['hour_of_day'] = df['time_start'].dt.hour
    df['minute_of_hour'] = df['time_start'].dt.minute
    df['day_of_week'] = df['time_start'].dt.weekday
    df['user_age'] = (pd.to_datetime('today') - pd.to_datetime(df['user_birthdate'])).dt.days // 365

    # Contar o número de viagens por estação
    station_counts = df['station_start'].value_counts().reset_index()
    station_counts.columns = ['station', 'count']

    # Juntar as informações com as coordenadas das estações
    merged_data = pd.merge(station_counts, df_stations, on='station')

    # Adicionar barra lateral para seleção de estação
    stations_selected = st.sidebar.multiselect(
        "Selecione as estações para visualizar no mapa:",
        options=merged_data['station'].unique(),
        default=merged_data['station'].iloc[:1]  # Padrão: primeira estação
    )

    # Filtrar dados com base na seleção
    if stations_selected:
        filtered_data = merged_data[merged_data['station'].isin(stations_selected)]
    else:
        # Se não houver estação selecionada, mostra todas as estações
        filtered_data = merged_data

    # Previsão de bicicletas disponíveis
    st.sidebar.subheader("")

    # Criar o modelo LGBMRegressor
    df_rides_model = df[['user_gender', 'user_age', 'station_start', 'hour_of_day', 'minute_of_hour', 'day_of_week']]
    df_rides_model = pd.get_dummies(df_rides_model, drop_first=True)

    # Calcular bike_count
    bike_counts = df.groupby('time_start_15min').size().reset_index(name='bike_count')
    valid_indices = df_rides_model.index.intersection(bike_counts.index)
    X = df_rides_model.loc[valid_indices]
    y = bike_counts.loc[valid_indices, 'bike_count']

    # Normalizar e treinar o modelo
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Usando LGBMRegressor
    lgbm = LGBMRegressor(n_estimators=100, random_state=42)  # 100 árvores
    lgbm.fit(X_train, y_train)

    # Gerar combinações de estação, hora, minuto e dia da semana (a cada 15 minutos)
    stations = merged_data['station'].unique()
    hours = list(range(24))  # Horas de 0 a 23
    minutes = [0, 15, 30, 45]  # Intervalos de 15 minutos
    days_of_week = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    day_of_week_dict = {'Segunda': 0, 'Terça': 1, 'Quarta': 2, 'Quinta': 3, 'Sexta': 4, 'Sábado': 5, 'Domingo': 6}

    # Lista para armazenar as previsões
    previsoes = []

    # Gerar previsões para todas as combinações (15 minutos)
    for station in stations:
        for hour in hours:
            for minute in minutes:
                for day in days_of_week:
                    day_num = day_of_week_dict[day]
                    input_data = pd.DataFrame({
                        'user_gender': [0],  # Assumindo '0' para masculino ou '1' para feminino
                        'user_age': [25],  # Idade simulada
                        'station_start': [station],
                        'hour_of_day': [hour],
                        'minute_of_hour': [minute],
                        'day_of_week': [day_num]
                    })
                    input_data = pd.get_dummies(input_data, drop_first=True)
                    input_data = input_data.reindex(columns=X.columns, fill_value=0)
                    input_data_scaled = scaler.transform(input_data)
                    predicted_bikes = lgbm.predict(input_data_scaled)  # Usando o modelo LGBMRegressor
                    previsao_bicicletas = int(predicted_bikes[0] // 490)

                    # Adicionar a previsão à lista
                    previsoes.append({
                        'station': station,
                        'hour': hour,
                        'minute': minute,
                        'day_of_week': day,
                        'predicted_bikes': previsao_bicicletas
                    })

    # Criar um DataFrame com as previsões
    previsoes_df = pd.DataFrame(previsoes)

    # Exportar para um CSV
    output_csv_path = 'previsoes_bicicletas_15min_lgbm.csv'
    previsoes_df.to_csv(output_csv_path, index=False)

    # Mensagem de sucesso
    st.success(f"As previsões foram salvas com sucesso em {output_csv_path}")

    # Exibir o mapa com PyDeck
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
                    "ScatterplotLayer",
                    data=filtered_data,
                    get_position="[lon, lat]",
                    get_color="[200, 30, 0, 160]",
                    get_radius=200,
                    pickable=True,
                    tooltip=True
                ),
            ],
        ), use_container_width=True  # Ajusta a largura do mapa para ocupar todo o container
    )
else:
    st.error("Os dados não foram carregados corretamente.")
