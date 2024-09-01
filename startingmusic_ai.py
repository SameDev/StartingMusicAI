import logging
from flask import Flask, request, jsonify
import pandas as pd
import requests
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import json

app = Flask(__name__)

# Configuração do logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Função para obter dados das APIs
def get_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Verifica se houve algum erro HTTP
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        logging.error("Erro ao obter dados da URL %s: %s", url, e)
        return {}

# URLs das APIs
users_api_url = "https://starting-music.onrender.com/user"
songs_api_url = "https://starting-music.onrender.com/music"

# Função para extrair o nome de itens em uma lista de dicionários
def extract_name(item_list):
    return [item['name'] for item in item_list] if isinstance(item_list, list) else []

# Função para carregar e processar os dados
# Carrega os dados dos usuários e músicas a partir das APIs e os processa
def load_data():
    try:
        users_data = get_data(users_api_url)
        songs_data = get_data(songs_api_url)

        # Criar DataFrames a partir dos dados
        users_df = pd.DataFrame(users_data.get('user', []))
        songs_df = pd.DataFrame(songs_data.get('songs', []))

        # Normaliza e processa os dados dos usuários
        if 'tags' in users_df.columns:
            users_df_normalized = pd.json_normalize(users_df['tags'])
            users_df = pd.concat([users_df.drop(columns=['tags']), users_df_normalized], axis=1)
        users_df = users_df.fillna(value="")

        if 'likes' in users_df.columns:
            users_df['likes'] = users_df['likes'].apply(lambda x: extract_name(x))
        if 'playlist' in users_df.columns:
            users_df['playlist'] = users_df['playlist'].apply(lambda x: extract_name(x))

        # Normaliza e processa os dados das músicas
        if 'tags' in songs_df.columns:
            songs_df_normalized = pd.json_normalize(songs_df['tags'])
            songs_df = pd.concat([songs_df.drop(columns=['tags']), songs_df_normalized], axis=1)
        songs_df = songs_df.fillna(value="")

        if 'tags' in songs_df.columns:
            songs_df['tags'] = songs_df['tags'].apply(lambda x: extract_name(x))
        if 'playlist' in songs_df.columns:
            songs_df['playlist'] = songs_df['playlist'].apply(lambda x: extract_name(x))
        if 'userLiked' in songs_df.columns:
            songs_df['userLiked'] = songs_df['userLiked'].apply(lambda x: extract_name(x))

        return users_df, songs_df

    except Exception as e:
        logging.error("Erro ao carregar e processar os dados: %s", e)
        return pd.DataFrame(), pd.DataFrame()

# Função para garantir que todos os valores são strings
# Isso evita problemas com tipos de dados mistos
def ensure_strings(data):
    if isinstance(data, dict):
        return {str(k): ensure_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [ensure_strings(item) for item in data]
    elif data is None:
        return ""
    elif isinstance(data, (int, float)):
        return str(data)
    return data

# Função para recomendar músicas
# Essa função utiliza o ID do usuário para buscar as músicas que ele gostou
# e recomenda músicas semelhantes com base no conteúdo e popularidade
def recommend_songs(user_id, users_df, songs_df):
    try:
        user_id = int(user_id)
    except ValueError:
        return {"error": "User ID must be an integer"}

    user_data = users_df[users_df['id'] == user_id]
    
    if user_data.empty:
        return {"error": "User not found"}

    # Verifica se o usuário curtiu alguma música
    liked_list = user_data['likes'].values[0]
    if not liked_list:
        # Se não tiver curtido nenhuma música, recomenda as 10 músicas mais populares ou recentes
        recommendations = songs_df.head(10).to_dict(orient='records')
        return {"songs": ensure_strings(recommendations)}
    
    liked_songs = set(liked_list)
    songs_to_recommend = set(songs_df['name']) - liked_songs

    songs_df['information'] = songs_df['name'] + ' ' + songs_df['artist'] + ' ' + songs_df['playlist'].apply(lambda x: ' '.join(x))

    # Aplicar a vetorização TF-IDF nas informações das músicas
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(songs_df['information'])

    # Calcula a similaridade de cosseno entre as músicas
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    similar_songs = {}
    for i, row in enumerate(cosine_similarities):
        similar_songs_list = [songs_df.iloc[j] for j in row.argsort()[-6:-1][::-1] if i != j]
        similar_songs[songs_df.iloc[i]['name']] = similar_songs_list

    recommendations = []
    for song in songs_to_recommend:
        if song in similar_songs:
            recommendations.extend(similar_songs[song])
    
    # Limita o número de recomendações a 10
    recommendations = recommendations[:10]
    recommendations = [song.to_dict() for song in recommendations]

    # Garantir que todos os valores são strings
    recommendations = [ensure_strings(song) for song in recommendations]

    # Verifica o tipo e conteúdo das recomendações
    logging.debug("Tipo de recommendations: %s", type(recommendations))
    logging.debug("Conteúdo de recommendations: %s", recommendations)

    return {"songs": recommendations}

# Rota para recomendar músicas
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    users_df, songs_df = load_data()
    if users_df.empty or songs_df.empty:
        return jsonify({"error": "Error loading data"}), 500

    recommendations = recommend_songs(user_id, users_df, songs_df)

    # Verifica o tipo de dados retornados
    if not isinstance(recommendations, dict):
        return jsonify({"error": "Returned data is not in the correct format"}), 500

    # Loga o tipo e conteúdo da resposta antes de enviar
    logging.debug("Final response: %s", recommendations)

    return jsonify(ensure_strings(recommendations))


if __name__ == '__main__':
    app.run(debug=True)
