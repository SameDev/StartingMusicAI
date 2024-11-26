import logging
from flask import Flask, request, jsonify
import pandas as pd
import requests
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS  # Importar CORS

app = Flask(__name__)

# Ativar o CORS
CORS(app)  # Isso permite que qualquer origem acesse a API. Se quiser restringir, passe o parâmetro `origins=['http://dominio.com']`.

# Configuração do logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# URLs das APIs
users_api_url = "https://starting-music.onrender.com/user"
songs_api_url = "https://starting-music.onrender.com/music"

# Função para obter dados das APIs
def get_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        logging.error("Erro ao obter dados da URL %s: %s", url, e)
        return {}
    
def extract_name(item_list):
    return [item.get('nome', "") for item in item_list] if isinstance(item_list, list) else []

# Função para carregar e processar os dados
def load_data():
    try:
        users_data = get_data(users_api_url)
        songs_data = get_data(songs_api_url)

        # Criar DataFrames a partir dos dados
        users_df = pd.DataFrame(users_data.get('user', []))
        songs_df = pd.DataFrame(songs_data.get('songs', []))

        # Validar e corrigir colunas esperadas em usuários
        if not users_df.empty:
            users_df = validate_and_fill_columns(users_df, ['id', 'nome', 'tags', 'gostei', 'playlist'])
            if 'tags' in users_df.columns:
                users_df = expand_json_column(users_df, 'tags')
            if 'gostei' in users_df.columns:
                users_df['gostei'] = users_df['gostei'].apply(lambda x: extract_name(x))
            if 'playlist' in users_df.columns:
                users_df['playlist'] = users_df['playlist'].apply(lambda x: extract_name(x))

        # Validar e corrigir colunas esperadas em músicas
        if not songs_df.empty:
            songs_df = validate_and_fill_columns(songs_df, [
                'id', 'nome', 'artista', 'url', 'duracao', 'data_lanc', 
                'image_url', 'albumId', 'tags', 'artistaId', 'playlist', 'userLiked'
            ])
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

# Função para validar e preencher colunas ausentes
def validate_and_fill_columns(df, expected_columns):
    for column in expected_columns:
        if column not in df.columns:
            logging.warning("Coluna '%s' está ausente no DataFrame. Preenchendo com valores vazios.", column)
            df[column] = ""
    return df

# Função para expandir colunas JSON
def expand_json_column(df, column_name):
    try:
        expanded_df = pd.json_normalize(df[column_name])
        return pd.concat([df.drop(columns=[column_name]), expanded_df], axis=1)
    except Exception as e:
        logging.error("Erro ao expandir a coluna JSON '%s': %s", column_name, e)
        return df
    
def ensure_strings(data):
    if isinstance(data, dict):
        return {str(k): ensure_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [ensure_strings(item) for item in data]
    elif data is None:
        return ""
    elif isinstance(data, (int, float, pd.Timestamp)):
        return int(data) if isinstance(data, (int, pd.Timestamp)) else int(data)
    elif isinstance(data, pd.Series):
        return data.apply(ensure_strings).tolist()  # Para Series do Pandas
    return str(data) 



# Função para formatar as músicas no formato desejado
def format_song(song):
    tags = song.get("tags", [])
    if isinstance(tags, str):  # Caso tags seja uma string (erro comum em serialização)
        tags = [{"id": tags.find(id), "nome": tags}]
    elif not isinstance(tags, list):  # Se tags não for uma lista, inicialize como vazia
        tags = []

    # Corrigir inconsistências nos objetos de tags
    formatted_tags = []
    for tag in tags:
        if isinstance(tag, dict):  # Garantir que é um dicionário
            formatted_tags.append({
                "id": tag.get("id", ""),
                "nome": tag.get("nome", "")
            })
        else:  # Caso contrário, crie um objeto padrão
            formatted_tags.append({"id": "", "nome": str(tag)})

    return {
        "id": song.get("id", ""),
        "nome": song.get("nome", ""),
        "artista": song.get("artista", ""),
        "url": song.get("url", ""),
        "duracao": song.get("duracao", ""),
        "data_lanc": song.get("data_lanc", ""),
        "image_url": song.get("image_url", ""),
        "albumId": song.get("albumId", ""),
        "tags": formatted_tags,
        "artistaId": song.get("artistaId", []),
        "playlist": song.get("playlist", []),
        "usuarioGostou": song.get("userLiked", [])
    }



# Função de recomendação
def recommend_songs(user_id, users_df, songs_df):
    try:
        user_id = int(user_id)
    except ValueError:
        return {"error": "User ID must be an integer"}

    user_data = users_df[users_df['id'] == user_id]
    if user_data.empty:
        return {"error": "User not found"}

    liked_list = user_data['gostei'].values[0]
    if not liked_list:
        return {"songs": songs_df.head(10).apply(format_song, axis=1).tolist()}

    liked_songs = set(liked_list)
    songs_to_recommend = set(songs_df['nome']) - liked_songs

    # Criar uma coluna 'information' para cálculos
    songs_df['information'] = (
        songs_df['nome'] + ' ' +
        songs_df['artista'] + ' ' +
        songs_df['playlist'].apply(lambda x: ' '.join(x))
    )

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(songs_df['information'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    similar_songs = {}
    for i, row in enumerate(cosine_similarities):
        similar_songs_list = [
            songs_df.iloc[j] for j in row.argsort()[-6:-1][::-1] if i != j
        ]
        similar_songs[songs_df.iloc[i]['nome']] = similar_songs_list

    recommendations = []
    for song in songs_to_recommend:
        if song in similar_songs:
            recommendations.extend(similar_songs[song])

    recommendations = recommendations[:10]
    recommendations = [format_song(song) for song in recommendations]

    return {"songs": recommendations}

# Endpoint de recomendação
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    users_df, songs_df = load_data()
    if users_df.empty or songs_df.empty:
        return jsonify({"error": "Error loading data"}), 500

    recommendations = recommend_songs(user_id, users_df, songs_df)

    # Garantir que todos os valores sejam serializáveis
    recommendations = ensure_strings(recommendations)

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run()
