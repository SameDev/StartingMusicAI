import logging
from flask import Flask, request, jsonify
import pandas as pd
import requests
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS
CORS(app)

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# URLs of the APIs
users_api_url = "https://starting-music.onrender.com/user"
songs_api_url = "https://starting-music.onrender.com/music"

# Function to fetch data from APIs
def get_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from URL {url}: {e}")
        return {}

# Function to load and process data
def load_data():
    try:
        users_data = get_data(users_api_url)
        songs_data = get_data(songs_api_url)

        users_df = pd.DataFrame(users_data.get('user', [])) if 'user' in users_data else pd.DataFrame()
        songs_df = pd.DataFrame(songs_data.get('songs', [])) if 'songs' in songs_data else pd.DataFrame()

        if not users_df.empty:
            if 'gostei' in users_df:
                users_df['gostei'] = users_df['gostei'].apply(lambda x: [item.get('nome', '') for item in x] if isinstance(x, list) else [])
            if 'tags' in users_df:
                users_df['tags'] = users_df['tags'].apply(lambda x: [item.get('nome', '') for item in x] if isinstance(x, list) else [])
            if 'playlist' in users_df:
                users_df['playlist'] = users_df['playlist'].apply(lambda x: [item.get('nome', '') for item in x] if isinstance(x, list) else [])

        if not songs_df.empty:
            if 'tags' in songs_df:
                songs_df['tags'] = songs_df['tags'].apply(lambda x: [item.get('nome', '') for item in x] if isinstance(x, list) else [])
            if 'playlist' in songs_df:
                songs_df['playlist'] = songs_df['playlist'].apply(lambda x: [item.get('nome', '') for item in x] if isinstance(x, list) else [])
            if 'userLiked' in songs_df:
                songs_df['userLiked'] = songs_df['userLiked'].apply(lambda x: [item.get('nome', '') for item in x] if isinstance(x, list) else [])

        return users_df, songs_df
    except Exception as e:
        logging.error(f"Error loading and processing data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Function to format songs
def format_song(song):
    return {
        "id": song.get("id", ""),
        "nome": song.get("nome", ""),
        "artista": song.get("artista", ""),
        "url": song.get("url", ""),
        "duracao": song.get("duracao", ""),
        "data_lanc": song.get("data_lanc", ""),
        "image_url": song.get("image_url", ""),
        "tags": song.get("tags", []),
        "playlist": song.get("playlist", []),
        "usuarioGostou": song.get("userLiked", [])
    }

# Function to recommend songs
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
    
    unliked_songs_df = songs_df[~songs_df['nome'].isin(liked_list)].reset_index(drop=True)

    if unliked_songs_df.empty:
        return {"message": "No unliked songs found for this user."}

    songs_df['information'] = (
        songs_df['nome'].fillna('') + ' ' +
        songs_df['artista'].fillna('') + ' ' +
        songs_df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '').fillna('')
    )

    # Vectorize with TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(songs_df['information'])

    liked_song_indices = songs_df[songs_df['nome'].isin(liked_list)].index.tolist()

    if liked_song_indices:
        liked_tfidf = tfidf_matrix[liked_song_indices]
        unliked_tfidf = tfidf_matrix[len(liked_list):]
        cosine_similarities = linear_kernel(liked_tfidf, unliked_tfidf)
        similar_indices = cosine_similarities.mean(axis=0).argsort()[::-1]

        # Ensure indices are within bounds
        similar_indices = [i for i in similar_indices if i < len(unliked_songs_df)]
        num_similar_songs = min(len(similar_indices), 10)
        recommendations = unliked_songs_df.iloc[similar_indices[:num_similar_songs]].apply(format_song, axis=1).tolist()
    else:
        recommendations = songs_df.head(10).apply(format_song, axis=1).tolist()

    return {"songs": recommendations}

# Recommendation endpoint
@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    users_df, songs_df = load_data()
    if users_df.empty or songs_df.empty:
        return jsonify({"error": "Error loading data"}), 500

    recommendations = recommend_songs(user_id, users_df, songs_df)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run()
