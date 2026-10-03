import logging
import pandas as pd
import requests
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

users_api_url = "https://starting-music.onrender.com/user"
songs_api_url = "https://starting-music.onrender.com/music"


def get_data(url):
    """Busca os dados da API"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao buscar dados da URL {url}: {e}")
        return {}


def load_data():
    """Carrega e formata os dados"""
    users_data = get_data(users_api_url)
    songs_data = get_data(songs_api_url)

    users_df = pd.DataFrame(users_data.get('user', []))
    songs_df = pd.DataFrame(songs_data.get('songs', []))

    for df, col in [(users_df, 'gostei'), (songs_df, 'tags'), (songs_df, 'playlist')]:
        if col in df:
            df[col] = df[col].apply(
                lambda x: [item.get('nome', '') for item in x] if isinstance(x, list) else []
            )

    return users_df, songs_df


def train_word2vec(songs_df):
    """Treina o modelo Word2Vec"""

    sentences = []

    for _, row in songs_df.iterrows():

        tokens = []

        tokens += row.get("tags", [])
        tokens += row.get("playlist", [])

        if "nome" in row:
            tokens.append(row["nome"].lower())

        if tokens:
            sentences.append(tokens)

    if not sentences:
        return None

    model = Word2Vec(
        sentences,
        vector_size=128,
        window=8,
        min_count=1,
        sg=1,
        epochs=50,
        workers=4
    )

    return model


def get_song_vector(tokens, model):
    """Gera vetor médio baseado nas tags"""

    vectors = [model.wv[word] for word in tokens if word in model.wv]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


def build_song_vectors(songs_df, model):
    """Cria vetor para todas músicas"""

    vectors = []

    for _, row in songs_df.iterrows():

        tokens = []
        tokens += row.get("tags", [])
        tokens += row.get("playlist", [])

        vectors.append(get_song_vector(tokens, model))

    songs_df["vector"] = vectors

    return songs_df


def get_user_vector(liked_list, songs_df):

    liked_vectors = []

    for song in liked_list:

        row = songs_df[songs_df["nome"] == song]

        if not row.empty:
            liked_vectors.append(row.iloc[0]["vector"])

    if not liked_vectors:
        return None

    return np.mean(liked_vectors, axis=0)


def recommend_songs(user_id, users_df, songs_df, model):

    try:
        user_id = int(user_id)
    except ValueError:
        return {"error": "Invalid User ID"}

    user_data = users_df[users_df["id"] == user_id]

    if user_data.empty:
        return {"error": "User not found"}

    liked_list = user_data.iloc[0].get("gostei", [])

    if not liked_list:
        return {"songs": songs_df.sample(min(10, len(songs_df))).to_dict(orient="records")}

    user_vector = get_user_vector(liked_list, songs_df)

    if user_vector is None:
        return {"songs": songs_df.sample(min(10, len(songs_df))).to_dict(orient="records")}

    song_vectors = np.vstack(songs_df["vector"])

    similarity_scores = cosine_similarity(song_vectors, [user_vector]).flatten()

    songs_df["score"] = similarity_scores

    # remover músicas já curtidas
    candidates = songs_df[~songs_df["nome"].isin(liked_list)].copy()

    liked_artists = songs_df[songs_df["nome"].isin(liked_list)]["artista"].unique()

    candidates["artist_penalty"] = candidates["artista"].apply(
        lambda a: 0.8 if a in liked_artists else 1.0
    )

    candidates["final_score"] = candidates["score"] * candidates["artist_penalty"]

    ranked = candidates.sort_values(by="final_score", ascending=False)

    exploration_size = 3

    exploration = candidates.sample(min(exploration_size, len(candidates)))

    exploitation = ranked.head(max(0, 10 - exploration_size))

    final = pd.concat([exploitation, exploration]).drop_duplicates(subset=["nome"])

    return {"songs": final.drop(columns=["vector"]).to_dict(orient="records")}
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Carrega dados
    users_df, songs_df = load_data()

    if users_df.empty or songs_df.empty:
        print("Erro ao carregar dados das APIs")
        exit()

    # Treina modelo
    model = train_word2vec(songs_df)

    if model is None:
        print("Erro ao treinar Word2Vec")
        exit()

    # Cria vetores das músicas
    songs_df = build_song_vectors(songs_df, model)

    # Teste de recomendação
    user_id = 1

    recommendations = recommend_songs(user_id, users_df, songs_df, model)

    print("\nRecomendações geradas:\n")

    for song in recommendations.get("songs", []):
        print(f"- {song.get('nome')} | {song.get('artista')}")