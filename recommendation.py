import logging
import pandas as pd
import requests
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

users_api_url = "https://starting-music.onrender.com/user"
songs_api_url = "https://starting-music.onrender.com/music"

def get_data(url):
    """ Busca os dados da API e retorna um dicionário JSON """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao buscar dados da URL {url}: {e}")
        return {}

def load_data():
    """ Carrega os dados dos usuários e músicas da API e formata corretamente """
    users_data = get_data(users_api_url)
    songs_data = get_data(songs_api_url)

    users_df = pd.DataFrame(users_data.get('user', []))
    songs_df = pd.DataFrame(songs_data.get('songs', []))

    for df, col in [(users_df, 'gostei'), (songs_df, 'tags'), (songs_df, 'playlist')]:
        if col in df:
            df[col] = df[col].apply(lambda x: [item.get('nome', '') for item in x] if isinstance(x, list) else [])
    
    return users_df, songs_df

def train_word2vec(songs_df):
    """ Treina um modelo Word2Vec com base nos dados das músicas """
    sentences = songs_df['tags'].tolist() + songs_df['playlist'].tolist()
    sentences = [tags for tags in sentences if tags]
    
    if not sentences:
        return None
    
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_song_vector(song, model):
    """ Obtém o vetor médio de uma música com base no modelo Word2Vec """
    vectors = [model.wv[word] for word in song if word in model.wv]
    
    if not vectors:
        return np.zeros(model.vector_size).tolist()  # Retorna vetor zerado
    
    return np.nan_to_num(np.mean(vectors, axis=0)).tolist()  # Substitui NaN por 0

def get_user_embedding(user_songs, songs_df, word2vec_model):
    """ Cria um vetor representando o usuário com base nas músicas curtidas """
    song_vectors = [get_song_vector(song, word2vec_model) for song in user_songs if song in songs_df['nome'].values]
    return np.mean(song_vectors, axis=0) if song_vectors else np.zeros(100)

def recommend_songs(user_id, users_df, songs_df, word2vec_model):
    """ Gera recomendações de músicas para um usuário específico """
    try:
        user_id = int(user_id)
    except ValueError:
        return {"error": "Invalid User ID"}
    
    user_data = users_df[users_df['id'] == user_id]
    if user_data.empty:
        return {"error": "User not found"}
    
    liked_list = user_data['gostei'].values[0] if 'gostei' in user_data else []
    if not liked_list:
        return {"songs": songs_df.sample(10).to_dict(orient='records')}
    
    # Criar vetores das músicas caso ainda não tenha sido feito
    if 'vector' not in songs_df:
        songs_df['vector'] = songs_df['tags'].apply(lambda x: get_song_vector(x, word2vec_model))
    
    liked_vectors = np.array([get_song_vector(song, word2vec_model) for song in liked_list if song in songs_df['nome'].values])
    
    if liked_vectors.size == 0:
        return {"songs": songs_df.sample(10).to_dict(orient='records')}
    
    # Calcula a similaridade corretamente
    song_vectors = np.vstack(songs_df['vector'])
    similarity_scores = cosine_similarity(song_vectors, liked_vectors).mean(axis=1)

    # Garante que a coluna 'score' existe
    songs_df['score'] = similarity_scores if similarity_scores.size > 0 else np.zeros(len(songs_df))

    ranked_songs = songs_df.sort_values(by="score", ascending=False).head(10)
    ranked_songs = ranked_songs.drop(columns=['vector'])


    # TODO: Corrigir e treinar o Word2Vec para melhores resultados através do Score.
    print(songs_df[['nome', 'tags']].head())  # Verifica se as tags estão carregadas corretamente
    print(word2vec_model.wv.index_to_key[:10])  # Lista algumas palavras conhecidas pelo modelo
    print(get_song_vector(['rock', 'pop'], word2vec_model))  # Testa se retorna um vetor válido

    return {"songs": ranked_songs.to_dict(orient='records')}
