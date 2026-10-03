from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

from recommendation import load_data, train_word2vec, build_song_vectors, recommend_songs

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Carrega e prepara tudo apenas uma vez ao iniciar o servidor
users_df, songs_df = load_data()
word2vec_model = train_word2vec(songs_df)

if word2vec_model is not None:
    songs_df = build_song_vectors(songs_df, word2vec_model)
else:
    logging.error("Falha ao treinar o modelo Word2Vec")

@app.route('/recommend', methods=['GET'])
def recommend():
    """Rota para gerar recomendações de músicas para um usuário"""

    user_id = request.args.get('user_id')

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    recommendations = recommend_songs(user_id, users_df, songs_df, word2vec_model)

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)