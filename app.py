import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from recommendation import load_data, train_word2vec, recommend_songs

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/recommend', methods=['GET'])
def recommend():
    """ Rota para gerar recomendações de músicas para um usuário """
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    users_df, songs_df = load_data()
    if users_df.empty or songs_df.empty:
        return jsonify({"error": "Error loading data"}), 500

    word2vec_model = train_word2vec(songs_df)
    recommendations = recommend_songs(user_id, users_df, songs_df, word2vec_model)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
