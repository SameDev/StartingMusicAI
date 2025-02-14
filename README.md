# Sistema de Recomendação de Músicas - Starting Music

Este projeto implementa um sistema de recomendação de músicas baseado em inteligência artificial utilizando o modelo **Word2Vec** para gerar sugestões personalizadas com base nas preferências dos usuários. O sistema acessa dados sobre usuários e músicas através de APIs e utiliza a similaridade entre músicas para recomendar as mais relevantes para cada usuário.

## Como Funciona

### Fluxo de Recomendação

1. **Coleta de Dados**: O sistema acessa duas APIs externas:
   - **Usuários**: Contém informações sobre os usuários, incluindo quais músicas eles gostaram.
   - **Músicas**: Contém dados sobre músicas, incluindo tags e playlists associadas.

2. **Pré-processamento**: A partir dos dados coletados, as tags e playlists das músicas são processadas para formar listas de palavras que descrevem cada música. Além disso, as músicas que os usuários gostaram são coletadas.

3. **Treinamento do Modelo Word2Vec**: O modelo Word2Vec é treinado com base nas tags e playlists das músicas. O Word2Vec é um algoritmo de aprendizagem de máquina que transforma palavras (tags/descrições) em vetores numéricos, permitindo calcular a similaridade entre elas.

4. **Geração de Vetores para Músicas**: Para cada música, um vetor é gerado utilizando o modelo Word2Vec, representando a música com base em suas tags e playlists.

5. **Cálculo de Similaridade**: Quando um usuário solicita recomendações, o sistema cria um vetor de preferências para o usuário com base nas músicas que ele já gostou. A similaridade entre as músicas preferidas e todas as músicas do catálogo é calculada usando a **similaridade de cosseno**.

6. **Recomendações**: As músicas mais semelhantes às preferências do usuário são recomendadas, levando em conta a similaridade entre as músicas.

### Componentes Principais

- **`load_data()`**: Carrega os dados dos usuários e das músicas a partir das APIs externas e organiza-os em DataFrames do Pandas.
- **`train_word2vec()`**: Treina o modelo Word2Vec a partir das tags e playlists das músicas.
- **`get_song_vector()`**: Converte as tags de uma música em um vetor numérico usando o modelo Word2Vec.
- **`get_user_embedding()`**: Cria um vetor de preferências para o usuário baseado nas músicas que ele gostou.
- **`recommend_songs()`**: Gera as recomendações para o usuário com base no cálculo da similaridade de cosseno entre os vetores das músicas.

## Como Usar

### Requisitos

- Python 3.7 ou superior
- Flask
- Flask-CORS
- Pandas
- Numpy
- Gensim
- scikit-learn
- Requests

### Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-repositorio/starting-music.git
   cd starting-music
   ```

2. Instale as dependências:
  ```bash
  pip install -r requirements.txt
  ```
3. Inicie o servidor Flask:
  ```bash
  python app.py
  ```
4. O servidor estará rodando em http://localhost:5000. Agora você pode testar as recomendações.

### Endpoint para Recomendação
- **GET** /recommend: Gera recomendações de músicas para um usuário.
#### Parâmetros
- user_id: ID do usuário para o qual as recomendações serão geradas.

Exemplo de requisição:
```bash
curl "http://localhost:5000/recommend?user_id=1"
```
Resposta de exemplo:
```json
{
  "songs": [
    {
      "nome": "Música A",
      "tags": ["rock", "pop"],
      "score": 0.89
    },
    {
      "nome": "Música B",
      "tags": ["indie", "alternativo"],
      "score": 0.85
    }
  ]
}
```
- Erro (quando user_id não é fornecido ou inválido):
```json 
{
  "error": "User ID is required"
}
```

## Contributing
- If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the Affero General Public License (AGPL) v3.0 - see the [LICENSE](LICENSE) file for details.