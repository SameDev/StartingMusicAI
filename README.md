---
# Starting Music Recommendation System

## Overview
The Starting Music Recommendation System is an AI-powered recommendation engine designed to suggest songs to users based on their listening history and preferences. It is built using Python and Flask, leveraging the power of machine learning techniques to provide personalized music recommendations. The system is a core component of the Starting Music platform, which helps emerging artists showcase their work and enables listeners to discover new music tailored to their tastes.

## How It Works
The recommendation engine operates by analyzing user data and music metadata retrieved from external APIs. It uses a combination of content-based filtering and similarity calculations to generate recommendations.

## Data Collection
**User Data:** Information about users, including their liked songs, playlists, and tags, is fetched from the user API.
**Song Data:** Information about songs, including song name, artist, album, tags, and playlists, is fetched from the music API.

## Data Processing
1. Normalization and Structuring:

    - The user and song data are normalized and transformed into a format suitable for analysis. Nested data structures like tags and playlists are extracted and flattened.

2. Content-Based Filtering:

    - A TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer is applied to the song metadata to convert text information (such as song names, artists, and tags) into numerical vectors.
    - The system calculates the cosine similarity between songs based on these vectors, identifying which songs are similar to each other.

## Recommendation Logic
1. User Preferences:

    - When a user requests recommendations, the system checks their liked songs. If the user has liked any songs, the system searches for similar songs using the cosine similarity scores.

2. Fallback Recommendations:

    - If the user has not liked any songs yet, the system provides a fallback recommendation list, which includes the most popular or recently added songs.

3. Recommendation Output:

    - The system outputs a list of up to 10 song recommendations, ensuring they are in a user-friendly format. All output values are converted to strings to maintain consistency.

## API Endpoint

- GET /recommend: This endpoint receives a user_id as a query parameter and returns a JSON response containing recommended songs for the user.


### Example Response:


```json
{
  "songs": [
    {
      "id": 35,
      "name": "Amiga da Minha Mulher",
      "artist": "Seu Sacani",
      "url": "https://example.com/audio/amiga_da_minha_mulher",
      "duration": "",
      "release_date": "2024-02-20T00:00:00.000Z",
      "image_url": "https://example.com/images/amiga_da_minha_mulher",
      "albumId": 47,
      "tags": [
        {
          "id": 16,
          "name": "Acoustic"
        },
        {
          "id": 17,
          "name": "Samba/Pagode"
        }
      ],
      "artistId": [],
      "playlist": [],
      "userLiked": []
    }
  ]
}
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/starting-music-recommendation.git
cd starting-music-recommendation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask application:

```bash
python app.py
```

4. Access the recommendation endpoint:

- Navigate to ***http://localhost:5000/recommend?user_id=1*** to get recommendations for a user with user_id=1.

## Dependencies
- **Flask:** For building the web application.
- **Pandas:** For data manipulation and processing.
- **Requests:** For making API calls to fetch user and song data.
- **Scikit-learn:** For implementing TF-IDF vectorization and cosine similarity calculations.
- **Logging:** For tracking and debugging application behavior.

## Contributing
- If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License

This project is licensed under the Affero General Public License (AGPL) v3.0 - see the [LICENSE](LICENSE) file for details.
