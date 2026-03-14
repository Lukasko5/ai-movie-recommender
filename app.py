
import streamlit as st
import pandas as pd
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process


df = pd.read_csv("tmdb_5000_movies.csv")


df = df[["title", "overview", "genres", "keywords", "vote_average", "runtime"]]


df["overview"] = df["overview"].fillna("")
df["genres"] = df["genres"].fillna("")
df["keywords"] = df["keywords"].fillna("")
df["vote_average"] = df["vote_average"].fillna(0)
df["runtime"] = df["runtime"].fillna(0)


df["combined"] = (
    df["overview"] + " " + df["overview"] + " " + df["overview"] + " " +
    df["keywords"] + " " + df["keywords"] + " " +
    df["genres"]
)

df["combined"] = df["combined"].str.lower()




def get_genres(genre_string):
    if not genre_string:
        return []
    genres = json.loads(genre_string)
    return [g["name"] for g in genres]

df["genre_names"] = df["genres"].apply(get_genres)

@st.cache_resource
def load_embeddings():

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(df["combined"].tolist(), show_progress_bar=True)

    cosine_sim = cosine_similarity(embeddings, embeddings)

    return cosine_sim

cosine_sim = load_embeddings()


indices = pd.Series(df.index, index=df["title"]).drop_duplicates()


titles = df["title"].tolist()

def find_closest_title(user_input):
    match = process.extractOne(user_input, titles)

    if match is None:
        return None

    best_match = match[0]
    score = match[1]

    if score >= 70:
        return best_match
    else:
        return None




def recommend_movies(title, num_recommendations=5):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]

    recommendations = []

    for i in movie_indices:

        movie_title = df["title"].iloc[i]
        movie_overview = df["overview"].iloc[i]

        movie_rating = df["vote_average"].iloc[i]
        movie_runtime = df["runtime"].iloc[i]
        movie_genres = df["genre_names"].iloc[i]

        recommendations.append((movie_title, movie_overview, movie_rating, movie_runtime, movie_genres))

    return recommendations




def get_movie_poster(title):

    API_KEY = st.secrets["TMDB_API_KEY"]

    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={title}"

    response = requests.get(url)
    data = response.json()

    if data["results"]:
        poster_path = data["results"][0]["poster_path"]

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    return None




st.title("🎬 AI Movie Recommender")

user_movie = st.text_input("Enter a movie title")

num_recommendations = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Recommend Movies"):

    closest = find_closest_title(user_movie)

    if closest is None:
        st.error("Movie not found.")
    else:
        st.success(f"Using: {closest}")

        results = recommend_movies(closest, num_recommendations)

        for i, (movie, overview, rating, runtime, genres) in enumerate(results, start=1):

            poster = get_movie_poster(movie)

            st.subheader(f"{i}. {movie}")

            col1, col2 = st.columns([1, 2])

            with col1:
                if poster:
                    st.image(poster, width=200)

            with col2:

                genre_text = " • ".join(genres)
                st.write(f"🎭 {genre_text}")

                st.write(f"⭐ Rating: {round(rating, 1)} / 10")

                runtime = int(runtime)
                hours = runtime // 60
                minutes = runtime % 60
                st.write(f"⏱ Duration: {hours}h {minutes}m")

                st.write(overview[:400] + "...")

            st.divider()

