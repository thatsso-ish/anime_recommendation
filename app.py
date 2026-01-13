import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader

# Load Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Helper to display anime cards
def display_anime_cards(recommendations):
    cols = st.columns(3)
    for idx, row in recommendations.iterrows():
        with cols[idx % 3]:
            # Handle different column names if necessary (distinguished by logic)
            name = row['name']
            rating = row.get('rating', row.get('predicted_rating', 'N/A'))
            # Genre might not be in the recommendations dataframe depending on proper merge
            # If not present, we can try to fetch it from the main anime df if available globally
            # For now, let's look for genre or default
            genre = row.get('genre', 'Genre N/A')
            type_ = row.get('type', 'TV')
            
            st.markdown(f"""
            <div class="anime-card">
                <div class="anime-title">{name}</div>
                <div style="color: #ccc; font-size: 0.8rem; margin-bottom: 0.5rem;">{genre}</div>
                <div class="anime-metric">
                    <span class="rating-badge">⭐ {rating}</span>
                    <span>{type_}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Load data
def load_data():
    anime = pd.read_csv('data/anime.csv')
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    ratings = pd.concat([train, test], ignore_index=True)
    return anime, ratings, train, test

anime, ratings, train, test = load_data()

# Extract unique types and genres
types = sorted(anime['type'].dropna().unique())
genres = set()
for genre_list in anime['genre'].dropna().str.split(', '):
    genres.update(genre_list)
genres = sorted(genres)

# Sidebar for navigation
st.sidebar.markdown('## 🧭 Navigation')
page = st.sidebar.radio("", ["About", "Recommendations"])

if page == "About":
    st.markdown('<div class="main-header"><h1>🎬 Anime Recommendation System</h1></div>', unsafe_allow_html=True)
    st.image('visuals/anime.png', use_column_width=True)
    st.markdown("""
    <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
    <h3>Welcome to your next obsession! 🌟</h3>
    <p>Discover new anime tailored just for you using our state-of-the-art recommendation algorithms.</p>
    <ul>
        <li><strong>🧠 Content-Based</strong>: Finds show similar to what you love.</li>
        <li><strong>🤝 Collaborative</strong>: Suggests hits based on community taste.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "Recommendations":
    st.markdown('<div class="main-header"><h1>🔍 Find Your Next Watch</h1></div>', unsafe_allow_html=True)

    # Algorithm selection
    algorithm = st.selectbox('Select Algorithm', ('Content-Based Filtering', 'Collaborative Filtering'))

    # Implement the chosen algorithm
    def content_based_filtering(anime):
        st.info("💡 **Content-Based Filtering**: Suggests anime based on similarity of genres and types.")
        
        # User can filter by types
        selected_types = st.multiselect('Select types to filter by', types, default=types)
        
        # Filter anime by selected types
        def filter_by_types(anime, selected_types):
            return anime[anime['type'].isin(selected_types)]
        
        filtered_anime_by_type = filter_by_types(anime, selected_types)
        
        # User can filter by genres
        selected_genres = st.multiselect('Select genres to filter by', genres, default=genres)
        
        # Filter anime by selected genres
        def filter_by_genres(anime, selected_genres):
            return anime[anime['genre'].apply(lambda x: any(genre in x for genre in selected_genres) if isinstance(x, str) else False)]
        
        filtered_anime = filter_by_genres(filtered_anime_by_type, selected_genres)
        
        if filtered_anime.empty:
            st.warning("No anime found for the selected types and genres. Please adjust your filters.")
            return
        
        # Example: Using genre to recommend similar animes
        filtered_anime['genre'] = filtered_anime['genre'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(filtered_anime['genre'])
        
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        filtered_indices = pd.Series(range(len(filtered_anime)), index=filtered_anime['name']).drop_duplicates()
        
        def get_recommendations(names, num_recommendations=10, cosine_sim=cosine_sim):
            idxs = [filtered_indices[name] for name in names]
            sim_scores = sum([list(enumerate(cosine_sim[idx])) for idx in idxs], [])
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:num_recommendations+1]
            anime_indices = [i[0] for i in sim_scores]
            return filtered_anime[['name', 'rating', 'genre', 'type']].iloc[anime_indices]

        # User can select between 1 to 3 animes
        selected_animes = st.multiselect('Select up to 3 animes you like:', filtered_anime['name'].values, default=filtered_anime['name'].values[:1])
        
        # Slider to select number of recommendations
        num_recommendations = st.slider('Number of recommendations', min_value=1, max_value=30, value=10)
        
        if st.button('🚀 Recommend'):
            recommendations = get_recommendations(selected_animes, num_recommendations)
            st.subheader('🌟 Top Picks for You:')
            display_anime_cards(recommendations)

    def collaborative_filtering(ratings, anime):
        st.info("🤝 **Collaborative Filtering**: Suggests anime based on what users like you enjoyed.")
        
        # User can filter by types
        selected_types = st.multiselect('Select types to filter by', types, default=types)
        
        # Filter anime by selected types
        def filter_by_types(anime, selected_types):
            return anime[anime['type'].isin(selected_types)]
        
        filtered_anime_by_type = filter_by_types(anime, selected_types)
        
        # User can filter by genres
        selected_genres = st.multiselect('Select genres to filter by', genres, default=genres)
        
        # Filter anime by selected genres
        def filter_by_genres(anime, selected_genres):
            return anime[anime['genre'].apply(lambda x: any(genre in x for genre in selected_genres) if isinstance(x, str) else False)]
        
        filtered_anime = filter_by_genres(filtered_anime_by_type, selected_genres)
        
        if filtered_anime.empty:
            st.warning("No anime found. Please adjust filters.")
            return
        
        # Load pre-trained SVD model
        try:
            with open('models/svd_model.pkl', 'rb') as f:
                svd = pickle.load(f)
        except FileNotFoundError:
            st.error("Model file not found. Please train the model first.")
            return
        
        # Check and rename columns if necessary
        if 'ID' in ratings.columns:
            ratings[['user_id', 'anime_id']] = ratings['ID'].str.split('_', expand=True)
            ratings['user_id'] = ratings['user_id'].astype(int)
            ratings['anime_id'] = ratings['anime_id'].astype(int)

        # Reader for Surprise
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(ratings[['user_id', 'anime_id', 'rating']], reader)
        
        def get_recommendations(user_id, n=10):
            user_ratings = ratings[ratings['user_id'] == user_id]
            if user_ratings.empty:
                return pd.DataFrame() # Return empty if no ratings for user
            
            all_anime_ids = set(filtered_anime['anime_id'])
            rated_anime_ids = set(user_ratings['anime_id'])
            unrated_anime_ids = all_anime_ids - rated_anime_ids
            
            # Predict ratings for unrated animes
            predictions = [(anime_id, svd.predict(user_id, anime_id).est) for anime_id in unrated_anime_ids]
            predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
            
            recommended_animes = pd.DataFrame(predictions, columns=['anime_id', 'predicted_rating'])
            recommended_animes = recommended_animes.merge(filtered_anime, on='anime_id')
            
            recommended_animes['predicted_rating'] = recommended_animes['predicted_rating'].round(2)
            
            return recommended_animes

        # Predict ratings for a specific user and recommend top animes
        user_id = st.number_input('Enter User ID', min_value=1, max_value=int(ratings['user_id'].max()) if not ratings.empty else 1)
        if st.button('🚀 Recommend'):
            recs = get_recommendations(user_id)
            if recs.empty:
                st.warning("User has no ratings or doesn't exist.")
            else:
                st.subheader('🌟 Top Picks for You:')
                display_anime_cards(recs)

    # Show results based on selected algorithm
    if algorithm == 'Content-Based Filtering':
        content_based_filtering(anime)
    elif algorithm == 'Collaborative Filtering':
        collaborative_filtering(ratings, anime)
