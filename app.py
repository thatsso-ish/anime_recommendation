import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader

# Load data
def load_data():
    anime = pd.read_csv('data/anime.csv')
    ratings = pd.read_csv('data/ratings.csv')
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return anime, ratings, train, test

anime, ratings, train, test = load_data()

# Extract unique types and genres
types = sorted(anime['type'].dropna().unique())
genres = set()
for genre_list in anime['genre'].dropna().str.split(', '):
    genres.update(genre_list)
genres = sorted(genres)

# Sidebar for navigation
page = st.sidebar.radio("Choose a page", ["About", "Recommendations"])

if page == "About":
    st.title('About')
    st.image('visuals/anime.png', caption='Anime Recommendation System', use_column_width=True)
    st.write("""
    ### Anime Recommendation System
    
    This Anime Recommendation System helps users discover new animes based on their preferences. The system offers two types of recommendation algorithms:
    
    - **Content-Based Filtering**: This algorithm recommends animes based on the genre and type of animes that a user likes. It uses the genre and type information to find similar animes.
    
    - **Collaborative Filtering**: This algorithm recommends animes based on the ratings given by users. It uses a collaborative approach, finding animes that similar users have liked.
    
    ### How It Works
    
    1. **Content-Based Filtering**:
        - Users can filter animes by type and genre.
        - Select up to 3 animes they like.
        - The system recommends similar animes based on the selected animes.
    
    2. **Collaborative Filtering**:
        - Users can filter animes by type and genre.
        - Enter their User ID.
        - The system recommends animes that similar users have rated highly.
    
    Explore the "Recommendations" page to start finding your next favorite anime!
    """)

elif page == "Recommendations":
    st.title('Anime Recommendation System')

    # Algorithm selection
    algorithm = st.selectbox('Select Algorithm', ('Content-Based Filtering', 'Collaborative Filtering'))

    # Implement the chosen algorithm
    def content_based_filtering(anime):
        st.write("Content-Based Filtering is selected.")
        
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
            st.write("No anime found for the selected types and genres. Please select different filters.")
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
            return filtered_anime[['name', 'rating']].iloc[anime_indices]

        # User can select between 1 to 3 animes
        selected_animes = st.multiselect('Select up to 3 animes to get recommendations', filtered_anime['name'].values, default=filtered_anime['name'].values[:1])
        
        # Slider to select number of recommendations
        num_recommendations = st.slider('Number of recommendations', min_value=1, max_value=30, value=10)
        
        if st.button('Recommend'):
            recommendations = get_recommendations(selected_animes, num_recommendations)
            st.write('Recommended Animes:')
            
            # Display recommendations in tabular format
            st.table(pd.DataFrame({
                'Rank': range(1, len(recommendations) + 1),
                'Anime Title': recommendations['name'],
                'Rating / 10': recommendations['rating'].round(2)
            }))

    def collaborative_filtering(ratings, anime):
        st.write("Collaborative Filtering is selected.")
        
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
            st.write("No anime found for the selected types and genres. Please select different filters.")
            return
        
        # Load pre-trained SVD model
        with open('models/svd_model.pkl', 'rb') as f:
            svd = pickle.load(f)
        
        # Check and rename columns if necessary
        if 'ID' in ratings.columns:
            ratings[['user_id', 'anime_id']] = ratings['ID'].str.split('_', expand=True)
            ratings['user_id'] = ratings['user_id'].astype(int)
            ratings['anime_id'] = ratings['anime_id'].astype(int)

        # Reader for Surprise
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(ratings[['user_id', 'anime_id', 'rating']], reader)
        
        # Predict ratings for a specific user and recommend top animes
        user_id = st.number_input('Enter User ID', min_value=1, max_value=ratings['user_id'].max())
        if st.button('Recommend'):
            user_ratings = ratings[ratings['user_id'] == user_id]
            if user_ratings.empty:
                st.write("No ratings found for this user.")
                return
            
            # Get all animes not rated by the user
            all_anime_ids = set(filtered_anime['anime_id'])
            rated_anime_ids = set(user_ratings['anime_id'])
            unrated_anime_ids = all_anime_ids - rated_anime_ids
            
            # Predict ratings for unrated animes
            predictions = [(anime_id, svd.predict(user_id, anime_id).est) for anime_id in unrated_anime_ids]
            predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
            
            recommended_animes = pd.DataFrame(predictions, columns=['anime_id', 'predicted_rating'])
            recommended_animes = recommended_animes.merge(filtered_anime, on='anime_id')
            
            recommended_animes['predicted_rating'] = recommended_animes['predicted_rating'].round(2)
            
            # Display recommendations in tabular format
            st.write('Recommended Animes:')
            st.table(pd.DataFrame({
                'Rank': range(1, len(recommended_animes) + 1),
                'Anime Title': recommended_animes['name'].values,
                'Predicted Rating': recommended_animes['predicted_rating'].values
            }))

    # Show results based on selected algorithm
    if algorithm == 'Content-Based Filtering':
        content_based_filtering(anime)
    elif algorithm == 'Collaborative Filtering':
        collaborative_filtering(ratings, anime)
