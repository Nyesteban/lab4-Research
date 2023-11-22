import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

movies = pd.read_csv('tmdb_movies_data.csv')
#movies_unpopular = movies[movies['popularity'] <= movies.loc[:, 'popularity'].mean()]

values_combined = movies['genres'].str.replace('|', ' ') + " " + movies['keywords'].str.replace('|', ' ')
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(values_combined.values.astype('U'))
cosine_similarity = cosine_similarity(tfidf_matrix)
similarity_df = pd.DataFrame(cosine_similarity, index=movies['original_title'], columns=movies['original_title'])

movie = input('Enter a movie you like: ')
movie_index = similarity_df.index.get_loc(movie)
top_10 = similarity_df.iloc[movie_index].sort_values(ascending=False)[1:11]
print(f'Top 10 similar movies to {movie}:')
print(top_10)
