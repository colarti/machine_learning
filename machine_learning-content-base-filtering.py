import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# print(os.getcwd())
# os.chdir('./App19-MachineLearning/machine_learning/')
movies = pd.read_csv('./csv_data/movies.csv')
credits = pd.read_csv('./csv_data/credits.csv')
ratings = pd.read_csv('./csv_data/ratings.csv')
small_movies = pd.read_csv('./csv_data/movies_small.csv', sep=',')

print(f'MOVIES---------\n{movies.head()}')
print(f'CREDITS--------\n{credits.tail()}')
print(f'RATINGS--------\n{ratings.head(10)}')
print(f'SMALL_MOVIES---\n{small_movies}')


# CONTENT-BASE FILTERING

tfidf = TfidfVectorizer(stop_words='english')
small_movies['overview'] = small_movies['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(small_movies['overview'])
print(tfidf_matrix.toarray())

print(pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out()))

print(tfidf_matrix.shape)   # (movies, word)

similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

print(similarity_matrix)

print(f'TITLES:\n{small_movies["original_title"]}')

movie_title = 'John Carter'

idx = small_movies.loc[small_movies['original_title'] == movie_title].index[0]

print(f'{idx = }')

scores = similarity_matrix[idx]

print(f'{scores = }')

scores = list(scores)

print(f'{scores = }')

scores = sorted(enumerate(scores), key=lambda x:x[1], reverse=True)

print(f'Sorted {scores = }')

recommended_indexes = [x[0] for x in scores if x[1] < 1 and x[1] > 0]

print(f'{recommended_indexes = }')

recommended_movies = small_movies['original_title'].iloc[recommended_indexes]

print(f'{recommended_movies = }\n')


# depending on the movie_title, find the similar movies 'nr' number of returns
def similar_movies(movie_title, nr_movies):
    movie_index = movies.loc[movies['original_title'] == movie_title].index[0]
    score = list(enumerate(similarity_matrix[movie_index]))
    score = sorted(score, key=lambda x:x[1], reverse=True)
    movie_indicies = [tpl[0] for tpl in score[1:] if tpl[1] > 0 and tpl[1] < 1]


    movie_recommendations = movies['original_title'].iloc[movie_indicies]


    if len(movie_recommendations) > nr_movies:
        movie_recommendations = movie_recommendations[:nr_movies]


    return movie_recommendations


print(similar_movies('Spectre', 12))