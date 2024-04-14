import pandas as pd
import os

file = 'machine_learning-popular-base-filtering.py'

try:
    cur_dir = os.getcwd()
    print(f'cur_dir: {cur_dir}')

    if file not in os.listdir(cur_dir):
        os.chdir('./App19-MachineLearning/machine_learning')
        cur_dir = os.getcwd()
except:
    print(f'{file} already exists in folder')
    cur_dir = os.getcwd()

print(f'Current Directory: {cur_dir}')


movies = pd.read_csv('./csv_data/movies.csv')
ratings = pd.read_csv('./csv_data/ratings.csv')
credits = pd.read_csv('./csv_data/credits.csv')

print(f'MOVIES:\n{movies.head()}')
print(f'RATINGS:\n{ratings.head()}')
print(f'CREDITS:\n{credits.head()}')

# Popular Content Equation
# Weight Rating (WR) = (v / (v+m))*R + (m / (v+m))*C
# v = number of votes (vote_count)
# m - min number of votes required
#       0.0-1.0, where 0 is 0% and 1.0 is 100%
# R - Average rating of the movie (vote_average)
# C - average rating across all movies
#       average rating for all movies, so sum all vote_counts / num_of_movies

m = movies['vote_count'].quantile(0.9)      # want the movies with the count that starts from the 90th percentile
print(f'minimum number of votes: {m =}')

C = movies['vote_average'].mean()           # want the average for all of the movies
print(f'average movie rating: {C =}')

# filter for the specific movies that has a 'vote_count' above the 90% percentile
filtered_movies = movies.copy().loc[movies['vote_count'] >= m]  #shallow copy, so if a change is made, then all same dataframes will have the change
print(f'Filtered Movies with vote_count equal or above {m =:.2f}\n{filtered_movies}')

# create a function to calculate the Weight Rating (WR)
def weight_rating(df, m=m, C=C):
    v = df['vote_count']
    R = df['vote_average']

    wr = (v / (v+m))*R + (m / (v+m))*C
    return wr

# create a new column to the movies pd-dataframe with the weighted-rate
filtered_movies['weighted_rate'] = filtered_movies.apply(weight_rating, axis=1)

print(f'Weight Rate column added to Movies:\n{filtered_movies}')

# lets sort by weighted_rating
data = filtered_movies.sort_values('weighted_rate', ascending=False)
print(f'{data = }')

print(f'{filtered_movies.columns =}')

# lets sort by weighted_rate, but select the specific columns to view
print(f"{filtered_movies.sort_values('weighted_rate', ascending=False)[['title', 'vote_count', 'vote_average', 'weighted_rate']].head(10) =}")

# move the sorted data to a dictionary to transfer over for html
data_dict = filtered_movies.sort_values('weighted_rate', ascending=False)[['title', 'vote_count', 'vote_average', 'weighted_rate']].head()
temp = data_dict.to_dict()

print(f'{temp =}')