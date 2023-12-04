import pickle
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.utils import shuffle

df = pd.read_csv("data/edited_rating.csv")

#I was trying to create train and test datasets
#However, we need all part of the dataset
#So slicing the dataset is wrong move.
df = shuffle(df)
df_train = df


#Init train dicts 
user_to_movie = {}
movie_to_user = {}
user_and_movie_to_rating = {}

#Creating user_to_movie, movie_to_user and user_and_movie_to_rating
def update_dicts(row):
    i = int(row.userId)
    j = int(row.movie_index)

    #Check if user is recorded
    #If not, record
    if i not in user_to_movie:
        user_to_movie[i] = [j]
    #If user is recorded,
    #Add new movie
    else:
        user_to_movie[i].append(j)

    #Check if movie is recorded
    #If not, record
    if j not in movie_to_user:
        movie_to_user[j] = [i]
    #If movie is recorded,
    #Add new user
    else:
        movie_to_user[j].append(i)

    #Add rating of the user for the current movie
    user_and_movie_to_rating[(i,j)] = row.rating

#Apply to all data
df_train.apply(update_dicts, axis=1)
    

#Save our dictionaries as json files
#Because we will use them in another file.
with open("jsons/user_to_movie.json", "wb") as f:
    pickle.dump(user_to_movie, f)

with open("jsons/movie_to_user.json", "wb") as f:
    pickle.dump(movie_to_user, f)

with open("jsons/user_and_movie_to_rating.json", "wb") as f:
    pickle.dump(user_and_movie_to_rating, f)
