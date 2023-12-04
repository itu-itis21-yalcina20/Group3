import pandas as pd

#Import original csv file
df = pd.read_csv("data/ratings.csv")

#Substract 1 because user id start from 1
df.userId = df.userId - 1


#Use set operation because there are many same row with same movieId
unique_movie = set(df.movieId.values)
movie_index = {}
count = 0


#Create an index for each movie
for id in unique_movie:
    movie_index[id] = count
    count +=1

#Add new index to main data
df["movie_index"] = df.apply(lambda row: movie_index[row.movieId], axis=1)

#Remove timestamp column, it will not be used
df = df.drop(columns=["timestamp"])

#Save the file.
df.to_csv("data/edited_rating.csv")


