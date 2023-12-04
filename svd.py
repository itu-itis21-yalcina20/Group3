import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime


#Load the dictionaries from folder.
with open("jsons/user_to_movie.json", "rb") as f:
    user_to_movie = pickle.load(f)

with open("jsons/movie_to_user.json", "rb") as f:
    movie_to_user = pickle.load(f)

with open("jsons/user_and_movie_to_rating.json", "rb") as f:
    user_and_movie_to_rating = pickle.load(f)

#Find the number of users and movies
N = np.max(list(user_to_movie.keys())) + 1
M = np.max(list(movie_to_user.keys())) + 1


#Create an aditional dictionary
#This dictionary is responsible for finding all rating of users
user_to_movierating = {}
for i, movies in user_to_movie.items():
    r = np.array([user_and_movie_to_rating[(i,j)] for j in movies])
    user_to_movierating[i] = (movies, r)

#Create an aditional dictionary
#This dictionary is responsible for finding all rating of movies
movies_to_userrating = {}
for j, users in movie_to_user.items():
    r = np.array([user_and_movie_to_rating[(i,j)] for i in users])
    movies_to_userrating[j] = (users, r)


#Error functions
def get_sse(m2u):
    #m2u = movie -> (user, rating)
    N = 0
    sse = 0
    for j, (user, r) in m2u.items():
        #Predict the rating using the weights of user and movie
        p = W[user].dot(U[j]) + b[user] + c[j] + rating_mean

        #Find the difference 
        delta = r - p
        #Because delta is a vector, we need dot product
        sse += delta.dot(delta)
        N += len(r)
    return sse/N

def get_mae(m2u):
    #m2u = movie -> (user, rating)
    N = 0
    mae = 0
    for j, (user, r) in m2u.items():
        #Predict the rating using the weights of user and movie
        p = W[user].dot(U[j]) + b[user] + c[j] + rating_mean
        for i in range(len(r)):
            #Find the absulute difference
            delta = abs(r[i]- p[i])
            mae += delta
        N += len(r)
    return mae/N

#Variables
KK = 50     #Number of weights of users and movies
k_mae = []  #A list for recording mae 
k_sse = []  #A list for recording sse

for K in range(KK):
    #Init weights and biases
    W = np.random.randn(N,K)
    b = np.zeros(N)
    U = np.random.randn(M,K)
    c = np.zeros(M)
    #Calculate the mean
    rating_mean = np.mean(list(user_and_movie_to_rating.values()))




    #Parametres for training
    epochs = 5
    reg = 0.1
    mae_losses = []
    sse_losses = []

    for epoch in range(epochs):
        print("Epoch", epoch)
        #Time calculation
        epoch_start = datetime.now()
        t0 = datetime.now()

        for i in range(N):
            m_ids, r = user_to_movierating[i]
            #Implement formulas
            matrix = U[m_ids].T.dot(U[m_ids]) + np.eye(K) * reg
            vector = (r - b[i] - c[m_ids] - rating_mean).dot(U[m_ids])
            bi = (r - U[m_ids].dot(W[i]) - c[m_ids] - rating_mean).sum()
            
            #Find new weight and bias
            W[i] = np.linalg.solve(matrix, vector)
            b[i] = bi/ (len(user_to_movie[i]) + reg)

            if i % 300 == 0:
                print("i: ", i, "N: ", N)
        print("Updated W and b: ", datetime.now() - t0)

        t1 = datetime.now()
        for j in range(M):
            try:
                #Implement formulas
                u_ids, r = movies_to_userrating[j]
                matrix = W[u_ids].T.dot(W[u_ids]) + np.eye(K)*reg
                vector = (r - b[u_ids] - c[j] - rating_mean).dot(W[u_ids])
                cj = (r - W[u_ids].dot(U[j]) - b[u_ids] - rating_mean).sum()

                #Find new weight and bias
                U[j] = np.linalg.solve(matrix, vector)
                c[j] = cj / (len(movie_to_user[j]) + reg)
            except:
                pass
            if j % 4000 == 0:
                print("j: ", j, "M: ", M)
        
            
        print("Updated U and c: ", datetime.now() - t1)
        print("Epoch ends: ", datetime.now()- epoch_start)

        t2 = datetime.now()
        sse_losses.append(get_sse(movies_to_userrating))
        mae_losses.append(get_mae(movies_to_userrating))

        print("Calculating errors", datetime.now() - t2)
        print("SSE: ", sse_losses[-1])
        print("MAE: ", mae_losses[-1])

    print("SSE: ", sse_losses[-1])
    print("MAE: ", mae_losses[-1])
    k_mae.append(mae_losses[-1])
    k_sse.append(sse_losses[-1])

plt.plot(k_sse, label="SSE")
plt.plot(k_mae, label="MAE")
plt.xlabel("K Value")
plt.ylabel("Rating error")
plt.title("Error for 5 epoch vs K value")
plt.legend()
plt.show()

def predict(userID, movieID):
    p = W[userID].dot(U[movieID]) + b[userID] + c[movieID] + rating_mean
    return p

print(predict(0,0))


