import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Import the dataset
df = pd.read_csv('/content/drive/MyDrive/ratings_Electronics.csv', header=None)
df.columns = ['user_id', 'prod_id', 'rating']
df = df.drop('timestamp', axis=1)
df_copy = df.copy(deep=True)

# Display dataset shape
rows, columns = df.shape
print("Number of rows =", rows)
print("Number of columns =", columns)

# Display dataset info
df.info()

# Find number of missing values in each column
missing_values = df.isna().sum()
print(missing_values)

# Summary statistics of 'rating' variable
rating_stats = df['rating'].describe()
print(rating_stats)

# Plot rating distribution
plt.figure(figsize=(12, 6))
df['rating'].value_counts(normalize=True).plot(kind='bar')
plt.show()

# Number of unique user IDs and product IDs in the data
unique_users = df['user_id'].nunique()
unique_products = df['prod_id'].nunique()
print('Number of unique USERS in Raw data =', unique_users)
print('Number of unique ITEMS in Raw data =', unique_products)

# Top 10 users based on rating count
most_rated = df['user_id'].value_counts().sort_values(ascending=False)[:10]
print(most_rated)

# Preprocessing
min_user_ratings = 50
counts = df['user_id'].value_counts()
df_final = df[df['user_id'].isin(counts[counts >= min_user_ratings].index)]
num_obs_final = len(df_final)
unique_users_final = df_final['user_id'].nunique()
unique_products_final = df_final['prod_id'].nunique()
print('The number of observations in the final data =', num_obs_final)
print('Number of unique USERS in the final data =', unique_users_final)
print('Number of unique PRODUCTS in the final data =', unique_products_final)

# Create interaction matrix
final_ratings_matrix = df_final.pivot(index='user_id', columns='prod_id', values='rating').fillna(0)
density = np.count_nonzero(final_ratings_matrix) / (final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]) * 100
print('Density: {:4.2f}%'.format(density))

# Model based Collaborative Filtering: Singular Value Decomposition

final_ratings_sparse = csr_matrix(final_ratings_matrix.values)

# Singular Value Decomposition
num_latent_features = 50
U, s, Vt = svds(final_ratings_sparse, k=num_latent_features)
sigma = np.diag(s)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# Predicted ratings
preds_df = pd.DataFrame(abs(all_user_predicted_ratings), columns=final_ratings_matrix.columns)

# Get recommendations
def recommend_items(user_index, interactions_matrix, preds_matrix, num_recommendations):
    user_ratings = interactions_matrix[user_index, :].toarray().reshape(-1)
    user_predictions = preds_matrix[user_index, :].toarray().reshape(-1)

    temp = pd.DataFrame({'user_ratings': user_ratings, 'user_predictions': user_predictions})
    temp['Recommended Products'] = np.arange(len(user_ratings))
    temp = temp.set_index('Recommended Products')
    
    temp = temp.loc[temp.user_ratings == 0]   
    temp = temp.sort_values('user_predictions', ascending=False)
    
    print('\nRecommended products for user(user_id = {}):\n'.format(user_index))
    print(temp['user_predictions'].head(num_recommendations))

# Test recommendations function
user_index = 121
num_recommendations = 5
recommend_items(user_index, final_ratings_sparse, preds_matrix, num_recommendations)

user_index2 = 100
num_recommendations2 = 10
recommend_items(user_index2, final_ratings_sparse, preds_matrix, num_recommendations2)

# Calculate RMSE
average_actual_ratings = final_ratings_matrix.mean()
average_predicted_ratings = preds_df.mean()
rmse_df = pd.concat([average_actual_ratings, average_predicted_ratings], axis=1)
rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']
RMSE = mean_squared_error(rmse_df['Avg_actual_ratings'], rmse_df['Avg_predicted_ratings'], squared=False)
print(f'RMSE SVD Model = {RMSE}')
