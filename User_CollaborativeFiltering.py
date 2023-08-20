import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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

# Get similar users
def similar_users(user_index, interactions_matrix):
    similarity = []
    for user in range(interactions_matrix.shape[0]):
        sim = cosine_similarity([interactions_matrix.loc[user_index]], [interactions_matrix.loc[user]])
        similarity.append((user, sim))
    similarity.sort(key=lambda x: x[1], reverse=True)
    most_similar_users = [tup[0] for tup in similarity]
    similarity_scores = [tup[1] for tup in similarity]
    most_similar_users.remove(user_index)
    similarity_scores.remove(similarity_scores[0])
    return most_similar_users, similarity_scores

# Get recommendations
def recommendations(user_index, num_of_products, interactions_matrix):
    most_similar_users = similar_users(user_index, interactions_matrix)[0]
    observed_interactions = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[user_index] > 0)]))
    recommendations = []
    for similar_user in most_similar_users:
        if len(recommendations) < num_of_products:
            similar_user_prod_ids = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[similar_user] > 0)]))
            recommendations.extend(list(similar_user_prod_ids.difference(observed_interactions)))
            observed_interactions = observed_interactions.union(similar_user_prod_ids)
        else:
            break
    return recommendations[:num_of_products]

# Test recommendations function
user_index = 3
num_of_recommendations = 5
recommended_products = recommendations(user_index, num_of_recommendations, final_ratings_matrix)
print('Recommended products for user {}:'.format(user_index))
print(recommended_products)

# Test recommendations for another user
user_index2 = 1521
recommended_products2 = recommendations(user_index2, num_of_recommendations, final_ratings_matrix)
print('Recommended products for user {}:'.format(user_index2))
print(recommended_products2)
