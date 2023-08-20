import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

def load_dataset(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = ['user_id', 'prod_id', 'rating']
    df = df.drop('timestamp', axis=1)
    return df

def preprocess_data(df, min_user_ratings):
    user_ratings = df['user_id'].value_counts()
    df_final = df[df['user_id'].isin(user_ratings[user_ratings >= min_user_ratings].index)]
    return df_final

def calculate_density(rating_matrix):
    given_num_of_ratings = np.count_nonzero(rating_matrix)
    possible_num_of_ratings = rating_matrix.shape[0] * rating_matrix.shape[1]
    density = (given_num_of_ratings / possible_num_of_ratings) * 100
    return density

def top_n_products(final_rating, n, min_interaction):
    recommendations = final_rating[final_rating['rating_count'] > min_interaction]
    recommendations = recommendations.sort_values('avg_rating', ascending=False)
    return recommendations.index[:n]

def main():
    file_path = '/content/drive/MyDrive/ratings_Electronics.csv'
    min_user_ratings = 50
    min_interaction = 100
    num_top_products = 5

    # Load dataset
    df = load_dataset(file_path)

    # Preprocess data
    df_final = preprocess_data(df, min_user_ratings)

    # Calculate density
    final_ratings_matrix = df_final.pivot(index='user_id', columns='prod_id', values='rating').fillna(0)
    density = calculate_density(final_ratings_matrix)
    print('Density: {:.2f}%'.format(density))

    # Rank-based recommendation
    average_rating = df_final.groupby('prod_id').mean()['rating']
    count_rating = df_final.groupby('prod_id').count()['rating']
    final_rating = pd.DataFrame({'avg_rating': average_rating, 'rating_count': count_rating})
    final_rating = final_rating.sort_values(by='avg_rating', ascending=False)

    top_products = top_n_products(final_rating, num_top_products, min_interaction)
    print('Top {} products with at least {} interactions:'.format(num_top_products, min_interaction))
    print(list(top_products))

if __name__ == '__main__':
    main()
