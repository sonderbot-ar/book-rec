import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('BAAI/bge-small-en-v1.5')
df = pd.read_csv('C:/Users/sudhi/Downloads/Book_Recommender_DeepLearning/book-rec/data/enriched_books_from_dump.csv')

#computing weighted_ratings
df['average_rating'] = df['average_rating'].fillna(0)
df['ratings_count'] = df['ratings_count'].fillna(0)

C = df['average_rating'].mean()
m = df['ratings_count'].quantile(0.90)

def calculate_weighted_rating(row, m, C):
    v = row['ratings_count']
    R = row['average_rating']
    if v == 0:
        return 0
    return (v / (v + m)) * R + (m / (v + m)) * C

df['weighted_rating'] = df.apply(calculate_weighted_rating, axis=1, m=m, C=C)


embeddings = np.load('C:/Users/sudhi/Downloads/Book_Recommender_DeepLearning/book-rec/embeddings/book_embeddings.npy')
print(df.info())

def get_recommendations(user_query, top_k=3):
    query_vector = model.encode([user_query])
    similarity_score = cosine_similarity(query_vector, embeddings)[0]
    top_20_indices = np.argsort(similarity_score)[::-1][:20]
    top_books = df.iloc[top_20_indices].copy()
    top_books['similarity_score'] = similarity_score[top_20_indices]

    final_recommendations = top_books.sort_values(
        by=['weighted_rating', 'similarity_score'],
        ascending=[False, False]).head(top_k)
    
    return final_recommendations 

if __name__ == "__main__":
    test_query = "I want a romance novel that has a strong character development."
    results = get_recommendations(test_query)

    for index, row in results.iterrows():
        print(f"Title: {row['title']}")
        print(f"Author(s): {row['authors']}")
        print(f"Average Rating: {row['average_rating']}")
        print(f"Categories: {row['categories']}")
        print(f"Published Year: {row['published_year']}")
        print(f"Match Score: {row['similarity_score']:.4f}")
        print("-" * 40)
