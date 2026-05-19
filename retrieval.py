import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv('/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/data/cleaned_books.csv')
embeddings = np.load('/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/embeddings/book_embeddings.npy')


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