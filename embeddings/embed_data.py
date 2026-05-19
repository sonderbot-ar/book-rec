
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_and_save_embeddings():
    df = pd.read_csv('/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/data/cleaned_books.csv')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(df['context_string'].tolist(), show_progress_bar=True)
    np.save('/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/embeddings/book_embeddings.npy', embeddings)
    print(f'Generated and saved embeddings for {len(df)} books.')

if __name__ == "__main__":
    generate_and_save_embeddings()
    