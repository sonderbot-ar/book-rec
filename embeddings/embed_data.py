
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def build_hybrid_context(row):
    orig = str(row.get('description', ''))
    api = str(row.get('api_description', ''))
    
    if orig.lower() == 'nan': orig = ''
    if api.lower() == 'nan': api = ''
        
    orig_words = orig.split()
    api_words = api.split()

    if len(orig_words) > 20 and len(api_words) > 20:
        part1 = " ".join(orig_words[:150])
        part2 = " ".join(api_words[:150])
        return f"{part1}... PLOT EXPANSION: {part2}"
    
    elif len(api_words) > 20:
        return api
        
    elif len(orig_words) > 20:
        return orig
        
    else:
        title = str(row.get('title', 'Unknown Title'))
        author = str(row.get('authors', 'Unknown Author'))
        return f"{title} by {author}. No detailed description available."
    


def main():
    df = pd.read_csv('C:/Users/sudhi/Downloads/Book_Recommender_DeepLearning/book-rec/data/enriched_books_from_dump.csv')
    df['ai_context'] = df.apply(build_hybrid_context, axis=1)
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    embeddings = model.encode(df['ai_context'].tolist(), show_progress_bar=True)
    np.save('book_embeddings.npy', embeddings)
    df.to_csv('enriched_books_for_rag.csv', index=False)

if __name__ == "__main__":
    main()