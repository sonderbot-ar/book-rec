import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

def get_top_matches(query_vector, doc_vectors, top_n=5):
    dot_product = np.dot(doc_vectors, query_vector)
    norm_doc = np.linalg.norm(doc_vectors, axis=1)
    norm_query = np.linalg.norm(query_vector)
    
    similarities = dot_product / (norm_doc * norm_query)
    
    # Sort and get the top N highest scoring indices
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return top_indices

def ask_librarian(user_query, context_books):
    system_prompt = f"""
    You are a charming, highly intelligent librarian. 
    A user has asked for a book recommendation: "{user_query}"
    
    Here are the top 5 books from our database that match their request:
    {context_books}
    
    Write a conversational, engaging response recommending 1 or 2 of these books. 
    Explain exactly why they fit the user's request based ONLY on the provided descriptions.
    """
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1",
        "prompt": system_prompt,
        "stream": False # Set to True later if we want the "typing" effect
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()['response']
    else:
        return "Error: The Librarian is asleep."
    
def main():
    print("Loading the Library Database...")
    df = pd.read_csv('/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/data/enriched_books_for_rag.csv')
    
    print("Loading the Vector Engine (bge-small)...")
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    print("Loading the Vector Database (.npy)...")
    book_vectors = np.load('/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/embeddings/book_embeddings.npy')
    
    print("\n" + "="*50)
    print("📚 The AI Librarian is ready! (Type 'quit' to exit)")
    print("="*50)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        print("\nThinking...")
        
        query_vector = model.encode(user_input)
        
        top_indices = get_top_matches(query_vector, book_vectors, top_n=5)
        
        context_string = ""
        for idx in top_indices:
            row = df.iloc[idx]
            context_string += f"- TITLE: {row['title']} by {row['authors']}\n"
            context_string += f"  PLOT: {row['ai_context']}\n\n"
            
        answer = ask_librarian(user_input, context_string)
        
        print("\nLibrarian: " + answer)

if __name__ == "__main__":
    main()