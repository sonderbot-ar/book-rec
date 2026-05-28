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

def ask_librarian(user_query, context_books, history=None):
    if history is None:
        history = []

    system_prompt = f"""
    You are a charming, highly intelligent librarian. 
    
    Here are the top 5 books from our database that match the user's latest request:
    {context_books}
    
    Write a conversational, engaging response. Answer their question based ONLY on 
    these provided descriptions and the context of your previous conversation.
    """
    
    url = "http://localhost:11434/api/chat"

    messages = [
        {'role': 'system', 'content': system_prompt}
    ]

    for msg in history:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            
            # If Gradio used the new OpenAI list format, extract just the text
            if isinstance(content, list):
                text_parts = [item["text"] for item in content if isinstance(item, dict) and "text" in item]
                content = " ".join(text_parts)
                
            # Ensure it is a clean string before sending to Ollama
            if isinstance(content, str) and content.strip():
                messages.append({"role": msg.get("role", "user"), "content": content})
                
        # Fallback for the old Gradio List/Tuple format
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            past_user, past_bot = msg
            
            # Only append if both parts are actual strings (ignoring file uploads)
            if isinstance(past_user, str) and isinstance(past_bot, str):
                messages.append({"role": "user", "content": past_user})
                messages.append({"role": "assistant", "content": past_bot})

    messages.append({'role': 'user', 'content': user_query})

    payload = {
        "model": "llama3.1",
        "messages": messages,
        "stream": False # Set to True later if we want the "typing" effect
    }

    print("\n--- DEBUG: SENDING PAYLOAD ---")
    print(f"Total messages in conversation history: {len(messages)}")
    
    try:
        response = requests.post(url, json=payload)
        
        # TERMINAL INTERCEPT 2: Check exactly what Ollama sends back
        print("\n--- DEBUG: OLLAMA RAW RESPONSE ---")
        print(response.json())
        
        if response.status_code == 200:
            return response.json()['message']['content']
        return "The Library is currently closed (API Error)."
    except Exception as e:
        print(f"\n--- DEBUG: CRASH --- \n{e}")
        return "Error: Cannot connect to Ollama."
    

    
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