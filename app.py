import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from librarian_chat import *

df = pd.read_csv('/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/data/enriched_books_for_rag.csv')
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
book_vectors = np.load('/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/embeddings/book_embeddings.npy')


def chat_with_rag(message, history):
    query_vector = model.encode(message)
    
    top_indices = get_top_matches(query_vector, book_vectors, top_n=5)
    
    context_string = ""
    for idx in top_indices:
        row = df.iloc[idx]
        context_string += f"- TITLE: {row['title']} by {row['authors']}\n"
        context_string += f"  PLOT: {row['ai_context']}\n\n"
        
    return ask_librarian(message, context_string)

demo = gr.ChatInterface(
    fn=chat_with_rag,
    title="📚 The AI Librarian",
    description="Ask for book recommendations! This runs 100% locally using Llama 3.1 and your custom vector database.",
    examples=[
        "I want a fast-paced sci-fi thriller with space battles.",
        "Recommend me a non-fiction book about historical espionage.",
        "I'm looking for a cozy mystery set in a snowy cabin."
    ]
)

if __name__ == "__main__":
    demo.launch(share=False)