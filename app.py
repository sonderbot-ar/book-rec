import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from librarian_chat import *

df = pd.read_csv('/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/data/enriched_books_for_rag.csv')
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
book_vectors = np.load('/Users/andresrodartee/Desktop/eada/Deep Learning/book-rec/embeddings/book_embeddings.npy')


def chat_with_rag(message, history, book_history_state):
    query_vector = model.encode(message)
    
    top_indices = get_top_matches(query_vector, book_vectors, top_n=5)
    
    context_string = ""
    new_books_list = []
    
    for idx in top_indices:
        row = df.iloc[idx]
        context_string += f"- TITLE: {row['title']} by {row['authors']}\n  PLOT: {row['ai_context']}\n\n"
        
        new_books_list.append({
            "Title": row['title'],
            "Author": row['authors']
        })
        
    bot_response = ask_librarian(message, context_string, history)
    new_books_df = pd.DataFrame(new_books_list)
    if not isinstance(book_history_state, pd.DataFrame):
        try:
            if isinstance(book_history_state, dict) and "data" in book_history_state:
                book_history_state = pd.DataFrame(book_history_state["data"], columns=["Title", "Author"])
            else:
                book_history_state = pd.DataFrame(book_history_state, columns=["Title", "Author"])
        except Exception:
            book_history_state = pd.DataFrame(columns=["Title", "Author"])
            
    if book_history_state is None or book_history_state.empty:
        updated_history_df = new_books_df
    else:
        updated_history_df = pd.concat([book_history_state, new_books_df])
        
        updated_history_df = updated_history_df.drop_duplicates(subset=['Title'])
        
        updated_history_df = updated_history_df.reset_index(drop=True)
        
    print("\n--- DEBUG: TABLE UPDATE ---")
    print(f"Old Table Rows: {len(book_history_state)} | New Table Rows: {len(updated_history_df)}")

    return bot_response, updated_history_df


initial_table = pd.DataFrame(columns=["Title", "Author"])

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 The AI Librarian")
    
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(placeholder="Ask the librarian for a recommendation...", label="Your Question")
            clear = gr.ClearButton([msg, chatbot])
            
        with gr.Column(scale=1):
            gr.Markdown("### 🕒 Recommendation History")
            book_table = gr.Dataframe(
                value=initial_table,
                headers=["Title", "Author"],
                interactive=False, 
                wrap=True
            )

    def user_sends_message(user_msg, chat_history):
        """Instantly updates the chat UI using strict Gradio 5 dictionaries."""
        chat_history.append({"role": "user", "content": user_msg})
        return "", chat_history

    def bot_responds(chat_history, current_table_df):
        """Calls our RAG logic and updates the chat and the table."""
        raw_content = chat_history[-1]["content"] 
        
        if isinstance(raw_content, list):
            text_parts = [item.get("text", "") for item in raw_content if isinstance(item, dict)]
            user_msg = " ".join(text_parts).strip()
        elif isinstance(raw_content, tuple):
            user_msg = str(raw_content[0])
        else:
            user_msg = str(raw_content)
        
        bot_reply, updated_table = chat_with_rag(user_msg, chat_history[:-1], current_table_df)
        
        chat_history.append({"role": "assistant", "content": bot_reply})
        
        return chat_history, updated_table

    msg.submit(
        user_sends_message, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot], 
        queue=False
    ).then(
        bot_responds,
        inputs=[chatbot, book_table], 
        outputs=[chatbot, book_table] 
    ) 

if __name__ == "__main__":
    demo.launch()