import gradio as gr 
from retrieval import get_recommendations

def chatbot(message, history):
    results = get_recommendations(message, top_k=3)
    response = "Here are some recommendations based on what you asked for:\n"

    for index, row in results.iterrows():
        response += f"### 📖 **{row['title']}**\n"
        response += f"**Author:** {row['authors']}\n"
        response += f"**Genre:** {row['categories']} | **Year:** {int(row['published_year'])}\n"
        response += f"**Weighted Rating:** {row['weighted_rating']:.2f}/5.0\n"
        response += '---\n'

    return response

demo = gr.ChatInterface(
    fn=chatbot,
    title="Book Recommendation Chatbot",
    description="Ask for book recommendations based on your preferences!",
)

if __name__ == "__main__":
    demo.launch()