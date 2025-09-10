


# rag_chatbot_en_dataset.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
# 🔑 Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 1. Load multilingual embedding model
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 2. Load English-only dataset
with open("faqs.txt", "r", encoding="utf-8") as f:
    docs = [line.strip() for line in f if line.strip()]

# 3. Create FAISS index
embeddings = embed_model.encode(docs)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# 4. Chat function
def rag_chat(user_query, k=2):
    # Embed user query
    q_emb = embed_model.encode([user_query])
    
    # Retrieve top k docs
    D, I = index.search(np.array(q_emb), k=k)
    context = "\n".join([docs[i] for i in I[0]])
    
    # Multilingual RAG prompt
    prompt = f"""
    You are a helpful campus assistant.
    
    Context information:
    {context}
    
    Question: {user_query}
    
    Rules:
    - The dataset is only in English.
    - User queries may be in English, Hindi, or Tamil (or other languages).
    - Always reply in the same language as the user’s query.
    - If the answer is not found in the context, politely say you don't know.
    """
    
    # Call Gemini
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text

# 5. Run chatbot in console
if __name__ == "__main__":
    print("🎓 Multilingual Campus Chatbot (English dataset, answers in query language)")
    print("Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = rag_chat(query)
        print("Bot:", answer)
