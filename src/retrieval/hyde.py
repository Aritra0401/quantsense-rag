# src/retrieval/hyde.py
# Hypothetical Document Embeddings — boosts retrieval quality
from groq import Groq
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
def generate_hypothetical_doc(query: str) -> str:
    '''Ask LLM to write what the answer-document would look like.'''
    prompt = (
        f'You are a financial analyst. Write a 3-sentence excerpt from '
        f'an annual report or earnings transcript that directly answers: '
        f'{query}. Be specific, use numbers if relevant.'
    )
    resp = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=200, temperature=0.3,
    )
    return resp.choices[0].message.content
def hyde_embed(query: str, model: SentenceTransformer) -> list:
    hyp_doc = generate_hypothetical_doc(query)
    # Embed the hypothetical doc, not the query
    emb = model.encode(hyp_doc, normalize_embeddings=True).tolist()
    return emb, hyp_doc # Return embedding + text for display

if __name__ == "__main__":
    print(generate_hypothetical_doc(
        "What are Apple's major AI investments?"
    ))