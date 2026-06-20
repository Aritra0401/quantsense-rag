# Detects contradictions between 10-K filings and earnings transcripts
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
llm = Groq(api_key=os.getenv('GROQ_API_KEY'))
CONTRADICTION_TOPICS = [
    'revenue guidance', 'margin expectations',
    'debt levels', 'capex plans', 'headcount',
]
def find_contradictions(doc_a: str, doc_b: str,
                            label_a='10-K', label_b='Transcript') -> list:
    prompt = (
        f'Compare these two financial documents for contradictions.\n'
        f'DOCUMENT A ({label_a}): {doc_a[:1000]}\n'
        f'DOCUMENT B ({label_b}): {doc_b[:1000]}\n\n'
        f'List any factual contradictions or inconsistencies about: '
        f'{CONTRADICTION_TOPICS}.\n'
        f'Format: [TOPIC]: Document A says X, Document B says Y.\n'
        f'If none found, reply: No contradictions detected.'
    )
    resp = llm.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=400, temperature=0.1
    )
    output = resp.choices[0].message.content
    if 'No contradictions' in output:
        return []
    # Parse lines
    lines = [l.strip() for l in output.split('\n') if l.strip().startswith('[')]
    return lines