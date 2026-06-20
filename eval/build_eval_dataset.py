# Auto-generates evaluation dataset using LLM
# Run this once after ingestion is done
import json, os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
llm = Groq(api_key=os.getenv('GROQ_API_KEY'))
SEED_QUESTIONS = [
    'What was Apple revenue in FY2023?',
    'What are the main risk factors for JPMorgan?',
    'What did HDFC Bank say about NIM in FY2024?',
    'What is Infosys revenue growth guidance?',
    'What is Reliance Industries EBITDA margin?',
]
def generate_variations(base_q: str, n: int = 10) -> list:
    prompt = (
        f'Generate {n} financial questions similar to: "{base_q}"\n'
        f'Make them specific, answerable from annual reports. '
        f'Output as JSON list of strings.'
    )
    resp = llm.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role':'user','content':prompt}],
        max_tokens=600
    )
    try:
        text = resp.choices[0].message.content
        start = text.find('[')
        end = text.rfind(']') + 1
        return json.loads(text[start:end])[:n]
    except:
        return [base_q]
if __name__ == '__main__':
    all_questions = []
    for q in SEED_QUESTIONS:
        all_questions.extend(generate_variations(q, n=10))
    all_questions = list(set(all_questions))[:50]
    with open('eval/eval_questions.json', 'w') as f:
        json.dump(all_questions, f, indent=2)
    print(f'Generated {len(all_questions)} eval questions')