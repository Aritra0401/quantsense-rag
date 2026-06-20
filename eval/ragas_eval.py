# eval/ragas_eval.py
# Run RAGAS evaluation — generates your PPT slide 12 numbers
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from src.agent.graph import agent_graph
from loguru import logger
def run_pipeline(question: str) -> dict:
    result = agent_graph.invoke({
        'query': question, 'docs': [], 'tool_output': '',
        'answer': '', 'confidence': 0.0, 'used_web_fallback': False,
        'hyp_text': None, 'retry_count': 0, 'route': None,
    })
    return result
def run_ragas_eval(questions_path='eval/eval_questions.json'):
    with open(questions_path) as f:
        questions = json.load(f)[:20] # Start with 20 for speed
    data = {'question':[], 'answer':[], 'contexts':[], 'ground_truth':[]}
    for i, q in enumerate(questions):
        logger.info(f'Evaluating {i+1}/{len(questions)}: {q[:60]}')
        try:
            result = run_pipeline(q)
            data['question'].append(q)
            data['answer'].append(result.get('answer',''))
            contexts = [d['text'] for d in result.get('docs',[])[:3]]
            data['contexts'].append(contexts if contexts else ['No context'])
            data['ground_truth'].append(q) # Self-eval; replace with manual GT
        except Exception as e:
            logger.error(f'Failed: {e}')
    ds = Dataset.from_dict(data)
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy,
                            context_precision, context_recall])
    print('\n=== RAGAS SCORES ===')
    print(scores)
    scores.to_pandas().to_csv('eval/ragas_results.csv', index=False)
    return scores
if __name__ == '__main__':
    run_ragas_eval()