# Detects management tone shifts across earnings call transcripts
# This is a real alternative data alpha signal used by quant funds
import os, glob
from groq import Groq
from dotenv import load_dotenv
import json
load_dotenv()
llm = Groq(api_key=os.getenv('GROQ_API_KEY'))
TOPICS = ['revenue_growth', 'margin_pressure', 'competition', 'guidance', 'risk']
def score_sentiment(text: str, topic: str) -> dict:
    '''Score management tone on a topic from -1 (very negative) to 1 (very positive).'''
    prompt = (
        f'Analyze the management tone about "{topic}" in this earnings call excerpt.\n'
        f'Return ONLY a JSON: {{"score": float -1 to 1, "key_phrase": str}}\n'
        f'Text: {text[:1500]}'
    )
    resp = llm.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role':'user','content':prompt}],
        max_tokens=100, temperature=0.1
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {'score': 0.0, 'key_phrase': 'parse error'}
def analyze_ticker(ticker: str, transcript_dir='data/raw/transcripts') -> list:
    '''Return tone scores per quarter per topic for a ticker.'''
    files = sorted(glob.glob(f'{transcript_dir}/{ticker}_*.txt'))
    results = []
    for fpath in files:
        period = os.path.basename(fpath).replace('.txt','').split('_',1)[1]
        with open(fpath) as f:
            text = f.read()
        scores = {topic: score_sentiment(text, topic) for topic in TOPICS}
        results.append({'period': period, 'scores': scores})
    return results
def get_drift_alerts(results: list) -> list:
    '''Flag topics where sentiment changed by more than 0.4 between periods.'''
    alerts = []
    if len(results) < 2:
        return alerts
    for topic in TOPICS:
        prev = results[-2]['scores'][topic]['score']
        curr = results[-1]['scores'][topic]['score']
        drift = curr - prev
        if abs(drift) > 0.4:
            direction = 'POSITIVE' if drift > 0 else 'NEGATIVE'
            alerts.append(f'{direction} drift on {topic}: {prev:.2f} fi {curr:.2f}')
    return alerts