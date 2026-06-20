# Cross-checks numerical claims in LLM answer vs ground truth CSV
import re, pandas as pd
from loguru import logger
def load_ground_truth(path='data/structured/ground_truth.csv') -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logger.warning('Ground truth CSV not found — skipping number check')
        return pd.DataFrame()
def extract_numbers(text: str) -> list:
    '''Extract all numbers (possibly with B/M/% suffix) from text.'''
    pattern = r'\$?([0-9,]+\.?[0-9]*)\s*([BbMmKk%]?)'
    matches = re.findall(pattern, text)
    return [(m[0].replace(',',''), m[1]) for m in matches]
def check_answer(answer: str, ticker: str, gt_df: pd.DataFrame) -> dict:
    '''
    Returns dict with:
      - flagged: bool (True if suspicious numbers found)
      - warnings: list of strings
      - annotated_answer: answer with [UNVERIFIED] tags
    '''
    if gt_df.empty:
        return {'flagged': False, 'warnings': [], 'annotated_answer': answer}
    company_gt = gt_df[gt_df['ticker'] == ticker]
    known_values = company_gt['value'].tolist()
    nums_in_answer = extract_numbers(answer)
    warnings = []
    annotated = answer
    for num_str, suffix in nums_in_answer:
        try:
            num = float(num_str)
            # Check if any GT value is within 15% of this number
            close = any(abs(num - gv)/max(abs(gv),1) < 0.15 for gv in known_values)
            if not close and num > 100: # Only flag significant numbers
                warnings.append(f'{num_str}{suffix} could not be verified')
        except ValueError:
            pass
    return {
        'flagged': len(warnings) > 0,
        'warnings': warnings,
        'annotated_answer': annotated,
    }
