import requests, os, time
from loguru import logger
HEADERS = {'User-Agent': 'QuantSense aritrabiswas0004@gmail.com'}
BASE = 'https://data.sec.gov'
TICKER_TO_CIK = {
    'AAPL': '0000320193',
    'JPM': '0000019617',
    'INFY': '0001067491',
}
def get_filings(cik: str, form_type: str = '10-K', max_count: int = 2):
    url = f'{BASE}/submissions/CIK{cik}.json'
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    filings = data['filings']['recent']
    results = []
    for i, ft in enumerate(filings['form']):
        if ft == form_type and len(results) < max_count:
            results.append({
                'accession': filings['accessionNumber'][i].replace('-',''),
                'date': filings['filingDate'][i],
                'form': ft,
            })
    return results
def download_filing_txt(cik: str, accession: str, out_path: str):
    '''Downloads filing index and gets the primary document.'''
    idx_url = f'{BASE}/Archives/edgar/data/{int(cik)}/{accession}/{accession}-index.json'
    r = requests.get(idx_url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        logger.warning(f'Index not found: {accession}')
        return False
    idx = r.json()
    # Find .htm or .txt primary doc
    primary = None
    for item in idx.get('files', []):
        if item.get('type') in ('10-K', '10-Q') and item['name'].endswith('.htm'):
            primary = item['name']
            break
    if not primary:
        return False
    doc_url = f'{BASE}/Archives/edgar/data/{int(cik)}/{accession}/{primary}'
    doc_r = requests.get(doc_url, headers=HEADERS, timeout=30)
    with open(out_path, 'w', encoding='utf-8', errors='replace') as f:
        f.write(doc_r.text)
    logger.info(f'Saved: {out_path}')
    return True
def download_all(save_dir: str = 'data/raw/filings'):
    os.makedirs(save_dir, exist_ok=True)
    for ticker, cik in TICKER_TO_CIK.items():
        for form in ['10-K', '10-Q']:
            filings = get_filings(cik, form, max_count=2)
            for filing in filings:
                fname = f'{save_dir}/{ticker}_{form}_{filing["date"]}.htm'
                if not os.path.exists(fname):
                    download_filing_txt(cik, filing['accession'], fname)
                    time.sleep(0.5) # Respect SEC rate limits
    logger.info('All downloads complete.')
if __name__ == '__main__':
    download_all()