# All quantitative tools the agent can call
import yfinance as yf
import pandas as pd
import numpy as np
from langchain_core.tools import tool
@tool
def get_financial_ratios(ticker: str) -> str:
    '''Fetch live financial ratios for a stock ticker.'''
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        ratios = {
            'P/E': round(info.get('trailingPE', 0), 2),
            'P/B': round(info.get('priceToBook', 0), 2),
            'ROE (%)': round((info.get('returnOnEquity',0) or 0)*100, 2),
            'Debt/Equity': round(info.get('debtToEquity', 0), 2),
            'EV/EBITDA': round(info.get('enterpriseToEbitda', 0), 2),
            'FCF (B USD)': round((info.get('freeCashflow', 0) or 0)/1e9, 2),
            'Market Cap (B)':round((info.get('marketCap', 0) or 0)/1e9, 2),
        }
        lines = [f'{k}: {v}' for k, v in ratios.items()]
        return f'Financial ratios for {ticker}:\n' + '\n'.join(lines)
    except Exception as e:
        return f'Error fetching ratios: {e}'
@tool
def compute_dcf(ticker: str, growth_rate: float = 0.05,
                  wacc: float = 0.10, years: int = 5) -> str:
    '''Compute a simple DCF valuation for a stock.'''
    try:
        stock = yf.Ticker(ticker)
        fcf = stock.info.get('freeCashflow', None)
        shares = stock.info.get('sharesOutstanding', None)
        if not fcf or not shares:
            return 'Insufficient data for DCF.'
        pv = sum([fcf*(1+growth_rate)**t / (1+wacc)**t for t in range(1, years+1)])
        terminal = (fcf*(1+growth_rate)**years * (1+0.02)) / (wacc - 0.02)
        terminal_pv = terminal / (1+wacc)**years
        intrinsic = (pv + terminal_pv) / shares
        return (f'DCF for {ticker}: growth={growth_rate*100}%, '
                f'WACC={wacc*100}%, Intrinsic Value = USD {intrinsic:.2f}/share')
    except Exception as e:
        return f'DCF error: {e}'
@tool
def get_price_history(ticker: str, period: str = '1y') -> str:
    '''Get price summary: current, 52w high/low, return.'''
    try:
        hist = yf.Ticker(ticker).history(period=period)
        ret = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
        return (f'{ticker} | Current: {hist["Close"].iloc[-1]:.2f} | '
                f'52w High: {hist["High"].max():.2f} | '
                f'52w Low: {hist["Low"].min():.2f} | '
                f'1Y Return: {ret:.1f}%')
    except Exception as e:
        return f'Price error: {e}'
    
