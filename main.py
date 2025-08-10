"""
Simple trading bot template (SmartAPI compatible)
Modes:
 - backtest: run a vectorized backtest on historical CSV data
 - paper: connect to SmartAPI (paper trading) and place simulated/real orders
Strategy included: SMA crossover (fast & slow)
"""

import os
import time
import json
from dataclasses import dataclass
from typing import Tuple, List
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # reads .env file

# ======= CONFIG =======
SMARTAPI_USERNAME = os.getenv("SMARTAPI_USERNAME", "")
SMARTAPI_PASSWORD = os.getenv("SMARTAPI_PASSWORD", "")
SMARTAPI_APIKEY = os.getenv("SMARTAPI_APIKEY", "")
# SMARTAPI_BASE = "https://api.smartapi.in"  # example, set if known
SMARTAPI_BASE = os.getenv("SMARTAPI_BASE", "")  # better to set in .env

SYMBOL = os.getenv("SYMBOL", "RELIANCE")   # example
TIMEFRAME = os.getenv("TIMEFRAME", "1day") # used for historical fetch naming
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))  # backtest capital

# ======= Simple SmartAPI wrapper (template) =======
class SmartAPIClient:
    def __init__(self, username: str, password: str, api_key: str, base_url: str):
        self.username = username
        self.password = password
        self.api_key = api_key
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.client_login_data = {}
        self.feed_token = None
        self.auth_token = None

    def login(self) -> bool:
        """
        Implement login according to SmartAPI docs.
        For Angel One SmartAPI, typically:
         - POST /session with apiKey, clientId and password -> returns data -> access token
         - then generate feed token / session token
        You must adapt endpoints and payload per the docs.
        """
        if not self.base:
            raise ValueError("SMARTAPI_BASE not set in .env")
        # Example placeholder (modify as per actual SmartAPI endpoints)
        url = f"{self.base}/session"  # placeholder
        payload = {"clientId": self.username, "password": self.password, "apiKey": self.api_key}
        resp = self.session.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            print("Login failed:", resp.status_code, resp.text)
            return False
        data = resp.json()
        # Parse tokens as per actual response
        # Example:
        # self.auth_token = data["data"]["jwtToken"]
        # self.feed_token = data["data"]["feedToken"]
        self.client_login_data = data
        print("Login response:", data)
        # set auth header if present
        if self.auth_token:
            self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})
        return True

    def get_historical(self, symbol: str, from_date: str, to_date: str, interval: str = "1day") -> pd.DataFrame:
        """
        Fetch historical OHLCV data from SmartAPI.
        If you don't want to use API, prepare a CSV and pass to backtest.
        Return DataFrame with columns: ['date','open','high','low','close','volume']
        """
        # Placeholder URL - replace with real endpoint from SmartAPI docs
        url = f"{self.base}/historical"
        params = {"symbol": symbol, "from": from_date, "to": to_date, "interval": interval, "apiKey": self.api_key}
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Convert to DataFrame according to response shape
        # Example expected: data["data"]["candles"] = [[ts, open, high, low, close, volume], ...]
        if "data" in data and "candles" in data["data"]:
            candles = data["data"]["candles"]
            df = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"], unit="ms", errors="coerce")
            df = df.set_index("date")
            return df
        else:
            raise RuntimeError("Unexpected historical response: " + str(data))

    def place_order(self, symbol: str, qty: int, side: str, order_type: str = "MARKET", price: float = None) -> dict:
        """
        Place an order. side='BUY' or 'SELL'. Modify endpoint & payload per SmartAPI docs.
        For paper trading you can just simulate.
        """
        url = f"{self.base}/order"  # placeholder
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "price": price,
            "apiKey": self.api_key
        }
        resp = self.session.post(url, json=payload, timeout=10)
        if resp.status_code not in (200, 201):
            print("Order failed:", resp.status_code, resp.text)
        try:
            return resp.json()
        except:
            return {"status_code": resp.status_code, "text": resp.text}

# ======= Strategy: SMA crossover =======
@dataclass
class Signal:
    date: pd.Timestamp
    type: str  # "BUY" or "SELL"
    price: float

def generate_signals(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> List[Signal]:
    """Return list of buy/sell signals from SMA crossover"""
    data = df.copy()
    data["sma_fast"] = data["close"].rolling(fast).mean()
    data["sma_slow"] = data["close"].rolling(slow).mean()
    data = data.dropna()
    signals = []
    prev_pos = 0  # 1 if we hold
    for idx in range(1, len(data)):
        prev = data.iloc[idx - 1]
        cur = data.iloc[idx]
        # buy signal: fast crosses above slow
        if (prev.sma_fast <= prev.sma_slow) and (cur.sma_fast > cur.sma_slow):
            signals.append(Signal(date=cur.name, type="BUY", price=float(cur.close)))
        # sell signal: fast crosses below slow
        elif (prev.sma_fast >= prev.sma_slow) and (cur.sma_fast < cur.sma_slow):
            signals.append(Signal(date=cur.name, type="SELL", price=float(cur.close)))
    return signals

# ======= Backtester (vectorized simple) =======
def backtest(df: pd.DataFrame, initial_capital: float = 100000, qty_per_trade: int = 1, fast=10, slow=30):
    df = df.copy().sort_index()
    signals = generate_signals(df, fast, slow)
    cash = initial_capital
    position = 0
    trades = []
    nav_list = []
    for i, (ts, row) in enumerate(df.iterrows()):
        # check any signal at this date
        todays = [s for s in signals if s.date == ts]
        for s in todays:
            if s.type == "BUY" and cash >= s.price * qty_per_trade:
                cash -= s.price * qty_per_trade
                position += qty_per_trade
                trades.append({"date": ts, "type": "BUY", "price": s.price, "qty": qty_per_trade})
            elif s.type == "SELL" and position >= qty_per_trade:
                cash += s.price * qty_per_trade
                position -= qty_per_trade
                trades.append({"date": ts, "type": "SELL", "price": s.price, "qty": qty_per_trade})
        nav = cash + position * row.close
        nav_list.append({"date": ts, "nav": nav, "cash": cash, "position": position, "price": row.close})
    nav_df = pd.DataFrame(nav_list).set_index("date")
    returns = nav_df["nav"].pct_change().fillna(0)
    total_return = (nav_df["nav"].iloc[-1] / initial_capital - 1) * 100
    print(f"Backtest finished. Start capital: {initial_capital}, Final NAV: {nav_df['nav'].iloc[-1]:.2f}, Return: {total_return:.2f}%")
    return {"nav": nav_df, "trades": trades, "total_return_pct": total_return}

# ======= Paper trading runner (simulate or place real via API) =======
def run_paper_trading(client: SmartAPIClient, df: pd.DataFrame, qty_per_trade: int = 1, fast=10, slow=30, dry_run=True):
    """
    If dry_run=True, we only simulate orders locally (print).
    If dry_run=False, attempt to call client.place_order (ensure client.login() done).
    """
    signals = generate_signals(df, fast, slow)
    print(f"Signals: {len(signals)} found")
    for s in signals:
        print(f"{s.date} => {s.type} @ {s.price}")
        if dry_run:
            print("DRY RUN: not placing order.")
        else:
            side = s.type
            resp = client.place_order(symbol=SYMBOL, qty=qty_per_trade, side=side, order_type="MARKET")
            print("Order response:", resp)
        time.sleep(0.5)  # throttle

# ======= Helper: load CSV if API not available =======
def load_csv_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date")
    # ensure numeric columns
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

# ======= CLI-like entrypoint =======
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Trading bot template (backtest + paper)")
    parser.add_argument("--mode", choices=["backtest", "paper"], default="backtest")
    parser.add_argument("--csv", help="Use local CSV file (for backtest or paper). CSV must have columns date,open,high,low,close,volume")
    parser.add_argument("--from", dest="from_date", help="From date yyyy-mm-dd (for API historical fetch)")
    parser.add_argument("--to", dest="to_date", help="To date yyyy-mm-dd (for API historical fetch)")
    parser.add_argument("--fast", type=int, default=10)
    parser.add_argument("--slow", type=int, default=30)
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument("--dry", action="store_true", help="For paper: dry run (do not place real orders)")
    parser.add_argument("--initial", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()

    if args.mode == "backtest":
        if args.csv:
            df = load_csv_data(args.csv)
        else:
            # try to fetch from API (requires SMARTAPI_BASE, credentials, and date range)
            client = SmartAPIClient(SMARTAPI_USERNAME, SMARTAPI_PASSWORD, SMARTAPI_APIKEY, SMARTAPI_BASE)
            if not client.login():
                print("Cannot login to SmartAPI; provide CSV with --csv for backtest.")
                return
            df = client.get_historical(SYMBOL, args.from_date, args.to_date, interval=TIMEFRAME)

        res = backtest(df, initial_capital=args.initial, qty_per_trade=args.qty, fast=args.fast, slow=args.slow)
        # print trades summary
        print("Trades:")
        for t in res["trades"]:
            print(t)
        # save NAV to csv
        res["nav"].to_csv("nav_out.csv")
        print("NAV saved to nav_out.csv")

    elif args.mode == "paper":
        # need either CSV (with recent prices) or API
        if args.csv:
            df = load_csv_data(args.csv)
            client = None
        else:
            client = SmartAPIClient(SMARTAPI_USERNAME, SMARTAPI_PASSWORD, SMARTAPI_APIKEY, SMARTAPI_BASE)
            ok = client.login()
            if not ok:
                print("Login failed; aborting paper trading.")
                return
            df = client.get_historical(SYMBOL, args.from_date, args.to_date, interval=TIMEFRAME)

        run_paper_trading(client if client else SmartAPIClient("", "", "", ""), df, qty_per_trade=args.qty, fast=args.fast, slow=args.slow, dry_run=args.dry)

if __name__ == "__main__":
    main()
