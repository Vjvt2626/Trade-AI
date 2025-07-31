import json
import logging
import time
import os
from datetime import datetime, timedelta
import pandas as pd
import requests
import yfinance as yf
import numpy as np
from fyers_apiv3 import fyersModel
from openai import AzureOpenAI
import asyncio
import threading
import signal
from newsapi import NewsApiClient

# --- Configuration and Initialization ---

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Load Fyers access token from token.txt
# This will still be loaded even in backtest, but MockFyersModel won't use it
with open('token.txt', 'r') as f:
    access_token = f.read().strip()

# Ensure log directory exists
log_dir = "./fyers_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging with console output - MOVED TO HERE
logging.basicConfig(
    filename='trade_logs.json',
    level=logging.INFO, # Set to logging.DEBUG for more verbose output during debugging
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Set to logging.DEBUG for more console output
logger.addHandler(console_handler)

# Global stop flag and backtest mode
stop_event = threading.Event()
backtest_mode = True  # Set to True for backtesting with historical data
backtest_start = datetime(2025, 6, 1)  # Adjusted to a historical range for backtesting
backtest_end = datetime(2025, 6, 15)  # Adjusted to a historical range for backtesting

# --- Mocking Classes for Backtesting ---
# These classes simulate the behavior of external APIs without making actual network calls.
# This is crucial for efficient and repeatable backtesting.

class MockFyersModel:
    """
    A mock class to simulate Fyers API calls during backtesting.
    """
    def __init__(self, client_id, token, log_path):
        logger.info("Initializing Mock Fyers Model for backtesting.")
        self.orders = {}
        self.order_id_counter = 0

    def quotes(self, data):
        symbol = data.get("symbol")
        logger.debug(f"Mock Fyers: Fetching quote for {symbol}")
        # Return mock data based on symbol type
        if "NIFTY-I" in symbol or "BANKNIFTY-I" in symbol:
            # Simple mock for index futures/spot
            return {"s": "ok", "d": [{"v": {"ltp": 22000 if "NIFTY" in symbol else 48000}}]}
        elif "CE" in symbol or "PE" in symbol:
            # Mock for options - for realistic backtesting, this would need
            # to fetch or calculate historical option data based on underlying, IV etc.
            # For now, it returns a fixed mock value.
            return {"s": "ok", "d": [{"v": {"ltp": 50.0, "iv": 15.0, "oi": 1000, "low_price": 45.0, "high_price": 55.0}}]}
        return {"s": "error", "message": "Invalid symbol"}

    def history(self, data):
        logger.debug(f"Mock Fyers: Fetching historical data for {data.get('symbol')}")
        # For historical options, this would need a more complex mock or actual data source
        return {"code": 200, "candles": []}

    def place_order(self, data):
        self.order_id_counter += 1
        order_id = f"MOCK_ORDER_{self.order_id_counter}"
        logger.info(f"Mock Fyers: Placing order {order_id} for {data.get('symbol')} Qty: {data.get('qty')} (Backtest)")
        self.orders[order_id] = data
        return {"s": "ok", "id": order_id, "message": "Order placed successfully (mock)"}


class MockAzureOpenAIClient:
    """
    A mock class to simulate Azure OpenAI API calls during backtesting.
    """
    def __init__(self, api_key, api_version, azure_endpoint, azure_deployment):
        logger.info("Initializing Mock Azure OpenAI Client for backtesting.")
        self._chat_instance = self.Chat() # Instantiate the Chat class once

    class Chat:
        def __init__(self):
            # Instantiate the Completions class once within Chat
            self.completions = self.Completions()

        class Completions:
            def create(self, model, messages, temperature):
                # Simulate sentiment based on keywords in the prompt
                content_str = messages[0]["content"].lower()
                if "upward" in content_str or "bullish" in content_str:
                    sentiment = "Bullish"
                    confidence = 0.8
                elif "downward" in content_str or "bearish" in content_str:
                    sentiment = "Bearish"
                    confidence = 0.8
                else:
                    sentiment = "Neutral"
                    confidence = 0.5
                
                # --- CORRECTED MOCK RESPONSE STRUCTURE (THIS IS THE KEY CHANGE) ---
                # 1. Create the object that will represent the 'message' part
                mock_message_object = type('MockMessage', (object,), {
                    'content': json.dumps({"sentiment": sentiment, "confidence": confidence}),
                    'role': 'assistant' # Typically, the model's message has a role
                })()

                # 2. Create the object that will represent a 'choice' in the choices list
                mock_choice_object = type('MockChoice', (object,), {
                    'message': mock_message_object, # The 'message' attribute holds the mock_message_object
                    'finish_reason': 'stop', # Optional, but good to mimic
                    'index': 0 # Optional, but good to mimic
                })()

                # 3. Create the top-level response object
                mock_completion_response_object = type('MockCompletionResponse', (object,), {
                    'choices': [mock_choice_object], # The 'choices' attribute is a list of mock_choice_objects
                    'id': 'mock_id_123', # Optional top-level attributes
                    'model': model,
                    'created': int(time.time())
                })()

                logger.debug(f"Mock OpenAI: Returning sentiment {sentiment} with confidence {confidence}")
                return mock_completion_response_object # Return the fully structured mock object

    @property
    def chat(self):
        # When client.chat is accessed, return the instantiated _chat_instance
        return self._chat_instance

class MockNewsApiClient:
    """
    A mock class to simulate NewsAPI calls during backtesting.
    """
    def __init__(self, api_key):
        logger.info("Initializing Mock News API Client for backtesting.")

    def get_everything(self, q, language, page_size):
        logger.debug(f"Mock News API: Fetching news for {q}")
        # Return fixed mock articles for backtesting
        return {"articles": [{"title": f"Mock News for {q}: Market is steady."} for _ in range(page_size)]}

# --- API Initialization ---

# Initialize APIs based on whether backtest mode is enabled
if backtest_mode:
    fyers = MockFyersModel(client_id=config["fyers_client_id"], token=access_token, log_path=log_dir)
    client = MockAzureOpenAIClient(
        api_key=config["openai_api_key"],
        api_version=config["openai_api_version"],
        azure_endpoint=config["openai_api_base"],
        azure_deployment=config["openai_deployment"]
    )
    newsapi = MockNewsApiClient(api_key=config["newsapi_key"])
else:
    fyers = fyersModel.FyersModel(client_id=config["fyers_client_id"], token=access_token, log_path=log_dir)
    client = AzureOpenAI(
        api_key=config["openai_api_key"],
        api_version=config["openai_api_version"],
        azure_endpoint=config["openai_api_base"],
        azure_deployment=config["openai_deployment"]
    )
    newsapi = NewsApiClient(api_key=config["newsapi_key"])

# --- Data Service Class ---

class DataService:
    """
    Handles fetching market data and calculating technical indicators.
    """
    def get_yfinance_symbol(self, instrument):
        """Maps an instrument name to its Yahoo Finance ticker symbol."""
        return {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}.get(instrument.upper(), "^NSEI")

    async def get_rsi_and_trend(self, instrument, historical_data=None):
        logger.info(f"Fetching RSI and trend for {instrument}")
        yf_symbol = self.get_yfinance_symbol(instrument)
        data = historical_data

        if data is None or data.empty:
            try:
                data = yf.download(yf_symbol, period="60d", interval="1d", progress=False, prepost=True)
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)
            except Exception as e:
                logger.error(f"Failed to download yfinance data for {yf_symbol}: {e}", exc_info=True)
                return 50.0, "Neutral"

        if 'Close' not in data.columns:
            logger.warning(f"No 'Close' column found in yfinance data for {yf_symbol}.")
            return 50.0, "Neutral"
        
        if isinstance(data['Close'], pd.DataFrame):
            close_prices_series_raw = data['Close'].iloc[:, 0]
            logger.warning(f"data['Close'] was a DataFrame; extracted first column for prices.")
        else:
            close_prices_series_raw = data['Close']

        close_prices = pd.to_numeric(close_prices_series_raw, errors='coerce').dropna().to_numpy()

        if len(close_prices) < 15:
            logger.warning(f"Insufficient numeric close prices after cleaning for RSI/Trend calculation for {yf_symbol}. Only {len(close_prices)} data points.")
            return 50.0, "Neutral"

        # Calculate RSI (Wilder's Smoothing Method)
        gains, losses = [], []
        for i in range(1, 15):
            diff = close_prices[i] - close_prices[i-1]
            gains.append(float(max(0, diff)))
            losses.append(float(max(0, -diff)))

        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0

        rsi_values = [50.0] * len(close_prices)

        if avg_loss == 0:
            rs = np.inf
        else:
            rs = avg_gain / avg_loss # CORRECTED LINE: Used avg_loss instead of rs_loss
        
        if len(close_prices) >= 14:
            rsi_values[13] = 100 - (100 / (1 + rs))

        for i in range(15, len(close_prices)):
            diff = close_prices[i] - close_prices[i-1]
            gain_val = max(0, diff)
            loss_val = max(0, -diff)
            avg_gain = ((avg_gain * 13) + gain_val) / 14
            avg_loss = ((avg_loss * 13) + loss_val) / 14
            if avg_loss == 0:
                rs = np.inf
            else:
                rs = avg_gain / avg_loss # CORRECTED LINE: Used avg_loss again
            rsi_values[i] = 100 - (100 / (1 + rs))
        
        rsi = round(rsi_values[-1], 2) if len(rsi_values) > 13 else 50.0

        if len(close_prices) < 10:
            trend = "Neutral"
        else:
            ma5 = np.mean(close_prices[-5:])
            ma10 = np.mean(close_prices[-10:])
            trend = "Up" if ma5 > ma10 else "Down" if ma5 < ma10 else "Neutral"

        logger.info(f"RSI: {rsi}, Trend: {trend} for {instrument}")
        return rsi, trend

    async def _get_last_close(self, symbol):
        """Fetches the last closing price for a given symbol, primarily for fallback."""
        logger.info(f"Fetching last close for {symbol}")
        try:
            if symbol.startswith("NSE:"):
                # Extract instrument name, remove "FUT" if present
                instrument_name = symbol.split(":")[1].split("-")[0].replace("FUT", "")
                yf_symbol = self.get_yfinance_symbol(instrument_name)
                
                # For options, a mock last close is appropriate for backtesting as real data isn't easily available
                if "CE" in symbol or "PE" in symbol:
                    return 50.0 # Mock value for options
                
                data = yf.download(yf_symbol, period="1d", interval="1d", progress=False, prepost=True)
                return data['Close'].iloc[-1] if not data.empty else 0
            return 0
        except Exception as e:
            logger.error(f"Failed to get last close for {symbol}: {e}")
            return 0

    async def get_historical_ohlc(self, symbol, days=10):
        """Fetches historical OHLC data from Fyers (only in live mode)."""
        logger.info(f"Fetching historical OHLC for {symbol}")
        if not backtest_mode:
            try:
                if symbol.startswith("NSE:"):
                    hist_data = fyers.history({
                        "symbol": symbol,
                        "resolution": "5",
                        "date_format": "0",
                        "range_from": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
                        "range_to": datetime.now().strftime("%Y-%m-%d"),
                        "cont_flag": "1"
                    })
                    if hist_data.get("code") == 200 and hist_data.get("candles"):
                        df = pd.DataFrame(hist_data["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                        return df.set_index("timestamp")
            except Exception as e:
                logger.error(f"Failed to fetch historical data for {symbol}: {e}")
        return pd.DataFrame()

# --- Utility Functions ---

def get_next_weekly_expiry_date(today=None, instrument="NIFTY"):
    """
    Calculates the next weekly expiry date for NIFTY (Thursday) or BANKNIFTY (Wednesday).
    Takes 'today' as an argument for backtesting purposes.
    """
    if today is None:
        today = datetime.now().date()
    
    # Fyers weekly expiry days: NIFTY=Thursday (3), BANKNIFTY=Wednesday (2)
    target_day = 3 if instrument.upper() == "NIFTY" else 2
    
    days_to_target = (target_day - today.weekday() + 7) % 7
    
    # If today is the target day and it's past market close, move to next week's expiry
    # This check is only relevant for live trading, not typically for daily backtesting loops.
    if days_to_target == 0 and datetime.now().time().hour >= 15 and not backtest_mode:
        days_to_target += 7
        
    return today + timedelta(days=days_to_target)

def format_fyers_symbol(instrument, strike, opt_type, expiry):
    """
    Formats the Fyers symbol string for options contracts.
    Fyers weekly options format is typically: INSTRUMENTYYMDDSTRIKEOPTTYPE
    where M is month number (1-9), 'O' for Oct, 'N' for Nov, 'D' for Dec.
    """
    # Convert month number to Fyers' specific single character format
    month_num = expiry.month
    if month_num < 10:
        month_char = str(month_num)
    elif month_num == 10:
        month_char = 'O'
    elif month_num == 11:
        month_char = 'N'
    elif month_num == 12:
        month_char = 'D'
    
    fyers_expiry_format = f"{expiry.strftime('%y')}{month_char}{expiry.strftime('%d')}"
    return f"NSE:{instrument.upper()}{fyers_expiry_format}{int(strike)}{opt_type.upper()}"


async def fetch_expiries(fyers, instrument):
    """Fetches valid expiries for an instrument (mocked in backtest)."""
    logger.info(f"Fetching expiries for {instrument}")
    try:
        calculated_expiry = get_next_weekly_expiry_date(instrument=instrument)
        # For simplicity in backtest, we primarily rely on the calculated expiry.
        # In a real scenario, you might query Fyers for available expiries.
        if backtest_mode:
            logger.info(f"Backtest mode: Mocking fetch_expiries, returning {calculated_expiry.date()}")
            return [calculated_expiry]
        
        # For live trading, attempt to validate a sample symbol to ensure expiry is valid
        sample_strike = 25000 if instrument.upper() == "NIFTY" else 50000
        sample_symbol = format_fyers_symbol(instrument, sample_strike, "CE", calculated_expiry)
        response = fyers.quotes({"symbol": sample_symbol}) # "type": "symbolData" is not strictly needed for basic quote
        if response.get("s") == "ok" and response["d"][0].get("v", {}).get("ltp", 0) > 0:
            return [calculated_expiry]
        
        logger.warning(f"No valid market data for {sample_symbol}. Using calculated expiry. Response: {response.get('message')}")
        return [calculated_expiry]
    except Exception as e:
        logger.error(f"Error fetching expiries for {instrument}: {e}")
        return [get_next_weekly_expiry_date(instrument=instrument)] # Fallback in case of error


async def fetch_market_data(fyers, symbol):
    """Fetches live market data for a given symbol (mocked in backtest)."""
    logger.info(f"Fetching market data for {symbol}")
    for attempt in range(3):
        try:
            response = fyers.quotes({"symbol": symbol, "type": "symbolData"})
            if response.get("s") == "ok" and response["d"][0].get("v", {}).get("ltp", 0) > 0:
                quote = response["d"][0]["v"]
                return {
                    "ltp": quote.get("ltp", 0),
                    "iv": quote.get("iv", 15.0), # Implied Volatility
                    "oi": quote.get("oi", 0), # Open Interest
                    "low": quote.get("low_price", 0),
                    "high": quote.get("high_price", 0)
                }
            logger.warning(f"Attempt {attempt + 1}/3 failed for {symbol}: {response.get('message')}. Retrying...")
            await asyncio.sleep(2 ** attempt) # Exponential backoff
        except Exception as e:
            logger.error(f"Market data error for {symbol} on attempt {attempt + 1}: {e}")
            await asyncio.sleep(2 ** attempt)
    logger.error(f"Failed to fetch market data for {symbol} after 3 attempts.")
    return {"ltp": 0, "iv": 15.0, "oi": 0, "low": 0, "high": 0}

async def get_sentiment(news, client):
    """
    Analyzes news articles to determine market sentiment using Azure OpenAI.
    Uses mock client in backtest mode.
    """
    logger.info("Fetching sentiment from news via AI.")
    try:
        # Check if client is a mock client (for backtesting)
        if isinstance(client, MockAzureOpenAIClient):
            # Pass relevant data for mock sentiment logic
            articles_content = [a["title"] for a in news[:5]]
            messages = [{"role": "system", "content": f"Articles: {json.dumps(articles_content)}"}]
            mock_chat_completion = client.chat.completions.create(None, messages, None)
            result = json.loads(mock_chat_completion.message.content)
            logger.info(f"Mock Sentiment: {result['sentiment']}, Confidence: {result['confidence']}")
            return result["sentiment"], result["confidence"]

        # Real Azure OpenAI API call
        prompt = f"""
        Analyze the following news articles for market sentiment.
        Return a JSON object: {{"sentiment": "Bullish/Bearish/Neutral", "confidence": float (0-1)}}.
        Articles: {json.dumps([a["title"] + ": " + (a.get("description") or "") for a in news[:5]])}
        """
        response = client.chat.completions.create(
            model=config["openai_deployment"],
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3 # Lower temperature for more deterministic output
        )
        result = json.loads(response.choices[0].message.content)
        logger.info(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']}")
        return result["sentiment"], result["confidence"]
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return "Neutral", 0.5 # Default to neutral if error occurs


async def build_market_data_chain(fyers, instrument, expiry, strikes, data_svc):
    """
    Builds a list of option contract data (LTP, OI, IV) for given strikes and expiry.
    Mocks data generation in backtest mode.
    """
    logger.info(f"Building market data chain for {instrument}, Expiry: {expiry.strftime('%Y-%m-%d')}, Strikes: {strikes}")
    market_data_chain = []
    if backtest_mode:
        logger.debug(f"Backtest mode: Generating mock market data for {len(strikes)} strikes.")
        if not strikes:
            logger.warning("No strikes available for mock data generation in backtest.")
        else:
            # Simple mock option pricing: center around ATM and spread out
            # For a more realistic backtest, this would require an option pricing model
            # like Black-Scholes using historical underlying prices and IV.
            atm_strike_index = len(strikes) // 2 # Approximate ATM strike in our list
            for strike in strikes:
                for opt_type in ["CE", "PE"]:
                    symbol = format_fyers_symbol(instrument, strike, opt_type, expiry)
                    
                    # Basic mock LTP simulation: calls increase with distance from ATM
                    # This is highly simplified and not indicative of real option behavior.
                    base_ltp = 50.0
                    if opt_type == "CE":
                        # Calls generally decrease as strike increases
                        mock_ltp = max(5.0, base_ltp - (strike - strikes[atm_strike_index]) * 0.1)
                    else: # PE
                        # Puts generally decrease as strike decreases
                        mock_ltp = max(5.0, base_ltp + (strike - strikes[atm_strike_index]) * 0.1)

                    market_data_chain.append({
                        "symbol": symbol,
                        "strike": strike,
                        "type": opt_type,
                        "ltp": round(mock_ltp, 2),
                        "iv": 15.0, # Mock IV
                        "oi": 1000, # Mock OI
                        "low": round(mock_ltp * 0.9, 2),
                        "high": round(mock_ltp * 1.1, 2)
                    })
    else:
        tasks = []
        for strike in strikes:
            for opt_type in ["CE", "PE"]:
                symbol = format_fyers_symbol(instrument, strike, opt_type, expiry)
                tasks.append(fetch_market_data(fyers, symbol)) # Actual Fyers API calls
        
        results = await asyncio.gather(*tasks)
        
        # Combine strikes and types to match the order of results
        all_options = [(s, t) for s in strikes for t in ["CE", "PE"]]

        for i, (strike, opt_type) in enumerate(all_options):
            symbol = format_fyers_symbol(instrument, strike, opt_type, expiry)
            data = results[i]
            if data["ltp"] > 0:
                market_data_chain.append({"symbol": symbol, "strike": strike, "type": opt_type, **data})
            else:
                # Fallback to last close if live LTP is not available
                fallback_ltp = await data_svc._get_last_close(symbol)
                if fallback_ltp > 0:
                    market_data_chain.append({"symbol": symbol, "strike": strike, "type": opt_type, "ltp": fallback_ltp, "iv": data["iv"], "oi": data["oi"], "low": 0, "high": 0})
                else:
                    logger.warning(f"Could not get market data or fallback for {symbol}. Skipping this option.")

    logger.info(f"Built market data chain with {len(market_data_chain)} entries.")
    return market_data_chain

async def suggest_trades(data_svc, instrument, market_data_chain, logs):
    """
    Suggests potential trades based on technical indicators and sentiment.
    """
    logger.info(f"Suggesting trades for {instrument}. Logs: {logs}")
    try:
        sentiment, sentiment_confidence = logs["sentiment"], logs["sentiment_confidence"]
        rsi = logs["rsi"]
        short_ma = logs.get("short_ma", 0)
        long_ma = logs.get("long_ma", 0)
        trend_10_50_ma = logs.get("trend_10_50_ma", "Neutral") # Get the 10/50 MA trend

        suggestions = []
        for data in market_data_chain:
            if data.get("ltp", 0) > 0:
                # Basic confidence derived from sentiment confidence
                confidence = min(sentiment_confidence, 0.9) if sentiment in ["Bullish", "Bearish"] else 0.5
                
                trade_direction = "Neutral"
                
                # Strategy 1: MACD-like Crossover + RSI + Sentiment
                if short_ma > long_ma and trend_10_50_ma == "Up": # Bullish crossover
                    if rsi < 70 and sentiment == "Bullish": # Not overbought and bullish sentiment
                        trade_direction = "Buy_CE"
                elif short_ma < long_ma and trend_10_50_ma == "Down": # Bearish crossover
                    if rsi > 30 and sentiment == "Bearish": # Not oversold and bearish sentiment
                        trade_direction = "Buy_PE"
                
                # Strategy 2: RSI Extremes + Sentiment Confirmation
                if trade_direction == "Neutral": # Only apply if no strong signal yet
                    if rsi < 30 and sentiment == "Bullish" and data["type"] == "CE": # Oversold and bullish sentiment
                        trade_direction = "Buy_CE"
                    elif rsi > 70 and sentiment == "Bearish" and data["type"] == "PE": # Overbought and bearish sentiment
                        trade_direction = "Buy_PE"
                
                if trade_direction == "Buy_CE" and data["type"] == "CE":
                    suggestions.append({
                        "symbol": data["symbol"],
                        "strike": data["strike"],
                        "type": data["type"],
                        "ltp": data["ltp"],
                        "confidence": confidence # You might want to refine confidence based on multiple signals
                    })
                elif trade_direction == "Buy_PE" and data["type"] == "PE":
                    suggestions.append({
                        "symbol": data["symbol"],
                        "strike": data["strike"],
                        "type": data["type"],
                        "ltp": data["ltp"],
                        "confidence": confidence
                    })
        
        logger.info(f"Generated {len(suggestions)} trade suggestions.")
        return suggestions
    except Exception as e:
        logger.error(f"Failed to parse trade suggestions: {e}")
        return []

async def execute_trade(fyers, best_suggestion, hedge_data, qty, logs):
    """
    Places the main trade and a hedging trade (mocked in backtest).
    """
    logger.info(f"Executing trade for {best_suggestion['symbol']} (Hedge: {hedge_data['symbol']}) with mock Fyers: {isinstance(fyers, MockFyersModel)}")
    
    # Place main order
    order = fyers.place_order(
        data={
            "symbol": best_suggestion["symbol"],
            "qty": qty,
            "side": 1 if best_suggestion["type"] == "CE" else -1, # 1 for Buy (CE), -1 for Sell (PE)
            "type": 2, # Limit order (adjust as needed for strategy, 1 for market)
            "limitPrice": best_suggestion["ltp"] * 1.01, # Set a small buffer for limit price
            "productType": "INTRADAY"
        }
    )
    if order.get("s") == "ok":
        logger.info(json.dumps({"event": "trade_entry", "main_symbol": best_suggestion["symbol"], 
                                "qty": qty, "entry_price": best_suggestion["ltp"], 
                                "order_id": order.get("id"), **logs}))
        
        # Place hedging order (opposite side)
        hedge_order = fyers.place_order(
            data={
                "symbol": hedge_data["symbol"],
                "qty": qty,
                "side": -1 if best_suggestion["type"] == "CE" else 1, # Sell hedge if main is CE, Buy hedge if main is PE
                "type": 2, # Limit order
                "limitPrice": hedge_data["ltp"] * 0.99 if best_suggestion["type"] == "CE" else hedge_data["ltp"] * 1.01, # Adjust hedge limit
                "productType": "INTRADAY"
            }
        )
        if hedge_order.get("s") == "ok":
            logger.info(json.dumps({"event": "hedge_entry", "hedge_symbol": hedge_data["symbol"], 
                                    "qty": qty, "entry_price": hedge_data["ltp"], 
                                    "order_id": hedge_order.get("id"), **logs}))
            # Proceed to monitor the trade after successful entry of both legs
            return await monitor_trade(fyers, best_suggestion, hedge_data, qty, logs)
        else:
            logger.error(f"Failed to execute hedge trade for {hedge_data['symbol']}. Hedge order response: {hedge_order}. Attempting to cancel main order.")
            # In a real system, you'd try to cancel the main order if hedge fails
            # fyers.cancel_order({"id": order.get("id")})
            return False
    else:
        logger.error(f"Failed to execute main trade for {best_suggestion['symbol']}. Main order response: {order}")
        return False

async def monitor_trade(fyers, best_suggestion, hedge_data, qty, logs):
    """
    Monitors an active trade for profit target or stop loss (mocked in backtest).
    """
    target_price = best_suggestion["ltp"] * (1 + config["profit_target"])
    stop_loss_price = best_suggestion["ltp"] * (1 - config["risk_per_trade"])
    
    # In backtest mode, simulate immediate exit based on fixed or mocked prices
    if backtest_mode:
        logger.info(f"Backtest mode: Simulating trade monitoring for {best_suggestion['symbol']}. Assuming immediate resolution.")
        
        # For a truly detailed backtest, you'd iterate through historical minute/tick data here.
        # This simplified mock assumes the trade immediately hits profit or stop loss.
        
        # You could add some randomness or logic based on indicators here
        # For instance, if overall trend is strong up, assume profit met more often.
        
        # Example: Assume profit is met in 70% of backtest trades, SL in 30%
        # This is for demonstration. A proper backtest would use historical price data.
        import random
        if random.random() < 0.7: # Simulate profit
            final_exit_price_main = target_price
            reason = "profit_target"
        else: # Simulate stop loss
            final_exit_price_main = stop_loss_price
            reason = "stop_loss"
        
        # In a real backtest, hedge exit price would also be based on its movement
        # For simplicity, let's assume it moves proportionally opposite or just exit at entry price for now.
        final_exit_price_hedge = hedge_data["ltp"] # Simplified for mock
            
        profit_loss_main = (final_exit_price_main - best_suggestion["ltp"]) * qty
        profit_loss_hedge = (final_exit_price_hedge - hedge_data["ltp"]) * qty * (-1 if best_suggestion["type"] == "CE" else 1) # Hedge profit/loss is opposite
        
        total_profit = (profit_loss_main + profit_loss_hedge) * config["lot_size_multiplier"] # Apply lot size multiplier once
        
        logger.info(json.dumps({"event": "backtest_trade_exit", 
                                "main_symbol": best_suggestion["symbol"], "qty": qty, 
                                "entry_price_main": best_suggestion["ltp"], "exit_price_main": final_exit_price_main,
                                "hedge_symbol": hedge_data["symbol"],
                                "entry_price_hedge": hedge_data["ltp"], "exit_price_hedge": final_exit_price_hedge,
                                "profit": total_profit, "reason": reason, **logs}))
        return True # Indicate trade completed in backtest
            
    # Live mode monitoring (existing logic, with minor improvements)
    start_time = time.time()
    trade_duration_limit = 300 # 5 minutes
    
    while time.time() - start_time < trade_duration_limit:
        current_data = await fetch_market_data(fyers, best_suggestion["symbol"])
        current_ltp = current_data["ltp"]
        
        if current_ltp <= 0:
            logger.warning(f"Current LTP for {best_suggestion['symbol']} is 0 or invalid, continuing monitoring.")
            await asyncio.sleep(5) # Shorter sleep if no valid data
            continue

        if current_ltp >= target_price:
            logger.info(f"Profit target reached for {best_suggestion['symbol']}. Exiting trade.")
            exit_order = fyers.place_order(
                data={
                    "symbol": best_suggestion["symbol"],
                    "qty": qty,
                    "side": -1 if best_suggestion["type"] == "CE" else 1, # Sell to exit
                    "type": 1, # Market order for quick exit
                    "productType": "INTRADAY"
                }
            )
            hedge_exit = fyers.place_order(
                data={
                    "symbol": hedge_data["symbol"],
                    "qty": qty,
                    "side": 1 if best_suggestion["type"] == "CE" else -1, # Buy to exit hedge
                    "type": 1, # Market order for quick exit
                    "productType": "INTRADAY"
                }
            )
            if exit_order.get("s") == "ok" and hedge_exit.get("s") == "ok":
                final_ltp_main = (await fetch_market_data(fyers, best_suggestion["symbol"]))["ltp"] # Get actual exit LTP
                final_ltp_hedge = (await fetch_market_data(fyers, hedge_data["symbol"]))["ltp"]
                
                profit_loss_main = (final_ltp_main - best_suggestion["ltp"]) * qty
                profit_loss_hedge = (final_ltp_hedge - hedge_data["ltp"]) * qty * (-1 if best_suggestion["type"] == "CE" else 1) # Hedge P/L is opposite
                total_profit = (profit_loss_main + profit_loss_hedge) * config["lot_size_multiplier"]

                logger.info(json.dumps({"event": "trade_exit", "main_symbol": best_suggestion["symbol"], "qty": qty, 
                                        "entry_price_main": best_suggestion["ltp"], "exit_price_main": final_ltp_main,
                                        "hedge_symbol": hedge_data["symbol"],
                                        "entry_price_hedge": hedge_data["ltp"], "exit_price_hedge": final_ltp_hedge,
                                        "profit": total_profit, "reason": "profit_target", **logs}))
                return True
            else:
                logger.error(f"Failed to exit trade for profit: Main exit: {exit_order}, Hedge exit: {hedge_exit}")
                return False # Failed to exit gracefully
        
        elif current_ltp <= stop_loss_price:
            logger.info(f"Stop loss hit for {best_suggestion['symbol']}. Exiting trade.")
            exit_order = fyers.place_order(
                data={
                    "symbol": best_suggestion["symbol"],
                    "qty": qty,
                    "side": -1 if best_suggestion["type"] == "CE" else 1, # Sell to exit
                    "type": 1, # Market order
                    "productType": "INTRADAY"
                }
            )
            hedge_exit = fyers.place_order(
                data={
                    "symbol": hedge_data["symbol"],
                    "qty": qty,
                    "side": 1 if best_suggestion["type"] == "CE" else -1, # Buy to exit hedge
                    "type": 1, # Market order
                    "productType": "INTRADAY"
                }
            )
            if exit_order.get("s") == "ok" and hedge_exit.get("s") == "ok":
                final_ltp_main = (await fetch_market_data(fyers, best_suggestion["symbol"]))["ltp"]
                final_ltp_hedge = (await fetch_market_data(fyers, hedge_data["symbol"]))["ltp"]

                profit_loss_main = (final_ltp_main - best_suggestion["ltp"]) * qty
                profit_loss_hedge = (final_ltp_hedge - hedge_data["ltp"]) * qty * (-1 if best_suggestion["type"] == "CE" else 1)
                total_profit = (profit_loss_main + profit_loss_hedge) * config["lot_size_multiplier"]

                logger.info(json.dumps({"event": "trade_exit", "main_symbol": best_suggestion["symbol"], "qty": qty, 
                                        "entry_price_main": best_suggestion["ltp"], "exit_price_main": final_ltp_main,
                                        "hedge_symbol": hedge_data["symbol"],
                                        "entry_price_hedge": hedge_data["ltp"], "exit_price_hedge": final_ltp_hedge,
                                        "profit": total_profit, "reason": "stop_loss", **logs}))
                return True
            else:
                logger.error(f"Failed to exit trade for stop loss: Main exit: {exit_order}, Hedge exit: {hedge_exit}")
                return False
        
        await asyncio.sleep(0.2) # Check every 200ms
        
    logger.warning(f"Trade timeout for {best_suggestion['symbol']}, forcing market exit (live mode).")
    # Forcing exit at market price if timeout occurs
    exit_order = fyers.place_order(
        data={
            "symbol": best_suggestion["symbol"],
            "qty": qty,
            "side": -1 if best_suggestion["type"] == "CE" else 1,
            "type": 1, # Market order
            "productType": "INTRADAY"
        }
    )
    hedge_exit = fyers.place_order(
        data={
            "symbol": hedge_data["symbol"],
            "qty": qty,
            "side": 1 if best_suggestion["type"] == "CE" else -1,
            "type": 1, # Market order
            "productType": "INTRADAY"
        }
    )
    if exit_order.get("s") == "ok" and hedge_exit.get("s") == "ok":
        current_ltp_main = (await fetch_market_data(fyers, best_suggestion["symbol"]))["ltp"]
        current_ltp_hedge = (await fetch_market_data(fyers, hedge_data["symbol"]))["ltp"]

        profit_loss_main = (current_ltp_main - best_suggestion["ltp"]) * qty
        profit_loss_hedge = (current_ltp_hedge - hedge_data["ltp"]) * qty * (-1 if best_suggestion["type"] == "CE" else 1)
        total_profit = (profit_loss_main + profit_loss_hedge) * config["lot_size_multiplier"]

        logger.info(json.dumps({"event": "trade_exit", "main_symbol": best_suggestion["symbol"], "qty": qty, 
                                "entry_price_main": best_suggestion["ltp"], "exit_price_main": current_ltp_main,
                                "hedge_symbol": hedge_data["symbol"],
                                "entry_price_hedge": hedge_data["ltp"], "exit_price_hedge": current_ltp_hedge,
                                "profit": total_profit, "reason": "timeout", **logs}))
    return True

# --- Main Trading Loop ---

async def main_loop():
    """
    The main loop that orchestrates data fetching, analysis, and trade execution.
    Handles both live trading and backtesting modes.
    """
    logger.info("Starting main trading loop.")
    instrument = config["instrument"]
    data_svc = DataService()
    capital = config["capital"]
    risk_per_trade = config["risk_per_trade"]
    profit_target = config["profit_target"]
    min_confidence = config["min_confidence"]
    max_trades = config["max_trades_per_day"]
    trade_count = 0

    if backtest_mode:
        # Generate daily dates for backtesting, excluding weekends
        dates = pd.date_range(start=backtest_start.date(), end=backtest_end.date(), freq="B").tolist() # 'B' for business day frequency
        logger.info(f"Backtesting from {backtest_start.date()} to {backtest_end.date()} ({len(dates)} trading days).")
    else:
        dates = [datetime.now()] # For live trading, just run for current time

    for current_date_or_time in dates:
        if stop_event.is_set():
            logger.info("Stop event received. Exiting main loop.")
            break

        current_time_for_logic = current_date_or_time if backtest_mode else datetime.now()
        
        logger.info(f"--- Processing Day/Time: {current_time_for_logic.strftime('%Y-%m-%d %H:%M')} ---")
        
        # In live mode, check market hours. In backtest, we process each valid day.
        if not backtest_mode and not (current_time_for_logic.weekday() < 5 and # Monday-Friday
                                    9 * 3600 + 15 * 60 <= current_time_for_logic.hour * 3600 + current_time_for_logic.minute * 60 <= 15 * 3600 + 30 * 60):
            logger.info("Outside trading hours in live mode, sleeping for 60 seconds.")
            await asyncio.sleep(config["live_mode_sleep_interval"])
            continue
        
        # --- Fetch Underlying Data ---
        yf_symbol_for_underlying = data_svc.get_yfinance_symbol(instrument)
        yf_data = pd.DataFrame() # Initialize yf_data
        ltp = 0

        if backtest_mode:
            # For backtest, get enough historical data for MA and RSI calculations up to the current backtest day
            # We need data PRIOR to current_time_for_logic to calculate indicators for *that* day.
            # So, fetch data up to the *end* of the current day being processed.
            temp_end_date = current_time_for_logic + timedelta(days=1) # Fetch up to (but not including) next day's start
            temp_start_date = current_time_for_logic - timedelta(days=90) # Sufficient history for 50-day MA + RSI
            
            try:
                # Fetch daily data for the relevant period
                # Ensure the end date covers the current day for .asof() to pick it up
                temp_end_date = current_time_for_logic + timedelta(days=1)
                temp_start_date = current_time_for_logic - timedelta(days=90)
                
                yf_data = yf.download(yf_symbol_for_underlying, start=temp_start_date.date(), end=temp_end_date.date(), interval="1d", progress=False, prepost=True, auto_adjust=True)
                
                if yf_data.empty or 'Close' not in yf_data.columns:
                    logger.warning(f"No valid yfinance data for {yf_symbol_for_underlying} up to {current_time_for_logic.date()}.")
                    ltp = 0
                else:
                    # Get the closest available close price for the current backtest day
                    # Use .asof() to get the row, then .iloc[0] or .item() to get the scalar
                    # .item() is safer if you expect a single scalar result, otherwise .iloc[0] for the first element
                    ltp_series_result = yf_data['Close'].asof(current_time_for_logic)
                    
                    if ltp_series_result.empty: # Check if the result Series is empty
                         # Fallback to the last available close if the exact date is missing (e.g., holidays)
                         ltp = yf_data['Close'].iloc[-1] if not yf_data.empty else 0
                         logger.warning(f"No exact close for {current_time_for_logic.date()}, using last available: {ltp}")
                    else:
                        # Extract the scalar value from the Series
                        ltp = ltp_series_result.item()
                    
                    logger.info(f"Retrieved LTP (backtest): {ltp} for {current_time_for_logic.date()}")

            except Exception as e:
                logger.error(f"Yfinance error in backtest for {yf_symbol_for_underlying} at {current_time_for_logic.date()}: {e}", exc_info=True) # Added exc_info=True
                ltp = 0
        else: # Live mode
            underlying_symbol_fyers = f"NSE:{instrument.upper()}-I" # Fyers futures symbol
            ltp_data = await fetch_market_data(fyers, underlying_symbol_fyers)
            ltp = ltp_data["ltp"]
            # For live MA/RSI, download recent data via yfinance for indicators
            yf_data = yf.download(yf_symbol_for_underlying, period="60d", interval="1d", progress=False, prepost=True, auto_adjust=True)

        if ltp <= 0:
            logger.error(f"No valid LTP for {yf_symbol_for_underlying} or {instrument}, skipping iteration for {current_time_for_logic}.")
            if not backtest_mode: await asyncio.sleep(config["live_mode_sleep_interval"])
            continue

        # --- Calculate Technical Indicators ---
        # Pass the downloaded yf_data to avoid redundant downloads
        rsi, trend_5_10_ma = await data_svc.get_rsi_and_trend(instrument, yf_data)

        short_ma, long_ma, trend_10_50_ma = 0, 0, "Neutral"
        if not yf_data.empty and 'Close' in yf_data.columns and len(yf_data) >= 50:
            # Ensure we're using data up to the current backtest date for MA calculations
            # .loc[:current_time_for_logic] ensures we only use data available *up to* this point in backtest
            relevant_close_prices = yf_data['Close'].loc[:current_time_for_logic].dropna()
            
            if len(relevant_close_prices) >= 50:
                short_ma = relevant_close_prices.rolling(window=10).mean().iloc[-1]
                long_ma = relevant_close_prices.rolling(window=50).mean().iloc[-1]
                
                if short_ma > long_ma:
                    trend_10_50_ma = "Up"
                elif short_ma < long_ma:
                    trend_10_50_ma = "Down"
                logger.info(f"Short MA (10-day): {round(short_ma, 2)}, Long MA (50-day): {round(long_ma, 2)}, Trend (10/50 MA): {trend_10_50_ma}")
            else:
                logger.warning(f"Not enough data ({len(relevant_close_prices)} days) for 10/50 MA calculation for {current_time_for_logic.date()}.")
        else:
            logger.warning(f"Not enough yfinance data or 'Close' column missing for MA calculation for {current_time_for_logic.date()}.")

        # --- Determine Strikes and Fetch Option Chain Data ---
        strike_interval = 50 if instrument.upper() == "NIFTY" else 100
        # Calculate ATM strike based on the current LTP of the underlying
        atm_strike = round(ltp / strike_interval) * strike_interval
        # Generate a range of strikes around ATM
        strikes = [atm_strike - 2 * strike_interval, atm_strike - strike_interval, atm_strike,
                   atm_strike + strike_interval, atm_strike + 2 * strike_interval]
        strikes = [s for s in strikes if s > 0] # Ensure strikes are positive
        logger.info(f"Calculated strikes around LTP {ltp}: {strikes}")

        expiry = get_next_weekly_expiry_date(current_time_for_logic.date(), instrument)
        market_data_chain = await build_market_data_chain(fyers, instrument, expiry, strikes, data_svc)

        # --- Fetch Sentiment ---
        if backtest_mode:
            # In backtest mode, provide mock news based on market trend for sentiment generation
            if trend_10_50_ma == "Up":
                news_titles = ["Market showing strong upward momentum.", "Positive economic indicators reported.", "Nifty rallies on strong buying."] * 2
            elif trend_10_50_ma == "Down":
                news_titles = ["Market experiencing downward pressure.", "Negative global cues.", "Nifty drops amid selling pressure."] * 2
            else:
                news_titles = ["Market remains stable.", "Mixed signals for investors."] * 2
            news = [{"title": title} for title in news_titles[:5]] # Take top 5 mock articles
        else:
            # In live mode, fetch real news from NewsAPI
            news_articles = newsapi.get_everything(q=instrument, language='en', page_size=5)["articles"]
            news = news_articles
        
        sentiment, sentiment_confidence = await get_sentiment(news, client)
        
        # Consolidate all relevant data for logging and strategy decisions
        logs = {"rsi": rsi,
                "trend_5_10_ma": trend_5_10_ma,
                "trend_10_50_ma": trend_10_50_ma,
                "sentiment": sentiment,
                "sentiment_confidence": sentiment_confidence,
                "underlying_ltp": ltp # Add current underlying LTP to logs
               }

        # --- Suggest and Execute Trades ---
        suggestions = await suggest_trades(data_svc, instrument, market_data_chain, logs)
        
        if suggestions and trade_count < max_trades:
            # Filter suggestions based on combined strategy logic
            filtered_suggestions = []
            for suggestion in suggestions:
                # Apply the trading strategy logic here to filter for the best trade
                # Example:
                is_bullish_signal = (trend_10_50_ma == "Up" and sentiment == "Bullish" and suggestion["type"] == "CE" and rsi < 70)
                is_bearish_signal = (trend_10_50_ma == "Down" and sentiment == "Bearish" and suggestion["type"] == "PE" and rsi > 30)

                if (is_bullish_signal or is_bearish_signal) and suggestion.get("confidence", 0) >= min_confidence:
                    filtered_suggestions.append(suggestion)

            if filtered_suggestions:
                # Select the best suggestion (e.g., highest confidence)
                best_suggestion = max(filtered_suggestions, key=lambda x: x.get("confidence", 0))
                logger.info(f"Best filtered suggestion: {best_suggestion['symbol']} (Confidence: {best_suggestion['confidence']})")
                
                # Determine hedge contract
                hedge_type = "PE" if best_suggestion["type"] == "CE" else "CE"
                hedge_symbol = format_fyers_symbol(instrument, best_suggestion["strike"], hedge_type, expiry)
                hedge_data = next((d for d in market_data_chain if d.get("symbol") == hedge_symbol), None)

                if hedge_data and hedge_data.get("ltp", 0) > 0:
                    qty = config["nifty_lot_size"] if instrument.upper() == "NIFTY" else config["banknifty_lot_size"]
                    
                    trade_value = best_suggestion["ltp"] * qty
                    # Simulate capital check, considering the cost of the main leg
                    if capital >= trade_value:
                        logger.info(f"Attempting to execute trade for {best_suggestion['symbol']} (Main) and {hedge_data['symbol']} (Hedge).")
                        trade_executed = await execute_trade(fyers, best_suggestion, hedge_data, qty, logs)
                        
                        if trade_executed:
                            trade_count += 1
                            # In backtesting, capital adjustment is usually done after P&L calculation
                            # For simple simulation, deduct max risk per trade
                            capital -= (capital * risk_per_trade) # Simulate capital reduction by max risk for this trade
                            logger.info(f"Trade executed. Current simulated capital: {capital}")
                        else:
                            logger.warning(f"Trade execution failed for {best_suggestion['symbol']}.")
                    else:
                        logger.warning(f"Insufficient simulated capital ({capital}) for trade value ({trade_value}) for {best_suggestion['symbol']}. Skipping trade.")
                else:
                    logger.warning(f"Hedge data not found or invalid for {hedge_symbol}. Skipping trade for {best_suggestion['symbol']}.")
            else:
                logger.info("No suitable trade suggestions after applying strategy filters.")
        else:
            logger.info("No trade suggestions or maximum trades per day reached.")

        if not backtest_mode:
            await asyncio.sleep(config.get("live_mode_sleep_interval", 60)) # Sleep in live mode
        else:
            # In backtest mode, we iterate day by day. No need for intra-day sleep.
            pass

# --- Signal Handling for Graceful Shutdown ---

def signal_handler(signum, frame):
    """Handles OS signals (like Ctrl+C) to stop the trading loop gracefully."""
    logger.info("Received shutdown signal, stopping trades gracefully.")
    stop_event.set() # Set the global stop flag

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- Main Execution Block ---

if __name__ == "__main__":
    # Ensure default values for new config parameters if they don't exist
    if "nifty_lot_size" not in config:
        config["nifty_lot_size"] = 50 # Default Nifty lot size
    if "banknifty_lot_size" not in config:
        config["banknifty_lot_size"] = 35 # Default BankNifty lot size
    if "live_mode_sleep_interval" not in config:
        config["live_mode_sleep_interval"] = 60 # Default sleep interval in live mode (seconds)
    if "lot_size_multiplier" not in config: # Add this to your config for profit calculation
        config["lot_size_multiplier"] = 1 # This multiplier is usually 1, unless you want to adjust for something else.

    try:
        logger.info("Starting auto trader application.")
        asyncio.run(main_loop())
    except Exception as e:
        logger.error(f"Critical error in main loop: {e}", exc_info=True) # Log full traceback
    finally:
        logger.info("Auto trader shutdown completed.")
