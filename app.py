import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import re
import json
import time
import os
import yfinance as yf
from fyers_apiv3 import fyersModel
from openai import AzureOpenAI
import asyncio
import threading
import signal
from newsapi import NewsApiClient
import logging
from dotenv import load_dotenv
import traceback
import httpx # <--- ADDED: Import httpx for handling HTTP errors

# --- Logging Setup ---
log_dir = "C:\\ProgramData\\OrionAI"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(log_dir, "orion_trader.log"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
global_logger = logging.getLogger(__name__)  # Explicit global logger

# --- Configuration & Setup ---
st.set_page_config(page_title="Orion AI Trader", layout="wide", initial_sidebar_state="expanded")

# --- Helper Functions ---
@st.cache_data
def get_next_weekly_expiry_date(today=None, instrument="NIFTY"):
    """Calculates the next weekly expiry (Thursday for NIFTY, Wednesday for BANKNIFTY)."""
    if today is None:
        today = date.today()
    target_day = 3 if instrument.upper() == "NIFTY" else 2  # Thursday=3, Wednesday=2
    days_to_target = (target_day - today.weekday() + 7) % 7
    if days_to_target == 0 and datetime.now().time().hour >= 15:
        days_to_target += 7
    calculated_expiry = today + timedelta(days=days_to_target)
    while calculated_expiry <= today:  # Ensure future expiry
        calculated_expiry += timedelta(days=7)
    global_logger.info(f"Calculated expiry for {instrument}: {calculated_expiry}")
    return calculated_expiry

@st.cache_data
def format_fyers_symbol(instrument, strike, option_type, expiry_date):
    """Formats a symbol string for Fyers API (e.g., NSE:NIFTY25OCT2525000CE)."""
    year_short = expiry_date.strftime("%y")
    month_short = expiry_date.strftime("%b").upper()[:3]  # e.g., OCT
    strike = int(strike)  # Ensure strike is an integer
    return f"NSE:{instrument.upper()}{year_short}{month_short}{strike}{option_type.upper()}"

@st.cache_data
def fetch_valid_expiries(_fyers, instrument):
    """Fetches valid expiry dates using sample symbol validation."""
    try:
        calculated_expiry = get_next_weekly_expiry_date(instrument=instrument)
        # Use a more realistic sample strike for validation
        sample_strike = 20000 if instrument.upper() == "NIFTY" else 45000 # Adjusted for typical ranges
        sample_symbol = format_fyers_symbol(instrument, sample_strike, "CE", calculated_expiry)
        # Using get_market_data which handles retries and returns a dict with 'ltp'
        response_data = _fyers.get_market_data(sample_symbol)
        if response_data and response_data.get("ltp", 0) > 0:
            return [calculated_expiry]
        global_logger.warning(f"No valid market data for sample symbol {sample_symbol}. Using calculated expiry.")
        return [calculated_expiry]
    except Exception as e:
        global_logger.error(f"Error fetching expiries for {instrument}: {e}")
        st.error(f"Failed to fetch expiries: {e}")
        return [get_next_weekly_expiry_date(instrument=instrument)]

# --- Core Service Classes ---
@st.cache_resource
def init_fyers_service(client_id, token):
    return FyersService(client_id, token)

class FyersService:
    def __init__(self, client_id, token):
        self.client_id = client_id
        self.token = token
        self.fyers = None
        if self.is_authenticated():
            self.initialize_model()

    def is_authenticated(self):
        return bool(self.client_id and self.token)

    def initialize_model(self):
        try:
            self.fyers = fyersModel.FyersModel(
                client_id=self.client_id, token=self.token, is_async=False, log_path=""
            )
            profile = self.fyers.get_profile()
            if profile.get("code") != 200:
                st.session_state.fyers_authenticated = False
                st.error(f"Fyers Authentication Failed: {profile.get('message')}")
            else:
                st.session_state.fyers_authenticated = True
        except Exception as e:
            st.error(f"Failed to initialize Fyers Model: {e}")
            global_logger.error(f"Fyers initialization failed: {e}")
            st.session_state.fyers_authenticated = False

    def get_market_data(self, symbol, retries=3, delay=2):
        for attempt in range(retries):
            try:
                global_logger.info(f"Attempting to fetch market data for {symbol}, Attempt {attempt + 1}/{retries}")
                response = self.fyers.quotes({"symbols": symbol})
                if response.get("code") == 200 and response.get("d"):
                    quote = response["d"][0]["v"]
                    return {
                        "ltp": quote.get("lp", 0),
                        "iv": quote.get("iv", 15.0),
                        "oi": quote.get("oi", 0),
                        "low": quote.get("low_price", 0),
                        "high": quote.get("high_price", 0),
                    }
                global_logger.warning(f"No data or non-200 code for {symbol}. Response: {response}. Retrying...")
                time.sleep(delay)
            except Exception as e:
                global_logger.warning(f"Attempt {attempt + 1}/{retries} failed for {symbol}: {e}")
                time.sleep(delay)
        global_logger.error(f"Failed to fetch market data for {symbol} after {retries} retries")
        return None

    def get_historical_data(self, symbol, resolution="5", days=10):
        if not self.is_authenticated():
            return pd.DataFrame()
        range_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        range_to = datetime.now().strftime("%Y-%m-%d")
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "0",
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1",
        }
        try:
            response = self.fyers.history(data=data)
            if response.get("code") == 200 and response.get("candles"):
                df = pd.DataFrame(
                    response["candles"],
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                return df.set_index("timestamp")
            return pd.DataFrame()
        except Exception as e:
            global_logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()

    def place_order(self, symbol, qty, side, stop_loss=None):
        if not self.is_authenticated():
            return {"code": -1, "message": "Not authenticated"}
        order = {
            "symbol": symbol,
            "qty": qty,
            "type": 2,  # Market order
            "side": side,  # 1 for buy, -1 for sell
            "productType": "INTRADAY",
            "validity": "DAY",
        }
        if stop_loss:
            order["stopPrice"] = stop_loss
            order["type"] = 4  # Stop order
        try:
            response = self.fyers.place_order(data=order)
            return response
        except Exception as e:
            global_logger.error(f"Order placement failed for {symbol}: {e}")
            return {"code": -1, "message": str(e)}

@st.cache_resource
def init_news_service():
    return NewsService()

class NewsService:
    def __init__(self):
        self.client = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
        self.enabled = bool(os.getenv("NEWSAPI_KEY"))

    def fetch_news(self, query, days=3):
        if not self.enabled:
            return []
        try:
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            response = self.client.get_everything(
                q=query, from_param=from_date, to=to_date, language="en", sort_by="relevancy"
            )
            articles = response.get("articles", [])
            global_logger.info(f"Fetched {len(articles)} news articles for {query}")
            return articles
        except Exception as e:
            global_logger.error(f"Failed to fetch news for {query}: {e}")
            return []

    def analyze_sentiment(self, gpt, articles):
        if not articles or not gpt.enabled:
            return "Neutral", 0.5
        prompt = f"""
        Analyze the following news articles for market sentiment on the given index (e.g., NIFTY or BANKNIFTY).
        Return a JSON object: {{"sentiment": "Bullish/Bearish/Neutral", "confidence": float (0-1)}}.
        Articles: {json.dumps([a["title"] + ": " + (a.get("description") or "") for a in articles])}
        """
        messages = [{"role": "system", "content": prompt}]
        response = gpt.ask(messages, temperature=0.3) # <--- Changed to call gpt.ask
        global_logger.info(f"Sentiment analysis response: {response}")
        try:
            result = json.loads(response) if response else {}
            return (
                result.get("sentiment", "Neutral"),
                float(result.get("confidence", 0.5)),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            global_logger.error(f"Failed to parse sentiment response: {e}, Response: {response}")
            return "Neutral", 0.5

@st.cache_resource
def init_gpt_service():
    return GptService()

class GptService:
    def __init__(self):
        try:
            self.client = AzureOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version=os.getenv("OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("OPENAI_API_BASE"),
            )
            self.deployment = os.getenv("OPENAI_DEPLOYMENT")
            self.enabled = True
        except Exception as e:
            st.error(f"Failed to initialize Azure OpenAI: {e}")
            global_logger.error(f"Azure OpenAI initialization failed: {e}")
            self.enabled = False

    # Renamed _make_request to ask for consistency with how it's called
    def ask(self, messages, temperature=0.4, max_retries=3, backoff_factor=2): # <--- RENAMED METHOD
        if not self.enabled:
            return None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment, messages=messages, temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                global_logger.error(f"Azure OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(backoff_factor ** attempt)
        return None

    def extract_trade_intent(self, user_text):
        system_prompt = """
        Parse the user's prompt to extract a trading symbol in JSON format: {"symbol": "<INDEX>_<STRIKE>_<CE/PE>", "instrument": "<INDEX>"}
        INDEX: NIFTY or BANKNIFTY
        STRIKE: Number
        CE/PE: Option type
        Example: "Analyze NIFTY 25000 CE" -> {"symbol": "NIFTY_25000_CE", "instrument": "NIFTY"}
        Respond ONLY with the JSON object. If unclear, return {}.
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
        response = self.ask(messages) # <--- Changed to call self.ask
        try:
            return json.loads(response)
        except:
            global_logger.error(f"Failed to parse trade intent: {response}")
            return {}

    def analyze_trade_opportunity(self, symbol, ltp, rsi, trend, iv, oi, sentiment, sentiment_confidence, news_summary):
        global_logger.info(f"Analyzing {symbol}: LTP={ltp}, RSI={rsi}, Trend={trend}, IV={iv}, OI={oi}")
        prompt = f"""
        You are an expert options analyst. Provide a professional-grade analysis for the given options contract in Markdown format, incorporating technical indicators and news sentiment.

        **Contract Details:**
        - **Symbol:** {symbol}
        - **Current Price (LTP):** ₹{ltp:.2f}
        - **Underlying RSI (14D):** {rsi}
        - **Underlying Trend (5/10 MA):** {trend}
        - **Implied Volatility (IV):** {iv:.2f}%
        - **Open Interest (OI):** {oi:,}
        - **Market Sentiment:** {sentiment} (Confidence: {sentiment_confidence:.2%})
        - **News Summary:** {news_summary}

        **Structure:**
        ### Quick Technical Observation:
        One-line summary of the current situation (strength, weakness, or indecision).

        ### Detailed Analysis:
        - **Price Action:** Comment on LTP (holding, breaking out, or rejecting).
        - **Indicators:**
          - **RSI:** Overbought (>70), oversold (<30), or neutral? Implications?
          - **Trend:** Does the trend support a trade?
          - **OI & IV:** High/low OI and IV implications (premium cost, interest).
          - **News Sentiment:** How does the sentiment ({sentiment}) affect the trade?

        ### Key Levels:
        - **Support:** Suggest a support level for the option price.
        - **Resistance:** Suggest a resistance level.

        ### Bias & Outlook:
        State bias (e.g., Bullish, Bearish, Neutral). Short-term direction?

        ### Actionable Advice:
        - **Aggressive Entry:** Entry point and stop-loss for high-risk traders.
        - **Safer Entry:** Entry on confirmation or breakout.
        - **General Counsel:** Emphasize risk management and market context.
        """
        messages = [{"role": "system", "content": prompt}]
        return self.ask(messages, temperature=0.5) # <--- Changed to call self.ask

    def suggest_trades(self, instrument, market_data_chain, news_sentiment, logs): # <--- Renamed market_data to market_data_chain for clarity
        global_logger.info(f"Suggesting trades for {instrument}, Market Data Chain (size: {len(market_data_chain)}), Sentiment: {news_sentiment}")
        valid_market_data_chain = [d for d in market_data_chain if d.get("ltp", 0) > 0 and d.get("strike", 0) > 0]
        if not valid_market_data_chain:
            global_logger.warning("No valid market data available for trade suggestions.")
            return []

        # No need to limit to top 10 here as it's done in render_suggester_page before passing
        # valid_market_data_chain = sorted(valid_market_data_chain, key=lambda x: abs(x["strike"] - valid_market_data_chain[0].get("ltp", 0)))[:20]

        prompt = f"""
        You are a trading expert tasked with suggesting high-confidence option trades for {instrument}. Use the following data:
        - Option Chain Data: {json.dumps(valid_market_data_chain)}
        - Market Sentiment: {json.dumps(news_sentiment)}
        - Past Trades (up to 5 recent): {json.dumps(logs[:5])}
        Suggest up to 3 trades in JSON format as an array of objects. Each object must contain:
        - "symbol": The full Fyers symbol (e.g., "NSE:NIFTY25JUL25000CE")
        - "strike": The strike price (e.g., 25000)
        - "type": The option type ("CE" or "PE")
        - "confidence": A float between 0 and 1 representing the confidence level (e.g., 0.95)
        - "reason": A brief explanation for the suggestion
        Only include trades with confidence > 0.8.
        Don't Hallucinate or invent data. Use only the provided market data and sentiment.
        Ensure the suggestions are actionable and relevant to current market conditions.
        Return the response as a VALID JSON array and nothing else. Ensure the JSON is parsable directly.
        Example of expected format:
        [
            {{"symbol": "NSE:NIFTY25JUL25000CE", "strike": 25000, "type": "CE", "confidence": 0.95, "reason": "Strong bullish signal"}},
            {{"symbol": "NSE:NIFTY25JUL24500PE", "strike": 24500, "type": "PE", "confidence": 0.92, "reason": "Bearish trend confirmed"}}
        ].
        If you cannot generate valid suggestions, return an empty array: [].
        """
        messages = [{"role": "system", "content": prompt}]
        try:
            # Removed timeout parameter as it's not standard for AzureOpenAI client.chat.completions.create
            response = self.ask(messages, temperature=0.3)
            global_logger.info(f"Raw GPT response in suggest_trades: {response}") # Log before any processing
            if not response:
                global_logger.error("No response from GPT for trade suggestions.")
                return []
        except httpx.RequestTimeout as e: # This exception handling relies on httpx, ensure it's imported
            global_logger.error(f"GPT API timeout: {e}")
            return []
        except httpx.HTTPStatusError as e:
            global_logger.error(f"GPT API HTTP error: {e}, Status: {e.response.status_code}")
            return []
        except Exception as e:
            global_logger.error(f"Unexpected error in GPT API call for suggestions: {e}")
            return []

        try:
            suggestions = json.loads(response)
            if not isinstance(suggestions, list):
                global_logger.error(f"Invalid response format from GPT: {response} (expected list)")
                return [] # Return empty list if not a list
            # Filter for confidence, assuming valid structure by now
            return [s for s in suggestions if isinstance(s, dict) and s.get("confidence", 0) > 0.9]
        except json.JSONDecodeError as e:
            global_logger.error(f"Failed to parse trade suggestions as JSON: {e}, Raw Response: {response}")
            return [] # Return empty list on JSON parsing failure
        except Exception as e:
            global_logger.error(f"Unexpected error parsing trade suggestions: {e}, Response: {response}")
            return []

@st.cache_resource
def init_data_service():
    return DataService()

class DataService:
    def get_yfinance_symbol(self, instrument):
        return {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}.get(instrument.upper())

    @st.cache_data
    def get_rsi_and_trend(_self, instrument):
        yf_symbol = _self.get_yfinance_symbol(instrument)
        if not yf_symbol:
            return 50.0, "Neutral"
        try:
            data = yf.download(yf_symbol, period="21d", interval="1d", progress=False)
            if not isinstance(data, pd.DataFrame) or data.empty or len(data) < 15:
                global_logger.warning(f"Insufficient or invalid yfinance data for {yf_symbol}, falling back to 50.0. Data: {str(data.index)}")
                return 50.0, "Neutral"
            try:
                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    global_logger.info(f"MultiIndex columns detected for {yf_symbol}, levels: {data.columns.names}")
                    # Access the correct column, e.g., ('Close', 'symbol')
                    if ('Close', yf_symbol) in data.columns:
                        close_series = data[('Close', yf_symbol)]
                    elif 'Close' in data.columns: # Fallback if yf_symbol isn't secondary index
                         close_series = data['Close']
                    else:
                        raise KeyError(f"Neither ('Close', '{yf_symbol}') nor 'Close' found in columns.")
                else:
                    global_logger.info(f"Single index columns detected for {yf_symbol}, columns: {data.columns}")
                    close_series = data['Close']
                close_prices = close_series.tolist()
            except (AttributeError, KeyError, ValueError) as e:
                global_logger.error(f"Failed to access Close column for {yf_symbol}: {e}, Data structure: {str(data.index)}-{str(data.columns)}")
                return 50.0, "Neutral"
            # Remove NaN values and ensure enough data
            close_prices = [x for x in close_prices if pd.notna(x)]
            if len(close_prices) < 15:
                global_logger.warning(f"Invalid data points for {yf_symbol} after NaN removal, falling back to 50.0")
                return 50.0, "Neutral"

            # Calculate RSI iteratively
            # Need at least 14 periods for RSI. Adjusted loop to cover enough data.
            if len(close_prices) < 15: # Need 14 differences plus initial value, so 15 data points
                global_logger.warning(f"Not enough data for RSI calculation for {yf_symbol}. Found {len(close_prices)} points.")
                return 50.0, "Neutral"

            avg_gain = 0
            avg_loss = 0
            # Initial 14-period average gain/loss
            for i in range(1, 15):
                diff = close_prices[i] - close_prices[i - 1]
                avg_gain += max(0, diff)
                avg_loss += max(0, -diff)

            avg_gain /= 14
            avg_loss /= 14

            # Subsequent RSI calculations
            for i in range(15, len(close_prices)):
                diff = close_prices[i] - close_prices[i-1]
                gain = max(0, diff)
                loss = max(0, -diff)
                avg_gain = (avg_gain * 13 + gain) / 14
                avg_loss = (avg_loss * 13 + loss) / 14

            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = round(100 - (100 / (1 + rs)), 2)


            # Calculate Trend with 5/10 MA
            if len(close_prices) < 10:
                trend = "Neutral"
            else:
                ma5 = sum(close_prices[-5:]) / 5
                ma10 = sum(close_prices[-10:]) / 10
                trend = "Up" if ma5 > ma10 else "Down" if ma5 < ma10 else "Neutral"

            global_logger.info(f"RSI and Trend for {instrument}: RSI={rsi}, Trend={trend}, Data points={len(close_prices)}")
            return rsi, trend
        except Exception as e:
            global_logger.error(f"Failed to fetch yfinance data for {yf_symbol}: {e}")
            return 50.0, "Neutral"

    def _get_last_close(self, instrument):
        yf_symbol = self.get_yfinance_symbol(instrument)
        if not yf_symbol:
            return None
        try:
            data = yf.download(yf_symbol, period="1d", interval="1d", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                if ('Close', yf_symbol) in data.columns:
                    close_series = data[('Close', yf_symbol)]
                elif 'Close' in data.columns:
                    close_series = data['Close']
                else:
                    return None
            else:
                close_series = data['Close']
            return close_series.iloc[-1] if not data.empty else None
        except Exception as e:
            global_logger.error(f"Failed to get last close for {yf_symbol}: {e}")
            return None

@st.cache_resource
def init_trade_logger():
    return TradeLogger()

class TradeLogger:
    LOG_FILE = "trade_log.json"
    SUGGESTION_FILE = "suggestions.json"
    TRADES_FILE = "persistent_trades.json"

    def _load_log(self, file=LOG_FILE):
        if not os.path.exists(file):
            return []
        try:
            with open(file, "r") as f:
                return json.load(f)
        except Exception as e:
            global_logger.error(f"Failed to load log file {file}: {e}")
            return []

    def _save_log(self, log_data, file=LOG_FILE):
        try:
            with open(file, "w") as f:
                json.dump(log_data, f, indent=4)
        except Exception as e:
            global_logger.error(f"Failed to save log file {file}: {e}")

    def _load_trades(self):
        if os.path.exists(self.TRADES_FILE):
            try:
                with open(self.TRADES_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                global_logger.error(f"Failed to load persistent trades: {e}")
        return []

    def _save_trades(self, trades):
        try:
            with open(self.TRADES_FILE, "w") as f:
                json.dump(trades, f, indent=4)
        except Exception as e:
            global_logger.error(f"Failed to save persistent trades: {e}")

    def log_entry(self, entry_data):
        logs = self._load_log()
        log_entry = {"timestamp": datetime.now().isoformat(), **entry_data}
        logs.insert(0, log_entry)
        self._save_log(logs)
        try:
            with open("fine_tune_log.jsonl", "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
        except Exception as e:
            global_logger.error(f"Failed to save fine-tune log: {e}")
        st.toast(f"{entry_data.get('type', 'Entry')} logged!")

    def log_suggestion(self, suggestion):
        suggestions = self._load_log(self.SUGGESTION_FILE)
        suggestions.insert(0, suggestion)
        self._save_log(suggestions, self.SUGGESTION_FILE)

    def get_logs(self, file=LOG_FILE):
        return self._load_log(file)

    def save_active_trades(self, trades):
        self._save_trades(trades)

    def load_active_trades(self):
        return self._load_trades()

# --- UI Rendering & Main App Logic ---
def render_trade_monitor(fyers, data_svc, logger):
    st.subheader("Trade Monitor")
    if not st.session_state.active_trades:
        st.info("No trades are being monitored.")
        return

    for i, trade in enumerate(st.session_state.active_trades):
        with st.container(border=True):
            # Update live LTP before processing
            market_data = fyers.get_market_data(trade["symbol"])
            if market_data:
                trade["current_ltp"] = market_data["ltp"]
            else:
                trade["current_ltp"] = trade.get("current_ltp", 0)  # Fallback to last known or 0
                global_logger.warning(f"Failed to fetch market data for {trade['symbol']}, using fallback LTP: {trade['current_ltp']}")
            global_logger.info(f"Trade {trade['symbol']} current_ltp: {trade['current_ltp']}")

            st.metric("Live LTP", f"₹{trade['current_ltp']:.2f}")

            if trade["status"] == "pending_entry":
                st.markdown(f"**PENDING: {trade['symbol']}**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Current LTP", f"₹{trade['current_ltp']:.2f}")
                col2.metric("Buy Between", f"₹{trade['entry_low']:.2f} - ₹{trade['entry_high']:.2f}")
                col3.metric("Target", f"₹{trade['target']:.2f}")
                if trade.get("stop_loss"):
                    col3.metric("Stop Loss", f"₹{trade['stop_loss']:.2f}")

                # Modify trade parameters
                with st.form(key=f"modify_pending_{i}"):
                    new_entry_low = st.number_input("New Entry Low", value=trade["entry_low"], step=1.0, format="%.2f", key=f"entry_low_{i}")
                    new_entry_high = st.number_input("New Entry High", value=trade["entry_high"], step=1.0, format="%.2f", key=f"entry_high_{i}")
                    new_target = st.number_input("New Target", value=trade["target"], step=1.0, format="%.2f", key=f"target_pending_{i}")
                    new_stop_loss = st.number_input("New Stop Loss", value=trade.get("stop_loss", 0.0), step=1.0, format="%.2f", key=f"sl_pending_{i}")
                    if st.form_submit_button("Update"):
                        if trade["current_ltp"] is None:
                            market_data = fyers.get_market_data(trade["symbol"])
                            trade["current_ltp"] = market_data["ltp"] if market_data else trade.get("current_ltp", 0)
                        if (new_entry_low != trade["entry_low"] or new_entry_high != trade["entry_high"] or
                            new_target != trade["target"] or new_stop_loss != trade.get("stop_loss", 0.0)):
                            trade["entry_low"] = new_entry_low
                            trade["entry_high"] = new_entry_high
                            trade["target"] = new_target
                            trade["stop_loss"] = new_stop_loss if new_stop_loss > 0 else None
                            logger.log_entry({"type": "updated_pending_trade", **trade})
                            logger.save_active_trades(st.session_state.active_trades)
                            st.success(f"Trade parameters updated for {trade['symbol']}!")
                            st.rerun()

                # Cancel trade
                if st.button("Cancel Trade", key=f"cancel_{i}", type="primary"):
                    logger.log_entry({"type": "cancelled_trade", "symbol": trade["symbol"], "reason": "user_cancelled"})
                    st.session_state.active_trades.pop(i)
                    logger.save_active_trades(st.session_state.active_trades)
                    st.success(f"Cancelled trade {trade['symbol']}.")
                    st.rerun()

                if trade["current_ltp"] is not None and trade["entry_low"] <= trade["current_ltp"] <= trade["entry_high"]:
                    st.success("✅ Entry condition met! Placing BUY order...")
                    res = fyers.place_order(trade["symbol"], trade["qty"], side=1, stop_loss=trade.get("stop_loss"))
                    if res.get("code") == 200 and res.get("id"):
                        st.session_state.active_trades[i]["status"] = "active"
                        st.session_state.active_trades[i]["order_id"] = res["id"]
                        st.session_state.active_trades[i]["entry_price"] = trade["current_ltp"]
                        logger.log_entry({"type": "executed_trade", **st.session_state.active_trades[i]})
                        logger.save_active_trades(st.session_state.active_trades)
                        st.rerun()
                    else:
                        st.error(f"Order placement failed: {res.get('message')}")
                        st.session_state.active_trades.pop(i)
                        logger.save_active_trades(st.session_state.active_trades)
                        st.rerun()

            elif trade["status"] == "active":
                st.markdown(f"**ACTIVE: {trade['symbol']}**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Current LTP", f"₹{trade['current_ltp']:.2f}", f"₹{trade['current_ltp'] - trade['entry_price']:.2f}")
                col2.metric("Entry Price", f"₹{trade['entry_price']:.2f}")
                new_target = col3.number_input("Target", value=trade["target"], key=f"target_{i}", step=1.0, format="%.2f")
                new_stop_loss = col3.number_input("Stop Loss", value=trade.get("stop_loss", 0.0), key=f"sl_{i}", step=1.0, format="%.2f")
                if new_target != trade["target"]:
                    st.session_state.active_trades[i]["target"] = new_target
                    logger.log_entry({"type": "updated_active_trade", "symbol": trade["symbol"], "target": new_target})
                    logger.save_active_trades(st.session_state.active_trades)
                    st.toast(f"Target updated for {trade['symbol']}!")
                if new_stop_loss != trade.get("stop_loss", 0.0) and new_stop_loss > 0:
                    st.session_state.active_trades[i]["stop_loss"] = new_stop_loss
                    logger.log_entry({"type": "updated_active_trade", "symbol": trade["symbol"], "stop_loss": new_stop_loss})
                    logger.save_active_trades(st.session_state.active_trades)
                    st.toast(f"Stop Loss updated for {trade['symbol']}!")

                if st.button("Exit Trade Manually", key=f"exit_{i}", type="primary"):
                    res = fyers.place_order(trade["symbol"], trade["qty"], side=-1)
                    if res.get("code") == 200:
                        st.success("Manual exit order placed!")
                        logger.log_entry({"type": "exited_trade", "symbol": trade["symbol"], "exit_price": trade["current_ltp"]})
                        st.session_state.active_trades.pop(i)
                        logger.save_active_trades(st.session_state.active_trades)
                        st.rerun()
                    else:
                        st.error(f"Exit failed: {res.get('message')}")

                if trade["current_ltp"] is not None and trade["current_ltp"] >= trade["target"]:
                    st.success(f"🎯 Target hit! Exiting trade...")
                    res = fyers.place_order(trade["symbol"], trade["qty"], side=-1)
                    logger.log_entry({"type": "exited_trade", "symbol": trade["symbol"], "exit_price": trade["current_ltp"]})
                    st.session_state.active_trades.pop(i)
                    logger.save_active_trades(st.session_state.active_trades)
                    st.rerun()
                elif trade.get("stop_loss") and trade["current_ltp"] is not None and trade["current_ltp"] <= trade["stop_loss"]:
                    st.warning(f"🛑 Stop Loss hit! Exiting trade...")
                    res = fyers.place_order(trade["symbol"], trade["qty"], side=-1)
                    logger.log_entry({"type": "exited_trade", "symbol": trade["symbol"], "exit_price": trade["current_ltp"], "reason": "stop_loss"})
                    st.session_state.active_trades.pop(i)
                    logger.save_active_trades(st.session_state.active_trades)
                    st.rerun()


def process_gpt_response(gpt, prompt):
    global_logger.info(f"Prompt for GPT: {prompt}")
    response = None
    try:
        response = gpt.ask(prompt) # <--- Changed to call gpt.ask
        global_logger.info(f"Raw GPT response in process_gpt_response: {response}")
        print(f"Debug: Raw GPT response - {response}")  # Console debug
    except Exception as e:
        global_logger.error(f"Error during GPT call in process_gpt_response: {e}\nStack trace: {traceback.format_exc()}")
        return []

    if response is None:
        global_logger.error("GPT response is None in process_gpt_response")
        return []

    try:
        suggestions = json.loads(response)
        if not isinstance(suggestions, list):
            global_logger.error(f"Invalid response format from GPT: {response} (expected list)")
            return []
        return suggestions
    except json.JSONDecodeError as e:
        global_logger.error(f"Failed to parse trade suggestions as JSON in process_gpt_response: {e}, Raw Response: {response}")
        return []
    except Exception as e:
        global_logger.error(f"Unexpected error parsing trade suggestions in process_gpt_response: {e}, Raw Response: {response}")
        return []

def render_suggester_page(fyers, gpt, data_svc, news_svc, trade_logger):
    st.header("Trade Suggester")
    try:
        # User inputs for customization
        instrument = st.selectbox("Select Instrument", ["NIFTY", "BANKNIFTY"], key="suggester_instrument")
        min_confidence = st.slider("Minimum Confidence (%)", 0, 100, 90, 5, key="min_confidence") / 100.0
        range_percentage = st.slider("Strike Range (% of LTP)", 2, 10, 4, 1, key="range_percentage") / 100.0
        strike_step = st.number_input("Strike Step Size", min_value=25, value=50 if instrument == "NIFTY" else 100, step=25, key="strike_step")

        if st.button("Generate Trade Suggestions", key="generate_suggestions"):
            # Fetch initial market data for the underlying index
            underlying_symbol = f"NSE:{instrument.upper()}-INDEX"
            market_data_underlying = fyers.get_market_data(underlying_symbol)
            if not market_data_underlying or market_data_underlying.get("ltp", 0) <= 0:
                fallback_ltp = data_svc._get_last_close(instrument)
                if fallback_ltp:
                    market_data_underlying = {"ltp": fallback_ltp}
                    global_logger.warning(f"Using fallback LTP {fallback_ltp} for {underlying_symbol} due to zero value")
                else:
                    st.error(f"Failed to fetch valid market data for {underlying_symbol}.")
                    return
            ltp = market_data_underlying["ltp"]
            expiry = min(fetch_valid_expiries(fyers, instrument), key=lambda x: abs((x - date.today()).days))
            # expiry_str = expiry.strftime("%y%b").upper() # Not directly used in Fyers symbol formatting now

            # Fetch option chain data within specified range
            market_data_chain = []
            min_strike = int(ltp * (1 - range_percentage))
            max_strike = int(ltp * (1 + range_percentage))

            # Adjust min_strike and max_strike to be multiples of strike_step
            min_strike = (min_strike // strike_step) * strike_step
            max_strike = ((max_strike + strike_step - 1) // strike_step) * strike_step

            available_strikes = range(min_strike, max_strike + 1, strike_step)

            # Collect market data for a reasonable number of strikes around LTP
            strikes_to_fetch = []
            for strike in available_strikes:
                strikes_to_fetch.append(strike)
            
            # Sort strikes by proximity to LTP and take a subset if too many
            strikes_to_fetch = sorted(strikes_to_fetch, key=lambda x: abs(x - ltp))[:20] # Limit to 20 strikes (10 CE, 10 PE)

            for strike in strikes_to_fetch:
                for opt_type in ["CE", "PE"]:
                    symbol = format_fyers_symbol(instrument, strike, opt_type, expiry)
                    data = fyers.get_market_data(symbol)
                    if data and data.get("ltp", 0) > 0:
                        market_data_chain.append({
                            "symbol": symbol,
                            "strike": strike,
                            "type": opt_type,
                            "ltp": data["ltp"],
                            "iv": data.get("iv", 15.0), # Include IV for GPT analysis
                            "oi": data.get("oi", 0)     # Include OI for GPT analysis
                        })
                    global_logger.info(f"Processed {symbol}, LTP: {data.get('ltp', 0) if data else 'None'}")

            if not market_data_chain:
                st.warning("No valid option data available in the specified range.")
                return

            # Fetch market sentiment
            news = news_svc.fetch_news(instrument)
            sentiment, sentiment_confidence = news_svc.analyze_sentiment(gpt, news)
            st.write(f"Market Sentiment: {sentiment} (Confidence: {sentiment_confidence:.2f})")

            # Gather past trade logs
            logs = trade_logger.get_logs()[:5]  # Limit to 5 recent logs

            with st.spinner("Analyzing market data and generating suggestions..."):
                # Call GPT service with collected data
                suggestions = gpt.suggest_trades(
                    instrument,
                    market_data_chain,
                    {"sentiment": sentiment, "confidence": sentiment_confidence},
                    logs
                )
                
                # Filter for high-confidence suggestions
                valid_suggestions = [s for s in suggestions if isinstance(s, dict) and s.get("confidence", 0) >= min_confidence]

                if not valid_suggestions:
                    st.warning(f"No high-confidence trade suggestions generated (confidence >= {min_confidence*100}%).")
                    return

                # Display suggestions
                st.subheader("Suggested Trades")
                for suggestion in valid_suggestions:
                    trade_logger.log_suggestion(suggestion)
                    st.markdown(f"""
                    ### Suggested Trade
                    - **Symbol:** {suggestion.get('symbol', 'N/A')}
                    - **Strike:** {suggestion.get('strike', 'N/A')}
                    - **Type:** {suggestion.get('type', 'N/A')}
                    - **Confidence:** {suggestion.get('confidence', 0):.2%}
                    - **Reason:** {suggestion.get('reason', 'N/A')}
                    """)
                    if st.button(f"Analyze {suggestion.get('symbol', 'N/A')}", key=f"analyze_{suggestion.get('symbol', 'N/A')}"):
                        market_data = fyers.get_market_data(suggestion.get("symbol", ""))
                        st.session_state.trade_details = {
                            "symbol": suggestion.get("symbol", ""),
                            "instrument": instrument,
                            "ltp": market_data.get("ltp", 0) if market_data else 0,
                            "rsi": data_svc.get_rsi_and_trend(instrument)[0],
                            "trend": data_svc.get_rsi_and_trend(instrument)[1],
                            "iv": market_data.get("iv", 15.0) if market_data else 15.0,
                            "oi": market_data.get("oi", 0) if market_data else 0,
                            "sentiment": sentiment,
                            "confidence": sentiment_confidence,
                        }
                        news_summary = " ".join([a.get("title", "") for a in news[:3]]) if news else "No recent news."
                        st.session_state.analysis_result = gpt.analyze_trade_opportunity(
                            suggestion.get("symbol", ""), st.session_state.trade_details["ltp"],
                            st.session_state.trade_details["rsi"], st.session_state.trade_details["trend"],
                            st.session_state.trade_details["iv"], st.session_state.trade_details["oi"],
                            sentiment, sentiment_confidence, news_summary
                        )
                        st.rerun()

        if st.button("Refresh Suggestions", key="refresh_suggestions"):
            st.rerun()

    except Exception as e:
        st.error(f"Trade suggester error: {e}")
        global_logger.error(f"Trade suggester error: {e}\nStack trace: {traceback.format_exc()}")


def keyboard_exit_handler(fyers, active_trades):
    def signal_handler(sig, frame):
        for trade in active_trades:
            if trade["status"] == "active":
                res = fyers.place_order(trade["symbol"], trade["qty"], side=-1)
                if res.get("code") == 200:
                    global_logger.info(f"Manually exited trade: {trade['symbol']}")
                else:
                    global_logger.error(f"Failed to exit trade {trade['symbol']}: {res.get('message')}")
        active_trades.clear()
        logger.save_active_trades(active_trades)
        global_logger.info("All active trades exited.")
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)

async def main():
    load_dotenv()
    fyers_client_id = os.getenv("FYERS_CLIENT_ID", "BMYI69N4EG-100")
    fyers_token = None
    if os.path.exists("token.txt"):
        try:
            with open("token.txt", "r") as f:
                fyers_token = f.read().strip()
        except Exception as e:
            global_logger.error(f"Failed to read token.txt: {e}")

    fyers = init_fyers_service(fyers_client_id, fyers_token)
    gpt = init_gpt_service()
    data_svc = init_data_service()
    news_svc = init_news_service()
    trade_logger = init_trade_logger()

    if "fyers_authenticated" not in st.session_state:
        st.session_state.fyers_authenticated = fyers.is_authenticated()
    if "active_trades" not in st.session_state:
        st.session_state.active_trades = trade_logger.load_active_trades()
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "trade_details" not in st.session_state:
        st.session_state.trade_details = {}

    threading.Thread(target=keyboard_exit_handler, args=(fyers, st.session_state.active_trades), daemon=True).start()

    def update_market_data():
        while True:
            # Create a copy of the list to iterate to avoid issues if trades are modified during iteration
            for trade in list(st.session_state.active_trades):
                # Ensure trade still exists in active_trades before processing
                if trade in st.session_state.active_trades:
                    market_data = fyers.get_market_data(trade["symbol"])
                    if market_data:
                        trade["current_ltp"] = market_data["ltp"]
                        trade["iv"] = market_data["iv"]
                        trade["oi"] = market_data["oi"]
                    # Do not save inside the loop; save once after all updates or periodically
            trade_logger.save_active_trades(st.session_state.active_trades) # Save outside loop to avoid excessive writes
            time.sleep(5)  # Update every 5 seconds

    # Start the market data updater only once
    if "market_data_updater_started" not in st.session_state:
        threading.Thread(target=update_market_data, daemon=True).start()
        st.session_state.market_data_updater_started = True

    with st.sidebar:
        st.header("Orion AI Trader")
        st.success("Fyers Authenticated!") if st.session_state.fyers_authenticated else st.error("Fyers Not Authenticated.")
        st.success("GPT Service Ready!") if gpt.enabled else st.error("GPT Not Configured.")
        st.success("News API Ready!") if news_svc.enabled else st.error("News API Not Configured.")
        st.divider()
        page = st.radio("Navigation", ["Trade Analysis", "Trade Suggester", "Trade Logs"], key="page_nav")

    if not st.session_state.fyers_authenticated or not gpt.enabled or not news_svc.enabled:
        st.warning("Please configure Fyers, Azure OpenAI, and NewsAPI to use the application.")
        return

    if page == "Trade Analysis":
        await render_trade_analysis_page(fyers, gpt, data_svc, news_svc, trade_logger)
    elif page == "Trade Suggester":
        render_suggester_page(fyers, gpt, data_svc, news_svc, trade_logger)
    elif page == "Trade Logs":
        render_trade_logs_page(trade_logger)

async def render_trade_analysis_page(fyers, gpt, data_svc, news_svc, trade_logger):
    st.header("Trade Analysis Engine")
    try:
        prompt = st.text_area("Enter your trade idea:", placeholder="e.g., Analyze NIFTY 25000 CE", key="trade_prompt")
        if st.button("Analyze Trade", type="primary", key="analyze_button"):
            if not prompt:
                st.error("Please enter a trade idea.")
                return

            with st.spinner("Parsing your prompt..."):
                intent = gpt.extract_trade_intent(prompt)

            if intent and "symbol" in intent and "instrument" in intent:
                match = re.match(r"(BANKNIFTY|NIFTY)_(\d+)_(CE|PE)", intent["symbol"])
                if match:
                    instrument, strike, opt_type = match.groups()
                    expiries = fetch_valid_expiries(fyers, instrument)
                    if not expiries:
                        st.error("No valid expiries found for instrument.")
                        return

                    expiry = min(expiries, key=lambda x: abs((x - date.today()).days))
                    fyers_sym = format_fyers_symbol(instrument, strike, opt_type, expiry)

                    with st.spinner(f"Fetching data for {fyers_sym}..."):
                        market_data = fyers.get_market_data(fyers_sym)
                        rsi, trend = data_svc.get_rsi_and_trend(instrument)
                        news = news_svc.fetch_news(instrument)
                        sentiment, confidence = news_svc.analyze_sentiment(gpt, news)
                        news_summary = " ".join([a.get("title", "") for a in news[:3]]) if news else "No recent news."

                    if market_data and market_data.get("ltp", 0) > 0:
                        with st.spinner("Analyzing trade opportunity..."):
                            analysis = gpt.analyze_trade_opportunity(
                                fyers_sym, market_data["ltp"], rsi, trend, market_data["iv"], market_data["oi"],
                                sentiment, confidence, news_summary
                            )
                            if analysis:
                                st.session_state.analysis_result = analysis
                                st.session_state.trade_details = {
                                    "symbol": fyers_sym,
                                    "instrument": instrument,
                                    "rsi": rsi,
                                    "trend": trend,
                                    "sentiment": sentiment,
                                    "confidence": confidence,
                                    **market_data,
                                }
                            else:
                                st.error("Failed to generate trade analysis.")
                    else:
                        st.error(f"Could not fetch valid market data for {fyers_sym}.")
                else:
                    st.error("Could not parse symbol from prompt.")
            else:
                st.error("Could not understand trade intent.")

        if st.session_state.get("analysis_result"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("AI Analysis Report")
                st.markdown(st.session_state.analysis_result)

                st.subheader("Charts")
                trade_details = st.session_state.trade_details
                underlying_symbol = f"NSE:{trade_details['instrument']}-INDEX"
                option_symbol = trade_details["symbol"]
                hist_df_underlying = fyers.get_historical_data(underlying_symbol, resolution="5")
                hist_df_option = fyers.get_historical_data(option_symbol, resolution="5")

                if not hist_df_underlying.empty:
                    fig = go.Figure(
                        data=[
                            go.Candlestick(
                                x=hist_df_underlying.index,
                                open=hist_df_underlying["open"],
                                high=hist_df_underlying["high"],
                                low=hist_df_underlying["low"],
                                close=hist_df_underlying["close"],
                                name="Underlying",
                            )
                        ]
                    )
                    fig.update_layout(title=f"{trade_details['instrument']} Chart", xaxis_rangeslider_visible=False, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

                if not hist_df_option.empty:
                    global_logger.info(f"OHLC data for {option_symbol}: {len(hist_df_option)} candles, Last Close={hist_df_option['close'].iloc[-1]}")
                    fig = go.Figure(
                        data=[
                            go.Candlestick(
                                x=hist_df_option.index,
                                open=hist_df_option["open"],
                                high=hist_df_option["high"],
                                low=hist_df_option["low"],
                                close=hist_df_option["close"],
                                name="Option",
                            )
                        ]
                    )
                    fig.update_layout(title=f"{trade_details['symbol']} Chart", xaxis_rangeslider_visible=False, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Trade Execution")
                trade = st.session_state.trade_details
                st.markdown(f"**Symbol:** `{trade['symbol']}`")
                with st.container(border=True):
                    st.markdown("### Live Indicators")
                    c1, c2 = st.columns(2)
                    c1.metric("Current LTP", f"₹{trade.get('ltp', 0):.2f}")
                    c2.metric("IV", f"{trade.get('iv', 0):.2f}%")
                    c1.metric("RSI", f"{trade.get('rsi', 0.0)}")
                    c2.metric("Trend", f"{trade.get('trend', 'N/A')}")
                    c1.metric("Sentiment", f"{trade.get('sentiment', 'N/A')} ({trade.get('confidence', 0):.0%})")

                with st.form("trade_form", clear_on_submit=False):
                    qty = st.number_input("Quantity", min_value=5, value=50, step=5, key="qty_input")
                    entry_low = st.number_input("Buy Range (Low)", value=round(trade.get("ltp", 0) * 0.98, 2), key="entry_low_input")
                    entry_high = st.number_input("Buy Range (High)", value=round(trade.get("ltp", 0) * 1.02, 2), key="entry_high_input")
                    target = st.number_input("Target Price", value=round(trade.get("ltp", 0) * 1.2, 2), key="target_input")
                    stop_loss = st.number_input("Stop Loss (Optional)", value=0.0, step=1.0, format="%.2f", key="sl_input")
                    submitted = st.form_submit_button("Monitor for Entry")
                    discard = st.form_submit_button("Discard Trade")

                    if submitted:
                        pending_trade = {
                            "status": "pending_entry",
                            "symbol": trade["symbol"],
                            "qty": qty,
                            "entry_low": entry_low,
                            "entry_high": entry_high,
                            "target": target,
                            "stop_loss": stop_loss if stop_loss > 0 else None,
                            "current_ltp": trade.get("ltp", 0) if trade.get("ltp") is not None else 0,
                            "instrument": trade["instrument"],
                        }
                        st.session_state.active_trades.append(pending_trade)
                        trade_logger.log_entry({"type": "pending_trade", **pending_trade})
                        trade_logger.save_active_trades(st.session_state.active_trades)
                        st.success(f"Started monitoring {trade['symbol']} for entry.")
                        st.rerun()

                    elif discard:
                        st.session_state.analysis_result = None
                        st.session_state.trade_details = {}
                        st.success("Trade discarded.")
                        st.rerun()

        st.divider()
        render_trade_monitor(fyers, data_svc, trade_logger)

    except Exception as e:
        st.error(f"Trade analysis page error: {e}")
        global_logger.error(f"Trade analysis page error: {e}")

def render_trade_logs_page(logger):
    st.header("Trade & Suggestion Logs")
    try:
        trades = logger.get_logs()
        suggestions = logger.get_logs(logger.SUGGESTION_FILE)

        st.subheader("Trade Logs")
        if trades:
            st.dataframe(pd.DataFrame(trades), use_container_width=True)
        else:
            st.info("No trade logs available.")

        st.subheader("Suggestion Logs")
        if suggestions:
            st.dataframe(pd.DataFrame(suggestions), use_container_width=True)
        else:
            st.info("No suggestion logs available.")
    except Exception as e:
        st.error(f"Error rendering logs: {e}")
        global_logger.error(f"Logs page error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
