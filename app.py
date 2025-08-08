import matplotlib # Import matplotlib first
matplotlib.use('Agg') # Set backend immediately

import backtrader as bt
import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
from kiteconnect import KiteConnect
from kiteconnect.exceptions import NetworkException # Import NetworkException
import threading
import webbrowser
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import json
import sys
import pytz # Import pytz for timezone awareness
from multiprocessing import Pool # For parallel execution
from flask import Flask, request # >>> THIS IS THE CRUCIAL LINE FOR FLASK <<<

# --- Configuration Loading (Recommended for API keys) ---
CONFIG_FILE = 'config.json'
config = {}
try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: {CONFIG_FILE} not found. Please create it with API_KEY, API_SECRET, and ZERODHA_USER_ID.", file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Invalid JSON in {CONFIG_FILE}.", file=sys.stderr)
    sys.exit(1)

API_KEY = config.get('API_KEY')
API_SECRET = config.get('API_SECRET')
ZERODHA_USER_ID = config.get('ZERODHA_USER_ID') # Useful for tracking orders

if not all([API_KEY, API_SECRET, ZERODHA_USER_ID]):
    print("Error: API_KEY, API_SECRET, or ZERODHA_USER_ID missing in config.json.", file=sys.stderr)
    sys.exit(1)

# Set up IST timezone
ist = pytz.timezone('Asia/Kolkata')

# --- Global Access Token Variable ---
# This will hold the access token obtained during initial authentication
# and will be passed to child processes for re-initializing KiteConnect.
global_access_token = None


# Function to calculate RSI manually using pandas
def calculate_rsi(data, period=14):
    delta = data.diff().dropna()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gains = gains.ewm(span=period, adjust=False, min_periods=period).mean()
    avg_losses = losses.ewm(span=period, adjust=False, min_periods=period).mean()
    
    rs = avg_gains / (avg_losses.replace(0, 1e-10))
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Flask for Kite Redirect
app = Flask(__name__)
request_token = None
flask_started = threading.Event()

@app.route('/login')
@app.route('/')
def login():
    global request_token
    request_token = request.args.get('request_token')
    if request_token:
        print(f"Captured request_token: {request_token}")
        return f"Request Token Received. You can close this tab."
    return "Error: No request token received"

def start_flask():
    """Starts the Flask server for OAuth redirect."""
    try:
        flask_started.set()
        app.run(host='127.0.0.1', port=5000, debug=False)
    except Exception as e:
        print(f"Flask server failed: {e}", file=sys.stderr)
        print("Please manually paste the request_token from browser.", file=sys.stderr)

def authenticate_kite(api_key, api_secret):
    """Authenticates with KiteConnect, handling the OAuth flow."""
    global request_token, global_access_token
    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()
    print(f"Open this URL in your browser to log in: {login_url}")
    
    # Start Flask in a separate thread to listen for the redirect
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    
    # Wait for Flask to confirm it's running
    flask_started.wait(timeout=10)
    if not flask_started.is_set():
        print("Warning: Flask server did not confirm startup. Proceeding anyway, but manual token entry might be needed.", file=sys.stderr)

    timeout = time.time() + 120
    while request_token is None and time.time() < timeout:
        time.sleep(1)
    
    if request_token is None:
        request_token = input("Flask server failed or timed out. Paste request_token from browser (from URL after login): ")
    
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data['access_token']
        kite.set_access_token(access_token)
        global_access_token = access_token # Store for global use, especially multiprocessing
        print(f"Generated access_token: {access_token}")
        
        # Test API access
        try:
            quote_data = kite.quote('NSE:NIFTY 50')
            print(f"API access test successful: NIFTY 50 last price = {quote_data['NSE:NIFTY 50']['last_price']}")
        except Exception as e:
            print(f"Warning: Failed to fetch NIFTY 50 quote for API access test: {e}", file=sys.stderr)
            print("Please ensure your API key and secret are correct and active.", file=sys.stderr)
            
        return kite, access_token
    except Exception as e:
        print(f"Authentication failed: {e}", file=sys.stderr)
        sys.exit(1)

# Dynamic Lot Size Calculation
def get_lot_size(kite, instrument_name):
    """Fetches lot size for an instrument from KiteConnect NFO segment."""
    try:
        instruments = pd.DataFrame(kite.instruments('NFO'))
        # Filter by name (e.g., 'NIFTY' or 'BANKNIFTY') and segment
        instrument_row = instruments[
            (instruments['name'] == instrument_name) & 
            (instruments['segment'].str.contains('OPT') | instruments['segment'].str.contains('FUT'))
        ]
        
        if not instrument_row.empty:
            return instrument_row.iloc[0].get('lot_size', 25)
    except Exception as e:
        print(f"Error fetching lot size for {instrument_name}: {e}", file=sys.stderr)
    
    print(f"Warning: Lot size not found for {instrument_name} from API. Defaulting to 25.", file=sys.stderr)
    return 25


# Option Selector
def get_option_chain(kite, instrument_name='NIFTY'):
    """Fetches and filters the option chain for a given instrument."""
    try:
        spot_symbol_quote = 'NSE:NIFTY 50' if instrument_name == 'NIFTY' else 'NSE:BANKNIFTY'
        spot_quote_data = kite.quote(spot_symbol_quote)
        if spot_symbol_quote not in spot_quote_data or 'last_price' not in spot_quote_data[spot_symbol_quote]:
            print(f"Could not fetch live spot price for {spot_symbol_quote}. Cannot select options.", file=sys.stderr)
            return pd.DataFrame(), 0 # Return 0 for spot_price if not found
        spot_price = spot_quote_data[spot_symbol_quote]['last_price']
        
        print(f"Fetching option chain for {instrument_name} with spot price: {spot_price:.2f}")

        instruments = kite.instruments('NFO')
        df = pd.DataFrame(instruments)
        df = df[(df['segment'] == 'NFO-OPT') & (df['name'] == instrument_name)]
        
        df['expiry'] = pd.to_datetime(df['expiry'])
        
        # Select the nearest weekly/monthly expiry, ensuring at least 1 day until expiry
        today = datetime.now(ist).date() # Use IST current date for timezone-aware comparison
        unique_expiries = sorted(df['expiry'].unique())
        
        nearest_expiry = None
        for exp in unique_expiries:
            if exp.date() > today: # Strictly greater than today's calendar date
                nearest_expiry = exp
                break
            # If current day is expiry day, and it's morning (before market close), can still consider
            # For simplicity and safety, prefer next expiry unless specifically handling expiry day.
            if exp.date() == today and datetime.now(ist).time() < ist.localize(datetime(2000,1,1,15,30)).time():
                if nearest_expiry is None: # Only if no future expiry found yet
                    nearest_expiry = exp

        if nearest_expiry:
            df = df[df['expiry'] == nearest_expiry]
            print(f"Selected nearest expiry: {nearest_expiry.strftime('%Y-%m-%d')}")
        else:
            print(f"No {instrument_name} options found with suitable expiry after today. Consider adjusting backtest dates.", file=sys.stderr)
            return pd.DataFrame(), spot_price # Return empty df and current spot price
        
        if df.empty:
            print(f"No {instrument_name} options found for selected expiry.", file=sys.stderr)
            return pd.DataFrame(), spot_price # Return empty df and current spot price

        # Fetch real-time quotes for volume and OI in batches
        symbols_to_quote = [f"NFO:{row['tradingsymbol']}" for _, row in df.iterrows()]
        
        all_quotes = {}
        batch_size = 50 # Max 50 instruments per request
        for i in range(0, len(symbols_to_quote), batch_size):
            batch_symbols = symbols_to_quote[i:i + batch_size]
            for attempt in range(3):  # Retry up to 3 times for rate limits or network issues
                try:
                    quotes_batch = kite.quote(batch_symbols)
                    all_quotes.update(quotes_batch)
                    time.sleep(0.2)   # Increased delay to 0.2s to avoid rate limits
                    break # Break if successful
                except NetworkException as e:
                    print(f"Retry {attempt + 1}/3 for batch {i}-{min(i+batch_size-1, len(symbols_to_quote)-1)}: Network error: {e}", file=sys.stderr)
                    time.sleep(1 + attempt)  # Exponential backoff
                except Exception as e:
                    print(f"Error fetching quotes for batch {i}-{min(i+batch_size-1, len(symbols_to_quote)-1)}: {e}", file=sys.stderr)
                    break # Break on other errors
        
        # Add volume, OI, and last_price to DataFrame from fetched quotes
        df['last_volume'] = df['tradingsymbol'].apply(lambda x: all_quotes.get(f"NFO:{x}", {}).get('volume', 0))
        df['oi'] = df['tradingsymbol'].apply(lambda x: all_quotes.get(f"NFO:{x}", {}).get('oi', 0))
        df['last_price'] = df['tradingsymbol'].apply(lambda x: all_quotes.get(f"NFO:{x}", {}).get('last_price', 0))
        
        df = df[df['last_price'] > 0] # Filter out options with no current price (illiquid or expired)
        
        if df.empty:
            print("No active options found after filtering for last price > 0.", file=sys.stderr)
            return pd.DataFrame(), spot_price # Return empty df and current spot price

        df['abs_diff'] = abs(df['strike'] - spot_price)
        # Rank by volume (higher is better), OI (higher is better), and absolute difference from spot (lower is better)
        df['volume_rank'] = df['last_volume'].rank(ascending=False, method='min')
        df['oi_rank'] = df['oi'].rank(ascending=False, method='min')
        df['abs_diff_rank'] = df['abs_diff'].rank(ascending=True, method='min') 

        # Combined scoring: emphasizes proximity to ATM, then liquidity
        # Scoring with inverse ranks means lower rank (better) gives higher score
        df['score'] = (
            0.4 * (1 / (df['abs_diff'] + 1)) +  # Smaller diff -> higher score
            0.3 * (1 / (df['volume_rank'] + 1)) + # Smaller rank (higher volume) -> higher score
            0.3 * (1 / (df['oi_rank'] + 1))       # Smaller rank (higher OI) -> higher score
        )
        # This new scoring means higher score is ALWAYS better.
        
        df = df.sort_values('score', ascending=False)
        
        return df[['tradingsymbol', 'strike', 'instrument_type', 'last_price', 'last_volume', 'oi', 'abs_diff', 'score']], spot_price
    except Exception as e:
        print(f"Failed to fetch option chain: {e}", file=sys.stderr)
        return pd.DataFrame(), 0 # Return empty df and 0 for spot_price

def select_option(df, strategy_signal, spot_price):
    """Selects an optimal option based on strategy signal and option chain data."""
    if df.empty:
        print("No options available. Cannot select a symbol.", file=sys.stderr)
        return None, 'UNKNOWN' # Return symbol=None and type='UNKNOWN'
    
    selected_option_df = pd.DataFrame()
    
    # Filter for the relevant option type first (CE for bullish, PE for bearish)
    if strategy_signal['direction'] == 'bullish':
        selected_option_df = df[df['instrument_type'] == 'CE'].copy()
        
    elif strategy_signal['direction'] == 'bearish':
        selected_option_df = df[df['instrument_type'] == 'PE'].copy()
    else:
        print(f"Invalid strategy direction: {strategy_signal['direction']}. No option type selected.", file=sys.stderr)
        return None, 'UNKNOWN' # Return symbol=None and type='UNKNOWN'
    
    if selected_option_df.empty:
        print(f"No suitable {strategy_signal['direction']} options found in the filtered chain. Cannot select.", file=sys.stderr)
        return None, 'UNKNOWN' # Return symbol=None and type='UNKNOWN'

    # Prioritize ATM or slightly OTM options for liquidity and affordability
    ATM_RANGE_POINTS = 200 # Adjust this based on Nifty's typical strike intervals and desired moneyness
    
    atm_options = selected_option_df[selected_option_df['abs_diff'] <= ATM_RANGE_POINTS].copy()
    
    chosen_option = None
    if not atm_options.empty:
        # Within ATM range, pick the one with the highest score (considering volume, OI, and proximity)
        chosen_option = atm_options.sort_values('score', ascending=False).iloc[0]
        print(f"Selected ATM option: {chosen_option['tradingsymbol']}, Strike: {chosen_option['strike']}, Type: {chosen_option['instrument_type']}, Score: {chosen_option['score']:.2f}, Price: {chosen_option['last_price']:.2f}")
    else:
        print(f"No ATM options found within {ATM_RANGE_POINTS} points. Falling back to best scored option from full list.", file=sys.stderr)
        # If no options within the tight ATM range, pick the best overall scored option
        chosen_option = selected_option_df.sort_values('score', ascending=False).iloc[0]
        print(f"Selected best scored non-ATM option: {chosen_option['tradingsymbol']}, Strike: {chosen_option['strike']}, Type: {chosen_option['instrument_type']}, Score: {chosen_option['score']:.2f}, Price: {chosen_option['last_price']:.2f}")

    if chosen_option is not None:
        return chosen_option['tradingsymbol'], chosen_option['instrument_type'] # Return both symbol and type
    else:
        return None, 'UNKNOWN' # Should not happen if selected_option_df is not empty

# AI Trade Suggester
def train_trade_suggester(log_file='trade_logs.csv'):
    """Trains an AI model to suggest trade outcomes based on historical logs."""
    try:
        df = pd.read_csv(log_file)
        # Ensure sufficient data for training - lowered threshold to 30
        if df.empty or len(df) < 30: 
            print("Not enough trade logs for AI training (need at least 30). AI model training skipped.", file=sys.stderr)
            return None
        
        # Ensure features are numeric and handle NaNs
        # Added bb_width_at_entry, macd_hist_at_entry as potential features
        features = ['rsi_at_entry', 'atr_at_entry', 'volume_at_entry', 'bb_width_at_entry', 'macd_hist_at_entry']
        # Filter for only existing columns from the log file
        features = [f for f in features if f in df.columns]

        if not features: # If no features are available, cannot train
            print("No valid features found in trade logs for AI training. AI model training skipped.", file=sys.stderr)
            return None

        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=features + ['trade_outcome'])
        
        if df.empty:
            print("No valid data after cleaning trade logs for AI training.", file=sys.stderr)
            return None

        X = df[features]
        y = df['trade_outcome'].apply(lambda x: 1 if x == 'win' else 0)
        
        # Ensure there are at least two classes for classification (wins and losses)
        if y.nunique() < 2:
            print("Trade logs only contain one outcome type ('win' or 'loss'). Cannot train AI for classification.", file=sys.stderr)
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratify for balanced classes
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced') # Use balanced for imbalanced win/loss
        model.fit(X_train, y_train)
        
        print(f"AI Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
        return model
    except FileNotFoundError:
        print("No trade logs found. AI model training skipped.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error training AI model: {e}", file=sys.stderr)
        return None

def suggest_trade(model, current_data):
    """Predicts if a trade is likely to be a win based on current market data."""
    if model is None:
        # Default to 'enter' if no model is trained yet (to generate initial logs for training)
        return {'trade': 'enter', 'confidence': 0.0}
    
    # Define all possible features AI might be trained on
    all_possible_features = ['rsi_at_entry', 'atr_at_entry', 'volume_at_entry', 'bb_width_at_entry', 'macd_hist_at_entry']
    
    # Filter for only features that exist in the current_data DataFrame (which must match AI training)
    features_for_prediction = [f for f in all_possible_features if f in current_data.columns]
    
    if not features_for_prediction:
        print(f"Error: No valid features found in current_data for AI suggestion. Returning skip.", file=sys.stderr)
        return {'trade': 'skip', 'confidence': 0.0}

    # Ensure current_data has the expected columns, fill NaNs if necessary (e.g., at start of data)
    current_data = current_data[features_for_prediction].fillna(current_data[features_for_prediction].mean(numeric_only=True))
    
    if current_data.empty or current_data.isnull().values.any(): # Check for NaNs even after filling
        print("Current data for AI is empty or contains NaNs after processing. Returning skip.", file=sys.stderr)
        return {'trade': 'skip', 'confidence': 0.0}

    try:
        prediction_proba = model.predict_proba(current_data[features_for_prediction])[0] # Select only relevant columns
        prediction = model.predict(current_data[features_for_prediction])[0]
        
        confidence = prediction_proba[prediction]
        return {'trade': 'enter' if prediction == 1 else 'skip', 'confidence': confidence}
    except Exception as e:
        print(f"Error during AI prediction: {e}. Returning skip.", file=sys.stderr)
        return {'trade': 'skip', 'confidence': 0.0}


# Intraday Base Strategy
class IntradayBaseStrategy(bt.Strategy):
    params = (
        ('initial_capital', 15000), # Initial capital for daily loss limit
        ('lot_size', 25),            # Lot size for the instrument (dynamically fetched)
        ('size_percent_capital', 0.05), # 5% of capital to allocate per trade
        ('stop_loss_perc', 0.01),  # 1% fixed percentage stop-loss (will be overridden by ATR if dynamic)
        ('profit_target_perc', 0.02), # 2% fixed percentage profit target (will be overridden by ATR if dynamic)
        ('max_hold_minutes', 60), # 60 minutes max hold
        ('ai_confidence_threshold', 0.0), # AI confidence required to take a trade (Set to 0.0 for initial log generation)
        ('atr_period', 14),    # ATR period
        ('rsi_period', 14),    # RSI period
        ('actual_option_type', 'UNKNOWN'), # Passed from main to strategy
        ('stop_loss_atr_mult', 1.5),  # ATR multiplier for dynamic stop-loss
        ('profit_target_atr_mult', 2.0), # ATR multiplier for dynamic profit target
        ('bb_period', 20), # Bollinger Band period (for AI feature and some strategies)
        ('bb_dev', 2.0),     # Bollinger Band deviation (for AI feature and some strategies)
        ('macd_fast', 8),  # MACD fast period (for AI feature)
        ('macd_slow', 21), # MACD slow period (for AI feature)
        ('macd_signal', 5),# MACD signal period (for AI feature)
        ('vol_period', 14), # Volume SMA period (for AI feature and some strategies)
    )

    def __init__(self):
        super().__init__()
        # Reset state variables for each new strategy run (important if cerebro is reused implicitly)
        self.entry_price = None
        self.position_opened = False 
        self.entry_time = None
        self.order = None # To keep track of pending orders
        self.current_trade_entry_data = {} # To store entry data for logging on exit
        
        self.actual_option_type = self.p.actual_option_type # Store it in an instance variable (use self.p for params)
        self.log(f"Strategy initialized with Loaded Option Type: '{self.actual_option_type}'") 
        
        # --- FIX: Determine the correct data source for indicators ---
        # self.datas[0] is always the option data
        # self.datas[1] is the underlying data (if added)

        self.indicator_data_source = None
        self.using_underlying_data = False

        if len(self.datas) > 1 and self.datas[1] is not None and not self.datas[1].empty:
            # Check if underlying data has enough bars (at least 1 to prevent immediate errors)
            # More thorough checks for sufficient bars for indicators are in next()
            if len(self.datas[1]) >= 1: 
                self.indicator_data_source = self.datas[1]
                self.using_underlying_data = True
                self.log("Using UNDERLYING data for indicators.", dt=datetime.now(ist))
            else:
                self.log("Warning: Underlying data feed exists but is empty. Falling back to option data for indicators.", dt=datetime.now(ist))
        
        if self.indicator_data_source is None: # If not set by underlying data, use primary option data
            self.indicator_data_source = self.datas[0]
            self.log("Using OPTION data (fallback) for indicators.", dt=datetime.now(ist))

        # Assign close, volume, OI lines from the chosen data source
        self.underlying_close = self.indicator_data_source.close
        self.underlying_volume = self.indicator_data_source.volume
        # Check for openinterest on the chosen data source, fall back to volume if not available
        self.underlying_oi = self.indicator_data_source.openinterest if hasattr(self.indicator_data_source.lines, 'openinterest') else bt.indicators.SMA(self.indicator_data_source.volume, period=1)


        # Initialize indicators using the determined 'self.indicator_data_source'
        # This allows them to correctly access 'high', 'low', 'open', etc.
        self.underlying_atr = bt.indicators.ATR(self.indicator_data_source, period=self.p.atr_period)
        self.underlying_rsi = bt.indicators.RSI(self.indicator_data_source.close, period=self.p.rsi_period)
        self.underlying_bb = bt.indicators.BollingerBands(self.indicator_data_source, period=self.p.bb_period, devfactor=self.p.bb_dev)
        self.underlying_macd_obj = bt.indicators.MACD(self.indicator_data_source.close, period_me1=self.p.macd_fast, period_me2=self.p.macd_slow, period_signal=self.p.macd_signal)
        self.underlying_macd_hist = self.underlying_macd_obj.macd - self.underlying_macd_obj.signal

        self.model = train_trade_suggester()
        # Capture initial cash from broker at the very start of the backtest
        # This will be used for daily loss limit calculation.
        self.initial_portfolio_value = self.broker.getcash() 

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.isoformat()} [{self.__class__.__name__}]: {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            self.log(f"Order '{order.getordername()}' Status: {order.getstatusname()}")
            return

        if order.status in [order.Completed]:
            if order.isbuy:
                self.log(f"BUY EXECUTED: Symbol={self.datas[0]._name}, Size={order.executed.size}, Price={order.executed.price:.2f}, Total Value={order.executed.value:.2f}, Commission={order.executed.comm:.2f}")
                self.entry_price = order.executed.price
                self.position_opened = True
                self.entry_time = self.datas[0].datetime.datetime(0)
                
                # Extract strike and expiry for logging (best effort parsing)
                strike_price = np.nan
                expiry_date_str = 'N/A'
                try:
                    symbol_name = self.datas[0]._name
                    import re
                    # Updated regex to be more flexible for Zerodha's common patterns
                    # E.g., NIFTY2581424500CE -> extract 24500
                    strike_match = re.search(r'(\d{5})(CE|PE)?$', symbol_name) # Catches 5 digits followed by CE/PE or end
                    if strike_match:
                        strike_price = float(strike_match.group(1))
                    
                    # Expiry can be tricky to parse reliably from all tradingsymbols without a full lookup.
                    # For example, 25814 might mean Aug 14, 2025.
                    # If symbol format is consistently YYMDD or YYYYMMDD, it's parseable.
                    # For now, leaving as N/A unless a precise pattern is guaranteed.
                    pass 
                except Exception as ex:
                    self.log(f"Error parsing option symbol for strike/expiry: {ex}", dt=self.entry_time)

                # Store entry data for AI logging later
                self.current_trade_entry_data = {
                    'datetime': self.entry_time,
                    'symbol': self.datas[0]._name,
                    'entry_price': self.entry_price,
                    'rsi_at_entry': self.underlying_rsi[0] if len(self.underlying_rsi) > 0 else np.nan,
                    'atr_at_entry': self.underlying_atr[0] if len(self.underlying_atr) > 0 else np.nan,
                    'volume_at_entry': self.underlying_volume[0] if len(self.underlying_volume) > 0 else np.nan, 
                    'bb_width_at_entry': (self.underlying_bb.top[0] - self.underlying_bb.bot[0]) if len(self.underlying_bb) > 0 else np.nan,
                    'macd_hist_at_entry': self.underlying_macd_hist[0] if len(self.underlying_macd_hist) > 0 else np.nan,
                    'is_buy': True,
                    'strategy': self.__class__.__name__, # Log strategy name
                    'strike': strike_price, # Log extracted strike
                    'expiry': expiry_date_str # Log extracted expiry
                }
            elif order.issell:
                self.log(f"SELL EXECUTED: Symbol={self.datas[0]._name}, Size={order.executed.size}, Price={order.executed.price:.2f}, Total Value={order.executed.value:.2f}, Commission={order.executed.comm:.2f}, PnL={order.executed.pnl:.2f}")
                
                # Log trade outcome when position is closed
                if hasattr(self, 'current_trade_entry_data') and self.current_trade_entry_data:
                    log_data = {
                        **self.current_trade_entry_data,
                        'exit_price': order.executed.price,
                        'pnl': order.executed.pnl,
                        'trade_outcome': 'win' if order.executed.pnl > 0 else 'loss',
                        'duration_minutes': (self.datas[0].datetime.datetime(0) - self.current_trade_entry_data['datetime']).total_seconds() / 60
                    }
                    log_df = pd.DataFrame([log_data])
                    log_df.to_csv('trade_logs.csv', mode='a', header=not os.path.exists('trade_logs.csv'), index=False)
                    self.current_trade_entry_data = {} # Clear after logging
                
                pass 

            self.order = None 

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order '{order.getordername()}' Status: {order.getstatusname()}. Clearing pending order reference.")
            self.order = None 
            if order.isbuy and not self.position: 
                 self.position_opened = False

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"TRADE PNL: Gross {trade.pnl:.2f}, Net {trade.pnlcomm:.2f}")
    
    def next(self):
        current_price = self.data.close[0] # This is the option's current price
        cash = self.broker.get_cash()
        current_time = self.datas[0].datetime.datetime(0) 
        
        min_periods_needed = max(self.p.rsi_period, self.p.atr_period, self.p.bb_period, self.p.macd_slow, self.p.vol_period)
        
        # Check if the chosen indicator data source has enough bars
        if len(self.indicator_data_source) < min_periods_needed:
            return 
            
        if self.broker.getvalue() < self.initial_portfolio_value * 0.95 and self.position:
            self.log("Daily loss limit reached. Closing all positions.")
            self.close() 
            self.position_opened = False 
            return 

        ai_data = pd.DataFrame({
            'rsi_at_entry': [self.underlying_rsi[0]],
            'atr_at_entry': [self.underlying_atr[0]],
            'volume_at_entry': [self.underlying_volume[0]], 
            'bb_width_at_entry': [(self.underlying_bb.top[0] - self.underlying_bb.bot[0])],
            'macd_hist_at_entry': [self.underlying_macd_hist[0]]
        })
        
        ai_data = ai_data.fillna(ai_data.mean(numeric_only=True)) 

        suggestion = suggest_trade(self.model, ai_data)
        
        self.log(f"AI Suggestion: {suggestion['trade']}, Confidence: {suggestion['confidence']:.2f}")

        if self.order: 
            self.log(f"Pending order detected: {self.order.getordername()}. Skipping new actions.")
            return 
        
        self.position_opened = bool(self.position) 

        if self.position_opened: 
            is_long = self.position.size > 0
            
            if len(self.underlying_atr) > 0 and self.underlying_atr[0] > 0: 
                stop_price = self.entry_price - (self.underlying_atr[0] * self.p.stop_loss_atr_mult)
                profit_price = self.entry_price + (self.underlying_atr[0] * self.p.profit_target_atr_mult)
            else: 
                stop_price = self.entry_price * (1 - self.p.stop_loss_perc)
                profit_price = self.entry_price * (1 + self.p.profit_target_perc)

            minutes_held = (current_time - self.entry_time).total_seconds() / 60
            
            if current_price <= stop_price: 
                self.log(f"Stop-loss triggered at {current_price:.2f} (Entry: {self.entry_price:.2f}, SL: {stop_price:.2f}). Closing position.")
                self.close() 
            elif current_price >= profit_price: 
                self.log(f"Profit target hit at {current_price:.2f} (Entry: {self.entry_price:.2f}, PT: {profit_price:.2f}). Closing position.")
                self.close() 
            elif minutes_held >= self.p.max_hold_minutes: 
                self.log(f"Max hold period reached ({minutes_held:.0f} mins) at {current_price:.2f}. Closing position.")
                self.close() 
        
        elif not self.position_opened: 
            strategy_signal = self.should_enter(current_price) 
            
            if strategy_signal:
                self.log(f"Strategy Signal: Action='{strategy_signal['action']}', Direction='{strategy_signal['direction']}'. Loaded Option Type: '{self.actual_option_type}'.")
            else:
                self.log(f"No Strategy Signal. Loaded Option Type: '{self.actual_option_type}'.")

            if strategy_signal and \
                suggestion['trade'] == 'enter' and \
                suggestion['confidence'] >= self.p.ai_confidence_threshold:
                
                can_execute = False
                if strategy_signal['action'] == 'buy':
                    if self.actual_option_type == 'PE' and strategy_signal['direction'] == 'bearish':
                        can_execute = True 
                    elif self.actual_option_type == 'CE' and strategy_signal['direction'] == 'bullish':
                        can_execute = True 
                
                if can_execute:
                    current_price_per_unit = current_price 
                    
                    if current_price_per_unit <= 0:
                        self.log("Option current price is zero or negative. Cannot place trade.")
                        return

                    target_capital_for_trade = self.initial_portfolio_value * self.p.size_percent_capital
                    cost_per_lot = current_price_per_unit * self.p.lot_size
                    
                    if cost_per_lot <= 0:
                        self.log("Calculated cost per lot is zero or negative. Cannot place trade.")
                        return

                    num_lots_from_allocation = int(target_capital_for_trade / cost_per_lot)
                    
                    calculated_lots = num_lots_from_allocation
                    if calculated_lots == 0:
                        if (cash / cost_per_lot) >= 1: 
                            calculated_lots = 1
                        else:
                            self.log("Not enough cash for even one full lot with current market price. Cannot place trade.")
                            return
                    
                    size_for_trade = calculated_lots * self.p.lot_size
                    
                    if size_for_trade == 0: 
                        self.log("Calculated size is zero after all adjustments. Cannot place trade.")
                        return
                    
                    self.log(f"Placing Order: {strategy_signal['action'].upper()} {size_for_trade} units of {self.actual_option_type} at {current_price:.2f}. AI Conf: {suggestion['confidence']:.2f}")
                    self.order = self.buy(size=size_for_trade)
                else:
                    self.log(f"Skipping trade: Signal direction '{strategy_signal['direction']}' or action '{strategy_signal['action']}' does not match loaded option type '{self.actual_option_type}' for buying. Or invalid action.")
            else:
                if not strategy_signal:
                    self.log(f"Skipping trade: No strategy signal generated.")
                elif suggestion['trade'] != 'enter':
                    self.log(f"Skipping trade: AI suggested '{suggestion['trade']}' (confidence: {suggestion['confidence']:.2f}).")
                elif suggestion['confidence'] < self.p.ai_confidence_threshold:
                    self.log(f"Skipping trade: AI confidence ({suggestion['confidence']:.2f}) below threshold ({self.p.ai_confidence_threshold}).")

    def should_enter(self, current_price):
        """
        Define entry logic in subclass. Must return:
        {'action': 'buy' or 'sell', 'direction': 'bullish' or 'bearish'} or None
        """
        raise NotImplementedError("Define entry logic in subclass")

# Intraday RSI Mean Reversion (CustomRSIMeanReversion)
class CustomRSIMeanReversion(IntradayBaseStrategy):
    params = (
        ('rsi_period', 14),
        ('rsi_lower', 30), 
        ('rsi_upper', 70), 
        ('atr_period', 14), 
        ('atr_sma_period', 10), 
    )

    def __init__(self):
        super().__init__()
        self.atr_sma = bt.indicators.SMA(self.underlying_atr, period=self.p.atr_sma_period)

    def should_enter(self, current_price):
        min_periods_needed = max(self.p.rsi_period, self.p.atr_period, self.p.atr_sma_period)
        
        if len(self.indicator_data_source) < min_periods_needed:
            return None
            
        indicator_dt = self.indicator_data_source.datetime.datetime(0)
        self.log(f"RSI: {self.underlying_rsi[0]:.2f}, ATR: {self.underlying_atr[0]:.2f}, ATR_SMA: {self.atr_sma[0]:.2f}", dt=indicator_dt)
        
        if self.underlying_rsi[0] < self.p.rsi_lower and self.underlying_atr[0] > self.atr_sma[0]:
            if self.actual_option_type == 'CE': 
                self.log(f"RSI Buy Signal (Bullish): RSI={self.underlying_rsi[0]:.2f}, ATR={self.underlying_atr[0]:.2f}, Option Price={current_price:.2f}", dt=indicator_dt)
                return {'action': 'buy', 'direction': 'bullish'}
            self.log(f"RSI Buy Signal (Bullish) but Loaded Option is PE. Skipping.", dt=indicator_dt)
            
        elif self.underlying_rsi[0] > self.p.rsi_upper and self.underlying_atr[0] > self.atr_sma[0]:
            if self.actual_option_type == 'PE': 
                self.log(f"RSI Sell Signal (Bearish): RSI={self.underlying_rsi[0]:.2f}, ATR={self.underlying_atr[0]:.2f}, Option Price={current_price:.2f}", dt=indicator_dt)
                return {'action': 'buy', 'direction': 'bearish'}
            self.log(f"RSI Sell Signal (Bearish) but Loaded Option is CE. Skipping.", dt=indicator_dt)
            
        return None 

# Intraday Bollinger Breakout (CustomBollingerBreakout)
class CustomBollingerBreakout(IntradayBaseStrategy):
    params = (
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('rsi_period', 14), 
        ('rsi_threshold_buy', 60), 
        ('rsi_threshold_sell', 40), 
        ('vol_period', 14), 
    )

    def __init__(self):
        super().__init__()
        self.bb = bt.indicators.BollingerBands(self.indicator_data_source, period=self.p.bb_period, devfactor=self.p.bb_dev)
        
        self.oi_or_vol_for_sma = self.underlying_oi if hasattr(self.indicator_data_source.lines, 'openinterest') else self.underlying_volume
        self.vol_sma = bt.indicators.SMA(self.oi_or_vol_for_sma, period=self.p.vol_period) 
        
    def should_enter(self, current_price): 
        min_periods_needed = max(self.p.bb_period, self.p.vol_period, self.p.rsi_period)
        
        if len(self.indicator_data_source) < min_periods_needed:
            return None

        underlying_current_close = self.underlying_close[0]
        indicator_dt = self.indicator_data_source.datetime.datetime(0)

        self.log(f"Underlying Close: {underlying_current_close:.2f}, BB_Top: {self.bb.top[0]:.2f}, BB_Bot: {self.bb.bot[0]:.2f}, Activity: {self.oi_or_vol_for_sma[0]:.2f}, Activity_SMA: {self.vol_sma[0]:.2f}, RSI: {self.underlying_rsi[0]:.2f}", dt=indicator_dt)

        if underlying_current_close > self.bb.top[0] and \
           self.underlying_rsi[0] > self.p.rsi_threshold_buy and \
           self.oi_or_vol_for_sma[0] > self.vol_sma[0]:
            if self.actual_option_type == 'CE': 
                self.log(f"Bullish Breakout Signal: Price={underlying_current_close:.2f}, Option Price={current_price:.2f}, Activity={self.oi_or_vol_for_sma[0]:.2f}", dt=indicator_dt)
                return {'action': 'buy', 'direction': 'bullish'}
            else:
                self.log(f"Bullish Breakout Signal but Loaded Option is PE. Skipping.", dt=indicator_dt)
                return None
        
        elif underlying_current_close < self.bb.bot[0] and \
             self.underlying_rsi[0] < self.p.rsi_threshold_sell and \
             self.oi_or_vol_for_sma[0] > self.vol_sma[0]:
            if self.actual_option_type == 'PE': 
                self.log(f"Bearish Breakout Signal: Price={underlying_current_close:.2f}, Option Price={current_price:.2f}, Activity={self.oi_or_vol_for_sma[0]:.2f}", dt=indicator_dt)
                return {'action': 'buy', 'direction': 'bearish'}
            else:
                self.log(f"Bearish Breakout Signal but Loaded Option is CE. Skipping.", dt=indicator_dt)
                return None
        return None 

# New Strategy: SMA Crossover (CustomSMACrossover)
class CustomSMACrossover(IntradayBaseStrategy):
    params = (
        ('fast_period', 5),
        ('slow_period', 15),
        ('roc_period', 14), 
    )

    def __init__(self):
        super().__init__()
        self.fast_sma = bt.indicators.SMA(self.underlying_close, period=self.p.fast_period)
        self.slow_sma = bt.indicators.SMA(self.underlying_close, period=self.p.slow_period)
        self.roc = bt.indicators.ROC(self.underlying_close, period=self.p.roc_period)
        
    def should_enter(self, current_price):
        min_periods_needed = max(self.p.fast_period, self.p.slow_period, self.p.roc_period)
        
        if len(self.indicator_data_source) < min_periods_needed:
            return None

        indicator_dt = self.indicator_data_source.datetime.datetime(0)
        self.log(f"Fast SMA: {self.fast_sma[0]:.2f}, Slow SMA: {self.slow_sma[0]:.2f}, ROC: {self.roc[0]:.2f}", dt=indicator_dt)

        if self.fast_sma[0] > self.slow_sma[0] and self.fast_sma[-1] <= self.slow_sma[-1] and self.roc[0] > 0:
            if self.actual_option_type == 'CE': 
                self.log(f"SMA Crossover Buy Signal (Bullish): Fast={self.fast_sma[0]:.2f}, Slow={self.slow_sma[0]:.2f}, ROC={self.roc[0]:.2f}, Option Price={current_price:.2f}", dt=indicator_dt)
                return {'action': 'buy', 'direction': 'bullish'}
            else:
                self.log(f"SMA Crossover Buy Signal (Bullish) but Loaded Option is PE. Skipping.", dt=indicator_dt)
                return None
        
        elif self.fast_sma[0] < self.slow_sma[0] and self.fast_sma[-1] >= self.slow_sma[-1] and self.roc[0] < 0:
            if self.actual_option_type == 'PE': 
                self.log(f"SMA Crossover Sell Signal (Bearish): Fast={self.fast_sma[0]:.2f}, Slow={self.slow_sma[0]:.2f}, ROC={self.roc[0]:.2f}, Option Price={current_price:.2f}", dt=indicator_dt)
                return {'action': 'buy', 'direction': 'bearish'} 
            else:
                self.log(f"SMA Crossover Sell Signal (Bearish) but Loaded Option is CE. Skipping.", dt=indicator_dt)
                return None
        return None 

# New Strategy: MACD Crossover (CustomMACDCrossover)
class CustomMACDCrossover(IntradayBaseStrategy):
    params = (
        ('macd_fast', 8),
        ('macd_slow', 21),
        ('macd_signal', 5),
        ('vol_period', 14), 
    )

    def __init__(self):
        super().__init__()
        self.macd_obj = bt.indicators.MACD(self.indicator_data_source.close, period_me1=self.p.macd_fast, period_me2=self.p.macd_slow, period_signal=self.p.macd_signal)
        self.macd_line = self.macd_obj.macd
        self.signal_line = self.macd_obj.signal
        
        self.oi_or_vol_for_sma = self.underlying_oi if hasattr(self.indicator_data_source.lines, 'openinterest') else self.underlying_volume
        self.vol_sma = bt.indicators.SMA(self.oi_or_vol_for_sma, period=self.p.vol_period)
        
    def should_enter(self, current_price):
        min_periods_needed = max(self.p.macd_slow, self.p.macd_signal, self.p.vol_period)
        
        if len(self.indicator_data_source) < min_periods_needed:
            return None

        indicator_dt = self.indicator_data_source.datetime.datetime(0)
        self.log(f"MACD: {self.macd_line[0]:.2f}, Signal: {self.signal_line[0]:.2f}, Activity: {self.oi_or_vol_for_sma[0]:.2f}, Activity_SMA: {self.vol_sma[0]:.2f}", dt=indicator_dt)

        if self.macd_line[0] > self.signal_line[0] and self.macd_line[-1] <= self.signal_line[-1] and self.oi_or_vol_for_sma[0] > self.vol_sma[0]:
            if self.actual_option_type == 'CE': 
                self.log(f"MACD Crossover Buy Signal (Bullish): MACD={self.macd_line[0]:.2f}, Signal={self.signal_line[0]:.2f}, Option Price={current_price:.2f}", dt=indicator_dt)
                return {'action': 'buy', 'direction': 'bullish'}
            else: 
                self.log(f"MACD Crossover Buy Signal (Bullish) but Loaded Option is PE. Skipping.", dt=indicator_dt)
                return None

        if self.macd_line[0] < self.signal_line[0] and self.macd_line[-1] >= self.signal_line[-1] and self.oi_or_vol_for_sma[0] > self.vol_sma[0]:
            if self.actual_option_type == 'PE': 
                self.log(f"MACD Crossover Sell Signal (Bearish): MACD={self.macd_line[0]:.2f}, Signal={self.signal_line[0]:.2f}, Option Price={current_price:.2f}", dt=indicator_dt)
                return {'action': 'buy', 'direction': 'bearish'} 
            else: 
                self.log(f"MACD Crossover Sell Signal (Bearish) but Loaded Option is CE. Skipping.", dt=indicator_dt)
                return None
        return None 


# Function to place live orders (called from run_strategy in live mode)
def place_live_order(kite_conn, symbol, size, action):
    """
    Places a live order via KiteConnect.
    :param kite_conn: Authenticated KiteConnect object.
    :param symbol: Trading symbol (e.g., 'NFO:NIFTY25AUG2525000CE').
    :param size: Quantity of units to trade (should be a multiple of lot size).
    :param action: 'BUY' or 'SELL'.
    """
    if size <= 0:
        print(f"WARNING: Attempted to place live order with size {size}. Skipping.", file=sys.stderr)
        return

    # Real-Time Market Hours Check
    if not is_market_open():
        print("ERROR: Live order attempted outside market hours. Skipping order.", file=sys.stderr)
        return

    try:
        margin_response = kite_conn.margins(segment='equity') 
        available_cash_margin = margin_response.get('equity', {}).get('available', {}).get('cash', 0)
        
        quote = kite_conn.quote(f"NFO:{symbol}") 
        live_price = quote.get(f"NFO:{symbol}", {}).get('last_price', 0)
        
        if live_price > 0:
            estimated_order_cost = size * live_price
        else:
            estimated_order_cost = size * 100 
            print(f"Warning: Could not get live price for {symbol}, using estimated cost {estimated_order_cost:.2f}.", file=sys.stderr)
            
        if available_cash_margin < estimated_order_cost:
            print(f"Insufficient margin for {symbol} ({action} {size}). Available: {available_cash_margin:.2f}, Needed (estimated): {estimated_order_cost:.2f}. Skipping.", file=sys.stderr)
            return
        
        confirm = input(f"Confirm live {action} order for {size} units of {symbol}? (y/n): ")
        if confirm.lower() != 'y':
            print("Live order cancelled by user.", file=sys.stderr)
            return

        order = kite_conn.place_order(
            variety='regular', exchange='NFO', tradingsymbol=symbol,
            transaction_type=action.upper(), quantity=size, product='MIS', order_type='MARKET' 
        )
        print(f"Successfully placed live order: {order}")
    except Exception as e:
        print(f"ERROR: Live order placement failed for {symbol} ({action} {size}): {e}", file=sys.stderr)

# Real-Time Market Hours Check
def is_market_open():
    """Checks if current time is within Indian market hours (9:15 AM to 3:30 PM IST, Mon-Fri)."""
    now = datetime.now(ist)
    if not (0 <= now.weekday() <= 4): 
        return False
    
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close


# Function to fetch historical data
def fetch_historical_data(kite, symbol_name, start_date, end_date, interval='5minute', exchange_segment='NFO'):
    """Fetches historical data for a given symbol from KiteConnect."""
    try:
        instrument_token = None
        instrument_type = 'UNKNOWN' 
        
        instruments_in_segment = pd.DataFrame(kite.instruments(exchange_segment)) 
        
        if exchange_segment == 'NSE':
            instrument_row = instruments_in_segment[
                (instruments_in_segment['exchange'] == 'NSE') & 
                (instruments_in_segment['tradingsymbol'] == symbol_name)
            ]
            
            if instrument_row.empty:
                print(f"Attempting broader search for {symbol_name} in all segments for token...", file=sys.stderr)
                all_instruments_df = pd.DataFrame(kite.instruments()) 
                instrument_row = all_instruments_df[
                    (all_instruments_df['exchange'] == 'NSE') & 
                    (all_instruments_df['tradingsymbol'] == symbol_name)
                ]
            
            if not instrument_row.empty:
                instrument_type = instrument_row.iloc[0].get('instrument_type', 'INDEX') 
        
        else: 
            instrument_row = instruments_in_segment[
                (instruments_in_segment['segment'].str.contains('OPT') | instruments_in_segment['segment'].str.contains('FUT')) &
                (instruments_in_segment['tradingsymbol'] == symbol_name)
            ]
            if not instrument_row.empty:
                instrument_type = instrument_row.iloc[0].get('instrument_type', 'UNKNOWN_NFO')

        if not instrument_row.empty:
            instrument_token = instrument_row.iloc[0]['instrument_token']
        
        if instrument_token is None:
            print(f"Instrument token not found for {symbol_name} in {exchange_segment} segment.", file=sys.stderr)
            return pd.DataFrame(), 'UNKNOWN' 

        print(f"Fetching historical data for {symbol_name} (Token: {instrument_token}) from {start_date} to {end_date} at {interval} interval.")
        
        data = kite.historical_data(instrument_token, start_date, end_date, interval, continuous=False, oi=True) 
        df = pd.DataFrame(data)
        
        if not df.empty:
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'oi']]
            df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest'] 
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.dropna(subset=['close'])
            df = df.set_index('datetime') 
        
        return df, instrument_type 
    except Exception as e:
        print(f"Failed to fetch historical data for {symbol_name}: {e}", file=sys.stderr)
        return pd.DataFrame(), 'UNKNOWN' 

# Function to run a single strategy
# THIS FUNCTION MUST BE DEFINED IN THE GLOBAL SCOPE for multiprocessing to work.
def run_strategy(strategy_class, option_data, kite_conn_for_strategy, selected_option_symbol=None, initial_capital=15000, is_paper_trading=True, underlying_data=None, instrument_name='NIFTY', actual_option_type='UNKNOWN'):
    """
    Runs a single Backtrader strategy.
    
    kite_conn_for_strategy: This is the KiteConnect object, which might be re-initialized in a child process.
                            It's None if is_paper_trading is True.
    """
    cerebro = bt.Cerebro()
    
    # Lot size is derived from instrument_name, it's consistent across options of the same underlying.
    # Fetch it using the kite object available in this context (either the re-initialized one or None).
    # If kite_conn_for_strategy is None (paper trading), use default 25.
    lot_size = get_lot_size(kite_conn_for_strategy, instrument_name) if kite_conn_for_strategy else 25 
    
    cerebro.addstrategy(strategy_class,
                        initial_capital=initial_capital, 
                        lot_size=lot_size,             
                        actual_option_type=actual_option_type) 
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Minutes, riskfreerate=0.06)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions') 

    cerebro.adddata(bt.feeds.PandasData(dataname=option_data, timeframe=bt.TimeFrame.Minutes, compression=5, tz=ist)) 
    if underlying_data is not None and not underlying_data.empty: 
        cerebro.adddata(bt.feeds.PandasData(dataname=underlying_data, timeframe=bt.TimeFrame.Minutes, compression=5, tz=ist)) 
    else:
        print(f"Warning: Underlying data is empty or not provided for {strategy_class.__name__}. AI/indicators might be less effective.", file=sys.stderr)

    cerebro.broker.setcash(initial_capital)
    
    cerebro.broker.setcommission(commission=20.0, mult=1.0, margin=None, commtype=bt.CommInfoBase.COMM_FIXED)

    print(f"\nStarting Portfolio Value for {strategy_class.__name__}: {initial_capital:.2f}")
    
    results = cerebro.run()
    strat = results[0] 

    final_value = strat.broker.getvalue()
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades_analyzer = strat.analyzers.trades.get_analysis()
    annual_return = strat.analyzers.annual_return.get_analysis() if hasattr(strat.analyzers, 'annual_return') else {}

    print(f"--- Results for {strategy_class.__name__} ---")
    print(f"Ending Portfolio Value: {final_value:.2f}")
    
    sharpe_ratio = sharpe.get('sharperatio', 'N/A')
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 'N/A'):.2f}%")
    
    total_trades = trades_analyzer.get('total', {}).get('total', 0)
    winning_trades = trades_analyzer.get('won', {}).get('total', 0)
    losing_trades = trades_analyzer.get('lost', {}).get('total', 0)
    
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Losing Trades: {losing_trades}")
    print(f"Win Rate: {winning_trades / total_trades * 100:.2f}%" if total_trades > 0 else "Win Rate: N/A")
    
    if total_trades > 0:
        avg_win = trades_analyzer.get('won', {}).get('pnl', {}).get('average', 0.0)
        avg_loss = trades_analyzer.get('lost', {}).get('pnl', {}).get('average', 0.0)
        print(f"Avg Win: {avg_win:.2f}")
        print(f"Avg Loss: {avg_loss:.2f}")
        
        total_pnl_won = trades_analyzer.get('won', {}).get('pnl', {}).get('total', 0.0)
        total_pnl_lost = trades_analyzer.get('lost', {}).get('pnl', {}).get('total', 0.0)
        
        if total_pnl_lost != 0:
            profit_factor = total_pnl_won / abs(total_pnl_lost)
        else:
            profit_factor = float('inf') if total_pnl_won > 0 else 0.0
        print(f"Profit Factor: {profit_factor:.2f}")
    else:
        print("No trade statistics available.")

    if not is_paper_trading: 
        if strat.position and strat.position.size != 0: 
            action = 'SELL' if strat.position.size > 0 else 'BUY' 
            print(f"Live trading is enabled and strategy ended with an open position of size {strat.position.size}.")
            if kite_conn_for_strategy:
                place_live_order(kite_conn_for_strategy, selected_option_symbol, abs(strat.position.size), action) 
            else:
                print("Cannot place live order: KiteConnect object not available in this context (likely multiprocessing without re-init).", file=sys.stderr)
        else:
            print("Live trading enabled but strategy ended with no open position to manage.")

    return (strategy_class.__name__, final_value, sharpe, drawdown, trades_analyzer, cerebro) 


# Wrapper function for multiprocessing pool.map
def run_strategy_wrapper(args):
    """Unpacks arguments and calls run_strategy. Designed for multiprocessing pool."""
    strategy_class, option_data, access_token_for_child, selected_option_symbol, initial_capital, is_paper_trading, underlying_data, instrument_name, actual_option_type = args
    
    child_kite = None
    if not is_paper_trading:
        try:
            child_kite = KiteConnect(api_key=API_KEY)
            child_kite.set_access_token(access_token_for_child)
        except Exception as e:
            print(f"Error re-initializing KiteConnect in child process (for {strategy_class.__name__}): {e}", file=sys.stderr)
            return None 

    return run_strategy(strategy_class, option_data, child_kite, selected_option_symbol, initial_capital, is_paper_trading, underlying_data, instrument_name, actual_option_type)

# Sequential wrapper for fallback if parallel execution fails
def run_strategy_wrapper_sequential(args):
    """Sequential wrapper for run_strategy - for fallback or debugging parallel."""
    strategy_class, option_data, access_token_for_child, selected_option_symbol, initial_capital, is_paper_trading, underlying_data, instrument_name, actual_option_type = args
    
    child_kite = None
    if not is_paper_trading:
        try:
            child_kite = KiteConnect(api_key=API_KEY)
            child_kite.set_access_token(access_token_for_child)
        except Exception as e:
            print(f"Error re-initializing KiteConnect in sequential run: {e}", file=sys.stderr)
            return None 

    return run_strategy(strategy_class, option_data, child_kite, selected_option_symbol, initial_capital, is_paper_trading, underlying_data, instrument_name, actual_option_type)


# --- Main Execution Block ---
def main_execution():
    global global_access_token 

    print("Attempting KiteConnect authentication...")
    kite, access_token = authenticate_kite(API_KEY, API_SECRET)
    global_access_token = access_token 
    print("KiteConnect authentication successful! ✅")

    INSTRUMENT_NAME = 'NIFTY' 
    INITIAL_CAPITAL = 15000.0
    IS_PAPER_TRADING = True 
    END_DATE = datetime.now(ist).replace(hour=15, minute=30, second=0, microsecond=0) 
    START_DATE = END_DATE - timedelta(days=20) 

    print(f"\n--- Running for {INSTRUMENT_NAME} from {START_DATE.strftime('%Y-%m-%d %H:%M')} to {END_DATE.strftime('%Y-%m-%d %H:%M')} ---")

    print(f"\nFetching option chain for {INSTRUMENT_NAME}...")
    option_chain_df, current_spot_price = get_option_chain(kite, instrument_name=INSTRUMENT_NAME)
    if option_chain_df.empty or current_spot_price == 0:
        print("No suitable options found or could not get spot price. Exiting. ❌", file=sys.stderr)
        sys.exit(1)
    
    simulated_strategy_signal = {'action': 'buy', 'direction': 'bullish'} 
    
    selected_option_symbol, actual_option_type = select_option(option_chain_df, simulated_strategy_signal, current_spot_price)
    
    if selected_option_symbol is None:
        print("Failed to select an option symbol. Exiting. ❌", file=sys.stderr)
        sys.exit(1)

    print(f"Selected option for trading: {selected_option_symbol} (Type: {actual_option_type})")

    print(f"\nFetching historical data for {selected_option_symbol}...")
    option_data, _ = fetch_historical_data(kite, selected_option_symbol, START_DATE, END_DATE, interval='5minute', exchange_segment='NFO')
    
    print(f"Fetching historical data for underlying {INSTRUMENT_NAME}...")
    underlying_tradingsymbol = 'NIFTY 50' if INSTRUMENT_NAME == 'NIFTY' else 'BANKNIFTY'
    underlying_data, _ = fetch_historical_data(kite, underlying_tradingsymbol, START_DATE, END_DATE, interval='5minute', exchange_segment='NSE') 

    if option_data.empty:
        print("No option historical data found. Exiting. ❌", file=sys.stderr)
        sys.exit(1)

    strategies_to_test = [
        CustomRSIMeanReversion,
        CustomBollingerBreakout,
        CustomSMACrossover,
        CustomMACDCrossover,
    ]

    strategy_args = []
    for strat_class in strategies_to_test:
        strategy_args.append((
            strat_class,
            option_data,
            global_access_token, 
            selected_option_symbol,
            INITIAL_CAPITAL,
            IS_PAPER_TRADING,
            underlying_data,
            INSTRUMENT_NAME,
            actual_option_type 
        ))

    print("\nStarting strategy backtests/live runs... (May take a while) ⏳")
    results_summary = []

    try:
        with Pool(processes=len(strategies_to_test)) as pool:
            all_results = pool.map(run_strategy_wrapper, strategy_args)
            
        for result in all_results:
            if result: 
                strategy_name, final_value, sharpe, drawdown, trades_analyzer, cerebro = result
                results_summary.append({
                    'Strategy': strategy_name,
                    'Final Value': final_value,
                    'Sharpe Ratio': sharpe.get('sharperatio', 'N/A'),
                    'Max Drawdown': drawdown.get('max', {}).get('drawdown', 'N/A'),
                    'Total Trades': trades_analyzer.get('total', {}).get('total', 0),
                    'Win Rate': (trades_analyzer.get('won', {}).get('total', 0) / trades_analyzer.get('total', {}).get('total', 1) * 100) if trades_analyzer.get('total', {}).get('total', 0) > 0 else 'N/A'
                })
    except Exception as e:
        print(f"Multiprocessing failed: {e}. Falling back to sequential execution. ⚠️", file=sys.stderr)
        for args in strategy_args:
            result = run_strategy_wrapper_sequential(args) 
            if result:
                strategy_name, final_value, sharpe, drawdown, trades_analyzer, cerebro = result
                results_summary.append({
                    'Strategy': strategy_name,
                    'Final Value': final_value,
                    'Sharpe Ratio': sharpe.get('sharperatio', 'N/A'),
                    'Max Drawdown': drawdown.get('max', {}).get('drawdown', 'N/A'),
                    'Total Trades': trades_analyzer.get('total', {}).get('total', 0),
                    'Win Rate': (trades_analyzer.get('won', {}).get('total', 0) / trades_analyzer.get('total', {}).get('total', 1) * 100) if trades_analyzer.get('total', {}).get('total', 0) > 0 else 'N/A'
                })

    print("\n--- Backtest Results Summary ---")
    for summary in results_summary:
        print(json.dumps(summary, indent=2))
        print("-" * 30)

    print("\nApplication finished. Enjoy your insights! ✨")

if __name__ == '__main__':
    main_execution()
