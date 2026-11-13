import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor

# --- SECTION 1: PAGE SETUP & SIDEBAR CONTROLS ---
st.set_page_config(page_title="AI Market Analytics & Signals", layout="wide")
st.title("🤖 AI Market Analytics & Automated Trading Signals")

st.sidebar.header("Trading Controls")
symbol = st.sidebar.selectbox(
    "Select NSE Stock",
    ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "TATAMOTORS.NS"]
)
period = st.sidebar.selectbox("Historical Window", ["6mo", "1y", "2y", "5y"], index=1)
forecast_days = st.sidebar.slider("Days to Predict Ahead", min_value=1, max_value=10, value=5)

if st.sidebar.button("Refresh Live Data"):
    st.rerun()

# --- SECTION 2: DATA FETCHING ---
@st.cache_data(ttl=60)
def load_data(ticker, time_period):
    df = yf.download(ticker, period=time_period, interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

data = load_data(symbol, period)

if data.empty or len(data) < 40:
    st.error("Not enough market data fetched. Try selecting a longer Historical Window (e.g., 1y).")
    st.stop()

# --- SECTION 3: TECHNICAL INDICATORS ---
# 1. Moving Averages & Bollinger Bands
data["SMA_10"] = data["Close"].rolling(window=10).mean()
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["Std_20"] = data["Close"].rolling(window=20).std()
data["Upper_Band"] = data["SMA_20"] + (data["Std_20"] * 2)
data["Lower_Band"] = data["SMA_20"] - (data["Std_20"] * 2)

# 2. RSI (14-day)
delta = data["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

# 3. MACD (12, 26, 9)
ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = ema_12 - ema_26
data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

# --- SECTION 4: AUTOMATED BUY / SELL SIGNAL LOGIC ---
# Buy Signal: MACD crosses above Signal Line AND RSI is not overbought (< 68)
macd_bull_cross = (data["MACD"] > data["MACD_Signal"]) & (data["MACD"].shift(1) <= data["MACD_Signal"].shift(1))
buy_condition = macd_bull_cross & (data["RSI"] < 68)

# Sell Signal: MACD crosses below Signal Line OR RSI crosses down from overbought (> 72)
macd_bear_cross = (data["MACD"] < data["MACD_Signal"]) & (data["MACD"].shift(1) >= data["MACD_Signal"].shift(1))
rsi_overbought_exit = (data["RSI"] < 72) & (data["RSI"].shift(1) >= 72)
sell_condition = macd_bear_cross | rsi_overbought_exit

buy_signals = data[buy_condition]
sell_signals = data[sell_condition]

# --- SECTION 5: GRADIENT BOOSTING AI PREDICTION ---
# Build lag & technical features so the AI learns price patterns, not just dates
df_ml = data.copy()
df_ml["Return_1d"] = df_ml["Close"].pct_change()
df_ml["Dist_SMA10"] = (df_ml["Close"] - df_ml["SMA_10"]) / df_ml["SMA_10"]
df_ml["Dist_SMA20"] = (df_ml["Close"] - df_ml["SMA_20"]) / df_ml["SMA_20"]
df_ml["Target_Next_Close"] = df_ml["Close"].shift(-1)

features = ["Close", "SMA_10", "SMA_20", "RSI", "MACD", "MACD_Hist", "Return_1d", "Dist_SMA10", "Dist_SMA20"]
train_df = df_ml.dropna().copy()

model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
model.fit(train_df[features], train_df["Target_Next_Close"])

# Step-by-step recursive forecast for future days
predicted_prices = []
sim_df = data.copy()

for _ in range(forecast_days):
    # Recompute indicators on simulated series
    s10 = sim_df["Close"].rolling(10).mean().iloc[-1]
    s20 = sim_df["Close"].rolling(20).mean().iloc[-1]
    d = sim_df["Close"].diff()
    g = d.where(d > 0, 0).rolling(14).mean().iloc[-1]
    l = (-d.where(d < 0, 0)).rolling(14).mean().iloc[-1]
    rsi_val = 100 - (100 / (1 + (g / l if l != 0 else 1)))
    e12 = sim_df["Close"].ewm(span=12, adjust=False).mean().iloc[-1]
    e26 = sim_df["Close"].ewm(span=26, adjust=False).mean().iloc[-1]
    macd_val = e12 - e26
    macd_sig = sim_df["Close"].ewm(span=12, adjust=False).mean().sub(
        sim_df["Close"].ewm(span=26, adjust=False).mean()
    ).ewm(span=9, adjust=False).mean().iloc[-1]
    macd_h = macd_val - macd_sig
    ret_1d = sim_df["Close"].pct_change().iloc[-1]
    c_last = sim_df["Close"].iloc[-1]

    row_feat = pd.DataFrame([[
        c_last, s10, s20, rsi_val, macd_val, macd_h, ret_1d,
        (c_last - s10) / s10, (c_last - s20) / s20
    ]], columns=features)

    next_price = float(model.predict(row_feat)[0])
    predicted_prices.append(next_price)

    # Append predicted close to simulate the next day
    new_row = pd.DataFrame({"Close": [next_price]})
    sim_df = pd.concat([sim_df, new_row], ignore_index=True)

last_date = pd.to_datetime(data["Date"].iloc[-1])
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

# --- SECTION 6: LIVE KPI METRICS ---
current_price = float(data["Close"].iloc[-1])
prev_price = float(data["Close"].iloc[-2])
current_rsi = float(data["RSI"].iloc[-1])
predicted_end = float(predicted_prices[-1])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Last Close (₹)", f"₹{current_price:,.2f}", f"{current_price - prev_price:+.2f}")
col2.metric(f"AI {forecast_days}-Day Target (₹)", f"₹{predicted_end:,.2f}", f"{predicted_end - current_price:+.2f}")

if current_rsi > 70:
    rsi_status = "Overbought (Caution)"
elif current_rsi < 30:
    rsi_status = "Oversold (Bounce Zone)"
else:
    rsi_status = "Healthy Momentum"
col3.metric("14-Day RSI", f"{current_rsi:.1f}", rsi_status)

# Check latest active signal
if buy_condition.iloc[-1]:
    latest_action = "🟢 NEW BUY SIGNAL"
elif sell_condition.iloc[-1]:
    latest_action = "🔴 NEW SELL SIGNAL"
else:
    latest_action = "⚪ HOLD / WAIT"
col4.metric("Today's Signal", latest_action)

# --- SECTION 7: MULTI-PANEL CHART WITH BUY/SELL MARKERS ---
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    vertical_spacing=0.05, row_heights=[0.58, 0.2, 0.22],
    subplot_titles=(f"{symbol} Price, AI Forecast & Automated Buy/Sell Signals", "RSI (14)", "MACD (12, 26, 9)")
)

# Candlesticks
fig.add_trace(go.Candlestick(
    x=data["Date"], open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
    name="Price"
), row=1, col=1)

# Bollinger Bands
fig.add_trace(go.Scatter(x=data["Date"], y=data["Upper_Band"], name="Upper BB", line=dict(color="rgba(173,216,230,0.4)", width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=data["Date"], y=data["Lower_Band"], name="Lower BB", line=dict(color="rgba(173,216,230,0.4)", width=1), fill="tonexty", fillcolor="rgba(173,216,230,0.06)"), row=1, col=1)

# Green BUY Triangles (placed slightly below the candle Low)
fig.add_trace(go.Scatter(
    x=buy_signals["Date"], y=buy_signals["Low"] * 0.985,
    mode="markers+text", name="BUY Signal",
    marker=dict(symbol="triangle-up", size=14, color="#00FF7F", line=dict(color="black", width=1)),
    text=["BUY"] * len(buy_signals), textposition="bottom center",
    textfont=dict(color="#00FF7F", size=10)
), row=1, col=1)

# Red SELL Triangles (placed slightly above the candle High)
fig.add_trace(go.Scatter(
    x=sell_signals["Date"], y=sell_signals["High"] * 1.015,
    mode="markers+text", name="SELL Signal",
    marker=dict(symbol="triangle-down", size=14, color="#FF4B4B", line=dict(color="black", width=1)),
    text=["SELL"] * len(sell_signals), textposition="top center",
    textfont=dict(color="#FF4B4B", size=10)
), row=1, col=1)

# Gradient Boosting Forecast Line
fig.add_trace(go.Scatter(
    x=future_dates, y=predicted_prices,
    name="AI Forecast", mode="lines+markers",
    line=dict(color="#FFD700", width=3, dash="dash")
), row=1, col=1)

# RSI Panel
fig.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], name="RSI", line=dict(color="#AB63FA")), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

# MACD Panel
fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD"], name="MACD", line=dict(color="#00CC96")), row=3, col=1)
fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD_Signal"], name="Signal", line=dict(color="#FFA15A")), row=3, col=1)
fig.add_trace(go.Bar(x=data["Date"], y=data["MACD_Hist"], name="Histogram", marker_color="gray"), row=3, col=1)

fig.update_layout(height=780, template="plotly_dark", xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
