import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor

# --- SECTION 1: PAGE SETUP & INTRADAY CONTROLS ---
# st.set_page_config(page_title="Intraday AI Trading Terminal", layout="wide")
# st.title("⚡ Intraday AI Terminal: Live Auto-Refresh & Risk Levels")

st.sidebar.header("Intraday Controls")
symbol = st.sidebar.selectbox(
    "Select NSE Stock / Index",
    ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "^NSEI", "^NSEBANK"]
)
interval = st.sidebar.selectbox("Candle Timeframe", ["5m", "15m", "30m", "1h"], index=1)
period_map = {"5m": "5d", "15m": "15d", "30m": "1mo", "1h": "1mo"}
forecast_candles = st.sidebar.slider("Candles to Predict Ahead", min_value=1, max_value=12, value=4)

st.sidebar.divider()
auto_refresh = st.sidebar.toggle("Auto-Refresh Every 30s", value=True)
atr_sl_mult = st.sidebar.slider("Stop-Loss ATR Multiplier", 1.0, 3.0, 1.5, 0.5)
atr_tp_mult = st.sidebar.slider("Take-Profit ATR Multiplier", 1.5, 5.0, 3.0, 0.5)

# --- SECTION 2: DATA FETCHING ---
@st.cache_data(ttl=25)
def load_intraday_data(ticker, time_interval):
    lookback = period_map[time_interval]
    df = yf.download(ticker, period=lookback, interval=time_interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={time_col: "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)
    return df

# --- SECTION 3: LIVE AUTO-REFRESHING DASHBOARD FRAGMENT ---
refresh_interval = 30 if auto_refresh else None

@st.fragment(run_every=refresh_interval)
def render_live_terminal():
    data = load_intraday_data(symbol, interval)

    if data.empty or len(data) < 40:
        st.error("Not enough intraday data available right now. Try switching the timeframe to 15m or 1h.")
        return

    # 1. Session VWAP
    data["Trade_Date"] = data["Timestamp"].dt.date
    data["Typical_Price"] = (data["High"] + data["Low"] + data["Close"]) / 3
    data["TP_Vol"] = data["Typical_Price"] * data["Volume"]
    data["VWAP"] = data.groupby("Trade_Date")["TP_Vol"].cumsum() / data.groupby("Trade_Date")["Volume"].cumsum()

    # 2. Fast Intraday EMAs (9 & 21)
    data["EMA_9"] = data["Close"].ewm(span=9, adjust=False).mean()
    data["EMA_21"] = data["Close"].ewm(span=21, adjust=False).mean()

    # 3. ATR (14-Period Volatility for Stop-Loss & Target)
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift(1)).abs()
    low_close = (data["Low"] - data["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR"] = true_range.rolling(window=14).mean()

    # 4. RSI (14-Period)
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # 5. MACD (12, 26, 9)
    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    # 6. Automated Intraday Buy / Sell Signals
    ema_bull_cross = (data["EMA_9"] > data["EMA_21"]) & (data["EMA_9"].shift(1) <= data["EMA_21"].shift(1))
    buy_condition = ema_bull_cross & (data["Close"] > data["VWAP"]) & (data["RSI"] < 68)

    ema_bear_cross = (data["EMA_9"] < data["EMA_21"]) & (data["EMA_9"].shift(1) >= data["EMA_21"].shift(1))
    sell_condition = (ema_bear_cross & (data["Close"] < data["VWAP"])) | ((data["RSI"] < 72) & (data["RSI"].shift(1) >= 72))

    data["Signal"] = np.where(buy_condition, "BUY", np.where(sell_condition, "SELL", None))
    buy_signals = data[data["Signal"] == "BUY"]
    sell_signals = data[data["Signal"] == "SELL"]

    # 7. Gradient Boosting AI Prediction
    df_ml = data.copy()
    df_ml["Return_1"] = df_ml["Close"].pct_change()
    df_ml["Dist_VWAP"] = (df_ml["Close"] - df_ml["VWAP"]) / df_ml["VWAP"]
    df_ml["Dist_EMA9"] = (df_ml["Close"] - df_ml["EMA_9"]) / df_ml["EMA_9"]
    df_ml["Target_Next_Close"] = df_ml["Close"].shift(-1)

    features = ["Close", "EMA_9", "EMA_21", "RSI", "MACD", "MACD_Hist", "Return_1", "Dist_VWAP", "Dist_EMA9"]
    train_df = df_ml.replace([np.inf, -np.inf], np.nan).dropna().copy()

    model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(train_df[features], train_df["Target_Next_Close"])

    predicted_prices = []
    sim_df = data.copy()

    for _ in range(forecast_candles):
        e9 = sim_df["Close"].ewm(span=9, adjust=False).mean().iloc[-1]
        e21 = sim_df["Close"].ewm(span=21, adjust=False).mean().iloc[-1]
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
        ret_1 = sim_df["Close"].pct_change().iloc[-1]
        c_last = sim_df["Close"].iloc[-1]
        vwap_last = sim_df["VWAP"].iloc[-1] if pd.notna(sim_df["VWAP"].iloc[-1]) else c_last

        row_feat = pd.DataFrame([[
            c_last, e9, e21, rsi_val, macd_val, macd_h, ret_1,
            (c_last - vwap_last) / vwap_last, (c_last - e9) / e9
        ]], columns=features)

        next_price = float(model.predict(row_feat).item())
        predicted_prices.append(next_price)

        new_row = pd.DataFrame({"Close": [next_price], "VWAP": [vwap_last]})
        sim_df = pd.concat([sim_df, new_row], ignore_index=True)

    step_minutes = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}[interval]
    last_ts = data["Timestamp"].iloc[-1]
    future_times = [last_ts + pd.Timedelta(minutes=step_minutes * (i + 1)) for i in range(forecast_candles)]

    # 8. Calculate Active Trade Risk-Reward Levels (Latest Signal)
    active_signals = data.dropna(subset=["Signal"])
    current_price = float(data["Close"].iloc[-1])
    prev_price = float(data["Close"].iloc[-2])
    predicted_end = float(predicted_prices[-1])
    current_atr = float(data["ATR"].iloc[-1])

    if not active_signals.empty:
        last_sig_row = active_signals.iloc[-1]
        sig_type = last_sig_row["Signal"]
        sig_entry = float(last_sig_row["Close"])
        sig_atr = float(last_sig_row["ATR"]) if pd.notna(last_sig_row["ATR"]) else current_atr
        sig_time = last_sig_row["Timestamp"].strftime("%d-%b %H:%M")

        if sig_type == "BUY":
            stop_loss = sig_entry - (sig_atr * atr_sl_mult)
            take_profit = sig_entry + (sig_atr * atr_tp_mult)
        else:
            stop_loss = sig_entry + (sig_atr * atr_sl_mult)
            take_profit = sig_entry - (sig_atr * atr_tp_mult)
    else:
        sig_type, sig_entry, stop_loss, take_profit, sig_time = "NONE", current_price, current_price, current_price, "N/A"

    # 9. Top KPI & Risk Management Cards
    st.caption(f"Last Candle Timestamp: **{last_ts.strftime('%d %b %Y, %H:%M')}** | Auto-Refresh: **{'ON (30s)' if auto_refresh else 'PAUSED'}**")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Last Close", f"₹{current_price:,.2f}", f"{current_price - prev_price:+.2f}")
    col2.metric(f"AI {forecast_candles}-Candle Target", f"₹{predicted_end:,.2f}", f"{predicted_end - current_price:+.2f}")

    if sig_type != "NONE":
        badge = "🟢 BUY" if sig_type == "BUY" else "🔴 SELL"
        col3.metric(f"Latest Signal ({sig_time})", f"{badge} @ ₹{sig_entry:,.2f}")
        col4.metric("Stop-Loss (ATR)", f"₹{stop_loss:,.2f}", f"{stop_loss - sig_entry:+.2f}", delta_color="inverse")
        col5.metric("Take-Profit (ATR)", f"₹{take_profit:,.2f}", f"{take_profit - sig_entry:+.2f}")
    else:
        col3.metric("Latest Signal", "⚪ NO SIGNAL")
        col4.metric("Stop-Loss (ATR)", "—")
        col5.metric("Take-Profit (ATR)", "—")

    # 10. Intraday Multi-Panel Chart with SL & TP Lines
    plot_df = data.tail(100)
    first_plot_ts = plot_df["Timestamp"].iloc[0]
    plot_buys = buy_signals[buy_signals["Timestamp"] >= first_plot_ts]
    plot_sells = sell_signals[sell_signals["Timestamp"] >= first_plot_ts]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.58, 0.2, 0.22],
        subplot_titles=(f"{symbol} ({interval}) — VWAP, Signals, Stop-Loss/Target & AI Forecast", "RSI (14)", "MACD")
    )

    x_labels = plot_df["Timestamp"].dt.strftime("%d-%b %H:%M")

    fig.add_trace(go.Candlestick(
        x=x_labels, open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"],
        name="Price"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=x_labels, y=plot_df["VWAP"], name="VWAP", line=dict(color="#00BFFF", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=plot_df["EMA_9"], name="EMA 9", line=dict(color="#00FF7F", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_labels, y=plot_df["EMA_21"], name="EMA 21", line=dict(color="#FF69B4", width=1, dash="dot")), row=1, col=1)

    # Buy & Sell Markers
    fig.add_trace(go.Scatter(
        x=plot_buys["Timestamp"].dt.strftime("%d-%b %H:%M"),
        y=plot_buys["Low"] * 0.998,
        mode="markers+text",
        name="BUY",
        marker=dict(symbol="triangle-up", size=13, color="#00FF7F"),
        text=["BUY"] * len(plot_buys),
        textposition="bottom center",
        textfont=dict(color="#00FF7F", size=10)
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plot_sells["Timestamp"].dt.strftime("%d-%b %H:%M"),
        y=plot_sells["High"] * 1.002,
        mode="markers+text",
        name="SELL",
        marker=dict(symbol="triangle-down", size=13, color="#FF4B4B"),
        text=["SELL"] * len(plot_sells),
        textposition="top center",
        textfont=dict(color="#FF4B4B", size=10)
        ), row=1, col=1)
    # Horizontal Stop-Loss & Take-Profit Lines for Latest Signal
    if sig_type != "NONE":
        fig.add_hline(y=take_profit, line_dash="dash", line_color="#00FF7F", annotation_text=f"Target: ₹{take_profit:,.2f}", row=1, col=1)
        fig.add_hline(y=stop_loss, line_dash="dash", line_color="#FF4B4B", annotation_text=f"Stop-Loss: ₹{stop_loss:,.2f}", row=1, col=1)
        # AI Forecast Line
        future_labels = [ts.strftime("%d-%b %H:%M") for ts in future_times]
        fig.add_trace(go.Scatter(
        x=future_labels,
        y=predicted_prices,
        name="AI Forecast",
        mode="lines+markers",
        line=dict(color="#FFD700", width=3, dash="dash")
        ), row=1, col=1)
        
    # RSI Panel
        fig.add_trace(go.Scatter(x=x_labels, y=plot_df["RSI"], name="RSI", line=dict(color="#AB63FA")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        # MACD Panel
        fig.add_trace(go.Scatter(x=x_labels, y=plot_df["MACD"], name="MACD", line=dict(color="#00CC96")), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_labels, y=plot_df["MACD_Signal"], name="Signal", line=dict(color="#FFA15A")), row=3, col=1)
        fig.add_trace(go.Bar(x=x_labels, y=plot_df["MACD_Hist"], name="Histogram", marker_color="gray"), row=3, col=1)
        fig.update_layout(height=780, template="plotly_dark", xaxis_rangeslider_visible=False)
        fig.update_xaxes(type="category", nticks=12)
        st.plotly_chart(fig, use_container_width=True)
        render_live_terminal()