
import time
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.stats as si
import streamlit as st
import yfinance as yf

# App setup
st.set_page_config(page_title="SPY Pro CSP & Portfolio Matrix", layout="wide", page_icon="📈")
st.title("📈 SPY Professional Cash-Secured Put Dashboard")
st.markdown("### Real-Time Chain Scanner, Sizing Optimization & Risk Telemetry")

# Sidebar Controls
st.sidebar.header("⚙️ Strategy Parameters")
dte_min, dte_max = st.sidebar.slider("Days to Expiration (DTE) Window", 1, 30, (3, 14))
otm_min, otm_max = st.sidebar.slider("Out-of-The-Money (OTM) Filter Target %", 0.0, 10.0, (2.5, 5.0), step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("💰 Account & Risk Parameters")
account_size = st.sidebar.number_input("Total Portfolio Capital ($)", min_value=5000.0, value=100000.0, step=5000.0)
sweep_rate = st.sidebar.slider("Broker Cash Sweep Interest Rate (%)", 0.0, 7.0, 4.5, step=0.1) / 100
kelly_fraction = st.sidebar.selectbox("Kelly Sizing Model Scale", options=[0.25, 0.50, 1.0], format_func=lambda x: f"Quarter-Kelly ({x}x)" if x == 0.25 else f"Half-Kelly ({x}x)" if x == 0.50 else f"Full-Kelly ({x}x)")
min_pop = st.sidebar.slider("Minimum Probability of Profit (PoP) Floor %", 50, 95, 80) / 100

# Live Data Telemetry
@st.cache_data(ttl=60)
def fetch_live_market_matrix():
    spy = yf.Ticker("SPY")
    vix = yf.Ticker("^VIX")
    try:
        spy_p = spy.fast_info["lastPrice"]
        vix_v = vix.fast_info["lastPrice"] / 100
    except Exception:
        spy_p = spy.history(period="1d")["Close"].iloc[-1]
        vix_v = vix.history(period="1d")["Close"].iloc[-1] / 100
    return spy, spy_p, vix_v

spy, spy_price, vix_val = fetch_live_market_matrix()

# Top KPI Badges
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("🎯 SPY Underlying Price", f"${spy_price:,.2f}")
kpi2.metric("📊 Live Market VIX Index", f"{vix_val*100:.2f}%")
kpi3.metric("🏦 Cash Sweep Base Yield", f"{sweep_rate*100:.1f}%")
kpi4.metric("🕒 Execution Time", datetime.now().strftime("%H:%M:%S"))

# Scanner Logic Pipeline
opportunities = []
today = datetime.now().date()

with st.spinner("Streaming active option matrix from clearinghouse..."):
    for expiry_str in spy.options:
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            dte = (expiry_date - today).days
            if not (dte_min <= dte <= dte_max): continue

            opt_chain = spy.option_chain(expiry_str)
            puts = opt_chain.puts

            for _, row in puts.iterrows():
                strike = row["strike"]
                pct_otm = ((spy_price - strike) / spy_price) * 100
                if not (otm_min <= pct_otm <= otm_max): continue

                bid, ask = row["bid"], row["ask"]
                premium = (bid + ask) / 2 if (bid > 0 and ask > 0) else row["lastPrice"]
                if premium <= 0: continue

                # Probability calculation via log-normal distribution
                t_fraction = dte / 365.0
                sigma_t = vix_val * np.sqrt(t_fraction)
                d2 = (np.log(spy_price / strike) + (-0.5 * (vix_val**2)) * t_fraction) / sigma_t
                pop = si.norm.cdf(d2)
                if pop < min_pop: continue

                # Position math
                collateral = strike * 100
                breakeven = strike - premium
                interest_earned = collateral * (sweep_rate / 365) * dte
                total_return = (premium * 100) + interest_earned
                ann_yield = (total_return / collateral) * (365 / dte)

                # Kelly Criterion sizing calculation
                b_ratio = (premium * 100) / (collateral - (premium * 100))
                full_kelly = pop - ((1.0 - pop) / b_ratio) if b_ratio > 0 else 0
                adjusted_kelly = max(0.0, full_kelly * kelly_fraction)
                suggested_contracts = int((account_size * adjusted_kelly) / collateral)

                opportunities.append({
                    "Expiry": expiry_str, "DTE": dte, "Strike": strike, "OTM %": round(pct_otm, 2),
                    "Premium": round(premium, 2), "Breakeven": round(breakeven, 2), "PoP %": round(pop * 100, 1),
                    "Total Profit/Cont.": round(total_return, 2), "Ann. Blended Yield %": round(ann_yield * 100, 2),
                    "Kelly Weight %": round(adjusted_kelly * 100, 1), "Recommended Contracts": max(0, suggested_contracts)
                })
        except Exception:
            continue

# Dynamic UI Output Generation
if opportunities:
    df = pd.DataFrame(opportunities).sort_values(by="Ann. Blended Yield %", ascending=False)
    best = df.iloc

    # Real-Time Risk Warning Banner
    st.markdown("---")
    distance_to_breakeven = ((spy_price - best["Breakeven"]) / spy_price) * 100
    if spy_price <= best["Breakeven"]:
        st.error(f"💥 **CRITICAL BREACH**: SPY (${spy_price:.2f}) dropped below your position breakeven (${best['Breakeven']:.2f})!")
    elif distance_to_breakeven <= 1.0:
        st.warning(f"⚠️ **HIGH RISK**: SPY (${spy_price:.2f}) is within {distance_to_breakeven:.2f}% of your optimal breakeven (${best['Breakeven']:.2f})!")
    else:
        st.success(f"✅ **Margin Secure**: SPY is **{distance_to_breakeven:.2f}%** above your optimal breakeven line.")

    # Display Top Pick and Payoff Profile
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"### 👑 Optimal Selection: {best['Expiry']} | ${best['Strike']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("🎲 Win Prob (PoP)", f"{best['PoP %']}%")
        m2.metric("📈 Max Ann. Yield", f"{best['Ann. Blended Yield %']}%")
        m3.metric("🎯 Sizing Suggestion", f"{best['Recommended Contracts']} Lots")
        st.write(pd.DataFrame([best]).T.rename(columns={0: "Parameter Values"}))

    with col_right:
        st.markdown("### 📊 Position Payoff Profile")
        strikes_arr = np.linspace(best["Strike"] * 0.90, spy_price * 1.05, 100)
        payoff_dollar = [(best["Total Profit/Cont."] - (max(0.0, best["Strike"] - s) * 100)) * max(1, best['Recommended Contracts']) for s in strikes_arr]
        st.line_chart(pd.DataFrame({"Simulated SPY Expiry Settlement": strikes_arr, "Net Profit/Loss ($)": payoff_dollar}).set_index("Simulated SPY Expiry Settlement"))

    # Display full scan results grid
    st.markdown("---")
    st.markdown("### 🔎 Screened Options Chain Matrix")
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("⚠️ No active options matched your settings right now. Try widening the sliders on the left menu panel.")
