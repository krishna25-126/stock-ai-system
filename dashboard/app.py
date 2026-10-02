import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="AI Stock Prediction & Portfolio Optimizer", layout="wide")

UNIVERSE = ['AAPL', 'MSFT', 'JPM', 'XOM', 'JNJ', 'PG', 'NVDA', 'KO', 'CAT', 'HD']


@st.cache_data
def load_data():
    panel = pd.read_parquet('data/processed/features.parquet')
    recs = pd.read_csv('reports/recommendations.csv', index_col=0)
    weights = pd.read_csv('reports/optimal_weights.csv', index_col=0).iloc[:, 0]
    ml_board = pd.read_csv('reports/ml_leaderboard.csv')
    dl_board = pd.read_csv('reports/dl_leaderboard.csv')
    try:
        backtest = pd.read_csv('reports/backtest_curves.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        backtest = None
    return panel, recs, weights, ml_board, dl_board, backtest


panel, recs, weights, ml_board, dl_board, backtest = load_data()

st.title("AI Stock Prediction & Portfolio Optimization")
st.warning("Educational research project - not financial advice.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Price & Forecast", "Model Comparison", "Portfolio Allocation", "Recommendations", "Backtest"]
)

with tab1:
    ticker = st.selectbox("Select a stock", UNIVERSE)
    prices = panel[(ticker, 'Adj Close')].dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values, mode='lines', name=f'{ticker} Adj Close'))
    fig.update_layout(title=f"{ticker} Price History", xaxis_title="Date", yaxis_title="Price ($)", height=450)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${prices.iloc[-1]:.2f}")
    col2.metric("21-day Volatility", f"{panel[(ticker, 'vol21')].iloc[-1]:.2%}")
    col3.metric("RSI (14)", f"{panel[(ticker, 'rsi14')].iloc[-1]:.1f}")

with tab2:
    st.subheader("Classical ML Models")
    st.dataframe(ml_board, use_container_width=True)

    st.subheader("Deep Learning Models")
    st.dataframe(dl_board, use_container_width=True)

    st.caption("Note: directional accuracy near the base rate indicates limited predictive skill - see report for full discussion.")

with tab3:
    st.subheader("Optimized Portfolio Weights (Max Sharpe)")
    w_df = weights[weights > 0.001].sort_values(ascending=False)

    fig2 = go.Figure(data=[go.Pie(labels=w_df.index, values=w_df.values, hole=0.4)])
    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(w_df.apply(lambda x: f"{x:.1%}").rename("Weight"), use_container_width=True)

with tab4:
    st.subheader("Buy / Hold / Sell Signals")
    st.dataframe(recs, use_container_width=True)
    st.caption("Recommendations are relative rankings within this universe on a given day, not absolute buy/sell signals.")

with tab5:
    st.subheader("Out-of-Sample Backtest: Optimized vs. Equal-Weight vs. Benchmark")
    if backtest is not None:
        fig3 = go.Figure()
        for col in backtest.columns:
            fig3.add_trace(go.Scatter(x=backtest.index, y=backtest[col], mode='lines', name=col))
        fig3.update_layout(title="Portfolio Growth ($1 invested)", height=450, yaxis_title="Growth multiple")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Run src/backtest.py to generate backtest results.")