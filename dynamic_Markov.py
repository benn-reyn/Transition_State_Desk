import datetime
from dynamic_Markov import RegimeAnalyzer, CrossAssetAnalytics
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.stattools import coint

st.set_page_config(page_title="Regime Analysis", layout="wide")

@st.cache_resource(ttl=3600, show_spinner=False)
def execute_cached_estimation(ticker: str, start: str, end: str, k_regimes: int) -> RegimeAnalyzer:
    analyzer = RegimeAnalyzer(ticker=ticker, start=start, end=end, k_regimes=k_regimes)
    analyzer.fit_model()
    return analyzer

st.title("Regime Analysis")
st.markdown("---")

st.sidebar.markdown("### Navigation")
workspace = st.sidebar.selectbox("View", options=["1. Allocation", "2. Diagnostics"])
st.sidebar.markdown("### Inputs")

selected_tickers = st.sidebar.multiselect(
    "Assets",
    options=["BTC-USD", "GLD", "SLV", "SPY", "QQQ", "^TNX", "^VIX"],
    default=["BTC-USD", "GLD", "SPY"]
)

k_selection = st.sidebar.slider("Regimes (K)", min_value=2, max_value=3, value=2)
start_date = st.sidebar.date_input("Start Date", datetime.date(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2026, 6, 14))

fitted_analyzers = {}

if len(selected_tickers) < 1:
    st.info("Select assets to begin.")
else:
    for ticker in selected_tickers:
        try:
            analyzer_obj = execute_cached_estimation(
                ticker=ticker, start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"), k_regimes=k_selection
            )
            fitted_analyzers[ticker] = analyzer_obj
            if analyzer_obj.ms_garch_cascade:
                st.sidebar.info(f"{ticker}: Zero-Mean GARCH(1,1) active.")
        except Exception as e:
            st.sidebar.warning(f"Error loading {ticker}: {str(e)}")

pipeline_valid = len(fitted_analyzers) > 0

# ---------------------------------------------------------
# VIEWPORT 1: ALLOCATION
# ---------------------------------------------------------
if workspace == "1. Allocation" and pipeline_valid:
    st.subheader("Forward Allocation")
    
    forward_rows = []
    for ticker, analyzer in fitted_analyzers.items():
        forecast = analyzer.get_blended_forecast()
        if not forecast:
            continue
        
        curr_risk = float(analyzer.res.smoothed_marginal_probabilities[analyzer.k_regimes - 1].iloc[-1])
        f_p1 = float(forecast["forward_probabilities"][-1])

        forward_rows.append({
            "Asset": ticker,
            "Model": "MS + GARCH(1,1)" if analyzer.ms_garch_cascade else "Markov",
            "Current Vol Prob": curr_risk,
            "Next Day Vol Prob": f_p1,
            "Blended Daily Vol": float(forecast['blended_vol_daily']),
            "Blended Annual Vol": float(forecast['blended_vol_annualized'])
        })

    forward_df = pd.DataFrame(forward_rows)
    display_df = forward_df.copy()
    display_df["Current Vol Prob"] = display_df["Current Vol Prob"].map(lambda x: f"{x:.4f}")
    display_df["Next Day Vol Prob"] = display_df["Next Day Vol Prob"].map(lambda x: f"{x:.4f}")
    display_df["Blended Daily Vol"] = display_df["Blended Daily Vol"].map(lambda x: f"{x:.5f}")
    display_df["Blended Annual Vol"] = display_df["Blended Annual Vol"].map(lambda x: f"{x:.2f}%")

    col_table, col_exp = st.columns([5, 3])
    with col_table:
        st.dataframe(display_df.sort_values(by="Next Day Vol Prob", ascending=False), width="stretch", hide_index=True)
    with col_exp:
        st.info("Out-of-sample filtered probabilities are used for residual testing. Assets exhibiting autocorrelation trigger a Zero-Mean GARCH(1,1) overlay.")

    st.markdown("---")
    st.markdown("##### Impulse Response")
    irf_cols = st.columns(len(fitted_analyzers))
    for idx, (ticker, analyzer) in enumerate(fitted_analyzers.items()):
        with irf_cols[idx]:
            st.plotly_chart(analyzer.generate_contagion_plot(), use_container_width=True)

    st.markdown("---")
    st.markdown("##### Efficient Frontier")
    if len(fitted_analyzers) >= 2:
        col_ef_chart, col_ef_metrics = st.columns([5, 3])
        active_keys = list(fitted_analyzers.keys())
        with col_ef_chart:
            try:
                returns_data = pd.DataFrame({t: fitted_analyzers[t].data for t in active_keys}).dropna()
                
                dynamic_span = CrossAssetAnalytics.get_regime_ewma_span(fitted_analyzers)
                regime_covariance_matrix = returns_data.ewm(span=dynamic_span).cov().iloc[-len(active_keys):].values
                
                asset_means = np.array([fitted_analyzers[t].get_blended_forecast()["blended_mean_daily"] for t in active_keys])
                asset_vols = np.array([fitted_analyzers[t].get_blended_forecast()["blended_vol_daily"] for t in active_keys])
                
                num_simulations = 1500
                num_assets = len(active_keys)
                sim_vols = np.zeros(num_simulations)
                sim_returns = np.zeros(num_simulations)
                sim_sharpe = np.zeros(num_simulations)
                sim_weights = np.zeros((num_simulations, num_assets))
                
                for i in range(num_simulations):
                    weights = np.random.random(num_assets)
                    weights /= np.sum(weights)
                    sim_weights[i, :] = weights
                    
                    p_ret = np.sum(weights * asset_means) * 252
                    p_vol = np.sqrt(weights.T @ regime_covariance_matrix @ weights) * np.sqrt(252)
                    sim_returns[i] = p_ret * 100
                    sim_vols[i] = p_vol * 100
                    
                    sim_sharpe[i] = p_ret / p_vol if p_vol > 0 else 0
                
                ef_fig = go.Figure()
                ef_fig.add_trace(go.Scatter(
                    x=sim_vols, y=sim_returns, mode='markers',
                    marker=dict(size=5, color=sim_sharpe, colorscale='Blues', showscale=True, colorbar=dict(title='Sharpe')),
                    name='Simulated Portfolios'
                ))
                
                for ticker in active_keys:
                    f_cast = fitted_analyzers[ticker].get_blended_forecast()
                    ef_fig.add_trace(go.Scatter(
                        x=[f_cast["blended_vol_annualized"]], y=[f_cast["blended_mean_daily"] * 252 * 100],
                        mode='markers+text', text=[ticker], textposition="top center",
                        marker=dict(size=10, color='#DC2626', symbol='diamond'), name=ticker
                    ))
                    
                ef_fig.update_layout(
                    template="plotly_white", height=380,
                    xaxis_title="Annualized Volatility (%)", yaxis_title="Annualized Expected Return (%)",
                    margin=dict(l=40, r=20, t=20, b=40), showlegend=False,
                    font=dict(family="system-ui, -apple-system, sans-serif")
                )
                st.plotly_chart(ef_fig, use_container_width=True)
            except Exception as ef_error:
                st.caption(f"Error calculating efficient frontier: {str(ef_error)}")
        with col_ef_metrics:
            st.markdown("**Target Weights**")
            try:
                st.caption(f"Covariance Matrix: EWMA (Span: {dynamic_span} days).")
                best_sharpe_idx = np.argmax(sim_sharpe)
                min_vol_idx = np.argmin(sim_vols)
                
                msr_allocations = sim_weights[best_sharpe_idx]
                mvp_allocations = sim_weights[min_vol_idx]
                
                allocation_records = []
                for idx, t in enumerate(active_keys):
                    allocation_records.append({
                        "Asset": t,
                        "Max Sharpe": f"{msr_allocations[idx] * 100:.1f}%",
                        "Min Variance": f"{mvp_allocations[idx] * 100:.1f}%"
                    })
                st.dataframe(pd.DataFrame(allocation_records), width="stretch", hide_index=True)
            except Exception:
                st.caption("Allocation metrics unavailable.")

    st.markdown("---")
    st.markdown("##### Cross-Asset Correlation")
    col_sync, col_policy = st.columns([4, 4])
    
    with col_sync:
        sync_matrix = CrossAssetAnalytics.compute_regime_sync(fitted_analyzers)
        if not sync_matrix.empty:
            st.dataframe(sync_matrix.style.background_gradient(cmap="Blues", axis=None), width="stretch")
        else:
            st.caption("Not enough data to calculate correlation.")

    with col_policy:
        st.markdown("**Status**")
        high_correlation = False
        if not sync_matrix.empty and sync_matrix.shape[0] > 1:
            upper_vals = sync_matrix.values[np.triu_indices_from(sync_matrix.values, k=1)]
            if np.any(upper_vals > 0.70):
                high_correlation = True

        if high_correlation:
            st.warning("High correlation in tail risk states detected. Scale down sizing.")
        else:
            st.success("Assets show independent variance tracking.")
            
        mean_forward_risk = np.mean([x["Next Day Vol Prob"] for x in forward_rows]) if forward_rows else 0.0
        st.metric("Average Forward Volatility Prob", f"{mean_forward_risk:.4f}")

    st.markdown("---")
    st.markdown("##### Pairs Trading")
    if len(fitted_analyzers) >= 2:
        col_coint_grid, col_coint_advice = st.columns([5, 3])
        with col_coint_grid:
            coint_records = []
            active_keys = list(fitted_analyzers.keys())
            for i, tA in enumerate(active_keys):
                for j, tB in enumerate(active_keys):
                    if i < j:
                        try:
                            series_A = np.log(fitted_analyzers[tA].raw_prices)
                            series_B = np.log(fitted_analyzers[tB].raw_prices)
                            combined = pd.concat([series_A, series_B], axis=1).dropna()
                            
                            _, p_val, _ = coint(combined.iloc[:, 0], combined.iloc[:, 1])
                            
                            f_castA = fitted_analyzers[tA].get_blended_forecast()
                            f_castB = fitted_analyzers[tB].get_blended_forecast()
                            
                            p1_A = float(f_castA["forward_probabilities"][-1])
                            p1_B = float(f_castB["forward_probabilities"][-1])
                            
                            coint_records.append({
                                "Pair": f"{tA} / {tB}",
                                "Models": f"{'MS+GARCH' if fitted_analyzers[tA].ms_garch_cascade else 'MS'} / {'MS+GARCH' if fitted_analyzers[tB].ms_garch_cascade else 'MS'}",
                                "Coint P-Value": f"{p_val:.4f}",
                                "Status": "Stationary" if p_val < 0.05 else "Non-Stationary",
                                "Max Risk Prob": max(p1_A, p1_B)
                            })
                        except Exception:
                            pass
            coint_df = pd.DataFrame(coint_records)
            if not coint_df.empty:
                st.dataframe(coint_df, width="stretch", hide_index=True)
            else:
                st.caption("No integrated vectors available.")

        with col_coint_advice:
            if not coint_df.empty:
                valid_pairs = coint_df[coint_df["Status"] == "Stationary"]
                if valid_pairs.empty:
                    st.info("No cointegrated pairs found.")
                else:
                    highest_pair_risk = valid_pairs["Max Risk Prob"].max()
                    if highest_pair_risk >= 0.65:
                        st.warning("Pair is cointegrated, but forward risk is too high to trade.")
                    else:
                        st.success("Pair is cointegrated and forward risk is within bounds.")
    else:
        st.caption("Select 2 or more assets to calculate pairs.")

# ---------------------------------------------------------
# VIEWPORT 2: DIAGNOSTICS
# ---------------------------------------------------------
elif workspace == "2. Diagnostics" and pipeline_valid:
    st.subheader("Model Diagnostics")
    
    for ticker, analyzer in fitted_analyzers.items():
        with st.expander(f"Model Profile: {ticker}", expanded=True):
            tab_hist, tab_diag = st.tabs(["Historical Regimes", "Residual Diagnostics"])
            
            with tab_hist:
                col_chart, col_metrics = st.columns([5, 3])
                with col_chart:
                    st.plotly_chart(analyzer.generate_regime_plot(), use_container_width=True)
                with col_metrics:
                    st.markdown("##### Regime Parameters")
                    st.dataframe(analyzer.get_regime_statistics(), width="stretch", hide_index=True)

            with tab_diag:
                st.markdown("##### Standardized Residuals")
                st.caption("White-noise tests run using out-of-sample filtered probabilities.")
                diag_plots = analyzer.generate_individual_diagnostic_plots()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.plotly_chart(diag_plots[0], use_container_width=True)
                with col2:
                    st.plotly_chart(diag_plots[1], use_container_width=True)
                with col3:
                    st.plotly_chart(diag_plots[2], use_container_width=True)
