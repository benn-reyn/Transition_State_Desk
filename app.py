import datetime
from dynamic_Markov import RegimeAnalyzer, CrossAssetAnalytics
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.stattools import coint

st.set_page_config(page_title="Regime Analysis Terminal", layout="wide")

# Added a 1-hour expiration limit so we regularly catch data updates when running live
@st.cache_resource(ttl=3600, show_spinner=False)
def execute_cached_estimation(ticker: str, start: str, end: str, k_regimes: int) -> RegimeAnalyzer:
    analyzer = RegimeAnalyzer(ticker=ticker, start=start, end=end, k_regimes=k_regimes)
    analyzer.fit_model()
    return analyzer

st.title("Regime Analytics Terminal")
st.caption("Regime Analytics terminal guiding portfolio allocation")
st.markdown("---")

st.sidebar.markdown("### Navigation")
workspace = st.sidebar.selectbox("Interface Viewport", options=["1. Forecasting Desk", "2. Econometric Desk"])
st.sidebar.markdown("### Model Controls")

selected_tickers = st.sidebar.multiselect(
    "Asset Selection",
    options=["BTC-USD", "GLD", "SLV", "SPY", "QQQ", "^TNX", "^VIX"],
    default=["BTC-USD", "GLD", "SPY"]
)

k_selection = st.sidebar.slider("Regimes Count (K)", min_value=2, max_value=3, value=2)
start_date = st.sidebar.date_input("Start Date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2025, 1, 1))

fitted_analyzers = {}

if len(selected_tickers) < 1:
    st.info("Select assets in the sidebar menu.")
else:
    for ticker in selected_tickers:
        try:
            analyzer_obj = execute_cached_estimation(
                ticker=ticker, start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"), k_regimes=k_selection
            )
            fitted_analyzers[ticker] = analyzer_obj
            if analyzer_obj.use_garch_fallback:
                st.sidebar.info(f"{ticker}: Residuals functionally non-i.i.d. GARCH best specified.")
        except Exception as e:
            # Let the dashboard continue loading surviving parameters if one asset data pull breaks
            st.sidebar.warning(f"Skipping {ticker} due to loading error: {str(e)}")

# Ensure we have at least something calibrated to render tabs
pipeline_valid = len(fitted_analyzers) > 0

# ---------------------------------------------------------
# VIEWPORT 1: TACTICAL FORWARD ALLOCATION DESK
# ---------------------------------------------------------
if workspace == "1. Forecasting Desk" and pipeline_valid:
    st.subheader("Forecasting Analytics")
    
    forward_rows = []
    for ticker, analyzer in fitted_analyzers.items():
        forecast = analyzer.get_blended_forecast()
        if not forecast:
            continue
        
        if analyzer.use_garch_fallback:
            # Map current volatility path placement into an absolute empirical distribution score
            hist_vols = analyzer.garch_res.conditional_volatility / 100
            curr_vol = hist_vols.iloc[-1] if len(hist_vols) > 0 else 0.0
            
            curr_risk = float(np.sum(hist_vols <= curr_vol) / len(hist_vols)) if len(hist_vols) > 0 else 0.5
            f_p1 = float(forecast["forward_probabilities"][-1])
        else:
            curr_risk = float(analyzer.res.smoothed_marginal_probabilities[analyzer.k_regimes - 1].iloc[-1])
            f_p1 = float(forecast["forward_probabilities"][-1])

        forward_rows.append({
            "Asset": ticker,
            "Engine": "GARCH Fallback" if analyzer.use_garch_fallback else "Markov Switching",
            "Current Risk Prob": curr_risk,
            "Tomorrow Risk Prob": f_p1,
            "Blended Daily Vol": float(forecast['blended_vol_daily']),
            "Blended Annual Vol": float(forecast['blended_vol_annualized'])
        })

    forward_df = pd.DataFrame(forward_rows)
    display_df = forward_df.copy()
    display_df["Current Risk Prob"] = display_df["Current Risk Prob"].map(lambda x: f"{x:.4f}")
    display_df["Tomorrow Risk Prob"] = display_df["Tomorrow Risk Prob"].map(lambda x: f"{x:.4f}")
    display_df["Blended Daily Vol"] = display_df["Blended Daily Vol"].map(lambda x: f"{x:.5f}")
    display_df["Blended Annual Vol"] = display_df["Blended Annual Vol"].map(lambda x: f"{x:.2f}%")

    col_table, col_exp = st.columns([5, 3])
    with col_table:
        st.markdown("##### Forward Risk Forecast Matrix")
        st.dataframe(display_df.sort_values(by="Tomorrow Risk Prob", ascending=False), use_container_width=True, hide_index=True)
    with col_exp:
        st.info("Assets exhibiting persistent residual autocorrelation in variance are automatically re-routed away from standard Markov chains into conditional heteroskedasticity equations.")

    st.markdown("---")
    st.markdown("##### KvD Impulse Response Engine")
    irf_cols = st.columns(len(fitted_analyzers))
    for idx, (ticker, analyzer) in enumerate(fitted_analyzers.items()):
        with irf_cols[idx]:
            st.plotly_chart(analyzer.generate_contagion_plot(), use_container_width=True)

    st.markdown("---")
    st.markdown("##### Frontier Mapping")
    if len(fitted_analyzers) >= 2:
        col_ef_chart, col_ef_metrics = st.columns([5, 3])
        active_keys = list(fitted_analyzers.keys())
        with col_ef_chart:
            try:
                returns_data = pd.DataFrame({t: fitted_analyzers[t].data for t in active_keys}).dropna()
                historical_corr = returns_data.corr()
                
                asset_means = np.array([fitted_analyzers[t].get_blended_forecast()["blended_mean_daily"] for t in active_keys])
                asset_vols = np.array([fitted_analyzers[t].get_blended_forecast()["blended_vol_daily"] for t in active_keys])
                
                covariance_matrix = np.diag(asset_vols) @ historical_corr.values @ np.diag(asset_vols)
                
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
                    p_vol = np.sqrt(weights.T @ covariance_matrix @ weights) * np.sqrt(252)
                    sim_returns[i] = p_ret * 100
                    sim_vols[i] = p_vol * 100
                    
                    # Sharpe ratio uses simple annual return divided by annual risk
                    sim_sharpe[i] = p_ret / p_vol if p_vol > 0 else 0
                
                ef_fig = go.Figure()
                ef_fig.add_trace(go.Scatter(
                    x=sim_vols, y=sim_returns, mode='markers',
                    marker=dict(size=4, color=sim_sharpe, colorscale='Viridis', showscale=True, colorbar=dict(title='Sharpe Ratio')),
                    name='Simulated Portfolios'
                ))
                
                for ticker in active_keys:
                    f_cast = fitted_analyzers[ticker].get_blended_forecast()
                    ef_fig.add_trace(go.Scatter(
                        x=[f_cast["blended_vol_annualized"]], y=[f_cast["blended_mean_daily"] * 252 * 100],
                        mode='markers+text', text=[ticker], textposition="top center",
                        marker=dict(size=10, color='red', symbol='diamond'), name=ticker
                    ))
                    
                ef_fig.update_layout(
                    template="plotly_white", height=380,
                    xaxis_title="Annualized Blended Volatility (%)", yaxis_title="Annualized Expected Return (%)",
                    margin=dict(l=40, r=20, t=20, b=40), showlegend=False
                )
                st.plotly_chart(ef_fig, use_container_width=True)
            except Exception as ef_error:
                st.caption(f"Could not generate efficient frontier calculation: {str(ef_error)}")
        with col_ef_metrics:
            st.markdown("**Deterministic Targets**")
            try:
                best_sharpe_idx = np.argmax(sim_sharpe)
                min_vol_idx = np.argmin(sim_vols)
                
                msr_allocations = sim_weights[best_sharpe_idx]
                mvp_allocations = sim_weights[min_vol_idx]
                
                allocation_records = []
                for idx, t in enumerate(active_keys):
                    allocation_records.append({
                        "Asset": t,
                        "Max Sharpe Alloc": f"{msr_allocations[idx] * 100:.1f}%",
                        "Min Variance Alloc": f"{mvp_allocations[idx] * 100:.1f}%"
                    })
                st.dataframe(pd.DataFrame(allocation_records), use_container_width=True, hide_index=True)
            except Exception:
                st.caption("Allocation vectors optimization metrics uncompiled.")

    st.markdown("---")
    st.markdown("##### Cross-Asset Risk Co-Movement & Allocation Panel")
    col_sync, col_policy = st.columns([4, 4])
    
    with col_sync:
        st.markdown("**Regime Co-Movement Metrics**")
        sync_matrix = CrossAssetAnalytics.compute_regime_sync(fitted_analyzers)
        if not sync_matrix.empty:
            st.dataframe(sync_matrix.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)
        else:
            st.caption("Insufficient historical data to map system synchronization.")

    with col_policy:
        st.markdown("**Sizing Advice**")
        excessive_co_movement = False
        if not sync_matrix.empty and sync_matrix.shape[0] > 1:
            upper_vals = sync_matrix.values[np.triu_indices_from(sync_matrix.values, k=1)]
            if np.any(upper_vals > 0.70):
                excessive_co_movement = True

        if excessive_co_movement:
            st.warning("High co-movement in tail risk states detected, may want to hedge/diversify")
        else:
            st.success("Variance tracks independently across assets, well diversified for current regime.")
            
        mean_forward_risk = np.mean([x["Tomorrow Risk Prob"] for x in forward_rows]) if forward_rows else 0.0
        st.metric("Aggregate Portfolio Risk State Probability", f"{mean_forward_risk:.4f}")
        
        if mean_forward_risk >= 0.60:
            st.error("High conditional volatility regimes dominant, may want to risk off.")
        elif 0.35 <= mean_forward_risk < 0.60:
            st.warning("Transitional shifts active, be mindful of macro-events if/when rebalancing")
        else:
            st.success("Systemic low-variance parameters dominant.")

    st.markdown("---")
    st.markdown("##### Pairs Trading Statistical Arbitrage Engine")
    if len(fitted_analyzers) >= 2:
        col_coint_grid, col_coint_advice = st.columns([5, 3])
        with col_coint_grid:
            coint_records = []
            active_keys = list(fitted_analyzers.keys())
            for i, tA in enumerate(active_keys):
                for j, tB in enumerate(active_keys):
                    if i < j:
                        try:
                            # Swap cumsum returns for raw log asset price vectors to isolate structural I(1) properties
                            series_A = np.log(fitted_analyzers[tA].raw_prices)
                            series_B = np.log(fitted_analyzers[tB].raw_prices)
                            combined = pd.concat([series_A, series_B], axis=1).dropna()
                            
                            _, p_val, _ = coint(combined.iloc[:, 0], combined.iloc[:, 1])
                            
                            f_castA = fitted_analyzers[tA].get_blended_forecast()
                            f_castB = fitted_analyzers[tB].get_blended_forecast()
                            
                            p1_A = float(f_castA["forward_probabilities"][-1])
                            p1_B = float(f_castB["forward_probabilities"][-1])
                            
                            coint_records.append({
                                "Asset Pair": f"{tA} vs {tB}",
                                "Engines": f"{'GARCH' if fitted_analyzers[tA].use_garch_fallback else 'Markov'}/{'GARCH' if fitted_analyzers[tB].use_garch_fallback else 'Markov'}",
                                "Cointegration P-Value": f"{p_val:.4f}",
                                "Status": "Stationary Spread" if p_val < 0.05 else "Non-Stationary",
                                "Max Bound Risk Parameter": max(p1_A, p1_B)
                            })
                        except Exception:
                            pass
            coint_df = pd.DataFrame(coint_records)
            if not coint_df.empty:
                st.dataframe(coint_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No integrated vectors available.")

        with col_coint_advice:
            st.markdown("**Note**")
            if not coint_df.empty:
                valid_pairs = coint_df[coint_df["Status"] == "Stationary Spread"]
                if valid_pairs.empty:
                    st.info("No statistically tied pairs available.")
                else:
                    highest_pair_risk = valid_pairs["Max Bound Risk Parameter"].max()
                    if highest_pair_risk >= 0.65:
                        st.warning("Cointegration validated, but conditional volatility thresholds are too high for variance convergence trades within relevant time frame.")
                    else:
                        st.success("Cointegrated spreads confirmed alongside optimal stationary variance regimes.")
    else:
        st.caption("Select 2 or more nodes to activate structural arbitrage scanning.")

# ---------------------------------------------------------
# VIEWPORT 2: ECONOMETRIC RESEARCH WORKSPACE
# ---------------------------------------------------------
elif workspace == "2. Econometric Desk" and pipeline_valid:
    st.subheader("Econometric Desk")
    
    for ticker, analyzer in fitted_analyzers.items():
        with st.expander(f"Historical Calibration Profile: {ticker}", expanded=True):
            tab_hist, tab_diag = st.tabs(["Historical Regimes & Baselines", "Residual Memory Checks"])
            
            with tab_hist:
                col_chart, col_metrics = st.columns([5, 3])
                with col_chart:
                    st.plotly_chart(analyzer.generate_regime_plot(), use_container_width=True)
                with col_metrics:
                    st.markdown("##### Calculated Regime Parameters")
                    st.dataframe(analyzer.get_regime_statistics(), use_container_width=True, hide_index=True)
                    st.markdown("##### State Transition Matrix")
                    st.dataframe(analyzer.get_transition_matrix(), use_container_width=True)
                    st.markdown("##### Framework Benchmark Comparison")
                    st.dataframe(analyzer.get_model_comparison(), use_container_width=True, hide_index=True)

            with tab_diag:
                st.markdown("##### Standardized Model Residual Analysis")
                diag_plots = analyzer.generate_individual_diagnostic_plots()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.plotly_chart(diag_plots[0], use_container_width=True)
                with col2:
                    st.plotly_chart(diag_plots[1], use_container_width=True)
                with col3:
                    st.plotly_chart(diag_plots[2], use_container_width=True)
                    
                stationarity_res = analyzer.is_stationary()
                st.info(
                    f"Active Diagnostics Pipeline: {'GARCH Fallback' if analyzer.use_garch_fallback else 'Markov switching System'}. "
                    f"ADF Score: {stationarity_res['statistic']:.4f} (p-value: {stationarity_res['p_value']:.5f})."
                )