import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from statsmodels.tsa.stattools import acf
from msvar_model import MSIAHVAR
from msgarch_model import WeightedGARCH

# ----------------------------------------------------------------------
# Page Configuration & Modern Slate UI Styling
# ----------------------------------------------------------------------
st.set_page_config(page_title="Regime Risk Terminal", layout="wide")

st.markdown("""
<style>
    /* Global Overrides */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0B0E14;
        color: #94A3B8;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Headers & Text */
    h1, h2, h3, h4, [data-testid="stMarkdownContainer"] h3 {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    
    /* Clean Modern Grid Card */
    .terminal-card {
        background: #111622;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .metric-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        line-height: 1;
        color: #F8FAFC;
    }
    
    .metric-status {
        font-size: 0.8rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Custom Navigation Tabs Adjustment */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111622;
        border: 1px solid #1E293B;
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        color: #94A3B8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border-bottom: 2px solid #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Core Data State Engine
# ----------------------------------------------------------------------
STATE_FILE = "model_state.pkl"

#@st.cache_resource(show_spinner=False)
def load_terminal_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            state = pickle.load(f)
        return state["msvar"], state["garch_models"], state["residuals"], state["data"]
    return None, None, None, None

msvar, garch_models, residuals, data = load_terminal_state()

if msvar is None:
    st.error("Terminal state payload missing. Please run 'update_models.py' to generate parameters.")
    st.stop()

# Cache state attributes locally
xi_now = msvar.filtered_probs[-1]
pred_tomorrow = msvar.P.T @ xi_now
prob_bear_tomorrow = pred_tomorrow[1]
prob_bear_now = xi_now[1]

# Dynamic UI Variant Parameters
if prob_bear_tomorrow < 0.35:
    regime_color, regime_bg, regime_label = "#10B981", "rgba(16,185,129,0.15)", "Expansionary / Calm"
elif prob_bear_tomorrow < 0.65:
    regime_color, regime_bg, regime_label = "#F59E0B", "rgba(245,158,11,0.15)", "Regime Transition / Indeterminate"
else:
    regime_color, regime_bg, regime_label = "#EF4444", "rgba(239,68,68,0.15)", "Contractionary / Bear Market"

# ----------------------------------------------------------------------
# Main Dashboard Interface Header
# ----------------------------------------------------------------------
col_logo, col_meta = st.columns([3, 1])
with col_logo:
    st.title("Regime & Volatility Terminal")
    st.caption("Markov-Switching VAR (MSIAH-VAR) & Joint Regime GARCH")

# Top Level KPI Row
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.markdown(f"""
    <div class="terminal-card" style="border-left: 4px solid {regime_color};">
        <div class="metric-title">Forecasted Bear Probability (Tomorrow)</div>
        <div class="metric-value" style="color: {regime_color};">{prob_bear_tomorrow*100:.1f}%</div>
        <div class="metric-status" style="color: {regime_color};">● Current Outlook: {regime_label}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    dur_bear = 1 / (1 - msvar.P[1,1]) if msvar.P[1,1] < 1 else np.inf
    st.markdown(f"""
    <div class="terminal-card">
        <div class="metric-title">Expected Regime Durations</div>
        <div class="metric-value">{dur_bear:.1f} <span style="font-size:1.25rem; color:#64748B;">Days</span></div>
        <div class="metric-status" style="color: #64748B;">Calm Regime Expected Base: {1/(1-msvar.P[0,0]):.1f} Days</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    vol_forecasts = {}
    for name in data.columns:
        g = garch_models[name]
        e_last = residuals[name][-1]
        h_last = {0: g.h[0][-1], 1: g.h[1][-1]}
        vol_forecasts[name] = np.sqrt(g.forecast(e_last, h_last, xi_now)) * 100
    
    top_vol_asset = max(vol_forecasts, key=vol_forecasts.get)
    st.markdown(f"""
    <div class="terminal-card">
        <div class="metric-title">Highest Predicted Next-Day Vol</div>
        <div class="metric-value">{vol_forecasts[top_vol_asset]:.2f}%</div>
        <div class="metric-status" style="color: #38BDF8;">Asset Identifier: {top_vol_asset}</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Tabbed Workspace Architecture
# ----------------------------------------------------------------------
tab_monitor, tab_transmission, tab_audit = st.tabs([
    "Regime & Volatility Monitor", 
    "Transmission & Impulse Responses", 
    "Model Parameterization & Audits"
])

# ======================================================================
# TAB 1: REGIME & VOLATILITY MONITOR
# ======================================================================
with tab_monitor:
    st.markdown("### Regime‑Conditional Variance")
    
    col_sel, col_tbl = st.columns([3, 2])
    with col_sel:
        asset_variance = st.selectbox("Isolate Conditional Volatility Target:", data.columns, label_visibility="collapsed")
    
    garch_asset = garch_models[asset_variance]
    
    # Extract raw tracks and latent Markov probabilities
    p_calm = msvar.smoothed_probs[:, 0]
    p_bear = msvar.smoothed_probs[:, 1]
    h_calm = garch_asset.h[0]
    h_bear = garch_asset.h[1]
    
    # Blend the tracks mathematically to match system state reality
    blended_daily_vol = np.sqrt(p_calm * h_calm + p_bear * h_bear) * 100
    dates = data.index[1:]
    
    fig_var = go.Figure()
    max_y = float(blended_daily_vol.max()) * 1.15
    
    # Background panel overlay: Subtle Blue for Calm Regime Dominance
    fig_var.add_trace(go.Scatter(
        x=dates, y=np.where(p_calm >= 0.5, max_y, 0),
        fill='tozeroy', fillcolor='rgba(14, 165, 233, 0.02)',
        line=dict(width=0), name='Calm Regime Territory', hoverinfo='skip', showlegend=False
    ))
    
    # Background panel overlay: Faint Crimson Red for Bear Regime Dominance
    fig_var.add_trace(go.Scatter(
        x=dates, y=np.where(p_bear > 0.5, max_y, 0),
        fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.08)',
        line=dict(width=0), name='Panic Regime Territory', hoverinfo='skip', showlegend=False
    ))
    
    # Base Volatility Trace Line: Understated Clean Electric Blue
    fig_var.add_trace(go.Scatter(
        x=dates, y=blended_daily_vol,
        mode='lines', name='Calm Mode Vol',
        line=dict(color='#0EA5E9', width=1.5)
    ))
    
    # Foreground Highlight Overlay: Warning Red (Masked to show only during active Bear cycles)
    bear_line_mask = np.where(p_bear >= 0.5, blended_daily_vol, np.nan)
    fig_var.add_trace(go.Scatter(
        x=dates, y=bear_line_mask,
        mode='lines', name='Bear Mode Vol (Active Risk)',
        line=dict(color='#EF4444', width=2.0),
        connectgaps=False
    ))
    
    fig_var.update_layout(
        height=380, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_var.update_xaxes(showgrid=True, gridcolor='#1E293B', zeroline=False)
    fig_var.update_yaxes(showgrid=True, gridcolor='#1E293B', title_text="Daily Volatility (%)", range=[0, max_y])
    
    st.plotly_chart(fig_var, use_container_width=True, config={'displayModeBar': False})
    
    with col_tbl:
        st.markdown("<div style='margin-top:-45px;'></div>", unsafe_allow_html=True)
        vol_summary_df = pd.DataFrame({
            "Systemic Asset Base": list(vol_forecasts.keys()),
            "Next-Day Blended Vol Forecast": [f"{v:.3f}%" for v in vol_forecasts.values()]
        })
        st.dataframe(vol_summary_df, hide_index=True, use_container_width=True)

# ======================================================================
# TAB 2: TRANSMISSION & IMPULSE RESPONSES
# ======================================================================
with tab_transmission:
    st.markdown("### Ergodicity Validation & Structural Tail Risk")
    st.markdown(
        "Linear structural VARs assume steady-state ergodicity and Gaussian normal errors. "
        "When **Excess Kurtosis ($>0$)** is present, distributions violate normal parameters. This diagnostic validates the mathematical requirement of utilizing a path-dependent Markov-Switching design."
    )
    
    excess_kurtosis = data.kurtosis()
    skewness = data.skew()
    kurt_df = pd.DataFrame({
        "Asset Matrix": data.columns,
        "Excess Kurtosis": excess_kurtosis.values,
        "Skewness Coefficient": skewness.values
    })
    
    col_k_chart, col_k_metrics = st.columns([3, 2])
    with col_k_chart:
        fig_kurt = go.Figure()
        fig_kurt.add_trace(go.Bar(
            x=kurt_df["Asset Matrix"], y=kurt_df["Excess Kurtosis"],
            marker_color=['#EF4444' if k > 1.2 else '#38BDF8' for k in kurt_df["Excess Kurtosis"]],
            text=kurt_df["Excess Kurtosis"].round(2), textposition='auto',
            textfont=dict(color='#F8FAFC')
        ))
        fig_kurt.add_shape(type="line", x0=-0.5, x1=len(data.columns)-0.5, y0=0, y1=0,
                           line=dict(color="#64748B", width=1.5, dash="dash"))
        fig_kurt.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(title="Value", showgrid=True, gridcolor='#1E293B'),
            xaxis=dict(showgrid=False), margin=dict(l=10, r=10, t=10, b=10), height=240
        )
        st.plotly_chart(fig_kurt, use_container_width=True, config={'displayModeBar': False})
        
    with col_k_metrics:
        st.dataframe(kurt_df, hide_index=True, use_container_width=True)
        st.caption("*Note: Red markers represent significant leptokurtosis, highlighting high-probability tail risks that cause classic linear IRF bounds to collapse under standard unconditioned expectations.*")

    st.markdown("---")
    
    st.markdown("### Generalised Macro Transmission Vectors")
    irf_cols = st.columns([2, 2, 1])
    with irf_cols[0]:
        shock_var = st.selectbox("Initiate Shock Origin:", data.columns, key="term_shk")
    with irf_cols[1]:
        shock_size = st.radio("Impulse Boundary Weighting:", ["+1% Shock", "+1 Std. Dev. Shock"], horizontal=True, key="term_sz")
        
    idx = list(data.columns).index(shock_var)
    shock_vec = np.zeros(len(data.columns))
    
    if "+1%" in shock_size:
        shock_vec[idx] = 0.01
    else:
        erg = msvar._ergodic_probs(msvar.P)
        uncond_var = np.zeros(len(data.columns))
        for k in range(msvar.K):
            uncond_var += erg[k] * np.diag(msvar.covs[k])
        shock_vec[idx] = np.sqrt(uncond_var[idx])

    # Universal Un-Nested Computation Execution Block
    try:
        mean_irf, lower_irf, upper_irf = msvar.simulate_irf(shock_vec, horizon=20, n_sim=500)
        use_simulation = True
    except Exception:
        mean_irf = msvar.impulse_response(shock_vec, horizon=20)
        use_simulation = False

    cols_plot = data.columns
    n_cols = len(cols_plot)
    fig_irf = make_subplots(rows=1, cols=n_cols, subplot_titles=list(cols_plot))
    
    for i, col_name in enumerate(cols_plot):
        h = np.arange(21)
        fig_irf.add_trace(go.Scatter(
            x=h, y=mean_irf[:, i]*100, mode='lines', name=col_name,
            line=dict(color='#38BDF8', width=2)
        ), row=1, col=i+1)
        
        if use_simulation:
            fig_irf.add_trace(go.Scatter(
                x=np.concatenate([h, h[::-1]]),
                # Fixed Python Operator Precedence Bug below using proper tuple wrapping
                y=np.concatenate([lower_irf[:, i]*100, (upper_irf[:, i]*100)[::-1]]),
                fill='toself', fillcolor='rgba(56,189,248,0.12)',
                line=dict(color='rgba(255,255,255,0)'), showlegend=False
            ), row=1, col=i+1)
        fig_irf.add_shape(type="line", x0=0, x1=20, y0=0, y1=0, line=dict(color="#475569", width=1), row=1, col=i+1)

    fig_irf.update_layout(
        height=280, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified"
    )
    fig_irf.update_xaxes(showgrid=True, gridcolor='#1E293B')
    fig_irf.update_yaxes(showgrid=True, gridcolor='#1E293B')
    
    st.plotly_chart(fig_irf, use_container_width=True, config={'displayModeBar': False})
    st.caption("Showing full system transmission tracing over a 20-day path baseline horizon.")

# ======================================================================
# TAB 3: MODEL PARAMETERIZATION & AUDITS
# ======================================================================
with tab_audit:
    st.markdown("### Regime Persistence and Decay Dynamics")
    
    col_p1, col_p2 = st.columns([2, 3])
    with col_p1:
        st.markdown("**Transition Matrix Probs ($P$)**")
        st.dataframe(pd.DataFrame(msvar.P, columns=["To Calm", "To Bear"], index=["From Calm", "From Bear"]).style.format("{:.4f}"), use_container_width=True)
        
    with col_p2:
        st.markdown("**GARCH Architecture Half-Lives**")
        param_records = []
        for name in data.columns:
            g = garch_models[name]
            for k, lbl in enumerate(["Calm", "Bear"]):
                omega, alpha, beta = g.params[k]
                pers = alpha + beta
                hl = np.log(0.5) / np.log(pers) if pers < 1 else np.inf
                param_records.append({"Asset": name, "Regime": lbl, "Persistence": pers, "Half-Life (Days)": f"{hl:.1f}" if hl != np.inf else "Infinite"})
        st.dataframe(pd.DataFrame(param_records), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Residual Orthogonality Check (ACF)")
    
    acf_cols = st.columns(len(data.columns))
    for idx, name in enumerate(data.columns):
        with acf_cols[idx]:
            resid = residuals[name]
            acf_vals, conf_int = acf(resid, nlags=12, alpha=0.05)
            lower = conf_int[:, 0] - acf_vals
            upper = conf_int[:, 1] - acf_vals
            lags = np.arange(len(acf_vals))
            
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Bar(x=lags[1:], y=acf_vals[1:], marker_color='#64748B', showlegend=False))
            fig_acf.add_trace(go.Scatter(x=lags[1:], y=upper[1:], mode='lines', line=dict(width=0), showlegend=False))
            fig_acf.add_trace(go.Scatter(x=lags[1:], y=lower[1:], mode='lines', fill='tonexty', fillcolor='rgba(148,163,184,0.1)', line=dict(width=0), showlegend=False))
            fig_acf.update_layout(
                title=f"{name} Residuals", template="plotly_dark", 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=5, r=5, t=30, b=5), height=180,
                xaxis=dict(showgrid=False, tickvals=[3, 6, 9, 12]), yaxis=dict(showgrid=True, gridcolor='#1E293B', range=[-0.3, 0.3])
            )
            st.plotly_chart(fig_acf, use_container_width=True, config={'displayModeBar': False})
