import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import acf
from msvar_model import MSIAHVAR
from msgarch_model import WeightedGARCH

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="Macro Regime Terminal", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #08090D;
    color: #94A3B8;
    font-family: 'Inter', system-ui, sans-serif;
}

[data-testid="stMainBlockContainer"] { padding: 1.5rem 2rem 2rem 2rem; }

h1, h2, h3, h4 { color: #F1F5F9 !important; font-weight: 500 !important; letter-spacing: -0.01em; }

/* KPI cards */
.kpi-card {
    background: #0F1117;
    border: 0.5px solid #1E293B;
    border-radius: 8px;
    padding: 14px 16px;
}
.kpi-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #475569;
    font-weight: 500;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 24px;
    font-weight: 500;
    line-height: 1;
    color: #F1F5F9;
}
.kpi-sub {
    font-size: 12px;
    color: #475569;
    margin-top: 5px;
}

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 500;
    border: 0.5px solid;
}
.pill-calm  { background: rgba(16,110,86,0.15); color: #34D399; border-color: #0F6E56; }
.pill-trans { background: rgba(180,120,0,0.15); color: #FCD34D; border-color: #854F0B; }
.pill-bear  { background: rgba(153,45,45,0.15); color: #F87171; border-color: #A32D2D; }

/* Ergodicity badge */
.erg-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 500;
}
.erg-pass  { background: rgba(16,110,86,0.15); color: #34D399; border: 0.5px solid #0F6E56; }
.erg-warn  { background: rgba(180,120,0,0.15); color: #FCD34D; border: 0.5px solid #854F0B; }
.erg-fail  { background: rgba(153,45,45,0.15); color: #F87171; border: 0.5px solid #A32D2D; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 0.5px solid #1E293B;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 8px 18px;
    color: #475569;
    font-size: 13px;
    font-weight: 400;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #F1F5F9 !important;
    border-bottom: 2px solid #F1F5F9 !important;
    font-weight: 500 !important;
}

/* Selectbox / radio */
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label { font-size: 12px; color: #475569; }

/* Divider */
hr { border: none; border-top: 0.5px solid #1E293B; margin: 1.25rem 0; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 0.5px solid #1E293B; border-radius: 6px; }

/* Caption */
.stCaption { color: #475569 !important; font-size: 11px !important; }

/* Note text */
.note-text {
    font-size: 12px;
    color: #475569;
    line-height: 1.6;
    margin-top: 6px;
}

/* Section eyebrow */
.eyebrow {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #475569;
    font-weight: 500;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Load state
# ----------------------------------------------------------------------
STATE_FILE = "model_state.pkl"

def load_terminal_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            state = pickle.load(f)
        return state["msvar"], state["garch_models"], state["residuals"], state["data"]
    return None, None, None, None

msvar, garch_models, residuals, data = load_terminal_state()

if msvar is None:
    st.error("Model state not found. Run `update_models.py` to generate parameters.")
    st.stop()

# ----------------------------------------------------------------------
# Derived state
# ----------------------------------------------------------------------
xi_now        = msvar.filtered_probs[-1]
pred_tomorrow = msvar.P.T @ xi_now
prob_bear_tm  = pred_tomorrow[1]
prob_bear_now = xi_now[1]

dur_calm = 1 / (1 - msvar.P[0, 0]) if msvar.P[0, 0] < 1 else np.inf
dur_bear = 1 / (1 - msvar.P[1, 1]) if msvar.P[1, 1] < 1 else np.inf

vol_forecasts = {}
for name in data.columns:
    g      = garch_models[name]
    e_last = residuals[name][-1]
    h_last = {0: g.h[0][-1], 1: g.h[1][-1]}
    vol_forecasts[name] = np.sqrt(g.forecast(e_last, h_last, xi_now)) * 100

top_vol_asset = max(vol_forecasts, key=vol_forecasts.get)

# Regime colour
if prob_bear_tm < 0.35:
    regime_color, pill_cls, regime_label = "#34D399", "pill-calm",  "Calm regime"
elif prob_bear_tm < 0.65:
    regime_color, pill_cls, regime_label = "#FCD34D", "pill-trans", "Indeterminate regime"
else:
    regime_color, pill_cls, regime_label = "#F87171", "pill-bear",  "Bear regime"

# Plotly base layout
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color="#94A3B8", size=11),
    margin=dict(l=8, r=8, t=8, b=8),
)
_GRID = dict(showgrid=True, gridcolor="#1E293B", zeroline=False)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
col_title, col_pill = st.columns([5, 1])
with col_title:
    st.markdown("## Regime Terminal")
with col_pill:
    st.markdown(f"""
    <div style="display:flex; justify-content:flex-end; padding-top:10px;">
        <span class="status-pill {pill_cls}">
            <span style="width:6px;height:6px;border-radius:50%;background:{regime_color};display:inline-block;"></span>
            {regime_label}
        </span>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 3px solid {regime_color};">
        <div class="kpi-label">Bear prob tomorrow</div>
        <div class="kpi-value" style="color:{regime_color};">{prob_bear_tm*100:.1f}%</div>
        <div class="kpi-sub">Today: {prob_bear_now*100:.1f}% smoothed</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Expected regime duration</div>
        <div class="kpi-value">{dur_calm:.0f} <span style="font-size:14px;color:#475569;">days calm</span></div>
        <div class="kpi-sub">Bear avg: {dur_bear:.0f} days</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Highest next-day vol</div>
        <div class="kpi-value">{vol_forecasts[top_vol_asset]:.2f}%</div>
        <div class="kpi-sub">{top_vol_asset} blended forecast</div>
    </div>""", unsafe_allow_html=True)

with k4:
    ergodic_probs = msvar._ergodic_probs(msvar.P)
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Ergodic regime weights</div>
        <div class="kpi-value">{ergodic_probs[0]*100:.0f}% <span style="font-size:14px;color:#475569;">/ {ergodic_probs[1]*100:.0f}%</span></div>
        <div class="kpi-sub">Calm / Bear long-run</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_monitor, tab_transmission, tab_diagnostics = st.tabs([
    "Regime & volatility",
    "Shock transmission",
    "Diagnostics",
])

# ======================================================================
# TAB 1 — REGIME & VOLATILITY
# ======================================================================
with tab_monitor:
    asset_sel = st.selectbox(
        "Asset",
        options=list(data.columns),
        label_visibility="collapsed"
    )

    garch_asset = garch_models[asset_sel]
    p_calm      = msvar.smoothed_probs[:, 0]
    p_bear      = msvar.smoothed_probs[:, 1]
    h_calm      = garch_asset.h[0]
    h_bear      = garch_asset.h[1]
    blended_vol = np.sqrt(p_calm * h_calm + p_bear * h_bear) * 100
    dates       = data.index[1:]
    max_y       = float(blended_vol.max()) * 1.2

    fig_vol = go.Figure()

    # Bear regime shading
    fig_vol.add_trace(go.Scatter(
        x=dates, y=np.where(p_bear > 0.5, max_y, 0),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.07)",
        line=dict(width=0), hoverinfo="skip", showlegend=False, name=""
    ))

    # Calm vol line
    fig_vol.add_trace(go.Scatter(
        x=dates, y=blended_vol,
        mode="lines", name="Blended vol",
        line=dict(color="#38BDF8", width=1.5)
    ))

    # Bear regime highlight
    bear_mask = np.where(p_bear >= 0.5, blended_vol, np.nan)
    fig_vol.add_trace(go.Scatter(
        x=dates, y=bear_mask,
        mode="lines", name="Bear vol",
        line=dict(color="#F87171", width=2),
        connectgaps=False
    ))

    fig_vol.update_layout(
        **_LAYOUT, height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig_vol.update_xaxes(**_GRID)
    fig_vol.update_yaxes(**_GRID, title_text="Daily vol (%)", range=[0, max_y])

    st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})
    st.markdown('<p class="note-text">Red shading = bear regime dominant (kim smoothed prob > 50%)</p>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Next-day volatility forecasts</div>', unsafe_allow_html=True)

    vol_df = pd.DataFrame({
        "Asset":            list(vol_forecasts.keys()),
        "Calm regime vol":  [f"{np.sqrt(garch_models[n].h[0][-1])*100:.3f}%" for n in data.columns],
        "Bear regime vol":  [f"{np.sqrt(garch_models[n].h[1][-1])*100:.3f}%" for n in data.columns],
        "Blended forecast": [f"{v:.3f}%" for v in vol_forecasts.values()],
        "Regime weight":    [f"Calm {xi_now[0]*100:.0f}% / Bear {xi_now[1]*100:.0f}%" for _ in data.columns],
    })
    st.dataframe(vol_df, hide_index=True, use_container_width=True)

# ======================================================================
# TAB 2 — SHOCK TRANSMISSION
# ======================================================================
with tab_transmission:
    st.markdown('<div class="eyebrow">How does a shock to one asset move through the system?</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:13px;color:#64748B;margin-bottom:1rem;line-height:1.6;">'
        'When the ergodicity assumptions hold, these are in theory closed-form solutions'
        'of how a shock propagates across each asset over a 20-day horizon. '
        'Check the Diagnostics tab to verify assumptions before interpreting.'
        '</p>',
        unsafe_allow_html=True
    )

    ctrl1, ctrl2 = st.columns([2, 2])
    with ctrl1:
        shock_var = st.selectbox("Shock origin", data.columns, key="shock_origin")
    with ctrl2:
        shock_size = st.radio(
            "Shock size",
            ["+1% shock", "+1 std. dev."],
            horizontal=True, key="shock_size"
        )

    idx       = list(data.columns).index(shock_var)
    shock_vec = np.zeros(len(data.columns))

    if "+1%" in shock_size:
        shock_vec[idx] = 0.01
    else:
        erg       = msvar._ergodic_probs(msvar.P)
        uncond_v  = sum(erg[k] * np.diag(msvar.covs[k]) for k in range(msvar.K))
        shock_vec[idx] = np.sqrt(uncond_v[idx])

    try:
        mean_irf, lower_irf, upper_irf = msvar.simulate_irf(shock_vec, horizon=20, n_sim=500)
        use_sim = True
    except Exception:
        mean_irf = msvar.impulse_response(shock_vec, horizon=20)
        use_sim  = False

    n_vars  = len(data.columns)
    fig_irf = make_subplots(
        rows=1, cols=n_vars,
        subplot_titles=list(data.columns),
        horizontal_spacing=0.04
    )

    for i, col_name in enumerate(data.columns):
        y_mid = mean_irf[:, i] * 100
        h     = np.arange(len(y_mid))
        color = "#38BDF8" if y_mid[-1] >= 0 else "#F87171"

        if use_sim:
            fig_irf.add_trace(go.Scatter(
                x=np.concatenate([h, h[::-1]]),
                y=np.concatenate([lower_irf[:, i]*100, (upper_irf[:, i]*100)[::-1]]),
                fill="toself", fillcolor="rgba(56,189,248,0.08)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"
            ), row=1, col=i+1)

        fig_irf.add_trace(go.Scatter(
            x=h, y=y_mid, mode="lines", name=col_name,
            line=dict(color=color, width=1.8), showlegend=False
        ), row=1, col=i+1)

        fig_irf.add_shape(
            type="line", x0=0, x1=h[-1] if len(h) > 0 else 20, y0=0, y1=0,
            line=dict(color="#334155", width=1),
            row=1, col=i+1
        )

    fig_irf.update_layout(**_LAYOUT, height=260, hovermode="x unified")
    fig_irf.update_xaxes(**_GRID)
    fig_irf.update_yaxes(**_GRID)

    st.plotly_chart(fig_irf, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<p class="note-text">20-day horizon · shaded band = 95% simulation bounds · '
        'response sign (blue positive / red negative) relative to zero line</p>',
        unsafe_allow_html=True
    )

# ======================================================================
# TAB 3 — DIAGNOSTICS
# ======================================================================
with tab_diagnostics:

    # ------------------------------------------------------------------
    # SECTION A: Ergodicity checker
    # ------------------------------------------------------------------
    st.markdown('<div class="eyebrow">Ergodicity check — are model assumptions valid?</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:13px;color:#64748B;margin-bottom:1rem;line-height:1.6;">'
        'Kole &amp; van Dijk impulse responses require each regime to have a stable long-run mean and variance '
        '(ergodicity). This checker tests whether the empirical data in each regime supports that. '
        'GARCH persistence close to 1 signals non-ergodic variance. '
        'ADF failure signals non-ergodic mean.'
        '</p>',
        unsafe_allow_html=True
    )

    from statsmodels.tsa.stattools import adfuller

    erg_records = []
    for name in data.columns:
        g   = garch_models[name]
        raw = data[name].dropna().values

        # ADF test on raw series (mean ergodicity proxy)
        try:
            adf_stat, adf_p, *_ = adfuller(raw, autolag="AIC")
            mean_ergodic = adf_p < 0.05
        except Exception:
            adf_p, mean_ergodic = np.nan, None

        # GARCH persistence per regime
        pers = {}
        for k, lbl in enumerate(["Calm", "Bear"]):
            omega, alpha, beta = g.params[k]
            pers[lbl] = alpha + beta

        var_ergodic_calm = pers["Calm"] < 1.0
        var_ergodic_bear = pers["Bear"] < 1.0
        overall = mean_ergodic and var_ergodic_calm and var_ergodic_bear

        if overall:
            badge_cls, badge_txt = "erg-pass", "Assumptions hold"
        elif mean_ergodic and (var_ergodic_calm or var_ergodic_bear):
            badge_cls, badge_txt = "erg-warn", "Monitor closely"
        else:
            badge_cls, badge_txt = "erg-fail", "Assumptions at risk"

        erg_records.append({
            "name":       name,
            "adf_p":      adf_p,
            "mean_erg":   mean_ergodic,
            "pers_calm":  pers["Calm"],
            "pers_bear":  pers["Bear"],
            "var_erg_c":  var_ergodic_calm,
            "var_erg_b":  var_ergodic_bear,
            "badge_cls":  badge_cls,
            "badge_txt":  badge_txt,
        })

    # Render ergodicity table
    erg_cols = st.columns(len(data.columns))
    for col_el, rec in zip(erg_cols, erg_records):
        with col_el:
            mean_color = "#34D399" if rec["mean_erg"] else "#F87171"
            vc_color   = "#34D399" if rec["var_erg_c"] else "#F87171"
            vb_color   = "#34D399" if rec["var_erg_b"] else "#F87171"
            st.markdown(f"""
            <div style="background:#0F1117;border:0.5px solid #1E293B;border-radius:8px;padding:12px 14px;">
                <div style="font-size:12px;font-weight:500;color:#F1F5F9;margin-bottom:8px;">{rec['name']}</div>
                <div class="erg-badge {rec['badge_cls']}" style="margin-bottom:10px;">{rec['badge_txt']}</div>
                <div style="font-size:11px;color:#475569;margin-top:4px;">Mean ergodic (ADF)</div>
                <div style="font-size:12px;color:{mean_color};margin-bottom:4px;">p = {rec['adf_p']:.4f}</div>
                <div style="font-size:11px;color:#475569;">Var persist · calm</div>
                <div style="font-size:12px;color:{vc_color};margin-bottom:4px;">{rec['pers_calm']:.4f}</div>
                <div style="font-size:11px;color:#475569;">Var persist · bear</div>
                <div style="font-size:12px;color:{vb_color};">{rec['pers_bear']:.4f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # SECTION B: Leptokurtosis distribution curves
    # ------------------------------------------------------------------
    st.markdown('<div class="eyebrow">Leptokurtosis — tail risk relative to normal</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:13px;color:#64748B;margin-bottom:1rem;line-height:1.6;">'
        'Each curve shows the empirical return distribution (KDE) against a matched normal. '
        'Heavier tails and sharper peaks confirm excess kurtosis — the statistical justification '
        'for a regime-switching model over a plain VAR.'
        '</p>',
        unsafe_allow_html=True
    )

    n_vars   = len(data.columns)
    kurt_fig = make_subplots(
        rows=1, cols=n_vars,
        subplot_titles=list(data.columns),
        horizontal_spacing=0.04
    )

    KURT_WARN  = 1.2   # amber
    KURT_HIGH  = 3.0   # red

    for i, col_name in enumerate(data.columns):
        series   = data[col_name].dropna().values
        excess_k = float(pd.Series(series).kurtosis())   # excess kurtosis
        skew_val = float(pd.Series(series).skew())

        mu, sigma = series.mean(), series.std()
        x_range   = np.linspace(mu - 4.5*sigma, mu + 4.5*sigma, 300)

        # KDE
        kde      = scipy_stats.gaussian_kde(series, bw_method="silverman")
        kde_vals = kde(x_range)

        # Matched normal
        norm_vals = scipy_stats.norm.pdf(x_range, mu, sigma)

        # Colour by severity
        if abs(excess_k) < KURT_WARN:
            curve_color = "#34D399"     # green — close to normal
        elif abs(excess_k) < KURT_HIGH:
            curve_color = "#FCD34D"     # amber — moderate fat tails
        else:
            curve_color = "#F87171"     # red — high leptokurtosis

        # Normal reference
        kurt_fig.add_trace(go.Scatter(
            x=x_range, y=norm_vals,
            mode="lines", name="Normal",
            line=dict(color="#334155", width=1, dash="dot"),
            showlegend=(i == 0), legendgroup="normal"
        ), row=1, col=i+1)

        # Empirical KDE
        kurt_fig.add_trace(go.Scatter(
            x=x_range, y=kde_vals,
            mode="lines", name=col_name,
            line=dict(color=curve_color, width=1.8),
            showlegend=False,
            hovertemplate=f"<b>{col_name}</b><br>x: %{{x:.4f}}<br>density: %{{y:.4f}}<extra></extra>"
        ), row=1, col=i+1)

        # Kurtosis annotation inside each subplot
        kurt_fig.add_annotation(
            x=0.5, y=0.97, xref=f"x{i+1} domain" if i > 0 else "x domain",
            yref=f"y{i+1} domain" if i > 0 else "y domain",
            text=f"κ = {excess_k:.2f}",
            showarrow=False, font=dict(size=10, color=curve_color),
            xanchor="center", yanchor="top"
        )

    kurt_fig.update_layout(
        **_LAYOUT, height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )
    kurt_fig.update_xaxes(**_GRID)
    kurt_fig.update_yaxes(**_GRID, showticklabels=False)

    st.plotly_chart(kurt_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<p class="note-text">'
        'κ = excess kurtosis (0 = normal). '
        'Green &lt; 1.2 · Amber 1.2 – 3.0 · Red &gt; 3.0. '
        'Dotted line = matched normal distribution.'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # SECTION C: Kurtosis & skew summary table
    # ------------------------------------------------------------------
    st.markdown('<div class="eyebrow">Distribution statistics</div>', unsafe_allow_html=True)

    dist_records = []
    for col_name in data.columns:
        series   = data[col_name].dropna().values
        excess_k = float(pd.Series(series).kurtosis())
        skew_val = float(pd.Series(series).skew())

        if abs(excess_k) < KURT_WARN:
            k_status = "Normal tails"
        elif abs(excess_k) < KURT_HIGH:
            k_status = "Moderate fat tails"
        else:
            k_status = "High leptokurtosis"

        dist_records.append({
            "Asset":           col_name,
            "Excess kurtosis": round(excess_k, 3),
            "Skewness":        round(skew_val, 3),
            "Tail assessment": k_status,
        })

    st.dataframe(pd.DataFrame(dist_records), hide_index=True, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # SECTION D: Residual ACF
    # ------------------------------------------------------------------
    st.markdown('<div class="eyebrow">Residual autocorrelation — model error quality</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:13px;color:#64748B;margin-bottom:1rem;line-height:1.6;">'
        'Residuals should be white noise. Bars outside the dashed confidence bounds '
        'indicate unexplained structure remaining in model errors.'
        '</p>',
        unsafe_allow_html=True
    )

    acf_fig = make_subplots(
        rows=1, cols=n_vars,
        subplot_titles=list(data.columns),
        horizontal_spacing=0.04
    )

    for i, name in enumerate(data.columns):
        resid               = residuals[name]
        acf_vals, conf_int  = acf(resid, nlags=12, alpha=0.05)
        lower_ci            = conf_int[:, 0] - acf_vals
        upper_ci            = conf_int[:, 1] - acf_vals
        lags                = np.arange(len(acf_vals))

        breach = [abs(acf_vals[j]) > abs(upper_ci[j]) for j in range(1, len(acf_vals))]
        bar_colors = ["#F87171" if b else "#475569" for b in breach]

        acf_fig.add_trace(go.Bar(
            x=lags[1:], y=acf_vals[1:],
            marker_color=bar_colors, showlegend=False,
            hovertemplate="Lag %{x}: %{y:.3f}<extra></extra>"
        ), row=1, col=i+1)

        acf_fig.add_trace(go.Scatter(
            x=lags[1:], y=upper_ci[1:], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ), row=1, col=i+1)
        acf_fig.add_trace(go.Scatter(
            x=lags[1:], y=lower_ci[1:], mode="lines",
            fill="tonexty", fillcolor="rgba(148,163,184,0.07)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ), row=1, col=i+1)

    acf_fig.update_layout(**_LAYOUT, height=220)
    acf_fig.update_xaxes(**_GRID, tickvals=[3, 6, 9, 12])
    acf_fig.update_yaxes(**_GRID, range=[-0.3, 0.3])

    st.plotly_chart(acf_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<p class="note-text">Red bars = outside 95% confidence bounds. Shaded band = confidence interval.</p>',
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # SECTION E: Transition matrix + GARCH persistence
    # ------------------------------------------------------------------
    st.markdown('<div class="eyebrow">Model parameters</div>', unsafe_allow_html=True)
    col_p1, col_p2 = st.columns([1, 2])

    with col_p1:
        st.markdown("**Transition matrix**")
        trans_df = pd.DataFrame(
            msvar.P,
            columns=["→ Calm", "→ Bear"],
            index=["From Calm", "From Bear"]
        )
        st.dataframe(trans_df.style.format("{:.4f}"), use_container_width=True)

    with col_p2:
        st.markdown("**GARCH persistence & volatility half-lives**")
        param_rows = []
        for name in data.columns:
            g = garch_models[name]
            for k, lbl in enumerate(["Calm", "Bear"]):
                omega, alpha, beta = g.params[k]
                pers = alpha + beta
                hl   = np.log(0.5) / np.log(pers) if pers < 1 else np.inf
                param_rows.append({
                    "Asset":        name,
                    "Regime":       lbl,
                    "α + β":        round(pers, 4),
                    "Half-life":    f"{hl:.1f} days" if hl != np.inf else "∞",
                    "Ergodic":      "Yes" if pers < 1 else "No",
                })
        st.dataframe(pd.DataFrame(param_rows), hide_index=True, use_container_width=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.caption(
        "Impulse response methods: Kole, E., & van Dijk, D. (2023). "
        "Moments, shocks and spillovers in Markov-switching VAR models. "
        "Journal of Econometrics, 236(2), 105474."
    )
