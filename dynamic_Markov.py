from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import acf, pacf
import yfinance as yf
from arch import arch_model


class RegimeAnalyzer:

    def __init__(self, ticker: str, start: str, end: str, k_regimes: int = 2) -> None:
        self.ticker: str = ticker
        self.start: str = start
        self.end: str = end
        self.k_regimes: int = k_regimes
        
        self.raw_prices: pd.Series = self._fetch_raw_data()
        self.data: pd.Series = self.raw_prices.pct_change().dropna()
        
        self.res: Any = None
        self.garch_res: Any = None
        self.ms_garch_cascade: bool = False
        self.converged: bool = False

    def _fetch_raw_data(self) -> pd.Series:
        df = yf.download(self.ticker, start=self.start, end=self.end, progress=False)
        if df.empty:
            raise ValueError(f"No data found for: {self.ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            prices = df["Adj Close"].iloc[:, 0] if "Adj Close" in df.columns.levels[0] else df["Close"].iloc[:, 0]
        else:
            prices = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        return prices

    def _compute_residuals(self, use_filtered: bool = True) -> np.ndarray:
        if self.res is None:
            return np.array([])
            
        adjusted_data = self.data.iloc[1:].to_numpy()
        
        if use_filtered:
            probs = self.res.filtered_marginal_probabilities.to_numpy()
        else:
            probs = self.res.smoothed_marginal_probabilities.to_numpy()
            
        means = np.array([float(self.res.params.get(f"const[{i}]", 0.0)) for i in range(self.k_regimes)])
        variances = np.array([float(self.res.params.get(f"sigma2[{i}]", 1e-6)) for i in range(self.k_regimes)])
        rho = float(self.res.params.get("ar.L1", 0.0))
        lagged_data = self.data.iloc[:-1].to_numpy()
        
        cond_mean = np.zeros(len(adjusted_data))
        cond_var = np.zeros(len(adjusted_data))
        
        for t in range(len(adjusted_data)):
            p = probs[t]
            state_means = means + rho * lagged_data[t]
            cond_mean[t] = np.sum(p * state_means)
            cond_var[t] = np.sum(p * variances)
            
        return (adjusted_data - cond_mean) / np.sqrt(cond_var)

    def get_standardized_residuals(self) -> np.ndarray:
        ms_resid = self._compute_residuals(use_filtered=False)
        if self.ms_garch_cascade and self.garch_res is not None:
            garch_std_resid = self.garch_res.resid / self.garch_res.conditional_volatility
            return garch_std_resid[~np.isnan(garch_std_resid)]
        return ms_resid

    def fit_model(self) -> None:
        model = MarkovRegression(self.data, k_regimes=self.k_regimes, order=1, trend="c", switching_variance=True)
        
        try:
            self.res = model.fit(disp=False, em_iter=15, search_reps=20, method='bfgs', cov_type="approx")
            self.converged = True
        except Exception:
            try:
                self.res = model.fit(disp=False, em_iter=5, search_reps=10, method='powell', cov_type="approx")
                self.converged = True
            except Exception:
                self.res = None
                self.converged = False
                return

        filtered_residuals = self._compute_residuals(use_filtered=True)
        if len(filtered_residuals) > 10:
            sq_acf_vals = acf(filtered_residuals**2, nlags=5, fft=True)
            c_bound = 1.96 / np.sqrt(len(filtered_residuals))
            
            if np.any(np.abs(sq_acf_vals[1:]) > c_bound):
                self.ms_garch_cascade = True
                self._fit_garch_cascade(filtered_residuals)

    def _fit_garch_cascade(self, ms_residuals: np.ndarray) -> None:
        try:
            scaled_resid = ms_residuals * 100
            garch_init = arch_model(scaled_resid, vol="Garch", p=1, q=1, mean="Zero", dist="Normal")
            self.garch_res = garch_init.fit(disp=False, update_freq=0)
        except Exception:
            self.ms_garch_cascade = False

    def get_regime_statistics(self) -> pd.DataFrame:
        if self.res is None:
            return pd.DataFrame()

        rows = []
        for i in range(self.k_regimes):
            mu = float(self.res.params.get(f"const[{i}]", 0.0))
            sigma2 = float(self.res.params.get(f"sigma2[{i}]", 1e-6))
            sigma = np.sqrt(sigma2)
            duration = float(self.res.expected_durations[i])
            rows.append({
                "Regime": f"State {i}",
                "Daily Return": f"{mu:.6f}",
                "Annualized Return": f"{(mu * 252 * 100):.2f}%",
                "Base Daily Vol": f"{sigma:.6f}",
                "Expected Duration (Days)": f"{duration:.2f}"
            })
            
        if self.ms_garch_cascade and self.garch_res is not None:
            rows.append({
                "Regime": "Zero-Mean GARCH(1,1)",
                "Daily Return": "-",
                "Annualized Return": "-",
                "Base Daily Vol": "Dynamic",
                "Expected Duration (Days)": "Time-Varying"
            })
            
        return pd.DataFrame(rows)

    def get_blended_forecast(self) -> Dict[str, Any]:
        if self.res is None:
            return {}
            
        p = self.res.regime_transition
        if isinstance(p, np.ndarray) and p.ndim == 3:
            p = p[:, :, 0]
            
        current_probs = np.zeros(self.k_regimes)
        for i in range(self.k_regimes):
            current_probs[i] = float(self.res.smoothed_marginal_probabilities[i].iloc[-1])
            
        forward_probs = np.zeros(self.k_regimes)
        for j in range(self.k_regimes):
            forward_probs[j] = sum(current_probs[i] * p[i, j] for i in range(self.k_regimes))
            
        means = np.array([float(self.res.params.get(f"const[{i}]", 0.0)) for i in range(self.k_regimes)])
        variances = np.array([float(self.res.params.get(f"sigma2[{i}]", 1e-6)) for i in range(self.k_regimes)])
        
        blended_mean = float(np.sum(forward_probs * means))
        ms_blended_variance = float(np.sum(forward_probs * variances) + np.sum(forward_probs * ((means - blended_mean) ** 2)))
        
        if self.ms_garch_cascade and self.garch_res is not None:
            garch_forecast = self.garch_res.forecast(horizon=1)
            garch_multiplier = float(garch_forecast.variance.iloc[-1, 0]) / 10000 
            total_blended_variance = ms_blended_variance * garch_multiplier
        else:
            total_blended_variance = ms_blended_variance
            
        blended_volatility = np.sqrt(total_blended_variance)
        
        return {
            "forward_probabilities": forward_probs,
            "blended_mean_daily": blended_mean,
            "blended_vol_daily": blended_volatility,
            "blended_vol_annualized": blended_volatility * np.sqrt(252) * 100
        }

    def generate_regime_plot(self) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, 
            subplot_titles=("Daily Log Returns", "Regime Probabilities")
        )
        # Price Trace - Slate Gray
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data.values, mode="lines", line=dict(color="#1E293B", width=0.8)), row=1, col=1)
        
        prob_series = self.res.smoothed_marginal_probabilities[self.k_regimes - 1]
        
        # Probability Fill - Muted Blue
        fig.add_trace(
            go.Scatter(x=prob_series.index, y=prob_series.values, mode="lines", line=dict(color="#3B82F6", width=1.0), fill="tozeroy", fillcolor="rgba(59, 130, 246, 0.15)"),
            row=2, col=1
        )
        
        # GARCH Overlay - Crimson Red
        if self.ms_garch_cascade and self.garch_res is not None:
            vol_overlay = (self.garch_res.conditional_volatility / 100)
            vol_overlay = vol_overlay / vol_overlay.max() 
            fig.add_trace(
                go.Scatter(x=self.data.index[1:], y=vol_overlay, mode="lines", line=dict(color="#DC2626", width=1.0, dash="dot")),
                row=2, col=1
            )

        fig.update_layout(
            template="plotly_white", 
            height=400, 
            showlegend=False, 
            margin=dict(l=40, r=20, t=40, b=30), 
            hovermode="x unified",
            font=dict(family="system-ui, -apple-system, sans-serif", color="#334155")
        )
        
        fig.update_annotations(font_size=12, font_color="#475569")
        fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", gridwidth=1)
        fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", gridwidth=1)
        
        return fig

    def generate_contagion_plot(self) -> go.Figure:
        horizons = 10
        days = np.arange(horizons)
        response = np.zeros(horizons)
        
        forecast = self.get_blended_forecast()
        shock = forecast.get("blended_vol_daily", 0.02)
        response[0] = shock

        p = self.res.regime_transition
        if isinstance(p, np.ndarray) and p.ndim == 3:
            p = p[:, :, 0]
            
        ms_persistence = max(0.05, min(0.99, p[0, 0] + p[1, 1] - 1.0))
        
        if self.ms_garch_cascade and self.garch_res is not None:
            alpha = self.garch_res.params.get("alpha[1]", 0.0)
            beta = self.garch_res.params.get("beta[1]", 0.0)
            garch_persistence = alpha + beta
            persistence = max(0.05, min(0.99, (ms_persistence * 0.5) + (garch_persistence * 0.5)))
        else:
            persistence = ms_persistence

        for h in range(1, horizons):
            response[h] = response[h-1] * persistence
            
        upper_band = response + 1.96 * shock * (persistence ** days) * 0.5
        lower_band = response - 1.96 * shock * (persistence ** days) * 0.5
        
        fig = go.Figure()
        # Muted Blue Fill
        fig.add_trace(go.Scatter(x=np.concatenate([days, days[::-1]]), y=np.concatenate([upper_band, lower_band[::-1]]), fill='toself', fillcolor='rgba(59, 130, 246, 0.1)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        fig.add_trace(go.Scatter(x=days, y=response, mode='lines+markers', line=dict(color='#3B82F6', width=2), marker=dict(size=6, color='#2563EB'), showlegend=False))
        fig.add_shape(type="line", x0=0, y0=0, x1=horizons-1, y1=0, line=dict(color="#94A3B8", width=1, dash="dash"))
        
        engine_lbl = "MS + GARCH(1,1)" if self.ms_garch_cascade else "Markov"
        fig.update_layout(
            title=dict(text=f"Impulse Response ({engine_lbl}): {self.ticker}", font=dict(size=12, color="#334155")),
            template="plotly_white", 
            height=260,
            xaxis=dict(title="Days Post-Shock", title_font=dict(size=10), tickmode="linear", tick0=0, dtick=1, showgrid=True, gridcolor="#F1F5F9"),
            yaxis=dict(title="Response", title_font=dict(size=10), showgrid=True, gridcolor="#F1F5F9"), 
            margin=dict(l=40, r=20, t=40, b=35), 
            showlegend=False,
            font=dict(family="system-ui, -apple-system, sans-serif")
        )
        return fig

    def generate_individual_diagnostic_plots(self) -> List[go.Figure]:
        std_residuals = self.get_standardized_residuals()
        if len(std_residuals) == 0:
            return [go.Figure() for _ in range(3)]
            
        nlags = min(20, len(std_residuals) // 5)
        acf_vals = acf(std_residuals, nlags=nlags, fft=True)
        pacf_vals = pacf(std_residuals, nlags=nlags)
        sq_acf_vals = acf(std_residuals**2, nlags=nlags, fft=True)

        lags = np.arange(len(acf_vals))
        c_bound = 1.96 / np.sqrt(len(std_residuals))
        engine_lbl = "MS + GARCH(1,1)" if self.ms_garch_cascade else "Markov"
        titles = [f"{engine_lbl} Residual ACF", f"{engine_lbl} Residual PACF", f"{engine_lbl} Squared Residual ACF"]

        plots = []
        for vals, title in zip([acf_vals, pacf_vals, sq_acf_vals], titles):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=lags, y=vals, marker_color="#64748B", width=0.5))
            fig.add_shape(type="line", x0=0, y0=c_bound, x1=nlags, y1=c_bound, line=dict(color="#DC2626", width=1, dash="dash"))
            fig.add_shape(type="line", x0=0, y0=-c_bound, x1=nlags, y1=-c_bound, line=dict(color="#DC2626", width=1, dash="dash"))
            fig.update_layout(
                title=dict(text=title, font=dict(size=11, color="#334155")),
                template="plotly_white", 
                height=220, 
                xaxis=dict(tickmode="linear", tick0=0, dtick=5, showgrid=True, gridcolor="#F1F5F9"),
                yaxis=dict(range=[-0.5, 1.0], showgrid=True, gridcolor="#F1F5F9"), 
                margin=dict(l=40, r=20, t=40, b=30),
                font=dict(family="system-ui, -apple-system, sans-serif")
            )
            plots.append(fig)
        return plots


class CrossAssetAnalytics:

    @staticmethod
    def get_regime_ewma_span(analyzers: Dict[str, RegimeAnalyzer]) -> int:
        spans = []
        for a in analyzers.values():
            if a.res is not None:
                current_state = np.argmax(a.res.smoothed_marginal_probabilities.iloc[-1].to_numpy())
                expected_dur = float(a.res.expected_durations[current_state])
                spans.append(expected_dur)
        if not spans:
            return 21
        return int(max(5, min(60, np.mean(spans))))

    @staticmethod
    def compute_regime_sync(analyzers: Dict[str, RegimeAnalyzer]) -> pd.DataFrame:
        series_dict = {}
        for ticker, analyzer in analyzers.items():
            if analyzer.res is not None:
                series_dict[ticker] = analyzer.res.smoothed_marginal_probabilities[analyzer.k_regimes - 1]
        if len(series_dict) < 2:
            return pd.DataFrame()
        return pd.DataFrame(series_dict).corr()
