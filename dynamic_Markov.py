from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import acf, adfuller, pacf
import yfinance as yf
from arch import arch_model


class RegimeAnalyzer:

    def __init__(self, ticker: str, start: str, end: str, k_regimes: int = 2) -> None:
        self.ticker: str = ticker
        self.start: str = start
        self.end: str = end
        self.k_regimes: int = k_regimes
        
        # Pull raw series first so we have the absolute price history
        self.raw_prices: pd.Series = self._fetch_raw_data()
        self.data: pd.Series = self.raw_prices.pct_change().dropna()
        
        self.res: Any = None
        self.garch_res: Any = None
        self.garch_order: tuple = (1, 1)
        self.converged: bool = False
        self.use_garch_fallback: bool = False

    def _fetch_raw_data(self) -> pd.Series:
        df = yf.download(self.ticker, start=self.start, end=self.end, progress=False)
        if df.empty:
            raise ValueError(f"No data found for: {self.ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            prices = df["Adj Close"].iloc[:, 0] if "Adj Close" in df.columns.levels[0] else df["Close"].iloc[:, 0]
        else:
            prices = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        return prices

    def _compute_ms_residuals(self) -> np.ndarray:
        if self.res is None:
            return np.array([])
        adjusted_data = self.data.iloc[1:].to_numpy()
        smoothed_probs = self.res.smoothed_marginal_probabilities.to_numpy()
        
        means = np.array([float(self.res.params.get(f"const[{i}]", 0.0)) for i in range(self.k_regimes)])
        variances = np.array([float(self.res.params.get(f"sigma2[{i}]", 1e-6)) for i in range(self.k_regimes)])
        
        rho = float(self.res.params.get("ar.L1", 0.0))
        lagged_data = self.data.iloc[:-1].to_numpy()
        
        cond_mean = np.zeros(len(adjusted_data))
        cond_var = np.zeros(len(adjusted_data))
        
        for t in range(len(adjusted_data)):
            p = smoothed_probs[t]
            state_means = means + rho * lagged_data[t]
            cond_mean[t] = np.sum(p * state_means)
            cond_var[t] = np.sum(p * variances)
            
        return (adjusted_data - cond_mean) / np.sqrt(cond_var)

    def get_standardized_residuals(self) -> np.ndarray:
        if self.use_garch_fallback and self.garch_res is not None:
            return (self.garch_res.resid / self.garch_res.conditional_volatility).dropna().to_numpy()
        return self._compute_ms_residuals()

    def is_stationary(self) -> Dict[str, Any]:
        std_resid = self.get_standardized_residuals()
        if len(std_resid) < 10:
            return {"stationary": False, "p_value": 1.0, "statistic": 0.0}
        result = adfuller(std_resid)
        return {
            "stationary": bool(result[1] < 0.05),
            "p_value": float(result[1]),
            "statistic": float(result[0]),
        }

    def fit_model(self) -> None:
        best_aic = float("inf")
        best_res = None
        best_order = (1, 1)
        candidate_orders = [(1, 0), (2, 0), (1, 1), (2, 1)]
        scaled_data = self.data * 100

        for p, q in candidate_orders:
            try:
                garch_init = arch_model(scaled_data, vol="Garch", p=p, q=q, mean="AR", lags=1, dist="Normal")
                garch_fit = garch_init.fit(disp=False)
                if garch_fit.aic < best_aic:
                    best_aic = garch_fit.aic
                    best_res = garch_fit
                    best_order = (p, q)
            except Exception:
                continue

        self.garch_res = best_res
        self.garch_order = best_order

        model = MarkovRegression(self.data, k_regimes=self.k_regimes, order=1, trend="c", switching_variance=True)
        try:
            self.res = model.fit(disp=False, search_reps=20, cov_type="approx")
            self.converged = True
        except Exception:
            try:
                self.res = model.fit(disp=False, search_reps=0, cov_type="none")
                self.converged = False
            except Exception:
                self.res = None
                self.converged = False

        # If residuals have autocorrelation in variance, flip the switch to use GARCH fallback
        if self.res is not None:
            ms_residuals = self._compute_ms_residuals()
            if len(ms_residuals) > 10:
                sq_acf_vals = acf(ms_residuals**2, nlags=5, fft=True)
                c_bound = 1.96 / np.sqrt(len(ms_residuals))
                if np.any(np.abs(sq_acf_vals[1:]) > c_bound):
                    self.use_garch_fallback = True
                else:
                    self.use_garch_fallback = False
            else:
                self.use_garch_fallback = True
        else:
            self.use_garch_fallback = True

    def get_regime_statistics(self) -> pd.DataFrame:
        if self.use_garch_fallback and self.garch_res is not None:
            omega = self.garch_res.params.get("omega", 0.0) / 10000
            alpha = self.garch_res.params.get("alpha[1]", 0.0)
            beta = self.garch_res.params.get("beta[1]", 0.0)
            uncond_var = omega / (1 - alpha - beta) if (1 - alpha - beta) > 0 else 1e-4
            uncond_vol = np.sqrt(uncond_var)
            return pd.DataFrame([{
                "Regime": "GARCH Pivot State",
                "Daily Return": f"{self.garch_res.params.get('Const', 0.0)/100:.6f}",
                "Annualized Return": f"{(self.garch_res.params.get('Const', 0.0) * 2.52):.2f}%",
                "Daily Volatility": f"{uncond_vol:.6f}",
                "Annualized Volatility": f"{(uncond_vol * np.sqrt(252) * 100):.2f}%",
                "Expected Duration (Days)": "N/A"
            }])

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
                "Daily Volatility": f"{sigma:.6f}",
                "Annualized Volatility": f"{(sigma * np.sqrt(252) * 100):.2f}%",
                "Expected Duration (Days)": f"{duration:.2f}"
            })
        return pd.DataFrame(rows)

    def get_transition_matrix(self) -> pd.DataFrame:
        if self.use_garch_fallback or self.res is None:
            return pd.DataFrame({"Status": ["Pivoted to GARCH Single-State Engine"]}, index=["Fixed"])
        matrix = self.res.regime_transition
        if isinstance(matrix, np.ndarray) and matrix.ndim == 3:
            matrix = matrix[:, :, 0]
        labels = [f"State {i}" for i in range(self.k_regimes)]
        return pd.DataFrame(matrix, columns=[f"To {l}" for l in labels], index=[f"From {l}" for l in labels])

    def get_model_comparison(self) -> pd.DataFrame:
        if self.res is None:
            return pd.DataFrame()
        ms_loglik = float(self.res.llf)
        ms_aic = float(self.res.aic)
        ms_bic = float(self.res.bic)
        n = len(self.data)
        scale_adj = n * np.log(100)
        garch_loglik = float(self.garch_res.loglikelihood + scale_adj) if self.garch_res else np.nan
        garch_aic = float(self.garch_res.aic - 2 * scale_adj) if self.garch_res else np.nan
        garch_bic = float(self.garch_res.bic - 2 * scale_adj) if self.garch_res else np.nan
        p, q = self.garch_order
        return pd.DataFrame({
            "Metric": ["Log-Likelihood", "AIC", "BIC", "Framework Active"],
            "Markov Switching": [f"{ms_loglik:.2f}", f"{ms_aic:.2f}", f"{ms_bic:.2f}", "False" if self.use_garch_fallback else "True"],
            f"GARCH({p},{q})": [f"{garch_loglik:.2f}", f"{garch_aic:.2f}", f"{garch_bic:.2f}", "True" if self.use_garch_fallback else "False"],
            "Selection": ["GARCH" if self.use_garch_fallback else "Markov"] * 4
        })

    def get_blended_forecast(self) -> Dict[str, Any]:
        if self.use_garch_fallback and self.garch_res is not None:
            forecasts = self.garch_res.forecast(horizon=1)
            pred_var_scaled = float(forecasts.variance.iloc[-1, 0])
            blended_variance = pred_var_scaled / 10000
            blended_mean = float(forecasts.mean.iloc[-1, 0]) / 100
            blended_volatility = np.sqrt(blended_variance)
            
            # Extract and rank the conditional variance timeline to prevent saturation limits
            hist_vols = self.garch_res.conditional_volatility / 100
            if len(hist_vols) > 0:
                pct_rank = float(np.sum(hist_vols <= blended_volatility) / len(hist_vols))
            else:
                pct_rank = 0.5
                
            return {
                # Format the shape as a pseudo-two-state matrix vector for seamless downstream compatibility
                "forward_probabilities": np.array([1.0 - pct_rank, pct_rank]),
                "blended_mean_daily": blended_mean,
                "blended_vol_daily": blended_volatility,
                "blended_vol_annualized": blended_volatility * np.sqrt(252) * 100
            }

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
        blended_variance = float(np.sum(forward_probs * variances) + np.sum(forward_probs * ((means - blended_mean) ** 2)))
        blended_volatility = np.sqrt(blended_variance)
        return {
            "forward_probabilities": forward_probs,
            "blended_mean_daily": blended_mean,
            "blended_vol_daily": blended_volatility,
            "blended_vol_annualized": blended_volatility * np.sqrt(252) * 100
        }

    def generate_regime_plot(self) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, 
            subplot_titles=("Daily Asset Log Returns", "Volatility Tracking Engine Subsystem")
        )
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data.values, mode="lines", line=dict(color="#2D3748", width=0.8)), row=1, col=1)
        
        if self.use_garch_fallback and self.garch_res is not None:
            vol_series = self.garch_res.conditional_volatility / 100
            fig.add_trace(
                go.Scatter(x=vol_series.index, y=vol_series.values, mode="lines", line=dict(color="#E53E3E", width=1.1), fill="tozeroy", fillcolor="rgba(229, 62, 62, 0.08)"),
                row=2, col=1
            )
            title_text = "GARCH Time-Varying Conditional Volatility Profile (Daily)"
        else:
            prob_series = self.res.smoothed_marginal_probabilities[self.k_regimes - 1]
            fig.add_trace(
                go.Scatter(x=prob_series.index, y=prob_series.values, mode="lines", line=dict(color="#3182CE", width=1.0), fill="tozeroy", fillcolor="rgba(49, 130, 206, 0.12)"),
                row=2, col=1
            )
            title_text = "Markov High Volatility Regime Probability Track"

        fig.layout.annotations[1].update(text=title_text)
        
        # Lock hover interactions so crosshair line displays across subplots simultaneously
        fig.update_layout(
            template="plotly_white", height=400, showlegend=False, 
            margin=dict(l=40, r=20, t=30, b=30), hovermode="x unified"
        )
        fig.update_traces(xaxis="x")
        return fig

    def generate_contagion_plot(self) -> go.Figure:
        horizons = 10
        days = np.arange(horizons)
        response = np.zeros(horizons)
        
        forecast = self.get_blended_forecast()
        shock = forecast.get("blended_vol_daily", 0.02)
        response[0] = shock

        if self.use_garch_fallback and self.garch_res is not None:
            alpha = self.garch_res.params.get("alpha[1]", 0.1)
            beta = self.garch_res.params.get("beta[1]", 0.8)
            persistence = max(0.05, min(0.99, alpha + beta))
        else:
            if self.res is not None:
                p = self.res.regime_transition
                if isinstance(p, np.ndarray) and p.ndim == 3:
                    p = p[:, :, 0]
                persistence = max(0.05, min(0.99, p[0, 0] + p[1, 1] - 1.0))
            else:
                persistence = 0.1

        for h in range(1, horizons):
            response[h] = response[h-1] * persistence
            
        upper_band = response + 1.96 * shock * (persistence ** days) * 0.5
        lower_band = response - 1.96 * shock * (persistence ** days) * 0.5
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.concatenate([days, days[::-1]]), y=np.concatenate([upper_band, lower_band[::-1]]), fill='toself', fillcolor='rgba(147, 51, 234, 0.08)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
        fig.add_trace(go.Scatter(x=days, y=response, mode='lines+markers', line=dict(color='#A855F7', width=1.8), marker=dict(size=5, color='#A855F7')))
        fig.add_shape(type="line", x0=0, y0=0, x1=horizons-1, y1=0, line=dict(color="#4A5568", width=1, dash="dash"))
        
        engine_lbl = "GARCH" if self.use_garch_fallback else "Markov"
        fig.update_layout(
            title=dict(text=f"KvD Contagion ({engine_lbl}): {self.ticker}", font=dict(size=11, color="#2D3748")),
            template="plotly_white", height=240,
            xaxis=dict(title="Days Post-Shock", title_font=dict(size=9), tickmode="linear", tick0=0, dtick=1),
            yaxis=dict(title="Response", title_font=dict(size=9)), margin=dict(l=40, r=20, t=35, b=35), showlegend=False
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
        engine_lbl = "GARCH" if self.use_garch_fallback else "Markov"
        titles = [f"{engine_lbl} Residual ACF", f"{engine_lbl} Residual PACF", f"{engine_lbl} Squared Residual ACF"]

        plots = []
        for vals, title in zip([acf_vals, pacf_vals, sq_acf_vals], titles):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=lags, y=vals, marker_color="#4A5568", width=0.5))
            fig.add_shape(type="line", x0=0, y0=c_bound, x1=nlags, y1=c_bound, line=dict(color="#E53E3E", width=1, dash="dash"))
            fig.add_shape(type="line", x0=0, y0=-c_bound, x1=nlags, y1=-c_bound, line=dict(color="#E53E3E", width=1, dash="dash"))
            fig.update_layout(
                title=dict(text=title, font=dict(size=11, color="#2D3748")),
                template="plotly_white", height=200, xaxis=dict(tickmode="linear", tick0=0, dtick=5),
                yaxis=dict(range=[-0.5, 1.0]), margin=dict(l=40, r=20, t=40, b=30)
            )
            plots.append(fig)
        return plots


class CrossAssetAnalytics:

    @staticmethod
    def compute_regime_sync(analyzers: Dict[str, RegimeAnalyzer]) -> pd.DataFrame:
        series_dict = {}
        for ticker, analyzer in analyzers.items():
            if analyzer.use_garch_fallback and analyzer.garch_res is not None:
                series_dict[ticker] = analyzer.garch_res.conditional_volatility
            elif analyzer.res is not None:
                series_dict[ticker] = analyzer.res.smoothed_marginal_probabilities[analyzer.k_regimes - 1]
        if len(series_dict) < 2:
            return pd.DataFrame()
        return pd.DataFrame(series_dict).corr()