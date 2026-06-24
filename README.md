# Macro Regime Terminal

An analytical dashboard for tracking asset volatility regimes and cross-asset shocks. This terminal combines a Markov-Switching Interaction Autoregressive Hidden Volatility Autoregressive (MSIAH-VAR) model with regime-conditional GARCH specifications to forecast market states and map closed-form impulse responses.

## System Architecture

___________________________________
│     Macroeconomic Data Input    │
│  (Daily Yahoo Finance & FRED)   │
-----------------------------------
│
___________________________________
│      MSIAH-VAR Regime Model     │
│   (Smooths Calm vs. Bear Prob)  │
-----------------------------------
│
___________________________________
│    Regime-Conditional GARCH     │
│  (Blended Volatility Forecasts) │
-----------------------------------
│
__________________________________
│  Kole & van Dijk (2023) IRFs    │
│  (Closed-Form Shock Spillovers) │
-----------------------------------

### Core Methodology
* **Regime Detection:** A hidden Markov model architecture that endogenously segments the state space into distinct regimes (e.g., *Calm* vs. *High-volatility*) without requiring structural break priors.
* **Volatility Dynamics:** Rather than assuming static variance, individual asset errors are mapped through unique GARCH parameters conditioned on the latently filtered state probabilities.
* **Transmission Framework:** Structural shocks are propagated through the system using the closed-form solutions established by Kole & van Dijk (2023), mapping complete 20-day spillover profiles.

---

## Features

* **Regime & Volatility Monitor:** Real-time visual overlays tracking asset returns alongside smoothed regime probabilities. Generates blended next-day volatility forecasts using current state weights.
* **Shock Transmission Analyzer:** Interactive simulation suite demonstrating how structural innovations to a single origin asset ripple through the covariance matrix over a 20-day horizon, complete with 95% simulation bounds.
* **Diagnostic Suite:**
    * **Ergodicity Verification:** Real-time ADF tests and GARCH persistence bounds ($a + \beta$) to ensure compliance with structural model assumptions.
    * **Leptokurtosis Curves:** Kernel Density Estimation (KDE) mapped against theoretical normals to capture empirical tail risk ($\kappa$).
    * **Residual Quality:** Serial autocorrelation (ACF) tests on structural residuals to confirm white noise conditions.

---

## Technical Stack

* **Core Logic:** Python, NumPy, SciPy, Statsmodels
* **UI Framework:** Streamlit
* **Visualization:** Plotly Dynamic Charts
* **Data Pipeline:** Yahoo Finance API / FRED API

---

## Getting Started

### Prerequisites
Ensure your local Python environment is running version `3.9` or higher.

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Initialize Parameters:
   ```bash
   python update_models.py

4. Run app:
   ```bash
   streamlit run app.py
