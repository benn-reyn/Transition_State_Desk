import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import pickle
from msvar_model import MSIAHVAR
from msgarch_model import WeightedGARCH

# Data fetching 
def fetch_data(start_date="2015-01-01"):
    tickers = ["SPY", "TLT", "^VIX"]
    raw = yf.download(tickers, start=start_date, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        for field in ['Adj Close', 'Adj. Close', 'Close']:
            if field in raw.columns.get_level_values(0):
                data_yf = raw.xs(field, axis=1, level=0)
                break
            elif field in raw.columns.get_level_values(1):
                data_yf = raw.xs(field, axis=1, level=1)
                break
        else:
            data_yf = raw.iloc[:, raw.columns.get_level_values(1) == 'Close']
    else:
        data_yf = raw.copy()
    data_yf = data_yf.dropna()
    log_rets = np.log(data_yf / data_yf.shift(1)).dropna()
    log_rets.columns = ["SPY", "TLT", "VIX"]

    # ----- FRED: Term spread (10Y - 2Y) -----
    d10y = web.DataReader("DGS10", "fred", start_date)  
    d2y  = web.DataReader("DGS2",  "fred", start_date)  
    # Convert from raw percentage units to decimal
    term_spread = ((d10y["DGS10"] - d2y["DGS2"]).dropna()) / 100
    term_spread.name = "TermSpread"

    # ----- FRED: Credit spread (BAA - AAA) -----
    baa = web.DataReader("BAA10Y", "fred", start_date)  
    aaa = web.DataReader("AAA10Y", "fred", start_date)  
    # Convert from raw percentage units to decimal units
    credit_spread = ((baa["BAA10Y"] - aaa["AAA10Y"]).dropna()) / 100
    credit_spread.name = "CreditSpread"

    # allign 
    spreads = pd.concat([term_spread, credit_spread], axis=1).dropna()
    common_idx = log_rets.index.intersection(spreads.index)
    log_rets = log_rets.loc[common_idx]
    spreads = spreads.loc[common_idx]

    # Merge
    df = pd.concat([log_rets, spreads], axis=1)
    df = df.dropna()
    return df

def fit_models(data):
    print("Fitting MSIAH‑VAR...")
    msvar = MSIAHVAR(k_regimes=2)
    msvar.fit(data, max_iter=50, tol=1e-4, shrinkage=0.05)


    print("Fitting independent conditional variance tracks...")
    # Extract matrices for current (Y) and lagged (X) variables from the VAR framework
    Y = data.values[1:]
    X = data.values[:-1]
    T = Y.shape[0]
    
    residuals = {}
    garch_models = {}
    
    for i, name in enumerate(data.columns):
        resid_calm = Y[:, i] - (msvar.intercepts[0][i] + X @ msvar.coefs[0].T[:, i])
        resid_bear = Y[:, i] - (msvar.intercepts[1][i] + X @ msvar.coefs[1].T[:, i])

        residuals[name] = msvar.smoothed_probs[:, 0] * resid_calm + msvar.smoothed_probs[:, 1] * resid_bear
        

        g = WeightedGARCH()

        g.fit(resid_calm, resid_bear, msvar.smoothed_probs)
        garch_models[name] = g
        
    return msvar, garch_models, residuals


def save_state(msvar, garch_models, residuals, data, filepath="model_state.pkl"):
    state = {
        "msvar": msvar,
        "garch_models": garch_models,
        "residuals": residuals,
        "data": data,
    }
    with open(filepath, "wb") as f:
        pickle.dump(state, f)
    print(f"State saved to {filepath}")


if __name__ == "__main__":
    START_DATE = "2015-01-01"
    print("Fetching data (Yahoo Finance + FRED)...")
    data = fetch_data(START_DATE)
    print(f"Data shape: {data.shape}  Columns: {list(data.columns)}")
    msvar, garch, residuals = fit_models(data)
    save_state(msvar, garch, residuals, data)
    print("Daily update complete.")