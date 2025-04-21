import random
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 🔐 Global seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_multi_boosting_models(df, feature_cols, target_cols):
    df = df.dropna(subset=feature_cols + target_cols)
    X = df[feature_cols]
    models = {}
    performance = {}

    for target_col in target_cols:
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.03)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5  
        r2 = r2_score(y_test, y_pred)

        models[target_col] = model
        performance[target_col] = {"RMSE": rmse, "R2": r2}

    return models, performance


def monte_carlo_predict(model, X, iterations=100, return_samples=False):
    preds = [model.predict(X)[0] + np.random.normal(0, 0.01) for _ in range(iterations)]
    samples = np.array(preds)
    if return_samples:
        return {
            "mean": samples.mean(),
            "std": samples.std(),
            "percentiles": np.percentile(samples, [5, 25, 50, 75, 95]),
            "samples": samples
        }
    else:
        return {
            "mean": samples.mean(),
            "std": samples.std(),
            "percentiles": np.percentile(samples, [5, 25, 50, 75, 95])
        }

