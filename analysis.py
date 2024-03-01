"""
File: analysis.py
Author: Chuncheng Zhang
Date: 2024-02-29
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis the data

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""

# %% ---- 2024-02-29 ------------------------
# Requirements and constants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from rich import print

from sklearn import metrics
from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
    TheilSenRegressor,
)


# %% ---- 2024-02-29 ------------------------
# Function and class
def load_data(path: Path) -> pd.DataFrame:
    path = Path(path)
    assert path.is_file, f"File not found {path}"

    df = pd.read_json(path)
    return df


def analysis_data(df: pd.DataFrame, figpath: Path = None):
    figpath = Path(figpath) if figpath is not None else None

    # --------------------
    cols = df.groupby("name")
    print(cols.first())

    p1 = cols.get_group("p1")[["x", "y"]]
    p2 = cols.get_group("p2")[["x", "y"]]
    p3 = cols.get_group("p3")[["x", "y"]]

    src = np.concatenate([p1.to_numpy(), p3.to_numpy()], axis=1)
    dst = p2.to_numpy()

    # --------------------

    n = int(src.shape[0] / 2)
    train_X = src[:n]
    train_y = dst[:n]
    test_X = src[n:]
    test_y = dst[n:]

    reg = LinearRegression()
    reg.fit(train_X, train_y)
    pred = reg.predict(test_X)
    print(metrics.mean_squared_error(y_true=test_y, y_pred=pred))

    diff = pred - test_y

    # --------------------
    with plt.style.context("ggplot"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax = axes[0]
        for e in ["p1", "p2", "p3"]:
            ax.scatter(x=cols.get_group(e)["x"], y=cols.get_group(e)["y"], s=4, label=e)
        ax.set_title("Point positions")
        ax.axis("off")
        ax.legend()

        ax = axes[1]
        im = ax.imshow(diff, aspect=diff.shape[1] / diff.shape[0])
        ax.set_title("Regression error")
        ax.axis("off")
        plt.colorbar(im, shrink=0.5)
        plt.tight_layout()

        if figpath is None:
            plt.show()
        else:
            figpath.parent.mkdir(exist_ok=True, parents=True)
            fig.savefig(figpath)
            print("Saved figure %s" % figpath)

    return


# %%

# %% ---- 2024-02-29 ------------------------
# Play ground
if __name__ == "__main__":
    df = load_data(Path("data/data1.json"))
    analysis_data(df, "data/data1.jpg")
    # analysis_data(df)


# %% ---- 2024-02-29 ------------------------
# Pending


# %% ---- 2024-02-29 ------------------------
# Pending
