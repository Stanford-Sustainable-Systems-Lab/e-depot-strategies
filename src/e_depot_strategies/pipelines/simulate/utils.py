import string

import numpy as np
import pandas as pd


def capitalize_keys(d: dict) -> dict:
    """Apply capwords and replace underscores with spaces in dict keys."""
    d_new = {string.capwords(k.replace("_", " ")): v for k, v in d.items()}
    return d_new


def scale_and_round(d: dict[str:float], n: int) -> dict:
    """Scale a dictionary with float keys to a specific integer sum."""
    df = pd.DataFrame.from_dict(d, orient="index", columns=["orig"])
    cum_arr = np.insert(df["orig"].values, 0, 0)
    cum_arr = cum_arr * n
    cum_arr = np.diff(cum_arr.cumsum().round()).astype(int)
    df["rounded"] = cum_arr
    return df["rounded"].to_dict()


def build_df_from_dict(d: dict, id_cols: list[str], value_col: str) -> pd.DataFrame:
    """Build a DataFrame with a multi-index from a multi-level dictionary of uniform depth."""

    def _recurse(d: dict) -> pd.DataFrame:
        vals = list(d.values())
        if isinstance(vals[0], dict):
            dfs = {k: _recurse(v) for k, v in d.items()}
            df = pd.concat(dfs.values(), keys=dfs.keys())
        else:
            if isinstance(vals[0], list):
                d = {
                    k: [np.array(v)] for k, v in d.items()
                }  # Wrapping in lists to create an array column
            df = pd.DataFrame.from_dict(d, orient="index")
        return df

    df = _recurse(d).reset_index()
    df.columns = id_cols + [value_col]
    return df
