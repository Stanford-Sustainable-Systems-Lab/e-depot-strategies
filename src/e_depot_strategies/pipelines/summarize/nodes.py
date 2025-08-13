"""
This is a boilerplate pipeline 'summarize'
generated using Kedro 0.18.13
"""

import re
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from e_depot_strategies.pipelines.simulate.depot_power import present_to_repeating


def collate_scenario_partitions(parts: dict, params: dict) -> pd.DataFrame:
    """Read in dataframes from select partitions and concatenate."""
    df_ls = []
    key_ls = []
    for part_key, part_load_func in parts.items():
        key_split = Path(part_key).parts
        if key_split[0] in params["include_partitions"]:
            cur_key = list(key_split)
            key_ls.append(tuple(cur_key))
            df_cur = part_load_func().reset_index()
            df_ls.append(df_cur)
    coll = pd.concat(df_ls, keys=key_ls, names=params["level_names"])
    coll.index = coll.index.droplevel(-1)
    return coll


def set_reporting_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Process the partition keys and counterfactuals into grouping columns."""
    df = df.reset_index()
    df[["voc_category_i", "voc_category_j"]] = df["voc_set"].str.split(
        "__", expand=True
    )
    df["scen_id"] = (
        df["scen_task_key"].str.extract(r"(?<=task_)(\d+)").astype(int)
    )  # Note opposite naming convention, because "task_id" actually gives unique scenario IDs.

    # Correction for temporary combination of single_vocation and pairwise scenarios
    sing_voc_rows = df["run_id"] == "single_vocation_paired"
    df.loc[sing_voc_rows, "scen_id"] = (
        df.loc[sing_voc_rows, "scen_id"] + df.loc[~sing_voc_rows, "scen_id"].max() + 1
    )

    # Back to normal flow
    df = df.drop(columns=["voc_set", "scen_task_key", "run_id"])
    df = df.set_index(["voc_category_i", "voc_category_j", "scen_id"])

    # Set shared-private dichotomy
    if "counterfact" in df.columns:
        df["shar_priv"] = df["counterfact"].str.replace(r"_[0-9]+", "", regex=True)
        df["shar_priv"] = df["shar_priv"].str.replace(r"_fleet", "", regex=True)
    return df


def eliminate_infeasible_scenarios(
    summs: pd.DataFrame, profs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eliminate from summary the scenarios which had infeasible optimizations."""
    broken_rows = summs.loc[summs["problem_status"] == "infeasible_or_unbounded"]
    broken_scens = list(broken_rows.index.get_level_values("scen_id").unique())
    summs = summs.drop(index=broken_scens, level="scen_id")
    profs = profs.drop(index=broken_scens, level="scen_id")
    return (summs, profs)


def calculate_omitted_costs(summs: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Calculate costs which may be omitted from some scenario versions."""
    # Annualized upfront costs
    f = partial(
        present_to_repeating,
        i_ann_pct=params["discount_rate_pct"],
        n_years=params["time_horizon_yrs"],
        freq="1D",
    )
    replace_idx = summs["cost_plugs_dollars"].isna()
    replace_vals = summs.loc[replace_idx, "present_value_plugs_dollars"].transform(f)
    summs.loc[replace_idx, "cost_plugs_dollars"] = replace_vals

    # Total cost
    cost_cols = list(filter(lambda x: re.compile(r"^cost_").match(x), summs.columns))
    summs["total_cost_dollars"] = summs[cost_cols].sum(axis=1)
    return summs


def build_contrast_table(
    summs: pd.DataFrame, profs: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Build DataFrame which contrasts between shared and private outcomes."""
    contrast = summs.groupby(params["group_cols"]).agg("sum")
    contrast = contrast.select_dtypes(exclude=["object"])

    # Calculate derived measurements
    contrast["potential_energy_kwh"] = contrast["depot_capacity_kw"] * 24
    contrast["utilization"] = (
        contrast["energy_delivered_kwh"] / contrast["potential_energy_kwh"]
    )
    contrast["cost_dollars_per_kwh"] = (
        contrast["total_cost_dollars"] / contrast["energy_delivered_kwh"]
    )

    # Melt for easier plotting
    contrast = contrast.melt(ignore_index=False)
    contrast = contrast.set_index("variable", append=True)
    contrast = contrast.unstack("shar_priv")
    contrast = contrast.droplevel(0, "columns")
    contrast["diff"] = contrast["shared"] - contrast["private"]
    contrast["pct_diff"] = contrast["diff"] / contrast["private"] * 100
    contrast.loc[contrast["diff"] == 0, "pct_diff"] = 0
    contrast = contrast.unstack("variable")
    return contrast


def plot_many_counterfact_profiles(
    profs: pd.DataFrame, params: dict
) -> matplotlib.figure.Figure:
    """Plot the load profiles for each depot combination."""
    fig_parts = {}
    for id, grp in tqdm(profs.groupby(params["level_names"])):
        fig_parts[str(Path(*id))] = plot_counterfact_profiles(grp)
    return fig_parts


def plot_counterfact_profiles(profs: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot the individual vehicle load profiles grouped by counterfactual depot."""
    profs = profs.reset_index()
    profs["veh_day_id"] = pd.Categorical(profs["veh_day_id"])
    g = sns.relplot(
        data=profs,
        x="time",
        y="avg_power_kw",
        hue="veh_day_id",
        col="counterfact",
        kind="line",
    )
    g.set_titles("Counterfactual = {col_name}")
    g.set_axis_labels("Hour of Day", "Vehicle Power [kW]")
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%I%p"))
    return g.figure
