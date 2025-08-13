"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.13
"""

import pandas as pd

from e_depot_strategies.pipelines.simulate.utils import build_df_from_dict


def mark_vehicle_classes(
    consump: pd.DataFrame, classes: pd.DataFrame, params: dict, wpars: dict
) -> pd.DataFrame:
    """Mark the vehicle weight classes of the consumption observations."""
    classes = classes.loc[:, params["keep_cols_classes"]]
    mrg_cols = params["merge_cols"]
    consump = consump.merge(
        classes,
        how="left",
        left_on=mrg_cols["consump"],
        right_on=mrg_cols["classes"],
    )
    wgts = wpars["weight_class_group"]
    wgt_corresp = build_df_from_dict(
        wgts["values"], id_cols=wgts["id_columns"], value_col="weight_class_group"
    )
    consump = consump.merge(wgt_corresp, how="left", on=wgts["id_columns"])
    return consump


def filter_consumptions(consump: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Filter the consumption values to reasonable ranges."""
    orig_cols = consump.columns
    iqrs = consump.groupby(params["grp_cols"])[params["consump_col"]].quantile(
        [0.25, 0.75]
    )
    iqrs = iqrs.unstack(level=1)
    iqrs["iqr"] = iqrs[0.75] - iqrs[0.25]
    iqrs["upper"] = iqrs[0.75] + params["iqr_mult"] * iqrs["iqr"]
    iqrs["lower"] = iqrs[0.25] - params["iqr_mult"] * iqrs["iqr"]
    consump_sel = consump.merge(iqrs, how="left", on=params["grp_cols"])
    keeps = (consump_sel[params["consump_col"]] <= consump_sel["upper"]) & (
        consump_sel[params["consump_col"]] >= consump_sel["lower"]
    )
    consump_sel = consump_sel.loc[keeps, orig_cols]
    return consump_sel


def collect_fleetdna_vehicle_days(
    dist: pd.DataFrame,
    time: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Collect training data for PGM into a single file."""
    merge_cols = ["veh_day_id"]
    time_cols = ["start_hrs", "end_hrs", "hrs_offshift", "hrs_break", "hrs_active"]
    time_select = time.loc[:, merge_cols + time_cols]
    train = dist.merge(time_select, how="inner", on=["veh_day_id"])

    wgts = params["weight_class_group"]
    wgt_corresp = build_df_from_dict(
        wgts["values"], id_cols=wgts["id_columns"], value_col="weight_class_group"
    )
    train = train.merge(wgt_corresp, how="left", on=wgts["id_columns"])

    train_cols = set(train.columns)
    dist_cols = ["distance_total_mi"]
    data_cols = dist_cols + time_cols
    select_cols = list(train_cols.difference(set(data_cols))) + data_cols
    train = train.loc[:, select_cols]
    return train
