# Process FleetDNA Data for modeling
# by Fletcher Passow

import numpy as np
import pandas as pd


def clean_composite(composite: pd.DataFrame, vocations: pd.DataFrame) -> pd.DataFrame:
    """Clean composite data for use by other preprocessing modules."""
    fleet_df = composite.rename(columns={"did": "deploy_id"})
    fleet_df["veh_day_id"] = fleet_df.index.get_level_values(0)
    fleet_df = fleet_df.merge(vocations, on="voc_id")
    fleet_df["voc_id_fac"] = fleet_df["voc_id"].astype("category").cat.codes
    fleet_df["voc_cat_fac"] = fleet_df["voc_category"].astype("category").cat.codes
    return fleet_df


def get_distances(fleet: pd.DataFrame) -> pd.DataFrame:
    """Get distances traveled by vehicle-day."""
    cols = fleet.columns
    keep_cols = [s for s in cols if "id" in s]  # for truck-day ids
    keep_cols += ["distance_total", "voc_category", "voc_cat_fac"]
    dist_df = fleet.loc[:, keep_cols]
    dist_df = dist_df.rename(
        columns={"distance_total": "distance_total_mi"}
    )  # FleetDNA documentation tells us units are miles
    return dist_df


def get_midshift_stops(fleet: pd.DataFrame) -> pd.DataFrame:
    """Get midshift stops."""
    cols = fleet.columns
    keep_cols = [s for s in cols if "id" in s]  # for truck-day ids
    keep_cols += ["total_stops"]
    keep_cols += [
        s for s in cols if "stop_duration" in s
    ]  # for descriptive statistics of stops
    keep_cols += [s for s in cols if "_hrs" in s]  # for time totals by time type
    keep_cols += [
        s for s in cols if "_ts" in s
    ]  # for day start and end timestamps in local time
    keep_cols += [
        s for s in cols if "_rts" in s
    ]  # for adjustments to start and end timestamps
    keep_cols += [
        s for s in cols if "_count" in s
    ]  # for counts of trips and microtrips
    keep_cols += ["voc_category", "voc_cat_fac"]
    stops_df = fleet.loc[:, keep_cols]

    stops_df["start_ts"] = pd.to_datetime(stops_df["start_ts"], format="mixed")
    stops_df["end_ts"] = pd.to_datetime(stops_df["end_ts"], format="mixed")
    stops_df["total_stops"] = stops_df["total_stops"].astype(int)

    ## Clean data
    # First, eliminate all rows which include stop durations below zero seconds.
    stop_dist_cols = [
        "min_stop_duration",
        "stop_duration_25th_percentile",
        "median_stop_duration",
        "stop_duration_75th_percentile",
        "max_stop_duration",
    ]
    stop_dist_df = stops_df.loc[:, stop_dist_cols]

    neg_dur_idx = stops_df.loc[np.any(stop_dist_df.values < 0, axis=1)].index
    stops_df = stops_df.drop(index=neg_dur_idx)
    return stops_df


def get_times(stops: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Get beginning and ending active times for each vehicle-day."""
    cols = stops.columns
    keep_cols = [s for s in cols if "id" in s]
    keep_cols += [s for s in cols if "_ts" in s]
    keep_cols += [s for s in cols if "_name" in s]
    keep_cols += [s for s in cols if "_cat" in s]
    keep_cols += ["max_stop_duration"]
    times_df = stops.loc[:, keep_cols]

    times_df["start_ts"] = pd.to_datetime(times_df["start_ts"], format="mixed")
    times_df["end_ts"] = pd.to_datetime(times_df["end_ts"], format="mixed")
    times_df["start_time"] = times_df["start_ts"] - times_df["start_ts"].dt.floor("D")
    times_df["end_time"] = times_df["end_ts"] - times_df["end_ts"].dt.floor("D")
    times_df["start_hrs"] = times_df["start_time"].dt.total_seconds() / 3600
    times_df["end_hrs"] = times_df["end_time"].dt.total_seconds() / 3600
    times_df["max_stop_hrs"] = times_df["max_stop_duration"] / 3600

    # Get time differences
    cross_mid = times_df["end_hrs"] < times_df["start_hrs"]
    times_df["hrs_offshift"] = (
        params["total_hrs"] * (1 - cross_mid)
        - times_df["end_hrs"]
        + times_df["start_hrs"]
    )
    times_df["hrs_break"] = times_df["max_stop_hrs"]
    times_df["hrs_active"] = (
        params["total_hrs"] - times_df["hrs_offshift"] - times_df["hrs_break"]
    )
    return times_df
