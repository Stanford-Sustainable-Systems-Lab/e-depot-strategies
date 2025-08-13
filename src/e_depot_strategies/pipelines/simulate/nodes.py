"""
This is a boilerplate pipeline 'simulate'
generated using Kedro 0.18.13
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from .depot_power import calc_depot_profiles
from .utils import capitalize_keys, scale_and_round


def calc_energy_required(
    veh_days: pd.DataFrame, energy_consump: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """Calculate the energy required for each vehicle day based on distance."""
    veh_days["energy_required_kwh"] = 0.0
    for wgt_cls in veh_days["weight_class_group"].unique():
        vd_sel = veh_days["weight_class_group"] == wgt_cls
        cn_sel = energy_consump["weight_class_group"] == wgt_cls
        consump_arr = (
            energy_consump.loc[cn_sel, "consump_kwh_per_mi"]
            .sample(
                n=veh_days.loc[vd_sel, :].shape[0],
                replace=True,
                random_state=params["seed"],
            )
            .values
        )
        veh_days.loc[vd_sel, "energy_required_kwh"] = (
            veh_days.loc[vd_sel, "distance_total_mi"] * consump_arr
        )
    return veh_days


def select_electrifiable_veh_days(
    veh_days: pd.DataFrame,
    costs: dict,
    sim_opts: dict,
) -> pd.DataFrame:
    """Select only electrifiable vehicle days for further use."""
    fastest_charge = max(list(costs["plug_power_kw"].values()))
    d_t = pd.Timedelta(sim_opts["timestep_freq"]) / pd.Timedelta("1H")
    hrs_avail = veh_days["hrs_offshift"] - np.mod(veh_days["hrs_offshift"], d_t)
    electrifiable = (hrs_avail * fastest_charge) > veh_days["energy_required_kwh"]
    selected = veh_days.loc[electrifiable, :]
    return selected


def sample_veh_days(
    veh_days: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Sample vehicle days from the observed: distance driven, begin time, end time."""
    # Get absolute numbers of vehicles for each fleet
    voc_nums = {}
    for fleet_id, voc_fracs in params["voc_fractions"].items():
        voc_nums[fleet_id] = scale_and_round(voc_fracs, params["n_vehs_per_fleet"])
        voc_nums[fleet_id] = capitalize_keys(voc_nums[fleet_id])

    # Set up frame and number of samples dataframes with indexing
    veh_days = veh_days.set_index("veh_day_id")
    veh_day_grp = veh_days.groupby(params["grp_cols"])

    rng = np.random.default_rng(seed=params["seed"])
    scens = {}
    for i in range(params["n_samples"]):
        for fleet_id, voc_nums in voc_nums.items():
            sample_df = veh_day_grp.apply(
                lambda grp: grp.sample(
                    n=voc_nums[grp.name], replace=True, random_state=rng
                )
            )
            scens.update({(i, fleet_id): sample_df})
    all_scens = pd.concat(scens, axis=0, names=["scen_id", "fleet_id"])
    retain_cols = set(all_scens.index.names) - set(params["grp_cols"])
    all_scens = all_scens.reset_index(params["grp_cols"], drop=True)
    all_scens = all_scens.reset_index(list(retain_cols), drop=False)
    all_scens = all_scens.rename(columns={"veh_day_id": "veh_day_id_orig"})
    all_scens["veh_day_id"] = np.arange(all_scens.shape[0])
    return all_scens


def create_depot_profiles(
    veh_days: pd.DataFrame,
    grp_cols: list[str],
    params: dict,
    costs: dict,
    optim: dict,
    license: dict,
) -> pd.DataFrame:
    """Create depot profiles individually for each scenario."""
    profs = {}
    summs = {}
    col_union = set.union(*[set(v) for v in grp_cols.values()])
    labs = {k: None for k in col_union}
    for grouping, cols in grp_cols.items():
        cols_to_fill = col_union - set(cols)
        for k in cols_to_fill:
            labs.update({k: grouping})
        for id, grp in tqdm(veh_days.groupby(cols)):
            for i, k in enumerate(cols):
                labs.update({k: id[i]})
            id_full = tuple(labs.values())
            profs[id_full], summs[id_full] = calc_depot_profiles(
                grp,
                params=params,
                costs=costs,
                optim=optim,
                license=license,
            )
    profs_df = pd.concat(profs.values(), keys=profs.keys(), names=col_union, sort=True)
    summs_df = pd.concat(summs.values(), keys=summs.keys(), names=col_union, sort=True)
    summs_df.index = summs_df.index.droplevel(-1)
    return (profs_df, summs_df)


def write_scenario_partition(df: pd.DataFrame, params: dict) -> dict:
    """Write out the partition for this scenario and dataset."""
    return {params["dir"]: df}
