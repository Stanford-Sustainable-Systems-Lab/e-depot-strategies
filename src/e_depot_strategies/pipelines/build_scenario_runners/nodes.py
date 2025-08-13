"""
This is a boilerplate pipeline 'build_scenario_runners'
generated using Kedro 0.18.13
"""

import copy
from itertools import combinations_with_replacement
from pathlib import Path

from .manage_scenarios import generate_configs, get_random_seed


def generate_scenario_configs(params: dict) -> dict:
    """Call the appropriate scenario configuration builder.

    This function is meant to be called from a kedro pipeline directly.

    Args:
        params: the whole "parameters" input dictionary, usually passed directly from
            a kedro pipeline input.

    Returns: The dictionary of scenario configuration partitions.
    """
    scen_name = params["scenario_params"]["name"]
    if scen_name == "single_vocation":
        func = build_single_voc_configs
    elif scen_name == "pairwise":
        func = build_pair_configs
    elif scen_name == "quads":
        func = build_quads_configs
    elif scen_name == "quads_big":
        func = build_quads_configs
    else:
        raise NotImplementedError(
            f"Scenario config generator for {scen_name} scenarios not yet implemented."
        )
    parts = func(params)
    return parts


@generate_configs
def build_single_voc_configs(params: dict) -> tuple[list[Path], list[dict]]:
    """Build single vocation scenarios."""
    veh_day_sets = params["veh_day_sets"]
    scen_params = params["scenario_params"]
    scen_name = scen_params["name"]
    paths = []
    scens = []
    vocs = veh_day_sets["voc_fractions"]["fleet_0"].keys()
    for voc in vocs:
        pair_name = f"{voc}__{voc}"
        for scen_id in range(scen_params["n_runs_per_vocation"]):
            vehs = copy.deepcopy(veh_day_sets)

            # Build new veh_day_sets
            vehs["voc_fractions"] = {}
            vehs["voc_fractions"][f"fleet_{0}"] = {
                k: (1 if k == voc else 0) for k in vocs
            }
            vehs["n_samples"] = scen_params["n_samples_per_run"]
            vehs["n_vehs_per_fleet"] = scen_params["n_vehs_per_fleet"]
            vehs["seed"] = get_random_seed(scen_id)

            # Build new groupings for shared-private
            grps = {"private": ["fleet_id"]}

            # Copy in depot_power_grp_cols
            cur_params = {"veh_day_sets": vehs, "depot_power_grp_cols": grps}
            scens.append(cur_params)
            paths.append(Path(scen_name) / pair_name)
    return (paths, scens)


@generate_configs
def build_pair_configs(params: dict) -> tuple[list[Path], list[dict]]:
    """Build pairwise scenarios."""
    veh_day_sets = params["veh_day_sets"]
    scen_params = params["scenario_params"]
    scen_name = scen_params["name"]
    paths = []
    scens = []
    vocs = veh_day_sets["voc_fractions"]["fleet_0"].keys()
    for voc_i, voc_j in combinations_with_replacement(vocs, 2):
        pair_tup = (voc_i, voc_j)
        pair_name = f"{voc_i}__{voc_j}"
        for scen_id in range(scen_params["n_runs_per_pair"]):
            vehs = copy.deepcopy(veh_day_sets)

            # Build new veh_day_sets
            vehs["voc_fractions"] = {}
            for cur_id, cur_voc in enumerate(pair_tup):
                vehs["voc_fractions"][f"fleet_{cur_id}"] = {
                    k: (1 if k == cur_voc else 0) for k in vocs
                }
            vehs["n_samples"] = scen_params["n_samples_per_run"]
            vehs["n_vehs_per_fleet"] = scen_params["n_vehs_per_fleet"]
            vehs["seed"] = get_random_seed(scen_id)

            cur_params = {"veh_day_sets": vehs}
            scens.append(cur_params)
            paths.append(Path(scen_name) / pair_name)
    return (paths, scens)


@generate_configs
def build_quads_configs(params: dict) -> tuple[list[Path], list[dict]]:
    """Build pairwise scenarios."""
    energy_reqd = params["energy_reqd"]
    veh_day_sets = params["veh_day_sets"]
    costs = params["depot_costs"]
    opt_sets = params["optimizer"]
    scen_params = params["scenario_params"]
    scen_name = scen_params["name"]
    paths = []
    scens = []
    vocs = veh_day_sets["voc_fractions"]["fleet_0"].keys()
    for voc_i, voc_j in combinations_with_replacement(vocs, 2):
        pair_tup = (voc_i, voc_j)
        pair_name = f"{voc_i}__{voc_j}"
        for scen_id in range(scen_params["n_runs_per_pair"]):
            ener = copy.deepcopy(energy_reqd)
            vehs = copy.deepcopy(veh_day_sets)
            opts = copy.deepcopy(opt_sets)

            # Build a new energy required seed
            ener["seed"] = get_random_seed(scen_id)

            # Build new veh_day_sets
            vehs["voc_fractions"] = {}
            for cur_id, cur_voc in enumerate(pair_tup):
                vehs["voc_fractions"][f"fleet_{cur_id}"] = {
                    k: (1 if k == cur_voc else 0) for k in vocs
                }
            vehs["n_samples"] = scen_params["n_samples_per_run"]
            vehs["n_vehs_per_fleet"] = scen_params["n_vehs_per_fleet"]
            vehs["seed"] = get_random_seed(scen_id)

            for valet_set in ["with_valet", "no_valet"]:
                cst = copy.deepcopy(costs)
                if valet_set == "no_valet":
                    cst["valet_shifts"]["max_count"] = 0
                    cst["valet_shifts"]["min_count"] = 0
                else:
                    cst["valet_shifts"]["max_count"] = 100
                    cst["valet_shifts"]["min_count"] = 1

                cur_params = {
                    "energy_reqd": ener,
                    "veh_day_sets": vehs,
                    "depot_costs": cst,
                    "optimizer": opts,
                }
                scens.append(cur_params)
                paths.append(
                    Path(scen_name) / pair_name / f"veh_scen_{scen_id}" / valet_set
                )
    return (paths, scens)
