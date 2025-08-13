"""
This is a boilerplate pipeline 'simulate'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calc_energy_required,
    create_depot_profiles,
    sample_veh_days,
    select_electrifiable_veh_days,
    write_scenario_partition,
)


def create_pipeline(**kwargs) -> Pipeline:
    pipe = pipeline(
        [
            node(
                func=calc_energy_required,
                inputs=[
                    "fleetdna_vehicle_days",
                    "energy_consumption",
                    "params:energy_reqd",
                ],
                outputs="vehicle_days_with_energy",
                name="calculate_energy_required",
            ),
            node(
                func=select_electrifiable_veh_days,
                inputs=[
                    "vehicle_days_with_energy",
                    "params:depot_costs",
                    "params:depot_power_options",
                ],
                outputs="vehicle_days_electrifiable",
                name="select_electrifiable_veh_days",
            ),
            node(
                func=sample_veh_days,
                inputs=["vehicle_days_electrifiable", "params:veh_day_sets"],
                outputs="selected_vehicle_days",
                name="sample_vehicle_days",
            ),
            node(
                func=write_scenario_partition,
                inputs=["selected_vehicle_days", "params:results_partition"],
                outputs="selected_vehicle_days_part",
                name="write_scenario_partition_vehicle_days",
            ),
            node(
                func=create_depot_profiles,
                inputs=[
                    "selected_vehicle_days",
                    "params:depot_power_grp_cols",
                    "params:depot_power_options",
                    "params:depot_costs",
                    "params:optimizer",
                    "params:gurobi_license",
                ],
                outputs=["depot_profiles", "depot_summaries"],
                name="create_depot_profiles",
            ),
            node(
                func=write_scenario_partition,
                inputs=["depot_summaries", "params:results_partition"],
                outputs="depot_summaries_part",
                name="write_scenario_partition_summaries",
            ),
            node(
                func=write_scenario_partition,
                inputs=["depot_profiles", "params:results_partition"],
                outputs="depot_profiles_part",
                name="write_scenario_partition_profiles",
            ),
        ],
        tags="simulate",
    )
    return pipe
