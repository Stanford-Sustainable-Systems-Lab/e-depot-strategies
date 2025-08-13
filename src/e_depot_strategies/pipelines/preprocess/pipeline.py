"""
This is a boilerplate pipeline 'preprocess'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    collect_fleetdna_vehicle_days,
    filter_consumptions,
    mark_vehicle_classes,
)
from .process_fleetnda import (
    clean_composite,
    get_distances,
    get_midshift_stops,
    get_times,
)
from .process_role_consumption import calc_energy_consump_rate
from .process_role_energy import label_charging_sessions


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_composite,
                inputs=["fleet_dna_composite", "fleet_dna_vocations"],
                outputs="fleet_vehicle_days",
                name="clean_fleet_dna_composite",
            ),
            node(
                func=get_distances,
                inputs="fleet_vehicle_days",
                outputs="fleetdna_distances",
                name="preprocess_distances",
            ),
            node(
                func=get_midshift_stops,
                inputs="fleet_vehicle_days",
                outputs="fleetdna_stops",
                name="preprocess_midshift_stops",
            ),
            node(
                func=get_times,
                inputs=["fleetdna_stops", "params:fleetdna_times"],
                outputs="fleetdna_times",
                name="preprocess_begin_end_times",
            ),
            node(
                func=label_charging_sessions,
                inputs=["role_energy_readings", "params:charge_sessions"],
                outputs="energy_sessions",
                name="preprocess_energy_sessions",
            ),
            node(
                func=calc_energy_consump_rate,
                inputs=["energy_sessions", "role_gps", "params:consump_rate"],
                outputs="energy_consumption_raw",
                name="preprocess_energy_consumption",
            ),
            node(
                func=mark_vehicle_classes,
                inputs=[
                    "energy_consumption_raw",
                    "role_classes",
                    "params:mark_classes",
                    "params:weight_corresp",
                ],
                outputs="energy_consumption_w_groups",
                name="mark_vehicle_classes",
            ),
            node(
                func=filter_consumptions,
                inputs=["energy_consumption_w_groups", "params:filter_consumps"],
                outputs="energy_consumption",
                name="filter_consumptions",
            ),
            node(
                func=collect_fleetdna_vehicle_days,
                inputs=[
                    "fleetdna_distances",
                    "fleetdna_times",
                    "params:weight_corresp",
                ],
                outputs="fleetdna_vehicle_days",
                name="collect_fleetdna_vehicle_days",
            ),
        ],
        tags="preprocess",
    )
