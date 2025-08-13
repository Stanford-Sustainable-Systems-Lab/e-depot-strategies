"""
This is a boilerplate pipeline 'summarize'
generated using Kedro 0.18.13
"""

from kedro.pipeline import Pipeline, node, pipeline

from e_depot_strategies.pipelines.build_scenario_runners.manage_scenarios import (
    collate_scenario_partitions,
)

from .nodes import (
    build_contrast_table,
    calculate_omitted_costs,
    eliminate_infeasible_scenarios,
    plot_many_counterfact_profiles,
    set_reporting_groups,
)


def create_pipeline(**kwargs) -> Pipeline:
    ingest_pipe = pipeline(
        [
            node(
                func=collate_scenario_partitions,
                inputs=[
                    "contrast_part",
                    "params:summarize_partitions",
                ],
                outputs="summarized",
                name="collate_scenario_partitions",
            ),
            node(
                func=set_reporting_groups,
                inputs="summarized",
                outputs="summarized_grouped",
                name="set_reporting_groups",
            ),
        ]
    )
    ingest_profiles = pipeline(
        pipe=ingest_pipe,
        inputs={"contrast_part": "depot_profiles_part"},
        parameters="params:summarize_partitions",
        namespace="profiles",
    )
    ingest_summaries = pipeline(
        pipe=ingest_pipe,
        inputs={"contrast_part": "depot_summaries_part"},
        parameters="params:summarize_partitions",
        namespace="summaries",
    )

    contrast_pipe = pipeline(
        [
            node(
                func=eliminate_infeasible_scenarios,
                inputs=[
                    "summaries.summarized_grouped",
                    "profiles.summarized_grouped",
                ],
                outputs=["summaries_optimized", "profiles_optimized"],
                name="eliminate_infeasible_scenarios",
            ),
            node(
                func=calculate_omitted_costs,
                inputs=[
                    "summaries_optimized",
                    "params:depot_costs",
                ],
                outputs="summaries_with_costs",
                name="calculate_omitted_costs",
            ),
            node(
                func=build_contrast_table,
                inputs=[
                    "summaries_with_costs",
                    "profiles_optimized",
                    "params:contrast",
                ],
                outputs="scenario_contrasts",
                name="build_contrast_table",
            ),
        ]
    )

    plot_pipe = pipeline(
        [
            node(
                func=plot_many_counterfact_profiles,
                inputs=[
                    "profiles.summarized_grouped",
                    "params:summarize_partitions",
                ],
                outputs="counterfact_profiles_part",
                name="plot_many_counterfact_profiles",
            ),
        ],
        tags="summarize",
    )
    return plot_pipe + ingest_profiles + ingest_summaries + contrast_pipe
