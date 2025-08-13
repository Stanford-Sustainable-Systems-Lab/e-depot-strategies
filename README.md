# Electrified Depot Strategies

This repository contains the code required for replicating the analysis from *Charging electrified commercial vehicle fleets with reduced grid capacity using low-capital-cost depot management strategies* by Fletcher Passow and Ram Rajagopal. DOI: [10.1016/j.apenergy.2025.126563](https://doi.org/10.1016/j.apenergy.2025.126563)

More broadly, this code can help inform fleet and depot operators as they plan for electrification.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
  - [Running Kedro Pipelines](#running-kedro-pipelines)
  - [Visualizing Pipelines with Kedro Viz](#visualizing-pipelines-with-kedro-viz)
  - [Pipeline Descriptions](#pipeline-descriptions)
- [Notebooks](#notebooks)
- [Replicating the Original Paper](#replicating-the-original-paper)
- [License](#license)

## Project Structure

The project is structured according to the system used by the Kedro framework to create
production-ready data science code. Please take a look at the excellent
[Kedro documentation](https://docs.kedro.org) for more information. We encourage you to
read through their [Introduction to Kedro](https://docs.kedro.org/en/1.0.0/getting-started/course/)
to get familiar with many of the features that we use in this project.

More specifically, our project has the following structure:

```text
pyproject.toml
uv.lock
conf/           # Configuration files (parameters, logging, credentials), see Kedro for how these are used
data/           # Data folders (raw, intermediate, processed, reporting), see Kedro for how these are used
docs/           # Documentation sources
notebooks/      # Jupyter notebooks for post-simulation analysis and exploration
src/            # Source code (Python modules) for data preprocessing and simulation
slurm_logs/     # Logs from SLURM jobs (if applicable)
```

## Installation

1. Clone the repository:

  ```sh
  git clone https://github.com/Stanford-Sustainable-Systems-Lab/e-depot-strategies.git
  cd e-depot-strategies
  ```

2. Install dependencies using [uv](https://github.com/astral-sh/uv):

  ```sh
  uv sync
  ```

3. Acquire a license for the Gurobi mathematical programming solver. Free (educational) or commercial Gurobi WLS Licenses can be obtained [here](https://www.gurobi.com). Download the license file (gurobi.lic) and place it in the `conf/base/credentials/` subdirectory, which you will need to create. Re-format it into a YAML file as other Kedro parameters would be. The name of the file does not matter.

```text
gurobi_license:
  WLSACCESSID: <your access id key>
  WLSSECRET: <your secret>
  LICENSEID: <your license id>
```

## Data

Please see the [original paper](https://doi.org/10.1016/j.apenergy.2025.126563) for 
information on how to download and/or request the raw data files which this work is
based on. Once you have the raw data:

- Place FleetDNA data in `data/01_raw/FleetDNA`
- Place Run on Less - Electric (RoL-E) data in `data/01_raw/ROLE`
- Processed and intermediate data are stored in subsequent subfolders.

## Usage

### Running Kedro Pipelines

It is recommended to start by running the `preprocess` and `simulate` pipelines to get a
feel for the functionality of the code.

To run a pipeline (e.g. preprocess, simulate), use:

```sh
uv run kedro run --pipeline <pipeline_name>
```

You can also specify configuration or parameters as needed. See the Kedro documentation for more options.

### Visualizing Pipelines with Kedro Viz

To visualize the project’s pipelines interactively using Kedro Viz:

```sh
uv run kedro viz run
```

This will launch a local web server and open the Kedro Viz interface in your browser.
There, you can explore the pipeline structure, nodes, and data dependencies visually.

### Pipeline Descriptions

- **preprocess**:  
  Cleans, transforms, and prepares raw data for simulation. This includes data validation, feature engineering, and formatting to ensure consistency and readiness for downstream analysis.

- **build_scenario_runners**:
  Generates configuration files and shell scripts for use in running multiple scenarios. This takes the "overriding" parameters from a subdirectory of `conf/scenario_runners/` and uses them to build a set of configuration files in `conf/scenarios`. To select which set of scenarios you want to set up, use the `--env` CLI argument when calling this pipeline using Kedro:

  ```sh
  uv run kedro run --pipeline build_scenario_runners --env <scenario_set_name>
  ```

  For example, to set up the `quads` scenarios for sets of *four* strategy scenarios on 5-vehicle charging depots, you would use `--env quads`.By doing this, you will generate a set of configurations in the `conf/scenarios/quads` directory **and** a `bash` script which will run that set of scenarios in parallel on a SLURM cluster, if you have access to one. For more information on how to set up multiple scenarios to run on a local machine, please see the comments and docstrings of the `build_scenario_runners` pipeline itself. 

- **simulate**:  
  Runs scenario-based simulations using the preprocessed data. This pipeline models depot operations, vehicle schedules, and other system behaviors under sets of strategy scenarios. By default, it will produce results in the `test` scenario set, but if you call it overriding the `results_partition` parameter, you can redirect the outputs to other scenario sets.

- **summarize**:  
  This pipeline contains convenience functions for summarizing results, and is currently not in use directly as a Kedro pipeline.

## Notebooks

Run the Jupyter notebooks included in the `notebooks` subdirectory to re-generate the figures from the original paper.

## Replicating the Original Paper

To replicate the results in the original paper, follow these steps:

1. Download and/or request the original data (see [data](#data) for more information).
2. Run the `preprocess` pipeline.
3. Create the configurations for the necessary scenario sets by running the `build_scenario_runners` pipeline:
    - `quads` for 5-vehicle charging depots
    - `quads_big` for 15-vehicle charging depots
4. Run these scenario sets. This will call the `simulate` pipeline many times with different parameters specified in the different configuration files generated in the previous step. Especially, for `quads_big`, it is recommended to use a computer with more resources than the typical laptop. See the `conf/scenario_runners/quads_big` subdirectory for parameters describing the resources used for the original paper.
5. Run the analysis Jupyter notebooks to re-generate the paper figures.

## License

MIT License Copyright (c) Fletcher Passow 2025

## Acknowledgements

This work was supported by the [Stanford Precourt Institute for Energy](https://energy.stanford.edu)'s Bits & Watts Program.
