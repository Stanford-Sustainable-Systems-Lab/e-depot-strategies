import string
from functools import partial
from itertools import combinations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


def write_outputs(power: pd.Series, plugs: pd.Series, vehs: pd.Series):
    """Show summaries of the model run in terms of operational metrics like
    utilization.

    The following metrics will be output:
        - Power demand over typical day plot
        - Load duration curve for typical day plot
        - Utilization metric of depot
        - Wasserstein distance from single depot to uniform distribution

    Args:
        power:
            series of power in units of kW, with indices for vehicle day
            and time of day.

    Returns:
        Nothing, but saves all summaries as a side effect.
    """
    n_scens = power["scen_id"].nunique()

    entropy_df = calc_vocation_entropy(
        vehs, grp_cols=["scen_id"]
    )  # Once per scenario, based on the shared scenario (could be expanded to sub-scenario level later)
    entropy_df = entropy_df.to_frame()
    entropy_df["counterfact"] = "shared"
    util_df = calc_utilization(
        power, plugs, grp_cols=["scen_id", "counterfact"]
    )  # Once per sub-scenario
    improve_df = calc_utilization_change(util_df, grp_cols=["scen_id"])
    improve_df["counterfact"] = "shared"
    scen_df = util_df.merge(entropy_df, on=["scen_id", "counterfact"], how="left")
    scen_df = scen_df.merge(improve_df, on=["scen_id", "counterfact"], how="left")

    if n_scens == 1:
        plot_load_profile(power, grp_cols=["counterfact"])  # Once per sub-scenario
        plot_load_duration(power, grp_cols=["counterfact"])  # Once per sub-scenario
    else:
        scen_shared_df = scen_df.loc[scen_df["counterfact"] == "shared", :]
        plot_util_vs_entropy(
            util=scen_shared_df["utilization"],
            entropy=scen_shared_df["vocation_entropy"],
        )  # Only really useful in a multi-run context, because otherwise this gives just a single point
        plot_util_diff_vs_entropy(
            util_diff=scen_shared_df["mean_util_diff"],
            entropy=scen_shared_df["vocation_entropy"],
            metric_name="Mean",
        )

        counterfact_idx = power.index.get_level_values("counterfact").to_series()
        power_shared = power.loc[counterfact_idx == "shared", :]

        plot_load_profile(
            power_shared, grp_cols=["scen_id"]
        )  # once per scenario only for the shared sub-scenario, but we pre-filter this with the columns staying the same
        plot_load_duration(
            power_shared, grp_cols=["scen_id"]
        )  # once per scenario only for the shared sub-scenario

        reduce_df, reducer = calc_reduced_dims(power_shared)
        plot_components(
            reducer, power_shared.index.get_level_values("time").unique().sort_values()
        )
        plot_reduced_dims(reduce_df)
        plot_reduced_dims(reduce_df, util_df["utilization"])

        reduce_df["counterfact"] = "shared"
        scen_df = scen_df.merge(reduce_df, how="left", on=["scen_id", "counterfact"])

    return scen_df


def calc_scalar_summaries(
    power: pd.DataFrame, scalars: pd.DataFrame, veh_days: pd.DataFrame
) -> pd.DataFrame:
    """Calculate scalar summaries for each scenario."""
    voc_count_ser = veh_days.groupby(["scen_id", "voc_cat_fac"])["veh_day_id"].count()
    entropy_ser = voc_count_ser.groupby("scen_id").agg(entropy)
    entropy_ser.name = "vocation_entropy"
    entropy_df = entropy_ser.to_frame()
    entropy_df["counterfact"] = "shared"

    # Calculate utilizations
    power["shar_priv"] = power["counterfact"].str.replace(r"_[0-9]+", "", regex=True)
    power = power.set_index(
        ["counterfact", "shar_priv"], append=True
    )  # Treatment is different because power is coming from feather, while scalars are coming from CSV
    scalars["shar_priv"] = scalars["counterfact"].str.replace(
        r"_[0-9]+", "", regex=True
    )
    scalars = scalars.set_index(["scen_id", "counterfact", "shar_priv"])
    util_df = calc_utilization(
        power["avg_power_kw"],
        scalars["depot_capacity_kw"],
        grp_cols=["scen_id", "counterfact"],
    )  # Once per sub-scenario

    # Improvements
    improve_df = calc_utilization(
        power["avg_power_kw"],
        scalars["depot_capacity_kw"],
        grp_cols=["scen_id", "shar_priv"],
    )
    improve_df = improve_df.unstack(
        level="shar_priv",
    )
    improve_df.columns = improve_df.columns.droplevel(0)
    improve_df["util_diff"] = improve_df["shared"] - improve_df["private"]
    improve_df = improve_df.drop(columns=["private", "shared"])
    improve_df["counterfact"] = "shared"
    improve_df = improve_df.set_index("counterfact", append=True)

    scalars = scalars.merge(entropy_df, on=["scen_id", "counterfact"], how="left")
    scalars = scalars.merge(util_df, on=["scen_id", "counterfact"], how="left")
    scalars = scalars.merge(improve_df, on=["scen_id", "counterfact"], how="left")
    scalars = scalars.reset_index()
    scalars = scalars.drop(columns=["Unnamed: 0"])
    return scalars


def plot_util_vs_entropy(util: pd.Series, entropy: pd.Series) -> None:
    """Plot the pattern of utilization against fleet entropy."""
    plot_df = pd.concat([util, entropy], axis=1)
    plot_df["utilization_pct"] = plot_df["utilization"] * 100
    fig = plt.figure()
    sns.scatterplot(data=plot_df, x="vocation_entropy", y="utilization_pct")
    plt.title("Illustration of Depot Utilization vs. Vocation Entropy")
    plt.xlabel("Vocation Entropy of the Depot Fleet")
    plt.ylabel("Utilization of the Depot Charging Power [%]")
    return fig  # 'util_vs_entropy.svg'


def plot_util_diff_vs_entropy(
    util_diff: pd.Series, entropy: pd.Series, metric_name: str
) -> None:
    """Plot the pattern of utilization differences against fleet entropy."""
    fig = plt.figure()
    sns.scatterplot(x=entropy, y=util_diff)
    plt.title(
        f"Illustration of {metric_name} Depot Utilization Change vs. Vocation Entropy"
    )
    plt.xlabel("Vocation Entropy of the Depot Fleet")
    plt.ylabel(f"{metric_name} Utilization Change of the Depot Charging Power [%]")
    return fig  # f'util_diff_{metric_name}_vs_entropy.svg'


def plot_load_profile(power: pd.Series, params: dict) -> None:
    """Plot the load profile for this depot."""
    time_grp_cols = params["group_cols"] + ["time"]
    depot_power = power.groupby(time_grp_cols).sum()
    plot_df = depot_power.reset_index()
    plot_df = plot_df.sort_values(time_grp_cols)

    n_grp = plot_df.drop_duplicates(subset=params["group_cols"]).shape[0]
    n_scen = plot_df.drop_duplicates(subset=["scen_id"]).shape[0]
    title_tail = " by Scenario"
    if n_grp > params["scenario_plot_thresh"]:
        g = plot_vocational_power(
            time=plot_df["time"],
            power=plot_df["avg_power_kw"],
            voc=plot_df["voc_category"],
        )
        g.set_axis_labels("Hour of Day", "Depot Power [kW]")
        for ax in g.axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%I%p"))
        fig = g.figure
        fig.subplots_adjust(top=0.9)
        title_tail = (
            title_tail
            + "\nwith min, 5th, 25th, 75th, 95th, and max quantiles around the mean"
        )
    else:
        if n_grp == 1:
            hue_col = None
            title_tail = ""
        elif n_grp > 1 and n_scen == 1:
            hue_col = "counterfact"
        elif n_grp > 1 and n_scen > 1:
            hue_col = "scen_id"
        fig, ax = plt.subplots()
        ax = sns.lineplot(
            data=plot_df, x="time", y="avg_power_kw", hue=hue_col, axes=ax
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%I%p"))
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Depot Power [kW]")
    fig.suptitle(f"Illustration of Depot Power for a Typical Day{title_tail}")
    return fig


def plot_load_duration(power: pd.Series, params: dict) -> None:
    """Plot the load duration curve for this depot."""
    time_grp_cols = params["group_cols"] + ["time"]
    depot_power_ser = power.groupby(time_grp_cols).sum()
    plot_df = depot_power_ser.reset_index()
    plot_df = plot_df.sort_values(
        params["group_cols"] + ["avg_power_kw"], ascending=False
    )

    def norm(x: pd.Series) -> pd.Series:
        return np.arange(1, x.count() + 1) / (x.count() + 1) * 100

    plot_df["util_rate_pct"] = plot_df.groupby(params["group_cols"])[
        "avg_power_kw"
    ].transform(norm)
    plot_df = plot_df.reset_index(drop=True)
    plot_df = plot_df.sort_values(params["group_cols"])

    n_grp = plot_df.drop_duplicates(subset=params["group_cols"]).shape[0]
    n_scen = plot_df.drop_duplicates(subset=["scen_id"]).shape[0]
    title_tail = " by Scenario"
    fig = plt.figure()
    if n_grp > params["scenario_plot_thresh"]:
        g = plot_vocational_power(
            time=plot_df["util_rate_pct"],
            power=plot_df["avg_power_kw"],
            voc=plot_df["voc_category"],
        )
        g.set_axis_labels("Capacity Utilization Rate [%]", "Capacity Requirements [kW]")
        fig = g.figure
        fig.subplots_adjust(top=0.9)
        title_tail = (
            title_tail
            + "\nwith min, 5th, 25th, 75th, 95th, and max quantiles around the mean"
        )
    else:
        if n_grp == 1:
            hue_col = None
            title_tail = ""
        elif n_grp > 1 and n_scen == 1:
            hue_col = "counterfact"
        elif n_grp > 1 and n_scen > 1:
            hue_col = "scen_id"
        fig, ax = plt.subplots()
        ax = sns.lineplot(
            data=plot_df, x="util_rate_pct", y="avg_power_kw", hue=hue_col, axes=ax
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%I%p"))
        ax.set_xlabel("Capacity Utilization Rate [%]")
        ax.set_ylabel("Capacity Requirements [kW]")
    fig.suptitle(f"Illustration of Load Duration Curve for a Typical Day{title_tail}")
    return fig


def get_quantile_err_band(x: np.ndarray, min: float, max: float) -> tuple[float, float]:
    return (np.quantile(x, min), np.quantile(x, max))


def plot_vocational_power(
    time: pd.Series,
    power: pd.Series,
    voc: pd.Series,
    voc_pair: pd.Series = None,
) -> sns.FacetGrid:
    """Build a FacetGrid of load profiles."""
    plot_df = pd.DataFrame(
        {
            "time": time,
            "power": power,
            "voc": voc.transform(lambda s: string.capwords(s.replace("_", " "))),
        }
    )
    if voc_pair is None:
        g = sns.relplot(
            plot_df,
            x="time",
            y="power",
            col="voc",
            col_wrap=3,
            kind="line",
            errorbar=partial(get_quantile_err_band, min=0.05, max=0.95),
            err_kws={"alpha": 0.1},
        )
        g.set_titles("Vocation = {col_name}")
    else:
        plot_df["voc_pair"] = voc_pair.transform(
            lambda s: string.capwords(s.replace("_", " "))
        )
        g = sns.relplot(
            plot_df,
            x="time",
            y="power",
            col="voc",
            row="voc_pair",
            kind="line",
            errorbar=partial(get_quantile_err_band, min=0.05, max=0.95),
            err_kws={"alpha": 0.1},
        )
    g.map_dataframe(
        sns.lineplot,
        x="time",
        y="power",
        errorbar=partial(get_quantile_err_band, min=0.25, max=0.75),
        err_kws={"alpha": 0.2},
    )
    return g


def plot_reduced_dims(reduce_df: pd.DataFrame, hue_ser: pd.Series = None) -> None:
    """Plot principal component loadings for each depot-day.

    Args:
        reduce_df:
            Has index of scenario names, and columns of the two principal
            component values.
        hue_ser:
            Has index of scenario names, and values that will become the hue
            of the plot.

    """
    if hue_ser is not None:
        plot_df = reduce_df.merge(hue_ser, left_index=True, right_index=True)
        hue_name = hue_ser.name
        # plot_name = f"reduced_dimension_{hue_name}.svg"
    else:
        plot_df = reduce_df
        hue_name = None
        # plot_name = "reduced_dimension.svg"
    fig = plt.figure()
    sns.scatterplot(data=plot_df, x="PC_1", y="PC_2", hue=hue_name)
    plt.title("Illustration of Scenarios Plotted in Principal Component Space")
    return fig  # plot_name


def calc_reduced_dims(power: pd.Series, params: dict) -> tuple[pd.DataFrame, PCA]:
    """Calculate 2-dimensional embedding of depot load profiles."""
    depot_power_ser = power.groupby(["scen_id", "time"]).sum()
    # Smooth by scenario across time
    smooth_span = params["smoothing_span"]

    def smoother(x: pd.Series) -> pd.Series:
        return x.ewm(span=smooth_span).mean()

    smooth_ser = depot_power_ser.groupby("scen_id").transform(smoother)
    if params["scale"]:
        smooth_ser = smooth_ser.groupby("scen_id").transform(minmax_scale)
    smooth_df = smooth_ser.to_frame().reset_index()
    piv_df = smooth_df.pivot(columns="time", index="scen_id", values="avg_power_kw")
    reducer = PCA(n_components=params["n_components"])
    reducer.fit(X=piv_df)

    reduce_df = pd.DataFrame(
        reducer.transform(X=piv_df),
        index=piv_df.index,
        columns=[f"PC_{i + 1}" for i in range(reducer.n_components_)],
    )
    return (reduce_df, reducer)


def plot_components(reducer: PCA, times: np.ndarray) -> None:
    """Plot components of PCA dimension reducer."""
    times_ser = pd.Series(times)
    times_disp = (times_ser - times_ser.dt.floor("1D")).dt.total_seconds() / 3600
    pc_df = pd.DataFrame(reducer.components_, columns=times_disp)
    pc_df.index.name = "component"
    pc_df = pc_df.reset_index()
    pc_df["component"] = pc_df["component"].transform(lambda c: f"PC_{c + 1}")
    pc_df = pc_df.set_index("component")
    pc_df = pc_df.melt(ignore_index=False)
    fig = plt.figure()
    sns.lineplot(data=pc_df, x="time", y="value", hue="component")
    plt.ylabel("Deviation from Mean of 1 Unit of Principal Component")
    plt.title("Illustration of Principal Component Vectors")
    return fig  # 'components.svg'


def calc_utilization(
    power: pd.Series, capacity: pd.Series, grp_cols: list[str]
) -> float:
    """Calculate the utilization metric of this depot, given load profiles and
    the capacity.

    Assumes that power is in units of kW.
    """
    depot_power_ser = power.groupby(grp_cols + ["time"]).sum()

    def get_step_hrs(ser):
        time_ser = ser.index.get_level_values("time")
        time_step_s = (time_ser[1] - time_ser[0]).total_seconds()
        return time_step_s * pd.Timedelta("1s") / pd.Timedelta("1H")

    depot_freq_ser = depot_power_ser.groupby(grp_cols).apply(get_step_hrs)
    depot_freq_ser.name = "step_hrs"

    energy_df = depot_power_ser.to_frame()
    energy_df = energy_df.merge(depot_freq_ser, on=grp_cols)
    energy_df["energy_delivered_kwh"] = energy_df[power.name] * energy_df["step_hrs"]
    energy_df = energy_df.groupby(grp_cols)["energy_delivered_kwh"].sum().to_frame()

    depot_capacity_ser = capacity.groupby(grp_cols).sum()
    capacity_kwh_ser = depot_capacity_ser * pd.Timedelta("1D") / pd.Timedelta("1H")
    capacity_kwh_ser.name = "capacity_kwh"
    util_df = energy_df.merge(capacity_kwh_ser, on=grp_cols)
    util_df["utilization"] = util_df["energy_delivered_kwh"] / util_df["capacity_kwh"]
    return util_df["utilization"].to_frame()


def calc_utilization_change(util: pd.Series, grp_cols: list[str]) -> pd.DataFrame:
    """Calculate improvement in utilization across counterfactuals."""
    util = util.reset_index("counterfact")
    shared_idx = util["counterfact"] == "shared"
    util_shared = util.loc[shared_idx, :].drop(columns=["counterfact"])
    util_private = util.loc[~shared_idx, :]

    util_diff = util_private.merge(
        util_shared, on=grp_cols, suffixes=["_private", "_shared"]
    )
    util_diff["utilization_diff"] = (
        util_diff["utilization_shared"] - util_diff["utilization_private"]
    )
    util_diff = util_diff.drop(columns=["utilization_private", "utilization_shared"])
    return util_diff


def calc_wasserstein(*args: pd.Series) -> float:
    """Calculate the Wasserstein distance between two load profiles.

    The Wasserstein distance gives us a unitless number which is a distance
    between two charging profiles. This could be used for creating distance
    matrices and clustering load profiles.

    However, it cannot be interpreted as the amount of energy needed to be
    shifted to turn one profile into another profile. There are two reasons
    for this: 1) The profiles would need to have the exact same amount of
    energy delivered, which is not going to happen frequently given our
    sampling pipeline. 2) The Wasserstein distance formula takes differences
    between CDFs, not PDFs, so the output of the difference of two charging
    profiles would be in units of kW * hr^2, not kWh.

    Args:
        power_a:
            first typical day load profile, must have an index labelled
            'time'
        power_b:
            second typical day load profile, must have an index labelled
            'time'. If None, then a uniform distribution will be assumed on
            the same grid of time points as power_a has.

    Returns:
        Wasserstein metric between power_a and power_b
    """
    two = 2  # Used to pass ruff linting tests
    if len(args) < 1:
        raise ValueError("calc_wasserstein needs at least one distribution specified.")
    elif len(args) > two:
        raise ValueError(
            "calc_wasserstein cannot operate on more than two distributions."
        )

    if args[0] is None and args[1] is None:
        raise ValueError("calc_wasserstein needs at least one distribution specified.")
    elif args[0] is None:
        ser_a = args[1]
        ser_b = None
    else:
        ser_a = args[0]
        ser_b = args[1]

    def standardize_dist(ser: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Returns values and weights to prep for Wasserstein calculation"""
        weights = ser.groupby("time").sum()
        n_steps_a = weights.shape[0]
        values = np.arange(1, n_steps_a + 1) / (n_steps_a + 1)
        return (values, weights)

    a_values, a_weights = standardize_dist(ser_a)
    if ser_b is not None:
        b_values, b_weights = standardize_dist(ser_b)
    else:
        b_values = a_values
        b_weights = np.ones(a_values.shape[0])

    dist = wasserstein_distance(
        u_values=a_values, v_values=b_values, u_weights=a_weights, v_weights=b_weights
    )
    return dist


def calc_pairwise_wasserstein(power: pd.Series) -> pd.Series:
    """Calculate pairwise Wasserstein distances between all scenarios."""
    scen_ls = list(power.index.get_level_values("scen_id").unique())
    if len(scen_ls) == 1:
        scen_ls.append("uniform")

    pairs_ls = list(combinations(scen_ls, 2))
    pairs_df = pd.DataFrame.from_records(pairs_ls, columns=["scen_a", "scen_b"])

    def calc_wasser_from_names(name_a: str, name_b: str):
        if name_a == "uniform" and name_b == "uniform":
            return 0
        if name_a == "uniform":
            power_a = None
        else:
            power_a = power.loc[name_a]
        if name_b == "uniform":
            power_b = None
        else:
            power_b = power.loc[name_b]
        return calc_wasserstein(power_a, power_b)

    pairs_df["wasser"] = pairs_df.apply(
        lambda row: calc_wasser_from_names(row["scen_a"], row["scen_b"]), axis=1
    )
    return pairs_df


def calc_vocation_entropy(
    self, veh_df: pd.DataFrame, grp_cols: list[str]
) -> pd.DataFrame:
    """Calculate the vocation entropy for each scenario.

    Args:


    Returns:
        Entropy of vehicles-by-vocation vector for each scenario.
    """
    entropy_ser = veh_df.groupby(grp_cols)["n_vehicles"].agg(entropy)
    entropy_ser.name = "vocation_entropy"
    return entropy_ser
