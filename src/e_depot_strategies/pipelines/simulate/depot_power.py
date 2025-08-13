import cvxpy as cp
import numpy as np
import pandas as pd

N_DAYS_PER_YEAR = 365
HRS_PER_DAY = 24


def hrs_to_timestamp(x):
    return pd.Timestamp(x, unit="h")


def calc_depot_profiles(
    sessions: pd.DataFrame,
    params: dict,
    costs: dict,
    optim: dict,
    license: dict,
) -> pd.Series:
    """Creates an actual set of charging events for all the given vehicles.

    Assumes that the input charge_begin and charge_end are "typical day"
    times, so the charge_end may be before the charge_begin, denoting that
    the charging session went across midnight.

    Args:
        sessions:
            DataFrame with columns for total energy consumed, beginning time for
            charging availability, and ending time for charging availability for
            each session.
        is_time_cyclic:
            If True, then treat the incoming time values as wrapping around
            in a cycle to form a typical day. For example, if an event ends
            at a time less than its beginning, then this implies that the
            event crossed midnight. If False, then treat times as literal
            with no cycles.

    Returns:
        Dataframe of average power at each time step on the typical day,
        indexed by each vehicle-day identifier.
    """
    d_t = pd.Timedelta(params["timestep_freq"]) / pd.Timedelta("1H")
    # Note that begin and end seem to be switched here. However, this is
    # correct, because all prior code was considering the beginning and ending
    # of active time, while this code is looking for the beginning and ending of
    # charging time, which are complementary.
    frame = create_session_framework(
        grp=sessions["veh_day_id"],
        begin=sessions["end_hrs"].transform(hrs_to_timestamp),
        end=sessions["start_hrs"].transform(hrs_to_timestamp),
        freq=params["timestep_freq"],
    )
    frame.index = frame.index.set_names({"grp": "veh_day_id"})

    # Build up warm-starts
    use_warm_start = costs["valet_shifts"]["min_count"] > 0
    if use_warm_start:
        sessions["slowest_charge_kw"] = (
            sessions["energy_required_kwh"] / sessions["hrs_offshift"]
        )
        power_levels = list(costs["plug_power_kw"].values())
        sessions["smallest_charger_kw"] = round_up_to_nearest(
            sessions["slowest_charge_kw"].values, power_levels
        )
        sessions["min_charge_hrs"] = (
            sessions["energy_required_kwh"] / sessions["smallest_charger_kw"]
        )

        sessions["time_sorter"] = sessions["end_hrs"] - 12  # Start from noon

        # Get the last possible time for the vehicle to start charging to be ready
        unroll = sessions["start_hrs"] < sessions["end_hrs"]
        sessions["start_hrs_unroll"] = sessions["start_hrs"] + unroll * HRS_PER_DAY
        sessions["last_charge_start_hrs"] = (
            sessions["start_hrs_unroll"] - sessions["min_charge_hrs"]
        )  # Start_hrs is the "start of active time hours"

        buff_hrs = pd.Timedelta(params["timestep_freq"]).total_seconds() / 3600

        def _set_heuristic_charging(grp: pd.DataFrame) -> pd.DataFrame:
            """Set heuristic charging by charging all vehicles in an offset fashion."""
            cur_charge_start = grp["end_hrs"].iloc[0]
            n_chargers = 1
            for id, veh in grp.iterrows():
                cur_charge_start = np.maximum(cur_charge_start, veh["end_hrs"])
                last_possible_start = veh["last_charge_start_hrs"]
                if (
                    cur_charge_start >= last_possible_start
                ):  # If we need to have started earlier, then start a new charger
                    chg_hr = veh["end_hrs"]
                    n_chargers += 1
                else:  # Continue blocking charging forward on this charger
                    chg_hr = cur_charge_start
                grp.loc[id, "charge_start_hrs_roll"] = chg_hr
                grp.loc[id, "charger_id"] = n_chargers
                cur_charge_start = chg_hr + veh["min_charge_hrs"] + buff_hrs
            return grp

        sessions = sessions.sort_values(
            "end_hrs"
        )  # Sort by the vehicles' arrival times back to the depot
        sessions["charge_start_hrs_roll"] = np.NaN
        sessions["charger_id"] = np.NaN
        sessions = sessions.groupby("smallest_charger_kw", group_keys=False).apply(
            _set_heuristic_charging
        )
        sessions["charge_end_hrs_roll"] = (
            sessions["charge_start_hrs_roll"] + sessions["min_charge_hrs"] + buff_hrs
        )
        sessions["charge_start_hrs"] = sessions["charge_start_hrs_roll"].mod(
            HRS_PER_DAY
        )
        sessions["charge_end_hrs"] = sessions["charge_end_hrs_roll"].mod(HRS_PER_DAY)

        sessions = sessions.sort_index()
        warm_plugs = create_session_framework(
            grp=sessions["veh_day_id"],
            begin=sessions["charge_start_hrs"].transform(hrs_to_timestamp),
            end=sessions["charge_end_hrs"].transform(hrs_to_timestamp),
            freq=params["timestep_freq"],
        )
        warm_plugs_mat = warm_plugs["avail"].unstack().values
        sessions["power_guess_kw"] = sessions["energy_required_kwh"] / (
            d_t * np.sum(warm_plugs_mat, axis=1)
        )
        pow_guess_vec = np.expand_dims(sessions["power_guess_kw"].values, axis=1)
        pow_avail_vec = np.expand_dims(sessions["smallest_charger_kw"].values, axis=1)
        pow_mat = warm_plugs_mat * pow_guess_vec
        warm_plugs_sel = {
            k: pow_avail_vec == pow for k, pow in costs["plug_power_kw"].items()
        }
        warm_plugs_mats = {k: warm_plugs_mat * vec for k, vec in warm_plugs_sel.items()}

    shift_info = pd.DataFrame.from_dict(
        costs["valet_shifts"]["end_hrs"], orient="index", columns=["end_hrs"]
    )
    shift_info.index.name = "shift"
    shift_info = shift_info.reset_index()
    shift_info["start_hrs"] = (
        shift_info["end_hrs"]
        .shift(1)
        .fillna(shift_info["end_hrs"].iloc[-1])
        .astype(int)
    )
    shifts = create_session_framework(
        grp=shift_info["shift"],
        begin=shift_info["start_hrs"].transform(hrs_to_timestamp),
        end=shift_info["end_hrs"].transform(hrs_to_timestamp),
        freq=params["timestep_freq"],
    )

    # Set up optimization matrices
    plug_power_ls = list(costs["plug_power_kw"].keys())

    avail_df = frame["avail"].unstack()
    A = avail_df.values

    shift_mask = shifts["avail"].unstack().values
    shift_durs = shift_mask.sum(axis=1)
    if np.unique(shift_durs).size > 1:
        raise Warning(
            "All shifts must have equal length, or the valet shift constraints will fail."
        )
    shift_dur = shift_durs[0]

    P = cp.Variable(A.shape, nonneg=True)
    G = {pow: cp.Variable(A.shape, boolean=True) for pow in plug_power_ls}
    if use_warm_start:
        P.value = pow_mat.astype(float)
        for s, G_s in G.items():
            G[s].value = warm_plugs_mats[s]

    E = (
        sessions.set_index("veh_day_id")
        .loc[avail_df.index, "energy_required_kwh"]
        .values
    )

    # Application constraints
    c = []
    plugged = sum_3D_cp(G, 0)
    c.append(cp.multiply(1 - A, plugged) == 0)  # no charging when active
    c.append(1 - plugged >= 0)  # no double-charging
    c.append(E - d_t * cp.sum(P, axis=1) == 0)  # energy adequacy
    c.append(P - sum_3D_cp(G, 0, costs["plug_power_kw"]) <= 0)  # charger power limit

    # Plug cost
    n_plugs = {s: cp.max(cp.sum(G_s, axis=0)) for s, G_s in G.items()}
    present_value_plugs = cp.sum(
        [n_plugs[s] * costs["plug_cost_dollars"][s] for s in n_plugs]
    )
    cost_plugs = present_to_repeating(
        p=present_value_plugs,
        i_ann_pct=costs["discount_rate_pct"],
        n_years=costs["time_horizon_yrs"],
        freq="1D",
    )

    # Demand cost
    rate = costs["electric_rate"]
    depot_power = cp.sum(P, axis=0)
    max_power = cp.max(depot_power)
    cost_demand = rate["flatdemandcost"] * max_power

    # Energy cost
    energy_times = pd.date_range(
        pd.Timestamp(0, unit="s"),
        pd.Timestamp(24 * 3600, unit="s"),
        freq=params["timestep_freq"],
        inclusive="left",
    ).to_series()
    hrs = energy_times.dt.hour
    interval_costs = hrs.transform(
        lambda h: rate["energyratestructure"][rate["energyweekdayschedule"][h]]
    )
    total_energy = cp.sum(depot_power * d_t)
    cost_energy = cp.sum(cp.multiply(depot_power * d_t, interval_costs.values))

    # Valet cost
    plug_change = take_wrapped_first_diff(plugged)
    driver_present = take_wrapped_first_diff(A)
    valet_switches = cp.sum(cp.maximum(plug_change - driver_present, 0), axis=0)
    valet_switches_stack = cp.vstack([valet_switches] * shift_mask.shape[0])
    valets_per_shift = cp.max(cp.multiply(shift_mask, valet_switches_stack), axis=1)
    n_valet_shifts_raw = cp.sum(valets_per_shift, axis=0)
    n_valet_shifts = cp.maximum(n_valet_shifts_raw, costs["valet_shifts"]["min_count"])
    cost_valet = costs["valet_cost_per_hr"] * d_t * shift_dur * n_valet_shifts

    # Apply bounds on the maximum number of valets
    c_warm = c + [n_valet_shifts_raw <= 0]
    c_ext = c + [n_valet_shifts <= costs["valet_shifts"]["max_count"]]

    # Charging power change sparsity penalty
    power_change = take_wrapped_first_diff(P)
    phi_sparsity = costs["power_change_penalty"] * cp.sum(power_change)

    # Set up an solve problem
    obj = cp.Minimize(
        cost_plugs + cost_demand + cost_energy + cost_valet + phi_sparsity
    )
    prob_warm = cp.Problem(obj, c_warm)
    prob = cp.Problem(obj, c_ext)

    if optim["solver"] == "GUROBI" and optim["use_personal_license"]:
        import gurobipy as gp

        env = gp.Env(params=license)
    else:
        env = None
    if optim["solver"] in optim:  # If there are solver options
        if optim["warm_start"]:
            prob_warm.solve(
                solver=optim["solver"],
                warm_start=optim["warm_start"],
                verbose=optim["verbose"],
                env=env,
                **optim[optim["solver"]],
            )
        pass
        prob.solve(
            solver=optim["solver"],
            warm_start=optim["warm_start"],
            verbose=optim["verbose"],
            env=env,
            **optim[optim["solver"]],
        )
    else:
        if optim["warm_start"]:
            prob_warm.solve(
                solver=optim["solver"],
                warm_start=optim["warm_start"],
                verbose=optim["verbose"],
                env=env,
            )
        prob.solve(
            solver=optim["solver"],
            warm_start=optim["warm_start"],
            verbose=optim["verbose"],
            env=env,
        )

    # Record results
    res_dict = {"problem_status": prob.status}
    res_dict.update({"n_vehicles": A.shape[0]})
    capacity_plugs = np.sum(
        [n_plugs[s].value * costs["plug_power_kw"][s] for s in n_plugs]
    )
    res_dict.update({f"n_plugs_{s}": int(n.value) for s, n in n_plugs.items()})
    res_dict.update(
        {
            "depot_capacity_kw": capacity_plugs,
            "present_value_plugs_dollars": round(present_value_plugs.value, 2),
            "max_power_kw": round(float(max_power.value), 2),
            "cost_plugs_dollars": round(float(cost_plugs.value), 2),
            "cost_demand_dollars": round(cost_demand.value, 2),
            "energy_delivered_kwh": round(total_energy.value, 2),
            "cost_energy_dollars": round(cost_energy.value, 2),
            "n_valet_shifts": round(n_valet_shifts_raw.value, 0),
            "valets_per_shift": str(valets_per_shift.value),
            "cost_valet_dollars": round(cost_valet.value, 2),
        }
    )
    res_profs = pd.DataFrame(P.value, index=avail_df.index, columns=avail_df.columns)

    res_scalars = pd.DataFrame(res_dict, index=pd.RangeIndex(0, 1))
    res_profs = res_profs.melt(ignore_index=False, value_name="avg_power_kw")
    res_profs = res_profs.set_index("time", append=True)

    return (res_profs, res_scalars)


def take_wrapped_first_diff(
    x: cp.Expression | np.ndarray,
) -> cp.Expression | np.ndarray:
    if isinstance(x, cp.Expression):
        x_wrap = cp.reshape(x[:, 0] - x[:, -1], (x.shape[0], 1))
        x_diff = cp.abs(cp.hstack([cp.diff(x, axis=1), x_wrap]))
    else:
        x_wrap = np.reshape(x[:, 0] - x[:, -1], (x.shape[0], 1))
        x_diff = np.abs(np.hstack([np.diff(x, axis=1), x_wrap]))
    return x_diff


def sum_3D_cp(
    D: dict[cp.Variable],
    axis: int,
    w: dict[float] = None,
) -> dict[cp.Variable] | cp.Variable:
    """Sum across an artificial 3D cp.Variable, where the third dimension is
    represented as a dict.

    Args:
        D:
            the dict of cp.Variables to sum across
        axis:
            the axis over which to sum the variable, where the first axis is
            represented by the dict keys.
        w:
            a dict of weights to multiply the cp.Variables by

    Returns: summed cp.Variable
    """
    if w is None:
        w = dict.fromkeys(D.keys(), 1)
    if axis >= 1:
        return {cp.sum(D_s, axis=axis - 1) for s, D_s in D.items()}
    else:
        res = None
        for s, D_s in D.items():
            if res is None:
                res = w[s] * D_s
            else:
                res += w[s] * D_s
        return res


def create_session_framework(
    grp: pd.Series,
    begin: pd.Series,
    end: pd.Series,
    freq: str,
    is_time_cyclic: bool = True,
) -> pd.DataFrame:
    """Create a discrete time framework from session id, beginning, and ending.

    Note that all events represented must be of the same type. There is no
    facility here to group by different types of events.

    Args:
        grp:
            Groups for events, often the entity for which the event occurs
            (e.g. truck, hot water heater)
        begin:
            Times of event beginnings as pandas Timestamp objects
        end:
            Times of event endings as pandas Timestamp objects
        freq:
            frequency string (as defined by `pandas`) for the final discrete
            time system
        is_time_cyclic:
            If True, then treat the incoming time values as wrapping around
            in a cycle to form a typical day. For example, if an event ends
            at a time less than its beginning, then this implies that the
            event crossed midnight. If False, then treat times as literal
            with no cycles.


    Returns: pandas DataFrame of discrete times which are inside of or outside
    of an event.
    """
    begin_end = pd.DataFrame(
        {
            "grp": grp,
            "sess_id": np.arange(grp.shape[0]) + 1,
            "begin": begin,
            "end": end,
        }
    )

    # Cycle times, if necessary
    if is_time_cyclic:
        crosses_midnight = begin_end["end"] < begin_end["begin"]
        begin_end["end"] = begin_end["end"] + pd.Timedelta("1D") * crosses_midnight

    melt_df = begin_end.melt(
        id_vars=["grp", "sess_id"], var_name="time_type", value_name="time"
    )
    melt_df["sess_id"] = np.where(melt_df["time_type"] == "end", melt_df["sess_id"], 0)

    frame_df = build_discrete_from_int_end(
        idx_ser=melt_df["sess_id"],
        time_ser=melt_df["time"],
        grp_ser=melt_df["grp"],
        freq=freq,
        time_extent=(pd.Timestamp(0, unit="s"), pd.Timestamp(24 * 3600, unit="s")),
    )
    frame_df = frame_df.reset_index()

    # Then roll back onto a typical day timeframe
    if is_time_cyclic:
        frame_df["time_mod"] = frame_df["time"] - frame_df["time"].dt.floor("1D")
        frame_df["avail"] = (frame_df["sess_id"] > 0).astype(int)
        frame_df = frame_df.drop(columns=["sess_id"])
        frame_df = frame_df.groupby(["grp", "time_mod"]).max().reset_index()
        frame_df = frame_df.drop_duplicates(["grp", "time_mod"], keep="last")
        frame_df["time"] = pd.to_datetime(
            frame_df["time_mod"].dt.total_seconds(), unit="s"
        )
        frame_df = frame_df.drop(columns=["time_mod"])

    frame_df = frame_df.set_index(["grp", "time"])
    frame_df = frame_df.sort_index()
    return frame_df


def build_discrete_from_int_end(
    idx_ser: pd.Series,
    time_ser: pd.Series,
    grp_ser: pd.Series = None,
    freq: str = "1H",
    time_extent: tuple[pd.Timestamp, pd.Timestamp] = (None, None),
) -> pd.Series:
    """Create a discrete time framework from an interval-ending time series
    and a matching set of event indices.

    Args:
        idx_ser:
            Event indices to align onto the discrete time frame
        time_ser:
            Timestamps for each event's ending time
        grp_ser:
            Groups for events, often the entity for which the event occurs
            (e.g. truck, hot water heater)
        freq:
            frequency string (as defined by `pandas`) for the final discrete
            time system
        time_extent:
            times to (begin, end) the frame at. If the maximum value in time_ser
            is greater than the end value here, then the end value will be overridden

    Returns:
        Series with MultiIndex of group and discrete time which contains
        indices for events occuring during those time periods.
    """
    if grp_ser is None:
        grp_ser = pd.Series(0, index=idx_ser.index)

    event_df = pd.DataFrame(
        {"grp": grp_ser.values, "time": time_ser.values, "sess_id": idx_ser.values}
    )
    event_df = event_df.set_index(["grp", "time"])
    event_df = event_df.sort_index()

    # Get minimum and maximum times
    start_lim, end_lim = time_extent
    start_obs, end_obs = time_ser.min(), time_ser.max()
    start = min(start_lim, start_obs)
    end = max(end_lim, end_obs)
    uniq_grps = grp_ser.unique()
    start_ts = pd.Series(start, index=uniq_grps)
    start_ts.name = "start"
    end_ts = pd.Series(end, index=uniq_grps)
    end_ts.name = "end"
    start_end_df = pd.concat([start_ts, end_ts], axis=1, join="outer")

    # Create time framework for each group
    def construct_periods(
        start: pd.Timestamp, end: pd.Timestamp, group: object
    ) -> pd.DataFrame:
        """Takes a start, end, and group and forms a DataFrame of interval-ending discrete time."""
        return pd.DataFrame(
            {
                "grp": group,
                "time": pd.date_range(
                    start=start, end=end, freq=freq, inclusive="right"
                ),
            }
        )

    period_df_ls = [
        construct_periods(v["start"], v["end"], i) for i, v in start_end_df.iterrows()
    ]
    period_df = pd.concat(period_df_ls, ignore_index=True)

    # Use merge_asof to merge on original charging session indices
    period_df = period_df.sort_values(["time"])
    event_df = event_df.sort_values(["time"])
    idx_df = pd.merge_asof(
        period_df, event_df, by="grp", on="time", direction="forward"
    )
    idx_df = idx_df.set_index(["grp", "time"])
    idx_df = idx_df.sort_index()

    # TODO: Consider modification here to include partial timestep
    # after the end of a charging session as part of that session
    idx_df = idx_df.fillna(0)
    idx_df = idx_df.astype(int)
    return idx_df


def present_to_repeating(
    p: float, i_ann_pct: float, n_years: float, freq: str
) -> float:
    """Calculate time-value of money conversion of present value to repeating payment.

    Args:
        p:
            present value of money
        i_ann_pct:
            interest rate in percent on an annual basis
        n_years:
            number of years that repeating payments will occur over
        freq:
            frequency of the repeating payments

    Returns: float giving value of recurring payments
    """
    ratio = pd.Timedelta(N_DAYS_PER_YEAR, "days") / pd.Timedelta(freq)
    n = n_years * ratio
    i = i_ann_pct / 100 / ratio
    comp = (1 + i) ** n
    numer = i * comp
    denom = comp - 1
    return p * numer / denom


def round_up_to_nearest(arr, values):
    """
    Round up each element in the numpy array `arr` to the nearest value in `values`.

    Parameters:
    arr (np.ndarray): The input numpy array.
    values (list): The list of values to round up to.

    Returns:
    np.ndarray: The rounded numpy array.
    """
    # Convert the list of values to a NumPy array for broadcasting
    values = np.array(values)

    # Ensure the values are sorted to efficiently find the nearest higher value
    values = np.expand_dims(np.sort(values), axis=0)

    # Reshape the input array to a column to simplify processing
    arr_col = np.expand_dims(arr.flatten(), axis=1)

    surplus_power = values - arr_col
    insuf_power = surplus_power < 0
    infeasible = np.all(insuf_power, axis=1, keepdims=True)
    surplus_power_adj = ~insuf_power * surplus_power + insuf_power * (
        np.max(values) + 1
    )
    power_idx = np.argmin(surplus_power_adj, axis=1, keepdims=True)

    power_idx = ~infeasible * power_idx + infeasible * -1
    rounded = values.flatten()[power_idx].flatten()
    return rounded
