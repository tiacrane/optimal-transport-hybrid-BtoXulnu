import numpy as np
import pandas as pd


def compute_normalization(df_incl, df_excl, branching_fractions, inclusive_bf,
                          pdg_col="X_gen_PDG", input_weight_col="input_weight"):
    """
    Compute per-event normalization weights so that the resonant and inclusive
    samples can be combined into a single hybrid model.

    The normalization follows the standard hybrid construction:

        excl_norm = (N_incl / N_excl) * (1 - BF_Xu / (sum_BF_excl + BF_Xu))

    This factor is applied uniformly to all exclusive events (independent of
    PDG code, since the exclusive sample is pre-mixed by EvtGen). Inclusive
    events get weight 1 from this factor. Both samples are further multiplied
    by their input_weight column (e.g. form-factor corrections; set to 1 if absent).

    The resulting column "norm_weight" is added in-place to both DataFrames.

    Parameters
    ----------
    df_incl : DataFrame
        Inclusive simulation sample.
    df_excl : DataFrame
        Combined exclusive (resonant) simulation sample.
    branching_fractions : dict
        PDG code (int) -> branching fraction (float, absolute). Only used to
        compute the total exclusive branching fraction for the normalization.
    inclusive_bf : float
        Inclusive B(B -> Xu l nu) branching fraction.
    pdg_col : str
        Column name for the hadronic system PDG code.
    input_weight_col : str
        Optional per-event weight column already present in the DataFrames
        (e.g. form-factor corrections). If the column does not exist, it is
        treated as 1 for all events.
    """
    sum_excl_bf = sum(branching_fractions.values())
    denom = sum_excl_bf + inclusive_bf

    n_incl = len(df_incl)
    n_excl = len(df_excl)
    excl_norm = (n_incl / n_excl) * (1.0 - inclusive_bf / denom)

    for df, bf_factor in ((df_incl, 1.0), (df_excl, excl_norm)):
        if input_weight_col in df.columns:
            input_w = df[input_weight_col].values
        else:
            print('WARNING : input weights set to 1.0')
            input_w = 1.0
        df["norm_weight"] = bf_factor * input_w


def apply_ot_weights(df, reweight_map, bin_width=None,
                     pplus_col="genPplus", pminus_col="genPminus"):
    """
    Assign per-event OT hybrid weights by looking up each event's (P+, P-) bin.

    Parameters
    ----------
    df : DataFrame
        Inclusive events.
    reweight_map : 2D array | str
        Weight map of shape (n_pplus_bins, n_pminus_bins) as returned by extract_reweights
        (bins of bin_width starting at 0), OR the path to a weight table csv written by
        compute_weights.py (the bins are then taken from the csv and bin_width is not needed).
    bin_width : float
        Bin size used when building the grid (same value as in solve_ot). Only needed for a 2D array.
    pplus_col, pminus_col : str
        Column names for P+ and P-.

    Returns
    -------
    ndarray, shape (n_events,)
        Per-event OT hybrid weights. Events outside the grid get the weight of the nearest
        edge bin, events with NaN P+/P- the weight of bin (0, 0) (warnings are printed).
    """
    from_csv = isinstance(reweight_map, str)
    if from_csv:
        reweight_map, pp_edges, pm_edges = read_weight_table(reweight_map)
    elif bin_width is None:
        raise ValueError("bin_width is needed when reweight_map is a 2D array")
    else:
        pp_edges = np.arange(reweight_map.shape[0] + 1) * bin_width
        pm_edges = np.arange(reweight_map.shape[1] + 1) * bin_width

    pplus, pminus = df[pplus_col].values.astype(float), df[pminus_col].values.astype(float)
    nan = ~np.isfinite(pplus) | ~np.isfinite(pminus)
    out = ((pplus < pp_edges[0]) | (pplus >= pp_edges[-1]) | (pminus < pm_edges[0]) | (pminus >= pm_edges[-1])) & ~nan
    if nan.any():
        print(f"WARNING: {int(nan.sum())} events have NaN {pplus_col}/{pminus_col}; they get the weight of bin (0, 0)")
    if out.any():
        print(f"WARNING: {int(out.sum())} events are outside the weight grid (P+ in [{pp_edges[0]:g}, {pp_edges[-1]:g}), "
              f"P- in [{pm_edges[0]:g}, {pm_edges[-1]:g}) GeV); they get the weight of the nearest edge bin")

    if from_csv:   # bins from the csv edges
        ix = np.searchsorted(pp_edges, pplus, side="right") - 1
        iy = np.searchsorted(pm_edges, pminus, side="right") - 1
    else:          # official lookup for a map from extract_reweights
        with np.errstate(invalid="ignore"):  # NaN -> int cast, reported above
            ix = (pplus / bin_width).astype(int)
            iy = (pminus / bin_width).astype(int)
    ix = np.clip(ix, 0, reweight_map.shape[0] - 1)
    iy = np.clip(iy, 0, reweight_map.shape[1] - 1)
    ix[nan], iy[nan] = 0, 0
    return reweight_map[ix, iy]


def read_weight_table(csv_path, weight_col="ot_hybrid_weight"):
    """
    Read a weight table csv (pplus_low, pplus_high, pminus_low, pminus_high, ot_hybrid_weight; one
    row per bin, any row order) into a 2D weight map and its bin edges, taken from the csv itself.

    Returns
    -------
    grid : ndarray, shape (n_pplus_bins, n_pminus_bins)
    pplus_edges, pminus_edges : ndarray
    """
    t = pd.read_csv(csv_path)
    edges, index = {}, {}
    for var in ("pplus", "pminus"):
        low, high = t[f"{var}_low"].round(9).to_numpy(), t[f"{var}_high"].round(9).to_numpy()
        edges[var] = np.union1d(low, high)
        index[var] = np.searchsorted(edges[var], low)
        if not np.allclose(edges[var][index[var] + 1], high):
            raise ValueError(f"{csv_path}: the {var} bins overlap or are not contiguous")

    grid = np.full((len(edges["pplus"]) - 1, len(edges["pminus"]) - 1), np.nan)
    if len(set(zip(index["pplus"], index["pminus"]))) != len(t):
        raise ValueError(f"{csv_path}: some (P+, P-) bins appear more than once")
    grid[index["pplus"], index["pminus"]] = t[weight_col].to_numpy()
    if np.isnan(grid).any():
        print(f"WARNING: {int(np.isnan(grid).sum())} (P+, P-) bins missing from {csv_path} (or NaN), "
              f"their weight is set to 0")
        grid = np.nan_to_num(grid, nan=0.0)
    return grid, edges["pplus"], edges["pminus"]
