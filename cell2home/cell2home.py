import sys
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib import rcParams

import scipy.sparse
import mudata
import muon as mu

# dictionary of chemokine receptors and their ligands based on CellPhoneDB

markers = {'CCR1':['CCL3', 'CCL5', 'CCL7', 'CCL8', 'CCL14', 'CCL15', 'CCL16', 'CCL23', 'CCL26', 'CCL3L1'],
           'CCR2':['CCL2', 'CCL7', 'CCL8', 'CCL13', 'CCL16', 'CCL26'],
           'CCR3':['CCL3', 'CCL5', 'CCL8', 'CCL11', 'CCL13', 'CCL14', 'CCL15', 'CCL24', 'CCL26', 'CCL3L1'],
           'CCR4':['CCL5', 'CCL17', 'CCL22'],
           'CCR5':['CCL3', 'CCL4', 'CCL5', 'CCL7', 'CCL8'],
           'CCR6':['CCL20'],
           'CCR7':['CCL19', 'CCL21'],
           'CCR8':['CCL1', 'CCL16', 'CCL18'],
           'CCR9':['CCL25'],
           'CCR10':['CCL27', 'CCL28'],
           'CXCR1':['CXCL1', 'CXCL5', 'CXCL6', 'CXCL8', 'PPBP'],
           'CXCR2':['CXCL1', 'CXCL2', 'CXCL3', 'CXCL5', 'CXCL6', 'CXCL8', 'PPBP'],
           'CXCR3':['CXCL9', 'CXCL10', 'CXCL11', 'PF4'],
           'CXCR4':['CXCL12', 'CXCL14'],
           'CXCR5':['CXCL13'],
           'CXCR6':['CXCL16'],
           'CX3CR1':['CX3CL1'],
           'XCR1':['XCL1', 'XCL2'],
           'CMKLR1':['RARRES2'],
           'CCRL2':['CCL19', 'RARRES2'],
           'SELL':['CD34', 'PODXL', 'PODXL2', 'SELPLG'],
           'SELP':['CD34', 'PODXL2', 'SELPLG', 'CD24'],
           'ITGAL':['ICAM1', 'ICAM2', 'ICAM3', 'ICAM4', 'F11R'],
           'ITGA4':['FN1', 'MADCAM1', 'VCAM1', 'PLA2G2A', 'JAM2', 'SPP1', 'PLAUR'],
           'ITGB7':['FN1', 'MADCAM1', 'VCAM1', 'CDH1'],
           'ITGAE':['CDH1']
          }

def category_pseudobulker(adata, obs_to_bulk, sep=":", compute_mean=False):
    """
    Compute the pseudobulks of an AnnData object based on unique value combinations 
    of specified observations. The number of cells that went into each pseudobulk 
    will be reported in ``.obs["cell_count"]`` of the resulting object.
    
    Input
    -----
    adata : ``AnnData``
        The object to pseudobulk, must have the desired expression space in ``.X``.
    obs_to_bulk : list of ``str``
        Column names in ``.obs`` to use to construct the pseudobulks. A pseudobulk 
        will be constructed for each unique value combination of the columns. Will 
        be present as ``.obs`` columns in the resulting object.
    sep : ``str``, optional (default: ``":"``)
        The ``.obs_names`` of the resulting pseudobulk will be the unique value 
        combinations of ``obs_to_bulk`` going into making the pseudobulk, matching 
        the provided order and separated by ``sep``.
    compute_mean : ``bool``, optional (default: ``False``)
        If ``True``, will return the mean of the ``.X`` for each of the pseudobulks. 
        If ``False``, will return the sum instead.
    """
    # deeply indebted to dandelion.tl.pseudobulk_gex()
    # but aggressively optimising resource use for specific use case here
    if type(obs_to_bulk) is list:
        # this will create a single value by pasting all the columns together
        tobulk = adata.obs[obs_to_bulk].T.astype(str).agg(sep.join)
    else:
        # we just have a single column
        tobulk = adata.obs[obs_to_bulk]
        # turn the obs_to_bulk variable to a list for later
        obs_to_bulk = [obs_to_bulk]
    # this pandas function creates the exact pseudobulk assignment we want
    # this needs to be different than the default uint8
    # as you can have more than 255 cells in a pseudobulk, it turns out
    # store as sparse to minimise memory footprint
    pbs = pd.get_dummies(tobulk, dtype="uint16", sparse=True)
    # obs can be derived from the column names of the pbs for now
    pbs_obs = pd.DataFrame(index=pbs.columns, columns=obs_to_bulk)
    for i, obs_col in enumerate(obs_to_bulk):
        pbs_obs[obs_col] = [it.split(sep)[i] for it in pbs.columns]
    # add up to get cell count, need to turn to array and flatten so it goes in
    pbs_obs["cell_count"] = np.asarray(pbs.sum(axis=0)).flatten()
    # at this point we can pull out the actual sparse and turn it to CSR
    pbs = pbs.sparse.to_coo().tocsr()
    # is our actual .X sparse?
    if scipy.sparse.issparse(adata.X):
        # the fact that pbs is sparse means the resulting X is sparse
        # gene-cell * cell-pseudobulk = gene-pseudobulk
        pbs_X = adata.X.T.dot(pbs).T
        # do we need to compute a mean rather than our current sum?
        if compute_mean:
            # pre-multiplying by a diagonal matrix multiplies each row by value
            # https://solitaryroad.com/c108.html
            # we've got a computed cell count, multiply by its inverse to get the means
            pbs_X = scipy.sparse.diags(1/pbs_obs["cell_count"].values).dot(pbs_X)
    else:
        # we've got a dense in .X, need to proceed a bit differently
        # first off, turn pbs dense too
        pbs = np.array(pbs.todense())
        # multiply the two denses together via np.dot instead
        pbs_X = np.dot(adata.X.T, pbs).T
        # do we need to compute a mean rather than our current sum?
        if compute_mean:
            # divide each row by the cell total
            pbs_X = pbs_X/pbs_obs["cell_count"].values[:,None]
    pb_adata = sc.AnnData(pbs_X, obs=pbs_obs, var=adata.var)
    return pb_adata

def construct_signatures(adata, obs_to_bulk, interactions, source_col="source"):
    """
    Construct a reference signature based on an AnnData object and a data frame 
    specifying the interactions. Will pseudobulk the AnnData based on provided 
    ``.obs`` keys, and extract the source gene expression for each pseudobulks, 
    resulting in the interactions data frame having each of its rows copied as 
    many times as there are pseudobulks to report the corresponding expression.
    
    Input
    -----
    adata : ``AnnData``
        The object to use for pseudobulk construction, with raw counts in ``.X`` 
        stored as a sparse matrix.
    obs_to_bulk : list of ``str``
        Column names in ``.obs`` to use to construct the pseudobulks. A pseudobulk 
        will be constructed for each unique value combination of the columns.
    interactions : ``pd.DataFrame``
        Must have at least two columns - the ``source_col``, specifying the gene 
        in the reference to extract the expression for, and a target gene column 
        for later use. All contents will be retained in the output.
    source_col : ``str``, optional (default: ``"source"``)
        The column in ``interactions`` to use to extract the relevant source gene 
        expression from the reference. Interactions where the gene is absent from 
        the object will be omitted.
    """
    # compute pseudobulk based on provided obs to bulk on
    pdata = category_pseudobulker(adata, obs_to_bulk=obs_to_bulk)
    # this is to have been raw counts, turn to log1p-normalised expression
    sc.pp.normalize_total(pdata, target_sum=1e4)
    sc.pp.log1p(pdata)
    sc.pp.scale(pdata, max_value=10)

    # check which of our interactions have their source present in the object
    interactions_present = interactions.loc[np.isin(interactions[source_col], pdata.var_names), :]
    # notify the user if some went missing
    if interactions_present.shape[0] < interactions.shape[0]:
        print("Skipping "+str(interactions.shape[0]-interactions_present.shape[0])+" interactions with ``source_col`` absent from ``adata.var_names``")
    # pull out the expression of the source genes from the pseudobulk object
    signatures = pdata[:, interactions_present[source_col].unique()].to_df()
    # melt the data frame, i.e. turn it from an MxN data frame of expression values
    # to an M*N list where each row has the pseudobulk name, the source gene, and the expression
    # the .reset_index() forces the pseudobulk names back into the data frame as a column
    # where the id_vars="index" picks it up to use for the melt
    signatures = pd.melt(signatures.reset_index(), id_vars = 'index')
    # rename the columns for consistent nomenclature
    signatures.columns = ["population", source_col, "expression"]
    # merge in the stuff from interactions
    # pandas is smart enough to find the source_col overlap in both of them
    # and then makes extra copies of signatures rows where there's one source_col
    # in the interactions, with multiple targets present
    # end result of all this - a data frame with n_pseudobulks*n_interactions rows
    signatures = signatures.merge(interactions)
    # turn the useless index to string to avoid downstream annoyance
    signatures.index = signatures.index.map(str)
    return signatures

def compute_cell_scores(adata, signatures, source_col="source", target_col="target", exp=True):
    """
    Take the query AnnData object and a set of signatures from ``construct_signatures()``, 
    retrieving the target gene expression for each interaction from the query and multiplying 
    it by the source expression from the signature to obtain cell scores. Yields a MuData 
    object with the original query as ``"rna"`` and the computed cell scores as 
    ``"cell_scores"``.
    
    Input
    -----
    adata : ``AnnData``
        The query object, with log-normalised expression in ``.X``.
    signatures : ``pd.DataFrame``
        Output of ``construct_signatures()``, must have ``source_col`` and ``target_col`` present.
    source_col : ``str``, optional (default: ``"source"``)
        The ``signatures`` column holding information on the source gene that was used to 
        construct the reference signature. Will be present in the output.
    target_col : ``str``, optional (default: ``"target"``)
        The ``signatures`` column holding information on the target gene, the expression of 
        which is to be extracted from the query. Will be present in the output.
    exp : ``bool``, optional (default: ``True``)
        Whether to use the exponential formulation or simple multiplication.
    """
    # check which of the signatures actually have the target gene in the query
    signatures_present = signatures.loc[np.isin(signatures[target_col], adata.var_names), :]
    # notify the user if some went missing
    if signatures_present.shape[0] < signatures.shape[0]:
        print("Skipping "+str(signatures.shape[0]-signatures_present.shape[0])+" signatures with ``target_col`` absent from ``adata.var_names``")
    # set up an AnnData object based on the query
    # where the vars are exactly in the order of target_col genes
    # it's fine for there to be duplicates, it doesn't matter
    bdata = adata[:, signatures_present[target_col]]
    # at this point our bdata matches the signatures_present data frame in shape
    # override its var with this data frame's relevant columns
    bdata.var = signatures_present[["population",source_col,target_col]]
    # are we sparse or dense?
    if scipy.sparse.issparse(bdata.X):
        # post-multiplying by a diagonal matrix multiplies each column by value
        # https://solitaryroad.com/c108.html
        # each column is a target gene, we want to multiply it by the corresponding source expression
        bdata.X = bdata.X.dot(scipy.sparse.diags(signatures_present["expression"].values))
    else:
        # we're dense, just multiply the values per column outright
        if exp:
            bdata.X = np.exp(bdata.X + signatures_present["expression"].values[None, :])  # original exponential version
        else:
            bdata.X = bdata.X * signatures_present["expression"].values[None, :]
    # initialise our MuData and stuff the things in there
    mdata = mudata.MuData({"rna":adata, "cell_scores":bdata})
    return mdata


def collapse_scores(mdata, var_to_bulk, score_key="cell_scores", collapse_key="collapsed"):
    """
    Average out scores across unique combinations of categories from ``.var``. Will be 
    inserted into the input MuData object under a new key.
    
    Input
    -----
    mdata : ``MuData``
        Needs to have an object under ``score_key`` to take and use.
    var_to_bulk : list of ``str``
        Column names in the object's ``.var`` to use to compute the score averages. Each 
        unique combination of values of the keys will be represented. Will  be present as 
        ``.var`` columns in the resulting object.
    score_key : ``str``, optional (default: ``"cell_scores"``)
        The ``mdata`` field to use to retrieve the scores to average out.
    collapse_key : ``str``, optional (default: ``"collapsed"``)
        The ``mdata`` field to store the averaged out scores into.
    """
    # our pseudobulker is written to compute pseudobulks on obs
    # not a worry, we can transpose the input and then the output
    # and this way we can perform pseudobulks on var instead
    bdata = category_pseudobulker(mdata[score_key].T, obs_to_bulk=var_to_bulk, compute_mean=True).T
    # we don't care about the cell count thing, avoid cluttering output
    del bdata.var["cell_count"]
    # stash in object
    mdata.mod[collapse_key] = bdata


import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap


def compute_top_interactions_one_group(
    mdata,
    modality: str,
    identity: str,
    *,
    var_key: str = "population",                      # e.g. "tissue" or "population"
    split_obs_key: str = "Manually_curated_celltype", # obs column holding cell type
    group_identity: str,
    top_n: int = 10,
) -> pd.Series:
    """
    Compute top-N ligand–receptor interactions (by mean within one cell type).

    Returns
    -------
    pd.Series
        Index: interaction labels ('source–target' or fallback).
        Values: mean scores within the selected group (descending).
    """
    if group_identity is None:
        raise ValueError("Provide group_identity (the single cell type to show).")

    adata = mdata[modality]
    if var_key not in adata.var:
        raise ValueError(f"'{var_key}' not found in .var of '{modality}'.")
    if split_obs_key not in adata.obs:
        raise ValueError(f"'{split_obs_key}' not found in .obs of '{modality}'.")

    # Select features by identity (e.g., tissue == 'Liver')
    feat_mask = adata.var[var_key] == identity
    if not np.any(feat_mask):
        raise ValueError(f"No features where {var_key!r} == {identity!r} in '{modality}'.")
    adata_f = adata[:, feat_mask]

    # Cells: one group
    cell_mask = (adata_f.obs[split_obs_key].astype(str).values == str(group_identity))
    if not np.any(cell_mask):
        raise ValueError(f"No cells with {split_obs_key} == {group_identity!r}.")
    adata_g = adata_f[cell_mask, :]

    # Labels
    vdf = adata_g.var
    if {"source", "target"}.issubset(vdf.columns):
        labels = (vdf["source"].astype(str) + "–" + vdf["target"].astype(str)).values
    elif "target" in vdf:
        labels = vdf["target"].astype(str).values
    else:
        labels = np.array([vn.split(":", 1)[-1] for vn in adata_g.var_names])
    vname_to_label = dict(zip(adata_g.var_names, labels))

    # Data → means
    X = adata_g.X.toarray() if hasattr(adata_g.X, "toarray") else adata_g.X
    df = pd.DataFrame(X, columns=adata_g.var_names)
    means = df.mean(axis=0)
    means.index = means.index.map(vname_to_label)  # index now 'source–target' (or fallback)
    means = means.groupby(level=0).mean()          # collapse if multiple vars map to same label

    # Top-N
    top = means.sort_values(ascending=False).head(min(top_n, len(means)))
    if top.empty:
        raise ValueError("No data after filtering.")
    return top


def plot_top_interactions_bar_one_group(
    top_series: pd.Series,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = "Reds",
    show: bool = True,
):
    """
    Plot a horizontal bar chart from the Series produced by compute_top_interactions_one_group.

    Parameters
    ----------
    top_series : pd.Series
        Index: labels for bars. Values: mean score.
    """
    if not isinstance(top_series, pd.Series) or top_series.empty:
        raise ValueError("top_series must be a non-empty pd.Series.")

    # Colors by mean
    cmap_obj = get_cmap(cmap)
    vmin, vmax = float(top_series.min()), float(top_series.max())
    if vmin == vmax:
        bar_colors = [cmap_obj(0.5)] * len(top_series)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        bar_colors = [cmap_obj(norm(v)) for v in top_series.values]

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        x=top_series.values,
        y=top_series.index.tolist(),
        orient="h",
        palette=bar_colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Mean score")
    ax.set_ylabel("")
    ax.set_title(title or f"Top {len(top_series)} ligand–receptor pairs")
    plt.tight_layout()
    if show:
        plt.show()
        return None
    return ax

