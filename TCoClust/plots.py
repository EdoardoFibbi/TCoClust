import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_daq as daq

from .config import *
from .poisson_utils import poisson_cell_posterior, poisson_cell_residual
from .utils import block_sort, get_row_col_outliers, part_matrix_to_dict

if TYPE_CHECKING:
    from .class_defs import TccResult


def _highlight_cell(x: float,
                    y: float,
                    ax: plt.Axes = None,
                    **kwargs) -> plt.Rectangle:

    """ highlight_cell is a utility function to highlight cells in plot_coclust or imshow.
        It basically draws a rectangle around the cell located at (x,y)

        Parameters
        ----------
        x : float
            First coordinate of cell to be highlighted
        y : float
            Second coordinate of cell to be highlighted
        ax : matplotlib.Axes (optional)
            If a Matplotlib Axis object is passed, it is used for plotting.
            defaults to :code:`None`.
        **kwargs :
            kwargs of plt.Rectangle.

        Returns
        -------
        rect : matplotlib.patches.Rectangle
            Rectangle around cell with specified location.

    """

    rect = plt.Rectangle((x - .5, y - .5), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def outlier_plots(result,
                  result0=None,
                  row_names=None,
                  col_names=None):

    """ outlier_plots produces a figure containing 2x2 subplots for outlier diagnostics, combining the plost produced by
    cell_posterior_plots and cell_residual_plots. Plots are produced with Plotly Express and are interactive.
    For more details about the plots' nature refer to the accompanying paper.
    [add citation in future versions]

    Parameters
    ----------
    result : TccResult
        Result object outputted by co-clustering procedure.
    result0 : TccResult or None (optional)
        Result object outputted by co-clustering procedure when no trimming is applied.
        If :code:`None`, then it is computed by the function (N.B.: this may require some time).
        Defaults to :code:`None`.
    row_names : list of str or None (optional)
        Names to be assigned to the rows. It must have length equal to the number of rows of X (n).
        If not provided, generic row labels will be used.
        Defaults to :code:`None`.
    col_names : list of str or None (optional)
        Names to be assigned to the columns. It must have length equal to the number of columns of X (p).
        If not provided, generic column labels will be used.
        Defaults to :code:`None`.

    Returns
    -------
    A Plotly Express figure containing four subplots

    """

    # deferred import to avoid circular dependency causing 'Import Error'
    from .cell_tricc import cell_tbsem

    X = result.Input.X
    n, p = X.shape
    # if row_names / col_names are not given, then use row / column indices (starting from 1) as labels
    if row_names is None:
        row_names = ["R" + str(row_) for row_ in range(1, n + 1)]
    if col_names is None:
        col_names = ["C" + str(col_) for col_ in range(1, p + 1)]

    # if not provided, perform non-robust co-clustering of the same data matrix, with same options (but no trimming)
    if result0 is None:
        # N.B.: depending on the data and options, this may take a while
        result0 = cell_tbsem(X, result.Input.g, result.Input.m, density=result.Input.density, alpha=0, beta=0,
                             n_init=result.Input.n_init, t_max=result.Input.t_max, t_burn_in=result.Input.t_burn_in,
                             until_converged=result.Input.until_converged, trim_quantile=result.Input.trim_quantile,
                             trim_global=result.Input.trim_global, seed=result.Input.seed)

    # -------- #
    # PLOTTING #
    # -------- #

    figure1_traces, figure2_traces = cell_posterior_plots(result,
                                                          result0,
                                                          row_names,
                                                          col_names,
                                                          in_outlier_plots=True)
    figure3_traces, figure4_traces = cell_residual_plots(result,
                                                         result0,
                                                         row_names,
                                                         col_names,
                                                         in_outlier_plots=True)

    all_traces = [figure1_traces, figure2_traces, figure3_traces, figure4_traces]

    # Create a 2x2 subplot figure
    subplots_fig = sp.make_subplots(rows=2, cols=2)
    x_axis_labels = np.array([["Cell log-posterior", "Cell value"], ["Cell residual", "Cell value"]])
    y_axis_labels = np.array([["Robust cell log-posterior"] * 2, ["Robust cell residual"] * 2])

    for figure_num, figure_traces in enumerate(all_traces):
        for traces in figure_traces:
            row_num = int(figure_num / 2)
            col_num = figure_num % 2
            # Add traces to current subplot
            subplots_fig.add_trace(traces, row=row_num + 1, col=col_num + 1)
            # Update axis labels of current subplot
            subplots_fig.update_xaxes(title_text=x_axis_labels[row_num, col_num], row=row_num + 1, col=col_num + 1)
            subplots_fig.update_yaxes(title_text=y_axis_labels[row_num, col_num], row=row_num + 1, col=col_num + 1)

    subplots_fig.update_layout(height=600)
    return subplots_fig


def cell_residual_plots(result: "TccResult",
                        result0: Union["TccResult", None] = None,
                        row_names: Union[list, None] = None,
                        col_names: Union[list, None] = None,
                        in_outlier_plots: bool = False):

    """ cell_residual_plots produces a figure containing 1x2 subplots for outlier diagnostics using cell residuals.
    Plots are produced with Plotly Express and are interactive.
    For more details about the plots' nature refer to the accompanying paper.
    [add citation in future versions]

    result : TccResult
        Result object outputted by co-clustering procedure.
    result0 : TccResult or None (optional)
        Result object outputted by co-clustering procedure when no trimming is applied.
        If :code:`None`, then it is computed by the function (N.B.: this may require some time).
        Defaults to :code:`None`.
    row_names : list of str or None (optional)
        Names to be assigned to the rows. It must have length equal to the number of rows of X (n).
        If not provided, generic row labels will be used.
        Defaults to :code:`None`.
    col_names : list of str or None (optional)
        Names to be assigned to the columns. It must have length equal to the number of columns of X (p).
        If not provided, generic column labels will be used.
        Defaults to :code:`None`.
    in_outlier_plots : bool (optional)
        Whether the output needs to be passed to function `outlier_plots`.
        Defaults to :code:`False`.

    Returns
    -------
    Tuple of two lists of Plotly figure traces if `in_outlier_plots` is True, otherwise returns a Plotly Express figure.

    """
    # deferred import to avoid circular dependency causing 'Import Error'
    from .cell_tricc import cell_tbsem

    X = result.Input.X
    n, p = X.shape
    # if row_names / col_names are not given, then use row / column indices (starting from 1) as labels
    if row_names is None:
        row_names = ["R" + str(row_) for row_ in range(1, n + 1)]
    if col_names is None:
        col_names = ["C" + str(col_) for col_ in range(1, p + 1)]

    # if not provided, perform non-robust co-clustering of the same data matrix, with same options (but no trimming)
    if result0 is None:
        # N.B.: depending on the data and options, this may take a while
        result0 = cell_tbsem(X, result.Input.g, result.Input.m, density=result.Input.density, alpha=0, beta=0,
                             n_init=result.Input.n_init, t_max=result.Input.t_max, t_burn_in=result.Input.t_burn_in,
                             until_converged=result.Input.until_converged, trim_quantile=result.Input.trim_quantile,
                             trim_global=result.Input.trim_global, seed=result.Input.seed)

    # get the partition matrices of the robust and non-robust co-clustering
    Z, W = result.Partitions.values()
    Z1, W1 = result0.Partitions.values()

    # -------- #
    # PLOTTING #
    # -------- #

    x = poisson_cell_residual(X, Z1, W1, result0.BlockParameters, kind="var_stab").ravel()
    y = poisson_cell_residual(X, Z, W, result.BlockParameters, kind="var_stab").ravel()

    df = pd.DataFrame(np.c_[x, y], columns=["Cell residual", "Robust cell residual"])

    df["Cell ID"] = ["(" + row_name + ", " + col_name + ")" for row_name, col_name in product(row_names, col_names)]

    df["Outlyingness"] = "Regular"
    df.loc[(result.M[1] == 0).ravel(), "Outlyingness"] = "Flagged"

    df["Cell value"] = np.nan_to_num(result.Input.X.ravel())

    fig1 = px.scatter(df, x="Cell residual", y="Robust cell residual",
                      color="Outlyingness",
                      opacity=.5,
                      hover_data="Cell ID",
                      )

    fig2 = px.line(x=[1.1 * np.nanmin(x), 1.1 * np.nanmax(x)], y=[1.1 * np.nanmin(x), 1.1 * np.nanmax(x)])
    fig2.update_traces(line=dict(color="gray", dash="dash"))
    fig3 = go.Figure(data=fig1.data + fig2.data, layout=fig1.layout)

    fig4 = px.scatter(df, x="Cell value", y="Robust cell residual",
                      color="Outlyingness",
                      opacity=.5,
                      hover_data="Cell ID",
                      )

    # For as many traces that exist per Express figure, get the traces from each plot and store them in an array.
    # This is essentially breaking down the Express fig into it's traces
    figure1_traces = []
    figure2_traces = []
    for trace in range(len(fig3["data"])):
        figure1_traces.append(fig3["data"][trace])
    for trace in range(len(fig4["data"])):
        fig4["data"][trace]["showlegend"] = False
        figure2_traces.append(fig4["data"][trace])

    # Create a 1x2 subplot
    subplots_fig = sp.make_subplots(rows=1, cols=2)

    for traces in figure1_traces:
        subplots_fig.add_trace(traces, row=1, col=1)
    for traces in figure2_traces:
        subplots_fig.add_trace(traces, row=1, col=2)

    subplots_fig.update_xaxes(title_text="Cell residual", row=1, col=1)
    subplots_fig.update_xaxes(title_text="Cell value", row=1, col=2)

    subplots_fig.update_yaxes(title_text="Robust cell residual", row=1, col=1)
    subplots_fig.update_yaxes(title_text="Robust cell residual", row=1, col=2)

    # subplots_fig.update_layout(template="simple_white")

    if in_outlier_plots:
        return figure1_traces, figure2_traces
    return subplots_fig


def cell_posterior_plots(result: "TccResult",
                         result0: Union["TccResult", None] = None,
                         row_names: Union[list, None] = None,
                         col_names: Union[list, None] = None,
                         in_outlier_plots: bool = False):

    """ cell_posterior_plots produces a figure containing 1x2 subplots for outlier diagnostics using cell posteriors.
    Plots are produced with Plotly Express and are interactive.
    For more details about the plots' nature refer to the accompanying paper.
    [add citation in future versions]

    Parameters
    ----------
    result : TccResult
        Result object outputted by co-clustering procedure.
    result0 : TccResult or None (optional)
        Result object outputted by co-clustering procedure when no trimming is applied.
        If :code:`None`, then it is computed by the function (N.B.: this may require some time).
        Defaults to :code:`None`.
    row_names : list of str or None (optional)
        Names to be assigned to the rows. It must have length equal to the number of rows of X (n).
        If not provided, generic row labels will be used.
        Defaults to :code:`None`.
    col_names : list of str or None (optional)
        Names to be assigned to the columns. It must have length equal to the number of columns of X (p).
        If not provided, generic column labels will be used.
        Defaults to :code:`None`.
    in_outlier_plots : bool (optional)
        Whether the output needs to be passed to function òutlier_plots`.
        Defaults to :code:`False`.

    Returns
    -------
    Tuple of two lists of Plotly figure traces if `in_outlier_plots` is True, otherwise returns a Plotly Express figure.

    """

    # deferred import to avoid circular dependency causing 'Import Error'
    from .cell_tricc import cell_tbsem

    X = result.Input.X
    n, p = X.shape
    # if row_names / col_names are not given, then use row / column indices (starting from 1) as labels
    if row_names is None:
        row_names = ["R" + str(row_) for row_ in range(1, n + 1)]
    if col_names is None:
        col_names = ["C" + str(col_) for col_ in range(1, p + 1)]

    # if not provided, perform non-robust co-clustering of the same data matrix, with same options (but no trimming)
    if result0 is None:
        # N.B.: depending on the data and options, this may take a while
        result0 = cell_tbsem(X, result.Input.g, result.Input.m, density=result.Input.density, alpha=0, beta=0,
                             n_init=result.Input.n_init, t_max=result.Input.t_max, t_burn_in=result.Input.t_burn_in,
                             until_converged=result.Input.until_converged, trim_quantile=result.Input.trim_quantile,
                             trim_global=result.Input.trim_global, seed=result.Input.seed)

    # get the partition matrices of the robust and non-robust co-clustering
    Z, W = result.Partitions.values()
    Z1, W1 = result0.Partitions.values()

    # -------- #
    # PLOTTING #
    # -------- #

    x = np.log(
        np.maximum(
            poisson_cell_posterior(X, Z1, W1, result0.BlockParameters).ravel(),
            0
        )
    )
    y = np.log(
        np.maximum(
            poisson_cell_posterior(X, Z, W, result.BlockParameters).ravel(),
            0
        )
    )

    df = pd.DataFrame(np.c_[x, y], columns=["Cell log-posterior", "Robust cell log-posterior"])

    df["Cell ID"] = ["(" + row_name + ", " + col_name + ")" for row_name, col_name in product(row_names, col_names)]

    df["Outlyingness"] = "Regular"
    df.loc[(result.M[1] == 0).ravel(), "Outlyingness"] = "Flagged"

    df["Cell value"] = np.nan_to_num(result.Input.X.ravel())

    fig1 = px.scatter(df, x="Cell log-posterior", y="Robust cell log-posterior",
                      color="Outlyingness",
                      opacity=.5,
                      hover_data="Cell ID",
                      )

    fig2 = px.line(x=[1.1 * np.nanmin(x), 1.1 * np.nanmax(x)], y=[1.1 * np.nanmin(x), 1.1 * np.nanmax(x)])
    fig2.update_traces(line=dict(color="gray", dash="dash"))
    fig3 = go.Figure(data=fig1.data + fig2.data, layout=fig1.layout)

    fig4 = px.scatter(df, x="Cell value", y="Robust cell log-posterior",
                      color="Outlyingness",
                      opacity=.5,
                      hover_data="Cell ID",
                      )

    # For as many traces that exist per Express figure, get the traces from each plot and store them in an array.
    # This is essentially breaking down the Express fig into its traces
    figure1_traces = []
    figure2_traces = []
    for trace in range(len(fig3["data"])):
        if in_outlier_plots:
            fig3["data"][trace]["showlegend"] = False
        figure1_traces.append(fig3["data"][trace])
    for trace in range(len(fig4["data"])):
        fig4["data"][trace]["showlegend"] = False
        figure2_traces.append(fig4["data"][trace])

    # Create a 1x2 subplot
    subplots_fig = sp.make_subplots(rows=1, cols=2)

    for traces in figure1_traces:
        subplots_fig.add_trace(traces, row=1, col=1)
    for traces in figure2_traces:
        subplots_fig.add_trace(traces, row=1, col=2)

    subplots_fig.update_xaxes(title_text="Cell log-posterior", row=1, col=1)
    subplots_fig.update_xaxes(title_text="Cell value", row=1, col=2)

    subplots_fig.update_yaxes(title_text="Robust cell log-posterior", row=1, col=1)
    subplots_fig.update_yaxes(title_text="Robust cell log-posterior", row=1, col=2)

    if in_outlier_plots:
        return figure1_traces, figure2_traces
    return subplots_fig


def plot_coclust(X: np.ndarray,
                 Z: np.ndarray,
                 W: np.ndarray,
                 row_names: Union[list, None] = None,
                 col_names: Union[list, None] = None,
                 vmin: float = 0,
                 vmax: float = 5,
                 figsize: Union[Tuple[float, float], None] = None,
                 cmap: str = "viridis",
                 partition_lines_color: str = "red",
                 colorbar: bool = False,
                 transpose: bool = False,
                 keep_aspect: bool = False,
                 return_graphics_objects: bool = False,
                 fig: Union[plt.figure, None] = None,
                 ax: Union[plt.axis, None] = None) \
        -> Union[tuple[plt.figure, plt.axis, plt.colorbar], tuple[plt.figure, plt.axis]]:

    """ plot_coclust is a utility function to plot trimmed co-clustering results.
        It allows the user to modify some parameters of the plot, using the same keywords as Matplotlib.
        If a finer control over the graphics settings is needed, plot_coclust can return Matplotlib's
        "Figure", "Axis" and "Colorbar" objects, so that the user can modify the output to their liking.

        Parameters
        ----------
        X : np.ndarray
            Original data matrix (i.e., not yet sorted according to co-clusters)
        Z : np.ndarray
            Row partition matrix
        W : np.ndarray
            Column partition matrix
        row_names : list of str (optional)
            Names to be assigned to the rows. It must have length equal to the number of rows of X (n).
            The list is assumed to be ordered according to the original matrix and will be sorted inside
            plot_coclust according to the given row partition.
            defaults to :code:`None`
        col_names : list of str (optional)
            Names to be assigned to the columns. It must have length equal to the number of columns of X (p).
            The list is assumed to be ordered according to the original matrix and will be sorted inside
            plot_coclust according to the given column partition.
            defaults to :code:`None`
        vmin : float (optional)
            See matplotlib.pyplot.imshow
        vmax : float (optional)
            See matplotlib.pyplot.imshow
        figsize : tuple (float, float) (optional)
            See matplotlib.pyplot.figure
        cmap : str (optional)
            See matplotlib.pyplot.imshow
            Defaults to "viridis"
        partition_lines_color : str (optional)
            Color for the lines delimiting the row and column partitions. Must be a valid matplotlib color.
            Defaults to "red"
        colorbar : bool (optional)
            If True a colorbar is added to the right of the plot.
            Defaults to :code:`False`
        transpose : bool (optional)
            If True the transposed matrix is plotted.
            Defaults to :code:`False`
        keep_aspect : bool (optional)
            If True preserves aspect ratio of plotted matrix, otherwise the aspect ratio is determined by figsize.
            Defaults to :code:`False`
        return_graphics_objects : bool (optional)
            If True "Figure", "Axis" and "Colorbar" matplotlib objects are returned.
            Defaults to :code:`False`
        fig : matplotlib.Figure (optional)
            If a Matplotlib Figure object is passed, it is used for plotting.
            defaults to :code:`None`
        ax : matplotlib.Axes (optional)
            If a Matplotlib Axis object is passed, it is used for plotting.
            defaults to :code:`None`

        Returns
        -------
        If return_graphics_objects is True:
            fig : matplotlib.figure
            ax : matplotlib.axes
            bar : matplotlib.colorbar
        Else, nothing is returned.

    """

    n, p = X.shape
    g = Z.shape[1]
    m = W.shape[1]

    trimmed_rows = not np.all(Z.sum(axis=1))
    trimmed_cols = not np.all(W.sum(axis=1))

    row_partition_fit = part_matrix_to_dict(Z, get_row_col_outliers(Z))
    col_partition_fit = part_matrix_to_dict(W, get_row_col_outliers(W))

    if row_names and not col_names:
        X_sort_fit, row_names_sorted = block_sort(X, row_partition_fit, col_partition_fit, row_names)
    if not row_names and col_names:
        X_sort_fit, col_names_sorted = block_sort(X, row_partition_fit, col_partition_fit, col_names)
    if row_names and col_names:
        X_sort_fit, row_names_sorted, col_names_sorted = block_sort(X, row_partition_fit, col_partition_fit, row_names,
                                                                    col_names)
    else:
        X_sort_fit = block_sort(X, row_partition_fit, col_partition_fit)
        row_names_sorted = []
        col_names_sorted = []

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if keep_aspect:
        aspect = "equal"
    else:
        aspect = "auto"

    if transpose:
        ax.set(title="Transposed sorted data matrix according to Trimmed LBM fit")
        plot = ax.imshow(X_sort_fit.T, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect=aspect)
        ax.set_ylim([p - .5, -.5])
        ax.set_xlim([-.5, n - .5])
        offset = 0
        for l in range(m):
            if l > 0:
                offset += np.sum(W[:, l - 1])
            if (l == m - 1) and trimmed_cols:
                color = "m"
                linewidth = 2.
                linestyle = "--"
            elif l == m - 1:
                break
            else:
                color = partition_lines_color
                linewidth = 1.5
                linestyle = "-"
            ax.plot(np.linspace(-.5, n - .5, n), np.sum(W[:, l]) * np.ones(n) + offset - .5,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle)
        offset = 0
        for k in range(g):
            if k > 0:
                offset += np.sum(Z[:, k - 1])
            if (k == g - 1) and trimmed_rows:
                color = "m"
                linewidth = 2.
                linestyle = "--"
            elif k == g - 1:
                break
            else:
                color = partition_lines_color
                linewidth = 1.5
                linestyle = "-"
            ax.plot(np.sum(Z[:, k]) * np.ones(p) + offset - .5, np.linspace(-.5, p - .5, p),
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle)

        if colorbar:
            bar = plt.colorbar(plot, location="right")
        ax.set(xticks=range(len(row_names_sorted)), xticklabels=row_names_sorted)
        ax.set(yticks=range(len(col_names_sorted)), yticklabels=col_names_sorted)

        # color in red tick labels corresponding to outliers
        if col_names:
            for i in range(len(get_row_col_outliers(Z))):
                ax.get_xticklabels()[-i - 1].set_color("red")

        if row_names:
            for j in range(len(get_row_col_outliers(W))):
                ax.get_yticklabels()[-j - 1].set_color("red")

    else:
        ax.set(title="Sorted data matrix according to Trimmed LBM fit")
        plot = ax.imshow(X_sort_fit, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect=aspect)
        ax.set_xlim([-.5, p - .5])
        ax.set_ylim([n - .5, -.5])
        offset = 0
        for l in range(g):
            if l > 0:
                offset += np.sum(Z[:, l - 1])
            if (l == g - 1) and trimmed_cols:
                color = "m"
                linewidth = 2.
                linestyle = "--"
            elif l == g - 1:
                break
            else:
                color = partition_lines_color
                linewidth = 1.5
                linestyle = "-"
            ax.plot(np.linspace(-.5, p - .5, p), np.sum(Z[:, l]) * np.ones(p) + offset - .5,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle)
        offset = 0
        for k in range(m):
            if k > 0:
                offset += np.sum(W[:, k - 1])
            if (k == m - 1) and trimmed_rows:
                color = "m"
                linewidth = 2.
                linestyle = "--"
            elif k == m - 1:
                break
            else:
                color = partition_lines_color
                linewidth = 1.5
                linestyle = "-"
            ax.plot(np.sum(W[:, k]) * np.ones(n) + offset - .5, np.linspace(-.5, n - .5, n),
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle)

        if colorbar:
            bar = plt.colorbar(plot, location="right")
        ax.set(xticks=range(len(col_names_sorted)), xticklabels=col_names_sorted)
        ax.set(yticks=range(len(row_names_sorted)), yticklabels=row_names_sorted)

        # color in red tick labels corresponding to rowwise and columnwise outliers
        if col_names:
            for i in range(len(get_row_col_outliers(Z))):
                ax.get_yticklabels()[-i - 1].set_color("red")

        if row_names:
            for j in range(len(get_row_col_outliers(W))):
                ax.get_xticklabels()[-j - 1].set_color("red")

    if return_graphics_objects:
        if colorbar:
            return fig, ax, bar
        else:
            return fig, ax
    else:
        plt.show()


def coclust_treemap(result: "TccResult",
                    row_names: Union[list, None] = None,
                    col_names: Union[list, None] = None):

    """ coclust_treemap is a Dash app displaying an interactive treemap representation of a co-clustering.
    Along the treemap, the app presents sliders and a switch to filter the data according to various features.

    Parameters
    ----------
    result : TccResult
        Result object outputted by co-clustering procedure.
    row_names : list of str (optional)
        Names to be assigned to the rows. It must have length equal to the number of rows of X (n).
        If not provided, generic row labels will be used.
        Defaults to :code:`None`.
    col_names : list of str (optional)
        Names to be assigned to the columns. It must have length equal to the number of columns of X (p).
        If not provided, generic column labels will be used.
        Defaults to :code:`None`.

    """

    # lazy import
    from .utils import part_matrix_to_dict

    # ------------------------ #
    #   Preparing input data   #
    # ------------------------ #

    X = result.Input.X
    n, p = X.shape
    # if row_names / col_names are not given, then use row / column indices (starting from 1) as labels
    if row_names is None:
        row_names = ["R" + str(row_) for row_ in range(1, n + 1)]
    if col_names is None:
        col_names = ["C" + str(col_) for col_ in range(1, p + 1)]

    #   df field                |       availability       |    notes                              #
    # --------------------------|--------------------------|-------------------------------------- #
    #   Count                   |       always             |                                       #
    #   Lambda                  |       always             |                                       #
    #   Block_ranking           |       always             |    Computed at function runtime       #
    #   Block_ranking_long      |       always             |    idem                               #
    #   Flagged                 |       always             |                                       #
    #   Row_name                |       optional           |    If not given, use default names    #
    #   Column_name             |       optional           |    idem                               #

    df = pd.DataFrame(columns=["Count",
                               "Lambda",
                               "Block_ranking",
                               "Block_ranking_long",
                               "Flagged",
                               "Row_name",
                               "Column_name",
                               ]
                      )

    # the field "Count" contains the cell values from X
    df["Count"] = X.ravel()

    #  the field "Flagged" can be obtained similarly from the mask matrix
    df["Flagged"] = (result.M[1].ravel() == 0)

    # row and column names to identify every cell (must be expanded since X values are a column of df)
    col_names_expanded, row_names_expanded = np.meshgrid(col_names, row_names)
    df["Row_name"] = row_names_expanded.ravel()
    df["Column_name"] = col_names_expanded.ravel()
    df.loc[df["Flagged"], "Column_name"] += "*"

    # get recovered partitions and appropriately index the estimated block parameters to add "Lambda" to df
    Z, W = result.Partitions.values()
    z, w = part_matrix_to_dict(Z), part_matrix_to_dict(W)
    expanded_group_labels = np.array(list(product(list(z.values()), list(w.values()))))
    I, J = expanded_group_labels[:, 0], expanded_group_labels[:, 1]
    df["Lambda"] = result.BlockParameters[I, J]

    # compute and assign block ranking
    block_ranking = np.argsort(np.argsort(result.BlockParameters, axis=None)[::-1], axis=None).reshape(
        result.BlockParameters.shape) + 1
    df["Block_ranking"] = block_ranking[I, J]
    df["Block_ranking_long"] = "Block #" + df["Block_ranking"].astype("str")

    filtered_df = df.query("Count > 0")

    app = Dash(__name__)

    # --------------------- #
    #   Define app layout   #
    # --------------------- #

    app.layout = html.Div(

        children=[

            html.Div(
                children=[

                    html.Div(
                        children=[

                            html.H4(
                                children="Top k-ranked blocks",
                                style={'textAlign': 'left'}
                            ),

                            # slider to filter entire blocks based on ranking
                            dcc.Slider(min=1,
                                       max=df.Block_ranking.max(),
                                       step=1,
                                       value=4,
                                       id="topk_slider"
                                       ),
                        ],

                        style={

                        }
                    ),

                    html.Div(
                        children=[

                            html.H4(
                                children="Range of cell counts",
                                style={'textAlign': 'left'}
                            ),

                            # range (double-ended) slider to filter leaves based on cell values
                            dcc.RangeSlider(min=0,
                                            max=df.Count.max(),
                                            step=1,
                                            value=[0, df.Count.max()],
                                            marks=None,
                                            tooltip={
                                                "placement": "top",
                                                "always_visible": True,
                                            },
                                            id="signals_slider"
                                            ),
                        ],

                        style={

                        }
                    )
                ],

                style={
                    'width': '100%',
                    'vertical-align': 'top'
                }
            ),

            html.Div(
                children=[

                    html.H4(
                        children="Show flagged only",
                        style={'textAlign': 'left'}
                    ),

                    # switch to show only flagged cells
                    daq.BooleanSwitch(on=False,
                                      labelPosition="top",
                                      id="flagged_switch"
                                      ),
                ],

                style={
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'width': '15%'
                }
            ),

            html.Div(
                children=[
                    dcc.Graph(id="treemap-plot")
                ],

                style={
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'width': '80%'
                }
            ),

        ],

        style={
            'width': '100%',
            'vertical-align': 'top',
            'font-family': 'Verdana',
        }
    )

    # ---------------------------------- #
    #   Define callback to update plot   #
    # ---------------------------------- #

    # using Dash's app.callback decorator
    @app.callback(
        Output("treemap-plot", "figure"),
        [
            Input("topk_slider", "value"),
            Input("signals_slider", "value"),
            Input("flagged_switch", "on")
        ]
    )
    def update_treemap(slider_value, range_slider_value, switch_on):
        df_to_plot = filtered_df.copy()
        df_to_plot = df_to_plot.query(f"Block_ranking <= {slider_value}")
        df_to_plot = df_to_plot.query(f"{range_slider_value[0]} <= Count <= {range_slider_value[1]}")

        if switch_on:
            df_to_plot = df_to_plot[df_to_plot["Flagged"]]

        fig = px.treemap(df_to_plot,
                         path=[px.Constant(f"All top {slider_value} blocks"), "Block_ranking_long", "Row_name",
                               "Column_name"],
                         values="Count",
                         color="Lambda",
                         color_continuous_scale="Blues",
                         color_continuous_midpoint=np.average(filtered_df["Lambda"]),
                         range_color=(0, df["Lambda"].max())
                         )

        fig.update_layout(margin=dict(t=50, l=20, r=0, b=20),
                          coloraxis_colorbar=dict(title="Estimated 𝜆", ),
                          font_family="Verdana",
                          )

        return fig

    app.run(debug=False, jupyter_mode="tab")
