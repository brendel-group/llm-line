import itertools
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# import fonts
for font in fm.findSystemFonts("fonts", "ttf"):
    fm.fontManager.addfont(font)

# we're designing all figures at 2x size
COLUMNWIDTH = 234.8775 / 72 * 2  # in inches
FULLWIDTH = 487.8225 / 72 * 2  # in inches

CHANCE_BY_BASE_NAME = {
    "hellaswag": 0.25,
    "piqa": 0.5,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "commonsense_qa": 0.2,
    "openbookqa": 0.25,
    "winogrande": 0.5,
    "copa": 0.5,
    "social_iqa": 1 / 3,
    "mmlu": 0.25,
}


def setup_style():
    sb.reset_defaults()
    sb.set_theme(style="whitegrid")
    sb.set_context("paper", font_scale=1.5)
    plt.rcParams["font.family"] = "Roboto"

    colors = sb.color_palette("deep")
    markers = ["o", "s", "d", "^", "v", "<", ">", "P", "x", "*", "h", "H"]
    return colors, markers


def plot_l2l_base(
    df: pd.DataFrame,
    arch: str,
    pretraining: str,
    tokenizer: str,
    save_path: Optional[str] = None,
):
    df = df.copy()
    df = df[df["Architecture"] == arch]
    df = df[df["Pretraining Data"] == pretraining]
    df = df[df["Tokenizer"] == tokenizer]
    print(df["Name"].unique())

    colors, _ = setup_style()

    fig = plt.figure(figsize=(FULLWIDTH, 4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.1, 1, 1], wspace=0.1)

    # Create axes with shared y-axis for subplots 2 and 3
    ax1 = fig.add_subplot(gs[0])  # First plot
    ax_legend1 = fig.add_subplot(gs[1])  # Space for first legend
    ax2 = fig.add_subplot(gs[2])  # Second plot
    ax3 = fig.add_subplot(gs[3], sharey=ax2)  # Third plot sharing y with second

    # Store axes in list for easier referencing
    axs = [ax1, ax2, ax3]

    # Hide the legend subplot (it's just for spacing)
    ax_legend1.set_visible(False)

    # Hide y-axis labels for the third subplot since they're shared
    plt.setp(ax3.get_yticklabels(), visible=False)

    axs[0].set_title("Validation-to-Validation", weight="bold")
    axs[0].set_xlabel("C4 Validation Loss")
    axs[0].set_ylabel("Validation Loss")
    val_sets = ["The Pile UC", "FineWeb-EDU", "RefineWeb", "Slimpajama"]
    axs[0].scatter(
        df["C4 Validation Loss"],
        df[[f"{val_set} Validation Loss" for val_set in val_sets]].mean(axis=1),
        color="black",
        s=5,
    )
    for color, val_set in zip(colors, val_sets):
        axs[0].scatter(
            df["C4 Validation Loss"],
            df[f"{val_set} Validation Loss"],
            color=color,
            s=1,
        )
    construct_legend(
        axs[0],
        [(["Average", *val_sets], [(0, 0, 0), *colors])],
        title="Validation Set",
        loc="center left",
        bbox_to_anchor=(1, 0.5),  # Position in the dedicated legend space
        frameon=False,
    )

    axs[1].set_title("Validation-to-Test", weight="bold")
    axs[1].set_xlabel("C4 Validation Loss")
    axs[1].set_ylabel("Test Loss")
    test_sets = [
        "ARC-Challenge",
        "ARC-Easy",
        "OpenBookQA",
        "PIQA",
        "COPA",
        "Winogrande",
        "HellaSwag",
    ]
    axs[1].scatter(
        df["C4 Validation Loss"],
        df[[f"{test_set} Loss" for test_set in test_sets]].mean(axis=1),
        color="black",
        s=5,
    )
    for color, test_set in zip(colors, test_sets):
        axs[1].scatter(
            df["C4 Validation Loss"],
            df[f"{test_set} Loss"],
            color=color,
            s=1,
        )

    axs[2].set_title("Test-to-Test", weight="bold")
    axs[2].set_xlabel("Hellaswag Loss")
    test_sets = [
        "ARC-Challenge",
        "ARC-Easy",
        "OpenBookQA",
        "PIQA",
        "COPA",
        "Winogrande",
    ]
    axs[2].scatter(
        df["HellaSwag Loss"],
        df[[f"{test_set} Loss" for test_set in test_sets]].mean(axis=1),
        color="black",
        s=5,
    )
    for color, test_set in zip(colors, test_sets):
        axs[2].scatter(
            df["HellaSwag Loss"],
            df[f"{test_set} Loss"],
            color=color,
            s=1,
        )
    construct_legend(
        axs[2],
        [(["Average", *test_sets], [(0, 0, 0), *colors])],
        title="Test Set",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )

    # Remove tight_layout since we're using gridspec
    if save_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_path}_{timestamp}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_fit_vs_accuracy(
    df: pd.DataFrame, model: str, pretraining: str, save_path: Optional[str] = None
):
    df = df.copy()
    df = df[df["arch"] == model]
    df = df[df["pretraining_data"] == pretraining]

    acc_columns = [col for col in df.columns if "/acc" in col]
    acc_df = df[acc_columns]
    acc_df = acc_df.melt(var_name="test_set", value_name="accuracy")

    # Set up the data and compute relative accuracy
    acc_df["test_set"] = [col.split("/")[0] for col in acc_df["test_set"]]
    acc_df["base_name"] = [
        "mmlu" if "mmlu" in col else col for col in acc_df["test_set"]
    ]
    acc_df["chance"] = [
        CHANCE_BY_BASE_NAME[base_name] for base_name in acc_df["base_name"]
    ]
    acc_df["relative_accuracy"] = (acc_df["accuracy"] - acc_df["chance"]) / (
        1 - acc_df["chance"]
    )

    # Sort the data by relative accuracy
    acc_df = acc_df.groupby("test_set")["relative_accuracy"].max()
    acc_df = acc_df.sort_values(ascending=False).reset_index()
    acc_df["base_name"] = [
        "mmlu" if "mmlu" in col else col for col in acc_df["test_set"]
    ]
    acc_df["fit"] = np.nan

    hellaswag_loss = df["hellaswag/loss"].values.reshape(-1, 1)

    # Fit a linear regression model for each test set
    for test_set in acc_df["test_set"].unique():
        try:
            loss = df[f"{test_set}/loss"]
        except KeyError:
            continue

        model = LinearRegression().fit(
            hellaswag_loss,
            loss,
        )
        fit_score = model.score(
            hellaswag_loss,
            loss,
        )
        acc_df.loc[acc_df["test_set"] == test_set, "fit"] = fit_score

    acc_df.dropna(inplace=True)

    grouped_data = acc_df.groupby("base_name")

    _, _ = setup_style()

    plt.figure(figsize=(COLUMNWIDTH, 4), constrained_layout=True)
    for base_name, group_df in grouped_data:
        plt.scatter(
            group_df["relative_accuracy"],
            group_df["fit"],
            label=base_name,
        )
    plt.xlabel("Relative Accuracy")
    plt.ylabel("R² of Linear")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    if save_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_path}_{timestamp}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_intervention_arch(
    df: pd.DataFrame, save_path: Optional[str] = None, avg_only: bool = False
):
    df = df.copy()
    archs = sorted(df["arch"].unique())

    colors, markers = setup_style()

    if avg_only:
        fig, axs = plt.subplots(1, 2, figsize=(COLUMNWIDTH, 4), constrained_layout=True)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(FULLWIDTH, 4), constrained_layout=True)

    axs[0].set_title("Validation-to-Validation", weight="bold")
    axs[0].set_xlabel("C4 Validation Loss")
    axs[0].set_ylabel("Validation Loss")
    val_sets = ["pile_uc", "fineweb_edu_100bt", "refineweb", "slimpajama"]
    if avg_only:
        grouped = df.groupby("arch")
        for (arch, group), color in zip(grouped, colors):
            axs[0].scatter(
                group["c4_val_loss"],
                group[[f"{val_set}_val_loss" for val_set in val_sets]].mean(axis=1),
                color=color,
            )
    else:
        for color, val_set in zip(colors, val_sets):
            grouped = df.groupby("arch")
            for (arch, group), marker in zip(grouped, markers):
                axs[0].scatter(
                    group["c4_val_loss"],
                    group[f"{val_set}_val_loss"],
                    color=color,
                    marker=marker,
                )
        construct_legend(
            axs[0],
            [
                "Validation Set",
                (val_sets, colors),
                None,
                "Architecture",
                (archs, markers),
            ],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )

    axs[1].set_title("Validation-to-Test", weight="bold")
    axs[1].set_xlabel("C4 Validation Loss")
    axs[1].set_ylabel("Test Loss")
    test_sets = [
        "arc_challenge",
        "arc_easy",
        "openbookqa",
        "piqa",
        "copa",
        "winogrande",
        "hellaswag",
    ]
    if avg_only:
        grouped = df.groupby("arch")
        for (arch, group), color in zip(grouped, colors):
            axs[1].scatter(
                group["c4_val_loss"],
                group[[f"{test_set}/loss" for test_set in test_sets]].mean(axis=1),
                color=color,
            )
        construct_legend(
            fig,
            [
                (archs, colors),
            ],
            title="Architecture",
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            frameon=False,
            ncol=4,
        )
    else:
        for color, test_set in zip(colors, test_sets):
            grouped = df.groupby("arch")
            for (arch, group), marker in zip(grouped, markers):
                axs[1].scatter(
                    group["c4_val_loss"],
                    group[f"{test_set}/loss"],
                    color=color,
                    marker=marker,
                )
        construct_legend(
            axs[1],
            ["Test Set", (test_sets, colors)],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )

    if save_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_path}_{timestamp}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_intervention_arch_tokenizer_pretraining(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    xdata: str = "C4 Validation Loss",
    ydata: List[str] = [
        "The Pile UC Validation Loss",
        "FineWeb-EDU Validation Loss",
        "RefineWeb Validation Loss",
        "Slimpajama Validation Loss",
    ],
):
    df = df.copy()
    archs = sorted(df["Architecture"].unique())
    pretrains = sorted(df["Pretraining Data"].unique())
    tokenizers = sorted(df["Tokenizer"].unique())

    dims = ["Architecture", "Pretraining Data", "Tokenizer"]
    dim_values = [archs, pretrains, tokenizers]
    n_dims = len(dims)

    colors, markers = setup_style()

    n_rows = max(len(pretrains), len(tokenizers), len(archs))
    n_cols = len(pretrains) + len(tokenizers) + len(archs)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=((n_cols + 1) * 2, (n_rows + 1) * 2),
        constrained_layout=True,
    )

    grouped = df.groupby(dims)

    col_offset = 0
    for i, dim in enumerate(dims):
        row_values = dim_values[(i + 1) % n_dims]
        col_values = dim_values[(i + 2) % n_dims]
        n_rows_group = len(row_values)
        n_cols_group = len(col_values)

        for dim_current_vals, group in grouped:
            row = row_values.index(dim_current_vals[(i + 1) % n_dims])
            col = col_values.index(dim_current_vals[(i + 2) % n_dims])

            axs[row, col + col_offset].scatter(
                group[xdata],
                group[ydata].mean(axis=1),
                color=colors[dim_values[i].index(dim_current_vals[i])],
                s=5,
            )

        for row, dim_value in enumerate(row_values):
            axs[row, 0 + col_offset].set_ylabel(dim_value)
        for col, dim_value in enumerate(col_values):
            axs[n_rows_group - 1, col + col_offset].set_xlabel(dim_value)
        for row, col in itertools.product(
            range(n_rows_group, n_rows), range(col_offset, col_offset + n_cols_group)
        ):
            axs[row, col].set_visible(False)

        construct_legend(
            fig,
            [(dim_values[i], colors)],
            title=dim,
            loc="lower center",
            bbox_to_anchor=((col_offset + n_cols_group / 2 + 0.5) / n_cols, 1),
            frameon=False,
            ncol=2,
        )

        col_offset += n_cols_group

    if save_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_path}_{timestamp}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def construct_legend(
    ax: plt.Axes, sections: List[Optional[str] | Tuple[List[str], List[str]]], **kwargs
):
    handles, labels = [], []
    for section in sections:
        if section is None:  # empty line
            handles.append(plt.Line2D([0], [0], color="white"))
            labels.append("")
        elif isinstance(section, str):  # title
            handles.append(plt.Line2D([0], [0], color="white", label=section))
            labels.append(section)  # Make title text bold using LaTeX
        else:  # list of markers or colors
            n = min(len(section[0]), len(section[1]))
            labels.extend(section[0][:n])

            for i in range(n):
                if isinstance(section[1][i], tuple):  # color
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color="white",
                            marker="o",
                            markerfacecolor=section[1][i],
                            markersize=7,
                        )
                    )
                else:  # marker
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            color="white",
                            marker=section[1][i],
                            markerfacecolor="black",
                            markersize=7,
                        )
                    )
    legend = ax.legend(handles, labels, **kwargs)

    # Adjust title size and weight
    legend.get_title().set_weight("bold")
    legend.get_title().set_fontsize(legend._fontsize)

    # Adjust intermediate titles weight
    for text in legend.get_texts():
        if text.get_text() in [s for s in sections if isinstance(s, str)]:
            text.set_weight("bold")


def plot_intervention_matched_dims(
    df: pd.DataFrame,
    intervention: str,
    match_on: List[str],
    matches: List[List[str]],
    x_data: str,
    y_data: Dict[str, List[str]],
    row_titles: List[str],
    save_path: Optional[str] = None,
    shorten_col_titles: bool = False,
    verbose: bool = False,
    title_x_pos: float = 0.5,
    show_compute_isolines: bool = False,
    fit_curve: bool = False,
    margin: float = 0.1,
    x_lim_upper: Optional[float] = None,
    legend_kwargs: Dict[str, Any] = {},
    x_name: Optional[str] = None,
):
    df = df.copy()

    intervention_vals = sorted(df[intervention].unique())
    for joined_value in matches:
        assert len(joined_value) == len(match_on)

    colors, markers = setup_style()

    n_rows = len(y_data.keys())
    n_cols = len(matches)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FULLWIDTH, 2.5 * n_rows),
        constrained_layout=True,
        sharex="col",
        sharey="row",
    )

    grouped = df.groupby([intervention, *match_on])

    # TODO: This is a hack, just make color a dict
    # TODO: Also would be nicer if legend entries were sorted
    present_intervention_val_and_colors = set()  # not all possible vals do appear
    for row, ydata_list in enumerate(y_data.values()):
        y_lims = (np.inf, -np.inf)
        for (col, joined_vals), (color, intervention_val) in itertools.product(
            enumerate(matches), zip(colors, intervention_vals)
        ):
            x_lims = (np.inf, -np.inf)
            group_key = (intervention_val, *joined_vals)
            if group_key in grouped.groups:
                group = grouped.get_group(group_key)

                if x_lim_upper is not None:
                    assert len(group["Pretraining Data"].unique()) == 1
                    pretrain = group["Pretraining Data"].unique()[0]
                    if pretrain in ["The Pile", "The Pile Deduped"]:
                        pretrain = "The Pile UC"
                    group = group[group[f"{pretrain} Validation Loss"] < 4]

                if row == 0 and verbose:
                    print(group_key, group["Name"].unique())

                x = group[x_data]
                y = group[ydata_list].mean(axis=1)

                axs[row, col].scatter(
                    x,
                    y,
                    color=color,
                    s=7,
                )
                x_lims = (min(x_lims[0], x.min()), max(x_lims[1], x.max()))
                y_lims = (min(y_lims[0], y.min()), max(y_lims[1], y.max()))

                present_intervention_val_and_colors.add((intervention_val, color))

                if fit_curve:
                    N = group["Size"]
                    D = group["Tokens"]
                    mask = np.isfinite(x) & np.isfinite(N) & np.isfinite(D)
                    x = x[mask]
                    N = N[mask]
                    D = D[mask]

                    if len(x) == 0:
                        if verbose:
                            print(f"  No samples to fit")
                        continue
                    ex, ey, popt = fit_loss_to_loss_power_law(
                        x, y, N, D, verbose=verbose
                    )
                    if popt is None:
                        continue  # Couldn't fit curve

                    x_fit = np.linspace(
                        x.min() - 10 * margin,
                        x.max() + 10 * margin,
                        100,
                    )
                    ex_fit = np.full_like(x_fit, ex)
                    ey_fit = np.full_like(x_fit, ey)
                    y_fit = loss_to_loss_curve_plot((x_fit, ex_fit, ey_fit), *popt)
                    axs[row, col].plot(
                        x_fit,
                        y_fit,
                        color=color,
                        zorder=1,
                        linewidth=3,
                        alpha=0.5,
                    )
            if x_lims[0] != np.inf and x_lims[1] != -np.inf:
                axs[row, col].set_xlim(x_lims[0] - margin, x_lims[1] + margin)
        if y_lims[0] != np.inf and y_lims[1] != -np.inf:
            axs[row, col].set_ylim(y_lims[0] - margin, y_lims[1] + margin)

    if show_compute_isolines:
        grouped = df.groupby(match_on)

        for row, ydata_list in enumerate(y_data.values()):
            for col, joined_vals in enumerate(matches):
                if tuple(joined_vals) not in grouped.groups:
                    continue

                group = grouped.get_group(tuple(joined_vals))
                models = group["Name"].unique()
                flop_vals = sorted(
                    group[group["Name"] == models[0]]["FLOPs"].dropna().unique()
                )[::5]
                if not flop_vals:
                    continue

                for flop_val in flop_vals:
                    datax, datay = [], []
                    for model in models:
                        compute_matched = group[(group["Name"] == model)]
                        idx = compute_matched["FLOPs"].sub(flop_val).abs().idxmin()
                        compute_matched = compute_matched.loc[[idx]]
                        datax.append(compute_matched[x_data].iloc[0])
                        datay.append(compute_matched[ydata_list].mean(axis=1).iloc[0])
                    axs[row, col].plot(datax, datay, color="black", zorder=0.1)

    construct_legend(
        fig,
        [
            (
                [v for v, _ in present_intervention_val_and_colors],
                [c for _, c in present_intervention_val_and_colors],
            )
        ],
        title=intervention,
        frameon=False,
        **legend_kwargs,
    )

    for col, ax in enumerate(axs[0, :]):
        fig.text(
            title_x_pos - 0.02,
            1.04,
            (
                "\n".join(
                    [m.split(" ")[0] for m in match_on]
                    if shorten_col_titles
                    else match_on
                )
            ),
            ha="right",
            va="bottom",
            fontweight="bold",
            transform=ax.transAxes,
        )
        fig.text(
            title_x_pos + 0.02,
            1.04,
            "\n".join(
                [
                    m.replace("The Pile", "Pile").replace("FineWeb", "FW")
                    for m in matches[col]
                ]
                if shorten_col_titles
                else matches[col]
            ),
            ha="left",
            va="bottom",
            transform=ax.transAxes,
        )
    for ax in axs[-1, :]:
        ax.set_xlabel(x_name or x_data)
    for row, ax in enumerate(axs[:, 0]):
        ax.set_ylabel(list(y_data.keys())[row])
        fig.text(
            -0.35,
            0.5,
            row_titles[row],
            ha="right",
            va="center",
            fontweight="bold",
            transform=ax.transAxes,
            rotation=90,
        )

    for ax in axs.flat:
        ax.axline((4, 4), slope=1, color="lightgray", zorder=0, linewidth=1)

    if save_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_path}_{timestamp}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_intervention_size(
    df: pd.DataFrame,
    x_data: str,
    y_data: Dict[str, List[str]],
    arch: Optional[Union[str, List[str]]] = [],
    tokenizer: Optional[Union[str, List[str]]] = [],
    pretrain: Optional[Union[str, List[str]]] = [],
    save_path: Optional[str] = None,
    markersize: int = 20,
):
    df = df.copy()
    if isinstance(arch, str):
        arch = [arch]
    if arch:
        df = df[df["Architecture"].isin(arch)]
    else:
        arch = [str(a) for a in df["Architecture"].unique()]
    if isinstance(tokenizer, str):
        tokenizer = [tokenizer]
    if tokenizer:
        df = df[df["Tokenizer"].isin(tokenizer)]
    else:
        tokenizer = [str(t) for t in df["Tokenizer"].unique()]
    if isinstance(pretrain, str):
        pretrain = [pretrain]
    if pretrain:
        df = df[df["Pretraining Data"].isin(pretrain)]
    else:
        pretrain = [str(p) for p in df["Pretraining Data"].unique()]

    colors, markers = setup_style()
    # markers = ["+", "x", "*"]

    fig, ax = plt.subplots(figsize=(COLUMNWIDTH, 4))

    grouped = df.groupby(["Pretraining Data"])

    for (_, group), marker in zip(grouped, markers):
        for ydata_name, ydata_list in y_data.items():
            scatter = ax.scatter(
                group[x_data],
                group[ydata_list].mean(axis=1),
                c=group["Size"],
                cmap="crest",
                marker=marker,
                # edgecolor="white",
                s=markersize,
            )
    fig.text(
        1.03, 1, "Size", ha="left", va="top", fontweight="bold", transform=ax.transAxes
    )
    fig.colorbar(scatter, ax=ax, shrink=0.85, anchor=(0, 0), panchor=(0, 0))
    ax.legend(
        [
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="k",
                markersize=7,
            )
            for marker in markers
        ],
        [name[0] for (name, _) in grouped],
        title="Pretraining Data",
        title_fontproperties={"weight": "bold"},
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
    )

    title_keys, title_vals = [], []
    if arch:
        title_keys.append("Architecture")
        title_vals.append(" | ".join(arch))
    if tokenizer:
        title_keys.append("Tokenizer")
        title_vals.append(" | ".join(tokenizer))
    if pretrain:
        title_keys.append("Pretraining")
        title_vals.append(" | ".join(pretrain))

    fig.text(
        0.3 - 0.02,
        1.04,
        "\n".join(title_keys),
        ha="right",
        va="bottom",
        fontweight="bold",
        transform=ax.transAxes,
    )
    fig.text(
        0.3 + 0.02,
        1.04,
        "\n".join(title_vals),
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.set_xlabel(x_data)
    ax.set_ylabel(list(y_data.keys())[0])

    if save_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_path}_{timestamp}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_intervention_optim(
    df: pd.DataFrame,
    x_data: str,
    y_data: Dict[str, List[str]],
    arch: Optional[Union[str, List[str]]] = [],
    tokenizer: Optional[Union[str, List[str]]] = [],
    pretrain: Optional[Union[str, List[str]]] = [],
    save_path: Optional[str] = None,
):
    df = df.copy()
    if isinstance(arch, str):
        arch = [arch]
    if arch:
        df = df[df["Architecture"].isin(arch)]
    else:
        arch = [str(a) for a in df["Architecture"].unique()]
    if isinstance(tokenizer, str):
        tokenizer = [tokenizer]
    if tokenizer:
        df = df[df["Tokenizer"].isin(tokenizer)]
    else:
        tokenizer = [str(t) for t in df["Tokenizer"].unique()]
    if isinstance(pretrain, str):
        pretrain = [pretrain]
    if pretrain:
        df = df[df["Pretraining Data"].isin(pretrain)]
    else:
        pretrain = [str(p) for p in df["Pretraining Data"].unique()]

    colors, markers = setup_style()
    markers[8] = "x"

    grouped = df.groupby(["Optimizer", "Scheduler", "Learning Rate", "Weight Decay"])
    # Repeat colors and markers to be the same length as grouped
    num_groups = len(grouped)
    colors = (colors * (num_groups // len(colors) + 1))[:num_groups]
    markers = (markers * (num_groups // len(markers) + 1))[:num_groups]

    fig, ax = plt.subplots(figsize=(COLUMNWIDTH, 4))
    for ydata_name, ydata_list in y_data.items():
        for (group_key, group), color, marker in zip(grouped, colors, markers):
            scatter = ax.scatter(
                group[x_data],
                group[ydata_list].mean(axis=1),
                color=color,
                marker=marker,
                s=40,
            )
    categories = [k for k, _ in grouped]
    category_labels = [f"{k[0]} | {k[1]} | {k[2]:.1e} | {k[3]:.1e}" for k in categories]
    category_labels = [
        label.replace("e+0", "e+").replace("e-0", "e-") for label in category_labels
    ]
    ax.legend(
        [
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                # color="w",
                linestyle="None",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=10,
            )
            for marker, color in zip(markers, colors)
        ],
        category_labels,
        title="Optimizer | Schedule | LR | Weight Decay",
        title_fontproperties={"weight": "bold"},
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
    )
    # construct_legend(
    #     ax,
    #     [
    #         (
    #             category_labels,
    #             colors,
    #         )
    #     ],
    #     title="Optimizer | Schedule | LR | Weight Decay",
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.2),
    #     frameon=False,
    # )

    title_keys, title_vals = [], []
    if arch:
        title_keys.append("Architecture")
        title_vals.append(" | ".join(arch))
    if tokenizer:
        title_keys.append("Tokenizer")
        title_vals.append(" | ".join(tokenizer))
    if pretrain:
        title_keys.append("Pretraining")
        title_vals.append(" | ".join(pretrain))

    fig.text(
        0.3 - 0.02,
        1.04,
        "\n".join(title_keys),
        ha="right",
        va="bottom",
        fontweight="bold",
        transform=ax.transAxes,
    )
    fig.text(
        0.3 + 0.02,
        1.04,
        "\n".join(title_vals),
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.set_xlabel(x_data)
    ax.set_ylabel(list(y_data.keys())[0])

    if save_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_path}_{timestamp}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_intervention_ctx(
    df: pd.DataFrame,
    x_data: str,
    y_data: Dict[str, List[str]],
    arch: Optional[Union[str, List[str]]] = [],
    tokenizer: Optional[Union[str, List[str]]] = [],
    pretrain: Optional[Union[str, List[str]]] = [],
    save_path: Optional[str] = None,
    markersize: int = 20,
):
    df = df.copy()
    if isinstance(arch, str):
        arch = [arch]
    if arch:
        df = df[df["Architecture"].isin(arch)]
    else:
        arch = [str(a) for a in df["Architecture"].unique()]
    if isinstance(tokenizer, str):
        tokenizer = [tokenizer]
    if tokenizer:
        df = df[df["Tokenizer"].isin(tokenizer)]
    else:
        tokenizer = [str(t) for t in df["Tokenizer"].unique()]
    if isinstance(pretrain, str):
        pretrain = [pretrain]
    if pretrain:
        df = df[df["Pretraining Data"].isin(pretrain)]
    else:
        pretrain = [str(p) for p in df["Pretraining Data"].unique()]

    colors, markers = setup_style()

    fig, ax = plt.subplots(figsize=(COLUMNWIDTH, 4))
    grouped = df.groupby("Pretraining Data")
    for (_, group), marker in zip(grouped, markers):
        for ydata_name, ydata_list in y_data.items():
            scatter = ax.scatter(
                group[x_data],
                group[ydata_list].mean(axis=1),
                c=group["Context Length"],
                cmap="crest",
                s=markersize,
                marker=marker,
            )
    fig.text(
        1.12,
        1,
        "Context\nLength",
        ha="center",
        va="top",
        fontweight="bold",
        transform=ax.transAxes,
    )
    fig.colorbar(scatter, ax=ax, shrink=0.85, anchor=(0, 0), panchor=(0, 0))
    ax.legend(
        [
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="k",
                markersize=7,
            )
            for marker in markers
        ],
        [name for (name, _) in grouped],
        title="Pretraining Data",
        title_fontproperties={"weight": "bold"},
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
    )

    title_keys, title_vals = [], []
    if arch:
        title_keys.append("Architecture")
        title_vals.append(" | ".join(arch))
    if tokenizer:
        title_keys.append("Tokenizer")
        title_vals.append(" | ".join(tokenizer))
    if pretrain:
        title_keys.append("Pretraining")
        title_vals.append(" | ".join(pretrain))

    fig.text(
        0.3 - 0.02,
        1.04,
        "\n".join(title_keys),
        ha="right",
        va="bottom",
        fontweight="bold",
        transform=ax.transAxes,
    )
    fig.text(
        0.3 + 0.02,
        1.04,
        "\n".join(title_vals),
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.set_xlabel(x_data)
    ax.set_ylabel(list(y_data.keys())[0])

    if save_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_path}_{timestamp}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_intervention(
    df: pd.DataFrame,
    x_data: str,
    y_data: Dict[str, List[str]],
    intervention: str,
    intervention_data_type: Literal["categorical", "continuous"] = "categorical",
    plot_group: Literal["all", "first"] = "all",
    save_path: Optional[str] = None,
    x_label: Optional[str] = None,
    title: Optional[str] = None,
    group_order: Optional[List[str]] = None,
    group_names: Optional[List[str]] = None,
    z_order: Optional[List[int]] = None,
    x_range: Optional[Tuple[float, float]] = None,
    entropy_df: Optional[pd.DataFrame] = None,
    fit_curve: Optional[Union[List[str], bool]] = False,
    subsample: Optional[int] = 1,
    margin: float = 0.1,
    legend_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    file_type: Literal["pdf", "png", "svg"] = "pdf",
):
    if len(y_data) > 1:
        raise NotImplementedError("Only one y_data is supported for now")
    assert z_order is None or len(z_order) == len(
        group_order
    ), "custom_z_order must be the same length as custom_group_order"

    df = df.copy()
    # Setup figure
    colors, markers = setup_style()

    fig, ax = plt.subplots(figsize=(COLUMNWIDTH, 4))
    ax.axline((4, 4), slope=1, color="lightgray", zorder=0, linewidth=1)

    # Plot data
    grouped = df.groupby(intervention, dropna=False)
    if group_order is not None:

        def fuzzy_match(object, name_list):
            for name in name_list:
                if name.lower() in str(object).lower():
                    return name
            return False

        grouped = [
            (fuzzy_match(name, group_order), group)
            for name, group in grouped
            if fuzzy_match(name, group_order)
        ]
        grouped.sort(key=lambda x: group_order.index(str(x[0])))
    else:
        grouped = list(grouped)

    if z_order is None:
        z_order = [i for i in range(len(grouped))]
    if group_names is not None:
        grouped = [(name, group) for name, (_, group) in zip(group_names, grouped)]

    x_lims = (np.inf, -np.inf)
    y_lims = (np.inf, -np.inf)
    if intervention_data_type == "categorical":
        legend_labels, legend_handles = [], []
        for (name, group), color, marker, z in zip(grouped, colors, markers, z_order):
            models = group["Name"].unique()
            if len(models) > 1 and plot_group == "first":
                group = group[group["Name"] == models[0]]

            if verbose:
                print(name, group["Name"].unique())

            if x_range is not None:
                group = group[
                    (group[x_data] >= x_range[0]) & (group[x_data] <= x_range[1])
                ]
            x = group[x_data]
            y = group[list(y_data.values())[0]].mean(axis=1)
            ax.scatter(
                x[::subsample],
                y[::subsample],
                color=color,
                marker=marker,
                zorder=z,
            )
            x_lims = (min(x_lims[0], x.min()), max(x_lims[1], x.max()))
            y_lims = (min(y_lims[0], y.min()), max(y_lims[1], y.max()))

            if fit_curve is True or name in fit_curve:
                N = group["Size"]
                D = group["Tokens"]
                ex, ey, popt = fit_loss_to_loss_power_law(x, y, N, D, verbose=verbose)
                if popt is None:
                    continue  # Couldn't fit curve

                x_fit = np.linspace(
                    x.min() - 10 * margin,
                    x.max() + 10 * margin,
                    100,
                )
                ex_fit = np.full_like(x_fit, ex)
                ey_fit = np.full_like(x_fit, ey)
                y_fit = loss_to_loss_curve_plot((x_fit, ex_fit, ey_fit), *popt)
                ax.plot(
                    x_fit,
                    y_fit,
                    color=color,
                    zorder=1,
                    linewidth=3,
                    alpha=0.5,
                )
                handle = plt.Line2D(
                    [0],
                    [0],
                    color=(*color, 0.5),
                    marker=marker,
                    linestyle="solid",
                    linewidth=3,
                    markerfacecolor=color,
                )
            else:
                handle = plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="None",
                    color=color,
                    markerfacecolor=color,
                )
            legend_labels.append(name)
            legend_handles.append(handle)

        ax.legend(
            legend_handles,
            legend_labels,
            title=intervention,
            title_fontproperties={"weight": "bold"},
            frameon=False,
            **(legend_kwargs or {}),
        )
    else:
        raise NotImplementedError(
            "Only categorical interventions are supported for now"
        )

    # Axes
    ax.set_xlim(round(x_lims[0] - margin, 1), round(x_lims[1] + margin, 1))
    ax.set_ylim(round(y_lims[0] - margin, 1), round(y_lims[1] + margin, 1))
    ax.set_xlabel(x_label or x_data)
    ax.set_ylabel(list(y_data.keys())[0])
    ax.set_title(title, fontweight="bold")

    # Save figure
    if save_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(
            f"{save_path}_{timestamp}.{file_type}", bbox_inches="tight", dpi=300
        )
    plt.show()


def plot_schematic(save_path: Optional[str] = None):
    np.random.seed(0)

    colors, markers = setup_style()
    fig, ax = plt.subplots(figsize=(COLUMNWIDTH, 4))
    markersize = 60
    x_range = (2, 5.5)
    # ax.axline((4, 4), slope=1, color="lightgray", zorder=0, linewidth=1)

    x = np.linspace(*x_range, 100)
    x_base = np.array([3, 4, 5]) + np.random.normal(0, 0.1, 3)
    x_intervention1 = x_base - 0.5 + np.random.normal(0, 0.1, 3)
    x_intervention2 = x_base - 0.5 + np.random.normal(0, 0.1, 3)

    p_base = [0.8, 2, 1.3, 0]
    p_intervention = [0.8, 1, 1.3, 1]

    line_base = train_to_test_curve(x, *p_base)
    line_intervention1 = train_to_test_curve(x, *p_intervention)

    ax.plot(x, line_base, color=colors[0], zorder=1, linewidth=3, alpha=0.5)
    ax.plot(x, line_intervention1, color=colors[2], zorder=1, linewidth=3, alpha=0.5)

    y_base = train_to_test_curve(x_base, *p_base)
    y_intervention1 = train_to_test_curve(x_intervention1, *p_intervention)
    y_intervention2 = train_to_test_curve(x_intervention2, *p_base)

    ax.scatter(
        x_base,
        y_base,
        color=colors[0],
        zorder=2,
        marker=markers[0],
        s=markersize,
        edgecolor="white",
    )
    ax.scatter(
        x_intervention1,
        y_intervention1,
        color=colors[2],
        zorder=2,
        marker=markers[1],
        s=markersize,
        edgecolor="white",
    )
    ax.scatter(
        x_intervention2,
        y_intervention2,
        color=colors[3],
        zorder=2,
        marker=markers[2],
        s=markersize,
        edgecolor="white",
    )

    ax.annotate(
        "",
        xy=(x_intervention1[1], y_intervention1[1]),
        xytext=(x_base[1], y_base[1]),
        arrowprops=dict(facecolor=colors[2], shrink=0.1, linewidth=2),
    )

    ax.annotate(
        "",
        xy=(x_intervention2[1], y_intervention2[1]),
        xytext=(x_base[1], y_base[1]),
        arrowprops=dict(facecolor=colors[3], shrink=0.1, linewidth=2),
    )

    ax.text(
        x_base[1] + 0.1,
        y_base[1],
        "Base Model",
        fontweight="bold",
        ha="left",
        va="top",
        fontsize=12,
        color=colors[0],
    )

    ax.text(
        x_intervention1[1] + 0.5,
        y_intervention1[1],
        "Effective\nIntervention",
        fontweight="bold",
        ha="center",
        va="top",
        fontsize=12,
        color=colors[2],
    )

    ax.text(
        x_intervention2[1] + 0.5,
        y_intervention2[1],
        "Ineffective\nIntervention",
        fontweight="bold",
        ha="center",
        va="top",
        fontsize=12,
        color=colors[3],
    )

    ax.set_xlim(*x_range)
    ax.set_xlabel("Loss 1")
    ax.set_xticklabels([])
    ax.set_ylabel("Loss 2")
    ax.set_yticklabels([])
    # Save figure
    if save_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_path}_{timestamp}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_l2l(
    df: pd.DataFrame,
    x_data: str,
    y_data: Dict[str, List[str]],
    save_path: Optional[str] = None,
    titles: Optional[List[str]] = [],
    fit_curves: bool = False,
    entropy_df: Optional[pd.DataFrame] = None,
    e_min: float = 0,
    # subsample: Optional[int] = 1,
    margin: float = 0.1,
    legend_kwargs: Optional[List[Dict[str, Any]]] = [None, None],
    verbose: bool = False,
):
    assert (
        len(df["Pretraining Data"].unique()) == 1
    ), "Only one pretraining data is supported for now"

    assert (
        len(titles) == len(y_data) or len(titles) == 0
    ), "Must provide a title for each y_data category"
    assert (
        len(legend_kwargs) == len(y_data) or len(legend_kwargs) == 0
    ), "Must provide a legend_kwargs for each y_data category"

    df = df.copy()

    # Setup figure
    colors, markers = setup_style()

    n_cols = len(y_data)
    fig, axs = plt.subplots(
        1, n_cols, figsize=(COLUMNWIDTH, 6), constrained_layout=True
    )
    if n_cols == 1:
        axs = np.array([axs])
    for ax in axs.flat:
        ax.axline((4, 4), slope=1, color="lightgray", zorder=0, linewidth=1)

    # Plot data
    for i, (ax, (y_data_category, y_data_list)) in enumerate(
        zip(axs.flat, y_data.items())
    ):
        x_lims = (np.inf, -np.inf)
        y_lims = (np.inf, -np.inf)
        legend_labels, legend_handles = [], []
        for y_data_name, color, marker in zip(y_data_list, colors, markers):
            if verbose:
                print(y_data_name)
            x = df[x_data]
            y = df[y_data_name]
            ax.scatter(
                x,
                y,
                color=color,
                marker=marker,
            )
            x_lims = (min(x_lims[0], x.min()), max(x_lims[1], x.max()))
            y_lims = (min(y_lims[0], y.min()), max(y_lims[1], y.max()))

            if fit_curves:
                if entropy_df is not None:
                    assert len(df["Pretraining Data"].unique()) == 1
                    pretrain = df["Pretraining Data"].unique()[0]
                    ex = entropy_df.loc[
                        (entropy_df["pretrain"] == pretrain)
                        & (entropy_df["loss"] == x_data),
                        "entropy",
                    ].values[0]
                    ey = entropy_df.loc[
                        (entropy_df["pretrain"] == pretrain)
                        & (entropy_df["loss"] == y_data_name),
                        "entropy",
                    ].values[0]
                    if verbose:
                        print(f"  ex {ex:.2f}, ey {ey:.2f}")
                    popt = fit_shifted_power_law(x, y, ex, ey, verbose=verbose)
                else:
                    N = df["Size"]
                    D = df["Tokens"]
                    ex, ey, popt = fit_loss_to_loss_power_law(
                        x, y, N, D, verbose=verbose, e_min=e_min
                    )
                if popt is None:
                    continue  # Couldn't fit curve

                x_fit = np.linspace(
                    x.min() - 10 * margin,
                    x.max() + 10 * margin,
                    100,
                )
                ex_fit = np.full_like(x_fit, ex)
                ey_fit = np.full_like(x_fit, ey)
                y_fit = loss_to_loss_curve_optim((x_fit, ex_fit, ey_fit), *popt)
                ax.plot(
                    x_fit,
                    y_fit,
                    color=color,
                    zorder=1,
                    linewidth=3,
                    alpha=0.5,
                )

            legend_labels.append(
                y_data_name.replace(" Validation Loss", "").replace(" Loss", "")
            )
            if fit_curves:
                handle = plt.Line2D(
                    [0],
                    [0],
                    color=(*color, 0.5),
                    marker=marker,
                    linestyle="solid",
                    linewidth=3,
                    markerfacecolor=color,
                )
            else:
                handle = plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="None",
                    color=color,
                    markerfacecolor=color,
                )
            legend_handles.append(handle)
        ax.legend(
            legend_handles,
            legend_labels,
            # title=y_data_category,
            title_fontproperties={"weight": "bold"},
            frameon=False,
            **(legend_kwargs[i] or {}),
        )

        # Axes
        ax.set_xlim(round(x_lims[0] - margin, 1), round(x_lims[1] + margin, 1))
        ax.set_ylim(round(y_lims[0] - margin, 1), round(y_lims[1] + margin, 1))
        ax.set_xlabel(x_data)
        ax.set_ylabel(y_data_category)
        if titles:
            ax.set_title(titles[i], fontweight="bold")

    # Save figure
    if save_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_path}_{timestamp}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def train_to_test_curve(x, a, b, c, d):
    return a * (x - b) ** c + d


def loss_to_loss_curve_plot(x, a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        _x, e1, e2 = x
        # Ensure the base is positive before applying the power
        return a * (_x - e1) ** b + e2


def loss_to_loss_curve_optim(x, a, b):
    _x, e1, e2 = x
    # Ensure the base is positive before applying the power
    return a * np.maximum((_x - e1), 1e-10) ** b + e2


def compute_to_loss_curve(x, a, b, c, d, e):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        N, D = x
        return a + ((b / N) ** (d / e) + c / D) ** e


def fit_compute_to_loss_curve(x, N, D, verbose: bool = False):
    try:
        popt, _ = curve_fit(
            compute_to_loss_curve,
            (N, D),
            x,
            p0=(0, 1, 1, 1, 1),
            bounds=(
                [0, -np.inf, -np.inf, -np.inf, -np.inf],
                [x.min(), np.inf, np.inf, np.inf, np.inf],
            ),
        )
        return popt[0]
    except RuntimeError as e:
        if verbose:
            print(f"  Optimizer failed, setting entropy by heuristic")
        return x.min()


def fit_entropies(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    results = []

    grouped_df = df.groupby("Pretraining Data")
    losses = [
        "ARC-Challenge Loss",
        "ARC-Easy Loss",
        "OpenBookQA Loss",
        "PIQA Loss",
        "COPA Loss",
        "Winogrande Loss",
        "HellaSwag Loss",
        "The Pile UC Validation Loss",
        "RefineWeb Validation Loss",
        "Slimpajama Validation Loss",
        "C4 Validation Loss",
        "FineWeb-Edu Validation Loss",
    ]

    for (pretrain, group), loss in tqdm(itertools.product(grouped_df, losses)):
        if verbose:
            print(pretrain, loss)
        N = group["Size"]
        D = group["Tokens"]
        x = group[loss]

        # Drop all indices where any of x, N, D is nan or not finite
        mask = np.isfinite(x) & np.isfinite(N) & np.isfinite(D)
        x = x[mask]
        N = N[mask]
        D = D[mask]

        if len(x) == 0:
            if verbose:
                print(f"  No samples to fit")
            # TODO: Possibly set to x.min(), since we should only run out of data points
            # here due to nans in N and D
            results.append({"pretrain": pretrain, "loss": loss, "entropy": None})
            continue
        if verbose:
            print(f"  Fitting with {len(x)} samples")

        e = fit_compute_to_loss_curve(x, N, D)
        results.append({"pretrain": pretrain, "loss": loss, "entropy": e})
        if verbose:
            print(f"  e {e:.2f}")
    return pd.DataFrame(results)


def fit_shifted_power_law(x, y, ex, ey, verbose: bool = False):
    try:
        ex = np.full_like(x, ex)
        ey = np.full_like(x, ey)

        popt, _ = curve_fit(
            loss_to_loss_curve_optim,
            (x, ex, ey),
            y,
            p0=(0.8, 1),
            maxfev=10000,
        )
        if verbose:
            print(f"  Fitted parameters {popt}")
    except RuntimeError as e:
        if verbose:
            print(f"  Could not fit curve")
        popt = None
    return popt


def fit_loss_to_loss_power_law(x, y, N, D, verbose: bool = False, e_min: float = 0):
    try:
        popt, _ = curve_fit(
            compute_to_loss_curve,
            (N, D),
            x,
            p0=(0.1, 1, 1, 1, 1),
            bounds=(
                [e_min, -np.inf, -np.inf, -np.inf, -np.inf],
                [x.min(), np.inf, np.inf, np.inf, np.inf],
            ),
        )
        e1 = popt[0]
    except (RuntimeError, ValueError) as e:
        if verbose:
            print(f"  Optimizer for e1 failed, setting by heuristic")
        e1 = x.min()
    try:
        popt, _ = curve_fit(
            compute_to_loss_curve,
            (N, D),
            y,
            p0=(e_min, 1, 1, 1, 1),
            bounds=(
                [0, -np.inf, -np.inf, -np.inf, -np.inf],
                [y.min(), np.inf, np.inf, np.inf, np.inf],
            ),
        )
        e2 = popt[0]
    except (RuntimeError, ValueError) as e:
        if verbose:
            print(f"  Optimizer for e2 failed, setting by heuristic")
        e2 = y.min()

    if verbose:
        print(f"  e1 {e1:.2f}, e2 {e2:.2f}")

    try:
        e1_expanded = np.full_like(x, e1)
        e2_expanded = np.full_like(x, e2)

        popt, _ = curve_fit(
            loss_to_loss_curve_optim,
            (x, e1_expanded, e2_expanded),
            y,
            p0=(0.8, 1),
            bounds=([-np.inf, 0], [np.inf, np.inf]),
            maxfev=10000,
        )
        if verbose:
            print(f"  Fitted parameters {popt}")
    except (RuntimeError, ValueError) as e:
        if verbose:
            print(f"  Could not fit curve")
        popt = None
    return e1, e2, popt


def tuple_contains_any_combination(row, vals):
    if row is None:
        if None in vals:
            return True
        return False
    return all(r in vals for r in row)
