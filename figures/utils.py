import itertools
import json
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def load_model_data(model_configs, base_dir):
    """Load data for all model types and pretraining sources"""
    data_dict = {}

    for model_type, config in model_configs.items():
        for folder_name, short_key in config["folders"].items():
            # Create composite key that includes model type
            dict_key = f"{model_type}_{short_key}"
            jsonl_path = os.path.join(base_dir, folder_name, "metrics.eval.jsonl")

            # Read the JSONL file
            data = []
            with open(jsonl_path, "r") as f:
                for line in f:
                    data.append(json.loads(line))

            data_dict[dict_key] = data

    return data_dict


def process_jsonl_to_df(data_dict):
    """Convert the JSONL data into a DataFrame with model type information"""
    rows = []

    for key, entries in data_dict.items():
        model_type, pretrain_data = key.split("_", 1)

        for entry in entries:
            step = entry["global_step"]
            row = {
                "steps": step,
                "pretrain_data": pretrain_data,
                "model_type": model_type,
            }

            # Add accuracy for each task
            for task, metrics in entry.items():
                if isinstance(metrics, dict) and "acc,none" in metrics:
                    row[f"{task}/acc"] = metrics["acc,none"]

            for task, metrics in entry.items():
                if isinstance(metrics, dict) and "loss" in metrics:
                    row[f"{task}/loss"] = metrics["loss"]

            rows.append(row)

    return pd.DataFrame(rows)


def create_performance_scatter(
    df, dataset1, dataset2, model_configs, color_mapping, metric="acc"
):
    """Create scatter plot with different markers for model types"""
    plt.figure(figsize=(12, 8))

    # Construct column names
    x_col = f"{dataset1}/{metric}"
    y_col = f"{dataset2}/{metric}"

    # Dictionary to store combined data for R² calculation
    pretrain_data_combined = {}

    # Plot points for each model type and pretraining data
    for model_type, config in model_configs.items():
        model_df = df[df["model_type"] == model_type]

        # Apply model-specific step filtering
        model_df = model_df[
            (model_df["steps"] % config["step_interval"] == 0)
            & (model_df["steps"] >= config["min_step"])
        ]

        # Plot points for each pretraining data source
        for pretrain in model_df["pretrain_data"].unique():
            subset = model_df[model_df["pretrain_data"] == pretrain]

            # Plot scatter points
            plt.scatter(
                subset[x_col],
                subset[y_col],
                c=color_mapping[pretrain],
                marker=config["marker"],
                alpha=0.6,
                label=f"{model_type}-{pretrain}",
            )

            # Collect data for combined R² calculation
            if pretrain not in pretrain_data_combined:
                pretrain_data_combined[pretrain] = {"x": [], "y": []}
            pretrain_data_combined[pretrain]["x"].extend(subset[x_col].values)
            pretrain_data_combined[pretrain]["y"].extend(subset[y_col].values)

    # Add regression lines for combined data
    for pretrain, data in pretrain_data_combined.items():
        x = np.array(data["x"])
        y = np.array(data["y"])

        # Calculate regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r2 = r_value**2

        # Plot regression line
        x_range = np.linspace(x.min(), x.max(), 100)
        plt.plot(
            x_range,
            slope * x_range + intercept,
            c=color_mapping[pretrain],
            linestyle="--",
            label=f"{pretrain} (R² = {r2:.3f})",
        )

    # Customize plot
    plt.title(
        f"Model Performance: {dataset1} vs {dataset2}\n(Different intervals/min_steps per model type)"
    )
    plt.xlabel(f"{dataset1} Accuracy")
    plt.ylabel(f"{dataset2} Accuracy")

    # Adjust legend
    plt.legend(
        title="Model-Pretraining & Fits", bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    plt.tight_layout()
    return plt.gcf()


def filter_min_tokens(df: pd.DataFrame, min_tokens: int) -> Tuple[pd.DataFrame, str]:
    """
    Filter the DataFrame to only include rows where the number of tokens is greater than min_tokens.
    """
    assert "name" in df.columns, "name column is missing"
    assert "tokens" in df.columns, "tokens column is missing"

    # Get max tokens for each model name
    max_tokens_per_model = df.groupby("name")["tokens"].max()

    # Keep only models that reached the minimum token count
    valid_models = max_tokens_per_model[max_tokens_per_model >= min_tokens].index
    df = df[df["name"].isin(valid_models)]

    return df, valid_models


def filter_within_chinchilla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame to only include rows where the size-to-tokens ratio is within
    Chinnchilla-scaling.
    """
    assert "tokens" in df.columns, "tokens column is missing"
    assert "size" in df.columns, "size column is missing"

    # Use .loc to avoid SettingWithCopyWarning
    df.loc[:, "within_chinchilla"] = df["tokens"] < df["size"] * 20
    return df[df["within_chinchilla"] == True]


def get_size_from_name(name: str) -> int:
    return int(name.split("_")[1].replace("M", "000000"))


def get_closest_number_in_list(number: int, numbers: list[int]) -> int:
    return min(numbers, key=lambda x: abs(x - number))


def get_size_bucket(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[int]]:
    """
    Add a "size_bucket" column to match models based on parameter count.
    """
    assert "name" in df.columns, "name column is missing"
    assert "arch" in df.columns, "arch column is missing"
    assert "size" in df.columns, "size column is missing"

    # Sizes are in the name, and models where designed to match Llama sizes
    llama_names = df[df["arch"] == "llama"]["name"].unique()
    llama_sizes = sorted(list(set([get_size_from_name(name) for name in llama_names])))

    df["size_bucket"] = df.apply(
        lambda row: get_closest_number_in_list(row["size"], llama_sizes),
        axis=1,
    )
    return df, llama_sizes


def get_token_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a "token_bucket" column to match models based on token count.
    """
    assert "name" in df.columns, "name column is missing"
    assert "tokens" in df.columns, "tokens column is missing"

    llama_tokens = sorted(df[df["arch"] == "llama"]["tokens"].unique())
    df["token_bucket"] = df.apply(
        lambda row: get_closest_number_in_list(row["tokens"], llama_tokens),
        axis=1,
    )
    return df, llama_tokens


def create_grid_comparison(
    df,
    columns,
    max_checkpoints_per_group: Optional[int] = None,
    fit_lines: Optional[bool] = False,
    transform: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Creates a grid of scatter plots comparing each column against others.

    Args:
        df: DataFrame with the data
        columns: list of column names to compare
        max_checkpoints_per_group: maximum number of checkpoints to sample per arch-pretraining combination
        fit_lines: whether to fit a line for each arch-pretraining combination
        transform: optional transformation to apply to the x-axis
        save_path: optional path to save figure
    """
    # Sample checkpoints if needed
    sampled_df = df.copy()
    archs = sorted(df["arch"].unique())
    p_datas = sorted(df["pretraining_data"].unique())
    if max_checkpoints_per_group is not None:
        samples = []
        for arch in archs:
            for p_data in p_datas:
                mask = (df["arch"] == arch) & (df["pretraining_data"] == p_data)
                group_data = df[mask]
                if len(group_data) > max_checkpoints_per_group:
                    samples.append(
                        group_data.sample(n=max_checkpoints_per_group, random_state=42)
                    )
                else:
                    samples.append(group_data)
        sampled_df = pd.concat(samples)

    n = len(columns)
    fig, axes = plt.subplots(n, n, figsize=(5 * n, 5 * n))

    # Style settings
    palette = sns.color_palette("deep", len(p_datas))
    marker_styles = ["o", "s", "^", "v", "d", "p", "X", "P"][: len(archs)]
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"][: len(archs)]
    colors = dict(zip(p_datas, palette))
    markers = dict(zip(archs, marker_styles))
    lines = dict(zip(archs, line_styles))

    # Off-diagonal plots
    for (i, col1), (j, col2) in itertools.permutations(enumerate(columns), 2):
        ax = axes[j, i]  # this way,all cols/rows share the same x/y axis
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.grid(color="lightgray")
        ax.axline((5, 5), slope=1, color="lightgray", zorder=0)  # y=x line

        # Fit one line per arch-pretraining combination and plot it
        # for arch in sampled_df["arch"].unique():
        for arch, p_data in itertools.product(
            sorted(sampled_df["arch"].unique()),
            sorted(sampled_df["pretraining_data"].unique()),
        ):
            data = sampled_df[
                (sampled_df["arch"] == arch)
                & (sampled_df["pretraining_data"] == p_data)
            ]
            if len(data) == 0:
                continue

            if fit_lines:
                slope, intercept, r_value, _, _ = stats.linregress(
                    data[col1], data[col2]
                )
                x_range = np.linspace(data[col1].min(), data[col1].max(), 100)
                y_range = slope * x_range + intercept
                ax.plot(
                    x_range,
                    y_range,
                    color=colors[p_data],
                    linestyle=lines[arch],
                )
            else:
                x_data = data[col1]
                if transform == "probit":
                    x_data = stats.norm.ppf(x_data)
                elif transform == "log":
                    x_data = np.log(x_data)
                ax.scatter(
                    x_data,
                    data[col2],
                    color=colors[p_data],
                    marker=markers[arch],
                    alpha=0.6,
                    label=f"{arch}-{p_data}",
                )

    # Add one fit line for all data combined
    # if len(sampled_df) > 1:
    #     slope, intercept, r_value, _, _ = stats.linregress(
    #         sampled_df[col1], sampled_df[col2]
    #     )
    #     x_range = np.linspace(
    #         sampled_df[col1].min(), sampled_df[col1].max(), 100
    #     )
    #     y_range = slope * x_range + intercept
    #     # Use black color for the overall correlation line
    #     ax.plot(x_range, y_range, c="black", linestyle="--")

    #     # Add R² annotation
    #     mid_idx = len(x_range) // 2
    #     ax.annotate(
    #         f"R² = {r_value**2:.2f}",
    #         xy=(x_range[mid_idx], y_range[mid_idx]),
    #         xytext=(10, 10),
    #         textcoords="offset points",
    #         color="black",
    #         bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    #     )

    # Add legend to the first subplot
    handles, labels = [], []
    # Empty handle for title
    handles.append(plt.Line2D([0], [0], color="white"))
    labels.append("Pretraining Data")
    # colors for each pretraining data
    for p_data, color in colors.items():
        handles.append(
            plt.Line2D([0], [0], color="white", marker="o", markerfacecolor=color)
        )
        labels.append(p_data)
    # Empty line for spacing
    handles.append(plt.Line2D([0], [0], color="white"))
    labels.append("")
    # Empty handle for title
    handles.append(plt.Line2D([0], [0], color="white"))
    labels.append("Architecture")
    # markers/lines for each architecture
    if fit_lines:
        for arch, linestyle in lines.items():
            handles.append(plt.Line2D([0], [0], color="black", linestyle=linestyle))
            labels.append(arch)
    else:
        for arch, marker in markers.items():
            handles.append(
                plt.Line2D(
                    [0], [0], color="white", marker=marker, markerfacecolor="black"
                )
            )
            labels.append(arch)

    # Diagonal plots
    for i, col in enumerate(columns):
        ax = axes[i, i]

        ax.set_xticks([])
        ax.set_yticks([])

        fig.legend(
            handles,
            labels,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            bbox_transform=ax.transAxes,
            frameon=False,
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_joint_train2train(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Reproduce train-to-train plots from Brandfonbrener et al. Points represent a pair of
    models "joined" on model size and dataset size.
    """
    df = df.copy()

    # get data and arch values
    data = sorted(df["pretraining_data"].unique())
    archs = sorted(df["arch"].unique())

    # bucket based on llama sizes and token counts
    df, _ = get_size_bucket(df)
    df, _ = get_token_bucket(df)
    # perform a self-merge to get size-and-token-paired data points
    df = pd.merge(df, df, on=["size_bucket", "token_bucket"], suffixes=["_1", "_2"])

    # set up plot
    n = len(data)
    fig, axes = plt.subplots(n, n, figsize=(5 * n, 5 * n), constrained_layout=True)

    palette = sns.color_palette("deep")
    colors = {
        ("gpt2", "gpt2"): palette[0],  # blue
        ("llama", "llama"): palette[3],  # red
        ("mamba", "mamba"): palette[8],  # yellow
        ("gpt2", "llama"): palette[4],  # purple
        ("gpt2", "mamba"): palette[2],  # green
        ("llama", "mamba"): palette[1],  # orange
    }

    # Plot pretraining data combinations in different subplots
    for (col, data1), (row, data2) in itertools.product(
        enumerate(data), enumerate(data)
    ):
        for (i, arch1), (j, arch2) in itertools.combinations_with_replacement(
            enumerate(archs), 2
        ):
            ax = axes[row, col]
            data1_stem = data1.split("_")[0]
            data2_stem = data2.split("_")[0]
            ax.set_xlabel(f"Loss on {data1_stem} (Trained on {data1_stem})")
            ax.set_ylabel(f"Loss on {data2_stem} (Trained on {data2_stem})")
            ax.grid(color="lightgray")
            ax.axline((5, 5), slope=1, color="lightgray", zorder=0)  # y=x line

            points = df[
                (df["arch_1"] == arch1)
                & (df["arch_2"] == arch2)
                & (df["pretraining_data_1"] == data1)
                & (df["pretraining_data_2"] == data2)
            ]

            ax.scatter(
                points[f"{data1}_val_loss_1"],
                points[f"{data2}_val_loss_2"],
                color=colors[(arch1, arch2)],
                # color=colors[i],
                # marker=markers[j],
            )

    # Add legend to the right of the figure
    handles, labels = [], []
    # Empty handle for title
    # handles.append(plt.Line2D([0], [0], color="white"))
    # labels.append("x-Axis Architecture")
    # for color, arch in zip(colors, archs):
    #     handles.append(
    #         plt.Line2D([0], [0], color="white", marker="o", markerfacecolor=color)
    #     )
    #     labels.append(arch)
    # Empty handle for spacing
    # handles.append(plt.Line2D([0], [0], color="white"))
    # labels.append("")
    # Empty handle for title
    # handles.append(plt.Line2D([0], [0], color="white"))
    # labels.append("y-Axis Architecture")
    # markers/lines for each architecture
    # for marker, arch in zip(markers, archs):
    #     handles.append(
    #         plt.Line2D([0], [0], color="white", marker=marker, markerfacecolor="black")
    #     )
    #     labels.append(arch)

    for arch1, arch2 in itertools.combinations_with_replacement(archs, 2):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color="white",
                marker="o",
                markerfacecolor=colors[(arch1, arch2)],
            )
        )
        labels.append(f"{arch1} | {arch2}")

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        title="x-Axis | y-Axis",
    )

    # plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_joint_train2train_v2(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Reproduce train-to-train plots from Brandfonbrener et al. Points represent a pair of
    models "joined" on model size and dataset size.
    """
    df = df.copy()

    # get data and arch values
    data = sorted(df["pretraining_data"].unique())
    archs = sorted(df["arch"].unique())
    arch_combinations = list(itertools.combinations_with_replacement(archs, 2))

    # bucket based on llama sizes and token counts
    df, _ = get_size_bucket(df)
    df, _ = get_token_bucket(df)
    # perform a self-merge to get size-and-token-paired data points
    df = pd.merge(df, df, on=["size_bucket", "token_bucket"], suffixes=["_1", "_2"])

    # set up plot
    n_cols = len(data)
    n_rows = len(arch_combinations)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), constrained_layout=True
    )

    palette = sns.color_palette("deep", len(data))

    # Plot pretraining data combinations in different subplots
    for (d1, data1), (d2, data2) in itertools.product(enumerate(data), enumerate(data)):
        for i, (arch1, arch2) in enumerate(arch_combinations):
            ax = axes[i, d1]
            data1_stem = data1.split("_")[0]
            ax.set_xlabel(f"Loss on {data1_stem} (Trained on {data1_stem})")
            if d1 == 0:
                ax.set_ylabel(f"{arch1} | {arch2}\nLoss on Data 2 (Trained on Data 2)")
            else:
                ax.set_ylabel(f"Loss on Data 2 (Trained on Data 2)")
            ax.grid(color="lightgray")
            ax.axline((5, 5), slope=1, color="lightgray", zorder=0)  # y=x line

            points = df[
                (df["arch_1"] == arch1)
                & (df["arch_2"] == arch2)
                & (df["pretraining_data_1"] == data1)
                & (df["pretraining_data_2"] == data2)
            ]

            ax.scatter(
                points[f"{data1}_val_loss_1"],
                points[f"{data2}_val_loss_2"],
                color=palette[d2],
            )

    # Add legend to the right of the figure
    handles, labels = [], []
    for d, _data in enumerate(data):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color="white",
                marker="o",
                markerfacecolor=palette[d],
            )
        )
        labels.append(_data)

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        title="Data 2",
    )

    # plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
