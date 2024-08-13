import typing as tp
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from adjustText import adjust_text
from gseapy import enrichment_map
from scipy.spatial.distance import pdist, squareform
from scipy.stats import false_discovery_control, ttest_ind
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def parse_data(path: Path) -> pd.DataFrame:
    """Reads gene expression data from excel file.

    Args:
        path: Path to excel file

    Returns:
        Dataframe of gene expressions
    """
    return pd.read_excel(path, index_col=0)


def clean_data(df: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    """Clean the data before analysis - if the number of samples are below the
    threshold, they are dropped. Otherwise, they are filled with 0; these are
    imputed in the next step.

    Args:
        df: Dataframe to be cleaned
        threshold: Threshold for dropping the gene of interest.

    Returns:
        Dataframe of cleaned data.
    """
    return df.replace(0, np.nan).dropna(thresh=threshold).replace(np.nan, 0)


def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing data - in the "clean_data" step we keep some missing data
    assuming it is a measurement error rather than no expression. We then use
    K Nearest Neighbours to impute this data with the most likely values.

    Args:
        df: Dataframe to be imputed.

    Returns:
        Dataframe with imputed values.
    """
    non_zero = df.replace(0, np.nan)
    imputer = KNNImputer(n_neighbors=5)
    imputed = imputer.fit_transform(non_zero)
    return pd.DataFrame(imputed, columns=non_zero.columns, index=non_zero.index)


def scale_data(df: pd.DataFrame) -> pd.DataFrame:
    """Scale and normalise the data using an arcsinh transformation. This arcsinh
    transformation is commonly used in bioinformatics to get a scaling similar
    to the natural log but allowing for zero values.

    Args:
        df: Dataframe to be scaled

    Returns:
        Scaled and transformed dataframe.
    """

    # This formula scales using arcsinh but follows natural log curve.
    df = (np.arcsinh(df) / np.log(2)) - 1

    # Normalise the data
    scaler = StandardScaler()

    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)


def count_up_down_genes(df: pd.DataFrame, lfc_threshold: float = 2) -> pd.DataFrame:
    """Count the number of up and down regulated genes in the dataframe.

    Args:
        df: Dataframe to count from.
        lfc_threshold: Log fold change factor to be counted.

    Returns:
        New dataframe with counting metrics
    """
    upregulated = df[df["Log Fold Change"] > lfc_threshold]
    downregulated = df[df["Log Fold Change"] < -lfc_threshold]

    counts_df = pd.DataFrame(
        {
            "Up-regulated Genes": [len(upregulated)],
            "Down-regulated Genes": [len(downregulated)],
        }
    )

    return counts_df


def process_one_file(path: Path, exp_name: str) -> None:
    """Shortcut function for running analysis on one file.

    Args:
        path: Path to raw gene expression data
        exp_name: Experiment name
    """
    df = parse_data(path)

    group_one = df.columns[: len(df.columns) // 2]
    group_two = df.columns[len(df.columns) // 2 :]

    df = clean_data(df)

    g1, g2 = impute_data(df.loc[:, group_one]), impute_data(df.loc[:, group_two])

    df = g1.join(g2, how="inner")

    output_path = Path("../out excel") / f"{path.stem}_imputed.xlsx"

    df.to_excel(output_path)

    df = scale_data(df)

    output_path = Path("../out excel") / f"{path.stem}_scaled.xlsx"

    df.to_excel(output_path)

    lfc = df.loc[:, group_two].mean(axis=1) - df.loc[:, group_one].mean(axis=1)
    df["Log Fold Change"] = lfc

    p_values = ttest_ind(df.loc[:, group_one], df.loc[:, group_two], axis=1)[1]
    df["p_values"] = p_values

    counts_df = count_up_down_genes(df)

    output_path = Path("../out excel") / f"{path.stem}_analysis.xlsx"
    df.to_excel(output_path)

    sig_genes = make_sig_gene_list(df, group_one, group_two, lfc)

    output_path = Path("../out excel") / f"{path.stem}_sigonly.xlsx"

    sig_genes.to_excel(output_path)

    sig_gene_list = list(sig_genes.index)

    create_enrichment_map(sig_gene_list, exp_name)

    analyse_genes(sig_gene_list, exp_name)

    counts_output_path = Path("../out excel") / f"{path.stem}_gene_counts.xlsx"

    counts_df.to_excel(counts_output_path)

    fig = make_volcano_plot(df)

    output_path = Path("../img") / f"{path.stem}_volcano.png"

    fig.savefig(output_path, dpi=800)

    fig = make_pca_plot(df, group_one, group_two)

    output_path = Path("../img") / f"{path.stem}_pca.png"

    fig.savefig(output_path, dpi=800)

    # Plots
    make_heatmap(df, exp_name)
    make_significant_genes_heatmap(df, sig_genes, exp_name)
    make_top_20_box_plot(df, group_one, group_two, exp_name, Path("../img"))

    specific_pathways = [
        "Oxidative Phosphorylation",
        "Myogenesis",
        "Hypoxia",
        "TNF-Alpha Signaling via NF-kB",
        "Oxidative phosphorylation",
        "Themogenesis",
        "Cardiac muscle contraction",
    ]

    plot_specific_pathways(sig_gene_list, exp_name, specific_pathways, Path("../img"))


def get_colormap_function(log_fold_change_thresh: float, p_val_thresh: float):
    """Calculate the colormap value for volcano plots

    Args:
        p_val_thresh: Significant p-value threshold
        log_fold_change_thresh: Significant log fold change threshold

    Returns:
        Function for getting the correct color for volcano plots.
    """

    def new_func(lfc: float, pval: float):
        if pval < p_val_thresh and lfc > log_fold_change_thresh:
            return 1
        elif pval < p_val_thresh and lfc < -log_fold_change_thresh:
            return -1
        else:
            return 0

    return new_func


def make_volcano_plot(
    df: pd.DataFrame,
    p_val_threshold: float = 0.05,
    lfc_threshold: float = 0.3,
    max_named_genes: int = 25,
    cmap: str = "coolwarm",
    alpha: float = 0.8,
    text_fontsize: float = 17,
    fig_size: tp.Tuple[float, float] = (10, 10),
    color_above_threshold: str = "red",
    color_below_threshold: str = "gray",
) -> plt.Figure:
    """Create a volcano plot for differential expression analysis

    Args:
        df: Dataframe containing cleaned and scaled values.
        p_val_threshold:p-value threshold to consider significant
        lfc_threshold: Log fold change threshold to consider significant
        max_named_genes: Maximum number of genes to label with text on plot
        cmap: matplotlib colormap to use
        alpha: Opacity of points on plot
        text_fontsize: Fontsize of text to label points with
        fig_size: Size of figure to export
        color_above_threshold: Color for the upregulated genes
        color_below_threshold: Color for the downregulated genes

    Returns:
        matplotlib figure for the volcano plot
    """

    fig, ax = plt.subplots(figsize=fig_size)
    df["absLFC"] = df["Log Fold Change"].abs()
    df.sort_values("absLFC", ascending=False, inplace=True)

    # Generate the colormap function for thresholds
    cmap_func = get_colormap_function(lfc_threshold, p_val_threshold)

    # Add the colors to the dataframe
    colors = df.apply(lambda x: cmap_func(x["Log Fold Change"], x["p_values"]), axis=1)

    # Plot the points
    sc = ax.scatter(
        df["Log Fold Change"],
        -np.log10(df["p_values"]),
        c=colors,
        cmap=cmap,
        alpha=alpha,
    )

    # Add significant lines to the plot
    ax.axvline(x=lfc_threshold, color=color_above_threshold)
    ax.axvline(x=-lfc_threshold, color=color_above_threshold)
    ax.axhline(y=-np.log10(p_val_threshold), color=color_above_threshold)

    # Add text labels to significant points
    texts = []
    i = 0
    j = 0
    for x, y, s in zip(df["Log Fold Change"], -np.log10(df["p_values"]), df.index):
        if (
            x > lfc_threshold
            and y > -np.log10(p_val_threshold)
            and not s.startswith("LOC")
            and i < max_named_genes
        ):
            texts.append(ax.text(x, y, s, fontsize=text_fontsize))
            i += 1
        elif (
            x < lfc_threshold
            and y > -np.log10(p_val_threshold)
            and not s.startswith("LOC")
            and j < max_named_genes
        ):
            texts.append(ax.text(x, y, s, fontsize=text_fontsize))
            j += 1

    # Move the text around to avoid overlapping
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5))

    max_lfc = df["Log Fold Change"].abs().max()
    ax.set_xlim(-max_lfc, max_lfc)
    ax.spines[["top", "right"]].set_visible(False)

    ax.set_xlabel("Log Fold Change")
    ax.set_ylabel("-Log10 p_values")

    sc.set_clim(-max_lfc, max_lfc)

    plt.tight_layout()

    return fig


def make_sig_gene_list(
    df: pd.DataFrame,
    group_one: tp.List[str],
    group_two: tp.List[str],
    lfc: pd.DataFrame,
) -> pd.DataFrame:
    """Make a list of significant genes given a gene expression dataset and
    list of log fold changes.

    Args:
        df: Raw gene expression dataframebb
        group_one: List of samples in group one
        group_two: List of samples in group two
        lfc: Dataframe of log fold changes

    Returns:
        [TODO:return]
    """
    p_values = ttest_ind(df.loc[:, group_one], df.loc[:, group_two], axis=1)[1]
    fdr_values = false_discovery_control(p_values)
    df["fdr"] = fdr_values

    p_values_df = pd.DataFrame(
        {"p_values": p_values, "FDR": fdr_values, "Log Fold Change": lfc},
        index=df.index,
    )
    sig_genes = p_values_df[
        (p_values_df["p_values"] < 0.05) & (df["Log Fold Change"].abs() > 0.3)
    ]

    return sig_genes


def analyse_genes(
    gene_list: tp.List[str],
    exp_name: str,
    output_path: Path = Path("./docs/enriched.xlsx"),
) -> None:
    """Analyse the genes using gseapy

    Args:
        gene_list: Gene list to analyse
        exp_name: Experiment name
        output_path: Output path to save results to
    """
    try:
        enriched = gp.enrichr(
            gene_list=gene_list,
            organism="human",
            gene_sets=["KEGG_2021_Human", "MSigDB_Hallmark_2020"],
        )

        dotplot_output_path = Path(f"../img/img_dotplot_{exp_name}.png")
        gp.dotplot(
            enriched.results,
            x="Gene_set",
            size=10,
            marker="o",
            xticklabels_rot=45,
            ofname=dotplot_output_path,
        )

        barplot_output_path = Path(f"../img/img_barplot_{exp_name}.png")

        gp.barplot(
            enriched.results,
            column="P-value",
            group="Gene_set",
            color=["darkorange", "gold"],
            ofname=barplot_output_path,
        )

        enriched_output_path = Path(f"../out excel/enriched_{exp_name}.xlsx")
        enriched.results.to_excel(enriched_output_path)

    except Exception as e:
        print("An error occurred during enrichment analysis:", e)


def compare_all_sigonly_files(input_folder: Path, output_path: Path) -> pd.DataFrame:
    """Find significant genes in multiple files in a folder

    Args:
        input_folder: Input folder to examine
        output_path: Output path to save results to

    Returns:
        Dataframe with crossover genes
    """
    sig_files = [file for file in input_folder.glob("*_sigonly.xlsx")]
    if not sig_files:
        print("No _sigonly files found in the folder.")

    all_sig_genes = {}

    for sig_file in sig_files:
        experiment_name = sig_file.stem.replace("_sigonly", "")
        df = pd.read_excel(sig_file, index_col=0)
        all_sig_genes[experiment_name] = set(df.index)

        common_genes = list(set.intersection(*all_sig_genes.values()))

        crossover_genes_df = pd.DataFrame(index=common_genes)
        for set_name, genes in all_sig_genes.items():
            crossover_genes_df[set_name] = [gene in genes for gene in common_genes]

        crossover_genes_output_path = output_path / "crossover_genes.xlsx"
        crossover_genes_df.to_excel(crossover_genes_output_path)

    return crossover_genes_df


def make_box_plot(
    df: pd.DataFrame, gene_list: list, group_one: tp.List[str], group_two: tp.List[str]
) -> plt.Figure:
    """Make a box plot for gene expression

    Args:
        df: Dataframe with raw gene expression
        gene_list: List of genes to plot
        group_one: Column sample labels for group one
        group_two: Column sample labels for group two

    Returns:
        matplotlib figure of the boxplot
    """
    num_genes = len(gene_list)
    num_cols = 2
    num_rows = (num_genes + 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows))
    axes = np.array(axes).flatten()

    for i, gene_name in enumerate(gene_list):
        ax = axes[i]
        if i < num_genes:
            try:
                gene_data_group_one = df.loc[gene_name, group_one]
                gene_data_group_two = df.loc[gene_name, group_two]
                ax.boxplot(
                    [gene_data_group_one, gene_data_group_two],
                    labels=[group_one, group_two],
                )
                ax.set_title(
                    f"Expression of {gene_name} between {group_one} and {group_two}"
                )
                ax.set_ylabel("Expression Level")
                ax.set_xlabel("Groups")
            except KeyError:
                ax.set_title(f"Gene {gene_name} not found in the dataset")
                ax.axis("off")

    plt.tight_layout()
    plt.show()


def create_enrichment_map(
    gene_list: tp.List[str],
    exp_name: str,
    output_path: Path = Path("../img/enrichment_map.html"),
) -> None:
    """Make an enrichr map and save it to file

    Args:
        gene_list: List of genes to enrich
        exp_name: Expriment name
        output_path: Path to save plot to
    """
    try:
        enriched = gp.enrichr(
            gene_list=gene_list,
            organism="human",
            gene_sets=["KEGG_2021_Human", "MSigDB_Hallmark_2020"],
        )

        emap_output_path = Path(f"../img/enrichment_map_{exp_name}.html")
        enrichment_map(enriched.results, emap_output_path)

    except Exception as e:
        print("An error occurred during enrichment map:", e)


def make_pca_plot(
    df: pd.DataFrame, group_one: tp.List[str], group_two: tp.List[str]
) -> plt.Figure:
    """PCA plot the samples using the gene expression

    Args:
        df: Raw gene expression values
        group_one: Column labels for the first group of samples
        group_two: Column labels for the second group of samples

    Returns:
        matplotlib figure of the PCA plot
    """
    pca = PCA(n_components=2)
    groups = [0] * len(group_one) + [1] * len(group_two)
    df = df.drop(
        columns=["Log Fold Change", "p_values", "fdr", "ID", "FDR", "absLFC"],
        errors="ignore",
    )
    pca_result = pca.fit_transform(df.T)
    df_pca = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
    print(df_pca)
    df_pca["Group"] = groups

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(
        df_pca["PC1"], df_pca["PC2"], c=df_pca["Group"], cmap="coolwarm", alpha=0.8
    )

    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Groups")
    ax.add_artist(legend1)

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA of Gene Expression Data")

    plt.tight_layout()

    return fig


def make_heatmap(df: pd.DataFrame, exp_name: str) -> None:
    """Make a heatmap of gene expression

    Args:
        df: Raw gene expression values
        exp_name: Experiment name
    """
    cols = [col for col in df.columns if col not in ["breadth", "uniprot_id", "avg"]]
    short_cols = [col[:20] for col in cols]
    short_cols = [short_cols[i] + str(i) for i in range(len(short_cols))]

    log_fold_change_heatmap = np.log2(df[cols])

    data_dist = pdist(log_fold_change_heatmap.values.T)
    heatmap_data = squareform(data_dist)

    data = [go.Heatmap(z=heatmap_data, x=short_cols, y=short_cols, colorscale="ylgnbu")]

    layout = go.Layout(
        title=f"Transcription profiling of {exp_name} samples",
        autosize=False,
        width=900,
        height=900,
        margin=dict(l=200, b=200, pad=4),
        xaxis=dict(
            showgrid=False,
            tickmode="linear",
            dtick=1,
            tickangle=-45,
        ),
        yaxis=dict(
            showgrid=False,
            tickmode="linear",
            dtick=1,
        ),
    )

    fig = go.Figure(data=data, layout=layout)
    output_path = Path("../img") / f"{exp_name}_heatmap.html"
    fig.write_html(str(output_path))
    print(f"Heatmap saved to {output_path}")


def make_significant_genes_heatmap(
    df: pd.DataFrame, sig_genes: pd.DataFrame, exp_name: str
) -> None:
    """Make a heatmap only for significant genes

    Args:
        df: Raw gene expressions
        sig_genes: Significant genes to plot
        exp_name: Experiment name
    """
    sig_gene_names = sig_genes.index.tolist()

    sig_gene_df = np.log2(df.loc[sig_gene_names, :])

    data = [
        go.Heatmap(
            z=sig_gene_df.values,
            x=sig_gene_df.columns,
            y=sig_gene_df.index,
            colorscale="ylgnbu",
        )
    ]

    layout = go.Layout(
        title=f"Significant Genes Heatmap of {exp_name}",
        autosize=False,
        width=1000,
        height=1000,
        margin=dict(l=200, b=200, pad=4),
        xaxis=dict(
            showgrid=False,
            tickmode="linear",
            dtick=1,
            tickangle=-45,
        ),
        yaxis=dict(
            showgrid=False,
            tickmode="linear",
            dtick=1,
            tickangle=-45,
        ),
    )

    fig = go.Figure(data=data, layout=layout)
    output_path = Path("../img") / f"{exp_name}_significant_genes_heatmap.html"
    fig.write_html(str(output_path))
    print(f"Significant Genes Heatmap saved to {output_path}")


def make_top_20_box_plot(
    df: pd.DataFrame,
    group_one: tp.List[str],
    group_two: tp.List[str],
    exp_name: str,
    output_path: Path,
) -> plt.Figure:
    """Make a box plot for the top 20 significant genes.

    Args:
        df: Dataframe of raw gene expressions
        group_one: List of column labels for the first group
        group_two: List of column labels for the second group
        output_path: Output path to save plot to
        exp_name: Experiment name

    Returns:
        matplotlib figure of the boxplot
    """
    top_10_pos = df.nlargest(10, "Log Fold Change").index
    top_10_neg = df.nsmallest(10, "Log Fold Change").index
    top_20_genes = top_10_pos.union(top_10_neg)

    num_genes = len(top_20_genes)
    num_rows = 2
    num_cols = (num_genes + 1) // num_rows
    fig, axes = plt.subplots(num_cols, num_rows, figsize=(6 * num_rows, 12))
    axes = np.array(axes).flatten()

    for i, gene_name in enumerate(top_20_genes):
        ax = axes[i]
        if i < num_genes:
            try:
                gene_data_group_one = df.loc[gene_name, group_one]
                gene_data_group_two = df.loc[gene_name, group_two]
                ax.boxplot(
                    [gene_data_group_one, gene_data_group_two],
                    labels=["Group One", "Group Two"],
                )
                ax.set_title(
                    f"Expression of {gene_name} between Group One and Group Two"
                )
                ax.set_ylabel("Expression Level")
                ax.set_xlabel("Groups")
            except KeyError:
                ax.set_title(f"Gene {gene_name} not found in the dataset")
                ax.axis("off")

    plt.tight_layout()
    output_path = output_path / f"top_20_genes_box_plot_{exp_name}.png"
    fig.savefig(output_path, dpi=800)
    print(f"Box plot saved to {output_path}")
    return fig


def plot_specific_pathways(
    gene_list: tp.List[str],
    exp_name: str,
    pathways: tp.List[str],
    output_path: Path = Path("../img"),
) -> None:
    """Plot pathway analysis for specific pathways

    Args:
        gene_list: List of genes to perform pathway analysis on
        exp_name: Experiment name
        pathways: List of pathways to analyse
        output_path: Outpuit path to save results to
    """
    try:
        enriched = gp.enrichr(
            gene_list=gene_list,
            organism="human",
            gene_sets=["KEGG_2021_Human", "MSigDB_Hallmark_2020"],
        )

        filtered_results = enriched.results[enriched.results["Term"].isin(pathways)]

        if filtered_results.empty:
            print(
                "None of the specified pathways are present in the enrichment results."
            )
            return

        dotplot_output_path = (
            output_path / f"img_dotplot_{exp_name}_specific_pathways.png"
        )
        barplot_output_path = (
            output_path / f"img_barplot_{exp_name}_specific_pathways.png"
        )
        enrichment_map_output_path = (
            output_path / f"enrichment_map_{exp_name}_specific_pathways.html"
        )

        gp.dotplot(
            filtered_results,
            x="Gene_set",
            size=10,
            marker="o",
            xticklabels_rot=45,
            ofname=dotplot_output_path,
        )

        gp.barplot(
            filtered_results,
            column="P-value",
            group="Gene_set",
            color=["darkred", "darkblue"],
            ofname=barplot_output_path,
        )

        enrichment_map(filtered_results, enrichment_map_output_path)

        print(f"Specific pathways dot plot saved to {dotplot_output_path}")
        print(f"Specific pathways bar plot saved to {barplot_output_path}")
        print(f"Specific pathways enrichment map saved to {enrichment_map_output_path}")

    except Exception as e:
        print("An error occurred during pathway analysis:", e)


def process_all_files(directory: Path) -> None:
    """Process all the files in a directory.

    Args:
        directory: Path of directory to analyse
    """
    excel_files = directory.glob("*.xlsx")
    for file in excel_files:
        experiment_name = file.stem.replace(" ", "_")
        process_one_file(file, experiment_name)


if __name__ == "__main__":
    # Note - data available in paper
    directory = Path("../data/Prdx2/Stim comparrisons")
    process_all_files(directory)
