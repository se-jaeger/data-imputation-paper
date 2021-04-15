import matplotlib.pyplot as plt
import seaborn as sns


def plot_metric_vs_missing_fraction(data, y, ylabel, ci, fpath, fname):
    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(
        data=data,
        x="Missing Fraction",
        y=y,
        hue="Imputer",
        style="Imputer",
        ci=ci
    )
    ax.set(ylabel=ylabel)
    ax.set_xticks(sorted(data["Missing Fraction"].unique()))
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")  # place legend in top right corner
    fpath.mkdir(parents=True, exist_ok=True)
    plt.savefig(fpath/fname)


def plot_rank_vs_fraction_by_type(data, ci, fpath, fname):
    sns.relplot(
        data=data,
        x="Missing Fraction",
        y="Imputation Rank",
        hue="Imputer",
        style="Imputer",
        col="Missing Type",
        row="metric",
        kind="line",
        height=5,
        ci=ci,
        col_order=["MCAR", "MAR", "MNAR"]
    )
    # ax.set_xticks(sorted(results["Missing Fraction"].unique()))
    plt.gcf().subplots_adjust(bottom=0.15, left=0.05)  # avoid x/ylabel cutoff in SVG export
    fpath.mkdir(parents=True, exist_ok=True)
    plt.savefig(fpath/fname)


def draw_cat_box_plot(data, y, ylim, fpath, fname, col_order=["MCAR", "MAR", "MNAR"], hue_order=None, row_order=None):
    g = sns.catplot(
        x="Missing Fraction",
        y=y,
        hue="Imputation Method",
        col="Missing Type",
        row="metric",
        data=data,
        kind="box",
        height=4,
        col_order=col_order,
        row_order=row_order,
        hue_order=hue_order,
        margin_titles=True
    )
    g.set_titles(row_template="{row_name}", col_template="{col_name}", size=18).set(ylim=ylim)

    plt.tight_layout(rect=(0, 0, 0.85, 1))
    fpath.mkdir(parents=True, exist_ok=True)
    plt.savefig(fpath/fname)
