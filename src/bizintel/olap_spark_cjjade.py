"""olap_spark_case.py - example.

An example of OLAP reporting with Apache Spark.

Spark is a distributed data processing engine.
It is not a replacement for Power BI.

This example uses Spark to perform the same basic OLAP operations
used in the DuckDB reporting example:

    - Slice: focus on one value from one dimension.
    - Dice: focus on selected values from multiple dimensions.
    - Rollup: summarize detailed data at a higher level.
    - Drilldown: move from a summary to more detailed data.

Run olap_case.py first to create the reporting-ready CSV file.

Author: CJ Jade
Date: 2026-07

Process:
    - Start a local Spark session.
    - Load the reporting-ready CSV file.
    - Slice sales by one region.
    - Dice sales by region and category.
    - Roll up sales into annual totals.
    - Drill down from annual totals to monthly totals.
    - Visualize the small reporting results.
    - Log a summary of findings.

Data Source:
- data/reporting/sales_reporting_case.csv

Terminal command to run this file from the root project folder:

uv run python -m bizintel.olap_spark_case

OBS:
  Don't edit this file - it should remain a working example.
  Copy it, rename it with your alias, and modify your copy.
  If you do, include your command to run it in the docstring above and in README.md.
"""

# === Section 1. Import dependencies and set up constants ===

# === DECLARE IMPORTS (bring in free code from elsewhere) ===

from pathlib import Path  # noqa: I001
from typing import Final

from datafun_toolkit.logger import log_path  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
from pyspark.sql import DataFrame, SparkSession  # type: ignore
from pyspark.sql import functions as F  # type: ignore

from bizintel.utils_logger import LOG, log_header  # type: ignore
from bizintel.utils_viz import plot_bar, plot_line

# === DECLARE GLOBAL CONSTANTS AND CONFIGURATION ===

# Folder containing reporting-ready data.
DATA_REPORTING: Final[Path] = Path("data/reporting")

# Reporting-ready CSV file created by olap_case.py.
REPORTING_FILE: Final[Path] = DATA_REPORTING / "sales_reporting_case.csv"

# Region used in the slice example.
SLICE_REGION: Final[str] = "East"

# Values used in the dice example.
DICE_REGIONS: Final[list[str]] = ["East", "West"]
DICE_CATEGORIES: Final[list[str]] = ["Clothing", "Electronics"]


# === Section 2. Define Reusable Functions ===

# === Section 2.1 DEFINE A CREATE SPARK SESSION FUNCTION ===


def create_spark_session() -> SparkSession:
    """Create a local Spark session.

    WHY: SparkSession is the starting point for working with Spark.
    It allows us to load data and create Spark DataFrames.

    Returns:
        Active SparkSession.
    """
    LOG.info("Creating Spark session")

    spark: SparkSession = (
        SparkSession.builder.appName("SmartSalesOLAP").master("local[*]").getOrCreate()
    )

    # Reduce routine Spark messages in the terminal.
    spark.sparkContext.setLogLevel("WARN")

    LOG.info("  Spark session created")
    return spark


# === Section 2.2 DEFINE A LOAD REPORTING DATA FUNCTION ===


def load_reporting_data(
    spark: SparkSession,
    file_path: Path,
) -> DataFrame:
    """Load reporting data into a Spark DataFrame.

    WHY: The reporting file already contains the sales facts
    joined with customer and product information.

    Args:
        spark: Active SparkSession.
        file_path: Path to the reporting CSV file.

    Returns:
        Spark DataFrame containing reporting data.
    """
    LOG.info("Loading reporting data")

    # header=True uses the first row as column names.
    # inferSchema=True asks Spark to detect data types.
    df_reporting: DataFrame = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(file_path))
    )

    LOG.info(f"  Loaded {df_reporting.count()} rows")
    return df_reporting


# === Section 2.3 DEFINE A SLICE FUNCTION ===

# Slice: filter one dimension to one selected value.
#
# We answer:
# What are total sales by category in one selected region?
#
# Use a slice when you want to focus on one business segment.


def slice_by_region(
    df_reporting: DataFrame,
    selected_region: str,
) -> DataFrame:
    """Slice sales by one selected region.

    Args:
        df_reporting: Complete reporting DataFrame.
        selected_region: Region to include.

    Returns:
        Spark DataFrame with Region, Category, and TotalSales.
    """
    LOG.info(f"Spark OLAP slice: {selected_region}")

    # filter() keeps only rows for the selected region.
    df_selected: DataFrame = df_reporting.filter(F.col("Region") == selected_region)

    # groupBy() creates one group for each category.
    # sum() calculates total sales for each group.
    df_slice: DataFrame = (
        df_selected.groupBy("Region", "Category")
        .agg(
            F.round(
                F.sum("SaleAmount"),
                2,
            ).alias("TotalSales")
        )
        .orderBy(F.desc("TotalSales"))
    )

    return df_slice


# === Section 2.4 DEFINE A DICE FUNCTION ===

# Dice: filter two or more dimensions.
#
# We answer:
# Which selected region and category combinations
# produce the highest sales?
#
# Use a dice when you want to examine a specific
# multidimensional subset of the data.


def dice_by_region_and_category(
    df_reporting: DataFrame,
    selected_regions: list[str],
    selected_categories: list[str],
) -> DataFrame:
    """Dice sales by selected regions and categories.

    Args:
        df_reporting: Complete reporting DataFrame.
        selected_regions: Regions to include.
        selected_categories: Categories to include.

    Returns:
        Spark DataFrame with Region, Category, and TotalSales.
    """
    LOG.info("Spark OLAP dice: region and category")

    # isin() keeps rows whose values appear in the selected lists.
    df_selected: DataFrame = df_reporting.filter(
        F.col("Region").isin(selected_regions)
        & F.col("Category").isin(selected_categories)
    )

    df_dice: DataFrame = (
        df_selected.groupBy("Region", "Category")
        .agg(
            F.round(
                F.sum("SaleAmount"),
                2,
            ).alias("TotalSales")
        )
        .orderBy(F.desc("TotalSales"))
    )

    return df_dice


# === Section 2.5 DEFINE A ROLLUP FUNCTION ===

# Rollup: move from detailed data to a higher-level summary.
#
# We answer:
# What are total sales for each year?
#
# Use a rollup when managers need a broader summary
# instead of detailed monthly or transaction-level data.


def rollup_yearly_sales(
    df_reporting: DataFrame,
) -> DataFrame:
    """Roll up sales into yearly totals.

    Args:
        df_reporting: Complete reporting DataFrame.

    Returns:
        Spark DataFrame with SalesYear and TotalSales.
    """
    LOG.info("Spark OLAP rollup: yearly sales")

    df_yearly: DataFrame = (
        df_reporting.groupBy("SalesYear")
        .agg(
            F.round(
                F.sum("SaleAmount"),
                2,
            ).alias("TotalSales")
        )
        .orderBy("SalesYear")
    )

    return df_yearly


# === Section 2.6 DEFINE A DRILLDOWN FUNCTION ===

# Drilldown: move from a summary to more detail.
#
# We answer:
# What do monthly sales look like inside one selected year?
#
# Use drilldown when a summary raises more questions.
# A yearly total may hide strong and weak months.


def drilldown_monthly_sales(
    df_reporting: DataFrame,
    selected_year: int,
) -> DataFrame:
    """Drill down from yearly sales to monthly sales.

    Args:
        df_reporting: Complete reporting DataFrame.
        selected_year: Year to investigate.

    Returns:
        Spark DataFrame with YearMonth and TotalSales.
    """
    LOG.info(f"Spark OLAP drilldown: monthly sales for {selected_year}")

    # First, filter to one selected year.
    df_selected_year: DataFrame = df_reporting.filter(
        F.col("SalesYear") == selected_year
    )

    # Then group the selected year by month.
    df_monthly: DataFrame = (
        df_selected_year.groupBy("YearMonth")
        .agg(
            F.round(
                F.sum("SaleAmount"),
                2,
            ).alias("TotalSales")
        )
        .orderBy("YearMonth")
    )

    return df_monthly


# === Section 2.7 DEFINE A SUMMARIZE FUNCTION ===


def summarize(
    df_slice: pd.DataFrame,
    df_dice: pd.DataFrame,
    df_yearly: pd.DataFrame,
    df_monthly: pd.DataFrame,
    selected_region: str,
    selected_year: int,
) -> None:
    """Log a brief summary of Spark OLAP findings.

    Args:
        df_slice: Slice result converted to pandas.
        df_dice: Dice result converted to pandas.
        df_yearly: Rollup result converted to pandas.
        df_monthly: Drilldown result converted to pandas.
        selected_region: Region used in the slice.
        selected_year: Year used in the drilldown.

    Returns:
        None
    """
    LOG.info("========================")
    LOG.info("SPARK SUMMARY")
    LOG.info("========================")

    # First row contains the largest value
    # because the slice was sorted descending.
    top_category: str = str(df_slice.iloc[0]["Category"])
    top_category_sales: float = float(df_slice.iloc[0]["TotalSales"])

    LOG.info(
        f"Slice: In {selected_region}, the leading category is "
        f"{top_category} (${top_category_sales:,.2f})"
    )

    top_region: str = str(df_dice.iloc[0]["Region"])
    top_dice_category: str = str(df_dice.iloc[0]["Category"])
    top_dice_sales: float = float(df_dice.iloc[0]["TotalSales"])

    LOG.info(
        "Dice: The strongest selected combination is "
        f"{top_region} / {top_dice_category} "
        f"(${top_dice_sales:,.2f})"
    )

    best_year: int = int(
        df_yearly.loc[
            df_yearly["TotalSales"].idxmax(),
            "SalesYear",
        ]  # type: ignore
    )

    best_year_sales: float = float(df_yearly["TotalSales"].max())

    LOG.info(f"Rollup: The strongest year is {best_year} (${best_year_sales:,.2f})")

    best_month: str = str(
        df_monthly.loc[
            df_monthly["TotalSales"].idxmax(),
            "YearMonth",
        ]
    )

    best_month_sales: float = float(df_monthly["TotalSales"].max())

    LOG.info(
        f"Drilldown: The strongest month in {selected_year} is "
        f"{best_month} (${best_month_sales:,.2f})"
    )

    LOG.info("========================")
    LOG.info("ANALYST NOTES:")
    LOG.info("Slice focuses on one dimension value.")
    LOG.info("Dice filters multiple dimensions.")
    LOG.info("Rollup creates a higher-level summary.")
    LOG.info("Drilldown reveals more detailed results.")
    LOG.info("========================")


# === DEFINE THE MAIN FUNCTION (WHERE THE MAGIC HAPPENS) ===


def main() -> None:
    """Main function to run the Spark OLAP logic.

    This is where the main logic starts
    when this script is run.
    """

    # First, log the header for the BI module.
    log_header(LOG, "BI")

    LOG.info("========================")
    LOG.info("START main()")
    LOG.info("========================")

    log_path(LOG, "Reporting data:", REPORTING_FILE)

    LOG.info("CALL a function to create a Spark session........")
    spark: SparkSession = create_spark_session()

    LOG.info("CALL a function to load reporting data........")
    df_reporting: DataFrame = load_reporting_data(
        spark,
        REPORTING_FILE,
    )

    LOG.info("CALL a function to slice sales by region........")
    df_slice_spark: DataFrame = slice_by_region(
        df_reporting,
        SLICE_REGION,
    )

    LOG.info("SHOW the slice result........")
    df_slice_spark.show()

    # The OLAP result is small after aggregation,
    # so it is safe to convert it to pandas for plotting.
    df_slice: pd.DataFrame = df_slice_spark.toPandas()

    LOG.info("CALL a function to plot the slice result........")
    plot_bar(
        df=df_slice,
        x="Category",
        y="TotalSales",
        title=f"Sales by Category in {SLICE_REGION}",
        xlabel="Category",
        ylabel="Total Sales ($)",
        palette="Blues_d",
    )

    LOG.info("CALL a function to dice sales by region and category........")
    df_dice_spark: DataFrame = dice_by_region_and_category(
        df_reporting,
        DICE_REGIONS,
        DICE_CATEGORIES,
    )

    LOG.info("SHOW the dice result........")
    df_dice_spark.show()

    df_dice: pd.DataFrame = df_dice_spark.toPandas()

    # Create one readable label for each combination.
    df_dice["RegionCategory"] = (
        df_dice["Region"].astype(str) + " / " + df_dice["Category"].astype(str)
    )

    LOG.info("CALL a function to plot the dice result........")
    plot_bar(
        df=df_dice,
        x="RegionCategory",
        y="TotalSales",
        title="Selected Region and Category Sales",
        xlabel="Region / Category",
        ylabel="Total Sales ($)",
        palette="Greens_d",
    )

    LOG.info("CALL a function to roll up yearly sales........")
    df_yearly_spark: DataFrame = rollup_yearly_sales(df_reporting)

    LOG.info("SHOW the rollup result........")
    df_yearly_spark.show()

    df_yearly: pd.DataFrame = df_yearly_spark.toPandas()

    LOG.info("CALL a function to plot yearly sales........")
    plot_bar(
        df=df_yearly,
        x="SalesYear",
        y="TotalSales",
        title="Yearly Sales Rollup",
        xlabel="Year",
        ylabel="Total Sales ($)",
        palette="Blues_d",
    )

    # Use the most recent year for the drilldown example.
    # F.max() returns a Row object, so we call first() to get the first row.
    # Then use alias() to rename the column to "LatestYear" for easier access.
    selected_year_row = df_reporting.agg(F.max("SalesYear").alias("LatestYear")).first()

    # Check if the selected year row is None,
    # which indicates that there is no reporting data available.
    # If it is None, raise a ValueError with an appropriate message.
    if selected_year_row is None:
        raise ValueError("No reporting data was available.")

    # Extract the latest year from the selected year row.
    latest_year = selected_year_row["LatestYear"]

    # Check if the latest year is None,
    # which indicates that there are no sales years in the reporting data.
    # If it is None, raise a ValueError with an appropriate message.
    if latest_year is None:
        raise ValueError("No sales year was found in the reporting data.")

    # Convert the latest year to an integer for further processing.
    selected_year: int = int(latest_year)
    LOG.info(f"Selected year for drilldown: {selected_year}")

    LOG.info("CALL a function to drill down to monthly sales........")
    df_monthly_spark: DataFrame = drilldown_monthly_sales(
        df_reporting,
        selected_year,
    )

    LOG.info("SHOW the drilldown result........")
    df_monthly_spark.show()

    # df_monthly_spark is a Spark DataFrame, which is not directly compatible with matplotlib.
    # To plot the data using matplotlib, we need to convert it to a pandas DataFrame.
    # Use the provided toPandas() method to convert it to a pandas DataFrame.
    df_monthly: pd.DataFrame = df_monthly_spark.toPandas()

    LOG.info("CALL a function to plot monthly sales........")
    plot_line(
        df=df_monthly,
        x="YearMonth",
        y="TotalSales",
        title=f"Monthly Sales Drilldown for {selected_year}",
        xlabel="Month",
        ylabel="Total Sales ($)",
    )

    LOG.info("CALL a function to summarize findings........")
    summarize(
        df_slice,
        df_dice,
        df_yearly,
        df_monthly,
        SLICE_REGION,
        selected_year,
    )

    LOG.info("CALL a function to show charts........")
    plt.show()

    # Stop Spark when the work is complete.
    spark.stop()

    LOG.info("Spark workflow complete")
    LOG.info("CLOSE chart windows to continue.")
    LOG.info("Terminate this process with CTRL+c as needed.")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    # This conditional ensures that the main() function is only executed
    # when this script is run directly, not when it is imported as a module.
    main()


def plot_top_bottom_sales(
    df_reporting: DataFrame,
    group_by_cols: list[str] | str = ["Region", "Category"],  # noqa: B006
    top_n: int = 5,
) -> None:
    """Aggregate sales and plot top N and bottom N groups.

    Args:
        df_reporting: Spark DataFrame with at least the grouping columns and SaleAmount.
        group_by_cols: Column name or list of column names to aggregate by.
        top_n: Number of top / bottom groups to show.

    Returns:
        None (creates matplotlib charts; don't call plt.show() here so caller can control display).
    """
    LOG.info("Preparing top/bottom sales chart")

    # Normalize group_by_cols to a list
    if isinstance(group_by_cols, str):
        cols = [group_by_cols]
    else:
        cols = list(group_by_cols)

    # Aggregate in Spark
    df_agg: DataFrame = df_reporting.groupBy(*cols).agg(
        F.round(F.sum("SaleAmount"), 2).alias("TotalSales")
    )

    # Build a readable label column (e.g., "East / Electronics")
    if len(cols) == 1:
        label_col = cols[0]
        df_labeled = df_agg.withColumn("Label", F.col(label_col).cast("string"))
    else:
        df_labeled = df_agg.withColumn(
            "Label", F.concat_ws(" / ", *[F.col(c).cast("string") for c in cols])
        )

    # Convert to pandas (aggregated result is small)
    df_pd = (
        df_labeled.select("Label", "TotalSales")
        .orderBy(F.desc("TotalSales"))
        .toPandas()
    )

    # Ensure numeric and handle missing
    df_pd["TotalSales"] = pd.to_numeric(df_pd["TotalSales"], errors="coerce").fillna(
        0.0
    )

    # If there are fewer rows than requested, adjust top_n
    n_available = len(df_pd)
    if n_available == 0:
        LOG.warning("No groups found to plot.")
        return

    n_show = min(top_n, max(1, n_available))

    # Select top N and bottom N
    df_top = df_pd.nlargest(n_show, "TotalSales").copy()
    df_bottom = df_pd.nsmallest(n_show, "TotalSales").copy()

    # Sort for horizontal bar aesthetics (small->large so bars grow left->right)
    df_top_sorted = df_top.sort_values("TotalSales", ascending=True)
    df_bottom_sorted = df_bottom.sort_values("TotalSales", ascending=True)

    # Plot side-by-side horizontal bar charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Top N (left)
    axes[0].barh(df_top_sorted["Label"], df_top_sorted["TotalSales"], color="tab:green")
    axes[0].set_title(f"Top {n_show} groups by Sales")
    axes[0].set_xlabel("Total Sales ($)")
    axes[0].tick_params(axis="y", labelsize=9)

    # Bottom N (right)
    axes[1].barh(
        df_bottom_sorted["Label"], df_bottom_sorted["TotalSales"], color="tab:red"
    )
    axes[1].set_title(f"Bottom {n_show} groups by Sales")
    axes[1].set_xlabel("Total Sales ($)")
    axes[1].tick_params(axis="y", labelsize=9)

    # If group labels are long, rotate x ticks a bit (horizontal bars so x ticks not usually a problem)
    for ax in axes:
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    LOG.info("Top/bottom sales charts prepared (call plt.show() in caller to display).")
