from pathlib import Path
from typing import List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Visualizer:
    """
    Data visualization class for credit card transactions dataset.
    
    Creates various types of plots using Matplotlib to visualize
    transaction patterns and insights.
    """

    def __init__(self, dataframe: pd.DataFrame, output_dir: Optional[str] = None) -> None:
        """
        Initialize the Visualizer with a DataFrame.
        
        Args:
            dataframe: Pandas DataFrame containing transaction data.
            output_dir: Optional directory to save plots. If None, plots are displayed.
        """
        self._df: pd.DataFrame = dataframe.copy()
        self._output_dir: Optional[str] = output_dir
        self._plot_count: int = 0
        if self._output_dir:
            Path(self._output_dir).mkdir(parents=True, exist_ok=True)

    def _save_or_show(self, filename: str) -> None:
        """
        Save plot to file or display it.
        
        Args:
            filename: Name of the file to save.
        """
        if self._output_dir:
            filepath = Path(self._output_dir) / filename
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {filepath}")
            self._plot_count += 1
        else:
            plt.show()
        plt.close()

    def plot_transaction_distribution(
        self, 
        amount_column: str, 
        bins: int = 50,
        title: str = "Transaction Amount Distribution"
    ) -> None:
        """
        Create a histogram showing the distribution of transaction amounts.
        Type 1: Histogram
        
        Args:
            amount_column: Name of the amount column.
            bins: Number of bins for the histogram.
            title: Title of the plot.
        """
        if amount_column not in self._df.columns:
            print(f"Column '{amount_column}' not found in dataset.")
            return
        
        amounts = self._df[amount_column].dropna()
        
        plt.figure(figsize=(10, 6))
        plt.hist(amounts, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
        plt.xlabel("Transaction Amount", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend(["Transaction Frequency"])
        
        self._save_or_show("transaction_distribution.png")

    def plot_category_totals(
        self,
        category_column: str,
        value_column: str,
        top_n: int = 10,
        title: str = "Top Categories by Total Value"
    ) -> None:
        """
        Create a bar chart showing top categories by total value.
        Type 2: Bar Chart
        
        Args:
            category_column: Column name to group by.
            value_column: Column name to sum.
            top_n: Number of top categories to display.
            title: Title of the plot.
        """
        if category_column not in self._df.columns or value_column not in self._df.columns:
            print(f"One or both columns not found in dataset.")
            return
        
        top_categories = (
            self._df.groupby(category_column)[value_column]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        
        plt.figure(figsize=(12, 6))
        top_categories.plot(kind="bar", color="steelblue", edgecolor="black")
        plt.xlabel(category_column, fontsize=12)
        plt.ylabel(f"Total {value_column}", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")
        plt.legend([f"Total {value_column}"])
        plt.tight_layout()
        
        self._save_or_show("category_totals.png")

    def plot_time_series(
        self,
        date_column: str,
        value_column: str,
        title: str = "Transaction Trends Over Time"
    ) -> None:
        """
        Create a line plot showing trends over time.
        Type 3: Line Plot
        
        Args:
            date_column: Column name containing dates.
            value_column: Column name to plot.
            title: Title of the plot.
        """
        if date_column not in self._df.columns or value_column not in self._df.columns:
            print(f"One or both columns not found in dataset.")
            return
        
        try:
            df_copy = self._df.copy()
            df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors="coerce")
            df_copy = df_copy.dropna(subset=[date_column, value_column])
            df_copy = df_copy.sort_values(date_column)
            
            daily_totals = df_copy.groupby(df_copy[date_column].dt.date)[value_column].sum()
            
            plt.figure(figsize=(12, 6))
            plt.plot(daily_totals.index, daily_totals.values, marker="o", linewidth=2, markersize=4, label=value_column)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel(f"Total {value_column}", fontsize=12)
            plt.title(title, fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha="right")
            plt.legend()
            plt.tight_layout()
            
            self._save_or_show("time_series.png")
        except Exception as e:
            print(f"Error creating time series plot: {e}")

    def plot_correlation_heatmap(
        self,
        numeric_columns: Optional[List[str]] = None,
        title: str = "Correlation Heatmap"
    ) -> None:
        """
        Create a heatmap showing correlations between numeric columns.
        Type 4: Heatmap
        
        Args:
            numeric_columns: List of numeric column names. If None, uses all numeric columns.
            title: Title of the plot.
        """
        if numeric_columns is None:
            numeric_columns = self._df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            print("Need at least 2 numeric columns for correlation heatmap.")
            return
        
        correlation_matrix = self._df[numeric_columns].corr()
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(correlation_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar(im, label="Correlation Coefficient")
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha="right")
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.title(title, fontsize=14, fontweight="bold")
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                        ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        self._save_or_show("correlation_heatmap.png")

    def plot_box_plot(
        self,
        category_column: str,
        value_column: str,
        title: str = "Distribution by Category"
    ) -> None:
        """
        Create a box plot showing distribution of values by category.
        Type 5: Box Plot
        
        Args:
            category_column: Column name to group by.
            value_column: Column name to plot.
            title: Title of the plot.
        """
        if category_column not in self._df.columns or value_column not in self._df.columns:
            print(f"One or both columns not found in dataset.")
            return
        
        categories = self._df[category_column].unique()[:10]  # Limit to top 10 categories
        data_to_plot = [self._df[self._df[category_column] == cat][value_column].dropna() 
                        for cat in categories]
        
        plt.figure(figsize=(12, 6))
        bp = plt.boxplot(data_to_plot, labels=categories)
        plt.xlabel(category_column, fontsize=12)
        plt.ylabel(value_column, fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        
        self._save_or_show("box_plot.png")

    def plot_monthly_trends(
        self,
        date_column: str,
        value_column: str,
        title: str = "Monthly Spending Trends"
    ) -> None:
        """
        Create a line plot showing monthly spending trends.
        This matches the abstract requirement for "monthly trends" visualization.
        
        Args:
            date_column: Column name containing dates.
            value_column: Column name to plot.
            title: Title of the plot.
        """
        if date_column not in self._df.columns or value_column not in self._df.columns:
            print(f"One or both columns not found in dataset.")
            return
        
        try:
            df_copy = self._df.copy()
            df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors="coerce")
            df_copy = df_copy.dropna(subset=[date_column, value_column])
            df_copy = df_copy.sort_values(date_column)
            
            # Group by month
            df_copy['year_month'] = df_copy[date_column].dt.to_period('M')
            monthly_totals = df_copy.groupby('year_month')[value_column].sum()
            
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(monthly_totals)), monthly_totals.values, 
                    marker="o", linewidth=2, markersize=6, label="Monthly Spending")
            plt.xlabel("Month", fontsize=12)
            plt.ylabel(f"Total {value_column}", fontsize=12)
            plt.title(title, fontsize=14, fontweight="bold")
            plt.xticks(range(len(monthly_totals)), 
                      [str(period) for period in monthly_totals.index], 
                      rotation=45, ha="right")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            self._save_or_show("monthly_trends.png")
        except Exception as e:
            print(f"Error creating monthly trends plot: {e}")
    
    def plot_spending_breakdown(
        self,
        category_column: str,
        value_column: str,
        title: str = "Spending Breakdown by Category"
    ) -> None:
        """
        Create a pie chart showing spending breakdown by category.
        This matches the abstract requirement for "spending breakdowns" visualization.
        
        Args:
            category_column: Column name to group by.
            value_column: Column name to sum.
            title: Title of the plot.
        """
        if category_column not in self._df.columns or value_column not in self._df.columns:
            print(f"One or both columns not found in dataset.")
            return
        
        category_totals = (
            self._df.groupby(category_column)[value_column]
            .sum()
            .sort_values(ascending=False)
            .head(10)  # Top 10 categories
        )
        
        plt.figure(figsize=(10, 8))
        plt.pie(category_totals.values, labels=category_totals.index, autopct='%1.1f%%', startangle=90)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.axis('equal')
        plt.tight_layout()
        
        self._save_or_show("spending_breakdown.png")

    def get_plot_count(self) -> int:
        """
        Get the number of plots created.
        
        Returns:
            Number of plots created.
        """
        return self._plot_count
