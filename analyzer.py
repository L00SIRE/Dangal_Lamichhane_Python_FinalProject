from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class Analyzer:
    """
    Data analysis class for credit card transactions dataset.
    
    Performs statistical analysis, computes descriptive statistics,
    correlations, and provides insights on transaction patterns.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize the Analyzer with a DataFrame.
        
        Args:
            dataframe: Cleaned pandas DataFrame containing transaction data.
        """
        self._df: pd.DataFrame = dataframe.copy()
        self._numeric_columns: List[str] = self._identify_numeric_columns()

    def _identify_numeric_columns(self) -> List[str]:
        """
        Identify numeric columns in the dataset.
        
        Returns:
            List of column names that contain numeric data.
        """
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols

    def get_descriptive_stats(self) -> pd.DataFrame:
        """
        Compute descriptive statistics for numeric columns.
        
        Returns:
            DataFrame containing mean, std, min, max, and quartiles.
        """
        if not self._numeric_columns:
            return pd.DataFrame()
        
        stats = self._df[self._numeric_columns].describe()
        return stats

    def compute_correlations(self) -> pd.DataFrame:
        """
        Compute correlation matrix for numeric columns.
        
        Returns:
            Correlation matrix as a DataFrame.
        """
        if len(self._numeric_columns) < 2:
            return pd.DataFrame()
        
        correlation_matrix = self._df[self._numeric_columns].corr()
        return correlation_matrix

    def analyze_transaction_amounts(self, amount_column: Optional[str] = None) -> Dict[str, float]:
        """
        Analyze transaction amounts using NumPy operations.
        
        Args:
            amount_column: Name of the amount column. If None, tries to find it automatically.
            
        Returns:
            Dictionary with statistical measures of transaction amounts.
        """
        if amount_column is None:
            amount_column = self._find_amount_column()
        
        if amount_column not in self._df.columns:
            return {}
        
        amounts = self._df[amount_column].values
        amounts = amounts[~np.isnan(amounts)]  # Remove NaN values
        
        if len(amounts) == 0:
            return {}
        
        analysis = {
            "total_transactions": len(amounts),
            "total_amount": float(np.sum(amounts)),
            "mean_amount": float(np.mean(amounts)),
            "median_amount": float(np.median(amounts)),
            "std_amount": float(np.std(amounts)),
            "min_amount": float(np.min(amounts)),
            "max_amount": float(np.max(amounts)),
            "q25": float(np.percentile(amounts, 25)),
            "q75": float(np.percentile(amounts, 75)),
        }
        
        return analysis

    def _find_amount_column(self) -> Optional[str]:
        """
        Try to find the transaction amount column by common names.
        
        Returns:
            Column name if found, None otherwise.
        """
        common_names = ["amount", "transaction_amount", "amt", "value", "price", "cost"]
        for col in self._df.columns:
            col_lower = col.lower()
            for name in common_names:
                if name in col_lower:
                    return col
        return None

    def get_column_info(self) -> Dict[str, any]:
        """
        Get information about all columns in the dataset.
        
        Returns:
            Dictionary with column information including data types and value counts.
        """
        info = {
            "total_columns": len(self._df.columns),
            "numeric_columns": self._numeric_columns,
            "categorical_columns": list(self._df.select_dtypes(include=["object"]).columns),
            "column_data_types": self._df.dtypes.to_dict(),
        }
        
        return info

    def analyze_by_category(self, category_column: str, value_column: str) -> pd.DataFrame:
        """
        Analyze transactions grouped by a category column.
        
        Args:
            category_column: Column name to group by.
            value_column: Column name to aggregate.
            
        Returns:
            DataFrame with aggregated statistics by category.
        """
        if category_column not in self._df.columns or value_column not in self._df.columns:
            return pd.DataFrame()
        
        grouped = self._df.groupby(category_column)[value_column].agg([
            ("count", "count"),
            ("sum", "sum"),
            ("mean", "mean"),
            ("median", "median"),
            ("std", "std"),
        ]).reset_index()
        
        return grouped

    def get_top_categories(self, category_column: str, value_column: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N categories by total value.
        
        Args:
            category_column: Column name to group by.
            value_column: Column name to sum.
            top_n: Number of top categories to return.
            
        Returns:
            DataFrame with top categories sorted by total value.
        """
        if category_column not in self._df.columns or value_column not in self._df.columns:
            return pd.DataFrame()
        
        top_categories = (
            self._df.groupby(category_column)[value_column]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        top_categories.columns = [category_column, "total_value"]
        
        return top_categories

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the current DataFrame.
        
        Returns:
            The DataFrame being analyzed.
        """
        return self._df.copy()

