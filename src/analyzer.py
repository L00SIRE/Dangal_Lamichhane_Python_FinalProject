from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


class BaseAnalyzer(ABC):
    """
    Abstract base class for data analyzers.
    Demonstrates inheritance and encapsulation in OOP design.
    """
    
    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize the base analyzer.
        
        Args:
            dataframe: Pandas DataFrame containing data to analyze.
        """
        self._df: pd.DataFrame = dataframe.copy()
        self._analysis_results: Dict[str, any] = {}
    
    @abstractmethod
    def analyze(self) -> Dict[str, any]:
        """
        Abstract method that must be implemented by subclasses.
        
        Returns:
            Dictionary containing analysis results.
        """
        pass
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the current DataFrame.
        
        Returns:
            The DataFrame being analyzed.
        """
        return self._df.copy()
    
    def save_results(self, key: str, value: any) -> None:
        """
        Save analysis results (encapsulation).
        
        Args:
            key: Key for the result.
            value: Value to store.
        """
        self._analysis_results[key] = value
    
    def get_results(self) -> Dict[str, any]:
        """
        Get all saved analysis results.
        
        Returns:
            Dictionary of saved results.
        """
        return self._analysis_results.copy()


class Analyzer(BaseAnalyzer):
    """
    Data analysis class for credit card transactions dataset.
    
    Inherits from BaseAnalyzer and performs statistical analysis,
    computes descriptive statistics, correlations, and provides
    insights on transaction patterns.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize the Analyzer with a DataFrame.
        
        Args:
            dataframe: Cleaned pandas DataFrame containing transaction data.
        """
        super().__init__(dataframe)
        self._numeric_columns: List[str] = self._identify_numeric_columns()
        self._categorical_columns: List[str] = self._identify_categorical_columns()

    def _identify_numeric_columns(self) -> List[str]:
        """
        Identify numeric columns in the dataset.
        
        Returns:
            List of column names that contain numeric data.
        """
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols

    def _identify_categorical_columns(self) -> List[str]:
        """
        Identify categorical columns in the dataset.
        
        Returns:
            List of column names that contain categorical data.
        """
        categorical_cols = self._df.select_dtypes(include=["object"]).columns.tolist()
        return categorical_cols

    def analyze(self) -> Dict[str, any]:
        """
        Perform comprehensive analysis (implements abstract method).
        
        Returns:
            Dictionary with all analysis results.
        """
        results = {
            "descriptive_stats": self.get_descriptive_stats().to_dict(),
            "correlations": self.compute_correlations().to_dict(),
            "column_info": self.get_column_info(),
            "transaction_analysis": self.analyze_transaction_amounts(),
        }
        self.save_results("full_analysis", results)
        return results

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
        amounts = amounts[~np.isnan(amounts)]  # Remove NaN values
        
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
        common_names: Set[str] = {"amount", "transaction_amount", "amt", "value", "price", "cost"}
        for col in self._df.columns:
            col_lower = col.lower()
            for name in common_names:
                if name in col_lower:
                    return col
        return None

    def get_column_info(self) -> Dict[str, any]:
        """
        Get information about all columns in the dataset.
        Demonstrates use of lists, dictionaries, and sets.
        
        Returns:
            Dictionary with column information including data types and value counts.
        """
        # Using lists, dictionaries, and sets
        column_names_list: List[str] = list(self._df.columns)
        column_types_dict: Dict[str, str] = self._df.dtypes.to_dict()
        unique_values_set: Set[int] = {len(self._df[col].unique()) for col in self._df.columns}
        
        info = {
            "total_columns": len(column_names_list),
            "column_names": column_names_list,
            "numeric_columns": self._numeric_columns,
            "categorical_columns": self._categorical_columns,
            "column_data_types": column_types_dict,
            "unique_value_counts": sorted(list(unique_values_set)),
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

    def get_top_categories(self, category_column: str, value_column: str, top_n: int = 10) -> Tuple[List[str], List[float]]:
        """
        Get top N categories by total value.
        Returns results as a tuple (demonstrating tuple usage).
        
        Args:
            category_column: Column name to group by.
            value_column: Column name to sum.
            top_n: Number of top categories to return.
            
        Returns:
            Tuple containing (list of category names, list of total values).
        """
        if category_column not in self._df.columns or value_column not in self._df.columns:
            return ([], [])
        
        top_categories = (
            self._df.groupby(category_column)[value_column]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        
        category_names: List[str] = top_categories.index.tolist()
        category_values: List[float] = top_categories.values.tolist()
        
        return (category_names, category_values)
    
    def get_numeric_columns(self) -> List[str]:
        """
        Get list of numeric column names.
        
        Returns:
            List of numeric column names.
        """
        return self._numeric_columns.copy()
    
    def get_categorical_columns(self) -> List[str]:
        """
        Get list of categorical column names.
        
        Returns:
            List of categorical column names.
        """
        return self._categorical_columns.copy()
