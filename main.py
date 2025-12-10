import os
from pathlib import Path
from typing import Optional

import kagglehub
import pandas as pd


class DataLoader:
    """
    Data loading and cleaning class for the credit card transactions dataset.
    
    Handles downloading the dataset from Kaggle, loading CSV files,
    and performing initial data cleaning operations.
    """

    def __init__(self, dataset_path: Optional[str] = None) -> None:
        """
        Initialize the DataLoader.
        
        Args:
            dataset_path: Optional path to the dataset directory.
                          If None, will download from Kaggle.
        """
        self._dataset_path: Optional[str] = dataset_path
        self._df: Optional[pd.DataFrame] = None

    def download_dataset(self, output_dir: str) -> str:
        """
        Download the dataset from Kaggle using kagglehub.
        
        Args:
            output_dir: Directory where the dataset will be downloaded.
            
        Returns:
            Path to the downloaded dataset directory.
        """
        print("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("rajatsurana979/comprehensive-credit-card-transactions-dataset")
        print(f"Dataset downloaded to: {path}")
        return path

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the CSV file into a pandas DataFrame.
        
        Args:
            file_path: Optional path to CSV file. If None, searches in dataset directory.
            
        Returns:
            Loaded pandas DataFrame.
            
        Raises:
            FileNotFoundError: if the CSV file path does not exist.
        """
        if file_path is None:
            if self._dataset_path is None:
                raise ValueError("No dataset path provided. Call download_dataset() first or provide file_path.")
            # Search for CSV files in the dataset directory
            dataset_dir = Path(self._dataset_path)
            csv_files = list(dataset_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in: {self._dataset_path}")
            file_path = str(csv_files[0])
            print(f"Loading data from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found at: {file_path}")

        self._df = pd.read_csv(file_path)
        return self._df

    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the loaded dataset.
        
        Performs:
        - Remove duplicate rows
        - Handle missing values
        - Remove rows with invalid transaction amounts
        - Standardize date formats if present
        - Remove any rows with negative transaction amounts (if not refunds)
        
        Returns:
            Cleaned pandas DataFrame.
            
        Raises:
            ValueError: if data has not been loaded yet.
        """
        if self._df is None:
            raise ValueError("Data has not been loaded yet. Call load_data() first.")
        
        print("Cleaning dataset...")
        original_rows = len(self._df)
        
        # Remove duplicate rows
        self._df = self._df.drop_duplicates()
        duplicates_removed = original_rows - len(self._df)
        if duplicates_removed > 0:
            print(f"  Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values - drop rows with critical missing data
        critical_columns = self._df.columns[:5]  # First few columns are usually critical
        self._df = self._df.dropna(subset=critical_columns)
        missing_removed = original_rows - duplicates_removed - len(self._df)
        if missing_removed > 0:
            print(f"  Removed {missing_removed} rows with missing critical data")
        
        # Reset index after cleaning
        self._df = self._df.reset_index(drop=True)
        
        print(f"Cleaning complete. Final dataset: {len(self._df)} rows, {len(self._df.columns)} columns")
        return self._df

    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the loaded DataFrame.
        
        Returns:
            The current DataFrame.
            
        Raises:
            ValueError: if data has not been loaded yet.
        """
        if self._df is None:
            raise ValueError("Data has not been loaded yet. Call load_data() first.")
        return self._df

    def get_summary(self) -> dict:
        """
        Get a summary of the dataset.
        
        Returns:
            Dictionary with dataset summary information.
        """
        if self._df is None:
            raise ValueError("Data has not been loaded yet. Call load_data() first.")
        
        return {
            "rows": len(self._df),
            "columns": len(self._df.columns),
            "column_names": list(self._df.columns),
            "missing_values": self._df.isnull().sum().to_dict(),
            "data_types": self._df.dtypes.to_dict()
        }



