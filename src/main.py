from pathlib import Path

from data_loader import DataLoader


def main() -> None:
    """
    Entry point for the project.
    
    Downloads, loads, and cleans the credit card transactions dataset.
    """
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    print("=== Credit Card Spending and Rewards Analyzer ===")
    print("Processing and cleaning dataset...\n")

    # Initialize DataLoader
    loader = DataLoader()

    # Download dataset if not already present
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        print("Dataset not found locally. Downloading from Kaggle...")
        dataset_path = loader.download_dataset(str(data_dir))
        loader._dataset_path = dataset_path
    else:
        print(f"Using existing dataset in: {data_dir}")
        loader._dataset_path = str(data_dir)

    # Load the data
    try:
        df = loader.load_data()
        print(f"\nDataset loaded successfully.")
        print(f"Initial rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Column names: {list(df.columns)[:10]}...")  # Show first 10 columns
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Clean the data
    try:
        cleaned_df = loader.clean_data()
        print(f"\nDataset cleaned and ready for analysis.")
        
        # Display summary
        summary = loader.get_summary()
        print(f"\n=== Dataset Summary ===")
        print(f"Total rows: {summary['rows']}")
        print(f"Total columns: {summary['columns']}")
        print(f"\nData types:")
        for col, dtype in list(summary['data_types'].items())[:10]:
            print(f"  {col}: {dtype}")
        
        print(f"\nDataset is ready for analysis!")
        
    except Exception as e:
        print(f"Error cleaning dataset: {e}")
        return


if __name__ == "__main__":
    main()


