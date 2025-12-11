"""
Smart Credit Card Spend Optimizer

Main entry point for the project. Demonstrates:
- OOP concepts (classes, inheritance, encapsulation)
- Data cleaning and analysis using NumPy and Pandas
- Data visualization using Matplotlib
- Use of Python data structures (lists, tuples, dictionaries, sets)
- Recursion for directory traversal and data processing
- Credit card rewards optimization
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple

from analyzer import Analyzer
from card_optimizer import CardOptimizer
from data_loader import DataLoader
from utils import (
    build_category_hierarchy,
    count_hierarchy_nodes,
    recursive_find_files,
    recursive_total_spending,
)
from visualizer import Visualizer


def demonstrate_data_structures(df) -> Dict[str, any]:
    """
    Demonstrate use of all Python data structures:
    lists, tuples, dictionaries, and sets.
    
    Args:
        df: Pandas DataFrame to analyze.
        
    Returns:
        Dictionary showing usage of all data structures.
    """
    print("\n=== Demonstrating Python Data Structures ===")
    
    # Lists: column names and values
    column_list: List[str] = list(df.columns)
    numeric_values_list: List[float] = df.select_dtypes(include=["number"]).iloc[:, 0].dropna().tolist()[:100]
    
    # Tuples: pairs of (column_name, data_type)
    column_types_tuple: Tuple[str, ...] = tuple((col, str(dtype)) for col, dtype in df.dtypes.items())
    
    # Dictionaries: mapping column names to statistics
    stats_dict: Dict[str, float] = {
        col: df[col].mean() if df[col].dtype in ["int64", "float64"] else 0.0
        for col in df.select_dtypes(include=["number"]).columns
    }
    
    # Sets: unique values in categorical columns
    unique_categories_set: Set[str] = set()
    for col in df.select_dtypes(include=["object"]).columns[:5]:
        unique_categories_set.update(df[col].dropna().astype(str).unique()[:50])
    
    print(f"  Lists: {len(column_list)} columns, {len(numeric_values_list)} numeric values")
    print(f"  Tuples: {len(column_types_tuple)} (column, type) pairs")
    print(f"  Dictionaries: {len(stats_dict)} column statistics")
    print(f"  Sets: {len(unique_categories_set)} unique categorical values")
    
    return {
        "lists": column_list,
        "tuples": column_types_tuple,
        "dictionaries": stats_dict,
        "sets": unique_categories_set,
    }


def main() -> None:
    """
    Entry point for the project.
    
    Downloads, loads, cleans, analyzes, and visualizes the credit card transactions dataset.
    """
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output"

    print("=" * 60)
    print("Smart Credit Card Spend Optimizer")
    print("=" * 60)
    print("\nProcessing and cleaning dataset...\n")

    # Step 1: Data Loading and Cleaning
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
        print(f"Column names: {list(df.columns)[:10]}...")
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
        print(f"\nData types (first 10):")
        for col, dtype in list(summary['data_types'].items())[:10]:
            print(f"  {col}: {dtype}")
        
    except Exception as e:
        print(f"Error cleaning dataset: {e}")
        return

    # Step 2: Demonstrate Python Data Structures
    demonstrate_data_structures(cleaned_df)

    # Step 3: Recursive Directory Traversal
    print("\n=== Demonstrating Recursion ===")
    print("Recursively searching for CSV files...")
    csv_files = recursive_find_files(data_dir, ".csv")
    print(f"Found {len(csv_files)} CSV file(s) recursively")
    if csv_files:
        print(f"  Files: {csv_files[:3]}...")  # Show first 3

    # Step 4: Data Analysis using OOP
    try:
        print(f"\n=== Performing Data Analysis (OOP) ===")
        analyzer = Analyzer(cleaned_df)
        
        # Perform comprehensive analysis (inheritance demonstration)
        analysis_results = analyzer.analyze()
        
        # Get descriptive statistics
        print("\nDescriptive Statistics:")
        stats = analyzer.get_descriptive_stats()
        if not stats.empty:
            print(stats.head(10).to_string())
        
        # Analyze transaction amounts
        amount_analysis = analyzer.analyze_transaction_amounts()
        if amount_analysis:
            print("\nTransaction Amount Analysis:")
            for key, value in amount_analysis.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:,.2f}")
                else:
                    print(f"  {key}: {value:,}")
        
        # Get column information
        col_info = analyzer.get_column_info()
        print(f"\nColumn Information:")
        print(f"  Numeric columns: {len(col_info['numeric_columns'])}")
        print(f"  Categorical columns: {len(col_info['categorical_columns'])}")
        
        # Compute correlations
        correlations = analyzer.compute_correlations()
        if not correlations.empty:
            print(f"\nCorrelation matrix computed ({len(correlations)}x{len(correlations)})")
        
        # Get top categories (returns tuple)
        if col_info['categorical_columns'] and col_info['numeric_columns']:
            cat_col = col_info['categorical_columns'][0]
            val_col = col_info['numeric_columns'][0]
            top_cats, top_vals = analyzer.get_top_categories(cat_col, val_col, top_n=5)
            if top_cats:
                print(f"\nTop 5 Categories by {val_col}:")
                for cat, val in zip(top_cats, top_vals):
                    print(f"  {cat}: {val:,.2f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

    # Step 5: Create Visualizations
    try:
        print(f"\n=== Creating Visualizations ===")
        visualizer = Visualizer(cleaned_df, output_dir=str(output_dir))
        
        # Find amount column for visualization
        amount_col = analyzer._find_amount_column()
        if amount_col:
            print(f"Creating transaction distribution histogram...")
            visualizer.plot_transaction_distribution(amount_col)
        
        # Create bar chart if we have categorical and numeric columns
        numeric_cols = analyzer.get_numeric_columns()
        categorical_cols = analyzer.get_categorical_columns()
        if categorical_cols and numeric_cols:
            print(f"Creating category totals bar chart...")
            visualizer.plot_category_totals(categorical_cols[0], numeric_cols[0], top_n=10)
        
        # Create correlation heatmap if we have numeric columns
        if len(numeric_cols) >= 2:
            print(f"Creating correlation heatmap...")
            visualizer.plot_correlation_heatmap(numeric_cols[:10])
        
        # Create box plot
        if categorical_cols and numeric_cols:
            print(f"Creating box plot...")
            visualizer.plot_box_plot(categorical_cols[0], numeric_cols[0])
        
        # Create monthly trends and spending breakdown if date column exists
        date_cols = [col for col in cleaned_df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_cols and numeric_cols:
            try:
                print(f"Creating monthly spending trends plot...")
                visualizer.plot_monthly_trends(date_cols[0], numeric_cols[0])
            except Exception as e:
                print(f"Monthly trends plot skipped: {e}")
            
            if categorical_cols:
                try:
                    print(f"Creating spending breakdown pie chart...")
                    visualizer.plot_spending_breakdown(categorical_cols[0], numeric_cols[0])
                except Exception as e:
                    print(f"Spending breakdown plot skipped: {e}")
        
        print(f"\nTotal plots created: {visualizer.get_plot_count()}")
        print(f"Visualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

    # Step 6: Demonstrate Hierarchy Building (Recursion)
    print("\n=== Recursive Hierarchy Building ===")
    if categorical_cols:
        sample_categories = cleaned_df[categorical_cols[0]].dropna().astype(str).unique()[:20].tolist()
        hierarchy = build_category_hierarchy(sample_categories)
        node_count = count_hierarchy_nodes(hierarchy)
        print(f"Built hierarchy with {node_count} nodes")
        print(f"Hierarchy structure: {list(hierarchy.keys())[:5]}...")
        
        # Recursive total spending across complex categories (as per abstract)
        if numeric_cols:
            amount_col = numeric_cols[0]
            spending_by_cat = {}
            for cat in sample_categories:
                cat_df = cleaned_df[cleaned_df[categorical_cols[0]].astype(str) == cat]
                if len(cat_df) > 0:
                    spending_by_cat[cat.lower()] = float(cat_df[amount_col].sum())
            
            total_spending = recursive_total_spending(hierarchy, spending_by_cat)
            print(f"Recursive total spending across categories: ${total_spending:,.2f}")

    # Step 7: Credit Card Optimization (as per abstract)
    try:
        print(f"\n=== Credit Card Rewards Optimization ===")
        optimizer = CardOptimizer(cleaned_df)
        
        # Find amount and category columns
        amount_col = analyzer._find_amount_column() or (numeric_cols[0] if numeric_cols else None)
        category_col = "Category" if "Category" in cleaned_df.columns else (categorical_cols[0] if categorical_cols else None)
        
        if amount_col:
            # Get spending by category
            spending_by_category = optimizer.get_spending_by_category(amount_col, category_col)
            print(f"\nSpending by Category:")
            for cat, amount in sorted(spending_by_category.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {cat}: ${amount:,.2f}")
            
            # Compare all cards
            print(f"\nComparing Credit Cards:")
            card_comparison = optimizer.compare_all_cards(amount_col, category_col)
            for i, (card_name, gross_rewards, net_rewards) in enumerate(card_comparison, 1):
                print(f"\n{i}. {card_name}:")
                print(f"   Gross Rewards: ${gross_rewards:,.2f}")
                print(f"   Net Rewards: ${net_rewards:,.2f}")
            
            # Find optimal card
            best_card, best_rewards, breakdown = optimizer.find_optimal_card(amount_col, category_col)
            print(f"\n{'='*60}")
            print(f"OPTIMAL CARD: {best_card.get_name()}")
            print(f"Estimated Annual Rewards: ${best_rewards:,.2f}")
            print(f"{'='*60}")
            print(f"\nRewards Breakdown by Category:")
            for cat, reward in sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {cat}: ${reward:,.2f}")
        
    except Exception as e:
        print(f"Error during card optimization: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("=== Analysis Complete ===")
    print("=" * 60)


if __name__ == "__main__":
    main()
