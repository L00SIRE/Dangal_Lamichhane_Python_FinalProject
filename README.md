# Smart Credit Card Spend Optimizer

## Project Title
Smart Credit Card Spend Optimizer

## Team Members
- Suman Dangal
- Diwash Lamichhane

## Course Information
**Course:** CISC 288-01, Fall 2025  
**Instructor:** Trina Dutta Barlow  
**Due Date:** December 09, 2025

---

## Project Description

This project is a Python tool designed to help users track their spending and maximize credit card rewards. Using the "Comprehensive Credit Card Transactions Dataset" from Kaggle, the program analyzes transaction patterns, including dates, merchant names, and spending categories. The main goal is to show users exactly where their money goes and calculate which credit card would give them the most cash back based on their spending habits.

The application is built using Object-Oriented Programming (OOP) to keep the code organized. We created specific classes to load data, calculate statistics, and simulate different credit card types (like Travel cards vs. Standard cards). The project uses the Pandas library to clean and organize the data, while Matplotlib is used to generate clear graphs of monthly trends and spending breakdowns. We also implemented a recursive function to total up spending across complex categories. This tool makes it easy to visualize financial habits and optimize rewards strategies.

## Dataset Details

**Dataset Name:** Comprehensive Credit Card Transactions Dataset  
**Source:** Kaggle  
**Link:** `https://www.kaggle.com/datasets/rajatsurana979/comprehensive-credit-card-transactions-dataset`

### Preprocessing Steps

1. **Data Download**: The dataset is automatically downloaded from Kaggle using the `kagglehub` library when the project is first run
2. **Data Loading**: CSV files are loaded into Pandas DataFrames
3. **Data Cleaning**:
   - Removal of duplicate rows
   - Handling missing values in critical columns
   - Data type validation and conversion
   - Index reset after cleaning operations
4. **Data Transformation**: 
   - Column type identification (numeric vs categorical)
   - Date parsing and standardization (where applicable)
   - Data normalization for analysis

---

## Project Structure

```
Dangal_Lamichhane_Python_FinalProject/
├── src/
│   ├── main.py              # Main entry point orchestrating all components
│   ├── data_loader.py       # DataLoader class for data loading and cleaning
│   ├── analyzer.py          # Analyzer class with inheritance (BaseAnalyzer)
│   ├── card_optimizer.py    # CardOptimizer class for simulating credit cards
│   ├── visualizer.py        # Visualizer class for creating plots
│   └── utils.py             # Utility functions including recursion
├── data/                    # Dataset directory (auto-created, gitignored)
├── output/                  # Generated visualizations (auto-created, gitignored)
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

---

## Instructions to Run

### Prerequisites

Before running the project, ensure you have:

1. **Python 3.8 or higher** installed on your system
   - Check your Python version: `python --version` or `python3 --version`
   - Download Python from [python.org](https://www.python.org/downloads/) if needed

2. **pip** (Python package manager)
   - Usually comes with Python installation
   - Verify installation: `pip --version` or `pip3 --version`

3. **Kaggle account** (for dataset download)
   - Create a free account at [kaggle.com](https://www.kaggle.com/)
   - Set up Kaggle API credentials (optional, but recommended):
     - Go to Kaggle Account Settings → API → Create New Token
     - Place `kaggle.json` in `~/.kaggle/` directory (or follow kagglehub documentation)

### Installation Steps

1. **Clone or download the repository:**
   ```bash
   git clone https://github.com/L00SIRE/Dangal_Lamichhane_Python_FinalProject.git
   cd Dangal_Lamichhane_Python_FinalProject
   ```
   
   Or download the ZIP file and extract it, then navigate to the project directory.

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or if you have both Python 2 and 3:
   ```bash
   pip3 install -r requirements.txt
   ```
   
   This will install:
   - `numpy` - Numerical computing
   - `pandas` - Data manipulation and analysis
   - `matplotlib` - Data visualization
   - `kagglehub` - Kaggle dataset download

3. **Run the project:**
   ```bash
   python src/main.py
   ```
   
   Or:
   ```bash
   python3 src/main.py
   ```

### Expected Runtime

- **First run**: 2-5 minutes (downloads dataset from Kaggle)
- **Subsequent runs**: 30 seconds - 2 minutes (uses cached dataset)

### Output Locations

After running, you will find:

- **Console Output**: Analysis results, statistics, and optimal card recommendation printed to terminal
- **Visualizations**: PNG files saved in `output/` directory:
  - `transaction_distribution.png` - Histogram of transaction amounts
  - `category_totals.png` - Bar chart of top spending categories
  - `correlation_heatmap.png` - Correlation matrix visualization
  - `box_plot.png` - Distribution comparison by category
  - `monthly_trends.png` - Monthly spending trends (if date column available)
  - `spending_breakdown.png` - Pie chart of spending by category

- **Dataset**: Downloaded CSV files stored in `data/` directory

### Troubleshooting

**Issue: ModuleNotFoundError**
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue: Kaggle dataset download fails**
- **Solution**: 
  - Verify your internet connection
  - Check Kaggle API credentials if using kagglehub
  - The dataset will be downloaded automatically on first run

**Issue: Permission denied errors**
- **Solution**: 
  - On Linux/Mac: Use `sudo` if needed or check directory permissions
  - Ensure you have write permissions in the project directory

**Issue: Date parsing errors**
- **Solution**: This is normal if the dataset doesn't have date columns. The program will skip date-dependent visualizations gracefully.

**Issue: Empty visualizations**
- **Solution**: Check that the dataset has the expected columns. The program will attempt to find relevant columns automatically.

### What Happens When You Run

The script executes the following workflow:

1. **Data Loading Phase**:
   - Checks for existing dataset in `data/` directory
   - Downloads from Kaggle if not found
   - Loads CSV file into Pandas DataFrame
   - Displays dataset dimensions and column information

2. **Data Cleaning Phase**:
   - Removes duplicate rows
   - Handles missing values
   - Displays cleaning statistics

3. **Data Structures Demonstration**:
   - Creates and uses lists, tuples, dictionaries, and sets
   - Demonstrates operations on each data structure type

4. **Recursion Demonstration**:
   - Recursively searches directory tree for CSV files
   - Builds and analyzes hierarchical category structures
   - Recursively totals spending across complex categories

5. **Data Analysis Phase**:
   - Computes descriptive statistics (mean, std, min, max, quartiles)
   - Calculates correlation matrices
   - Analyzes transaction amounts using NumPy operations
   - Performs category-based analysis

6. **Visualization Phase**:
   - Creates histogram of transaction distributions
   - Generates bar charts for top categories
   - Produces correlation heatmaps
   - Creates box plots for category comparisons
   - Creates monthly trends visualization (if date columns available)
   - Generates spending breakdown pie charts

7. **Credit Card Optimization Phase**:
   - Simulates different credit card types (Travel cards vs. Standard cards)
   - Calculates rewards for each card based on spending patterns
   - Determines which card provides the most cash back
   - Shows rewards breakdown by category

8. **Output**:
   - Console output with analysis results and optimal card recommendation
   - Visualization files saved to `output/` directory (PNG format)

---

## Code Implementation Details

### Object-Oriented Design

The project implements Object-Oriented Programming through multiple classes demonstrating encapsulation, inheritance, and polymorphism:

#### 1. DataLoader Class (`src/data_loader.py`)
- **Purpose**: Handles dataset downloading, loading, and initial cleaning
- **Encapsulation**: Uses private attributes (`_df`, `_dataset_path`) to protect internal state
- **Key Methods**:
  - `download_dataset()`: Downloads dataset from Kaggle
  - `load_data()`: Loads CSV files into DataFrame
  - `clean_data()`: Performs data cleaning operations
  - `get_summary()`: Returns dataset summary information

#### 2. BaseAnalyzer Abstract Class (`src/analyzer.py`)
- **Purpose**: Abstract base class defining the interface for data analyzers
- **Inheritance**: Demonstrates abstract base class pattern
- **Encapsulation**: Protected attributes and methods
- **Abstract Method**: `analyze()` must be implemented by subclasses
- **Key Methods**:
  - `get_dataframe()`: Returns the DataFrame
  - `save_results()`: Stores analysis results
  - `get_results()`: Retrieves saved results

#### 3. Analyzer Class (`src/analyzer.py`)
- **Purpose**: Performs statistical analysis on transaction data
- **Inheritance**: Inherits from `BaseAnalyzer` and implements abstract `analyze()` method
- **Encapsulation**: Private methods for internal operations (`_identify_numeric_columns()`, `_find_amount_column()`)
- **Key Methods**:
  - `analyze()`: Comprehensive analysis implementing abstract method
  - `get_descriptive_stats()`: Computes descriptive statistics using Pandas
  - `compute_correlations()`: Calculates correlation matrix
  - `analyze_transaction_amounts()`: Uses NumPy for statistical calculations
  - `analyze_by_category()`: Groups and aggregates by categories
  - `get_top_categories()`: Returns top categories as tuple (demonstrating tuple usage)

#### 4. CreditCard Class (`src/card_optimizer.py`)
- **Purpose**: Represents a credit card with specific reward rates
- **Encapsulation**: Private attributes for card properties
- **Key Methods**:
  - `calculate_rewards()`: Calculates rewards for a transaction
  - `get_name()`: Returns card name
  - `get_annual_fee()`: Returns annual fee

#### 5. CardOptimizer Class (`src/card_optimizer.py`)
- **Purpose**: Simulates different credit card types and finds optimal card
- **Key Methods**:
  - `find_optimal_card()`: Determines which card gives most cash back
  - `compare_all_cards()`: Compares all available cards
  - `calculate_total_rewards()`: Calculates total rewards for a card
  - `get_spending_by_category()`: Analyzes spending patterns by category
- **Default Cards**: Includes Travel Rewards Card, Standard Cashback Card, and Category Bonus Card

#### 6. Visualizer Class (`src/visualizer.py`)
- **Purpose**: Creates various types of data visualizations
- **Encapsulation**: Private methods for plot saving/displaying
- **Key Methods**:
  - `plot_transaction_distribution()`: Creates histogram
  - `plot_category_totals()`: Creates bar chart
  - `plot_time_series()`: Creates line plot
  - `plot_monthly_trends()`: Creates monthly trends visualization (as per abstract)
  - `plot_spending_breakdown()`: Creates spending breakdown pie chart (as per abstract)
  - `plot_correlation_heatmap()`: Creates heatmap
  - `plot_box_plot()`: Creates box plot

### Data Analysis Implementation

The project uses NumPy and Pandas extensively for data analysis:

1. **Pandas Operations**:
   - DataFrame creation and manipulation
   - Data cleaning (`drop_duplicates()`, `dropna()`)
   - Grouping and aggregation (`groupby()`, `agg()`)
   - Descriptive statistics (`describe()`)
   - Correlation computation (`corr()`)

2. **NumPy Operations**:
   - Array operations (`values`, `~np.isnan()`)
   - Statistical functions (`np.mean()`, `np.median()`, `np.std()`, `np.percentile()`)
   - Aggregation (`np.sum()`, `np.min()`, `np.max()`)

### Data Visualization Implementation

The project creates multiple types of visualizations using Matplotlib:

1. **Histogram** (`plot_transaction_distribution`):
   - Shows frequency distribution of transaction amounts
   - Includes labeled axes, title, legend, and grid

2. **Bar Chart** (`plot_category_totals`):
   - Displays top categories by total value
   - Rotated labels for readability
   - Includes legend and grid

3. **Monthly Trends** (`plot_monthly_trends`):
   - Shows monthly spending trends over time
   - Date parsing and monthly aggregation
   - Markers and line styling

4. **Spending Breakdown** (`plot_spending_breakdown`):
   - Pie chart showing spending distribution by category
   - Percentage breakdown visualization

5. **Heatmap** (`plot_correlation_heatmap`):
   - Visualizes correlation matrix between numeric variables
   - Color-coded with correlation coefficients displayed
   - Colorbar for reference

6. **Box Plot** (`plot_box_plot`):
   - Compares distributions across categories
   - Shows quartiles and outliers
   - Multiple categories side-by-side

All visualizations include:
- Properly labeled axes
- Descriptive titles
- Legends where appropriate
- Grid for better readability
- Professional styling

### Python Data Structures Usage

The project demonstrates comprehensive use of all core Python data structures:

1. **Lists**:
   - Column names: `list(df.columns)`
   - Numeric values: `df.select_dtypes(...).tolist()`
   - File paths: `recursive_find_files()` returns list
   - Used throughout for iteration and data storage

2. **Tuples**:
   - Column type pairs: `tuple((col, str(dtype)) for col, dtype in df.dtypes.items())`
   - Return values: `get_top_categories()` returns `Tuple[List[str], List[float]]`
   - Used for immutable data pairs and function returns

3. **Dictionaries**:
   - Statistics storage: `Dict[str, float]` for column statistics
   - Configuration: `Dict[str, any]` for analysis results
   - Column information: `dtypes.to_dict()`
   - Used extensively for key-value mappings

4. **Sets**:
   - Unique categorical values: `set()` for unique category collection
   - Column name lookups: `Set[str]` for common column name patterns
   - Used for efficient membership testing and uniqueness

### Recursion Implementation

The project implements three meaningful recursive functions:

1. **`recursive_find_files()`** (`src/utils.py`):
   - **Purpose**: Recursively traverses directory tree to find files with specific extension
   - **Base Case**: Directory doesn't exist or file found
   - **Recursive Case**: Process subdirectories recursively
   - **Use Case**: Finding CSV files in nested directory structures

2. **`build_category_hierarchy()`** and **`count_hierarchy_nodes()`** (`src/utils.py`):
   - **Purpose**: Builds and analyzes hierarchical category structures
   - **Base Case**: Empty category list or leaf node
   - **Recursive Case**: Process category levels, build nested dictionary structure
   - **Use Case**: Organizing categories into tree-like structures and counting nodes recursively

3. **`recursive_total_spending()`** (`src/utils.py`):
   - **Purpose**: Recursively totals up spending across complex category hierarchies
   - **Base Case**: Empty hierarchy
   - **Recursive Case**: Process each category level, accumulate spending amounts
   - **Use Case**: Matches abstract requirement: "recursive function to total up spending across complex categories"

---

## Key Visualizations and Insights

The project generates several visualizations saved in the `output/` directory:

1. **Transaction Distribution Histogram**: 
   - Reveals the frequency distribution of transaction amounts
   - Helps identify spending patterns and outliers

2. **Category Totals Bar Chart**: 
   - Identifies top spending categories
   - Useful for understanding spending priorities

3. **Monthly Trends Visualization** (as per abstract):
   - Shows monthly spending trends over time
   - Helps identify seasonal patterns and spending cycles

4. **Spending Breakdown Pie Chart** (as per abstract):
   - Visual breakdown of spending by category
   - Shows percentage distribution of spending

5. **Correlation Heatmap**: 
   - Reveals relationships between numeric variables
   - Helps identify correlated features

6. **Box Plots**: 
   - Compares distributions across categories
   - Shows quartiles, medians, and outliers

## Credit Card Optimization Features

The project includes a credit card optimization system that:

1. **Simulates Different Card Types**:
   - **Travel Rewards Card**: Higher rewards for travel, airline, hotel, and restaurant categories (5% travel, 4% hotels, 3% restaurants, 1% base)
   - **Standard Cashback Card**: Flat 2% cashback on all purchases
   - **Category Bonus Card**: Bonus rewards on groceries (6%), gas (3%), and dining (3%)

2. **Calculates Optimal Card**:
   - Analyzes spending patterns by category
   - Calculates total rewards for each card type
   - Accounts for annual fees
   - Recommends the card that provides maximum cash back

3. **Shows Rewards Breakdown**:
   - Displays rewards by category for the optimal card
   - Compares all available cards side-by-side
   - Shows gross and net rewards (after annual fees)

---

## Design Decisions and Architecture

### Class Design Rationale

1. **Separation of Concerns**: Each class has a single, well-defined responsibility
   - `DataLoader`: Data acquisition and cleaning
   - `Analyzer`: Statistical analysis
   - `Visualizer`: Plot generation

2. **Inheritance Strategy**: Used abstract base class (`BaseAnalyzer`) to:
   - Define common interface for analyzers
   - Allow future extensibility
   - Demonstrate inheritance concept

3. **Encapsulation**: Private attributes (leading underscore) protect internal state:
   - Prevents accidental modification
   - Maintains data integrity
   - Follows Python conventions

### Error Handling

- Try-except blocks around critical operations
- Graceful degradation when optional features fail
- Informative error messages for debugging

### Code Organization

- Modular design with separate files for each major component
- Clear function and method names
- Comprehensive docstrings
- Type hints for better code clarity

---

## Challenges Faced

1. **Dataset Structure Uncertainty**: 
   - Solution: Implemented flexible column detection using common naming patterns

2. **Date Format Variations**: 
   - Solution: Robust date parsing with error handling and fallback options

3. **Memory Management**: 
   - Solution: Efficient data processing using Pandas operations and data copying only when necessary

4. **Visualization Flexibility**: 
   - Solution: Created methods that handle missing columns gracefully

---

## Lessons Learned

1. **OOP Design**: Proper encapsulation and inheritance significantly improve code maintainability and extensibility

2. **Recursion**: Powerful tool for tree traversal and hierarchical data processing, but requires careful base case definition

3. **Data Structures**: Choosing the right data structure (list vs tuple vs dict vs set) improves both efficiency and code readability

4. **Visualization**: Clear labels, titles, and legends are essential for effective data communication

5. **Collaboration**: Git workflow and clear documentation facilitate effective team collaboration

---

## Future Enhancements

Potential improvements for future iterations:

- Credit card reward calculation logic
- Optimization algorithms for reward maximization
- Interactive visualizations using Plotly
- Machine learning models for spending prediction
- Web dashboard for real-time analysis
- User authentication and personalized dashboards

---

## Quick Start Guide

For a quick test run:

```bash
# 1. Navigate to project directory
cd Dangal_Lamichhane_Python_FinalProject

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the project
python src/main.py
```

The program will:
- Automatically download the dataset (first time only)
- Clean and process the data
- Perform comprehensive analysis
- Generate visualizations
- Recommend the optimal credit card

## File Structure Explained

- **`src/main.py`**: Main entry point - run this file to execute the entire project
- **`src/data_loader.py`**: Handles data downloading and cleaning
- **`src/analyzer.py`**: Performs statistical analysis (includes BaseAnalyzer abstract class)
- **`src/card_optimizer.py`**: Simulates credit cards and finds optimal card
- **`src/visualizer.py`**: Creates all data visualizations
- **`src/utils.py`**: Utility functions including recursive operations
- **`requirements.txt`**: Lists all Python package dependencies
- **`README.md`**: This documentation file
- **`data/`**: Directory where dataset is stored (created automatically)
- **`output/`**: Directory where visualizations are saved (created automatically)

## Notes

- The dataset is downloaded automatically on first run and cached for subsequent runs
- All visualizations are saved as PNG files in the `output/` directory
- The program handles missing columns gracefully and will skip features that require unavailable data
- Console output provides detailed information about each processing step

## License

This project is created for educational purposes as part of CISC 288-01 course requirements.

---

## Contact Information

For questions or issues, please contact:
- Suman Dangal
- Diwash Lamichhane

## GitHub Repository

**Repository Link:** `https://github.com/L00SIRE/Dangal_Lamichhane_Python_FinalProject`
