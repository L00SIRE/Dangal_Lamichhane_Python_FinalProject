## Credit Card Spending and Rewards Analyzer

This project analyzes credit card transaction data to understand spending patterns and estimate potential cashback / rewards under different credit card strategies.

### Dataset

**Dataset:** Comprehensive Credit Card Transactions Dataset (Kaggle)  
**Link:** `https://www.kaggle.com/datasets/rajatsurana979/comprehensive-credit-card-transactions-dataset`

The dataset is automatically downloaded using `kagglehub` when you run the project for the first time.

### Project Structure

- `src/main.py` - Main entry point that orchestrates data loading and cleaning
- `src/data_loader.py` - `DataLoader` class for downloading, loading, and cleaning the dataset

### Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the project:**
   ```bash
   python src/main.py
   ```

   The script will:
   - Download the dataset from Kaggle (if not already present)
   - Load the CSV file
   - Clean the data (remove duplicates, handle missing values)
   - Display a summary of the processed dataset

### Data Processing

The `DataLoader` class handles:
- **Downloading:** Automatically downloads the dataset from Kaggle using `kagglehub`
- **Loading:** Reads CSV files from the dataset directory
- **Cleaning:** 
  - Removes duplicate rows
  - Handles missing values in critical columns
  - Resets index after cleaning operations

The cleaned dataset is ready for analysis and visualization in subsequent phases of the project.

### Team Members

- Suman Dangal
- Diwash Lamichhane

### Future Enhancements

As the project grows, additional modules will be added:
- `analyzer.py` - Data analysis and statistics
- `visualizer.py` - Data visualization using Matplotlib
- `cards.py` - Credit card reward calculation logic
- `optimizer.py` - Optimization algorithms for reward maximization


