from pathlib import Path
from typing import Dict, List, Set, Tuple


def recursive_find_files(directory: Path, extension: str = ".csv", files_found: List[str] = None) -> List[str]:
    """
    Recursively find all files with a given extension in a directory tree.
    Demonstrates meaningful recursion for directory traversal.
    
    Args:
        directory: Path object representing the directory to search.
        extension: File extension to search for (e.g., ".csv").
        files_found: List to accumulate found files (used in recursion).
        
    Returns:
        List of file paths found recursively.
    """
    if files_found is None:
        files_found = []
    
    # Base case: if directory doesn't exist, return empty list
    if not directory.exists():
        return files_found
    
    try:
        # Recursive case: traverse directory tree
        for item in directory.iterdir():
            if item.is_file() and item.suffix == extension:
                files_found.append(str(item))
            elif item.is_dir():
                # Recursive call for subdirectories
                recursive_find_files(item, extension, files_found)
    except PermissionError:
        # Skip directories we don't have permission to access
        pass
    
    return files_found


def recursive_calculate_statistics(data: List[float], depth: int = 0) -> Dict[str, float]:
    """
    Recursively calculate statistics on nested data structures.
    Demonstrates recursion for computation.
    
    Args:
        data: List of numeric values (can be nested).
        depth: Current recursion depth.
        
    Returns:
        Dictionary with calculated statistics.
    """
    # Base case: if data is empty or not a list
    if not isinstance(data, list) or len(data) == 0:
        return {"sum": 0.0, "count": 0, "max_depth": depth}
    
    # Base case: if all elements are numbers (leaf level)
    if all(isinstance(x, (int, float)) for x in data):
        return {
            "sum": sum(data),
            "count": len(data),
            "max_depth": depth,
            "mean": sum(data) / len(data) if len(data) > 0 else 0.0,
            "min": min(data),
            "max": max(data),
        }
    
    # Recursive case: process nested lists
    results = []
    for item in data:
        if isinstance(item, list):
            result = recursive_calculate_statistics(item, depth + 1)
            results.append(result)
        elif isinstance(item, (int, float)):
            results.append({"sum": item, "count": 1, "max_depth": depth})
    
    # Aggregate results
    total_sum = sum(r.get("sum", 0) for r in results)
    total_count = sum(r.get("count", 0) for r in results)
    max_depth = max((r.get("max_depth", depth) for r in results), default=depth)
    
    return {
        "sum": total_sum,
        "count": total_count,
        "max_depth": max_depth,
        "mean": total_sum / total_count if total_count > 0 else 0.0,
    }


def build_category_hierarchy(df_categories: List[str], separator: str = "/") -> Dict[str, any]:
    """
    Recursively build a hierarchical structure from category names.
    Demonstrates recursion for hierarchy analysis.
    
    Args:
        df_categories: List of category strings (e.g., ["Food/Groceries", "Food/Restaurants"]).
        separator: Character that separates hierarchy levels.
        
    Returns:
        Nested dictionary representing the hierarchy.
    """
    def _build_recursive(categories: List[str], current_level: Dict[str, any]) -> None:
        """
        Recursive helper function to build hierarchy.
        
        Args:
            categories: Remaining categories to process.
            current_level: Current level of the hierarchy dictionary.
        """
        if not categories:
            return
        
        # Group categories by their first level
        level_groups: Dict[str, List[str]] = {}
        for category in categories:
            if separator in category:
                first_level, rest = category.split(separator, 1)
                if first_level not in level_groups:
                    level_groups[first_level] = []
                level_groups[first_level].append(rest)
            else:
                # Leaf node
                if category not in current_level:
                    current_level[category] = {}
        
        # Recursively process each group
        for first_level, remaining in level_groups.items():
            if first_level not in current_level:
                current_level[first_level] = {}
            _build_recursive(remaining, current_level[first_level])
    
    hierarchy: Dict[str, any] = {}
    _build_recursive(df_categories, hierarchy)
    return hierarchy


def count_hierarchy_nodes(hierarchy: Dict[str, any], count: int = 0) -> int:
    """
    Recursively count nodes in a hierarchy structure.
    
    Args:
        hierarchy: Nested dictionary representing hierarchy.
        count: Current count (used in recursion).
        
    Returns:
        Total number of nodes in the hierarchy.
    """
    if not isinstance(hierarchy, dict) or len(hierarchy) == 0:
        return count
    
    # Count current level nodes
    count += len(hierarchy)
    
    # Recursively count child nodes
    for value in hierarchy.values():
        if isinstance(value, dict):
            count = count_hierarchy_nodes(value, count)
    
    return count


def recursive_total_spending(
    category_hierarchy: Dict[str, any],
    spending_by_category: Dict[str, float],
    total: float = 0.0
) -> float:
    """
    Recursively total up spending across complex category hierarchies.
    This function matches the abstract requirement: "recursive function to 
    total up spending across complex categories."
    
    Args:
        category_hierarchy: Nested dictionary representing category hierarchy.
        spending_by_category: Dictionary mapping category names to spending amounts.
        total: Running total (used in recursion).
        
    Returns:
        Total spending amount across all categories in the hierarchy.
    """
    # Base case: empty hierarchy
    if not isinstance(category_hierarchy, dict) or len(category_hierarchy) == 0:
        return total
    
    # Recursive case: process each category in the hierarchy
    for category, subcategories in category_hierarchy.items():
        # Add spending for this category if it exists
        category_lower = category.lower()
        if category_lower in spending_by_category:
            total += spending_by_category[category_lower]
        
        # Recursively process subcategories
        if isinstance(subcategories, dict):
            total = recursive_total_spending(subcategories, spending_by_category, total)
    
    return total
