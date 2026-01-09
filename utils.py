from langchain_core.tools import tool
import os
import glob
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DatasetManager:
    """
    Manages dataset loading, caching, and operations.
    
    Encapsulates dataset state instead of using global variables.
    Provides memory management with configurable cache limits.
    """
    _cache: Dict[str, pd.DataFrame] = field(default_factory=dict)
    max_datasets: int = 10
    
    def load(self, path: str) -> pd.DataFrame:
        """Load a dataset, using cache if available."""
        if path not in self._cache:
            self._evict_if_needed()
            self._cache[path] = pd.read_csv(path)
        return self._cache[path]
    
    def get(self, path: str) -> Optional[pd.DataFrame]:
        """Get a dataset from cache, or None if not loaded."""
        return self._cache.get(path)
    
    def save(self, path: str, df: pd.DataFrame) -> None:
        """Save a DataFrame to the cache."""
        self._evict_if_needed()
        self._cache[path] = df
    
    def contains(self, path: str) -> bool:
        """Check if a dataset is in the cache."""
        return path in self._cache
    
    def list_cached(self) -> List[str]:
        """List all cached dataset names."""
        return list(self._cache.keys())
    
    def clear(self) -> None:
        """Clear all cached datasets."""
        self._cache.clear()
    
    def _evict_if_needed(self) -> None:
        """Remove oldest entry if cache is at capacity."""
        if len(self._cache) >= self.max_datasets:
            oldest = next(iter(self._cache))
            del self._cache[oldest]


def create_dataset_tools(manager: DatasetManager):
    """
    Factory function that creates tools bound to a specific DatasetManager instance.
    
    Args:
        manager: The DatasetManager instance to use for all operations.
    
    Returns:
        A dictionary of tool functions that can be passed to the agent.
    """
    
    @tool
    def list_csv_files() -> List[str]:
        """List all CSV file names in the local directory.

        Returns:
            A list containing CSV file names.
            If no CSV files are found, returns None.
        """
        csv_pattern = os.path.join(os.getcwd(), "*.csv")
        csv_files = glob.glob(csv_pattern)
        if not csv_files:
            return None
        return [os.path.basename(f) for f in csv_files]

    @tool
    def preload_datasets(paths: List[str]) -> str:
        """
        Loads CSV files into the cache if not already loaded.
        
        This function helps to efficiently manage datasets by loading them once
        and storing them in memory for future use.
        
        Args:
            paths: A list of file paths to CSV files.

        Returns:
            A message summarizing which datasets were loaded or already cached.
        """
        loaded = []
        cached = []
        for path in paths:
            if not manager.contains(path):
                manager.load(path)
                loaded.append(path)
            else:
                cached.append(path)
        
        return f"Loaded datasets: {loaded}\nAlready cached: {cached}"

    @tool
    def get_dataset_summaries(dataset_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple CSV files and return metadata summaries for each.

        Args:
            dataset_paths: A list of file paths to CSV datasets.

        Returns:
            A list of summaries, one per dataset, each containing:
            - "file_name": The path of the dataset file.
            - "column_names": A list of column names in the dataset.
            - "data_types": A dictionary mapping column names to their data types.
        """
        summaries = []
        for path in dataset_paths:
            df = manager.load(path)
            summary = {
                "file_name": path,
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict()
            }
            summaries.append(summary)
        return summaries

    @tool
    def call_dataframe_method(file_name: str, method: str) -> str:
        """
        Execute a method on a DataFrame and return the result.
        
        This tool lets you run simple DataFrame methods like 'head', 'tail', or 'describe'.
        
        Args:
            file_name: The path or name of the dataset.
            method: The name of the method to call (e.g., 'head', 'describe', 'info').
        
        Returns:
            The output of the method as a formatted string, or an error message.
        
        Example:
            call_dataframe_method(file_name="data.csv", method="head")
        """
        try:
            df = manager.load(file_name)
        except FileNotFoundError:
            return f"DataFrame '{file_name}' not found."
        except Exception as e:
            return f"Error loading '{file_name}': {str(e)}"
        
        func = getattr(df, method, None)
        if not callable(func):
            return f"'{method}' is not a valid DataFrame method."
        
        try:
            result = func()
            return str(result)
        except Exception as e:
            return f"Error calling '{method}' on '{file_name}': {str(e)}"

    @tool
    def drop_column(file_name: str, column_name: str) -> str:
        """
        Remove a column from a dataset.
        
        Args:
            file_name: The name of the dataset.
            column_name: The name of the column to remove.
        
        Returns:
            A success message or error description.
        """
        if not manager.contains(file_name):
            return f"Dataset '{file_name}' not loaded. Load it first with preload_datasets."
        
        df = manager.get(file_name)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found. Available: {df.columns.tolist()}"
        
        manager.save(file_name, df.drop(columns=[column_name]))
        return f"Successfully dropped column '{column_name}' from '{file_name}'."

    @tool
    def rename_column(file_name: str, old_name: str, new_name: str) -> str:
        """
        Rename a column in a dataset.
        
        Args:
            file_name: The name of the dataset.
            old_name: The current name of the column.
            new_name: The new name for the column.
        
        Returns:
            A success message or error description.
        """
        if not manager.contains(file_name):
            return f"Dataset '{file_name}' not loaded. Load it first with preload_datasets."
        
        df = manager.get(file_name)
        if old_name not in df.columns:
            return f"Column '{old_name}' not found. Available: {df.columns.tolist()}"
        
        manager.save(file_name, df.rename(columns={old_name: new_name}))
        return f"Successfully renamed '{old_name}' to '{new_name}' in '{file_name}'."

    @tool
    def drop_rows_with_missing(file_name: str, column_name: str = None) -> str:
        """
        Drop rows with missing values from a dataset.
        
        Args:
            file_name: The name of the dataset.
            column_name: Optional. If provided, only drop rows where this column has missing values.
        
        Returns:
            A message indicating how many rows were dropped.
        """
        if not manager.contains(file_name):
            return f"Dataset '{file_name}' not loaded. Load it first with preload_datasets."
        
        df = manager.get(file_name)
        original_count = len(df)
        
        if column_name:
            if column_name not in df.columns:
                return f"Column '{column_name}' not found in '{file_name}'."
            df = df.dropna(subset=[column_name])
        else:
            df = df.dropna()
        
        manager.save(file_name, df)
        dropped_count = original_count - len(df)
        return f"Dropped {dropped_count} rows. Dataset now has {len(df)} rows."

    @tool
    def fill_missing_values(file_name: str, column_name: str, fill_value: str) -> str:
        """
        Fill missing values in a specific column.
        
        Args:
            file_name: The name of the dataset.
            column_name: The column to fill missing values in.
            fill_value: Use 'mean', 'median', 'mode', or provide a specific value.
        
        Returns:
            A success message or error description.
        """
        if not manager.contains(file_name):
            return f"Dataset '{file_name}' not loaded. Load it first with preload_datasets."
        
        df = manager.get(file_name)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in '{file_name}'."
        
        missing_count = df[column_name].isna().sum()
        
        if fill_value.lower() == 'mean':
            df[column_name] = df[column_name].fillna(df[column_name].mean())
        elif fill_value.lower() == 'median':
            df[column_name] = df[column_name].fillna(df[column_name].median())
        elif fill_value.lower() == 'mode':
            df[column_name] = df[column_name].fillna(df[column_name].mode().iloc[0])
        else:
            df[column_name] = df[column_name].fillna(fill_value)
        
        manager.save(file_name, df)
        return f"Filled {missing_count} missing values in '{column_name}' with '{fill_value}'."

    @tool
    def filter_dataset(file_name: str, column_name: str, operator: str, value: str) -> str:
        """
        Filter dataset rows based on a condition.
        
        Args:
            file_name: The name of the dataset.
            column_name: The column to filter on.
            operator: Comparison operator ('==', '!=', '>', '<', '>=', '<=', 'contains').
            value: The value to compare against.
        
        Returns:
            A message indicating how many rows remain after filtering.
        """
        if not manager.contains(file_name):
            return f"Dataset '{file_name}' not loaded. Load it first with preload_datasets."
        
        df = manager.get(file_name)
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in '{file_name}'."
        
        original_count = len(df)
        
        # Try to convert value to numeric
        try:
            numeric_value = float(value)
        except ValueError:
            numeric_value = None
        
        if operator == '==':
            df = df[df[column_name] == (numeric_value if numeric_value is not None else value)]
        elif operator == '!=':
            df = df[df[column_name] != (numeric_value if numeric_value is not None else value)]
        elif operator == '>':
            df = df[df[column_name] > numeric_value]
        elif operator == '<':
            df = df[df[column_name] < numeric_value]
        elif operator == '>=':
            df = df[df[column_name] >= numeric_value]
        elif operator == '<=':
            df = df[df[column_name] <= numeric_value]
        elif operator == 'contains':
            df = df[df[column_name].astype(str).str.contains(value, case=False, na=False)]
        else:
            return f"Invalid operator '{operator}'. Use: ==, !=, >, <, >=, <=, or contains."
        
        manager.save(file_name, df)
        removed_count = original_count - len(df)
        return f"Filtered dataset. Removed {removed_count} rows. Now has {len(df)} rows."

    @tool
    def save_dataset(file_name: str, output_path: str = None) -> str:
        """
        Save a dataset to a CSV file.
        
        Args:
            file_name: The name of the dataset in cache.
            output_path: Optional. Path to save. If not provided, overwrites original.
        
        Returns:
            A success message or error description.
        """
        if not manager.contains(file_name):
            return f"Dataset '{file_name}' not loaded. Load it first with preload_datasets."
        
        df = manager.get(file_name)
        save_path = output_path if output_path else file_name
        
        try:
            df.to_csv(save_path, index=False)
            return f"Saved to '{save_path}' ({len(df)} rows, {len(df.columns)} columns)."
        except Exception as e:
            return f"Error saving dataset: {str(e)}"

    # Return all tools as a dictionary
    return {
        "list_csv_files": list_csv_files,
        "preload_datasets": preload_datasets,
        "get_dataset_summaries": get_dataset_summaries,
        "call_dataframe_method": call_dataframe_method,
        "drop_column": drop_column,
        "rename_column": rename_column,
        "drop_rows_with_missing": drop_rows_with_missing,
        "fill_missing_values": fill_missing_values,
        "filter_dataset": filter_dataset,
        "save_dataset": save_dataset,
    }
