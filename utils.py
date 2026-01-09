from langchain_core.tools import tool
import os, glob
import pandas as pd
from typing import List, Dict, Any

__all__ = [
    "list_csv_files", "preload_datasets", "get_dataset_summaries", "call_dataframe_method",
    "drop_column", "rename_column", "drop_rows_with_missing", "fill_missing_values",
    "filter_dataset", "save_dataset", "DATAFRAME_CACHE"
]

@tool
def list_csv_files() -> List[str]:
    """List all CSV file names in the local directory.

    Returns:
        A list containing CSV file names.
        If no CSV files are found, returns None.
    """
    csv_files = os.path.join(os.getcwd(), "*.csv") 
    csv_files = glob.glob(csv_files)
    if not csv_files:
        return None
    return [os.path.basename(file) for file in csv_files]


DATAFRAME_CACHE = {}

@tool
def preload_datasets(paths: List[str]) -> str:
    """
    Loads CSV files into a global cache if not already loaded.
    
    This function helps to efficiently manage datasets by loading them once
    and storing them in memory for future use. Without caching, you would
    waste tokens describing dataset contents repeatedly in agent responses.
    
    Args:
        paths: A list of file paths to CSV files.

    Returns:
        A message summarizing which datasets were loaded or already cached.
    """
    loaded = []
    cached = []
    for path in paths:
        if path not in DATAFRAME_CACHE:
            DATAFRAME_CACHE[path] = pd.read_csv(path)
            loaded.append(path)
        else:
            cached.append(path)
    
    return (
        f"Loaded datasets: {loaded}\n"
        f"Already cached: {cached}"
    )

@tool
def get_dataset_summaries(dataset_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze multiple CSV files and return metadata summaries for each.

    Args:
        dataset_paths (List[str]): 
            A list of file paths to CSV datasets.

    Returns:
        List[Dict[str, Any]]: 
            A list of summaries, one per dataset, each containing:
            - "file_name": The path of the dataset file.
            - "column_names": A list of column names in the dataset.
            - "data_types": A dictionary mapping column names to their data types (as strings).
    """
    summaries = []

    for path in dataset_paths:
        # Load and cache the dataset if not already cached
        if path not in DATAFRAME_CACHE:
            DATAFRAME_CACHE[path] = pd.read_csv(path)
        
        df = DATAFRAME_CACHE[path]

        # Build summary
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
   This tool lets you run simple DataFrame methods like 'head', 'tail', or 'describe' 
   on a dataset that has already been loaded and cached using 'preload_datasets'.
   Args:
       file_name (str): The path or name of the dataset in the global cache.
       method (str): The name of the method to call on the DataFrame. Only no-argument 
                     methods are supported (e.g., 'head', 'describe', 'info').
   Returns:
       str: The output of the method as a formatted string, or an error message if 
            the dataset is not found or the method is invalid.
   Example:
       call_dataframe_method(file_name="data.csv", method="head")
   """
   # Try to get the DataFrame from cache, or load it if not already cached
   if file_name not in DATAFRAME_CACHE:
       try:
           DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
       except FileNotFoundError:
           return f"DataFrame '{file_name}' not found in cache or on disk."
       except Exception as e:
           return f"Error loading '{file_name}': {str(e)}"
   
   df = DATAFRAME_CACHE[file_name]
   func = getattr(df, method, None)
   if not callable(func):
       return f"'{method}' is not a valid method of DataFrame."
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
        file_name: The name of the dataset in the cache.
        column_name: The name of the column to remove.
    
    Returns:
        A success message or error description.
    """
    if file_name not in DATAFRAME_CACHE:
        return f"Dataset '{file_name}' not found in cache. Load it first with preload_datasets."
    
    df = DATAFRAME_CACHE[file_name]
    if column_name not in df.columns:
        return f"Column '{column_name}' not found in '{file_name}'. Available columns: {df.columns.tolist()}"
    
    DATAFRAME_CACHE[file_name] = df.drop(columns=[column_name])
    return f"Successfully dropped column '{column_name}' from '{file_name}'."


@tool
def rename_column(file_name: str, old_name: str, new_name: str) -> str:
    """
    Rename a column in a dataset.
    
    Args:
        file_name: The name of the dataset in the cache.
        old_name: The current name of the column.
        new_name: The new name for the column.
    
    Returns:
        A success message or error description.
    """
    if file_name not in DATAFRAME_CACHE:
        return f"Dataset '{file_name}' not found in cache. Load it first with preload_datasets."
    
    df = DATAFRAME_CACHE[file_name]
    if old_name not in df.columns:
        return f"Column '{old_name}' not found in '{file_name}'. Available columns: {df.columns.tolist()}"
    
    DATAFRAME_CACHE[file_name] = df.rename(columns={old_name: new_name})
    return f"Successfully renamed column '{old_name}' to '{new_name}' in '{file_name}'."


@tool
def drop_rows_with_missing(file_name: str, column_name: str = None) -> str:
    """
    Drop rows with missing values from a dataset.
    
    Args:
        file_name: The name of the dataset in the cache.
        column_name: Optional. If provided, only drop rows where this column has missing values.
                    If not provided, drops rows with any missing values.
    
    Returns:
        A message indicating how many rows were dropped.
    """
    if file_name not in DATAFRAME_CACHE:
        return f"Dataset '{file_name}' not found in cache. Load it first with preload_datasets."
    
    df = DATAFRAME_CACHE[file_name]
    original_count = len(df)
    
    if column_name:
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in '{file_name}'."
        df = df.dropna(subset=[column_name])
    else:
        df = df.dropna()
    
    DATAFRAME_CACHE[file_name] = df
    dropped_count = original_count - len(df)
    return f"Dropped {dropped_count} rows with missing values. Dataset now has {len(df)} rows."


@tool
def fill_missing_values(file_name: str, column_name: str, fill_value: str) -> str:
    """
    Fill missing values in a specific column.
    
    Args:
        file_name: The name of the dataset in the cache.
        column_name: The column to fill missing values in.
        fill_value: The value to fill. Use 'mean', 'median', or 'mode' for numeric columns,
                   or provide a specific value.
    
    Returns:
        A success message or error description.
    """
    if file_name not in DATAFRAME_CACHE:
        return f"Dataset '{file_name}' not found in cache. Load it first with preload_datasets."
    
    df = DATAFRAME_CACHE[file_name]
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
    
    DATAFRAME_CACHE[file_name] = df
    return f"Filled {missing_count} missing values in column '{column_name}' with '{fill_value}'."


@tool
def filter_dataset(file_name: str, column_name: str, operator: str, value: str) -> str:
    """
    Filter dataset rows based on a condition.
    
    Args:
        file_name: The name of the dataset in the cache.
        column_name: The column to filter on.
        operator: The comparison operator ('==', '!=', '>', '<', '>=', '<=', 'contains').
        value: The value to compare against.
    
    Returns:
        A message indicating how many rows remain after filtering.
    """
    if file_name not in DATAFRAME_CACHE:
        return f"Dataset '{file_name}' not found in cache. Load it first with preload_datasets."
    
    df = DATAFRAME_CACHE[file_name]
    if column_name not in df.columns:
        return f"Column '{column_name}' not found in '{file_name}'."
    
    original_count = len(df)
    
    # Try to convert value to appropriate type
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
    
    DATAFRAME_CACHE[file_name] = df
    removed_count = original_count - len(df)
    return f"Filtered dataset. Removed {removed_count} rows. Dataset now has {len(df)} rows."


@tool
def save_dataset(file_name: str, output_path: str = None) -> str:
    """
    Save a dataset from cache to a CSV file.
    
    Args:
        file_name: The name of the dataset in the cache.
        output_path: Optional. The path to save the file. If not provided, overwrites the original.
    
    Returns:
        A success message or error description.
    """
    if file_name not in DATAFRAME_CACHE:
        return f"Dataset '{file_name}' not found in cache. Load it first with preload_datasets."
    
    df = DATAFRAME_CACHE[file_name]
    save_path = output_path if output_path else file_name
    
    try:
        df.to_csv(save_path, index=False)
        return f"Successfully saved dataset to '{save_path}' ({len(df)} rows, {len(df.columns)} columns)."
    except Exception as e:
        return f"Error saving dataset: {str(e)}"