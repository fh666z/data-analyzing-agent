from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from langchain_core.tools import tool
from typing import Dict
import pandas as pd

from utils import DatasetManager


def create_evaluation_tools(manager: DatasetManager):
    """
    Factory function that creates evaluation tools bound to a DatasetManager instance.
    
    Args:
        manager: The DatasetManager instance to use for dataset access.
    
    Returns:
        A dictionary of evaluation tool functions.
    """
    
    @tool
    def evaluate_classification_dataset(file_name: str, target_column: str) -> Dict[str, float]:
        """
        Train and evaluate a classifier on a dataset using the specified target column.
        
        Args:
            file_name: The name or path of the dataset.
            target_column: The name of the column to use as the classification target.
        
        Returns:
            A dictionary with the model's accuracy score.
        """
        try:
            df = manager.load(file_name)
        except FileNotFoundError:
            return {"error": f"Dataset '{file_name}' not found."}
        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}
        
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found in '{file_name}'."}
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        return {"accuracy": acc}

    @tool
    def evaluate_regression_dataset(file_name: str, target_column: str) -> Dict[str, float]:
        """
        Train and evaluate a regression model on a dataset using the specified target column.
        
        Args:
            file_name: The name or path of the dataset.
            target_column: The name of the column to use as the regression target.
        
        Returns:
            A dictionary with R² score and Mean Squared Error.
        """
        try:
            df = manager.load(file_name)
        except FileNotFoundError:
            return {"error": f"Dataset '{file_name}' not found."}
        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}
        
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found in '{file_name}'."}
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            "r2_score": r2,
            "mean_squared_error": mse
        }

    return {
        "evaluate_classification_dataset": evaluate_classification_dataset,
        "evaluate_regression_dataset": evaluate_regression_dataset,
    }
