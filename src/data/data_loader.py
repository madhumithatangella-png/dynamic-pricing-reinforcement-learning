"""Data Loader component for Dynamic Pricing RL.

Provides secure and logged data loading operations.
"""

from pathlib import Path
import pandas as pd
from src.logger import get_logger

logger = get_logger("data_loader")


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Loads raw dataset from a CSV file.

    Args:
        file_path: Path to the target CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If target file does not exist.
        ValueError: If file is not a valid CSV format or is empty.
    """
    logger.info(f"Attempting to load dataset from: {file_path}")

    # Check existence
    if not file_path.exists():
        err_msg = f"Dataset file not found at {file_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            err_msg = f"The dataset at {file_path} is empty."
            logger.error(err_msg)
            raise ValueError(err_msg)

        logger.info(
            f"Successfully loaded raw dataset. Shape: {df.shape[0]} rows, "
            f"{df.shape[1]} columns."
        )
        return df

    except pd.errors.EmptyDataError as ede:
        err_msg = f"EmptyDataError encountered while reading {file_path}: {ede}"
        logger.error(err_msg)
        raise ValueError(err_msg) from ede

    except pd.errors.ParserError as pe:
        err_msg = f"ParserError encountered while parsing CSV {file_path}: {pe}"
        logger.error(err_msg)
        raise ValueError(err_msg) from pe

    except Exception as e:
        err_msg = f"Unexpected error occurred while loading dataset {file_path}: {e}"
        logger.error(err_msg)
        raise RuntimeError(err_msg) from e
