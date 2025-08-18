import pandas as pd


def get_string_array_length(x):
    """Get the number of non-empty elements in a comma-separated string.

    Args:
        x (str): The input string.

    Returns:
        int: The number of non-empty elements in the string.
    """
    return len([c for c in str(x).split(',') if c.strip() != ''])


def range_to_average(x):
    """Convert a range string to its average value.

    Args:
        x (str): The input range string (e.g., "1-5").

    Returns:
        float: The average value of the range.
    """
    if pd.isna(x):
        return None
    try:
        start, end = map(int, x.split('-'))
        return int((start + end) / 2)
    except ValueError:
        return None
