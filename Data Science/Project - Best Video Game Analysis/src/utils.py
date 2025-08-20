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
    
def read_games_csv_with_broken_header(path):
    """Read the specific Games CSV file with a broken header.

    Args:
        path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame with fixed Headers.
    """
    with open(path, "r", encoding="utf-8") as f:
        broken_header = f.readline().rstrip("\n")

    names = broken_header.split(",") # 39 names right now
    names.insert(7, "test")  # missing header at index 7 (after index 6)

    assert len(names) == 40, f"Expected 40 names, got {len(names)}"

    # Read the CSV using new names, skip the broken header row
    games = pd.read_csv(path, header=None, names=names, skiprows=1)
    
    return games
