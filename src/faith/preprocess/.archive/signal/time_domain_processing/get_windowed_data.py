import pandas as pd


def get_window(
    df: pd.DataFrame, center_index: int, window_size: int = 5
) -> pd.DataFrame:
    """
    Get windowed data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    center_index : int
        The center index of the window.
    window_size : int
        The size of the window.

    Returns
    -------
    pd.DataFrame
        The windowed data.
    """
    start = max(center_index - window_size, 0)
    end = min(center_index + window_size + 1, len(df))
    windowed = df.iloc[start:end].drop(columns=["xdata"])

    return windowed


# Function to get windowed data
# @staticmethod
# def get_windowed_data(df, center_index, window_size=5):
#     start = max(center_index - window_size, 0)
#     end = min(center_index + window_size + 1, len(df))
#     return df.iloc[start:end].drop(columns=['xdata'])
