# Function to get windowed data
@staticmethod
def get_windowed_data(df, center_index, window_size=5):
    start = max(center_index - window_size, 0)
    end = min(center_index + window_size + 1, len(df))
    return df.iloc[start:end].drop(columns=['xdata'])