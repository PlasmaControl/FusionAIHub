import pandas as pd

def merge_times_1d(t)
# reference
@staticmethod
def time_matching_merge_asof_1d(time1, data1, time_std, left_window=0, right_window=0):
    # Convert input arrays to DataFrames
    df1 = pd.DataFrame({'time1': time1, 'data1': data1})
    df2 = pd.DataFrame({'time_std': time_std})
    
    # Sort the DataFrames by the time columns
    df1.sort_values('time1', inplace=True)
    df2.sort_values('time_std', inplace=True)
    
    # Perform the asof merge
    merged_df = pd.merge_asof(df2, df1, left_on='time_std',
                                right_on='time1', direction='nearest')

    # Drop unnecessary columns and handle NaN values
    merged_df.drop(columns='time_std', inplace=True)
    merged_df.dropna(inplace=True)

    # Extract the matched time and data
    matched_time = merged_df['time1'].values
    matched_data = merged_df['data1'].values

    return matched_time, matched_data