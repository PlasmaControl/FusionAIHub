import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import butter, lfilter
from scipy.sparse.linalg import eigs as Eigens
from sklearn.metrics import r2_score
import scipy.signal
import pandas as pd
import time
import pickle


def get_weights(input_dim, res_size, K_in, K_rec, insca, spra, bisca):
    # ---------- Initializing W_in ---------

    if K_in == -1 or input_dim < K_in:
        W_in = insca * (np.random.rand(res_size, input_dim) * 2 - 1)
    else:
        Ico = 0
        nrentries = np.int32(res_size * K_in)
        ij = np.zeros((2, nrentries))
        datavec = insca * (np.random.rand(nrentries) * 2 - 1)
        for en in range(res_size):
            Per = np.random.permutation(input_dim)[:K_in]
            ij[0][Ico:Ico + K_in] = en
            ij[1][Ico:Ico + K_in] = Per
            Ico += K_in
        W_in = scipy.sparse.csc_matrix((datavec, np.int32(ij)), shape=(res_size, input_dim), dtype='float32')
        if K_in > input_dim / 2:
            W_in = W_in.todense()

    # ---------- Initializing W_res ---------

    converged = False
    attempts = 50
    while not converged and attempts > 0:
        if K_rec == -1:
            W_res = np.random.randn(res_size, res_size)
        else:
            Ico = 0
            nrentries = np.int32(res_size * K_rec)
            ij = np.zeros((2, nrentries))
            datavec = np.random.randn(nrentries)
            for en in range(res_size):
                Per = np.random.permutation(res_size)[:K_rec]
                ij[0][Ico:Ico + K_rec] = en
                ij[1][Ico:Ico + K_rec] = Per
                Ico += K_rec
            W_res = scipy.sparse.csc_matrix((datavec, np.int32(ij)), shape=(res_size, res_size), dtype='float32')
            if K_rec > res_size / 2:
                W_res = W_res.todense()
        try:
            we = Eigens(W_res, return_eigenvectors=False, k=6)
            converged = True
        except:
            print("WARNING: No convergence! Redo %i times ... " % (attempts - 1))
            attempts -= 1
            pass

    W_res *= (spra / np.amax(np.absolute(we)))
    # ---------- Initializing W_bi ---------

    W_bi = bisca * (np.random.rand(res_size) * 2 - 1)
    return W_in, W_res, W_bi


##############################################################################

def res_exe(W_in, W_res, W_bi, leak, U):
    T = U.shape[0]  # size of the input vector
    nres = W_res.shape[0]  # Getting the size of the network (= 100)
    R = np.zeros((T + 1, nres),
                 dtype='float32')  # Initializing the RCN output matrix (one extra frame for the warming up)
    for t in range(T):  # for each frame
        if scipy.sparse.issparse(W_in):
            a = W_in * U[t, :]
        else:
            a = np.dot(W_in, U[t, :])

        if scipy.sparse.issparse(W_res):
            b = W_res * R[t, :]
        else:
            b = np.dot(W_res, R[t, :])
        R[t + 1, :] = np.tanh(a + b + W_bi)
        R[t + 1, :] = (1 - leak) * R[t, :] + leak * R[t + 1, :]
#     R = np.concatenate((np.ones((R.shape[0], 1)), R), 1)
    return R[1:, :]  # returns the reservoir output and the desired output

##############################################################################
def getX(W_in,W_res,W_bi,leak,bi_direc,u):
    x=res_exe(W_in,W_res,W_bi,leak,u)
    if bi_direc:
        x=np.concatenate((x,np.flipud(res_exe(W_in,W_res,W_bi,leak,np.flipud(u)))),1)
    x=np.concatenate((np.ones((x.shape[0],1)),x),1)
    return x 
##############################################################################

def res_train(xTx, xTy, xlen, regu):
    t1 = time.time()
    lmda = regu ** 2 * xlen
    inv_xTx = np.linalg.inv(xTx + lmda * np.eye(xTx.shape[0],dtype=np.float32))
    t2 = time.time()
    beta = np.dot(inv_xTx, xTy)  # beta is the output weight matrix
    return beta,inv_xTx,t2-t1

def rcn_infer(w_in,w_res,w_bi,w_out,leak,r_prev,u):
    if scipy.sparse.issparse(w_in): # applying input weights to the input. Sparse and dense matrix multiplication is different in Python 
        a1 = w_in * u 
    else:
        a1=np.dot(w_in, u)
    if scipy.sparse.issparse(w_res): # applying recurrent weights to the previous reservoir states 
        a2 = w_res * r_prev 
    else:
        a2 = np.dot(w_res, r_prev)
    
    r_now = np.tanh(a1 + a2 + w_bi) # adding bias and applying activation function
    r_now = (1 - leak) * r_prev + leak * r_now # applying leak rate
    y = np.dot(np.append([1],r_now),w_out) # applying the output weight
    return r_now,y

def moving_average(signal, window_length):
    """
    Calculate the moving average of a signal.

    Parameters:
        signal (list or numpy array): The input signal.
        window_length (int): The length of the moving average window.

    Returns:
        numpy array: The moving average of the signal.
    """
    if window_length <= 0:
        raise ValueError("Window length must be a positive integer.")
    if window_length > len(signal):
        raise ValueError("Window length cannot be greater than the signal length.")

    # Use numpy's convolution function to compute the moving average
    window = np.ones(window_length) / window_length
    moving_avg = np.convolve(signal, window, mode='same')

    return moving_avg

def calculate_derivatives(df, window_size,smooth_len):
    window_size = np.int32(window_size/df.index.to_series().diff().fillna(method='bfill').mean())
    smooth_len = np.int32(smooth_len/df.index.to_series().diff().fillna(method='bfill').mean())
    
    if window_size!=0 or smooth_len!=0:
    
        # Assume df is a DataFrame with a single column of data
        # Calculate time differences in seconds if the index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            time_diff = df.index.to_series().diff().dt.total_seconds().fillna(method='bfill')
            rolling_time_diff = time_diff.rolling(window=window_size, min_periods=1).sum()
        else:

            # For a non-datetime index, use the window size as the denominator directly
            rolling_time_diff = window_size

        # Calculate rolling differences for the data
        rolling_diff = df.rolling(smooth_len).mean().diff(window_size).fillna(method='bfill')

        # Calculate first derivative
        first_derivative = rolling_diff / rolling_time_diff

        # Calculate second derivative
        rolling_diff_first_derivative = first_derivative.rolling(smooth_len).mean().diff(window_size).fillna(method='bfill')
        second_derivative = rolling_diff_first_derivative / rolling_time_diff

        return first_derivative.add_suffix('_1dev'), second_derivative.add_suffix('_2dev')
    else:
        print('data length smaller than the windows! return 0s')
        return np.zeros(df.shape),np.zeros(df.shape)
    
    
def calculate_derivatives_simple(S):
    N = len(S)
    dS = np.zeros_like(S)  # First derivative
    dS2 = np.zeros_like(S)  # Second derivative

    # Calculate first derivative (backward difference)
    for i in range(1, N):
        dS[i] = S[i] - S[i-1]

    # Calculate second derivative (backward difference)
    for i in range(2, N):
        dS2[i] = dS[i] - dS[i-1]
    return dS,dS2

def extract_signal_chunks(df, timestamps, N):
    """
    Extract chunks of a signal from a DataFrame for each given timestamp.

    Parameters:
        df (pd.DataFrame): A timeseries DataFrame with the index as time in seconds.
        timestamps (list or array): An array of timestamps (in seconds) for which to extract chunks.
        N (int): Length of each chunk in seconds leading up to the given timestamp.

    Returns:
        pd.DataFrame: A DataFrame where each row is a chunk of the signal corresponding to an N-second interval.
    """
    # Ensure the dataframe index is sorted
    df = df.sort_index()
    fs = 1/np.mean(np.diff(df.index))
    print(fs)
    N_smple=np.int32(fs*N)
    # Create an empty DataFrame to hold the results
    result = []

    for ts in timestamps:
        # Define the time range for the chunk
        end_time_idx=np.argmin(np.abs(df.index-ts))
        start_time_idx = end_time_idx - N_smple
        # print(start_time_idx,end_time_idx)
        # Extract the chunk for the given timestamp
        chunk = df.iloc[start_time_idx:end_time_idx].rolling(10).mean().fillna(method='bfill').fillna(method='ffill')
        result.append(chunk)

    result=pd.DataFrame(np.hstack([x.to_numpy() for x in result]).T,columns=[f'temp{x+1}' for x in range(N_smple)],index=timestamps)
    
    return result


def low_pass_filter_with_time(signal, time, cutoff_freq, order=4):
    """
    Apply a low-pass Butterworth filter to a signal array with a corresponding time array.

    Parameters:
    - signal (array-like): The input signal to be filtered.
    - time (array-like): The time array corresponding to the signal.
    - cutoff_freq (float): The cutoff frequency of the low-pass filter (Hz).
    - order (int): The order of the Butterworth filter (default is 4).

    Returns:
    - filtered_signal (array-like): The filtered signal.
    """
    # Calculate sampling rate from the time array
    sampling_rate = 1 / np.mean(np.diff(time))

    # Normalize the cutoff frequency by the Nyquist frequency
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist

    # Design a Butterworth low-pass filter
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)

    # Apply the filter to the signal
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal

def interpolate_timeseries(df, new_time_array):
    """
    Interpolates a time series DataFrame based on a new time array.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with time as the index and one or more columns for data.
    - new_time_array (array-like): The new time array for interpolation.

    Returns:
    - interpolated_df (pd.DataFrame): A new DataFrame with the new time array as the index and interpolated values.
    """
    # Ensure the DataFrame index is sorted
    df = df.sort_index()

    # Create a new DataFrame with the new time array
    new_index = pd.Index(new_time_array, name="Time")
    interpolated_df = pd.DataFrame(index=new_index)

    # Interpolate each column
    for column in df.columns:
        interpolated_df[column] = np.interp(
            new_time_array, df.index.values, df[column].values
        )

    return interpolated_df

def rolling_average_past_window(signal, time, window_seconds):
    """
    Apply a rolling average on a signal array using a past window [t-N, t) without using pandas.

    Parameters:
    - signal (array-like): The signal values to be averaged.
    - time (array-like): The corresponding time values in seconds.
    - window_seconds (float): The rolling window size in seconds.

    Returns:
    - smoothed_signal (array-like): The signal after applying the rolling average.
    """
    smoothed_signal = np.zeros_like(signal)

    for i, t in enumerate(time):
        # Define the past window [t-N, t)
        lower_bound = t - window_seconds

        # Find indices within the window [t-N, t)
        indices = np.where((time >= lower_bound) & (time < t))[0]

        # Compute the average of signal values within the window
        if len(indices) > 0:
            smoothed_signal[i] = np.mean(signal[indices])
        else:
            smoothed_signal[i] = np.nan  # If no points in the window, set to NaN

    return smoothed_signal

def generate_c_array(matrix_list, matname, l_idx, paramprefix):
    """
    Generates a C-compatible 3D array initialization from a list of matrices and writes it to a file.

    Args:
        matrix_list (list of list of list of floats): The matrices to be converted.
        table_name (str): The name of the table in C.
        output_file (str): The path of the file to save the C code.
    """
    table_name = f'{paramprefix.lower()}_{matname}L{l_idx+1}'
    n_models = len(matrix_list)
    sparse_flag = True if scipy.sparse.issparse(matrix_list[0]) else False
    c_array = f"// Sparse matrix " if sparse_flag else f"// Dense matrix "
    PcsLineIn = ""
    PcsLineRec = ""
    if sparse_flag:
        firstelem = f"[{paramprefix}_RcnL{l_idx+1}Nonzero]"
        secondelem = f"[{paramprefix}_RcnNSparse]"
    if matname == "wIn":
        if sparse_flag:
            firstelem = f"[{paramprefix}_RcnL{l_idx+1}NonzeroIn]"
            secondelem = f"[{paramprefix}_RcnNSparse]"
            PcsLineIn = f"Sparse, {firstelem}, {secondelem}, wInL{l_idx+1}[i],"
        else:
            firstelem = f"[{paramprefix}_RcnL{l_idx+1}Size]"
            secondelem = f"[{paramprefix}_RcnL{l_idx+1}NInputs]"
            PcsLineIn = f"Dense, {firstelem}, {secondelem}, wInL{l_idx+1}[i],"
    elif matname == "wRec":
        if sparse_flag:
            firstelem = f"[{paramprefix}_RcnL{l_idx+1}NonzeroRec]"
            secondelem = f"[{paramprefix}_RcnNSparse]"
            PcsLineRec = f"Sparse, {firstelem}, {secondelem}, wRecL{l_idx+1}[i],"
        else:
            firstelem = f"[{paramprefix}_RcnL{l_idx+1}Size]"
            secondelem = f"[{paramprefix}_RcnL{l_idx+1}Size]"
            PcsLineRec = f"Dense, {firstelem}, {secondelem}, wRecL{l_idx+1}[i],"
    if matname == "wOut":
        firstelem = f"[{paramprefix}_RcnL{l_idx+1}Internal]"
        secondelem = f"[{paramprefix}_RcnL{l_idx+1}OutSize]"
    elif matname == "wBi":
        firstelem = f"[{paramprefix}_RcnL{l_idx+1}Size]"
        secondelem = ""
    elif matname == "rTr":
        firstelem = f"[{paramprefix}_RcnL{l_idx+1}Internal]"
        secondelem = f"[{paramprefix}_RcnL{l_idx+1}Internal]"
    elif matname == "rTd":
        firstelem = f"[{paramprefix}_RcnL{l_idx+1}Internal]"
        secondelem = f"[{paramprefix}_RcnL{l_idx+1}OutSize]"


    print(PcsLineIn,PcsLineRec)
    c_array += f"{table_name}\nstatic float const {table_name}[{paramprefix}_RcnNModels]{firstelem}{secondelem} = {{\n"

        # if matrix_list[0].ndim==1:
        #     # n_rows = matrix_list[0].shape[0]
        #     # n_cols=1
        #     c_array +="[1] = {{\n"
        # else:
        #     # n_rows,n_cols=matrix_list[0].shape
        #     c_array +="[1] = {{\n"

    # Start constructing the C code
    # c_array += f"static float const {table_name}[{paramprefix}NModels][{n_rows}][{n_cols}] = {{\n"
    for model_idx, matrix in enumerate(matrix_list):
        if scipy.sparse.issparse(matrix):
            c_array += "\t{\n"
            # matrix=np.asarray(matrix.copy().todense())
            # c_array+=f"static float const {table_name}[{n_models}][{lnonzero}][{3}] = {{\n"
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    value =  matrix[row,col]
                    if not value==0:
                        c_array+=f"\t\t{{ {row}, {col}, {value:.6f}f }},\n"
            
#             rows, cols = matrix.nonzero()
#             values = matrix.data
            
#             for row, col, value in zip(rows, cols, values):
#                 c_array+=f"\t\t{{ {row}, {col}, {value:.6f}f }},\n"
            # c_array += "\t},\n"
        elif matrix.ndim==1:
            c_array += "\t{\n"
            row_data = ",\n".join(f"\t\t{val:.6f}f" for val in matrix)
            c_array += f" {row_data},\n"
            # c_array += "\t},\n"
            # print
        else:
            c_array += "\t{\n"
            for row in matrix:
                row_data = ", ".join(f"{val:.6f}f" for val in row)
                c_array += f"\t\t{{ {row_data} }},\n"
        c_array += "\t},\n"
    
    c_array += "};\n"
    return c_array

def rcn2c(weights, param_text, main_path, norm_txt=''):
    param_prefix = param_text[0]
    param_postfix = param_text[1]

    matlist = ['wIn','wRec','wBi','wOut']
    num_layers = len(weights[0]['wIn'])
    nnzIn = np.zeros(num_layers)
    nnzRec = np.zeros(num_layers)
    for l_idx in range(num_layers):
        nnzIn[l_idx]  = weights[0]['wIn'][l_idx].nnz if scipy.sparse.issparse(weights[0]['wIn'][l_idx]) else -1
        nnzRec[l_idx] = weights[0]['wRec'][l_idx].nnz if scipy.sparse.issparse(weights[0]['wRec'][l_idx]) else -1

    fname = f"kstar_{param_prefix.lower()}_matrices{param_postfix}"
    output_file = f'{main_path}{fname}'
    with open(f'{output_file}.h', "w") as fid:
        fid.write("#pragma once\nenum {\n")
        fid.write(f"\t{param_prefix}_RcnNSparse = 3, //row_ind, col_ind, val\n")
        fid.write(f"\t{param_prefix}_RcnTotalNInputs = {weights['nInpTot']}, //Total number of inputs\n")
        fid.write(f"\t{param_prefix}_RcnOutSize = {weights['outDim']}, ////output dim of RCN (assuming all RCNs have same)\n")
        fid.write(f"\t{param_prefix}_RcnNModels = {weights['nEns']}, //num of ensemble models\n")
        fid.write(f"\t{param_prefix}_RcnNLayers = {num_layers}, //number of RCN layers per ensemble model\n")
        # fid.write(f"\t{param_prefix}_DataSize = {weights['nSamp']},//Number of training samples\n")
        fid.write("\n")

        for l_idx in range(num_layers): # num of RCN layers
            fid.write(f"\t//RCN {l_idx+1} params\n")
            fid.write(f"\t{param_prefix}_RcnL{l_idx+1}Size = {weights['resSize'][l_idx]}, //size of RCN\n")
            fid.write(f"\t{param_prefix}_RcnL{l_idx+1}Internal = {weights['resSize'][l_idx]+1}, //{param_prefix}_L{l_idx+1}Size + 1\n") 
            fid.write(f"\t{param_prefix}_RcnL{l_idx+1}NInputs = {weights['inDim'][l_idx]}, //input dim of first RCN\n")
            fid.write(f"\t{param_prefix}_RcnL{l_idx+1}OutSize = {weights['outDim']}, //output dim of first RCN\n")
            fid.write(f"\t{param_prefix}_RcnL{l_idx+1}KIn = {weights['kIn'][l_idx]}, // num of input connections per RCN node (-1 = fully connected)\n")
            fid.write(f"\t{param_prefix}_RcnL{l_idx+1}NonzeroIn = {nnzIn[l_idx]:.0f}, // total num of input connections for sparse wIn (Negative = Fully connected)\n")
            fid.write(f"\t{param_prefix}_RcnL{l_idx+1}KRec = {weights['kRec'][l_idx]}, // num of inrecurrent connections per RCN node (-1 = fully connected)\n")
            fid.write(f"\t{param_prefix}_RcnL{l_idx+1}NonzeroRec = {nnzRec[l_idx]:.0f}, // total num of recurrent connections for sparse wRec (Negative = Fully connected)\n\n")

        fid.write("};\n\n// RCN wights and parameters\n\n")

        for l_idx in range(num_layers):  # number of layers
            for matname in matlist:
                # print(f'{matname}L{l_idx+1}')
                matrices = [weights[x][matname][l_idx] for x in range(weights['nEns'])]
                c_array = generate_c_array(matrices, matname, l_idx, param_prefix)
                fid.write(c_array)
        fid.write(norm_txt)
        leak_out = f"static float const {param_prefix.lower()}_leak[{param_prefix}_RcnNLayers] = {{"
        leak_out +="f, ".join(map(str,weights['leak']))
        leak_out += "f};\n"
        fid.write(leak_out)

        # regu_out = f"static float const {param_prefix.lower()}_regu[{param_prefix}NLayers] = {{"
        # regu_out +="f, ".join(map(str,weights['regu']))
        # regu_out += "f};\n"
        # fid.write(regu_out)

    with open(f'{output_file}.pkl','wb') as fid:
        pickle.dump(weights,fid)
    return 1

def file_append(nfile, txline):
    fid = open(nfile, "a")
    fid.write(txline)
    fid.close()
    return 1

def filter_lp(x, t, tau):
    dt=np.mean(np.diff(t))
    alpha = dt / (dt + tau)
    x_filt=np.zeros(len(x) + 1)
    for k in range(len(x)):
        x_filt[k+1] = (x_filt[k] + alpha * (x[k] - x_filt[k]))
    return x_filt[1:]

def recursively_load_dict_from_group(h5file, path):
    """
    Recursively loads a nested dictionary from an h5py group.
    """
    result = {}
    group = h5file[path]
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            result[key] = recursively_load_dict_from_group(h5file, f"{path}/{key}")
        elif isinstance(item, h5py.Dataset):
            data = item[()]
            if item.dtype == h5py.special_dtype(vlen=str) or data.dtype.kind in ('S', 'U'):
                # Convert to list if it was a list of strings
                if isinstance(data, np.ndarray) and data.size > 1:
                    result[key] = [s.decode('utf-8') if isinstance(s, bytes) else s for s in data]
                else:
                    result[key] = data.decode('utf-8') if isinstance(data, bytes) else data
            else:
                result[key] = data
            # Decode bytes to string if needed
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            result[key] = data
    return result

def recursively_save_dict_to_group(h5file, path, dic):
    """
    Recursively saves a nested dictionary to an h5py group structure.
    """
    for key, item in dic.items():
        key_path = f"{path}/{key}"
        if isinstance(item, dict):
            # Create subgroup and recurse
            grp = h5file.require_group(key_path)
            recursively_save_dict_to_group(h5file, key_path, item)
        # else:
        #     # Convert scalars to arrays if needed
        #     if not isinstance(item, (np.ndarray, list, tuple)):
        #         item = np.array(item)
        #     h5file.create_dataset(key_path, data=item)

        elif isinstance(item, str):
            # Encode string to bytes (UTF-8) for HDF5 compatibility
            h5file.create_dataset(key_path, data=item, dtype=h5py.special_dtype(vlen=str))
        # Handle list of strings
        elif isinstance(item, list) and all(isinstance(item, str) for item in item):
            # Use a variable-length string type
            dt = h5py.special_dtype(vlen=str)
            dset = h5file.create_dataset(key_path, (len(item),), dtype=dt)
            dset[:] = item
        # Handle numerical data (int, float, numpy arrays)
        else:
            h5file[key_path] = item


def compute_beam_avg_power(time, values):
    time = np.asarray(time, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    # Threshold: beam is ON if power > 1% of the max observed power
    peak = values.max()
    if peak == 0:
        return np.zeros_like(values)
    threshold = 0.01 * peak

    is_on = values > threshold

    # Detect segment boundaries
    changes = np.diff(is_on.astype(int))
    change_idx = np.where(changes != 0)[0] + 1  # index where new state starts

    # Build segments: (start_idx, end_idx, is_on_state)
    boundaries = np.concatenate(([0], change_idx, [len(values)]))
    segments = []
    for i in range(len(boundaries) - 1):
        si, ei = boundaries[i], boundaries[i + 1]
        segments.append((si, ei, bool(is_on[si])))

    # Group into modulation cycles and compute average power per cycle.
    # A cycle = one ON segment + one adjacent OFF segment (in either order).
    # We integrate power over the full cycle: avg = integral / T_cycle.
    avg_power = np.full(len(values), np.nan)

    i = 0
    while i < len(segments):
        si, ei, state = segments[i]
        t_start = time[si]
        t_end = time[ei - 1]

        if not state:
            # OFF segment alone (e.g. before beam starts) -> avg = 0
            avg_power[si:ei] = 0.0
            i += 1
            continue

        # ON segment: look for a following OFF segment to form a cycle
        cycle_start = si
        cycle_end = ei
        if i + 1 < len(segments) and not segments[i + 1][2]:
            cycle_end = segments[i + 1][1]
            consumed = 2
        else:
            consumed = 1

        # Also check if there's a preceding OFF that belongs to this cycle
        # (for the first ON after a long OFF, the OFF already got assigned 0)

        # Compute average power over the cycle using trapezoidal integration
        cycle_time = time[cycle_start:cycle_end]
        cycle_vals = values[cycle_start:cycle_end]
        if len(cycle_time) > 1:
            integral = np.trapz(cycle_vals, cycle_time)
            duration = cycle_time[-1] - cycle_time[0]
            cycle_avg = integral / duration if duration > 0 else 0.0
        else:
            cycle_avg = cycle_vals[0]

        avg_power[cycle_start:cycle_end] = cycle_avg
        i += consumed

    # Forward-fill any remaining NaNs
    mask = np.isnan(avg_power)
    if mask.any():
        idx = np.where(~mask, np.arange(len(avg_power)), 0)
        np.maximum.accumulate(idx, out=idx)
        avg_power = avg_power[idx]
        still_nan = np.isnan(avg_power)
        if still_nan.any():
            first_valid = np.argmin(still_nan)
            avg_power[:first_valid] = avg_power[first_valid]
    return avg_power
