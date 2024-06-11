def read_file(discharge, file_suffix, df_time):

    path = find_path(discharge)
    file = h5py.File(f'{path}{discharge}_{file_suffix}.h5', 'r')
    keys = file.keys()

    for i, key in enumerate(keys):
        dict_tmp = {'xdata': file[key]['xdata']}
        if len(file[key]['zdata'].shape) == 2:
            for j in range(file[key]['zdata'].shape[0]):
                dict_tmp[key+str(j)] = file[key]['zdata'][j, :]
        elif len(file[key]['zdata'].shape) == 1:
            dict_tmp[key] = file[key]['zdata']
            
        df_tmp = pd.DataFrame(dict_tmp).astype('float32')
        if i == 0:
            df = pd.merge_asof(df_time, df_tmp, on='xdata',
                                direction='nearest')
        else:
            df = pd.merge_asof(df, df_tmp, on='xdata',
                                direction='nearest')
    file.close()
    return df