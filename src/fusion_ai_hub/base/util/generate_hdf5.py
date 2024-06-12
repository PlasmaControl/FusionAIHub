
# reference
def hdf5_generator(discharge_list, h5_profiles,
                    data_filename='diag2diag.pkl'):
    all_X =[]
    all_y = []
    all_time = []
    discharg_read_list = []
    len_list = []
    for discharge in tqdm(discharge_list):
        print(discharge)
        try:
            dfs = {}
            # creating the standard time
            path=find_path(discharge)
            
            file = h5py.File(f'{path}{discharge}_shape.h5', 'r')
            t_min = 0
            t_max = file['R0']['xdata'][-1]
            file.close()
            
            file = h5py.File(f'{path}{discharge}_TS.h5', 'r')
            df_time = pd.DataFrame({'xdata': file[list(file.keys())[0]]['xdata']})
            time = file[list(file.keys())[0]]['xdata'][:]
            
            time_index = (time >= t_min) & (time <= t_max)
            time_tmp = time[time_index]
            df_time = pd.DataFrame({'xdata': time_tmp})
            file.close()
            
            # Read all the files
            for file_suffix in h5_profiles:
                df = read_file(discharge, file_suffix, df_time)
                dfs[file_suffix] = df

            # summarize all the data in this dicharge
            df_tmp = np.concatenate(
                [dfs[key].to_numpy()[1:] for key in dfs.keys()], axis=1)

            key_list_dict = {}
            key_list = []
            for key in dfs.keys():
                key_list_dict[key]=list(dfs[key].keys())
                for key_ in key_list_dict[key]:
                    key_list.append(key_)
            
            # add this discharge to the total file
            all_X.append(df_tmp)
            all_time.append(df_time['xdata'])
            all_time_tmp= np.concatenate(all_time, axis=0)
            all_X_tmp = np.concatenate(all_X, axis=0)
            len_list.append(df_time['xdata'].shape[0])
            discharg_read_list.append(discharge)
            # Serialize the data and save to a file
            with open(data_filename, 'wb') as file:
                pickle.dump([all_X_tmp, all_time_tmp, discharg_read_list,
                                len_list, key_list, key_list_dict], file)

        except Exception as e:  # if 2==1:
            print(f"Error: {e}")
            continue
        finally:  # if 2==1:
            try:
                file.close()
            except:
                continue

    return [all_X_tmp]