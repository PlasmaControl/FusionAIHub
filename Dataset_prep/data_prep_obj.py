import h5py
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import os
from tqdm import tqdm

import glob
import cv2



# In[2]:

file_normal_size = {
            'ece_cali': 600 * 1024**2,      #600MB
            'ece_s':    25 * 1024**2,      
            'co2_pl':   600 * 1024**2,
            'co2_den':  600 * 1024**2,
            'co2_s':    0.2 * 1024**2,
            'ts':       0.5 * 1024**2,
            'cer':      8 * 1024**2,
            'mse':      4 * 1024**2,
            'mag':      50 * 1024**2,
            'mag_hi':   1000 * 1024**2,
            'actu':     100 * 1024**2,
            'basic':    40 * 1024**2,
            'profiles': 5* 1024**2,
        }

#names of the cyrotrons
ech_gytname = ['lei','luk','r2d']

multi_level=['cer','mag','profiles','basic','actu','ts','ts_error','ts_rz']
no_level=['ece_cali','ece_s']

file_keys={ 'co2_s':['r0', 'v1', 'v2', 'v3'],\
            'co2_den':['r0', 'v1', 'v2', 'v3'],\
            'co2_pl':['r0', 'v1', 'v2', 'v3'],\
           
           'ece_cali':[],\
           'ece_s':[],\
           
           'ts':{r'{}.{}'.format(area,sig):[r'{}.{}'.format(area,sig)] for area in ['core','divertor','tangential']
                                                                     for sig in ['dens','temp']},
           'ts_error':{r'{}.{}'.format(area,sig):[r'{}.{}'.format(area,sig)] for area in ['core','divertor','tangential']
                                                                     for sig in ['dens','temp']},
           
           'ts_rz':{r'{}.{}'.format(area,sig):[r'{}.{}'.format(area,sig)] for area in ['core','divertor','tangential']
                                                                     for sig in ['r', 'z']},

           'cer': {
                        output: [f'q.{output}.v{channel:02d}' for channel in range(1, 33)] +
                                [f'q.{output}.t{channel:02d}' for channel in range(1, 49)]
                        for output in ['amp', 'samp', 'ti', 'sti', 'rot', 'srot', 'r', 'phi', 'nz', 'fz', 'zeff', 'vb', 'svb']
                    },\
                                                     
           'mse':[r'%02d'%i for i in range(1, 70)],\

           'mag':{'dsl':['dsl.1u180', 'dsl.2u180', 'dsl.3u180', 'dsl.4u157', 'dsl.5u157', 'dsl.6u157'],\
                  'mpi':['mpi.11m067', 'mpi.11m322', 'mpi.1a139', 'mpi.1a322', 'mpi.1b139', 'mpi.1b157', 'mpi.1b322', \
                        'mpi.1l180', 'mpi.1u157', 'mpi.2a067', 'mpi.2a139', 'mpi.2a322', 'mpi.2b067', 'mpi.2b139', \
                        'mpi.2b322', 'mpi.2l180', 'mpi.2u157', 'mpi.3a139', 'mpi.3a322', 'mpi.3b139', 'mpi.3b322', \
                        'mpi.3l180', 'mpi.3u157', 'mpi.4a139', 'mpi.4a322', 'mpi.4b139', 'mpi.4b322', 'mpi.4u157', \
                        'mpi.5a139', 'mpi.5a322', 'mpi.5b139', 'mpi.5b322', 'mpi.5u157', 'mpi.66m067', 'mpi.66m157', \
                        'mpi.66m247', 'mpi.66m322', 'mpi.67a097', 'mpi.67a142', 'mpi.67a157', 'mpi.67a322', \
                        'mpi.67b097', 'mpi.67b157', 'mpi.67b322', 'mpi.6fa322', 'mpi.6fb142', 'mpi.6fb322', \
                        'mpi.6na132', 'mpi.6na157', 'mpi.6na322', 'mpi.6nb157', 'mpi.6nb322', 'mpi.6u157', \
                        'mpi.79a147', 'mpi.79b142', 'mpi.79b322', 'mpi.79fa322', 'mpi.79na322', 'mpi.7fa322', \
                        'mpi.7fb322', 'mpi.7na322', 'mpi.7nb142', 'mpi.7nb322', 'mpi.7u157', 'mpi.89a322', \
                        'mpi.89b322', 'mpi.8a322', 'mpi.8b322', 'mpi.9a322', 'mpi.9b322'],\
                  'psf':['psf.1a', 'psf.1b', 'psf.2a', 'psf.2b', 'psf.3a', 'psf.3b', 'psf.4a', 'psf.4b', \
                        'psf.5a', 'psf.5b', 'psf.6fa', 'psf.6fb', 'psf.6na', 'psf.6nb', 'psf.7fa', \
                        'psf.7fb', 'psf.7na', 'psf.7nb', 'psf.8a', 'psf.8b', 'psf.9a', 'psf.9b'],\
                  'psi':['psi.11m', 'psi.12a', 'psi.12b', 'psi.1l', 'psi.23a', 'psi.23b', 'psi.2l', 'psi.34a', \
                        'psi.34b', 'psi.3l', 'psi.45a', 'psi.45b', 'psi.58a', 'psi.58b', 'psi.6a', 'psi.6b', \
                        'psi.7a', 'psi.7b', 'psi.89fb', 'psi.89nb', 'psi.9a', 'psi.9b']\
                 }, \
            'mag_hi':[f'b{i}' for i in range(1,9)],\
            'profiles':{'pressure': ['betap','betan','pres'], \
                    'other': ['wmhd','li'],\
                    'q_info':['q0','q95','qmin','qpsi'],\
                    'q_rho_info':['rhoqmin'],\
                    'mag_geo_para':['alpha','r0','aminor',\
                                    'kappa','tritop','tribot',\
                                    'rmaxis','zmaxis',\
                                    'volume'],\
                    'mag_map':['psirz','ssibry', 'ssimag'],\
                    'divertor_geo':['drsep',\
                                    'gapbot','gapin','gapout','gaptop',\
                                    'zxpt1','zxpt2'],\
                    'profile':['edensfit', 'etempfit',\
                               'itempfit','idensfit',\
                                'trotfit'],\
                    'mag_mode_number':['n1rms','n2rms','n3rms']},\
                    
            'basic':{'mag':['ip', 'ipsip', 'iptipp','pcbcoil', 'bcoil','bt','vloop'],\
                    'neutron':[ 'plasticfix', 'fzns'],\
                    'd_alpha':['fs00','fs01','fs02','fs03','fs04','fs05']},\
            'actu': {'pinj': ['pinjf_%dl' % k for k in [15,21,30,33]]+['pinjf_%dr' % k for k in [15,21,30,33]],\
                    'tinj':['tinj_%dl' % k for k in [15,21,30,33]]+['tinj_%dr' % k for k in [15,21,30,33]],\
                    'ech':['echpwrc','echpwr']\
                    +['ec%sfpwrc' % (x) for x in ech_gytname]\
                    +['ec%sxmfrac' % (x) for x in ech_gytname]\
                    +['ec%spolang' % (x) for x in ech_gytname],
                    'gas':['gasa', 'gasb', 'gasc', 'gasd', 'gase'],
                     'rmp_current':['c19', 'c79', 'c139', 'c199', 'c259', 'c319', \
                              'iu30', 'iu90', 'iu150', 'iu210', 'iu270', 'iu330', \
                              'il30', 'il90', 'il150', 'il210', 'il270', 'il330'],
                    'coil_field_strength':['ecoila', 'ecoilb', 'e567up', 'e567dn', 'e89dn', 'e89up']\
                              +['f1a','f2a','f3a','f4a','f5a','f6a','f7a','f8a','f9a',\
                                'f1b','f2b','f3b','f4b','f5b','f6b','f7b','f8b','f9b'] }
            
            }

data_keys=['xdata','ydata','zdata']
unit_keys=['xunits','yunits','zunits']


spec_params_default={
        'window': 'hamm',
        'scaling': 'density', # {'density', 'spectrum'}
        'detrend': 'linear', # {'linear', 'constant', False}
        'eps': 1e-11} 

#object that contains the functions to maniupulate one discharge 
class DichargePerp():
    def __init__(self,discharge=174823,suffix_list=['co2_s']):
        self.discharge=discharge
        self.suffix_list=suffix_list
        
    def file_path_gen(self,discharge,suffix):
        return f'/scratch/gpfs/EKOLEMEN/big_d3d_data/{str(discharge)[:2]}0000/{discharge}_{suffix}.h5'
        
    def get_data(self,discharge,suffix):
        discharge_path=self.file_path_gen(discharge,suffix)
        input_file = h5py.File(discharge_path, 'r')
        return input_file

    #divide the data into sub catagory
    def data_division(self,input_file,input_suffix):
        if input_suffix in multi_level:
            input_multi_level={}
            for key in file_keys[input_suffix].keys():
                keys_of_this_catagory=file_keys[input_suffix][key]
                input_multi_level[key]={key_i:input_file[key_i] for key_i in keys_of_this_catagory}
        else:
            input_multi_level={'only': input_file}
        return input_multi_level

    def get_full_data(self):
        file_dict={}
        for suffix in self.suffix_list:
            input_file=self.get_data(self.discharge,suffix)
            file_dict[suffix]=self.data_division(input_file,suffix)
        self.file_dict=file_dict
        return file_dict
        
    @staticmethod
    def spec_filters(freq, time, amp_f_t,spec_params=spec_params_default,thr=0.9, gaussblr_win=(31,3)):
        def norm(amp_f_t):
            mn = amp_f_t.mean()
            std = amp_f_t.std()
            return((amp_f_t-mn)/std)
        
        def rescale(amp_f_t):
            return (amp_f_t-amp_f_t.min())/(amp_f_t.max()-amp_f_t.min())
        
        def quantfilt(amp_f_t,thr=0.9):
            filt = np.quantile(amp_f_t,thr,axis=0)
            out = np.where(amp_f_t<filt,0,amp_f_t)
            return out
        
        # gaussian filtering
        def gaussblr(amp_f_t,filt=(31, 3)):
            amp_f_t = (rescale(amp_f_t)*255).astype('uint8')
            out = cv2.GaussianBlur(amp_f_t,filt,0)
            return rescale(out)
        
        # mean filtering
        def meansub(amp_f_t):
            mn = np.mean(amp_f_t,axis=1)[:,np.newaxis]
            out = np.absolute(amp_f_t - mn)
            return rescale(out)
        
        # morphological filtering
        def morph(amp_f_t):
            amp_f_t = (rescale(amp_f_t)*255).astype('uint8')
            se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
            se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
            mask = cv2.morphologyEx(amp_f_t, cv2.MORPH_CLOSE, se1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
            return rescale(mask)
        
        def apply_all(freq, time, amp_f_t,spec_params,thr=thr, gaussblr_win=gaussblr_win):
            
            Sxx = np.log(amp_f_t + spec_params['eps'])
            Sxx=(Sxx-np.min(Sxx))/(np.max(Sxx)-np.min(Sxx)) # rescale the pixels to (0,1)
            
            Sxx_enhanced = quantfilt(Sxx,thr)
            Sxx_enhanced = gaussblr(Sxx_enhanced,gaussblr_win)
            Sxx_enhanced = meansub(Sxx_enhanced)    
            Sxx_enhanced = morph(Sxx_enhanced)
            Sxx_enhanced = meansub(Sxx_enhanced)    
            
            return freq, time, Sxx_enhanced

        return apply_all(freq, time, amp_f_t,spec_params,thr=thr, gaussblr_win=gaussblr_win)

    @staticmethod
    def spectro_calc(sig_time,data,spec_params=spec_params_default, plot=False):
        spec_params['fs']=1./np.mean(sig_time[1:]-sig_time[:-1])
        spec_params['nperseg']=max(int(0.6*spec_params['fs']),1) # default 1024
        spec_params['noverlap']=max(int(spec_params['nperseg']/4),1) # default: nperseg / 4
        print(spec_params)
        
        freq, time, amp_f_t = signal.spectrogram(data, \
                                             nperseg=spec_params['nperseg'], \
                                             noverlap=spec_params['noverlap'],\
                                             fs=spec_params['fs'], \
                                             window=spec_params['window'],\
                                             scaling=spec_params['scaling'], \
                                             detrend=spec_params['detrend'])
        if plot:
            plt.clf()
            plt.imshow(amp_f_t.T,aspect='auto',cmap='hot',\
                       extent=[time[0], time[-1],\
                               freq[-1], freq[0]])
            plt.colorbar()
            plt.ylabel('kHz')
            plt.xlabel('ms')
            plt.gca().invert_yaxis()
            plt.show()
        return freq, time, amp_f_t
        
    @staticmethod
    def spectro_plot(freq, time, amp_f_t):
        plt.clf()
        plt.imshow(amp_f_t,aspect='auto',cmap='hot',\
                    extent=[time[0], time[-1],\
                            freq[-1], freq[0]])
        plt.colorbar()
        plt.ylabel('kHz')
        plt.xlabel('ms')
        plt.gca().invert_yaxis()
        plt.show()
        
    @staticmethod
    def time_serie_plot(dict):
        plt.clf()
        if (dict['zdata'][:].shape)==1:
            plt.plot(dict['xdata'][:],dict['zdata'][:])
            
        else:
            plt.plot(dict['xdata'][:],dict['zdata'][:].T)
        plt.xlabel('Time (ms)')
        plt.show()

    @staticmethod
    def time_matching_merge_asof_1d(time1, data1, time_std):
        # Convert input arrays to DataFrames
        df1 = pd.DataFrame({'time1': time1, 'data1': data1})
        df2 = pd.DataFrame({'time_std': time_std})
        
        # Sort the DataFrames by the time columns
        df1.sort_values('time1', inplace=True)
        df2.sort_values('time_std', inplace=True)
        
        # Perform the asof merge
        merged_df = pd.merge_asof(df2, df1, left_on='time_std', right_on='time1', direction='nearest')

        # Drop unnecessary columns and handle NaN values
        merged_df.drop(columns='time_std', inplace=True)
        merged_df.dropna(inplace=True)

        # Extract the matched time and data
        matched_time = merged_df['time1'].values
        matched_data = merged_df['data1'].values

        return matched_time, matched_data
        
    @staticmethod
    def time_matching_merge_asof_2d(time1, data1, time_std):
        # Create DataFrames
        # Note: We assume time1 and time_std are already float arrays representing time in seconds
        df1 = pd.DataFrame(data1.T, index=time1)  # Transpose data1 to align time as rows
        df_std = pd.DataFrame(index=time_std)

        # Reset index to include time in the DataFrame directly for merging
        df1 = df1.reset_index().rename(columns={'index': 'time1'})
        df_std = df_std.reset_index().rename(columns={'index': 'time_std'})

        # Perform merge_asof to find the closest matches
        merged = pd.merge_asof(df_std.sort_values('time_std'), df1.sort_values('time1'), left_on='time_std', right_on='time1', direction='nearest')

        # Extract the aligned times (as float)
        matched_time = merged['time1'].values

        # Extract indices from the merged DataFrame
        indices = merged['time1'].apply(lambda x: np.where(df1['time1'] == x)[0][0] if x in df1['time1'].values else -1).values
        
        # Use indices to fetch data from the original 2D array, handle missing indices
        matched_data = np.array([data1[:, int(idx)] if idx != -1 else np.full(data1.shape[0], np.nan) for idx in indices]).T

        return matched_time, matched_data
        
    @staticmethod
    def time_matching_binary_search(time1, data1, time_std, mode='2d'):
        # Function to find the closest time in time1 to each time in time_std
        def find_closest(target):
            # Binary search for the closest timestamp
            low, high = 0, len(time1) - 1
            best_idx = low
            while low <= high:
                mid = (low + high) // 2
                if time1[mid] < target:
                    low = mid + 1
                elif time1[mid] > target:
                    high = mid - 1
                else:
                    return mid
                # Update the best index if the current mid is closer to the target
                if abs(time1[mid] - target) < abs(time1[best_idx] - target):
                    best_idx = mid
            return best_idx

        # Align data1 to time_std
        matched_data = []
        matched_time = []
        for t in time_std:
            closest_idx = find_closest(t)
            if mode=='2d':
                matched_data.append(data1[:,closest_idx])
            else:
                matched_data.append(data1[closest_idx])
            matched_time.append(time1[closest_idx])

        return matched_time, matched_data
        
    @classmethod
    def time_matching(cls,time, data, time_std, mode='merge_asof'):
        if len(data.shape)==1:
            if mode=='merge_asof':
                return cls.time_matching_merge_asof_1d(time, data, time_std)
            elif mode=='binary':
                return cls.time_matching_binary_search(time, data, time_std, mode='1d')
        elif len(data.shape)==2:
            if mode=='merge_asof':
                return cls.time_matching_merge_asof_2d(time, data, time_std)
            elif mode=='binary':
                return cls.time_matching_binary_search(time, data, time_std, mode='2d')
        else:
            print('The data has to be 1d arry or 2d array')
                
    @staticmethod
    def time_interp_past_looking(time, data, time_std, mode='extrapolate'):
        if mode=='extrapolate':
            pass
        elif mode=='fill':
            pass
            
    @staticmethod
    def time_interp(time, data, time_std):
        return np.interp(time_std,time, data)



class DatasetPrep(DichargePerp):
    def __init__(self,discharge_search_list,suffix_list):
        self.discharge_search_list=discharge_search_list
        self.suffix_list=suffix_list
    
    def filter_discharges(self):
        suffix_list=self.suffix_list
        discharge_search_list=self.discharge_search_list
        # Define the criteria for the files you're interested in
        
        criteria = {key:file_normal_size[key]*0.5 for key in suffix_list}

        discharge_list = {key: [] for key in suffix_list}
        
        for discharge in tqdm(discharge_search_list):
            for suffix, size_limit in criteria.items():
                discharge_path=self.file_path_gen(discharge,suffix)
                # Check if the file exists
                if os.path.isfile(discharge_path):
                    # Get the size of the file
                    file_size = os.path.getsize(discharge_path)
                    if file_size > size_limit:
                        discharge_list[suffix].append(discharge)
                    
                else:
                    pass
            
        return discharge_list


class data_obj_rest():
    
    def save_dict_to_hdf5(dictionary, h5file):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                group = h5file.create_group(key)
                save_dict_to_hdf5(value, group)
            else:
                h5file.create_dataset(key, data=value)

    def TS_interp_(discharge,write_h5=True,plot=False):
        TS_Z_min_set=[0.0, 0.03, 0.09, 0.1, 0.15, 0.16, 0.21, 0.22, 0.26, 0.27, 0.28,\
                      0.3, 0.31, 0.32, 0.36, 0.37, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, \
                      0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55,\
                      0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66,\
                      0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, \
                      0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, \
                      0.89, 0.9, 0.91, 0.92, 0.93]
        str_shot=str(discharge)[:2]
        path = f'/scratch/gpfs/EKOLEMEN/big_d3d_data/{str_shot}0000/'
        
        TS_file = h5py.File(path + str(discharge) + '_TS.h5', 'r')
        TS_RZ_file = h5py.File(path + str(discharge) + '_TS_RZ.h5', 'r')
        
        TS_Z=TS_RZ_file['S.BLESSED.CORE.Z']['zdata'][:]
        order_index=np.argsort(TS_Z)
        TS_Z_sort=TS_Z[order_index]
        
        TS_interp={}
        TS_keys=['TS.BLESSED.CORE.density','TS.BLESSED.CORE.temp']
        for key in TS_keys:
            TS_interp_list=[]
            TS_time=TS_file[key]['xdata'][:]
            for i in range(len(TS_time)):
                TS_data=TS_file[key]['zdata'][:,i]*0.1**19
                TS_data_sort=TS_data[order_index]
                TS_interp_tmp=np.interp(TS_Z_min_set,TS_Z_sort,TS_data_sort)
                TS_interp_list.append(TS_interp_tmp)
            #########*********************start herere
            TS_interp[key]={'xdata':np.array(TS_time),'ydata':np.array(TS_Z_min_set),'zdata':np.array(TS_interp_list).T*10.**19}

        if write_h5:
            
            with h5py.File(f'{path}{discharge}_TS_core_interp.h5', 'w') as h5file:
                save_dict_to_hdf5(TS_interp, h5file)

        if plot:
            plt.clf()
            plt.scatter(TS_Z_min_set,TS_interp[key]['zdata'][:,600],label='interp')
            plt.scatter(TS_Z_sort,(TS_file[key]['zdata'][:,600])[order_index],label='origin')
            plt.legend()
            plt.show()
            
        return 0
        
            
    def read_file(discharge,file_suffix,df_time):

        path=find_path(discharge)
        file=h5py.File(f'{path}{discharge}_{file_suffix}.h5', 'r')
        keys=file.keys()

            
        for i,key in enumerate(keys):
            dict_tmp={'xdata':file[key]['xdata']}
            if len(file[key]['zdata'].shape)==2:
                for j in range(file[key]['zdata'].shape[0]):
                    dict_tmp[key+str(j)]=file[key]['zdata'][j,:]
            elif len(file[key]['zdata'].shape)==1:
                dict_tmp[key]=file[key]['zdata']
                
            df_tmp=pd.DataFrame(dict_tmp).astype('float32')
                
            if i ==0:
                df= pd.merge_asof(df_time,df_tmp,on='xdata',direction='nearest')
            else:
                df= pd.merge_asof(df,df_tmp,on='xdata',direction='nearest')

        file.close()
        return df

    def hdf5_generator(discharge_list,h5_profiles,data_filename='diag2diag.pkl'):
        all_X=[]
        all_y=[]
        all_time=[]
        discharg_read_list=[]
        len_list=[]
        for discharge in tqdm(discharge_list):
            print(discharge)
            try:
                dfs={}
                #creating the standard time 
                path=find_path(discharge)
                
                file = h5py.File(f'{path}{discharge}_shape.h5', 'r')
                t_min=0
                t_max=file['R0']['xdata'][-1]
                file.close()
                
                file=h5py.File(f'{path}{discharge}_TS.h5', 'r')
                df_time=pd.DataFrame({'xdata':file[list(file.keys())[0]]['xdata']})
                time=file[list(file.keys())[0]]['xdata'][:]
                
                time_index=(time >= t_min) & (time <= t_max)
                time_tmp=time[time_index]
                df_time=pd.DataFrame({'xdata':time_tmp})
                file.close()
                
                #Read all the files 
                for file_suffix in h5_profiles:
                    df=read_file(discharge,file_suffix,df_time)
                    dfs[file_suffix]=df

                #summarize all the data in this dicharge 
                df_tmp=np.concatenate([dfs[key].to_numpy()[1:] for key in dfs.keys()], axis=1)

                key_list_dict={}
                key_list=[]
                for key in dfs.keys():
                    key_list_dict[key]=list(dfs[key].keys())
                    for key_ in key_list_dict[key]:
                        key_list.append(key_)
                
                #add this discharge to the total file 
                all_X.append(df_tmp)
                all_time.append(df_time['xdata'])
                all_time_tmp= np.concatenate(all_time, axis=0)
                all_X_tmp = np.concatenate(all_X, axis=0)
                len_list.append(df_time['xdata'].shape[0])
                discharg_read_list.append(discharge)
                # Serialize the data and save to a file
                with open(data_filename, 'wb') as file:
                    pickle.dump([all_X_tmp,all_time_tmp,discharg_read_list,len_list,key_list,key_list_dict], file)

            except Exception as e:
            #if 2==1:
                print(f"Error: {e}")
                continue
            finally:
            #if 2==1:
                try:
                    file.close()
                except:
                    continue

        return [all_X_tmp]       
