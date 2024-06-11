def order_of_magnitude_normal_factor_calc(self,discharge):
    norm_factor_list_tmp={}
    for suffix in self.file_keys.keys():
        file_dict=self.get_data(discharge, suffix, norm=False)
        norm_factor_list_tmp[suffix]={}
        for key in file_dict.keys():
            mean_tmp=abs(np.mean(file_dict[key]['zdata'][:]))
            try:
                exponent=self.get_order_of_magnitude(mean_tmp)
                norm_factor_list_tmp[suffix][key]=10**exponent
            except:
                norm_factor_list_tmp[suffix][key]=1.
    return norm_factor_list_tmp

def avg_factor_calc(self,discharge):
    avg_factor={}
    for suffix in self.file_keys.keys():
        file_dict=self.get_data(discharge, suffix, norm=True)
        avg_factor[suffix]={}
        for key in file_dict.keys():
            data=file_dict[key]['zdata'][:]
            avg_tmp=np.mean(data,axis=len(data.shape)-1)
            if np.isnan(avg_tmp).any():
                avg_tmp=0.

            avg_factor[suffix][key]=avg_tmp

    return avg_factor

def std_factor_calc(self,discharge):
    std_factor={}
    for suffix in self.file_keys.keys():
        file_dict=self.get_data(discharge, suffix, norm=True)
        std_factor[suffix]={}
        for key in file_dict.keys():
            data=file_dict[key]['zdata'][:]
            std_tmp=np.std(data,axis=len(data.shape)-1)
            if np.isnan(std_tmp).any():
                std_tmp=1.
            std_factor[suffix][key]=std_tmp

    return std_factor