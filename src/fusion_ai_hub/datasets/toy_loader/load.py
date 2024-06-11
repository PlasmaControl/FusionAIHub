def get_data(self,discharge, suffix, norm=True):
    discharge_path=self.file_path_gen(discharge, suffix)
    input_file = h5py.File(discharge_path, 'r')
    input_dict_tmp = self.hdf5_to_dict(input_file)
    if suffix in no_level:
        input_dict={suffix:input_dict_tmp}
    else:
        input_dict=input_dict_tmp

    if norm and (suffix in self.norm_factor_list):
        for key in input_dict.keys():
            if self.norm_factor_list[suffix][key]=='log':
                input_dict[key]['zdata']=np.log(np.array(input_dict[key]['zdata'][:]))
            else:
                input_dict[key]['zdata']=np.array(input_dict[key]['zdata'][:])/self.norm_factor_list[suffix][key]
    
    return input_dict