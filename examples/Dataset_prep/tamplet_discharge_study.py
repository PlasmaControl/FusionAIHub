import data_prep_obj as data_prep

#************start of user block************
discharge=174823
suffix_list=['ts','ece_s']

#setting for finding flat top
Ip_window_size=500
Ip_std_threshold=0.01
plot_Ip=False

#mode of normalization
norm_mode='all'

#the key for the standard time [suffix, key]
custom_time_std=False
time_std_key=['ts','core.dens']
time_std=[]

#the list suffix and key to e
interp_suffix=[['ts','core.dens'],['ts','core.dens']]  #e.g. [['ts','core.dens'],['ts','core.dens']]
interp_mode='normal'

#time_matching_mode = ['merge_asof','binary','dynamic'] only 'dynamic' works for now (04/29/2024)
time_matching_mode='dynamic'


left_window={'ece_s':50}
right_window={'ece_s':50}

time_matching_padding='zeros' #['zeros', 'last', 'nan']

plot_matched_data=True #plot the matched data

#************end of user block************
#initalize the object
discharge_obj=data_prep.DichargePerp()

dict_=discharge_obj.time_series_full_pipeline(discharge,suffix_list,time_std_key,time_std=time_std,\
                                        custom_time_std=custom_time_std,\
                                        Ip_window_size=Ip_window_size, \
                                        Ip_std_threshold=Ip_std_threshold, \
                                        plot_Ip=plot_Ip, norm_mode=norm_mode, \
                                        interp_suffix=interp_suffix,  \
                                        interp_mode=interp_mode, \
                                        time_matching_mode=time_matching_mode, \
                                        left_window=left_window, \
                                        right_window=right_window, \
                                        time_matching_padding=time_matching_padding, \
                                        plot_matched_data=plot_matched_data)
