import numpy as np


def norm_data(data,avg_,std_,mode='all'):
    avg_=np.array(avg_)
    std_=np.array(std_)
    if mode == 'all':
        std_all=(np.mean(std_**2))**0.5
        avg_all=np.mean(avg_)
    elif mode=='std_all_avg_individual':
        std_all=(np.mean(std_**2))**0.5
        avg_all=np.expand_dims(avg_,axis=1)

    elif mode=='individual':
        std_all=np.expand_dims(avg_,axis=1)
        avg_all=np.expand_dims(avg_,axis=1)


    data_norm=(data-avg_all)/std_all

    return data_norm
