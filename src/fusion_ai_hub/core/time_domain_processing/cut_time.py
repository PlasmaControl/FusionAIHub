@staticmethod
def cut_time(time, data, t_min, t_max):
    t_indx_min=np.argmin(abs(np.array(time)-t_min))
    t_indx_max=np.argmin(abs(np.array(time)-t_max))

    return time[t_indx_min:t_indx_max], data[...,t_indx_min:t_indx_max]
