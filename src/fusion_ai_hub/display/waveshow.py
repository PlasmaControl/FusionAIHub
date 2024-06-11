@staticmethod
def time_serie_plot(dict):
    plt.clf()
    if dict['zdata'][:].shape == 1:
        plt.plot(dict['xdata'][:],dict['zdata'][:])
    else:
        plt.plot(dict['xdata'][:],dict['zdata'][:].T)
    plt.xlabel('Time (ms)')
    plt.show()