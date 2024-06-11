@staticmethod
def spectro_plot(freq, time, amp_f_t):
    plt.clf()
    plt.imshow(amp_f_t,aspect='auto',cmap='hot',
                extent=[time[0], time[-1], freq[-1], freq[0]])
    plt.colorbar()
    plt.ylabel('kHz')
    plt.xlabel('ms')
    plt.gca().invert_yaxis()
    plt.show()