import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Any, Optional

from matplotlib.collections import QuadMesh

def specshow(data: np.ndarray,
    *,
    x_coords: Optional[np.ndarray] = None,
    y_coords: Optional[np.ndarray] = None,
    x_axis: Optional[str] = None,
    y_axis: Optional[str] = None,
    sr: float = 22050,
    hop_length: int = 512,
    n_fft: Optional[int] = None,
    win_length: Optional[int] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    auto_aspect=True,
    **kwargs: Any,
    ) -> QuadMesh:
    """
    Display a spectrogram/chromagram/cqt/etc.
    
    Parameters
    ----------
    data : np.ndarray [shape=(d, n)]
        The audio samples. Multichannel audio will be downmixed.
    x_coords : np.ndarray [shape=(n,)]
        The x-coordinates for the waveform. By default, these are
        assumed to be sample indices.
    y_coords : np.ndarray [shape=(n,)]
        The y-coordinates for the waveform. By default, these are
        assumed to be sample values.
    x_axis : str
        Label for the x-axis. By default, this is 'Time'.
    y_axis : str
        Label for the y-axis. By default, this is 'Amplitude'.
    sr : number > 0 [scalar]
        The sample rate of the data.
    hop_length : int > 0
        The number of samples between successive frames.
    n_fft : int > 0
        The number of samples per frame.
    win_length : int <= n_fft
        The number of samples in each STFT window.
    fmin : float > 0
        The frequency of the lowest spectrogram bin.
    fmax : float > 0
        The frequency of the highest spectrogram bin.
    auto_aspect : bool
        Automatically set the aspect ratio of the plot to match the
        spectrogram's aspect ratio.
    kwargs : additional keyword arguments
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the waveform and spectrogram plots.
    ax : matplotlib.axes.Axes
        The axis containing the waveform and spectrogram plots.
    """
    
    plt.clf()
    plt.imshow(data,aspect='auto',cmap='hot',
                extent=[x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]])
    plt.colorbar()
    plt.ylabel('kHz')
    plt.xlabel('ms')
    plt.gca().invert_yaxis()
    plt.show()

# @staticmethod
# def time_serie_plot(dict):
#     plt.clf()
#     if dict['zdata'][:].shape == 1:
#         plt.plot(dict['xdata'][:],dict['zdata'][:])
#     else:
#         plt.plot(dict['xdata'][:],dict['zdata'][:].T)
#     plt.xlabel('Time (ms)')
#     plt.show()

# @staticmethod
# def spectro_plot(freq, time, amp_f_t):
#     plt.clf()
#     plt.imshow(amp_f_t,aspect='auto',cmap='hot',
#                 extent=[time[0], time[-1], freq[-1], freq[0]])
#     plt.colorbar()
#     plt.ylabel('kHz')
#     plt.xlabel('ms')
#     plt.gca().invert_yaxis()
#     plt.show()