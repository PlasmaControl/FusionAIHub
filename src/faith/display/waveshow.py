from typing import Any, Optional, Union

import matplotlib.axes as mplaxes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path as MplPath
from matplotlib.util import _deprecate as Deprecated


def waveshow(
    y: np.ndarray,
    *,
    sr: float = 22050,
    max_points: int = 11025,
    axis: Optional[str] = "time",
    offset: float = 0.0,
    marker: Union[str, MplPath, MarkerStyle] = "",
    where: str = "post",
    label: Optional[str] = None,
    transpose: bool = False,
    ax: Optional[mplaxes.Axes] = None,
    x_axis: Optional[Union[str, Deprecated]] = None,
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
    See Also
    --------
    librosa.display.waveshow
    librosa.display.specshow
    """

    plt.clf()
    plt.plot(y)
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
