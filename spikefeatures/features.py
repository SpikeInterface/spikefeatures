"""
Functions based on
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/mean_waveforms/waveform_metrics.py
22/04/2020
"""

import numpy as np
from scipy.stats import linregress
import pandas as pd

all_1D_features = ['peak_to_valley', 'halfwidth', 'peak_trough_ratio',
                   'repolarization_slope', 'recovery_slope']


def calculate_features(waveforms, sampling_frequency, feature_names=None,
                       recovery_slope_window=0.7):
    """ Calculate features for all waveforms

    Parameters
    ----------
    waveforms  : numpy.ndarray (num_waveforms x num_samples)
        waveforms to compute features for
    sampling_frequency  : float
        rate at which the waveforms are sampled (Hz)
    feature_names : list or None (if None, compute all)
        features to compute
    recovery_slope_window : float
        windowlength in ms after peak wherein recovery slope is computed

    Returns
    -------
    metrics : pandas.DataFrame  (num_waveforms x num_metrics)
        one column for each metric
        one row per waveforms

    """
    metrics = pd.DataFrame()

    if feature_names is None:
        feature_names = all_1D_features
    else:
        for name in feature_names:
            assert name in all_1D_features, f'{name} not in {all_1D_features}'

    if 'peak_to_valley' in feature_names:
        metrics['peak_to_valley'] = peak_to_valley(waveforms=waveforms,
                                                   sampling_frequency=sampling_frequency)
    if 'peak_trough_ratio' in feature_names:
        metrics['peak_trough_ratio'] = peak_trough_ratio(waveforms=waveforms)

    if 'halfwidth' in feature_names:
        metrics['halfwidth'] = halfwidth(waveforms=waveforms,
                                         sampling_frequency=sampling_frequency)

    if 'repolarization_slope' in feature_names:
        metrics['repolarization_slope'] = repolarization_slope(
            waveforms=waveforms,
            sampling_frequency=sampling_frequency,
        )

    if 'recovery_slope' in feature_names:
        metrics['recovery_slope'] = recovery_slope(
            waveforms=waveforms,
            sampling_frequency=sampling_frequency,
            window=recovery_slope_window,
        )
    return metrics


def peak_to_valley(waveforms, sampling_frequency):
    """ time in s between throug and peak

    :param waveforms: numpy.ndarray (num_waveforms x num_samples)
    :param sampling_frequency: rate at which the waveforms are sampled (Hz)
    :return: peak_to_valley; np.ndarray (num_waveforms)
    """
    trough_idx, peak_idx = _get_trough_and_peak_idx(waveforms)
    ptv = (peak_idx - trough_idx) * (1/sampling_frequency)
    return ptv


def peak_trough_ratio(waveforms):
    """ waveform peak height / trough depth

    :param waveforms: numpy.ndarray (num_waveforms x num_samples)
    :return: peak_trough_ratio; np.ndarray (num_waveforms)
    """
    trough_idx, peak_idx = _get_trough_and_peak_idx(waveforms)
    ptratio = np.empty(trough_idx.shape[0])
    for i, (thidx, pkidx) in enumerate(zip(trough_idx, peak_idx)):
        ptratio[i] = waveforms[i, pkidx] / waveforms[i, thidx]
    return ptratio


def halfwidth(waveforms, sampling_frequency, return_idx=False):
    """ Width of the waveform peak at its half ampltude heigth

    :param return_idx: bool, if true return halfwidth, index of crossing threhold pre peak, index of crossing post peak
    :param waveforms: numpy.ndarray (num_waveforms x num_samples)
    :param sampling_frequency: rate at which the waveforms are sampled (Hz)
    :return: peak_to_valley; np.ndarray (num_waveforms)
    """
    trough_idx, peak_idx = _get_trough_and_peak_idx(waveforms)
    hw = np.empty(waveforms.shape[0])
    cross_pre_pk = np.empty(waveforms.shape[0], dtype=np.int)
    cross_post_pk = np.empty(waveforms.shape[0], dtype=np.int)

    for i in range(waveforms.shape[0]):
        peak_val = waveforms[i, peak_idx[i]]
        threshold = 0.5 * peak_val  # threshold is half of peak heigth (assuming baseline is 0)

        cpre_idx = np.where(waveforms[i, :peak_idx[i]] < threshold)[0]
        cpost_idx = np.where(waveforms[i, peak_idx[i]:] < threshold)[0]

        if len(cpre_idx) == 0 or len(cpost_idx) == 0:
            continue

        cross_pre_pk[i] = cpre_idx[-1] + 1  # last occurence of waveform lower than thr, before peak
        cross_post_pk[i] = cpost_idx[0] - 1 + peak_idx[i]  # first occurence of waveform lower than peak, after peak
        hw[i] = (cross_post_pk[i] - cross_pre_pk[i] + peak_idx[i]) * (1/sampling_frequency)

    if not return_idx:
        return hw
    else:
        return hw, cross_pre_pk, cross_post_pk


def repolarization_slope(waveforms, sampling_frequency, return_idx=False):
    """ waveform slope between trough and first crossing of baseline thereafter

    :param return_idx: bool, if true return halfwidth, index of crossing baseline after trough
    :param waveforms: numpy.ndarray (num_waveforms x num_samples)
    :param sampling_frequency: rate at which the waveforms are sampled (Hz)
    :return: repolarization slope; np.ndarray (num_waveforms)
    """
    trough_idx, peak_idx = _get_trough_and_peak_idx(waveforms)

    rslope = np.empty(waveforms.shape[0])
    return_to_base_idx = np.empty(waveforms.shape[0], dtype=np.int)

    time = np.arange(0, waveforms.shape[1]) * (1/sampling_frequency)  # in s
    for i in range(waveforms.shape[0]):
        rtrn_idx = np.where(waveforms[i, trough_idx[i]:] >= 0)[0]
        if len(rtrn_idx) == 0:
            continue

        return_to_base_idx[i] = rtrn_idx[0] + trough_idx[i]  # first time after  trough, where waveform is at baseline
        rslope[i] = linregress(time[trough_idx[i]:return_to_base_idx[i]],
                               waveforms[i, trough_idx[i]: return_to_base_idx[i]])[0]

    if not return_idx:
        return rslope
    else:
        return rslope, return_to_base_idx


def recovery_slope(waveforms, sampling_frequency, window):
    """ slope of waveform after peak, within specified window

    :param waveforms: numpy.ndarray (num_waveforms x num_samples)
    :param sampling_frequency: rate at which the waveforms are sampled (Hz)
    :param window: windowlength in ms after peak wherein recovery slope is computed
    :return: peak_to_valley; np.ndarray (num_waveforms)
    """
    _, peak_idx = _get_trough_and_peak_idx(waveforms)
    rslope = np.empty(waveforms.shape[0])
    time = np.arange(0, waveforms.shape[1]) * (1/sampling_frequency)  # in s

    for i in range(waveforms.shape[0]):
        max_idx = int(peak_idx[i] + ((window/1000)*sampling_frequency))
        max_idx = np.min([max_idx, waveforms.shape[1]])
        slope = _get_slope(time[peak_idx[i]:max_idx], waveforms[i, peak_idx[i]:max_idx])
        rslope[i] = slope[0]
    return rslope


def _get_slope(x, y):
    """ slope of x and y data

    :param x: np.ndarray (n_samples)
    :param y: np.ndarray (n_samples)
    :return: scipy.linregress output (slope, intercept, rvalue, pvalue, stderr)
    """
    slope = linregress(x, y)
    return slope


def _get_trough_and_peak_idx(waveform):
    """ detect trough and peak in waveform, peak always after trough

    :param waveform: np.ndarray (num_samples)
    :return: index_of_trough, index_of_peak
    """
    trough_idx = np.argmin(waveform, axis=1)
    peak_idx = np.empty(trough_idx.shape, dtype=int)
    for i, tridx in enumerate(trough_idx):
        if tridx == waveform.shape[1]-1:
            continue
        idx = np.argmax(waveform[i, tridx:])
        peak_idx[i] = idx + tridx
    return trough_idx, peak_idx


