import numpy as np


def generate_templates_with_gt(fs, n_points, neg_amp=-50, pos_amp=20, neg_pos_ratio=0.3):
    times_idxs = np.arange(n_points).astype('int64')
    neg_idxs = times_idxs[:int(n_points * neg_pos_ratio)]
    pos_idxs = times_idxs[int(n_points * neg_pos_ratio):]
    vals = np.zeros_like(times_idxs).astype('float')

    neg_idx_mid = len(neg_idxs) // 2
    slope_neg = neg_amp / neg_idx_mid
    vals[neg_idxs[:neg_idx_mid]] = slope_neg * (neg_idxs[:neg_idx_mid] - neg_idxs[0])
    vals[neg_idxs[neg_idx_mid:]] = -slope_neg * (neg_idxs[neg_idx_mid:] - neg_idxs[neg_idx_mid]) + neg_amp

    pos_idx_mid = len(pos_idxs) // 2
    slope_pos = pos_amp / pos_idx_mid
    vals[pos_idxs[:pos_idx_mid]] = slope_pos * (pos_idxs[:pos_idx_mid] - pos_idxs[0])
    vals[pos_idxs[pos_idx_mid:]] = -slope_pos * (pos_idxs[pos_idx_mid:] - pos_idxs[pos_idx_mid]) + pos_amp

    peak_to_valley = (pos_idxs[pos_idx_mid] - neg_idxs[neg_idx_mid]) / fs
    halfwidth = (len(neg_idxs) // 2) / fs
    peak_to_through_ratio = pos_amp / neg_amp
    repolarization_slope = -slope_neg * fs
    recovery_slope = -slope_pos * fs
    waveform = vals[np.newaxis]

    return waveform, peak_to_valley, halfwidth, peak_to_through_ratio, repolarization_slope, recovery_slope