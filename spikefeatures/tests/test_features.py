import spikefeatures as sf
from .utils import generate_templates_with_gt
import numpy as np


def test_calculate_features():
    fs = 30000.
    neg_amps = [-10, -30, -50, -100]
    pos_amps = [10, 20, 30, 40]
    neg_pos_ratios = [0.2, 0.5, 0.7]
    npoints = 200

    for neg in neg_amps:
        for pos in pos_amps:
            for ratio in neg_pos_ratios:
                waveform, ptv_gt, hw_gt, \
                ptr_gt, rp_gt, rs_gt = generate_templates_with_gt(fs, npoints, neg, pos, ratio)
                features = sf.calculate_features(waveform, fs)
                assert np.isclose(features['peak_to_valley'], ptv_gt)
                assert np.isclose(features['halfwidth'], hw_gt)
                assert np.isclose(features['peak_trough_ratio'], ptr_gt)
                assert np.isclose(features['repolarization_slope'], rp_gt)
                assert np.isclose(features['recovery_slope'], rs_gt)


def test_peak_to_valley():
    fs = 30000.
    neg_amps = [-10, -30, -50, -100]
    pos_amps = [10, 20, 30, 40]
    neg_pos_ratios = [0.2, 0.5, 0.7]
    npoints = 200

    for neg in neg_amps:
        for pos in pos_amps:
            for ratio in neg_pos_ratios:
                waveform, ptv_gt, hw_gt, \
                ptr_gt, rp_gt, rs_gt = generate_templates_with_gt(fs, npoints, neg, pos, ratio)
                ptv = np.squeeze(sf.peak_to_valley(waveform, fs))
                assert np.isclose(ptv, ptv_gt)


def test_halfwidth():
    fs = 30000.
    neg_amps = [-10, -30, -50, -100]
    pos_amps = [10, 20, 30, 40]
    neg_pos_ratios = [0.2, 0.5, 0.7]
    npoints = 200

    for neg in neg_amps:
        for pos in pos_amps:
            for ratio in neg_pos_ratios:
                waveform, ptv_gt, hw_gt, \
                ptr_gt, rp_gt, rs_gt = generate_templates_with_gt(fs, npoints, neg, pos, ratio)
                hw = np.squeeze(sf.halfwidth(waveform, fs))
                assert np.isclose(hw, hw_gt)


def test_peak_trough_ratio():
    fs = 30000.
    neg_amps = [-10, -30, -50, -100]
    pos_amps = [10, 20, 30, 40]
    neg_pos_ratios = [0.2, 0.5, 0.7]
    npoints = 200

    for neg in neg_amps:
        for pos in pos_amps:
            for ratio in neg_pos_ratios:
                waveform, ptv_gt, hw_gt, \
                ptr_gt, rp_gt, rs_gt = generate_templates_with_gt(fs, npoints, neg, pos, ratio)
                ptr = np.squeeze(sf.peak_trough_ratio(waveform))
                assert np.isclose(ptr, ptr_gt)


def test_repolarization_slope():
    fs = 30000.
    neg_amps = [-10, -30, -50, -100]
    pos_amps = [10, 20, 30, 40]
    neg_pos_ratios = [0.2, 0.5, 0.7]
    npoints = 200

    for neg in neg_amps:
        for pos in pos_amps:
            for ratio in neg_pos_ratios:
                waveform, ptv_gt, hw_gt, \
                ptr_gt, rp_gt, rs_gt = generate_templates_with_gt(fs, npoints, neg, pos, ratio)
                rp = np.squeeze(sf.repolarization_slope(waveform, fs))
                assert np.isclose(rp, rp_gt)


def test_recovery_slope():
    fs = 30000.
    neg_amps = [-10, -30, -50, -100]
    pos_amps = [10, 20, 30, 40]
    neg_pos_ratios = [0.2, 0.5, 0.7]
    npoints = 200

    for neg in neg_amps:
        for pos in pos_amps:
            for ratio in neg_pos_ratios:
                waveform, ptv_gt, hw_gt, \
                ptr_gt, rp_gt, rs_gt = generate_templates_with_gt(fs, npoints, neg, pos, ratio)
                rs = np.squeeze(sf.recovery_slope(waveform, fs, window=0.7))
                assert np.isclose(rs, rs_gt)
