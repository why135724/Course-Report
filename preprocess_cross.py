"""
eeg data preprocess and feature extractor
quan xueliang, 2021.06

整体流程:
1) 使用 load_eeg_data 读取 EDF 原始数据 (约 300 Hz, 多通道).
2) 对前 24 个 EEG 通道做 data_preprocess (0.5-49 Hz 带通滤波).
3) 利用第 25 个通道的触发信号切出 sad / happy 时间段.
4) sliding_window_sample 对各时间段按 1s 窗 + 1s 步长切窗.
5) 对每个窗口调用 de_feature_extractor 提取 5 个频带 × 24 通道的 DE 特征,
   展平成长度为 5*24 的特征向量, 分别保存为 sad_de_features.npy / happy_de_features.npy.
6) 后续在 SVM.py 中读取这两个特征文件, 拼成样本矩阵 X 和标签 y, 划分 train/test,
   用 StandardScaler + SVM (GridSearchCV) 做二分类建模与评估.
"""
import numpy as np
import scipy
import scipy.io
import scipy.fftpack
import scipy.linalg
import math
from scipy.signal import butter, lfilter
from scipy.signal.windows import hann
from typing import List, Tuple
import mne
import os

def data_preprocess(raw_data):
    """
    :param raw_data: [channels, timpoints], 300 Hz
    :return: filtered data, [channels, timpoints]
    """

    # 不再降采样，直接用 300 Hz 采样率
    eeg_rawdata = raw_data

    # 0.5-49 Hz 带通滤波
    low_cut, high_cut, fs, order = 0.5, 49.0, 300, 4
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    eeg_rawdata = lfilter(b, a, eeg_rawdata, axis=1)

    return eeg_rawdata


def psd_feature_extractor(raw_data):
    """
    :param raw_data: filtered data, [channels, timepoints], 300 Hz
    :return: psd feature, shape = [n_windows, n_bands, n_channels]
    """
    # 采样率和频带设置与滤波一致
    sample_freq = 300
    stft_n = 512
    window_size = 1  # second
    freq_bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 49)]

    n_channels, n_samples = raw_data.shape
    point_per_window = int(sample_freq * window_size)
    window_num = n_samples // point_per_window
    if window_num == 0:
        raise ValueError("Not enough samples for 1-second windowing at 300 Hz.")

    psd_feature = np.zeros((window_num, len(freq_bands), n_channels), dtype=np.float32)

    for window_index in range(window_num):
        start_index = point_per_window * window_index
        end_index = point_per_window * (window_index + 1)
        window_data = raw_data[:, start_index:end_index]

        # 汉宁窗
        h = hann(point_per_window)
        hdata = window_data * h  # 广播到 [channels, points]

        # FFT
        fft_data = np.fft.fft(hdata, n=stft_n, axis=1)
        # 只保留正频部分
        energy_graph = np.abs(fft_data[:, : stft_n // 2])  # [channels, freqs]

        for band_index, band in enumerate(freq_bands):
            band_ave_psd = _get_average_psd(energy_graph, band, sample_freq, stft_n)
            psd_feature[window_index, band_index, :] = band_ave_psd.astype(np.float32)

    return psd_feature


def _get_average_psd(energy_graph, freq_band, sample_freq, stft_n=512):
    """
    :param energy_graph: [channels, freq_bins] = |FFT| (magnitude)
    :param freq_band: (f_low, f_high)
    """
    f_low, f_high = freq_band
    # 这里使用与 FFT 一致的频率分辨率
    freq_res = sample_freq / stft_n
    start_index = int(np.floor(f_low / freq_res))
    end_index = int(np.floor(f_high / freq_res))
    start_index = max(start_index, 0)
    end_index = min(end_index, energy_graph.shape[1] - 1)
    if end_index <= start_index:
        # 避免空频带
        end_index = start_index + 1
    # 均值功率谱密度（平方）
    ave_psd = np.mean(energy_graph[:, start_index:end_index + 1] ** 2, axis=1)
    return ave_psd


def de_feature_extractor(EEGdata):
    """
    Differential Entropy(DE) 特征提取
    :param EEGdata: filtered data, [channels, timepoints], 300 Hz
    :return: DE feature, shape = [n_windows, n_bands, n_channels]
    """
    assert EEGdata.shape[0] == 24

    fs = 300  # 与实际 EDF 采样率一致
    channels = EEGdata.shape[0]
    # 与 PSD 相同的频带
    fStart = [1, 4, 8, 14, 31]
    fEnd = [4, 8, 14, 31, 49]
    window = 1  # second
    stftn = 512  # FFT 点数

    window_points = fs * window  # 每个窗口采样点数
    n, m = EEGdata.shape
    n_windows = m // window_points
    if n_windows == 0:
        raise ValueError("Not enough samples for 1-second windowing at 300 Hz.")

    # 频率向量,只取非负频率
    freqs = np.fft.rfftfreq(stftn, d=1.0 / fs)  # [0, fs/2]

    # 预计算各频带对应的索引
    band_indices = []
    for fl, fh in zip(fStart, fEnd):
        idx = np.where((freqs >= fl) & (freqs <= fh))[0]
        if len(idx) == 0:
            # 避免空索引
            idx = np.array([np.argmin(np.abs(freqs - fl))])
        band_indices.append(idx)

    # 输出: [n_windows, n_bands, n_channels]
    de_feature = np.zeros((n_windows, len(fStart), channels), dtype=np.float32)

    for w in range(n_windows):
        start = w * window_points
        end = (w + 1) * window_points
        data_now = EEGdata[:, start:end]  # [channels, window_points]

        # 加窗(可选)
        h = hann(window_points)
        data_now_win = data_now * h

        # 对每个通道做 FFT
        fft_data = np.fft.rfft(data_now_win, n=stftn, axis=1)  # [channels, stftn//2+1]
        power_spectrum = np.abs(fft_data) ** 2  # 功率

        for b_idx, idx in enumerate(band_indices):
            # 该频带上的平均功率
            band_power = np.mean(power_spectrum[:, idx], axis=1)  # [channels]
            # Differential Entropy: log(power)  (log10 更常见,避免 log(0))
            de_vals = np.log10(band_power + 1e-9)
            de_feature[w, b_idx, :] = de_vals.astype(np.float32)

    return de_feature


def load_eeg_data(edf_path: str) -> Tuple[np.ndarray, dict]:
    """
    加载 .dsi 和 .edf 文件,返回 EEG 脑电数据

    :param dsi_path: .dsi 文件路径 (通常包含标签等元信息)
    :param edf_path: .edf 文件路径 (包含原始 EEG 数据)
    :return: (eeg_data, metadata)
        eeg_data: [channels, timepoints] 形状的 numpy 数组
        metadata: 包含采样率、通道名等信息的字典
    """

    # 加载 .edf 文件 (EDF 格式)
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        raise ValueError(f"Failed to load .edf file: {e}")

    # 获取 EEG 数据 [channels, timepoints]
    eeg_data = raw.get_data()

    # 获取元信息
    metadata = {
        'sfreq': raw.info['sfreq'],  # 采样率
        'ch_names': raw.info['ch_names'],  # 通道名
        'n_channels': len(raw.info['ch_names']),
        'n_samples': eeg_data.shape[1],
    }

    return eeg_data, metadata


import matplotlib.pyplot as plt


# 使用示例
def sliding_window_sample(data: np.ndarray, window_size: int, stride: int, frequency: int) -> List[np.ndarray]:
    """
    使用滑动窗口对数据进行采样

    :param data: EEG数据 [channels, timepoints]
    :param window_size: 窗口大小(秒)
    :param stride: 滑动步长(秒)
    :param frequency: 采样频率(Hz)
    :return: 样本列表,每个样本形状为 [channels, window_points]
    """
    n_channels, n_timepoints = data.shape
    window_points = window_size * frequency
    stride_points = stride * frequency

    samples = []
    start = 0

    while start + window_points <= n_timepoints:
        sample = data[:, start:start + window_points]
        samples.append(sample)
        start += stride_points

    return samples


def CAR_process_and_save(sad_sample, happy_sample,subject_id):
    """
    对输入的脑电窗口做共同平均参考(CAR)并保存.
    输入:
        sad_sample: list of [channels, timepoints]
        happy_sample: list of [channels, timepoints]
    输出:
        保存为:
            ./processed_data/sad_car_windows.npy   # shape: [n_sad_windows, channels, timepoints]
            ./processed_data/happy_car_windows.npy # shape: [n_happy_windows, channels, timepoints]
    """
    # sad 窗口 CAR
    sad_car_list = []
    for sample in sad_sample:
        # sample: [channels, timepoints]
        mean_ref = np.mean(sample, axis=0, keepdims=True)  # [1, timepoints]
        car_sample = sample - mean_ref  # [channels, timepoints]
        sad_car_list.append(car_sample)
    sad_car_windows = np.stack(sad_car_list, axis=0)  # [n_sad, channels, timepoints]
    #np.save("./processed_data/sad_car_2D.npy", sad_car_windows)
    output_dir = "./processed_data"
    sad_car_2D_path = os.path.join(output_dir, f"subject_{subject_id}_sad_car_2D.npy")
    np.save(sad_car_2D_path, sad_car_windows)


    sad_car_features = np.array(sad_car_windows).reshape(sad_car_windows.shape[0], -1)
    # happy 窗口 CAR
    happy_car_list = []
    for sample in happy_sample:
        mean_ref = np.mean(sample, axis=0, keepdims=True)
        car_sample = sample - mean_ref
        happy_car_list.append(car_sample)

    happy_car_windows = np.stack(happy_car_list, axis=0)  # [n_happy, channels, timepoints]
    output_dir = "./processed_data"
    happy_car_2D_path = os.path.join(output_dir, f"subject_{subject_id}_happy_car_2D.npy")
    np.save(happy_car_2D_path, happy_car_windows)

    print("CAR windows saved.")



if __name__ == "__main__":
    # 可以尝试不同的EDF文件

    windowsize = 1  # second
    stride = 1  # second
    frequency = 300  # Hz

    #edf_file = "dataset/S1_raw.edf"
    #edf_file = "dataset/S2_raw.edf"
    edf_file = "dataset/S3_raw.edf"

    eeg_data, metadata = load_eeg_data(edf_file)
    processed_data = data_preprocess(eeg_data[:24, :])
    triggers = eeg_data[24, :]
    start_indices = np.where(triggers == 5)[0]
    end_indices = start_indices + 54317
    # end_indices = np.where(triggers == 10)[0]

    # 这里根据触发信号切分出 sad / happy 段落
    # 注意：这里只是按照标签切分原始时间段，然后提取特征并保存，
    # 真正的训练 / 测试划分在 SVM.py 里完成，因此不会造成“训练用到了测试标签”的泄露。
    sad = processed_data[:, start_indices[0]:end_indices[0]]
    happy = processed_data[:, start_indices[1]:end_indices[1]]

    sad_sample = sliding_window_sample(sad, windowsize, stride, frequency)
    happy_sample = sliding_window_sample(happy, windowsize, stride, frequency)
    if edf_file == "dataset/KSYsession1_raw.edf":
        subject_id = "01"
    elif edf_file == "dataset/CZY251129_001_raw.edf":
        subject_id = "02"
    elif edf_file == "dataset/whb251129_001_raw.edf":
        subject_id = "03"
    CAR_process_and_save(sad_sample, happy_sample, subject_id)
    # 如需 PSD 特征，打开下一行