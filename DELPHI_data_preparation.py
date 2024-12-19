import json
import time
import numpy as np
import pandas as pd

import metrics_utils
import utils

class TrainDataGenerator(object):
    def __init__(self, data_index, max_depth=4, min_samples_split=2):
        self.algorithm_list = ['DAWA', 'Privelet', 'MWEM', 'HB', 'DPcube', 'identity', 'AHP', 'PHP', 'EFPA']
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.error_cache = {}
        self.tree = None
        self.error_data = {}
        self.data_index = data_index
        self.metrics = ['Max_C', 'Min_C', 'Rank_C', 'Outliers_C',
           'MSE_S', 'MAE_S', 'Max_S', 'Anomaly_S',  # 'Cluster_S',
           'Distribution_Dispersion_S',
           'EarthMoversDistance_S', ]

    def fit(self, X, dataset):
        algorithm_list = self.algorithm_list
        t = 20
        best_algorithm_list = []
        best_error_list = []
        best_algorithm = None
        module_list = []
        metrics = self.metrics
        for metric in metrics:
            self.error_data[metric] = {}
            for epsilon in np.arange(0.01, 1.01, 0.01):
                epsilon_str = f'{epsilon:.2f}'
                self.error_data[metric][epsilon_str] = {}
                for algorithm in algorithm_list:
                    self.error_data[metric][epsilon_str][algorithm] = [0] * 495


        for epsilon in np.arange(0.01, 1.01, 0.01):
            epsilon_str = f'{epsilon:.2f}'
            for algorithm in algorithm_list:
                for dataIdx, data in enumerate(dataset):
                    for _ in range(t):
                        func = getattr(utils, 'run' + algorithm)
                        data = np.array(data)
                        data = data.astype(int)
                        noise_data = func(data, epsilon)
                        for metric in metrics:
                            self.error_data[metric][epsilon_str][algorithm][dataIdx] += self._error_calculation(data, noise_data, metric) / t

            print(f'epsilon {epsilon_str} done')
        with open(f'DELPHI_data/DELPHI_data_{self.data_index}.json', 'w', encoding='utf-8') as jsonFile:
            json.dump(self.error_data, jsonFile, ensure_ascii=False)
    def _error_calculation(self, data, noise_data, metric):
        metric_func = getattr(metrics_utils, metric)
        error = metric_func(data, noise_data)
        return error



data_index = 1

for data_index in range(1, 11):
    # 获取特征向量
    histogram_feature = pd.read_csv(f"data/generated_lists_{data_index}_features.csv")

    # 获取直方图数据
    with open(f'data/generated_lists_{data_index}.json') as f:
        histograms_data = json.load(f)
        histograms_data = np.array(histograms_data, dtype=object)
    start = time.time()
    dt = TrainDataGenerator(data_index)

    dt.fit(histogram_feature, histograms_data)

    end = time.time()

    print(f'runtime: {end-start} s')

