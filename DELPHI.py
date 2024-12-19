import importlib
import json
import math
import time
from collections import Counter
from math import sqrt

import numpy as np
import pandas as pd
from graphviz import Digraph

import metrics_utils
import utils
from utils import find_two_largest, Max_S, Max_C, Min_C, Rank_S

class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(MyJSONEncoder, self).default(obj)


class DecisionTree:
    def __init__(self, metric, max_depth=4, min_samples_split=2):
        self.algorithm_list = ['DAWA', 'Privelet', 'MWEM', 'HB', 'DPcube', 'identity', 'AHP']
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.error_cache = {}
        self.tree = None
        self.metric = metric

        # 读取 error_data
        with open('DELPHI_data/DELPHI_data_1.json', 'r') as f:
            self.error_data = json.load(f)[metric]
            # {
            # 评估指标：
            # {
            # 隐私预算：
            # {
            # 算法：【】

    def fit(self, X, raw_data):
        # 开始在根节点调用递归构建函数
        start = time.time()
        self.tree = self._build_tree(X, raw_data, np.array(range(len(raw_data))), 0, None)
        end = time.time()
        print(f"计算的时间为 {end-start}s")
    def _build_tree(self, X, raw_data, data_indices, depth, epsilon_division):
        num_samples, num_features = X.shape
        features = X.columns.values
        # 检查停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or num_samples < self.min_samples_split:
            [leaf_best_algorithm, leaf_FoBA] = self._fraction_of_best_algorithm(raw_data, data_indices, epsilon_division)
            return {
                'best_algorithm': leaf_best_algorithm,
                'FoBA': leaf_FoBA,
                'num': len(X)
            }

        # 初始化最优特征、分割点和最小不纯度
        best_feature, best_split, min_impurity = None, None, float('inf')

        # 遍历每个特征寻找最优分割
        final_best_algorithm = None
        final_FoBA = None
        if epsilon_division is None:
            for epsilon in np.arange(0.02, 1.01, 0.01):
                best_algorithm, FoBA, impurity = self._weighted_impurity(raw_data, data_indices, raw_data,
                                                                         data_indices, raw_data, data_indices, epsilon)
                # 如果不纯度更低，则更新最优分割点
                if impurity < min_impurity:
                    final_FoBA = FoBA
                    final_best_algorithm = best_algorithm
                    min_impurity = impurity
                    best_feature, best_split = 'epsilon', epsilon
        else:
            for featureIdx in range(num_features):
                feature = features[featureIdx]
                values = np.unique(X[feature])
                for split in values:
                    # 计算该分割点的加权节点不纯度
                    left_data = raw_data[X[feature] <= split]
                    left_indices = data_indices[X[feature] <= split]
                    right_data = raw_data[X[feature] > split]
                    right_indices = data_indices[X[feature] > split]
                    if len(right_data) == 0:
                        continue
                    best_algorithm, FoBA, impurity = self._weighted_impurity(raw_data, data_indices, left_data, left_indices, right_data, right_indices, epsilon_division)

                    # 如果不纯度更低，则更新最优分割点
                    if impurity < min_impurity:
                        final_FoBA = FoBA
                        final_best_algorithm = best_algorithm
                        min_impurity = impurity
                        best_feature, best_split = feature, split

        # 如果未找到最优分割，创建叶节点
        if best_feature is None:
            return 1

        # 创建分割后的子树
        # if epsilon_division is None:
        #     left_child = self._build_tree(X, raw_data, data_indices, depth + 1, [0.01, round(best_split-0.01, 2)])
        #     right_child = self._build_tree(X, raw_data, data_indices, depth + 1, [best_split, 1.00])
        # else:
        #     left_indices_bool = X[best_feature] <= best_split
        #     left_indices = data_indices[left_indices_bool]
        #     right_indices_bool = X[best_feature] > best_split
        #     right_indices = data_indices[right_indices_bool]
        #     left_child = self._build_tree(X[left_indices_bool], raw_data[left_indices_bool], left_indices, depth + 1, epsilon_division)
        #     right_child = self._build_tree(X[right_indices_bool], raw_data[right_indices_bool], right_indices, depth + 1, epsilon_division)

        # 返回当前节点的字典结构
        return {
            'best_algorithm': final_best_algorithm,
            'FoBA': final_FoBA,
            "feature": best_feature,
            "split": best_split,
            'num': len(raw_data),
            # "left": left_child,
            # "right": right_child
        }

    def _weighted_impurity(self, raw_data, data_indices, left_data, left_indices, right_data, right_indices, epsilon_division):
        if isinstance(epsilon_division, list):
            cur_scope = left_scope = right_scope = epsilon_division
        else:
            cur_scope = [0.01, 1.00]
            left_scope = [0.01, round(epsilon_division-0.01, 2)]
            right_scope = [epsilon_division, 1.00]
        [best_algorithm, FoBA] = self._fraction_of_best_algorithm(raw_data, data_indices, cur_scope)
        NIV = 1 - FoBA * FoBA

        [_, left_FoBA] = self._fraction_of_best_algorithm(left_data, left_indices, left_scope)
        left_NIV = 1 - left_FoBA * left_FoBA

        [_, right_FoBA] = self._fraction_of_best_algorithm(right_data, right_indices, right_scope)
        right_NIV = 1 - right_FoBA * right_FoBA

        return best_algorithm, FoBA, NIV - (left_NIV + right_NIV)

    def _fraction_of_best_algorithm(self, dataset, indices, epsilon_scope):
        best_algorithm_list = []
        m = 0
        for epsilon in np.arange(epsilon_scope[0], round(epsilon_scope[1]+0.01, 2), 0.01):
            if epsilon > 1.00:
                continue
            m += 1
            epsilon_str = f'{epsilon:.2f}'
            cur_error_data = self.error_data[epsilon_str]
            for i in indices:
                # 找到对应索引处的最小值对应的key
                min_key = min(cur_error_data, key=lambda k: cur_error_data[k][i])
                best_algorithm_list.append(min_key)
        best_algorithm_counter = Counter(best_algorithm_list)
        most_best_algorithm_fraction = best_algorithm_counter.most_common(1)[0][1] / len(best_algorithm_list)
        most_best_algorithm = best_algorithm_counter.most_common(1)[0][0]
        return [most_best_algorithm, most_best_algorithm_fraction]



    def _error_calculation(self, data, noise_data, metric):
        metric_func = getattr(metrics_utils, metric)
        error = metric_func(data, noise_data)
        return error


    def _node_impurity(self, y):
        # 使用基尼不纯度或熵来计算节点不纯度，简单起见，这里使用基尼不纯度
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        gini = 1 - np.sum(p ** 2)
        return gini

    def _create_leaf(self, y):
        pass

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, tree):
        # 递归遍历树以预测样本类别
        if not isinstance(tree, dict):
            return tree
        feature, split = tree["feature"], tree["split"]
        if x[feature] <= split:
            return self._predict_sample(x, tree["left"])
        else:
            return self._predict_sample(x, tree["right"])

    def visualize_tree(self, filename="tree"):
        # 创建Graphviz的有向图对象
        dot = Digraph()
        self._add_nodes_edges(self.tree, dot)
        # 将图保存为文件
        dot.render(filename, format="png", cleanup=True)
        print(f"决策树图已保存为 {filename}.png")

    def _add_nodes_edges(self, node, dot, parent=None, edge_label=""):
        # 如果节点是叶节点，标记叶节点
        if "feature" not in node:
            node_label = f"{node['best_algorithm']} {int(node['FoBA']*node['num'])} / {node['num']}"
            dot.node(str(id(node)), label=node_label, shape="box", style="filled", color="lightgrey")
            if parent:
                dot.edge(parent, str(id(node)), label=edge_label)
            return

        # 创建分割特征和阈值的节点
        node_label = f"{node['feature']} <= {node['split']}"
        dot.node(str(id(node)), label=node_label, shape="ellipse", style="filled", color="lightblue")

        # 连接到父节点
        if parent:
            dot.edge(parent, str(id(node)), label=edge_label)

        # 递归添加左子树和右子树
        self._add_nodes_edges(node["left"], dot, parent=str(id(node)), edge_label="True")
        self._add_nodes_edges(node["right"], dot, parent=str(id(node)), edge_label="False")


data_index = 1
metrics = ['Max_C', 'Min_C', 'Rank_C', 'Outliers_C',
           'MSE_S', 'MAE_S', 'Max_S', 'Anomaly_S',  # 'Cluster_S',
           'Distribution_Dispersion_S',
           'EarthMoversDistance_S', ]

for data_index in range(1, 11):
    # 获取特征向量
    for metric in metrics:
        histogram_feature = pd.read_csv(f"../data/generated_lists_{data_index}_features.csv")
        # 获取直方图数据
        with open(f'../data/generated_lists_{data_index}.json') as f:
            histograms_data = json.load(f)
            histograms_data = np.array(histograms_data, dtype=object)
        start = time.time()
        dt = DecisionTree(metric)

        dt.fit(histogram_feature, histograms_data)

        print(dt.tree)

        dt.visualize_tree(f"output/decision_tree_{data_index}_{metric}.png")
        with open(f'output/decision_tree_{data_index}_{metric}.json', 'w') as f:
            json.dump(dt.tree, f, cls=MyJSONEncoder)

        end = time.time()

        print(f'runtime: {end-start} s')



