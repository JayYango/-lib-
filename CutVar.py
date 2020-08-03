# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from y2s.chimerge import ChiMerge
from . import Refaat
from .IV import IV_CrossTable
from .Gini import Gini_CrossTable
from .Entropy import Entropy_CrossTable
from .Chi2 import Chi2_CrossTable
from .OddsRatio import OddsRatio_CrossTable
from .LikeRatio import LikeRatio_CrossTable

# 分割点分箱--连续变量
def CutVar_CutPoints(data, x_var, y_var=None, cut_points=None):
    intv_list = []
    n = len(cut_points)
    L = -np.inf    #负无穷大
    i = 0
    k = 0
    while i < n:
        U = cut_points[i]
        if len(data.loc[(data[x_var] >= L) & (data[x_var] < U), x_var]) > 0:
            k += 1
            intv_list.append([k, L, U])
            L = U
        i += 1
    if len(data.loc[data[x_var] >= L, x_var]) > 0:
        intv_list.append([k + 1, L, np.inf])
    else:
        intv_list[k - 1][2] = np.inf
    var_map = pd.DataFrame(intv_list, columns=['_bin', '_LL', '_UL'])   
    return var_map

# 等宽分箱--连续变量
def CutVar_EqW(data, x_var, y_var=None, n_bins=10):
    x_max = data[x_var].max()
    x_min = data[x_var].min()
    bin_size = (x_max - x_min) / n_bins
    intv_list = []
    L = -np.inf
    k = 0
    i = 1
    while i < n_bins:
        U = x_min + i * bin_size
        if len(data.loc[(data[x_var] >= L) & (data[x_var] < U), x_var]) > 0:
            k += 1
            intv_list.append([k, L, U])
            L = U
        i += 1
    if len(data.loc[data[x_var] >= L, x_var]) > 0:
        intv_list.append([k + 1, L, np.inf])
    else:
        intv_list[k - 1][1] = np.inf
    var_map = pd.DataFrame(intv_list, columns=['_bin', '_LL', '_UL'])   
    return var_map

# 等频分箱--连续变量
def CutVar_EqF(data, x_var, y_var=None, n_bins=10):
    cut_point = [-np.inf]
    for i in range(1, n_bins):
        cut_point.append(np.quantile(data[x_var], i / n_bins, interpolation='higher'))
    cut_point.append(np.inf)
    
    tmp = np.unique(cut_point)
    intv_list = []
    for i in range(len(tmp) - 1):
        intv_list.append([i + 1, tmp[i], tmp[i + 1]])
    
    var_map = pd.DataFrame(intv_list, columns=['_bin', '_LL', '_UL'])
    return var_map

# 按偏离均值程度分箱--连续变量
def CutVar_Means(data, x_var, y_var=None, cut_points=[0]):
    mu = data[x_var].mean()
    sigma = data[x_var].std()
    cuts = []
    for x in cut_points:
        cuts.append(mu + x * sigma)
    return CutVar_CutPoints(data, x_var, y_var=y_var, cut_points=cuts)

def CutVar_Percentiles(data, x_var, y_var=None, cut_points=[0.5]):
    cuts = []
    for x in cut_points:
        cuts.append(np.quantile(data[x_var], x, interpolation='higher'))

    return CutVar_CutPoints(data, x_var, y_var=y_var, cut_points=cuts)

# K均值分箱--连续变量
def CutVar_KMeans(data, x_var, y_var=None, n_bins=10):
    model = KMeans(n_clusters=n_bins)
    model = model.fit(data[[x_var]])
    cut_point = [-np.inf]
    centers = np.sort(model.cluster_centers_.ravel())
    for i in range(len(centers) - 1):
        cut_point.append((centers[i] + centers[i + 1]) / 2)
    cut_point.append(np.inf)
    
    tmp = np.unique(cut_point)
    intv_list = []
    for i in range(len(tmp) - 1):
        intv_list.append([i + 1, tmp[i], tmp[i + 1]])
    
    var_map = pd.DataFrame(intv_list, columns=['_bin', '_LL', '_UL'])
    return var_map

# 决策树分箱--连续变量
def CutVar_DecisionTree(data, x_var, y_var, max_depth=3, min_percent=0.05):
    n, r = data.shape
    clf = DecisionTreeClassifier(criterion='entropy',
                                 max_depth=max_depth,
                                 min_samples_leaf=int(n * min_percent))
    clf = clf.fit(data[x_var].values.reshape(-1, 1), data[y_var].values.reshape(-1, 1)) #reshape(-1, 1)改变行列数 -1为不指定，1为1列

    cut_point = clf.tree_.threshold[clf.tree_.threshold != -2]
    n_cuts = len(cut_point)
    intv_list = []
    if n_cuts == 1:
        intv_list = [[1, -np.inf, cut_point[0]], [2, cut_point[0], np.inf]]
    elif n_cuts > 1:
        cut_point.sort()
        intv_list = [[1, -np.inf, cut_point[0]]]
        for i in range(n_cuts - 1):
            intv_list = intv_list + [[i + 2, cut_point[i], cut_point[i + 1]]]
        intv_list = intv_list + [[n_cuts + 1, cut_point[n_cuts - 1], np.inf]]
    else:
        intv_list = [[1, -np.inf, np.inf]]
    var_map = pd.DataFrame(intv_list, columns=['_bin', '_LL', '_UL'])
    return var_map

# ChiMerge分箱--连续变量   卡方分箱
def CutVar_ChiMerge(data, x_var, y_var, max_number_intervals=4):
    chi = ChiMerge(min_expected_value=0.5, max_number_intervals=max_number_intervals, threshold=np.inf)
    chi.loadData(np.matrix(data.loc[:, [x_var, y_var]].values), False)
    chi.generateFrequencyMatrix()
    chi.chimerge()
    intv_list = chi.frequency_matrix_intervals
    intv_list[0] = -np.inf
    intv_data = []
    n = len(intv_list)
    for i in range(n):
        if i + 1 >= n:
            intv_data.append([i + 1, intv_list[i], np.inf])
        else:
            intv_data.append([i + 1, intv_list[i], intv_list[i + 1]])
    var_map = pd.DataFrame(intv_data, columns=['_bin', '_LL', '_UL'])
    return var_map

'''
《信用风险评分卡研究》一书中提到的分箱方法
metric: 'Gini', 'Entropy', 'Chi2', 'IV'
'''
def CutVar_Refaat(data, x_var, y_var, metric='IV', var_type='cont', n_bins=10, acc=0.05):
    if var_type == 'cat':
        return Refaat.ReduceCats(data, x_var, y_var, method=metric, n_bins=n_bins)
    else:
        return Refaat.BinContVar(data, x_var, y_var, method=metric, n_bins=n_bins, acc=acc)

# 变量分箱

def CutVar(var_maps, data, x_var, y_var=None, method="EqF", metric="IV", var_type='cont', **kwargs):
    kwargs1 = kwargs.copy()
    dist_flag = False
    if kwargs1.__contains__('return_dist'):
        if kwargs['return_dist']:
            dist_flag = True
        kwargs1.__delitem__('return_dist')

    plot_flag = False
    if kwargs1.__contains__('plot'):
        if kwargs['plot']:
            plot_flag = True
        kwargs1.__delitem__('plot')
    
    x_type = 'cont'
    if method == 'CutPoints':
        var_map = CutVar_CutPoints(data, x_var, **kwargs1)
    elif method == 'EqW':
        var_map = CutVar_EqW(data, x_var, **kwargs1)
    elif method == 'EqF':
        var_map = CutVar_EqF(data, x_var, **kwargs1)
    elif method == 'Means':
        var_map = CutVar_Means(data, x_var, **kwargs1)
    elif method == 'Percentiles':
        var_map = CutVar_Percentiles(data, x_var, **kwargs1)
    elif method == 'KMeans':
        var_map = CutVar_KMeans(data, x_var, **kwargs1)
    elif method == 'ChiMerge':
        var_map = CutVar_ChiMerge(data, x_var, y_var, **kwargs1)
    elif method == 'DecisionTree':
        var_map = CutVar_DecisionTree(data, x_var, y_var, **kwargs1)
    elif method == 'Refaat':
        var_map = CutVar_Refaat(data, x_var, y_var, metric=metric, var_type=var_type, **kwargs1)
        x_type = var_type
    else:
        raise ValueError('%s: this method is not supported' % method)

    # 名义/分类变量按 x 变量的值统计分布情况
    if (dist_flag & (x_type == 'cat')):
        cat_freq = pd.crosstab(data[x_var], data[y_var]).reset_index() #交叉取值，第一个参数是列，第二个是行
        cat_freq.columns = [x_var, '_good', '_bad']
        var_map = var_map.merge(cat_freq, on=x_var)
        var_map['_total'] = var_map['_good'] + var_map['_bad']
        var_map['_percent'] = var_map._total / sum(var_map._total)
        var_map['_good_rate'] = var_map._good / var_map._total

    if ((dist_flag & (x_type == 'cont')) | plot_flag):
        if x_type == 'cat':
            bin_data = ApplyMap1(data, x_var, var_map)
        else:
            bin_data = ApplyMap2(data, x_var, var_map)
        freq = pd.crosstab(bin_data[x_var], bin_data[y_var]).reset_index()
        freq.columns = ['_bin', '_good', '_bad']
        freq['_total'] = freq['_good'] + freq['_bad']
        freq['_percent'] = freq._total / sum(freq._total)
        freq['_good_rate'] = freq._good / freq._total
        if x_type == 'cont':
            freq = var_map.merge(freq, on='_bin')
            if dist_flag:
                var_map = freq
        if plot_flag:
            percent = freq['_percent']
            percent.index = map(str, freq['_bin'])
            good_rate = freq['_good_rate']
            good_rate.index = map(str, freq['_bin'])
            plt.figure()
            ax1 = percent.plot(kind='bar', color='g', alpha=0.3)
            ax2 = ax1.twinx()
            good_rate.plot(ax=ax2, color='k', marker='o', markersize=9, alpha=0.6)
            ax1.set_xlim([-0.5, len(freq._bin) - 0.5])
            ax1.set_ylabel('percent')
            ax2.set_ylabel('good rate')
            ax1.set_xlabel(x_var)
            if x_type == 'cont':
                xticks = []
                for i in range(percent.shape[0]):
                    xticks.append('[%s, %s)' % (freq.loc[i, '_LL'], freq.loc[i, '_UL']))
                ax1.set_xticklabels(xticks)
            else:
                tmp = var_map.groupby('_bin')[x_var].unique()
                for i in tmp.index:
                    print('%s bin%d = ' % (x_var, i), tmp[i])
    var_maps[(x_type, x_var)] = var_map
    return var_map

# 将单个变量分箱结果应用于数据集--名义变量
def ApplyMap1(data, x_var, var_map, new_var=None):
    if new_var == None:
        new_var = x_var
    out_data = data.copy()
    out_data = out_data.merge(var_map, on=x_var)
    out_data.loc[:, new_var] = out_data['_bin'].astype(int)
    out_data.drop('_bin', axis=1, inplace=True)
    return out_data

# 将单个变量分箱结果应用于数据集--连续变量
def ApplyMap2(data, x_var, var_map, new_var=None):
    if new_var == None:
        new_var = x_var
    n = var_map.shape[0]
    out_data = data.copy()
    for i in range(n):
        out_data.loc[(data[x_var] >= var_map.loc[i, '_LL']) & (data[x_var] < var_map.loc[i, '_UL']), new_var] = var_map.loc[i, '_bin']
    out_data.loc[:, new_var] = out_data[new_var].astype(int)
    return out_data

# 多个变量分箱--连续变量
def CutVars(var_maps, data, X_var, y_var=None, method="EqF", **kwargs):
    for x_var in X_var:
        CutVar(var_maps, data, x_var, y_var, method, **kwargs)

# 将多个变量分箱结果应用于数据集--连续变量
def ApplyMaps(data, var_maps, prefix='', suffix=''):
    tmp1 = data.copy()
    for x in var_maps.keys():
        x_type, x_var = x
        new_var = prefix + x_var + suffix
        if x_type == 'cat':
            tmp2 = ApplyMap1(tmp1, x_var, var_maps[x], new_var)
        else:
            tmp2 = ApplyMap2(tmp1, x_var, var_maps[x], new_var)
        tmp1 = tmp2.copy()
    return tmp1

def PowerVar(var_map, metric='IV', var_type='cont'):
    if var_type == 'cat':
        cross_table = var_map.groupby('_bin').sum()[['_good', '_bad']]
    else:
        cross_table = var_map[['_good', '_bad']]
    if metric == 'Chi2':
        return Chi2_CrossTable(cross_table)
    elif metric == 'LikeRatio':
        return LikeRatio_CrossTable(cross_table)
    elif metric == 'OddsRatio':
        OR, p_value, _, _ = OddsRatio_CrossTable(cross_table)
        return OR, p_value
    elif metric == 'Entropy':
        return Entropy_CrossTable(cross_table)
    elif metric == 'Gini':
        return Gini_CrossTable(cross_table)
    elif metric == 'IV':
        return IV_CrossTable(cross_table)
    else:
        raise ValueError('%s: this metric is not supported' % metric)

def PowerVars(var_maps, metric='IV', sort=True):
    v_data = []
    for x in var_maps.keys():
        x_type, x_var = x
        v = PowerVar(var_maps[x], metric=metric, var_type=x_type)
        if metric not in ['Chi2', 'LikeRatio', 'OddsRatio']:
            v1 = v
            p  = 0
        else:
            v1 = v[0]
            p  = v[1]
        v_data.append([x_var, v1, p])
    v_name = '_' + metric
    result = pd.DataFrame(v_data, columns=['_var_name', v_name, '_p_value'])
    if metric not in ['Chi2', 'LikeRatio', 'OddsRatio']:
        result.drop('_p_value', axis=1, inplace=True)
    if sort:
        result.sort_values(by=v_name, inplace=True, ascending=False)
    return result
