import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os, glob, csv
import statistics
from collections import Counter
from sklearn_classifier import open_file, get_model, CV_Para_selection
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectPercentile
from imblearn.pipeline import Pipeline
from sklearn import feature_selection
from util import label_features, write_list_to_csv, del_file
import pandas as pd
import pickle as pk
import time
from datetime import timedelta
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, SparsePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
import seaborn as sns
from mne.stats import fdr_correction
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import scipy.stats
import operator
from tqdm import tqdm

def non_param_unpaired_CI(sample1, sample2, conf):
    """
    Calculate the non-parametric unpaired confidence interval
    """
    n1 = len(sample1)
    n2 = len(sample2)
    alpha = 1 - conf
    N = stats.norm.ppf(1 - alpha / 2)

    # The confidence interval for the difference between the two population
    # medians is derived through the n x m differences.
    diffs = sorted([i - j for i in sample1 for j in sample2])

    # the Kth smallest to the Kth largest of the n x m differences then determine
    # the confidence interval, where K is:
    k = np.math.ceil(n1 * n2 / 2 - (N * (n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5))

    diff_median = np.median(list([a - b for a in sample1 for b in sample2]))
    CI = (round(diffs[k - 1], 8), round(diffs[len(diffs) - k], 8))

    return diff_median, CI


def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate the mean and confidence interval of the data
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

class ML_temp_file(object):
    '''
    This class is aiming to save the necessary metrics and information in each cross validation loop.
    The saved information could used to reproduce the experimental results, and test accuracies of combinations of
    different feature sets.
    '''

    def __init__(self, cv_num, feature_set_name, clf_model, fs_model, best_score_in_grid_search, clf_scores,
                 important_features,
                 important_features_name, doc_label, true_label):
        self.cv_num = cv_num
        self.feature_set_name = feature_set_name
        self.clf_model = clf_model  # classification model
        self.fs_model = fs_model  # feature selection model
        self.best_score_in_grid_search = best_score_in_grid_search
        self.clf_scores = clf_scores
        self.important_features = important_features
        self.important_features_name = important_features_name
        self.doc_label = doc_label
        self.true_label = true_label

    def to_pickle(self, path):
        fp = open(path, "wb")
        pk.dump(self, fp)


def main():
    Classifier_list = ['Logistic Regression', 'SVM', 'Gradient Boosting', 'AdaBoost', 'RandomForest']
    save_file_address = './result/Audio_cf_results.csv'
    imp_features_add = './result/Important_features.csv'

    Random_seed = 21  # fix random seed for a fully deterministically-reproducible run

    # Load feature sets and label the data set
    # If no 'Label' column, I label samples based on the sample ID, see details in label_features function
    file_addresses = glob.glob('./features/*csv')
    if file_addresses != []:
        for file_address in file_addresses:
            label_features(file_address)
    else:
        print("ERROR: No feature set input")

    save_directory = './statistical_test/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # load feature sets in csv files
    list_ = []
    LoadFeatures_pool = {}
    FeatureSetName_pool = []
    LoadFeatureNames_pool = []
    for file_ in file_addresses:
        FeatureSetName = os.path.basename(file_).split("_")[0]
        df = pd.read_csv(file_, index_col=0, header=0)

        if FeatureSetName == 'DisVoice':  # large number to nan
            for cols in df.columns[:-1]:
                sli = pd.Series(df[cols])
                sli[sli > 1000000000000000] = np.nan
                df[cols] = sli

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median())

        LoadFeatures_pool[FeatureSetName] = df
        FeatureSetName_pool.append(FeatureSetName)
        LoadFeatureNames = df.columns.tolist()[0:-1]
        LoadFeatureNames_pool.append(LoadFeatureNames)
        print("Info: Got the " + FeatureSetName + " Feature Set")

    task_abbreviation = '.'.join(os.path.basename(file_).split(".")[:-1]).split("_")[-1]
    # LoadFeatures_pool: features

    # create new diction
    pResults = {}
    pResults_confidence_interval = {}

    # create new list
    LoadFeatureNames_pool_filtered = []
    LoadFeatures_pool_filtered = []
    FeatureSetName_pool_filtered = []

    LoadFeatureNames_pool_save = []
    Features_pValue_pool_save = []
    FeatureSetName_pool_save = []

    for FeaSet_id in range(len(FeatureSetName_pool)):

        if FeatureSetName_pool[FeaSet_id] not in ['LDA', "conversational", "opensmile", "pyAudioAnalysis", "disvoice"] \
            and "related-zs" not in FeatureSetName_pool[FeaSet_id] and "related-gt" not in FeatureSetName_pool[FeaSet_id] :

            features_name = []
            features = []

            features_name_save = []
            features_pValue_save = []

            df_data=LoadFeatures_pool[FeatureSetName_pool[FeaSet_id]].copy()
            doc_label = df_data['Label'].values

            # Identify non-numeric label
            if ~str(doc_label)[0].isdigit():
                le = preprocessing.LabelEncoder()
                le.fit(doc_label)
                doc_label =le.transform(doc_label)

            df_data['Label'] = doc_label
            LoadFeatures_pool[FeatureSetName_pool[FeaSet_id]] = df_data

            for Fea_id in tqdm(range(len(LoadFeatureNames_pool[FeaSet_id]))):

                dis_overview = '' # show which class has a higher mean
                if len(set(doc_label)) == 2:
                    class_a = df_data.iloc[:, Fea_id][doc_label == 0]
                    class_b = df_data.iloc[:, Fea_id][doc_label == 1]

                    if np.nanmedian(class_a)>=np.nanmedian(class_b):
                        dis_overview = le.inverse_transform([0])[0]
                    else:
                        dis_overview = le.inverse_transform([1])[0]

                    try:
                        stat, p = mannwhitneyu(class_a, class_b)
                    except:
                        p = 1.0
                elif len(set(doc_label)) == 3:
                    class_a = LoadFeatures_pool[FeaSet_id][:, Fea_id][doc_label == 0]
                    class_b = LoadFeatures_pool[FeaSet_id][:, Fea_id][doc_label == 1]
                    class_c = LoadFeatures_pool[FeaSet_id][:, Fea_id][doc_label == 2]

                    try:
                        stat, p = stats.kruskal(class_a, class_b, class_c)
                    except:
                        p = 1.0

                # print(FeatureSetName_pool[FeaSet_id],LoadFeatureNames_pool[FeaSet_id][Fea_id],stat, p,Fea_id)

                features_name.append(LoadFeatureNames_pool[FeaSet_id][Fea_id])
                features.append(df_data.iloc[:, Fea_id].values.tolist())

                # save all p values
                dis_overview = str(dis_overview)
                features_name_save.append(LoadFeatureNames_pool[FeaSet_id][Fea_id]+ '_' + dis_overview)
                features_pValue_save.append(p)

                # save p-values in diction
                pResults[LoadFeatureNames_pool[FeaSet_id][Fea_id]+ '_' + dis_overview] = p
                # r = mean_confidence_interval(class_a)

                median_diff, r = non_param_unpaired_CI (class_a, class_b, conf=0.95)
                pResults_confidence_interval[LoadFeatureNames_pool[FeaSet_id][Fea_id]] = str(np.round(median_diff,4))+", 95% CI − "+str(np.round(r[0],4))+" to "+str(np.round(r[1],4))+"; P = "

            print(FeatureSetName_pool[FeaSet_id], len(features_name))

            # reject_fdr, features_pValue_save = fdr_correction(features_pValue_save, alpha=0.05, method='indep')
            # features_pValue_save = multipletests(features_pValue_save, alpha=0.05, method='bonferroni')[1]

            LoadFeatureNames_pool_filtered.append(features_name)
            LoadFeatures_pool_filtered.append(np.array(features).T)
            FeatureSetName_pool_filtered.append(FeatureSetName_pool[FeaSet_id])

            LoadFeatureNames_pool_save.append(features_name_save)
            Features_pValue_pool_save.append(features_pValue_save)
            FeatureSetName_pool_save.append(FeatureSetName_pool[FeaSet_id])

    # Apply multiple comparison correction
    all_p_values = list(pResults.values())
    rejected, corrected_p_values, _, _ = multipletests(all_p_values, method='fdr_bh')
    
    for i, key in enumerate(pResults.keys()):
        pResults[key] = corrected_p_values[i]

    # form results
    df = []
    for FeaSet_id in range(len(FeatureSetName_pool_save)):
        data_frame_header = [[FeatureSetName_pool_save[FeaSet_id], FeatureSetName_pool_save[FeaSet_id]]]
        fn_pvalue = []
        rank = np.argsort(Features_pValue_pool_save[FeaSet_id])
        for i in range(len(rank)):
            fn_pvalue.append([LoadFeatureNames_pool_save[FeaSet_id][rank[i]], pResults[LoadFeatureNames_pool_save[FeaSet_id][rank[i]]]])
        df.append(pd.DataFrame(data_frame_header + fn_pvalue, index=None, columns=None))

    df_all = pd.concat(df, axis=1, sort=False)
    df_all.to_csv('./statistical_test/Feature_P_ranking_' + task_abbreviation + '.csv', index=None, columns=None, header=None, encoding="GBK")
    print("has ouput all p values")

    p_value_list_path = './statistical_test/Feature_P_ranking_' + task_abbreviation + '.csv'
    entire_features = pd.concat(LoadFeatures_pool.values(), axis=1)
    entire_features = entire_features.loc[:, ~entire_features.columns.duplicated()]
    plot_his_2Class(p_value_list_path, entire_features, task_abbreviation)
    print("has output the plot")


def plot_his_2Class(p_value_list_path, features_list, abbr):

    plt.rcParams['font.sans-serif'] = ['SimHei']

    salient_features_list = pd.read_csv(p_value_list_path, header=0, index_col=None, encoding="GBK")
    save_directory = './histogram'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    else:
        del_file(save_directory)

    # select feature set
    for j in range(0, int(len(salient_features_list.columns) / 2)):

        plot_fig_num = 10
        for i in range(plot_fig_num):
            
            feature_name = salient_features_list.iloc[i, j * 2]

            print('plotting: (', salient_features_list.columns[j * 2], ',', feature_name, ')')

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

            label = features_list['Label']

            try:
                df = features_list[feature_name[:-2]]
            except:
                continue

            filter_col_A = [location for location, label_ in enumerate(label.values.tolist()) if label_ == 0]
            sampleA = df.loc[df.index[filter_col_A]].dropna().values.tolist()

            filter_col_B = [location for location, label_ in enumerate(label.values.tolist()) if label_ == 1]
            sampleB = df.loc[df.index[filter_col_B]].dropna().values.tolist()

            all_data = [sampleA, sampleB]
            # plot violin plot
            axes.violinplot(all_data, showmeans=False, showmedians=True)
            axes.set_title(feature_name)

            # adding horizontal grid lines
            for ax in [axes]:
                ax.yaxis.grid(True)
                ax.set_xticks([y + 1 for y in range(len(all_data))])
                # ax.set_xlabel('xlabel')
                # ax.set_ylabel('ylabel')

            # add x-tick labels
            plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))], xticklabels=[abbr[0], abbr[1]])
            fig.savefig(
                save_directory + '/' + abbr + '_' + salient_features_list.columns[j * 2] + '_' + feature_name + '.png',
                dpi=fig.dpi)


def plot_his_3Class(p_value_list_path, features_list, abbr):

    salient_features_list = pd.read_csv(p_value_list_path, header=None, index_col=None)
    save_directory = './histogram'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # select feature set
    for j in range(0, len(salient_features_list.columns)):

        plot_fig_num = 10
        for i in range(plot_fig_num):

            feature_name = salient_features_list.iloc[i, j * 2]

            print('plotting: (', salient_features_list.columns[j], ',', feature_name, ')')

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

            label = features_list['Label']
            df = features_list[feature_name]

            filter_col_A = [location for location, label_ in enumerate(label.values.tolist()) if label_ == 0]
            sampleA = df.loc[df.index[filter_col_A]].values.tolist()

            filter_col_B = [location for location, label_ in enumerate(label.values.tolist()) if label_ == 1]
            sampleB = df.loc[df.index[filter_col_B]].values.tolist()

            filter_col_C = [location for location, label_ in enumerate(label.values.tolist()) if label_ == 2]
            sampleC = df.loc[df.index[filter_col_C]].values.tolist()

            all_data = [sampleA, sampleB, sampleC]

            # plot violin plot
            axes.violinplot(all_data, showmeans=False, showmedians=True)
            axes.set_title(feature_name)

            # adding horizontal grid lines
            for ax in [axes]:
                ax.yaxis.grid(True)
                ax.set_xticks([y + 1 for y in range(len(all_data))])

            # add x-tick labels
            plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))], xticklabels=[abbr[0], abbr[1], abbr[2]])
            fig.savefig(save_directory + '/' + abbr + '_' + salient_features_list[j * 2][0] + '_' + feature_name + '.png', dpi=fig.dpi)


if __name__ == "__main__":
    main()