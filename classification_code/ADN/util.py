#the aim of this code is to read the txt file and convert to a form which LIWC
#can directly use. Additionally, add the number of text into the excel log file

import os
import csv
import numpy as np
import pandas as pd

class ML_temp_file(object):
    '''
    This class is aiming to save the necessary metrics and information in each cross validation loop.
    The saved information could used to reproduce the experimental results, and test accuracies of combinations of
    different feature sets.
    '''

    def __init__(self, cv_num, feature_set_name, clf_model, fs_model, best_score_in_grid_search, clf_scores, doc_label,
                 true_label, sample_id, session_name, baseline):
        self.cv_num = cv_num
        self.feature_set_name = feature_set_name
        self.clf_model = clf_model  # classification model
        self.fs_model = fs_model  # feature selection model
        self.best_score_in_grid_search = best_score_in_grid_search
        self.clf_scores = clf_scores
        self.doc_label = doc_label
        self.true_label = true_label
        self.session_name = session_name
        self.sample_id = sample_id
        self.baseline = baseline
        self.train_cv_prob = []
        self.orig_score = []
        self.training_id = []

    def to_pickle(self, path):
        fp = open(path, "wb")
        pk.dump(self, fp)

def label_features(file_address):
    labeled_data=[]
    with open(file_address, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:

            # check whether have label already
            if row[-1]=="Label":
                print("Info: Labeled file -",file_address)
                return

            if row[0]=="ID":
                labeled_data.append(row+["Label"])
            elif row[0].startswith("C") or row[0].startswith("S"):
                labeled_data.append(row+["Patient"])
            elif row[0].startswith("H"):
                labeled_data.append(row+["Healthy"])
            elif row[0].startswith("D"):
                labeled_data.append(row + ["Patient"])
            else:
                labeled_data.append(row + ["None"])
                
    csvfile.close()

    with open(file_address,'w',newline="") as f:
        writer = csv.writer(f, delimiter=',')
        for item in labeled_data:
            #print(item)
            writer.writerow(item)
    f.close()
    print("Info: Has labeled the output file")

def write_list_to_csv(save_name,content_list):
    with open(save_name,'w',newline="") as f:
        writer = csv.writer(f, delimiter=',')
        for item in content_list:
            writer.writerow(item)
    f.close()

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def read_csv(file_address):
    file_contents = []
    with open(file_address, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            file_contents.append(row)
    return file_contents

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def check_file_exists(file_address):
    return os.path.isfile(file_address)

def write_prediction_title_in_csv(save_file_address, abbr, pos_class=None):

    # define positive class
    if pos_class is None:
        if 'L' not in abbr:
            if 'S' in abbr:
                pos_class = list(abbr.split('_')).index('S')
            elif 'D' in abbr:
                pos_class = list(abbr.split('_')).index('D')
            else:
                pos_class = 0
        else:
            if 'H' in abbr:
                pos_class = list(abbr.split('_')).index('H')
            else:
                pos_class = 0

    title = [['Feature', 'Score','CM'] + abbr.split('_') + ['SEN', "SPE", "F1", 'MCC', 'AUC', 'ROC', 'ACC', 'BAC', 'MB',
                                                           'THR','NOTE']]
    if not os.path.exists(save_file_address):
        df = pd.DataFrame(title, index=None, columns=None)
        df.to_csv(save_file_address, index=False, columns=None, header=False, mode='a')

def write_prediction_result_in_csv(feature_used, nsa_name, F1_weighted, MCC, Acc, BAC, AUC_max, ConfuMatrix_max,
                                             save_file_address, ROC, ClfReport_, baseline, thresh, Note, abbr, pos_class=None):
    class_num = len(ClfReport_[0])

    # define positive class
    if pos_class is None:
        if 'L' not in abbr:
            if 'S' in abbr:
                pos_class = list(abbr.split('_')).index('S')
            elif 'D' in abbr:
                pos_class = list(abbr.split('_')).index('D')
            else:
                pos_class = 0
        else:
            if 'H' in abbr:
                pos_class = list(abbr.split('_')).index('H')
            else:
                pos_class = 0

    # get all metrics
    cnf_matrix = ConfuMatrix_max
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    contents = []
    for clf in range(class_num):
        contents.append([feature_used, nsa_name,abbr.split('_')[clf]] +
                        ConfuMatrix_max[clf].tolist() +
                        [TPR[pos_class], TNR[pos_class], F1_weighted, MCC, AUC_max, ROC, Acc, BAC,
                         baseline, thresh, Note])

    df = pd.DataFrame(contents, index=None, columns=None)
    df.to_csv(save_file_address, index=False, columns=None, header=False, mode='a')

def write_classification_title_in_csv(save_file_address, abbr, pos_class=None):

    # define positive class
    if pos_class is None:
        if 'L' not in abbr:
            if 'A' in abbr:
                pos_class = list(abbr.split('_')).index('A')
            elif 'D' in abbr:
                pos_class = list(abbr.split('_')).index('D')
            else:
                pos_class = 0
        else:
            if 'H' in abbr:
                pos_class = list(abbr.split('_')).index('H')
            else:
                pos_class = 0

    title = [['Feature', 'CM']+abbr.split('_')+['SEN', "SPE", "F1", 'MCC','AUC', 'ROC', 'ACC', 'BAC', 'MB', 'WM']]
    if not os.path.exists(save_file_address):
        df = pd.DataFrame(title, index=None, columns=None)
        df.to_csv(save_file_address, index=False, columns=None, header=False, mode='a')

def write_classification_results_to_csv(feature_used, F1_weighted, MCC, Acc, BAC, AUC_max, ROC, ConfuMatrix_max, save_file_address, ClfReport_,
                                        baseline, classifier_name, abbr, pos_class=None):
    class_num = len(ClfReport_[0])

    # define positive class
    if pos_class is None:
        if 'L' not in abbr:
            if 'A' in abbr:
                pos_class = list(abbr.split('_')).index('A')
            elif 'D' in abbr:
                pos_class = list(abbr.split('_')).index('D')
            else:
                pos_class = 0
        else:
            if 'H' in abbr:
                pos_class = list(abbr.split('_')).index('H')
            else:
                pos_class = 0

    # Calculate confusion matrix metrics
    FP = ConfuMatrix_max.sum(axis=0) - np.diag(ConfuMatrix_max)
    FN = ConfuMatrix_max.sum(axis=1) - np.diag(ConfuMatrix_max)
    TP = np.diag(ConfuMatrix_max)
    TN = ConfuMatrix_max.sum() - (FP + FN + TP)

    # Convert to float for division operations
    FP, FN, TP, TN = map(lambda x: x.astype(float), [FP, FN, TP, TN])

    # Calculate performance metrics for each class
    TPR = TP / (TP + FN)  # Sensitivity / Recall
    TNR = TN / (TN + FP)  # Specificity
    PPV = TP / (TP + FP)  # Precision
    NPV = TN / (TN + FN)  # Negative Predictive Value
    FPR = FP / (FP + TN)  # False Positive Rate
    FNR = FN / (TP + FN)  # False Negative Rate
    FDR = FP / (TP + FP)  # False Discovery Rate
    ACC = (TP + TN) / (TP + FP + FN + TN)  # Accuracy

    # Prepare results for each class
    contents = []
    for clf in range(class_num):
        contents.append([
            feature_used,
            abbr.split('_')[clf],
            *ConfuMatrix_max[clf].tolist(),
            TPR[clf],
            TNR[clf],
            F1_weighted[clf] if isinstance(F1_weighted, np.ndarray) else F1_weighted,
            MCC[clf] if isinstance(MCC, np.ndarray) else MCC,
            AUC_max[clf] if isinstance(AUC_max, np.ndarray) else AUC_max,
            ROC[clf] if isinstance(ROC, np.ndarray) else ROC,
            Acc[clf] if isinstance(Acc, np.ndarray) else Acc,
            BAC[clf] if isinstance(BAC, np.ndarray) else BAC,
            baseline,
            classifier_name
        ])

    # Write results to CSV
    write_classification_title_in_csv(save_file_address, abbr)
    df = pd.DataFrame(contents, index=None, columns=None)
    df.to_csv(save_file_address, index=False, header=False, mode='a')

def merge_all_tmp_file(in_path = './tmp/',out_path = './pk/'):
    import pickle as pk
    from shutil import copy2

    # # save all pickle files into one pickle file
    fs = glob.glob(os.path.join(in_path, '*.pickle'))

    # read the first level file names (split by '_')
    score_name = list(set([os.path.basename(f).split('_')[0] for f in fs]))
    print(score_name)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for score in score_name:
        print(score)
        files = {}
        for f in [x for x in fs if str(score) in x]:
            with open(f, 'rb') as fr:
                load_file = pk.load(fr)
            files[f] = load_file

        # save individual scores in a pickle file
        path = os.path.join(out_path,str(score)+'_row-score.pickle')
        with open(path, "wb") as fp:
            pk.dump(files,fp)

    # copy .csv files
    fs = glob.glob(os.path.join(in_path, '*.csv'))
    for f in fs:
        copy2(f, os.path.join(out_path, str(os.path.basename(f).split('.')[0]) + '_row-score.csv'))

def recover_all_tmp_files(in_path = './pk/',out_path = './tmp/'):
    import pickle as pk
    from shutil import copy2
    from tqdm import tqdm

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # copy .csv files
    fs = glob.glob(os.path.join(in_path, '*.csv'))
    for f in fs:
        copy2(f, out_path)

    pk_files = glob.glob(os.path.join(in_path,'*.pickle'))

    for f in tqdm(pk_files):
        with open(f, 'rb') as fr:
            load_file = pk.load(fr)
        for key in load_file.keys():
            with open(key, "wb") as fp:
                # print(key)
                pk.dump(load_file[key], fp)

if __name__ == '__main__':
    import glob

    # read labels
    df_score = pd.read_csv("./score/PHQ_score.csv", index_col=0, header=0)
    df_score.columns = ["Label"]
    df_score.index = [x.replace("_", "-") for x in df_score.index]

    # relabel the files
    for f in glob.glob("./features/*.csv"):
        df = pd.read_csv(f, index_col=0, header=0)
        df.loc[:, "Label"] = [df_score.loc[x, "Label"] for x in df.index]
        df.index.name = "ID"
        df.to_csv(f)
