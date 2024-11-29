import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
import glob
from sklearn_classifier import get_model
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
import pandas as pd
from util import write_classification_results_to_csv
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
import os
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import RobustScaler
from sklearn_classifier import calibrate_probabilities
from sklearn.multiclass import OneVsRestClassifier
import pickle
from sklearn.linear_model import LogisticRegression

def to_labels(pos_probs, threshold):
    if np.array(pos_probs).ndim == 1:
        return (pos_probs >= threshold).astype('int')
    elif np.array(pos_probs).ndim == 2:
        return (pos_probs[:, 1] >= threshold).astype('int')
    else:
        print("Wrong dimension of to_labels function.")
        quit()

def PRC_curve(scores, real_label, save_path, le=None):
    # plot precision-recall curve
    from sklearn.metrics import average_precision_score, auc
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.preprocessing import label_binarize

    if len(set(real_label)) == 2:
        y_train = np.array([1 - np.array(real_label), real_label]).T
    else:
        y_train = label_binarize(real_label, classes=sorted(set(real_label)))
    n_classes = y_train.shape[1]
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    precision = dict()
    recall = dict()
    average_precision = dict()
    thresholds = dict()

    for i in range(n_classes):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(y_train[:, i], scores[:, i])
        average_precision[i] = auc(recall[i], precision[i])

    # A "micro-average": quantifying score on all classes jointly
    # precision["weighted"], recall["weighted"], _ = precision_recall_curve(Y_test.ravel(), scores.ravel())
    # average_precision['weighted'] = average_precision_score(Y_test, scores, average='weighted')
    AUPRC = average_precision_score(y_train, scores, average='weighted')

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')

    if n_classes == 2:
        threshold = np.linspace(0, 1, num=100)
        l, = plt.plot(threshold, [balanced_accuracy_score(real_label, to_labels(scores[:, 1], x)) for x in threshold],
                      color='gold', lw=2)
        lines.append(l)
        labels.append('Balanced Accuracy')

    # l, = plt.plot(recall["weighted"], precision["weighted"], color='gold', lw=2)
    # lines.append(l)
    # labels.append('weighted-average Precision-recall (area = {0:0.2f})'
    #               ''.format(AUPRC))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        if le is not None:
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(le.inverse_transform([i])[0], average_precision[i]))
        else:
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(str(i), average_precision[i]))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Weighted Precision: {0:0.2f}'.format(AUPRC))
    plt.legend(lines, labels, loc='lower left', prop=dict(size=12))

    plt.savefig(save_path, dpi=200)

    return AUPRC


def ROC_curve(scores, real_label, save_path, le=None):
    # plot ROC curve
    # Compute ROC curve and ROC area for each class
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import label_binarize
    from itertools import cycle

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    if len(set(real_label)) == 2:
        y_train = np.array([1 - np.array(real_label), real_label]).T
    else:
        y_train = label_binarize(real_label, classes=sorted(set(real_label)))
    n_classes = y_train.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_train[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_train.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(7, 8))
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        if le is not None:
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(le.inverse_transform([i])[0], roc_auc[i]))
        else:
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    # plt.show()

    plt.savefig(save_path, dpi=100)

    return roc_auc_score(y_train, scores, multi_class="ovo", average="weighted")

def cross_validation(dataframes, feature_sets, abbr, Classifier_list, test_feature_sets=None, prob_calibration=True, n_splits=20, cache_file='./tmp/cv_cache.pkl', weighting_method = "equal"):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    master_df = pd.concat([df[label_column] for df, _, label_column in dataframes], axis=1)
    master_df.columns = [f'label_{i}' for i in range(len(dataframes))]
    master_df['combined_label'] = master_df.mode(axis=1)[0]
    
    for i, (df, feature_columns, _) in enumerate(dataframes):
        master_df = master_df.join(df[feature_columns], rsuffix=f'_feature_{i}')
    
    all_fold_results = []
    for fold, (train_index, val_index) in enumerate(skf.split(master_df, master_df['combined_label']), 1):
        print(f"===== {abbr} == Running fold ", fold, " ===========")
        if fold not in cache or not all(fs in cache[fold] for fs in feature_sets):

            # Create a new cache for this fold
            if fold not in cache:
                cache[fold] = {}
                fold_results = {}
            else:
                fold_results = cache[fold]
            
            # Run each feature set
            for i, (df, feature_columns, label_column) in enumerate(dataframes):
                
                # skip if feature set is already in cache
                if feature_sets[i] in cache[fold]:
                    print(f"Skipping {feature_sets[i]}")
                    continue

                print(f"Running {feature_sets[i]}")

                aligned_df = master_df.loc[:,['combined_label']].join(df[feature_columns], how='left')
                
                X_train = aligned_df.iloc[train_index][feature_columns]
                X_val = aligned_df.iloc[val_index][feature_columns]
                y_train = aligned_df.iloc[train_index]['combined_label']
                y_val = aligned_df.iloc[val_index]['combined_label']
                
                valid_train = X_train.notna().mean(axis=1) > 0.5
                valid_val = X_val.notna().mean(axis=1) > 0.5
                X_train, y_train = X_train[valid_train], y_train[valid_train]
                X_val, y_val = X_val[valid_val], y_val[valid_val]
                
                if len(y_val) == 0:
                    continue
                
                pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler()),
                    ('feature_selection', SelectFromModel(get_model(Classifier_list[0]), threshold="mean")),
                    ('oversample', SMOTE(random_state=2024)), 
                    ('classifier', OneVsRestClassifier(get_model(Classifier_list[0])))
                ])
                
                if prob_calibration:
                    train_cv_prob, y_pred_proba, balanced_acc  = calibrate_probabilities(X_train, y_train, X_val, pipeline, Classifier_list)
                    train_cv_prob = pd.DataFrame(train_cv_prob, index=X_train.index, columns=np.unique(y_train))
                else:
                    pipeline.fit(X_train, y_train)
                    y_pred_proba = pipeline.predict_proba(X_val)
                    train_cv_prob = None
                    balanced_acc = 0.8

                unique_classes = np.unique(y_train)
                fold_results[feature_sets[i]] = {
                    'proba': pd.DataFrame(y_pred_proba, index=X_val.index, columns=unique_classes),
                    'true': y_val,
                    'balanced_acc': balanced_acc,
                    'train_cv_prob': train_cv_prob,
                    'train_y': y_train
                }
            
            cache[fold] = fold_results
            
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
        
        all_fold_results.append(cache[fold])
    
    if test_feature_sets is None:
        test_feature_sets = feature_sets
    
    all_true_labels = []
    all_pred_probas = []
    
    accuracies, f1_scores, balanced_accuracies = [], [], []
    
    for fold, fold_results in enumerate(all_fold_results, 1):

        # load the results for this fold
        y_pred_probas = [fold_results[fs]['proba'] for fs in test_feature_sets if fs in fold_results]
        y_val_combined_list = [fold_results[fs]['true'] for fs in test_feature_sets if fs in fold_results]
        y_val_combined = pd.concat(y_val_combined_list).groupby(level=0).agg(lambda x: x.value_counts().index[0])
        BACs = [fold_results[fs].get('balanced_acc', 0.8) for fs in test_feature_sets if fs in fold_results]
        train_cv_probs = [fold_results[fs].get('train_cv_prob', None) for fs in test_feature_sets if fs in fold_results]
        train_y = [fold_results[fs].get('train_y', None) for fs in test_feature_sets if fs in fold_results]
    

        if weighting_method == 'balanced_accuracy':
            # Balanced accuracy weighting with majority baseline normalization
            majority_baseline = max(Counter(y_val_combined).values()) / len(y_val_combined)
            normalized_BACs = np.array(BACs) - majority_baseline
            weights = normalized_BACs / np.sum(normalized_BACs)

        elif weighting_method == 'Best':
            
            # select the best feature set's weight to 1 and other to 0.01
            weights = [0.01 for x in range(len(BACs))]
            best_idx = np.argmax(BACs)
            weights[best_idx] = 1

        elif weighting_method == 'inverse_error':
            # Inverse Error Weighting
            errors = 1 - np.array(BACs)
            weights = (1 / errors) / np.sum(1 / errors)

        elif weighting_method == 'equal':
            # Equal weighting (simple average)
            weights = np.ones(len(BACs)) / len(BACs)

        elif weighting_method == 'meta_learner':
            # concate results
            train_cv_probs = pd.concat(train_cv_probs, axis=1).fillna(0.5)
            train_y = pd.concat(train_y).groupby(level=0).agg(lambda x: x.value_counts().index[0])
            y_pred_probas = pd.concat(y_pred_probas, axis=1).fillna(0.5)
            
            # Use a meta-learner to learn weights
            # Ensure meta_X and meta_y have the same index
            common_index = train_cv_probs.index.intersection(train_y.index)
            meta_X = train_cv_probs.loc[common_index]
            meta_y = train_y.loc[common_index]

            # Train a meta-learner 
            meta_learner = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler()),
                    ('feature_selection', SelectFromModel(get_model(Classifier_list[0]), threshold="mean")),
                    ('oversample', SMOTE(random_state=2024)), 
                    ('classifier', OneVsRestClassifier(get_model(Classifier_list[0])))
                ])
            meta_learner.fit(meta_X, meta_y)

            # Use the meta-learner to predict probabilities for the test set
            y_pred_proba_combined = meta_learner.predict_proba(y_pred_probas)
            y_pred_proba_combined = pd.DataFrame(y_pred_proba_combined, index=y_pred_probas.index, columns=meta_learner.classes_)
        else:
            # Default to equal weighting if an unknown method is specified
            weights = np.ones(len(BACs)) / len(BACs)
            y_pred_proba_combined = pd.concat([df * weight for df, weight in zip(y_pred_probas, weights)]).groupby(level=0).mean()

        if weighting_method != 'meta_learner':
            y_pred_proba_combined = pd.concat([df * weight for df, weight in zip(y_pred_probas, weights)]).groupby(level=0).mean()
        # Check if the index is the same for all predictions
        common_index = y_val_combined.index.intersection(y_pred_proba_combined.index)
        y_val_combined = y_val_combined.loc[common_index]
        y_pred_proba_combined = y_pred_proba_combined.loc[common_index]
        
        if len(common_index) != len(y_pred_proba_combined):
            print(f"Warning: Mismatch in index lengths for fold {fold}")
            print(f"Common index length: {len(common_index)}")
            print(f"y_pred_proba_combined length: {len(y_pred_proba_combined)}")
            print(f"y_val_combined length: {len(y_val_combined)}")

        y_pred_combined = y_pred_proba_combined.idxmax(axis=1)
        
        accuracy = accuracy_score(y_val_combined, y_pred_combined)
        f1 = f1_score(y_val_combined, y_pred_combined, average='weighted')
        balanced_accuracy = balanced_accuracy_score(y_val_combined, y_pred_combined)
        
        accuracies.append(accuracy)
        f1_scores.append(f1)
        balanced_accuracies.append(balanced_accuracy)
        
        print(f"Fold {fold}: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}, Balanced Accuracy = {balanced_accuracy:.4f}")
        
        all_true_labels.extend(y_val_combined)
        all_pred_probas.append(y_pred_proba_combined)
    
    print("\nAverage Scores:")
    print(f"Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"F1 Score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    print(f"Balanced Accuracy: {np.mean(balanced_accuracies):.4f} (+/- {np.std(balanced_accuracies):.4f})")
    
    real_label = np.array(all_true_labels)
    scores = np.concatenate(all_pred_probas)
    predict_label = scores.argmax(axis=1)
    
    acc_result = accuracy_score(real_label, predict_label)
    BAC = balanced_accuracy_score(real_label, predict_label)
    MCC = metrics.matthews_corrcoef(real_label, predict_label)
    F1_weighted = f1_score(real_label, predict_label, average='weighted')

    # Save PRC and ROC curve
    save_path = './PRC/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_address = save_path + abbr + '_PRC_' + '_'.join(test_feature_sets) + '.png'
    AUPRC_weighted = PRC_curve(scores, real_label, save_address, le=None)

    save_address = save_path + abbr + '_ROC_' + '_'.join(test_feature_sets) + '.png'
    ROC_weighted = ROC_curve(scores, real_label, save_address, le=None)

    # Save other classification metrics
    CM = confusion_matrix(real_label, predict_label)
    CR_print = classification_report(real_label, predict_label)
    CR = precision_recall_fscore_support(real_label, predict_label)
    
    print("Classification accuracy:", acc_result)
    print("Confusion Matrix:", CM)
    print(CR_print)
    print(CR)
    print("AUPRC weighted:", AUPRC_weighted)

    counter = Counter(real_label)
    baseline = counter.most_common(1)[0][1] / len(real_label)

    save_file_address = f'./result/Audio_cf_results_{abbr}.csv'
    os.makedirs(os.path.dirname(save_file_address), exist_ok=True)

    if abbr == 'DS_H':
        abbr = 'PH'
    
    abbr_ = '_'.join([x for x in abbr])
    feature_used = '_'.join(sorted(test_feature_sets))
    write_classification_results_to_csv(feature_used, F1_weighted, MCC, acc_result, BAC, AUPRC_weighted,
                                        ROC_weighted, CM, save_file_address, CR, baseline, "_".join(Classifier_list) + "_FS", abbr_)
    
    return acc_result, F1_weighted, BAC, AUPRC_weighted, ROC_weighted

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Run classification with early or late fusion')
    parser.add_argument('--fusion_type', type=str, default='late', choices=['early', 'late'], help='Fusion type: early or late')
    parser.add_argument('--classifiers', nargs='+', default=['RandomForest'], help='List of classifiers to use')
    parser.add_argument('--features', nargs='+', default=[], help='List of feature sets to use')
    parser.add_argument('--folder_path', type=str, default='./features', help='Path to the folder containing CSV files')
    parser.add_argument('--skip_single_feature_set', action="store_true", default=False, help='Path to the folder containing CSV files')
    args = parser.parse_args()

    # List all CSV files in the specified folder
    csv_files = glob.glob(f"{args.folder_path}/*.csv")
    args.features = ["LIWC", "TFIDF", "clinical-related", "Assessment-related"]

    # Load each CSV file into a DataFrame and extract feature columns and label column
    Classifier_list = args.classifiers
    F_List = args.features
    dataframes = []
    feature_sets = []
    all_labels = pd.DataFrame()
    full_feature_List = [os.path.basename(x).split('_')[0] for x in csv_files]

    for file in [x for x in csv_files if os.path.basename(x).split('_')[0] in full_feature_List]:

        df = pd.read_csv(file, index_col=0, header=0, low_memory=False)
        feature_columns = df.columns[:-1]
        label_column = 'Label'
        feature_set_name = os.path.basename(file).split('_')[0]

        # Merge labels
        if all_labels.empty:
            all_labels = df[['Label']]
        else:
            all_labels = all_labels.combine_first(df[['Label']])

        dataframes.append((df, feature_columns, "Label"))
        print('Have read file: ', file, 'with shape: ', df.shape)
        feature_sets.append(feature_set_name)
        abbr = os.path.basename(file).split('.')[0].split('_')[-1]


    # First run to cache all results
    acc, f1, bac, auprc, auroc = cross_validation(dataframes, feature_sets, abbr, Classifier_list)
    print(f"All features - Accuracy: {acc:.4f}, F1: {f1:.4f}, BAC: {bac:.4f}, AUPRC: {auprc:.4f}, AUROC: {auroc:.4f}")

    # Subsequent runs to test different feature combinations
    test_sets = [["opensmile"],
                 ['Assessment-related'], 
                 ["clinical-related-gt"], 
                 ["clinical-related-zs"], 
                 ["clinical-related-sft"], 
                 ["TFIDF"],
                 ['LIWC'], 
                 ['Assessment-related', "clinical-related-sft"],
                 ['Assessment-related', "clinical-related-sft", "opensmile"],
                 ['Assessment-related', "clinical-related-sft", 'LIWC', "TFIDF"],
                 ['Assessment-related', "clinical-related-zs", 'LIWC', "TFIDF", "opensmile"],
                 ['Assessment-related', "clinical-related-sft", 'LIWC', "TFIDF", "opensmile"],
                 ]
    
    for test_set in test_sets:
        acc, f1, bac, auprc, auroc = cross_validation(dataframes, feature_sets, abbr, Classifier_list, 
                                                      test_feature_sets=test_set)
        print(f"{test_set} - Accuracy: {acc:.4f}, F1: {f1:.4f}, BAC: {bac:.4f}, AUPRC: {auprc:.4f}, AUROC: {auroc:.4f}")
