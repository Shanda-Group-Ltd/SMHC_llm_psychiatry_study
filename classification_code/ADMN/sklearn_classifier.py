import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import balanced_accuracy_score
# load data
def open_file(file_name):
    '''
    This function is aiming to open the .csv file as following structure, and return the
    features set array, label array and the list of feature names.

    ID         feature-1  feature-2  feature-3  feature-4  ... feature-n Label
    sample1      xxx         xxx        xxx        xxx           xxx       a
    sample2      xxx         xxx        xxx        xxx           xxx       a
    sample3      xxx         xxx        xxx        xxx           xxx       b
    ...          ...         ...        ...        ...           ...      ...
    samplen      xxx         xxx        xxx        xxx           xxx       b

    '''
    ini = 0
    ClassLabel = []
    Data = []
    with open(file_name, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # first line is feature names, last column is label
        for row in spamreader:
            if ini == 0:
                FeatureNames = row[1:-1]
            else:
                ClassLabel.append(row[-1])
                Data.append([float(value) for value in row[1:-1]])
            ini = ini + 1

    # Identify non-numeric label
    if ~ClassLabel[0].isdigit():
        le = preprocessing.LabelEncoder()
        le.fit(ClassLabel)
        ClassLabel = le.transform(ClassLabel)

    # return numpy format
    X_tensor = np.array(Data)
    Y_tensor = np.array(ClassLabel)
    return X_tensor, Y_tensor, FeatureNames

def CV_Para_selection(X_tensor, Y_tensor, Classifier_list,random_seed_num=21,FeatureSetName='Doc2Vec'):

    '''
    this funtion is aiming to perform the hyper-parameter tuning and feature selection

    Parameters
    ----------
    X_tensor:
            input features
    Y_tensor:
            input labels
    Classifier_list:
            list of classifier names that you want to use, should match the name in get_model() function
    random_seed_num:
            fix the random seed of classifier and oversampling method
    FeatureSetName:
            use to provide the name of current feature set name, as we may not want to apply feature selection for some
            specific feature set (e.g., like doc2vec)

    Return
    ----------
    Selected_cf:
            the classifier model after tuning the hyper-parameters
    Selected_fs
            the feature selection model
    best_scores
            the classification accuracy (within grid search) of best classifier model and best feature selection model
    '''
    pipeline_step_variance = 'variance_reduce'
    pipeline_step_smote = 'oversample'
    pipeline_step0 = 'StandardScaler'
    pipeline_step1 = 'reduce_dim'
    pipeline_step2 = 'classifier'
    num_of_feature=np.array(X_tensor).shape[1]

    pipe = Pipeline([
        (pipeline_step_variance, VarianceThreshold(threshold=(0.0000000001))),
        (pipeline_step_smote, SMOTE(random_state=random_seed_num)),
        (pipeline_step0, preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)),
        (pipeline_step1, SelectPercentile(feature_selection.f_classif)),
        (pipeline_step2, get_model(Classifier_list[0]))
    ])

    Selected_cf = []
    Selected_fs = []
    best_scores=[]

    for cf_item in Classifier_list:
        para_steps = {}
        if FeatureSetName == 'Doc2Vec':
            para_steps.update({pipeline_step1: [SelectPercentile(feature_selection.f_classif)],
                               pipeline_step1 + '__percentile': [100]
                               }
                              )
        else:
            para_steps.update({pipeline_step1: [SelectPercentile(feature_selection.f_classif)],
                               pipeline_step1 + '__percentile': [100]
                               }
                              )
        #print(cf_item)
        if cf_item == 'Logistic Regression':
            para_steps.update({pipeline_step2: [get_model('Logistic Regression')],
                               # pipeline_step2 + '__C': [0.01, 0.1, 1],
                               # pipeline_step2 + '__solver': [ 'liblinear', 'sag']
                               }
                              )
        elif cf_item == 'SVM':
            para_steps.update({pipeline_step2: [get_model('SVM')],
                               pipeline_step2 + '__kernel': ['linear', 'rbf'],
                               # pipeline_step2 + '__C': [0.1, 0.5, 1],
                               # pipeline_step2 + '__gamma': [0.1, 0.01, 0.001]
                               }
                              )
        elif cf_item == 'Gradient Boosting':
            para_steps.update({pipeline_step2: [get_model('Gradient Boosting')],
                               # pipeline_step2 + '__learning_rate': [0.1, 0.5, 1],
                               # pipeline_step2 + '__max_depth': [3, 4, 5, 7],
                               # pipeline_step2 + '__n_estimators': [100, 150, 200]
                               }
                              )
        elif cf_item == 'AdaBoost':
            para_steps.update({pipeline_step2: [get_model('AdaBoost')],
                               # pipeline_step2 + '__learning_rate': [0.1, 0.5, 1],
                               # pipeline_step2 + '__n_estimators': [50, 100, 150]
                               }
                              )
        elif cf_item == 'RandomForest':
            para_steps.update({pipeline_step2: [get_model('RandomForest')],
                               # pipeline_step2 + '__n_estimators': [100, 150, 200],
                               # pipeline_step2 + '__max_depth': [3, 5, 7, None]
                               }
                              )
        elif cf_item == 'MLP':
            para_steps.update({pipeline_step2: [get_model('MLP')],
                               # pipeline_step2 + '__hidden_layer_sizes': [100, 150, 200],
                               # pipeline_step2 + '__activation': ['identity', 'logistic', 'tanh', 'relu']
                               }
                              )
        else:
            print("Error: No such classifier")

        param_grid = para_steps

        skf = StratifiedKFold(n_splits=10,random_state=random_seed_num)
        loo = LeaveOneOut()
        grid = GridSearchCV(pipe, cv=skf, n_jobs=16, param_grid=param_grid, return_train_score="False")
        # print(X_tensor,Y_tensor)
        grid.fit(X_tensor, Y_tensor)
        # print(grid.cv_results_)
        # input()
        best_classifier, best_evaluator = grid.best_params_['classifier'], grid.best_params_['reduce_dim']

        Selected_cf.append(best_classifier)
        Selected_fs.append(best_evaluator)
        best_scores.append(grid.best_score_)

        #print(best_classifier)
        #print(best_evaluator)
        print(grid.best_score_, " ", cf_item)

    return Selected_cf, Selected_fs, best_scores

def get_model(name):
    '''
    This function is used to return classifier model with a classifier name as input
    '''

    random_seed_num=2024
    if name == 'Logistic Regression':
        return LogisticRegression(solver='liblinear',random_state=random_seed_num,class_weight="balanced")
    elif name == 'SVM':
        return SVC(kernel='linear', probability=True, random_state=random_seed_num,class_weight="balanced")
    elif name == 'Decision tree':
        return tree.DecisionTreeClassifier(random_state=random_seed_num,class_weight="balanced")
    elif name == 'SVC':
        return SVC(kernel='poly', probability=True, random_state=random_seed_num,class_weight="balanced")
    elif name == 'SVM_rbf':
        return SVC(kernel='rbf', probability=True, random_state=random_seed_num,class_weight="balanced")
    elif name == 'MultinomialNB':
        return MultinomialNB()
    elif name == 'Gradient Boosting':
        return GradientBoostingClassifier(n_estimators=100, random_state=random_seed_num)
    elif name == 'KNeighborsClassifier':
        return KNeighborsClassifier(3)
    elif name == 'MLP':
        return MLPClassifier(solver='lbfgs', random_state=random_seed_num)
    elif name == 'NaiveBayes':
        return GaussianNB()
    elif name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=100,random_state=random_seed_num)
    elif name == "RandomForest":
        return RandomForestClassifier(max_depth=7, n_estimators=100, criterion="gini", random_state=random_seed_num,class_weight="balanced")
    else:
        raise ValueError('No such model')
    
def process_fold(args):
    train_index, val_index, X_train, Y_train, Classifier_list, Random_seed = args
    x_train, x_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train, y_val = Y_train.iloc[train_index], Y_train.iloc[val_index]

    pipe_ = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Use median imputation for robustness
        ('scaler', RobustScaler()),  # Use RobustScaler to handle outliers
        ('feature_selection', SelectFromModel(get_model(Classifier_list[0]), threshold="mean")),  # Feature selection using random forest
        ('oversample', SMOTE(random_state=21, k_neighbors=min(6, len(X_train) - 1))), 
        ('classifier', OneVsRestClassifier(get_model(Classifier_list[0])))  # Use the first classifier in the list
    ])

    pipe_.fit(x_train, y_train)
    proba = pipe_.predict_proba(x_val)

    scores = []
    for i in range(len(set(Y_train))):
        x_val_ = pipe_.named_steps['imputer'].transform(x_val)
        x_val_ = pipe_.named_steps['scaler'].transform(x_val_)
        x_val_ = pipe_.named_steps['feature_selection'].transform(x_val_)
        score = pipe_.named_steps['classifier'].estimators_[i].predict_proba(x_val_)
        scores.append(score)
    
    return proba, scores

def calibrate_probabilities(X_train, Y_train, X_test, pipe, Classifier_list):
    """
    This function is used to calibrate the probabilities of the classifier.
    It takes the training and testing data, the pipeline, and the list of classifiers as input.
    It returns the calibrated probabilities for both training and test sets, and balanced accuracy.
    """

    def get_smote_pipeline(Classifier_list, k_neighbors=6):
        smote_params = {'random_state': 2024}
        if k_neighbors:
            smote_params['k_neighbors'] = k_neighbors
        return Pipeline([
                ('imputer', SimpleImputer(strategy='median')),  # Use median imputation for robustness
                ('scaler', RobustScaler()),  # Use RobustScaler to handle outliers
                ('feature_selection', SelectFromModel(get_model(Classifier_list[0]), threshold="mean")),  # Feature selection using random forest
                ('oversample', SMOTE(random_state=21, k_neighbors=min(k_neighbors, len(X_train) - 1))), 
                ('classifier', OneVsRestClassifier(get_model(Classifier_list[0])))  # Use the first classifier in the list
            ])

    def multiclass_probability_calibration(X_train, Y_train, X_test, Classifier_list, Random_seed=2024):
        # Initialize KFold
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=Random_seed)
        
        from multiprocessing import Pool, cpu_count
        
        with Pool(processes=min(10, cpu_count())) as pool:
            results = pool.map(process_fold, [(train_index, val_index, X_train, Y_train, Classifier_list, Random_seed) for train_index, val_index in kf.split(X_train, Y_train)])

        proba = []
        proba_each_estimator = []
        for result in results:
            proba.extend(result[0])
            proba_each_estimator.append(result[1])

        proba_each_estimator = np.concatenate(proba_each_estimator, axis=1)
        proba = np.array(proba)

        # Fit the model on all training data and predict on test data
        pipe_ = get_smote_pipeline(Classifier_list)
        pipe_.fit(X_train, Y_train)

        score_train = pipe_.predict_proba(X_train)
        score_test = pipe_.predict_proba(X_test)

        score_testing_each_estimator = []
        score_training_each_estimator = []
        for i in range(len(set(Y_train))):
            x_test_ = pipe_.named_steps['imputer'].transform(X_test)
            x_test_ = pipe_.named_steps['scaler'].transform(x_test_)
            x_test_ = pipe_.named_steps['feature_selection'].transform(x_test_)
            score_test = pipe_.named_steps['classifier'].estimators_[i].predict_proba(x_test_)
            score_testing_each_estimator.append(score_test)

            x_train_ = pipe_.named_steps['imputer'].transform(X_train)
            x_train_ = pipe_.named_steps['scaler'].transform(x_train_)
            x_train_ = pipe_.named_steps['feature_selection'].transform(x_train_)
            score_train = pipe_.named_steps['classifier'].estimators_[i].predict_proba(x_train_)
            score_training_each_estimator.append(score_train)

        score_testing_each_estimator = np.array(score_testing_each_estimator)
        score_training_each_estimator = np.array(score_training_each_estimator)

        # one-vs-rest
        score_norm_test = []
        score_norm_train = []
        balanced_accuracies = []
        for dim in range(len(set(Y_train))):
            proba_binarized = proba_each_estimator[dim]
            Y_train_binarized = (Y_train == dim).astype(int)
            probs = proba_binarized[:, 1]
            fpr, tpr, thresholds = roc_curve(Y_train_binarized, probs)

            gmeans = np.sqrt(tpr * (1 - fpr))
            ix = np.argmax(gmeans)

            testing_binarized = score_testing_each_estimator[dim]
            testing_norm = proba_normalization(proba_binarized, thresholds[ix], testing_binarized)
            score_norm_test.append(testing_norm[:, 1])

            training_binarized = score_training_each_estimator[dim]
            training_norm = proba_normalization(proba_binarized, thresholds[ix], training_binarized)
            score_norm_train.append(training_norm[:, 1])

            # Calculate balanced accuracy for this class
            y_pred = (training_norm[:, 1] > 0.5).astype(int)
            balanced_acc = balanced_accuracy_score(Y_train_binarized, y_pred)
            balanced_accuracies.append(balanced_acc)

        score_norm_test = np.array(score_norm_test).T
        score_norm_test = score_norm_test / score_norm_test.sum(axis=1, keepdims=1)

        score_norm_train = np.array(score_norm_train).T
        score_norm_train = score_norm_train / score_norm_train.sum(axis=1, keepdims=1)

        return score_norm_train, score_norm_test, np.mean(balanced_accuracies)

    try:
        proba = cross_val_predict(pipe, X_train, Y_train, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=21), method='predict_proba', n_jobs=-1)
        pipe1 = get_smote_pipeline(Classifier_list)
    except:
        pipe = get_smote_pipeline(Classifier_list, k_neighbors=3)
        proba = cross_val_predict(pipe, X_train, Y_train, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=21), method='predict_proba', n_jobs=-1)
        pipe1 = get_smote_pipeline(Classifier_list, k_neighbors=3)

    if len(set(Y_train)) == 2:
        probs = proba[:, 1]
        fpr, tpr, thresholds = roc_curve(Y_train, probs)
        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)
        best_threshold = thresholds[ix]
        
        pipe1.fit(X_train, Y_train)
        score_train = pipe1.predict_proba(X_train)
        score_test = pipe1.predict_proba(X_test)
        
        score_norm_train = proba_normalization(proba, best_threshold, score_train)
        score_norm_test = proba_normalization(proba, best_threshold, score_test)
        
        # Calculate balanced accuracy
        y_pred = (proba[:, 1] > best_threshold).astype(int)
        balanced_acc = balanced_accuracy_score(Y_train, y_pred)
        
        return score_norm_train, score_norm_test, balanced_acc
    
    elif len(set(Y_train)) >= 3:
        return multiclass_probability_calibration(X_train, Y_train, X_test, Classifier_list)
    
    else:
        raise ValueError("Error: cannot handle multiclass classification task with more than 3 classes right now.")

def proba_normalization(proba, threshold, new_prob):
    if np.array(proba).ndim != 2:
        raise ValueError("Input dimension is ", proba.ndim, ", while 2 is expected.")

    max_p = max(proba[:, 1])
    min_p = min(proba[:, 1])

    if threshold == max_p or threshold == min_p:
        return new_prob

    p = []
    for x in list(new_prob):
        if x[1] >= threshold:
            p_new = 0.5 + (x[1] - threshold) / (max_p - threshold) / 2
            if p_new > 1:
                p_new = 1
            p_new = [1 - p_new, p_new]
        else:
            p_new = (x[1] - min_p) / (threshold - min_p) / 2
            if p_new < 0:
                p_new = 0
            p_new = [1 - p_new, p_new]
        p.append(p_new)

    return np.array(p)