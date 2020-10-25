import argparse
import numpy as np
import matplotlib.pyplot as pl
from time import time


# Import the random forest package
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from sklearn.decomposition import PCA
from sklearn import svm

__author__ = 'gulinvladimir'

def main():
    args = parse_args()

    train_data = np.loadtxt(args.train)
    test_data  = np.loadtxt(args.test)

    total_data = np.concatenate(([train_data, test_data]), axis=0)

    # Visualizations of data
    #visualize_classes(total_data[0::, 1::], total_data[0::, 0])
    #visualize_data(total_data[0::, 1::], len(train_data[:,0]), len(test_data[:,0]))

    number_of_features = len(train_data[0, :])

    use_features_in_tree = (int)(args.features_percent * number_of_features)

    # Create the random forest classifier
    print ("Build random forest classifier...")
    forest = RandomForestClassifier(n_estimators = args.trees, max_features=use_features_in_tree)
    sta = time()
    forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
    sto = time()
    prediction_randomforest = forest.predict(test_data[0::, 1::])

    #Create adaboost classifier
    print ("Build adaboost classifier...")
    adaboost_classifier = AdaBoostClassifier(n_estimators=args.trees)
    adaboost_classifier.fit(train_data[0::, 1::], train_data[0::, 0])
    prediction_adaboost = adaboost_classifier.predict(test_data[0::, 1::])

    #Create gbm classifier
    print ("Build gbm classifier...")
    gbm_classifier = GradientBoostingClassifier(n_estimators=args.trees)
    gbm_classifier.fit(train_data[0::, 1::], train_data[0::, 0])
    prediction_gbm = gbm_classifier.predict(test_data[0::, 1::])

    #Create svm classifier
    print ("Build svm classifier...")
    svm_classifier = svm.SVC()
    svm_classifier.fit(train_data[0::, 1::], train_data[0::, 0])
    prediction_svm = svm_classifier.predict(test_data[0::, 1::])
    
    #Create XGBoost classifier
    print ("Build XGBoost classifier...")
    xgb_classifier = XGBClassifier(n_estimators = args.trees)
    xgb_classifier.fit(train_data[0::, 1::], train_data[0::, 0])
    prediction_xgb = xgb_classifier.predict(test_data[0::, 1::])
    
    #Create DART classifier
    print ("Build DART classifier...")
    dart_classifier = LGBMClassifier(boosting_type='dart', n_estimators = args.trees)
    dart_classifier.fit(train_data[0::, 1::], train_data[0::, 0])
    prediction_dart = dart_classifier.predict(test_data[0::, 1::])
    
    print (classification_report(test_data[0::, 0], prediction_randomforest, 'Random Forest'))
    print (classification_report(test_data[0::, 0], prediction_adaboost, 'AdaBoost'))
    print (classification_report(test_data[0::, 0], prediction_gbm, 'Gradient Boosting Machine'))
    print (classification_report(test_data[0::, 0], prediction_svm, 'SVM'))
    print (classification_report(test_data[0::, 0], prediction_xgb, 'XGB'))
    print (classification_report(test_data[0::, 0], prediction_dart, 'DART'))
    
    nums = [5, 10, 50, 100, 250, 500]
    rf = [[],[]]
    ada = [[],[]]
    gbm = [[],[]]
    xgb = [[],[]]
    dart = [[],[]]
    for i in nums:
        forest = RandomForestClassifier(n_estimators = i, max_features=use_features_in_tree)
        sta = time()
        forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
        sto = time()
        prediction_randomforest = forest.predict(test_data[0::, 1::])
        rf[0].append(sto - sta)
        rf[1].append((float)(classification_report(test_data[0::, 0], prediction_randomforest).split()[-2]))
        adaboost_classifier = AdaBoostClassifier(n_estimators=i)
        sta = time()
        adaboost_classifier.fit(train_data[0::, 1::], train_data[0::, 0])
        sto = time()
        prediction_adaboost = adaboost_classifier.predict(test_data[0::, 1::])
        ada[0].append(sto - sta)
        ada[1].append((float)(classification_report(test_data[0::, 0], prediction_adaboost).split()[-2]))
        gbm_classifier = GradientBoostingClassifier(n_estimators=i)
        sta = time()
        gbm_classifier.fit(train_data[0::, 1::], train_data[0::, 0])
        sto = time()
        prediction_gbm = gbm_classifier.predict(test_data[0::, 1::])
        gbm[0].append(sto - sta)
        gbm[1].append((float)(classification_report(test_data[0::, 0], prediction_gbm).split()[-2]))
        xgb_classifier = XGBClassifier(n_estimators = i)
        sta = time()
        xgb_classifier.fit(train_data[0::, 1::], train_data[0::, 0])
        sto = time()
        prediction_xgb = xgb_classifier.predict(test_data[0::, 1::])
        xgb[0].append(sto - sta)
        xgb[1].append((float)(classification_report(test_data[0::, 0], prediction_xgb).split()[-2]))
        dart_classifier = LGBMClassifier(boosting_type='dart', n_estimators = i)
        sta = time()
        dart_classifier.fit(train_data[0::, 1::], train_data[0::, 0])
        sto = time()
        prediction_dart = dart_classifier.predict(test_data[0::, 1::])
        dart[0].append(sto - sta)
        dart[1].append((float)(classification_report(test_data[0::, 0], prediction_dart).split()[-2]))
        
    fig = pl.figure(figsize=(16, 16))
    pl.subplot(211)
    pl.plot(nums, rf[0], color='red', label='rf time')
    pl.plot(nums, ada[0], color='blue', label='ada time')
    pl.plot(nums, gbm[0], color='orange', label='gbm time')
    pl.plot(nums, xgb[0], color='yellow', label='xgb time')
    pl.plot(nums, dart[0], color='pink', label='dart time')
    pl.legend()
    pl.subplot(212)
    pl.plot(nums, rf[1], color='red', label='rf score')
    pl.plot(nums, ada[1], color='blue', label='ada score')
    pl.plot(nums, gbm[1], color='orange', label='gbm score')
    pl.plot(nums, xgb[1], color='yellow', label='xgb score')
    pl.plot(nums, dart[1], color='pink', label='dart score')
    pl.legend()
    pl.show()
        


def visualize_classes(total_data, response):
    ''' Visualization of total spam data
    :param total_data: Train and test data
    :param response: answers
    :return:
    '''
    pca = PCA(n_components=2)
    projection = pca.fit_transform(total_data)

    not_spam_class = np.where(response == 0)
    spam_class = np.where(response == 1)

    fig = pl.figure(figsize=(8, 8))
    pl.rcParams['legend.fontsize'] = 10

    pl.plot(projection[not_spam_class, 0], projection[not_spam_class, 1],
            'o', markersize=7, color='blue', alpha=0.5, label='Not spam')

    pl.plot(projection[spam_class, 0], projection[spam_class, 1],
            'o', markersize=7, color='red', alpha=0.5, label='Spam')

    pl.title('Spam data')
    pl.show()


def visualize_data(total_data, train_size, test_size):
    ''' Visualization of total spam data
    :param total_data: Train and test data
    :param train_size: Size of train set
    :param test_size: Size of test set
    :return:
    '''
    pca = PCA(n_components=2)
    projection = pca.fit_transform(total_data)

    fig = pl.figure(figsize=(8, 8))

    pl.rcParams['legend.fontsize'] = 10
    pl.plot(projection[0:train_size, 0], projection[0:train_size, 1],
            'o', markersize=7, color='blue', alpha=0.5, label='Train')
    pl.plot(projection[train_size:train_size+test_size, 0], projection[train_size:train_size+test_size, 1],
            'o', markersize=7, color='red', alpha=0.5, label='Test')
    pl.title('Spam data')
    pl.show()


def classification_report(y_true, y_pred, alg_name=None):
    ''' Computes clasification metrics

    :param y_true - original class label
    :param y_pred - predicted class label
    :return presicion, recall for each class; micro_f1 measure, macro_f1 measure
    '''

    if (alg_name != None):
        print (alg_name + " :")

    last_line_heading = 'avg / total'
    final_line_heading = 'final score'

    labels = unique_labels(y_true, y_pred)

    width = len(last_line_heading)
    target_names = ['{0}'.format(l) for l in labels]

    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None)

    f1_macro = 0
    precision_macro = 0
    recall_macro = 0

    for i, label in enumerate(labels):
        values = [target_names[i]]
        f1_macro += f1[i]
        precision_macro += p[i]
        recall_macro += r[i]
        for v in (p[i], r[i], f1[i]):
            values += ["{0:0.5f}".format(v)]
        values += ["{0}".format(s[i])]
        report += fmt % tuple(values)

    report += '\n'

    # compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["{0:0.5f}".format(v)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)

    values = [final_line_heading]
    for v in (precision_macro, recall_macro, f1_macro):
        values += ["{0:0.5f}".format(v / labels.size)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)

    return report

def parse_args():
    parser = argparse.ArgumentParser(description='Random Forest Tutorial')
    parser.add_argument("-tr", "--train", action="store", type=str, help="Train file name")
    parser.add_argument("-te", "--test", action="store", type=str, help="Test file name")
    parser.add_argument("-t", "--trees", action="store", type=int, help="Number of trees in random forest", default=5)
    parser.add_argument("-fp", "--features_percent", action="store", type=float, help="Percent of features in each tree", default=0.9)
    return parser.parse_args()

if __name__ == "__main__":
    main()