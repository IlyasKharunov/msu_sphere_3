import argparse
import numpy as np
import matplotlib.pyplot as pl
import time


# Import the random forest package
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from sklearn.decomposition import PCA

# Import pytorch package
import torch

__author__ = 'gulinvladimir'

def main():
    args = parse_args()

    train_data = np.loadtxt(args.train)
    test_data  = np.loadtxt(args.test)

    total_data = np.concatenate(([train_data, test_data]), axis=0)

    visualize_data(total_data[0::, 1::], len(train_data[:,0]), len(test_data[:,0]))

    number_of_features = len(train_data[0, :])

    use_features_in_tree = (int)(args.features_percent * number_of_features)
    
    if args.fit_params == True:
        tree_numbers = [10, 50, 100, 200]
        #, 400, 800, 1200, 1600, 2000
        treescore = []
        percent_numbers = [ 0.2, 0.5,  0.7,  0.9]
        # , 0.3, 0.4 0.1,0.6,0.8,
        perscore = []
        for i in tree_numbers:
            forest = RandomForestClassifier(n_estimators = i, max_features=0.2)
            # Fit the training data to the Survived labels and create the decision trees
            forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
            # Take the same decision trees and run it on the test data
            prediction = forest.predict(test_data[0::, 1::])
            fscore = (float)(classification_report(test_data[0::, 0], prediction).split()[-2])
            treescore.append(fscore)
            
        for i in percent_numbers:
            forest = RandomForestClassifier(n_estimators = 500, max_features=i)
            # Fit the training data to the Survived labels and create the decision trees
            forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
            # Take the same decision trees and run it on the test data
            prediction = forest.predict(test_data[0::, 1::])
            fscore = (float)(classification_report(test_data[0::, 0], prediction).split()[-2])
            perscore.append(fscore)
        
        f = lambda i: perscore[i]
        maxperidx = max(range(len(perscore)), key=f)
        print('best percent: ', percent_numbers[maxperidx])
        f = lambda i: treescore[i]
        maxtreeidx = max(range(len(treescore)), key=f)
        print('best treenum: ', tree_numbers[maxtreeidx])

        fig = pl.figure(figsize=(16, 16))
        pl.subplot(211)
        pl.plot(tree_numbers, treescore)
        pl.subplot(212)
        pl.plot(percent_numbers, perscore)
        pl.show()
    else:
        startrf = time.time()
        # Create the random forest object which will include all the parameters
        # for the fit
        forest = RandomForestClassifier(n_estimators = args.trees, max_features=use_features_in_tree)
    
        # Fit the training data to the Survived labels and create the decision trees
        forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
    
        # Take the same decision trees and run it on the test data
        prediction = forest.predict(test_data[0::, 1::])
        #print(prediction[0:20])
    
        print('RF:\n',classification_report(test_data[0::, 0], prediction))
        endrf = time.time()
        print('time: ',endrf - startrf)
        
        startnn = time.time()
        # My code:
        # Neural network
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        D_in = train_data.shape[1] - 1
        D_latent = (train_data.shape[1] - 1)*4
        D_out = 2
        
        x_train = torch.from_numpy(train_data[0::, 1::]).float().to(device)
        y_train = torch.from_numpy(train_data[0::, 0]).long().to(device)
        #print(y_train[0])
        
        model = torch.nn.Sequential(
                  torch.nn.Linear(D_in, D_latent),
                  torch.nn.ReLU(),
                  torch.nn.Linear(D_latent, D_out),
                  torch.nn.LogSoftmax(dim=1)
                )
        
        model.to(device)
        
        loss_fn = torch.nn.NLLLoss()
        
        learning_rate = 1e-2
        for t in range(500):
            prediction = model(x_train)
        
            loss = loss_fn(prediction, y_train)
        
            model.zero_grad()
        
            loss.backward()
        
            for param in model.parameters():
                param.data -= learning_rate * param.grad.data
        
        x_test = torch.from_numpy(test_data[0::, 1::]).float().to(device)
        prediction = model(x_test).detach().cpu().numpy()
        #print(prediction.dtype,'\n',prediction[0:20],prediction.shape)
        prediction = np.where(prediction[:,1] > prediction[:,0], 1., 0.)
        
        print('NN:\n',classification_report(test_data[0::, 0], prediction))
        endnn = time.time()
        print('time: ',endnn - startnn)
        
        startknn = time.time()
        # KNN
        kneighbors = KNeighborsClassifier()
        kneighbors.fit(train_data[0::, 1::], train_data[0::, 0])
        prediction = kneighbors.predict(test_data[0::, 1::])
    
        print('KNN:\n',classification_report(test_data[0::, 0], prediction))
        endknn = time.time()
        print('time: ',endknn - startknn)
    

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


def classification_report(y_true, y_pred):
    ''' Computes clasification metrics

    :param y_true - original class label
    :param y_pred - predicted class label
    :return presicion, recall for each class; micro_f1 measure, macro_f1 measure
    '''
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
    parser.add_argument("-t", "--trees", action="store", type=int, help="Number of trees in random forest", default=10)
    parser.add_argument("-fp", "--features_percent", action="store", type=float, help="Percent of features in each tree", default=0.9)
    parser.add_argument("-f","--fit_params", action="store", type=bool, help="fit number of trees and percent of features", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    main()