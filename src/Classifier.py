import numpy as np
from sklearn import svm
import pylab as plt


class Classifier:
    """
        __author__ = Nicolas Perez-Nieves
        __email__ = nicolas.perez14@imperial.ac.uk
        
        This class implements a classifier which is to learn from the features extracted by
        the SDNN.
        
        It trains a SVM by default
        
        It allows to perform cross-validation with any of the parameters specified in 
        the dictionary classifier_params    
    
    """
    def __init__(self, X_train, y_train, X_test, y_test, classifier_params, classifier_type='SVM'):
        """
            Classifier initialization routine
            
            Inputs:
            - X_train: A numpy array containing the training data of shape (N, D)
                     Each row corresponds to a sample, each column to a feature
            - y_train: A numpy array containing the labels for the training data of shape (N, 1)
            - X_test: A numpy array containing the test data of shape (M, D)
                    Each row corresponds to a sample, each column to a feature
            - y_test: A numpy array containing the labels for the test data of shape (M, 1)
            - classifier_params: A dictionary containing the parameters for the specific classifier to be used
            - classifier_type: A string specifying the classifier type to be used (SVM by default)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.classifier_type = classifier_type
        self.classifier_params = classifier_params

        self.classifier = []
        self.train_score = []
        self.test_score = []

        self.cvs_mean = []  # Cross validation scores
        self.cvs_std = []  # Cross validation std
        self.cval_param = {}  # Best cross validated parameter
        self.plots = []

    def train_classifier_svm(self):
        """
            Trains a SVM classifier
            
            The parameters C, gamma, kernel and prob_flag specified under self.classifier_params are used.
            
            Each classifier instance is appended to self.classifier
            The training error is appended to self.train_error  
        """
        if(self.classifier_type=='SVM'):
            try:
                C = self.classifier_params['C']
            except:
                if 'C' in self.cval_param:
                    C = self.cval_param['C']
                else:
                    C = 1.0
                print('C was not specified')
            try:
                gamma = self.classifier_params['gamma']
            except:
                if 'gamma' in self.cval_param:
                    gamma = self.cval_param['gamma']
                else:
                    gamma = 'auto'
                print('gamma was not specified')
            try:
                kernel = self.classifier_params['kernel']
            except:
                kernel = 'rbf'
                print('kernel was not specified')
            try:
                prob_flag = self.classifier_params['prob_flag']
            except:
                prob_flag = False
                print('probability_flag was not specified')

            # Train SVM
            clf = svm.SVC(C=C, gamma=gamma, kernel=kernel, probability=prob_flag)
            clf.fit(self.X_train, self.y_train)
            self.classifier.append(clf)

            # Obtain the Training Error
            self.train_score.append(clf.score(self.X_train, self.y_train))
        else:
            print("Error, SVM classifier not specified")

    def test_classifier_svm(self):
        """
            Evaluates the test score using the last classifier trained and appends it to self.test_error   
        """
        try:
            clf = self.classifier[-1]
        except:
            print('No classifier has been trained')
        self.test_score.append(clf.score(self.X_test, self.y_test))


    def run_classiffier(self):
        """
            Trains the SVM with self.X_train and self.y_train data 
            and tests it with self.X_test and self.y_test data
            
            Returns a tuple containing two doubles for the train and test error respectively    
        """
        if (self.classifier_type == 'SVM'):
            self.train_classifier_svm()
            self.test_classifier_svm()
            return self.train_score[-1], self.test_score[-1]

    def cross_val_svm(self, cv_param, cv):
        """
            This method computes the cross validation error fo a svm classifier
             
            Input:
                - cv_param: A dictionary with one single key specifying the parameter 
                            to cross validate with respect to. The key must be either 'C' or 'gamma'.
                -cv: An integer that specifies the number of k-folds
                -plot: A flag to specify if the results will be plotted or not
            
            Returns a tuple of:
                - param : An numpy array with the cross validated parameter
                - cve_mean: An numpy array with the cross-validation error per parameter value
                - cve_std: An numpy array with the cross-validation std per parameter value
                - cv: An integer specifying the number of k-folds used
        """

        from sklearn.model_selection import cross_val_score
        if len(cv_param) > 1:
            print('Crossvalidation w.r.t one parameter only')
            return

        # Check which parameter is to be cross validated
        try:
            C = cv_param['C']
            N = len(C)
        except:
            C = 1.0
        try:
            gamma = cv_param['gamma']
            N = len(gamma)
        except:
            gamma = 'auto'

        # Cross-validate, calculate the cross-validation score and std
        # and append it to the self.cve_mean and self.cve_std
        for i in range(N):
            if type(C) is list:
                clf = svm.SVC(C=C[i], gamma=gamma, kernel='rbf')
            elif type(gamma) is list:
                clf = svm.SVC(C=C, gamma=gamma[i], kernel='rbf')
            else:
                print('Parameters should be specified as a list')
                return
            scores = cross_val_score(clf, self.X_train, self.y_train, cv=cv)
            self.cvs_mean.append(scores.mean())
            self.cvs_std.append(scores.std())

        # Plot the cross validation error and save the best parameter
        if type(C) is list:
            # Save the best
            self.cval_param['C'] = C[np.argmax(np.array(self.cvs_mean))]
            # Plot
            x = np.array(C)
            y = np.array(self.cvs_mean)
            e = np.array(self.cvs_std)
            fig = plt.plot(x, y, 'k-')
            plt.fill_between(x, y - e, y + e)
            plt.title('%s-fold Cross Validation error' %cv)
            plt.xlabel('C')
            plt.ylabel('CVE')
            self.plots.append(fig)
            return np.array(C), np.array(self.cvs_mean), np.array(self.cvs_std), cv
        else:
            # Save the best
            self.cval_param['gamma'] = gamma[np.argmax(np.array(self.cvs_mean))]
            # Plot
            x = np.array(gamma)
            y = np.array(self.cvs_mean)
            e = np.array(self.cvs_std)
            fig = plt.plot(x, y, 'k-')
            plt.fill_between(x, y - e, y + e)
            plt.title('%s-fold Cross Validation error' %cv)
            plt.xlabel('gamma')
            plt.ylabel('CVE')
            self.plots.append(fig)
            return np.array(gamma), np.array(self.cvs_mean), np.array(self.cvs_std), cv
