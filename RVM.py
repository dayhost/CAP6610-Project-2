#from skrvm import RVC
from sklearn import datasets
from skbayes.rvm_ard_models import RegressionARD,ClassificationARD,RVR,RVC
from sklearn.svm import SVR
import scipy.io as io
from sklearn.utils.extmath import pinvh,log_logistic,safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.metrics.pairwise import pairwise_kernels

iris = datasets.load_iris()

import numpy as np

train = io.loadmat('Proj2FeatVecsSet1.mat')
label = io.loadmat('Proj2TargetOutputsSet1.mat')
train_data = train['Proj2FeatVecsSet1']
label_data = label['Proj2TargetOutputsSet1']
label_data = np.argmax(label_data, axis=1)


#================== To be discarded ================
def RVM(XEstimate,ClassLabels, XValidate, Parameters):
    clf = RVC(verbose='true')
    clf.fit(XEstimate, ClassLabels)
    Yvalidate = clf.predict(XValidate)
    EstParameters = clf.get_params()
    return Yvalidate,EstParameters



#=================== Function to be used ============
def RVM2(XEstimate,ClassLabels, XValidate, Parameters):
    clf = RVC(n_iter = Parameters.get('n_iter'),
    tol = Parameters.get('tol'),
    n_iter_solver = Parameters.get('n_iter_solver'),
    tol_solver = Parameters.get('tol_solver'),
    fit_intercept = Parameters.get('fit_intercept'),
    verbose = Parameters.get('verbose'),
    kernel = Parameters.get('kernel'),
    degree = Parameters.get('degree'),
    gamma = Parameters.get('gamma'),
    coef0 = Parameters.get('coef0'),
    kernel_params = Parameters.get('kernel_params') )
    clf.fit(XEstimate, ClassLabels)
    Yvalidate = clf.predict_proba(XValidate)
    EstParameters = { 'relevant_vectors':clf.relevant_vectors_ ,'coef':clf.coef_,'active':clf.active_,'intercept':clf.active_}
    return Yvalidate,EstParameters




def _decision_function_active(normalize, X, coef_, active_, intercept_):
        ''' Constructs decision function using only relevant features '''
        if normalize:
            X = (X - np.mean(X,0)[active_]) / np.std(X,0)[active_]
        decision = safe_sparse_dot(X, coef_[active_]) + intercept_
        return decision



def get_kernel( X, Y, gamma, degree, coef0, kernel, kernel_params ):
    if callable(kernel):
        params = kernel_params or {}
    else:
        params = {"gamma": gamma,
                  "degree": degree,
                  "coef0": coef0  }
    return pairwise_kernels(X, Y, metric=kernel,
                            filter_params=True, **params)



def TestMyClassifier(XTest,Parameters,EstParameters):
    #check_is_fitted('coef_')
    XTest = check_array(XTest, accept_sparse=None, dtype=np.float64)
    n_features = EstParameters.get('relevant_vectors')[0].shape[1]
    if XTest.shape[1] != n_features:
        raise ValueError("X has %d features per sample; expecting %d"
                         % (XTest.shape[1], n_features))
    kernel = lambda rvs: get_kernel(XTest, rvs, Parameters.get('gamma'), Parameters.get('degree'),
                                    Parameters.get('coef0'), Parameters.get('kernel'), Parameters.get('kernel_params'))
    decision = []
    for rv, cf, act, b in zip(EstParameters.get('relevant_vectors'), EstParameters.get('coef'), EstParameters.get('active'),
                              EstParameters.get('intercept')):
        # if there are no relevant vectors => use intercept only
        if rv.shape[0] == 0:
            decision.append(np.ones(XTest.shape[0]) * b)
        else:
            decision.append(_decision_function_active(Parameters.get('normalize'),kernel(rv), cf, act, b))
    prob = np.asarray(decision).squeeze().T
    if prob.ndim == 1:
        prob = np.vstack([1 - prob, prob]).T
    prob = prob / np.reshape(np.sum(prob, axis=1), (prob.shape[0], 1))
    Ytest = prob;
    return Ytest


Parameters = { 'n_iter':100, 'tol':1e-4, 'n_iter_solver':15, 'tol_solver':1e-4,
                 'fit_intercept':True, 'verbose': False, 'kernel':'rbf', 'degree': 2,
                 'gamma': None, 'coef0':1, 'kernel_params':None }


#RVM2(train_data,label_data,train_data,Parameters)
Yvalidate,EstParameters = RVM2(iris.data, iris.target, iris.data ,Parameters)



print(TestMyClassifier(iris.data,Parameters,EstParameters))