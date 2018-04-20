#
# Created by Qi Le on April 19, 2018
#

import numpy as np
import scipy.io as io
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def my_svm(estimate, labels, validate, parameters):
    # @para estimate is an N by 60 matrix
    # @para labels is an N by 1 array
    # @para validate is an M by 60 matrix
    # @para parameters is other parameters required by svm
    # @output is an M by 5 matrix which contains the probability of each sample in each class

    svc = get_svc(parameters)
    svc_train(svc, estimate, labels)
    prob = svc_probability(svc, validate)
    return prob


def get_svc(parameters):
    svc = SVC(C=parameters['parameters']['C'], kernel=parameters['parameters']['kernel'],
              degree=parameters['parameters']['degree'], gamma=parameters['parameters']['gamma'],
              coef0=parameters['parameters']['coef0'], probability=parameters['parameters']['probability'],
              tol=parameters['parameters']['tol'], cache_size=parameters['parameters']['cache_size'],
              class_weight=parameters['parameters']['class_weight'], shrinking=parameters['parameters']['shrinking'],
              verbose=parameters['parameters']['verbose'], max_iter=parameters['parameters']['max_iter'],
              decision_function_shape=parameters['parameters']['decision_function_shape'],
              random_state=parameters['parameters']['random_state'])
    return svc


def svc_train(svc, estimate, labels):
    svc.fit(estimate, labels)


def svc_predict(svc, validate):
    return svc.predict(validate)


def svc_probability(svc, validate):
    return svc.predict_proba(validate)


def svc_score(svc, validate, v_labels):
    return svc.score(validate, v_labels)


def svc_get_para(svc):
    support = svc.support_
    support_vectors = svc.support_vectors_
    n_support = svc.n_support_
    dual_coef = svc.dual_coef_
    intercept = svc.intercept_
    sparse = svc._sparse
    shape_fit = svc.shape_fit_
    prob_a = svc.probA_
    prob_b = svc.probB_
    gamma = svc._gamma
    classes = svc.classes_
    # hyper = svc.get_params(deep=True)
    ret = {'support': support, 'support_vectors': support_vectors, 'n_support': n_support, 'dual_coef': dual_coef,
           'intercept': intercept, 'sparse': sparse, 'shape_fit': shape_fit, 'prob_a': prob_a, 'prob_b': prob_b,
           'gamma': gamma, 'classes': classes}
    return ret


def svc_set_para(svc, svc_para):
    # svc.set_params(**svc_para['hyper'])
    svc.support_ = svc_para['support']
    svc.support_vectors_ = svc_para['support_vectors']
    svc.n_support_ = svc_para['n_support']
    svc._dual_coef_ = svc_para['dual_coef']
    svc._intercept_ = svc_para['intercept']
    svc._sparse = svc_para['sparse']
    svc.shape_fit_ = svc_para['shape_fit']
    svc.probA_ = svc_para['prob_a']
    svc.probB_ = svc_para['prob_b']
    svc._gamma = svc_para['gamma']
    svc.classes_ = svc_para['classes']


if __name__ == '__main__':
    train = io.loadmat("Proj2FeatVecsSet1.mat")["Proj2FeatVecsSet1"]
    label = np.argmax(io.loadmat("Proj2TargetOutputsSet1.mat")["Proj2TargetOutputsSet1"], axis=1)
    e_train, _, e_label, _ = train_test_split(train, label, train_size=0.1, random_state=0)
    _, v_train, _, v_label = train_test_split(train, label, test_size=0.2, random_state=0)
    s_para = {'C': 1.0, 'kernel': 'rbf', 'degree': 3, 'gamma': 'auto', 'coef0': 0.0, 'probability': True,
              'shrinking': True, 'tol': 1e-3, 'cache_size': 800, 'class_weight': 'balanced', 'verbose': False,
              'max_iter': -1, 'decision_function_shape': 'ovr', 'random_state': None}
    # res = my_svm(e_train, e_label, v_train, {'parameters': s_para})
    c_svc = get_svc({'parameters': s_para})
    svc_train(c_svc, e_train, e_label)
    trained_para = svc_get_para(c_svc)
    n_svc = SVC(s_para)
    svc_set_para(n_svc, trained_para)
    # n_svc.set_params(**svc.get_params())
    print(svc_score(c_svc, v_train, v_label))
    print(svc_score(n_svc, v_train, v_label))
