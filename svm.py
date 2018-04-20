#
# Created by Qi Le on April 19, 2018
#

import scipy.io as io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def my_svm(estimate, labels, validate, parameters):
    # @para estimate is an N by 60 matrix
    # @para labels is an N by 1 array
    # @para validate is an M by 60 matrix
    # @para parameters is other parameters required by svm
    # @output is an M by 5 matrix which contains the probability of each sample in each class

    svc = SVC(C=parameters['parameters']['C'], kernel=parameters['parameters']['kernel'],
              degree=parameters['parameters']['degree'], gamma=parameters['parameters']['gamma'],
              coef0=parameters['parameters']['coef0'], probability=parameters['parameters']['probability'],
              tol=parameters['parameters']['tol'], cache_size=parameters['parameters']['cache_size'],
              class_weight=parameters['parameters']['class_weight'], shrinking=parameters['parameters']['shrinking'],
              verbose=parameters['parameters']['verbose'], max_iter=parameters['parameters']['max_iter'],
              decision_function_shape=parameters['parameters']['decision_function_shape'],
              random_state=parameters['parameters']['random_state'])
    svc.fit(estimate, labels)
    return svc.predict_proba(validate)


if __name__ == '__main__':
    train = io.loadmat("Proj2FeatVecsSet1.mat")["Proj2FeatVecsSet1"]
    label = np.argmax(io.loadmat("Proj2TargetOutputsSet1.mat")["Proj2TargetOutputsSet1"], axis=1)
    e_train, _, e_label, _ = train_test_split(train, label, train_size=0.2, random_state=0)
    _, v_train, _, v_label = train_test_split(train, label, test_size=0.2, random_state=0)
    s_para = {'C': 1.0, 'kernel': 'rbf', 'degree': 3, 'gamma': 'auto', 'coef0': 0.0, 'probability': True,
              'shrinking': True, 'tol': 1e-3, 'cache_size': 800, 'class_weight': 'balanced', 'verbose': False,
              'max_iter': -1, 'decision_function_shape': 'ovr', 'random_state': None}
    res = my_svm(e_train, e_label, v_train, {'parameters': s_para})