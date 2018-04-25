import numpy as np


from time import sleep
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.gaussian_process.kernels import RBF

from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

from TrainMyClassifier import *
from MyConfusionMatrix import *

def CVhelper(args):
    XEstimate=args[0]
    XValidate= args[1]
    YEstimate=args[2]
    YValidateTrue=args[3]
    algo=args[4]
    params=args[5]

    # calling TrainMyClassifier
    YValidate, Estparameters = TrainMyClassifier(XEstimate, YEstimate, XValidate,
                                                 {'algorithm':algo, 'parameters':params })


    score = (np.sum(YValidateTrue == np.argmax(YValidate, axis=1))/np.size(YValidateTrue))*100
    return score

def MyCrossValidate(XTrain, ClassLabels, Nf):
    """
    Xtrain: Training data with labels
    ClassLabels: Class labels for train set.
    Nf: Number of folds

    returns:
            Array of Ytrain:
            Array of EstParameters:
            Array of EstConfMatrices:
            Array of ConfMatrix:
    """

    algorithms = ['RVM']
    parameters = {'SVM':{'C' : [1, 5, 10], 'kernel' : ['rbf','poly'], 'degree' : [2, 3, 5], 'gamma' : ['auto'],
                            'coef0' : [0.0], 'probability' : [True], 'shrinking' : [True], 'tol' : [1e-3, 1e-4],
                            'class_weight' : ['balanced'], 'verbose' : [False], 'max_iter' : [-1],
                            'decision_function_shape' : ['ovo'], 'random_state' : [None], 'cache_size': [800]},

                 'RVM':{ 'kernel':['rbf'],'degree':[3],'coef1':[None],'coef0':[0.0],'n_iter':[5],'tol':[1e-3],
                        'alpha':[100,1,1e-6],'threshold_alpha':[1e9],'beta':[1.e-2,1.e-6,1.e-8],'beta_fixed':[False],
                        'bias_used':[True],'verbose':[False]},
                 'GPR':{"kernel": [ 1.0 * RBF(1),
                                    2.0 * RBF(1),
                                    3.0 * RBF(1),
                                    1.0 * RBF(2),
                                    1.0 * RBF(3)
                                    ],
                              "optimizer": ["fmin_l_bfgs_b"],
                              "n_restarts_optimizer": [0],
                              "max_iter_predict": [100],
                              "warm_start": [True],
                              "copy_X_train": [True],
                              "random_state": [0],
                              "multi_class": ["one_vs_one"],
                              "n_jobs": [-1]}}
    # decode ClassLabels to numbers
    ClassLabels = np.argmax(ClassLabels, axis=1)


    algo_score = []
    algo_params = []

    Ytrain = []
    EstParameter = []
    EstConfMatrices = []
    ConfMatrix = []
    for algo in algorithms:

        # generating hyper parameter array for hyper parameter search.
        grid = ParameterGrid(parameters[algo])
        grid_search_score = []
        pbar = tqdm(list(grid))
        for params in pbar:
            pbar.set_description("Searching Parameters for {}".format(algo))
            # scikit-learn object to divide data set in Estimate and Validate sets.
            k_fold = KFold(n_splits=Nf, random_state=None, shuffle=False)

            # Array of scores for each split.
            cv_scores = []
            p = Pool(8)
            p_params =[]
            for train_index, val_index in k_fold.split(XTrain):

                # Spliting data in Estimate and validate set.
                XEstimate, XValidate = XTrain[train_index], XTrain[val_index]
                YEstimate, YValidateTrue = ClassLabels[train_index], ClassLabels[val_index]
                p_params.append([XEstimate, XValidate, YEstimate, YValidateTrue, algo, params])

            cv_scores = p.map(CVhelper, p_params)

            # average accuracy for selected hyper parameters
            score = np.mean(cv_scores)
            grid_search_score.append(score)

        # calculating best hyper parameters
        idx = np.argmax(grid_search_score)
        best_params = list(grid)[idx]

        # storing best algorithm score and hyper parameters
        algo_score.append(np.max(grid_search_score))
        algo_params.append(best_params)


        # calculating method out for current algorithm
        _Ytrain = []
        _EstParameter = []
        _EstConfMatrices = []
        _ConfMatrix = []

        # scikit-learn object to divide data set in Estimate and Validate sets.
        k_fold = KFold(n_splits=Nf, random_state=None, shuffle=False)

        _all_Yval_true = []
        _all_Yval_pred = []
        index = 0

        print("Average cross validation score for {} is {}%.".format(algo, np.max(grid_search_score)))
        for train_index, val_index in k_fold.split(XTrain):

            # Spliting data in Estimate and validate set.
            XEstimate, XValidate = XTrain[train_index], XTrain[val_index]
            YEstimate, YValidateTrue = ClassLabels[train_index], ClassLabels[val_index]

            # calling TrainMyClassifier
            YValidate, Estparameters = TrainMyClassifier(XEstimate, YEstimate, XValidate,
                                                         {'algorithm':algo, 'parameters':best_params })

            # storing TrainMyClassifier's output
            _Ytrain.append(YValidate)
            _EstParameter.append(Estparameters)

            # printing kernel numbers
            if  algo == 'SVM':
                print('Number of support vector for validation set {}: {}'.format(index, len(Estparameters['support_vectors'])))
            elif  algo == 'RVM':
                if len(Estparameters) == 1:
                    relevant_vectors = np.shape(Estparameters.get('clf').relevance_)[0]
                else:
                    relevant_vectors = 0
                    for clf in Estparameters[0].multi_.estimators_:
                        relevant_vectors = relevant_vectors + np.shape(clf.relevance_)[0]
                    relevant_vectors = relevant_vectors/2
                print('Number of relavance vector for validation set {}: {}'.format(index, relevant_vectors))

            # calculating confusion matrix for validation set
            print("Confusion matrix for validation set {}: ".format(index))
            _EstConfMatrices.append(MyConfusionMatrix(np.argmax(YValidate, axis=1), YValidateTrue))


            _all_Yval_true.extend(YValidateTrue.tolist())
            _all_Yval_pred.extend(np.argmax(YValidate, axis=1).tolist())

            index+=1
        print("Over all Confusion matrix for {}: ".format(algo))
        # calculating over all confusion matrix for validation set
        _all_Yval_pred = np.array(_all_Yval_pred).reshape(len(_all_Yval_pred),1)
        _all_Yval_true = np.array(_all_Yval_true).reshape(len(_all_Yval_true),1)
        _ConfMatrix = MyConfusionMatrix(_all_Yval_pred, _all_Yval_true)

        # append Main outputs
        Ytrain.append(_Ytrain)
        EstParameter.append(_EstParameter)
        EstConfMatrices.append(_EstConfMatrices)
        ConfMatrix.append(_ConfMatrix)

        # waiting for all print commands to execute.
        sleep(.5)

    return Ytrain, EstParameter, EstConfMatrices, ConfMatrix
