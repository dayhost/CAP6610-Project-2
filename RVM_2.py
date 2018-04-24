from skbayes.rvm_ard_models import RVC
from multiprocessing import Pool
"""
Parameter range
    {
    "kernel": [ 1.0 * RBF(1), 
            2.0 * RBF(1),
            3.0 * RBF(1),
            1.0 * RBF(2),
            1.0 * RBF(3)
            ]
    "optimizer": ["fmin_l_bfgs_b"],
    "n_restarts_optimizer": [0],
    "max_iter_predict": [100],
    "warm_start": [True],
    "copy_X_train": [True],
    "random_state": [0],
    "multi_class": ["one_vs_one"],
    "n_jobs": [-1]
    }
"""

core_num = 4


def train_classifier(tuple_data):
    return tuple_data[0].fit(tuple_data[1], tuple_data[2])

def TrainMyClassifier(XEstimate, ClassLabels, XValidate, Parameters):
    """
    This function will return all the
    :param XEstimate: (data, label), data in [N, 60] shape
    :param ClassLabels: label in [N] shape
    :param XValidate: data in shape [N, 60]
    :param Parameters:
    {
        "algorithm": "RVM",
        "parameters": {
            "kernel": 6,
            "optimizer": "fmin_l_bfgs_b",
            "n_restarts_optimizer": 0,
            "max_iter_predict": 100,
            "warm_start": True,
            "copy_X_train": True,
            "random_state": 0,
            "multi_class": "one_vs_one",
            "n_jobs": -1
        }
    }
    :return:
    """
    if Parameters["algorithm"] == "RVM":
        # get the classes from the labels
        classes = np.unique(ClassLabels, axis=0)
        sorted(classes, reverse=True)
        num_class = len(classes)

        # get data and label based on classes
        data = []
        for cla in classes:
            data.append(XEstimate[ClassLabels == cla])

        target = []
        for cla in classes:
            target.append(ClassLabels[ClassLabels == cla])

        # put data and label into a matrix, so that we could do a easier calculation for probability
        # the following calculation is all based on the matrix
        data_matrix = []
        for i in range(num_class - 1):
            data_matrix.append([])
            for j in range(num_class - 1):
                data_matrix[i].append(None)

        target_matrix = []
        for i in range(num_class - 1):
            target_matrix.append([])
            for j in range(num_class - 1):
                target_matrix[i].append(None)

        for i in range(num_class-1):
            for j in range(i, num_class-1):
                data_matrix[i][j] = np.concatenate([data[i], data[j+1]], axis=0)
                target_matrix[i][j] = np.concatenate([target[i], target[j+1]], axis=0)

        classifier_matrix = []
        for i in range(num_class-1):
            classifier_matrix.append([])
            for j in range(num_class-1):
                classifier_matrix[i].append(None)

        pool_list = []
        for i in range(num_class-1):
            for j in range(i, num_class-1):
                rvm_classifier = RVC(
                    n_iter=Parameters["parameters"]['n_iter'],
                    tol=Parameters["parameters"]['tol'],
                    n_iter_solver=Parameters["parameters"]['n_iter_solver'],
                    tol_solver=Parameters["parameters"]['tol_solver'],
                    fit_intercept=Parameters["parameters"]['fit_intercept'],
                    verbose=Parameters["parameters"]['verbose'],
                    kernel=Parameters["parameters"]['kernel'],
                    degree=Parameters["parameters"]['degree'],
                    gamma=Parameters["parameters"]['gamma'],
                    coef0=Parameters["parameters"]['coef0'],
                    kernel_params=Parameters["parameters"]['kernel_params']
                )
                # rvm_classifier.fit(data_matrix[i][j], target_matrix[i][j])
                pool_list.append((rvm_classifier, data_matrix[i][j], target_matrix[i][j]))
                classifier_matrix[i][j] = rvm_classifier

        with Pool(processes=core_num) as pool:
            classifier_list = pool.map(train_classifier, pool_list)

        classifier_counter = 0
        for i in range(num_class-1):
            for j in range(i, num_class-1):
                classifier_matrix[i][j] = classifier_list[classifier_counter]
                classifier_counter += 1

        out_matrix = []
        for i in range(num_class-1):
            out_matrix.append([])
            for j in range(num_class-1):
                out_matrix[i].append(None)

        for i in range(num_class-1):
            for j in range(i, num_class-1):
                out_matrix[i][j] = classifier_matrix[i][j].predict_proba(XValidate)

        # calculate the whole prediction prob
        val_shape = XValidate.shape[0]
        predict_prob_list = []
        for i in range(num_class):
            predict_prob_list.append(np.zeros(shape=[val_shape, 1]))

        for i in range(num_class-1):
            for j in range(i, num_class-1):
                predict_prob_list[i] += out_matrix[i][j][:, 0][:, np.newaxis] / (num_class * 2)
                predict_prob_list[j + 1] += out_matrix[i][j][:, 1][:, np.newaxis] / (num_class * 2)

        # get the result of num_class probability
        result = np.concatenate(predict_prob_list, axis=1)

        # calculate the probability for the one more class
        std = np.std(result, axis=1)[:, np.newaxis]
        other_prob = np.exp(-std) / (1 + np.exp(std * 5))
        result = np.concatenate([result, other_prob], axis=1)
        result = result / np.repeat((other_prob + 1), axis=1, repeats=num_class + 1)

        # put all the parameters into a dict
        estParameters = {}
        estParameters["class_num"] = num_class
        estParameters["parameters"] = []
        for i in range(num_class-1):
            for j in range(i, num_class-1):
                estParameters["parameters"].append(
                    {
                        'relevant_vectors': classifier_matrix[i][j].relevant_vectors_,
                        'coef': classifier_matrix[i][j].coef_,
                        'active':  classifier_matrix[i][j].active_,
                        'intercept': classifier_matrix[i][j].intercept_,
                        'mean': classifier_matrix[i][j]._x_mean,
                        'std': classifier_matrix[i][j]._x_std,
                        'classes': classifier_matrix[i][j].classes_,
                        'lambda': classifier_matrix[i][j].lambda_,
                        'sigma': classifier_matrix[i][j].sigma_,
                        'relevant': classifier_matrix[i][j].relevant_
                    }
                )

        return result, estParameters


def TestMyClassifier(XTest, Parameters, EstParameters):
    """

    :param XTest: [N, 60]
    :param Parameters: hypter parameter
    :param EstParameters: parameter in the model
    :return:
    """
    num_class = EstParameters["class_num"]
    classifier = []
    # init all the classifiers
    for param_dict in EstParameters["parameters"]:
        rvm_classifier = RVC(
            n_iter=Parameters["parameters"]['n_iter'],
            tol=Parameters["parameters"]['tol'],
            n_iter_solver=Parameters["parameters"]['n_iter_solver'],
            tol_solver=Parameters["parameters"]['tol_solver'],
            fit_intercept=Parameters["parameters"]['fit_intercept'],
            verbose=Parameters["parameters"]['verbose'],
            kernel=Parameters["parameters"]['kernel'],
            degree=Parameters["parameters"]['degree'],
            gamma=Parameters["parameters"]['gamma'],
            coef0=Parameters["parameters"]['coef0'],
            kernel_params=Parameters["parameters"]['kernel_params']
        )
        rvm_classifier.relevant_vectors_ = param_dict.get('relevant_vectors')
        rvm_classifier.relevant_ = param_dict.get('relevant')
        rvm_classifier.active_ = param_dict.get('active')
        rvm_classifier.coef_ = param_dict.get('coef')
        rvm_classifier.intercept_ = param_dict.get('intercept')
        rvm_classifier._x_mean = param_dict.get('mean')
        rvm_classifier._x_std = param_dict.get('std')
        rvm_classifier.classes_ = param_dict.get('classes')
        rvm_classifier.lambda_ = param_dict.get('lambda')
        rvm_classifier.sigma_ = param_dict.get('sigma')
        classifier.append(rvm_classifier)

    # put all the classifiers into a matrix, so it is easier for calculation
    classifier_matrix = []
    for i in range(num_class-1):
        classifier_matrix.append([])
        for j in range(num_class-1):
            classifier_matrix[i].append(None)

    count = 0
    for i in range(num_class-1):
        for j in range(i, num_class-1):
            classifier_matrix[i][j] = classifier[count]
            count += 1

    # calculate the output for XTest
    out_matrix = []
    for i in range(num_class - 1):
        out_matrix.append([])
        for j in range(num_class - 1):
            out_matrix[i].append(None)

    for i in range(num_class - 1):
        for j in range(i, num_class - 1):
            out_matrix[i][j] = classifier_matrix[i][j].predict_proba(XTest)

    # calculate the whole prediction prob
    val_shape = XTest.shape[0]
    predict_prob_list = []
    for i in range(num_class):
        predict_prob_list.append(np.zeros(shape=[val_shape, 1]))

    for i in range(num_class - 1):
        for j in range(i, num_class - 1):
            predict_prob_list[i] += out_matrix[i][j][:, 0][:, np.newaxis] / (num_class * 2)
            predict_prob_list[j + 1] += out_matrix[i][j][:, 1][:, np.newaxis] / (num_class * 2)

    result = np.concatenate(predict_prob_list, axis=1)

    # calculate the probability for the one more class
    std = np.std(result, axis=1)[:, np.newaxis]
    other_prob = np.exp(-std) / (1 + np.exp(std * 10))
    result = np.concatenate([result, other_prob], axis=1)
    result = result / np.repeat((other_prob + 1), axis=1, repeats=num_class + 1)

    return result

if __name__ == "__main__":
    import scipy.io as io
    import numpy as np

    train = io.loadmat("Proj2FeatVecsSet1.mat")
    label = io.loadmat("Proj2TargetOutputsSet1.mat")
    train_data = train["Proj2FeatVecsSet1"]
    label_data = label["Proj2TargetOutputsSet1"]
    label_data = np.argmax(label_data, axis=1)[:, np.newaxis]
    dataset = np.concatenate([train_data, label_data], axis=1)
    np.random.shuffle(dataset)
    training = dataset[:9000]
    validation = dataset[1000:1100]
    xeval = training[:, :-1]
    xlabel = np.squeeze(training[:, -1:], axis=1)
    xval = validation[:, :-1]
    result, params = TrainMyClassifier(xeval, xlabel, xval,
                      {
                          "algorithm": "RVM",
                          "parameters":{
                              'n_iter':100,
                              'tol':1e-4,
                              'n_iter_solver':100,
                              'tol_solver':1e-4,
                              'fit_intercept':True,
                              'verbose': False,
                              'kernel':'rbf',
                              'degree': 2,
                              'gamma': None,
                              'coef0':1,
                              'kernel_params': None
                          }
                      })
    xval_target = validation[:, -1:]
    print(result)
    result = np.argmax(result, axis=1)
    print("eval result: " + str(result))
    count = 0
    for i in range(len(result)):
        if result[i] == xval_target[i][0]:
            count += 1
    print("eval accurcy: %.3f" % (count / (len(result) * 1.0)))

    result = TestMyClassifier(xval, {
                          "algorithm": "RVM",
                          "parameters":{
                              'n_iter':100,
                              'tol':1e-4,
                              'n_iter_solver':15,
                              'tol_solver':1e-4,
                              'fit_intercept':True,
                              'verbose': False,
                              'kernel':'rbf',
                              'degree': 2,
                              'gamma': None,
                              'coef0':1,
                              'kernel_params': None
                          }
                      }, params)
    result = np.argmax(result, axis=1)
    print("test result: " + str(result))
    count = 0
    for i in range(len(result)):
        if result[i] == xval_target[i][0]:
            count += 1
    print("test accurcy: %.3f" % (count / (len(result) * 1.0)))

