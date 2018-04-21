from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import numpy as np

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

def TrainMyClassifier(XEstimate, ClassLabels, XValidate, Parameters):
    """
    This function will return all the
    :param XEstimate: (data, label), data in [N, 60] shape
    :param ClassLabels: label in [N] shape
    :param XValidate: data in shape [N, 60]
    :param Parameters:
    {
        "algorithm": "GPR",
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
    if Parameters["algorithm"] == "GPR":
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

        for i in range(num_class-1):
            for j in range(i, num_class-1):
                gpc_classifier = GaussianProcessClassifier(
                    kernel=Parameters["parameters"]["kernel"],
                    optimizer=Parameters["parameters"]["optimizer"],
                    n_restarts_optimizer=Parameters["parameters"]["n_restarts_optimizer"],
                    max_iter_predict=Parameters["parameters"]["max_iter_predict"],
                    warm_start=Parameters["parameters"]["warm_start"],
                    copy_X_train=Parameters["parameters"]["copy_X_train"],
                    random_state=Parameters["parameters"]["random_state"],
                    multi_class="one_vs_rest",
                    n_jobs=Parameters["parameters"]["n_jobs"]
                )
                gpc_classifier.fit(data_matrix[i][j], target_matrix[i][j])
                classifier_matrix[i][j] = gpc_classifier

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
                        "log_marginal_likelihood_value_": classifier_matrix[i][j].log_marginal_likelihood_value_,
                        "classes_": classifier_matrix[i][j].classes_,
                        "n_classes_": classifier_matrix[i][j].n_classes_,
                        "base_estimator_": classifier_matrix[i][j].base_estimator_
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
        gpc_classifier = GaussianProcessClassifier(
            kernel=Parameters["parameters"]["kernel"],
            optimizer=Parameters["parameters"]["optimizer"],
            n_restarts_optimizer=Parameters["parameters"]["n_restarts_optimizer"],
            max_iter_predict=Parameters["parameters"]["max_iter_predict"],
            warm_start=Parameters["parameters"]["warm_start"],
            copy_X_train=Parameters["parameters"]["copy_X_train"],
            random_state=Parameters["parameters"]["random_state"],
            multi_class="one_vs_rest",
            n_jobs=Parameters["parameters"]["n_jobs"]
        )
        gpc_classifier.log_marginal_likelihood_value_ = param_dict["log_marginal_likelihood_value_"]
        gpc_classifier.classes_ = param_dict["classes_"]
        gpc_classifier.n_classes_ = param_dict["n_classes_"]
        gpc_classifier.base_estimator_ = param_dict["base_estimator_"]
        classifier.append(gpc_classifier)

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
    other_prob = np.exp(-std) / (1 + np.exp(std * 5))
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
    training = dataset[:1000]
    validation = dataset[1000:1100]
    xeval = training[:, :-1]
    xlabel = np.squeeze(training[:, -1:], axis=1)
    xval = validation[:, :-1]
    result, params = TrainMyClassifier(xeval, xlabel, xval,
                      {
                          "algorithm": "GPR",
                          "parameters": {
                              "kernel": 1.0 * RBF(1),
                              "optimizer": "fmin_l_bfgs_b",
                              "n_restarts_optimizer": 0,
                              "max_iter_predict": 100,
                              "warm_start": True,
                              "copy_X_train": True,
                              "random_state": 0,
                              "multi_class": "one_vs_one",
                              "n_jobs": -1
                          }
                      })
    xval_target = validation[:, -1:]
    result = np.argmax(result, axis=1)
    print("eval result: " + str(result))
    count = 0
    for i in range(len(result)):
        if result[i] == xval_target[i][0]:
            count += 1
    print("eval accurcy: %.3f" % (count / (len(result) * 1.0)))

    result = TestMyClassifier(xval, {
                          "algorithm": "GPR",
                          "parameters": {
                              "kernel": 1.0 * RBF(1),
                              "optimizer": "fmin_l_bfgs_b",
                              "n_restarts_optimizer": 0,
                              "max_iter_predict": 100,
                              "warm_start": True,
                              "copy_X_train": True,
                              "random_state": 0,
                              "multi_class": "one_vs_one",
                              "n_jobs": -1
                          }
                      }, params)
    result = np.argmax(result, axis=1)
    print("test result: " + str(result))
    count = 0
    for i in range(len(result)):
        if result[i] == xval_target[i][0]:
            count += 1
    print("test accurcy: %.3f" % (count / (len(result) * 1.0)))

