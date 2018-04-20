from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ExpSineSquared
import numpy as np


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
        data = []
        for i in range(5):
            data.append(XEstimate[ClassLabels == i])

        target = []
        for i in range(5):
            target.append(ClassLabels[ClassLabels == i])

        combined_data = []
        combined_target = []
        for i in range(5):
            for j in range(i+1, 5):
                combined_data.append(np.concatenate([data[i], data[j]], axis=0))
                combined_target.append(np.concatenate([target[i], target[j]], axis=0))

        classifier = []
        for i in range(10):
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
            gpc_classifier.fit(combined_data[i], combined_target[i])
            classifier.append(gpc_classifier)

        out = []
        for i in range(10):
            tmp_out = classifier[i].predict_proba(XValidate)
            out.append(tmp_out)

        # calculate the whole prediction prob
        predict_prob0 = np.expand_dims((out[0][:, 0] + out[1][:, 0] + out[2][:, 0] + out[3][:, 0]) / 10, axis=1)
        predict_prob1 = np.expand_dims((out[0][:, 1] + out[4][:, 0] + out[5][:, 0] + out[6][:, 0]) / 10, axis=1)
        predict_prob2 = np.expand_dims((out[1][:, 1] + out[4][:, 1] + out[7][:, 0] + out[8][:, 0]) / 10, axis=1)
        predict_prob3 = np.expand_dims((out[2][:, 1] + out[5][:, 1] + out[7][:, 1] + out[9][:, 0]) / 10, axis=1)
        predict_prob4 = np.expand_dims((out[3][:, 1] + out[6][:, 1] + out[8][:, 1] + out[9][:, 1]) / 10, axis=1)

        result = np.concatenate([predict_prob0, predict_prob1, predict_prob2, predict_prob3, predict_prob4], axis=1)

        std = np.std(result, axis=1)[:, np.newaxis]
        other_prob = np.exp(-std) / (1 + np.exp(-std))
        result = np.concatenate([result, other_prob], axis=1)
        result = result / np.repeat((other_prob + 1), axis=1, repeats=6)

        estParameters = {}
        estParameters["parameters"] = []
        for i in range(10):
            estParameters["parameters"].append(
                {
                    "log_marginal_likelihood_value_": classifier[i].log_marginal_likelihood_value_,
                    "classes_": classifier[i].classes_,
                    "n_classes_": classifier[i].n_classes_,
                    "base_estimator_": classifier[i].base_estimator_
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
    classifier = []
    for param_dict in EstParameters["paramemters"]:
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

    out = []
    for i in range(10):
        tmp_out = classifier[i].predict_proba(XTest)
        out.append(tmp_out)

    # calculate the whole prediction prob
    predict_prob0 = np.expand_dims((out[0][:, 0] + out[1][:, 0] + out[2][:, 0] + out[3][:, 0]) / 10, axis=1)
    predict_prob1 = np.expand_dims((out[0][:, 1] + out[4][:, 0] + out[5][:, 0] + out[6][:, 0]) / 10, axis=1)
    predict_prob2 = np.expand_dims((out[1][:, 1] + out[4][:, 1] + out[7][:, 0] + out[8][:, 0]) / 10, axis=1)
    predict_prob3 = np.expand_dims((out[2][:, 1] + out[5][:, 1] + out[7][:, 1] + out[9][:, 0]) / 10, axis=1)
    predict_prob4 = np.expand_dims((out[3][:, 1] + out[6][:, 1] + out[8][:, 1] + out[9][:, 1]) / 10, axis=1)

    result = np.concatenate([predict_prob0, predict_prob1, predict_prob2, predict_prob3, predict_prob4], axis=1)

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
    training = dataset[:3000]
    validation = dataset[1000:1100]
    xeval = training[:, :-1]
    print(xeval.shape)
    xlabel = np.squeeze(training[:, -1:], axis=1)
    print(xlabel.shape)
    xval = validation[:, :-1]
    result, params = TrainMyClassifier(xeval, xlabel, xval,
                      {
                          "algorithm": "GPR",
                          "parameters": {
                              # "kernel": ExpSineSquared(0.0003, 1.2), # best one
                              # "kernel": ExpSineSquared(1e-3, 1e-0),
                              "kernel": None,
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
    count = 0
    for i in range(len(result)):
        if result[i] == xval_target[i][0]:
            count += 1
    print(result)
    print("accurcy: %.3f" % (count / (len(result) * 1.0)))

