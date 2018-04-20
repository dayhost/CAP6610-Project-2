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
        data0 = XEstimate[ClassLabels == 0]
        data1 = XEstimate[ClassLabels == 1]
        data2 = XEstimate[ClassLabels == 2]
        data3 = XEstimate[ClassLabels == 3]
        data4 = XEstimate[ClassLabels == 4]
        target0 = ClassLabels[ClassLabels == 0]
        target1 = ClassLabels[ClassLabels == 1]
        target2 = ClassLabels[ClassLabels == 2]
        target3 = ClassLabels[ClassLabels == 3]
        target4 = ClassLabels[ClassLabels == 4]
        # we need to do the One v.s One mode ourselves
        # for class 0, 1
        p1 = GaussianProcessClassifier(
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
        data01 = np.concatenate([data0, data1], axis=0)
        target01 = np.concatenate([target0, target1], axis=0)
        p1.fit(data01, target01)
        # for class 0, 2
        p2 = GaussianProcessClassifier(
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
        data02 = np.concatenate([data0, data2], axis=0)
        target02 = np.concatenate([target0, target2], axis=0)
        p2.fit(data02, target02)
        # for class 0, 3
        p3 = GaussianProcessClassifier(
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
        data03 = np.concatenate([data0, data3], axis=0)
        target03 = np.concatenate([target0, target3], axis=0)
        p3.fit(data03, target03)
        # for class 0, 4
        p4 = GaussianProcessClassifier(
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
        data04 = np.concatenate([data0, data4], axis=0)
        target04 = np.concatenate([target0, target4], axis=0)
        p4.fit(data04, target04)
        # for class 1, 2
        p5 = GaussianProcessClassifier(
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
        data12 = np.concatenate([data1, data2], axis=0)
        target12 = np.concatenate([target1, target2], axis=0)
        p5.fit(data12, target12)
        # for class 1, 3
        p6 = GaussianProcessClassifier(
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
        data13 = np.concatenate([data1, data3], axis=0)
        target13 = np.concatenate([target1, target3], axis=0)
        p6.fit(data13, target13)
        # for class 1, 4
        p7 = GaussianProcessClassifier(
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
        data14 = np.concatenate([data1, data4], axis=0)
        target14 = np.concatenate([target1, target4])
        p7.fit(data14, target14)
        # for class 2, 3
        p8 = GaussianProcessClassifier(
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
        data23 = np.concatenate([data2, data3], axis=0)
        target23 = np.concatenate([target2, target3], axis=0)
        p8.fit(data23, target23)
        # for class 2, 4
        p9 = GaussianProcessClassifier(
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
        data24 = np.concatenate([data2, data4], axis=0)
        target24 = np.concatenate([target2, target4], axis=0)
        p9.fit(data24, target24)
        # for class 3, 4
        p10 = GaussianProcessClassifier(
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
        data34 = np.concatenate([data3, data4], axis=0)
        target34 = np.concatenate([target3, target4], axis=0)
        p10.fit(data34, target34)

        out1 = p1.predict_proba(XValidate)
        out2 = p2.predict_proba(XValidate)
        out3 = p3.predict_proba(XValidate)
        out4 = p4.predict_proba(XValidate)
        out5 = p5.predict_proba(XValidate)
        out6 = p6.predict_proba(XValidate)
        out7 = p7.predict_proba(XValidate)
        out8 = p8.predict_proba(XValidate)
        out9 = p9.predict_proba(XValidate)
        out10 = p10.predict_proba(XValidate)

        # calculate the whole prediction prob
        predict_prob0 = np.expand_dims((out1[:, 0] + out2[:, 0] + out3[:, 0] + out4[:, 0]) / 10, axis=1)
        predict_prob1 = np.expand_dims((out1[:, 1] + out5[:, 0] + out6[:, 0] + out7[:, 0]) / 10, axis=1)
        predict_prob2 = np.expand_dims((out2[:, 1] + out5[:, 1] + out8[:, 0] + out9[:, 0]) / 10, axis=1)
        predict_prob3 = np.expand_dims((out3[:, 1] + out6[:, 1] + out8[:, 1] + out10[:, 0]) / 10, axis=1)
        predict_prob4 = np.expand_dims((out4[:, 1] + out7[:, 1] + out9[:, 1] + out10[:, 1]) / 10, axis=1)

        result = np.concatenate([predict_prob0, predict_prob1, predict_prob2, predict_prob3, predict_prob4], axis=1)

        estParameters = {
            "p1": {
                "log_marginal_likelihood_value_": p1.log_marginal_likelihood_value_,
                "classes_": p1.classes_,
                "n_classes_": p1.n_classes_,
                "base_estimator_": p1.base_estimator_
            },
            "p2": {
                "log_marginal_likelihood_value_": p2.log_marginal_likelihood_value_,
                "classes_": p2.classes_,
                "n_classes_": p2.n_classes_,
                "base_estimator_": p2.base_estimator_
            },
            "p3": {
                "log_marginal_likelihood_value_": p3.log_marginal_likelihood_value_,
                "classes_": p3.classes_,
                "n_classes_": p3.n_classes_,
                "base_estimator_": p3.base_estimator_
            },
            "p4": {
                "log_marginal_likelihood_value_": p4.log_marginal_likelihood_value_,
                "classes_": p4.classes_,
                "n_classes_": p1.n_classes_,
                "base_estimator_": p1.base_estimator_
            },
            "p5": {
                "log_marginal_likelihood_value_": p5.log_marginal_likelihood_value_,
                "classes_": p5.classes_,
                "n_classes_": p5.n_classes_,
                "base_estimator_": p5.base_estimator_
            },
            "p6": {
                "log_marginal_likelihood_value_": p6.log_marginal_likelihood_value_,
                "classes_": p6.classes_,
                "n_classes_": p6.n_classes_,
                "base_estimator_": p6.base_estimator_
            },
            "p7": {
                "log_marginal_likelihood_value_": p7.log_marginal_likelihood_value_,
                "classes_": p7.classes_,
                "n_classes_": p7.n_classes_,
                "base_estimator_": p7.base_estimator_
            },
            "p8": {
                "log_marginal_likelihood_value_": p8.log_marginal_likelihood_value_,
                "classes_": p8.classes_,
                "n_classes_": p8.n_classes_,
                "base_estimator_": p8.base_estimator_
            },
            "p9": {
                "log_marginal_likelihood_value_": p9.log_marginal_likelihood_value_,
                "classes_": p9.classes_,
                "n_classes_": p9.n_classes_,
                "base_estimator_": p9.base_estimator_
            },
            "p10": {
                "log_marginal_likelihood_value_": p10.log_marginal_likelihood_value_,
                "classes_": p10.classes_,
                "n_classes_": p10.n_classes_,
                "base_estimator_": p10.base_estimator_
            }
        }

        return result, estParameters


def TestMyClassifier(XTest, Parameters, EstParameters):
    """

    :param XTest: [N, 60]
    :param Parameters: hypter parameter
    :param EstParameters: parameter in the model
    :return:
    """
    p1 = GaussianProcessClassifier(
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
    p1.log_marginal_likelihood_value_ = EstParameters["p1"]["log_marginal_likelihood_value_"]
    p1.classes_ = EstParameters["p1"]["classes_"]
    p1.n_classes_ = EstParameters["p1"]["n_classes_"]
    p1.base_estimator_ = EstParameters["p1"]["base_estimator_"]
    # for class 0, 2
    p2 = GaussianProcessClassifier(
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
    p2.log_marginal_likelihood_value_ = EstParameters["p2"]["log_marginal_likelihood_value_"]
    p2.classes_ = EstParameters["p2"]["classes_"]
    p2.n_classes_ = EstParameters["p2"]["n_classes_"]
    p2.base_estimator_ = EstParameters["p2"]["base_estimator_"]
    # for class 0, 3
    p3 = GaussianProcessClassifier(
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
    p3.log_marginal_likelihood_value_ = EstParameters["p3"]["log_marginal_likelihood_value_"]
    p3.classes_ = EstParameters["p3"]["classes_"]
    p3.n_classes_ = EstParameters["p3"]["n_classes_"]
    p3.base_estimator_ = EstParameters["p3"]["base_estimator_"]
    # for class 0, 4
    p4 = GaussianProcessClassifier(
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
    p4.log_marginal_likelihood_value_ = EstParameters["p4"]["log_marginal_likelihood_value_"]
    p4.classes_ = EstParameters["p4"]["classes_"]
    p4.n_classes_ = EstParameters["p4"]["n_classes_"]
    p4.base_estimator_ = EstParameters["p4"]["base_estimator_"]
    # for class 1, 2
    p5 = GaussianProcessClassifier(
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
    p5.log_marginal_likelihood_value_ = EstParameters["p5"]["log_marginal_likelihood_value_"]
    p5.classes_ = EstParameters["p5"]["classes_"]
    p5.n_classes_ = EstParameters["p5"]["n_classes_"]
    p5.base_estimator_ = EstParameters["p5"]["base_estimator_"]
    # for class 1, 3
    p6 = GaussianProcessClassifier(
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
    p6.log_marginal_likelihood_value_ = EstParameters["p6"]["log_marginal_likelihood_value_"]
    p6.classes_ = EstParameters["p6"]["classes_"]
    p6.n_classes_ = EstParameters["p6"]["n_classes_"]
    p6.base_estimator_ = EstParameters["p6"]["base_estimator_"]
    # for class 1, 4
    p7 = GaussianProcessClassifier(
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
    p7.log_marginal_likelihood_value_ = EstParameters["p7"]["log_marginal_likelihood_value_"]
    p7.classes_ = EstParameters["p7"]["classes_"]
    p7.n_classes_ = EstParameters["p7"]["n_classes_"]
    p7.base_estimator_ = EstParameters["p7"]["base_estimator_"]
    # for class 2, 3
    p8 = GaussianProcessClassifier(
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
    p8.log_marginal_likelihood_value_ = EstParameters["p8"]["log_marginal_likelihood_value_"]
    p8.classes_ = EstParameters["p8"]["classes_"]
    p8.n_classes_ = EstParameters["p8"]["n_classes_"]
    p8.base_estimator_ = EstParameters["p8"]["base_estimator_"]
    # for class 2, 4
    p9 = GaussianProcessClassifier(
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
    p9.log_marginal_likelihood_value_ = EstParameters["p9"]["log_marginal_likelihood_value_"]
    p9.classes_ = EstParameters["p9"]["classes_"]
    p9.n_classes_ = EstParameters["p9"]["n_classes_"]
    p9.base_estimator_ = EstParameters["p9"]["base_estimator_"]
    # for class 3, 4
    p10 = GaussianProcessClassifier(
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
    p10.log_marginal_likelihood_value_ = EstParameters["p10"]["log_marginal_likelihood_value_"]
    p10.classes_ = EstParameters["p10"]["classes_"]
    p10.n_classes_ = EstParameters["p10"]["n_classes_"]
    p10.base_estimator_ = EstParameters["p10"]["base_estimator_"]

    out1 = p1.predict_proba(XTest)
    out2 = p2.predict_proba(XTest)
    out3 = p3.predict_proba(XTest)
    out4 = p4.predict_proba(XTest)
    out5 = p5.predict_proba(XTest)
    out6 = p6.predict_proba(XTest)
    out7 = p7.predict_proba(XTest)
    out8 = p8.predict_proba(XTest)
    out9 = p9.predict_proba(XTest)
    out10 = p10.predict_proba(XTest)

    # calculate the whole prediction prob
    predict_prob0 = np.expand_dims((out1[:, 0] + out2[:, 0] + out3[:, 0] + out4[:, 0]) / 10, axis=1)
    predict_prob1 = np.expand_dims((out1[:, 1] + out5[:, 0] + out6[:, 0] + out7[:, 0]) / 10, axis=1)
    predict_prob2 = np.expand_dims((out2[:, 1] + out5[:, 1] + out8[:, 0] + out9[:, 0]) / 10, axis=1)
    predict_prob3 = np.expand_dims((out3[:, 1] + out6[:, 1] + out8[:, 1] + out10[:, 0]) / 10, axis=1)
    predict_prob4 = np.expand_dims((out4[:, 1] + out7[:, 1] + out9[:, 1] + out10[:, 1]) / 10, axis=1)

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
    training = dataset[:100]
    validation = dataset[1000:1100]
    xeval = training[:, :-1]
    print(xeval.shape)
    xlabel = np.squeeze(training[:, -1:], axis=1)
    print(xlabel.shape)
    xval = validation[:, :-1]
    result = TrainMyClassifier(xeval, xlabel, xval,
                      {
                          "algorithm": "GPR",
                          "parameters": {
                              "kernel": ExpSineSquared(1e-3, 1e-3),
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
    print(result)

