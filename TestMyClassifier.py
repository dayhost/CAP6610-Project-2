import numpy as np

from TrainMyClassifier import *

def TestMyClassifier(XTest,Parameters,EstParameters):
    if Parameters['algorithm'] == 'SVM':
        n_svc = get_svc(Parameters)
        svc_set_para(n_svc, EstParameters)
        Ytest = svc_probability(n_svc, XTest)

        return Ytest

    elif Parameters['algorithm'] == 'RVM':

        Parameters = Parameters['parameters']

        if len(EstParameters) == 1:
            clf = EstParameters.get('clf')
        else:
            clf = EstParameters[0]
        if np.shape(clf.classes_)[0] == 2:
            Ytest = clf.predict_proba(XTest)
        else:
            Ytest = predict_proba(clf,XTest)

        return Ytest

    elif Parameters['algorithm'] == 'GPR':
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


def predict_proba(clf,XValidate):
    noOfClasses = np.shape(clf.classes_)[0]
    noOfClassifiers = (noOfClasses * (noOfClasses-1))/2
    print('no of Classifiers',noOfClassifiers)
    dataSize = np.shape(XValidate)[0]
    Yvalidate = np.zeros((dataSize, np.shape(clf.classes_)[0]))
    c = 0
    prob = clf.multi_.estimators_[c].predict_proba(XValidate)
    #Summing Fkm(X) where k!=m
    for i in range(0,noOfClasses):
        for j in range(i, noOfClasses):
            if (i < j):
                Yvalidate[:, i] =  Yvalidate[:, i]+ prob[:, 0]
                Yvalidate[:, j] =  Yvalidate[:, j]+ prob[:, 1]
                c = c + 1;
                if(c<noOfClassifiers):
                    prob = clf.multi_.estimators_[c].predict_proba(XValidate)
    #Calculating 1/G(summation(ykm))
    Yvalidate = Yvalidate / np.shape(clf.classes_)[0]
    #Calculating probability of XValidate  not belonging to any class
    prob_std = np.ndarray.std(Yvalidate, axis=1)[:, np.newaxis]
    sigmoid = 1 - expit(prob_std)
    Yvalidate = np.concatenate([Yvalidate, sigmoid], axis=1)
    Yvalidate = Yvalidate / np.repeat((sigmoid + 1), axis=1, repeats=np.shape(clf.classes_)[0] + 1)
    return Yvalidate
