import numpy as np
from scipy.special import expit
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from skrvm import RVC

def TrainMyClassifier(XEstimate, ClassLabels, XValidate, Parameters ):
# RVM
    if Parameters['algorithm'] == 'RVM':
        Parameters = Parameters['parameters']

        clf =  RVC(alpha=Parameters.get('alpha'),beta=Parameters.get('beta'),n_iter=Parameters.get('n_iter'))
        clf.fit(XEstimate, ClassLabels)
        if np.shape(clf.classes_)[0] == 2:
            Yvalidate = clf.predict_proba(XValidate)
        else:
            Yvalidate = predict_proba(clf,XValidate)
        EstParameters = get_params(clf)

        return Yvalidate, EstParameters
#SVM

    elif Parameters['algorithm'] == 'SVM':

        svc = get_svc(Parameters)
        svc_train(svc, XEstimate, ClassLabels)
        prob = svc_probability(svc, XValidate)
        EstParameters = svc_get_para(svc)

        prob_std = np.ndarray.std(prob, axis=1)[:, np.newaxis]
        sigmoid = 1 - expit(prob_std)
        Yvalidate = np.concatenate([prob, sigmoid], axis=1)
        Yvalidate = Yvalidate / np.repeat((sigmoid + 1), axis=1, repeats=len(svc.classes_)+1)

        return Yvalidate, EstParameters
#GPR
    elif Parameters["algorithm"] == "GPR":
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

# Helper Funtions for TrainMyClassifier
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
    hyper = svc.get_params(deep=True)
    ret = {'support': support, 'support_vectors': support_vectors, 'n_support': n_support, 'dual_coef': dual_coef,
           'intercept': intercept, 'sparse': sparse, 'shape_fit': shape_fit, 'prob_a': prob_a, 'prob_b': prob_b,
           'gamma': gamma, 'classes': classes, 'hyper': hyper}
    return ret


def svc_set_para(svc, svc_para):
    svc.set_params(**svc_para['hyper'])
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

def get_params(clf):
    if np.shape(clf.classes_)[0] == 2:
        Parameters = [{'phi': clf.phi,'relevance': clf.relevance_,'alpha': clf.alpha_,'beta': clf.beta_,'m': clf.m_,
                       'gamma': clf.gamma,'bias': clf.bias,'clf': clf}]
    else :
        Parameters = [clf]
        for c in clf.multi_.estimators_:
            Parameter = {'phi': c.phi,'relevance': c.relevance_,'alpha': c.alpha_,'beta': c.beta_,'m': c.m_,
                         'gamma': c.gamma,'bias': c.bias,'clf': c}
            Parameters.append(Parameter)
    return Parameters

# Helper function for Cross validation.
