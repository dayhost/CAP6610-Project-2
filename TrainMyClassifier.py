import numpy as np
from scipy.special import expit
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from skbayes.rvm_ard_models import RVC

def TrainMyClassifier(XEstimate, ClassLabels, XValidate, Parameters ):
# RVM
    if Parameters['algorithm'] == 'RVM':
        Parameters = Parameters['parameters']
        clf = RVC(n_iter = Parameters.get('n_iter'), tol = Parameters.get('tol'),
                    n_iter_solver = Parameters.get('n_iter_solver'), tol_solver = Parameters.get('tol_solver'),
                    fit_intercept = Parameters.get('fit_intercept'),
                    verbose = Parameters.get('verbose'),
                    kernel = Parameters.get('kernel'),
                    degree = Parameters.get('degree'),
                    gamma = Parameters.get('gamma'),
                    coef0 = Parameters.get('coef0'),
                    kernel_params = Parameters.get('kernel_params') )

        clf.fit(XEstimate, ClassLabels)
        prob = clf.predict_proba(XValidate)

        prob_std = np.ndarray.std(prob, axis=1)[:, np.newaxis]
        sigmoid = 1 - expit(prob_std)
        Yvalidate = np.concatenate([prob, sigmoid], axis=1)
        Yvalidate = Yvalidate / np.repeat((sigmoid + 1), axis=1, repeats = np.shape(clf.classes_)[0] + 1)

        EstParameters = { 'relevant_vectors':clf.relevant_vectors_ ,'coef':clf.coef_,'active':clf.active_,
                          'intercept':clf.intercept_,'mean':clf._x_mean, 'std':clf._x_std,'classes':clf.classes_,
                          'lambda':clf.lambda_,'sigma':clf.sigma_,'relevant':clf.relevant_}

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

# Helper function for Cross validation.
