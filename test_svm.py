import svm


def test_svm(xtest, parameters, est_parameters):
    svc_tes = svm.get_svc(parameters)
    svm.svc_set_para(svc_tes, est_parameters)
    ret = svm.svc_probability(svc_tes, xtest)
    return ret
