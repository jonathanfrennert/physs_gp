class NG_Parameterisation(object):
    pass

class NG_Moment(NG_Parameterisation):
    """ Gaussian parameterisation by mean and variance """
    pass

class NG_Natural(NG_Parameterisation):
    """ Gaussian parameterisation by natural parameters """
    pass

class NG_Precision(NG_Parameterisation):
    """ Gaussian parameterisation by mean and precision"""
    pass

def get_parameterisation_class(approx_posterior):
    if str(type(approx_posterior).__name__) in ['MeanFieldConjugateGaussian', 'MeanFieldApproximatePosterior']:
        approx_posterior = approx_posterior.approx_posteriors[0]

    if str(type(approx_posterior).__name__) in ['ConjugateGaussian', 'FullConjugateGaussian']:
        return NG_Moment()

    elif str(type(approx_posterior).__name__) in ['ConjugatePrecisionGaussian', 'FullConjugatePrecisionGaussian']:
        return NG_Precision()

    breakpoint()
    raise RuntimeError('Parameterisation type not found!')
