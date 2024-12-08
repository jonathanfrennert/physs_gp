import CRPS
import CRPS.CRPS as pscore
import numpy as np

def avg_crps(y_true, y_pred_mu, y_pred_var, num_samples=1000, seed=0):
    y_true = np.squeeze(y_true)
    y_pred_mu = np.squeeze(y_pred_mu)
    y_pred_var = np.squeeze(y_pred_var)

    N = y_true.shape[0]

    np.random.seed(seed)
    samples = np.array([np.random.normal(y_pred_mu, np.sqrt(y_pred_var)) for i in range(num_samples)]).T
    crps = np.mean([pscore(samples[n], y_true[n]).compute()[0] for n in range(N)])
    return crps
