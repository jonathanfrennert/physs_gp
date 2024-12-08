def gaussian_gaussian_conditional_diagional(XS:np.ndarray, X: np.ndarray, kernel, approximate_posterior) -> np.ndarray:
    """
        Args:
            g1 defines the distribution of the conditional p(f*|f)
            g2 defines the distribution that the expectation is wrt E_{q(f)} [ . ]
    """

    k_zz = kernel.K(X, X)
    k_xz = kernel.K(XS, X)
    k_xsxs_diag = kernel.K_diag(XS)

    mu = approximate_posterior.m
    sig = approximate_posterior.S

    k_xsxs_diag = np.squeeze(k_xsxs_diag)

    k_zz_chol = cholesky(k_zz+Settings.jitter*np.eye(X.shape[0]))
    #k_sig_chol = np.linalg.cholesky(sig+Settings.jitter*np.eye(mu.shape[0]))
    
    A = triangular_solve(k_zz_chol, k_xz.T, lower=True) # M x N
    A1 = triangular_solve(k_zz_chol.T, A, lower=False) # M x N
    A2 = (sig.T @ A1).T # N x M 

    #mu = k_xz @ cholesky_solve(k_zz_chol, mu) # N x 1
    mu = A1.T @ mu # N x 1

    sig = k_xsxs_diag - np.sum(np.square(A), axis=0) + np.sum(np.square(A2), axis=1) #N x 1

    #ensure correct shapes
    mu = np.reshape(mu, [mu.shape[0], 1])
    sig = np.reshape(sig, [sig.shape[0], 1])

    return mu, sig

