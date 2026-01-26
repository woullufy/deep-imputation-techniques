import numpy as np
from numpy.linalg import solve
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp
from sklearn.cluster import KMeans


class GaussianMixtureModel:
    def __init__(
            self,
            n_components=2,
            max_iter=100,
            tol=1e-3,
            reg_covar=1e-6,
            random_state=None
    ):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.rng = np.random.default_rng(random_state)

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.log_likelihood_ = []

    def _initialize(self, X):
        n_samples, n_features = X.shape

        kmeans = KMeans(n_clusters=self.K, n_init=1, random_state=self.rng.integers(10 ** 9))
        labels = kmeans.fit_predict(X)

        self.means_ = kmeans.cluster_centers_

        self.covariances_ = np.zeros((self.K, n_features, n_features))
        for k in range(self.K):
            diff = X[labels == k] - self.means_[k]
            cov = (diff.T @ diff) / max((labels == k).sum(), 1)
            self.covariances_[k] = cov + self.reg_covar * np.eye(n_features)

        self.weights_ = np.bincount(labels, minlength=self.K) / n_samples

    def _estimate_log_gaussian(self, X):
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.K))

        for k in range(self.K):
            mu = self.means_[k]
            cov = self.covariances_[k]

            L = cholesky(cov, lower=True)
            diff = X - mu

            sol = solve_triangular(L, diff.T, lower=True)
            m_dist = np.sum(sol ** 2, axis=0)

            log_det = 2 * np.sum(np.log(np.diag(L)))

            log_prob[:, k] = -0.5 * (m_dist + log_det + n_features * np.log(2 * np.pi))

        return log_prob

    def _e_step(self, X):
        weighted_log_probs = self._estimate_log_gaussian(X) + np.log(self.weights_)
        log_prob_norm = logsumexp(weighted_log_probs, axis=1)
        log_resp = weighted_log_probs - log_prob_norm[:, None]
        resp = np.exp(log_resp)
        return resp, np.sum(log_prob_norm)

    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        Nk = resp.sum(axis=0)

        self.weights_ = Nk / n_samples
        self.means_ = (resp.T @ X) / Nk[:, None]

        self.covariances_ = np.zeros((self.K, n_features, n_features))
        for k in range(self.K):
            diff = X - self.means_[k]
            self.covariances_[k] = ((resp[:, k][:, None] * diff).T @ diff) / Nk[k]
            self.covariances_[k] += self.reg_covar * np.eye(n_features)

    def fit(self, X):
        X = np.asarray(X, float)
        self._initialize(X)

        prev_ll = -np.inf

        for it in range(self.max_iter):
            resp, ll = self._e_step(X)
            self._m_step(X, resp)

            self.log_likelihood_.append(ll)

            if abs(ll - prev_ll) < self.tol * X.shape[0]:
                break

            prev_ll = ll

        return self

    def predict_proba(self, X):
        log_prob = self._estimate_log_gaussian(X) + np.log(self.weights_)
        log_norm = logsumexp(log_prob, axis=1)
        return np.exp(log_prob - log_norm[:, None])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X):
        weighted_log_probs = self._estimate_log_gaussian(X) + np.log(self.weights_)
        log_prob_norm = logsumexp(weighted_log_probs, axis=1)
        return np.mean(log_prob_norm)

    def sample(self, n_samples=1):
        n_features = self.means_.shape[1]

        comp = self.rng.choice(self.K, size=n_samples, p=self.weights_)

        samples = np.zeros((n_samples, n_features))
        for i, k in enumerate(comp):
            samples[i] = self.rng.multivariate_normal(
                mean=self.means_[k],
                cov=self.covariances_[k]
            )
        return samples, comp


class GMMMissing:
    def __init__(self, n_components=2, max_iter=100, tol=1e-3, reg_covar=1e-6, random_state=None):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.rng = np.random.default_rng(random_state)

        self.pi_ = None
        self.mu_ = None
        self.sigma_ = None
        self.log_likelihood_ = []

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        X_filled = np.where(np.isnan(X), np.nanmean(X, axis=0), X)

        kmeans = KMeans(n_clusters=self.K, n_init=5, random_state=self.rng.integers(10 ** 9))
        labels = kmeans.fit_predict(X_filled)

        self.mu_ = kmeans.cluster_centers_
        self.sigma_ = np.array([np.cov(X_filled[labels == k].T) + self.reg_covar * np.eye(n_features)
                                for k in range(self.K)])
        self.pi_ = np.bincount(labels, minlength=self.K) / n_samples

    def _e_step(self, X):
        n_samples, n_features = X.shape
        log_weighted_probs = np.zeros((n_samples, self.K))

        exp_X = np.zeros((n_samples, self.K, n_features))
        exp_XX = np.zeros((n_samples, self.K, n_features, n_features))

        for k in range(self.K):
            mu_k = self.mu_[k]
            sigma_k = self.sigma_[k]

            for i in range(n_samples):
                x = X[i]
                missing = np.isnan(x)
                obs = ~missing
                n_obs = np.sum(obs)

                # Fully missing
                if n_obs == 0:
                    log_p_xo = 0.0

                    x_exp = mu_k.copy()
                    cov_exp = sigma_k.copy()

                #  Partially observed
                else:
                    mu_o = mu_k[obs]
                    Sigma_oo = sigma_k[np.ix_(obs, obs)]

                    L = cholesky(Sigma_oo, lower=True)
                    diff = x[obs] - mu_o
                    sol = solve_triangular(L, diff, lower=True)
                    dist = np.sum(sol ** 2)
                    log_det = 2 * np.sum(np.log(np.diag(L)))
                    log_p_xo = -0.5 * (dist + log_det + n_obs * np.log(2 * np.pi))

                    if not np.any(missing):
                        x_exp = x
                        cov_exp = np.zeros((n_features, n_features))
                    else:
                        mu_m = mu_k[missing]
                        Sigma_mo = sigma_k[np.ix_(missing, obs)]
                        Sigma_mm = sigma_k[np.ix_(missing, missing)]

                        reg_mo = solve(Sigma_oo, Sigma_mo.T).T

                        cond_mean_m = mu_m + reg_mo @ (x[obs] - mu_o)
                        cond_cov_m = Sigma_mm - reg_mo @ Sigma_mo.T

                        x_exp = x.copy()
                        x_exp[missing] = cond_mean_m
                        cov_exp = np.zeros((n_features, n_features))
                        cov_exp[np.ix_(missing, missing)] = cond_cov_m

                log_weighted_probs[i, k] = log_p_xo + np.log(self.pi_[k])
                exp_X[i, k] = x_exp
                exp_XX[i, k] = np.outer(x_exp, x_exp) + cov_exp

        log_prob_norm = logsumexp(log_weighted_probs, axis=1)
        resp = np.exp(log_weighted_probs - log_prob_norm[:, np.newaxis])

        return resp, exp_X, exp_XX, np.sum(log_prob_norm)

    def _m_step(self, X, resp, exp_X, exp_XX):
        n_samples, n_features = X.shape
        Nk = resp.sum(axis=0) + 1e-10

        for k in range(self.K):
            w_resp = resp[:, k][:, np.newaxis]
            self.mu_[k] = np.sum(w_resp * exp_X[:, k], axis=0) / Nk[k]

            w_resp_sq = resp[:, k][:, np.newaxis, np.newaxis]
            Exx = np.sum(w_resp_sq * exp_XX[:, k], axis=0) / Nk[k]

            self.sigma_[k] = Exx - np.outer(self.mu_[k], self.mu_[k])
            self.sigma_[k] += self.reg_covar * np.eye(n_features)

        self.pi_ = Nk / n_samples

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self._initialize_parameters(X)
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            resp, exp_X, exp_XX, ll = self._e_step(X)
            self._m_step(X, resp, exp_X, exp_XX)

            self.log_likelihood_.append(ll)
            if abs(ll - prev_ll) < self.tol * X.shape[0]:
                break
            prev_ll = ll
        return self

    def predict(self, X):
        resp, _, _, _ = self._e_step(np.asarray(X, dtype=float))
        return np.argmax(resp, axis=1)

    def impute(self, X, stochastic=True):
        X = np.asarray(X, dtype=float)
        X_imputed = X.copy()
        resp, _, _, _ = self._e_step(X)
        n_samples, n_features = X.shape

        for i in range(n_samples):
            x = X[i]
            missing = np.isnan(x)
            if not np.any(missing): continue

            k = self.rng.choice(self.K, p=resp[i])
            mu_k, sigma_k = self.mu_[k], self.sigma_[k]
            obs = ~missing

            if np.sum(obs) == 0:
                if stochastic:
                    X_imputed[i] = self.rng.multivariate_normal(mu_k, sigma_k)
                else:
                    X_imputed[i] = mu_k
            else:
                mu_o, mu_m = mu_k[obs], mu_k[missing]
                Sigma_oo = sigma_k[np.ix_(obs, obs)]
                Sigma_mo = sigma_k[np.ix_(missing, obs)]
                Sigma_mm = sigma_k[np.ix_(missing, missing)]

                reg_coeff = solve(Sigma_oo, Sigma_mo.T).T
                cond_mean = mu_m + reg_coeff @ (x[obs] - mu_o)

                if stochastic:
                    cond_cov = Sigma_mm - reg_coeff @ Sigma_mo.T
                    cond_cov += 1e-9 * np.eye(cond_cov.shape[0])
                    X_imputed[i, missing] = self.rng.multivariate_normal(cond_mean, cond_cov)
                else:
                    X_imputed[i, missing] = cond_mean
        return X_imputed
