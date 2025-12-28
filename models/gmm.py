import numpy as np
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

            # Cholesky for stable inversion
            L = cholesky(cov, lower=True)
            diff = X - mu

            # Solve L * y = diff.T   Mahalanobis distance
            sol = solve_triangular(L, diff.T, lower=True)
            m_dist = np.sum(sol ** 2, axis=0)

            # Log determinant
            log_det = 2 * np.sum(np.log(np.diag(L)))

            # Full log Gaussian density
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

        # Choose mixture components
        comp = self.rng.choice(self.K, size=n_samples, p=self.weights_)

        samples = np.zeros((n_samples, n_features))
        for i, k in enumerate(comp):
            samples[i] = self.rng.multivariate_normal(
                mean=self.means_[k],
                cov=self.covariances_[k]
            )
        return samples, comp
