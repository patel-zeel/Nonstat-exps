import numpy as np
import scipy
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

## Terminology is consistent with paper 
# Paper name: Nonstationary Gaussian Process Regression Using Point Estimates of Local Smoothness

class LLS:
  def __init__(self, input_dim, N=1): # N = Number of local points
    self.input_dim = input_dim
    self.N = N
    self.name = 'LLS'

    # Keeping track of refiting the model to modify K_XX efficiently
    self.fit_count = 0
    self.refit_count = 0

  def rbf(self, x1, x2, sigma_l_bar): # RBF for GP_l without variance or noise
    return np.exp(-(x1-x2.T)**2/(2*(sigma_l_bar**2)))
  """
  # Deprecated
  def K_XX(self, x1, x2, sigma_f, sigma_n, l): # From Eq. 7 
    p = l.T@l
    P_r = p
    P_c = p.T
    P_s = P_r + P_c
    E = np.exp(-((x1-x2.T)**2) / P_s)

    return (sigma_f**2)*(0.5**-0.5)*(P_r**0.25)*(P_c**0.25)*(P_s**-0.5)*E  # Eq. 7
  """
  def K(self, X, X2, sigma_f, l_bar, sigma_l_bar): # Main kernel function by Generalizing Eq. 7
    ans = 1
    for dim in range(self.input_dim):
        lx1 = self.predict_lengthscales(X, dim, l_bar, sigma_l_bar)
        lx2 = self.predict_lengthscales(X2, dim, l_bar, sigma_l_bar)
        P_rc = lx1@lx2.T
        P_s = 0.5 * ((lx1**2) + (lx2.T)**2)
        E = np.exp(-(X[:,dim:dim+1].reshape(-1,1)-X2[:,dim:dim+1].reshape(1,-1))**2 / P_s)
        ans = ans * (P_rc**0.5)*(P_s**-0.5)*E
    return sigma_f * ans

  def K_(self, X, X2):
    return self.K(X, X2, self.sigma_f, self.l_bar, self.sigma_l_bar)

  def predict_lengthscales(self, X_hat, dim, l_bar, sigma_l_bar):
        
    K_X_barX_bar = self.rbf(self.X_bar[:,dim:dim+1].reshape(-1,1), self.X_bar[:,dim:dim+1].reshape(-1,1), sigma_l_bar[dim])
    K_XX_bar = self.rbf(X_hat[:,dim:dim+1].reshape(-1,1), self.X_bar[:,dim:dim+1].reshape(-1,1), sigma_l_bar[dim])
    L = np.linalg.cholesky(K_X_barX_bar)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, np.log(l_bar[:,dim:dim+1].reshape(-1,1))))
    l = np.exp(K_XX_bar@alpha)
    return l

  def predict_lengthscales_(self, X_hat, dim):
     return self.predict_lengthscales(X_hat, dim, self.l_bar, self.sigma_l_bar)

  def objective(self, params): # From equation 6 in section 4.1
    sigma_f = params[0]
    sigma_n = params[1]
    l_bar = params[2:2+self.N*self.input_dim].reshape(self.N, self.input_dim)
    sigma_l_bar = params[2+self.N*self.input_dim:].reshape(self.input_dim, )
    
    # print(sigma_f, sigma_n, l_bar, sigma_l_bar)
    # Evaluating global kernel
    K_XX = self.K(self.X, self.X, sigma_f, l_bar, sigma_l_bar) + (sigma_n**2)*np.eye(self.X.shape[0])
    K_X_barX_bar = 1
    for dim in range(self.input_dim):
        K_X_barX_bar *= self.rbf(self.X_bar[:,dim:dim+1].reshape(-1,1), 
                                 self.X_bar[:,dim:dim+1].reshape(-1,1), sigma_l_bar[dim])

    L_theta =  self.y.T@np.linalg.pinv(K_XX)@self.y +\
               np.log(np.linalg.det(K_XX)) +\
               np.log(np.linalg.det(K_X_barX_bar)) # Eq. 6

    return L_theta[0,0]

  def fit(self, X, y, sigma_f=1., sigma_n=1., n_restarts_optimizer=10):
    assert len(X.shape) == 2, "X must be 2D"
    assert len(y.shape) == 2, "y must be 2D"
    assert y.shape[1] == 1, "y must be of shape (*,1)"

    self.X = X
    self.y = y
    if self.N <= self.X.shape[0]:
      kmeans = KMeans(n_clusters=self.N)
      self.X_bar = kmeans.fit(X).cluster_centers_
    else:
      self.X_bar = X
    sigma_f = float(sigma_f)
    sigma_n = float(sigma_n)
    sigma_l_bar = np.ones((self.input_dim, ), dtype=np.float64)
    
    # Fitting
    self.refit_count += 1
    optim_fun = np.inf
    for cycle in range(n_restarts_optimizer):
      # initialize lengthscales for support points
      l_bar = np.ones((self.X_bar.shape[0], self.input_dim), dtype=np.float64)+np.random.rand(self.X_bar.shape[0], self.input_dim)*0.01
      
      # print('initial l_bar', l_bar)
      try:
        params = [sigma_f]+[sigma_n]+l_bar.flatten().tolist()+sigma_l_bar.tolist()
        res = scipy.optimize.minimize(self.objective, params, bounds=[(10**-3,10**3) for i in range(len(params))])
      except np.linalg.LinAlgError:
        print('cycle',cycle,'did not converge')
        continue
#       except ValueError:
#         print('cycle',cycle,'did not converge')
#         continue
      # print(res.fun, 'optim value in cycle', cycle)
      if res.fun==-np.inf:
        continue
      if res.fun<optim_fun:
        optim_fun = res.fun
        self.sigma_f = res.x[0]
        self.sigma_n = res.x[1]
        self.l_bar = res.x[2:2+self.N*self.input_dim].reshape(self.N, self.input_dim)
        self.sigma_l_bar = res.x[2+self.N*self.input_dim:].reshape(self.input_dim, )
    
    self.params = {'likelihood':optim_fun, 'global variance':self.sigma_f, 'noise_level':self.sigma_n**2, 
                   'N_lengthscales':self.l_bar, 'GP_l_lengthscale':self.sigma_l_bar}
    return self
    
  def get_params(self):
    return self.params

  def predict(self, X_hat, return_cov=False):
    if self.refit_count < 1:
      raise AssertionError("Model is not fitted yet. Please fit using .fit(X, y) method first.")
    if self.refit_count>self.fit_count:
      self.fit_count=self.refit_count
      self.K_XX_inv = np.linalg.pinv(self.K_(self.X, self.X) + (self.sigma_n**2)*np.eye(self.X.shape[0]))
    K_X_hatX = self.K_(X_hat, self.X)
    mean = K_X_hatX@self.K_XX_inv@self.y
    if return_cov:
      cov = self.K_(X_hat, X_hat) - K_X_hatX@self.K_XX_inv@K_X_hatX.T
    return mean, cov