import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy import linalg as LA

X_truth = np.ones((5,6))
X_pred = 1.7*np.ones((5,6))

# print 'LA error: ', (1.0/5)*LA.norm(X_truth - X_pred)**2
# print 'LA error: ', (1.0/X_truth.shape[0])*LA.norm(X_truth - X_pred)**2
# print 'LA error:: ', (1.0/5)*LA.norm(X_truth - X_pred, 'fro')**2
# print 'LA error: ', LA.norm(X_truth - X_pred)**2
# print 'LA error: ', LA.norm(X_truth - X_pred)
# print 'LA error: ', (1.0/X_truth.shape[0])*LA.norm(X_truth - X_pred)

print X_truth.size
print 'LA error:', LA.norm(X_truth - X_pred)**2/X_truth.size
print 'numpy error: ', LA.norm(X_truth - X_pred)**2/X_truth.size


print 'sklearn MSE error: ', mean_squared_error(X_truth, X_pred)
