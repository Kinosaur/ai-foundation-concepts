import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def f(x):   
    return np.cos(x) + 0.2*np.random.normal(size=(x.shape))

X = np.linspace(-4,4,20).reshape(-1,1)
y = f(X)


def poly_features(X, K):
    X = X.flatten()
    N = X.shape[0]
    Phi = np.zeros((N, K+1))
    for k in range(K+1):
        Phi[:,k] = X**k 
    return Phi

def nonlinear_features_maximum_likelihood(Phi, y):
    kappa = 1e-08 # 'jitter'term; good for numerical stability
    D = Phi.shape[1]  
    Pt = Phi.T @ y # Phi^T*y
    PP = Phi.T @ Phi + kappa*np.eye(D) # Phi^T*Phi + kappa*I
    C = scipy.linalg.cho_factor(PP)
    theta_ml = scipy.linalg.cho_solve(C, Pt) # inv(Phi^T*Phi)*Phi^T*y 
    
    return theta_ml

K = 10 # Define the degree of the polynomial we wish to fit
Phi = poly_features(X, K) # N x (K+1) feature matrix

theta_ml = nonlinear_features_maximum_likelihood(Phi, y) # maximum likelihood estimator

# test inputs
Xtest = np.linspace(-10,10,1000).reshape(-1,1)

# feature matrix for test inputs
Phi_test = poly_features(Xtest, K)

y_pred_test = Phi_test @ theta_ml # predicted y-values



# test inputs
ytest = f(Xtest) # ground-truth y-values

plt.figure()
plt.plot(X, y, '+', markersize=10, label='Data points')
plt.plot(Xtest, ytest, label=f'ground truth observations' )
plt.plot(Xtest, y_pred_test, label=f'prediction' )
plt.xlabel("X-axis ($x$)")
plt.ylabel("Y-axis ($y$)")
plt.title("Plot of Training Data Set")
plt.xlim([-10, 10]) # Setting x-axis limits
plt.legend()
plt.grid(True)
plt.show()



def RMSE(y, ypred):
    rmse = np.sqrt(np.mean((y-ypred)**2))
    return rmse

y_pred = Phi @ theta_ml # predicted y-values

loss_value = RMSE(y, y_pred)
print(loss_value)
