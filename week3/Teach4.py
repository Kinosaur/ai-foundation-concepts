import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

X = np.array([-3, -1, 0.0, 1, 3]).reshape(-1,1) # 5x1 vector, N=5, D=1
y = np.array([2.2, 3.7, 3.14, 3.67, 4.67]).reshape(-1,1) # 5x1 vector


def max_lik_estimate(X, y):
    theta_ml = np.linalg.solve(  (X.T @ X) ,  (X.T @ y)    ) ## <-- SOLUTION
    return theta_ml


def predict_with_estimate(Xtest, theta):
    prediction = Xtest @ theta 
    return prediction 


theta_ml = max_lik_estimate(X,y)
print(theta_ml[0][0])
# define a test set
Xtest = np.linspace(-5,5,100).reshape(-1,1) # 100 x 1 vector of test inputs
ml_prediction = predict_with_estimate(Xtest, theta_ml)

plt.figure()
plt.plot(X, y, '+', markersize=10, label='Data points')
plt.plot(Xtest, ml_prediction, label=f'y={theta_ml[0][0]}x' )
plt.xlabel("X-axis ($x$)")
plt.ylabel("Y-axis ($y$)")
plt.title("Plot of Training Data Set")
plt.xlim([-5, 5]) # Setting x-axis limits
plt.legend()
plt.grid(True)
plt.show()


