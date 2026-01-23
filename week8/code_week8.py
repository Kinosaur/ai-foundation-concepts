import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.5, 2.3, 2.9]).reshape(-1,1) # 5x1 vector, N=5, D=1
y = np.array([1.4, 1.9, 3.2]).reshape(-1,1) # 5x1 vector

def max_lik_estimate(X, y):
    theta_ml = np.linalg.solve(X.T @ X, X.T @ y) 
    return theta_ml

def predict_with_estimate(Xtest, theta):
    prediction = Xtest @ theta 
    return prediction 

# =====================================================================================
print("="*10)
N, D = X.shape
X_aug = np.hstack([np.ones((N,1)), X]) # augmented training inputs of size N x (D+1)
print("X_aug = ")
print(X_aug)
print("="*10)
# =====================================================================================
theta_ml = max_lik_estimate(X_aug,y)
print(theta_ml)
# define a test set
Xtest = np.linspace(-5,5,100).reshape(-1,1) # 100 x 1 vector of test inputs
# =====================================================================================
Xtest_aug = np.hstack([np.ones((Xtest.shape[0],1)), Xtest]) # 100 x (D + 1) vector of test inputs
# =====================================================================================
ml_prediction = predict_with_estimate(Xtest_aug, theta_ml)
y_pred = predict_with_estimate(X_aug, theta_ml)
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

# RMSE Calculation Function
def RMSE(y, ypred):
    return np.sqrt(np.mean((y - ypred)**2))

# Calculate RMSE for the test set
rmse_test = RMSE(y, y_pred)
print("RMSE of MLE :", rmse_test)
print("================================= END MLE =================================")
# RMSE of MLE : 0.3849741916091627

















print("================================ Start GD =================================")
def loss_function(Phi, y, theta):
    """ Compute cost for linear regression """
    return  np.sum((y- Phi.dot(theta)) ** 2)


def gradient_descent(Phi, y, theta, alpha, iterations):
    """ Perform gradient descent to learn theta """
    cost_history = [0] * iterations
    for it in range(iterations):
        prediction = Phi.dot(theta)
        error = y - prediction 
        gradient = -2* Phi.T.dot(error)
        # ============ Optional ===============
        # alpha_new = alpha / (1+0.00000001*it)
        # theta -= alpha_new * gradient
        # =====================================
        # theta -= alpha * gradient
        theta = theta  - (alpha * gradient)
        cost_history[it] = loss_function(Phi, y, theta)
        # Predict test outcomes
        if np.isnan(cost_history[it]) or (it > 0 and cost_history[it] > cost_history[it - 1]):
            print(f"Breaking at iteration {it} due to NaN or increase in cost.")
            break
    return theta, cost_history


np.random.seed(41)
numberTheta = 2
theta = np.random.uniform(-1, 1, (numberTheta, 1))  # Initialize theta
# alpha = 0.0002  # Learning rate
alpha = 0.0002  # Learning rate
iterations = 100000  # Number of iterations

# Run Gradient Descent
theta_gd, cost_history= gradient_descent(X_aug, y, theta, alpha, iterations)
print("theta_gd", theta_gd)
# Plot the cost history over iterations
plt.figure()
plt.plot(range(iterations), cost_history, 'b.')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Reduction over Time')
plt.show()

# Xtest = np.linspace(-5,5,100).reshape(-1,1) # 100 x 1 vector of test inputs
# =====================================================================================
Xtest_aug = np.hstack([np.ones((Xtest.shape[0],1)), Xtest]) # 100 x (D + 1) vector of test inputs
# =====================================================================================
ml_prediction = predict_with_estimate(Xtest_aug, theta_gd)
y_pred_gd = predict_with_estimate(X_aug, theta_gd)

# Calculate RMSE for the test set
rmse_test = RMSE(y, y_pred_gd)
print("RMSE of GD :", rmse_test)