from PCA import df

import numpy as np
from sklearn import model_selection
from dtuimldmtools import rlr_validate
from matplotlib.pylab import figure, plot, subplot, hist, xlabel, ylabel, legend, show, semilogx, grid, title, loglog
import sklearn.linear_model as lm 

# Standardize data
df_standardized = (df - df.mean()) / df.std()

# Define attributes
attribute_names = ['power','toughness', 'rarity']
class_name      = ['price']

# Prepare data
X = df_standardized[attribute_names].to_numpy(dtype=np.float32)
y = df_standardized[class_name].to_numpy(dtype=np.float32).ravel()

# Store dimensions of X as local variables
N = np.shape(X)[0] ## Number of observations
M = np.shape(X)[1] ## Number of attributes

#######################################################
# REGRESSION ANALYSIS 
#######################################################

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict DiabetesPedigreeFunction
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('price (true)'); ylabel('price (estimated)');
subplot(2,1,2)
hist(residual,40)

show()

N, M = X.shape
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attribute_names = [u'Offset']+attribute_names
M = M+1

## Crossvalidation parameters
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
   
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using any input data (baseline)
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda on whole training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    
    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Do linear regression with sklearn.linear_model package
    m = lm.LinearRegression().fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-')
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    k+=1

show()

# Show results + visualize
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attribute_names[m], np.round(w_rlr[m,-1],2)))
