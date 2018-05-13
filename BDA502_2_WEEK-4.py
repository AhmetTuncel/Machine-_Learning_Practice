
from sklearn import linear_model
# class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
from sklearn.metrics import mean_squared_error, r2_score

reg = linear_model.LinearRegression()
reg = reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(reg.coef_)