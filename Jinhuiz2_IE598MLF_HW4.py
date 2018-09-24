import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet



#############################################################################
print(50 * '-')
print('Part 1: Exploratory Data Analysis')
print(50 * '-')

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'housing/housing.data',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print('Dataset excerpt:\n\n', df.head())


#############################################################################

sns.set(style='whitegrid', context='notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

sns.pairplot(df[cols], size=2.5)
# plt.tight_layout()
# plt.savefig('./figures/scatter.png', dpi=300)
plt.show()


cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

# plt.tight_layout()
# plt.savefig('./figures/corr_mat.png', dpi=300)
plt.show()

sns.reset_orig()



#############################################################################


class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


X = df[['RM']].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=42)








def lin_regplot(X, y, model):
    plt.scatter(X, y, c='lightblue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    return


#############################################################################
print(50 * '-')
print('Part 2: Linear regression')
print(50 * '-')

slr = LinearRegression()
slr.fit(X_train, y_train)
y_pred = slr.predict(X_test)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

lin_regplot(X_train, y_train, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.title('Linear Model')
plt.show()





X = df.iloc[:, :-1].values
y = df['MEDV'].values



slr = LinearRegression()

slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
# plt.tight_layout()

# plt.savefig('./figures/slr_residuals.png', dpi=300)
plt.show()


print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))




#Ridge
print(50 * '-')
print('Part 3.1: Ridge regression')
print(50 * '-')
for i in range(1,10,1):
    ridge = Ridge(alpha=i*0.5)
    ridge.fit(X_train,y_train)
    y_pred = ridge.predict(X_test)
    print('Slope: %.3f' % ridge.coef_[0])
    print('Intercept: %.3f' % ridge.intercept_)
    lin_regplot(X_train, y_train, ridge)
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000\'s [MEDV]')
    plt.title('Ridge Model')
    plt.show()
    X = df.iloc[:, :-1].values
    y = df['MEDV'].values
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    plt.scatter(y_train_pred,  y_train_pred - y_train,
                c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
                c='limegreen', marker='s', edgecolor='white',
                label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.tight_layout()
    plt.title('Ridge Model')
    plt.show()
    print('aplha=',i*0.5)
    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
    print(50 * '>')


#Lasso
print(50 * '-')
print('Part 3.2: LASSO regression')
print(50 * '-')
for i in range(1,10,1):
    lasso = Lasso(alpha=i*0.1)
    lasso.fit(X_train,y_train)
    y_pred = lasso.predict(X_test)
    print('Slope: %.3f' % lasso.coef_[0])
    print('Intercept: %.3f' % lasso.intercept_) 
    lin_regplot(X_train, y_train, lasso)
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000\'s [MEDV]')
    plt.title('LASSO Model')
    plt.show()
    
    X = df.iloc[:, :-1].values
    y = df['MEDV'].values
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    plt.scatter(y_train_pred,  y_train_pred - y_train,
                c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
                c='limegreen', marker='s', edgecolor='white',
                label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.tight_layout()
    plt.title('Lasso Model')
    plt.show()
    print('aplha=',i*0.1)
    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
    print(50 * '>')



#ElasticNet
print(50 * '-')
print('Part 3.3: Elastic Net regression')
print(50 * '-')
for i in range(1,10,1):
    elanet = ElasticNet(l1_ratio=i*0.1)
    elanet.fit(X_train,y_train)
    y_pred = elanet.predict(X_test)
    print('Slope: %.3f' % elanet.coef_[0])
    print('Intercept: %.3f' % elanet.intercept_)
    lin_regplot(X_train, y_train, elanet)
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000\'s [MEDV]')
    plt.title('Elastic Net Model')
    plt.show()
    X = df.iloc[:, :-1].values
    y = df['MEDV'].values
    elanet.fit(X_train, y_train)
    y_train_pred = elanet.predict(X_train)
    y_test_pred = elanet.predict(X_test)
    
    plt.scatter(y_train_pred,  y_train_pred - y_train,
                c='steelblue', marker='o', edgecolor='white',
                label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test,
                c='limegreen', marker='s', edgecolor='white',
                label='Test data')
    
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
    plt.xlim([-10, 50])
    plt.tight_layout()
    plt.title('ElasticNet Model')
    plt.show()
    print('l1_ritio=',i*0.1)
    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
    print(50 * '>')

print("My name is {Jinhui Zhang}")
print("My NetID is: {jinhuiz2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")