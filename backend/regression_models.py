from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor


def get_regression_models():

    models = {

        "Linear Regression":
            LinearRegression(),

        "Ridge Regression":
            Ridge(),

        "Lasso Regression":
            Lasso(),

        "Decision Tree Regressor":
            DecisionTreeRegressor(),

        "Random Forest Regressor":
            RandomForestRegressor(),

        "Gradient Boosting Regressor":
            GradientBoostingRegressor(),

        "Support Vector Regressor":
            SVR(),

        "Neural Network Regressor":
            MLPRegressor(max_iter=1000)
    }

    return models