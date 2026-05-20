from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def get_classification_models():

    models = {

        "Logistic Regression":
            LogisticRegression(),

        "Decision Tree":
            DecisionTreeClassifier(),

        "Random Forest":
            RandomForestClassifier(),

        "Support Vector Machine":
            SVC(),

        "K-Nearest Neighbors":
            KNeighborsClassifier(),

        "Naive Bayes":
            GaussianNB(),

        "Neural Network":
            MLPClassifier(max_iter=1000)
    }

    return models