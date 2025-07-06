from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def train_and_evaluate(X_train, X_test, y_train, y_test):
    parameters = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(LinearSVC(), parameters, cv=5)
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    clf = grid.best_estimator_
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.2f}")
    return clf
