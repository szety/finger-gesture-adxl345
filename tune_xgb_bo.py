"""
Tune xgb classifier using Bayesian Optimization on
two AXDL345 accelerometers data
"""

# Author: Tom Sze <sze.takyu@gmail.com>

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import dump, load

from bayes_opt import BayesianOptimization
from termcolor import colored
from xgboost import XGBClassifier
from numpy import loadtxt


def optimize_xgb(data, targets, n_iter):
    """Apply Bayesian Optimization to xgb.

    This optimizes for hyperparameters of xgb.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The data to fit.

    targets : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    n_iter : int
        The number of iterations of optimization.

    Returns
    -------
    best_param : dict
        Result of best hyperparameters after optimization.
    """

    def xgb_crossval(max_depth, learning_rate):
        """Wrapper of xgb cross validation.
        """
        return xgb_cv(
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            data=data,
            targets=targets,
        )

    optimizer = BayesianOptimization(
        f=xgb_crossval,
        pbounds={
            "max_depth": (0, 5),
            "learning_rate": (0.1, 1)
        },
        random_state=1234,
        verbose=2
    )
    optimizer.maximize(n_iter=n_iter)

    print("Final result:", optimizer.max['params'])
    return optimizer.max['params']


def xgb_cv(max_depth, learning_rate, data, targets):
    """Evaluate cross-validation score of AUC of a xgb classifier

    A pipeline standardizes data and fits data to a xgb classifier with
    given max_depth and learning_rate parameters. The cross validation
    score is found on this pipeline with given data and targets.

    Parameters
    ----------
    max_depth :

    learning_rate :

    data : array-like of shape (n_samples, n_features)
        The data to fit.

    targets : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    Returns
    -------
    score : numpy.float64
        Mean AUC score of the estimator for each run of the cross validation.
    """

    # Create the classifier with new parameters in a pipeline
    pipe = make_pipeline(StandardScaler(),
                         XGBClassifier(learning_rate=learning_rate,
                                       max_depth=max_depth))
    cval = cross_val_score(pipe,
                           data,
                           targets,
                           cv=10,
                           scoring='roc_auc')

    return cval.mean()


def fit_and_save_model(params, data, targets):
    """Fit xgb classifier pipeline with params parameters and
    save it to disk"""
    pipe = make_pipeline(StandardScaler(),
                         XGBClassifier(learning_rate=params['learning_rate'],
                                       max_depth=int(params['max_depth'])))

    pipe.fit(data, targets)

    # Persist the pipeline to disk
    dump(pipe, 'ADXL345_xgb_gesture.joblib')
    print('Done saving')

    return pipe


def evaluate_model(estimator, test_x, test_y):
    """Show estimator test accuracy on test set"""
    y_pred = estimator.predict(test_x)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(test_y, predictions)
    print("Test accuracy: %.2f%%" % (accuracy * 100.0))


if __name__ == "__main__":
    dataset = loadtxt('all_record.csv', delimiter=",")
    X = dataset[:, 0:6]
    y = dataset[:, 6]

    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=seed)

    print(colored('--- Optimizing xgb ---', 'green'))
    best_params = optimize_xgb(X_train, y_train, n_iter=100)

    print(colored('--- Fit and save model ---', 'green'))
    model = fit_and_save_model(best_params, X_train, y_train)

    print(colored('--- Evaluate model ---', 'green'))
    evaluate_model(model, X_test, y_test)

    print(colored('--- Load and evaluate model  ---', 'green'))
    model = load('ADXL345_xgb_gesture.joblib')
    evaluate_model(model, X_test, y_test)

    print(colored('--- Done ---', 'red'))
