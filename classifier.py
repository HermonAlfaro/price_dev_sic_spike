# imports

# external
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_curve, auc, average_precision_score, plot_precision_recall_curve, plot_confusion_matrix
import joblib

class Classifier(object):
    """
    To run classification and calculates metrics.

    workflow:
        1. perform feature selection
        2. perfrom feature scaling
        3. perform hyper parameters tuning and fitting classifier model
        4. compute and plot metric of performance

    Attributes:
        feature_selector_obj: feature selector to select features for training
        feature_scaler_obj: feature scaler for scalling features
        classifier_obj: classifier object to fit and predicting probabilities
        parameters: parameters of the classifier
        cv: cross validation folder object
        scoring: scoring function for the cross validation
        search_obj: object to perform hypter parameters tuning
    """

    def __init__(self, classifier_obj, parameters, cv, scoring,
                 search_cls):

        self.classifier_obj = classifier_obj
        self.parameters = parameters
        self.cv = cv
        self.scoring = scoring
        self.search_cls = search_cls
        self.search_obj = None

    def learning(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        to perform hyper parameter tuning and fitting classifier with best parameters

        :param X: features's DataFrame
        :param y: label's DataFrame
        :return:
        """

        # pipeline
        steps = [('classifier', self.classifier_obj)]

        pipeline = Pipeline(steps)

        # tuning
        self.search_obj = self.search_cls(pipeline,
                                          self.parameters,
                                          cv=self.cv,
                                          n_jobs=-1,
                                          scoring=self.scoring)

        self.search_obj.fit(X, y)

    def predicting(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        To predict label values makeing use of the classifier already tuned

        :param X: features's DataFrame
        :return: predictions
        """

        # predicting
        y_pred = self.search_obj.predict(X)

        return y_pred

    def predicting_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        To predict probabilities of the labels
        :param X: features's DataFrame
        :return: probabilities
        """

        # predicting
        y_pred = self.search_obj.predict_proba(X)

        return y_pred

    def calculate_score(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        To calculate the score of the classifier when trying to predict labels values for features values
        :param X: features's DataFrame
        :param y: label's DataFrame
        :return: score of the classifier
        """

        # score
        score = self.search_obj.score(X, y)

        return score

    @staticmethod
    def get_classification_report(y: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
        """
        create a classification report making use of the classification_report method form scikit-learn

        :param y: true labels's DataFrame
        :param y_pred: predicted's DataFrame
        :return: report of classification
        """
        report = classification_report(y, y_pred)

        return report

    def calculate_roc_auc(self, X: pd.DataFrame, y: pd.DataFrame, index: int = 1) -> float:
        """
        To calculate are under roc curve. It is useful to evaluate performance of the classification.
        The roc curve is the collection of pairs (false positive rate, true positive rate).
        Bigger it is the area, best is the classification.
        Because that means that the true positive rate is bigger than the false positive rate

        :param X: features's DataFrame
        :param y: labels's DataFrame
        :param index: 0 for class 0, 1 for class 1
        :return: roc area under curve
        """

        # probqbilities
        y_score = self.search_obj.predict_proba(X)
        y_score = y_score[:, index]

        # false positive rates and true positve rates
        fpr, tpr, _ = roc_curve(y, y_score)

        return auc(fpr, tpr)

    def plot_roc(self, X: pd.DataFrame, y: pd.DataFrame, index=1):
        """
        To plot roc curve.

        :param X: features's DataFrame
        :param y: labels's DataFrame
        :param index:
        :return:
        """

        # probabilities
        y_score = self.search_obj.predict_proba(X)
        y_score = y_score[:, index]

        # rates
        fpr, tpr, _ = roc_curve(y, y_score)

        # roc auc
        roc_auc = auc(fpr, tpr)

        # plotting
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Roc')
        plt.legend(loc="lower right")
        plt.show()

    def calculate_precision_recall_auc(self, X: pd.DataFrame, y: pd.DataFrame, index=1) -> float:
        """
        To calculate are under precision-recall curve. It is useful to evaluate performance of the classification.
        The precision-recall curve is the collection of pairs (recall, positive).
        Bigger it is the area, best is the classification.
        Because that means that not only the precision is high, but also the recall

        :param X: features's DatFrame
        :param y: labels's DataFrame
        :param index: 0 for label 0, 1 for label 1
        :return:
        """

        # probabilities
        y_score = self.search_obj.predict_proba(X)
        y_score = y_score[:, index]

        return average_precision_score(y, y_score)
    
    def plot_confusion_matrix(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Calculates prediction of classes for examples in X and compares against y to
        creates a confusion matrix.

        :param X: features's DatFrame
        :param y: labels's DataFrame
        :return:
        """

        plot_confusion_matrix(self.search_obj, X, y)
        plt.show()
        
      

    def plot_precision_recall(self, X: pd.DataFrame, y: pd.DataFrame, index=1):
        """
        To plot precision-recall curve.

        :param X: features's DataFrame
        :param y: labels's DataFrame
        :param index: 0 for label 0, 1 for label 1
        :return:
        """

        # precision-recall area under curve
        aps = self.calculate_precision_recall_auc(X, y, index=index)

        # plotting curve
        lw = 2
        plot_precision_recall_curve(self.search_obj.best_estimator_, X, y)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"Precision-Recall curve: {aps}")
        plt.legend(loc="lower right")
        plt.show()

    def save_classifier_obj(self, fname: str):
        """
        To persist model classifier object as pkl file

        :param fname: file name to persist
        :return:
        """
        joblib.dump(self.search_obj.best_estimator_, fname)

    def load_classifier_obj(self, fname: str):
        """
        to read persisted model as pkl file

        :param fname: file name where persist the classifier
        :return:
        """

        self.classifier_obj = joblib.load(fname)


def make_classification(classifier_obj: object, params: dict,
                        X_train: pd.DataFrame, y_train: pd.DataFrame,
                        X_test: pd.DataFrame, y_test: pd.DataFrame,
                        search: object, cv: object, scoring: str) -> Classifier:
    """
    To train, valid, test, and show results of classifications.
    workflow:
    1. training
    2. testing
    3. reporting
        3.1. roc
        3.2. confusion matrix

    :param classifier_obj: instance of a sklearn classifier class
    :param params: dict of parameters of the classifier object
    :param X_train: features to train
    :param y_train: target to train
    :param X_test: features to test
    :param y_test: target to test
    :param search: sklearn hyper-parameters optimization class
    :param cv: sklearn cross validation object
    :param scoring: name of the function to score predictions
    :return: classifier_obj
    """
    
    classifier = Classifier(
               classifier_obj=classifier_obj,
               parameters=params,
               cv = cv,
               scoring=scoring,
               search_cls=search)
        
    
    # learning from X_train_enc
    classifier.learning(X_train, y_train)

    # predicting
    y_pred_train = classifier.predicting(X_train)

    y_pred_test = classifier.predicting(X_test)


    # reports
    report_train = classifier.get_classification_report(y_train, y_pred_train)
    report_test = classifier.get_classification_report(y_test, y_pred_test)

    # roc
    roc_train = classifier.calculate_roc_auc(X_train, y_train)
    roc_test = classifier.calculate_roc_auc(X_test, y_test)

    print("Main results trainset:")
    print(report_train)
    print(f"roc auc train: {roc_train}")
    classifier.plot_roc(X_train, y_train)
    classifier.plot_confusion_matrix(X_train, y_train)

    print("-----------------------------------------")

    print("Main results testset:")
    print(report_test)
    print(f"roc auc test: {roc_test}")
    classifier.plot_roc(X_test, y_test)
    classifier.plot_confusion_matrix(X_test, y_test)
    
    return classifier

