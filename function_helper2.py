## kaiku
## libraires
import time
from tqdm import tqdm
import pandas as pd
pd.set_option('display.max_colwidth', None)  # Show full content of each column
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# kbest
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
#RFE
from sklearn.feature_selection import RFE
#SFFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import warnings
warnings.filterwarnings('ignore')

#########################################################################################################
def train_and_evaluate(X, y, model,
                       test_size=0.3,
                       random_state=42):
    """
    train_and_evaluate

    This function trains and evaluates a classification model using a pipeline that includes feature scaling and classification.

    Parameters:
    - X (pd.DataFrame or np.ndarray): Features of the dataset.
    - y (pd.Series or np.ndarray): Target variable of the dataset.
    - model (object): A scikit-learn classifier instance (e.g., LogisticRegression, RandomForestClassifier, LinearRegression)
      to be used for training and prediction.
    - test_size (float, optional): Proportion of the dataset to be used as the test set. Default is 0.3.
    - random_state (int, optional): Seed used by the random number generator for reproducibility. Default is 42.

    Returns:
    - train_score (float): Accuracy score of the model on the training set.
    - test_score (float): Accuracy score of the model on the test set.
    """

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Construct the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # Time and progress bar for training
    with tqdm(total=1, desc="Training", unit="task") as pbar:
        start_train_time = time.time()
        pipeline.fit(X_train, y_train)
        end_train_time = time.time()
        pbar.update(1)  # Update progress bar

    # Time and progress bar for evaluation
    with tqdm(total=1, desc="Evaluation", unit="task") as pbar:
        start_eval_time = time.time()
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        end_eval_time = time.time()
        pbar.update(1)  # Update progress bar

    return train_score, test_score

##########################################################################################################
def K_best_score_list(X, y, score_func):
    """
    K_best_score_list

    This function evaluates the importance of each feature in a dataset using the SelectKBest feature selection method. It calculates and returns a DataFrame containing the scores and p-values for each feature.

    Parameters:
    - score_func (callable): The scoring function to use for feature selection. This function should be compatible with scikit-learn's `SelectKBest`, such as `f_classif` for classification or `f_regression` for regression.

    Returns:
    - feature_scores (pd.DataFrame): A DataFrame containing the scores and p-values for each feature, sorted by the score in descending order. The DataFrame has three columns:
      - 'Feature': The name of each feature.
      - 'Score': The score assigned to each feature by the scoring function.
      - 'p-Value': The p-value associated with each feature.

    The function performs the following steps:
    1. Splits the dataset into training and testing sets with a test size of 0.3 and a random state of 42.
    2. Normalizes the features using `StandardScaler`.
    3. Applies `SelectKBest` with the specified scoring function to the normalized features.
    4. Creates a DataFrame with feature names, scores, and p-values.
    5. Sorts the DataFrame by the scores in descending order.
    6. Returns the sorted DataFrame.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    selector = SelectKBest(score_func, k='all')
    x_train_kbest = selector.fit_transform(X_train, y_train)
    x_test_kbest = selector.transform(X_test)

    feature_scores = pd.DataFrame({'Feature': X.columns,
                                   'Score': selector.scores_,
                                   'p-Value': selector.pvalues_})

    feature_scores = feature_scores.sort_values(by='Score', ascending=False)
    return feature_scores

##########################################################################################################
def kbest_evaluate_features(X, y,
                            model,
                            score_func,
                            task='classification',
                            test_size=0.3,
                            random_state=42):
    """
    Evaluates the performance of a model using different numbers of features selected by the K Best method.
    This function performs feature selection for varying values of `k`, trains the model, and calculates
    accuracy scores for classification tasks and R² scores for regression tasks.

    Parameters:
    - X (pd.DataFrame or np.ndarray): Features of the dataset.
    - y (pd.Series or np.ndarray): Target variable of the dataset.
    - model (object): A scikit-learn model instance (e.g., LogisticRegression, RandomForestClassifier,
      LinearRegression) to be used for training and prediction.
    - score_func (callable): The scoring function to use with `SelectKBest` for feature selection
      (e.g., `f_classif` or `mutual_info_classif` for classification; `f_regression` for regression).
    - task (str, optional): The type of task ('classification' or 'regression'). Default is 'classification'.
    - test_size (float, optional): Proportion of the dataset to be used as the test set. Default is 0.3.
    - random_state (int, optional): Seed used by the random number generator for reproducibility. Default is 42.

    Returns:
    - result_df (pd.DataFrame): A DataFrame containing the results of the evaluation.
      The DataFrame has the following columns:
      - 'k': The number of features selected.
      - 'score': The performance score of the model with the selected features (accuracy or R²).
      - 'selected_features': A list of feature names selected for each value of `k`.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    score_list = []
    selected_features_list = []

    # Add tqdm progress bar
    for k in tqdm(range(1, len(X.columns) + 1), desc="Evaluating Features", unit="feature"):
        selector = SelectKBest(score_func, k=k)
        X_train_kbest = selector.fit_transform(X_train_scaled, y_train)
        X_test_kbest = selector.transform(X_test_scaled)

        model.fit(X_train_kbest, y_train)
        y_preds_kbest = model.predict(X_test_kbest)

        if task == 'classification':
            # Calculate the accuracy score
            score = accuracy_score(y_test, y_preds_kbest)

            # Optionally calculate AUC score if it's a binary classification
            if len(np.unique(y)) == 2:
                y_probs_kbest = model.predict_proba(X_test_kbest)[:, 1]
                auc_score_kbest = roc_auc_score(y_test, y_probs_kbest)
            else:
                auc_score_kbest = None

        elif task == 'regression':
            # Calculate the R² score
            score = r2_score(y_test, y_preds_kbest)
            auc_score_kbest = None

        else:
            raise ValueError("Invalid task type. Choose 'classification' or 'regression'.")

        score_list.append(score)

        # Get selected feature names
        selected_feature_mask = selector.get_support()
        selected_features = X.columns[selected_feature_mask].tolist()
        selected_features_list.append(selected_features)

    x = np.arange(1, len(X.columns) + 1)
    result_df = pd.DataFrame({
        'k': x,
        'score': score_list,
        'selected_features': selected_features_list
    })

    return result_df

##########################################################################################################
def rfe_evaluate_features(X, y, model, task='classification'):
    """
    Evaluates the performance of a model using Recursive Feature Elimination (RFE) for different numbers
    of features. This function performs feature selection for varying values of `k`, trains the model, and calculates
    appropriate metrics based on the task type (classification or regression). It also provides a list of selected features
    for each value of `k`.

    Parameters:
    - X (pd.DataFrame or np.ndarray): Features of the dataset.
    - y (pd.Series or np.ndarray): Target variable of the dataset.
    - model (object): A scikit-learn model instance (e.g., LogisticRegression, RandomForestClassifier, LinearRegression)
      to be used for training and prediction.
    - task (str, optional): The type of task ('classification' or 'regression'). Default is 'classification'.

    Returns:
    - result_df (pd.DataFrame): A DataFrame containing the results of the evaluation. The DataFrame includes the
      following columns:
      - 'k': The number of features selected.
      - 'score': The performance score of the model with the selected features (accuracy, AUC, MSE, or R²).
      - 'selected_features': A list of feature names selected for each value of `k`.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    score_list = []
    selected_features_list = []

    # Add tqdm progress bar
    for k in tqdm(range(1, len(X.columns) + 1), desc="Evaluating Features", unit="feature"):
        # Initialize RFE
        rfe = RFE(estimator=model, n_features_to_select=k)
        X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
        X_test_rfe = rfe.transform(X_test_scaled)

        # Fit the model
        model.fit(X_train_rfe, y_train)
        y_preds_rfe = model.predict(X_test_rfe)

        # Evaluate based on the task type
        if task == 'classification':
            # Calculate accuracy score
            score = accuracy_score(y_test, y_preds_rfe)

            # Optionally calculate AUC score for binary classification
            if len(np.unique(y)) == 2:
                y_probs_rfe = model.predict_proba(X_test_rfe)[:, 1]
                auc_score_rfe = roc_auc_score(y_test, y_probs_rfe)
                score = auc_score_rfe  # Use AUC score if it's available

        elif task == 'regression':
            # Calculate MSE and R² score
#             mse = mean_squared_error(y_test, y_preds_rfe)
            r2 = r2_score(y_test, y_preds_rfe)
            # You can choose to return MSE or R², currently returning MSE
            score = r2  # Or use r2 if preferred

        else:
            raise ValueError("Invalid task type. Choose 'classification' or 'regression'.")

        score_list.append(score)

        # Get selected feature names
        selected_feature_mask = rfe.get_support()
        selected_features = X.columns[selected_feature_mask].tolist()
        selected_features_list.append(selected_features)

    x = np.arange(1, len(X.columns) + 1)
    result_df = pd.DataFrame({
        'k': x,
        'score': score_list,
        'selected_features': selected_features_list
    })

    return result_df

##########################################################################################################
def sffs_sbfs_evaluate_features(X, y,
                                model,
                                task='classification',
                                forward=True,
                                floating=False,
                                scoring='accuracy',
                                cv=0,
                                test_size=0.3,
                                random_state=42,
                                ):
    """
    Performs feature selection using Sequential Forward Selection (SFS) or Sequential Backward Selection (SBS)
    for both classification and regression tasks.

    Parameters:
    - X (pd.DataFrame or np.ndarray): Features of the dataset.
    - y (pd.Series or np.ndarray): Target variable of the dataset.
    - model (object): A scikit-learn model instance (e.g., LogisticRegression, RandomForestClassifier, LinearRegression)
      to be used for training and prediction.
    - test_size (float, optional): Proportion of the dataset to be used as the test set. Default is 0.3.
    - random_state (int, optional): Seed used by the random number generator for reproducibility. Default is 42.
    - forward (bool, optional): Whether to use forward selection. Default is True.
    - floating (bool, optional): Whether to use floating feature selection. Default is False.
    - scoring (str, optional): The scoring metric to use ('accuracy', 'roc_auc' for classification; 'neg_mean_squared_error' for regression). Default is 'accuracy'.
    - cv (int, optional): Number of cross-validation folds. Default is 5.
    - task (str, optional): The type of task ('classification' or 'regression'). Default is 'classification'.

    Returns:
    - results_df (pd.DataFrame): A DataFrame containing the results of the evaluation. The DataFrame includes the
      following columns:
      - 'Number of Features': The number of features selected.
      - 'Selected Features': A list of feature names selected for each value of `k`.
      - 'Score': The performance score of the model with the selected features.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize lists to store results
    selected_features = []
    scores = []

    # Iterate over different numbers of features
    for k in tqdm(range(1, X.shape[1] + 1), desc="Evaluating Features", unit="feature"):
        # Initialize the Sequential Feature Selector
        sfs = SFS(model,
                  k_features=k,
                  forward=forward,
                  floating=floating,
                  scoring=scoring,
                  cv=cv)

        # Fit the Sequential Feature Selector to the training data
        sfs.fit(X_train_scaled, y_train)

        # Transform the data to only include the selected features
        X_train_selected = sfs.transform(X_train_scaled)
        X_test_selected = sfs.transform(X_test_scaled)

        # Train the model using only the selected features
        model.fit(X_train_selected, y_train)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test_selected)

        if task == 'classification':
            if scoring == 'accuracy':
                score = accuracy_score(y_test, y_pred)
            elif scoring == 'roc_auc':
                y_probs = model.predict_proba(X_test_selected)[:, 1]
                score = roc_auc_score(y_test, y_probs)
            else:
                raise ValueError("Unsupported scoring metric for classification. Use 'accuracy' or 'roc_auc'.")

        elif task == 'regression':
            if scoring == 'neg_mean_squared_error':
                score = -mean_squared_error(y_test,
                                            y_pred)  # Note: SFS returns negative MSE, so negate to get positive MSE
            elif scoring == 'r2':
                score = r2_score(y_test, y_pred)
            else:
                raise ValueError("Unsupported scoring metric for regression. Use 'neg_mean_squared_error' or 'r2'.")

        else:
            raise ValueError("Invalid task type. Choose 'classification' or 'regression'.")

        # Store results
        selected_features.append([X.columns[i] for i in sfs.k_feature_idx_])
        scores.append(score)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Number of Features': list(range(1, X.shape[1] + 1)),
        'Selected Features': selected_features,
        'Score': scores
    })

    return results_df

##########################################################################################################
def count_outliers(df):
    outlier_counts = {}
    for col in df.columns:
        if df[col].dtype != 'object':  # Exclude non-numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            lower_bound_outliers = df[df[col] < lower_bound]
            upper_bound_outliers = df[df[col] > upper_bound]
            total_outliers = len(lower_bound_outliers) + len(upper_bound_outliers)
            outlier_counts[col] = total_outliers
    return outlier_counts

##########################################################################################################
def replace_outliers_with_mean(df):
    for col in df.columns:
        if df[col].dtype != 'object':  # Exclude non-numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            lower_bound_outliers = df[col] < lower_bound
            upper_bound_outliers = df[col] > upper_bound

            # Replace outliers with the column mean
            col_mean = df[col].mean()
            df[col][lower_bound_outliers | upper_bound_outliers] = col_mean

    return df
