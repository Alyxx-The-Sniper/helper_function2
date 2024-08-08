{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0b02d67b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_and_evaluate(X, y, model,\n",
    "                       test_size=0.3, \n",
    "                       random_state=42\n",
    "                       ):\n",
    "\n",
    "    \n",
    "            \"\"\"\n",
    "            train_and_evaluate\n",
    "\n",
    "            This function trains and evaluates a classification model using a pipeline that includes feature scaling and classification. \n",
    "\n",
    "            Parameters:\n",
    "            - X (pd.DataFrame or np.ndarray): Features of the dataset.\n",
    "            - y (pd.Series or np.ndarray): Target variable of the dataset.\n",
    "            - classifier (object): A scikit-learn classifier instance (e.g., LogisticRegression, RandomForestClassifier, LinearRegression) to be used for training and prediction.\n",
    "            - test_size (float, optional): Proportion of the dataset to be used as the test set. Default is 0.3.\n",
    "            - random_state (int, optional): Seed used by the random number generator for reproducibility. Default is 42.\n",
    "\n",
    "            Returns:\n",
    "            - train_score (float): Accuracy score of the model on the training set.\n",
    "            - test_score (float): Accuracy score of the model on the test set.\n",
    "\n",
    "            The function performs the following steps:\n",
    "            1. Splits the dataset into training and testing sets based on the provided `test_size` and `random_state`.\n",
    "            2. Constructs a pipeline that scales features and applies the specified classifier.\n",
    "            3. Trains the model on the training set.\n",
    "            4. Evaluates and returns the accuracy scores for both the training and testing sets.\n",
    "            \"\"\"\n",
    "\n",
    "            # Split the dataset into training and testing sets\n",
    "            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)\n",
    "\n",
    "\n",
    "            pipeline = Pipeline([\n",
    "                ('scaler', StandardScaler()),\n",
    "        #         ('selector', SelectKBest(score_func=f_classif, k=2)),\n",
    "                ('model', model)\n",
    "            ])\n",
    "\n",
    "            # Fit the model\n",
    "            pipeline.fit(X_train, y_train)\n",
    "\n",
    "            # Evaluate the model\n",
    "            train_score = pipeline.score(X_train, y_train)\n",
    "            test_score = pipeline.score(X_test, y_test)\n",
    "\n",
    "            return train_score, test_score"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
