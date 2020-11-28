import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def train_and_test():
    """
    Create and test TF-IDF vectorizer and classifier.
    """

    # Run/return CLI control
    run_script = ""
    while run_script != "y":
        run_script = input(
            "\nWARNING: If tfidf_vectorizer.joblib and classifier.joblib exist, this"
            " script will replace them.\nContinue? [y/n]\n")
        if run_script == "n":
            print("Returning to main menu...")
            return
        elif run_script != "y":
            print("Invalid input")

    print('\n####################\nTraining and testing\n####################')

    # Get and split records into training and testing sets from csv
    training_df = pd.read_csv(
        '../data/training_data.csv', encoding='cp1252', header=None,
        names=['Article', 'Publish_date', 'Percentage_change', 'Classification'])
    X_train, X_test, y_train, y_test = train_test_split(
        training_df['Article'],
        training_df['Classification'],
        test_size=0.2, random_state=0)

    # Initialise TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        sublinear_tf=True, min_df=6, max_df=50,
        ngram_range=(1, 3), stop_words='english')

    # Learn vocabulary and idf from training corpus
    # Calculate document-term matrix for training and testing corpora
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    print('\nTF-IDF vectorizer created')

    print('\nFinding most correlated n-grams for each classification...')

    # Find most correlated n-grams for each classification
    # N = number of most correlated n-grams to display
    N = 2
    for classification in sorted(set(training_df['Classification'])):

        # Calculate chi^2 for each feature relevant to the loop classification
        feature_chi2 = chi2(tfidf_train, y_train == classification)

        # Sort feature names (n-grams) by chi-squared
        # chi2 returns array and shape, hence [0]
        idx = np.argsort(feature_chi2[0])
        feature_names = np.array(tfidf_vectorizer.get_feature_names())[idx]

        # Create list of features, sorted by chi-squared, for each n-gram
        unigrams = [feature for feature in feature_names if len(
            feature.split(' ')) == 1]
        bigrams = [feature for feature in feature_names if len(
            feature.split(' ')) == 2]
        trigrams = [feature for feature in feature_names if len(
            feature.split(' ')) == 3]

        # Display N most correlated features for each n-gram type
        print("\n# Classification: '{}':".format(classification))
        print(
            "Most correlated unigrams:\n- {}".format('\n- '.join(unigrams[-N:])))
        print(
            "Most correlated bigrams:\n- {}".format('\n- '.join(bigrams[-N:])))
        print(
            "Most correlated trigrams:\n- {}".format('\n- '.join(trigrams[-N:])))

    print('\nBenchmarking models...\n')

    # List of classification models to benchmark and select from
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]

    # Train classifiers using each model and get accuracy using test data
    bm_results_list = []
    for model in models:
        clf = model.fit(tfidf_train, y_train)
        pred = clf.predict(tfidf_test)
        accuracy = metrics.accuracy_score(y_test, pred)
        bm_results_list.append([model.__class__.__name__, accuracy])

    # Display results from benchmarking
    bm_results_df = pd.DataFrame(
        bm_results_list, columns=['Model', 'Accuracy'])
    print(bm_results_df)

    # Get index of most accurate model
    model_idx = bm_results_df['Accuracy'].idxmax(0)

    print('\n{} selected as model'.format(bm_results_df['Model'][model_idx]))

    # Train classifier using most accurate model
    model = models[model_idx]
    clf = model.fit(tfidf_train, y_train)

    print('\nClassifier created')

    # Save TF-IDF vectorizer and classifier for monitor_and_predict.py
    dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
    dump(clf, 'classifier.joblib')

    print('\nTF-IDF vectorizer and classifier saved')

    print('\nDisplaying confusion matrix...')

    # Create and plot confusion matrix
    pred = clf.predict(tfidf_test)
    confusion_matrix = metrics.confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt='d',
                xticklabels=sorted(set(training_df['Classification'])),
                yticklabels=sorted(set(training_df['Classification'])))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    fig.canvas.manager.set_window_title("Confusion matrix")
    plt.show()

    print('\nReturning to main menu...')
    return
