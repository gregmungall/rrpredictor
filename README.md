# Rolls-Royce share price predictor

## Disclaimer

**Investing in the stock market involves risk and can lead to monetary loss. The
predictions made by this program are not to be taken as financial advice. This
project is purely educational and the predictions are not particularly
accurate**

## Overview

This project uses web scraping and machine learning techniques to predict
Rolls-Royce plc. share price movement from articles on the
[Reuters news website](https://www.reuters.com/companies/RR.L/news). The program
is split into four scripts. The
[rr_predictor.py](rrpredictor/rrpredictor.py) script is the main CLI for the
program where the user can select the action for the program to perform. The
[monitor_and_predict.py](rrpredictor/monitor_and_predict.py) script contains the
function that checks the Reuters website for the latest article and then uses
the saved machine learning model to predict the share price movement based on
the article. The [create_training_data.py](rrpredictor/create_training_data.py)
script contains the function that allows the user to update the data used to
train the machine learning model by scraping the Reuters website. The
[train_and_test.py](rrpredictor/train_and_test.py) script contains the function
that allows the user to retrain the machine learning model.  

More detail on each script is given below after usage.

## Usage

The training data and machine learning model are saved with this repository, so
there is no need to collect data and train the model before setting the
program to monitor and predict unless the user wishes to update the data used to
train the model.  

This program has been created and tested with Python 3.8.5. At the time of
writing, some packages are only available through Conda for this python version.
To ensure the program runs correctly, create an environment with the required
packages by running:

    conda env create -f environment.yml

Then, to run the program, navigate to rrpredictor directory (the directory that
contains [rr_predictor.py](rrpredictor/rrpredictor.py)) and run:

    python rr_predictor.py

Then follow the instructions given.

## Script detail

### [rr_predictor.py](rrpredictor/rrpredictor.py)

This script is the main CLI for the program. The user can select which action
for the program to perform from the given list. Each function from the other
scripts is imported here.

### [monitor_and_predict.py](rrpredictor/monitor_and_predict.py)

This script contains a function that checks the Reuters website every 10 minutes
for the latest Rolls-Royce news article. It uses the saved machine learning
model to predict the share price movement based on the article.

Key features:

-   **Web scraping:** The function uses
    [Requests](https://requests.readthedocs.io/en/master/) and
    [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) to
    get and parse the HTML from the
    [Rolls-Royce plc. news page on Reuters](https://www.reuters.com/companies/RR.L/news).
    Beautiful Soup was used here because it's quick and lightweight compared to
    Selenium which is used in
    [create_training_data.py](rrpredictor/create_training_data.py).
-   **Prediction:** The function uses the TF-IDF vectorizer and
    classifier, created and saved in
    [train_and_test.py](rrpredictor/train_and_test.py), to calculate the TF-IDF
    vector of the latest article and then from this predict it's classification.    

### [create_training_data.py](rrpredictor/create_training_data.py)

This script contains a function that extracts the 240 most recent Rolls-Royce
news articles from the Reuters website, calculates the percentage
change in share price across each article's publish date and assigns a
classification to each article (increase, neutral, decrease). This data is used
in [train_and_test.py](rrpredictor/train_and_test.py) to train and test the
TF-IDF vectorizer and classifier.

Key features:

-   **Web scraping:** The function uses
    [Selenium](https://selenium-python.readthedocs.io/) to load 240 articles on
    [Rolls-Royce plc. news page on Reuters](https://www.reuters.com/companies/RR.L/news)
    and then extract the required text (title, date, body text) from each
    article. The Reuters website uses a JavaScript infinite scroll mechanism to
    load more articles. Furthermore, each article's date is also added using
    JavaScript. Therefore, Selenium is used as it drives a browser and thus
    allows JavaScript to run. Requests, which is used in
    [monitor_and_predict.py](rrpredictor/monitor_and_predict.py), will only get
    the site's static HTML which does not include content added to the page
    using JavaScript.
-   **Memory utilisation monitoring:** Using Selenium can be memory intensive,
    and to reduce the script run time it's beneficial to hold all the article
    data as a list in memory before saving it to a CSV file. To prevent crashes,
    the function uses [psutil](https://github.com/giampaolo/psutil) to monitor
    memory usage whilst looping through articles. If the memory usage exceeds
    95% of the system's capacity, the function saves the list to the CSV file
    and clears the list in memory. If the memory usage remains over 95% the
    function exits the program.
-   **Numpy for searching:** The function uses [Numpy's](https://numpy.org/)
    .argmin() to find the index of the price record for each article's publish
    date (or the closest date after publish). This is quicker than other methods
    for longer lists
    ([see this benchmarking](https://stackoverflow.com/a/11825864)).
    This is beneficial as this method will be run for each iteration of the
    loop.

### [train_and_test.py](rrpredictor/train_and_test.py)

This script contains a function that trains and tests the TF-IDF vectorizer and
classifier used in [monitor_and_predict.py](rrpredictor/monitor_and_predict.py)
using the data created in
[create_training_data.py](rrpredictor/create_training_data.py). The machine
learning library used is [scikit-learn](https://scikit-learn.org/stable/).

Key features:

-   **Split train and test data:** To avoid overfitting, the function randomly
    selects 20% of the training data and reserves it for testing the model.
-   **TF-IDF vectorizer:** The machine learning models used in this program
    expect numerical feature vectors with a fixed size. Therefore,
    the function uses TF-IDF vectorization to transform the corpus into this
    format. Due to the size of the corpus,
    [TF-IDF term weighting](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
    is used to reduce the impact of common features between articles. The
    vectorizer is trained with the training corpus.
-   **Model selection:** To create the optimal classifier, the function
    benchmarks four classification models: Naive Bayes (multinomial variant),
    Logistic Regression, Linear Support Vector, and Random Forest. The
    function selects the most accurate model for the classifier by testing each
    one with the test data. The accuracy of each model and the model used in the
    saved classifier are shown in the Test results section below.
-   **Confusion matrix:** Once the classifier is created, the function creates
    a confusion matrix to visualise the accuracy. For the saved classifier, this
    is shown in the Test results section below.

## Test results

The training corpus size is limited as there are only 240 articles available on
the
[Rolls-Royce plc. news page on Reuters](https://www.reuters.com/companies/RR.L/news).
Therefore, the classifier accuracy will also be limited. Furthermore, this
project only implements a simple testing procedure. This is suitable here as
this is purely educational, but a commercial or academic classifier would
benefit from a significantly larger corpus and a more rigorous testing
procedure.

### Most correlated n-grams

Below is a list of n-grams most correlated with each classification. These have
been determined using the training data saved in this repository.

#### Classification: 'decrease':

Most correlated unigrams:

-   covid
-   19

Most correlated bigrams:

-   19 pandemic
-   covid 19

Most correlated trigrams:

-   raise billion pounds
-   covid 19 pandemic

#### Classification: 'increase':

Most correlated unigrams:

-   xwb
-   systems

Most correlated bigrams:

-   cash flow
-   600 jobs

Most correlated trigrams:

-   east told investors
-   free cash flow

#### Classification: 'neutral':

Most correlated unigrams:

-   looked
-   project

Most correlated bigrams:

-   air new
-   royce trent

Most correlated trigrams:

-   royce trent 1000
-   rolls royce trent

### Model benchmarking

Below are the results from benchmarking the four classification models to
determine the most accurate model to be used for the classifier. These results
have been determined using the training data saved in this repository.

| Model                  | Accuracy |
| ---------------------- | :------: |
| RandomForestClassifier | 0.562500 |
| LinearSVC              | 0.666667 |
| MultinomialNB          | 0.562500 |
| LogisticRegression     | 0.666667 |

The LinearSVC and LogisticRegression models have equal accuracy so LinearSVC was
arbitrarily chosen.

### Confusion matrix

Below is the confusion matrix generated from the saved classifier's test
results using the test data.

![Confusion matrix](images/Confusion_matrix.png?raw=true "Confusion matrix")
