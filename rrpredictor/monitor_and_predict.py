import ctypes
import time

import requests
from bs4 import BeautifulSoup
from joblib import load


def monitor_and_predict():
    """
    Monitor Reuters for new articles and predict share change based latest.
    """

    # Run/return CLI control
    run_script = ""
    while run_script != "y":
        run_script = input(
            "\nThis script checks uk.reuters.com/companies/RR.L/news for new articles "
            "every 10 minutes and predicts share price movement based on the latest "
            "article.\nTo stop the script, press Ctrl-C.\n\n"
            "Continue? [y/n]\n")
        if run_script == "n":
            print("Returning to main menu...")
            return
        elif run_script != "y":
            print("Invalid input")

    print('\n####################\nBeginning monitoring\n####################')

    # For comparison
    old_article_URL = None
    while True:

        # Get and parse article list page HTML
        URL = 'https://uk.reuters.com/companies/RR.L/news'
        list_page = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(list_page.content, 'html.parser')

        # Find and store latest article URL
        new_article_URL = soup.find(
            "a", class_="MarketStoryItem-headline-2cgfz")['href']

        # Only process latest article if not done in a previous loop
        if new_article_URL != old_article_URL:

            # Get and parse latest article HTML
            article = requests.get(new_article_URL, headers={
                                   "User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(article.content, 'html.parser')

            # Find and get required text
            title = soup.find(
                'h1', class_='ArticleHeader-headline-NlAqj').get_text()
            text_list = []
            text_list.append(title)
            for para in soup.find_all('p', class_='ArticleBody-para-TD_9x'):
                text_list.append(para.text)

            # Prep text for tfidf vectorizer (must be string)
            text = ' '.join(text_list)

            # Load saved TF-IDF vectorizer and calculate vector for article text
            tfidf_vectorizer = load('tfidf_vectorizer.joblib')
            tfidf = tfidf_vectorizer.transform(
                [text])  # Text must be passed as string

            # Load saved classifier and make price change prediction
            clf = load('classifier.joblib')
            pred = clf.predict(tfidf)[0]

            print('\nLatest article: {}\nURL: {}\nPrediction: {}'.format(
                title, new_article_URL, pred.title()))

            # Alert user of new latest article
            ctypes.windll.user32.MessageBoxW(
                0, "Latest article: {}\nPrediction: {}".format(
                    title, pred.title()),
                "RR Predictor", 0)

        # Store article URL for comparison in next loop
        old_article_URL = new_article_URL

        # Run loop every 10 minutes
        time.sleep(600)
