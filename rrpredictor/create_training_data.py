import csv
import os
from datetime import datetime, timedelta

import chromedriver_binary
import numpy as np
import psutil
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys


def create_training_data():
    """
    Get articles and store with share price change over article date.
    """

    # Run/return CLI control
    run_script = ""
    while run_script != "y":
        run_script = input(
            "\nWARNING: If ..\data\\training_data.csv exists, this script will replace it."
            " Check that ../data/RR.L.csv is up to date with "
            "uk.reuters.com/companies/RR.L/news.\nContinue? [y/n]\n")
        if run_script == "n":
            print("Returning to main menu...")
            return
        elif run_script != "y":
            print("Invalid input")

    print(
        '\n######################\nCreating training data\n######################')

    # Start webdriver and get the article list page
    print('\nStarting webdriver...')
    opts = Options()
    opts.set_headless()
    opts.add_argument('log-level=3')  # Supress log (not fatal errors)
    opts.add_experimental_option(
        'excludeSwitches', ['enable-logging'])  # As above
    driver = webdriver.Chrome(options=opts)
    driver.implicitly_wait(120)
    URL = 'https://uk.reuters.com/companies/RR.L/news'
    driver.get(URL)

    # Close cookie pop up
    print('\nClosing cookie popup...')
    button = driver.find_element_by_id("_evidon-banner-acceptbutton")
    button.click()

    # Scroll down to load required no. of articles and store their anchor
    # tags
    print('\nFinding articles...')
    articles = driver.find_elements_by_class_name(
        "MarketStoryItem-headline-2cgfz")
    no_of_articles = len(articles)
    # Multiples of 20 (there are 20 articles per page)
    while no_of_articles < 240:
        driver.find_element_by_tag_name("body").send_keys(Keys.PAGE_DOWN)
        articles = driver.find_elements_by_class_name(
            "MarketStoryItem-headline-2cgfz")
        no_of_articles = len(articles)
    print("\nNumber of articles found: {}".format(no_of_articles))

    # Prep price data csv for reading
    print('\nGetting share price data...')
    prices_csv = open('../data/RR.L.csv', 'r', newline='')
    prices_reader = csv.reader(prices_csv, delimiter=',')
    next(prices_reader, None)  # Skip header

    # Create list of closing prices and dates
    price_list = []
    for record in prices_reader:
        if record[4] != 'null':
            record_list = [datetime.strptime(record[0], '%Y-%m-%d'), record[4]]
            price_list.append(record_list)

    prices_csv.close()

    print('\nNumber of closing price records found: {}'.format(len(price_list)))

    print('\nCreating training data [memory usage limit 95%]...')

    # Delete existing training data if it exists
    try:
        os.remove('../data/training_data.csv')
    except OSError:
        pass

    # Create training data csv and prepare for writing
    training_data_csv = open('../data/training_data.csv',
                             'a', newline='', encoding='cp1252')
    training_data_writer = csv.writer(training_data_csv)

    # Loop to extract relevent text from each article
    data_list = []
    for idx, article in enumerate(articles):

        # Open article link in new tab and switch to it
        article.send_keys(Keys.CONTROL + Keys.RETURN)
        driver.switch_to.window(driver.window_handles[1])

        # Find and get text from relevent elements, store as string
        text_list = []
        text_list.append(
            driver.find_element_by_class_name('ArticleHeader-headline-NlAqj').text)
        for para in driver.find_elements_by_class_name('ArticleBody-para-TD_9x'):
            text_list.append(para.text)
        text = ' '.join(text_list)

        # Find and get publish date
        date = driver.find_element_by_class_name(
            'ArticleHeader-date-Goy3y').text
        date = datetime.strptime(date, '%B %d, %Y')

        # Find index of price record on date of publish, or closest date after
        price_date_index = np.array(
            [record[0] - date
             if record[0] >= date
             else timedelta.max
             for record in price_list]
        ).argmin(0)

        # Find percentage change from previous close price to next close price
        perc_change = 100 * (
            (float(price_list[price_date_index + 1][1])
             - float(price_list[price_date_index - 1][1]))
            / float(price_list[price_date_index - 1][1]))

        # Assign article classification based on percentage change
        if abs(perc_change) < 1:
            classification = 'neutral'
        elif perc_change < 0:
            classification = 'decrease'
        else:
            classification = 'increase'

        # Append article record to full data list
        record = [text, date, perc_change, classification]
        data_list.append(record)

        print('\nNumber of articles processed: {}'.format(idx + 1))

        # Monitor memory usage to ensure does not exceed set limit of 95%
        # If memory exceeds 95% save to disk and check again
        # If continues to exceed 95% close dirver and exit program
        memory_usage = psutil.virtual_memory().percent
        print("Current memory usage: {}%".format(memory_usage))
        if memory_usage > 95.0:
            print("Current memory usage exceeds 95%. Saving data to disk..")
            training_data_writer.writerows(data_list)
            data_list = []
            if memory_usage > 95.0:
                print("Memory usage still exceeds 95%. Exiting...")
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                driver.close()
                exit()

        # Close tab and switch back to article list page
        driver.close()
        driver.switch_to.window(driver.window_handles[0])

    # Once all articles processed close driver
    driver.close()

    # Save data list if has not been saved due to exceeding memory limit
    if len(data_list) > 0:
        training_data_writer.writerows(data_list)

    print('\nTraining data created. Number of records in training data: {}'.format(
        no_of_articles))

    training_data_csv.close()

    print('\nReturning to main menu...')
    return
