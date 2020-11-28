from create_training_data import create_training_data
from monitor_and_predict import monitor_and_predict
from train_and_test import train_and_test

# This script is the main CLI for the program. The user can select which
# action for the program to perform from the given list. Each function
# is contained within it's own script in this directory.

script_select = ""
while True:
    if script_select == "":
        script_select = input(
            "\n########################\n"
            "RR Share Price Predictor"
            "\n########################\n\n"
            "DISCLAIMER:\n"
            "Investing in the stock market involves risk and can lead to monetary loss. "
            "The predictions made by this program are not to be taken as financial advice."
            "\n\nWelcome to the RR Share Price predictor. This programs uses machine "
            "learning to predict Rolls-Royce plc. share price movement based on the latest"
            " related news article on the Reuters website.\n"
            "Please select from the following options [type 1, 2, 3 or 4 and press enter]:"
            "\n1. Activate monitoring\n"
            "2. Create training data\n"
            "3. Train and test classifier\n"
            "4. Exit\n")
    elif script_select == "1":
        monitor_and_predict()
        script_select = ""
    elif script_select == "2":
        create_training_data()
        script_select = ""
    elif script_select == "3":
        train_and_test()
        script_select = ""
    elif script_select == "4":
        print('Exiting...')
        exit()
    else:
        script_select = input("Invalid selection, must be 1, 2, 3 or 4\n")
