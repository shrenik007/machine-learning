# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def performPreprocessing(titanic):

    #TODO: Complete preprocessing here
    # dropping columns
    titanic = titanic.drop(columns=['Name', 'Cabin', 'Ticket'])

    # handling missing values and converting values of NaN to mean for column Age and Fare. #
    # For column Embarked converting NaN to most frequent values
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(titanic[['Age']])
    titanic['Age'] = imp.transform(titanic[['Age']]).ravel()

    imp.fit(titanic[['Fare']])
    titanic['Fare'] = imp.transform(titanic[['Fare']]).ravel()

    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp.fit(titanic[['Embarked']])
    titanic['Embarked'] = imp.transform(titanic[['Embarked']]).ravel()

    # Normalize Data - MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(titanic[['Age']])
    titanic['Age'] = scaler.transform(titanic[['Age']])

    scaler.fit(titanic[['SibSp']])
    titanic['SibSp'] = scaler.transform(titanic[['SibSp']])

    scaler.fit(titanic[['Parch']])
    titanic['Parch'] = scaler.transform(titanic[['Parch']])

    scaler.fit(titanic[['Fare']])
    titanic['Fare'] = scaler.transform(titanic[['Fare']])

    scaler.fit(titanic[['Pclass']])
    titanic['Pclass'] = scaler.transform(titanic[['Pclass']])

    # Step 6 - Encoding Categorical Variables
    titanic['Sex'] = titanic['Sex'].map({'male': 1, 'female': 0})
    titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    print(titanic)
    return titanic


def main():


    # Open the training dataset as a dataframe and perform preprocessing
    titanic_train = pd.read_csv("data/lab9/train.csv", delimiter=",")
    titanic_test = pd.read_csv("data/lab9/test.csv", delimiter=",")


    # Merge the two datasets into one dataframe
    titanic_test["Survived"] = np.zeros(len(titanic_test))
    titanic_test["Survived"] = -1
    frameList = [titanic_train, titanic_test]
    allData = pd.concat(frameList, ignore_index=True, sort='True')

    # Run preprocessing. Seperate the resulting data into test and train

    allData = performPreprocessing(allData)

    # separating train and test data
    train_data = allData[allData['Survived'] != -1]
    test_data = allData[allData['Survived'] == -1]

    train_data = train_data.drop(columns=['PassengerId'])

    train_features = train_data.drop(columns=['Survived'])
    train_labels = train_data['Survived']

    test_passengerID_series = test_data['PassengerId']
    test_data = test_data.drop(columns=['PassengerId'])
    test_features = test_data.drop(columns=['Survived'])

    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(train_features, train_labels)

    predicted = neigh.predict(test_features)
    resultSeries = pd.Series(data=predicted, name='Survived', dtype='int64')
    df = pd.DataFrame({"PassengerId": test_passengerID_series.reset_index(drop=True), "Survived": resultSeries})
    df.to_csv("submission.csv", index=False, header=True)

main()
