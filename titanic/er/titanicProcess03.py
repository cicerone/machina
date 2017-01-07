# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 08:34:00 2016

@author: erfan
"""

cabinMap = {'A': 1, 'B' : 2, 'C' : 3, 'D' : 4, 'E' : 5, 'F' : 6, 'G' : 7 , 'T' : 8}
def mapCabin(str):
    
    if not isinstance(str, float):
        listOfSplit = str.split( )
        numList = []
#        for cabin in listOfSplit:
#            if len(cabin)>1:
#                numList.append(cabinMap[str[0]]*1000 + int(cabin[1:]))
#            else:
#                numList.append(cabinMap[str[0]]*1000)
            
#            num = sum(numList)*1.0/len(numList)
        num = cabinMap[str[0]]
        return num
    else:
        return str

def map2Familyname(str):
    if not isinstance(str, float):
        tmp = str.split(',')
        return tmp[0]
    else:
        return str

import pandas as pd

uniqueTitleDic = {'Capt': 1, 'Col': 2, 'Don': 3, 'Dona': 4, 'Dr': 5, 'Jonkheer': 6, 'Lady': 7, 'Major': 8, \
'Master': 9, 'Miss': 10, 'Mlle': 11, 'Mme': 12, 'Mr': 13, 'Mrs': 14, 'Ms': 15, 'Rev': 16, 'Sir':17, 'the Countess': 18};
def findTitle(fullName):
    nameParts = fullName.split(', ')
    title = nameParts[1].split('.')
    return uniqueTitleDic[title[0]]

dfTrain = pd.read_csv('train.csv', header=0)
dfTrain['Sex'] = dfTrain['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
dfTrain['Embarked'] = dfTrain['Embarked'].map({'S': 0, 'C': 1, 'Q':2})
dfTrain['Cabin'] = dfTrain['Cabin'].map(lambda strObj: mapCabin(strObj))
dfTrain['Title'] = dfTrain['Name'].map(lambda strObj: findTitle(strObj))
dfTrain['Name'] = dfTrain['Name'].map(lambda strObj: map2Familyname(strObj))

dfTest = pd.read_csv('test.csv', header=0)
dfTest['Sex'] = dfTest['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
dfTest['Embarked'] = dfTest['Embarked'].map({'S': 0, 'C': 1, 'Q':2})
dfTest['Cabin'] = dfTest['Cabin'].map(lambda strObj: mapCabin(strObj))
dfTest['Title'] = dfTest['Name'].map(lambda strObj: findTitle(strObj))
dfTest['Name'] = dfTest['Name'].map(lambda strObj: map2Familyname(strObj))


listOfUniqueNamesInTrainTest = list(set(list(dfTrain['Name'].unique()) + list(dfTest['Name'].unique())))

def familyname2Num(str):
    numOfNewFamilynameInTrainTest = 0
    if str in listOfUniqueNamesInTrainTest:
        return listOfUniqueNamesInTrainTest.index(str)
    else:
        numOfNewFamilynameInTrainTest = numOfNewFamilynameInTrainTest + 1
        return numOfNewFamilynameInTrainTest
    
dfTrain['Name'] = dfTrain['Name'].map(lambda str: familyname2Num(str))    
del dfTrain['Ticket']
del dfTrain['PassengerId']

dfTest['Name'] = dfTest['Name'].map(lambda str: familyname2Num(str))    
del dfTest['Ticket']
del dfTest['PassengerId']


from randomForest import fillNan, Rfm

traindf = fillNan(dfTrain)
testdf = fillNan(dfTest)

featuresWithNoNan = []
featuresWithNan = []
for feature in list(dfTrain.columns.values):
    if pd.notnull(dfTrain[feature]).all():
        featuresWithNoNan.append(feature)
    else:
        featuresWithNan.append(feature)

entireFeatureSet = featuresWithNan + featuresWithNoNan
del entireFeatureSet[entireFeatureSet.index('Survived')]

trainData = traindf[sorted(entireFeatureSet)].values
targetData = traindf['Survived'].values
testData = testdf[sorted(entireFeatureSet)].values

rfm = Rfm(trainData, targetData)
rfm.train()
testVotes = rfm.getVotesOnTest(testData)

import numpy as np
testDecisions = rfm.uniqueClassCodes[np.argmax(testVotes, axis = 1)]
results = [[index+892, d] for index, d in enumerate(testDecisions)]
results = np.array(results, dtype=int)
dfTestLabeled = pd.read_csv('testLabeled.csv', header=0)
actualClassCodes = dfTestLabeled['Survived'].values
print(1 - np.sum(np.abs(actualClassCodes - results[:, 1]))/418.0)

maxRowIndex = results.shape[0] - 1
maxColIndex = results.shape[1] - 1
outfile = open('result.csv', 'w')
outfile.write('PassengerId,Survived\n')
for rowIndex in range(results.shape[0]):
    for colIndex in range(results.shape[1]):
        outfile.write('%d' % results[rowIndex, colIndex])
        if colIndex < maxColIndex:
            outfile.write(',')
    if rowIndex < maxRowIndex:
        outfile.write('\n')

outfile.close()