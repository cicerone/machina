# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 19:39:41 2016

@author: erfan
"""

from sklearn.ensemble import RandomForestClassifier 
from numpy import argmax, zeros, inf, log2, array, histogram, sum

class Rfm:
    def __init__(self, trainData=None, trainTargets=None,\
    testData=None, testTargets=None, numOfTrees = 500):
        self.trainData = trainData
        self.trainTargets =  trainTargets
        self.testData = testData
        self.testTargets = testTargets
        self.numOfTrees = numOfTrees
        
    def train(self):
        self.rf = RandomForestClassifier(oob_score=True, min_samples_split = 1, n_estimators = self.numOfTrees)
        self.rf = self.rf.fit(self.trainData, self.trainTargets)
        self._calcTrainOob()
        self._calcTrainEntropy()
        
    def _calcTrainOob(self):
        self.uniqueClassCodes = self.rf.classes_
        self.unique2Index = {}
        cnt = 0
        for uniqueClassCode in self.uniqueClassCodes:
            self.unique2Index[uniqueClassCode] = cnt
            cnt += 1
        self.oobVotes = self.rf.oob_decision_function_
        self.oobDecisions = argmax(self.oobVotes, axis = 1)
        self.oobDecisions = self.uniqueClassCodes[self.oobDecisions]
        self.oobConfMatrix = zeros((len(self.uniqueClassCodes), len(self.uniqueClassCodes)), dtype = float)
        for d, t in zip(self.oobDecisions, self.trainTargets):
            self.oobConfMatrix[self.unique2Index[d], self.unique2Index[t]] += 1
            
    def _calcTrainEntropy(self):
        tmp = log2(self.oobVotes)
        tmp[tmp == -inf] = 0
        tmp = -self.oobVotes*tmp
        self.trainEntropy = sum(tmp, axis = 1)/log2(len(self.uniqueClassCodes))
        self.trainHist = histogram(self.trainEntropy, array(range(0, 11, 1))/10.0)
        
    def getVotesOnTest(self, testData):
        return self.rf.predict_proba(testData)


class RfmWithFeatures:
    def __init__(self, rfm, usedFeatures):
        self.rfm = rfm
        self.usedFeatures = usedFeatures



def fillNan(dataFrame):
    import pandas as pd
    featuresWithNoNan = []
    featuresWithNan = []
    for feature in list(dataFrame.columns.values):
        if pd.notnull(dataFrame[feature]).all():
            featuresWithNoNan.append(feature)
        else:
            featuresWithNan.append(feature)
        
    allFeatures = featuresWithNan + featuresWithNoNan
    
    dfUpdatedTrain = dataFrame.copy();
    
    nanFeatureCountList = []
    for notAddedFeature in featuresWithNan:
        notAddedData = dataFrame[notAddedFeature].copy()
        nullData = notAddedData[notAddedData.isnull()]
        nanFeatureCountList.append([notAddedFeature, nullData.shape[0]])
        
    nanFeatureCountList.sort(key=lambda x: x[1])    
    
    from sklearn.ensemble import RandomForestRegressor
    
    featureSet = [x for x in allFeatures if x not in featuresWithNan]        
    
    for numOfFixedFeatures in range(len(nanFeatureCountList)):
        featureToFix = nanFeatureCountList[numOfFixedFeatures][0]
        if numOfFixedFeatures > 0:
            featureSet.append(nanFeatureCountList[numOfFixedFeatures-1][0])
        
        dfDataForFeatureSet = dfUpdatedTrain[featureSet].copy()
        dfDataForFeatureToFix = dfUpdatedTrain[featureToFix].copy()
    
        nullRows = dfDataForFeatureToFix.isnull()
        dfTestSet = dfDataForFeatureSet[nullRows].copy()
        dfTarget = dfDataForFeatureToFix.dropna(axis = 0, how = 'any').copy()
        dfTrainSet = dfDataForFeatureSet.loc[list(dfTarget.index)].copy()
        
        testSet = dfTestSet.values
        target = dfTarget.values
        trainSet = dfTrainSet.values
    
        regressor = RandomForestRegressor(n_estimators=150, min_samples_split=1)
        regressor.fit(trainSet, target)
        regressedData = regressor.predict(testSet)
        
        colIndex = dfUpdatedTrain.columns.get_loc(featureToFix)
        rowIndexes = dfTestSet.index
        dfUpdatedTrain.iloc[rowIndexes, colIndex] = regressedData
    
    return dfUpdatedTrain

