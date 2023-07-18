import random
from enum import Enum
import pandas as pd
from joblib import dump, load
from datetime import datetime
from DataLoader.SupportedScalers import SupportedScalers
from Core.PreprocessInterface import Preprocess

from Utils.Settings import ProjectSettings

class TitanicData(Preprocess):

    # survival 	Survival 	            0 = No, 1 = Yes
    # pclass 	Ticket class 	        1 = 1st, 2 = 2nd, 3 = 3rd
    # sex 	    Sex 	
    # Age 	    Age in years 	
    # sibsp 	# of siblings / spouses aboard the Titanic 	
    # parch 	# of parents / children aboard the Titanic 	
    # ticket 	Ticket number 	
    # fare 	    Passenger fare 	
    # cabin 	Cabin number 	
    # embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton

    def __init__(self, scaler: SupportedScalers=SupportedScalers.Standard):
        self.scaler=scaler
        
        self.trainData = pd.read_csv(ProjectSettings.Settings()['DatasetLocation']['Titanic']['Train'])
        self.testData = pd.read_csv(ProjectSettings.Settings()['DatasetLocation']['Titanic']['Test'])

        self.train_x = self.trainData.drop(["Survived"], axis=1, inplace=False)
        self.train_Y = self.trainData["Survived"]

        self.test_x = self.testData
        self.test_Y = pd.read_csv(ProjectSettings.Settings()['DatasetLocation']['Titanic']['TestTarget'])['Survived']

    def Get_TrainData(self):
        return self.trainData
    
    def Get_TestData(self):
        return self.testData
    
    def RunPreprocessPipeline(self):
        self.train_x = self.PreprocessTrainData()
        self.test_x = self.PreprocessTestData()

        featuresToScale = ['Age', 'Fare']
        self.FitScaler(self.train_x, featuresToScale)
        self.train_x = self.ScaleData(self.train_x, featuresToScale)
        self.test_x = self.ScaleData(self.test_x, featuresToScale)

        return self.train_x, self.train_Y, self.test_x, self.test_Y

    
    def PreprocessTrainData(self):
        self.train_x['Age'] = self.train_x['Age'].fillna(self.train_x['Age'].median())
        self.train_x['Fare'] = self.train_x['Fare'].fillna(self.train_x['Fare'].median())
        # Cabin has several empty values. Do we omit or add it?
        # NOTE: I do not recommend assign a random Cabin to all the passengers, maybe we can find a feature
            # having a high correlation with Cabin, and assign it to each passenger according to the correlation value.
        self.train_x['Cabin'] = self.train_x['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'Z')
        self.train_x['Embarked'] = self.train_x['Embarked'].fillna(random.choice(self.train_x["Embarked"].dropna().unique()))

        class_dummies = pd.get_dummies(self.train_x['Pclass'], drop_first=False, prefix="TicketClass")
        
        self.train_x["Sex"].replace({'male' : 0,'female' : 1},inplace=True)

        embarked_dummies = pd.get_dummies(self.train_x['Embarked'], drop_first=True, prefix="Embarked")

        self.train_x['FamilySize'] = self.train_x['Parch'] + self.train_x['SibSp']
        self.train_x['FamilySizeCategory'] = self.train_x['FamilySize'].apply(lambda x: 0 if x==0 else 1 if (x>=1 and x<=4) else 2 if (x>4 and x<=7) else 3)
        # 0 for alone
        # 1 for small family
        # 2 for medium family
        # 3 for large family

        # Divide the Cabin into: First, Second and Third part.
        self.train_x['Cabin'] = self.train_x['Cabin'].apply(lambda x: 0 if x in {'A','B','C','T'} else 1 if x in {'D','E'} else 2 if x in {'F','G'} else 3)
        # 0 for cabin A, B, C or T (there is only a passenger in Cabin T)
        # 1 for cabin D, E
        # 2 for cabin F, G
        # 3 for unknown Cabin (Z)

        self.train_x = self.train_x.drop(['PassengerId', 'Pclass', 'Embarked', 'FamilySize', 'Parch', 'SibSp', 'Name', 'Ticket', 'Name', 'Cabin'], axis=1)
        self.train_x = self.train_x.join([class_dummies, embarked_dummies])


        return self.train_x

    def PreprocessTestData(self):

        self.test_x['Age'] = self.test_x['Age'].fillna(self.test_x['Age'].median())
        self.test_x['Fare'] = self.test_x['Fare'].fillna(self.test_x['Fare'].median())
        self.test_x['Embarked'] = self.test_x['Embarked'].fillna(random.choice(self.test_x["Embarked"].dropna().unique()))
        
        self.test_x["Sex"].replace({'male' : 0,'female' : 1},inplace=True)
        self.test_x['FamilySize'] = self.test_x['Parch'] + self.test_x['SibSp']
        self.test_x['FamilySizeCategory'] = self.test_x['FamilySize'].apply(lambda x: 0 if x==0 else 1 if (x>=1 and x<=4) else 2 if (x>4 and x<=7) else 3)
        self.test_x['Embarked'] = self.test_x['Embarked'].fillna(random.choice(self.test_x["Embarked"].dropna().unique()))
        
        class_dummies = pd.get_dummies(self.test_x['Pclass'], drop_first=False, prefix="TicketClass")
        embarked_dummies = pd.get_dummies(self.test_x['Embarked'], drop_first=True, prefix="Embarked")
        
        self.test_x = self.test_x.drop(['PassengerId','Pclass', 'Embarked', 'FamilySize', 'Parch', 'SibSp', 'Name', 'Ticket', 'Name', 'Cabin'], axis=1)
        self.test_x = self.test_x.join([class_dummies, embarked_dummies])

        return self.test_x