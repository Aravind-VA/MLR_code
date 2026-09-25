

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# here pip fereeze we use to know any libraries in venv or not
# here we are creating docstring
"""
in this file we are going to load data and create a model for multiple linear regression
"""



import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import pickle


class MLR:
    def __init__(self,path): # self is nothing but object refrence and this whole process is called constructor
        self.path = path
        self.data = pd.read_csv(self.path)
        self.data["State"] = self.data["State"].map({'New York':0,'California':1,'Florida':3}).astype(int)
        self.X = self.data.iloc[ : , : -1] # fpr independent column
        self.y = self.data.iloc[ : ,-1] # for dependent columns
        self.X_train,self.X_test,self.y_train,self.y_test  = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        print(f"traing data size: {len(self.X_train)} : {len(self.y_train)}")
        print(f"testing data size: {len(self.X_test)} : {len(self.y_test)}")
    def train(self):
        self.reg = LinearRegression()
        self.reg.fit(self.X_train, self.y_train)
        self.y_pred = self.reg.predict(self.X_train)
        print(f"training accuracy : {r2_score(self.y_train,self.y_pred)}")
        print(f"training loss: {root_mean_squared_error(self.y_train,self.y_pred)}")
    def test(self):
        self.y_test_pred = self.reg.predict(self.X_test)
        print(f"testing accuracy : {r2_score(self.y_test,self.y_test_pred)}")
        print(f"testing loss: {root_mean_squared_error(self.y_test,self.y_test_pred)}")
    def check_own_data(self):
        rd = 1200
        admin = 1800
        ms = 1900
        s =1
        self.reg.predict([[rd,admin,ms,s]])
        print(self.reg.predict([[rd,admin,ms,s]])[0]) # here we are index number as to get only value
    def saving_model(self):
        with open('model.pkl', 'wb') as file: # here wb means write binary
            pickle.dump(self.reg, file)
        print(f"______ load and check_____")
        with open('model.pkl', 'rb') as t:  # rb means read binary
            loaded_model = pickle.load(t)
            rd = 1200
            admin = 1800
            ms = 1900
            s = 1
            self.reg.predict([[rd, admin, ms, s]])
            print(f"load_model_prediction : {loaded_model.predict([[rd,admin,ms,s]])}")







if __name__ == "__main__":
     path = "50_Startups.csv"
     obj = MLR(path) # object
     obj.train()
     obj.test()
     obj.check_own_data()
     obj.saving_model()