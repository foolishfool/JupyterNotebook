#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
@author: Riley Smith
Created: 1-27-21
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from numpy import array
from numpy import argmax
from keras.utils import to_categorical

class DATAREAD():
    def __init__(self):
         return
    def stringToIntOnlineShoppersPurchasingIntentionDataset(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
            if  X.at[i,name]== 'Jan':
                X.at[i,name] =1            
            elif  X.at[i,name]== 'Feb':
                    X.at[i,name]=2 
            elif  X.at[i,name]== 'Mar': 
                X.at[i,name]=3 
            elif  X.at[i,name]== 'Apr': 
                X.at[i,name]=4 
            elif  X.at[i,name]== 'May': 
                X.at[i,name]=5 
            elif  X.at[i,name]== 'June': 
                X.at[i,name]=6  
            elif  X.at[i,name]== 'Jul': 
                X.at[i,name]=7 
            elif  X.at[i,name]== 'Aug':
                X.at[i,name]=8 
            elif  X.at[i,name]== 'Sep':
                X.at[i,name]=9 
            elif  X.at[i,name]== 'Oct':
                X.at[i,name]=10
            elif  X.at[i,name]== 'Nov':
                X.at[i,name]=11 
            elif  X.at[i,name]== 'Dec': 
                X.at[i,name]=12
            elif  X.at[i,name]== 'Returning_Visitor':  
                X.at[i,name]=0
            elif  X.at[i,name]== 'New_Visitor': 
                X.at[i,name]=1
            elif  X.at[i,name]== 'Other': 
                X.at[i,name]=3
            elif  X.at[i,name]== 'False': 
                X.at[i,name]=0        
            elif  X.at[i,name]== 'True': 
                X.at[i,name]=1
            else: 
                print("Unkonwn value {}".format(X.at[i,name]))

    def stringToIntCrowdsourcedMappingDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
            if  X.at[i,name]== 'water':
                X.at[i,name]=0            
            elif  X.at[i,name]== 'forest':
                X.at[i,name]=1
            elif  X.at[i,name]== 'grass': 
                X.at[i,name]=2 
            elif  X.at[i,name]== 'orchard': 
                X.at[i,name]=3 
            elif  X.at[i,name]== 'impervious': 
                X.at[i,name]=4
            elif  X.at[i,name]== 'farm': 
                X.at[i,name]=5  
            else: 
                print("Unkonwn value {}".format(X.at[i,name]))

    def stringToIntIrisDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
            if  X.at[i,name]== 'Iris-setosa':
                X.at[i,name]=0            
            elif  X.at[i,name]== 'Iris-versicolor':
                X.at[i,name]=1
            elif  X.at[i,name]== 'Iris-virginica': 
                X.at[i,name]=2 
            else: 
                print("Unkonwn value {}".format(X.at[i,name]))

    def stringToIntFrogCallDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
            if  X.at[i,name]== 'AdenomeraAndre' or  X.at[i,name]== 'Bufonidae' or  X.at[i,name]== 'Adenomera':
                X.at[i,name]=0            
            elif  X.at[i,name]== 'AdenomeraHylaedactylus'or  X.at[i,name]== 'Dendrobatidae' or  X.at[i,name]== 'Ameerega':
                X.at[i,name]=1
            elif  X.at[i,name]== 'Ameeregatrivittata'or  X.at[i,name]== 'Hylidae' or  X.at[i,name]== 'Dendropsophus':
                X.at[i,name]=2 
            elif  X.at[i,name]== 'HylaMinuta'or  X.at[i,name]== 'Leptodactylidae' or  X.at[i,name]== 'Hypsiboas':
                X.at[i,name]=3 
            elif  X.at[i,name]== 'HypsiboasCinerascens'or  X.at[i,name]== 'Leptodactylus':
                X.at[i,name]=4
            elif  X.at[i,name]== 'HypsiboasCordobae'or  X.at[i,name]== 'Osteocephalus':
                X.at[i,name]=5  
            elif  X.at[i,name]== 'LeptodactylusFuscus'or  X.at[i,name]== 'Rhinella':
                X.at[i,name]=6  
            elif  X.at[i,name]== 'OsteocephalusOophagus'or  X.at[i,name]== 'Scinax':
                X.at[i,name]=7  
            elif  X.at[i,name]== 'Rhinellagranulosa': 
                X.at[i,name]=8  
            elif  X.at[i,name]== 'ScinaxRuber': 
                X.at[i,name]=9  
            else: 
                print("Unkonwn value {}".format(X.at[i,name]))
    
    def stringToIntCustomerSegmentationDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
            if  X.at[i,name]== ''  or  X.at[i,name]== 'nan'or  (X.at[i,name]== -1 and name != 'Var_1') :
                X.at[i,name]= -1   
            if  X.at[i,name]== 'Male' or  X.at[i,name]== 'Yes' or  X.at[i,name]== 'Artist' or  X.at[i,name]== 'Average'or  X.at[i,name]== 'A'or  (X.at[i,name]== -1 and name == 'Var_1'):
                X.at[i,name]= 0            
            elif  X.at[i,name]== 'Female'or  X.at[i,name]== 'No' or  X.at[i,name]== 'Doctor'or  X.at[i,name]== 'Low'or  X.at[i,name]== 'B':
                X.at[i,name]=1
            elif  X.at[i,name]== 'Engineer'or  X.at[i,name]== 'High' or  X.at[i,name]== 'Cat_1'or  X.at[i,name]== 'C':
                X.at[i,name]=2 
            elif  X.at[i,name]== 'Entertainment'or  X.at[i,name]== 'Cat_2' or  X.at[i,name]== 'D':
                X.at[i,name]=3 
            elif  X.at[i,name]== 'Executive'or  X.at[i,name]== 'Cat_3':
                X.at[i,name]=4
            elif  X.at[i,name]== 'Healthcare'or  X.at[i,name]== 'Cat_4':
                X.at[i,name]=5  
            elif  X.at[i,name]== 'Homemaker'or  X.at[i,name]== 'Cat_5':
                X.at[i,name]=6  
            elif  X.at[i,name]== 'Lawyer'or  X.at[i,name]== 'Cat_6':
                X.at[i,name]=7  
            elif  X.at[i,name]== 'Marketing'or  X.at[i,name]== 'Cat_7':
                X.at[i,name]=8  
            else: 
                print("Unkonwn value {} at {}".format(X.at[i,name], name))

    
    def stringToIntObseityDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
            if  X.at[i,name]== 'Sometimes' or  X.at[i,name]== 'Public_Transportation' or  X.at[i,name]== 'Normal_Weight' or  X.at[i,name]== 'Female'or  X.at[i,name]== 'yes':
                X.at[i,name]=0            
            elif  X.at[i,name]== 'no' or  X.at[i,name]== 'Walking' or  X.at[i,name]== 'Overweight_Level_I'or  X.at[i,name]== 'Male':
                X.at[i,name]=1
            elif  X.at[i,name]== 'Frequently'or  X.at[i,name]== 'Automobile' or  X.at[i,name]== 'Overweight_Level_II':
                X.at[i,name]=2 
            elif  X.at[i,name]== 'Always' or  X.at[i,name]== 'Motorbike' or  X.at[i,name]== 'Obesity_Type_I':
                X.at[i,name]=3 
            elif  X.at[i,name]== 'Bike'or  X.at[i,name]== 'Obesity_Type_II':
                X.at[i,name]=4
            elif  X.at[i,name]== 'Obesity_Type_III':
                X.at[i,name]=5  
            elif  X.at[i,name]== 'Insufficient_Weight':
                X.at[i,name]=6  
            else: 
                print("Unkonwn value {}".format(X.at[i,name]))

    def stringToIntDataUserModelDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
            if  X.at[i,name]== 'very_low':
                X.at[i,name]=0            
            elif  X.at[i,name]== 'Low' :
                X.at[i,name]=1
            elif  X.at[i,name]== 'Middle':
                X.at[i,name]=2 
            elif  X.at[i,name]== 'High':
                X.at[i,name]=3 
            else: 
                print("Unkonwn value {}".format(X.at[i,name]))

    def replaceNANinDataset(self,X):
        for col_name in X.columns:
            X[col_name] = X[col_name].astype(str).str.strip()
            for i in range(0, X.shape[0]):
                if  X.at[i,col_name]== '' or  X.at[i,col_name]== 'NaN'  or  X.at[i,col_name]== 'nan' or  X.at[i,col_name]== '?'or  X.at[i,col_name]== 'NA' :
                    X.at[i,col_name]= -1    

    def stringToIntMiceProteinDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
            if  X.at[i,name]== '' :
                X.at[i,name]= -1            
            elif  X.at[i,name]== 'Control' or  X.at[i,name]== 'Memantine' or  X.at[i,name]== 'C/S'or  X.at[i,name]== 'c-CS-m':
                X.at[i,name]=0 
            elif  X.at[i,name]== 'Ts65Dn'or  X.at[i,name]== 'Saline' or  X.at[i,name]== 'S/C' or  X.at[i,name]== 'c-CS-s':
                X.at[i,name]=1 
            elif  X.at[i,name]== 'c-SC-m'  :
                X.at[i,name]=2
            elif  X.at[i,name]== 'c-SC-s':
                X.at[i,name]=3
            elif  X.at[i,name]== 't-CS-m':
                X.at[i,name]=4  
            elif  X.at[i,name]== 't-CS-s':
                X.at[i,name]=5  
            elif  X.at[i,name]== 't-SC-m':
                X.at[i,name]=6  
            elif  X.at[i,name]== 't-SC-s':
                X.at[i,name]=7 
            else: 
                print("Unkonwn value {}".format(X.at[i,name]))

    def stringToIntHCVDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
            if  X.at[i,name]== '' :
                X.at[i,name]= -1            
            elif  X.at[i,name]== 'm' or  X.at[i,name]== '0=Blood Donor':
                X.at[i,name]=0 
            elif  X.at[i,name]== 'f'or  X.at[i,name]== '1=Hepatitis':
                X.at[i,name]=1 
            elif  X.at[i,name]== '2=Fibrosis'  :
                X.at[i,name]=2
            elif  X.at[i,name]== '3=Cirrhosis':
                X.at[i,name]=3
            elif  X.at[i,name]== '0s=suspect Blood Donor':
                X.at[i,name]=4  
            else: 
                print("Unkonwn value {}".format(X.at[i,name]))
                

    def removeminorpartdata(self,Label,Data):
        allindexes = {}
       # print("Label {}".format(Label))   
        for i in Label.index:    
           # print("i {}".format(i))    
            if Label[i] in allindexes:
               # print("i {}".format(i))   
              #  print("allindexes {}".format(allindexes))   
               # print("Label[i] {}".format(Label[i]))   
                allindexes[Label[i]].append(i)
            else: 
                allindexes[Label[i]] = []
             
       # print(" Label1 {}".format(len(Label)))  
        for item in allindexes.values():         
            if len(item) < 10:
              # print("item {}".format(item))
               Label = Label.drop(item)
               Data = Data.drop(item)
        self.cleanedLabel = Label
        self.cleanedData = Data
        


    def initializedataset(self,X,Y,attributute):
         self.X = X.sample(n =X.shape[0])
         self.data_test =  Y
         
         data_train = self.X 
         data_test =  self.data_test

         label_train = data_train[attributute]
         label_test = data_test[attributute]
         data_train = data_train.drop(attributute,axis = 1)
         data_test = data_test.drop(attributute,axis = 1)

         #print("label_train len1 {}".format(len(label_train)))

         self.removeminorpartdata(label_train,data_train)
         #print(" self.refinedLabel  len {}".format(len( self.refinedLabel )))
        # print("self.refinedData len2 {}".format(len(self.refinedData)))
       
         self.data_continuous_indexes = []
         self.data_discrete_indexes = []
    
         for (column_name, column) in data_train.transpose().iterrows():
            if len(X[column_name].unique())>10: 
                self.data_continuous_indexes.append(column_name)
              #  print("continuous label : {}".format(column_name))
            else: 
                self.data_discrete_indexes.append(column_name)
              #  print("discrete label : {}".format(column_name))
    

         data_train_continuous = data_train[self.data_continuous_indexes]
         data_train_discrete = data_train[self.data_discrete_indexes]  
         data_test_continuous = data_test[self.data_continuous_indexes]
         data_test_discrete = data_test[self.data_discrete_indexes]  

        # transfer to numpy array
         self.data_train = data_train.to_numpy(dtype=np.float64)
         self.data_test = data_test.to_numpy(dtype=np.float64)
         self.cleanedData = self.cleanedData.to_numpy(dtype=np.float64)
         self.label_train = label_train.to_numpy(dtype=np.float64)
         self.cleanedLabel =  self.cleanedLabel.to_numpy(dtype=np.float64)
         self.label_test = label_test.to_numpy(dtype=np.float64)
         self.data_train_continuous = data_train_continuous.to_numpy(dtype=np.float64)
         self.data_train_discrete = data_train_discrete.to_numpy(dtype=np.float64)
         self.data_test_continuous = data_test_continuous.to_numpy(dtype=np.float64)
         self.data_test_discrete = data_test_discrete.to_numpy(dtype=np.float64)


        # data = array(data)
        # print(data)
        # one hot encode
         encoded = to_categorical(self.data_train_discrete)
         print(encoded)
         # invert encoding
         inverted = argmax(encoded[0])
         print(inverted)


         encoded2 = to_categorical(self.data_test_discrete)
         print(encoded2)
         # invert encoding
         inverted2 = argmax(encoded2[0])
         print(inverted2)

         scaler = StandardScaler().fit(self.data_train)
         self.data_train = scaler.transform(self.data_train)
         scaler2 = StandardScaler().fit(self.data_test)
         self.data_test = scaler2.transform(self.data_test)
         scaler3 = StandardScaler().fit(self.data_train_continuous)
         self.data_train_continuous = scaler3.transform(self.data_train_continuous)
         scaler4 = StandardScaler().fit(self.data_test_continuous)
         self.data_test_continuous = scaler4.transform(self.data_test_continuous)
        # scaler5 = StandardScaler().fit(self.data_test_discrete)
       #  self.data_test_discrete = scaler5.transform(self.data_test_discrete)
       #   scaler6 = StandardScaler().fit(self.data_train_discrete)
       #   self.data_train_discrete = scaler6.transform(self.data_train_discrete)
         scaler7 = StandardScaler().fit(self.cleanedData)
         self.cleanedData = scaler7.transform(self.cleanedData)

        # discrete data do not use scaler


    def initializedataset_frog(self,X,Y):
         self.X = X.sample(n =X.shape[0])
         self.data_test =  Y
         data_train = self.X 
         data_test =  self.data_test
         label_train = data_train["Species"]
         label_test = data_test["Species"]
         data_train = data_train.drop("Species",axis = 1)
         data_test = data_test.drop("Species",axis = 1)

        # transfer to numpy array
         self.data_train = data_train.to_numpy(dtype=np.float64)
         self.data_test = data_test.to_numpy(dtype=np.float64)
         self.label_train = label_train.to_numpy(dtype=np.float64)
         self.label_test = label_test.to_numpy(dtype=np.float64)
    
    def initializedataset_iris(self,X,Y):
         self.X = X.sample(n =X.shape[0])
         self.data_test =  Y
         data_train = self.X 
         data_test =  self.data_test
         label_train = data_train["Species"]
         label_test = data_test["Species"]
         data_train = data_train.drop("Species",axis = 1)
         data_test = data_test.drop("Species",axis = 1)

        # transfer to numpy array
         self.data_train = data_train.to_numpy(dtype=np.float64)
         self.data_test = data_test.to_numpy(dtype=np.float64)
         self.label_train = label_train.to_numpy(dtype=np.float64)
         self.label_test = label_test.to_numpy(dtype=np.float64)

    def initializedataset_CM(self,X,Y):
        self.X = X.sample(n =X.shape[0])
        self.data_test =  Y
        data_train = self.X 
        # transfer to numpy array
        data_train = data_train.to_numpy(dtype=np.float64)
        data_test = self.data_test.to_numpy(dtype=np.float64)
        self.label_train = data_train[:,0]
        self.label_test = data_test[:,0]

        # delete first column
        self.data_train= data_train[:,1:]
        self.data_test =data_test[:,1:]
