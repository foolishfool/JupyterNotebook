#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
@author: Riley Smith
Created: 1-27-21
"""
import pandas as pd
import category_encoders as ce
import matplotlib.pyplot as plt
from fcmeans import FCM
from math import e
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders.binary import BinaryEncoder
from sklearn.preprocessing import MultiLabelBinarizer
class DATAREAD():
    def __init__(self):
         return

    
    def initializedataset(self,Z,X,Y,attributute,unique_num=20 ):
         #X is training data  Y is test data all_data is X+Y
         self.X = X
         self.data_test =  Y
         self.all_data = Z #when there is no test.csv X+Y = Z
         data_train = self.X 
         data_test =  self.data_test
         label_train = data_train[attributute]
         label_test = data_test[attributute]
         data_train = data_train.drop(attributute,axis = 1)
         data_test = data_test.drop(attributute,axis = 1)
       
         self.data_continuous_indexes = []
         self.data_discrete_indexes = []
    
         for (column_name, column) in data_train.transpose().iterrows():
            #print(len(X[column_name].unique()))
            if len(X[column_name].unique())>unique_num: 
                self.data_continuous_indexes.append(column_name)
            else: 
                self.data_discrete_indexes.append(column_name)
  
         if self.data_continuous_indexes == []:
            self.continuous_feature_num =0
         else:
            self.continuous_feature_num = len(self.data_continuous_indexes)
         print(f"continuous_feature_num {self.continuous_feature_num}"  )   
         #print(self.continuous_feature_num )
         if self.data_discrete_indexes == []:
            self.discrete_feature_num = 0
         else:
            self.discrete_feature_num = len(self.data_discrete_indexes)

        
      
         data_train_continuous = data_train[self.data_continuous_indexes]
         data_train_discrete = data_train[self.data_discrete_indexes]  
         data_test_continuous = data_test[self.data_continuous_indexes]
         data_test_discrete = data_test[self.data_discrete_indexes]  
         print(f"discrete_feature_num {len(self.data_discrete_indexes)}"  )       
         print(f"self.data_discrete_indexes {self.data_discrete_indexes} " )
         
        # transfer to numpy array
         self.data_train = data_train.to_numpy(dtype=np.float64)
         self.all_data= self.all_data.to_numpy(dtype=np.float64)
         self.data_test = data_test.to_numpy(dtype=np.float64)

         self.label_train = label_train.to_numpy(dtype=np.float64)
         self.label_test = label_test.to_numpy(dtype=np.float64)
       #  print("___________________________________________")
         self.data_train_continuous = data_train_continuous.to_numpy(dtype=np.float64)  
         self.data_train_discrete = data_train_discrete.to_numpy(dtype=np.float64)


         self.data_train_discrete_before_transfer = data_train_discrete.to_numpy(dtype=np.float64)
         self.data_test_continuous = data_test_continuous.to_numpy(dtype=np.float64)
         self.data_test_discrete = data_test_discrete.to_numpy(dtype=np.float64)
         self.data_test_discrete_before_transfer = data_test_discrete.to_numpy(dtype=np.float64)

     
         self.uniqueNumbers =[]
   
         scaler = StandardScaler().fit(self.data_train)
         #print(f"self.data_train {self.data_train}")
         self.data_train_scaled = scaler.transform(self.data_train)
         #print(f"data_train_scaled {self.data_train_scaled}")
         scaler2 = StandardScaler().fit(self.data_test)
         self.data_test_scaled = scaler2.transform(self.data_test)
        
         if self.data_train_continuous != [] :
            scaler3 = StandardScaler().fit(self.data_train_continuous)
            self.data_train_continuous_normalized = scaler3.transform(self.data_train_continuous)
         if self.data_test_continuous != [] :
            scaler4 = StandardScaler().fit(self.data_test_continuous)
            self.data_test_continuous_normalized = scaler4.transform(self.data_test_continuous)
        
         if self.data_test_discrete != []:
             scaler5 = StandardScaler().fit(self.data_test_discrete)
             self.data_test_discrete_normalized = scaler5.transform(self.data_test_discrete)
         else:
            self.data_test_discrete_normalized =[]
         if self.data_train_discrete != []:
            scaler6 = StandardScaler().fit(self.data_train_discrete)
            self.data_train_discrete_normalized = scaler6.transform(self.data_train_discrete)
         else:
            self.data_train_discrete_normalized =[]

    def PCA_Comparision(self):
        pca1 = PCA()
        pca1.fit_transform(self.data_train_discrete)
       # print(self.data_train_discrete.shape)
        #print("Base Line PCA feature importance")
        print(pca1.components_)
        self.RankingFeatureImportance(abs(pca1.components_),self.data_train_discrete.shape[1])
        #print("Base Line PCA explained_variance_ratio_")
        print(pca1.explained_variance_ratio_)
        plt.bar(
            range(1,len(pca1.explained_variance_)+1),
            pca1.explained_variance_
            )
        
        
        plt.xlabel('PCA Feature')
        plt.ylabel('Explained variance')
        plt.title('Feature Explained Variance')
        plt.show()

    def label_encoding(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        
        le = LabelEncoder()
        X[name] = le.fit_transform(X[name])  
        #print(X)
    def RankingFeatureImportance(self,X,pc_num):
        d ={}
        for i in range(0,pc_num):
           if i< pc_num:
               d[i] = []
        for x in X:
           for j in range(0,len(x)):
               d[j].append(x[j])
        
        
        for l in range(0,pc_num):
            sorted_index = [sorted(d[l]).index(x) for x in d[l]]
            sorted_list = sorted(d[l], reverse=True)
            transfered_list = self.transferValueToproporation(sorted_list)
            print(f"Sorted feature index for component {l} {sorted_index}" )
            #print(f"Sorted imporance porporation for component {l}  {sorted_list}" )
            print(f"Sorted imporance porporation for component {l}  {transfered_list}" )
            #print(f"sum  {sum(sorted_list)}")
    
    def transferValueToproporation(self,list):
        sum_num = sum(list)
        transfered_list =[x/sum_num for x in list]
        return transfered_list
    
    def test(self):
        my_model = FCM(n_clusters=2) # we use two cluster as an example
        my_model.fit(self.data_train) ## X, numpy array. rows:samples columns:features