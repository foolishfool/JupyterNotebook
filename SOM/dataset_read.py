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
                X.at[i,name] =1            
            elif  X.at[i,name]== 'forest':
                    X.at[i,name]=2 
            elif  X.at[i,name]== 'grass': 
                X.at[i,name]=3 
            elif  X.at[i,name]== 'orchard': 
                X.at[i,name]=4 
            elif  X.at[i,name]== 'impervious': 
                X.at[i,name]=5 
            elif  X.at[i,name]== 'farm': 
                X.at[i,name]=6  
            else: 
                print("Unkonwn value {}".format(X.at[i,name]))