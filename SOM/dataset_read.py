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