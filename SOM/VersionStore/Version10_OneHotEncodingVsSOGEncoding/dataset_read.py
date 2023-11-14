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
from math import e
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

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
           
            if  (X.at[i,name]== ''   or  X.at[i,name]== 'nan') and name !="Var_1":
                X.at[i,name]= -1   
            elif X.at[i,name]== 'Male' or  X.at[i,name]== 'Yes' or  X.at[i,name]== 'Artist' or  X.at[i,name]== 'Average'or  X.at[i,name]== 'A':
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
            elif (X.at[i,name]== ''   or  X.at[i,name]== 'nan') and name =="Var_1":
                X.at[i,name]= 1  
            else: 
                print("Unkonwn value {} at {}".format(X.at[i,name], name))


    def stringToIntAirlinePassengerSatisfactionDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
           
            if  X.at[i,name]== ''   or  X.at[i,name]== 'nan':
                X.at[i,name]= -1   
            elif X.at[i,name]== 'Male' or  X.at[i,name]== 'Loyal Customer' or  X.at[i,name]== 'Business travel' or  X.at[i,name]== 'neutral or dissatisfied'or  X.at[i,name]== 'Eco Plus':
                X.at[i,name]= 0            
            elif  X.at[i,name]== 'Female'or  X.at[i,name]== 'disloyal Customer' or  X.at[i,name]== 'Personal Travel'or  X.at[i,name]== 'satisfied'or  X.at[i,name]== 'Business':
                X.at[i,name]=1
            elif  X.at[i,name]== 'Eco':
                X.at[i,name]=2
            else: 
                print("Unkonwn value {} at {}".format(X.at[i,name], name))

    def stringToIntToddlerAutismDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
           
            if  X.at[i,name]== ''   or  X.at[i,name]== 'nan':
                X.at[i,name]= -1   
            elif X.at[i,name]== 'f' or  X.at[i,name]== 'asian' or  X.at[i,name]== 'yes' or  X.at[i,name]== 'family member'or  X.at[i,name]== 'Yes':
                X.at[i,name]= 0            
            elif  X.at[i,name]== 'm'or  X.at[i,name]== 'black' or  X.at[i,name]== 'no'or  (X.at[i,name]== 'Health care professional'or  X.at[i,name]== 'Health Care Professional')or  X.at[i,name]== 'No':
                X.at[i,name]=1
            elif  X.at[i,name]== 'Others'or  X.at[i,name]== 'Self':
                X.at[i,name]=2
            elif  X.at[i,name]== 'Latino':
                X.at[i,name]=3
            elif  X.at[i,name]== 'middle eastern':
                X.at[i,name]=4
            elif  X.at[i,name]== 'mixed':
                X.at[i,name]=5
            elif  X.at[i,name]== 'Native Indian':
                X.at[i,name]=6
            elif  X.at[i,name]== 'Hispanic':
                X.at[i,name]=7
            elif  X.at[i,name]== 'Pacifica':
                X.at[i,name]=8
            elif  X.at[i,name]== 'south asian':
                X.at[i,name]=9
            elif  X.at[i,name]== 'White European':
                X.at[i,name]=10
            else: 
                print("Unkonwn value {} at {}".format(X.at[i,name], name))

    def stringToHotelReservationDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
           
            if  (X.at[i,name]== ''   or  X.at[i,name]== 'nan'):
                X.at[i,name]= -1   
            elif ( X.at[i,name]== 'Not Selected' and name == 'type_of_meal_plan')  or  X.at[i,name]== 'Online' or  X.at[i,name]== 'Not_Canceled' :
                X.at[i,name]= 0            
            elif  X.at[i,name]== 'Meal Plan 1'or  X.at[i,name]== 'Room_Type 1' or  X.at[i,name]== '2017' or  X.at[i,name]== 'Offline'or  X.at[i,name]== 'Canceled':
                X.at[i,name]=1
            elif  X.at[i,name]== 'Meal Plan 2' or  X.at[i,name]== 'Room_Type 2'or  X.at[i,name]== '2018' or  X.at[i,name]== 'Aviation' :
                X.at[i,name]=2 
            elif  X.at[i,name]== 'Meal Plan 3'or  X.at[i,name]=='Room_Type 3'  or  X.at[i,name]== 'Complementary' :
                X.at[i,name]=3 
            elif  X.at[i,name]== 'Executive'or  X.at[i,name]== 'Room_Type 4'or  X.at[i,name]== 'Corporate' :
                X.at[i,name]=4
            elif  X.at[i,name]== 'Healthcare'or  X.at[i,name]== 'Room_Type 5':
                X.at[i,name]=5  
            elif  X.at[i,name]== 'Homemaker'or  X.at[i,name]== 'Room_Type 6':
                X.at[i,name]=6  
            elif  X.at[i,name]== 'Lawyer'or  X.at[i,name]== 'Room_Type 7':
                X.at[i,name]=7  
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
        #print(X[name])
        for i in range(0, X.shape[0]):
            if isinstance(X.at[i,name], str):
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

    def stringToIntNetflixDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
     
        for i in range(0, X.shape[0]):
            if  X.at[i,name]== '' :
                X.at[i,name]= -1      
            elif  X.at[i,name]== 'Basic'or  X.at[i,name]== 'United States' or  X.at[i,name]== 'Male' or  X.at[i,name]== 'Smartphone' :
                X.at[i,name]=0            
            elif  X.at[i,name]== 'Premium'or  X.at[i,name]== 'Canada' or  X.at[i,name]== 'Female' or  X.at[i,name]== 'Tablet' :
                X.at[i,name]=1
            elif  X.at[i,name]== 'Standard'or  X.at[i,name]== 'United Kingdom' or  X.at[i,name]== 'Smart TV' :
                X.at[i,name]=2 
            elif  X.at[i,name]== 'Germany' or  X.at[i,name]== 'Laptop' :
                X.at[i,name]=3 
            elif  X.at[i,name]== 'France' :
                X.at[i,name]=4 
            elif  X.at[i,name]== 'Brazil' :
                X.at[i,name]=5             
            elif  X.at[i,name]== 'Mexico' :
                X.at[i,name]=6             
            elif  X.at[i,name]== 'Spain' :
                X.at[i,name]=7             
            elif  X.at[i,name]== 'Italy' :
                X.at[i,name]=8
            elif  X.at[i,name]== 'Australia' :
                X.at[i,name]=9
            else: 
                print(f"Unkonwn value {X.at[i,name]} type {type(X.at[i,name])}")



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

    def stringToIntCarPriceDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
            if  X.at[i,name]== '' :
                X.at[i,name]= -1            
            elif  X.at[i,name]== '-1':
                X.at[i,name]=4
            elif  X.at[i,name]== '-2':
                X.at[i,name]=5 
            else: 
                print("Unkonwn value {}".format(X.at[i,name]))

    def stringToIntSalarydataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
            if  X.at[i,name]== '' :
                X.at[i,name]= -1            
            elif  X.at[i,name]== 'Male' or  X.at[i,name]== 'Bachelor\'s'or  X.at[i,name]== 'Bachelor\'s Degree' or  X.at[i,name]== 'Australia'or  X.at[i,name]== 'White':
                X.at[i,name]=0 
            elif  X.at[i,name]== 'Female'or  X.at[i,name]== 'UK'or  X.at[i,name]== 'Black':
                X.at[i,name]=1 
            elif  X.at[i,name]== ' 'or  X.at[i,name]== 'High School'or  X.at[i,name]== 'USA'or  X.at[i,name]== 'Hispanic':
                X.at[i,name]=2
            elif  X.at[i,name]== 'Other'or  X.at[i,name]== 'Canada'or  X.at[i,name]== 'Korean':
                X.at[i,name]=3
            elif  X.at[i,name]== 'Master\'s Degree'or  X.at[i,name]== 'Master\'s'or  X.at[i,name]== 'China'or  X.at[i,name]== 'Welsh':
                X.at[i,name]=4  
            elif  X.at[i,name]== 'PhD'or  X.at[i,name]== 'phD' or  X.at[i,name]== 'Asian':
                X.at[i,name]=5  
            elif   X.at[i,name]== 'African American':
                X.at[i,name]=6  
            elif   X.at[i,name]== 'Chinese':
                X.at[i,name]=7  
            elif   X.at[i,name]== 'Mixed':
                X.at[i,name]=8  
            elif   X.at[i,name]== 'Australian':
                X.at[i,name]=8  
            elif   X.at[i,name]== 'nan':
                X.at[i,name]=9 
            else: 
                print("Unkonwn value {}".format(X.at[i,name]))



    def stringToIntCardiovascularDataSet(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        #print(X[name])
        for i in range(0, X.shape[0]):
            if  X.at[i,name]== '' :
                X.at[i,name]= -1            
            elif  X.at[i,name]== 'Poor' or  X.at[i,name]== 'Within the past 2 years'or  X.at[i,name]== 'Yes'or  X.at[i,name]== 'Female'or  X.at[i,name]== '18-24':
                X.at[i,name]=0 
            elif  X.at[i,name]== 'Very Good'or  X.at[i,name]== 'Within the past year'or  X.at[i,name]== 'No'or  X.at[i,name]== 'Male'or  X.at[i,name]== '25-29':
                X.at[i,name]=1 
            elif  X.at[i,name]== 'Good'  or  X.at[i,name]== '5 or more years ago'or  X.at[i,name]== 'No, pre-diabetes or borderline diabetes' or  X.at[i,name]== '30-34':
                X.at[i,name]=2
            elif  X.at[i,name]== 'Fair'or  X.at[i,name]== 'Within the past 5 years'or  X.at[i,name]== 'Yes, but female told only during pregnancy'or  X.at[i,name]== '35-39':
                X.at[i,name]=3
            elif  X.at[i,name]== 'Excellent'or  X.at[i,name]== 'Never'or  X.at[i,name]== '40-44':
                X.at[i,name]=4  
            elif  X.at[i,name]== '45-49':
                X.at[i,name]=5
            elif  X.at[i,name]== '50-54':
                X.at[i,name]=6
            elif  X.at[i,name]== '55-59':
                X.at[i,name]=7
            elif  X.at[i,name]== '60-64':
                X.at[i,name]=8
            elif  X.at[i,name]== '65-69':
                X.at[i,name]=9
            elif  X.at[i,name]== '70-74':
                X.at[i,name]=10
            elif  X.at[i,name]== '75-79':
                X.at[i,name]=11
            elif  X.at[i,name]== '80+':
                X.at[i,name]=11
            else: 
                print("Unkonwn value {}".format(X.at[i,name]))       

    def changeCategoryTextToInt(self,X,name) :
        X[name] = X[name].astype(str).str.strip()
        names =[]
        num = 0
        for i in range(0, X.shape[0]):
            if  X.at[i,name] in names:
                X.at[i,name] = names.index( X.at[i,name] )
            else: 
                names.append( X.at[i,name] )
                X.at[i,name] = num
                num +=1

    #remove data and label which in a class, but the total number is too small in another way to reduce noise data
    def removeminorpartdata(self,Label,Data):
        allindexes = {}
        # i is the label index such as 329 1009
        for i in Label.index:    
            if Label[i] in allindexes:
                allindexes[Label[i]].append(i)
            else: 
                allindexes[Label[i]] = []
        #print("allindexes {}".format(allindexes))
        # allindexes {0: [308, 459, 268, 141],1:[2,5,77]}
        for item in allindexes.values():   
           # print("item {}".format(item))      
            if len(item) < 10:
               Label = Label.drop(item)
               Data = Data.drop(item)
        self.cleanedLabel = Label
        self.cleanedData = Data
        


    def initializedataset(self,Z,X,Y,X_original_encoded,Y_original_encoded,attributute,unique_num=20):
         #X is traiing data  Y is test data all_data is X+Y
         self.X = X.sample(n =X.shape[0])
         self.data_test =  Y
         self.all_data = Z #when there is no test.csv X+Y = Z
         data_train = self.X 
         data_test =  self.data_test
         data_train_original_encoded = X_original_encoded
         data_test_original_encoded  = Y_original_encoded
         #attributute is the feature name of the label, which feature will be used as the class label

         label_train = data_train[attributute]
         label_test = data_test[attributute]
         data_train = data_train.drop(attributute,axis = 1)
         data_test = data_test.drop(attributute,axis = 1)
         data_train_original_encoded = data_train_original_encoded.drop(attributute,axis = 1)
         data_test_original_encoded = data_test_original_encoded.drop(attributute,axis = 1)
         #remove class that has few data
        # self.removeminorpartdata(label_train,data_train)
       
         self.data_continuous_indexes = []
         self.data_discrete_indexes = []
    
         for (column_name, column) in data_train.transpose().iterrows():
            #print(len(X[column_name].unique()))
            if len(X[column_name].unique())>unique_num: 
                self.data_continuous_indexes.append(column_name)
            else: 
                self.data_discrete_indexes.append(column_name)
         print(f"self.data_continuous_indexes {self.data_continuous_indexes}"  )   
         if self.data_continuous_indexes == []:
            self.continuous_feature_num =0
         else:
            self.continuous_feature_num = len(self.data_continuous_indexes)
         
         #print(self.continuous_feature_num )
         if self.data_discrete_indexes == []:
            self.discrete_feature_num = 0
         else:
            self.discrete_feature_num = len(self.data_discrete_indexes)

        
      
         data_train_continuous = data_train[self.data_continuous_indexes]
         data_train_discrete = data_train[self.data_discrete_indexes]  
         data_test_continuous = data_test[self.data_continuous_indexes]
         data_test_discrete = data_test[self.data_discrete_indexes]  

         #print(f"self.data_discrete_indexes {self.data_discrete_indexes} " )
        # transfer to numpy array
         self.data_train = data_train.to_numpy(dtype=np.float64)
         self.all_data= self.all_data.to_numpy(dtype=np.float64)
         self.data_test = data_test.to_numpy(dtype=np.float64)

         self.data_train_original_encoded = data_train_original_encoded.to_numpy(dtype=np.float64)
         self.data_test_original_encoded = data_test_original_encoded.to_numpy(dtype=np.float64)


         #data that remove class data which has fewer data
         #self.cleanedData = self.cleanedData.to_numpy(dtype=np.float64)
         #self.cleanedLabel =  self.cleanedLabel.to_numpy(dtype=np.float64)

         self.label_train = label_train.to_numpy(dtype=np.float64)
        # print(len( self.label_train))
         self.label_test = label_test.to_numpy(dtype=np.float64)

         self.data_train_continuous = data_train_continuous.to_numpy(dtype=np.float64)
         self.data_train_discrete = data_train_discrete.to_numpy(dtype=np.float64)

         print(f" column number self.data_train_discrete { self.data_train_discrete.shape[1]}")

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
        

        #*** data_test_discrete is normailzed after one-hot encoding
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
       #  if self.cleanedData != []:
        #    scaler7 = StandardScaler().fit(self.cleanedData)
       #     self.cleanedData = scaler7.transform(self.cleanedData)

         #self.transfertoonehotcode(self.data_train_discrete,self.data_test_discrete)

    #transfer discrete data to onehot code
    def transfertoonehotcode(self, train_data, test_data):
        self.minValues =[]
        self.maxValues = []
        for i in range(0, self.data_train_discrete.shape[1]):
            #print("self.data_train_discrete[:,i] {} i {} ".format(self.data_train_discrete[:,i] , i))
            #item  = to_categorical(self.data_train_discrete[:,i])
            uniquelist = np.unique(self.data_train_discrete[:,i])
           # print("uniquelist 1  {}, i {}".format(uniquelist, i))
            uniqueNumber = len(uniquelist)
            self.uniqueNumbers.append(uniqueNumber)
            if uniquelist[0] == -1:
                uniquelist = np.delete(uniquelist,0)      
            maxValue = np.amax(uniquelist)
            minValue = np.amin(uniquelist)
            self.minValues.append(minValue)
            self.maxValues.append(maxValue)
            #print("maxValue {} minValue {}  i {}".format(maxValue , minValue,i))

        self.data_train_discrete_after_transfer = self.add_zero_columns_to_data_and_transfer_data(train_data)
        self.data_test_discrete_after_transfer = self.add_zero_columns_to_data_and_transfer_data(test_data)


    def add_zero_columns_to_data_and_transfer_data(self, data):
        empty_list = []
        for i in range(0, data.shape[1]):
            hasmissingvalue = False
            uniquelist1 = np.unique(data[:,i])
            if self.uniqueNumbers[i] > len(uniquelist1):
                    hasmissingvalue = True
                    difference = self.uniqueNumbers[i] - len(uniquelist1)
                    #print("i {} difference {}".format(i,difference))
            for idx, j in np.ndenumerate(data[:,i]):
                if data[:,i][idx] == -1:        
                    if self.minValues[i] == 0:             
                        data[:,i][idx]  = self.maxValues[i] +1
                    else : 
                        data[:,i][idx]  = 0
                # one hot encode
            if hasmissingvalue:
                # print("self.data_test_discrete[:,i] 1 {}".format(self.data_test_discrete[:,i]))
                zero_column = np.zeros((data[:,i].shape[0],difference))
                if self.maxValues[i] in np.unique(data[:,i]):
                     encoded = to_categorical(data[:,i])
                  #   print("maxValue {} np.unique(data[:,i]) {}, i {}".format(self.maxValues[i], np.unique(data[:,i]), i))
                else:
                    encoded =np.append ( to_categorical(data[:,i]),zero_column, axis=1)
                  #  print("difference{}, zero_column {}".format(difference, zero_column))
                    #print("to_categorical(data[:,i]) {}, encoded 2{}i {}".format(to_categorical(data[:,i]), encoded, i))
            else: 
                 #print("to_categorical(data[:,i]) {},i {}".format(to_categorical(data[:,i]), i)) 
                 encoded = to_categorical(data[:,i])
                
                
                
            empty_list.append(encoded)
        return empty_list
    

    def onehot_encoding(self,df,names):
        for name in names:
            df = pd.get_dummies(df, columns=[name])       

        self.original_encoding_data = df
        #*** -1 the class labe is not included 
        self.original_encoded_feature_num = df.shape[1]-1
       # print(  self.original_encoded_feature_num )

    def label_encoding(self,X,name):
        X[name] = X[name].astype(str).str.strip()
        le = LabelEncoder()
        X[name] = le.fit_transform(X[name])  