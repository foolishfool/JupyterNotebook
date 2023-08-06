#!/usr/bin/env python
# coding: utf-8
from copy import deepcopy
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
from typing import List
import newSom
import UTDSM
import numpy as np
from sklearn.metrics import silhouette_score
import scipy.stats as stats
import pandas as pd
import collections
import researchpy as rp
class Experiment():
        def __init__(self):
         return

        def smooth(self, scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
            last = scalars[0]  # First value in the plot (first timestep)
            smoothed = list()
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
                smoothed.append(smoothed_val)                        # Save it
                last = smoothed_val                                  # Anchor the last smoothed value

            return smoothed
        

        def findBestClusterNo (self,X,class_num, dim_num, scope_num,unstable_repeate_num,smoothWeight):
            inertias_repeat = []
            silhouette_score_repeat= []
            y=1
            while y<= unstable_repeate_num:
                unit_list = [] 
                inertias = []
                silhouette_scores = []
                for x in range(class_num, class_num*(scope_num)+1):
                    unit_list.append(x)
                    som = newSom.SOM(m=x, n= 1, dim=dim_num) 
                    som.fit(X)
                    inertias.append(som._inertia_)
                    silhouette_scores.append(silhouette_score(X,som.predict(X,som.weights0)))
                inertias_repeat.append(inertias)
                silhouette_score_repeat.append(silhouette_scores)
                
                y=y+1



            multiple_lists = inertias_repeat
            arrays = [np.array(x) for x in multiple_lists] 
            inertias_average = [np.mean(k) for k in zip(*arrays)]

            multiple_lists2 = silhouette_score_repeat
            arrays2 = [np.array(x) for x in multiple_lists2] 
            silhouette_score_average = [np.mean(k) for k in zip(*arrays2)]

            for i in range(0,len(inertias_average)):
                print("inertias_average {}  cluster_num {}".format(inertias_average[i], unit_list[i]))

            for j in range(0,len(silhouette_score_average)):
                print("silhouette_score_average {}  cluster_num {}".format(silhouette_score_average[j], unit_list[j]))

            plt.plot(unit_list, self.smooth(inertias_average, smoothWeight))
            plt.plot(unit_list, inertias_average)

       
       
        def topology_som(self, som_num):
            start = int(np.sqrt(som_num))
            factor = som_num / start
            while not self.is_integer(factor):
                start += 1
                factor = som_num / start
            return int(factor), start
        
        def is_integer(self,number):
            if int(number) == number:
                return True
            else:
                return False
            
        def UTtest(self,dataread,label_train,class_num,dim_num,best_num,scope_num,unstable_repeat_num,type,elbow_num , row,column):
            
            all_train_score_W0_n =[]
            all_train_score_W_combine_n =[]
            test_score_W0_n = []
            test_score_W_combine_n = []


            all_train_score_W0_a =[]
            all_train_score_W_combine_a =[]
            test_score_W0_a = []
            test_score_W_combine_a = []


          
            plot_unit = [1]
            
            if type == 0:
                y = class_num

                while y <= scope_num:
                    m, n = self.topology_som(y)
                    print(f" mxn 0 {m*n}") 
                    print("neuron unit number: {}".format(y))
                    if y % 2 == 0:
                        som = newSom.SOM(m , n, dim=dim_num) 
                        print(f" mxn {int(y/2)*int(y/2)}") 
                    else:
                     som = newSom.SOM(m , n, dim=dim_num)  
                     print(f" mxn {y}") 
                    optimize_W = UTDSM.UTDSM_SOM(som,dataread.data_train,dataread.data_test,dataread.label_train,dataread.label_test,elbow_num,row,column)
                                    
                    optimize_W.run()
                    all_train_score_W0_n.append(optimize_W.all_train_score_W0_n)
                    all_train_score_W_combine_n.append(optimize_W.all_train_score_W_Combined_n)
                    test_score_W0_n.append(optimize_W.test_score_W0_n)
                    test_score_W_combine_n.append(optimize_W.test_score_W_Combined_n)


                    all_train_score_W0_a.append(optimize_W.all_train_score_W0_a)
                    all_train_score_W_combine_a.append(optimize_W.all_train_score_W_Combined_a)
                    test_score_W0_a.append(optimize_W.test_score_W0_n)
                    test_score_W_combine_a.append(optimize_W.test_score_W_Combined_a)
        
                    y =y+1
                    if(y<= scope_num):
                        plot_unit.append(y)
            
            
            if type == 1:
                y = 1
                while y <= unstable_repeat_num:
                    if best_num % 2 == 0:
                        som = newSom.SOM(m= int(best_num/2), n= int(best_num/2) , dim=dim_num)  
                    else:
                        som = newSom.SOM(m= best_num , n= 1, dim=dim_num)  

                    optimize_W = UTDSM.UTDSM_SOM(som,dataread.data_train,dataread.data_test,dataread.label_train,dataread.label_test,elbow_num,row,column)
                    optimize_W.run()
                    all_train_score_W0_n.append(optimize_W.all_train_score_W0_n)
                    all_train_score_W_combine_n.append(optimize_W.all_train_score_W_Combined_n)
                    test_score_W0_n.append(optimize_W.test_score_W0_n)
                    test_score_W_combine_n.append(optimize_W.test_score_W_Combined_n)


                    all_train_score_W0_a.append(optimize_W.all_train_score_W0_a)
                    all_train_score_W_combine_a.append(optimize_W.all_train_score_W_Combined_a)
                    test_score_W0_a.append(optimize_W.test_score_W0_n)
                    test_score_W_combine_a.append(optimize_W.test_score_W_Combined_a)
        
                    y =y+1
                    if(y<= unstable_repeat_num):
                        plot_unit.append(y)
                
                        
            figure, axis = plt.subplots(1, 2,figsize =(12, 5))
            axis[0].set_title("NMI Score")               
            axis[1].set_title("ARI Score")

            #print("all_train_score_W0_n len {}".format(len(all_train_score_W0_n)))
           # print("plot_unit {}".format(plot_unit))
            if type == 0:
                axis[0].set_xlabel('Neuron number')
            if type == 1:
                axis[0].set_xlabel('Repeat number')
            axis[0].plot(plot_unit,all_train_score_W0_n,'r',label ='all_train_score_W0')
            axis[0].plot(plot_unit,all_train_score_W_combine_n,'c',label ='all_train_score_W\'')
            axis[0].plot(plot_unit,test_score_W0_n,'y',label ='test_score_W0')
            axis[0].plot(plot_unit,test_score_W_combine_n,'k',label ='test_score_W\'')
            axis[0].legend(loc='best')



            if type == 0:
                axis[1].set_xlabel('Neuron number')
            if type == 1:
                axis[1].set_xlabel('Repeat number')
            axis[1].plot(plot_unit,all_train_score_W0_a,'r',label ='all_train_score_W0')
            axis[1].plot(plot_unit,all_train_score_W_combine_a,'c',label ='all_train_score_W\'')
            axis[1].plot(plot_unit,test_score_W0_a,'y',label ='test_score_W0')
            axis[1].plot(plot_unit,test_score_W_combine_a,'k',label ='test_score_W\'')
         
            axis[1].legend(loc='best')
            plt.show()
            
           # print("New Neuron Number : {}".format (int(np.mean(new_neuron_num))))
                                                    
                        
            df1_n = pd.DataFrame(test_score_W0_n, columns = ['test_score_W0'])
            df2_n = pd.DataFrame(test_score_W_combine_n, columns = ['test_score_W\''])

            df3_n = pd.DataFrame(all_train_score_W0_n, columns = ['train_score_W0'])
            df4_n = pd.DataFrame(all_train_score_W_combine_n, columns = ['train_score_W\''])

            summary, results = rp.ttest(group1= df1_n['test_score_W0'], group1_name= "test_score_W0",
                                        group2= df2_n['test_score_W\''], group2_name= "test_score_W\'")
            
            print("NMI T-Test")
            print(summary)
            print(results)

            stats.ttest_ind(test_score_W0_n, test_score_W_combine_n,alternative = 'less')

            summary1, results1 = rp.ttest(group1= df3_n['train_score_W0'], group1_name= "train_score_W0",
                                        group2= df4_n['train_score_W\''], group2_name= "train_score_W\'")
            
           # print(summary1)
            #print(results1)

            #stats.ttest_ind(all_train_score_W0_global_n, all_train_score_W_combine_global_n,alternative = 'less')



            df1_a = pd.DataFrame(test_score_W0_a, columns = ['test_score_W0'])
            df2_a = pd.DataFrame(test_score_W_combine_a, columns = ['test_score_W\''])     
            df3_a = pd.DataFrame(all_train_score_W0_a, columns = ['train_score_W0'])
            df4_a = pd.DataFrame(all_train_score_W_combine_a, columns = ['train_score_W\''])

            summary, results = rp.ttest(group1= df1_a['test_score_W0'], group1_name= "test_score_W0",
                                        group2= df2_a['test_score_W\''], group2_name= "test_score_W\'")
            
            print("ARI T-Test")
            print(summary)
            print(results)

            stats.ttest_ind(test_score_W0_a, test_score_W_combine_a,alternative = 'less')

            summary1, results1 = rp.ttest(group1= df3_n['train_score_W0'], group1_name= "train_score_W0",
                                        group2= df4_n['train_score_W\''], group2_name= "train_score_W\'")
            
            #print(summary1)
            #print(results1)

            #stats.ttest_ind(all_train_score_W0_global_a, all_train_score_W_combine_global_a,alternative = 'less')