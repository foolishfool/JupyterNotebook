#!/usr/bin/env python
# coding: utf-8
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
from typing import List
import newSom
import TDSM_SOM
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


        def Ttest(self,dataread, class_num,dim_num,scope_num,unstable_repeat_num):
        
            unit_list = []    
            all_train_score_W0_p =[]
            all_train_score_W_combine_p =[]
            test_score_W0_p = []
            test_score_W_combine_p= []
            
            all_train_score_W0_n =[]
            all_train_score_W_combine_n =[]
            test_score_W0_n = []
            test_score_W_combine_n = []


            all_train_score_W0_a =[]
            all_train_score_W_combine_a =[]
            test_score_W0_a = []
            test_score_W_combine_a = []
    
            initial_som_result = dict()
            initial_som_result_train = dict()
            splitted_number_result = dict()
            splitted_number_result_train = dict()


            all_train_score_W0_global_p =[]
            all_train_score_W_combine_global_p =[]
            test_score_W0_global_p = []
            test_score_W_combine_global_p = []

            all_train_score_W0_global_n =[]
            all_train_score_W_combine_global_n =[]
            test_score_W0_global_n = []
            test_score_W_combine_global_n = []

            all_train_score_W0_global_a =[]
            all_train_score_W_combine_global_a =[]
            test_score_W0_global_a = []
            test_score_W_combine_global_a = []

            p_values = []
            y = 1


            while y <= unstable_repeat_num:
                x = 1
                while x <= scope_num:
                    unit_list.append(class_num*x)
                    print("neuron unit number: {}".format(class_num*x))
                    print("*******************\n")
                    som = newSom.SOM(m= class_num, n= x, dim=dim_num)  
                    optimize_W = TDSM_SOM.TDSM_SOM(som,dataread.data_train,dataread.data_test,dataread.label_train,dataread.label_test,class_num)
                    optimize_W.run()
                    

                    all_train_score_W0_p.append(optimize_W.all_train_score_W0_p)
                    all_train_score_W_combine_p.append(optimize_W.all_train_score_W_Combined_p)
                    test_score_W0_p.append(optimize_W.test_score_W0_p)
                    test_score_W_combine_p.append(optimize_W.test_score_W_combined_p)


                    all_train_score_W0_n.append(optimize_W.all_train_score_W0_n)
                    all_train_score_W_combine_n.append(optimize_W.all_train_score_W_Combined_n)
                    test_score_W0_n.append(optimize_W.test_score_W0_n)
                    test_score_W_combine_n.append(optimize_W.test_score_W_combined_n)


                    all_train_score_W0_a.append(optimize_W.all_train_score_W0_a)
                    all_train_score_W_combine_a.append(optimize_W.all_train_score_W_Combined_a)
                    test_score_W0_a.append(optimize_W.test_score_W0_n)
                    test_score_W_combine_a.append(optimize_W.test_score_W_combined_a)
     

                    
                    all_train_score_W0_global_p.append(optimize_W.all_train_score_W0_p)
                    all_train_score_W_combine_global_p.append(optimize_W.all_train_score_W_Combined_p)
                    test_score_W0_global_p.append(optimize_W.test_score_W0_p)
                    test_score_W_combine_global_p.append(optimize_W.test_score_W_combined_p)   


                    all_train_score_W0_global_n.append(optimize_W.all_train_score_W0_n)
                    all_train_score_W_combine_global_n.append(optimize_W.all_train_score_W_Combined_n)
                    test_score_W0_global_n.append(optimize_W.test_score_W0_n)
                    test_score_W_combine_global_n.append(optimize_W.test_score_W_combined_n)   



                    all_train_score_W0_global_a.append(optimize_W.all_train_score_W0_a)
                    all_train_score_W_combine_global_a.append(optimize_W.all_train_score_W_Combined_a)
                    test_score_W0_global_a.append(optimize_W.test_score_W0_a)
                    test_score_W_combine_global_a.append(optimize_W.test_score_W_combined_a)   


                    splitted_number_result[class_num*x*(optimize_W.split_num +1)] = optimize_W.test_score_W_combined_p 
                    splitted_number_result_train[class_num*x*(optimize_W.split_num +1)] = optimize_W.all_train_score_W_Combined_p 
                    initial_som_result[class_num*x] = optimize_W.test_score_W0_p 
                    initial_som_result_train[class_num*x] = optimize_W.all_train_score_W0_p 
                    
                    x=x+1

                
                figure, axis = plt.subplots(1, 2,figsize=(12, 5))
                axis[0].set_title("NMI Score")               
                axis[1].set_title("ARI Score")




                axis[0].set_xlabel('Neuron number')
                axis[0].plot(unit_list,all_train_score_W0_n,'r',label ='all_train_score_W0')
                axis[0].plot(unit_list,all_train_score_W_combine_n,'c',label ='all_train_score_W\'')
                axis[0].plot(unit_list,test_score_W0_n,'y',label ='test_score_W1')
                axis[0].plot(unit_list,test_score_W_combine_n,'k',label ='test_score_W\'')
                axis[0].legend(loc='best')



                axis[1].set_xlabel('Neuron number')
                axis[1].plot(unit_list,all_train_score_W0_a,'r',label ='all_train_score_W0')
                axis[1].plot(unit_list,all_train_score_W_combine_a,'c',label ='all_train_score_W\'')
                axis[1].plot(unit_list,test_score_W0_a,'y',label ='test_score_W1')
                axis[1].plot(unit_list,test_score_W_combine_a,'k',label ='test_score_W\'')
                axis[1].legend(loc='best')
                plt.show()
                
                y =y+1
                #reset
                unit_list = []  
                all_train_score_W0_p =[]
                all_train_score_W_combine_p =[]
                test_score_W0_p = []
                test_score_W_combine_p= []


                all_train_score_W0_n =[]
                all_train_score_W_combine_n =[]
                test_score_W0_n = []
                test_score_W_combine_n= []


                all_train_score_W0_a =[]
                all_train_score_W_combine_a  =[]
                test_score_W0_a  = []
                test_score_W_combine_a = []
                       
                                      

             
            test_score_W0_global_p =[] #reset test_score_W0_global
            test_score_W_combine_global_p = []
            all_train_score_W0_global_p =[] 
            all_train_score_W_combine_global_p = []
                
            od = collections.OrderedDict(sorted(splitted_number_result.items()))
            od2 = collections.OrderedDict(sorted(splitted_number_result_train.items()))
            
            keys = od.keys()
            for s in keys:
                    if s in initial_som_result:
                        test_score_W_combine_global_p.append(od[s])
                        test_score_W0_global_p.append(initial_som_result[s])
                        unit_list.append(s)
                        print("s {}  value {} ".format(s,od[s]))
                
            keys2 = od2.keys()   
            for k in keys2:
                    if k in initial_som_result_train:
                        all_train_score_W_combine_global_p.append(od2[k])
                        all_train_score_W0_global_p.append(initial_som_result_train[k])

                    
            df1_p = pd.DataFrame(test_score_W0_global_p, columns = ['test_score_W0'])                                
            df2_p = pd.DataFrame(test_score_W_combine_global_p, columns = ['test_score_W\''])                          

            figure, axis = plt.subplots(1, 2,figsize=(12, 5))
            axis[0].set_title("Purity Score")   
            axis[0].set_xlabel('Neuron number')
            axis[0].plot(unit_list,all_train_score_W0_global_p,'r',label ='all_train_score_W0')
            axis[0].plot(unit_list,all_train_score_W_combine_global_p,'c',label ='all_train_score_W\'')
            axis[0].plot(unit_list,test_score_W0_global_p,'y',label ='test_score_W1')
            axis[0].plot(unit_list,test_score_W_combine_global_p,'k',label ='test_score_W\'')
            axis[0].legend(loc='best')
            plt.show() 


          # fig2 = plt.figure(figsize=(5,5))
          # axis[0].set_title("Purity Score")         
          # plt.title("Purity Score")      
          # plt.xlabel('Neuron number')
          # plt.plot(unit_list,all_train_score_W0_global_p,'r',label ='all_train_score_W0')
          # plt.plot(unit_list,all_train_score_W_combine_global_p,'c',label ='all_train_score_W\'')
          # plt.plot(unit_list,test_score_W0_global_p,'y',label ='test_score_W1')
          # plt.plot(unit_list,test_score_W_combine_global_p,'k',label ='test_score_W\'')
          # plt.legend()
          # plt.show() 

            summary, results = rp.ttest(group1= df1_p['test_score_W0'], group1_name= "test_score_W0",
                                        group2= df2_p['test_score_W\''], group2_name= "test_score_W\'")
            
            print("Purity T-Test")
            print(summary)
            print(results)

            stats.ttest_ind(test_score_W0_global_p, test_score_W_combine_global_p,alternative = 'less')
                                               
            
   

                        
            df1_n = pd.DataFrame(test_score_W0_global_n, columns = ['test_score_W0'])
            df2_n = pd.DataFrame(test_score_W_combine_global_n, columns = ['test_score_W\''])



            summary, results = rp.ttest(group1= df1_n['test_score_W0'], group1_name= "test_score_W0",
                                        group2= df2_n['test_score_W\''], group2_name= "test_score_W\'")
            
            print("NMI T-Test")
            print(summary)
            print(results)

            stats.ttest_ind(test_score_W0_global_n, test_score_W_combine_global_n,alternative = 'less')

            df1_a = pd.DataFrame(test_score_W0_global_a, columns = ['test_score_W0'])
            df2_a = pd.DataFrame(test_score_W_combine_global_a, columns = ['test_score_W\''])     

            summary, results = rp.ttest(group1= df1_a['test_score_W0'], group1_name= "test_score_W0",
                                        group2= df2_a['test_score_W\''], group2_name= "test_score_W\'")
            
            print("ARI T-Test")
            print(summary)
            print(results)

            stats.ttest_ind(test_score_W0_global_a, test_score_W_combine_global_a,alternative = 'less')