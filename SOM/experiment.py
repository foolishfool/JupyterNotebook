#!/usr/bin/env python
# coding: utf-8
from copy import deepcopy
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
from typing import List
import newSom
import TDSM_SOM
import UTDSM
import UTDSM_EFOSOM
import UTDSM_ONEHOTCODE
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
                    print("neuron unit number: {}".format(y))
                    if y % 2 == 0:
                        som = newSom.SOM(m=int(y/2) , n= int(y/2), dim=dim_num)  
                    else:
                     som = newSom.SOM(m= y , n= 1, dim=dim_num)  
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

        def UTtest_Discrete_Continuous( self,
                                        dataread,
                                        class_num,dim_num,
                                        dim_num_continuous,
                                        dim_num_discrete,
                                        best_num,scope_num,
                                        unstable_repeat_num,
                                        type,
                                        row,
                                        column
                                       ):
            
            all_train_score_W0_n =[]
            all_train_score_W_combine_n =[]
            test_score_W0_n = []
            test_score_W_combine_n = []


            all_train_score_W0_a =[]
            all_train_score_W_combine_a =[]
            test_score_W0_a = []
            test_score_W_combine_a = []

            all_train_score_W0_discrete_n =[]
            all_train_score_W_combine_discrete_n =[]
            test_score_W0__discrete_n = []
            test_score_W_combine_discrete_n = []

            
            all_train_score_W0_continuous_n =[]
            all_train_score_W_combine_continuous_n =[]
            test_score_W0_continuous_n = []
            test_score_W_combine_continuous_n = []

            all_train_score_W0_continuous_a =[]
            all_train_score_W_combine_continuous_a =[]
            test_score_W0_continuous_a = []
            test_score_W_combine_continuous_a = []
          
            plot_unit = [1]
            
            if type == 0:
                y = class_num
                while y <= scope_num:
                    print("neuron unit number: {}".format(y))
                    if y % 2 == 0:
                        som = newSom.SOM(m=int(y/2) , n= int(y/2), dim=dim_num)  
                        som_continusous = newSom.SOM(m=int(y/2) , n= int(y/2), dim= dim_num_continuous)  
                        som_discrete = newSom.SOM(m=int(y/2) , n= int(y/2), dim= dim_num_discrete)  
                    else:
                     som = newSom.SOM(m= y , n= 1, dim=dim_num)  
                     som_continusous = newSom.SOM(m= y , n= 1, dim= dim_num_continuous)  
                     som_discrete = newSom.SOM(m= y , n= 1, dim= dim_num_discrete)  
                   # optimize_W = UTDSM.UTDSM_SOM(som,dataread.data_train_continuous,dataread.data_test_continuous,dataread.label_train_continuous,dataread.label_test_continuous,elbow_num,row,column)        
                    optimize_W = UTDSM.UTDSM_SOM(som,
                             som_continusous,
                             som_discrete,
                             dataread.data_train,
                             dataread.data_train_continuous,
                             dataread.data_train_discrete,
                             dataread.data_test,
                             dataread.data_test_continuous,
                             dataread.data_test_discrete,
                             dataread.label_train,
                             dataread.label_test,
                             row,column)                          
                    optimize_W.run()

                    all_train_score_W0_continuous_n.append(optimize_W.all_train_score_W0_n)
                    all_train_score_W_combine_continuous_n.append(optimize_W.all_train_score_W_Combined_n)
                    test_score_W0_continuous_n.append(optimize_W.test_score_W0_n)
                    test_score_W_combine_continuous_n.append(optimize_W.test_score_W_Combined_n)


                    all_train_score_W0_continuous_a.append(optimize_W.all_train_score_W0_a)
                    all_train_score_W_combine_continuous_a.append(optimize_W.all_train_score_W_Combined_a)
                    test_score_W0_continuous_a.append(optimize_W.test_score_W0_n)
                    test_score_W_combine_continuous_a.append(optimize_W.test_score_W_Combined_a)


        
                    y =y+1
                    if(y<= scope_num):
                        plot_unit.append(y)
            
            
            if type == 1:
                y = 1
                while y <= unstable_repeat_num:
                    if best_num % 2 == 0:
                        som = newSom.SOM(m= int(best_num/2), n= int(best_num/2) , dim=dim_num) 
                        som_continusous = newSom.SOM(m=int(y/2) , n= int(y/2), dim= dim_num_continuous)  
                        som_discrete = newSom.SOM(m=int(y/2) , n= int(y/2), dim= dim_num_discrete)   
                    else:
                        som = newSom.SOM(m= best_num , n= 1, dim=dim_num)  

                    optimize_W = UTDSM.UTDSM_SOM(som,dataread.data_train,dataread.data_test,dataread.label_train,dataread.label_test,row,column)
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

            print("all_train_score_W0_n len {}".format(len(all_train_score_W0_n)))
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

        def CompareTwoSOM( self,
                                        dataread,
                                        class_num,dim_num,
                                        best_num,
                                        best_num_cleaned,
                                        scope_num,
                                        unstable_repeat_num,
                                        type,
                                        row,
                                        column
                                       ):
            
            all_train_score_W0_n =[]
            test_score_W0_n = []


            all_train_score_W0_a =[]
            test_score_W0_a = []

       
            all_train_score_W0_cleaned_n =[]
            test_score_W0_cleaned_n = []
            all_train_score_W0_cleaned_a =[]
            test_score_W0_cleaned_a = []

            plot_unit = [1]
            
            if type == 0:
                y = class_num
                while y <= scope_num:
                    print("neuron unit number: {}".format(y))
                    if y % 2 == 0:
                        som = newSom.SOM(m=int(y/2) , n= int(y/2), dim=dim_num)  
                        som_cleaned = newSom.SOM(m=int(y/2) , n= int(y/2), dim=dim_num)  
                    else:
                     som = newSom.SOM(m= y , n= 1, dim=dim_num)  
                     som_cleaned = newSom.SOM(m= y , n= 1, dim=dim_num) 

                   # optimize_W = UTDSM.UTDSM_SOM(som,dataread.data_train_continuous,dataread.data_test_continuous,dataread.label_train_continuous,dataread.label_test_continuous,elbow_num,row,column)        
                    optimize_W = UTDSM_NORMALSOM.UTDSM_NORMALSOM(som,
                             som_cleaned,
                             dataread.data_train,
                             dataread.cleanedData,
                             dataread.data_test,
                             dataread.label_train,
                             dataread.cleanedLabel,
                             dataread.label_test,
                             row,column)                          
                    optimize_W.run()

                    all_train_score_W0_n.append(optimize_W.all_train_score_W0_n)
                    test_score_W0_n.append(optimize_W.test_score_W0_n)


                    all_train_score_W0_cleaned_n.append(optimize_W.all_train_score_W0_cleaned_n)
                    test_score_W0_cleaned_n.append(optimize_W.test_score_W0_cleaned_n)

                    all_train_score_W0_a.append(optimize_W.all_train_score_W0_a)
                    test_score_W0_a.append(optimize_W.test_score_W0_a)


                    all_train_score_W0_cleaned_a.append(optimize_W.all_train_score_W0_cleaned_a)
                    test_score_W0_cleaned_a.append(optimize_W.test_score_W0_cleaned_a)
                  
                    y =y+1
                    if(y<= scope_num):
                        plot_unit.append(y)
            
            
            if type == 1:
                y = 1
                while y <= unstable_repeat_num:
                    if best_num % 2 == 0:
                        som = newSom.SOM(m= int(best_num/2), n= int(best_num/2) , dim=dim_num) 
                        som_cleaned = newSom.SOM(m= int(best_num_cleaned/2), n= int(best_num_cleaned/2) , dim=dim_num) 
                    else:
                        som = newSom.SOM(m= best_num , n= 1, dim=dim_num)  
                        som_cleaned = newSom.SOM(m= best_num_cleaned , n= 1, dim=dim_num)  
                   
                    optimize_W = UTDSM_NORMALSOM.UTDSM_NORMALSOM(som,
                             som_cleaned,
                             dataread.data_train,
                             dataread.cleanedData,
                             dataread.data_test,
                             dataread.label_train,
                             dataread.cleanedLabel,
                             dataread.label_test,
                             row,column)    
                    optimize_W.run()
                  
                    all_train_score_W0_n.append(optimize_W.all_train_score_W0_n)
                    test_score_W0_n.append(optimize_W.test_score_W0_n)


                    all_train_score_W0_cleaned_n.append(optimize_W.all_train_score_W0_cleaned_n)
                    test_score_W0_cleaned_n.append(optimize_W.test_score_W0_cleaned_n)

                    all_train_score_W0_a.append(optimize_W.all_train_score_W0_a)
                    test_score_W0_a.append(optimize_W.test_score_W0_a)


                    all_train_score_W0_cleaned_a.append(optimize_W.all_train_score_W0_cleaned_a)
                    test_score_W0_cleaned_a.append(optimize_W.test_score_W0_cleaned_a)
                     
                    y =y+1
                    if(y<= unstable_repeat_num):
                        plot_unit.append(y)
                
                        
            figure, axis = plt.subplots(1, 2,figsize =(12, 5))
            axis[0].set_title("NMI Score")               
            axis[1].set_title("ARI Score")

           # print("all_train_score_W0_n len {}".format(len(all_train_score_W0_n)))
           # print("plot_unit {}".format(plot_unit))
            if type == 0:
                axis[0].set_xlabel('Neuron number')
            if type == 1:
                axis[0].set_xlabel('Repeat number')
            
            axis[0].plot(plot_unit,all_train_score_W0_n,'r',label ='all_train_score')
            axis[0].plot(plot_unit,all_train_score_W0_cleaned_n,'c',label ='all_train_score_cleaned')
            axis[0].plot(plot_unit,test_score_W0_n,'y',label ='test_score')
            axis[0].plot(plot_unit,test_score_W0_cleaned_n,'k',label ='test_score_cleaned')
            axis[0].legend(loc='best')



            if type == 0:
                axis[1].set_xlabel('Neuron number')
            if type == 1:
                axis[1].set_xlabel('Repeat number')

            axis[1].plot(plot_unit,all_train_score_W0_a,'r',label ='all_train_score')
            axis[1].plot(plot_unit,all_train_score_W0_cleaned_a,'c',label ='all_train_score_cleaned')
            axis[1].plot(plot_unit,test_score_W0_a,'y',label ='test_score')
            axis[1].plot(plot_unit,test_score_W0_cleaned_a,'k',label ='test_score_cleaned')
         
            axis[1].legend(loc='best')
            plt.show()
            
           # print("New Neuron Number : {}".format (int(np.mean(new_neuron_num))))
                                                    
                        
            df1_n = pd.DataFrame(test_score_W0_n, columns = ['test_score_W0'])
            df2_n = pd.DataFrame(test_score_W0_cleaned_n, columns = ['test_score_cleaned'])



            summary, results = rp.ttest(group1= df1_n['test_score_W0'], group1_name= "test_score_W0",
                                        group2= df2_n['test_score_cleaned'], group2_name= "test_score_cleaned")
            
            print("NMI T-Test")
            print(summary)
            print(results)

            stats.ttest_ind(test_score_W0_n, test_score_W0_cleaned_n,alternative = 'less')



            df1_a = pd.DataFrame(test_score_W0_a, columns = ['test_score_W0'])
            df2_a = pd.DataFrame(test_score_W0_cleaned_a, columns = ['test_score_cleaned'])     


            summary, results = rp.ttest(group1= df1_a['test_score_W0'], group1_name= "test_score_W0",
                                        group2= df2_a['test_score_cleaned'], group2_name= "test_score_cleaned")
            
            print("ARI T-Test")
            print(summary)
            print(results)

            stats.ttest_ind(test_score_W0_a, test_score_W0_cleaned_a,alternative = 'less')



        def UTest_PureDiscreteData( self,
                                        dataread,
                                        class_num,dim_num,
                                        original_som_best_num,
                                        onehot_som_best_num,
                                        scope_num,
                                        unstable_repeat_num,
                                        type,
                                        row,
                                        column
                                       ):
            
            all_train_score_W0_n =[]
            test_score_W0_n = []


            all_train_score_W0_a =[]
            test_score_W0_a = []

       
            all_train_score_W0_onehot_n =[]
            test_score_W0_onehot_n = []
            all_train_score_W0_onehot_a =[]
            test_score_W0_onehot_a = []

            plot_unit = [1]
            
            if type == 0:
                y = class_num
                while y <= scope_num:
                    print("neuron unit number: {}".format(y))
                    if y % 2 == 0:
                        som = newSom.SOM(m=int(y/2) , n= int(y/2), dim=dim_num)  
                        som_onehot = newSom.SOM(m=int(y/2) , n= int(y/2), dim=dim_num)  
                    else:
                     som = newSom.SOM(m= y , n= 1, dim=dim_num)  
                     som_onehot = newSom.SOM(m= y , n= 1, dim=dim_num) 

                   # optimize_W = UTDSM.UTDSM_SOM(som,dataread.data_train_continuous,dataread.data_test_continuous,dataread.label_train_continuous,dataread.label_test_continuous,elbow_num,row,column)        
                    optimize_W = UTDSM_ONEHOTCODE.UTDSM_ONEHOTCODE(som,
                             som_onehot,
                             dataread.data_train,
                             dataread.data_train_discrete,
                             dataread.data_test,
                             dataread.data_test_discrete,
                             dataread.label_train,
                             dataread.label_test,
                             row,column)                          
                    optimize_W.run()

                    all_train_score_W0_n.append(optimize_W.all_train_score_W0_n)
                    test_score_W0_n.append(optimize_W.test_score_W0_n)


                    all_train_score_W0_onehot_n.append(optimize_W.all_train_score_W0_onehot_n)
                    test_score_W0_onehot_n.append(optimize_W.test_score_W0_onehot_n)

                    all_train_score_W0_a.append(optimize_W.all_train_score_W0_a)
                    test_score_W0_a.append(optimize_W.test_score_W0_a)


                    all_train_score_W0_onehot_a.append(optimize_W.all_train_score_W0_onehot_a)
                    test_score_W0_onehot_a.append(optimize_W.test_score_W0_onehot_a)
                  
                    y =y+1
                    if(y<= scope_num):
                        plot_unit.append(y)
            
            
            if type == 1:
                y = 1
                while y <= unstable_repeat_num:
                    if original_som_best_num % 2 == 0:
                        som = newSom.SOM(m= int(original_som_best_num/2), n= int(original_som_best_num/2) , dim=dim_num) 
                        som_onehot = newSom.SOM(m= int(onehot_som_best_num/2), n= int(onehot_som_best_num/2) , dim=dim_num) 
                    else:
                        som = newSom.SOM(m= original_som_best_num , n= 1, dim=dim_num)  
                        som_onehot = newSom.SOM(m= onehot_som_best_num , n= 1, dim=dim_num)  
                   
                    optimize_W = UTDSM_ONEHOTCODE.UTDSM_ONEHOTCODE(som,
                             som_onehot,
                             dataread.data_train,
                             dataread.data_train_discrete,
                             dataread.data_test,
                             dataread.data_test_discrete,
                             dataread.label_train,
                             dataread.label_test,
                             row,column)    
                    optimize_W.run()
                  
                    all_train_score_W0_n.append(optimize_W.all_train_score_W0_n)
                    test_score_W0_n.append(optimize_W.test_score_W0_n)


                    all_train_score_W0_onehot_n.append(optimize_W.all_train_score_W0_onehot_n)
                    test_score_W0_onehot_n.append(optimize_W.test_score_W0_onehot_n)

                    all_train_score_W0_a.append(optimize_W.all_train_score_W0_a)
                    test_score_W0_a.append(optimize_W.test_score_W0_a)


                    all_train_score_W0_onehot_a.append(optimize_W.all_train_score_W0_onehot_a)
                    test_score_W0_onehot_a.append(optimize_W.test_score_W0_onehot_a)
                     
                    y =y+1
                    if(y<= unstable_repeat_num):
                        plot_unit.append(y)
                
                        
            figure, axis = plt.subplots(1, 2,figsize =(12, 5))
            axis[0].set_title("NMI Score")               
            axis[1].set_title("ARI Score")

           # print("all_train_score_W0_n len {}".format(len(all_train_score_W0_n)))
           # print("plot_unit {}".format(plot_unit))
            if type == 0:
                axis[0].set_xlabel('Neuron number')
            if type == 1:
                axis[0].set_xlabel('Repeat number')
            
            axis[0].plot(plot_unit,all_train_score_W0_n,'r',label ='all_train_score')
            axis[0].plot(plot_unit,all_train_score_W0_onehot_n,'c',label ='all_train_score_onehot')
            axis[0].plot(plot_unit,test_score_W0_n,'y',label ='test_score')
            axis[0].plot(plot_unit,test_score_W0_onehot_n,'k',label ='test_score_onehot')
            axis[0].legend(loc='best')



            if type == 0:
                axis[1].set_xlabel('Neuron number')
            if type == 1:
                axis[1].set_xlabel('Repeat number')

            axis[1].plot(plot_unit,all_train_score_W0_a,'r',label ='all_train_score')
            axis[1].plot(plot_unit,all_train_score_W0_onehot_a,'c',label ='all_train_score_onehot')
            axis[1].plot(plot_unit,test_score_W0_a,'y',label ='test_score')
            axis[1].plot(plot_unit,test_score_W0_onehot_a,'k',label ='test_score_onehot')
         
            axis[1].legend(loc='best')
            plt.show()
            
           # print("New Neuron Number : {}".format (int(np.mean(new_neuron_num))))
                                                    
                        
            df1_n = pd.DataFrame(test_score_W0_n, columns = ['test_score_W0'])
            df2_n = pd.DataFrame(test_score_W0_onehot_n, columns = ['test_score_onehot'])



            summary, results = rp.ttest(group1= df1_n['test_score_W0'], group1_name= "test_score_W0",
                                        group2= df2_n['test_score_onehot'], group2_name= "test_score_onehot")
            
            print("NMI T-Test")
            print(summary)
            print(results)

            stats.ttest_ind(test_score_W0_n, test_score_W0_onehot_n,alternative = 'less')



            df1_a = pd.DataFrame(test_score_W0_a, columns = ['test_score_W0'])
            df2_a = pd.DataFrame(test_score_W0_onehot_a, columns = ['test_score_onehot'])     


            summary, results = rp.ttest(group1= df1_a['test_score_W0'], group1_name= "test_score_W0",
                                        group2= df2_a['test_score_onehot'], group2_name= "test_score_onehot")
            
            print("ARI T-Test")
            print(summary)
            print(results)

            stats.ttest_ind(test_score_W0_a, test_score_W0_onehot_a,alternative = 'less')

            

        def UTest_FESOM( self,
                                        dataread,
                                        class_num,
                                        dim_num,
                                        original_som_best_num,
                                        scope_num,
                                        unstable_repeat_num,
                                        type,
                                        row,
                                        column
                                       ):
            
            all_train_score_W0_n =[]
            test_score_W0_n = []


            all_train_score_W0_a =[]
            test_score_W0_a = []

    
            all_train_score_W0_transferred_n =[]
            all_test_score_W0_transferred_n = []
            all_train_score_W0_transferred_a =[]
            all_test_score_W0_transferred_a = []

            plot_unit = [1]
            

            soms = []
            for i in range(0, dataread.all_data.shape[1]):

               # print("dataread.data_test_discrete[:,i] {} {}".format(i, dataread.data_train_discrete_before_transfer[:,i]))
                sum_num = len(np.unique(dataread.all_data[:,i]))  
                #print("sum_num {} i{} npunique {}".format(sum_num,i,np.unique(dataread.all_data[:,i])));   
              #  print("np.unique(dataread.data_test_discrete[:,i] {}".format(np.unique(dataread.data_train_discrete_before_transfer[:,i])) )
                m, n = self.topology_som(sum_num)
                #som_feature = newSom.SOM(m, n, dim=sum_num) 
                if 0 in np.unique(dataread.all_data[:,i]):
                    som_feature = newSom.SOM(m, n, dim=sum_num)  
                else: som_feature = newSom.SOM(m , n, dim=sum_num+1)  

                soms.append(som_feature)
            
            if type == 0:
                y = class_num
                while y <= scope_num:
                    m, n = self.topology_som(y)
                    print("neuron unit number: {}".format(y))

                    som = newSom.SOM(m, n, dim=dim_num)  
                    som_transferred = newSom.SOM(m, n, dim=dim_num)  

                   # optimize_W = UTDSM.UTDSM_SOM(som,dataread.data_train_continuous,dataread.data_test_continuous,dataread.label_train_continuous,dataread.label_test_continuous,elbow_num,row,column)        
                    optimize_W = UTDSM_EFOSOM.UTDSM_EFOSOM(som,
                             soms,
                             som_transferred,
                             dataread.data_train,
                             dataread.data_train_discrete_after_transfer,
                             dataread.data_test,
                             dataread.data_test_discrete_after_transfer,
                             dataread.label_train,
                             dataread.label_test,
                             row,column)                          
                    optimize_W.run()

                    
                    all_train_score_W0_n.append(optimize_W.all_train_score_W0_n)
                    test_score_W0_n.append(optimize_W.test_score_W0_n)


                    all_train_score_W0_transferred_n.append(optimize_W.all_train_score_W0_transferred_n)
                    all_test_score_W0_transferred_n.append(optimize_W.test_score_W0_transferred_n)

                    all_train_score_W0_a.append(optimize_W.all_train_score_W0_a)
                    test_score_W0_a.append(optimize_W.test_score_W0_a)


                    all_train_score_W0_transferred_a.append(optimize_W.all_train_score_W0_transferred_a)
                    all_test_score_W0_transferred_a.append(optimize_W.test_score_W0_transferred_a)
                  
                    y =y+5
                    if(y<= scope_num):
                        plot_unit.append(y)
            
            
            if type == 1:
                y = 1
                while y <= unstable_repeat_num:
                    m, n = self.topology_som(original_som_best_num)

                    som = newSom.SOM(m, n, dim=dim_num)  
                    som_transferred = newSom.SOM(m, n, dim=dim_num)  

                    optimize_W = UTDSM_EFOSOM.UTDSM_EFOSOM(som,
                             soms,
                             som_transferred,
                             dataread.data_train,
                             dataread.data_train_discrete_after_transfer,
                             dataread.data_test,
                             dataread.data_test_discrete_after_transfer,
                             dataread.label_train,
                             dataread.label_test,
                             row,column)    
                    optimize_W.run()
                  
                    all_train_score_W0_n.append(optimize_W.all_train_score_W0_n)
                    test_score_W0_n.append(optimize_W.test_score_W0_n)


                    all_train_score_W0_transferred_n.append(optimize_W.all_train_score_W0_transferred_n)
                    all_test_score_W0_transferred_n.append(optimize_W.test_score_W0_transferred_n)

                    all_train_score_W0_a.append(optimize_W.all_train_score_W0_a)
                    test_score_W0_a.append(optimize_W.test_score_W0_a)


                    all_train_score_W0_transferred_a.append(optimize_W.all_train_score_W0_transferred_a)
                    all_test_score_W0_transferred_a.append(optimize_W.test_score_W0_transferred_a)
                     
                    y =y+1
                    if(y<= unstable_repeat_num):
                        plot_unit.append(y)
                
                        
            figure, axis = plt.subplots(1, 2,figsize =(12, 5))
            axis[0].set_title("NMI Score")               
            axis[1].set_title("ARI Score")

           # print("all_train_score_W0_n len {}".format(len(all_train_score_W0_n)))
           # print("plot_unit {}".format(plot_unit))
            if type == 0:
                axis[0].set_xlabel('Neuron number')
            if type == 1:
                axis[0].set_xlabel('Repeat number')
            
            axis[0].plot(plot_unit,all_train_score_W0_n,'r',label ='all_train_score')
            axis[0].plot(plot_unit,all_train_score_W0_transferred_n,'c',label ='all_train_score_transferred')
            axis[0].plot(plot_unit,test_score_W0_n,'y',label ='test_score')
            axis[0].plot(plot_unit,all_test_score_W0_transferred_n,'k',label ='test_score_transferred')
            axis[0].legend(loc='best')



            if type == 0:
                axis[1].set_xlabel('Neuron number')
            if type == 1:
                axis[1].set_xlabel('Repeat number')

            axis[1].plot(plot_unit,all_train_score_W0_a,'r',label ='all_train_score')
            axis[1].plot(plot_unit,all_train_score_W0_transferred_a,'c',label ='all_train_score_transferred')
            axis[1].plot(plot_unit,test_score_W0_a,'y',label ='test_score')
            axis[1].plot(plot_unit,all_test_score_W0_transferred_a,'k',label ='test_score_transferred')
         
            axis[1].legend(loc='best')
            plt.show()
            
           # print("New Neuron Number : {}".format (int(np.mean(new_neuron_num))))
                                                    
                        
            df1_n = pd.DataFrame(test_score_W0_n, columns = ['test_score_W0'])
            df2_n = pd.DataFrame(all_test_score_W0_transferred_n, columns = ['test_score_transferred'])



            summary, results = rp.ttest(group1= df1_n['test_score_W0'], group1_name= "test_score_W0",
                                        group2= df2_n['test_score_transferred'], group2_name= "test_score_transferred")
            
            print("NMI T-Test")
            print(summary)
            print(results)

            stats.ttest_ind(test_score_W0_n, all_test_score_W0_transferred_n,alternative = 'less')



            df1_a = pd.DataFrame(test_score_W0_a, columns = ['test_score_W0'])
            df2_a = pd.DataFrame(all_test_score_W0_transferred_a, columns = ['test_score_transferred'])     


            summary, results = rp.ttest(group1= df1_a['test_score_W0'], group1_name= "test_score_W0",
                                        group2= df2_a['test_score_transferred'], group2_name= "test_score_transferred")
            
            print("ARI T-Test")
            print(summary)
            print(results)

            stats.ttest_ind(test_score_W0_a, all_test_score_W0_transferred_a,alternative = 'less')


