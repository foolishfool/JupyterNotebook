#!/usr/bin/env python
# coding: utf-8

from operator import truediv
from re import escape
from tarfile import NUL
import matplotlib.pyplot as plt
from typing import List
import newSom
import CDSOM
import numpy as np
from sklearn.metrics import silhouette_score
import scipy.stats as stats
import pandas as pd
import researchpy as rp
class Experiment():
        def __init__(self):
         return

        # smooth the curve
        def smooth(self, scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
            last = scalars[0]  # First value in the plot (first timestep)
            smoothed = list()
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
                smoothed.append(smoothed_val)                        # Save it
                last = smoothed_val                                  # Anchor the last smoothed value

            return smoothed
        
        #find base cluster number based on the minia inertia value
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
        #return start row and factor column  the topology of som is startxfactor
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


        def InitializedExperimentDataList(self,
                                        dataread,
                                        y,
                                        all_train_score_W0_n,
                                        all_train_score_W_combine_n,
                                        test_score_W0_n,
                                        test_score_W_combine_n,
                                        all_train_score_W0_a,
                                        all_train_score_W_combine_a,
                                        test_score_W0_a,
                                        test_score_W_combine_a,
                                        train_continuous_score_W0_n,
                                        train_continuous_score_W_continuous_n,
                                        test_continuous_score_W0_n,
                                        test_continuous_score_W_continuous_n,
                                        train_continuous_score_W0_a,
                                        train_continuous_score_W_continuous_a,
                                        test_continuous_score_W0_a,
                                        test_continuous_score_W_continuous_a,
                                        train_discrete_score_W0_n,
                                        train_discrete_score_W_discrete_n,
                                        test_discrete_score_W0_n,
                                        test_discrete_score_W_discrete_n,
                                        train_discrete_score_W0_a,
                                        train_discrete_score_W_discrete_a,
                                        test_discrete_score_W0_a,
                                        test_discrete_score_W_discrete_a,
                                        iscontinous_data_test,
                                        compareType):

            m, n = self.topology_som(y)
       
            som = newSom.SOM(m , n, dim=dataread.continuous_feature_num + dataread.discrete_feature_num)  
            if  dataread.continuous_feature_num != 0:
             som_continusous = newSom.SOM(m , n, dim= dataread.continuous_feature_num)  
            if  dataread.discrete_feature_num != 0:
                som_discrete_original = newSom.SOM(m , n, dim= dataread.discrete_feature_num) 
                som_total_discrete = newSom.SOM(m , n, dim= dataread.discrete_feature_num) 
            else: 
                # som_discrete_original will be used in CDSOM initialize, so it cannot be null, although won't be used
                som_discrete_original = newSom.SOM(m , n, dim= dataread.continuous_feature_num) 
                som_total_discrete = som_continusous  


            discrete_feature_soms = []
            for i in range(0, dataread.data_train_discrete.shape[1]):
                current_feature_unique_num = len(np.unique(dataread.data_train_discrete[:,i]))  
                #get dimensionality of som
                m, n = self.topology_som(current_feature_unique_num)

                # get dim of feateures
                if 0 in np.unique(dataread.data_train_discrete[:,i]):
                    som_feature = newSom.SOM(m, n, dim=current_feature_unique_num)  
                else: som_feature = newSom.SOM(m , n, dim=current_feature_unique_num+1)  



                discrete_feature_soms.append(som_feature)
            




            optimize_W = CDSOM.CDSOM(som,
                     som_continusous,
                     discrete_feature_soms,
                     som_discrete_original,
                     som_total_discrete,
                     dataread.data_train,
                     dataread.data_train_continuous,
                     dataread.data_train_discrete,
                     dataread.data_train_discrete_normalized,
                     dataread.data_train_discrete_after_transfer,
                     dataread.data_test,
                     dataread.data_test_continuous,
                     dataread.data_test_discrete,
                     dataread.data_test_discrete_normalized,
                     dataread.data_test_discrete_after_transfer,
                     dataread.label_train,
                     dataread.label_test)                          

            if compareType == 0: # normalized and then do the COSOM
                optimize_W.do_COSOM(optimize_W.som_discrete_original,optimize_W.data_train_discrete_normalized, optimize_W.data_train_discrete_unnormalized , optimize_W.data_test_discrete_normalized,optimize_W.data_test_discrete_unnormalized, iscontinous_data_test,0)
            if compareType == 1: # for the discrete data directly do the cosom without normalization
                optimize_W.do_COSOM(optimize_W.som_discrete_original,optimize_W.data_train_discrete_unnormalized,optimize_W.data_train_discrete_unnormalized ,optimize_W.data_test_discrete_unnormalized,optimize_W.data_test_discrete_unnormalized,iscontinous_data_test,0)
            if compareType == 2:
                # do the normaliziation but compared with som with unrmalization
                optimize_W.do_COSOM(optimize_W.som_discrete_original,optimize_W.data_train_discrete_normalized, optimize_W.data_train_discrete_unnormalized, optimize_W.data_test_discrete_normalized,optimize_W.data_test_discrete_unnormalized,iscontinous_data_test,1)
          # all_train_score_W0_n.append(optimize_W.all_train_score_W0_n)
          # all_train_score_W_combine_n.append(optimize_W.all_train_score_W_combined_n)
          # test_score_W0_n.append(optimize_W.test_score_W0_n)
          # test_score_W_combine_n.append(optimize_W.test_score_W_combined_n)


         #   all_train_score_W0_a.append(optimize_W.all_train_score_W0_a)
         #   all_train_score_W_combine_a.append(optimize_W.all_train_score_W_combined_a)
         #   test_score_W0_a.append(optimize_W.test_score_W0_a)
         #   test_score_W_combine_a.append(optimize_W.test_score_W_combined_a)

           
            if iscontinous_data_test == True:
                train_continuous_score_W0_n.append(optimize_W.train_continuous_score_W0_n)
                train_continuous_score_W_continuous_n.append(optimize_W.train_continuous_score_W_continuous_n)
                test_continuous_score_W0_n.append(optimize_W.test_continuous_score_W0_n)
                test_continuous_score_W_continuous_n.append(optimize_W.test_continuous_score_W_continuous_n)

      
                train_continuous_score_W0_a.append(optimize_W.train_continuous_score_W0_a)
                train_continuous_score_W_continuous_a.append(optimize_W.train_continuous_score_W_continuous_a)
                test_continuous_score_W0_a.append(optimize_W.test_continuous_score_W0_a)
                test_continuous_score_W_continuous_a.append(optimize_W.test_continuous_score_W_continuous_a)
            else:



                train_discrete_score_W0_n.append(optimize_W.train_discrete_score_W0_n)
                train_discrete_score_W_discrete_n.append(optimize_W.train_discrete_score_W_discrete_n)
                test_discrete_score_W0_n.append(optimize_W.test_discrete_score_W0_n)
                test_discrete_score_W_discrete_n.append(optimize_W.test_discrete_score_W_discrete_n)

        
                train_discrete_score_W0_a.append(optimize_W.train_discrete_score_W0_a)
                train_discrete_score_W_discrete_a.append(optimize_W.train_discrete_score_W_discrete_a)
                test_discrete_score_W0_a.append(optimize_W.test_discrete_score_W0_a)
                test_discrete_score_W_discrete_a.append(optimize_W.test_discrete_score_W_discrete_a)



        def UTtest_Discrete_Continuous( self,
                                        dataread,
                                        class_num,
                                        best_num,neuron_scope_num,
                                        unstable_repeat_num,
                                        type,
                                        interval,
                                        iscontinous_data_test,
                                        compareType
                                       ):
            

            # type = 0 use scope_num type = 1 use unstable_repeat_num
            
            all_train_score_W0_n =[]
            all_train_score_W_combine_n =[]
            test_score_W0_n = []
            test_score_W_combine_n = []


            all_train_score_W0_a =[]
            all_train_score_W_combine_a =[]
            test_score_W0_a = []
            test_score_W_combine_a = []

            train_continuous_score_W0_n =[]
            train_continuous_score_W_continuous_n =[]

            test_continuous_score_W0_n = []
            test_continuous_score_W_continuous_n = []

            train_discrete_score_W0_n =[]
            train_discrete_score_W_discrete_n =[]
            
            test_discrete_score_W0_n = []
            test_discrete_score_W_discrete_n = []

            
            train_continuous_score_W0_a =[]
            train_continuous_score_W_continuous_a =[]

            test_continuous_score_W0_a = []
            test_continuous_score_W_continuous_a = []

            train_discrete_score_W0_a =[]
            train_discrete_score_W_discrete_a =[]
            
            test_discrete_score_W0_a = []
            test_discrete_score_W_discrete_a = []

            plot_unit = [class_num]
            if type == 0:
                y = class_num
                while y <= neuron_scope_num:
                    print("neuron unit number: {}".format(y))           
                    self.InitializedExperimentDataList(
                                        dataread,
                                        y,
                                        all_train_score_W0_n,
                                        all_train_score_W_combine_n,
                                        test_score_W0_n,
                                        test_score_W_combine_n,
                                        all_train_score_W0_a,
                                        all_train_score_W_combine_a,
                                        test_score_W0_a,
                                        test_score_W_combine_a,
                                        train_continuous_score_W0_n,
                                        train_continuous_score_W_continuous_n,
                                        test_continuous_score_W0_n,
                                        test_continuous_score_W_continuous_n,
                                        train_continuous_score_W0_a,
                                        train_continuous_score_W_continuous_a,
                                        test_continuous_score_W0_a,
                                        test_continuous_score_W_continuous_a,
                                        train_discrete_score_W0_n,
                                        train_discrete_score_W_discrete_n,
                                        test_discrete_score_W0_n,
                                        test_discrete_score_W_discrete_n,
                                        train_discrete_score_W0_a,
                                        train_discrete_score_W_discrete_a,
                                        test_discrete_score_W0_a,
                                        test_discrete_score_W_discrete_a,
                                        iscontinous_data_test,
                                        compareType
                                        )        
                    y =y + interval
                    if(y<= neuron_scope_num):
                        plot_unit.append(y)
            
            
            if type == 1:
                y = 1
                while y <= unstable_repeat_num:
                    self.InitializedExperimentDataList(
                                        dataread,
                                        best_num,
                                        all_train_score_W0_n,
                                        all_train_score_W_combine_n,
                                        test_score_W0_n,
                                        test_score_W_combine_n,
                                        all_train_score_W0_a,
                                        all_train_score_W_combine_a,
                                        test_score_W0_a,
                                        test_score_W_combine_a,
                                        train_continuous_score_W0_n,
                                        train_continuous_score_W_continuous_n,
                                        test_continuous_score_W0_n,
                                        test_continuous_score_W_continuous_n,
                                        train_continuous_score_W0_a,
                                        train_continuous_score_W_continuous_a,
                                        test_continuous_score_W0_a,
                                        test_continuous_score_W_continuous_a,
                                        train_discrete_score_W0_n,
                                        train_discrete_score_W_discrete_n,
                                        test_discrete_score_W0_n,
                                        test_discrete_score_W_discrete_n,
                                        train_discrete_score_W0_a,
                                        train_discrete_score_W_discrete_a,
                                        test_discrete_score_W0_a,
                                        test_discrete_score_W_discrete_a,
                                        iscontinous_data_test,
                                        compareType
                                        )          
                    y =y+1
                    if(y<= unstable_repeat_num):
                        plot_unit.append(y)
                
                        
            figure, axis = plt.subplots(1, 2,figsize =(12, 5))
            axis[0].set_title("NMI Score")               
            axis[1].set_title("ARI Score")

            if type == 0:
                axis[0].set_xlabel('Neuron number')
            if type == 1:
                axis[0].set_xlabel('Repeat number')
        

            if iscontinous_data_test == True:
                 
              
                #continous data    
                axis[0].plot(plot_unit,train_continuous_score_W0_n,'r',label ='train_con_W0')
                axis[0].plot(plot_unit,train_continuous_score_W_continuous_n,'c',label ='train_con_W_con')
                axis[0].plot(plot_unit,test_continuous_score_W0_n,'y',label ='test_con_W0')
                axis[0].plot(plot_unit,test_continuous_score_W_continuous_n,'b',label ='test_con_W_con')
                axis[0].legend(loc='best')
            else:  
                # discrete data
                axis[0].plot(plot_unit,train_discrete_score_W0_n,'r',label ='train_dis_W0')
                axis[0].plot(plot_unit,train_discrete_score_W_discrete_n,'c',label ='train_dis_W_dis')
                axis[0].plot(plot_unit,test_discrete_score_W0_n,'y',label ='test_dis_W0')
                axis[0].plot(plot_unit,test_discrete_score_W_discrete_n,'b',label ='test_dis_W_dis')
                axis[0].legend(loc='best')

            if type == 0:
                axis[1].set_xlabel('Neuron number')
            if type == 1:
                axis[1].set_xlabel('Repeat number')

            if iscontinous_data_test == True:

          
                axis[1].plot(plot_unit,train_continuous_score_W0_a,'r',label ='train_con_W0')
                axis[1].plot(plot_unit,train_continuous_score_W_continuous_a,'c',label ='train_con_W_con')
                axis[1].plot(plot_unit,test_continuous_score_W0_a,'y',label ='test_con_W0')
                axis[1].plot(plot_unit,test_continuous_score_W_continuous_a,'b',label ='test_con_W_con')
                axis[1].legend(loc='best')

            else:
                axis[1].plot(plot_unit,train_discrete_score_W0_a,'r',label ='train_dis_W0')
                axis[1].plot(plot_unit,train_discrete_score_W_discrete_a,'c',label ='train_dis_W_dis')
                axis[1].plot(plot_unit,test_discrete_score_W0_a,'y',label ='test_dis_W0')
                axis[1].plot(plot_unit,test_discrete_score_W_discrete_a,'b',label ='test_dis_W_dis')
                axis[1].legend(loc='best')
            
            plt.show()
            
            if iscontinous_data_test == True:
           # print("New Neuron Number : {}".format (int(np.mean(new_neuron_num))))
                                                    
                      
                df1_n = pd.DataFrame(train_continuous_score_W0_n, columns = ['train_con_W0'])
                df2_n = pd.DataFrame(train_continuous_score_W_continuous_n, columns = ['train_con_W_con'])

                df3_n = pd.DataFrame(test_continuous_score_W0_n, columns = ['test_con_W0'])
                df4_n = pd.DataFrame(test_continuous_score_W_continuous_n, columns = ['test_con_W_con'])

                summary, results = rp.ttest(group1= df1_n['train_con_W0'], group1_name= "train_con_W0",
                                            group2= df2_n['train_con_W_con'], group2_name= "train_con_W_con")
            
            else:     

                df1_n = pd.DataFrame(train_discrete_score_W0_n, columns = ['train_dis_W0'])
                df2_n = pd.DataFrame(train_discrete_score_W_discrete_n, columns = ['train_dis_W_dis'])

                df3_n = pd.DataFrame(test_discrete_score_W0_n, columns = ['test_dis_W0'])
                df4_n = pd.DataFrame(test_discrete_score_W_discrete_n, columns = ['test_dis_W_dis'])

                summary, results = rp.ttest(group1= df1_n['train_dis_W0'], group1_name= "train_dis_W0",
                                            group2= df2_n['train_dis_W_dis'], group2_name= "train_dis_W_dis")
            
            print("NMI T-Test")
            print(summary)
            print(results)

            if iscontinous_data_test == True:
                stats.ttest_ind(train_continuous_score_W0_n, train_continuous_score_W_continuous_n,alternative = 'less')

                summary1, results1 = rp.ttest(group1= df3_n['test_con_W0'], group1_name= "test_con_W0",
                                            group2= df4_n['test_con_W_con'], group2_name= "test_con_W_con")
        
                print(summary1)
                print(results1)

                stats.ttest_ind(test_continuous_score_W0_n, test_continuous_score_W_continuous_n,alternative = 'less')
            
            else:
                stats.ttest_ind(train_discrete_score_W0_n, train_discrete_score_W_discrete_n,alternative = 'less')

                summary1, results1 = rp.ttest(group1= df3_n['test_dis_W0'], group1_name= "test_dis_W0",
                                            group2= df4_n['test_dis_W_dis'], group2_name= "test_dis_W_dis")
        
                print(summary1)
                print(results1)

                stats.ttest_ind(test_continuous_score_W0_n, test_continuous_score_W_continuous_n,alternative = 'less')


            
            if iscontinous_data_test == True:
                df1_a = pd.DataFrame(train_continuous_score_W0_a, columns = ['train_con_W0'])
                df2_a = pd.DataFrame(train_continuous_score_W_continuous_a, columns = ['train_con_W_con'])

                df3_a = pd.DataFrame(test_continuous_score_W0_a, columns = ['test_con_W0'])
                df4_a = pd.DataFrame(test_continuous_score_W_continuous_a, columns = ['test_con_W_con'])

                summary, results = rp.ttest(group1= df1_a['train_con_W0'], group1_name= "train_con_W0",
                                            group2= df2_a['train_con_W_con'], group2_name= "train_con_W_con")
            
            else:
                df1_a = pd.DataFrame(train_discrete_score_W0_a, columns = ['train_dis_W0'])
                df2_a = pd.DataFrame(train_discrete_score_W_discrete_a, columns = ['train_dis_W_dis'])

                df3_a = pd.DataFrame(test_discrete_score_W0_a, columns = ['test_dis_W0'])
                df4_a = pd.DataFrame(test_discrete_score_W_discrete_a, columns = ['test_dis_W_dis'])

                summary, results = rp.ttest(group1= df1_a['train_dis_W0'], group1_name= "train_dis_W0",
                                            group2= df2_a['train_dis_W_dis'], group2_name= "train_dis_W_dis")
            
            
            
            print("ARI T-Test")
            print(summary)
            print(results)


            if iscontinous_data_test == True:
            
                stats.ttest_ind(train_continuous_score_W0_a, train_continuous_score_W_continuous_a,alternative = 'less')

                summary1, results1 = rp.ttest(group1= df3_a['test_con_W0'], group1_name= "test_con_W0",
                                            group2= df4_a['test_con_W_con'], group2_name= "test_con_W_con")
                
                print(summary1)
                print(results1)

                
            else:

                stats.ttest_ind(train_discrete_score_W0_a, train_discrete_score_W_discrete_a,alternative = 'less')

                summary1, results1 = rp.ttest(group1= df3_a['test_dis_W0'], group1_name= "test_dis_W0",
                                            group2= df4_a['test_dis_W_dis'], group2_name= "test_dis_W_dis")
                
                print(summary1)
                print(results1)

              