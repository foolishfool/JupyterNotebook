#!/usr/bin/env python
# coding: utf-8

from pickle import TRUE
import matplotlib.pyplot as plt
from typing import List
import newSom
import CDSOM
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
import researchpy as rp



class Experiment():

        def __init__(self):
         return

        def __defaults__(self):
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
                                        test_score_W0_p,
                                        test_score_W0_n,
                                        atest_score_W0_a,
                                        test_score_W_combine_p,
                                        test_score_W_combine_n,
                                        test_score_W_combine_a,
                                        features_choosen,
                                        feature_combination_number
                                        ):

            m, n = self.topology_som(y)
       
       
            som = newSom.SOM(m , n, dim=dataread.continuous_feature_num + dataread.discrete_feature_num)  
            if  dataread.continuous_feature_num != 0:
             som_continusous = newSom.SOM(m , n, dim= dataread.continuous_feature_num)  
            else:
                som_continusous =  newSom.SOM(m , n, dim= dataread.discrete_feature_num) 
            
            if  dataread.discrete_feature_num != 0:
                som_discrete_original = newSom.SOM(m , n, dim= dataread.discrete_feature_num) 
                som_total_discrete = newSom.SOM(m , n, dim= dataread.discrete_feature_num) 
            else: 
                # som_discrete_original will be used in CDSOM initialize, so it cannot be null, although won't be used
                som_discrete_original = newSom.SOM(m , n, dim= dataread.continuous_feature_num) 
                som_total_discrete = som_continusous  



            optimize_W = CDSOM.CDSOM(som,
                     som_continusous,
                     som_discrete_original,
                     som_total_discrete,
                     dataread.data_train,
                     dataread.data_train_continuous,
                     dataread.data_train_discrete,
                     dataread.data_train_discrete_normalized,
                     dataread.data_test,
                     dataread.data_test_continuous,
                     dataread.data_test_discrete,
                     dataread.data_test_discrete_normalized,
                     dataread.label_train,
                     dataread.label_test)                        


            optimize_W.do_DOSOM(features_choosen,feature_combination_number)

              


            test_score_W0_p.append(optimize_W.test_score_W0_p)
            test_score_W0_n.append(optimize_W.test_score_W0_n)
            atest_score_W0_a.append(optimize_W.test_score_W0_a)
          #  all_f1_score_original.append(optimize_W.f1_score_original)
            #all_log_loss_score_original.append(optimize_W.log_loss_original)

            test_score_W_combine_p.append(optimize_W.test_score_W_combine_p)
            test_score_W_combine_n.append(optimize_W.test_score_W_combine_n)
            test_score_W_combine_a.append(optimize_W.test_score_W_combine_a)
            #all_log_loss_score_sog.append(optimize_W.log_loss_sog)


        def UTtest_Discrete( self, dataread, class_num, scope_num, interval,  features_choosen ,feature_combination_number):
            

            # type = 0 use scope_num type = 1 use unstable_repeat_num
            all_test_score_W0_p =[]
            all_test_score_W0_n =[]
            all_test_score_W0_a =[]


            all_test_score_W_combine_p =[]
            all_test_score_W_combine_n =[]
            all_test_score_W_combine_a =[]

            plot_unit = [class_num]

            y = class_num
            while y <= scope_num:
                print("Neuron number: {}".format(y))           
                self.InitializedExperimentDataList(
                                        dataread,
                                        y,
                                        all_test_score_W0_p,
                                        all_test_score_W0_n,
                                        all_test_score_W0_a,
                                        all_test_score_W_combine_p ,
                                        all_test_score_W_combine_n ,
                                        all_test_score_W_combine_a ,
                                        features_choosen,feature_combination_number                                  
                                        )        
                y =y + interval
                if(y<= scope_num):
                    plot_unit.append(y)
            
            
               
            figure, axis = plt.subplots(1, 3,figsize =(12, 5))
            axis[0].set_title("Purity Score")               
            axis[1].set_title("NMI Score")
            axis[2].set_title("ARI Score") 

   
            print(f"all_test_score_W0_p mean {np.mean(all_test_score_W0_p)}")
            print(f"all_test_score_W0_n mean {np.mean(all_test_score_W0_n)}")
            print(f"all_test_score_W0_a mean {np.mean(all_test_score_W0_a)}")

            print(f"all_test_score_W_combine_p mean {np.mean(all_test_score_W_combine_p)}")
            print(f"all_test_score_W_combine_n mean {np.mean(all_test_score_W_combine_n)}")
            print(f"all_test_score_W_combine_a mean {np.mean(all_test_score_W_combine_a)}")




            axis[0].set_xlabel('Neuron number')
            axis[1].set_xlabel('Neuron number')
            axis[2].set_xlabel('Neuron number')


                   
            axis[0].plot(plot_unit,all_test_score_W0_p,'r',label ='all_test_score_W0_p')
            axis[0].plot(plot_unit,all_test_score_W_combine_p,'b',label ='all_test_score_W_combine_p')
            axis[0].legend(loc='best')

            axis[1].plot(plot_unit,all_test_score_W0_n,'r',label ='all_test_score_W0_n')
            axis[1].plot(plot_unit,all_test_score_W_combine_n,'b',label ='all_test_score_W_combine_n')
            axis[1].legend(loc='best')


            axis[2].plot(plot_unit,all_test_score_W0_a,'r',label ='all_test_score_W0_a')
            axis[2].plot(plot_unit,all_test_score_W_combine_a,'b',label ='all_test_score_W_combine_a')
            axis[2].legend(loc='best')





            plt.show()
            

                                                    
                      
            df1 = pd.DataFrame(all_test_score_W0_p, columns = ['all_test_score_W0_p'])
            df2 = pd.DataFrame(all_test_score_W_combine_p, columns = ['all_test_score_W_combine_p'])

               
            print("Purity T-Test")
                
            summary, results = rp.ttest(group1= df1['all_test_score_W0_p'], group1_name= "all_test_score_W0_p",
                                            group2= df2['all_test_score_W_combine_p'], group2_name= "all_test_score_W_combine_p")
            print(summary)
            print(results)


            df1 = pd.DataFrame(all_test_score_W0_n, columns = ['all_test_score_W0_n'])
            df2 = pd.DataFrame(all_test_score_W_combine_n, columns = ['all_test_score_W_combine_n'])

               
            print("NMI T-Test")
                
            summary, results = rp.ttest(group1= df1['all_test_score_W0_n'], group1_name= "all_test_score_W0_n",
                                            group2= df2['all_test_score_W_combine_n'], group2_name= "all_test_score_W_combine_n")
            print(summary)
            print(results)

            df1 = pd.DataFrame(all_test_score_W0_a, columns = ['all_test_score_W0_a'])
            df2 = pd.DataFrame(all_test_score_W_combine_a, columns = ['all_test_score_W_combine_a'])

               
            print("ARI Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_test_score_W0_a'], group1_name= "all_test_score_W0_a",
                                            group2= df2['all_test_score_W_combine_a'], group2_name= "all_test_score_W_combine_a")
            print(summary)
            print(results)












        