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
                                        all_accuracy_score_original,
                                        all_recall_score_original,
                                        all_precision_score_original,
                                        all_f1_score_original,
                                        all_accuracy_score_sog,
                                        all_recall_score_sog,
                                        all_precision_score_sog,
                                        all_f1_score_sog,
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

              


            all_accuracy_score_original.append(optimize_W.accuracy_score_original)
            all_recall_score_original.append(optimize_W.recall_score_original)
            all_precision_score_original.append(optimize_W.precision_score_original)
            all_f1_score_original.append(optimize_W.f1_score_original)
            #all_log_loss_score_original.append(optimize_W.log_loss_original)

            all_accuracy_score_sog.append(optimize_W.accuracy_score_sog)
            all_recall_score_sog.append(optimize_W.recall_score_sog)
            all_precision_score_sog.append(optimize_W.precision_score_sog)
            all_f1_score_sog.append(optimize_W.f1_score_sog)
            #all_log_loss_score_sog.append(optimize_W.log_loss_sog)


        def UTtest_Discrete( self, dataread, class_num, unstable_repeat_num, interval,  features_choosen ,feature_combination_number):
            

            # type = 0 use scope_num type = 1 use unstable_repeat_num
            all_accuracy_score_original =[]
            all_recall_score_original =[]
            all_precision_score_original =[]
            all_f1_score_original =[]
            all_log_loss_score_original =[]

            all_accuracy_score_sog =[]
            all_recall_score_sog =[]
            all_precision_score_sog =[]
            all_f1_score_sog =[]
            all_log_loss_sog =[]
            plot_unit = [class_num]

            y = 1
            while y <= unstable_repeat_num:
                print("Experiment number: {}".format(y))           
                self.InitializedExperimentDataList(
                                        dataread,
                                        class_num,
                                        all_accuracy_score_original,
                                        all_recall_score_original,
                                        all_precision_score_original,
                                        all_f1_score_original,
                                        all_accuracy_score_sog ,
                                        all_recall_score_sog ,
                                        all_precision_score_sog ,
                                        all_f1_score_sog,
                                        features_choosen,feature_combination_number                                  
                                        )        
                y =y + 1
                if(y<= unstable_repeat_num):
                    plot_unit.append(y)
            
            
               
            figure, axis = plt.subplots(1, 4,figsize =(12, 5))
            axis[0].set_title("Accuracy Score")               
            axis[1].set_title("Recall Score")
            axis[2].set_title("Precision Score") 
            axis[3].set_title("F1 Score")
          #  axis[4].set_title("Log Loss Score")

           # print(f"test_discrete_score_W0_p !!!!!! {test_discrete_score_W0_p}")
            print(f"all_accuracy_score_original mean {np.mean(all_accuracy_score_original)}")
            print(f"all_recall_score_original mean {np.mean(all_recall_score_original)}")
            print(f"all_precision_score_original mean {np.mean(all_precision_score_original)}")
            print(f"all_f1_score_original mean {np.mean(all_f1_score_original)}")
           # print(f"all_log_loss_score_original mean {np.mean(all_log_loss_score_original)}")
            print(f"all_accuracy_score_sog mean {np.mean(all_accuracy_score_sog)}")
            print(f"all_recall_score_sog mean {np.mean(all_recall_score_sog)}")
            print(f"all_precision_score_sog mean {np.mean(all_precision_score_sog)}")
            print(f"all_f1_score_sog mean {np.mean(all_f1_score_sog)}")
            #print(f"all_log_loss_sog mean {np.mean(all_log_loss_sog)}")



            axis[0].set_xlabel('Experiment Time')
            axis[1].set_xlabel('Experiment Time')
            axis[2].set_xlabel('Experiment Time')

            axis[3].set_xlabel('Experiment Time')
            #axis[4].set_xlabel('Neuron number')

                   
            axis[0].plot(plot_unit,all_accuracy_score_original,'r',label ='all_accuracy_score_original')
            axis[0].plot(plot_unit,all_accuracy_score_sog,'b',label ='all_accuracy_score_sog')
            axis[0].legend(loc='best')

            axis[1].plot(plot_unit,all_recall_score_original,'r',label ='all_recall_score_original')
            axis[1].plot(plot_unit,all_recall_score_sog,'b',label ='all_recall_score_sog')
            axis[1].legend(loc='best')


            axis[2].plot(plot_unit,all_precision_score_original,'r',label ='all_precision_score_original')
            axis[2].plot(plot_unit,all_precision_score_sog,'b',label ='all_precision_score_sog')
            axis[2].legend(loc='best')


            axis[3].plot(plot_unit,all_f1_score_original,'r',label ='all_f1_score_original')
            axis[3].plot(plot_unit,all_f1_score_sog,'b',label ='all_f1_score_sog')
            axis[3].legend(loc='best')

            
          #  axis[4].plot(plot_unit,all_log_loss_score_original,'r',label ='all_log_loss_score_original')
           # axis[4].plot(plot_unit,all_log_loss_sog,'b',label ='all_log_loss_sog')
          #  axis[4].legend(loc='best')



            plt.show()
            

                                                    
                      
            df1 = pd.DataFrame(all_accuracy_score_original, columns = ['all_accuracy_score_original'])
            df2 = pd.DataFrame(all_accuracy_score_sog, columns = ['all_accuracy_score_sog'])

               
            print("Accuracy Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_accuracy_score_original'], group1_name= "all_accuracy_score_original",
                                            group2= df2['all_accuracy_score_sog'], group2_name= "all_accuracy_score_sog")
            print(summary)
            print(results)


            df1 = pd.DataFrame(all_recall_score_original, columns = ['all_recall_score_original'])
            df2 = pd.DataFrame(all_recall_score_sog, columns = ['all_recall_score_sog'])

               
            print("Recall Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_recall_score_original'], group1_name= "all_recall_score_original",
                                            group2= df2['all_recall_score_sog'], group2_name= "all_recall_score_sog")
            print(summary)
            print(results)

            df1 = pd.DataFrame(all_precision_score_original, columns = ['all_precision_score_original'])
            df2 = pd.DataFrame(all_precision_score_sog, columns = ['all_precision_score_sog'])

               
            print("Precision Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_precision_score_original'], group1_name= "all_precision_score_original",
                                            group2= df2['all_precision_score_sog'], group2_name= "all_precision_score_sog")
            print(summary)
            print(results)

            df1 = pd.DataFrame(all_f1_score_original, columns = ['all_f1_score_original'])
            df2 = pd.DataFrame(all_f1_score_sog, columns = ['all_f1_score_sog'])

               
            print("F1 Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_f1_score_original'], group1_name= "all_f1_score_original",
                                            group2= df2['all_f1_score_sog'], group2_name= "all_f1_score_sog")
            print(summary)
            print(results)









        