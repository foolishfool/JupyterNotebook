#!/usr/bin/env python
# coding: utf-8

from pickle import TRUE
import matplotlib.pyplot as plt
from typing import List

import FCG
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


        def InitializedExperimentDataList(self,
                                        dataread,
                                        test_score_baseline_accuracy,
                                        test_score_baseline_recall,
                                        test_score_baseline_precision,
                                        test_score_baseline_f1,
                                        test_score_fcg_accuracy,
                                        test_score_fcg_recall,
                                        test_score_fcg_precision,
                                        test_score_fcg_f1,
                                        topology_ratio
                                      ):


            Comparision = FCG.FCG(
                     dataread.data_train,
                     dataread.data_train_continuous,
                     dataread.data_train_discrete,
                     dataread.data_test,
                     dataread.data_test_continuous,
                     dataread.data_test_discrete,
                     dataread.label_train,
                     dataread.label_test,topology_ratio)                        


            Comparision.do_FCG(topology_ratio)

              


            test_score_baseline_accuracy.append(Comparision.accuracy_score_baseline)
            test_score_baseline_recall.append(Comparision.recall_score_baseline)
            test_score_baseline_precision.append(Comparision.precision_score_baseline)
            test_score_baseline_f1.append(Comparision.f1_score_baseline)

            test_score_fcg_accuracy.append(Comparision.accuracy_score_fcg)
            test_score_fcg_recall.append(Comparision.recall_score_fcg)
            test_score_fcg_precision.append(Comparision.precision_score_fcg)
            test_score_fcg_f1.append(Comparision.f1_score_fcg)


        def Ttest( self, dataread, scope_num, topology_ratio):
            
            all_test_score_baseline_accuracy =[]
            all_test_score_baseline_recall =[]
            all_test_score_baseline_precision =[]
            all_test_score_baseline_f1 =[]

            
            all_test_score_fcg_accuracy =[]
            all_test_score_fcg_recall =[]
            all_test_score_fcg_precision =[]
            all_test_score_fcg_f1 =[]

            plot_unit = [1]

            y = 1
            while y <= scope_num:
                print("Experiment number: {}".format(y))           
                self.InitializedExperimentDataList(
                                        dataread,
                                        all_test_score_baseline_accuracy,
                                        all_test_score_baseline_recall,
                                        all_test_score_baseline_precision,
                                        all_test_score_baseline_f1,
                                        all_test_score_fcg_accuracy,
                                        all_test_score_fcg_recall,
                                        all_test_score_fcg_precision,   
                                        all_test_score_fcg_f1,
                                        topology_ratio                      
                                        )        
                y =y + 1
                if(y<= scope_num):
                    plot_unit.append(y)
            
            
               
            figure, axis = plt.subplots(1, 4,figsize =(12, 5))
            axis[0].set_title("Accuracy Score")               
            axis[1].set_title("Recall Score")
            axis[2].set_title("Precision Score") 
            axis[3].set_title("F1 Score")

   
          

            print(f"all_accuracy_score_baseline mean {np.mean(all_test_score_baseline_accuracy)}")
            print(f"all_recall_score_baseline mean {np.mean(all_test_score_baseline_recall)}")
            print(f"all_precision_score_baseline mean {np.mean(all_test_score_baseline_precision)}")
            print(f"all_f1_score_baseline mean {np.mean(all_test_score_baseline_f1)}")

            print(f"all_accuracy_score_fcg mean {np.mean(all_test_score_fcg_accuracy)}")
            print(f"all_recall_score_fcg mean {np.mean(all_test_score_fcg_recall)}")
            print(f"all_precision_score_fcg mean {np.mean(all_test_score_fcg_precision)}")
            print(f"all_f1_score_fcg mean {np.mean(all_test_score_fcg_f1)}")




            axis[0].set_xlabel('Experiment number')
            axis[1].set_xlabel('Experiment number')
            axis[2].set_xlabel('Experiment number')

            axis[3].set_xlabel('Experiment number')
            #axis[4].set_xlabel('Neuron number')

            print(f"len plot_unit {len(plot_unit)}  len (all_test_score_baseline_accuracy) {len(all_test_score_baseline_accuracy)}")
            axis[0].plot(plot_unit,all_test_score_baseline_accuracy,'r',label ='all_accuracy_score_baseline')
            axis[0].plot(plot_unit,all_test_score_fcg_accuracy,'b',label ='all_accuracy_score_fcg')
            axis[0].legend(loc='best')

            axis[1].plot(plot_unit,all_test_score_baseline_recall,'r',label ='all_recall_score_baseline')
            axis[1].plot(plot_unit,all_test_score_fcg_recall,'b',label ='all_recall_score_fcg')
            axis[1].legend(loc='best')


            axis[2].plot(plot_unit,all_test_score_baseline_precision,'r',label ='all_precision_score_baseline')
            axis[2].plot(plot_unit,all_test_score_fcg_precision,'b',label ='all_precision_score_fcg')
            axis[2].legend(loc='best')


            axis[3].plot(plot_unit,all_test_score_baseline_f1,'r',label ='all_f1_score_baseline')
            axis[3].plot(plot_unit,all_test_score_fcg_f1,'b',label ='all_f1_score_fcg')
            axis[3].legend(loc='best')

            
     


            plt.show()
            

                                                    
                      
            df1 = pd.DataFrame(all_test_score_baseline_accuracy, columns = ['all_accuracy_score_baseline'])
            df2 = pd.DataFrame(all_test_score_fcg_accuracy, columns = ['all_accuracy_score_fcg'])

               
            print("Accuracy Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_accuracy_score_baseline'], group1_name= "all_accuracy_score_baseline",
                                            group2= df2['all_accuracy_score_fcg'], group2_name= "all_accuracy_score_fcg")
            print(summary)
            print(results)


            df1 = pd.DataFrame(all_test_score_baseline_recall, columns = ['all_recall_score_baseline'])
            df2 = pd.DataFrame(all_test_score_fcg_recall, columns = ['all_recall_score_fcg'])

               
            print("Recall Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_recall_score_baseline'], group1_name= "all_recall_score_baseline",
                                            group2= df2['all_recall_score_fcg'], group2_name= "all_recall_score_fcg")
            print(summary)
            print(results)

            df1 = pd.DataFrame(all_test_score_baseline_precision, columns = ['all_precision_score_baseline'])
            df2 = pd.DataFrame(all_test_score_fcg_precision, columns = ['all_precision_score_fcg'])

               
            print("Precision Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_precision_score_baseline'], group1_name= "all_precision_score_baseline",
                                            group2= df2['all_precision_score_fcg'], group2_name= "all_precision_score_fcg")
            print(summary)
            print(results)

            df1 = pd.DataFrame(all_test_score_baseline_f1, columns = ['all_f1_score_baseline'])
            df2 = pd.DataFrame(all_test_score_fcg_f1, columns = ['all_f1_score_fcg'])

               
            print("F1 Score T-Test")
                
            summary, results = rp.ttest(group1= df1['all_f1_score_baseline'], group1_name= "all_f1_score_baseline",
                                            group2= df2['all_f1_score_fcg'], group2_name= "all_f1_score_fcg")
            print(summary)
            print(results)

