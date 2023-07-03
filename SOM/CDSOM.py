"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
find intra communiyt in each neuron memberships and do the whole mapping and retest 
"""
from asyncio.windows_events import NULL
from distutils.errors import DistutilsArgError
from enum import Flag
#from curses.ascii import NULL
from importlib import resources
from pickle import TRUE
from telnetlib import PRAGMA_HEARTBEAT
from traceback import format_exc
import turtle
from zlib import DEF_BUF_SIZE
from sklearn import metrics
from scipy.spatial import distance
from scipy import spatial
import numpy as np
import statistics
from scipy.stats import norm
from numpy import array
import random
import copy
import math
import operator
import collections
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import Counter
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from kneed import KneeLocator

# unsupervised continus and discrete som
class CDSOM():
    """
     Unsupervised SOM with continusous data, discrete data will be removed
    """
    def __init__(self, som,som_continuous,som_discrete, 
                            data_train_all, 
                            data_train_continuous,
                            data_train_discrete,
                            data_test_all,
                            data_test_continuous,
                            data_test_discrete,
                            label_train_all,                       
                            label_test_all,
                            row,column):
        """
        Parameters
        ----------
    
        som: original som model
        data_train : training data
        data_test : test data
        label_train: predicted labels by som for training data
        label_test : predicted labels by som for test data
        class_num: external validation class number
     
        """
       
        self.som = som  
        self.som_continuous = som_continuous 
        self.som_discrete = som_discrete 
        self.row = row
        self.column = column
        # initial cluster numbers in TDSM_SOM, which is the neuron number in som
        self.predicted_classNum= int(som.m*som.n)
        self.community_distance = 1

        self.data_train_all = data_train_all
        self.data_test_all = data_test_all

        self.data_train_continuous = data_train_continuous
        self.data_train_discrete = data_train_discrete
        
        self.data_test_continuous = data_test_continuous
        self.data_test_discrete = data_test_discrete


        self.train_label_all = label_train_all
        self.train_label_all = self.train_label_all.astype(int)
        self.test_label_all = label_test_all
        self.test_label_all = self.test_label_all.astype(int)


        # continuous features generated datas
        self.all_split_datas_continuous = []
        # continuous features generated datas indexes   
        self.all_split_datas_indexes_continuous = []

        self.outliner_clusters = []

    def purity_score(self,scorename, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        #print(" purity_score y_true{}  y_pred {} ".format(y_true,y_pred))
        if(scorename == "all_train_score_W0" ):
            self.all_train_score_W0_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
        if(scorename == "all_train_score_W_Combine" ):
            self.all_train_score_W_Combine_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
        if(scorename == "test_score_W0" ):
            self.test_score_W0_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
          #  print("test_score_W0_p{}".format(self.test_score_W0_p ))
        if(scorename == "test_score_W_Combine" ):
            self.test_score_W_Combine_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


    def nmiScore(self,scorename, y_true, y_pred):
        #print(" nmi y_true{}  y_pred {} ".format(y_true,y_pred))
        if(scorename == "all_train_score_W0" ):
            self.all_train_score_W0_n = normalized_mutual_info_score(y_true,y_pred)
            print("all_train_score_W0_n {}".format(self.all_train_score_W0_n ))  
        if(scorename == "all_train_score_W_Combine" ):          
            self.all_train_score_W_combine_n = normalized_mutual_info_score(y_true,y_pred)
            # if  self.all_train_score_W_Combined_n >= self.all_train_score_W0_n :
            #     print("all_train_score_W_Combined increased in nmi")
            print("all_train_score_W_combine_n {}".format(self.all_train_score_W_combine_n ))  
        if(scorename == "test_score_W0" ):
            self.test_score_W0_n = normalized_mutual_info_score(y_true,y_pred)
            print("test_score_W0_n {}".format(self.test_score_W0_n ))  
        if(scorename == "test_score_W_combine" ):
            self.test_score_W_combine_n = normalized_mutual_info_score(y_true,y_pred)
            print("test_score_W_combine_n {}".format(self.test_score_W_combine_n ))  


        if(scorename == "train_continuous_score_W0" ):
            self.train_continuous_score_W0_n = normalized_mutual_info_score(y_true,y_pred)
            print("train_continuous_score_W0_n {}".format(self.train_continuous_score_W0_n ))  

        if(scorename == "train_continuous_score_W_continuous" ):
            self.train_continuous_score_W_continuous_n = normalized_mutual_info_score(y_true,y_pred)
            print("train_continuous_score_W_continuous_n {}".format(self.train_continuous_score_W_continuous_n ))  

        if(scorename == "test_continuous_score_W0" ):
            #print("test W0 y_true {}  y_pred{}".format(y_true ,y_pred ))  
            self.test_continuous_score_W0_n = normalized_mutual_info_score(y_true,y_pred)
            print("test_continuous_score_W0_n {}".format(self.test_continuous_score_W0_n ))  
    
        if(scorename == "test_continuous_score_W_continuous" ):
           # print("test W continuous y_true {}  y_pred{}".format(y_true ,y_pred ))  
            self.test_continuous_score_W_continuous_n = normalized_mutual_info_score(y_true,y_pred)
            print("test_continuous_score_W_continuous_n {}".format(self.test_continuous_score_W_continuous_n ))  
    
    def ariScore(self,scorename, y_true, y_pred):
       # print(" ariScore y_true{}  y_pred {} ".format(y_true,y_pred))
        if(scorename == "all_train_score_W0" ):
            self.all_train_score_W0_a = adjusted_rand_score(y_true,y_pred)
            print("all_train_score_W0_a {}".format(self.all_train_score_W0_a ))  
        if(scorename == "all_train_score_W_combine" ):
            self.all_train_score_W_combine_a = adjusted_rand_score(y_true,y_pred)
            # if  self.all_train_score_W_Combined_a >= self.all_train_score_W0_a :
            #     print("all_train_score_W_Combined increased in ari")
            # else: print(-1)
            print("all_train_score_W_combine_a {}".format(self.all_train_score_W_combine_a ))  
        if(scorename == "test_score_W0" ):
            self.test_score_W0_a = adjusted_rand_score(y_true,y_pred)
            print("test_score_W0_a  {}".format(self.test_score_W0_a ))  
        if(scorename == "test_score_W_Combine" ):
            self.test_score_W_combine_a = adjusted_rand_score(y_true,y_pred)
            print("test_score_W_Combine_a {}".format(self.test_score_W_combine_a ))  

        if(scorename == "train_continuous_score_W0" ):
            self.train_continuous_score_W0_a = normalized_mutual_info_score(y_true,y_pred)
            print("train_continuous_score_W0_a {}".format(self.train_continuous_score_W0_a ))  

        if(scorename == "train_continuous_score_W_continuous" ):
            self.train_continuous_score_W_continuous_a = normalized_mutual_info_score(y_true,y_pred)
            print("train_continuous_score_W_continuous_a {}".format(self.train_continuous_score_W_continuous_a ))  

        if(scorename == "test_continuous_score_W0" ):
            self.test_continuous_score_W0_a = normalized_mutual_info_score(y_true,y_pred)
            print("test_continuous_score_W0_a {}".format(self.test_continuous_score_W0_a ))  
    
        if(scorename == "test_continuous_score_W_continuous" ):
            self.test_continuous_score_W_continuous_a = normalized_mutual_info_score(y_true,y_pred)
            print("test_continuous_score_W_continuous_a {}".format(self.test_continuous_score_W_continuous_a ))  
    
    def get_indices_and_data_in_predicted_clusters(self,class_num_predicted,predicted_label,data_set):
            
            """
            predicted_label = [1,1,2,3,1,1,2,1]
            idx start from 0 to n
            class_label index also start from 0 to n
            """

            clusters_indexes = []
            clusters_datas = []
            
            for i in range(0,class_num_predicted):
                newlist = []
                newdatalist = []
                for idx, y in enumerate(predicted_label): 
                    # is the cluster label
                    if(y == i):
                        x = idx
                        x = int(x)                      
                        newlist.append(x)  
                        newdatalist.append(data_set[x])                        
                clusters_indexes.append(newlist)
                clusters_datas.append(np.array(newdatalist))
            
            return clusters_indexes,clusters_datas

    # from data index to  [[2,35,34,3,23],[211,12,2,1]] get cluster index [[0,0,1,1] [0,0,1]]
    def get_mapped_class_in_clusters(self,clusters_indexes):
        mapped_class_in_clusters = []
        #print(f"clusters_indexes {clusters_indexes} ")
        #initialize mapped_clases_in_clusters
        for i in range(0, len(clusters_indexes)):
            mapped_class_in_clusters.append([])

        for j in range(0, len(clusters_indexes)):
            for item in clusters_indexes[j]:
                mapped_class_in_clusters[j].append(self.train_label_all[item])

        # mapped_clases_in_clusters = [[1,2,1,2,1,1],[2,2,2,2],[0,1,0]]
       # print(f" mapped_class_in_clusters {mapped_class_in_clusters} ")
        return mapped_class_in_clusters
          
    def getLabelMapping(self,predicted_class_label_in_each_cluster,Wtype  = 0):
        """
         predicted_class_label  = [[1,2,1,1],[3,3,3]]  the value in is the true value in class_label
         it means that predicted cluster 0 is 1 in class lable, cluster label 2 is 3 in class label
        """
        predicted_label_convert_to_class_label = []
       
        for item in predicted_class_label_in_each_cluster:
            if item != []:
                # the first item is for cluster0       
                # transfer to true class value based on indices in predict lables          
                predicted_label_convert_to_class_label.append(self.getMaxRepeatedElements(item))
            else:
                # -1 means there is no data in current neuron
                predicted_label_convert_to_class_label.append(-1)
        
        if Wtype == 0 :
            #print(f"predicted_class_label_in_each_cluster 0 {predicted_class_label_in_each_cluster} ")
            self.PLabel_to_Tlabel_Mapping_W_Original = predicted_label_convert_to_class_label
            #*********** remove null neurons in original null

            #self.removeNullNeuronsInW(predicted_label_convert_to_class_label,self.som_continuous.weights0)

        if Wtype == 1 :
            #print(f"predicted_class_label_in_each_cluster {predicted_class_label_in_each_cluster} ")
            self.PLabel_to_Tlabel_Mapping_W_Continous = predicted_label_convert_to_class_label

        if Wtype == 2 :
            self.PLabel_to_Tlabel_Mapping_W_Discrete = predicted_label_convert_to_class_label

    def getMaxRepeatedElements(self, list):
        #Count number of occurrences of each value in array of non-negative ints.
        counts = np.bincount(list)
        #Returns the indices of the maximum values along an axis.
        return np.argmax(counts)

    def convertPredictedLabelValue(self,predicted_cluster_labels, PLable_TLabel_Mapping):
        # PLabel_CLabel_Mapping the mapping of cluster label to class label
        # PLable_TLabel_Mapping size is the som.m*som.n* stop_split_num
        #print(f"predicted_cluster_labels {predicted_cluster_labels} PLable_TLabel_Mapping {PLable_TLabel_Mapping} ")
        for i in range(0,len(predicted_cluster_labels)):
            predicted_cluster_value =  predicted_cluster_labels[i]
            predicted_cluster_labels[i] = PLable_TLabel_Mapping[predicted_cluster_value]      

  
        return predicted_cluster_labels

    def removeNullNeuronsInW(self,predicted_cluster_index,weight):
        #print(f"weight1 {weight}")
        non_null_list = []
        for i in range(0,len(predicted_cluster_index)):
            if  predicted_cluster_index[i] != []:
                non_null_list.append(i) 
            else: print(i)
        #print(f"non_null_list {non_null_list}")
        weight = np.take(weight,non_null_list, axis=0)
        return weight
    def transferClusterLabelToClassLabel(self, mapping ,predicted_cluster_labels): 
        """
        map = self.PLabel_to_Tlabel_Mapping_W_Original self.PLabel_to_Tlabel_Mapping_W_continous
        """     
        predicted_class_labels =  self.convertPredictedLabelValue(predicted_cluster_labels,mapping)

        return predicted_class_labels


    
    #get a dictionary with nodes has decsending distance with cluster center
    def split_continuous_data(self, targetgroup_index,cluster_center):
        """
        targetgroup_index : the group (cluster) which will be split
        """
        sorted_data_dict = {}
        for idx in targetgroup_index:     
            distance = np.linalg.norm((self.data_train_continuous[idx] - cluster_center).astype(float))
            if distance >0:
                sorted_data_dict[idx] = distance
            #if distance == 0:
              #  print("zero distcance for data 1 idx {}".format(idx))       
        sorted_index_distance_dict = dict(sorted(sorted_data_dict.items(), key=operator.itemgetter(1),reverse=True))
        # return the sorted dictiony, key is index, value is distance
        return sorted_index_distance_dict



    def getfarthest_intra_node_index(self,sorted_dict):
        find_node = next(iter(sorted_dict))
        return find_node

    # get all nodes' distance to the cluster center
    def get_allnode_distance_to_center(self, target_node,  group_index, group_center):
        sorted_data_dict = {}
        distances = {}

        for idx in group_index:     
            distance = np.linalg.norm((self.data_train_continuous[idx] - target_node).astype(float))
            if distance >0 :
                sorted_data_dict[idx] =distance  

        sorted_dict = dict(sorted(sorted_data_dict.items(), key=operator.itemgetter(1),reverse=False))

        for key in sorted_dict:
            distance_intra = np.linalg.norm((self.data_train_continuous[key] - group_center).astype(float))
            distances[key] = distance_intra        
       
        return sorted_dict,distances

    # get all the inter node that has smaller distance to the target data, then target data to its cluster center 
    def get_intra_continuous_community_nodes(self,sorted_dict, intra_center):
        community_nodes = []
        community_nodes_keys = []

        for key in sorted_dict:  
            #**** cannot <= when == is itself, may cause one data one community       
            distance_intra = np.linalg.norm((self.data_train_continuous[key] - intra_center).astype(float))
            if sorted_dict[key] < distance_intra*self.community_distance:
                #print("sorted_dict[key {} key {} distance_intra{}".format(sorted_dict[key],key,distance_intra ))
                community_nodes.append(self.data_train_continuous[key])
                #print("key intra {}".format(key))
                community_nodes_keys.append(key)
        return community_nodes,community_nodes_keys


    def get_inter_continuous_community_nodes(self,sorted_dict,distances_inter):
        community_nodes = []
        community_nodes_keys = []
      
        for key in sorted_dict:
            if sorted_dict[key] < distances_inter[key]*self.community_distance:
                community_nodes.append(self.data_train_continuous[key])
                community_nodes_keys.append(key)
    
        return community_nodes,community_nodes_keys


    def _find_belonged_continuous_neuron(self,x,Y):
        """
        Find the index of the best matching unit for the input vector x.
        """  
        #initial distance

        Y = np.array(Y)
        firstindex = 0
        for i in range(len(Y)):
            if Y[i]!= []:
                firstindex = i
                break
                
        distance = math.dist(x,Y[firstindex][0])
        
        w_index = 0
        for i in range(0,len(Y)):
            if Y[i] != []:
                tree = spatial.KDTree(Y[i])
                currentdistance = tree.query(x)[0]
                if currentdistance < distance:
                    distance = currentdistance
                    w_index = i                               
  
        return  w_index

    def predict_based_continuous_splitdata(self,X,Y):
        """
        Predict cluster for each element in X.
        Parameters
        ----------
        X : ndarray
            An ndarray of shape (n, self.dim) where n is the number of samples.
            The data to predict clusters for.
        Returns
        -------
        labels : ndarray
            An ndarray of shape (n,). The predicted cluster index for each item
            in X.
        """

        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'


        labels =[]
        for x in X:
            b = self._find_belonged_continuous_neuron(x,Y)
            labels.append(b)

        # winWindexlabels, labels = np.array([ for x in X])
        # labels will be always from 0 - (m*n)*stop_split_num-1
        return np.array(labels)
    """
    def predict_based_overlapped_splitdata(self,test_data_continuous, test_data_discrete):

        assert len(test_data_continuous.shape) == 2, f'test_data_continuous should have two dimensions, not {len(test_data_continuous.shape)}'
        assert len(test_data_discrete.shape) == 2, f'test_data_discrete should have two dimensions, not {len(test_data_discrete.shape)}'

        labels =[]
        for x in test_data_continuous:
            a = self._find_belonged_neuron(x,self.all_split_datas_continuous)
            for d in test_data_discrete:
                b = self._find_belonged_discrete_neuron(d,self.all_split_datas_discrete)

                key = str(a) + str(b)
                labels.append(self.split_data_combination_index_dic[key])
        # winWindexlabels, labels = np.array([ for x in X])
        # labels will be always from 0 - (m*n)*stop_split_num-1
        return np.array(labels)
    """


    def predict_based_overlapped_splitdata(self,test_data_continuous, test_data_discrete):

            assert len(test_data_continuous.shape) == 2, f'test_data_continuous should have two dimensions, not {len(test_data_continuous.shape)}'
            assert len(test_data_discrete.shape) == 2, f'test_data_discrete should have two dimensions, not {len(test_data_discrete.shape)}'

            labels =[]

            for i in range(0,len(test_data_continuous)):
                a = self._find_belonged_continuous_neuron(test_data_continuous[i],self.all_split_datas_continuous)
              #  print (a)
               # print (self.all_split_datas_indexes_continuous)
                for i in range(0,len(self.all_split_datas_indexes_continuous) ):
                    if a in self.all_split_datas_indexes_continuous[i]:
                        belonged_a_index = i
               
               # a = self.som_continuous._find_bmu(test_data_continuous[i],self.som_continuous.weights0)
                #print("self.som_continuous.weights0 shape {}".format(self.som_continuous.weights0.shape[0]))
                b = self._find_belonged_discrete_neuron(test_data_discrete[i],self.all_split_datas_discrete)
                
                for j in range(0,len(self.all_split_datas_indexes_discrete) ):
                    if b in self.all_split_datas_indexes_discrete[j]:
                        belonged_b_index = j

                key = str(belonged_a_index) + "-" + str(belonged_b_index)
                labels.append(self.split_data_combination_index_dic[key])
            # winWindexlabels, labels = np.array([ for x in X])
            # labels will be always from 0 - (m*n)*stop_split_num-1
            return np.array(labels)


    def getScore(self,scorename, y_true, y_pred):
        #print(scorename)
        self.purity_score(scorename,y_true,y_pred)
        self.nmiScore(scorename,y_true,y_pred)
        self.ariScore(scorename,y_true,y_pred)



    def initializedata(self):
        self.train_W0_predicted_label = []
        self.train_W_continuous_predicted_label = []
        self.test_W0_predicted_label =   []
        self.test_W_Continuous_predicted_label =  []
        
    def getRightdataDistribution(self,predicted_clusters,weight0):           
        for i in range(0, len(predicted_clusters)):
            unit_list= []
            distances = []
            distances1 = []
            current_cluster_data = []
            for j in range(0, len(predicted_clusters[i])):
                current_cluster_data.append(self.data_train[j])
        
            for j in range(0, len(predicted_clusters[i])):
                unit_list.append(j)
            # type 0 distance to belonged neuron, 1
                #print("len self.right_datas[split_number][j] {}".format(self.right_datas[split_number][j]))
                distances.append(np.linalg.norm ((self.data_train[j] -weight0[i] )).astype(float))
                distances1.append(np.linalg.norm((self.data_train[j]- np.mean(current_cluster_data, axis=0)).astype(float), axis=0))
           # distances = np.sort(distances)
            #distances1 = np.sort(distances1)
            if  unit_list!= [] and len(unit_list) >1 :
                plt.xlabel('data number in cluster {}'.format(len(unit_list)))
                plt.plot(unit_list,distances,'g',label ='distance to cluster center')
                plt.plot(unit_list,distances1,'b',label ='distance to cluster mean')
                plt.legend()
                plt.show()  


     
   # utility delete multiple obj in a list

    def delete_multiple_element(self,list_object, indices):
        indices = sorted(indices, reverse=True)
        for idx in indices:
         if idx < len(list_object):
                list_object.pop(idx)
    

    def test_W_continuous(self):



        self.train_W_continuous_predicted_label = self.predict_based_continuous_splitdata(self.data_train_continuous,self.all_split_datas_continuous)    
        #self.train_W_continuous_predicted_label = self.som_continuous.predict(self.data_train_continuous,self.som_continuous.weights0)   

        predicted_clusters_indexes,b = self.get_indices_and_data_in_predicted_clusters(len(self.all_split_datas_continuous),self.train_W_continuous_predicted_label,self.data_train_continuous)
        self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters_indexes),1)  

        
        transferred_predicted_label_train_W_continuous =  self.transferClusterLabelToClassLabel(self.PLabel_to_Tlabel_Mapping_W_Continous,self.train_W_continuous_predicted_label)                 
        self.getScore("train_continuous_score_W_continuous",self.train_label_all,transferred_predicted_label_train_W_continuous)      

        
        self.test_W_Continuous_predicted_label = self.predict_based_continuous_splitdata(self.data_test_continuous,self.all_split_datas_continuous) 
        #self.test_W_Continuous_predicted_label = self.som_continuous.predict(self.data_test_continuous,self.som_continuous.weights0)   
        transferred_predicted_label_test_W_continuous = self.transferClusterLabelToClassLabel(self.PLabel_to_Tlabel_Mapping_W_Continous,self.test_W_Continuous_predicted_label)   
        #print("transferred_predicted_label_test {}".format(transferred_predicted_label_test))     
        self.getScore("test_continuous_score_W_continuous",self.test_label_all,transferred_predicted_label_test_W_continuous)

        if self.test_continuous_score_W_continuous_n < self.test_continuous_score_W0_n:
             print("Not good nmi result for continuous features !!!!!")
        if self.test_continuous_score_W_continuous_a < self.test_continuous_score_W0_a:
            print("Not good ari result for continuous features  !!!!!")
        print("New Continuous Feateure Neurons Number :{}".format(len(self.all_split_datas_continuous)))

    
    def drawnormaldistributonplot(self, predicted_clusters_index, i,color):
        #print("predicted_clusters_index {}".format(predicted_clusters_index))
        if len(predicted_clusters_index) >=1:
            print("neuron data  **************** {}".format(i))
            total_data_in_each_dim = []
            for i in range(0,self.som.dim):
                total_data_in_each_dim.append([])
                #print("total_data_in_each_dim {}".format(total_data_in_each_dim))
            for item in predicted_clusters_index:
               # print("item {}".format(item))
                data = self.data_train_all[item]
                for i in  range(0,self.som.dim):
                    total_data_in_each_dim[i].append(data[i])
           # print("total_data_in_each_dim 2 {}".format(total_data_in_each_dim))
            if self.row != 1:
                fig, axs = plt.subplots(self.row, self.column,figsize=(12, 12))
            else: fig, axs = plt.subplots(1, self.column,figsize=(12, 12))
            for i in range(0,self.som.dim):
                x_axis = total_data_in_each_dim[i]
                if self.row != 1:
                    m = int(i/self.column)
                
                n = int(i%self.column)      
                #print(" m {}  n {} ".format(i, x_axis))  
                a = np.min(self.data_train_all[:, i])
                b = np.max(self.data_train_all[:, i])
                #print("a {}  b{}".format(a, b))
                if self.row != 1:
                    axs[m,n].hist(x_axis, bins='auto', range =[a,b], color = color)
                else: axs[n].hist(x_axis, bins='auto', range =[a,b], color = color)
                
                #print("x_axis {}".format(x_axis))
            plt.show()
            print("****************")
        else:
            self.outliner_clusters.append(i)


    def getWcontinouswithSplitData(self,newclustered_datas):
        self.continuous_weights = []
        for i in range (0,len(newclustered_datas)):
           #print(f"newclustered_datas[i] {newclustered_datas[i]}" )

            neuron = np.average(newclustered_datas[i], axis=0)
           # print(f"neuron {neuron}" )
            self.continuous_weights.append(neuron) 

        self.continuous_weights = np.array( self.continuous_weights)
        #print(f"self.continuous_weights {self.continuous_weights }" )

    def run(self):
        """
        score_type 0 purity 1 numi 2 rai
        
        """
        # true distributon
        """
        self.cluster_data_based_true_label()
        print("True Distribution : ")
        for i in range(0,len(self.all_true_clustered_datas_indexes)) :
            self.drawnormaldistributonplot(self.all_true_clustered_datas_indexes[i],i,"green")
        """

        """
        *****************************************************************************************************************************************
        All data trained by som 
        """
        """
        self.som.fit(self.data_train_all)
        weight_continuous = self.som.weights0
        self.train_W0_predicted_label = self.som.predict(self.data_train_all,weight_continuous)   
        predicted_clusters_indexes, current_clustered_datas = self.get_indices_and_data_in_predicted_clusters(self.som.m*self.som.n, self.train_W0_predicted_label,self.data_train_all)   
  
        self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters_indexes) ,0)  
        # the value in predicted_clusters are true label value    
        transferred_predicted_label_train_W0 =  self.convertPredictedLabelValue(self.train_W0_predicted_label,self.PLabel_to_Tlabel_Mapping_W_Original)      
        self.getScore("all_train_score_W0",self.train_label_all,transferred_predicted_label_train_W0)

        self.test_W0_predicted_label = self.som.predict(self.data_test_all,weight_continuous)   
        transferred_predicted_label_test_W0 = self.transferClusterLabelToClassLabel(weight_continuous.shape[0],self.test_W0_predicted_label,self.data_test_all)    
        self.getScore("test_score_W0",self.test_label_all,transferred_predicted_label_test_W0)
        """

        """
        *****************************************************************************************************************************************
        continuous data trained by som 
        """
        # continuous
        self.som_continuous.fit(self.data_train_continuous)
        #print(f"weight_continuous 0 {weight_continuous }" )
        self.train_continuous_W0_predicted_label = self.som_continuous.predict(self.data_train_continuous,self.som_continuous.weights0)   
       # print(f"train_continuous_W0_predicted_label  {self.train_continuous_W0_predicted_label } ")
        predicted_clusters_indexes, current_clustered_datas = self.get_indices_and_data_in_predicted_clusters(self.som_continuous.weights0.shape[0], self.train_continuous_W0_predicted_label,self.data_train_continuous)   
        """predicted_clusters  [[23,31,21,2],[1,3,5,76],[45,5,12]] index in data_train in some situation, it will have [] """
        # the value in predicted_clusters are true label value  
        self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters_indexes),0)  
        transferred_predicted_label_train_continuous_W0 = self.transferClusterLabelToClassLabel(self.PLabel_to_Tlabel_Mapping_W_Original,self.train_continuous_W0_predicted_label)    
        self.getScore("train_continuous_score_W0",self.train_label_all,transferred_predicted_label_train_continuous_W0)

        self.test_continuous_W0_predicted_label = self.som_continuous.predict(self.data_test_continuous,self.som_continuous.weights0)  
        
        transferred_predicted_label_test_continuous_W0 = self.transferClusterLabelToClassLabel(self.PLabel_to_Tlabel_Mapping_W_Original,self.test_continuous_W0_predicted_label)    
        self.getScore("test_continuous_score_W0",self.test_label_all,transferred_predicted_label_test_continuous_W0)


        # the new cluster data generated by finding community nodes
        newclustered_datas = []
        newclustered_datas_index = []


        searched_datas = copy.deepcopy(current_clustered_datas)
        searched_datas = array(searched_datas).tolist()

        for i in range(0,len(searched_datas)):
            # get discasending fartheset node in current clustered_data
            sorted_index_distance_dict = self.split_continuous_data(predicted_clusters_indexes[i],  self.som_continuous.weights0[i])   
            while len(sorted_index_distance_dict) >0:          
                farthest_intra_node_index = self.getfarthest_intra_node_index(sorted_index_distance_dict)
                current_check_node = self.data_train_continuous[farthest_intra_node_index]
                del sorted_index_distance_dict[farthest_intra_node_index]
                #*** check if current_check_node is in other community
                already_in_community = False
   
                for k in range(0,len(newclustered_datas)):
                    if  current_check_node in np.array(newclustered_datas[k]):
                     already_in_community = True          
                     break
                
                if already_in_community :
                    continue


                newclustered_data =[]
                new_predicted_clusters =[]
                # get inter community nodes
                for j in range(0,len(searched_datas)):
                    if j != i:
                        sorted_dict_inter, distances_inter =  self.get_allnode_distance_to_center(current_check_node,predicted_clusters_indexes[j],self.som_continuous.weights0[j])
                        new_inter_community_nodes,new_inter_community_nodes_keys = self.get_inter_continuous_community_nodes(sorted_dict_inter,distances_inter)
                      #  print(f"new_inter_community_nodes_keys {new_inter_community_nodes_keys}")
                        if new_inter_community_nodes != []:
                            for item in new_inter_community_nodes:
                                newclustered_data.append(item)
                      
                        if new_inter_community_nodes_keys != []:
                            for item in new_inter_community_nodes_keys:
                                predicted_clusters_indexes[j].remove(item)
                                new_predicted_clusters.append(item)
                                #print("put item 1 in community {}".format(item))
                            # udpate predicted_clusters_indexes[j]
                            if predicted_clusters_indexes[j] != []: 
                               current_clustered_datas[j] = list(self.data_train_continuous[predicted_clusters_indexes[j]])
                            else:
                                current_clustered_datas[j] =[]
 
                sorted_dict_intra, distances_intra =  self.get_allnode_distance_to_center(current_check_node,predicted_clusters_indexes[i],self.som_continuous.weights0[i])
                #print(" i {}".format( i))

                new_intra_community_nodes,new_intra_community_nodes_keys = self.get_intra_continuous_community_nodes(sorted_dict_intra,self.som_continuous.weights0[i])
                #add self to the community
                #print(f"new_intra_community_nodes_keys {new_intra_community_nodes_keys}")
                if new_intra_community_nodes!=[]:
                    for item1 in new_intra_community_nodes:
                        newclustered_data.append(item1)    
               
                if new_intra_community_nodes_keys!=[]:
                    #print(" b1 {}".format( b1))
                    for item in new_intra_community_nodes_keys:
                        predicted_clusters_indexes[i].remove(item)
                        new_predicted_clusters.append(item)
                  
                    #change to np.array
                    current_clustered_datas[i] = np.array(current_clustered_datas[i])      
                    #update  predicted_clusters_indexes[i] 
                    if predicted_clusters_indexes[i] != [] :
                        current_clustered_datas[i] = list(self.data_train_continuous[predicted_clusters_indexes[i]] )
                    else:
                        current_clustered_datas[i] =[]                      
                 # add current data to the community generated
                if new_inter_community_nodes!=[] or new_intra_community_nodes!=[]:
                     newclustered_data.append(current_check_node)
                     new_predicted_clusters.append(farthest_intra_node_index)

                    
                     #*** remove current_check_node
                     predicted_clusters_indexes[i].remove(farthest_intra_node_index)
                     current_clustered_datas[i] = list(self.data_train_continuous[predicted_clusters_indexes[i]] )

            

                if newclustered_data != []:
                    newclustered_datas.append(newclustered_data)
                    newclustered_datas_index.append(new_predicted_clusters)

        #**** when item is [] do not remove as in the original W there will be also [] once is removed will cause problem  len( self.all_split_datas_continuous) is different or smaller then W0
      
        for item in current_clustered_datas:
            self.all_split_datas_continuous.append(item)
              #  print("len i {}".format(item))

        for item in newclustered_datas:
            self.all_split_datas_continuous.append(item)
             #   print("len j {}".format(item))  

        for item in predicted_clusters_indexes:
            self.all_split_datas_indexes_continuous.append(item)
        
        for item in newclustered_datas_index:
            self.all_split_datas_indexes_continuous.append(item)
        
       
        
        self.getWcontinouswithSplitData(self.all_split_datas_continuous)
        
        self.test_W_continuous()       