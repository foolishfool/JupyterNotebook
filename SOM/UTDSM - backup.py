"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
@author: Riley Smith
Created: 1-27-21
"""
from asyncio.windows_events import NULL
from distutils.errors import DistutilsArgError
from enum import Flag
#from curses.ascii import NULL
from importlib import resources
from pickle import TRUE
from telnetlib import PRAGMA_HEARTBEAT
import turtle
from zlib import DEF_BUF_SIZE
from sklearn import metrics
from scipy import spatial
import numpy as np
from numpy import array
import random
import copy
import math
import operator
import collections
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from kneed import KneeLocator

class UTDSM_SOM():
    """
    The 2-D, rectangular grid self-organizing map class using Numpy.
    """
    def __init__(self, som, data_train, data_test,label_train,label_test, elbow_num):
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
        self.right_datas = []
        self.all_rights_data_indexes = []

        self.combinedW = []
        self.combinedWMapping = []
        self.som = som  
        self.elbow_num = elbow_num
        self.elbowed_time = 0
        # initial cluster numbers in TDSM_SOM, which is the neuron number in som
        self.predicted_classNum= int(som.m*som.n)

        # Predicted label value convert to class class value when using specific W
        #self.PLabel_to_Tlabel_Mapping_W0 = np.zeros(self.predicted_classNum, dtype=object)
        #self.PLabel_to_Tlabel_Mapping_W1 = np.zeros(self.predicted_classNum, dtype=object)
        #self.PLabel_to_Tlabel_Mapping_WCombined = np.zeros(self.predicted_classNum, dtype=object)

        self.data_train = data_train
        print("self.data_train  initial {}".format(len(self.data_train )))
        self.data_test = data_test
        self.train_label = label_train
        self.train_label = self.train_label.astype(int)
       # print(self.train_label)
        self.test_label = label_test

        self.all_split_datas = []
        self.all_split_datas_indexes = []
        

    
       

        self.newmappings = []   
        self.all_left_train_data_label = np.arange(len(self.train_label))
        print("self.all_left_train_data_label  initial {}".format(len(self.all_left_train_data_label )))
    def purity_score(self,scorename, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        #print(" purity_score y_true{}  y_pred {} ".format(y_true,y_pred))
        if(scorename == "all_train_score_W0" ):
            #print(1111111111111)
            self.all_train_score_W0_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
            #print("all_train_score_W0_p {}".format(self.all_train_score_W0_p ))
        if(scorename == "all_train_score_W_Combined" ):
            self.all_train_score_W_Combined_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
           # print("all_train_score_W_Combined_p {}".format(self.all_train_score_W_Combined_p ))
        if(scorename == "test_score_W0" ):
            self.test_score_W0_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
          #  print("test_score_W0_p{}".format(self.test_score_W0_p ))
        if(scorename == "test_score_W_combined" ):
            self.test_score_W_Combined_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
           # print("test_score_W_combined_p{}".format(self.test_score_W_combined_p ))     
        if(scorename == "rest_score_W0" ):  
            self.rest_data_W0_p  = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
            print("self.rest_data_W0_p {}".format(self.rest_data_W0_p))

    def nmiScore(self,scorename, y_true, y_pred):
        #print(" nmi y_true{}  y_pred {} ".format(y_true,y_pred))
        if(scorename == "all_train_score_W0" ):
            self.all_train_score_W0_n = normalized_mutual_info_score(y_true,y_pred)
            print("all_train_score_W0_n {}".format(self.all_train_score_W0_n ))  
        if(scorename == "all_train_score_W_Combined" ):          
            self.all_train_score_W_Combined_n = normalized_mutual_info_score(y_true,y_pred)
            # if  self.all_train_score_W_Combined_n >= self.all_train_score_W0_n :
            #     print("all_train_score_W_Combined increased in nmi")
            print("all_train_score_W_Combined_n {}".format(self.all_train_score_W_Combined_n ))  
        if(scorename == "test_score_W0" ):
            self.test_score_W0_n = normalized_mutual_info_score(y_true,y_pred)
            print("test_score_W0_n {}".format(self.test_score_W0_n ))  
        if(scorename == "test_score_W_combined" ):
            self.test_score_W_Combined_n = normalized_mutual_info_score(y_true,y_pred)
            # if  self.test_score_W_Combined_n >= self.test_score_W0_n :
            #     print("test_score_W_Combined increased in nmi")
            # else: print(-1)
            print("test_score_W_combined_n {}".format(self.test_score_W_Combined_n ))  
        if(scorename == "rest_score_W0" ):
            self.rest_data_W0_n  = normalized_mutual_info_score(y_true,y_pred)
          #  if self.rest_data_W0_n == 0.0 :
             #   print("y_true {} y_pred {} ".format((y_true),(y_pred)))
            #print("self.rest_data_W0_n {}".format(self.rest_data_W0_n))


    def ariScore(self,scorename, y_true, y_pred):
       # print(" ariScore y_true{}  y_pred {} ".format(y_true,y_pred))
        if(scorename == "all_train_score_W0" ):
            self.all_train_score_W0_a = adjusted_rand_score(y_true,y_pred)
            print("all_train_score_W0_a {}".format(self.all_train_score_W0_a ))  
        if(scorename == "all_train_score_W_Combined" ):
            self.all_train_score_W_Combined_a = adjusted_rand_score(y_true,y_pred)
            # if  self.all_train_score_W_Combined_a >= self.all_train_score_W0_a :
            #     print("all_train_score_W_Combined increased in ari")
            # else: print(-1)
            print("all_train_score_W_Combined_a {}".format(self.all_train_score_W_Combined_a ))  
        if(scorename == "test_score_W0" ):
            self.test_score_W0_a = adjusted_rand_score(y_true,y_pred)
            print("test_score_W0_a  {}".format(self.test_score_W0_a ))  
        if(scorename == "test_score_W_combined" ):
            self.test_score_W_Combined_a = adjusted_rand_score(y_true,y_pred)
            print("test_score_W_combined_a {}".format(self.test_score_W_Combined_a ))  
        if(scorename == "rest_score_W0" ): 
            self.rest_data_W0_a  = adjusted_rand_score(y_true,y_pred)
           # print("y_true len {} y_pred {} ".format(len(y_true),len(y_pred)))
            #print("self.rest_data_W0_a {}".format(self.rest_data_W0_a))
            # if  self.test_score_W_Combined_a >= self.test_score_W0_a :
            #     print("test_score_W_Combined increased in ari")
            # else: print(-1)

    
    def get_indices_in_predicted_clusters(self,class_num_predicted,predicted_label,data_predict_indexes):
            
            """
            class_label is the true class label
            predicted_label = [1,1,2,3,1,1,2,1]
            idx start from = to n
            class_label index also start from 0 to n
            """

            #print("predicted_label 2{}".format(predicted_label))
            #print("data_predict_indexes 2{}".format(data_predict_indexes))
            # class labels in each cluster = [[1,1,1,1],[2,2,2,2],[1,1]]]
            clusters_indexes = []
            clusters_datas = []
            
            for i in range(0,class_num_predicted):
                newlist = []
                newdatalist = []
                for idx, y in enumerate(predicted_label): 
                    # is the cluster label
                    if(y == i):
                        x = data_predict_indexes[idx]
                        x = int(x)
                        #print(x)
                        newlist.append(x)  
                        newdatalist.append(self.data_train[x])                        
                clusters_indexes.append(newlist)
                clusters_datas.append(np.array(newdatalist))
            
         
            return clusters_indexes,clusters_datas

    # knewo [[2,35,34,3,23],[211,12,2,1]] get [[0,0,1,1] [0,0,1]]
    def get_mapped_class_in_clusters(self,clusters_indexes):
        mapped_clases_in_clusters = []
        for i in range(0, len(clusters_indexes)):
            mapped_clases_in_clusters.append([])
        for j in range(0, len(clusters_indexes)):
            for item in clusters_indexes[j]:
                mapped_clases_in_clusters[j].append(self.train_label[item])
        #print("mapped_clases_in_clusters 2{}".format(mapped_clases_in_clusters))
        # mapped_clases_in_clusters = [[1,2,1,2,1,1],[2,2,2,2],[0,1,0]]
        return mapped_clases_in_clusters
    
    def delete_multiple_element(self,list_object, indices):
        indices = sorted(indices, reverse=True)
        for idx in indices:
            if idx < len(list_object):
                list_object.pop(idx)


    def getMaxIndex(self,error_rate,clustered_indexes):
        #clustered_indexes = [[indices ],[],[]]
        maxIndex = []
        dataNum_list = [] #dataNum_list = [5,6,2,0] the number is data number in each cluster
        for x in clustered_indexes:
            dataNum_list.append(len(x))
        min_value = np.min(dataNum_list)
        reference_value = min_value-1
        for n in dataNum_list:
            if n == 0: # n is a empty neurons and make it to be min value, so will never access to the calculation of getting min value
                dataNum_list[n] = reference_value
        max_datas_index = []
        dataNum_list = np.array(dataNum_list)
        for i in range(0,error_rate):
            if np.max(dataNum_list) != reference_value:
                maxresult = np.where(dataNum_list == np.amax(dataNum_list)) 
                for j in range(0,len(maxresult[0])):
                    if j not in max_datas_index:
                        max_datas_index.append(maxresult[0][j])
                        dataNum_list[maxresult[0][j]] = min_value
        #max_datas_index is a sorted list of dataNum_list max_datas_index = [4,5,6,1,0,3]  index of neurons that has data increases from neuron 4 to neuron3
        for i in range(0,error_rate):           
            maxIndex.append(max_datas_index[i])
        return maxIndex


          
    def getLabelMapping(self,predicted_class_label_in_each_cluster,Wtype  = 0):
        """
         predicted_class_label  = [[1,2,1,1],[3,3,3]]  the value in is the true value in class_label
         it means that predicted cluster 0 is 1 in class lable, cluster label 2 is 3 in class label
        """
        #print("predicted_class_label_in_each_cluster {}".format(predicted_class_label_in_each_cluster))
        predicted_label_convert_to_class_label = []
        for item in predicted_class_label_in_each_cluster:
            if item != []:
                # the first item is for cluster0       
                # transfer to true class value based on indices in predict lables          
                predicted_label_convert_to_class_label.append(self.getMaxRepeatedElements(item))
            else:
                predicted_label_convert_to_class_label.append(-1)
        
        if Wtype == 0 :
            self.PLabel_to_Tlabel_Mapping_W0 = predicted_label_convert_to_class_label
            #print("self.PLabel_to_Tlabel_Mapping_W0 {}".format(self.PLabel_to_Tlabel_Mapping_W0 ))

        if Wtype == 1 :
            self.PLabel_to_Tlabel_Mapping_WCombined = predicted_label_convert_to_class_label
            #print("self.PLabel_to_Tlabel_Mapping_WCombined {}".format(self.PLabel_to_Tlabel_Mapping_WCombined ))

        if Wtype == 2 :
            self.PLabel_to_Tlabel_Mapping_W1 = predicted_label_convert_to_class_label

    def getMaxRepeatedElements(self, list):
        #print("list{}".format(list))
        #Count number of occurrences of each value in array of non-negative ints.
        counts = np.bincount(list)
        #print("counts {}".format(counts))
        #Returns the indices of the maximum values along an axis.
        #print("np.argmax(counts) {}".format(np.argmax(counts)))
        return np.argmax(counts)

    def convertPredictedLabelValue(self,predicted_cluster_labels, PLable_TLabel_Mapping):
        # PLabel_CLabel_Mapping the mapping of cluster label to class label
        # PLable_TLabel_Mapping size is the som.m*som.n* stop_split_num
        #print("predicted_cluster_labels {}".format(predicted_cluster_labels))
        #print("PLable_TLabel_Mapping {}".format(PLable_TLabel_Mapping))
        for i in range(0,len(predicted_cluster_labels)):
            predicted_cluster_value =  predicted_cluster_labels[i]
            predicted_cluster_labels[i] = PLable_TLabel_Mapping[predicted_cluster_value]      

  
        return predicted_cluster_labels

    

    def transferClusterLabelToClassLabel(self,mapping, predicted_cluster_number,predicted_cluster_labels, data_to_predict ,Wtype = 0): 
        """
         winW_indexes : the W that needs to use
         mapping_cluster_class_values whether to map cluster label and class label or not
         train_counter: split number
         Wtype: the tyee of W
                0:W0
                1:W1
                2:WCombine

        """      
        if mapping == True:
            predicted_clusters,b = self.get_indices_in_predicted_clusters(predicted_cluster_number,predicted_cluster_labels,data_to_predict)
            #print("predicted_clusters {}".format(predicted_clusters)) 
          
            if(Wtype == 0):
             self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters),0)  
              # the value in predicted_clusters are true label value       
             predicted_class_labels =  self.convertPredictedLabelValue(predicted_cluster_labels,self.PLabel_to_Tlabel_Mapping_W0)
             self.initial_current_traindata = b
             return predicted_class_labels
            
            if(Wtype == 1):   

             self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters),1)      
           
             predicted_class_labels =  self.convertPredictedLabelValue(predicted_cluster_labels,self.PLabel_to_Tlabel_Mapping_WCombined)
            
             return predicted_class_labels

        else:
            if(Wtype == 0):
                predicted_class_labels = self.convertPredictedLabelValue(predicted_cluster_labels,self.PLabel_to_Tlabel_Mapping_W0)
                return predicted_class_labels
            if(Wtype == 1):
                predicted_class_labels = self.convertPredictedLabelValue(predicted_cluster_labels,self.PLabel_to_Tlabel_Mapping_WCombined)
                return predicted_class_labels



    #get a dictionary with nodes has decsending distance with cluster center
    def split_data(self, gargetgroup_index,cluster_center):
        sorted_data_dict = {}
       # print("gargetgroup_index {} ".format(gargetgroup_index))
        for idx in gargetgroup_index:     
            distance = np.linalg.norm((self.data_train[idx] - cluster_center).astype(float))
            if distance >0:
                #print("idx {} distance  {}".format(idx,distance))
                sorted_data_dict[idx] = distance
            #if distance == 0:
              #  print("zero distcance for data 1 idx {}".format(idx))       
       # print("sorted_dict 1{} ".format(sorted_data_dict))
        sorted_dict = dict(sorted(sorted_data_dict.items(), key=operator.itemgetter(1),reverse=True))
        #print("sorted_dict 2{} ".format(sorted_dict))
        #collections.OrderedDict(sorted(sorted_data_dict.items(), reverse=True, key=lambda t: t[1]))
        return sorted_dict

    def getfarthest_intra_node_index(self,sorted_dict):
        #print("sorted_dict {}".format(sorted_dict))
        find_node = next(iter(sorted_dict))

        #print("find_node {}".format(find_node))
        return find_node

    def get_allnode_distance_in_a_group(self, target_node,  group_index, group_center):
        sorted_data_dict = {}
        distances_intra = {}
       # print("group_index {}".format(group_index))   
        for idx in group_index:     
            distance = np.linalg.norm((self.data_train[idx] - target_node).astype(float))
            if distance >0 :
                sorted_data_dict[idx] =distance  

        sorted_dict = dict(sorted(sorted_data_dict.items(), key=operator.itemgetter(1),reverse=False))

        for key in sorted_dict:
            distance_intra = np.linalg.norm((self.data_train[key] - group_center).astype(float))
            distances_intra[key] = distance_intra        
       # print(" distances_intra   {}  ".format(distances_intra))
        return sorted_dict,distances_intra
    # get all the inter node that has smaller distance to the target data, then target data to its cluster center 
    def get_intra_community_nodes(self,sorted_dict, intra_center):
        community_nodes = []
        community_nodes_keys = []
        #print("sorted_dict intro {}".format(sorted_dict))
        #for key, value in sorted_dict.items():
        for key in sorted_dict:  
           # print("key {}".format(key))
            #**** cannot <= when == is itself, may cause one data one community       
            distance_intra = np.linalg.norm((self.data_train[key] - intra_center).astype(float))
            if sorted_dict[key] < distance_intra:
                #print("sorted_dict[key {} key {} distance_intra{}".format(sorted_dict[key],key,distance_intra ))
                community_nodes.append(self.data_train[key])
                #print("key intra {}".format(key))
                community_nodes_keys.append(key)
        return community_nodes,community_nodes_keys


    def get_inter_community_nodes(self,sorted_dict,distances_intra):
        community_nodes = []
        community_nodes_keys = []
        #print("sorted_dictlenth{}".format(len(sorted_dict)))
        #print("distances_intra lenth1 {}".format(len(distances_intra)))
        for key in sorted_dict:
            #print("key {}".format(key))
            #print("sorted_dict[key]  {}".format(sorted_dict[key]))
            #print(" distances_intra[key] {}".format( distances_intra[key]))
            if sorted_dict[key] < distances_intra[key]:
                community_nodes.append(self.data_train[key])
                community_nodes_keys.append(key)
                #print("KEY IN COMMUNITY {}".format(key))
        #print("community_nodes {} community_nodes_keys {} ".format(community_nodes,community_nodes_keys))
        return community_nodes,community_nodes_keys


    def _find_bmu_based_neuron_representation_ineachW(self,x, weightIndex, newWeights):
        """
        neurons_representations = [[x1,x2, x3],[],[x4,x5]] measn the data that each neuron in newWeights represents
        """
        #print("self.empty_neurons[weightIndex] {}".format(self.empty_neurons[weightIndex]))
        #print("Before delte {}".format(newWeights.shape[0]))
        newWeights = np.delete(newWeights, self.empty_neurons[weightIndex], 0) 
       # print("After delte {}".format(newWeights.shape[0]))
        x_stack = np.stack([x]*(newWeights.shape[0]), axis=0)
       

        distance = np.linalg.norm((x_stack - newWeights).astype(float), axis=1)
        # Find index of best matching unit
        #print("_find_bmu_based_neuron_representation_ineachW {}".format(self.non_empty_neurons[weightIndex][np.argmin(distance)]))
        return self.non_empty_neurons[weightIndex][np.argmin(distance)]


    def _find_belonged_neuron(self,x,Y):
        """
        Find the index of the best matching unit for the input vector x.
        """  
        #initial distance
        #print("len Y {}".format(len(Y)))
        Y = np.array(Y)
        firstindex = 0
        for i in range(len(Y)):
            if Y[i]!= []:
                firstindex = i
                break
                

        #*** self.rightdatas = [[[],[]],[data in rightdata2],[]]
        #if len(Y[firstindex]) == 1 :
        #    print("x {} Y[0] {}".format(x,Y[firstindex][0]))
        #    distance = math.dist(x,Y[firstindex])
        #else:
        #print("x {} Y[0] {}".format(x,Y[firstindex]))
        #print("x {} Y[0][0] {}".format(x,Y[firstindex][0]))
        distance = math.dist(x,Y[firstindex][0])
        
        w_index = 0
        for i in range(0,len(Y)):
            if Y[i] != []:
                #print("i {}".format(i))
                tree = spatial.KDTree(Y[i])
                currentdistance = tree.query(x)[0]
                if currentdistance < distance:
                    distance = currentdistance
                    w_index = i                               
        #print("distance {}".format(distance))
        #print("w_index {}".format(w_index))
        return  w_index


    def getBestWAmongAllW(self,x,comparedData):
        distance = math.dist(x,comparedData[0][0])
        best_W_index = 0
        for i in range(0,len(comparedData)):
              #change type object to float
              array = np.array(comparedData[i]) 
              array = array.astype(float)
    
              tree = spatial.KDTree(array)
              currentdistance = tree.query(x)[0]
              if currentdistance < distance:
                distance = currentdistance
                best_W_index = i
        #print("for x W {} can use for representation".format(best_W_index))
        return best_W_index
    

    def predict_based_splitdata(self,X,Y):
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
            b = self._find_belonged_neuron(x,Y)
            labels.append(b)

        # winWindexlabels, labels = np.array([ for x in X])
        # labels will be always from 0 - (m*n)*stop_split_num-1
        return np.array(labels)

    

    def getScore(self,scorename, y_true, y_pred):
        #print(scorename)
       # self.purity_score(scorename,y_true,y_pred)
        self.nmiScore(scorename,y_true,y_pred)
        self.ariScore(scorename,y_true,y_pred)



    def initializedata(self):
        self.train_W0_predicted_label = []
        self.train_W_combined_predicted_label = []
        self.test_W0_predicted_label =   []
        self.test_W_combined_predicted_label =  []
        
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


    def run(self,train_data_indexes):
        """
        score_type 0 purity 1 numi 2 rai
        
        """
        # get train and test dataset 
        print("new round train data len {}".format(len(train_data_indexes)))
        data_train = []
        true_train_label = []

        for item in train_data_indexes:
            item = int(item)
            data_train.append(self.data_train[item])
            true_train_label.append(self.train_label[item])

        data_train = np.array(data_train)

        self.som.fit(data_train)
        weight0 = self.som.weights0
        newclustered_datas = []
        newclustered_datas_index = []

        self.train_W0_predicted_label = self.som.predict(data_train,weight0)   
       # print(" self.train_W0_predicted_label {}".format( self.train_W0_predicted_label ))
        predicted_clusters, current_clustered_datas = self.get_indices_in_predicted_clusters(self.som.m*self.som.n, self.train_W0_predicted_label,train_data_indexes)   
        print("current train data generated predicted_clusters len {}".format(len(predicted_clusters)))
        # predicted_clusters  [[23,31,21,2],[1,3,5,76],[45,5,12]] index in data_train
        self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters) ,0)  
        #print("initial   mapping {}".format(self.PLabel_to_Tlabel_Mapping_W0))
        # the value in predicted_clusters are true label value       
        transferred_predicted_label_train_W0 =  self.convertPredictedLabelValue(self.train_W0_predicted_label,self.PLabel_to_Tlabel_Mapping_W0)      
        #print("transferred_predicted_label_train_W0 {}".format(transferred_predicted_label_train_W0))
       
        if self.all_split_datas == []:
            self.getScore("all_train_score_W0",true_train_label,transferred_predicted_label_train_W0)
            self.test_W0_predicted_label = self.som.predict(self.data_test,weight0)   
            transferred_predicted_label_test_W0 = self.transferClusterLabelToClassLabel(False,weight0.shape[0],self.test_W0_predicted_label,self.data_test)    
            self.getScore("test_score_W0",self.test_label,transferred_predicted_label_test_W0)
        
       
        
        searched_datas = copy.deepcopy(current_clustered_datas)
        searched_datas = array(searched_datas).tolist()
      
        for i in range(0,len(searched_datas)):
            # get discasending fartheset node in current clustered_data
            sorted_dict = self.split_data(predicted_clusters[i],  weight0[i])
            #print(" the data searched amount is {} in predicterd clusters {}".format(len(sorted_dict),i))
            while len(sorted_dict) >0:    
            
                farthest_intra_node_index = self.getfarthest_intra_node_index(sorted_dict)
               # print("farthest_intra_nnode_index {} searched_datas[i] len {} i {}".format(farthest_intra_nnode_index,len(searched_datas[i]),i))
                current_check_node = self.data_train[farthest_intra_node_index]
               
                del sorted_dict[farthest_intra_node_index]
                #print("len sorted_dict {}".format(sorted_dict))
                #print("scurrent_check_node {}".format(current_check_node))
                #*** check if current_check_node is in other community
                already_in_community = False
   
                for k in range(0,len(newclustered_datas)):
                    #print("len(self.newclustered_datas) {} lenth {}".format(self.newclustered_datas[i],len(self.newclustered_datas[i])))
                    if  current_check_node in np.array(newclustered_datas[k]):
                        already_in_community = True     
                        #print("already_in_community {}".format(farthest_intra_node_index))                  
                        break
                
                if already_in_community :
                    continue


                
                newclustered_data = []
                new_predicted_clusters = []
     
                for j in range(0,len(searched_datas)):
                    if j != i:
                       # print("i {} j {}".format(i,j))
                       # print("inter data searched  i {} self.current_clustered_datas[j] j {} len{} )".format(i, j,len(self.current_clustered_datas[j] )))
                        sorted_dict_inter, distances_inter =  self.get_allnode_distance_in_a_group(current_check_node,predicted_clusters[j],weight0[j])
                       # print("sorted_dict_inter {} ".format(sorted_dict_inter))
                        a,b = self.get_inter_community_nodes(sorted_dict_inter,distances_inter)
                        if a != []:
                            for item in a:
                                newclustered_data.append(item)
                      
                        if b != []:
                           # print(" self.predicted_clusters 0[j] {}".format( predicted_clusters[j]))
                           # print("b {}".format(b))
                            for item in b:
                                predicted_clusters[j].remove(item)
                                new_predicted_clusters.append(item)
                           
                            if predicted_clusters[j] != []: 
                               current_clustered_datas[j] = list(self.data_train[predicted_clusters[j]])
                            else:
                                current_clustered_datas[j] =[]
                               #print("predicted_clusters[j] {}".format( predicted_clusters[j]) )
                               #print("j {}   current_clustered_datas[j] {}".format(j,  current_clustered_datas[j])) 
                        #print("len predicted_clusters[j] {} j".format(len(predicted_clusters[j]),j) )   
                       # print("len current_clustered_datas[j] {} j".format(len(current_clustered_datas[j]),j) )   
                sorted_dict_intra, distances_intra =  self.get_allnode_distance_in_a_group(current_check_node,predicted_clusters[i],weight0[i])
                #print(" i {}".format( i))

                a1,b1 = self.get_intra_community_nodes(sorted_dict_intra,weight0[i])
                #add self to the community
                #print(" predicted_clusters[i] initial {}".format( predicted_clusters[i]))
                if a1!=[]:
                    for item1 in a1:
                        newclustered_data.append(item1)    
               
                if b1!=[]:
                    #print(" b1 {}".format( b1))
                    for item in b1:
                        predicted_clusters[i].remove(item)
                        new_predicted_clusters.append(item)
                  
                    current_clustered_datas[i] = np.array(current_clustered_datas[i])                  
                    if predicted_clusters[i] != [] :
                        current_clustered_datas[i] = list(self.data_train[predicted_clusters[i]] )
                    else:
                        current_clustered_datas[i] =[]
                       # print("i {} predicted_clusters[i]  {} len  current_clustered_datas[i] {}".format(i, predicted_clusters[i] , len(current_clustered_datas[i])) )
                 # add current data to the community generated
                if a!=[] or a1!=[]:
                     newclustered_data.append(current_check_node)
                     new_predicted_clusters.append(farthest_intra_node_index)
                     #*** remove current_check_node
                     predicted_clusters[i].remove(farthest_intra_node_index)
                     current_clustered_datas[i] = list(self.data_train[predicted_clusters[i]] )

                #print("len predicted_clusters[i] {} j".format(len(predicted_clusters[i]),i) )   
                #print("len current_clustered_datas[i] {} i".format(len(current_clustered_datas[i]),i) )   
                newclustered_data = np.array(newclustered_data)

                if newclustered_data != []:
                    newclustered_datas.append(newclustered_data)
                    newclustered_datas_index.append(new_predicted_clusters)
                    #print("self.newclustered_datas 2 {}".format(self.newclustered_datas))
        

        #print("before elbow")
        #self.test_rest_data(predicted_clusters,current_clustered_datas,weight0)
        self.test_combineW() 
        #if self.elbowed_time < self.elbow_num:
            
        #for i in range(0,len(predicted_clusters)):
        #    intra_distances_dict = {}
        #    distancelist = []
        #    key_list = []
        #    #unit_list = []
        #    newclustered_data = []
        #    new_predicted_clusters = []
        #    for index in predicted_clusters[i]:
        #        intra_distances_dict[index] = np.linalg.norm((self.data_train[index] - weight0[i]).astype(float))
        #    intra_distances_dict = dict(sorted(intra_distances_dict.items(), key=operator.itemgetter(1),reverse=False))
#
        #
#
        #    for key,value in intra_distances_dict.items():
        #        distancelist.append(value)
        #        key_list.append(key)
        #        #unit_list.append(x)
        #    x  = range(1, len(distancelist)+1)
        #   # print(x)
        #   # print(distancelist)
#
        #    if  len(distancelist) >3 :
        #        kn = KneeLocator(x, distancelist, curve='convex', direction='increasing')
        #        elbow = kn.knee
        #        if elbow != None:
        #           # print("elbow kn {}".format(kn.knee))
        #            for j in range(0, len(key_list)):
        #                if  j >= elbow:
        #                
        #                    predicted_clusters[i].remove(key_list[j])
        #                    new_predicted_clusters.append(key_list[j])
        #                    newclustered_data.append(self.data_train[key_list[j]])
        #            
        #            if newclustered_data != []:
        #                    newclustered_datas.append(newclustered_data)
        #                    newclustered_datas_index.append(new_predicted_clusters)
        #plt.xlabel('rest data intra distances')
        #plt.plot(unit_list,distancelist,'g')
        #plt.legend()
        #plt.show()  
        #self.elbowed_time = self.elbowed_time+1
        """
        new_split_data_used_in_next_round = []
        new_split_data_label_used_in_next_round = []


        if newclustered_datas !=[]:
            for item in newclustered_datas:
                for j in item:
                    new_split_data_used_in_next_round.append(j)

            for item1 in newclustered_datas_index:
                for k in item1:
                    new_split_data_label_used_in_next_round.append(k)
            new_split_data_used_in_next_round = np.array(new_split_data_used_in_next_round)
            new_split_data_label_used_in_next_round = np.array(new_split_data_label_used_in_next_round)
            print("Split one time! with new cluster data len{}".format(len (newclustered_datas)))
            print("new split data len {}".format(len(new_split_data_used_in_next_round)))
            print("test_rest_data ")
            #preidcited _clusteder is generated by training current_train_lebel
           # self.test_rest_data(predicted_clusters,current_clustered_datas,weight0)

          

            #alreay has new mapping of W0
            if self.rest_data_W0_n >= 0.9:
                 new_generated_perfectW = []
                 new_generated_perfecgtW_mapping =[]
                 self.right_datas.append(self.rest_data)
                 print("lenself.right_datas {} ".format(len(self.right_datas)))
                 print("len rest_data {} self.rest_data_indexes len{} ".format(len(self.rest_data),len(self.rest_data_indexes)))
                 new_generated_perfectW.append(weight0)
                 new_generated_perfecgtW_mapping.append(self.newmappings)
                 self.combinedW.append(new_generated_perfectW)      
                 self.combinedWMapping.append(new_generated_perfecgtW_mapping)    
                 #self.left_train_data = np.delete(self.left_train_data, self.rest_data_true_label)
                 #print("len self.rest_data {}".format(len(self.rest_data )))
                 #print("len self.all_left_train_data_label 0  {}".format(len(self.all_left_train_data_label)))
                 for item in self.rest_data_indexes:      
                    self.all_left_train_data_label = [x for x in self.all_left_train_data_label if x != item]           
                 print("len self.all_left_train_data_label   {}".format(len(self.all_left_train_data_label)) )
                   # self.left_train_data_label.remove(item)
                 ## tod REMOVE RESTDATA IN SELF.TRAIN_DATA 
                 if self.all_left_train_data_label !=[]:             
                    self.run(self.all_left_train_data_label)
                    print(" run    all_left_train_data_label   ")
                 else:
                    # do the test_combineW
                    print("44444")
                    #TODO
                    self.test_combineW_basedon_restdata()
            else:
                if self.rest_data !=[] :
                    #print("rest_data_W0_n <0.9 and repeate run rest data")         
                    self.run(self.rest_data_indexes)
                else:
                 print("self.rest_data is []")  
                 #last round but  self.rest_data_W0_n is not 1 ** may get inifite loop
                 self.run(self.all_left_train_data_label)
            
           #if self.rest_data !=[]:             
           #    self.run(nextround_train_label)
           #
           #else:
           #     for item in current_clustered_datas:
           #        if item!=[]:
           #            self.all_split_datas.append(item)
           #     for item in self.PLabel_to_Tlabel_Mapping_W0:
           #        if item != -1:
           #            self.newmappings.append(item)
           #     self.test_combineW() 

                #self.getRightdataDistribution(predicted_clusters,weight0)
        else:
            #print("predicted_clusters len {}".format(len(predicted_clusters)))
            for item in current_clustered_datas:
                if item!=[]:
                    self.all_split_datas.append(item)
            #print("add split data last len{}".format(len( self.all_split_datas)))
            #print("get last mapping {} self.newmappings len".format(self.PLabel_to_Tlabel_Mapping_W0,len(self.newmappings )))
            for item in self.PLabel_to_Tlabel_Mapping_W0:
                if item != -1:
                    self.newmappings.append(item)
           # print("len self.newmappings last {}".format(len(self.newmappings)))
            #self.test_combineW()  
            if len(self.right_datas) ==0:
                #self.test_combineW() 
                print("repeat one time")
                self.run(train_data_indexes)
            elif len(self.all_left_train_data_label) != 0:
                print("no new split data and still has all_left_train_data_label len {} ".format(len(self.all_left_train_data_label)))
                self.run(self.all_left_train_data_label)
            elif len(self.all_left_train_data_label) == 0:
                self.test_combineW_basedon_restdata()
      
        else: self.test_combineW() 
        """

    def begin(self):
        while   self.all_left_train_data_label !=[]:
            self.run(self.all_left_train_data_label)

    def test_rest_data(self,predicted_clusters,current_clustered_datas,weight0):
            self.rest_data_true_label = []
            self.rest_data =[]
            self.rest_data_indexes =[]
            #remapping
            #print("predicted_clusters {}" .format(predicted_clusters))
            self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters) ,0)  
            #print("self.PLabel_to_Tlabel_Mapping_W0: {} len {}" .format(self.PLabel_to_Tlabel_Mapping_W0,len(self.PLabel_to_Tlabel_Mapping_W0)))
            for item in self.PLabel_to_Tlabel_Mapping_W0:
                if item != -1:
                    self.newmappings.append(item)
           # print("len self.newmappings {}".format(len(self.newmappings)))

            for item in predicted_clusters:
                if  item !=[]:
                    for i in range(0,len(item)):
                        self.rest_data_true_label.append(self.train_label[item[i]])
                        self.rest_data.append(self.data_train[item[i]])
                        self.rest_data_indexes.append(item[i])

            print("len rest_data_indexes {}".format(len(self.rest_data_indexes)))
            if self.rest_data !=[]:
                self.rest_data = np.array(self.rest_data)    

                for item in current_clustered_datas:
                    if item!=[]:
                        self.all_split_datas.append(item)
               # print("add split data len{}".format(len( self.all_split_datas)))
                #print("rest_data len {}".format(len(rest_data)))
                restdata_W0_predicted_label = self.som.predict(self.rest_data,weight0)  

                transferred_predicted_label_restdata_W0 =  self.convertPredictedLabelValue(restdata_W0_predicted_label,self.PLabel_to_Tlabel_Mapping_W0)     
                print("lenth rest data {}".format(len(self.rest_data)))
            # print("transferred_predicted_label_rest_data_W0{}".format(transferred_predicted_label_restdata_W0))
                self.getScore("rest_score_W0",self.rest_data_true_label,transferred_predicted_label_restdata_W0)

    def test_combineW(self):     
       # print("self.newmappings len {} ".format(len(self.newmappings)))  
       # print("self.all_split_datas len {} ".format(len(self.all_split_datas)))


        #print("len self.newmappings {}".format(len(self.newmappings)))
        self.train_W_combined_predicted_label = self.predict_based_splitdata(self.data_train,self.all_split_datas)    

        #print("self.train_W_combined_predicted_label {}".format(self.train_W_combined_predicted_label))  
        #*** needs to use self.combinedweight.shape[0] cannot use self.som.m*self.som.n as neruons is changed
        predicted_clusters,current_clustered_datas = self.get_indices_in_predicted_clusters(len(self.all_split_datas), self.train_W_combined_predicted_label,self.train_label)  
      #  self.getLabelMapping(self.get_mapped_class_in_clusters(predicted_clusters),1)  
        # the value in predicted_clusters are true label value
        # 
        predict_all_data_label_combineW = []
        for  item in self.train_W_combined_predicted_label:
            predict_all_data_label_combineW.append(self.newmappings[item])

        #transferred_predicted_label_train_WCombine =  self.transferClusterLabelToClassLabel(False,len(self.all_split_datas),self.train_W_combined_predicted_label,self.data_test,Wtype = 1)                 
        self.getScore("all_train_score_W_Combined",self.train_label,predict_all_data_label_combineW)      


        self.test_W_combined_predicted_label = self.predict_based_splitdata(self.data_test,self.all_split_datas)   

        predict_test_data_label_combineW = []
        for  item in self.test_W_combined_predicted_label:
            predict_test_data_label_combineW.append(self.newmappings[item])


       # transferred_predicted_label_test = self.transferClusterLabelToClassLabel(False,len(self.all_split_datas),self.test_W_combined_predicted_label,self.data_test,Wtype = 1)   
        #print("transferred_predicted_label_test {}".format(transferred_predicted_label_test))     
        self.getScore("test_score_W_combined",self.test_label,predict_test_data_label_combineW)

        if self.test_score_W_Combined_n < self.test_score_W0_n:
             print("Not good nmi result !!!!!")
        if self.test_score_W_Combined_a < self.test_score_W0_a:
            print("Not good ari result !!!!!")
        print("Combine Neurons Number :{}".format(len(self.all_split_datas)))


        #for item in self.all_split_datas:
        #    print("Member number in item :{}".format(len(item)))
        #if score_type == 0:
        #    print("Purity Score")   
        #elif score_type == 1:
        #    print("NMI Score")
        #elif score_type == 2:
        #    print("ARI Score")     
#
        #
        #print("train_score_W0 : {}".format( self.all_train_score_W0))
        #print("train_score_W\': {}".format(self.all_train_score_W_Combined))
        #print("test_score_W0 : {}".format( self.test_score_W0))
       # print("test_score_W\': {}".format(self.test_score_W_combined))

     
    def test_combineW_basedon_restdata(self):     
       # print("self.newmappings len {} ".format(len(self.newmappings)))  
       # print("self.all_split_datas len {} ".format(len(self.all_split_datas)))


        print("len self.right_datas{}".format(len(self.right_datas)))
        self.train_W_combined_predicted_label = self.predict_based_splitdata(self.data_train,self.right_datas)    

        #print("self.train_W_combined_predicted_label {}".format(self.train_W_combined_predicted_label))  
        #*** needs to use self.combinedweight.shape[0] cannot use self.som.m*self.som.n as neruons is changed
       # predicted_clusters,current_clustered_datas = self.get_indices_in_predicted_clusters(len(self.all_generated_datas), self.train_W_combined_predicted_label,self.train_label)  
      #  self.getLabelMapping(self.get_mapped_class_in_clusters(predicted_clusters),1)  
        # the value in predicted_clusters are true label value
        # 
        predict_all_data_label_combineW = []
        for  item in self.train_W_combined_predicted_label:
            predict_all_data_label_combineW.append(self.newmappings[item])

        #transferred_predicted_label_train_WCombine =  self.transferClusterLabelToClassLabel(False,len(self.all_split_datas),self.train_W_combined_predicted_label,self.data_test,Wtype = 1)                 
        self.getScore("all_train_score_W_Combined",self.train_label,predict_all_data_label_combineW)      


        self.test_W_combined_predicted_label = self.predict_based_splitdata(self.data_test,self.right_datas)   

        predict_test_data_label_combineW = []
        for  item in self.test_W_combined_predicted_label:
            predict_test_data_label_combineW.append(self.newmappings[item])


       # transferred_predicted_label_test = self.transferClusterLabelToClassLabel(False,len(self.all_split_datas),self.test_W_combined_predicted_label,self.data_test,Wtype = 1)   
        #print("transferred_predicted_label_test {}".format(transferred_predicted_label_test))     
        self.getScore("test_score_W_combined",self.test_label,predict_test_data_label_combineW)

        if self.test_score_W_Combined_n < self.test_score_W0_n:
             print("Not good nmi result !!!!!")
        if self.test_score_W_Combined_a < self.test_score_W0_a:
            print("Not good ari result !!!!!")
        print("Combine Neurons Number :{}".format(len(self.right_datas)))




        
    
