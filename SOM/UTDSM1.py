"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
#TODO  only do the intra community 
find intra communiyt in each neuron memberships and do the whole mapping and retest 
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

class UTDSM_SOM():
    """
    The 2-D, rectangular grid self-organizing map class using Numpy.
    """
    def __init__(self, som, data_train, data_test,label_train,label_test, elbow_num, x,y):
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
        self.row = x
        self.column = y
        # initial cluster numbers in TDSM_SOM, which is the neuron number in som
        self.predicted_classNum= int(som.m*som.n)

        # Predicted label value convert to class class value when using specific W
        #self.PLabel_to_Tlabel_Mapping_W0 = np.zeros(self.predicted_classNum, dtype=object)
        #self.PLabel_to_Tlabel_Mapping_W1 = np.zeros(self.predicted_classNum, dtype=object)
        #self.PLabel_to_Tlabel_Mapping_WCombined = np.zeros(self.predicted_classNum, dtype=object)

        self.data_train = data_train
       # print("self.data_train  initial {}".format(len(self.data_train )))
        self.data_test = data_test
        self.train_label = label_train
        self.train_label = self.train_label.astype(int)
       # print(self.train_label)
        self.test_label = label_test

        self.all_split_datas = []
        self.all_split_datas_indexes = []

        self.outliner_clusters = []
        self.newmappings = []   
        self.all_left_train_data_label = np.arange(len(self.train_label))
       # print("self.all_left_train_data_label  initial {}".format(len(self.all_left_train_data_label )))
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
            print("self.rest_data_W0_n {}".format(self.rest_data_W0_n))


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
            print("self.rest_data_W0_a {}".format(self.rest_data_W0_a))
            # if  self.test_score_W_Combined_a >= self.test_score_W0_a :
            #     print("test_score_W_Combined increased in ari")
            # else: print(-1)

    
    def get_indices_in_predicted_clusters(self,class_num_predicted,predicted_label):
            
            """
            class_label is the true class label
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
        #print("gargetgroup_index {} ".format(gargetgroup_index))
        for idx in gargetgroup_index:     
            distance = np.linalg.norm((self.data_train[idx] - cluster_center).astype(float))
            if distance >0:
                #print("idx {} distance  {}".format(idx,distance))
                sorted_data_dict[idx] = distance
            #if distance == 0:
              #  print("zero distcance for data 1 idx {}".format(idx))       
       # print("sorted_dict 1{} ".format(sorted_data_dict))
        sorted_dict = dict(sorted(sorted_data_dict.items(), key=operator.itemgetter(1),reverse=True))
       # print("sorted_dict 2{} ".format(sorted_dict))
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
            #print("key {}".format(key))
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
        self.purity_score(scorename,y_true,y_pred)
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
        self.som.fit(self.data_train)
        weight0 = self.som.weights0

        # the new cluster data generated by finding community nodes
        newclustered_datas = []
        newclustered_datas_index = []

        self.train_W0_predicted_label = self.som.predict(self.data_train,weight0)   
        # will be used in rest data test score, the data left after remove the newclustered_datas
        self.rest_data_predicted_label_dict ={}

        for idx, y in enumerate( self.train_W0_predicted_label): 
            self.rest_data_predicted_label_dict[idx] = y
           # print(idx)

        predicted_clusters, current_clustered_datas = self.get_indices_in_predicted_clusters(self.som.m*self.som.n, self.train_W0_predicted_label)   
        """predicted_clusters  [[23,31,21,2],[1,3,5,76],[45,5,12]] index in data_train in some situation, it will have [] """
        self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters) ,0)  
        # the value in predicted_clusters are true label value       
        transferred_predicted_label_train_W0 =  self.convertPredictedLabelValue(self.train_W0_predicted_label,self.PLabel_to_Tlabel_Mapping_W0)      
        #self.all_split_datas new generated clusters by finding community nodes
        if self.all_split_datas == []:

            self.getScore("all_train_score_W0",self.train_label,transferred_predicted_label_train_W0)
            self.test_W0_predicted_label = self.som.predict(self.data_test,weight0)   
            transferred_predicted_label_test_W0 = self.transferClusterLabelToClassLabel(False,weight0.shape[0],self.test_W0_predicted_label,self.data_test)    
            self.getScore("test_score_W0",self.test_label,transferred_predicted_label_test_W0)
        
       
        
        searched_datas = copy.deepcopy(current_clustered_datas)
        searched_datas = array(searched_datas).tolist()
      
        for i in range(0,len(searched_datas)):
            # get discasending fartheset node in current clustered_data
            sorted_dict = self.split_data(predicted_clusters[i],  weight0[i])   
            while len(sorted_dict) >0:    
            
                farthest_intra_node_index = self.getfarthest_intra_node_index(sorted_dict)
                current_check_node = self.data_train[farthest_intra_node_index]
               # print("farthest_intra_node_index  {}".format(farthest_intra_node_index))
                del sorted_dict[farthest_intra_node_index]
                #*** check if current_check_node is in other community
                already_in_community = False
   
                for k in range(0,len(newclustered_datas)):
                    if  current_check_node in np.array(newclustered_datas[k]):
                     already_in_community = True        
                   #  print("already in community ! {}".format(farthest_intra_node_index))      
                     break
                
                if already_in_community :
                    continue


                newclustered_data =[]
                new_predicted_clusters =[]
                # get inter community nodes
                for j in range(0,len(searched_datas)):
                    if j != i:
                        sorted_dict_inter, distances_inter =  self.get_allnode_distance_in_a_group(current_check_node,predicted_clusters[j],weight0[j])
                        a,b = self.get_inter_community_nodes(sorted_dict_inter,distances_inter)
                        if a != []:
                            for item in a:
                                newclustered_data.append(item)
                      
                        if b != []:
                            for item in b:
                                predicted_clusters[j].remove(item)
                                new_predicted_clusters.append(item)
                                del self.rest_data_predicted_label_dict[item]
                                #print("put item 1 in community {}".format(item))
                            if predicted_clusters[j] != []: 
                               current_clustered_datas[j] = list(self.data_train[predicted_clusters[j]])
                            else:
                                current_clustered_datas[j] =[]
 
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
                        del self.rest_data_predicted_label_dict[item]
                        #print("put item 2 in community {}".format(item))
                  
                    current_clustered_datas[i] = np.array(current_clustered_datas[i])                  
                    if predicted_clusters[i] != [] :
                        current_clustered_datas[i] = list(self.data_train[predicted_clusters[i]] )
                    else:
                        current_clustered_datas[i] =[]                      
                 # add current data to the community generated
                if a!=[] or a1!=[]:
                     newclustered_data.append(current_check_node)
                     new_predicted_clusters.append(farthest_intra_node_index)
                     #print("rest_data_predicted_label_dict {}".format(self.rest_data_predicted_label_dict))
                     #print("put farthest_intra_node_index  node  in commuity{}".format(farthest_intra_node_index))
                     del self.rest_data_predicted_label_dict[farthest_intra_node_index]
                    
                     #*** remove current_check_node
                     predicted_clusters[i].remove(farthest_intra_node_index)
                     current_clustered_datas[i] = list(self.data_train[predicted_clusters[i]] )

            

                if newclustered_data != []:
                    newclustered_datas.append(newclustered_data)
                    newclustered_datas_index.append(new_predicted_clusters)
                    #print("add new community cluster  !!!!!!!!")




        for item in current_clustered_datas:
            if item!=[]:
                self.all_split_datas.append(item)
        for item in newclustered_datas:
            if item!=[]:
                self.all_split_datas.append(item)


        
        for item in predicted_clusters:
            if item!=[]:
                self.all_split_datas_indexes.append(item)
        for item in newclustered_datas_index:
            if item!=[]:
                self.all_split_datas_indexes.append(item)

        
        
        #for i in range(0,len(self.all_split_datas_indexes)):
        #    print ("cluster index  number {}".format(len(self.all_split_datas_indexes[i])))
        #    self.drawnormaldistributonplot(self.all_split_datas_indexes[i],i,"blue")

       # print("len self.all_split_datas 1 {}".format(len(self.all_split_datas)))
        self.delete_multiple_element(self.all_split_datas,self.outliner_clusters)
        self.delete_multiple_element(self.all_split_datas_indexes,self.outliner_clusters)
       # print("len self.all_split_datas 2 {}".format(len(self.all_split_datas)))

        #self.test_optimized_clusters()
        
        self.test_combineW()         
      

    def delete_multiple_element(self,list_object, indices):
        indices = sorted(indices, reverse=True)
        for idx in indices:
         if idx < len(list_object):
                list_object.pop(idx)
    
    def test_optimized_clusters(self):


        self.getLabelMapping( self.get_mapped_class_in_clusters(self.all_split_datas_indexes) ,0)  

        for i in range(0,len(self.all_split_datas_indexes)):
            generated_data_true_label = []
            generated_data_predict_label = []
            for item in self.all_split_datas_indexes[i]:
                generated_data_predict_label.append(i)
                generated_data_true_label.append(self.train_label[item])
           # print("optimized neurons test {}".format(i))
            self.all_split_datas[i] = np.array(self.all_split_datas[i])
           # print("generated_data_predict_label {}".format(generated_data_predict_label))
           #print("generated_data_true_label {}".format(generated_data_true_label))
            generated_data_predicted_label  =  self.convertPredictedLabelValue(generated_data_predict_label,self.PLabel_to_Tlabel_Mapping_W0)   
            self.getScore("rest_score_W0",generated_data_true_label,generated_data_predicted_label)


    def test_new_generated_data(self,generated_data_true_label,current_clustered_datas,weight0):
        self.newgenerated_W0_predicted_label = self.som.predict(current_clustered_datas,weight0)   

        #self.getScore("rest_score_W0",generated_data_true_label,new_generated_data_W0)
        predicted_clusters, current_clustered_datas = self.get_indices_in_predicted_clusters(self.som.m*self.som.n, self.newgenerated_W0_predicted_label)   
        # predicted_clusters  [[23,31,21,2],[1,3,5,76],[45,5,12]] index in data_train in some situation, it will have [] 
        self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters) ,0)  
        # the value in predicted_clusters are true label value       
        new_generated_data_W0 =  self.convertPredictedLabelValue(self.newgenerated_W0_predicted_label,self.PLabel_to_Tlabel_Mapping_W0)   
         #restdata_W0_predicted_label = self.som.predict(data_rest,weight0)   

        self.getScore("rest_score_W0",generated_data_true_label,new_generated_data_W0)

    def test_rest_data_one_time(self,predicted_clusters,current_clustered_datas,weight0):
     #   print("predicted_clusters {}".format(predicted_clusters))
       # self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters) ,0)  
        data_rest = []
        for item in current_clustered_datas:
            if item != []:
                for x in item:
                 data_rest.append(x)
        data_rest = np.array(data_rest)
        rest_data_true_label = []
        for item in predicted_clusters:
            if item != []:
                for x in item:
                 rest_data_true_label.append(self.train_label[x])
        print("rest_data_true_label len {}".format(len(rest_data_true_label)))
        """
        Do not need to predict again for the rest _data as each predict will cause different result
        """
        restdata_W0_predicted_label =[]
        for idx, item in self.rest_data_predicted_label_dict.items():
            restdata_W0_predicted_label.append(item)
           
        #print("len  self.rest_data_predicted_label_dict {}".format(len(self.rest_data_predicted_label_dict)))
        print("restdata_W0_predicted_label  {}".format((restdata_W0_predicted_label)))
        print("rest_data_true_label  {}".format((rest_data_true_label)))
        #restdata_W0_predicted_label = self.som.predict(data_rest,weight0)   
        transferred_predicted_label_restdata_W0 =  self.convertPredictedLabelValue(restdata_W0_predicted_label,self.PLabel_to_Tlabel_Mapping_W0)   
        self.getScore("rest_score_W0",rest_data_true_label,transferred_predicted_label_restdata_W0)

    def test_rest_data(self,predicted_clusters,current_clustered_datas,weight0):
            self.rest_data_true_label = []
            self.rest_data =[]
            self.rest_data_indexes =[]
            #remapping
            self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters) ,0)  
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
        self.train_W_combined_predicted_label = self.predict_based_splitdata(self.data_train,self.all_split_datas)    

     
        # predicted_clusters = [[2,4,14,23,21],[1,43,24,5]]
       # print("current train data generated predicted_clusters len {}".format(len(predicted_clusters)))
        # predicted_clusters  [[23,31,21,2],[1,3,5,76],[45,5,12]] index in data_train
      
        #print("self.train_W_combined_predicted_label {}".format(self.train_W_combined_predicted_label))  
        #*** needs to use self.combinedweight.shape[0] cannot use self.som.m*self.som.n as neruons is changed
        predicted_clusters,current_clustered_datas = self.get_indices_in_predicted_clusters(len(self.all_split_datas), self.train_W_combined_predicted_label)  
        self.getLabelMapping(self.get_mapped_class_in_clusters(predicted_clusters),1)  
        # the value in predicted_clusters are true label value
        # 
        #predict_all_data_label_combineW = []
        #for  item in self.train_W_combined_predicted_label:
        #    predict_all_data_label_combineW.append(self.newmappings[item])

        transferred_predicted_label_train_WCombine =  self.transferClusterLabelToClassLabel(False,len(self.all_split_datas),self.train_W_combined_predicted_label,self.data_test,Wtype = 1)                 
        self.getScore("all_train_score_W_Combined",self.train_label,transferred_predicted_label_train_WCombine)      


        self.test_W_combined_predicted_label = self.predict_based_splitdata(self.data_test,self.all_split_datas)   

        #predict_test_data_label_combineW = []
        #for  item in self.test_W_combined_predicted_label:
        #    predict_test_data_label_combineW.append(self.newmappings[item])


        transferred_predicted_label_test = self.transferClusterLabelToClassLabel(False,len(self.all_split_datas),self.test_W_combined_predicted_label,self.data_test,Wtype = 1)   
        #print("transferred_predicted_label_test {}".format(transferred_predicted_label_test))     
        self.getScore("test_score_W_combined",self.test_label,transferred_predicted_label_test)

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
                data = self.data_train[item]
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
                a = np.min(self.data_train[:, i])
                b = np.max(self.data_train[:, i])
                #print("a {}  b{}".format(a, b))
                if self.row != 1:
                    axs[m,n].hist(x_axis, bins='auto', range =[a,b], color = color)
                else: axs[n].hist(x_axis, bins='auto', range =[a,b], color = color)
                
                #print("x_axis {}".format(x_axis))
            plt.show()
            print("****************")
        else:
            self.outliner_clusters.append(i)

    
    def cluster_data_based_true_label(self):
        self.all_true_clustered_datas_indexes = []
        for i in range(0,self.predicted_classNum):
            grouped_indexes = []
            for idx, label in enumerate(self.train_label):
                if  label == i:
                    grouped_indexes.append(idx)
            self.all_true_clustered_datas_indexes.append(grouped_indexes)
        


