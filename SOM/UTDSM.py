"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
@author: Riley Smith
Created: 1-27-21
"""
from asyncio.windows_events import NULL
from enum import Flag
#from curses.ascii import NULL
from importlib import resources
from pickle import TRUE
from telnetlib import PRAGMA_HEARTBEAT
from zlib import DEF_BUF_SIZE
from sklearn import metrics
from scipy import spatial
import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
class UTDSM_SOM():
    """
    The 2-D, rectangular grid self-organizing map class using Numpy.
    """
    def __init__(self, som, data_train, data_test,label_train,label_test,classNum = 2):
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
        self.classNum = classNum 

        # initial cluster numbers in TDSM_SOM, which is the neuron number in som
        self.predicted_classNum= int(som.m*som.n)

        # Predicted label value convert to class class value when using specific W
        #self.PLabel_to_Tlabel_Mapping_W0 = np.zeros(self.predicted_classNum, dtype=object)
        #self.PLabel_to_Tlabel_Mapping_W1 = np.zeros(self.predicted_classNum, dtype=object)
        #self.PLabel_to_Tlabel_Mapping_WCombined = np.zeros(self.predicted_classNum, dtype=object)

        self.data_train = data_train
        self.data_test = data_test
        self.label_train = label_train
        self.test_label = label_test
        self.combinedweight = som.weights0



        self.rightdatas = []
        self.rightWs = []
     


    def _initialdatasetsize(self):

        # score of right or error data when training with different W, W1 is the W generated in each split by som
        self.right_data_score_W0  =  []
        self.right_data_score_W_combine  =  []
        self.error_data_score_W1 =  []
        self.error_data_score_W0 =  []


        #predicted lables with different W in test or train data
        self.train_W0_predicted_label = []
        self.train_W_combined_predicted_label = []
        self.test_W0_predicted_label =   []
        self.test_W_combined_predicted_label =  []

        #all the error rate for each split
        self.error_rates =   []
        self.weights =   []

       
        # The training data that each neuron can represent in each W
        # self.neuron_represent_datas =[[split_data0],[split_data1],[split_data2]]
        # split_datai = [[data that n0 represents],[data that n1 represents],[data that n2 represents]]
        # data that ni representst = [[[index1,index2],[],[index3]],[[index5,index6][index7,index8][]],[[index9,index10][][]]]
        self.neuron_represent_datas = []



    def purity_score(self,scorename, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
       
        if(scorename == "all_train_score_W0" ):
            self.all_train_score_W0 = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
        if(scorename == "all_train_score_W_Combined" ):
            self.all_train_score_W_Combined = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

        if(scorename == "right_data_score_W0" ):
            self.right_data_score_W0.append( np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
        if(scorename == "right_data_score_W_combine" ):
            self.right_data_score_W_combine.append( np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
        if(scorename == "error_data_score_W1" ):
            self.error_data_score_W1.append( np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
        if(scorename == "error_data_score_W0" ):
            self.error_data_score_W0 .append( np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))

       
        if(scorename == "test_score_W0" ):
            self.test_score_W0 = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
        if(scorename == "test_score_W_combined" ):
            self.test_score_W_combined = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)      


    def nmiScore(self,scorename, y_true, y_pred):

        if(scorename == "all_train_score_W0" ):
            self.all_train_score_W0 = normalized_mutual_info_score(y_true,y_pred)
        if(scorename == "all_train_score_W_Combined" ):
            self.all_train_score_W_Combined = normalized_mutual_info_score(y_true,y_pred)

        if(scorename == "right_data_score_W0" ):
            self.right_data_score_W0.append(normalized_mutual_info_score(y_true,y_pred))
        if(scorename == "right_data_score_W_combine" ):
            self.right_data_score_W_combine.append(normalized_mutual_info_score(y_true,y_pred))
        if(scorename == "error_data_score_W1" ):
            self.error_data_score_W1.append(normalized_mutual_info_score(y_true,y_pred))
        if(scorename == "error_data_score_W0" ):
            self.error_data_score_W0.append(normalized_mutual_info_score(y_true,y_pred))
     
        if(scorename == "test_score_W0" ):
            self.test_score_W0 = normalized_mutual_info_score(y_true,y_pred)
        if(scorename == "test_score_W_combined" ):
            self.test_score_W_combined = normalized_mutual_info_score(y_true,y_pred)

    def ariScore(self,scorename, y_true, y_pred):

        if(scorename == "all_train_score_W0" ):
            self.all_train_score_W0 = adjusted_rand_score(y_true,y_pred)
        if(scorename == "all_train_score_W_Combined" ):
            self.all_train_score_W_Combined = adjusted_rand_score(y_true,y_pred)

        if(scorename == "right_data_score_W0" ):
            self.right_data_score_W0.append(adjusted_rand_score(y_true,y_pred))
        if(scorename == "right_data_score_W_combine" ):
            self.right_data_score_W_combine.append(adjusted_rand_score(y_true,y_pred))
        if(scorename == "error_data_score_W1" ):
            self.error_data_score_W1.append(adjusted_rand_score(y_true,y_pred))
        if(scorename == "error_data_score_W0" ):
            self.error_data_score_W0.append(adjusted_rand_score(y_true,y_pred))

        if(scorename == "test_score_W0" ):
            self.test_score_W0 = adjusted_rand_score(y_true,y_pred)
        if(scorename == "test_score_W_combined" ):
            self.test_score_W_combined = adjusted_rand_score(y_true,y_pred)



    def get_indices_in_predicted_clusters(self,class_num_predicted,predicted_label,data_to_predict):
            
            """
            """
            #print("predicted_label {}".format(predicted_label))
            # class labels in each cluster = [[1,1,1,1],[2,2,2,2],[1,1]]]
            clusters_indexes = []
            clusters_datas = []
            for i in range(0,class_num_predicted):
                newlist = []
                newdatalist = []
                for idx, y in enumerate(predicted_label):     
                    # is the cluster label
                    if(y == i):
                        newlist.append(idx)  
                        newdatalist.append(data_to_predict[idx])
                clusters_indexes.append(newlist)
                clusters_datas.append(newdatalist)
            # clusters_labels = [[1,2,3,12,24],[0,4,5,6],[9,11]]  indices of data that are grouped in each clusters
            # clusters_datas = [[data in cluster 0],[data in cluster 1],[data in cluster 2]]
            j=0
            print("clusters_indexes  {}".format(clusters_indexes))
            for x in clusters_indexes:
                if x != []:
                    j =j+1
            if j ==1 :
                self.NoErrorDataExist = True
            #print("clusters_indexes {}".format(clusters_indexes))
            print("clusters_datas size {}".format(len(clusters_datas)))
            return clusters_indexes,clusters_datas
      

    def getErrorDataBasedOnErrorRate(self, error_rate,current_train_data):
        # first needs to make sure each x in current_train_data is not []
        # error_rate = [0,current_train_data.size]]
        #print("current_train_data {}".format(current_train_data))
        #current_iteration_errordata = []
        #for i in range(0 , len(current_train_data)) :
        right_data,rightDataW,errordata=self.getErrorDataRightDataInSingleNeurons(error_rate,current_train_data)
        #print("append right_data  {}".format(right_data))
        if right_data != []:
            self.rightdatas.append(right_data)
            self.rightWs.append(rightDataW)
        #current_iteration_errordata.append(errordata)
        #print("errordata size {} errordata {}".format(len(errordata),errordata))
        return errordata

    def delete_multiple_element(self,list_object, indices):
        indices = sorted(indices, reverse=True)
        for idx in indices:
            if idx < len(list_object):
                list_object.pop(idx)


    def getErrorDataRightDataInSingleNeurons(self,error_rate,current_neuron_data):
        current_neuron_data = np.array(current_neuron_data)
        self.som.fit(current_neuron_data,1)
   
        current_neuron_data_predicted_label = self.som.predict(current_neuron_data,self.som.weights1)   
     
        clustered_indexes,clustered_datas =   self.get_indices_in_predicted_clusters(self.som.weights1.shape[0],current_neuron_data_predicted_label,current_neuron_data)
        empty_list_indices = []
        j = 0
        for i in range(0,len(clustered_datas)):
            if clustered_datas[i] !=[]:
                j=j+1
            else:
                empty_list_indices.append(i)
        if j > error_rate:
            minindex = self.getMinIndex(error_rate,clustered_indexes)
            reduce_index = minindex + empty_list_indices       
            rightDataW = np.delete(self.som.weights1,reduce_index,0)
            rightdata = copy.deepcopy(clustered_datas)
            self.delete_multiple_element(rightdata, reduce_index)
           #print("clustered_datas {}".format(clustered_datas))
            errordata = [clustered_datas[index] for index in minindex ]
        elif j == error_rate:
            #no right data 
            rightDataW = []
            rightdata =  []
            errordata = clustered_datas
        else:
            maxindex = self.getMaxIndex(error_rate,clustered_indexes)
            reduce_index = maxindex + empty_list_indices        
            rightDataW = np.delete(self.som.weights1,reduce_index,0)
            rightdata =  copy.deepcopy(clustered_datas)
            self.delete_multiple_element(rightdata, reduce_index)
            errordata = [ clustered_datas[index] for index in maxindex ]
        #print("return rightdata {} ".format(rightdata))
        return rightdata, rightDataW,errordata

    def getMinIndex(self,error_rate,clustered_indexes):
        #clustered_indexes = [[indices ],[],[]]
        minIndex = []
        dataNum_list = [] #dataNum_list = [5,6,2,0] the number is data number in each cluster
        for x in clustered_indexes:
            dataNum_list.append(len(x))
        #print("dataNum_list 1 {} ".format(dataNum_list))
        max_value = np.max(dataNum_list)
        reference_value = max_value +1
        for n in dataNum_list:
            if n == 0: # n is a empty neurons and make it to be max value, so will never access to the calculation of getting min value
                dataNum_list[n] = reference_value
        min_datas_index = []
        dataNum_list = np.array(dataNum_list)
       # print("dataNum_list2 {} max_value {}".format(dataNum_list,max_value))
        for i in range(0,error_rate): 
            if np.min(dataNum_list) != reference_value:
                #*** minresult = array([m,n])  m and n is the minum value index in minresult 
                #*** Get the indices of minimum element in numpy array it return an array of array  array[0] = [m,n] the minimum element indices )
                minresult = np.where(dataNum_list == np.amin(dataNum_list)) 
                for j in range(0,len(minresult[0])):
                    if j not in min_datas_index:
                        min_datas_index.append(minresult[0][j])
                        dataNum_list[minresult[0][j]] = max_value
        #min_datas_index is a sorted list of dataNum_list min_datas_index = [4,5,6,1,0,3]  index of neurons that has data increases from neuron 4 to neuron3
        #print("min_datas_index {}".format(min_datas_index))
        for i in range(0,error_rate):           
            minIndex.append(min_datas_index[i])
        #minIndex = [4,5] when error_rate = 2
        return minIndex

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

        if Wtype == 1 :
            self.PLabel_to_Tlabel_Mapping_WCombined = predicted_label_convert_to_class_label

        if Wtype == 2 :
            self.PLabel_to_Tlabel_Mapping_W1 = predicted_label_convert_to_class_label

    def getMaxRepeatedElements(self, list):
        #print("list{}".format(list))
        #Count number of occurrences of each value in array of non-negative ints.
        counts = np.bincount(list)
       # print("counts {}".format(counts))
        #Returns the indices of the maximum values along an axis.
        #print("most common 1 {}".format(b.most_common(1)))
        return np.argmax(counts)

    def convertPredictedLabelValue(self,predicted_cluster_labels, PLable_TLabel_Mapping):
        # PLabel_CLabel_Mapping the mapping of cluster label to class label
        # PLable_TLabel_Mapping size is the som.m*som.n* stop_split_num
        print("predicted_cluster_labels {}".format(predicted_cluster_labels))
        print("PLable_TLabel_Mapping {}".format(PLable_TLabel_Mapping))
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
            print("predicted_clusters {}".format(predicted_clusters))
          
           
          
            if(Wtype == 0):
             self.getLabelMapping( predicted_clusters,0)  
              # the value in predicted_clusters are true label value       
             predicted_class_labels =  self.convertPredictedLabelValue(predicted_cluster_labels,self.PLabel_to_Tlabel_Mapping_W0)
             self.initial_current_traindata = b
             return predicted_class_labels
            
            if(Wtype == 1):   

             self.getLabelMapping( predicted_clusters,1)      
           
             predicted_class_labels =  self.convertPredictedLabelValue(predicted_cluster_labels,self.PLabel_to_Tlabel_Mapping_WCombined)
            
             return predicted_class_labels

        else:
            if(Wtype == 0):
                predicted_class_labels = self.convertPredictedLabelValue(predicted_cluster_labels,self.PLabel_to_Tlabel_Mapping_W0)
                return predicted_class_labels
            if(Wtype == 1):
                predicted_class_labels = self.convertPredictedLabelValue(predicted_cluster_labels,self.PLabel_to_Tlabel_Mapping_WCombined)
                return predicted_class_labels



    def find_empty_neurons_ineachW(self):
        self.empty_neurons = [] # index of neurons that do not represent any data   in each weight or split data
        self.non_empty_neurons = []# index of neurons that can represent some data  

        for i in range(0,len(self.neuron_represent_datas)):   # self.neuron_represent_datas is [[[],[]],[[],[]],[[]]]
            self.empty_neurons.append([])
            self.non_empty_neurons.append([])
            for j in range(0,len(self.neuron_represent_datas[i])):
                if self.neuron_represent_datas[i][j] == []:
                    self.empty_neurons[i].append(j)
                else:                  
                    self.non_empty_neurons[i].append(j)


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


    def _find_bmu_among_multipleW(self,x):
        """
        Find the index of the best matching unit for the input vector x.
        """  
        #initial distance
        #print("self.rightdatas {} ".format(self.rightdatas))
        #print("x {} self.rightdatas[0][0] {} ".format(x,self.rightdatas[0][0]))
        #*** self.rightdatas = [[[],[]],[data in rightdata2],[]]
        distance = math.dist(x,self.rightdatas[0][0][0])
        
        w_index = 0
        for i in range(0,len(self.rightdatas)):
            for j in range(0,len(self.rightdatas[i])):
                tree = spatial.KDTree(self.rightdatas[i][j])
                currentdistance = tree.query(x)[0]
                if currentdistance < distance:
                    distance = currentdistance
                    w_index = i
                                
        x_stack = np.stack([x]*(self.rightWs[w_index].shape[0]), axis=0)
        distance = np.linalg.norm((x_stack -self.rightWs[w_index]).astype(float), axis=1)
        neuronsnumber_before_bestW = 0
        for j in range (0,w_index):
            neuronsnumber_before_bestW = neuronsnumber_before_bestW + self.rightWs[j].shape[0]
        predicted_cluster_index = neuronsnumber_before_bestW +  np.argmin(distance)
        return  predicted_cluster_index

    # get predicted cluster label in all W (the neruons in W that has no data represented will be ignored)
    def predict_among_nearestW_representedData(self,X,stop_split_num,weights):
        predict_labels =[]
        for x in X:
            nearest_neuron_in_eachW = []
            #**** split_number is n but weights number is n-1 as the last split doesn't generate new weights and self.empty_neurons or self.nonempty_neurons 
            for i in range(0,stop_split_num+1):
                #get nearest neurons in each W and make sure each neurons have data to represent
                bmu_index = self._find_bmu_based_neuron_representation_ineachW(x,i,weights[i]) 
                nearest_neuron_in_eachW.append(bmu_index) 
                #nearest_neuron_in_eachW = [1,2,0..] the value is the nearest neurons index in each W
                #print("nearest W index {} in W {}".format(bmu_index, i))
            all_compared_data = []
            for j in range(0,len(nearest_neuron_in_eachW)):
                #if len(self.neuron_represent_datas[j][nearest_neuron_in_eachW[j]]) >0:
                #_find_bmu_based_neuron_representation_ineachW already make sure that len(self.neuron_represent_datas[j][nearest_neuron_in_eachW[j]]) >0 
                all_compared_data.append(self.neuron_represent_datas[j][nearest_neuron_in_eachW[j]])
            #print("all_compared_data size {}".format(len(all_compared_data)))
            best_w_index = self.getBestWAmongAllW(x,all_compared_data)
            #****predicte_label range from 0 to (split_number+1)*self.som.n*self.som.n
            predict_labels.append(self.som.n*self.som.n* best_w_index + bmu_index)

        return np.array(predict_labels)

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
    

    def predict_among_multipleW(self,X):
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
            b = self._find_bmu_among_multipleW(x)
            labels.append(b)

        # winWindexlabels, labels = np.array([ for x in X])
        # labels will be always from 0 - (m*n)*stop_split_num-1
        return np.array(labels)


    def getScore(self,scorename, y_true, y_pred, scoretype):
        if scoretype == 0:
            self.purity_score(scorename,y_true,y_pred)
        elif scoretype == 1:
            self.nmiScore(scorename,y_true,y_pred)
        elif scoretype == 2:
            self.ariScore(scorename,y_true,y_pred)

    def run(self,error_rate =1,score_type = 0):
        """
        score_type 0 purity 1 numi 2 rai
        """
        self.NoErrorDataExist = False       
        # get train and test dataset 
        self._initialdatasetsize()
        #train som to get W0
        self.som.fit(self.data_train)
        self.weights.append(self.som.weights0)
        
        self.train_W0_predicted_label = self.som.predict(self.data_train,self.som.weights0)   
        transferred_predicted_label_all_train = self.transferClusterLabelToClassLabel(True,self.som.weights0.shape[0],self.train_W0_predicted_label,self.data_train)        
        self.getScore("all_train_score_W0",self.label_train,transferred_predicted_label_all_train,score_type)

        self.test_W0_predicted_label = self.som.predict(self.data_test,self.som.weights0)   
        transferred_predicted_label_test_W0 = self.transferClusterLabelToClassLabel(False,self.som.weights0.shape[0],self.test_W0_predicted_label,self.data_test)                                    
        self.getScore("test_score_W0",self.test_label,transferred_predicted_label_test_W0,score_type)
        #initialize
        current_data_train =   self.initial_current_traindata
        #self.get_indices_in_predicted_clusters(self.predicted_classNum,self.train_W0_predicted_label,self.data_train)
 
        while(self.NoErrorDataExist != True):
            next_iteration_errordata = []
            #print("len(current_data_train) {}current_data_train  {} " .format(len(current_data_train),current_data_train))
            if len(current_data_train) == 1:
                error_datas = self.getErrorDataBasedOnErrorRate(error_rate,current_data_train[0])
                for x in error_datas:
                    next_iteration_errordata.append(x)
            else:
                for x in current_data_train:
                    if x !=[]:
                        error_datas = self.getErrorDataBasedOnErrorRate(error_rate,x)
                    for item in error_datas:
                        next_iteration_errordata.append(item)
            
            current_data_train = next_iteration_errordata
        
        #get combined W

        self.combinedweight = self.rightWs[0]
        for i in range(1,len(self.rightWs)):
            print("self.rightWs[i]{}".format(self.rightWs[i]))
            self.combinedweight =  np.row_stack((self.combinedweight, self.rightWs[i]))
        print("self.combinedweight{}".format(self.combinedweight))
        self.train_W_combined_predicted_label = self.som.predict(self.data_train,self.combinedweight)    
           
        print("self.combinedweight.shape[0] {}".format(self.combinedweight.shape[0]))
        self.train_W_combined_predicted_label = self.predict_among_multipleW(self.data_train)    
        print("train_W_combined_predicted_label {}".format(self.train_W_combined_predicted_label))
        transferred_predicted_label_train_WCombine = self.transferClusterLabelToClassLabel(True,self.combinedweight.shape[0],self.train_W_combined_predicted_label,self.data_train,Wtype = 1)   
        self.getScore("all_train_score_W_Combined",self.label_train,transferred_predicted_label_train_WCombine,score_type)       

        self.test_W_combined_predicted_label = self.predict_among_multipleW(self.data_test)   


        transferred_predicted_label_test = self.transferClusterLabelToClassLabel(False,self.combinedweight.shape[0],self.test_W_combined_predicted_label,self.data_test,Wtype = 1)   
        self.getScore("test_score_W_combined",self.test_label,transferred_predicted_label_test,score_type)

        if score_type == 0:
            print("Purity Score")   
        elif score_type == 1:
            print("NMI Score")
        elif score_type == 2:
            print("ARI Score")     

        
        print("train_score_W0 : {}".format( self.all_train_score_W0))
        print("train_score_W\': {}".format(self.all_train_score_W_Combined))
        print("test_score_W0 : {}".format( self.test_score_W0))
        print("test_score_W\': {}".format(self.test_score_W_combined))

     
        


        
    
