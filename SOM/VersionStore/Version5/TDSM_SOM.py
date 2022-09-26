"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
@author: Riley Smith
Created: 1-27-21
"""
from asyncio.windows_events import NULL
#from curses.ascii import NULL
from importlib import resources
from pickle import TRUE
from telnetlib import PRAGMA_HEARTBEAT
from sklearn import metrics
from scipy import spatial
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
class TDSM_SOM():
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
        self.PLabel_to_Tlabel_Mapping_W0 = np.zeros(self.predicted_classNum, dtype=object)
        self.PLabel_to_Tlabel_Mapping_W1 = np.zeros(self.predicted_classNum, dtype=object)
        self.PLabel_to_Tlabel_Mapping_WCombined = np.zeros(self.predicted_classNum, dtype=object)

        self.data_train = data_train
        self.data_test = data_test
        self.label_train = label_train
        self.test_label = label_test
        self.combinedweight = som.weights0


    def _initialdatasetsize(self):

        # score of right or error data when training with different W, W1 is the W generated in each split by som
        self.right_data_score_W0_p  =  []
        self.right_data_score_W_combine_p  =  []
        self.error_data_score_W1_p =  []
        self.error_data_score_W0_p =  []


        self.right_data_score_W0_n  =  []
        self.right_data_score_W_combine_n  =  []
        self.error_data_score_W1_n =  []
        self.error_data_score_W0_n =  []


        self.right_data_score_W0_a  =  []
        self.right_data_score_W_combine_a  =  []
        self.error_data_score_W1_a =  []
        self.error_data_score_W0_a =  []
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
            self.all_train_score_W0_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
        if(scorename == "all_train_score_W_Combined" ):
            self.all_train_score_W_Combined_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

        if(scorename == "right_data_score_W0" ):
            self.right_data_score_W0_p.append( np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
        if(scorename == "right_data_score_W_combine" ):
            self.right_data_score_W_combine_p.append( np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
        if(scorename == "error_data_score_W1" ):
            self.error_data_score_W1_p.append( np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
        if(scorename == "error_data_score_W0" ):
            self.error_data_score_W0_p.append( np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))

       
        if(scorename == "test_score_W0" ):
            self.test_score_W0_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
        if(scorename == "test_score_W_combined" ):
            self.test_score_W_combined_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)      


    def nmiScore(self,scorename, y_true, y_pred):

        if(scorename == "all_train_score_W0" ):
            self.all_train_score_W0_n = normalized_mutual_info_score(y_true,y_pred)
        if(scorename == "all_train_score_W_Combined" ):
            self.all_train_score_W_Combined_n = normalized_mutual_info_score(y_true,y_pred)

        if(scorename == "right_data_score_W0" ):
            self.right_data_score_W0_n.append(normalized_mutual_info_score(y_true,y_pred))
        if(scorename == "right_data_score_W_combine" ):
            self.right_data_score_W_combine_n.append(normalized_mutual_info_score(y_true,y_pred))
        if(scorename == "error_data_score_W1" ):
            self.error_data_score_W1_n.append(normalized_mutual_info_score(y_true,y_pred))
        if(scorename == "error_data_score_W0" ):
            self.error_data_score_W0_n.append(normalized_mutual_info_score(y_true,y_pred))
     
        if(scorename == "test_score_W0" ):
            self.test_score_W0_n = normalized_mutual_info_score(y_true,y_pred)
        if(scorename == "test_score_W_combined" ):
            self.test_score_W_combined_n = normalized_mutual_info_score(y_true,y_pred)

    def ariScore(self,scorename, y_true, y_pred):

        if(scorename == "all_train_score_W0" ):
            self.all_train_score_W0_a = adjusted_rand_score(y_true,y_pred)
        if(scorename == "all_train_score_W_Combined" ):
            self.all_train_score_W_Combined_a = adjusted_rand_score(y_true,y_pred)

        if(scorename == "right_data_score_W0" ):
            self.right_data_score_W0_a.append(adjusted_rand_score(y_true,y_pred))
        if(scorename == "right_data_score_W_combine" ):
            self.right_data_score_W_combine_a.append(adjusted_rand_score(y_true,y_pred))
        if(scorename == "error_data_score_W1" ):
            self.error_data_score_W1_a.append(adjusted_rand_score(y_true,y_pred))
        if(scorename == "error_data_score_W0" ):
            self.error_data_score_W0_a.append(adjusted_rand_score(y_true,y_pred))

        if(scorename == "test_score_W0" ):
            self.test_score_W0_a = adjusted_rand_score(y_true,y_pred)
        if(scorename == "test_score_W_combined" ):
            self.test_score_W_combined_a = adjusted_rand_score(y_true,y_pred)

    def get_indices_in_clusters(self,class_num_predicted,category,predicted_label,split_number):
            
            """
            show element indice in each cluster [[2,8,1,],[4,5,6],[0,3,9]], value here is the element indice in each data set
            class_num: total cluster Number in training data, the neuron number in som
            category : indicates which data set need to be loaded 
                     0: all training data 
                     1: right data in each split
                     2: error data in each split
                     3: test data
            """
            # class labels in each cluster = [[1,1,1,1],[2,2,2,2],[1,1]]]
            clusters_labels = []
            # the training data in each clusterclusters_data  = [[ data1, data2, data3],[data3, data5],[data6,data7]]
            clusters_data = [] 
            for i in range(0,class_num_predicted):
                newlist = []
                newdatalist = []

                # ************ this idx is not the indice in train_data   it is the indice in predicted_label such as  idx:y-> 0:1, 1:2,3:1, 4:1      
                # the indices of  predicted_label is the same with related category label set:  label_trains,label_trains_right_data or label_trains_error_data
                # idx start from 0,1,2,3....  y is the predicted class value
                for idx, y in enumerate(predicted_label):     
                    # is the cluster label
                    if(y == i):
                        if(category == 0):
                            # if category ==0 predicted_label is generatd by train data without resampling, the whole training data, so idx is the same with label_train[idx]
                            #self.label_trains[idx] is the true label value 
                            newlist.append(self.label_train[idx]) 
                        if(category == 1):
                            newlist.append(self.right_data_labels[split_number][idx]) 
                            newdatalist.append(self.right_datas[split_number][idx])
                        if(category == 2):
                            newlist.append(self.error_data_labels[split_number][idx])
                        if(category == 3):
                            newlist.append(self.test_label[idx])        
               
                clusters_labels.append(newlist) 
                if(category == 1):
                    clusters_data.append(newdatalist)
          
            # [[indices of cluster 0],[indices of cluster 1],[indices of cluster 2 ]...]
            #*** include such situation: [[],[],[indices of cluster 2]], some clusters have no data

            if(category == 1):
               #initialize self.neuron_represent_datas
              # print("self.neuron_represent_datas  before {}".format(self.neuron_represent_datas))
               self.neuron_represent_datas.append([])
              # print("self.neuron_represent_datas  after {}".format(self.neuron_represent_datas))
               self.neuron_represent_datas[split_number] = clusters_data
               #print("self.neuron_represent_datas  after2 {}".format(self.neuron_represent_datas))
            return clusters_labels


    def getErrorDataIndicesFromPredictedLabels(self,class_label,predicted_class_label):
        """
        get the error data indices in the training data from predicted labels, the value in predicted_class_label is the class label
        
        """
        errordata_indices =[]
        for i in range(0,class_label.size):        
            if(class_label[i]!= predicted_class_label[i] ):
                errordata_indices.append(i)
        return errordata_indices        
      
          
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
        for i in range(0,len(predicted_cluster_labels)):
            predicted_cluster_value =  predicted_cluster_labels[i]
            predicted_cluster_labels[i] = PLable_TLabel_Mapping[predicted_cluster_value]      

  
        return predicted_cluster_labels


    def transferClusterLabelToClassLabel(self,category,predicted_cluster_labels, mapping_cluster_class_values = False, Wtype = 0,split_number = 0): 
        """
         winW_indexes : the W that needs to use
         mapping_cluster_class_values whether to map cluster label and class label or not
         train_counter: split number
         Wtype: the tyee of W
                0:W0
                1:W1
                2:WCombine

        """      
        if(mapping_cluster_class_values == True):
           
            predicted_clusters= self.get_indices_in_clusters(self.predicted_classNum,category,predicted_cluster_labels,split_number)         
            self.getLabelMapping( predicted_clusters,Wtype)  
            
            if(Wtype == 0):
                # the value in predicted_clusters are true label value       
                #print("predicted_cluster_labels before {}" .format(predicted_cluster_labels))           
                predicted_class_labels =  self.convertPredictedLabelValue(predicted_cluster_labels,self.PLabel_to_Tlabel_Mapping_W0)
                #print("predicted_cluster_labels before {}" .format(predicted_class_labels))    
                return predicted_class_labels
            
            if(Wtype == 1):       
                predicted_class_labels =  self.convertPredictedLabelValue(predicted_cluster_labels,self.PLabel_to_Tlabel_Mapping_W1)
                # update PLabel_to_Tlabel_Mapping_WCombined
                if(split_number == 0):
                    self.PLabel_to_Tlabel_Mapping_WCombined = np.concatenate((self.PLabel_to_Tlabel_Mapping_W0 , self.PLabel_to_Tlabel_Mapping_W1), axis = 0)     
                else:
                    self.PLabel_to_Tlabel_Mapping_WCombined = np.concatenate((self.PLabel_to_Tlabel_Mapping_WCombined , self.PLabel_to_Tlabel_Mapping_W1), axis = 0)        
                
                # PLabel_to_Tlabel_Mapping_WCombined = [L0,L1,...Ln-1] n is the nummber of neurons, L0 is the true label value
                return predicted_class_labels

        else:
            # do not map , just transfer
            if(Wtype == 2): 
                #print("self.PLabel_to_Tlabel_Mapping_WCombined {}" .format(self.PLabel_to_Tlabel_Mapping_WCombined))
                predicted_class_labels = self.convertPredictedLabelValue(predicted_cluster_labels,self.PLabel_to_Tlabel_Mapping_WCombined)
                return predicted_class_labels
            if(Wtype == 0):
                predicted_class_labels = self.convertPredictedLabelValue(predicted_cluster_labels,self.PLabel_to_Tlabel_Mapping_W0)
                return predicted_class_labels
            if(Wtype == 1):
                predicted_class_labels = self.convertPredictedLabelValue(predicted_cluster_labels,self.PLabel_to_Tlabel_Mapping_W1)
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


    def _find_bmu_among_multipleW(self,x,stop_split_num, Split_Data, Weights):
        """
        Find the index of the best matching unit for the input vector x.
        """  
        #initial distance

        distance = math.dist(x,Split_Data[0][0])
        w_index = 0
        for i in range(0,stop_split_num +1):
            tree = spatial.KDTree(Split_Data[i])
            currentdistance = tree.query(x)[0]
            if currentdistance < distance:
                distance = currentdistance
                w_index = i
                                
        x_stack = np.stack([x]*(self.som.m*self.som.n), axis=0)
        distance = np.linalg.norm((x_stack - Weights[w_index]).astype(float), axis=1)
        predicted_cluster_index = self.som.m*self.som.n* w_index +  np.argmin(distance)
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
    

    def predict_among_multipleW(self,X,stop_split_num,Train_Split_Data, weights):
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
            b = self._find_bmu_among_multipleW(x,stop_split_num,Train_Split_Data,weights)
            labels.append(b)

        # winWindexlabels, labels = np.array([ for x in X])
        # labels will be always from 0 - (m*n)*stop_split_num-1

        return np.array(labels)

    def randomly_choose_error_data(self,noisy_list):
            newlist = []
            for x in noisy_list:
                newlist.append(x)
            newlist = random.sample(newlist, int(len(newlist)))
            
            return newlist

    def get_subset(self,reduced_indices,X,category = 0, split_number = 0):

            if(category == 0): 
                if(split_number == 0):          
                    self.right_datas= np.array([np.delete(X,reduced_indices, axis=0)], dtype=object)                              
                else:
                    self.right_datas = self.combineTwoRaggedArray(self.right_datas,np.array(np.delete(X,reduced_indices, axis=0)))               
                    
            if(category == 1): 
                if(split_number == 0):               
                    self.right_data_labels = np.array([np.delete(X,reduced_indices, axis=0)], dtype=object)
                else:
                  
                    self.right_data_labels = self.combineTwoRaggedArray(self.right_data_labels,np.array(np.delete(X,reduced_indices, axis=0)))
           
            return 


    def combineTwoRaggedArray(self,A,B):
        #print("A len {}".format(len(A)))
        #print("B {}".format(B))
        newlist = list(A)
        #print("newlist0 {}".format(newlist))
        newlist.append(B)
        #print("newlist1 {}".format(newlist))
        #*** is A is self.variable then must return A and change A in code outside the function 
        A = np.array(newlist)
       # print("A len {}".format(len(A)))
        return A


    def getScore(self,scorename, y_true, y_pred):

        self.purity_score(scorename,y_true,y_pred)
        self.nmiScore(scorename,y_true,y_pred)
        self.ariScore(scorename,y_true,y_pred)

    def run(self):
        """
        score_type 0 purity 1 numi 2 rai
        """
        hasNoErorData = False
        current_split_number = 0             
        # get train and test dataset 
        self._initialdatasetsize()
        #train som to get W0
        self.som.fit(self.data_train)
        self.weights.append(self.som.weights0)
        
        self.train_W0_predicted_label = self.som.predict(self.data_train,self.som.weights0)   
        #mapping_cluster_class_values = true, get W0 mapping 
        transferred_predicted_label_all_train = self.transferClusterLabelToClassLabel(0,self.train_W0_predicted_label,True,split_number=current_split_number)
        
     

        self.getScore("all_train_score_W0",self.label_train,transferred_predicted_label_all_train)

        self.test_W0_predicted_label = self.som.predict(self.data_test,self.som.weights0)   
        transferred_predicted_label_test_W0 = self.transferClusterLabelToClassLabel(3,self.test_W0_predicted_label,split_number=current_split_number)                                    
        self.getScore("test_score_W0",self.test_label,transferred_predicted_label_test_W0)

        #initialize
        current_data_train = self.data_train
        current_label_train = self.label_train
        current_transferred_predicted_label = transferred_predicted_label_all_train


        while(hasNoErorData != True):
            self.error_lists = self.getErrorDataIndicesFromPredictedLabels(current_label_train,current_transferred_predicted_label)
            #**** when self.error_lists==0 means come to the last rounas as generate self.error_rates[n] first and then n+1 so when comes to self.error_lists==0 , no need to add , otherwise will have  self.error_rates[n+1] with split_num = n
            if(self.error_lists!=[]):
                self.error_rates.append(len(self.error_lists)/len(self.data_train)) 
            
            if current_split_number == 0:
                if len(self.error_lists) == 0:
                    print("W0 can represent the training data, no need to split!")
                    break
            else:
                if(self.error_lists ==[]):
                    hasNoErorData = True
                    # add train_data to  self.data_train_right_datas
                    self.right_datas = self.combineTwoRaggedArray(self.right_datas,current_data_train)
                    # this training has not been finished but current_split_number already +1, so needs  to reduce back
                    print(" NO Error Data, Finish Training!")
                    # *** split_num is the same with self.right_datas[split_num], in the last state right_datas size has increased so although is not training , but the split number is used 
                    #*** so current_split_number do not need to recuduce 1
                    self.split_num = current_split_number 
                    print("total split_number {}".format(self.split_num))
                    break
               
            #if(current_split_number == max_split_number):
            #   print(" Max Split Time !")
            #   current_split_number = current_split_number-1
            #   break

            #get samples of error data, used in cross validation
            #reduced_indices = self.randomly_choose_error_data(self.error_lists)         
           
            #*********** make it sotred, so when nptake error data it will be also from small indices to big indices, then can compared with label_error_data
            reduced_indices_sorted = np.sort(self.error_lists)
          
            self.get_subset(reduced_indices_sorted,current_data_train,0,current_split_number)  #get train_right_datas
            self.get_subset(reduced_indices_sorted,current_label_train,1,current_split_number)  #get label_train_right_datas
           
            #_________________train right data to see result
                
            if(current_split_number == 0):
                self.rightdata_W0_predicted_labels = np.array([self.som.predict(self.right_datas[0],self.som.weights0)], dtype=object)
            
                transferred_predicted_label_right_data =  self.transferClusterLabelToClassLabel(1,self.rightdata_W0_predicted_labels[current_split_number],split_number=current_split_number)
            else:
                #**** as label data is each element is one dimisision , so conbineTwoRaggedArray do not need to use np.array([element]) only use np.array(element) will be OK
                #print(" self.som.predict(self.right_datas[current_split_number],self.som.weights1) {}".format(self.som.predict(self.right_datas[current_split_number],self.som.weights1)))
                self.rightdata_W0_predicted_labels = self.combineTwoRaggedArray(self.rightdata_W0_predicted_labels,self.som.predict(self.right_datas[current_split_number],self.som.weights1))

                #print(" self.rightdata_W0_predicted_labels {}".format( self.rightdata_W0_predicted_labels[current_split_number]))
                # use W1 as W0
                transferred_predicted_label_right_data =  self.transferClusterLabelToClassLabel(1,self.rightdata_W0_predicted_labels[current_split_number],Wtype = 1,split_number=current_split_number)
            
            # get self.neuron_represent_datas
            self.get_indices_in_clusters(self.predicted_classNum,1,self.rightdata_W0_predicted_labels[current_split_number],current_split_number)
            self.find_empty_neurons_ineachW()
            
            self.getScore("right_data_score_W0",self.right_data_labels[current_split_number],transferred_predicted_label_right_data)
           
            print("right_data{}_score_W0 purity {} ".format(current_split_number,self.right_data_score_W0_p[current_split_number]))
            print("right_data{}_score_W0 nmi {} ".format(current_split_number,self.right_data_score_W0_n[current_split_number]))
            print("right_data{}_score_W0 api {} ".format(current_split_number,self.right_data_score_W0_a[current_split_number]))
            
            if(current_split_number == 0):
                self.data_train_error_datas =  np.array([np.take(current_data_train, reduced_indices_sorted,axis=0)], dtype=object)
                self.error_data_labels =  np.array([np.take(current_label_train, reduced_indices_sorted,axis=0)], dtype=object)
            else:
                self.data_train_error_datas = self.combineTwoRaggedArray(self.data_train_error_datas,np.array(np.take(current_data_train, reduced_indices_sorted,axis=0)))
                self.error_data_labels = self.combineTwoRaggedArray(self.error_data_labels,np.array(np.take(current_label_train, reduced_indices_sorted,axis=0)))




            #______________________get weights3 : the weight in error dataset
          
            self.som.fit(self.data_train_error_datas[current_split_number],1)
            #*** in looop n generate W(n+1)
            self.weights.append(self.som.weights1)
   
            if(current_split_number == 0):
                self.train_error_W1_predicted_labels = np.array([self.som.predict(self.data_train_error_datas[0],self.som.weights1)], dtype=object)
            else:
               self.train_error_W1_predicted_labels = self.combineTwoRaggedArray(self.train_error_W1_predicted_labels,self.som.predict(self.data_train_error_datas[current_split_number],self.som.weights1))
               
            #mapping_cluster_class_values = True, each W1 needs to do the mapping
            transferred_predicted_label_error_data = self.transferClusterLabelToClassLabel(2,self.train_error_W1_predicted_labels[current_split_number],True,1,split_number=current_split_number)
            
            #________ update current_normalized_predicted_label_train used in next loop
            current_transferred_predicted_label = transferred_predicted_label_error_data
           
           
    
            self.getScore("error_data_score_W1",self.error_data_labels[current_split_number],transferred_predicted_label_error_data)
            print("error_data{}_score_W1 purity {} ".format(current_split_number,self.error_data_score_W1_p[current_split_number]))
            print("error_data{}_score_W1 nmi {} ".format(current_split_number,self.error_data_score_W1_n[current_split_number]))
            print("error_data{}_score_W1 api {} ".format(current_split_number,self.error_data_score_W1_a[current_split_number]))
            
            if(current_split_number == 0):
                self.train_error_W0_predicted_labels = np.array([self.som.predict(self.data_train_error_datas[0],self.som.weights0)], dtype=object)
                
            else:
                self.train_error_W0_predicted_labels = self.combineTwoRaggedArray(self.train_error_W0_predicted_labels,self.som.predict(self.data_train_error_datas[current_split_number],self.som.weights1))
            
            transferred_predicted_label_error_data_W0 = self.transferClusterLabelToClassLabel(2,self.train_error_W0_predicted_labels[current_split_number],split_number=current_split_number)                   

            self.getScore("error_data_score_W0",self.error_data_labels[current_split_number],transferred_predicted_label_error_data_W0)

            print("error_data{}_score_W0 purity {}".format(current_split_number,self.error_data_score_W0_p[current_split_number]))
            print("error_data{}_score_W0 nmi {}".format(current_split_number,self.error_data_score_W0_n[current_split_number]))
            print("error_data{}_score_W0 api {}".format(current_split_number,self.error_data_score_W0_a[current_split_number]))

            if(self.error_data_score_W0_p[current_split_number] == 1):
                print("Error data can be represented by W0 purity {}".format(current_split_number))
            if(self.error_data_score_W0_n[current_split_number] == 1):
                print("Error data can be represented by W0 nmi {}".format(current_split_number))
            if(self.error_data_score_W0_a[current_split_number] == 1):
                print("Error data can be represented by W0 api {}".format(current_split_number))
            #______________________combinedweights
            if(current_split_number == 0):
                self.combinedweight =  np.concatenate((self.som.weights0, self.som.weights1), axis=0)
                #print("combinedweight shape 0: {}".format( self.combinedweight.shape))
            else:
                self.combinedweight =  np.concatenate((self.combinedweight, self.som.weights1), axis=0)
                #print("combinedweight shape1 : {}".format( self.combinedweight.shape))


            if(current_split_number == 0):
                self.rightdata_W_Combine_predicted_labels = np.array([self.som.predict(self.right_datas[current_split_number],self.som.weights0)], dtype=object)
            else:
                self.rightdata_W_Combine_predicted_labels = self.combineTwoRaggedArray(self.rightdata_W_Combine_predicted_labels,self.som.predict(self.right_datas[current_split_number],self.som.weights1))
            
            
            transferred_predicted_label_right_data_W_combined = self.transferClusterLabelToClassLabel(1,self.rightdata_W_Combine_predicted_labels[current_split_number],Wtype = 2,split_number=current_split_number)   
            self.getScore("right_data_score_W_combine",self.right_data_labels[current_split_number],transferred_predicted_label_right_data_W_combined)
            print("right_data{}_score_W\' purity {} ".format(current_split_number,self.right_data_score_W_combine_p[current_split_number]))
            print("right_data{}_score_W\' nmi {} ".format(current_split_number,self.right_data_score_W_combine_n[current_split_number]))
            print("right_data{}_score_W\' api {} ".format(current_split_number,self.right_data_score_W_combine_a[current_split_number]))
                        #_________ update current_data_train and current_label_train
            
            print("Finish one round splitting {} *********\n".format(current_split_number))


            current_data_train =  self.data_train_error_datas[current_split_number]
            current_label_train = self.error_data_labels[current_split_number]

            current_split_number = current_split_number +1
        
    

        unit_list = []
        all_train_scores_W_Combined_p = []
        test_scores_W_combined_p = []
        all_train_scores_W0_p = []
        test_scores_W0_p = []

        all_train_scores_W_Combined_n = []
        test_scores_W_combined_n = []
        all_train_scores_W0_n = []
        test_scores_W0_n = []

        all_train_scores_W_Combined_a = []
        test_scores_W_combined_a = []
        all_train_scores_W0_a = []
        test_scores_W0_a = []

        for stop_split_num in range(0, self.split_num+1):
            unit_list.append(stop_split_num)
            print("stop_split_num {}".format(stop_split_num))
            
            #self.train_W_combined_predicted_label = self.predict_among_nearestW_representedData(self.data_train,stop_split_num,self.weights)    
            self.train_W_combined_predicted_label = self.predict_among_multipleW(self.data_train,stop_split_num,self.right_datas,self.weights)    
           
            transferred_predicted_label_train_WCombine = self.transferClusterLabelToClassLabel(0,self.train_W_combined_predicted_label,Wtype = 2,split_number=stop_split_num)   
            self.getScore("all_train_score_W_Combined",self.label_train,transferred_predicted_label_train_WCombine)         

        #______________________ test data               
            #self.test_W_combined_predicted_label = self.predict_among_nearestW_representedData(self.data_test,stop_split_num,self.weights)   
            self.test_W_combined_predicted_label = self.predict_among_multipleW(self.data_test,stop_split_num,self.right_datas,self.weights)   


            transferred_predicted_label_test = self.transferClusterLabelToClassLabel(3,self.test_W_combined_predicted_label,Wtype = 2,split_number=stop_split_num)   
            self.getScore("test_score_W_combined",self.test_label,transferred_predicted_label_test)

            all_train_scores_W_Combined_p.append(self.all_train_score_W_Combined_p)
            test_scores_W_combined_p.append(self.test_score_W_combined_p)

            all_train_scores_W_Combined_n.append(self.all_train_score_W_Combined_n)
            test_scores_W_combined_n.append(self.test_score_W_combined_n)

            all_train_scores_W_Combined_a.append(self.all_train_score_W_Combined_a)
            test_scores_W_combined_a.append(self.test_score_W_combined_a)
            
            print("all_train_score_W\' purity split_num : {} {}".format(stop_split_num, self.all_train_score_W_Combined_p))
            print("all_train_score_W\' nmi split_num : {} {}".format(stop_split_num, self.all_train_score_W_Combined_n))
            print("all_train_score_W\' ari split_num : {} {}".format(stop_split_num, self.all_train_score_W_Combined_a))
                     
            print("test_score_W0 purity: {}".format( self.test_score_W0_p))
            print("test_score_W0 nmi: {}".format( self.test_score_W0_n))
            print("test_score_W0 api: {}".format( self.test_score_W0_a))
            print("test_score_W\': purity split_num : {} {}".format(stop_split_num,  self.test_score_W_combined_p))
            print("test_score_W\': nmi split_num : {} {}".format(stop_split_num,  self.test_score_W_combined_n))
            print("test_score_W\': ari split_num : {} {}".format(stop_split_num,  self.test_score_W_combined_a))
        

            all_train_scores_W0_p.append(self.all_train_score_W0_p)
            test_scores_W0_p.append(self.test_score_W0_p)
            all_train_scores_W0_n.append(self.all_train_score_W0_n)
            test_scores_W0_n.append(self.test_score_W0_n)
            all_train_scores_W0_a.append(self.all_train_score_W0_a)
            test_scores_W0_a.append(self.test_score_W0_a)

        #get last split result
        self.all_train_scores_W_Combined_last_p = all_train_scores_W_Combined_p[len(all_train_scores_W_Combined_p)-1]
        self.all_train_scores_W_Combined_last_n = all_train_scores_W_Combined_n[len(all_train_scores_W_Combined_n)-1]
        self.all_train_scores_W_Combined_last_a = all_train_scores_W_Combined_a[len(all_train_scores_W_Combined_a)-1]
        
        print("all_train_scores_W_Combined_last_p {}".format(self.all_train_scores_W_Combined_last_p))
        print("all_train_scores_W_Combined_last_n {}".format(self.all_train_scores_W_Combined_last_n))
        print("all_train_scores_W_Combined_last_a {}".format(self.all_train_scores_W_Combined_last_a))


        self.test_scores_W_Combined_last_p = test_scores_W_combined_p[len(test_scores_W_combined_p)-1]
        self.test_scores_W_Combined_last_n = test_scores_W_combined_n[len(test_scores_W_combined_n)-1]
        self.test_scores_W_Combined_last_a = test_scores_W_combined_a[len(test_scores_W_combined_a)-1]

        print("test_scores_W_Combined_last_p {}".format(self.test_scores_W_Combined_last_p))
        print("test_scores_W_Combined_last_n {}".format(self.test_scores_W_Combined_last_n))
        print("test_scores_W_Combined_last_a {}".format(self.test_scores_W_Combined_last_a))
        #if score_type == 0:
        #    plt.title("Purity Score")   
        #elif score_type == 1:
        #    plt.title("NMI Score")
        #elif score_type == 2:
        #    plt.title("ARI Score") 
#
        #plt.xlabel('Split Number')
        #plt.plot(unit_list,self.error_rates,'g',label ='error data percentage')
        #plt.plot(unit_list,all_train_scores_W_Combined,'c',label ='all_train_score_W\'')
        #plt.plot(unit_list,test_scores_W_combined,'k',label ='test_score_W\'')
        #plt.plot(unit_list,all_train_scores_W0,'r',label ='all_train_score_W0')
        #plt.plot(unit_list,test_scores_W0,'y',label ='test_score_W0')
        #plt.legend()
        #plt.show()          
        


        
    
