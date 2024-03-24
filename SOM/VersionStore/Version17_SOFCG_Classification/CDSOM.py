"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
find intra communiyt in each neuron memberships and do the whole mapping and retest 
"""

#from curses.ascii import NULL
from sklearn import metrics
from scipy import spatial
import numpy as np
from operator import add
import math
import sys
import operator
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from fcmeans import FCM


import itertools
import newSom
# unsupervised continus and discrete som
class CDSOM():
    """
     Unsupervised SOM with continusous data,and discrete data combination
     som the orignal som
     som_continuous when continusous = all train data som = som_continuous
     soms_discrete is the soms for each discrete features

    """
    def __init__(self, som,
                 som_continuous,
                 som_discrete_original, #used for comparision as a original som
                 som_total_discrete_transferred, #the last som got 
                 data_train_all, 
                 data_train_continuous,
                 data_train_discrete,
                 data_train_discrete_normalized,
                 data_test_all,
                 data_test_continuous,
                 data_test_discrete,
                 data_test_discrete_normalized,
                 label_train_all,                       
                 label_test_all,
                 ):
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
        self.som_total_discrete_transferred = som_total_discrete_transferred  
        self.som_discrete = som_discrete_original  
        # initial cluster numbers in TDSM_SOM, which is the neuron number in som
        self.predicted_classNum= int(som.m*som.n)
        self.community_distance = 1

        self.data_train_all = data_train_all
        self.data_test_all = data_test_all

        self.data_train_continuous = data_train_continuous   
        self.data_test_continuous = data_test_continuous


        self.data_train_discrete_unnormalized = data_train_discrete 
        self.data_train_discrete_normalized = data_train_discrete_normalized 
        self.data_test_discrete_unnormalized = data_test_discrete   
        self.data_test_discrete_normalized = data_test_discrete_normalized


        self.train_label_all = label_train_all
        self.train_label_all = self.train_label_all.astype(int)
        self.test_label_all = label_test_all
        self.test_label_all = self.test_label_all.astype(int)


        # continuous features generated datas
        self.all_split_datas_continuous = []
        # continuous features generated datas indexes   
        self.all_split_datas_indexes_continuous = []

        self.outliner_clusters = [] 

    def shannon_entropy(self,A, mode="auto", verbose=False):
        """
        https://stackoverflow.com/questions/42683287/python-numpy-shannon-entropy-array
        """
        A = np.asarray(A)

        # Determine distribution type
        if mode == "auto":
            condition = np.all(A.astype(float) == A.astype(int))
            #print(condition)
            if condition:
                mode = "discrete"
            else:
                mode = "continuous"
        if verbose:
            print(mode, file=sys.stderr)
        # Compute shannon entropy
        pA = A / A.sum()
        #print(f"A.sum() {A.sum()}")
        # Remove zeros
        pA = pA[np.nonzero(pA)[0]]
        if mode == "continuous":
            return -np.sum(pA*np.log2(A))  
        if mode == "discrete":
            return -np.sum(pA*np.log2(pA))   

    def mutual_information(self,x,y, mode="auto", normalized=False):
        """
        I(X, Y) = H(X) + H(Y) - H(X,Y)
        https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
        """
        x = np.asarray(x)
        y = np.asarray(y)
        # Determine distribution type
        #print(mode)
        if mode == "auto":
            condition_1 = np.all(x.astype(float) == x.astype(int))
            condition_2 = np.all(y.astype(float) == y.astype(int))
            #print(condition_1)
            #print(condition_2)
            if all([condition_1, condition_2]):
                mode = "discrete"
            else:
                mode = "continuous"

        H_x = self.shannon_entropy(x, mode=mode)
        print(H_x)
        H_y = self.shannon_entropy(y, mode=mode)
        print(H_y)
        H_xy = self.shannon_entropy(np.concatenate([x,y]), mode=mode)

        # Mutual Information
        I_xy = H_x + H_y - H_xy
        if normalized:
            return I_xy/np.sqrt(H_x*H_y)
        else:
            return  I_xy  

    def purity_score(self,scorename, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        #print(" purity_score y_true{}  y_pred {} ".format(y_true,y_pred))
        if(scorename == "all_train_score_W0" ):
            self.all_train_score_W0_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
            print("all_train_score_W0_p {}".format(self.all_train_score_W0_p ))  
        if(scorename == "all_train_score_W_combine" ):
            self.all_train_score_W_combine_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
            print("all_train_score_W_combine_p {}".format(self.all_train_score_W_combine_p ))  
        if(scorename == "test_score_W0" ):
            self.test_score_W0_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
            print("test_score_W0_p{}".format(self.test_score_W0_p ))
        if(scorename == "test_score_W_combine" ):
            self.test_score_W_combine_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
            print("test_score_W_combine_p {}".format(self.test_score_W_combine_p ))
        if(scorename == "test_discrete_score_W_discrete" ):
            self.test_discrete_score_W_discrete_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
            print("test_discrete_score_W_discrete_p {}".format(self.test_discrete_score_W_discrete_p ))  
        if(scorename == "train_discrete_score_W_discrete" ):
            self.train_discrete_score_W_discrete_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
            print("train_discrete_score_W_discrete_p {}".format(self.train_discrete_score_W_discrete_p ))  

        if(scorename == "train_discrete_score_W0" ):
            self.train_discrete_score_W0_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
            print("train_discrete_score_W0_p {}".format(self.train_discrete_score_W0_p ))  

        if(scorename == "test_discrete_score_W0" ):
            self.test_discrete_score_W0_p = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
            print("test_discrete_score_W0_p {}".format(self.test_discrete_score_W0_p ))  

    def nmiScore(self,scorename, y_true, y_pred):
       # print(" nmi y_true{} unique{} y_pred {} unique {}".format(y_true,np.unique(y_true),y_pred,np.unique(y_pred)))
        if(scorename == "all_train_score_W0" ):
            self.all_train_score_W0_n = normalized_mutual_info_score(y_true,y_pred)
            print("all_train_score_W0_n {}".format(self.all_train_score_W0_n ))  
        if(scorename == "all_train_score_W_combine" ):          
            self.all_train_score_W_combine_n = normalized_mutual_info_score(y_true,y_pred) 
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
            

        if(scorename == "train_discrete_score_W0" ):
            self.train_discrete_score_W0_n = normalized_mutual_info_score(y_true,y_pred)

            print("train_discrete_score_W0_n {}".format(self.train_discrete_score_W0_n ))  

        if(scorename == "test_discrete_score_W0" ):
            self.test_discrete_score_W0_n = normalized_mutual_info_score(y_true,y_pred)
            print("test_discrete_score_W0_n {}".format(self.test_discrete_score_W0_n ))  

        if(scorename == "test_continuous_score_W0" ):
            #print("test W0 y_true {}  y_pred{}".format(y_true ,y_pred ))  
            self.test_continuous_score_W0_n = normalized_mutual_info_score(y_true,y_pred)
            print("test_continuous_score_W0_n {}".format(self.test_continuous_score_W0_n ))  
    
        if(scorename == "test_continuous_score_W_continuous" ):
           # print("test W continuous y_true {}  y_pred{}".format(y_true ,y_pred ))  
            self.test_continuous_score_W_continuous_n = normalized_mutual_info_score(y_true,y_pred)
            print("test_continuous_score_W_continuous_n {}".format(self.test_continuous_score_W_continuous_n ))  

        if(scorename == "test_discrete_score_W_discrete" ):
            self.test_discrete_score_W_discrete_n = normalized_mutual_info_score(y_true,y_pred)
            print("test_discrete_score_W_discrete_n {}".format(self.test_discrete_score_W_discrete_n ))  
        if(scorename == "train_discrete_score_W_discrete" ):
            self.train_discrete_score_W_discrete_n = normalized_mutual_info_score(y_true,y_pred)
            print("train_discrete_score_W_discrete_n {}".format(self.train_discrete_score_W_discrete_n ))  
    
    def ariScore(self,scorename, y_true, y_pred):
       # print(" nmi y_true{} unique{} y_pred {} unique {} ".format(y_true,np.unique(y_true),y_pred,np.unique(y_pred)))
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
        if(scorename == "test_score_W_combine" ):
            self.test_score_W_combine_a = adjusted_rand_score(y_true,y_pred)
            print("test_score_W_combine_a {}".format(self.test_score_W_combine_a ))  

        if(scorename == "train_continuous_score_W0" ):
            self.train_continuous_score_W0_a = adjusted_rand_score(y_true,y_pred)
            print("train_continuous_score_W0_a {}".format(self.train_continuous_score_W0_a ))  
        
        if(scorename == "train_discrete_score_W0" ):
            self.train_discrete_score_W0_a = adjusted_rand_score(y_true,y_pred)
            print("train_discrete_score_W0_a  {}".format(self.train_discrete_score_W0_a ))  

        if(scorename == "train_continuous_score_W_continuous" ):
            self.train_continuous_score_W_continuous_a = adjusted_rand_score(y_true,y_pred)
            print("train_continuous_score_W_continuous_a {}".format(self.train_continuous_score_W_continuous_a ))  

        if(scorename == "test_continuous_score_W0" ):
            self.test_continuous_score_W0_a = adjusted_rand_score(y_true,y_pred)
            print("test_continuous_score_W0_a {}".format(self.test_continuous_score_W0_a ))  
        
        if(scorename == "test_discrete_score_W0" ):
            self.test_discrete_score_W0_a = adjusted_rand_score(y_true,y_pred)
            print("test_discrete_score_W0_a {}".format(self.test_discrete_score_W0_a ))  

        if(scorename == "test_continuous_score_W_continuous" ):
            self.test_continuous_score_W_continuous_a = adjusted_rand_score(y_true,y_pred)
            print("test_continuous_score_W_continuous_a {}".format(self.test_continuous_score_W_continuous_a ))  

        if(scorename == "test_discrete_score_W_discrete" ):
            self.test_discrete_score_W_discrete_a = adjusted_rand_score(y_true,y_pred)
            print("test_discrete_score_W_discrete_a {}".format(self.test_discrete_score_W_discrete_a ))  
        if(scorename == "train_discrete_score_W_discrete" ):
            self.train_discrete_score_W_discrete_a = adjusted_rand_score(y_true,y_pred)
            print("train_discrete_score_W_discrete_a {}".format(self.train_discrete_score_W_discrete_a ))  
    
    def get_indices_and_data_in_predicted_clusters(self,class_num_predicted,predicted_label):
            
            """
            predicted_label = [1,1,2,3,1,1,2,1]
            idx start from 0 to n
            class_label index also start from 0 to n
            """

            clusters_indexes = []
            #clusters_datas = []
            #print(class_num_predicted)  
            for i in range(0,class_num_predicted):
                newlist = []
                #newdatalist = []
                for idx, y in enumerate(predicted_label): 
                    # is the cluster label
                    if(y == i):
                        x = idx
                        x = int(x)                      
                        newlist.append(x)  
                        #newdatalist.append(data_set[x])                        
                clusters_indexes.append(newlist)
                #clusters_datas.append(np.array(newdatalist))
            #return clusters_indexes,clusters_datas
            #DO NOT NEED clusters_datas TO SAVE ram
            return clusters_indexes
           

    # from data index to  [[2,35,34,3,23],[211,12,2,1]] get cluster index [[0,0,1,1] [0,0,1]]
    def get_mapped_class_in_clusters(self,clusters_indexes,real_class_label):
        mapped_class_in_clusters = []
        #print(f"clusters_indexes {clusters_indexes} ")
        #initialize mapped_clases_in_clusters
        for i in range(0, len(clusters_indexes)):
            mapped_class_in_clusters.append([])

        for j in range(0, len(clusters_indexes)):
            for item in clusters_indexes[j]:
                mapped_class_in_clusters[j].append(real_class_label[item])
        #print(f"mapped_class_in_clusters {mapped_class_in_clusters}")
        # mapped_clases_in_clusters = [[1,2,1,2,1,1],[2,2,2,2],[0,1,0]]
       # for x in mapped_class_in_clusters:
       #    # print(f"x : {x}")
       #    print(f" x {self.realpropationofclasslabelinclusters(x,mapped_class_in_clusters.index(x))} ")
        return mapped_class_in_clusters
    
    def realpropationofclasslabelinclusters(self,clusters,i):
        #print(f" cluster len {len(clusters)}")
        keys, values = np.unique(clusters, return_counts =True)
        self.drawnDistrubitonofrealclassIneachNeuron(keys,values,i)

    def getLabelMapping(self,predicted_class_label_in_each_cluster,Wtype  = 0):
        """
         predicted_class_label  = [[1,2,1,1],[3,3,3]]  the value in is the true value in class_label
         it means that predicted cluster 0 is 1 in class lable, cluster label 2 is 3 in class label
        """
        predicted_label_convert_to_class_label = []
       
        for item in predicted_class_label_in_each_cluster:
            #print(f"item{item}" )   
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
       # print(f"counts  {counts}  list len{len(list)}")
        #print(f"self.realpropationofclasslabelinclusters(list) {self.realpropationofclasslabelinclusters(list)}")
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

    #def enchaned_som(self, original_som_weight):
    #    for x in original_som_weight:

    
    #get a dictionary with nodes has decsending distance with cluster center
    def split_continuous_data(self,data_train_continuous, targetgroup_index,cluster_center):
        """
        targetgroup_index : the group (cluster) which will be split
        """
        sorted_data_dict = {}
        for idx in targetgroup_index:     
            distance = np.linalg.norm((data_train_continuous[idx] - cluster_center).astype(float))
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
    def get_allnode_distance_to_center(self, data_train_continuous,target_node,  group_index, group_center):
        sorted_data_dict = {}
        distances = {}

        for idx in group_index:         
            distance = np.linalg.norm((data_train_continuous[idx] - target_node).astype(float))
            if distance >0 :
                sorted_data_dict[idx] =distance  

        sorted_dict = dict(sorted(sorted_data_dict.items(), key=operator.itemgetter(1),reverse=False))

        for key in sorted_dict:
            distance_intra = np.linalg.norm((data_train_continuous[key] - group_center).astype(float))
            distances[key] = distance_intra        
       
        return sorted_dict,distances

    # get all the inter node that has smaller distance to the target data, then target data to its cluster center 
    def get_intra_continuous_community_nodes(self,data_train_continuous,sorted_dict, intra_center):
        community_nodes = []
        community_nodes_keys = []

        for key in sorted_dict:  
            #**** cannot <= when == is itself, may cause one data one community       
            distance_intra = np.linalg.norm((data_train_continuous[key] - intra_center).astype(float))
            if sorted_dict[key] < distance_intra*self.community_distance:
                #print("sorted_dict[key {} key {} distance_intra{}".format(sorted_dict[key],key,distance_intra ))
                community_nodes.append(data_train_continuous[key])
                #print("key intra {}".format(key))
                community_nodes_keys.append(key)
        return community_nodes,community_nodes_keys


    def get_inter_continuous_community_nodes(self,data_train_continuous,sorted_dict,distances_inter):
        community_nodes = []
        community_nodes_keys = []
      
        for key in sorted_dict:
            if sorted_dict[key] < distances_inter[key]*self.community_distance:
                community_nodes.append(data_train_continuous[key])
                community_nodes_keys.append(key)
    
        return community_nodes,community_nodes_keys


    def _find_belonged_continuous_neuron(self,x,Y):
        """
        Find the index of the best matching unit for the input vector x.
        """  
        #initial distance
       # print(f"Y {(Y)}")
        #Y = np.array(Y)
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
        print(len(y_true))
        print(len(y_pred))
        self.purity_score(scorename,y_true,y_pred)
        self.nmiScore(scorename,y_true,y_pred)
        self.ariScore(scorename,y_true,y_pred)

    def getScores(self, score_tpye, y_true, y_pred):
        #print(f" y_true {y_true}")
        #print(f" y_pred {y_pred}")
        if score_tpye  == "Original":
            self.accuracy_score_original = accuracy_score(y_true,y_pred)
            print(f"accuracy_score_orignial {self.accuracy_score_original}")
            self.recall_score_original = recall_score(y_true,y_pred,average='macro')
            print(f"recall_score_original {self.recall_score_original}")
            self.precision_score_original = precision_score(y_true,y_pred,average='macro')
            print(f"precision_score_original {self.precision_score_original}")
            self.f1_score_original = f1_score(y_true,y_pred,average='macro')
            print(f"f1_score_original {self.f1_score_original}")
            #self.log_loss_original = log_loss(y_true,y_pred)
            #print(f"log_loss_original {self.log_loss_original}")
        elif score_tpye  == "SOG":
            self.accuracy_score_sog = accuracy_score(y_true,y_pred)
            print(f"accuracy_score_sog {self.accuracy_score_sog}")
            self.recall_score_sog = recall_score(y_true,y_pred,average='macro')
            print(f"recall_score_sog {self.recall_score_sog}")
            self.precision_score_sog = precision_score(y_true,y_pred,average='macro')
            print(f"precision_score_sog {self.precision_score_sog}")
            self.f1_score_sog = f1_score(y_true,y_pred,average='macro')
            print(f"f1_score_sog {self.f1_score_sog}")
           # self.log_loss_sog = log_loss(y_true,y_pred)
           # print(f"log_loss_sog {self.log_loss_sog}")

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

    def drawnDistrubitonofrealclassIneachNeuron(self,keys,values,i):

        
        fig = plt.figure(figsize = (4, 2))
        
        # creating the bar plot
        plt.bar(keys, values, color ='maroon',
                width = 0.4)
        
        plt.xlabel("Class Index")
        plt.ylabel("No. of class index")
        plt.title("Distribution of ground truth class index in neuron " + str(i))
        plt.show()


    def getWcontinouswithSplitData(self,newclustered_datas):
        self.continuous_weights = []
        for i in range (0,len(newclustered_datas)):
           #print(f"newclustered_datas[i] {newclustered_datas[i]}" )

            neuron = np.average(newclustered_datas[i], axis=0)
           # print(f"neuron {neuron}" )
            self.continuous_weights.append(neuron) 

        self.continuous_weights = np.array( self.continuous_weights)
        #print(f"self.continuous_weights {self.continuous_weights }" )
    
    def getuniquevalueindiscretedata(self, discrete_data):
        self.discrete_unique_values = []
        for i in range(0, len(discrete_data)):
           # print(f"discrete_data i {discrete_data[i]}")
            self.discrete_unique_values.append(np.unique(discrete_data[i]))
           # print(f"np.nuique(discrete_data[i]) {np.unique(discrete_data[i])}")

    def getCommanIndexesRatioInNeurons(self, feature_group, one_neuron_predict_group):
        #using Jaccard similarity 
       # print(f" feature_group {feature_group} one_neuron_predict_group {one_neuron_predict_group}")
        #print(f"len feature_group  {len(feature_group)} len one_neuron_predict_group {len(one_neuron_predict_group)}")  
       # similiary_score = jaccard_score(feature_group,one_neuron_predict_group)
       # print(f" similiary_score {similiary_score}")
        #return  similiary_score
        return round(len(np.intersect1d(feature_group, one_neuron_predict_group))/len(feature_group),3)
    
    def getCommanIndexesRatioInNeurons_fuzzy(self, feature_group_intersection, one_neuron_predict_group): # membership funciton
        # difference with getCommanIndexesRatioInNeurons is the denomitor
        if len(one_neuron_predict_group) !=0:
            return round(len(np.intersect1d(feature_group_intersection, one_neuron_predict_group))/len(one_neuron_predict_group),3) 
        else: return 0
    def getCommanIndexesRatioInNeurons_fuzzy_union(self, feature_group_intersection, feature_group_union, one_neuron_predict_group): # membership funciton
        # difference with getCommanIndexesRatioInNeurons is the denomitor
        if len(one_neuron_predict_group) !=0:
            return round(len(np.intersect1d(feature_group_intersection, one_neuron_predict_group))/len(feature_group_union),3) 
        else: return 0


    def getFeatureGroups(self, feature_column_data):  #cluster feature data based on unique value
        #feature_group = dictionary{value:[indexes], value2: [indexes]} feature_column_data =
        feature_group = {}
       # print(f" feature_column_data{ feature_column_data} ")
        for i in range(0,len(feature_column_data)):
            if len(feature_group)>=1:  
                if  feature_column_data[i] in feature_group.keys():
                    feature_group[feature_column_data[i]].append(i) 
                else:
                    feature_group[feature_column_data[i]] = [i]
            else:
                 #print(f" i  {i}, self.feature_column_data[i] {feature_column_data[i]}")
                 feature_group[feature_column_data[i]] = [i]
        #print(f" feature_group{ feature_group} ") 
        return feature_group
    
    
    def clusterMultipleFeatures(self,X,mutilple_feature_index_group): #with [0,1,] get mutiple_feature_group, whose key is three valuse in f0,f1,f2 
        #print(f" mutilple_feature_index_group {mutilple_feature_index_group}")
        #mutilple_feature_index_group =[0]
        mutiple_feature_group = {}
        mutiple_feature_group_union = {}  #mutilple_feature_index_group = [0,1]
        #print(f".shape[0] P{X.shape[0]}")
       
        
        for i in range(0,X.shape[0]): #choose one column in mutilple_feature_columns, as they are the same 
             key = tuple(self.getOneRowDataBaseOneMutipleFeatures(X,mutilple_feature_index_group,i))
             #print(f"key {key} mutiple_feature_group.keys() {mutiple_feature_group.keys()}")
             
             if len(mutiple_feature_group)>=1:
                    if key in mutiple_feature_group.keys():
                        mutiple_feature_group[key].append(i) 
                    else: mutiple_feature_group[key] =[i] 
             else:
                mutiple_feature_group[key] =[i] 
             values =[]
             for j in range(0,len(mutilple_feature_index_group)):       
                #try:
                    if X[:,mutilple_feature_index_group[j]][i] == key[j]:
                    #print(f"i {i} c {c}")
                        if i not in values :
                           values.append(i)
             mutiple_feature_group_union[key]  = values
        
             #print(f" mutiple_feature_group[key] inter { mutiple_feature_group[key]} ")
             #uNIVON
             #mutiple_feature_group_union[key]  = self.getUnionIndexesBasedOnKeyValues(X,mutilple_feature_index_group,key)
             #print(f" mutiple_feature_group_union[key] Union { mutiple_feature_group_union[key]} ")
        
             #print(f"mutiple_feature_group {key}  len value {len(mutiple_feature_group[key] )}") #mutiple_feature_group  = {(15.0, 1.0): [0, 82, 130, 139, 161, 166, 245,1558, 1665], (12.0, 9.0): [2, 130, 139, 161, 166, 24]}}
        return mutiple_feature_group,mutiple_feature_group_union
    

    def getUnionIndexesBasedOnKeyValues(self,X,columns,key):  #columns = [0,1] columns, key = [13,2] values for a data
        values =[]
        for i in range(0,X.shape[0]): 
            for j in range(0,len(columns)):       
                #try:
                    if X[:,columns[j]][i] == key[j]:
                    #print(f"i {i} c {c}")
                        if i not in values:
                            values.append(i)
                #except:
                   # print(f"c {c}  i{i} X[:,c][i]  {X[:,c][i] }  key{key}")

        #print(f"value {values}")
        return values 
             




    def getOneRowDataBaseOneMutipleFeatures(self,X,mutilple_feature_columns,row_num):   #X is training data from  [[F1column],[F2column],[F3column]] get one row F1_1,F2_1,F3_1
        features_num = len(mutilple_feature_columns)
        combination =[]
        for i in range(0,features_num):
            combination.append(X[:,mutilple_feature_columns[i]][row_num])
       # print(f"getOneRowDataBaseOneMutipleFeatures key: {combination} ")
        return combination 
    

    def getAllFeatureCombinationsGroupGranular(self,X): # the same with getAllfeatureGroups but for multiple features
        self.all_feature_combinations ={}
        for i in range(0,len(self.multiple_feature_groups)):  # self.multple_feature_groups =[[0,1,2], [3,4,5]]
            self.all_feature_combinations[i] = self.clusterMultipleFeatures(X,self.multiple_feature_groups[i])
        # for the left
        if self.left_feature_group != []:
            self.all_feature_combinations[len(self.multiple_feature_groups)] = self.clusterMultipleFeatures(X,self.left_feature_group)
        #print(f"self.all_feature_combinations {self.all_feature_combinations}")  # {0: {(15.0, 7.0): [0, 27, 41, 150, 156,], (13.0, 8.0): [4, 26, 1]}, 1: {(15.0, 7.0): [0, 27, 41, 150, 156,], (13.0, 8.0): [4, 26, 1]}
            

        
    
        #print(f"self.all_feature_combinations {self.all_feature_combinations}")

    def getAllfeatureGroups(self):
        self.all_feature_groups = {}
        #print(f"np.shape(self.data_train_discrete_unnormalized)[1]{ np.shape(self.data_train_discrete_unnormalized)[1]}")
        for i in range(0,np.shape(self.data_train_discrete_unnormalized)[1]):
            # i is the comlumn number
            self.all_feature_groups[i] = self.getFeatureGroups(self.data_train_discrete_unnormalized[:,i])
            #print(f" i {i} self.all_feature_gropus {  self.all_feature_groups[i] } ") #{15.0: [0, 1, 5, 6, 19, 12],1:[43,45,47]}
    def getAllfeatureGroupsCombination(self):
        self.all_feature_groups_combinations = {}
        #print(f"np.shape(self.data_train_discrete_unnormalized)[1]{ np.shape(self.data_train_discrete_unnormalized)[1]}")
        for i in range(0,np.shape(self.train_combinationns)[1]):
            # i is the comlumn number
            self.all_feature_groups_combinations[i] = self.getFeatureGroups(self.train_combinationns[:,i])

    def transferdataToprobabilityrepresentation(self, x, neuron_num):
        result_probability_sum_vector = np.zeros(neuron_num)
        for i in range(0, len(x)):
            for key in self.all_features_mapping[i].keys():
                if x[i] == key:
                    current_feature_probability_vector = self.all_features_mapping[i][key]  #current_feature_probability_vector = [0.1,0,0.2,0.7]
                    #print(f" current_feature_probability sum {np.sum(current_feature_probability)}")
                    if i == 0:
                        result_probability_sum_vector = current_feature_probability_vector
                        #print(f" reslut_probability 2 {result_probability_multiply}")
                    #print(f"i {i} x[i] {x[i]} key{key}")
                    else:                   
                        result_probability_sum_vector = np.add(result_probability_sum_vector,current_feature_probability_vector) 
        #print(f" result_probability_sum_vector {result_probability_sum_vector}")
        return result_probability_sum_vector

                        
    def findmaxprobablity(self,x,neuron_num):
        result_probability_sum = np.zeros(neuron_num)
        result_matrix = []
        for i in range(0, len(x)):
            #print(f"i{ i} self.all_features_mapping[i].keys() {self.all_features_mapping[i].keys()}")
            for key in self.all_features_mapping[i].keys():
                #print(f" i {j} len(x){len(x)} x {x}")
                if x[i] == key:
                    current_feature_probability = self.all_features_mapping[i][key]
                    #print(f" current_feature_probability sum {np.sum(current_feature_probability)}")
                    if i == 0:
                        result_probability_sum = current_feature_probability
                        #print(f" reslut_probability 2 {result_probability_multiply}")
                    #print(f"i {i} x[i] {x[i]} key{key}")
                    else:                   
                        result_probability_sum = np.add(result_probability_sum,current_feature_probability) 

                    result_matrix.append(current_feature_probability)                   

        result_matrix = np.array(result_matrix)
        match_neuron_index = self.getAdjustedNeuronProbability(result_matrix,result_probability_sum)
        #print(f"np.argmax(reslut_probability) {reslut_probability}")
        return match_neuron_index


    def getAdjustedNeuronProbability(self, result_probability_matrix,result_probability_multiply):
        choosen_neurons_index = []
        for i in range(len(result_probability_multiply)):
            if result_probability_multiply[i]  != 0:
                choosen_neurons_index.append(i)
        result_probability_matrix_remove_zero = result_probability_matrix[:,choosen_neurons_index] # only choose columns that have no zero

        result_probability = self.getNewProbabiltyBasedOnNewMatrix(result_probability_matrix_remove_zero)
        # print(f" result_probability {result_probability}")
        return choosen_neurons_index[np.argmax(result_probability)]
        # get zero value in each row:

    def getNewProbabiltyBasedOnNewMatrix(self,result_probability_matrix_remove_zero ):
        for i in range(0,result_probability_matrix_remove_zero.shape[1]) :
            current_column = result_probability_matrix_remove_zero[:,i]
            denominator = np.sum(current_column)
            for j in range(0, len(current_column)):
                current_column[j] = current_column[j] /denominator
            result_probability_matrix_remove_zero[:,i] = current_column #update old data

        #get last probablity based on new result_probability_matrix_remove_zero
        result = []
        denominator2 = 0
        for j in range(0,result_probability_matrix_remove_zero.shape[0]) :
            current_row = result_probability_matrix_remove_zero[j:]
            denominator2 = np.prod(current_row) + denominator2
            for i in range(0, len(current_row)):
                current_row[i] = current_row[i] /denominator2
                result.append(current_row[i])
        return result

    def predictBasedonNeuronProbability(self,X):
        #print(f" X {X}")
       # print(f" self.all_features_mapping {self.all_features_mapping}")

             # the som which will be used in new representation of data (the probality of each neuron) which is mxn n is the number of neuron and m is the number of data
        labels = np.array([self.findmaxprobablity(x,self.som.weights0.shape[0]) for x in X])
       # print(f" labels {labels}")
        return labels
    
    def trainNewSomWithFeatureProbabilityData(self,X):
        dim= self.som_discrete.weights0.shape[0]
        m, n = self.topology_som(dim)
        self.som_probability = newSom.SOM(m=m, n= n, dim=dim) 
       # print(f" X {X}")

        transefered_X = np.array([self.transferdataToprobabilityrepresentation(x,dim) for x in X])
       
      
        self.som_probability.fit(transefered_X)
        return self.som_probability.predict(transefered_X, self.som_probability.weights0)   
    
    def predcitTestDatwithSOMProbability(self,X,neuron_num):
        transefered_X = np.array([self.transferdataToprobabilityrepresentation(x,neuron_num) for x in X])
        return self.som_probability.predict(transefered_X, self.som_probability.weights0)   
    
    def getOneSingleFeatureNeuronProbability(self, one_feature_dic, neuron_predicted_groups):
        onesinglefeatureneuronprobablity = {}
        #print(f"one_feature_dic {one_feature_dic} ")
        for key in one_feature_dic.keys():
            probability_list = []
            for i in range(0,len(neuron_predicted_groups)) :  
                #print("New neuron _predicted group !!!")       
                probability = self.getCommanIndexesRatioInNeurons(one_feature_dic[key],neuron_predicted_groups[i]) 
                #print(f"i {i} neuron_predicted_groups[i] {len(neuron_predicted_groups[i])}")  
                probability_list.append(probability)
            #print(f"key {key} probability_list{probability_list}")
            probability_list = np.array(probability_list)
            onesinglefeatureneuronprobablity[key] = probability_list
           # print(f"key {key} probability_list) {probability_list} ")
        return onesinglefeatureneuronprobablity


    def getOneSingleFeatureNeuronProbability_fuzzy(self, one_feature_dic, neuron_predicted_groups):
        onesinglefeatureneuronprobablity = {}
        #print(f"neuron_predicted_groups {neuron_predicted_groups} ")
        for key in one_feature_dic.keys():
            probability_list = []
            for i in range(0,len(neuron_predicted_groups)) :  
                #print("New neuron _predicted group !!!")       
                probability = self.getCommanIndexesRatioInNeurons_fuzzy(one_feature_dic[key],neuron_predicted_groups[i]) 
                #print(f"i {i} neuron_predicted_groups[i] {len(neuron_predicted_groups[i])}")  
                probability_list.append(probability)
            #print(f"key {key} probability_list{probability_list}")
            probability_list = np.array(probability_list)
            onesinglefeatureneuronprobablity[key] = probability_list
           # print(f"key {key} probability_list fuzzy) {probability_list} ")
        return onesinglefeatureneuronprobablity


    def getMultipleFeaturesNeuronProbability_fuzzy(self, one_goup_feature_dic_value,one_goup_feature_dic_value_union, neuron_predicted_groups):
        onegroupfeatureneuronprobablity = {}
        #print(f"one_goup_feature_dic_value  len {len(one_goup_feature_dic_value)} ")
        for key in one_goup_feature_dic_value:
            #print(f"dic {dic}")
            probability_list = []
            for i in range(0,len(neuron_predicted_groups)) :  
                #print("New neuron _predicted group !!!")       
                probability = self.getCommanIndexesRatioInNeurons_fuzzy_union(one_goup_feature_dic_value[key],one_goup_feature_dic_value_union[key],neuron_predicted_groups[i]) 
                #print(f"i {i} neuron_predicted_groups[i] {len(neuron_predicted_groups[i])}")  
                probability_list.append(probability)
            #print(f"key {key} probability_list{probability_list}")
            probability_list = np.array(probability_list)
            onegroupfeatureneuronprobablity[key] = probability_list
            #print(f"key {key} probability_list fuzzy) {probability_list} ") # onegroupfeatureneuronprobablity{(13.0, 1.0): array([0.002, 0.   ]), (14.0, 1.0): array([0.002, 0.   ]), (15.0, 2.0): array([0.001, 0.001]), (10.0, 0.0): array([0.002, 0.   ]), (10.0, 2.0): array([0.002, 0.   ]), 
       # print(f"onegroupfeatureneuronprobablity{onegroupfeatureneuronprobablity}")
        return onegroupfeatureneuronprobablity
    

    
    def getEachNeuronProbabilityOfEachFeatureValue(self,neuron_predicted_groups):
        self.all_features_mapping ={}
        for i in range(0, len(self.all_feature_groups)):
           # print(f"  feature  {i}!!!!!!!!!!!")
            self.all_features_mapping[i] = self.getOneSingleFeatureNeuronProbability(self.all_feature_groups[i],neuron_predicted_groups)
           # self.all_features_mapping[i] = self.getOneSingleFeatureNeuronProbability_subjective(self.all_feature_groups[i],neuron_predicted_groups)
            #print(f"  self.all_features_mapping[i]  {self.all_features_mapping[i]}!!!!!!!!!!!")
    def getEachNeuronProbabilityOfEachFeatureValue_fuzzy(self,neuron_predicted_groups):
        self.all_features_mapping_fuzzy ={}
        for i in range(0, len(self.all_feature_groups)):
           # print(f"  feature  {i}!!!!!!!!!!!")
            self.all_features_mapping_fuzzy[i] = self.getOneSingleFeatureNeuronProbability_fuzzy(self.all_feature_groups[i],neuron_predicted_groups)
        #print(f"  self.all_features_mapping_fuzzy {self.all_features_mapping_fuzzy}")


    def getEachNeuronProbabilityOfEachFeatureValue_fuzzy_combination(self,neuron_predicted_groups):
        self.all_features_mapping_fuzzy_combination ={}
        for i in range(0, len(self.all_feature_groups_combinations)):
           # print(f"  feature  {i}!!!!!!!!!!!")
            self.all_features_mapping_fuzzy_combination[i] = self.getOneSingleFeatureNeuronProbability_fuzzy(self.all_feature_groups_combinations[i],neuron_predicted_groups)
        #print(f"  self.all_features_mapping_fuzzy {self.all_features_mapping_fuzzy}")

    def getEachNeuronProbabilityOfMultipleFeatureValue_fuzzy(self,neuron_predicted_groups): #neuron_predicted_groups is ngranule self.all_feature_combinations = {0: {(1,2):{2,34,5,6}}}
        self.all_multiple_features_mapping_fuzzy ={}
        for i in range(0, len(self.all_feature_combinations)):
           # print(f"  feature  {i}!!!!!!!!!!!")
            self.all_multiple_features_mapping_fuzzy[i] = self.getMultipleFeaturesNeuronProbability_fuzzy(self.all_feature_combinations[i],neuron_predicted_groups)

    def getEachNeuronProbabilityOfSpecificFeatureValue_fuzzy(self,neuron_predicted_groups): #neuron_predicted_groups is ngranule self.all_feature_combinations = {0: {(1,2):{2,34,5,6}}}
        self.multiple_features_mapping_fuzzy ={}
        for i in range(0, len(self.all_feature_combinations)):
           # print(f"  feature  {i}!!!!!!!!!!!")
            self.multiple_features_mapping_fuzzy[i] = self.getMultipleFeaturesNeuronProbability_fuzzy(self.all_feature_combinations[i],neuron_predicted_groups)
            #print(f"all_multiple_features_mapping_fuzzy {self.all_multiple_features_mapping_fuzzy}")
    def getEmbeddingWithNeuronProbablity(self,X):
        newX =[]
        
        for x in X:
            newdata =[]
            for j in range(0, len(x)):
                #**** for certain situation, in the trainng set there is too many data , so we resampled them , as adata reslut in the test data the value has but in trainig data it doesnt have
                if x[j] in self.all_features_mapping[j].keys():
                #print(f"j {j} x[j] {x[j]}self.all_features_mapping[j]   {self.all_features_mapping[j]}")
                    for value in self.all_features_mapping[j][x[j]]:
                        newdata.append(value) 
                else:
                   # print(f"j{j} x[j]   {x[j] }  self.all_features_mapping[j].keys() {self.all_features_mapping[j].keys()}")
                    fakekey = list(self.all_features_mapping[j])[0]
                    #**** it is not correct, just for a certain dataset, which has lots of data but certain features have very small propration, so when resample the traiing data, that feature is not incluced, but in the test data it has such feature value 
                    for value in self.all_features_mapping[j][fakekey]:
                        newdata.append(value) 
           # print(f"the original discrete data : {x} ")
            #print(f"the proposed encoded data representation: {newdata} ")
            newX.append(newdata)
       # print(f"new embedding {newX}")
        #print(f"the original discrete data : {X} and proposed encoded data representation: {newX} ")
        return np.array(newX)
    def zerolistmaker(self, n):
        listofzeros = [0] * n
        return listofzeros   
    

    def getOrignalDataNewCombinationEmbedding(self,X,all_feature_combination):
        newX =[]
       # print(f"all_feature_combination  {all_feature_combination}")
        for x in X:
           # print(f"x  {x}")
            x_combiantions =[]
           # print(f" proposed encoded data representation shape111: {len(all_features_data)} ")
            for i in range(0,len(all_feature_combination)):
                feature_group = all_feature_combination[i] 
                for i in feature_group:
                    x_combiantions.append(x[i])
            newX.append(x_combiantions)
           #print(f"newX  {newX}")
        a = np.array(newX) 
        return a

    def getEmbeddingWithNeuronProbablity_multiple_features_fuzzy(self,X, all_feature_combination): #self.multple_feature_groups =[[0,1], [0,2], [1,2]]
        newX =[]
        #print(f"all_feature_combination {all_feature_combination}")  
        for x in X:
            all_features_data =[]
           # print(f" proposed encoded data representation shape111: {len(all_features_data)} ")
            for i in range(0,len(all_feature_combination)):
                feature_group = all_feature_combination[i] 
                #print(f"feature_group {feature_group}")
                one_combination_data =[]
                pair_real_value =[]
                for i in feature_group:
                    pair_real_value.append(x[i])
                pair_real_value = tuple(pair_real_value)
                find_the_key = False
                #print(f"pair_real_value {pair_real_value}")
                for key in self.all_feature_combinations_sog_embedding[feature_group]: #*** dic is key                
                    #print(f"self.all_multiple_features_mapping_fuzzy[i][dic] {self.all_multiple_features_mapping_fuzzy[i][dic]}")
                    if pair_real_value == key:  
                       # print(f"key {key} pair_real_value {pair_real_value}")
                        find_the_key = True                      
                        for item in  self.all_feature_combinations_sog_embedding[feature_group][key]:  
                           # print(f"item {item}")         
                            one_combination_data.append(item)
                if find_the_key == False:
                    for value in self.all_feature_combinations_sog_embedding[feature_group][list(self.all_feature_combinations_sog_embedding[feature_group].keys())[0]]:
                        #print(f"item2  {item} key{key}") 
                        one_combination_data.append(value)                  
                all_features_data.extend(one_combination_data)
                #print(f" all_features_data1 {all_features_data}")
            #print(f"orignal {x}  all_features_data {all_features_data}")
                #print(f" proposed encoded data representation shape: {len(all_features_data)} ")
            newX.append(all_features_data)
        a = np.array(newX)   
        #print(f"x1 {a.shape}")
        #print(f"newX shape {np.array(newX).shape}")
        #a = a[:, np.any(a, axis=0)]    #remove zero column
        #print(f"x2 {a.shape}")
        return a
    
    
    def getEmbeddingWithNeuronProbablity_fuzzy(self,X):
        newX =[]
        for x in X:
            newdata =[]
            for j in range(0, len(x)): 
                #**** for certain situation, in the trainng set there is too many data , so we resampled them , as data reslut in the test data the value has but in trainig data it doesnt have
                #print(f"self.all_features_mapping_fuzzy[j].keys() {self.all_features_mapping_fuzzy[j].keys()}")
                if x[j] in self.all_features_mapping_fuzzy[j].keys():
                    #print(f"j {j} x[j] {x[j]}self.all_features_mapping_fuzzy[j]   {self.all_features_mapping_fuzzy[j]}")
                    for value in self.all_features_mapping_fuzzy[j][x[j]]:
                        newdata.append(value) 
                else:
                   # print(f"j{j} x[j]   {x[j] }  self.all_features_mapping[j].keys() {self.all_features_mapping[j].keys()}")
                    fakekey = list(self.all_features_mapping_fuzzy[j])[0]
                    #**** it is not correct, just for a certain dataset, which has lots of data but certain features have very small propration, so when resample the traiing data, that feature is not incluced, but in the test data it has such feature value 
                    for value in self.all_features_mapping_fuzzy[j][fakekey]:
                        newdata.append(value) 
            #print(f"the original discrete data : {x} and proposed encoded data representation: {newdata} ")
            newX.append(newdata)
           
       # print(f"new embedding {newX}")
        #print(f"the original discrete data : {X} and proposed encoded data representation: {newX} ")
        return np.array(newX)
    
    def getEmbeddingWithNeuronProbablity_fuzzy_combination(self,X):
        newX =[]
        for x in X:         
            newdata =[]
            for j in range(0, len(x)): 
                #**** for certain situation, in the trainng set there is too many data , so we resampled them , as data reslut in the test data the value has but in trainig data it doesnt have
                #print(f"self.all_features_mapping_fuzzy[j].keys() {self.all_features_mapping_fuzzy[j].keys()}")
                if x[j] in self.all_features_mapping_fuzzy_combination[j].keys():
                    #print(f"j {j} x[j] {x[j]}self.all_features_mapping_fuzzy[j]   {self.all_features_mapping_fuzzy[j]}")
                    for value in self.all_features_mapping_fuzzy_combination[j][x[j]]:
                        newdata.append(value) 
                else:
                   # print(f"j{j} x[j]   {x[j] }  self.all_features_mapping[j].keys() {self.all_features_mapping[j].keys()}")
                    fakekey = list(self.all_features_mapping_fuzzy_combination[j])[0]
                    #**** it is not correct, just for a certain dataset, which has lots of data but certain features have very small propration, so when resample the traiing data, that feature is not incluced, but in the test data it has such feature value 
                    for value in self.all_features_mapping_fuzzy_combination[j][fakekey]:
                        newdata.append(value) 
           # print(f"the original discrete data : {x} and proposed encoded data representation: {newdata} ")
            newX.append(newdata)
           
       # print(f"new embedding {newX}")
        #print(f"the original discrete data : {X} and proposed encoded data representation: {newX} ")
        return np.array(newX)
    def getEmbeddingWithNeuronProbablity_fuzzy1(self,X,features_choosen):
        newX =[]
        newX_features_choosen =[]
        for x in X:
            newdata =[]
            newdata_features_choosen =[]
            for j in range(0, len(x)): 
                #**** for certain situation, in the trainng set there is too many data , so we resampled them , as data reslut in the test data the value has but in trainig data it doesnt have
                #print(f"self.all_features_mapping_fuzzy[j].keys() {self.all_features_mapping_fuzzy[j].keys()}")
                if x[j] in self.all_features_mapping_fuzzy[j].keys():
                    #print(f"j {j} x[j] {x[j]}self.all_features_mapping_fuzzy[j]   {self.all_features_mapping_fuzzy[j]}")
                    for value in self.all_features_mapping_fuzzy[j][x[j]]:
                        newdata.append(value) 
                        newdata_features_choosen.append(value)
                else:
                   # print(f"j{j} x[j]   {x[j] }  self.all_features_mapping[j].keys() {self.all_features_mapping[j].keys()}")
                    fakekey = list(self.all_features_mapping_fuzzy[j])[0]
                    #**** it is not correct, just for a certain dataset, which has lots of data but certain features have very small propration, so when resample the traiing data, that feature is not incluced, but in the test data it has such feature value 
                    for value in self.all_features_mapping_fuzzy[j][fakekey]:
                        newdata.append(value) 
                        newdata_features_choosen.append(value)
            
            extra_embeddings = self.zerolistmaker(self.som.weights0.shape[0])

           # print(f"x {x}")
            #print(f"extra_embeddings initial {extra_embeddings}")

            for dt in features_choosen:
                if dt > len(x)-1:
                    raise Exception(f"Choosen feature index {dt}  out of range of total discrete feature  {len(x)}")

            for l in range(0, len(x)):  
                extra_embedding_one_feature = []
                #print(f"l {l}  features_choosen {features_choosen}")
                if l in features_choosen:
                     #print(f"x[l] {x[l]}  self.all_features_mapping_fuzzy[l].keys() {self.all_features_mapping_fuzzy[l].keys()}    ")
                     if x[l] in self.all_features_mapping_fuzzy[l].keys():
                       # print(f"self.all_features_mapping_fuzzy[l][x[l]] {self.all_features_mapping_fuzzy[l][x[l]]}")
                        for value in self.all_features_mapping_fuzzy[l][x[l]]:
                            extra_embedding_one_feature.append(value)
                     else: 
                        extra_embedding_one_feature = self.zerolistmaker(self.som.weights0.shape[0]) #when there is situation where   x[l] not in  self.all_features_mapping_fuzzy[l].keys(), some data in test dataset not in trainning dataset
                     #print(f"extra_embedding_one_feature {extra_embedding_one_feature}")
                     #print(f"extra_embeddings1  {extra_embeddings}  extra_embedding_one_feature {extra_embedding_one_feature}")
                     extra_embeddings = list( map(add, extra_embeddings, extra_embedding_one_feature) )
                     #print(f"extra_embeddings  {extra_embeddings} ")
                     #print(f"newdata_features_choosen {newdata_features_choosen }  extra_embeddings {extra_embeddings}")     
            newdata_features_choosen = newdata_features_choosen + extra_embeddings
            #print(f"newdata_features_choosen length: {len(newdata_features_choosen) }  newdata_features_choosen {newdata_features_choosen}")
            #print(f"the proposed encoded data representation: {newdata}  newdata_features_choosen {newdata_features_choosen} ")
            newX.append(newdata)
            newX_features_choosen.append(newdata_features_choosen)
       # print(f"np.array(newX_features_choosen) shape {np.array(newX_features_choosen).shape}")
        #print(f"new newX_features_choosen {newX_features_choosen}")
        #print(f"the original discrete data : {X} and proposed encoded data representation: {newX} ")
        return np.array(newX),np.array(newX_features_choosen)
       # print(f"new embedding {newX}")
        #print(f"the original discrete data : {X} and proposed encoded data representation: {newX} ")
        return np.array(newX)
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

    def PCA_Comparision(self):
        pca1 = PCA()
        pca1.fit_transform(self.data_train_discrete_unnormalized)
        #print(self.data_train_baseline_encoded.shape)
        #print("Base Line PCA feature importance")
        #print(abs(pca1.components_))
        self.RankingFeatureImportance(abs(pca1.components_),self.data_train_discrete_unnormalized.shape[1])
        print("Base Line PCA explained_variance_ratio_")
        print(pca1.explained_variance_ratio_)
        plt.bar(
            range(1,len(pca1.explained_variance_)+1),
            pca1.explained_variance_
            )
        
        
        plt.xlabel('PCA Feature')
        plt.ylabel('Explained variance')
        plt.title('Feature Explained Variance')
        plt.show()

       # pca2 = PCA()
       # print(self.discrete_data_embedding_sog.shape)
       # pca2.fit_transform(self.discrete_data_embedding_sog)
       # #print("SOG PCA feature importance")
       # #print(abs(pca2.components_))
       # self.RankingFeatureImportance(abs(pca2.components_),self.discrete_data_embedding_sog.shape[1])
       # print("SOG PCA explained_variance_ratio_")
       # print(pca2.explained_variance_ratio_)
       # plt.bar(
       #     range(1,len(pca2.explained_variance_)+1),
       #     pca2.explained_variance_
       #     )
       # 
       # 
       # plt.xlabel('PCA Feature')
       # plt.ylabel('Explained variance')
       # plt.title('Feature Explained Variance')
       # plt.show()

    def RankingFeatureImportance(self,X,pc_num):
        d ={}
        for i in range(0,pc_num):
           if i< pc_num:
               d[i] = []
        for x in X:
           for j in range(0,len(x)):
               d[j].append(x[j])
        
        
        for l in range(0,pc_num):
            sorted_index = [sorted(d[l]).index(x) for x in d[l]]
            sorted_list = sorted(d[l], reverse=True)
            print(f"Sorted index for feature {l} {sorted_index}" )
            print(f"Sorted list for feature {l}  {sorted_list}" )
            #print(f"sum  {sum(sorted_list)}")


    def getFeatureGroupsfromMultipleFeatures1(self, X, feature_num):

        divid_group = int(X.shape[1]/feature_num) # how many combination groups will have
        reminder = int(X.shape[1]%feature_num) # if there are columns that not in a group
        combinations = []
        for i in range(1,divid_group+1):
            new_combination =[]
            for j in range(feature_num*(i-1),feature_num*i):
                new_combination.append(j)
            combinations.append(new_combination)
        left=[]
        if reminder != 0:          
            for m in range(divid_group*feature_num, X.shape[1]):
                left.append(m)
        self.multiple_feature_groups = combinations
        self.left_feature_group = left
        print(f" self.multple_feature_groups  {self.multiple_feature_groups }  self.left_feature_group { self.left_feature_group} ")
        #return combinations,left  # combinations =[[0,1,2],[3,4,5]] reminder = [6]

    def getFeatureGroupsfromMultipleFeatures(self, X, feature_selected):

        self.multiple_feature_groups = [feature_selected]
        self.left_feature_group = []
        print(f" self.multple_feature_groups  {self.multiple_feature_groups }  self.left_feature_group { self.left_feature_group} ")
        #return combinations,left  # combinations =[[0,1,2],[3,4,5]] reminder = [6]

    def getAllCombinationsInOneFeatureGroup(self,combination_group,X):
        column_num = len(combination_group)
        if column_num<=1:
           raise Exception("at least two featues needed!")
        unique_combinations = []
        for i in range(0,column_num):
            #print(f"append combination_group[i] {combination_group[i]}")
            unique_combinations.append(np.unique(X[:,combination_group[i]]))
        #print(f"unique_combinations {unique_combinations}")
        return list(itertools.product(*unique_combinations))

    def getFeatureCombination(self,feature_list,feature_choosen_num):
        #print(f"feature_list {feature_list}")
        self.all_combinations = list(itertools.combinations(feature_list, feature_choosen_num))
        #print(f"self.all_combinations {self.all_combinations}") 


    def my_func(self,min, max)->list:     
        return list(range(min, max))
    
    def checkRepeat(self, list):
        for i in range(0,len(list)-1):
            if list[i] == list[i+1]:
                return True
        return False
    
    def do_DOSOM(self,features_chosoen_number):
        """
        do discrete optimized SOM 
        features_chosoen_number = 2, or 3, or 4 the feausure numbers for graunule
        """
       #1 all_feature_column = self.my_func(0,self.data_train_discrete_unnormalized.shape[1])  #all_feature_column =[0,1,2,3]
        
       # print(f" all_feature_column {all_feature_column}")
        

        #self.getFeatureGroupsfromMultipleFeatures(self.data_test_discrete_unnormalized,features_chosoen_number)   #     get    self.multiple_feature_groups = [feature_selected] self.left_feature_group = []
        """
        get all combinations of features
        total_combinations = []
        for i in range(0,len( self.multple_feature_groups)):
            combinations =self.getAllCombinationsInOneFeatureGroup( self.multple_feature_groups[i],self.data_test_discrete_unnormalized)
            total_combinations.append(combinations)
        if len(self.left_feature_group)>2:
            left_combinations =self.getAllCombinationsInOneFeatureGroup(self.left_feature_group,self.data_test_discrete_unnormalized)
            print(f" left_combinations {left_combinations} ")
            self.reminder_combinations = left_combinations #left_combinations =  combiantion of feature  left_reminder_list [6,7]
        self.total_combinations = total_combinations   # self.total_combinations = [[a], [b]] a = combiantion of feature (0,1,2)b = combiantion of feature (3,4,5)
        print(f" self.total_combinations  {self.total_combinations }")
        print(f" self.reminder_combinations  {self.reminder_combinations }")
        """
        # first step,  get discrete data classification 


        #clf = LinearDiscriminantAnalysis()
       # clf = RandomForestClassifier(n_estimators = 500, criterion = 'gini',max_features =None)
        #clf = KNeighborsClassifier(weights="distance")
        #clf = svm.SVC(kernel='sigmoid')
        #clf = GaussianNB()
        #clf = LogisticRegression(penalty ='l2',multi_class ='multinomial', random_state=0,l1_ratio =0)
        #clf = LogisticRegression()
        #clf = DecisionTreeClassifier(criterion='entropy', max_features='sqrt')


       # clf.fit(self.data_train_discrete_unnormalized, self.train_label_all)

        #class_result_test = clf.predict(self.data_test_discrete_unnormalized)
       # self.getScores("Original",self.test_label_all,class_result_test)
    
      
        
        #self.getAllFeatureCombinationsGroupGranular(self.data_train_discrete_unnormalized)



        self.getAllfeatureGroups()  #group each column by feature value get    self.all_feature_groups    
        self.som_discrete.fit(self.data_train_discrete_unnormalized)   
     #   print(f"elf.som.weights0 shape {self.som.weights0.shape}")
        weight_original = self.som_discrete.weights0
        self.train_W0_predicted_label = self.som_discrete.predict(self.data_train_discrete_unnormalized,weight_original)   

       # print(f"self.train_W0_predicted_label 1{self.train_W0_predicted_label}")
        predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(self.som_discrete.weights0.shape[0], self.train_W0_predicted_label) 




     #1   self.getFeatureCombination(all_feature_column,features_chosoen_number)  #get self.all_combinations [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        
        #self.train_combinationns = self.getOrignalDataNewCombinationEmbedding(self.data_train_discrete_unnormalized,self.all_combinations)
        #self.test_cominations = self.getOrignalDataNewCombinationEmbedding(self.data_test_discrete_unnormalized,self.all_combinations)

        
        
        
        # get self.all_feature_combinations_sog_embedding
       #1 self.all_feature_combinations_sog_embedding ={}
       #1 for element in self.all_combinations:            
       #1        #print(f"pair {element} ")  #element =(0,1)  pair of features
       #1        one_tuple_feature_clusters ,one_tuple_feature_clusters_union = self.clusterMultipleFeatures(self.data_train_discrete_unnormalized,element)
       #1        #print(f"one_tuple_feature_clusters len  {len(one_tuple_feature_clusters)}")
       #1        self.all_feature_combinations_sog_embedding[element] = self.getMultipleFeaturesNeuronProbability_fuzzy(one_tuple_feature_clusters,one_tuple_feature_clusters_union, predicted_clusters_indexes)
       #1       # print(f" self.all_feature_combinations_sog_embedding[element]  { self.all_feature_combinations_sog_embedding[element]}")
       # print(f"all_feature_combinations_sog_embedding {self.all_feature_combinations_sog_embedding}")

    
        self.getEachNeuronProbabilityOfEachFeatureValue_fuzzy(predicted_clusters_indexes)


        self.train_new_embedding_sog_fuzzy = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_train_discrete_unnormalized)
        self.test_new_embedding_sog_fuzzy = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_test_discrete_unnormalized)
        print(f"self.train_new_embedding_sog_fuzzy.shape  {self.train_new_embedding_sog_fuzzy.shape}")
        #self.train_new_embedding_sog_fuzzy,self.train_new_embedding_sog_fuzzy_features  = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_train_discrete_unnormalized,features_chosoen)
        #print(f"self.train_new_embedding_sog_fuzzy_features {self.train_new_embedding_sog_fuzzy_features.shape}")
        #self.test_new_embedding_sog_fuzzy,self.test_new_embedding_sog_fuzzy_features  = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_test_discrete_unnormalized,features_chosoen)
        #print(self.discrete_data_embedding_sog_fuzzy.shape)

        #self.getEachNeuronProbabilityOfMultipleFeatureValue_fuzzy(predicted_clusters_indexes)

       
        fcm = FCM(n_clusters=self.som_discrete.weights0.shape[0])
        fcm.fit(self.data_train_discrete_unnormalized)

        self.train_W0_predicted_label = fcm.predict(self.data_train_discrete_unnormalized)   
      #  print(f"self.train_W0_predicted_label 2{self.train_W0_predicted_label}")
        predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(self.som_discrete.weights0.shape[0], self.train_W0_predicted_label) 

        self.getEachNeuronProbabilityOfEachFeatureValue_fuzzy(predicted_clusters_indexes)

        self.train_new_embedding_fuzzyc = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_train_discrete_unnormalized)
        self.test_new_embedding_fuzzyc = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_test_discrete_unnormalized)

        self.train_discrete_extra_sog_fuzzyc =  np.concatenate((self.train_new_embedding_sog_fuzzy,self.train_new_embedding_fuzzyc), axis=1)  
        self.test_discrete_extra_sog_fuzzyc =  np.concatenate((self.test_new_embedding_sog_fuzzy,self.test_new_embedding_fuzzyc), axis=1)  


        #1 self.discrete_data_embedding_sog_multiple_feature_fuzzy = self.getEmbeddingWithNeuronProbablity_multiple_features_fuzzy(self.data_train_discrete_unnormalized,self.all_combinations)
        #1 self.test_new_embedding_sog_multiple_feature_fuzzy = self.getEmbeddingWithNeuronProbablity_multiple_features_fuzzy(self.data_test_discrete_unnormalized,self.all_combinations)

        #print(f"self.discrete_data_embedding_sog_multiple_feature_fuzzy shape {self.discrete_data_embedding_sog_multiple_feature_fuzzy.shape}")
        
        print(f"self.train_discrete_extra_sog_fuzzyc { self.train_discrete_extra_sog_fuzzyc.shape}") 
        #self.train_new_embedding_sog_fuzzy_remove_zero  = self.train_new_embedding_sog_fuzzy [:, np.any(self.train_new_embedding_sog_fuzzy , axis=0)]
        #self.test_new_embedding_sog_fuzzy_remove_zero  = self.test_new_embedding_sog_fuzzy [:, np.any(self.test_new_embedding_sog_fuzzy , axis=0)]

       #1 self.train_discrete_extra_sog_fuzzy =  np.concatenate((self.train_new_embedding_sog_fuzzy,self.discrete_data_embedding_sog_multiple_feature_fuzzy), axis=1)  
       #1 self.test_discrete_extra_sog_fuzzy =  np.concatenate((self.test_new_embedding_sog_fuzzy,self.test_new_embedding_sog_multiple_feature_fuzzy), axis=1)  

      #1  print(f"self.discrete_data_embedding_sog_multiple_feature_fuzzy.shape  {self.discrete_data_embedding_sog_multiple_feature_fuzzy.shape}")
        #clf2 =  LinearDiscriminantAnalysis(solver ='eigen',shrinkage ='auto' )
        #clf2 = KNeighborsClassifier()
        #clf2 = svm.SVC(C = 10000)
        clf2 = RandomForestClassifier(n_estimators = 500, criterion = 'gini',max_features =None)
        #clf2 =GaussianNB(var_smoothing = 0.2)
        #clf2 = DecisionTreeClassifier(criterion='entropy', max_features='sqrt')
        #clf2 = LogisticRegression(penalty ='l2',multi_class ='multinomial', random_state=0,l1_ratio =0)
       # clf2 = LogisticRegression()
        clf2 = LogisticRegression(penalty ='l2' ,multi_class ='auto',class_weight = None,max_iter = 10, solver = 'liblinear')
        #clf2.fit(self.data_train_discrete_unnormalized, self.train_label_all)
        clf2.fit(self.train_new_embedding_sog_fuzzy, self.train_label_all)
        class_result_test2 = clf2.predict(self.test_new_embedding_sog_fuzzy)
        #class_result_test2 = clf2.predict(self.data_test_discrete_unnormalized)
        #print("SOG Classficiation")
       # print(f"self.train_new_embedding_sog_fuzzy shape {self.train_new_embedding_sog_fuzzy.shape}")
        self.getScores("Original",self.test_label_all,class_result_test2)
  

        

    #    self.getAllfeatureGroupsCombination() # get 
#
    #    self.som_combination = newSom.SOM(self.som_discrete.m , self.som_discrete.n, dim= self.train_combinationns.shape[1])  
    #    self.som_enchanced_weight = self.getOrignalDataNewCombinationEmbedding( self.som_discrete.weights0,self.all_combinations)
    #    self.som_combination.trained = True
    #   # self.som_combination.fit(self.train_combinationns)   
    # #   print(f"elf.som.weights0 shape {self.som.weights0.shape}")
    #   # weight_combination = self.som_combination.weights0
    #    self.train_W0_predicted_label = self.som_combination.predict(self.train_combinationns,self.som_enchanced_weight)   
    #    predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(self.som_combination.shape[0], self.train_W0_predicted_label) 
    #    self.getEachNeuronProbabilityOfEachFeatureValue_fuzzy_combination(predicted_clusters_indexes)
#
    #    self.train_combination_sog_fuzzy = self.getEmbeddingWithNeuronProbablity_fuzzy_combination(self.train_combinationns)
    #    self.test_combination_sog_fuzzy = self.getEmbeddingWithNeuronProbablity_fuzzy_combination(self.test_cominations)
#
#
    #    self.train_discrete_extra_sog_fuzzy =  np.concatenate((self.train_new_embedding_sog_fuzzy,self.train_combination_sog_fuzzy), axis=1)  
    #    self.test_discrete_extra_sog_fuzzy =  np.concatenate((self.test_new_embedding_sog_fuzzy,self.test_combination_sog_fuzzy), axis=1)  
    #    print(f"self.train_discrete_extra_sog_fuzzy shape {self.train_discrete_extra_sog_fuzzy.shape}")

        #clf3 = svm.SVC(C = 10000)
        clf3 = RandomForestClassifier(n_estimators = 500, criterion = 'gini',max_features =None)
        #clf3 = GaussianNB(var_smoothing = 0.2)
       # clf3 =  LinearDiscriminantAnalysis(solver ='eigen',shrinkage ='auto' )
       # clf3 = KNeighborsClassifier()
        #clf3 = DecisionTreeClassifier(criterion='entropy', max_features='sqrt')
        #clf3 = LogisticRegression(penalty ='l2' ,multi_class ='auto',class_weight = None,max_iter =10, solver = 'liblinear')
        #clf3 = LogisticRegression()
        #clf3 = LogisticRegression(penalty ='l2' ,multi_class ='auto',class_weight = None,max_iter = 5000, solver = 'liblinear')
        clf3.fit(self.train_discrete_extra_sog_fuzzyc, self.train_label_all)
        class_result_test3 = clf3.predict(self.test_discrete_extra_sog_fuzzyc)
        #print("SOG Classficiation")
       # print(f"self.train_new_embedding_sog_fuzzy shape {self.train_new_embedding_sog_fuzzy.shape}")
        self.getScores("SOG",self.test_label_all,class_result_test3)
  

        if self.accuracy_score_sog < self.accuracy_score_original:
             print("Not good accuracy result !!!!!")
        if self.recall_score_sog < self.recall_score_original:
             print("Not good recall_score result !!!!!")
        if self.precision_score_sog < self.precision_score_original:
            print("Not good precision_score_original !!!!!")
        if self.f1_score_sog < self.f1_score_original:
             print("Not good accuracy result !!!!!")
       # if self.log_loss_sog > self.log_loss_original:
       #      print("Not good log_loss result !!!!!")

     

      
    def do_DCOSOM(self,features_chosoen):
        """
        do discrete optimized SOM 
        
        """

        # first step,  get discrete data classification 


        #clf = LinearDiscriminantAnalysis(solver='lsqr',shrinkage ='auto')
        #clf = KNeighborsClassifier(n_neighbors = 10, weights="distance",leaf_size = 10,algorithm ='ball_tree',p =1,n_jobs = -1)
        #clf = svm.SVC(C=1, gamma=1, kernel='rbf')# # #kernel='rbf',gamma ='auto',decision_function_shape = 'ovr'
        #clf = GaussianNB()
        clf =  RandomForestClassifier(n_estimators = 1000, criterion = 'log_loss',max_features =None)
        #clf = LogisticRegression(penalty ='l2',class_weight ='balanced',solver ='sag',max_iter = 200, multi_class ='multinomial')
        #clf = DecisionTreeClassifier(criterion='gini', splitter = 'best',min_samples_split = 100, max_depth = 100,max_features='log2')


        clf.fit(self.data_train_all, self.train_label_all)

        class_result_test = clf.predict(self.data_test_all)
        self.getScores("Original",self.test_label_all,class_result_test)
        #print(f"edata_test_all shape {self.data_test_all.shape}")

        self.getAllfeatureGroups()  #group each column by feature value get    self.all_feature_groups    
        self.som.fit(self.data_train_all)   
   
        weight_original = self.som.weights0
        self.train_W0_predicted_label = self.som.predict(self.data_train_all,weight_original)   

     
        predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(self.som.weights0.shape[0], self.train_W0_predicted_label) 

        
        self.getEachNeuronProbabilityOfEachFeatureValue_fuzzy(predicted_clusters_indexes)
        #self.discrete_data_embedding_sog_fuzzy = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_train_discrete_unnormalized)
        self.train_new_embedding_sog_fuzzy,self.train_new_embedding_sog_fuzzy_features  = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_train_discrete_unnormalized,features_chosoen)
        #print(f"self.train_new_embedding_sog_fuzzy_features {self.train_new_embedding_sog_fuzzy_features.shape}")
        self.test_new_embedding_sog_fuzzy,self.test_new_embedding_sog_fuzzy_features  = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_test_discrete_unnormalized,features_chosoen)
        #print(f"self.test_new_embedding_sog_fuzzy_features {self.test_new_embedding_sog_fuzzy_features.shape}")
        self.train_hybrid_embedding_sog = np.concatenate((self.data_train_continuous,self.train_new_embedding_sog_fuzzy), axis=1)  
        self.test_hybrid_embedding_sog = np.concatenate((self.data_test_continuous,self.test_new_embedding_sog_fuzzy), axis=1)  

        self.train_hybrid_embedding_sog_features = np.concatenate((self.data_train_continuous,self.train_new_embedding_sog_fuzzy_features), axis=1)  
        #print(f" self.data_test_continuous shape {self.data_test_continuous.shape}   self.test_new_embedding_sog_fuzzy_features shape {self.test_new_embedding_sog_fuzzy_features.shape}")
        self.test_hybrid_embedding_sog_features = np.concatenate((self.data_test_continuous,self.test_new_embedding_sog_fuzzy_features), axis=1)  
        #print(f" self.test_hybrid_embedding_sog_features shape {self.test_hybrid_embedding_sog_features.shape}")
        #clf2 =  LinearDiscriminantAnalysis(solver='lsqr',shrinkage ='auto')
        #clf2 =  KNeighborsClassifier(n_neighbors = 10, weights="distance",leaf_size = 10, algorithm ='ball_tree',p =1,n_jobs = -1)
        clf2 = svm.SVC(C=1, gamma=1, kernel='rbf')#rbf
        #clf2 = GaussianNB()
        #clf2 = RandomForestClassifier(n_estimators = 1000, criterion = 'log_loss',max_features =None)
       # clf2 = DecisionTreeClassifier(criterion='gini', splitter = 'best', min_samples_split = 100,  max_depth = 100,max_features='log2')
        #clf2 = LogisticRegression(penalty ='l2',class_weight ='balanced',solver ='sag',max_iter = 200,multi_class ='multinomial')
        clf2.fit(self.train_hybrid_embedding_sog_features, self.train_label_all)
        
        class_result_test2 = clf2.predict(self.test_hybrid_embedding_sog_features)
        print("SOG Classficiation")
        #print(f"self.train_new_embedding_sog_fuzzy shape {self.train_hybrid_embedding_sog.shape}")
        self.getScores("SOG",self.test_label_all,class_result_test2)
  
        if self.accuracy_score_sog < self.accuracy_score_original:
             print("Not good accuracy result !!!!!")
        if self.recall_score_sog < self.recall_score_original:
             print("Not good recall_score result !!!!!")
        if self.precision_score_sog < self.precision_score_original:
            print("Not good precision_score_original !!!!!")
        if self.f1_score_sog < self.f1_score_original:
             print("Not good accuracy result !!!!!")
       # if self.log_loss_sog > self.log_loss_original:
       #      print("Not good log_loss result !!!!!")
