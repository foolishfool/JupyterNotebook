"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
find intra communiyt in each neuron memberships and do the whole mapping and retest 
"""

#from curses.ascii import NULL
from pickletools import optimize
from sklearn.decomposition import PCA
from sklearn.cluster import Birch
#from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import BisectingKMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
from sklearn.cluster import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from scipy.special import softmax
from sklearn import metrics
from scipy import spatial
import numpy as np
from operator import add
import math
import sys
import operator
import copy
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score
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
                 soms_discrete,  # soms
                 som_discrete_original, #used for comparision as a original som
                 som_total_discrete_transferred, #the last som got 
                 data_train_all, 
                 data_train_continuous,
                 data_train_discrete,
                 data_train_discrete_normalized,
                 data_train_baseline_encoded,
                 data_test_all,
                 data_test_continuous,
                 data_test_discrete,
                 data_test_discrete_normalized,
                 data_test_baseline_encoded,
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
        self.soms_discrete = soms_discrete  
        self.som_total_discrete_transferred = som_total_discrete_transferred  
        self.som_discrete_baseline_encoder = som_discrete_original  
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

        self.data_train_baseline_encoded = data_train_baseline_encoded
        self.data_test_baseline_encoded = data_test_baseline_encoded

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
       # print(len(y_true))
       # print(len(y_pred))
        self.purity_score(scorename,y_true,y_pred)
        self.nmiScore(scorename,y_true,y_pred)
        self.ariScore(scorename,y_true,y_pred)



        
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
    
    def getCommanIndexesRatioInNeurons_fuzzy(self, feature_group, one_neuron_predict_group):
        # difference with getCommanIndexesRatioInNeurons is the denomitor
        if len(one_neuron_predict_group) !=0:
            return round(len(np.intersect1d(feature_group, one_neuron_predict_group))/len(one_neuron_predict_group),3)
        else: return 0
    def getFeatureGroups(self, feature_column_data):
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
    
    def getAllfeatureGroups(self):
        self.all_feature_groups = {}
        #print(f"np.shape(self.data_train_discrete_unnormalized)[1]{ np.shape(self.data_train_discrete_unnormalized)[1]}")
        for i in range(0,np.shape(self.data_train_discrete_unnormalized)[1]):
            # i is the comlumn number
            self.all_feature_groups[i] = self.getFeatureGroups(self.data_train_discrete_unnormalized[:,i])
            #print(f" i {i} self.all_feature_gropus {  self.all_feature_groups[i] } ")


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
        dim= self.som_discrete_baseline_encoder.weights0.shape[0]
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

    def getOneSingleFeatureNeuronProbability_subjective(self, one_feature_dic, neuron_predicted_groups):
        onesinglefeatureneuronprobablity = {}
        sum=0
        for key in one_feature_dic.keys():
            sum+= len(one_feature_dic[key])

        #print(f"one_feature_dic.keys() {one_feature_dic.keys()} ")
        for key in one_feature_dic.keys():
            probability_list = []
            for i in range(0,len(neuron_predicted_groups)) :  
                #print("New neuron _predicted group !!!")       
                probability =  len(one_feature_dic[key])/sum
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

    def getOneSingleFeatureNeuronProbability_softmax(self, one_feature_dic, neuron_predicted_groups):
        onesinglefeatureneuronprobablity = {}
        number_list= []
        for key in one_feature_dic.keys():
            number_list.append(len(one_feature_dic[key]))
        softmax_result = softmax(number_list)
        i = 0
        #print(f"number_list { number_list} softmax_result{softmax_result} ")
        for key in one_feature_dic.keys():
            probability_list = []
            for j in range(0,len(neuron_predicted_groups)) :  
                #print("New neuron _predicted group !!!")       
                probability_list.append(softmax_result[i])
            #print(f"key {key} probability_list{probability_list}")
            probability_list = np.array(probability_list)
            onesinglefeatureneuronprobablity[key] = probability_list
            i+=1
            #print(f"key {key} probability_list) {probability_list} ")
        return onesinglefeatureneuronprobablity
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
            #print(f"  feature  {i} self.all_features_mapping_fuzzy[i] {self.all_features_mapping_fuzzy[i]}")
    def getEachNeuronProbabilityOfEachFeatureValue_subjective(self,neuron_predicted_groups):
        self.all_features_mapping ={}
        for i in range(0, len(self.all_feature_groups)):
           # print(f"  feature  {i}!!!!!!!!!!!")
            #self.all_features_mapping[i] = self.getOneSingleFeatureNeuronProbability(self.all_feature_groups[i],neuron_predicted_groups)
            self.all_features_mapping[i] = self.getOneSingleFeatureNeuronProbability_subjective(self.all_feature_groups[i],neuron_predicted_groups)
            #print(f"  self.all_features_mapping[i]  {self.all_features_mapping[i]}!!!!!!!!!!!")

   
    def getEachNeuronProbabilityOfEachFeatureValue_softmax(self,neuron_predicted_groups):
        self.all_features_mapping ={}
        for i in range(0, len(self.all_feature_groups)):
           # print(f"  feature  {i}!!!!!!!!!!!")
            #self.all_features_mapping[i] = self.getOneSingleFeatureNeuronProbability(self.all_feature_groups[i],neuron_predicted_groups)
            self.all_features_mapping[i] = self.getOneSingleFeatureNeuronProbability_softmax(self.all_feature_groups[i],neuron_predicted_groups)
            #print(f"  self.all_features_mapping[i]  {self.all_features_mapping[i]}!!!!!!!!!!!")
    
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
    
    def getEmbeddingWithNeuronProbablity_fuzzy(self,X,features_choosen):
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
            for l in range(0, len(x)):
                extra_embedding_one_feature = []
                if l in features_choosen:
                     if x[l] in self.all_features_mapping_fuzzy[l].keys():
                       # print(f"self.all_features_mapping_fuzzy[l][x[l]] {self.all_features_mapping_fuzzy[l][x[l]]}")
                        for value in self.all_features_mapping_fuzzy[l][x[l]]:
                            extra_embedding_one_feature.append(value)
                    # print(f"extra_embedding_one_feature {extra_embedding_one_feature}")
                     #print(f"extra_embeddings1  {extra_embeddings}  extra_embedding_one_feature {extra_embedding_one_feature}")
                     extra_embeddings = list( map(add, extra_embeddings, extra_embedding_one_feature) )
                     #print(f"extra_embeddings2  {extra_embeddings} ")
            newdata_features_choosen = newdata_features_choosen + extra_embeddings
           # print(f"the original discrete data : {x} ")
            #print(f"the proposed encoded data representation: {newdata}  newdata_features_choosen {newdata_features_choosen} ")
            newX.append(newdata)
            newX_features_choosen.append(newdata_features_choosen)
       # print(f"new embedding {newX}")
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


        
   
    def do_DOSOM(self,features_chosoen,fuzzy_set_no_probability = False):
        """
        do discrete optimized SOM 
        
        """

        self.getAllfeatureGroups()  #group each column by feature value get    self.all_feature_groups
        
    

        self.som.fit(self.data_train_all)   
        #print(self.data_train_all)
        weight_original = self.som.weights0
        self.train_W0_predicted_label = self.som.predict(self.data_train_all,weight_original)   
       # print(f" self.train_W0_predicted_label  { self.train_W0_predicted_label }")
        #clustering  = AffinityPropagation(random_state=5,damping = 0.9, max_iter = 1000).fit(self.data_train_all)
       # clustering  = DBSCAN(eps =10).fit(self.data_train_all)
       # clustering  = HDBSCAN(min_cluster_size=20).fit(self.data_train_all)
       # clustering  = OPTICS(min_samples=20).fit(self.data_train_all)
       # clustering  = GaussianMixture(n_components=self.som.m, random_state=0).fit(self.data_train_all)

      #  clustering  = Birch(n_clusters=None ).fit(self.data_train_all)        #print(np.unique(clustering.labels_))      
         # clustering  = SpectralClustering(n_clusters=self.som.m,  assign_labels='discretize', random_state=0).fit(self.data_train_all)
        #clustering  = AgglomerativeClustering().fit(self.data_train_all)
       # clustering = MeanShift().fit(self.data_train_all)

        #self.train_W0_predicted_label = clustering.labels_
        #self.train_W0_predicted_label = clustering.predict(self.data_train_all)
        #predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(clustering.n_clusters_, self.train_W0_predicted_label) 
        #predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(len(clustering.cluster_centers_), self.train_W0_predicted_label) 
        #predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(len(clustering.cluster_centers_), self.train_W0_predicted_label) 
        #predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(len(clustering.means_), self.train_W0_predicted_label) 
        #predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(len(clustering.subcluster_centers_), self.train_W0_predicted_label) 
        predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(self.som.weights0.shape[0], self.train_W0_predicted_label) 
       # print(f"self.som.m {self.som.m}")
        #Get SOG Fuzzy Mapping
      
    
        if fuzzy_set_no_probability == True:
            self.getEachNeuronProbabilityOfEachFeatureValue(predicted_clusters_indexes)
           
        #train by baseline encoder
        if  fuzzy_set_no_probability == False:
            
          

            self.getEachNeuronProbabilityOfEachFeatureValue_fuzzy(predicted_clusters_indexes)
            #self.discrete_data_embedding_sog_fuzzy = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_train_discrete_unnormalized)
            self.train_new_embedding_sog_fuzzy,self.train_new_embedding_sog_fuzzy_features= self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_train_discrete_unnormalized,features_chosoen)
            self.test_new_embedding_sog_fuzzy,self.test_new_embedding_sog_fuzzy_features= self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_test_discrete_unnormalized,features_chosoen)
            self.som_sog = newSom.SOM(self.som.m , self.som.m,self.train_new_embedding_sog_fuzzy.shape[1]) 
            print(f"self.train_new_embedding_sog_fuzzy.shape[1] {self.train_new_embedding_sog_fuzzy.shape[1]}")
            self.som_sog.fit(self.train_new_embedding_sog_fuzzy)
            self.test_discrete_W0_predicted_label = self.som_sog.predict(self.test_new_embedding_sog_fuzzy,self.som_sog.weights0) 
           # self.som_discrete_baseline_encoder.fit(self.data_train_baseline_encoded)
           # weight_discrete_baseline = self.som_discrete_baseline_encoder.weights0
           # self.train_discrete_W0_predicted_label = self.som_discrete_baseline_encoder.predict(self.data_train_baseline_encoded,weight_discrete_baseline)   
           #predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(self.som_discrete_baseline_encoder.weights0.shape[0], self.train_discrete_W0_predicted_label)   






            # the value in predicted_clusters are true label value    
           # transferred_predicted_label_train_W0 =  self.convertPredictedLabelValue(self.train_discrete_W0_predicted_label,self.PLabel_to_Tlabel_Mapping_W_Original)      
            #print(f"transferred_predicted_label_train_W0 {transferred_predicted_label_train_W0}  np.unique  {np.unique(transferred_predicted_label_train_W0)}"  )
           # self.getScore("train_discrete_score_W0",self.train_label_all,transferred_predicted_label_train_W0)
            
            #*** when validate needs to use current mapping(the real situation mappping) rather than the training sessino mapping
            #self.test_discrete_W0_predicted_label = self.som_discrete_baseline_encoder.predict(self.data_test_baseline_encoded,weight_discrete_baseline) 
            predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(self.som_sog.weights0.shape[0], self.test_discrete_W0_predicted_label)   
            
          #  clustering2  = SpectralClustering(n_clusters=self.som.m,  assign_labels='discretize', random_state=0).fit(self.data_test_baseline_encoded)
           # self.test_discrete_W0_predicted_label = clustering2.labels_


            #clustering  = AffinityPropagation(random_state=5,damping = 0.9, max_iter = 1000).fit(self.data_train_baseline_encoded)
           # clustering  = SpectralClustering(n_clusters=self.som.m,  assign_labels='discretize', random_state=0).fit(self.data_test_baseline_encoded)
           # clustering = MeanShift().fit(self.data_train_baseline_encoded)
            #clustering = DBSCAN(eps =10).fit(self.data_test_baseline_encoded)
           # clustering  = HDBSCAN(min_cluster_size=20).fit(self.data_test_baseline_encoded)
            #clustering  = OPTICS(min_samples=20).fit(self.data_test_baseline_encoded)
            #clustering  = GaussianMixture(n_components=self.som.m, random_state=0).fit(self.data_test_baseline_encoded)
            #clustering  = Birch(n_clusters=None ).fit(self.data_test_baseline_encoded)      
           # self.test_discrete_W0_predicted_label = clustering.predict(self.data_test_baseline_encoded)
            #clustering  = BisectingKMeans(n_clusters=self.som.m, random_state=0).fit(self.data_test_baseline_encoded)
            #self.test_discrete_W0_predicted_label = clustering.labels_
              
            #self.test_discrete_W0_predicted_label = clustering.predict(self.data_test_baseline_encoded)
            #predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(len(clustering.cluster_centers_), self.test_discrete_W0_predicted_label)   
            #predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(len(set(np.unique(clustering.labels_))), self.test_discrete_W0_predicted_label)   
            #AgglomerativeClustering
          #  clustering  = AgglomerativeClustering().fit(self.data_test_baseline_encoded)
            #self.test_discrete_W0_predicted_label = clustering.labels_  
            #predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(clustering.n_clusters_, self.test_discrete_W0_predicted_label)   

            self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters_indexes,self.test_label_all) ,0)    
            transferred_predicted_label_test_W0 =  self.convertPredictedLabelValue(self.test_discrete_W0_predicted_label,self.PLabel_to_Tlabel_Mapping_W_Original)   
            self.getScore("test_discrete_score_W0",self.test_label_all,transferred_predicted_label_test_W0)

        else:
           # self.discrete_data_embedding_sog_fuzzy = self.getEmbeddingWithNeuronProbablityAndMultipleFeatures_fuzzy(self.data_train_discrete_unnormalized)
            self.test_new_embedding_sog_fuzzy = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_test_discrete_unnormalized,features_chosoen)
            #self.KmeansClusterEncoding(self.discrete_data_embedding_sog_fuzzy,self.test_new_embedding_sog_fuzzy,"train_discrete_score_W0","test_discrete_score_W0")
            self.SpectralClustering(self.discrete_data_embedding_sog_fuzzy,self.test_new_embedding_sog_fuzzy,"test_discrete_score_W0")
   

 

        '''
        
        #*** when validate needs to use current mapping(the real situation mappping) rather than the training sessino mapping
        self.test_discrete_W0_predicted_label = self.som.predict(self.data_test_all,weight_original)   
        predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(weight_original.shape[0], self.test_discrete_W0_predicted_label)   
        self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters_indexes,self.test_label_all) ,0)    
        transferred_predicted_label_test_W0 =  self.convertPredictedLabelValue(self.test_discrete_W0_predicted_label,self.PLabel_to_Tlabel_Mapping_W_Original)   
        self.getScore("test_discrete_score_W0",self.test_label_all,transferred_predicted_label_test_W0)

        '''
        #get new embedding data with SOG mapping
        #print(f"self.data_test_discrete_unnormalized {self.data_test_discrete_unnormalized}")
       # self.discrete_data_embedding_sog = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_train_discrete_unnormalized)   
      #  self.test_new_embedding_sog = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_test_discrete_unnormalized)
       # print(f"self.test_new_embedding_sog {self.test_new_embedding_sog}")
        '''
        self.test_discrete_baseline_predicted_label = self.som.predict(self.test_new_embedding_sog,weight_original) 

        predicted_clusters_transferred = self.get_indices_and_data_in_predicted_clusters(weight_original.shape[0],  self.test_discrete_baseline_predicted_label) 
        self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters_transferred,self.test_label_all) ,2)  
        transferred_predicted_label_test_W_transferred =  self.convertPredictedLabelValue( self.test_discrete_baseline_predicted_label,self.PLabel_to_Tlabel_Mapping_W_Discrete) 
        self.getScore("test_discrete_score_W_discrete",self.test_label_all,transferred_predicted_label_test_W_transferred)
        '''

       # self.KmeansClusterEncoding(self.discrete_data_embedding_sog,self.test_new_embedding_sog,"train_discrete_score_W_discrete","test_discrete_score_W_discrete")
       # self.AgglomerativeClustering(self.discrete_data_embedding_sog,self.test_new_embedding_sog,"test_discrete_score_W_discrete")
      #  self.AffinityPropagation(self.discrete_data_embedding_sog,self.test_new_embedding_sog,"test_discrete_score_W_discrete")
        #self.SpectralClustering(self.discrete_data_embedding_sog,self.test_new_embedding_sog,"test_discrete_score_W_discrete")   
        #self.MeanShift(self.discrete_data_embedding_sog,self.test_new_embedding_sog,"test_discrete_score_W_discrete")    
        #self.HDBSCAN(self.test_new_embedding_sog,"test_discrete_score_W_discrete")    
        #self.OPTICS(self.test_new_embedding_sog,"test_discrete_score_W_discrete")    
       # self.GaussianMixture(self.test_new_embedding_sog,"test_discrete_score_W_discrete")    
       # self.Birch(self.discrete_data_embedding_sog,self.test_new_embedding_sog,"test_discrete_score_W_discrete")
        #self.BisectingKMeans(self.discrete_data_embedding_sog,self.test_new_embedding_sog,"test_discrete_score_W_discrete")
        #print(f"self.training_new_embedding  {self.discrete_data_embedding.shape} " )
        # new som neuron number is not changed, m,n not change
       # if  fuzzy_set_no_probability == False:
       #     self.soself.som_sog = newSom.SOM(self.som_discrete_baseline_encoder.weights0.shape[0] , self.som_discrete_baseline_encoder.weights0.shape[1],dim) 
       # else:
       #     self.som_sog = newSom.SOM(self.som_sog_fuzzy.weights0.shape[0] , self.som_sog_fuzzy.weights0.shape[1],dim) 
       # self.som_sog.fit(self.discrete_data_embedding_sog)
       # 
       # weight_sog = self.som_sog.weights0

      # self.train_W_baseline_predicted_label = self.som_sog.predict(self.discrete_data_embedding_sog,weight_sog)    
      # predicted_clusters_indexes, current_clustered_datas = self.get_indices_and_data_in_predicted_clusters(self.som_sog.weights0.shape[0], self.train_W_baseline_predicted_label,self.discrete_data_embedding_sog)   

      # self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters_indexes,self.train_label_all) ,0)  
      # # the value in predicted_clusters are true label value    
      # transferred_predicted_label_train_W0 =  self.convertPredictedLabelValue(self.train_W_baseline_predicted_label,self.PLabel_to_Tlabel_Mapping_W_Original)      

      # self.getScore("train_discrete_score_W_discrete",self.train_label_all,transferred_predicted_label_train_W0)

      # self.test_new_embedding_sog = self.getEmbeddingWithNeuronProbablity(self.data_test_discrete_unnormalized)
        
        self.som_sog_features = newSom.SOM(self.som.m , self.som.m,self.train_new_embedding_sog_fuzzy_features.shape[1]) 

        self.som_sog_features.fit(self.train_new_embedding_sog_fuzzy_features)

        self.test_discrete_features_predicted_label = self.som_sog_features.predict(self.test_new_embedding_sog_fuzzy_features,self.som_sog_features.weights0) 

        predicted_clusters_transferred, current_clustered_datas_cleaned = self.get_indices_and_data_in_predicted_clusters(self.som_sog_features.weights0.shape[0],  self.test_discrete_features_predicted_label) 
        self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters_transferred,self.test_label_all) ,2)  
        transferred_predicted_label_test_W_transferred =  self.convertPredictedLabelValue( self.test_discrete_features_predicted_label,self.PLabel_to_Tlabel_Mapping_W_Discrete) 
        self.getScore("test_discrete_score_W_discrete",self.test_label_all,transferred_predicted_label_test_W_transferred)
      #   
        if self.test_discrete_score_W_discrete_p < self.test_discrete_score_W0_p:
             print("Not good purity result for discrete features !!!!!")
        if self.test_discrete_score_W_discrete_n < self.test_discrete_score_W0_n:
             print("Not good nmi result for discrete features !!!!!")
        if self.test_discrete_score_W_discrete_a < self.test_discrete_score_W0_a:
            print("Not good ari result for discrete features  !!!!!")
     


    def KmeansClusterEncoding(self,X,Y,scoreName):
        #X  is self.train_new_embedding_sog or self.train_new_embedding_sog_fuzzy
        kmeans = KMeans(n_clusters=self.som.weights0.shape[0], random_state=0, n_init="auto").fit(X)
        #self.train_W_baseline_predicted_label = kmeans.labels_  
        #predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(self.som.weights0.shape[0], self.train_W_baseline_predicted_label)   
        #self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters_indexes,self.train_label_all) ,0)  
        # the value in predicted_clusters are true label value    
        #transferred_predicted_label_train_W0 =  self.convertPredictedLabelValue(self.train_W_baseline_predicted_label,self.PLabel_to_Tlabel_Mapping_W_Original)      

        #self.getScore(scoreName1,self.train_label_all,transferred_predicted_label_train_W0)

        #self.test_new_embedding_sog_fuzzy = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_test_discrete_unnormalized)
        
        #Y is self.test_new_embedding_sog or self.test_new_embedding_sog_fuzzy
        self.test_discrete_features_predicted_label = kmeans.predict(Y)  
        predicted_clusters_transferred = self.get_indices_and_data_in_predicted_clusters(self.som.weights0.shape[0], self.test_discrete_features_predicted_label) 
        
        #self.test_discrete_baseline_predicted_label = self.som_sog_fuzzy.predict(self.test_new_embedding_sog_fuzzy,weight_sog_fuzzy) 
        self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters_transferred,self.test_label_all) ,2)  
        transferred_predicted_label_test_W_transferred =  self.convertPredictedLabelValue( self.test_discrete_features_predicted_label,self.PLabel_to_Tlabel_Mapping_W_Discrete) 
        self.getScore(scoreName,self.test_label_all,transferred_predicted_label_test_W_transferred)     



     
    def AffinityPropagation(self,X,Y,scoreName):
        clustering  = AffinityPropagation(random_state=5,damping = 0.9, max_iter = 1000).fit(X)

        self.test_discrete_features_predicted_label = clustering.predict(Y)  
        predicted_clusters_transferred = self.get_indices_and_data_in_predicted_clusters(len(clustering.cluster_centers_), self.test_discrete_features_predicted_label) 
        
        self.getTestScoreBasedonPredicted_clusters_transferred(scoreName,predicted_clusters_transferred)
    def BisectingKMeans(self,X,Y,scoreName):
        clustering  = BisectingKMeans(n_clusters=self.som.m, random_state=0).fit(X)

        self.test_discrete_features_predicted_label = clustering.predict(Y)  
        predicted_clusters_transferred = self.get_indices_and_data_in_predicted_clusters(self.som.m, self.test_discrete_features_predicted_label) 
        
        self.getTestScoreBasedonPredicted_clusters_transferred(scoreName,predicted_clusters_transferred)
    
    def getTestScoreBasedonPredicted_clusters_transferred(self, scoreName,predicted_clusters_transferred):
        self.getLabelMapping( self.get_mapped_class_in_clusters(predicted_clusters_transferred,self.test_label_all) ,2)  
        transferred_predicted_label_test_W_transferred =  self.convertPredictedLabelValue( self.test_discrete_features_predicted_label,self.PLabel_to_Tlabel_Mapping_W_Discrete) 
        self.getScore(scoreName,self.test_label_all,transferred_predicted_label_test_W_transferred)     


    def SpectralClustering(self,Y,scoreName):
        clustering  = SpectralClustering(n_clusters=self.som.m,  assign_labels='discretize', random_state=0).fit(Y)

        self.test_discrete_features_predicted_label = clustering.labels_
        predicted_clusters_transferred = self.get_indices_and_data_in_predicted_clusters(self.som.m, self.test_discrete_features_predicted_label) 
        
        self.getTestScoreBasedonPredicted_clusters_transferred(scoreName,predicted_clusters_transferred)

        
    def AgglomerativeClustering(self,X,Y,scoreName):
        clustering  = AgglomerativeClustering().fit(X)

        self.test_discrete_features_predicted_label = clustering.labels_
        predicted_clusters_transferred = self.get_indices_and_data_in_predicted_clusters(clustering.n_clusters_, self.test_discrete_features_predicted_label) 
        
        self.getTestScoreBasedonPredicted_clusters_transferred(scoreName,predicted_clusters_transferred)        
    
    def MeanShift(self,X, Y,scoreName):
        clustering  = MeanShift(bandwidth=2).fit(X)
        self.test_discrete_features_predicted_label = clustering.predict(Y)
        predicted_clusters_transferred = self.get_indices_and_data_in_predicted_clusters(len(clustering.cluster_centers_), self.test_discrete_features_predicted_label) 
        
        self.getTestScoreBasedonPredicted_clusters_transferred(scoreName,predicted_clusters_transferred)    
    def Birch(self,X, Y,scoreName):
        clustering  = Birch(n_clusters=None).fit(X)
        self.test_discrete_features_predicted_label = clustering.predict(Y)
        predicted_clusters_transferred = self.get_indices_and_data_in_predicted_clusters(len(clustering.subcluster_centers_), self.test_discrete_features_predicted_label) 
        
        self.getTestScoreBasedonPredicted_clusters_transferred(scoreName,predicted_clusters_transferred)    

    def DBSCAN(self,Y,scoreName):
        clustering  = DBSCAN(eps =10).fit(Y)

        self.test_discrete_features_predicted_label = clustering.labels_
        predicted_clusters_transferred = self.get_indices_and_data_in_predicted_clusters(len(set(np.unique(clustering.labels_))), self.test_discrete_features_predicted_label) 
        
        self.getTestScoreBasedonPredicted_clusters_transferred(scoreName,predicted_clusters_transferred)  
        
    def HDBSCAN(self,Y,scoreName):
        clustering  = HDBSCAN(min_cluster_size=20).fit(Y)

        self.test_discrete_features_predicted_label = clustering.labels_
        predicted_clusters_transferred = self.get_indices_and_data_in_predicted_clusters(len(set(np.unique(clustering.labels_))), self.test_discrete_features_predicted_label) 
        
        self.getTestScoreBasedonPredicted_clusters_transferred(scoreName,predicted_clusters_transferred)  
    
    def OPTICS(self,Y,scoreName):
        clustering  = OPTICS(min_samples=20).fit(Y)

        self.test_discrete_features_predicted_label = clustering.labels_
        predicted_clusters_transferred = self.get_indices_and_data_in_predicted_clusters(len(set(np.unique(clustering.labels_))), self.test_discrete_features_predicted_label) 
        
        self.getTestScoreBasedonPredicted_clusters_transferred(scoreName,predicted_clusters_transferred)  
    
    def GaussianMixture(self,Y,scoreName):
        clustering  = GaussianMixture(n_components=self.som.m, random_state=0).fit(Y)

        self.test_discrete_features_predicted_label = clustering.predict(Y)
        predicted_clusters_transferred = self.get_indices_and_data_in_predicted_clusters(len(clustering.means_), self.test_discrete_features_predicted_label) 
        
        self.getTestScoreBasedonPredicted_clusters_transferred(scoreName,predicted_clusters_transferred)  


    def get_SOG_Embedding_PCA(self,fuzzy_set_no_probability = False):
        """
        do discrete optimized SOM 
        
        """

        self.getAllfeatureGroups()  #group each column by feature value get    self.all_feature_groups
        
        self.som.fit(self.data_train_all)   
        weight_original = self.som.weights0
        self.train_W0_predicted_label = self.som.predict(self.data_train_all,weight_original)   
        predicted_clusters_indexes = self.get_indices_and_data_in_predicted_clusters(self.som.weights0.shape[0], self.train_W0_predicted_label) 
      
        #Get SOG Fuzzy Mapping
        self.getEachNeuronProbabilityOfEachFeatureValue_fuzzy(predicted_clusters_indexes)
    
        self.discrete_data_embedding_sog = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_train_discrete_unnormalized)   
        self.test_new_embedding_sog = self.getEmbeddingWithNeuronProbablity_fuzzy(self.data_test_discrete_unnormalized)
        #print( self.discrete_data_embedding_sog)
        self.PCA_Comparision()

  
    def PCA_Comparision(self):
        pca1 = PCA()
        pca1.fit_transform(self.data_train_baseline_encoded)
        print(self.data_train_baseline_encoded.shape)
        #print("Base Line PCA feature importance")
        #print(abs(pca1.components_))
        self.RankingFeatureImportance(abs(pca1.components_),self.data_train_baseline_encoded.shape[1])
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

        pca2 = PCA()
        print(self.discrete_data_embedding_sog.shape)
        pca2.fit_transform(self.discrete_data_embedding_sog)
        #print("SOG PCA feature importance")
        #print(abs(pca2.components_))
        self.RankingFeatureImportance(abs(pca2.components_),self.discrete_data_embedding_sog.shape[1])
        print("SOG PCA explained_variance_ratio_")
        print(pca2.explained_variance_ratio_)
        plt.bar(
            range(1,len(pca2.explained_variance_)+1),
            pca2.explained_variance_
            )
        
        
        plt.xlabel('PCA Feature')
        plt.ylabel('Explained variance')
        plt.title('Feature Explained Variance')
        plt.show()

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
               

    def extractFeatuesFromSOG(self, extracted_original_data, features):
        extracted_data = []
        #for x in features:
