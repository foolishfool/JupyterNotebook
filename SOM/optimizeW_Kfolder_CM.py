"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
@author: Riley Smith
Created: 1-27-21
"""
from importlib import resources
from pickle import TRUE
from sklearn import metrics
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

class OptimizeW():
    """
    The 2-D, rectangular grid self-organizing map class using Numpy.
    """
    def __init__(self, som, originalfeatureNum, X, Y,classNum = 2,k_folder_num_min =3, k_folder_num_max = 3,K_folder_list =[10], subset_percentage =0.1, keepFeatureColumns = [], cross_validation = True ):
        """
        Parameters
        ----------
        originalfeatureNum: the inital total feature num
        X : original csv dataset 
        train_num : int, default=5000
            The train data sample number
        test_num : int, default= 5000
            The test data sample number
        max_iter : int, optional
            Optional parameter to stop training if you reach this many
            interation.
        nLabels : label_true in train data 
        nsubLabels : label_true in sub train data 
        keepFeatureColumns: is the columns that needs to keep (choosen as the feature )  if feature n is choosen the input should be n-1, as start from 0: in   def _initializedataset(self, indice = 0, k_num = 2, feature_num =2): self.data_trains[indice]
        """
        self.som = som
        self.originalfeatureNum = originalfeatureNum
        #print("X:{}".format(X))
        self.X = X.sample(n =X.shape[0]) # randomly sample the dataframe to make it more average
        #print("self.X:{}".format(self.X))
        self.classNum = classNum
        self.K_folder_list = K_folder_list
        self.k_folder_num = k_folder_num_max
        self.subset_percentage = subset_percentage     
        self.data_tests =  Y
        self.keepFeatureColumns = keepFeatureColumns
        self.min_k_num = k_folder_num_min
        self.classtestdata =  X.iloc[[0,68,8202,9110,7718,7692],1:].to_numpy(dtype=np.float64)
        self.cross_validate = cross_validation
         #[1,2,3,4,5]it means that predicted class 0 is 1 in true lables, 1 is 2 in true
        self.predicted_classNum= int(som.m*som.n)

        #for w1 matrix predictlabel 1 is 0 in true_label
        self.PLabel_value_convert_to_Tlabel_value_W1 = np.zeros(self.predicted_classNum, dtype=object)
        self.PLabel_value_convert_to_Tlabel_value_W3 = np.zeros(self.predicted_classNum, dtype=object)
        self.PLabel_value_convert_to_Tlabel_value_WCombined = np.zeros(self.predicted_classNum*2, dtype=object)
        #print(self.classtestdata)

    # reset lists size based on kum
    def _initialdatasetsize(self,k_num =2):
        self.som_weights1s =  np.zeros(k_num, dtype=object)
        self.som_weights2s =  np.zeros(k_num, dtype=object)

        self.som_weights3s =  np.zeros(k_num, dtype=object)
        self.som_weights13_difference =  np.zeros(k_num, dtype=object)
        #self.train_score_W1 =  np.zeros(k_folder_num)
        #self.train_subset_score_W2 =  np.zeros(k_folder_num)
        
        self.validate_score_W1 =  np.zeros(k_num)
        self.validate_score_W2 =  np.zeros(k_num)
        self.validate_score_W_Combined =  np.zeros(k_num)

        self.all_train_score_W1 =  np.zeros(k_num)
        self.all_train_score_W_Combined =  np.zeros(k_num)
        self.right_data_score_W1  =  np.zeros(k_num)
        # the score of error dataset 
        self.error_data_score_W3 =  np.zeros(k_num)
        self.error_data_score_W1 =  np.zeros(k_num)

        self.test_score_W1 =  np.zeros(k_num)
        self.test_score_W2 =  np.zeros(k_num)
        self.test_score_W_combined =  np.zeros(k_num)


        self.validate_score_W1_predicted_labels =  np.zeros(k_num, dtype=object)
        self.validate_score_W2_predicted_labels =  np.zeros(k_num, dtype=object)

        self.rightdata_score_W1_predicted_labels =  np.zeros(k_num, dtype=object)

        self.all_train_score_W1_predicted_labels =  np.zeros(k_num, dtype=object)
        self.all_train_score_W_combined_predicted_labels =  np.zeros(k_num, dtype=object)

        self.train_error_score_W1_predicted_labels =  np.zeros(k_num, dtype=object)
        self.train_error_score_W3_predicted_labels =  np.zeros(k_num, dtype=object)

        self.test_score_W1_predicted_labels =  np.zeros(k_num, dtype=object)
        self.test_score_W2_predicted_labels =  np.zeros(k_num, dtype=object)
        self.test_score_W_combined_predicted_labels =  np.zeros(k_num, dtype=object)
        
        self.data_trains =  np.zeros(k_num, dtype=object)
        self.data_trains_error_data =  np.zeros(k_num, dtype=object)
        self.data_trains_right_data =  np.zeros(k_num, dtype=object)
        self.data_validates = np.zeros(k_num, dtype=object)
        
        self.label_trains = np.zeros(k_num, dtype=object)
        self.label_trains_error_data = np.zeros(k_num, dtype=object)
        self.label_trains_right_data = np.zeros(k_num, dtype=object)
        self.label_validates = np.zeros(k_num, dtype=object)
        self.label_tests = np.zeros(k_num, dtype=object)

        # the trainsubsets and train_subset labels
        self.train_subdatas =  np.zeros(k_num, dtype=object)
        self.train_sublabels=  np.zeros(k_num, dtype=object)

        self.train_sublabels_right=  np.zeros(k_num, dtype=object)
        self.train_sublabels_error=  np.zeros(k_num, dtype=object)
        # array that store error data indices for each iteration
        self.error_lists =   np.zeros(k_num, dtype=object)

        self.validate_score_W1_average = 0
        self.validate_score_W2_average = 0
        self.test_score_W1_average = 0
        self.test_score_W2_average = 0

    def _initializedataset(self, indice = 0, k_num = 2, feature_num =2):

        """
        Parameters
        ----------
        indice : the iteration index in a K folder loop, the maximum value is k_num
        k_num : int, k-folder number
        feature_num:  the feature selected, 
        """
        #print("k_num {}".format(k_num))
        # Initialize train and test data set
        #*** self.X.shape[0] = m  [:] range from 0 [0:m-1] so -1 at last

        data_validate = self.X.iloc[int(self.X.shape[0] * indice*(1/k_num)):int((self.X.shape[0]) *(1/k_num)*(indice+1))-1, :]
        #print("data_validate {}".format(data_validate.index))
        #print("data_validate {}".format(data_validate))

        #print("data_validate df0:{}".format(data_validate.iloc[0,:]))
        #print("data_validate df2:{}".format(data_validate.iloc[2,:]))
        if(self.cross_validate):
            data_train = self.X.drop(data_validate.index) # reduce data_validate from dataset
        else: 
            data_train = self.X
        #print("data_train {}".format(data_train.index))
       # data_test = data_test.sample(self.test_num) # get random test_num samples from data_test
        # transfer to numpy array
        data_train = data_train.to_numpy(dtype=np.float64)
        data_validate = data_validate.to_numpy(dtype=np.float64)
        data_test = self.data_tests.to_numpy(dtype=np.float64)
      #  data_test = data_test.to_numpy(dtype=np.float64)
        # Initialize  train label and test label
        #print("data_validate 0:{}".format(data_validate[0,:]))
        #print("data_validate 2:{}".format(data_validate[2,:]))
        self.label_trains[indice] = data_train[:,0]
        self.label_validates[indice] = data_validate[:,0]
        self.label_tests = data_test[:,0]
        #print(self.label_trains[indice].shape[0])

        # delete first column
        self.data_trains[indice]= data_train[:,1:feature_num+1]
        self.data_validates[indice] = data_validate[:,1:feature_num+1] # column number should be feature_num +1 as the first colum is class label
        self.data_test =data_test[:,1:feature_num+1]


        if (self.keepFeatureColumns!=[]):
            allfeaturecolumns = np.arange(self.originalfeatureNum)
            deletefeatureColumns = [item for item in allfeaturecolumns if item not in self.keepFeatureColumns]
            #print("deleteFeatures columns : {}".format(deletefeatureColumns))
            #print(self.data_trains[indice].shape)
            #print(self.data_validates[indice].shape)
           # print(self.data_test[indice].shape)

            self.data_trains[indice]= np.delete(self.data_trains[indice], deletefeatureColumns,1)
            self.data_validates[indice]= np.delete(self.data_validates[indice], deletefeatureColumns,1)
            self.data_test= np.delete(self.data_test, deletefeatureColumns,1)
            #print(self.data_trains[indice])
            #print(self.data_validates[indice].shape)
            #print(self.data_test[indice].shape)

        #print("self.data_trains[indice]{}".format(self.data_trains[indice].shape))



    def purity_score(self,scorelist, y_true, y_pred,iter_index = 0, isprinted = False):
        if isprinted:
            error_data =  self.getErrorDataIndicesFromPredictedLabeles(y_true,y_pred)
        #print("y_true  {} y_pred size  {}".format(y_true, y_pred))
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        #print(iter_index)
        #print(contingency_matrix)
        # return purity      
        scorelist[iter_index] = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
        #print ("scorelist {} scorelist[iter_index] {} iter_index : {}".format(scorelist, scorelist[iter_index],iter_index ))
    def groupClusterList(self,class_num_predicted,category,predicted_label,index):
            """
            transfer all label in a list of list
            class_num: total cluster Number in training data

            category : the indices data set need to be loaded
            """
            #print("preicited Labels lens: {}".format(Y.size))
            clusters = []
            for i in range(0,class_num_predicted):
                newlist = []
                #************ this idx is not the indice in train_date
                #print("label_trains size {}".format(len(self.label_trains[index])))
                for idx, y in enumerate(predicted_label):     
                    # the indices of  predicted_label is the same with realted category label set label_trains,label_trains_right_data or label_trains_error_data
                    if(y == i):
                        if(category == 0):
                            newlist.append(self.label_trains[index][idx]) #self.label_trains[idx] is the true label value 
                        if(category == 1):
                            newlist.append(self.label_trains_right_data[index][idx]) 
                        if(category == 2):
                            newlist.append(self.label_trains_error_data[index][idx])
                        if(category == 3):
                            newlist.append(self.label_tests[index][idx])        
                clusters.append(newlist) 
            #print("predicted_cluster {} {}".format(category, clusters))
            # [[indices of cluster 0],[indices of cluster 1],[indices of cluster 2 ]...]
            return clusters

    def getAlltraiingdataindices(self,i):
            """
            transfer all label in a list
            m: cluster Number
            """
            clusters = []
            for idx, y in enumerate(self.data_trains[i]):
                 clusters.append(idx)
            # print(clusters)
            # [[indices of cluster 1],[indices of cluster 2],[indices of cluster 3 ]...]
            return clusters


    
    def getErrorDataIndicesFromPredictedLabeles(self,true_label,predicted_label_normalized):
        """
        get the wrong data indices in the training data from predicted labels with weight
        
        """
        errordata_indices =[]
        for i in range(0,true_label.size):
            if(true_label[i]!= predicted_label_normalized[i] ):
                errordata_indices.append(i)
        return errordata_indices        
      
          
    def PredictedLabelsConvertToTrueLabels(self,predicted_labels_true_value_clusters,Wtype  = 0):
        """
         update self.predicted_labels_convert_to_true_labels
         predicted_labels_true_value_clusters  = [[1,2,1,1],[3,3,3]]  the value in is the true value in true_label
        """
        #print(true_class_labels)
        predicted_labels_convert_to_true_labels = []
        #[1,2,3,4,5]it means that predicted class 0 is 1 in true lables, 1 is 2 in true
        for item in predicted_labels_true_value_clusters:
            if item != []:
                # the first item is for cluster0       
                #transfer to true class value based on indices in predict lables          
                predicted_labels_convert_to_true_labels.append(self.GetMaxRepeatedElementsInaList(item))
            else:
                predicted_labels_convert_to_true_labels.append(-1)
        
        if Wtype == 0 :
            self.PLabel_value_convert_to_Tlabel_value_W1 = predicted_labels_convert_to_true_labels
            #print("self.PLabel_value_convert_to_Tlabel_value_W1 !!!! {}".format(self.PLabel_value_convert_to_Tlabel_value_W1))
        if Wtype == 1 :
            self.PLabel_value_convert_to_Tlabel_value_W3 = predicted_labels_convert_to_true_labels
            #print("self.PLabel_value_convert_to_Tlabel_value_W3 !!!! {}".format(self.PLabel_value_convert_to_Tlabel_value_W3))



    def NomalizeCombinedWPredictedLabels(self, predicted_labels):
        """
        for combined predicted_labels. as the neuron units number doubles, so the output range will double too.
        normalized the labels to be in the range of som.m*som.n
        """
        for i in range(0,predicted_labels.size):
            if (predicted_labels[i]>= self.predicted_classNum):
                currentNum = predicted_labels[i]
                predicted_labels[i] = currentNum - self.predicted_classNum
        #print("NomalizeCombinedWPredictedLabels {}".format())
        return predicted_labels


    def GetMaxRepeatedElementsInaList(self, list):
        #print(list)
        counts = np.bincount(list)
        #print("list {}".format( list))
        #print("max repeated {}".format( np.argmax(counts)))
        #b = Counter(list)
        #print("most common 1 {}".format(b.most_common(1)))
        return np.argmax(counts)



    def TransferPredictedLabelsToTrueLabelsValue(self,category, predicted_labels,iter_index, convert_predict_value_to_true_value = False, Wtype = 0):       
            #if  W 3 use PLabel_value_convert_to_Tlabel_value_W3 if W1 use PLabel_value_convert_to_Tlabel_value_W1, if combined?
        if(convert_predict_value_to_true_value == True):
            if(Wtype == 0):
                predicted_clusters= self.groupClusterList(self.predicted_classNum,category,predicted_labels,iter_index)
                self.PredictedLabelsConvertToTrueLabels( predicted_clusters,Wtype)           
                predicted_labels =  self.ConvertLabelValue(predicted_labels,self.PLabel_value_convert_to_Tlabel_value_W1)
                return predicted_labels
            if(Wtype == 1):
                predicted_clusters= self.groupClusterList(self.predicted_classNum,category,predicted_labels,iter_index)
                self.PredictedLabelsConvertToTrueLabels( predicted_clusters,Wtype)           
                predicted_labels =  self.ConvertLabelValue(predicted_labels,self.PLabel_value_convert_to_Tlabel_value_W3)
                return predicted_labels
        else:
            if(Wtype == 2):
                self.PLabel_value_convert_to_Tlabel_value_WCombined = np.concatenate((self.PLabel_value_convert_to_Tlabel_value_W1 , self.PLabel_value_convert_to_Tlabel_value_W3), axis = 0)
                #print("self.PLabel_value_convert_to_Tlabel_value_WCombined !!!! {}".format(self.PLabel_value_convert_to_Tlabel_value_WCombined))
                predicted_labels = self.ConvertLabelValue(predicted_labels,self.PLabel_value_convert_to_Tlabel_value_WCombined)
                return predicted_labels
            if(Wtype == 0):
                predicted_labels = self.ConvertLabelValue(predicted_labels,self.PLabel_value_convert_to_Tlabel_value_W1)
                return predicted_labels
            if(Wtype == 1):
                predicted_labels = self.ConvertLabelValue(predicted_labels,self.PLabel_value_convert_to_Tlabel_value_W3)
                return predicted_labels
    
    def NormalizeLables(self,predicted_labels,category = 0,iter_index = 0,convert_predict_value_to_true_value = False,Wtype = 0):       
        normalized_label = self.TransferPredictedLabelsToTrueLabelsValue(category,predicted_labels,iter_index,convert_predict_value_to_true_value,Wtype)
        #print("normalized_label {}" .format(normalized_label))
        return normalized_label

    def ConvertLabelValue(self, predicted_labels, PLabel_value_convert_to_Tlabel_value):
        #print("predicted_labels size {} PLabel_value_convert_to_Tlabel_value size {}".format(len(predicted_labels), len(PLabel_value_convert_to_Tlabel_value )))
        for i in range(0,predicted_labels.size):
                    predicte_value =  predicted_labels[i]
                    predicted_labels[i] = PLabel_value_convert_to_Tlabel_value[predicte_value]             
        return predicted_labels

    def reduce_error_data(self,noisy_list, percent):
            newlist = []
            #print("noisy_list {}".format(noisy_list))
            for x in noisy_list:
                newlist.append(x)
            # print(newlist)
            #print( "len(newlist) {}".format(len(newlist)))
            #print("percent * len(newlist) {}".format(int(percent * len(newlist))))
            newlist = random.sample(newlist, int(percent * len(newlist)))
            
            return newlist

    def get_subset(self,reduced_indices,X,category = 0, indice = 0):
            
            # X is the data sets that needs to delete error data
            if(category == 0): 
                #print("X[indice] shape : {}".format(X[indice].shape))
                #print("reduces_indices : {}".format(reduced_indices))
                self.train_subdatas[indice] = np.delete(X[indice], reduced_indices, axis=0)     
                #self.train_subdatas[indice] = data_train_subset
               
            if(category == 1): 
                self.train_sublabels[indice] = np.delete(X[indice],reduced_indices, axis=0)
            
            if(category == 4):             
               self.data_trains_right_data[indice] = np.delete(X[indice],reduced_indices, axis=0)
            
            if(category == 5):             
               self.label_trains_right_data[indice] = np.delete(X[indice],reduced_indices, axis=0)
               #print("self.label_trains_error_data[indice] {}".format(self.label_trains_error_data[indice]))
            return 




    def runOptimize(self):
        
        validate_score_W1_averages = []
        validate_score_W2_averages = []
        validate_score_W_combine_averages = []
        all_train_score_W1_averages = []     
        all_train_score_W_combine_averages = []
        test_score_W1_averages = []
        test_score_W2_averages = []
        test_score_W_combine_averages = []
        k_num_range = np.arange(self.min_k_num,self.k_folder_num+1)
        #for j in self.K_folder_list:
            #print("k_folder : {}".format(j))
        if(self.cross_validate == False):
            self.min_k_num = 1
            self.k_folder_num = 1

        for j in range(self.min_k_num, self.k_folder_num + 1):
            self._initialdatasetsize(j)
            for i in range(0, j): 
                hasNoErroData = False              
                # get train and test dataset 
                self._initializedataset(i,j,self.originalfeatureNum)

                #allincices = self.getAlltraiingdataindices(i)
                #print("all indices {}".format(allincices))
                #train som to get W1
                self.som.fit(self.data_trains[i])
                #print("self.data_trains[i] size {}    {}".format(i,self.data_trains[i].shape[0]))
                self.som_weights1s[i] = self.som.weights1
                self.validate_score_W1_predicted_labels[i] = self.som.predict(self.data_validates[i],self.som.weights1)
             
                #self.NormalizeLables(self.validate_score_W1_predicted_labels[i],0,i)
                #get cluster accuracy in train_data with W1
                #self.purity_score(self.validate_score_W1,self.label_validates[i],self.nLabels_predict[i],i)

                self.all_train_score_W1_predicted_labels[i] = self.som.predict(self.data_trains[i],self.som.weights1)
                #print("self.data_trains[i]{}" .format(self.data_trains[i]))
                #print("self.all_train_score_W1_predicted_labels[i] {}".format(self.all_train_score_W1_predicted_labels[i] ))
                normalized_predicted_label_all_train = self.NormalizeLables(self.all_train_score_W1_predicted_labels[i],category = 0, iter_index = i,convert_predict_value_to_true_value =True)
                self.purity_score(self.all_train_score_W1,self.label_trains[i],normalized_predicted_label_all_train,i)

                # get W2 in train data
                #self.error_lists[i] = self.getErrorClusters(self.groupClusterList(self.classNum,self.label_trains[i]),self.groupClusterList(self.classNum,self.nLabels_predict[i]))
                self.error_lists[i] = self.getErrorDataIndicesFromPredictedLabeles(self.label_trains[i],normalized_predicted_label_all_train)
                #print(" self.error_lists[i] size {}".format( len(self.error_lists[i])))
                if(self.error_lists[i] ==[]):
                    hasNoErroData = True
                    print(" Has NO Error !!!")
                #get reduced error data indices
                if(self.cross_validate == False):
                    self.subset_percentage = 1

                # reduced_indices is randomized
                reduced_indices = self.reduce_error_data(self.error_lists[i],self.subset_percentage)            
                #*********** make it sotred, so when nptake error data it will be also from small indices to big indices, then can compared with label_error_data
                reduced_indices_sorted = np.sort(reduced_indices)
                #print("reduced_indices_sorted {}" .format(reduced_indices_sorted))
                self.get_subset(reduced_indices_sorted,self.data_trains,0,i) # get train_subdatas 
                self.get_subset(reduced_indices_sorted,self.label_trains,1,i)   # get train_sub_label

                self.get_subset(reduced_indices_sorted,self.data_trains,4,i)  #get train_right_data
                self.get_subset(reduced_indices_sorted,self.label_trains,5,i)  #get label_train_right_label
                
                #_________________train right data to see result
                self.rightdata_score_W1_predicted_labels[i] = self.som.predict(self.data_trains_right_data[i],self.som.weights1)
               # print("self.data_trains_right_data[i] {}" .format(self.data_trains_right_data[i]))
                normalized_predicted_label_right_data =  self.NormalizeLables(self.rightdata_score_W1_predicted_labels[i],category = 1, iter_index = i)
                self.purity_score(self.right_data_score_W1,self.label_trains_right_data[i],normalized_predicted_label_right_data,i, True)
                print("right_data_score_W1 {}".format(self.right_data_score_W1[i]))
               
                self.data_trains_error_data[i] = np.take(self.data_trains[i], reduced_indices_sorted,axis=0)
                self.label_trains_error_data[i] =np.take(self.label_trains[i], reduced_indices_sorted,axis=0)
                #print("self.data_trains_error_data[i]   {}".format((self.data_trains_error_data[i])))
               # print("self.label_trains_error_data[i]  {}".format((self.label_trains_error_data[i])))
                #______________________get weights3 : the weight in error dataset
    
                if(hasNoErroData == False):
                    #print("self.data_trains_error_data[i] shape {}".format(self.data_trains_error_data[i].shape ))
                    self.som.fit(self.data_trains_error_data[i],2)

                    self.som_weights3s[i] = self.som.weights3
                    self.train_error_score_W3_predicted_labels[i] = self.som.predict(self.data_trains_error_data[i],self.som.weights3)
                    #get nLabels_errordata_predict
                    normalized_predicted_label =  self.NormalizeLables(self.train_error_score_W3_predicted_labels[i],category = 2, iter_index = i,convert_predict_value_to_true_value =True,Wtype=1)
                    #print("normalized_predicted_label W1  {}".format(normalized_predicted_label))
                    self.purity_score(self.error_data_score_W3,self.label_trains_error_data[i],normalized_predicted_label,i)
                    # TODO: why train_score_W3 is so slow
                    print("error_data_score_W3 {}".format(self.error_data_score_W3[i]))
                    #self.som_weights13_difference[i] = self.som.weights3 - self.som.weights1 #no use
                    
                    
                    self.train_error_score_W1_predicted_labels[i] = self.som.predict(self.data_trains_error_data[i],self.som.weights1)
                    normalized_predicted_label = self.NormalizeLables(self.train_error_score_W1_predicted_labels[i],category = 2, iter_index =i)
                    #print("normalized_predicted_label W1  {}".format(normalized_predicted_label))
                    self.purity_score(self.error_data_score_W1,self.label_trains_error_data[i],normalized_predicted_label,i)
                    print("error_data_score_W1 {}".format(self.error_data_score_W1[i]))


                    #train error with W1

                    #______________________combinedweights

                    combinedweights =  np.concatenate((self.som.weights1, self.som.weights3), axis=0)
                    #self.nLabels_W_Combined_predicted[i] = self.som.predict(self.data_validates[i],combinedweights,combined= True)
                    #print("Difference of Weights: {} {}".format(newweights,newweights.shape) )
                #else:
                #    self.nLabels_W_Combined_predicted[i] = self.som.predict(self.data_validates[i],self.som.weights1)

                #self.NormalizeLables(self.nLabels_W_Combined_predicted[i],3,i)
                # get cluster accuracy in train_sub_data with W2
                #self.purity_score(self.validate_score_W_Combined,self.label_validates[i],self.nLabels_W_Combined_predicted[i],i)

                self.all_train_score_W_combined_predicted_labels[i] = self.som.predict(self.data_trains[i],combinedweights,combined = True)
                normalized_predicted_label = self.NormalizeLables(self.all_train_score_W_combined_predicted_labels[i],category = 0, iter_index = i, Wtype= 2 )
                self.purity_score(self.all_train_score_W_Combined,self.label_trains[i],normalized_predicted_label,i)



                # train som in subset to get W2
                #self.som.fit(self.train_subdatas[i],1)
                #self.som_weights2s[i] = self.som.weights2
                #self.validate_score_W2_predicted_labels[i] = self.som.predict(self.data_validates[i],self.som.weights2)
                #self.NormalizeLables(self.validate_score_W2_predicted_labels[i],1,i)
                #get cluster accuracy in train_sub_data with W2
                #self.purity_score(self.validate_score_W2,self.label_validates[i],self.nsubLabels_predict[i],i)
                #print("self.som.validate_score_W2_predicted_labels {}{}".format(i,self.validate_score_W2_predicted_labels[i]))
                # print("self.som.weights2 {} {}".format(i,self.som.weights2))
                #print("self.validate_score_W2 {} {}".format(i,self.validate_score_W2[i] ))
           

                #______________________ test data 

                self.test_score_W1_predicted_labels[i] = self.som.predict(self.data_test,self.som.weights1)
                normalized_predicted_label = self.NormalizeLables(self.test_score_W1_predicted_labels[i],category = 3, iter_index =i)
            ## get cluster accuracy in train_sub_data with W2
                self.purity_score(self.test_score_W1,self.label_tests,normalized_predicted_label,i)
                
                #self.test_score_W2_predicted_labels[i] = self.som.predict(self.data_test,self.som.weights2)
               # self.NormalizeLables(self.test_score_W2_predicted_labels[i],1,i)
                #self.purity_score(self.test_score_W2,self.label_tests,self.nsubLabels_predict[i],i)
                #print("self.test_score_W2 {}".format(self.test_score_W2))
                
                if(hasNoErroData == False):
                    self.test_score_W_combined_predicted_labels [i] = self.som.predict(self.data_test,combinedweights,combined= True)
                else:
                    self.test_score_W_combined_predicted_labels [i] = self.som.predict(self.data_test,self.som.weights1)  
                
                normalized_predicted_label = self.NormalizeLables(self.test_score_W_combined_predicted_labels[i],category = 3, iter_index = i,Wtype=2)
                self.purity_score(self.test_score_W_combined,self.label_tests,normalized_predicted_label,i)

            #
            #print("Test W2 nLabels_predict ! {}".format(self.nLabels_predict[i]))
           # print("self.validate_score_W1 size {}".format(self.validate_score_W1))


            
            #validate_score_W3_average = np.average(self.train_score_W3)


            if(self.cross_validate):
                print("k number :{}".format(j))
                validate_score_W1_average = np.average(self.validate_score_W1)
                validate_score_W2_average = np.average(self.validate_score_W2)
                validate_score_W_combine_average = np.average(self.validate_score_W_Combined)
                print("validate_score_W1_average : {}".format( validate_score_W1_average))
                print("validate_score_W2_average : {}".format( validate_score_W2_average))
                print("validate_score_W_combine_average : {}".format( validate_score_W_combine_average))
                validate_score_W1_averages.append( validate_score_W1_average )
                validate_score_W2_averages.append( validate_score_W2_average )
                validate_score_W_combine_averages.append( validate_score_W_combine_average )
            else:
                all_train_score_W1_average = np.average(self.all_train_score_W1)
                all_train_score_W_combine_average = np.average(self.all_train_score_W_Combined)
                print("all_train_score_W1_average : {}".format( all_train_score_W1_average))
                print("all_train_score_W_combine_average : {}".format( all_train_score_W_combine_average))
                all_train_score_W1_averages.append( all_train_score_W1_average )
                all_train_score_W_combine_averages.append( all_train_score_W_combine_average )
            
            #print("train_error_score_W3_average : {}".format( validate_score_W3_average))
            
           
            test_score_W1_average = np.average(self.test_score_W1)
            test_score_W2_average = np.average(self.test_score_W2)
            test_score_W_combine_average = np.average(self.test_score_W_combined)
           
           
            print("test_score_W1_average : {}".format( test_score_W1_average))
            #print("test_score_W2_average : {}".format( test_score_W2_average))
            print("test_score_W_combine_average : {}".format( test_score_W_combine_average))

            test_score_W1_averages.append( test_score_W1_average )
            test_score_W2_averages.append( test_score_W2_average )
            test_score_W_combine_averages.append( test_score_W_combine_average )

        
        betterScoreList_train = []
        if(self.cross_validate):    
            for i in range(0,len(validate_score_W1_averages)):
                if(validate_score_W_combine_averages[i]>=validate_score_W1_averages[i]):
                    betterScoreList_train.append(validate_score_W_combine_averages[i] -validate_score_W1_averages[i])
        else:
            for i in range(0,len(validate_score_W1_averages)):
                if(all_train_score_W_combine_averages[i]>=all_train_score_W1_averages[i]):
                    betterScoreList_train.append(all_train_score_W_combine_averages[i] -all_train_score_W1_averages[i])
        
        betterScoreList_test = []
        for i in range(0,len(test_score_W1_averages)):
            if(test_score_W_combine_averages[i]>=test_score_W1_averages[i]):
                betterScoreList_test.append(test_score_W_combine_averages[i] -test_score_W1_averages[i])

        betterscore_train = len(betterScoreList_train)
        betterscore_test = len(betterScoreList_test)
        
        if(self.cross_validate):
            plt.plot(k_num_range,validate_score_W1_averages,'g', label ='validate_score_W1_averages')
            #plt.plot(k_num_range,validate_score_W2_averages,'r', label ='validate_score_W2_averages')
            plt.plot(k_num_range,validate_score_W_combine_averages,'c', label ='validate_score_W_combine_averages')
        
            plt.plot(k_num_range,test_score_W1_averages,'b', label ='test_score_W1_averages')     
            #plt.plot(k_num_range,test_score_W2_averages,'y', label ='test_score_W2_averages')
            plt.plot(k_num_range,test_score_W_combine_averages,'k', label ='test_score_W_combine_averages')

            plt.xlabel("Feature Number: {} Better Score train: {} Better Score test: {}".format(self.originalfeatureNum,betterscore_train,betterscore_test))          
            plt.legend()
            plt.show()
        
        print("betterScoreList_train: {}".format(betterScoreList_train))
        print("betterScoreList_test: {}".format(betterScoreList_test))
        
        return self.originalfeatureNum, betterscore_train, betterscore_test


        
    
