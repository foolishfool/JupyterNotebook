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
    def __init__(self, som, X, Y,classNum = 2):
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
        #print("X:{}".format(X))
        self.X = X.sample(n =X.shape[0]) # randomly sample the dataframe to make it more average
        #print("self.X:{}".format(self.X))
        self.classNum = classNum 
        self.data_tests =  Y
         #[1,2,3,4,5]it means that predicted class 0 is 1 in true lables, 1 is 2 in true
        self.predicted_classNum= int(som.m*som.n)

        #for w1 matrix predictlabel 1 is 0 in true_label
        self.PLabel_value_convert_to_Tlabel_value_W1 = np.zeros(self.predicted_classNum, dtype=object)
        self.PLabel_value_convert_to_Tlabel_value_W3 = np.zeros(self.predicted_classNum, dtype=object)
        self.PLabel_value_convert_to_Tlabel_value_WCombined = np.zeros(self.predicted_classNum*2, dtype=object)
        #print(self.classtestdata)

    # reset lists size based on kum
    def _initialdatasetsize(self):
       
        
        self.all_train_score_W1 =  []
        self.all_train_score_W_Combined =  []
        self.right_data_score_W1  =  []
        # the score of error dataset 
        self.error_data_score_W3 =  []
        self.error_data_score_W1 =  []

        self.test_score_W1 =  []
        self.test_score_W_combined =[]


        self.rightdata_score_W1_predicted_label =  []

        self.all_train_score_W1_predicted_label = []
        self.all_train_score_W_combined_predicted_label = []

        self.train_error_score_W1_predicted_label =  []
        self.train_error_score_W3_predicted_label =  []

        self.test_score_W1_predicted_label =  []
        self.test_score_W2_predicted_label = []
        self.test_score_W_combined_predicted_label =  []
        
        self.data_train_error_data =  []
        self.data_train_right_data = []


    def _initializedataset(self):

        """
        Parameters
        ----------
        feature_num:  the feature selected, 
        """   
        data_train = self.X 
        # transfer to numpy array
        data_train = data_train.to_numpy(dtype=np.float64)
        data_test = self.data_tests.to_numpy(dtype=np.float64)
        self.label_train = data_train[:,0]
        self.label_test = data_test[:,0]

        # delete first column
        self.data_train= data_train[:,1:]
        self.data_test =data_test[:,1:]

        self._initialdatasetsize()

    def purity_score(self,scoretype, y_true, y_pred, isprinted = False):
        if isprinted:
            error_data =  self.getErrorDataIndicesFromPredictedLabeles(y_true,y_pred)
        #print("y_true  {} y_pred size  {}".format(y_true, y_pred))
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        #print(contingency_matrix)
        # return purity 
        if(scoretype == "all_train_score_W1" ):
            self.all_train_score_W1.append(np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
        if(scoretype == "right_data_score_W1" ):
            self.right_data_score_W1.append( np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
        if(scoretype == "error_data_score_W3" ):
            self.error_data_score_W3.append( np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
        if(scoretype == "error_data_score_W1" ):
            self.error_data_score_W1 .append( np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
        if(scoretype == "all_train_score_W_Combined" ):
            self.all_train_score_W_Combined .append( np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
        if(scoretype == "test_score_W1" ):
            self.test_score_W1.append(np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
        if(scoretype == "test_score_W_combined" ):
            self.test_score_W_combined.append(np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))

    def groupClusterList(self,class_num_predicted,category,predicted_label):
            """
            transfer all label in a list of list
            class_num: total cluster Number in training data

            category : the indices data set need to be loaded
            """
            #print("preicited Labels lens: {}".format(Y.size))
            clusters = []
            for i in range(0,class_num_predicted):
                newlist = []
                #************ this idx is not the indice in train_data
                #print("label_trains size {}".format(len(self.label_trains[index])))
                for idx, y in enumerate(predicted_label):     
                    # the indices of  predicted_label is the same with related category label set label_trains,label_trains_right_data or label_trains_error_data
                    if(y == i):
                        if(category == 0):
                            newlist.append(self.label_train[idx]) #self.label_trains[idx] is the true label value 
                        if(category == 1):
                            newlist.append(self.label_train_right_data[idx]) 
                        if(category == 2):
                            newlist.append(self.label_train_error_data[idx])
                        if(category == 3):
                            newlist.append(self.label_test[idx])        
                clusters.append(newlist) 
            #print("predicted_cluster {} {}".format(category, clusters))
            # [[indices of cluster 0],[indices of cluster 1],[indices of cluster 2 ]...]
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



    def TransferPredictedLabelsToTrueLabelsValue(self,category, predicted_labels, convert_predict_value_to_true_value = False, Wtype = 0):       
            #if  W 3 use PLabel_value_convert_to_Tlabel_value_W3 if W1 use PLabel_value_convert_to_Tlabel_value_W1, if combined?
        if(convert_predict_value_to_true_value == True):
            if(Wtype == 0):
                predicted_clusters= self.groupClusterList(self.predicted_classNum,category,predicted_labels)
                self.PredictedLabelsConvertToTrueLabels( predicted_clusters,Wtype)           
                predicted_labels =  self.ConvertLabelValue(predicted_labels,self.PLabel_value_convert_to_Tlabel_value_W1)
                return predicted_labels
            if(Wtype == 1):
                predicted_clusters= self.groupClusterList(self.predicted_classNum,category,predicted_labels)
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
    
    def NormalizeLables(self,predicted_labels,category = 0,convert_predict_value_to_true_value = False,Wtype = 0):       
        normalized_label = self.TransferPredictedLabelsToTrueLabelsValue(category,predicted_labels,convert_predict_value_to_true_value,Wtype)
        #print("normalized_label {}" .format(normalized_label))
        return normalized_label

    def ConvertLabelValue(self, predicted_labels, PLabel_value_convert_to_Tlabel_value):
        #print("predicted_labels size {} PLabel_value_convert_to_Tlabel_value size {}".format(len(predicted_labels), len(PLabel_value_convert_to_Tlabel_value )))
        for i in range(0,predicted_labels.size):
                    predicte_value =  predicted_labels[i]
                    predicted_labels[i] = PLabel_value_convert_to_Tlabel_value[predicte_value]             
        return predicted_labels

    def reduce_error_data(self,noisy_list):
            newlist = []
            for x in noisy_list:
                newlist.append(x)
            newlist = random.sample(newlist, int(len(newlist)))
            
            return newlist

    def get_subset(self,reduced_indices,X,category = 0):
            
            if(category == 0):             
               self.data_train_right_data= np.delete(X,reduced_indices, axis=0)
            
            if(category == 1):             
               self.label_train_right_data = np.delete(X,reduced_indices, axis=0)
               #print("self.label_trains_error_data[indice] {}".format(self.label_trains_error_data[indice]))
            return 




    def runOptimize(self,traing_time = 2):
        hasNoErroData = False              
        # get train and test dataset 
        self._initializedataset()
        #train som to get W1
        self.som.fit(self.data_train)
        self.all_train_score_W1_predicted_label = self.som.predict(self.data_train,self.som.weights1)
        normalized_predicted_label_all_train = self.NormalizeLables(self.all_train_score_W1_predicted_label,category = 0,convert_predict_value_to_true_value =True)
        self.purity_score("all_train_score_W1",self.label_train,normalized_predicted_label_all_train)

        self.error_lists = self.getErrorDataIndicesFromPredictedLabeles(self.label_train,normalized_predicted_label_all_train)
        
        #print(" self.error_lists[i] size {}".format( len(self.error_lists[i])))
        if(self.error_lists ==[]):
            hasNoErroData = True
            print(" Has NO Error !!!")
   
        # reduced_indices is randomized
        reduced_indices = self.reduce_error_data(self.error_lists)            
        #*********** make it sotred, so when nptake error data it will be also from small indices to big indices, then can compared with label_error_data
        reduced_indices_sorted = np.sort(reduced_indices)
        #print("reduced_indices_sorted {}" .format(reduced_indices_sorted))
        self.get_subset(reduced_indices_sorted,self.data_train,0)  #get train_right_data
        self.get_subset(reduced_indices_sorted,self.label_train,1)  #get label_train_right_label
        
        #_________________train right data to see result
        self.rightdata_score_W1_predicted_labels= self.som.predict(self.data_train_right_data,self.som.weights1)
       # print("self.data_trains_right_data {}" .format(self.data_trains_right_data))
        normalized_predicted_label_right_data =  self.NormalizeLables(self.rightdata_score_W1_predicted_labels,category = 1)
        self.purity_score("right_data_score_W1",self.label_train_right_data,normalized_predicted_label_right_data,True)
        print("right_data_score_W1 {}".format(self.right_data_score_W1))
        
        self.data_train_error_data = np.take(self.data_train, reduced_indices_sorted,axis=0)
        self.label_train_error_data =np.take(self.label_train, reduced_indices_sorted,axis=0)
        #print("self.label_trains_error_data  {}".format((self.label_trains_error_data)))
        #______________________get weights3 : the weight in error dataset
    
        if(hasNoErroData == False):
            #print("self.data_trains_error_data[i] shape {}".format(self.data_trains_error_data[i].shape ))
            self.som.fit(self.data_train_error_data,2)

            self.train_error_score_W3_predicted_label = self.som.predict(self.data_train_error_data,self.som.weights3)
            #get nLabels_errordata_predict
            normalized_predicted_label =  self.NormalizeLables(self.train_error_score_W3_predicted_label,category = 2,convert_predict_value_to_true_value =True,Wtype=1)
            #print("normalized_predicted_label W1  {}".format(normalized_predicted_label))
            self.purity_score("error_data_score_W3",self.label_train_error_data,normalized_predicted_label)
            # TODO: why train_score_W3 is so slow
            print("error_data_score_W3 {}".format(self.error_data_score_W3))
            #self.som_weights13_difference[i] = self.som.weights3 - self.som.weights1 #no use
            
            
            self.train_error_score_W1_predicted_label = self.som.predict(self.data_train_error_data,self.som.weights1)
            normalized_predicted_label = self.NormalizeLables(self.train_error_score_W1_predicted_label,category = 2)
            #print("normalized_predicted_label W1  {}".format(normalized_predicted_label))
            self.purity_score("error_data_score_W1",self.label_train_error_data,normalized_predicted_label)
            print("error_data_score_W1 {}".format(self.error_data_score_W1))

            #______________________combinedweights

            combinedweights =  np.concatenate((self.som.weights1, self.som.weights3), axis=0)


            self.all_train_score_W_combined_predicted_label = self.som.predict(self.data_train,combinedweights,combined = True)
            normalized_predicted_label = self.NormalizeLables(self.all_train_score_W_combined_predicted_label,category = 0, Wtype= 2 )
            self.purity_score("all_train_score_W_Combined",self.label_train,normalized_predicted_label)

           

            #______________________ test data 

            self.test_score_W1_predicted_label = self.som.predict(self.data_test,self.som.weights1)
            normalized_predicted_label = self.NormalizeLables(self.test_score_W1_predicted_label,category = 3)
            ## get cluster accuracy in train_sub_data with W2
            self.purity_score("test_score_W1",self.label_test,normalized_predicted_label)
                
                
            if(hasNoErroData == False):
                self.test_score_W_combined_predicted_label = self.som.predict(self.data_test,combinedweights,combined= True)
            else:
                self.test_score_W_combined_predicted_label = self.som.predict(self.data_test,self.som.weights1)  
            
            normalized_predicted_label = self.NormalizeLables(self.test_score_W_combined_predicted_label,category = 3,Wtype=2)
            self.purity_score("test_score_W_combined",self.label_test,normalized_predicted_label)

            print("all_train_score_W1 : {}".format( self.all_train_score_W1))
            print("all_train_score_W_combine: {}".format( self.all_train_score_W_Combined))
            
        
           
            print("test_score_W1 : {}".format( self.test_score_W1))
            print("test_score_W : {}".format( self.test_score_W_combined))
        



        
    
