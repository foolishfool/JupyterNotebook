"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
@author: Riley Smith
Created: 1-27-21
"""
from sklearn import metrics
import numpy as np
import random
import newSom

class OptimizeW():
    """
    The 2-D, rectangular grid self-organizing map class using Numpy.
    """
    def __init__(self, som, X, classNum = 2, k_folder_num=10, subset_percentage =0.1 ):
        """
        Parameters
        ----------
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
        """
        self.som = som
        self.X = X
        self.classNum = classNum
        self.k_folder_num = k_folder_num
        self.subset_percentage = subset_percentage
        
        self.som_weights1s =  np.zeros(k_folder_num, dtype=object)
        self.som_weights2s =  np.zeros(k_folder_num, dtype=object)

        #self.train_score_W1 =  np.zeros(k_folder_num)
        #self.train_subset_score_W2 =  np.zeros(k_folder_num)
        
        self.validate_score_W1 =  np.zeros(k_folder_num)
        self.validate_score_W2 =  np.zeros(k_folder_num)
        
        # the predcit labels and sublabels
        self.nLabels_predict =  np.zeros(k_folder_num, dtype=object)
        self.nsubLabels_predict =  np.zeros(k_folder_num, dtype=object)

        #self.train_score_W1_predicted_labels =  np.zeros(k_folder_num, dtype=object)
        #self.train_subset_score_W2_predicted_labels =  np.zeros(k_folder_num, dtype=object)

        self.validate_score_W1_predicted_labels =  np.zeros(k_folder_num, dtype=object)
        self.validate_score_W2_predicted_labels =  np.zeros(k_folder_num, dtype=object)

        self.data_trains =  np.zeros(k_folder_num, dtype=object)
        self.data_validates = np.zeros(k_folder_num, dtype=object)
        
        self.label_trains = np.zeros(k_folder_num, dtype=object)
        self.label_validates = np.zeros(k_folder_num, dtype=object)

        # the trainsubsets and train_subset labels
        self.train_subdatas =  np.zeros(k_folder_num, dtype=object)
        self.train_sublabels=  np.zeros(k_folder_num, dtype=object)
        # array that store error data indices for each iteration
        self.error_lists =   np.zeros(k_folder_num, dtype=object)

        self.validate_score_W1_average = 0
        self.validate_score_W2_average = 0

        # percentage of each fraction for 1 k_folder
        self.k_fraction = 1/k_folder_num

    def _initialdataset(self, indice = 0):

        # Initialize train and test data set
        data_validate = self.X.iloc[int(self.X.shape[0] * indice*self.k_fraction):int(self.X.shape[0] *self.k_fraction*(indice+1)), :]
        #print("data_validate {}".format(data_validate))
        data_train = self.X.drop(data_validate.index) # reduce data_validate from dataset
       # data_test = data_test.sample(self.test_num) # get random test_num samples from data_test
        # transfer to numpy array
        data_train = data_train.to_numpy(dtype=np.float64)
        data_validate = data_validate.to_numpy(dtype=np.float64)
      #  data_test = data_test.to_numpy(dtype=np.float64)
        # Initialize  train label and test label
        self.label_trains[indice] = data_train[:,data_train.shape[1]-1]
        self.label_validates[indice] = data_validate[:,data_validate.shape[1]-1]


        # delete last column
        self.data_trains[indice]= data_train[:,:-1]
        self.data_validates[indice] =data_validate[:,:-1]

    def purity_score(self,scorelist, y_true, y_pred,iter_index = 0):
        # compute contingency matrix (also called confusion matrix)
        #print(y_true.shape)
        #print(y_pred.shape)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        #print(iter_index)
        #print(contingency_matrix)
        # return purity
    # print (np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
        scorelist[iter_index] = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def groupClusterList(self,m,Y):
            """
            transfer all label in a list
            m: cluster Number
            """
            clusters = []
            for i in range(0,m):
                newlist = []
                for idx, y in enumerate(Y): 
                    if(y == i):
                        newlist.append(idx)
                clusters.append(newlist) 
            # print(clusters)
            return clusters

    def getErrorClusters(self,list_true,list_pred):
            """
            get the wrong data indices in the training data when predicted with weight1
            """
            errorlist = []
            for i in range(0,len(list_true)):
                newlist = [item for item in list_pred[i] if item not in list_true[i]]
                errorlist.append(newlist)
           # print("error data:{}".format(errorlist))
            return errorlist

    def NormalizeLables(self,Y,category = 0, iter_index = 0):
        """
        X, Y is the label_true and label_predict
        transfrer predcited label to match the range of true label
        a = np.amax(X)+1,b = np.amax(Y)+1 the number of classes in X and Y
        b= m*n,  to make it easier , to make b can be divided by a
        category = 0 noramlize label
        category = 1 normalize sub label
        iter_index the interation number in max_iter
        """
        div = int((self.som.m*self.som.n)/(self.classNum))

        if(category == 0):
            nLabel = np.arange(Y.size)
        
        if(category == 1):
            nsubLabel = np.arange(Y.size)
        
        index = 0;    
        for idx, y in enumerate(Y): 
        # print("idx {}".format(idx))
            for i in range(1,div+1):
                if(y < i*div):
                    index+=1
                    #print(i-1)
                    if(category == 0):
                        nLabel[idx] = i-1
                    if(category == 1):
                        nsubLabel[idx] = i-1
                    break               
        if(category == 0):
            #print("{} normalized predicted nLabel:\n {} ".format(iter_index,nLabel))
            self.nLabels_predict[iter_index] = nLabel
        if(category == 1):
            #print("{} normalized predicted nsubLabel: \n{} ".format(iter_index,nsubLabel))
            self.nsubLabels_predict[iter_index] = nsubLabel




    def reduce_error_data(self,noisy_list, percent):
            newlist = []
            for x in noisy_list:
                for e in x:
                    newlist.append(e)
            # print(newlist)
            # print(len(newlist))
            # print(int(percent * len(newlist)))
            newlist = random.sample(newlist, int(percent * len(newlist)))
            # print(newlist)
            return newlist

    def get_subset(self,reduced_indices,X,category = 0, indice = 0):
            
            # X is the data sets that needs to delete error data
            if(category == 0): 
                data_train_subset= np.delete(X[indice], reduced_indices, axis=0)     
                self.train_subdatas[indice] = data_train_subset
                #self.train_subdatas[indice] = data_train_subset
               
            if(category == 1): 
                data_train_sublabel=np.delete(X[indice],reduced_indices, axis=0)
                self.train_sublabels[indice] = data_train_sublabel

            return 



    def runOptimize(self):
        for i in range(0, self.k_folder_num):
            
            # get w1 in train data
            self._initialdataset(i)
            #train som to get W1
            self.som.fit(self.data_trains[i])
            self.som_weights1s[i] = self.som.weights1
            self.validate_score_W1_predicted_labels[i] = self.som.predict(self.data_validates[i],self.som.weights1)
            self.NormalizeLables(self.validate_score_W1_predicted_labels[i],0,i)
              # get cluster accuracy in train_data with W1
            self.purity_score(self.validate_score_W1,self.label_validates[i],self.nLabels_predict[i],i)
            #print("self.som.validate_score_W1_predicted_labels {}{}".format(i,self.validate_score_W1_predicted_labels[i]))
            #print("self.som.weights1 {} {}".format(i,self.som.weights1))
            #print("self.validate_score_W1 {} {}".format(i,self.validate_score_W1[i] ))
            # get W2 in train data
            self.error_lists[i] = self.getErrorClusters(self.groupClusterList(self.classNum,self.label_trains[i]),self.groupClusterList(self.classNum,self.nLabels_predict[i]))
            #get reduced error data indices
            reduced_indices = self.reduce_error_data(self.error_lists[i],self.subset_percentage)
            self.get_subset(reduced_indices,self.data_trains,0,i) # get train_sub_data
            self.get_subset(reduced_indices,self.label_trains,1,i)   # get train_sub_label
            # train som in subset to get W2
            self.som.fit(self.train_subdatas[i],1)
            self.som_weights2s[i] = self.som.weights2
            self.validate_score_W2_predicted_labels[i] = self.som.predict(self.data_validates[i],self.som.weights2)
            self.NormalizeLables(self.validate_score_W2_predicted_labels[i],1,i)
            # get cluster accuracy in train_sub_data with W2
            self.purity_score(self.validate_score_W2,self.label_validates[i],self.nsubLabels_predict[i],i)
            #print("self.som.validate_score_W2_predicted_labels {}{}".format(i,self.validate_score_W2_predicted_labels[i]))
           # print("self.som.weights2 {} {}".format(i,self.som.weights2))
            #print("self.validate_score_W2 {} {}".format(i,self.validate_score_W2[i] ))
           #self.test_score_W1_predicted_labels[i] = self.som.predict(self.data_tests[i],self.som.weights1)
           #xself.NormalizeLables(self.label_tests[i], self.test_score_W1_predicted_labels[i],0,i)
           ## get cluster accuracy in train_sub_data with W2
           #self.purity_score(self.test_score_W1,self.label_tests[i],self.nLabels_predict[i],i)
           #
           #print("Test W1 nLabels_predict ! {}".format(self.nLabels_predict[i]))
           #
           #self.test_score_W2_predicted_labels[i] = self.som.predict(self.data_tests[i],self.som.weights2)
           #self.NormalizeLables(self.label_tests[i], self.test_score_W2_predicted_labels[i],0,i)
           ## get cluster accuracy in train_sub_data with W2
           #self.purity_score(self.test_score_W2,self.label_tests[i],self.nLabels_predict[i],i)
           #
           #print("Test W2 nLabels_predict ! {}".format(self.nLabels_predict[i]))
        
        self.validate_score_W1_average = np.average(self.validate_score_W1)
        self.validate_score_W2_average = np.average(self.validate_score_W2)

        print("validate_score_W1_average : {}".format( self.validate_score_W1_average))
        print("validate_score_W2_average : {}".format( self.validate_score_W2_average))
       # self.test_score_W1_average = np.average(self.som_weights1s)
        self.weights2_average = np.average(self.som_weights2s)
        #print("weights2_average : {}".format( self.som_weights2s))

       # print("test_score_W1_average : {}".format( self.test_score_W1_average))
       # print("test_score_W2_average : {}".format( self.test_score_W2_average))

        return
            # if train_subset_score_W2.average > train_score_W1.average
            # get W'
            # train_score_W'>train_score_W1
            # test_score_W'> test_score_W1
            #get W1 and W2 score in test data
