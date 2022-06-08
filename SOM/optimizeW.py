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
    def __init__(self, som, X, classNum = 2,train_num =5000, validate_num =0, max_iter=100, subset_percentage =0.1 ):
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
        self.train_num = train_num
        self.validate_num = validate_num
        self.max_iter = max_iter
        self.subset_percentage = subset_percentage
        
        self.train_score_W1 =  np.zeros(max_iter)
        self.train_subset_score_W2 =  np.zeros(max_iter)
        
        self.validate_score_W1 =  np.zeros(max_iter)
        self.validate_score_W2 =  np.zeros(max_iter)
        
        # the predcit labels and sublabels
        self.nLabels_predict =  np.zeros(max_iter, dtype=object)
        self.nsubLabels_predict =  np.zeros(max_iter, dtype=object)

        self.train_score_W1_predicted_labels =  np.zeros(max_iter, dtype=object)
        self.train_subset_score_W2_predicted_labels =  np.zeros(max_iter, dtype=object)

        self.test_score_W1_predicted_labels =  np.zeros(max_iter, dtype=object)
        self.test_score_W2_predicted_labels =  np.zeros(max_iter, dtype=object)

        self.data_trains =  np.zeros(max_iter, dtype=object)
        self.data_tests = np.zeros(max_iter, dtype=object)
        
        self.label_trains = np.zeros(max_iter, dtype=object)
        self.label_tests = np.zeros(max_iter, dtype=object)

        # the trainsubsets and train_subset labels
        self.train_subdatas =  np.zeros(max_iter, dtype=object)
        self.train_sublabels=  np.zeros(max_iter, dtype=object)
        # array that store error data indices for each iteration
        self.error_lists =   np.zeros(max_iter, dtype=object)

        self.train_score_W1_average = 0
        self.train_subset_score_W2_average = 0
        

    def _initialdataset(self, indice = 0):
        # Initialize train and test data set
        data_train = self.X.sample(self.train_num) # get random train_num samples from X 
       # data_test = self.X.drop(data_train.index) # reduce data_train from dataset
       # data_test = data_test.sample(self.test_num) # get random test_num samples from data_test
        # transfer to numpy array
        data_train = data_train.to_numpy(dtype=np.float64)
      #  data_test = data_test.to_numpy(dtype=np.float64)
        # Initialize  train label and test label
        #np.put(self.label_trains,indice,data_train[:,data_train.shape[1]-1])
        #np.put(self.label_tests,indice,data_test[:,data_test.shape[1]-1])
        self.label_trains[indice] = data_train[:,data_train.shape[1]-1]
       # self.label_tests[indice] = data_test[:,data_test.shape[1]-1]

        # delete last column
        #np.put(self.data_trains,indice,data_train[:,:-1])
        #np.put(self.data_tests,indice,data_test[:,:-1])
        self.data_trains[indice]= data_train[:,:-1]
       # self.data_tests[indice] =data_test[:,:-1]

    def purity_score(self,scorelist, y_true, y_pred,iter_index = 0):
        # compute contingency matrix (also called confusion matrix)
        #print(y_true.shape)
        #print(y_pred.shape)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
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
        #print("div : {}".format(div) )
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
            #print("{} normalized predicted nLabel:\n {} ".format(Y,nLabel))
            self.nLabels_predict[iter_index] = nLabel
        if(category == 1):
            #print("normalized predicted nsubLabel: \n{} ".format(nsubLabel))
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
        for i in range(0, self.max_iter):
            
            # get w1 in train data
            self._initialdataset(i)
            #train som to get W1
            self.som.fit(self.data_trains[i])
            self.train_score_W1_predicted_labels[i] = self.som.predict(self.data_trains[i],self.som.weights1)
            self.NormalizeLables(self.train_score_W1_predicted_labels[i],0,i)
              # get cluster accuracy in train_data with W1
            self.purity_score(self.train_score_W1,self.label_trains[i],self.nLabels_predict[i],i)
            # get W2 in train data
            self.error_lists[i] = self.getErrorClusters(self.groupClusterList(self.classNum,self.label_trains[i]),self.groupClusterList(self.classNum,self.nLabels_predict[i]))
            #get reduced error data indices
            reduced_indices = self.reduce_error_data(self.error_lists[i],self.subset_percentage)
            self.get_subset(reduced_indices,self.data_trains,0,i) # get train_sub_data
            self.get_subset(reduced_indices,self.label_trains,1,i)   # get train_sub_label
            # train som in subset to get W2
            self.som.fit(self.train_subdatas[i],1)
            self.train_subset_score_W2_predicted_labels[i] = self.som.predict(self.train_subdatas[i],self.som.weights2)
            self.NormalizeLables(self.train_subset_score_W2_predicted_labels[i],1,i)
            # get cluster accuracy in train_sub_data with W2
            self.purity_score(self.train_subset_score_W2,self.train_sublabels[i],self.nsubLabels_predict[i],i)

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
        
        self.train_score_W1_average = np.average(self.train_score_W1)
        self.train_subset_score_W2_average = np.average(self.train_subset_score_W2)
        #self.test_score_W1_average = np.average(self.test_score_W1)
        #self.test_score_W2_average = np.average(self.test_score_W2)

        print("train_score_W1_average : {}".format( self.train_score_W1_average))
        print("train_subset_score_W2_average : {}".format( self.train_subset_score_W2_average))
       # print("test_score_W1_average : {}".format( self.test_score_W1_average))
       # print("test_score_W2_average : {}".format( self.test_score_W2_average))

        return
            # if train_subset_score_W2.average > train_score_W1.average
            # get W'
            # train_score_W'>train_score_W1
            # test_score_W'> test_score_W1
            #get W1 and W2 score in test data
