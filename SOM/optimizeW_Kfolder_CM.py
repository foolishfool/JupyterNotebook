"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
@author: Riley Smith
Created: 1-27-21
"""
from importlib import resources
from sklearn import metrics
import numpy as np
import random
import matplotlib.pyplot as plt

class OptimizeW():
    """
    The 2-D, rectangular grid self-organizing map class using Numpy.
    """
    def __init__(self, som, originalfeatureNum, X, Y,classNum = 2,k_folder_num_min =3, k_folder_num_max = 3,K_folder_list =[10], subset_percentage =0.1, keepFeatureColumns = [] ):
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
        self.X = X
        self.classNum = classNum
        self.K_folder_list = K_folder_list
        self.k_folder_num = k_folder_num_max
        self.subset_percentage = subset_percentage     
        self.data_tests =  Y
        self.keepFeatureColumns = keepFeatureColumns
        self.min_k_num = k_folder_num_min

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
        # the score of error dataset 
        self.train_score_W3 =  np.zeros(k_num)
        self.test_score_W1 =  np.zeros(k_num)
        self.test_score_W2 =  np.zeros(k_num)
        # the predcit labels and sublabels
        self.nLabels_predict =  np.zeros(k_num, dtype=object)
        self.nsubLabels_predict =  np.zeros(k_num, dtype=object)
        self.nLabels_errordata_predict =  np.zeros(k_num, dtype=object)

        self.validate_score_W1_predicted_labels =  np.zeros(k_num, dtype=object)
        self.validate_score_W2_predicted_labels =  np.zeros(k_num, dtype=object)
        self.train_error_score_W3_predicted_labels =  np.zeros(k_num, dtype=object)

        self.test_score_W1_predicted_labels =  np.zeros(k_num, dtype=object)
        self.test_score_W2_predicted_labels =  np.zeros(k_num, dtype=object)

        self.data_trains =  np.zeros(k_num, dtype=object)
        self.data_trains_error =  np.zeros(k_num, dtype=object)
        self.data_validates = np.zeros(k_num, dtype=object)
        
        self.label_trains = np.zeros(k_num, dtype=object)
        self.label_validates = np.zeros(k_num, dtype=object)
        self.label_tests = np.zeros(k_num, dtype=object)

        # the trainsubsets and train_subset labels
        self.train_subdatas =  np.zeros(k_num, dtype=object)
        self.train_sublabels=  np.zeros(k_num, dtype=object)
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
        data_train = self.X.drop(data_validate.index) # reduce data_validate from dataset
       # data_test = data_test.sample(self.test_num) # get random test_num samples from data_test
        # transfer to numpy array
        data_train = data_train.to_numpy(dtype=np.float64)
        data_validate = data_validate.to_numpy(dtype=np.float64)
        data_test = self.data_tests.to_numpy(dtype=np.float64)
      #  data_test = data_test.to_numpy(dtype=np.float64)
        # Initialize  train label and test label
        
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



    def purity_score(self,scorelist, y_true, y_pred,iter_index = 0):
        # compute contingency matrix (also called confusion matrix)
        #print(y_true.shape)
        #print(y_pred.shape)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        #print(iter_index)
        #print(contingency_matrix)
        # return purity
      
        scorelist[iter_index] = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
        #print ("scorelist {} scorelist[iter_index] {} iter_index : {}".format(scorelist, scorelist[iter_index],iter_index ))
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
                        newlist.append(idx) #idx is indice
                clusters.append(newlist) 
            # print(clusters)
            # [[indices of cluster 1],[indices of cluster 2],[indices of cluster 3 ]...]
            return clusters

    def getAlltraiingdataincies(self,i):
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

    def getErrorClusters(self,list_true,list_pred):
            """
            get the wrong data indices in the training data when predicted with weight1
            """
            #print("list_true {}".format(list_true))
            #print("list_pred {}".format(list_pred))           
            errorlist = []
            for i in range(0,len(list_true)):
                newlist = [item for item in list_pred[i] if item not in list_true[i]]
                # item is indices
                #print("newlist {} {} ".format(i, newlist))
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
        if(category == 2):
            #print("{} normalized predicted nsubLabel: \n{} ".format(iter_index,nsubLabel))
            self.nLabels_errordata_predict[iter_index] = nLabel



    def reduce_error_data(self,noisy_list, percent):
            newlist = []
            for x in noisy_list:
                for e in x:
                    newlist.append(e)
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
                data_train_subset= np.delete(X[indice], reduced_indices, axis=0)     
                self.train_subdatas[indice] = data_train_subset
                #self.train_subdatas[indice] = data_train_subset
               
            if(category == 1): 
                data_train_sublabel=np.delete(X[indice],reduced_indices, axis=0)
                self.train_sublabels[indice] = data_train_sublabel

            if(category == 2): 
                data_train_error=np.delete(X[indice],reduced_indices, axis=0)
                self.data_trains_error[indice] = data_train_error

            return 



    def runOptimize(self):

        validate_score_W1_averages = []
        validate_score_W2_averages = []
        test_score_W1_averages = []
        test_score_W2_averages = []
        k_num_range = np.arange(self.min_k_num,self.k_folder_num+1)
        for j in self.K_folder_list:
            print("k_folder : {}".format(j))
        #for j in range(self.min_k_num, self.k_folder_num+1):
            self._initialdatasetsize(j)
            for i in range(0, j):               
                # get train and test dataset 
                self._initializedataset(i,j,self.originalfeatureNum)
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

                right_data_indices = [item for item in self.data_trains.indices if item not in reduced_indices]
                reduced_indices_error = self.reduce_error_data(self.error_lists[i],0)
                #self.get_subset(reduced_indices_error,self.data_trains,3,i) # get train_error_data
                self.get_subset(reduced_indices_error,self.label_trains,2,i) 
                

                self.som.fit(self.data_trains_error[i],2)
                self.som_weights3s[i] = self.som.weights3

                self.som_weights13_difference[i] = self.som.weights3 - self.som.weights1
                print("Difference of Weights: {}".format(self.som_weights13_difference[i]))
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
                self.test_score_W1_predicted_labels[i] = self.som.predict(self.data_test,self.som.weights1)
                self.NormalizeLables(self.test_score_W1_predicted_labels[i],0,i)
            ## get cluster accuracy in train_sub_data with W2
                self.purity_score(self.test_score_W1,self.label_tests,self.nLabels_predict[i],i)
                
                self.test_score_W2_predicted_labels[i] = self.som.predict(self.data_test,self.som.weights2)
                self.NormalizeLables(self.test_score_W2_predicted_labels[i],1,i)
            ## get cluster accuracy in train_sub_data with W2
                self.purity_score(self.test_score_W2,self.label_tests,self.nsubLabels_predict[i],i)
                #print("self.test_score_W2 {}".format(self.test_score_W2))
            #
            #print("Test W2 nLabels_predict ! {}".format(self.nLabels_predict[i]))
           # print("self.validate_score_W1 size {}".format(self.validate_score_W1))

            validate_score_W1_average = np.average(self.validate_score_W1)
            validate_score_W2_average = np.average(self.validate_score_W2)
            validate_score_W3_average = np.average(self.train_score_W3)
            test_score_W1_average = np.average(self.test_score_W1)
            test_score_W2_average = np.average(self.test_score_W2)
            
              

            print("k number :{}".format(j))
            print("validate_score_W1_average : {}".format( validate_score_W1_average))
            print("validate_score_W1_average : {}".format( validate_score_W1_average))
            print("train_error_score_W3_average : {}".format( validate_score_W3_average))
            print("test_score_W1_average : {}".format( test_score_W1_average))
            print("test_score_W2_average : {}".format( test_score_W2_average))
        # self.test_score_W1_average = np.average(self.som_weights1s)
            #self.weights2_average = np.average(self.som_weights2s)
            #print("weights2_average : {}".format( self.som_weights2s))
            validate_score_W1_averages.append( validate_score_W1_average )
            validate_score_W2_averages.append( validate_score_W2_average )
            test_score_W1_averages.append( test_score_W1_average )
            test_score_W2_averages.append( test_score_W2_average )

        betterScoreList_train = []
        for i in range(0,len(validate_score_W1_averages)):
            if(validate_score_W2_averages[i]>=validate_score_W1_averages[i]):
                betterScoreList_train.append(validate_score_W2_averages[i] -validate_score_W1_averages[i])

        betterScoreList_test = []
        for i in range(0,len(test_score_W1_averages)):
            if(test_score_W2_averages[i]>=test_score_W1_averages[i]):
                betterScoreList_test.append(test_score_W2_averages[i] -test_score_W1_averages[i])

        betterscore_train = len(betterScoreList_train)
        betterscore_test = len(betterScoreList_test)

        plt.plot(k_num_range,validate_score_W1_averages,'g', label ='validate_score_W1_averages')
        plt.plot(k_num_range,validate_score_W2_averages,'r', label ='validate_score_W2_averages')
        plt.plot(k_num_range,test_score_W1_averages,'b', label ='test_score_W1_averages')     
        plt.plot(k_num_range,test_score_W2_averages,'y', label ='test_score_W2_averages')
        plt.xlabel("Feature Number: {} Better Score train: {} Better Score test: {}".format(self.originalfeatureNum,betterscore_train,betterscore_test))          
        plt.legend()
        plt.show()
        
        print("betterScoreList_train: {}".format(betterScoreList_train))
        print("betterScoreList_test: {}".format(betterScoreList_test))


        return self.originalfeatureNum, betterscore_train
    
