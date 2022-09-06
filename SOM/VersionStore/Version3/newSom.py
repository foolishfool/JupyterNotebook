"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
@author: Riley Smith
Created: 1-27-21
"""
import copy
import math
import numpy as np
from mayavi import mlab
from scipy import spatial
class SOM():
    """
    The 2-D, rectangular grid self-organizing map class using Numpy.
    """
    def __init__(self, m=3, n=3, dim=3, lr=1, sigma=1, max_iter=3000,
                    random_state=None):
        """
        Parameters
        ----------
        m : int, default=3
            The shape along dimension 0 (vertical) of the SOM.
        n : int, default=3
            The shape along dimesnion 1 (horizontal) of the SOM.
        dim : int, default=3
            The dimensionality (number of features) of the input space.
        lr : float, default=1
            The initial step size for updating the SOM weights.
        sigma : float, optional
            Optional parameter for magnitude of change to each weight. Does not
            update over training (as does learning rate). Higher values mean
            more aggressive updates to weights.
        max_iter : int, optional
            Optional parameter to stop training if you reach this many
            interation.
        random_state : int, optional
            Optional integer seed to the random number generator for weight
            initialization. This will be used to create a new instance of Numpy's
            default random number generator (it will not call np.random.seed()).
            Specify an integer for deterministic results.
        """
        # Initialize descriptive features of SOM
 
        self.currentWIndex = 0
        self.m =m
        self.n =n
        self.dim =dim
        self.lr = lr
        self.initial_lr = lr
        self.sigma =sigma
        self.max_iter = max_iter
        self.random_max_iter  = 100

        self.trained = False
        self.shape = (m, n)
    
        # Initialize weights
        self.random_state = random_state
        rng = np.random.default_rng(random_state)

        self.weights= rng.normal(size=(m * n, dim))
        #print("initila self.weigts {}".format(self.weights))
        self.weights0= rng.normal(size=(m * n, dim))
        self.weights1= rng.normal(size=(m * n, dim))
        self._locations = self._get_locations(m, n)
         
        #print(self.weights)
        # Set after fitting
        self._inertia = None
        self._n_iter_ = None
        self._trained = False

    def _get_locations(self, m, n):
        """
        Return the indices of an m by n array.
        """
        # the element in these indices are non-zero
        #return a group of indices for each suitable elements in a group or matrix 
        #print("m n {} {}".format(m,n))
        #print("np.ones(shape=(m, n){}".format(np.ones(shape=(m, n))))
        #print("_get_locations( m, n){}".format(np.argwhere(np.ones(shape=(m, n))).astype(np.int64)))
        return np.argwhere(np.ones(shape=(m, n))).astype(np.int64)

    def _find_bmu(self,x, newWeights, combined = False, train_counter = 0):
        """
        Find the index of the best matching unit for the input vector x.
        """  
       # print("x:{}".format(x)) 
        # Stack x to have one row per weight *********** get the all the element for one row
        if(combined== False):
            x_stack = np.stack([x]*(self.m*self.n), axis=0)
        else:  
            x_stack = np.stack([x]*(self.m*self.n* (train_counter +2)), axis=0)
        # Calculate distance between x and each weight  ， it use the norm to represent the distance of the concept of vector x_stack - newWeights
        #print("x_stack:{}".format(x_stack ))   #, axis =1  process by row
        #print("newWeights:{}".format(newWeights ))   #, axis =1  process by row
        #print("x_stack - newWeights:{}".format(x_stack - newWeights )) 
        #a = np.array(x_stack - newWeights,dtype=object) 
        #arr_float64 = a.astype(float)
          #, axis =1  process by row
        distance = np.linalg.norm((x_stack - newWeights).astype(float), axis=1)
        #print("distance:{}".format(distance ))   
       # print("min distance:")   
        #print(np.argmin(distance))
        # Find index of best matching unit
        return np.argmin(distance)


    def _find_bmu_among_multipleW(self,x, Train_Split_Data, Weights):
        """
        Find the index of the best matching unit for the input vector x.
        """  
        #initial distance

        distance = math.dist(x,Train_Split_Data[0][0])
        w_index = 0
        #print("initial distance {}".format(distance)) 
        #print("len(Train_Split_Data) {}".format(len(Train_Split_Data))) 
        for i in range(0,len(Train_Split_Data)):
            tree = spatial.KDTree(Train_Split_Data[i])
            #print("i {}".format(i)) 
            #print("Train_Split_Data[i] {}".format(Train_Split_Data[i])) 
            currentdistance = tree.query(x)[0]
            nearestDataindex = tree.query(x)[1]
            #print("nearestDataindex {}".format(nearestDataindex)) 
       
            if currentdistance < distance:
               # print("currentdistance {} distance {}".format(currentdistance,distance)) 
                distance = currentdistance
                w_index = i
                #print("currentdistance {}  i {} distance {} ".format(currentdistance, i, distance ))
                 
        # Stack x to have one row per weight *********** get the all the element for one row
        #print("w_index {} distance {}".format(w_index,distance) )
        x_stack = np.stack([x]*(self.m*self.n), axis=0)
          #, axis =1  process by row
        distance = np.linalg.norm((x_stack - Weights[w_index]).astype(float), axis=1)
        #print("distance:{}".format(distance ))   
       # print("min distance:")   
        #print(np.argmin(distance))
        # Find index of best matching unit and belonged w index
        return w_index, np.argmin(distance)


    def step(self,x):
        """
        Do one step of training on the given input vector.
        """
        #print(x)
        # Stack x to have one row per weight 
        x_stack = np.stack([x]*(self.m*self.n), axis=0)
        #print("x_stack {}".format(x_stack))
        #print("self.weights{}".format(self.weights));
        #print("x_stack{}".format(x_stack));
        # x_stack , with mxn row , each row has the same array: x
        # Get index of best matching unit
        bmu_index = self._find_bmu(x,self.weights)
        #print("bmu_index{}".format(bmu_index));
        # Find location of best matching unit, _locations is all the indices for a given matrix for array
        # bmu_location is the bmu_indexth element in _locations, such as if bmu_index = 4 in [[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]] it return [2,0]
        bmu_location = self._locations[bmu_index,:]
        #print("bmu_location{}".format(bmu_location));
        # Find square distance from each weight to the BMU
        #print("[bmu_location]*(m*n){}".format([bmu_location]*(m*n)));
        stacked_bmu = np.stack([bmu_location]*(self.m*self.n), axis=0)
        #print("stacked_bmu: {}".format(stacked_bmu))
        #the distance among unit is calcuated by the distance among unit's indices
        #bmu_distance is an array with distance to each unit
        bmu_distance = np.sum(np.power(self._locations.astype(np.float64) - stacked_bmu.astype(np.float64), 2), axis=1)
       # print("bmu_distance:{}".format(bmu_distance))
        # Compute update neighborhood
        neighborhood = np.exp((bmu_distance / (self.sigma ** 2)) * -1)
       # print("neighborhood:{}".format(neighborhood))
        #local_step is an array with stepchanges to each unit
        local_step = self.lr * neighborhood
        #print("local_step:{}".format(local_step))
        # Stack local step to be proper shape for update
        local_multiplier = np.stack([local_step]*(self.dim), axis=1)
        #print("local_multiplier:{}".format(local_multiplier))
        # Multiply by difference between input and weights
        delta = local_multiplier * (x_stack - self.weights).astype(float)
        #print("delta:{}".format(delta))
       # print("weights:{}".format(self.weights))
        # Update weights
        self.weights += delta
       
    
    def _compute_point_intertia(self, x):
        """
        Compute the inertia of a single point. Inertia defined as squared distance
        from point to closest cluster center (BMU)
        """
        
        # Find BMU
        bmu_index = self._find_bmu(x,self.weights)
        bmu = self.weights[bmu_index]
        # Compute sum of squared distance (just euclidean distance) from x to bmu
        return np.sum(np.square(x - bmu))


    def fit( self, X, weightIndex = 0,epochs=1, shuffle=True):
        """
        Take data (a tensor of type float64) as input and fit the SOM to that
        data for the specified number of epochs.
        Parameters
        ----------
        X : ndarray
            Training data. Must have shape (n, self.dim) where n is the number
            of training samples.
        epochs : int, default=1
            The number of times to loop through the training data when fitting.
        shuffle : bool, default True
            Whether or not to randomize the order of train data when fitting.
            Can be seeded with np.random.seed() prior to calling fit.
        Returns
        -------
        None
            Fits the SOM to the given data but does not return anything.
        """

        #print("X {}".format(X))
        # Count total number of iterations
        global_iter_counter = 0
    # the number of samples   
        n_samples = X.shape[0] 
        total_iterations = np.minimum(epochs * n_samples, self.max_iter)
        for epoch in range(epochs):
            # Break if past max number of iterations
            if global_iter_counter > self.max_iter:
                break

            if shuffle:
                rng = np.random.default_rng(self.random_state)
                indices = rng.permutation(n_samples)
                #print("indices1 {}".format(indices))
                # permute the index of samples
                indices = np.array(indices)
                #print("indices2 {}".format(indices))
            else:
                indices = np.arange(n_samples)                       
            

         # Train
            for idx in indices:

             # Break if past max number of iterations
                if global_iter_counter > self.max_iter:
                    break
                #print("idx =  {}  ".format( idx))
                #print(X[idx] )
                
                input = X[idx]
                #if (type(input) is np.float64):
                #    input = [input]
                # Do one step of training
                self.step(input)
                # Update learning rate
                global_iter_counter += 1
                lr = (1 - (global_iter_counter / total_iterations)) * self.initial_lr
    
        # Compute inertia
          
        inertia = np.sum(np.array([float(self._compute_point_intertia(x)) for x in X]))
        self._inertia_ = inertia

    # Set n_iter_ attribute
        self._n_iter_ = global_iter_counter

    # Set trained flag
        self.trained = True
        if(weightIndex == 0):
            self.weights0 = copy.deepcopy(self.weights)

        if(weightIndex == 1):
            self.weights1 = copy.deepcopy(self.weights)

        return
  
    def predict(self,X, newWeights, combined= False, train_counter = 0):
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

        #print("weights used:\n")
        #print(newWeights)
        # Check to make sure SOM has been fit
        if not self.trained:
            raise NotImplementedError('SOM object has no predict() method until after calling fit().')

        # Make sure X has proper shape
        #print("len(X.shape) {}".format(len(X.shape)))
        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'
        assert X.shape[1] == self.dim, f'This SOM has dimesnion {self.dim}. Received input with dimension {X.shape[1]}'
     
        labels = np.array([self._find_bmu(x,newWeights,combined,train_counter) for x in X])
    
        #print("predicted label:\n {}".format(labels))
        return labels

    def predict_among_multipleW(self,X,Train_Split_Data, weights):
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

        #print("weights used:\n")
        #print(newWeights)
        # Check to make sure SOM has been fit
        if not self.trained:
            raise NotImplementedError('SOM object has no predict() method until after calling fit().')

        # Make sure X has proper shape
        #print("len(X.shape) {}".format(len(X.shape)))
        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'
        assert X.shape[1] == self.dim, f'This SOM has dimesnion {self.dim}. Received input with dimension {X.shape[1]}'
        #  def _find_bmu_among_multipleW(self,x, Train_Split_Data, Weights):
        winWindexlabels = []
        labels =[]
        i = 0
        for x in X:
            a,b = self._find_bmu_among_multipleW(x,Train_Split_Data,weights)
            winWindexlabels.append(a)
            labels.append(b)
            i = i+1

       # winWindexlabels, labels = np.array([ for x in X])
        # labels will be always from 0 - m*n-1
        #print("predicted label:\n {}".format(labels))
        return np.array(winWindexlabels), np.array(labels)


    def transform(self, X):
        """
        Transform the data X into cluster distance space.
        Parameters
        ----------
        X : ndarray
            Data of shape (n, self.dim) where n is the number of samples. The
            data to transform.
        Returns
        -------
        transformed : ndarray
            Transformed data of shape (n, self.n*self.m). The Euclidean distance
            from each item in X to each cluster center.
        """
        # Stack data and cluster centers
        X_stack = np.stack([X]*(self.m*self.n), axis=1)
        cluster_stack = np.stack([self.weights]*X.shape[0], axis=0)

        # Compute difference
        diff = X_stack - cluster_stack

        # Take and return norm
        return np.linalg.norm(diff, axis=2)

    def fit_predict(self, X, **kwargs):
        """
        Convenience method for calling fit(X) followed by predict(X).
        Parameters
        ----------
        X : ndarray
            Data of shape (n, self.dim). The data to fit and then predict.
        **kwargs
            Optional keyword arguments for the .fit() method.
        Returns
        -------
        labels : ndarray
            ndarray of shape (n,). The index of the predicted cluster for each
            item in X (after fitting the SOM to the data in X).
        """
        # Fit to data
        self.fit(X, **kwargs)

        # Return predictions
        return self.predict(X)
        

    def fit_transform(self, X, **kwargs):
        """
        Convenience method for calling fit(X) followed by transform(X). Unlike
        in sklearn, this is not implemented more efficiently (the efficiency is
        the same as calling fit(X) directly followed by transform(X)).
        Parameters
        ----------
        X : ndarray
            Data of shape (n, self.dim) where n is the number of samples.
        **kwargs
            Optional keyword arguments for the .fit() method.
        Returns
        -------
        transformed : ndarray
            ndarray of shape (n, self.m*self.n). The Euclidean distance
            from each item in X to each cluster center.
        """
        # Fit to data
        self.fit(X, **kwargs)

        # Return points in cluster distance space
        return self.transform(X)

    @property
    def cluster_centers_(self):
        return self.weights.reshape(self.m, self.n, self.dim)

    @property
    def inertia_(self):
        if self._inertia_ is None:
            raise AttributeError('SOM does not have inertia until after calling fit()')
        return self._inertia_

    @property
    def n_iter_(self):
        if self._n_iter_ is None:
            raise AttributeError('SOM does not have n_iter_ attribute until after calling fit()')
        return self._n_iter_