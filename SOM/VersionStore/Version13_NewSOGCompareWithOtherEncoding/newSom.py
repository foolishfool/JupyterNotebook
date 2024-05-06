"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
@author: Riley Smith
Created: 1-27-21
from readline import append_history_file
"""

import copy
import numpy as np
from sklearn import preprocessing
from scipy.spatial import distance

class SOM():
    """
    The 2-D, rectangular grid self-organizing map class using Numpy.
    """
    def __init__(self, m=3, n=3, dim=3, lr=1, sigma=1, max_iter=3000,
                    ):
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
 
        self.m =m
        self.n =n
        self.dim =dim
        self.lr = lr
        self.initial_lr = lr
        self.sigma =sigma
        self.max_iter = max_iter

        self.trained = False
        self.shape = (m, n)
    
        # Initialize weights
        self.random_state = None
        rng = np.random.default_rng(None)

        self.weights= rng.normal(size=(m * n, dim))
       # self.weights = np.zeros((m*n, dim))
        #print("initila self.weigts {} ".format(self.weights.shape))
        self.weights0= rng.normal(size=(m * n, dim))
        self.weights1= rng.normal(size=(m * n, dim))
        self.weights_onehot = rng.normal(size=(m * n, dim))
        #print("self.weights_onehot{}".format( self.weights_onehot))
        encoder = preprocessing.OneHotEncoder(max_categories= 6)
        
        totalweight =[]
       
        for i in range(0, self.weights_onehot.shape[0]):              
            emptyrow = []
            #print("self.weights_onehot[i:]{} i {}".format(self.weights_onehot[i,:],i))
            for item in self.weights_onehot[i,:]:                   
                emptyelement =np.append([],item)  
                emptyrow.append(emptyelement)
            emptyrow = np.array(emptyrow)
            #emptyrow = emptyrow.toarray()
            #print("emptyrow {}".format(emptyrow))
            emptyrow = encoder.fit_transform(emptyrow)       
            emptyrow = emptyrow.toarray()
            #print("emptyrow 2 {}".format(emptyrow))
            totalweight.append(emptyrow)
        #print("totalweight {}".format(totalweight))
        #input = [[0.], [0.76666665], [0.5], [0.23333333], [1.]]
        #self.weights_onehot = encoder.fit_transform(totalweight)
        #print("self.weights_onehot {}".format(self.weights_onehot))
       # print("1111111111111")
        self.weights_onehot = totalweight
        #print(" self.weights_onehot 2 {}".format( self.weights_onehot ))

        self._locations = self._get_locations(m, n)
        
       # print(self._locations)
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
    
    def _find_bmu(self,x, newWeights,showlog = False):
        """
        Find the index of the best matching unit for the input vector x.
        """
        #if showlog:  
        #    print("x len{} newWeights.shape[0]  {}".format(len(x), newWeights.shape[0])) 
        # Stack x to have one row per weight *********** get the all the element for one row
        # when split_nubmer = 0 corresponds to weight0, split_nubmer n represent Wn
        x_stack = np.stack([x]*(newWeights.shape[0]), axis=0)
        # Calculate distance between x and each weight  ， it use the norm to represent the distance of the concept of vector x_stack - newWeights
       # if showlog:
        #    print("x {} x_stack{}  newWeights {} m {} n{} dim{}".format(x, x_stack, newWeights, self.m,self.n,self.dim))
       # if x_stack.shape != newWeights.shape:
        #    print("x {} x_stack{}  newWeights {} m {} n{} dim{}".format(x, x_stack, newWeights, self.m,self.n,self.dim))
        distance = np.linalg.norm((x_stack - newWeights).astype(float), axis=1)
        # Find index of best matching unit
        return np.argmin(distance)
    
    
    def _find_bmu_hamming(self,x, newWeights):
        hamming_distances =[]
        for i in range(0,newWeights.shape[0] ):
            hdistance = distance.hamming(x, newWeights[i])
            hamming_distances.append(hdistance)
        
        mindex = min(hamming_distances)

        return hamming_distances.index(mindex)



    def _find_bmu_hamming_onehot(self,x, newWeights):
        hamming_distances =[]
       # print("x {} newweights {}".format(x, newWeights))
        for i in range(0,len(newWeights)):
            hdistance = 0
           # print("x {} newWeights[i] {}".format(x, newWeights[i]))
            for item in newWeights[i]:
             #   print("weighgt{}".format(item))
             #   print("distance.hamming(x, item){}".format( distance.hamming(x, item)))
                hdistance = hdistance + distance.hamming(x, item)
            hamming_distances.append(hdistance)
          #  print("hdistance {}".format( hdistance))
        #print("  {}".format(hamming_distances))
        mindex = min(hamming_distances)
        print("mindex {}".format( hamming_distances.index(mindex)))
        return hamming_distances.index(mindex)
    
    
    def step(self,x, showlog):
        """
        Do one step of training on the given input vector.
        """
      #  print(f"x {x}")
        # Stack x to have one row per weight 
        x_stack = np.stack([x]*(self.m*self.n), axis=0)
        #print("x_stack {}".format(x_stack))
        #print("self.weights{}".format(self.weights));
        #print("x_stack{}".format(x_stack));
        # x_stack , with mxn row , each row has the same array: x
        # Get index of best matching unit
       # print(showlog)
        if showlog == True:
            print("x {} ".format(x) )
        bmu_index = self._find_bmu(x,self.weights,showlog)
        if showlog == True:
            print("bmu_index{}".format(bmu_index))
       # print("self.weights{}".format(self.weights))
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
        #print("delta:{}".format(delta))
        delta = local_multiplier * (x_stack - self.weights).astype(float)
       
        #print("delta:{}".format(delta))
        #print("weights:{}".format(self.weights))
        # Update weights
        self.weights += delta
       # self.weights = np.round(self.weights,3)
        #print(f"self.weights at last {self.weights}")


    def step_hamming(self,x):
        """
        Do one step of training on the given input vector.
        """
        #print(x)
        # Stack x to have one row per weight 
        x_stack = np.stack([x]*(self.m*self.n), axis=0)
        # x_stack , with mxn row , each row has the same array: x
        # Get index of best matching unit
        bmu_index = self._find_bmu_hamming(x,self.weights)
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

    def step_hamming_onehot(self,x):
        """
        Do one step of training on the given input vector.
        """

        for item in x:
           # print("item {}".format(item))
        # Stack x to have one row per weight 
            x_stack = np.stack([x]*(self.m*self.n), axis=0)
        # x_stack , with mxn row , each row has the same array: x
        # Get index of best matching unit
            bmu_index = self._find_bmu_hamming_onehot(item,self.weights_onehot)
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
       ## print("x_stack:{}".format(x_stack.shape))
        #print("self.weights_onehot:{}".format(self.weights_onehot.shape))
        # Multiply by difference between input and weights
        delta = local_multiplier * (x_stack - self.weights_onehot).astype(float)
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
        #print("np.sum(np.square(x - bmu)) {}".format(np.sum(np.square(x - bmu))))
        # Compute sum of squared distance (just euclidean distance) from x to bmu
        return np.sum(np.square(x - bmu))

    
    def fit( self, X, weightIndex = 0,epochs=1, shuffle=False, showlog = False):
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
        #@print(111111)
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
            
          #  print(f"indices {indices}")
         # Train
            for idx in indices:
                
             # Break if past max number of iterations
                if global_iter_counter > self.max_iter:
                    break
                #print("idx =  {}  ".format( idx))
                #print(X[idx] )
                
                input = X[idx]
              #  print(f"idx {idx}") 
                #if (type(input) is np.float64):
                #    input = [input]
                # Do one step of training
                self.step(input,showlog)
                # Update learning rate
                global_iter_counter += 1
                self.lr = (1 - (global_iter_counter / total_iterations)) * self.initial_lr
    
        # Compute inertia
          
        inertia = np.sum(np.array([float(self._compute_point_intertia(x)) for x in X]))
        #print("inertia {}".format(inertia))
        self._inertia_ = inertia
    
    # Set n_iter_ attribute
        self._n_iter_ = global_iter_counter

    # Set trained flag
        self.trained = True
        if(weightIndex == 0):
            self.weights0 = copy.deepcopy(self.weights)
            #print(f"self.weights0 = {self.weights0}")
        if(weightIndex == 1):
            self.weights1 = copy.deepcopy(self.weights)
     
        return

    def fit_hamming( self, X, weightIndex = 0,epochs=1, shuffle=True):
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
        #print("X {}".format())

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
                self.step_hamming(input)
                # Update learning rate
                global_iter_counter += 1
                self.lr = (1 - (global_iter_counter / total_iterations)) * self.initial_lr
    
        # Compute inertia
          
        inertia = np.sum(np.array([float(self._compute_point_intertia(x)) for x in X]))
        #print("inertia {}".format(inertia))
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
    

    def fit_hamming_onehot( self, X, weightIndex = 0,epochs=1, shuffle=True):
        global_iter_counter = 0
    # the number of samples   
        n_samples = X.shape[0] 
        #print("n_samples {}".format(n_samples))
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
               # print("idx {} input : {}" .format(idx, input))
                #if (type(input) is np.float64):
                #    input = [input]
                # Do one step of training
                # input = [[onehotcode],[onehotcode],[onehotcode]]
                self.step_hamming_onehot(input)
                # Update learning rate
                global_iter_counter += 1
                self.lr = (1 - (global_iter_counter / total_iterations)) * self.initial_lr
    
        # Compute inertia
          
        inertia = np.sum(np.array([float(self._compute_point_intertia(x)) for x in X]))
        #print("inertia {}".format(inertia))
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
    


    
    def predict(self,X, newWeights):
        """
        train_data_clusters = [[1,2,6]]
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
     
        labels = np.array([self._find_bmu(x,newWeights) for x in X])
        #print(f" labels {labels}")
        return labels
    
    
    def predict_with_probaility(self,X, newWeights,train_data_clusters):
        """
        train_data_clusters = [[1,2,6].[4,61],[34.56]]
        the predicted clusters data
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
     
        labels = np.array([self._find_bmu_withprobability(x,newWeights,train_data_clusters) for x in X])
        return labels

    def predict_hamming(self,X, newWeights):
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
     
        labels = np.array([self._find_bmu_hamming(x,newWeights) for x in X])
        return labels      
    
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


    def map_vects(self, input_vects,newweights):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(newweights))],
                            key=lambda x: np.linalg.norm(vect-
                                                         newweights[x]))
            to_return.append(self._locations[min_index])

        return to_return

        
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

    