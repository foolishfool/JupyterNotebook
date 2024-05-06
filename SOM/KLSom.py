"""
Script to implement simple self organizing map using PyTorch, with methods
similar to clustering method in sklearn.
@author: Riley Smith
Created: 1-27-21
from readline import append_history_file
"""

import copy
from math import log2
import random
import numpy as np
from sklearn import preprocessing
from scipy.spatial import distance
from scipy.special import rel_entr, kl_div
from scipy.stats import entropy
from numpy.linalg import norm
class KLSOM():
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

       

     #   print("initila self.weigts {} ".format(self.weights))
        self.weights0= rng.normal(size=(m * n, dim))
        self.weights1= rng.normal(size=(m * n, dim))
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
    def kl_divergence(self, p, q):
        #print(f" p {p}  q {q}")
        return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
    
    def _find_bmu(self,x, newWeights,showlog = False):
        """
        Find the index of the best matching unit for the input vector x.
        """

        self.get_membership_distribution_based_neuron(x)

        all_kl =[]
       # print(f"newWeights {newWeights}")
       # for key in self.converted_x:
         #   print(f"self.converted_x[key]{self.converted_x[key]}nnewWeights[key]) {newWeights[key]}")
            #all_kl.append(self.JSD(self.converted_x[key],newWeights[key],True))
        for weight in newWeights :
           all_kl.append(self.JSD(x,weight))
           
        #print(f" all_kl {all_kl}")
       # distance = np.linalg.norm(all_kl, axis=2)
    #    print(f" all_kl {all_kl}np.argmin(all_kl) { np.argmin(all_kl)}")
        # Find index of best matching unit
        return np.argmin(all_kl)
    def _find_bmu2(self,x, newWeights,showlog = False):
        """
        Find the index of the best matching unit for the input vector x.
        """

        self.get_membership_distribution_based_neuron(x)

        all_kl =[]
        
        for key in self.converted_x:
           # print(f"self.converted_x[key]{self.converted_x[key]}nnewWeights[key]) {newWeights[key]}")
            all_kl.append(self.JSD(self.converted_x[key],newWeights[key],True))
        #for weight in newWeights :
        #f   all_kl.append(self.JSD(x,weight))
           
      #  print(f" all_kl {all_kl}   np.argmin(all_kl) { np.argmin(all_kl)}")
       # distance = np.linalg.norm(all_kl, axis=2)
    #    print(f" all_kl {all_kl}np.argmin(all_kl) { np.argmin(all_kl)}")
        # Find index of best matching unit
        return np.argmin(all_kl)
    def get_membership_distribution_based_neuron(self,x):
      
        sliced_x = np.array_split(x,self.m*self.n)
        #print(f"x  { x} sliced_x {sliced_x}  ")
        self.converted_x = {}
        for i in range(0, self.m*self.n):
               if i in self.converted_x:
                    #print(f"converted_x { self.converted_x} ")
                    self.converted_x[i].append(sliced_x[i])
               else: self.converted_x[i] = [sliced_x[i]]
     
     #   print(f"converted_x {self.converted_x}")
        return self.converted_x

    def JSD(self,P, Q, list_of_array = False):
      #  print(f"P  { P} Q {Q}")
       # return  np.linalg.norm( P[0]-Q[0])
        #print(f"P  { P} Q {Q}  ")
        if list_of_array == True:
            _P = P[0] / norm(P, ord=1)
            _Q = Q[0]/ norm(Q, ord=1)
        else:   
          #  print(P)
            _P = P / norm(P, ord=1)
            _Q = Q/ norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
       # print(0.5 * (entropy(_P, _M) + entropy(_Q, _M)))
        return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))
    
    def step(self,x, showlog):
        """
        Do one step of training on the given input vector.
        """
        #print(f"x {x}")
        # Stack x to have one row per weight 
      #  x_stack = np.stack([x]*(self.m*self.n), axis=0)
        #print("x_stack {}".format(x_stack))
        #print("self.weights{}".format(self.weights));
        #print("x_stack{}".format(x_stack));
        # x_stack , with mxn row , each row has the same array: x
        # Get index of best matching unit
       # print(showlog)
        #if showlog == True:
        #    print("x {} {}".format(x, self.weights.shape) )
        bmu_index = self._find_bmu(x,self.weights,showlog)
        #print(f"self.weights 1 {self.weights}  " )
        
      #  self.weights[bmu_index] =   [sum(x)/2 for x in zip(self.weights[bmu_index], self.converted_x[bmu_index] )]
        
       # self.weights[bmu_index] = [x / 2 for x in self.weights[bmu_index]]
       # print(f" bmu_index {bmu_index} self.converted_x[bmu_index] { self.converted_x[bmu_index] }  " )
       # print(f"self.weights 2 {self.weights}  " )
        #return

        #print("bmu_index{}".format(bmu_index));
        # Find location of best matching unit, _locations is all the indices for a given matrix for array
        # bmu_location is the bmu_indexth element in _locations, such as if bmu_index = 4 in [[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]] it return [2,0]
        bmu_location = self._locations[bmu_index,:]
        # Find square distance from each weight to the BMU
      #  print("bmu_location{}".format(bmu_location))
        stacked_bmu = np.stack([bmu_location]*(self.m*self.n), axis=0)
        #the distance among unit is calcuated by the distance among unit's indices
        #bmu_distance is an array with distance to each unit
       # print(f"np.power {np.power(self._locations.astype(np.float64) - stacked_bmu.astype(np.float64), 2)}")
        #bmu_distance =[]
        
        #for weight in self.weights:         
           # print(f" weight {weight} self.weights[bmu_index] {self.weights[bmu_index]}")  
           # bmu_distance.append(self.JSD(weight,self.weights[bmu_index]))
        #print("bmu_distance:{}".format(bmu_distance))
        #bmu_distance =  np.array(bmu_distance)
        bmu_distance = np.sum(np.power(self._locations.astype(np.float64) - stacked_bmu.astype(np.float64), 2), axis=1) #bmu_distance is the toplogy distance each unit to bmu
       # print("bmu_distance:{}".format(bmu_distance))
      #  bmu_distance = np.sum(kl_div(self._locations,bmu_location))
        #print("self._locations.astype(np.float64) - stacked_bmu.astype(np.float64) {}".format(self._locations.astype(np.float64) - stacked_bmu.astype(np.float64)))
        # Compute update neighborhood
        neighborhood = np.exp((bmu_distance / (self.sigma ** 2)) * -1)
      #  print("neighborhood:{}".format(neighborhood))
        #local_step is an array with stepchanges to each unit
        local_step = self.lr * neighborhood
        #print("local_step:{}".format(local_step))
        # Stack local step to be proper shape for update
        local_multiplier = np.stack([local_step]*(self.dim), axis=1)
       # print("local_multiplier:{}".format(local_multiplier))
        # Multiply by difference between input and weights
        #print("x_stack -:{}".format(x_stack ))
        x_stack_weights_difference = []
        for i in range(0, len(self.weights)):
              x_stack_weights_difference.append(self.JSD(x, self.weights[i]))
        #print("x_stack_weights_difference:{}".format(x_stack_weights_difference))
        #print("x_stack - self.weights:{}".format(np.absolute(x_stack - self.weights))) #meembeship value cannot be less than zero

       # delta = local_step * (x_stack - self.weights).astype(float)
       # delta = local_multiplier * (np.absolute(self.converted_x - self.weights)).astype(float)
       # delta ={}
      #  for key in self.converted_x:
       #     delta[key] =  local_step[key] * abs(self.converted_x[key][0] - self.weights[key][0])
      
        delta = local_step * (x_stack_weights_difference)
       # print("delta:{}".format(delta))
        delta =  np.stack([delta]*(self.dim), axis=1)
        #print("delta2:{}".format(delta))
        #print("weights:{}".format(self.weights))
        # Update weights
        self.weights += delta
       # for key in self.weights:
        #    self.weights[key] =   self.weights[key]+ delta[key]
       # self.weights = np.round(self.weights,3)
     #   print(f"self.weights at last {self.weights}")
   

    
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
        self.JSD(x,bmu)
        return np.sum(np.square(x - bmu))

    
    def fit( self, X, weightIndex = 0,epochs=1, shuffle=True, showlog = False):
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
       # self.weights = np.abs(  self.weights )
        self.weights = []
        randomindex =  random.sample(range(0, X.shape[0]), self.m * self.n)
        #random_x = random.choice(X)
       # self.weights = self.get_membership_distribution_based_neuron(random_x )
        #print(" self.weights  {}".format( self.weights ))
        for i in range(0, len(randomindex)):
           self.weights.append(X[randomindex[i]])
      #  print("self.weights. {}".format(self.weights))
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
                self.step(input,showlog)
                # Update learning rate
                global_iter_counter += 1
                self.lr = (1 - (global_iter_counter / total_iterations)) * self.initial_lr
    
        # Compute inertia
          
        #inertia = np.sum(np.array([float(self._compute_point_intertia(x)) for x in X]))
        #print("inertia {}".format(inertia))
        #self._inertia_ = inertia
    
    # Set n_iter_ attribute
        self._n_iter_ = global_iter_counter

    # Set trained flag
        self.trained = True
        if(weightIndex == 0):
            self.weights0 = copy.deepcopy(self.weights)
         #   print(f"self.weights0 = {self.weights0}")
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
        if (len(X.shape) == 1):
            print(f"X{X}")
        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'
        assert X.shape[1] == self.dim, f'This SOM has dimesnion {self.dim}. Received input with dimension {X.shape[1]}'
       # print(11111111111)
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

    