import numpy as np
import pandas as pd
import xlsxwriter
import  newSom 
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class GHSOM(object):
    def __init__(self, input_data,dim):
     #  if filetype == 0 :
     #   df = pd.read_excel(xlsxnmae)
     #  else:
     #   df = pd.read_csv(xlsxnmae)
# ---#-------------------------------input data random pre-process------------------------------------
     #  df = df.sample(frac=1)
     #  
     #  self.df_nominal = df.index
     #  print(self.df_nominal)
     #  df_numerical_tmp = df.iloc[:,1:]
     #
     #  self.df_numerical = df_numerical_tmp.apply(pd.to_numeric, errors='coerce').fillna(-1)
     #  # get data dim to latter SOM prcess
     #  self.input_dim = len(self.df_numerical.columns)
     #  self.input_num = len(self.df_numerical.index)
     #  # -----------------------------------input data random pre-process------------------------------------
     #  # change data to np array (SOM accept nparray format)
     #  input_data = np.array(self.df_numerical).astype(float)

    #    scaler = StandardScaler().fit(input_data)
    #    self.input_data = scaler.transform(input_data)
        self.input_data = input_data
        self.input_dim = dim
        self.weightDict = {}

    # input_data = np.array([[2., 1., 1., 1., 3.],
    #                         [2., 1., 0., 0., 3.],
    #                         [2., 1., 0., 1., 2.],
    #                         [2., 1., 0., 1., 3.],
    #                         [4., 0., 0., 0., 2.],
    #                         [4., 0., 1., 0., 1.],
    #                         [4., 0., 1., 0., 3.],
    #                         [4., 0., 1., 0., 2.],
    #                         [0., 0., 1., 0., 3.],
    #                         [0., 0., 1., 0., 2.]])
    # input_dim = 5
    # input_data = np.array([[10., 5., 5., 5., 15.],
    #                        [10., 5., 0., 0., 15.],
    #                        [10., 5., 0., 5., 10.],
    #                        [10., 5., 0., 5., 15.],
    #                        [20., 0., 0., 0., 10.],
    #                        [20., 0., 5., 0., 5.],
    #                        [20., 0., 5., 0., 15.],
    #                        [20., 0., 5., 0., 10.],
    #                        [0., 0., 5., 0., 15.],
    #                        [0., 0., 5., 0., 10.]])

    # calculate mqe with input,  if no result_index then cal mqe with all input
    # @param result_index: python list format

    #cal mqe in each clusters
    def cal_clustered_mqe(self,input_data, result_index = None):

        mqe = []

        if result_index is None:
            return self.cal_mqe(input_data)
        else:
            for i in result_index:
                indices = i
                if indices == []:
                    mqe.append(0.0)
                else:
                    clustered_input = np.take(input_data, indices, 0)
                    # print(clustered_input)
                    mqe.append(self.cal_mqe(clustered_input))
        return mqe


    def cal_mqe(self,input_data):
        # if input data equal 1 dimensions array , then expaned to 2 diemnsion  [2., 1., 0., 0., 3.] ==> [[2., 1., 0., 0., 3.]]
        if input_data.ndim == 1:
            input_data = np.expand_dims(input_data, axis=0)
        # np.mean(input_data, axis=0) get mean for each column
        # np.stack( np.mean(input_data, axis=0) for i in range(len(input_data)) repeat mean for len(input_data) times to make its dim is the same as input_data
        return np.mean(
                    np.sqrt(
                        np.sum(
                            np.power(
                                np.subtract(
                                    input_data,
                                    np.stack( np.mean(input_data, axis=0) for i in range(len(input_data)) )
                                )
                            , 2)
                        , axis=1)
                    )
                )

    def call_som(self,m, n, dim, input_data, weight_after_insertion=None):
        som = newSom.SOM(m, n, dim) 
        if weight_after_insertion is None:
            som.fit(input_data)
            trained_weight = som.weights0
        else:
            trained_weight = weight_after_insertion
            
        mapped = som.map_vects(input_data,trained_weight)  
        result = np.array(mapped)
        return trained_weight, result


    def som_neuron_locations(self,m, n):
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])


    # get each som result cluster (node) contains how many input
    def clustered_location_input_index(self,m, n, trained_weight, som_result_map, input_data):
        filter_map_value = np.array(list(self.som_neuron_locations(m, n)))
        # filter map output:
        # 3-D array
        # [[[0 0]   [[0 1]   [[1 0]   [[1 1]
        # [0 0]     [0 1]    [1 0]    [1 1]
        # [0 0]     [0 1]    [1 0]    [1 1]
        # [0 0]     [0 1]    [1 0]    [1 1]
        # [0 0]     [0 1]    [1 0]    [1 1]
        # [0 0]     [0 1]    [1 0]    [1 1]
        # [0 0]     [0 1]    [1 0]    [1 1]
        # [0 0]     [0 1]    [1 0]    [1 1]
        # [0 0]     [0 1]    [1 0]    [1 1]
        # [0 0]],   [0 1]]   [1 0]]   [1 1]]]
        # for i in range(0, m*n*2, 2):
        #     filter_map.append(np.stack( [np.squeeze(filter_map_value.reshape((1, -1))) for i in range(len(input_data))])[:,[i,i+1]])
        filter_map = []
        som_clusting_result_location_list = []
        for i in range(0, m*n*2, 2):
            filter_map_tmp = np.stack([np.squeeze(filter_map_value.reshape((1, -1))) for i in range(len(input_data))])[:,[i,i+1]]
            # print(filter_map_tmp)
            myarray_tmp = np.array(np.squeeze(np.array(np.where(np.all(som_result_map == filter_map_tmp, axis=1)))))

            if myarray_tmp.size == 1:
                som_clusting_result_location_list.append([myarray_tmp.tolist()])
            else:
                som_clusting_result_location_list.append(myarray_tmp.tolist())
        # return [[indices in cluster0],[ndices in cluster1],[ ndices in cluster2],[ndices in clustern]]
        return som_clusting_result_location_list

    # find which topology location is near to error unit
    def find_neighborhood_location(self, topology_map, m, n, error_unit_location):
        #print("error_unit_location {}".format(error_unit_location))
        stacked_error_unit = np.stack(error_unit_location  for i in range(m*n))
        #print("stacked_error_unit {}".format(stacked_error_unit))
        # print(np.absolute(np.sum(np.subtract(stacked_error_unit,topology_map), axis=1)))
        neighborhood_location_index_tmp = np.sum(np.absolute(np.subtract(stacked_error_unit,topology_map)), axis=1)

        #print("neighborhood_location_index_tmp {}".format(neighborhood_location_index_tmp))
        neighborhood_location_index = np.squeeze(np.array(np.where(neighborhood_location_index_tmp == 1)))
        #print("neighborhood_location_index {}".format(neighborhood_location_index))
        return neighborhood_location_index


    def get_dissimilar_weight_location(self,topology_map, error_unit_index, neighborhood_location_index, trained_weight):
        error_unit_weight = np.take(trained_weight, error_unit_index, 0)
        neighborhood_location_weight = np.take(trained_weight, neighborhood_location_index, 0)
        # get max value among neighbours
        dissimilar_weight_location_tmp_index = np.argmax(
                                                    np.sqrt(
                                                        np.sum(
                                                            np.power(
                                                                np.subtract(
                                                                    np.stack([error_unit_weight for i in range(len(neighborhood_location_weight))])
                                                                , neighborhood_location_weight)
                                                            , 2)
                                                        ,axis=1)
                                                    )
                                                )
        dissimilar_weight_location_index = np.take(neighborhood_location_index, dissimilar_weight_location_tmp_index)
        dissimilar_weight_location = np.take(topology_map, dissimilar_weight_location_index, 0)
        return dissimilar_weight_location, dissimilar_weight_location_index


    def insert_units(self,slice_point, weight_topology_map):
        print('------after insertation -------')
        units_to_be_inserted = np.divide(np.add(np.take(weight_topology_map, slice_point, 0), np.take(weight_topology_map, slice_point-1, 0)), 2)
        new_weight_topology_map = np.insert(weight_topology_map, slice_point, units_to_be_inserted, 0)

        return new_weight_topology_map


    def get_map_weight_after_unit_insertion(self, m, n, topology_map, error_unit_location, error_unit_index, dissimilar_weight_location, dissimilar_weight_location_index, trained_weight, input_dim):
        print('-----------get_map_weight_before_unit_insertion----------')
        #print(trained_weight)
        #print(topology_map)
        # print(error_unit_location)
        # print(dissimilar_weight_location)

        # check which layer should be insert
        if error_unit_index > dissimilar_weight_location_index:
            slice_point = error_unit_index
        else:
            slice_point = dissimilar_weight_location_index
        # insert at max index

        # check insert row or column
        if  np.argmax(np.absolute(np.subtract(error_unit_location, dissimilar_weight_location))) == 1:
            # insert one row
            print('insert row - add y direction')

            # slice point rem
            print('slice_point')
            slice_point = slice_point%n

            print(slice_point)

            print('-----------AfterReshape-----------')
            # rearrange weight array by row index
            trained_weight_default_index = np.arange(m*n)
            print('m {} n {}'.format(m,n))
            
            #print('trained_weight_default_index {}'.format(trained_weight_default_index))

            new_order = []

            for i in range(n):
                trained_weight_remainder = trained_weight_default_index%n
                #print('trained_weight_remainder {} i {}'.format(trained_weight_remainder,i))
                new_order = np.append(new_order, np.where(trained_weight_remainder == i)[0])
                #print('new_order {} i {}'.format(new_order,i))
            new_order = new_order.astype(int)
            # print('-----------new_order-----------')
            # print(new_order)
            # print(new_order)
            weight_topology_map  = trained_weight[new_order,:][:].reshape(n, m, -1).astype(np.float32)

            # print('------default_weight_topology_map----------')
            # print(weight_topology_map)

            new_weight_topology_map = self.insert_units(slice_point, weight_topology_map)
            # print('------new_weight_topology_map-------')
            # print(new_weight_topology_map)


            # reformat back to generator order
            # TODO: number 5 in reshpae must be equal to input_data ttribute length
            new_weight_after_insertion = np.swapaxes(new_weight_topology_map,0,1).reshape(-1, input_dim).astype(np.float32)
            # print('---------reshape back to default weight result-----------')
            # print(new_weight_after_insertion)

            # add row to initial
            new_n = n+1

            return new_weight_after_insertion, m, new_n

        else:
            # insert one column
            print('insert column - add x dirction')

            # check if slice point equal 0 if 0 then return 1 
            # if in the first row
            print('slice_point')
            if slice_point//n == 0:
                slice_point = 1
            else:
                slice_point = slice_point//n
            print(slice_point)

            print('-----------AfterReshape-----------')
            # todo : this 5 must be input_data attribute lengths
            weight_topology_map = trained_weight.reshape((m,n,-1))
            # print('------default_weight_topology_map')
            #print("weight_topology_map2 {}".format(weight_topology_map))
            new_weight_topology_map = self.insert_units(slice_point, weight_topology_map)
            # print('------new_weight_topology_map-------')
            # print(new_weight_topology_map)

            # print('---------reshape back to default weight result-----------')
            # print(new_weight_after_insertion)
            new_weight_after_insertion = new_weight_topology_map.reshape(-1, input_dim).astype(np.float32)


            new_m = m+1

            return new_weight_after_insertion, new_m, n



    def check_tau2_condition (self,clustered_result_by_index, input_data, reault_mqe, mqe0, input_dim,m,mapname,level):
        tau2 = 0.6
        print('---------------tau2*mqe0--------------')
        print(tau2*mqe0)
        print("for each reault_mqe size {}".format(len(reault_mqe)))
        for idx, each_mqe in enumerate(reault_mqe):
            if each_mqe < tau2*mqe0:
                print('satisfy tau2 condition')
            else:
                self.parentid = idx
                self.parent = self.trained_weight
                row=idx//m
                column=idx%m
                input_index = clustered_result_by_index[idx]
                print('-------------------')

                new_input_that_not_satisfy_tau2_condition = np.take(input_data, input_index, 0)

                m = 2
                n = 2
                level += 1

                new_mqe0 = self.cal_mqe(new_input_that_not_satisfy_tau2_condition)
                print("level {} ".format(level))
                print("new mqe0: "+ str(new_mqe0))

                self.check_tau1_condition(m, n, new_mqe0, new_input_that_not_satisfy_tau2_condition, input_dim,level,mapname=mapname,row=row,column=column)
                level -= 1



    def check_tau1_condition (self, m, n,  mqe0, input_data, dim,level, row=0,column=0, mapname="Average_Map0"):

        tau1 = 0.4

        tau1_iter_time = 1

        # if not satisfy then call som
        # is the location of each nurons that inputdata belongs to in the som
        trained_weight, som_result_map = self.call_som(m, n, dim, input_data)
        # find inputs in each unit
        clustered_result_by_index = self.clustered_location_input_index(m, n, trained_weight, som_result_map, input_data)
        # print('input that belong to same cluster:')
        # print(clustered_result_by_index)
        mqe = self.cal_clustered_mqe(input_data, clustered_result_by_index)

        print('each unit mqe:')
        print(mqe)
        n_mqe = list(filter(lambda x: x!=0.0,mqe))

        while True:
            if len(n_mqe)<=1:

                #all data in one cluster
                break
            if np.mean(mqe) < tau1*mqe0:

                print('mqe < tau1*mqe0')
                print('-------------------------------------after_tau1_iter_time-------------------------------------')
                print(tau1_iter_time)
                print('-------------------------------------mqe after insertation-------------------------------------')
                print(mqe)
                print('-------------------------------------input belong to unit  after insertation-------------------------------------')
                print('input that belong to same cluster:')

               # print("clustered_result_by_index {}".format(clustered_result_by_index))
               # result_by_ID = [[self.df_nominal[i] for i in j] for j in clustered_result_by_index]


              #  print("result_by_ID {}".format(result_by_ID))
              #  results=np.array(result_by_ID)
               # if level!=0:
               #     mapname = mapname  + '__unit' + '(' + str(row) + 'x' + str(column) + ')' + 'level' + str(level)+"shape="+str(m)+"x"+str(n)

               # workbook = xlsxwriter.Workbook(mapname+".xlsx")
               # worksheet = workbook.add_worksheet("shape="+str(m)+"x"+str(n))

               # for col, data in enumerate(results):
               #     worksheet.write_column(0,col , data)
               # workbook.close()


                print('-------------------------------------topology_map  after insertation-------------------------------------')
                print('m: ')
                print(m)
                print('n:')
                print(n)

                if len(self.weightDict) == 0:
                    self.weightDict = {level, [self.trained_weight] }
                else:
                    if self.weightDict.has_key(level):
                        currentweightdic = {'weight': self.trained_weight, 'parent':self.parent, 'parentnodeId':self.parentid }
                        self.weightDict[level].append(currentweightdic)
                    else:
                        self.weightDict.update({level,[self.trained_weight]})

                print('-------------------------------------begin tau2 check-------------------------------------')
                self.check_tau2_condition(clustered_result_by_index, input_data, mqe, mqe0, dim,m,mapname,level)

                break

            else:
                print('mqe > tau1*mqe0')

                topology_map = np.array(list(self.som_neuron_locations(m, n)))
                # find max mqe index
                error_unit_index = np.argmax(mqe)
                loaciton_map = np.array(list(self.som_neuron_locations(m, n)))
                error_unit_location = np.take(loaciton_map, error_unit_index, 0)
                #print('error_unit_location:')


                neighborhood_location_index = self.find_neighborhood_location(topology_map, m, n, error_unit_location)
                print('dissimilar_weight_location 2D coordinate:')
                dissimilar_weight_location, dissimilar_weight_location_index = self.get_dissimilar_weight_location(topology_map, error_unit_index, neighborhood_location_index, trained_weight)
                print(dissimilar_weight_location)
                # insert Unit
                new_weight_after_insertion, m, n = self.get_map_weight_after_unit_insertion(m, n, topology_map, error_unit_location, error_unit_index, dissimilar_weight_location, dissimilar_weight_location_index, trained_weight, dim)
                print('re-call SOM:')
                print(m)
                print(n)
                # print(new_weight_after_insertion)
                print("level {} ".format(level))
                # 2,3,4.... SOM call
                trained_weight, som_result_map = self.call_som(m, n, dim, input_data, new_weight_after_insertion)
                self.trained_weight = trained_weight
                # find how many input in each unit
                clustered_result_by_index = self.clustered_location_input_index(m, n, trained_weight, som_result_map, input_data)
                mqe = self.cal_clustered_mqe(input_data, clustered_result_by_index)
                tau1_iter_time += 1

    
    def train(self):
    #
    # Below is GHSOM Train step
    #
    # initial variable


        t0=time.time()
        mqe0 = self.cal_clustered_mqe(self.input_data)
        #initial mqe
        print("mqe0 {}".format(mqe0))
        # m => x direction, n => y direction
        m = 2
        n = 2

        self.check_tau1_condition(m, n, mqe0, self.input_data, self.input_dim,level=0)
        t=time.time()
        print (str(t-t0))


    def getlastweight(self):
        last level self.weightDict.keys()[-1]
        return
class Dict(dict):
     def __add__(self, other):
         copy = self.copy()
         copy.update(other)
         return copy
     def __radd__(self, other):
         copy = other.copy()
         copy.update(self)
         return copy