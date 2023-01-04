# -*- coding: utf-8 -*-
import torch
import numpy as np
import time


class Cluster_GPU():
    '''
    A pytorch GPU implementation for K-Means algorithm,
    which is used for real-time clutering in SCRL.
    '''
    def __init__(self, 
        num_clusters, 
        shift_threshold=1e-2, 
        max_iter=20, 
        device=torch.device('cuda'), 
        debug=False):
        self.cluster_fuc = KMeans_Mixed(
            num_clusters=num_clusters, 
            shift_threshold=shift_threshold, 
            max_iter=max_iter, 
            device=device
        )
        self.device = device
        self.debug = debug
    
    def __call__(self, x):
        dimension = len(x.size())
        x = x.to(self.device)
        B = x.size(0)
        output_vector = x.clone().detach()
        # D == 2
        if dimension == 2:
            _, choice_cluster, choice_points = self.cluster_fuc(output_vector, debug=self.debug)
        # D >= 3
        elif dimension == 3:
            choice_cluster_list, cluster_points_list = [], []
            for batch in range(B):
                y = output_vector.narrow(dim=0, start=batch, length=1).squeeze(0)
                _, choice_cluster, choice_points = self.cluster_fuc(y, debug=self.debug)
                choice_cluster_list.append(choice_cluster)
                cluster_points_list.append(choice_points)
            choice_cluster = np.stack(choice_cluster_list)
            choice_points = np.stack(cluster_points_list)
        else:
            raise ValueError('Dimension of input must <= 3, got {dimension} instead')
        return choice_cluster, choice_points


class KMeans_Mixed():
    '''
    This version uses GPU for tensor computation and 
    CPU for indexing to improve the speed of the algorithm.
    '''
    def __init__(self,
                num_clusters,
                shift_threshold,
                max_iter,
                cluster_centers = [],
                device=torch.device('cuda')):

        self.num_clusters = num_clusters
        self.shift_threshold = shift_threshold
        self.max_iter = max_iter
        self.cluster_centers = cluster_centers
        self.device = device
        self.pairwise_distance_func = pairwise_distance

    def initialize(self, X):
        num_samples = len(X)
        initial_indices = np.random.choice(num_samples, self.num_clusters, replace=False)
        initial_state = X[initial_indices]
        return initial_state

    def __call__(self, tensor_input, debug=False):
        if debug:
            time_start=time.time()

        X = tensor_input
        X = X.to(self.device)
        choice_points = np.ones(self.num_clusters)
        # init cluster center
        if type(self.cluster_centers) == list:
            initial_state = self.initialize(X)
        else:
            if debug:
                print('resuming cluster')
            initial_state = self.cluster_centers
            dis = self.pairwise_distance_func(X, initial_state, self.device)
            choice_points = torch.argmin(dis, dim=0)
            initial_state = X[choice_points]
            initial_state = initial_state.to(self.device)
        iteration = 0
        status = 0
        while status == 0:
            # CPU is better at indexing, so transfer the data to the cpu
            dis = self.pairwise_distance_func(X, initial_state, self.device).cpu().numpy()
            choice_cluster = np.argmin(dis, axis=1)
            initial_state_pre = initial_state.clone()
            for index in range(self.num_clusters):
                selected = np.where(choice_cluster == index)
                selected = X[selected]
                initial_state[index] = selected.mean(dim=0)
                dis_new = self.pairwise_distance_func(X, 
                    initial_state[index].unsqueeze(0), 
                    self.device).cpu().numpy()
                culuster_pos = np.argmin(dis_new, axis=0)
                # a cluster has at least one sample
                while culuster_pos in choice_points[:index]:
                    dis_new[culuster_pos] = np.inf
                    culuster_pos = np.argmin(dis_new, axis=0)

                choice_points[index] = culuster_pos
            initial_state = X[choice_points]

            center_shift = torch.sum(torch.sum((initial_state - initial_state_pre) ** 2, dim=1))

            iteration = iteration + 1

            if center_shift **2 < self.shift_threshold:
                status = 1
            if iteration >= self.max_iter:
                status = 2

            if debug:
                print("Iter: {} center_shift: {:.5f}".format(iteration, center_shift))

        if debug:
            if status == 1:
                time_end=time.time()
                print('Time cost: {:.3f}'.format(time_end-time_start))
                print("Stopped for the center_shift!")
            else:
                time_end=time.time()
                print('Time cost: {:.3f}'.format(time_end-time_start))
                print("Stopped for the max_iter!")
        return initial_state, choice_cluster, choice_points

# utils
def pairwise_distance(data1, data2, device=torch.device('cuda')):
    data1, data2 = data1.to(device), data2.to(device)
    # N*1*M
    A = data1.unsqueeze(dim=1)
    # 1*N*M
    B = data2.unsqueeze(dim=0)
    dis = (A - B) ** 2.0
    dis = dis.sum(dim=-1)
    return dis
