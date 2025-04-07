# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause
#
# Contact: Mattia Milani (Nokia) <mattia.milani@nokia.com>

"""
class module
============

Use this module to manage a class object.
"""

import ctypes
import numpy as np
from typing_extensions import Self
from typing import List, Tuple, Type, Union
from scipy.spatial import KDTree
from tf_cure.src.contextManagers.contextManagers import timeit_cnt as timeit
import tensorflow as tf

from .tree import tf_rtree

# Is not possible to use neither split or unstuck, such function are
# not available in graph mode.
# Up to now the only solution seems to be concat
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),
                              tf.TensorSpec(shape=[], dtype=tf.int64),
                              tf.TensorSpec(shape=[], dtype=tf.int64)))
def tf_concat(points, slice_idx, delta) -> tf.Tensor:
    return tf.concat([points[:slice_idx], points[slice_idx+delta:]], 0)

@tf.function(input_signature=(tf.RaggedTensorSpec(shape=[None, None, None],
                                                  dtype=tf.float32,
                                                  ragged_rank=1,
                                                  row_splits_dtype=tf.int64),
                              tf.TensorSpec(shape=[], dtype=tf.int64),
                              tf.TensorSpec(shape=[], dtype=tf.int64)))
def ragged_tf_concat(points, slice_idx, delta) -> tf.Tensor:
    return tf.concat([points[:slice_idx], points[slice_idx+delta:]], 0)

@tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.RaggedTensorSpec(shape=[None, None, None],
                                                  dtype=tf.float32,
                                                  ragged_rank=1,
                                                  row_splits_dtype=tf.int64)))
def compute_dst(rep: tf.Tensor, adv: tf.RaggedTensor) -> tf.Tensor:
    rep_r = tf.tile(tf.expand_dims(rep, axis=0), multiples=[adv.nrows(), 1, 1])

    # Count the number of elements for each row of the adv
    row_len = tf.cast(adv.row_lengths(), tf.int32)
    rep_len = tf.cast(tf.shape(rep)[0], tf.int32)

    rep_r = tf.tile(rep_r, multiples=[1, tf.reduce_max(row_len), 1])
    adv_r = adv.to_tensor()
    adv_r = tf.repeat(adv_r, repeats=rep_len, axis=1)
    rep_r = tf.where(adv_r == tf.zeros(tf.shape(rep)[1], dtype=tf.float32), adv_r, rep_r)

    # # Calculate the euclidean distance between each row of adv_r and rep_r
    dst = tf.math.square(tf.norm(tf.math.subtract(adv_r, rep_r), axis=-1))
    dst = tf.where(dst == 0., np.inf, dst)
    dst = tf.reduce_min(dst, axis=1)
    return dst

@tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None], dtype=tf.float32)))
def pp_dst(points: tf.Tensor, adv: tf.Tensor) -> tf.Tensor:
    points_shape = tf.shape(points)
    adv_shape = tf.shape(adv)
    points_r = tf.repeat(points, repeats=adv_shape[0], axis=0)
    adv_r = tf.tile(adv, multiples=[tf.shape(points_r)[0] // adv_shape[0], 1])
    all_dst = tf.math.square(tf.norm(tf.math.subtract(points_r, adv_r), axis=-1))
    all_dst = tf.reshape(all_dst, (points_shape[0], adv_shape[0]))
    all_dst = tf.transpose(all_dst)
    return all_dst

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.bool),
                              tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None], dtype=tf.float32)))
def rep_loop_body(selected_rep, representors, points):
    update = tf.constant([False])
    adv = tf.cond(tf.reduce_all(selected_rep), lambda: tf.expand_dims(tf.reduce_mean(points, axis=0), 0),
                  lambda: tf.gather(representors, tf.where(tf.equal(selected_rep, False))[:, 0], axis=0))

    # Compute best point
    all_dst = pp_dst(points, adv)
    min_dst = tf.reduce_min(all_dst, axis=0)
    max_point = tf.gather(points, tf.math.argmax(min_dst), axis=0)
    exp_max_p = tf.expand_dims(max_point, 0)

    idx = tf.expand_dims(tf.expand_dims(tf.where(tf.equal(selected_rep, True))[0][0], 0), 0)
    selected_rep = tf.tensor_scatter_nd_update(selected_rep, idx, update)
    representors = tf.tensor_scatter_nd_update(representors, idx, exp_max_p)

    return selected_rep, representors, points

@tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[], dtype=tf.int32)))
def static_select_representors(points: tf.Tensor,
                               n_rep: tf.Tensor) -> tf.Tensor:
    """select_representors.

    Select the representors for the given cluster.

    Parameters
    ----------
    points : tf.Tensor
        points of the cluster
    n_rep : tf.Tensor
        number of representors to select

    Returns
    -------
    tf.Tensor
        representors for the cluster
    """
    # Tensor full of False for n_rep
    selected_rep = tf.fill([n_rep], True)

    # Representors tf variable that contains the current selected representors
    # At the beginning is full of the points average value repeated n_rep times
    representors = tf.expand_dims(tf.reduce_mean(points, axis=0), 0)
    representors = tf.tile(representors, [n_rep, 1])

    # Condition, as long as all selected rep are false then return True
    cond = lambda _x, _y, _w: tf.reduce_any(_x)

    # Loop body
    selected_rep, representors, _ = tf.while_loop(cond, rep_loop_body, [selected_rep, representors, points])

    return representors

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),
                              tf.TensorSpec(shape=[None], dtype=tf.int32)))
def static_get_idx(c_id: tf.Tensor, ids: tf.Tensor) -> tf.Tensor:
    mask = tf.equal(ids[:, tf.newaxis], c_id[tf.newaxis, :])
    return tf.where(mask)[: , 0]

@tf.function(input_signature=(tf.RaggedTensorSpec(shape=[None, None, None],
                                                  dtype=tf.float32,
                                                  ragged_rank=1,
                                                  row_splits_dtype=tf.int64),
                              tf.TensorSpec(shape=[None], dtype=tf.int64)))
def static_merge_clst(clusters: tf.RaggedTensor,
                      idx: tf.Tensor) -> tf.Tensor:
    clst = tf.gather(clusters, idx, axis=0)
    return tf.reshape(clst, [-1, tf.shape(clst)[-1]])

@tf.function(input_signature=(tf.RaggedTensorSpec(shape=[None, None, None],
                                                  dtype=tf.float32,
                                                  ragged_rank=1,
                                                  row_splits_dtype=tf.int64),
                              tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[], dtype=tf.int64),
                              tf.TensorSpec(shape=[], dtype=tf.int64)))
def ragged_tf_swap(points, new_point, slice_idx, delta) -> tf.Tensor:
    return tf.concat([points[:slice_idx], [new_point], points[slice_idx+delta:]], 0)

class tf_clusters():
    """TfClusters.

    Manage a queue of clusters.
    To apply operations to the tree, all functions requires the tree object
    passed as parameter, and it's later returned as part of the output.
    This is done such that the tree object is not stored in the class.
    And it's later updated by the caller.
	"""

    def __init__(self, points: tf.Tensor,
                 n_rep: int = 5,
                 compression_factor: float = 0.5):
        self.n_rep = n_rep
        self.compression_factor = compression_factor
        self.timeit_flag = False

        with timeit("Cluster init with ragged tensors", active=self.timeit_flag):
            # At init each point is a different cluster, all clusters are in
            # a ragged tensor with shape ([n_points, 1, n_features])
            points = tf.expand_dims(points, 1)
            self.__r_clusters = tf.RaggedTensor.from_tensor(points,
                                                            ragged_rank=1)
            self.__r_reps = tf.RaggedTensor.from_tensor(points,
                                                        ragged_rank=1)
            self.__ids = tf.range(points.shape[0])

            # Define the closest tensor as full of -1 at the beginning
            self.__closest = tf.Variable(tf.zeros([points.shape[0]]))

            # Define the distances tensor as all 0s at the beginning
            self.__distances = tf.Variable(tf.zeros([points.shape[0]]))

        # self.apply_rep_compression()

    def get_idx(self, c_id: Union[int, tf.Tensor]) -> Union[int, tf.Tensor]:
        """get_idx.

        Get the index of the cluster given the id.

        Parameters
        ----------
        c_id : Union[int, tf.Tensor]
            if int the id of the cluster to get the index
            if a tensor then gather the indexes of the clusters

        Returns
        -------
        Union[int, tf.Tensor]
            index of the cluster, if c_id is int then the output is an int
            otherwise a tensor with the indexes.

        Raises
        ------
        ValueError
            If the id is not in the range of the clusters
        """
        if c_id.shape == () or len(c_id) == 1:
            # if c_id in a tensor int64 cast it to int32
            c_id = tf.cast(c_id, tf.int32)
            tf_idx = tf.where(self.ids == c_id)
            if tf_idx.shape[0] == 0:
                raise ValueError("The id is not in the range of the clusters")
            return tf_idx[0][0]
        else:
            mask = tf.equal(self.ids[:, tf.newaxis], c_id[tf.newaxis, :])
            tf_idx = tf.where(mask)[: , 0]
            return tf_idx

    def best_closest(self,
                     c_id: int,
                     rep: Union[tf.Tensor, tf.RaggedTensor],
                     tree: tf_rtree,
                     debug: bool = False) -> Tuple[int, float]:
        """best_closest.

        Compute the closest cluster to the given set of representors.

        Parameters
        ----------
        c_id : int
            c_id of the cluster
        rep : Union[tf.Tensor, tf.RaggedTensor]
            representors to compute the closest cluster
        tree : tf_rtree
            tree with all the clusters
        debug : bool
            flag to enable debug print

        Returns
        -------
        Tuple[int, float]
            id of the closest cluster and the distance from the cluster
        """
        # The representors tensor is ragged on the second dimension, to compute
        # n I need to get the current length on that dimension
        with timeit("init n_reprers and n_request", active=debug):
            if isinstance(rep, tf.RaggedTensor):
                n_dim = rep.shape[-1]
                rep = tf.reshape(rep.to_tensor(), [-1, n_dim])
            n_reprers = rep.shape[0]
            n_request = n_reprers + 1

        with timeit("Compute closest distance", active=debug):
            closest_ids = tree.nearest(rep, n=n_request)

        # The closest cluster is usually the second element becuse the first is
        # the cluster itself. Remove from the tensor where == id
        with timeit("Clean closest ids", active=debug):
            closest_ids = tf.reshape(closest_ids, [-1])
            tf_mask = closest_ids != c_id
            # closest_ids = tf.boolean_mask(closest_ids, tf_mask) # Super slow
            closest_ids = tf.gather(closest_ids, tf.where(tf_mask))
            closest_ids = tf.reshape(closest_ids, [-1]) # Ensure flatten 1D shape
            # Remove duplicate ids (possible when multiple representors are used)
            closest_ids = tf.unique(closest_ids).y

        with timeit("Get closest idx", active=debug):
            closest_idx = static_get_idx(closest_ids, self.ids)
            # If closest_idx is an int expand it to a tensor of shape [1]
            if closest_idx.shape == ():
                closest_idx = tf.expand_dims(closest_idx, 0)

        with timeit("Obtain closest reps", active=debug):
            # Compute distance from the remaining ids
            closest_rep = tf.gather(self.reps, closest_idx)

        with timeit("Compute closest distance", active=debug):
            closest_dst = compute_dst(rep, closest_rep)

        # Return the c_id of the closest cluster with the minimum distance
        with timeit("Get min distance", active=debug):
            min_idx = tf.argmin(closest_dst)
        return closest_ids[min_idx], closest_dst[min_idx]

    def compute_closest(self, tree: tf_rtree) -> None:
        """compute_closest.

        Compute the closest cluster for each cluster in the queue.

        Parameters
        ----------
        tree : tf_rtree
            tree with all the clusters

        Returns
        -------
        None
        """
        self.__closest, self.__distances = tf.map_fn(
            lambda x: self.best_closest(x[0], x[1], tree),
            elems=(self.ids, self.reps),
            dtype=(tf.int32, tf.float32),
            fn_output_signature=(tf.TensorSpec(shape=[], dtype=tf.int32),
                                 tf.TensorSpec(shape=[], dtype=tf.float32)))


    def update_closest(self, new_c_id: int,
                       old_c_id: int,
                       tree: tf_rtree) -> None:
        """update_closest.

        Function to update the closest data structure.
        The computation procceds in 2 steps:
            1) Identify all the clusters which have old_c_id as closest cluster
               and compute the new closest cluster for those object.
               the cluster with new_c_id falls into this category
            2) Compute the distance between each cluster and new_c_id and
               update the closest data structure only if the new distance is
               lower than the current one

        Assumption:
            - The closest cluster of new_c_id is currently old_c_id

        Parameters
        ----------
        new_c_id : int
            id of the new cluster
        old_c_id : int
            id of the old cluster
        tree : tf_rtree
            tree with all the clusters

        Returns
        -------
        None
        """
        # Init, identify the index of the new clsuter:
        new_idx = self.get_idx(new_c_id)

        # Step 1, identify all the clusters which have old_c_id as closest
        # cluster and compute the new closest cluster for those object
        s1_mask = self.closest == old_c_id
        s1_idx = tf.where(s1_mask)
        s1_c_id = tf.gather(self.ids, s1_idx)
        s1_c_reps = tf.gather(self.reps, s1_idx)

        s1_closest, s1_distances = tf.map_fn(
            lambda x: self.best_closest(x[0], x[1], tree),
            elems=(s1_c_id, s1_c_reps),
            dtype=(tf.int32, tf.float32),
            fn_output_signature=(tf.TensorSpec(shape=[], dtype=tf.int32),
                                 tf.TensorSpec(shape=[], dtype=tf.float32)))

        # Swap closest[s1_idx] with the new s1_closest and also distances
        self.__closest = tf.tensor_scatter_nd_update(self.closest,
                                                     s1_idx,
                                                     s1_closest)
        self.__distances = tf.tensor_scatter_nd_update(self.distances,
                                                       s1_idx,
                                                       s1_distances)

        # Step 2, compute the distance between each cluster and new_c_id
        # if the new distance is lower than the previous one update the closest
        # and distance with the new one
        rep = self.reps[new_idx]

        new_dst = compute_dst(rep, self.reps)

        # Get the mask of clusters which have a lower new_distance in comparison
        # to the current one, and also which are not the new_c_id
        new_mask = tf.logical_and(new_dst < self.distances, self.ids != new_c_id)

        self.__closest = tf.where(new_mask, new_c_id, self.closest)
        self.__distances = tf.where(new_mask, new_dst, self.distances)

    def apply_rep_compression(self) -> None:
        """apply_rep_compression.

        Apply the compression factor to the representors.
        """
        self.__r_reps += self.compression_factor * (self.mean - self.reps)

    def compute_representors(self, c_id:int) -> None:
        """recompute_representors.

        Recompute the representors for the given cluster.

        Parameters
        ----------
        c_id : int
            id of the cluster to recompute the representors

        Returns
        -------
        None
        """
        idx = self.get_idx(c_id)
        c_points = self.clusters[idx]

        time_flag = False
        if c_points.shape[0] > self.n_rep:
            with timeit("Representors selection algorithm", active=time_flag):
                c_rep = static_select_representors(c_points, tf.constant(self.n_rep))
        else:
            c_rep = c_points

        with timeit("Representors compression", active=time_flag):
            c_rep += self.compression_factor * (tf.reduce_mean(c_points, axis=0) - c_rep)

        with timeit("Representors update", active=time_flag):
            self.__r_reps = ragged_tf_swap(self.reps, c_rep, idx, tf.constant(1, dtype=tf.int64))

    def merge(self, u_id: int, v_id: int) -> None:
        """merge.

        Merge two clusters given their indexes.

        Parameters
        ----------
        u_idx : int
            index of the first cluster to merge
        v_idx : int
            index of the second cluster to merge

        Returns
        -------
        None
        """
        if u_id == v_id:
            raise ValueError("Cannot merge the same cluster")
        __u_idx = 0 if u_id < v_id else 1

        idx = static_get_idx(tf.stack([u_id, v_id]), self.ids)

        merge_points = static_merge_clst(self.clusters, idx)

        self.__r_clusters = ragged_tf_swap(self.clusters, merge_points,
                                           idx[__u_idx], tf.constant(1, dtype=tf.int64))

    def __delitem__(self, c_id: int) -> None:
        """__delitem__.

        Delete a cluster given the id.

        Parameters
        ----------
        c_id : int
            id of the cluster to delete

        Returns
        -------
        None
        """
        idx = tf.constant(self.get_idx(c_id))
        delta = tf.constant(1, dtype=tf.int64)
        self.__r_clusters = ragged_tf_concat(self.clusters, idx, delta)
        self.__r_reps = ragged_tf_concat(self.reps, idx, delta)
        self.__ids = tf.cast(tf_concat(tf.cast(self.ids, tf.float32), idx, delta), tf.int32)
        self.__closest = tf.cast(tf_concat(tf.cast(self.closest, tf.float32), idx, delta), tf.int32)
        self.__distances = tf_concat(self.distances, idx, delta)

    @property
    def closest(self) -> tf.Tensor:
        """closest.

        Return the tensor with all the closest clusters.

        Returns
        -------
        tf.Tensor
            tensor with all the closest clusters
        """
        return self.__closest

    @property
    def distances(self) -> tf.Tensor:
        """distances.

        Return the tensor with all the distances from the closest clusters.

        Returns
        -------
        tf.Tensor
            tensor with all the distances from the closest clusters
        """
        return self.__distances

    @property
    def mean(self) -> tf.Tensor:
        """mean.

        Return the mean of all the points.

        Returns
        -------
        tf.Tensor
            mean of all the points
        """
        return tf.reduce_mean(self.__r_clusters, axis=0)

    @property
    def clusters(self) -> tf.RaggedTensor:
        """clusters.

        Return the ragged tensor with all the clusters.

        Returns
        -------
        tf.RaggedTensor
            ragged tensor with all the clusters
        """
        return self.__r_clusters

    @property
    def reps(self) -> tf.RaggedTensor:
        """reps.

        Return the ragged tensor with all the representors.

        Returns
        -------
        tf.RaggedTensor
            ragged tensor with all the representors for each cluster
        """
        return self.__r_reps

    @property
    def ids(self) -> tf.Tensor:
        """ids.

        Return the tensor with all the ids.

        Returns
        -------
        tf.Tensor
            tensor with all the ids for each cluster
        """
        return self.__ids

    def __setitem__(self, *args, **kwargs) -> None:
        raise NotImplementedError("This method is not implemented and a cluster cannot be set")

    def __getitem__(self, id: int) -> Tuple[float, tf.Tensor, tf.Tensor, float, float]:
        """__getitem__.

        returns all the information about a cluster given
        a specific index.

        Parameters
        ----------
        idx : int
            index of the cluster to return

        Returns
        -------
        Tuple[float, tf.Tensor, tf.Tensor, float, float]
            Returns the cluster id, the cluster itself, the representors,
            the closest cluster id and the distance from the closest cluster
        """
        idx = self.get_idx(id)
        return self.ids[idx], self.clusters[idx], self.reps[idx], \
                self.__closest[idx], self.__distances[idx]

    def __iter__(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """__iter__.

        Returns an iterator with all the information about the clusters.

        Returns
        -------
        Tuple[float, tf.Tensor, tf.Tensor, float, float]
            Returns the cluster id, the cluster itself, the representors,
            the closest cluster id and the distance from the closest cluster
        """
        for i in range(self.ids.shape[0]):
            yield self[self.ids[i]]

    @property
    def size(self) -> int:
        """size.

        Return the number of clusters with more than 0 points

        Returns
        -------
        int
            size of the queue
        """
        return self.clusters.shape[0]
