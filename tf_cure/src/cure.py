# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause
#
# Contact: Mattia Milani (Nokia) <mattia.milani@nokia.com>

"""
cure module
===========

Module built to comptue the CURE clustering algorithm on top
of tensorflow vector, taking advantage of GPU acceleration when possible.
"""

from typing import Union

import numpy as np
import tensorflow as tf

from tf_cure.src.contextManagers.contextManagers import timeit_cnt as timeit
from tf_cure.src.contextManagers.contextManagers import TimeitContext as TiT
from .tf_cluster import tf_clusters
from .tree import tf_rtree

class TfCURE:
    """TfCURE.

    This class is the entry point to execute the CURE algorithm using tensorflow
	"""


    def __init__(self, data: tf.Tensor,
                 n_clusters: int = 2,
                 n_rep: int = 5,
                 compression_factor: float = 0.5):
        """__init__.
        Initialization method for the TfCURE object
        The dimension of the objects is obtained from the last dimension
        of the data tensor.

        Parameters
        ----------
        data : tf.Tensor
            The data to cluster
        n_clusters : int
            The number of clusters to obtain
        n_rep : int
            The number of representatives to use
        compression_factor : float
            The compression factor to apply for representatives
        """
        self.__clusters: Union[tf.Tensor, None] = None
        self.__tree: Union[tf_rtree, None] = None

        self.__data: tf.Tensor = data
        self.__dimension: int = self.data.shape[-1]
        self.__n_clusters: int = n_clusters
        self.__n_rep: int = n_rep
        self.__compression_factor: float = compression_factor
        self.timeit_flag: bool = False

        self.__validate_arguments()

    def __validate_arguments(self) -> None:
        assert not tf.equal(tf.size(self.data), 0), \
                f"The size of data points cannot be 0, shape: {self.data.shape}, size: {tf.size(self.data)}"
        assert self.n_clusters > 0, \
                f"The number of clusters must be higher than 0, {self.n_clusters} has been provided"
        assert self.n_rep > 0, \
                f"The number of representatives must be higher than 0, {self.n_rep} has been provided"
        assert self.compression_factor > 0.0, \
                f"The compression factor must be higher than 0, {self.compression_factor} has been provided"

    def process(self) -> None:
        self.init_rtree()
        self.init_queue()
        self.iterate()

    def init_rtree(self, **kwargs) -> None:
        self.tree = tf_rtree(leaf_capacity=200,
                             fill_factor=0.8,
                             dimension=self.dimension)

    def init_queue(self) -> None:
        self.clusters = tf_clusters(self.data, n_rep=self.n_rep,
                                    compression_factor=self.compression_factor)
        with timeit("Inserting clusters in the tree", active=self.timeit_flag):
            self.tree.add_clusters(self.clusters.ids, self.clusters.reps)
        with timeit("Computing closest cluster", active=self.timeit_flag):
            self.clusters.compute_closest(self.tree)


    def iterate(self) -> None:
        """iterate."""
        cycle_timer = TiT()
        merge_timer = TiT()
        clean_timer = TiT()
        update_timer = TiT()
        compute_closest_timer = TiT()
        compute_rep_timer = TiT()
        add_to_tree_timer = TiT()
        v_remove_timer = TiT()
        v_delete_timer = TiT()
        u_remove_timer = TiT()
        while self.clusters.size > self.n_clusters:
            with cycle_timer():
                # Get the cluster closest to it's neighbor
                min_dst_idx = tf.argmin(self.clusters.distances, output_type=tf.int32)
                u_id = self.clusters.ids[min_dst_idx]
                u_dst = self.clusters.distances[min_dst_idx]
                v_id = self.clusters.closest[min_dst_idx]

                #### Merge the clusters ####
                with merge_timer():
                    # Add v points to u
                    self.clusters.merge(u_id, v_id)

                #### Clean data structures ####
                with clean_timer():
                    # Remove v rep from the tree
                    with v_remove_timer():
                        v_idx = self.clusters.get_idx(v_id)
                        self.tree.remove_cluster(v_id, self.clusters.reps[v_idx])
                    # Empty v points
                    with v_delete_timer():
                        del self.clusters[v_id]
                    # Remove u rep from the tree
                    with u_remove_timer():
                        u_idx = self.clusters.get_idx(u_id)
                        self.tree.remove_cluster(u_id, self.clusters.reps[u_idx])

                #### Update the remaining clusters ####
                with update_timer():
                    # Recocompute u representors
                    with compute_rep_timer():
                        self.clusters.compute_representors(u_id)
                    # Add the new u rep to the tree
                    with add_to_tree_timer():
                        self.tree.add_cluster(u_id, self.clusters.reps[u_idx])

                # Recompute the closest cluster to u
                # Recompute the distances of the clusters BUT ONLY in respect
                # to the new cluster, if the current dst is < than the new one
                # then do not update.
                with compute_closest_timer():
                    self.clusters.update_closest(u_id, v_id, self.tree)

        print(f"[{cycle_timer.n_runs}] Average cycle execution time: {cycle_timer.mean*1000:.4f}ms")
        print(f"[{merge_timer.n_runs}] Average merge execution time: {merge_timer.mean*1000:.4f}ms")
        print(f"[{clean_timer.n_runs}] Average clean execution time: {clean_timer.mean*1000:.4f}ms")
        print(f"\t[{v_remove_timer.n_runs}] Average v_remove execution time: {v_remove_timer.mean*1000:.4f}ms")
        print(f"\t[{v_delete_timer.n_runs}] Average v_delete execution time: {v_delete_timer.mean*1000:.4f}ms")
        print(f"\t[{u_remove_timer.n_runs}] Average u_remove execution time: {u_remove_timer.mean*1000:.4f}ms")
        print(f"[{update_timer.n_runs}] Average update execution time: {update_timer.mean*1000:.4f}ms")
        print(f"\t[{compute_rep_timer.n_runs}] Average compute_rep execution time: {compute_rep_timer.mean*1000:.4f}ms")
        print(f"\t[{add_to_tree_timer.n_runs}] Average add_to_tree execution time: {add_to_tree_timer.mean*1000:.4f}ms")
        print(f"[{compute_closest_timer.n_runs}] Average update execution time: {compute_closest_timer.mean*1000:.4f}ms")

    @property
    def data(self) -> tf.Tensor:
        """data.

        Returns the data tensor

        Parameters
        ----------

        Returns
        -------
        tf.Tensor
            The data tensor
        """
        return self.__data

    @property
    def dimension(self) -> int:
        """dimension.

        Returns the number of dimensions for the points in the data tensor

        Parameters
        ----------

        Returns
        -------
        int
            The number of dimensions
        """
        return self.__dimension

    @property
    def n_clusters(self) -> int:
        """n_clusters.

        Returns the number of clusters to obtain

        Parameters
        ----------

        Returns
        -------
        int
            The number of clusters to obtain
        """
        return self.__n_clusters

    @property
    def n_rep(self) -> int:
        """n_rep.

        Returns the number of representatives to use

        Parameters
        ----------

        Returns
        -------
        int
            The number of representatives to use
        """
        return self.__n_rep

    @property
    def compression_factor(self) -> float:
        """compression_factor.

        Returns the compression factor to apply for representatives

        Parameters
        ----------

        Returns
        -------
        float
            The compression factor to apply for representatives
        """
        return self.__compression_factor

    @property
    def tree(self) -> Union[tf_rtree, None]:
        """tree.

        Returns the rtree object

        Parameters
        ----------

        Returns
        -------
        tf_rtree
            The rtree object
        """
        return self.__tree

    @tree.setter
    def tree(self, tree: tf_rtree) -> None:
        """tree.

        Set the rtree object

        Parameters
        ----------
        tree : tf_rtree
            The rtree object

        Returns
        -------
        """
        self.__tree = tree

    @property
    def clusters(self) -> Union[tf.Tensor, None]:
        """clusters.

        Returns current avilable clusters

        Parameters
        ----------

        Returns
        -------
        tf.Tensor
            The clusters
        """
        return self.__clusters

    @clusters.setter
    def clusters(self, clusters: tf_clusters) -> None:
        """clusters.

        Set the current available clusters

        Parameters
        ----------
        clusters : TfClusters
            The clusters

        Returns
        -------
        """
        self.__clusters = clusters
