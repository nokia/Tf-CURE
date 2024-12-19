# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2024 Mattia Milani <mattia.milani@nokia.com>

"""
tree module
===========

This module is used to manage a rtree easily, masking the complexity of
introducing tensors in the tree structure.
"""

from typing import Tuple

import numpy as np
import tensorflow as tf
from rtree import index

from tf_cure.src.contextManagers.contextManagers import timeit_cnt

class tf_rtree:
    """tf_rtree.

    Class to manage a rtree with tensors
	"""


    def __init__(self, *args, **kwargs):
        """__init__.

		Parameters
		----------
		"""
        self.__prop = index.Property(**kwargs)
        self.__tree = index.Index(properties=self.__prop)

        def take_n_nearest(rep, n) -> np.array:
            return np.array(list(self.tree.nearest(rep, n, objects=False)))

        def insert(c_id, rep) -> None:
            self.tree.insert(c_id, rep)

        def delete(c_id, rep) -> None:
            self.tree.delete(c_id, rep)

        self.v_delete = np.vectorize(delete, signature='(),(n)->()')
        self.vinsert = np.vectorize(insert, signature='(),(n)->()')
        self.v_take_n_nearest = np.vectorize(take_n_nearest, signature='(n),()->(m)')

    def nearest(self, rep: tf.Tensor, n: int = 1) -> tf.Tensor:
        """nearest.

        Get the nearest n clusters to the representation rep

        Parameters
        ----------
        rep : tf.Tensor
            The representation of the cluster
        n : int
            The number of clusters to return
        **kwargs : Any
            kwargs to pass to the nearest method

        Returns
        -------
        tf.Tensor
            Tensor with 2 dimensions, for each rep n closest clusters are piked
            shape (n_rep, n)
        """
        # Transform the rep in a flat tensor
        flat_rep = rep.numpy()
        n_close = np.repeat(n, flat_rep.shape[0])
        # Get the n nearest clusters using a vectorize version to operate
        # on all rep at the same time
        all_nearest = self.v_take_n_nearest(flat_rep, n_close)
        return tf.convert_to_tensor(all_nearest, dtype=tf.int32)

    def add_clusters(self, ids: tf.Tensor, reps: tf.RaggedTensor) -> None:
        """add_clusters.

        Insert all the clusters in the rtree

        Parameters
        ----------
        ids : tf.Tensor
            The ids of the clusters
        reps : tf.RaggedTensor
            The representations of the clusters
        """
        # Transform ids and repeat each id for each rep of the same cluster
        ids = tf.repeat(ids, reps.row_lengths()).numpy()
        # Transform reps in a flat tensor
        flat_reps = reps.flat_values.numpy()
        self.vinsert(ids, flat_reps)

    def add_cluster(self, c_id: int, reps: tf.Tensor) -> None:
        """add_cluster.

        Insert a cluster in the rtree

        Parameters
        ----------
        c_id : int
            The id of the cluster
        reps : tf.Tensor
            The representation of the cluster
        """
        # Transform reps in a flat tensor
        flat_reps = reps.numpy()
        c_ids = np.repeat(c_id, flat_reps.shape[0])
        self.vinsert(c_ids, flat_reps)

    def remove_cluster(self, c_id: int,
                       reps: tf.Tensor) -> None:
        """remove_cluster.

        Remove a cluster from the rtree

        Parameters
        ----------
        c_id : int
            The id of the cluster to remove
        """
        # Transform reps in a flat tensor
        flat_reps = reps.numpy()
        c_ids = np.repeat(c_id, flat_reps.shape[0])
        self.v_delete(c_ids, flat_reps)

    @property
    def tree(self) -> index.Index:
        """tree.

        Returns
        -------
        index.Index
            The current tree
        """
        return self.__tree

