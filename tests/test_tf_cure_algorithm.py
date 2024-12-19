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
# Copyright (C) 2020 Mattia Milani <mattia.milani@nokia.com>

"""
CURE algorithm test module
==========================

Use this module to test the CURE algorithm module
"""

from typing import List
import numpy as np
import tensorflow as tf
import pytest

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.cure import cure
from rtree import index

from SNN2.src.model.layers.CURELayer.tf_CURE import TfCURE
from SNN2.src.model.layers.CURELayer.tf_cluster import TfCluster

def tf_round(t, decimals=0):
    mul = tf.constant(10**decimals, dtype=t.dtype)
    return tf.round(t * mul)/mul

class TestTfCluster():

    def test_closest_by_pointer(self) -> None:
        c1 = tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
        c2 = tf.convert_to_tensor([[2.0, 3.0], [4.0, 5.0]])
        c1 = TfCluster(1, c1)
        c2 = TfCluster(2, c2)
        c3 = tf.convert_to_tensor([[2.0, 3.0], [8.0, 5.0]])
        c3 = TfCluster(3, c3)
        c1.closest = c2
        c2.closest = c3
        assert c1.closest.closest == c3

    def test_cluster_dst(self) -> None:
        c1 = tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
        c2 = tf.convert_to_tensor([[2.0, 3.0], [4.0, 5.0]])
        c1 = TfCluster(1, c1)
        c2 = TfCluster(2, c2)
        assert c1.dst(c2) == 0.0

    def test_unbalanced_dst(self) -> None:
        c1 = tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
        c2 = tf.convert_to_tensor([[2.0, 3.0], [4.0, 5.0], [5.0, 6.0]])
        c3 = tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [6.0, 7.0]])
        c1 = TfCluster(1, c1)
        c2 = TfCluster(2, c2)
        c3 = TfCluster(3, c3)
        dc1c2 = float(c1.dst(c2).numpy())
        dc1c3 = float(c1.dst(c3).numpy())
        dc2c3 = float(c2.dst(c3).numpy())
        assert f"{dc1c2:.8f}" == "0.47140458"
        assert f"{dc1c3:.8f}" == "0.17677669"
        assert f"{dc2c3:.8f}" == "0.05892568"

    def test_merge(self) -> None:
        c1 = tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0]])
        c2 = tf.convert_to_tensor([[2.0, 3.0], [4.0, 5.0]])
        c1 = TfCluster(1, c1)
        c2 = TfCluster(2, c2)
        c3 = TfCluster.merge(c1, c2)
        tf.debugging.assert_equal(c3.points, tf.concat([c1.points, c2.points], 0))
        assert c3.id == 1
        print(c3.mean)
        tf.debugging.assert_equal(c3.mean, tf.convert_to_tensor([2.5, 3.5]))
        tf.debugging.assert_equal(c3.rep, tf.convert_to_tensor([[1.75, 2.75],
                                                                [2.75, 3.75],
                                                                [2.25, 3.25],
                                                                [3.25, 4.25]]))

    def test_merge_repChoice(self) -> None:
        c1 = tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [4.0, 5.0], [6.0, 7.0]])
        c2 = tf.convert_to_tensor([[2.0, 3.0], [4.0, 5.0], [5.0, 6.0], [7.0, 8.0]])
        c1 = TfCluster(1, c1)
        c2 = TfCluster(2, c2)
        c3 = TfCluster.merge(c1, c2)
        tf.debugging.assert_equal(c3.rep, tf.convert_to_tensor([[3.125, 4.125],
                                                                [4.875, 5.875],
                                                                [3.875, 4.875],
                                                                [4.375, 5.375],
                                                                [3.625, 4.625]]))

    def test_add_to_tree(self) -> None:
        c1 = tf.convert_to_tensor([[1.0, 2.0]])
        c2 = tf.convert_to_tensor([[2.0, 3.0]])
        idx = index.Index()
        c1 = TfCluster(1, c1, tree=idx)
        c2 = TfCluster(2, c2, tree=idx)
        assert list(idx.intersection((0.9, 1.9, 2.1, 3.1))) == [1, 2]

    def test_remove_from_tree(self) -> None:
        c1 = tf.convert_to_tensor([[1.0, 2.0]])
        c2 = tf.convert_to_tensor([[2.0, 3.0]])
        idx = index.Index()
        c1 = TfCluster(1, c1, tree=idx)
        c2 = TfCluster(2, c2, tree=idx)
        assert list(idx.intersection((0.9, 1.9, 2.1, 3.1))) == [1, 2]
        c1.remove_from_tree()
        assert list(idx.intersection((0.9, 1.9, 2.1, 3.1))) == [2]

    def test_compute_closest(self) -> None:
        c1 = tf.convert_to_tensor([[1.0, 2.0], [3.0, 4.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0]])
        c2 = tf.convert_to_tensor([[2.0, 3.0], [4.0, 5.0], [5.0, 6.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0]])
        idx = index.Index()
        c1 = TfCluster(1, c1, tree=idx)
        c2 = TfCluster(2, c2, tree=idx)
        c1.nearest_from_tree(idx)
        assert False


class TestTfCURE():
    """TestTfCURE.

    Test class for the CURE algorithm implemented through TF
	"""

    def test_default(self) -> None:
        assert True

    def test_process(self, tf_sample_LSUN) -> None:
        input_data = tf_sample_LSUN
        tf_cure_instance = TfCURE(input_data, 3)
        tf_cure_instance.process()
        assert False

    def test_create_tree(self, tf_sample_LSUN) -> None:
        input_data = tf_sample_LSUN
        tf_cure_instance = TfCURE(input_data, 3)
        tf_cure_instance.create_rtree()
        assert False
        # assert tf_cure_instance.tree.query([0.0, 1.0], k=[1])[1][0] == 39

