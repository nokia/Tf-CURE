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

from pyclustering.cluster import cluster_visualizer;
from pyclustering.cluster.cure import cure;
from pyclustering.utils import read_sample;
from pyclustering.samples.definitions import FCPS_SAMPLES;

from tf_cure.src.CURELayer.tf_CURE import TfCURE
from tf_cure.src.CURELayer.tf_cluster import TfCluster
from tf_cure.src import TfCURE

class TestTfCURE():
    """TestTfCURE.

    Test class for the CURE algorithm implemented through TF
	"""

    def test_clustering(self) -> None:
        input_data = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)
        # Allocate three clusters.
        cure_instance = cure(input_data, 3, ccore=False)
        cure_instance.process()
        clusters = cure_instance.get_clusters()
        with tf.device('/GPU:0'):
            input_data_tf = tf.convert_to_tensor(input_data)
            tf_cure_instance = TfCURE(input_data_tf, 3)
            tf_cure_instance.process()

        pyclst_rep = np.array(cure_instance.get_representors())
        pyclst_rep = np.round(pyclst_rep, decimals=4)
        # tf_clusters = tf_cure_instance.clusters
        # assert tf_clusters is not None
        #
        # np.testing.assert_equal(clusters, tf_clusters.numpy())
        tf_reps = tf_cure_instance.clusters.reps.numpy()
        tf_reps = np.round(tf_reps, decimals=4)
        # Assert the two array have the same shape
        assert pyclst_rep.shape == tf_reps.shape
        # Assert that all elements from pyclst_rep are in tf_reps independently from the order
        assert all([any([np.allclose(x, y) for y in tf_reps]) for x in pyclst_rep])
        assert False

