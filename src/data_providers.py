# -*- coding: utf-8 -*-
"""Data providers
Originally proprosed by 'Pawel Swietojanski', 'Steve Renals', 'Matt Graham'
which can be found in: https://github.com/CSTR-Edinburgh/mlpractical/blob/mlp2017-8/master/mlp/data_providers.py

This module provides classes for loading datasets and iterating over batches of
data points.
"""

import pickle
import gzip
import numpy as np
import os
import pandas as pd
from constant import VALID_MALL_ID, DEFAULT_SEED, SHOP_IN_MALL, TRAIN_DATA_DIR


class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.
        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        """Number of data points to include in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def max_num_batches(self):
        """Maximum number of batches to iterate over in an epoch."""
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        """Updates number of batches to iterate over."""
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.
        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def new_epoch(self):
        """Starts a new epoch (pass through data), possibly shuffling first."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def __next__(self):
        return self.next()

    def reset(self):
        """Resets the provider to the initial state."""
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        """Randomly shuffles order of data."""
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch
    

class WIFIDataProvider(DataProvider):
    """Data provider for WiFi project data information."""

    def __init__(self, mall_id, which_set='train', is_eval=False, batch_size=128, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new WiFi data provider object.
        Args:
            mall_id: The mall_id that is used to locate the sub-data sets.
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the WiFi data this object should provide.
            is_eval: whether the datasets is for evaluation
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        valid_mall_id = VALID_MALL_ID

        assert mall_id in valid_mall_id, (
                               'Expected mall to be in {0} '
                               'Got {1}'.format(valid_mall_id, mall_id)
                           )
        if not is_eval:
            assert which_set in ['train', 'valid'], (
                'Expected which_set to be either train, valid, eval. '
                'Got {0}'.format(which_set)
            )
        else:
            assert which_set == 'eval', (
                'Expected which_set to be eval. '
                'Got {0}'.format(which_set)
            )
        self.which_set = which_set
        self.mall_id = mall_id
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # the csv file should put int ./data/ and the source file should be in ./src/
        # when using the source file, the current directory should be in ./src/ 
        # otherwise, please change the data_path 
        data_path = os.path.join(TRAIN_DATA_DIR, '{0}-{1}.csv'.format(self.mall_id, self.which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )

        # load data from compressed csv file
        loaded = pd.read_csv(data_path, delimiter = ',')

        # get number of classes
        self.shop_list = SHOP_IN_MALL[mall_id]
        self.num_classes = len(self.shop_list)
        
        inputs = loaded.iloc[:, 8:].values # the inputs is just wifi information and they are in the columns of [:, 8:]
        self.row_id = None
        if not is_eval:
            targets = loaded.loc[:, 'shop_id'].values
            self.one_of_k_targets = self.to_one_of_k(targets)
        else:
            targets = np.zeros([inputs.shape[0]])
            self.one_of_k_targets = np.zeros([inputs.shape[0], self.num_classes])
            self.row_id = loaded.loc[:, 'row_id'].values
        inputs = inputs.astype(np.float32)
        # pass the loaded data to the parent class __init__
        super(WIFIDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(WIFIDataProvider, self).next()
        return inputs_batch, self.to_one_of_k(targets_batch)

    def to_one_of_k(self, targets):
        """Converts integer coded class target to 1 of K coded targets.
        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,).
        Returns:
            Array of 1 of K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """        
        
        one_of_k_targets = np.zeros((targets.shape[0], self.num_classes))
        for i in range(targets.shape[0]):
            one_of_k_targets[i, self.shop_list.index(targets[i])] = 1
        return one_of_k_targets

class WIFIDataProviderLatLongAdded(DataProvider):
    """Data provider for WiFi project data information."""

    def __init__(self, mall_id, which_set='train', is_eval=False, batch_size=128, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new WiFi data provider object.
        Args:
            mall_id: The mall_id that is used to locate the sub-data sets.
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the WiFi data this object should provide.
            is_eval: whether the datasets is for evaluation
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        valid_mall_id = VALID_MALL_ID

        assert mall_id in valid_mall_id, (
                               'Expected mall to be in {0} '
                               'Got {1}'.format(valid_mall_id, mall_id)
                           )
        if not is_eval:
            assert which_set in ['train', 'valid'], (
                'Expected which_set to be either train, valid, eval. '
                'Got {0}'.format(which_set)
            )
        else:
            assert which_set == 'eval', (
                'Expected which_set to be eval. '
                'Got {0}'.format(which_set)
            )
        self.which_set = which_set
        self.mall_id = mall_id
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # the csv file should put int ./data/ and the source file should be in ./src/
        # when using the source file, the current directory should be in ./src/ 
        # otherwise, please change the data_path 
        data_path = os.path.join(TRAIN_DATA_DIR, '{0}-{1}.csv'.format(self.mall_id, self.which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )

        # load data from compressed csv file
        loaded = pd.read_csv(data_path, delimiter = ',')

        # get number of classes
        self.shop_list = SHOP_IN_MALL[mall_id]
        self.num_classes = len(self.shop_list)


        feature_index = [i for i in range(8, loaded.shape[1])] # Bssids features
        for i in [5, 4]:  # Add longitude and latitude features
            feature_index.insert(0, i)
        # the inputs containts bssids strength with longitude and latitude
        inputs = loaded.iloc[:, feature_index].values 
        self.row_id = None
        if not is_eval:
            targets = loaded.loc[:, 'shop_id'].values
            self.one_of_k_targets = self.to_one_of_k(targets)
        else:
            targets = np.zeros([inputs.shape[0]])
            self.one_of_k_targets = np.zeros([inputs.shape[0], self.num_classes])
            self.row_id = loaded.loc[:, 'row_id'].values
        inputs = inputs.astype(np.float32)
        # pass the loaded data to the parent class __init__
        super(WIFIDataProviderLatLongAdded, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(WIFIDataProviderLatLongAdded, self).next()
        return inputs_batch, self.to_one_of_k(targets_batch)

    def to_one_of_k(self, targets):
        """Converts integer coded class target to 1 of K coded targets.
        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,).
        Returns:
            Array of 1 of K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """        
        
        one_of_k_targets = np.zeros((targets.shape[0], self.num_classes))
        for i in range(targets.shape[0]):
            one_of_k_targets[i, self.shop_list.index(targets[i])] = 1
        return one_of_k_targets