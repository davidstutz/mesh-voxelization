import os
import h5py
import argparse
import numpy as np
from skimage import morphology

def write_hdf5(file, tensor, key = 'tensor'):
    """
    Write a simple tensor, i.e. numpy array ,to HDF5.

    :param file: path to file to write
    :type file: str
    :param tensor: tensor to write
    :type tensor: numpy.ndarray
    :param key: key to use for tensor
    :type key: str
    """

    assert type(tensor) == np.ndarray, 'expects numpy.ndarray'

    h5f = h5py.File(file, 'w')

    chunks = list(tensor.shape)
    if len(chunks) > 2:
        chunks[2] = 1
        if len(chunks) > 3:
            chunks[3] = 1
            if len(chunks) > 4:
                chunks[4] = 1

    h5f.create_dataset(key, data = tensor, chunks = tuple(chunks), compression = 'gzip')
    h5f.close()

def read_hdf5(file, key = 'tensor'):
    """
    Read a tensor, i.e. numpy array, from HDF5.

    :param file: path to file to read
    :type file: str
    :param key: key to read
    :type key: str
    :return: tensor
    :rtype: numpy.ndarray
    """

    assert os.path.exists(file), 'file %s not found' % file

    h5f = h5py.File(file, 'r')

    assert key in h5f.keys(), 'key %s not found in file %s' % (key, file)
    tensor = h5f[key][()]
    h5f.close()

    return tensor

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fill occupancy grids.')
    parser.add_argument('input', type=str, help='The input HDF5 file.')
    parser.add_argument('output', type=str, help='The output HDF5 file.')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print('Input file does not exist.')
        exit(1)

    occupancy = read_hdf5(args.input)
    if len(occupancy.shape) < 4:
        occupancy = np.expand_dims(occupancy, axis=0)

    filled = np.zeros(occupancy.shape)

    for n in range(occupancy.shape[0]):
        labels, num_labels = morphology.label(occupancy[n], background=1, connectivity=1, return_num=True)
        outside_label = labels[0][0][0]

        filled[n][labels != outside_label] = 1
        # filled[n][labels == outside_label] = 0

        print('Filled %d.' % n)

    write_hdf5(args.output, filled)
    print('Wrote %s.' % args.output)