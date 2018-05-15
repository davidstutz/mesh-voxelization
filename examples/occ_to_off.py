import os
import h5py
import argparse
import numpy as np

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

def write_off(file, vertices, faces):
    """
    Writes the given vertices and faces to OFF.

    :param vertices: vertices as tuples of (x, y, z) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        fp.write('OFF\n')
        fp.write(str(num_vertices) + ' ' + str(num_faces) + ' 0\n')

        for vertex in vertices:
            assert len(vertex) == 3, 'invalid vertex with %d dimensions found (%s)' % (len(vertex), file)
            fp.write(str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')

        for face in faces:
            assert face[0] == 3, 'only triangular faces supported (%s)' % file
            assert len(face) == 4, 'faces need to have 3 vertices, but found %d (%s)' % (len(face), file)

            for i in range(len(face)):
                assert face[i] >= 0 and face[i] < num_vertices, 'invalid vertex index %d (of %d vertices) (%s)' % (face[i], num_vertices, file)

                fp.write(str(face[i]))
                if i < len(face) - 1:
                    fp.write(' ')

            fp.write('\n')

        # add empty line to be sure
        fp.write('\n')

class Mesh:
    """
    Represents a mesh.
    """

    def __init__(self, vertices = [[]], faces = [[]]):
        """
        Construct a mesh from vertices and faces.

        :param vertices: list of vertices, or numpy array
        :type vertices: [[float]] or numpy.ndarray
        :param faces: list of faces or numpy array, i.e. the indices of the corresponding vertices per triangular face
        :type faces: [[int]] fo rnumpy.ndarray
        """

        self.vertices = np.array(vertices, dtype = float)
        """ (numpy.ndarray) Vertices. """

        self.faces = np.array(faces, dtype = int)
        """ (numpy.ndarray) Faces. """

        assert self.vertices.shape[1] == 3
        assert self.faces.shape[1] == 3

    def to_off(self, filepath):
        """
        Write mesh to OFF.

        :param filepath: path to write file to
        :type filepath: str
        """

        faces = np.ones((self.faces.shape[0], 4), dtype = int)*3
        faces[:, 1:4] = self.faces[:, :]

        write_off(filepath, self.vertices.tolist(), faces.tolist())

    @staticmethod
    def from_volume(volume):
        """
        Create a mesh from a voxel grid/volume.

        :param volume: volume
        :type volume: numpy.ndarray
        :return: mesh
        :rtype: Mesh
        """

        vertices = []
        faces = []

        xx, yy, zz = np.where(volume > 0.5)

        for i in range(len(xx)):
            v000 = (yy[i], xx[i], zz[i])                # 0
            v001 = (yy[i], xx[i], zz[i] + 1)            # 1
            v010 = (yy[i], xx[i] + 1, zz[i])            # 2
            v011 = (yy[i], xx[i] + 1, zz[i] + 1)        # 3
            v100 = (yy[i] + 1, xx[i], zz[i])            # 4
            v101 = (yy[i] + 1, xx[i], zz[i] + 1)        # 5
            v110 = (yy[i] + 1, xx[i] + 1, zz[i])        # 6
            v111 = (yy[i] + 1, xx[i] + 1, zz[i] + 1)    # 7

            n = len(vertices)
            f1 = [n + 0, n + 2, n + 4]
            f2 = [n + 4, n + 2, n + 6]
            f3 = [n + 1, n + 3, n + 5]
            f4 = [n + 5, n + 3, n + 7]
            f5 = [n + 0, n + 1, n + 2]
            f6 = [n + 1, n + 3, n + 2]
            f7 = [n + 4, n + 5, n + 7]
            f8 = [n + 4, n + 7, n + 6]
            f9 = [n + 4, n + 0, n + 1]
            f10 = [n + 4, n + 5, n + 1]
            f11 = [n + 2, n + 3, n + 6]
            f12 = [n + 3, n + 7, n + 6]

            vertices.append(v000)
            vertices.append(v001)
            vertices.append(v010)
            vertices.append(v011)
            vertices.append(v100)
            vertices.append(v101)
            vertices.append(v110)
            vertices.append(v111)

            faces.append(f1)
            faces.append(f2)
            faces.append(f3)
            faces.append(f4)
            faces.append(f5)
            faces.append(f6)
            faces.append(f7)
            faces.append(f8)
            faces.append(f9)
            faces.append(f10)
            faces.append(f11)
            faces.append(f12)

        return Mesh(vertices, faces)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert occupancy grids to meshes.')
    parser.add_argument('input', type=str, help='The input HDF5 file.')
    parser.add_argument('output', type=str, help='The output directory for OFF files.')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print('Input file does not exist.')
        exit(1)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print('Created output directory.')
    else:
        print('Output directory exists; potentially overwriting contents.')

    occupancy = read_hdf5(args.input)
    if len(occupancy.shape) < 4:
        occupancy = np.expand_dims(occupancy, axis=0)

    for n in range(occupancy.shape[0]):
        mesh = Mesh.from_volume(occupancy[n])

        off_file = os.path.join(args.output, '%d.off' % n)
        mesh.to_off(off_file)
        print('Wrote %s.' % off_file)