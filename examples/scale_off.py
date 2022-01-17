import os
import math
import argparse
import numpy as np

def write_off(file, vertices, faces, colors):
    """
    Writes the given vertices and faces to OFF.

    :param vertices: vertices as tuples of (x, y, z) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)
    num_colors = len(colors)

    assert num_vertices > 0
    assert num_faces > 0

    with open(file, 'w') as fp:
        fp.write('OFF\n')
        fp.write(str(num_vertices) + ' ' + str(num_faces) + ' 0\n')

        for vertex in vertices:
            assert len(vertex) == 3, 'invalid vertex with %d dimensions found (%s)' % (len(vertex), file)
            fp.write(str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')

        if num_colors == 0:
            for face in faces:
                assert face[0] == 3, 'only triangular faces supported (%s)' % file
                assert len(face) == 4, 'faces need to have 3 vertices, but found %d (%s)' % (len(face), file)

                for i in range(len(face)):
                    assert face[i] >= 0 and face[i] < num_vertices, 'invalid vertex index %d (of %d vertices) (%s)' % (face[i], num_vertices, file)

                    fp.write(str(face[i]))
                    if i < len(face) - 1:
                        fp.write(' ')
                fp.write('\n')

        else:
            for fi in range(num_faces):
                assert faces[fi][0] == 3, 'only triangular faces supported (%s)' % file
                assert len(faces[fi]) == 4, 'faces need to have 3 vertices, but found %d (%s)' % (len(faces[fi]), file)

                for vi in range(len(faces[fi])):
                    assert faces[fi][vi] >= 0 and faces[fi][vi] < num_vertices, 'invalid vertex index %d (of %d vertices) (%s)' % (faces[fi][vi], num_vertices, file)

                    fp.write(str(faces[fi][vi]))
                    fp.write(' ')
                
                for cc in range(len(colors[fi])):

                    fp.write(str(colors[fi][cc]))
                    if cc < len(colors[fi]) - 1:
                        fp.write(' ')

                fp.write('\n')

        # add empty line to be sure
        fp.write('\n')

def read_off(file, color=False):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :param color: if file contains color info
    :type color: bool
    :return: vertices and faces (and colors if true) as lists of tuples
    :rtype: [(float)], [(int)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        # Fix for ModelNet bug were 'OFF' and the number of vertices and faces are
        # all in the first line.
        if len(lines[0]) > 3:
            assert lines[0][:3] == 'OFF' or lines[0][:3] == 'off', 'invalid OFF file %s' % file

            parts = lines[0][3:].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 1
        # This is the regular case!
        else:
            assert lines[0] == 'OFF' or lines[0] == 'off', 'invalid OFF file %s' % file

            parts = lines[1].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 2

        vertices = []
        for i in range(num_vertices):
            vertex = lines[start_index + i].split(' ')
            vertex = [float(point.strip()) for point in vertex if point != '']
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        colors = []    
        for i in range(num_faces):
            face = lines[start_index + num_vertices + i].split(' ')
            face = [index.strip() for index in face if index != '']

            # check to be sure
            for index in face:
                assert index != '', 'found empty vertex index: %s (%s)' % (lines[start_index + num_vertices + i], file)

            face = [int(index) for index in face]

            if color == False:
                face[0] == len(face) - 1, 'face should have %d vertices but has %d (%s)' % (face[0], len(face) - 1, file)
                for index in face:
                    assert index >= 0 and index < num_vertices, 'vertex %d (of %d vertices) does not exist (%s)' % (index, num_vertices, file)
            else: 
                assert face[0] == len(face) - 4, 'face should have %d vertices but has %d (%s)' % (face[0], len(face) - 1, file)
                for index in range(len(face)-3):
                    assert index >= 0 and index < num_vertices, 'vertex %d (of %d vertices) does not exist (%s)' % (index, num_vertices, file)
                for cc in range(len(face)-3,len(face)):
                    assert cc>=0 and cc<=255, 'rgb color value %d is incorrect, check face no. %d (%s)' % (cc, i, file)
                rgb = face[4:]
                face = face[:4]
                colors.append(rgb)

            assert face[0] == 3, 'only triangular meshes supported (%s)' % file
            assert len(face) > 1

            faces.append(face)
        
    return vertices, faces, colors

class Mesh:
    """
    Represents a mesh.
    """

    def __init__(self, vertices = [[]], faces = [[]], colors = [[]]):
        """
        Construct a mesh from vertices and faces.

        :param vertices: list of vertices, or numpy array
        :type vertices: [[float]] or numpy.ndarray
        :param faces: list of faces or numpy array, i.e. the indices of the corresponding vertices per triangular face
        :type faces: [[int]] fo rnumpy.ndarray
        :param colors: list RGB colors corresponding to face
        :type colors: [[int]] fo rnumpy.ndarray
        """

        self.vertices = np.array(vertices, dtype = float)
        """ (numpy.ndarray) Vertices. """

        self.faces = np.array(faces, dtype = int)
        """ (numpy.ndarray) Faces. """

        self.colors = np.array(colors, dtype = int)
        """ (numpy.ndarray) RGB colors. """

        assert self.vertices.shape[1] == 3
        assert self.faces.shape[1] == 3
        if self.colors.size > 0:     
            assert self.colors.shape[0] == self.faces.shape[0], "number of faces (%d) does not equal number of colors (%d)" % (self.faces.shape[2], self.colors.shape[2]) 

    def extents(self):
        """
        Get the extents.

        :return: (min_x, min_y, min_z), (max_x, max_y, max_z)
        :rtype: (float, float, float), (float, float, float)
        """

        min = [0]*3
        max = [0]*3

        for i in range(3):
            min[i] = np.min(self.vertices[:, i])
            max[i] = np.max(self.vertices[:, i])

        return tuple(min), tuple(max)

    def scale(self, scales):
        """
        Scale the mesh in all dimensions.

        :param scales: tuple of length 3 with scale for (x, y, z)
        :type scales: (float, float, float)
        """

        assert len(scales) == 3

        for i in range(3):
            self.vertices[:, i] *= scales[i]

    def translate(self, translation):
        """
        Translate the mesh.

        :param translation: translation as (x, y, z)
        :type translation: (float, float, float)
        """

        assert len(translation) == 3

        for i in range(3):
            self.vertices[:, i] += translation[i]

    @staticmethod
    def from_off(filepath, color):
        """
        Read a mesh from OFF.

        :param filepath: path to OFF file
        :type filepath: str
        :param color: whether file has color information
        :type color: bool
        :return: mesh
        :rtype: Mesh
        """

        vertices, faces, colors = read_off(filepath, color)

        real_faces = []
        for face in faces:
            assert len(face) == 4
            real_faces.append([face[1], face[2], face[3]])

        return Mesh(vertices, real_faces, colors)

    def to_off(self, filepath):
        """
        Write mesh to OFF.

        :param filepath: path to write file to
        :type filepath: str
        :param color: whether to include color information
        :type color: bool
        """

        faces = np.ones((self.faces.shape[0], 4), dtype = int)*3
        faces[:, 1:4] = self.faces[:, :]

        write_off(filepath, self.vertices.tolist(), faces.tolist(), self.colors.tolist())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert OFF to OBJ.')
    parser.add_argument('input', type=str, help='The input directory containing OFF files.')
    parser.add_argument('output', type=str, help='The output directory for OBJ files.')
    parser.add_argument('--padding', type=float, default=0.1, help='Padding on each side.')
    parser.add_argument('--height', type=int, default=32, help='Height to scale to.')
    parser.add_argument('--width', type=int, default=32, help='Width to scale to.')
    parser.add_argument('--depth', type=int, default=32, help='Depth to scale to.')
    parser.add_argument('--color', type=bool, default=False, help='Whether to keep/use color information in off file, default is False')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print('Input directory does not exist.')
        exit(1)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print('Created output directory.')
    else:
        print('Output directory exists; potentially overwriting contents.')

    n = 0
    scale = max(args.height, args.width, args.depth)

    for filename in os.listdir(args.input):
        filepath = os.path.join(args.input, filename)
        mesh = Mesh.from_off(filepath, args.color)

        # Get extents of model.
        min, max = mesh.extents()
        total_min = np.min(np.array(min))
        total_max = np.max(np.array(max))

        print('%s extents before %f - %f, %f - %f, %f - %f.' % (os.path.basename(filepath), min[0], max[0], min[1], max[1], min[2], max[2]))

        # Set the center (although this should usually be the origin already).
        centers = (
            (min[0] + max[0]) / 2,
            (min[1] + max[1]) / 2,
            (min[2] + max[2]) / 2
        )
        # Scales all dimensions equally.
        sizes = (
            total_max - total_min,
            total_max - total_min,
            total_max - total_min
        )
        translation = (
            -centers[0],
            -centers[1],
            -centers[2]
        )
        scales = (
            1 / (sizes[0] + 2 * args.padding * sizes[0]),
            1 / (sizes[1] + 2 * args.padding * sizes[1]),
            1 / (sizes[2] + 2 * args.padding * sizes[2])
        )

        mesh.translate(translation)
        mesh.scale(scales)

        mesh.translate((0.5, 0.5, 0.5))
        mesh.scale((scale, scale, scale))

        min, max = mesh.extents()
        print('%s extents after %f - %f, %f - %f, %f - %f.' % (os.path.basename(filepath), min[0], max[0], min[1], max[1], min[2], max[2]))

        mesh.to_off(os.path.join(args.output, '%d.off' % n))
        n += 1