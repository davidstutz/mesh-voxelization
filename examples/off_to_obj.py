import os
import argparse

def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        assert lines[0] == 'OFF' or lines[0] == 'off', 'invalid OFF file %s' % file

        parts = lines[1].split(' ')
        assert len(parts) == 3

        num_vertices = int(parts[0])
        assert num_vertices > 0

        num_faces = int(parts[1])
        assert num_faces > 0

        vertices = []
        for i in range(num_vertices):
            vertex = lines[2 + i].split(' ')
            vertex = [float(point) for point in vertex]
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[2 + num_vertices + i].split(' ')
            face = [index.strip() for index in face]

            # check to be sure
            for index in face:
                assert index != '', 'found empty vertex index: %s (%s)' % (lines[2 + num_vertices + i], file)

            face = [int(index) for index in face]

            assert face[0] == len(face) - 1, 'face should have %d vertices but as %d (%s)' % (face[0], len(face) - 1, file)
            assert face[0] == 3, 'only triangular meshes supported (%s)' % file
            for index in face:
                assert index >= 0 and index < num_vertices, 'vertex %d (of %d vertices) does not exist (%s)' % (index, num_vertices, file)

            faces.append(face[1:])

        return vertices, faces

def write_obj(file, vertices, faces):
    """
    Writes the given vertices and faces to OBJ.

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
        for vertex in vertices:
            assert len(vertex) == 3, 'invalid vertex with %d dimensions found (%s)' % (len(vertex), file)
            fp.write('v' + ' ' + str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')

        for face in faces:
            assert len(face) == 3, 'only triangular faces supported (%s)' % file
            fp.write('f' + ' ' + str(face[0] + 1) + ' ' + str(face[1] + 1) + ' ' + str(face[2] + 1) + '\n')

        # add empty line to be sure
        fp.write('\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert OFF to OBJ.')
    parser.add_argument('input', type=str, help='The input directory containing OFF files.')
    parser.add_argument('output', type=str, help='The output directory for OBJ files.')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print('Input directory does not exist.')
        exit(1)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print('Created output directory.')
    else:
        print('Output directory exists; potentially overwriting contents.')

    for filename in os.listdir(args.input):
        filepath = args.input + '/' + filename
        vertices, faces = read_off(filepath)
        print('Read %s.' % filepath)

        filepath = args.output + '/' + filename[:-4] + '.obj'
        write_obj(filepath, vertices, faces)
        print('Wrote %s.' % filepath)
