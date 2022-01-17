# Mesh Voxelization

This is a parallel C++ implementation for efficiently voxelizing (watertight, triangular)
meshes into signed distance functions (SDFs) and/or occupancy grids.

If you use this tool, please cite the following work:

    @inproceedings{Stutz2018CVPR,
        title = {Learning 3D Shape Completion from Laser Scan Data with Weak Supervision },
        author = {Stutz, David and Geiger, Andreas},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        publisher = {IEEE Computer Society},
        year = {2018}
    }
    @misc{Stutz2017,
        author = {David Stutz},
        title = {Learning Shape Completion from Bounding Boxes with CAD Shape Priors},
        month = {September},
        year = {2017},
        institution = {RWTH Aachen University},
        address = {Aachen, Germany},
        howpublished = {http://davidstutz.de/},
    }

![Example of voxelizations.](screenshot.jpg?raw=true "Example of voxelizations.")

## Overview

**SDFs:**
The implementation uses the triangle-point distance implementation from
[christopherbatty/SDFGen](https://github.com/christopherbatty/SDFGen)
and triangle-ray intersection implementation by 
[Tomas Akenine-Möller](http://fileadmin.cs.lth.se/cs/personal/tomas_akenine-moller/raytri/).
To determine the distance of each voxel (its corner or center) to the
nearest mesh face as well as its sign. Negative sign corresponds to
interior voxels.

Publicly available marching cubes implementations, such as
[PyMCubes](https://github.com/pmneila/PyMCubes) or
[skimage](http://scikit-image.org/docs/dev/auto_examples/edges/plot_marching_cubes.html),
can be used to derive triangular meshes at sub-voxel accuracy
from the computed SDFs. Most implementations use the voxel
corner as reference points; this tool supports both the voxel corner
and its center. For a marching cubes implementation also using the
center, see the `voxel_centers` branch of
[this PyMCubes fork](https://github.com/davidstutz/PyMCubes).

**Occupancy:** Occupancy grids can either be derived from the computed
SDFs (not included) or computed separately using triangle-box intersections;
the code from [Tomas Akenine-Möller](http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/)
is used. This has the advantage that voxels intersecting the mesh surface
are definitely identified as occupied - this is not always the case
when deriving occupancy from SDFs.

The computed SDFs and occupancy grids are saved using
the [HDF5](https://support.hdfgroup.org/HDF5/) file format.
C++ and Python utilities for reading this
format are provided; as well as C++ and Python utilities for the
used (triangular) mesh format: [OFF](http://segeval.cs.princeton.edu/public/off_format.html).

Color of the original mesh can be preserved in ocupancy grid mode.

## Installation

Requirements for C++ tool:

* CMake;
* Boost;
* HDF5;
* Eigen;
* OpenMP;
* C++11.

Requirements for Python tool:

* Numpy;
* h5py;
* PyMCubes or skimage.

On Ubuntu and related Linux distributions, these requirements can be installed
as follows:

    sudo apt-get install build-essential cmake libboost-all-dev libhdf5-dev libeigen3-dev

Make sure that the used compiler supports OpenMp and that the installed Eigen version
include the [unsupported Tensor module](https://eigen.tuxfamily.org/dox/unsupported/group__CXX11__Tensor__Module.html).

For using the Python tools, also make sure to install numpy, h5py and skimage (or PyMCubes):

    pip install numpy
    pip install h5py
    pip install scikit-image

Follow the instructions [here](https://github.com/pmneila/PyMCubes) to install
PyMCubes as alternative to skimage (however, only one of both is required).

To build, **first adapt `cmake/FindEigen3.cmake` to include the correct path
to Eigen3's include directory and remove `NO_CMAKE_SYSTEM_PATH` if necessary**, and run:

    mkdir build
    cd build
    cmake ..
    make

To test the installation you can run (form within the `build` directory):

    ../bin/voxelize occ ../examples/input ../examples/output.h5

To obtain occupancy grids; or

    ../bin/voxelize sdf ../examples/input ../examples/output.h5

To obtain SDFs.

Also install [MeshLab](http://www.meshlab.net/) to visualize OFF files.

## Usage

The general procedure can be summarized as follows:

1. Scale the raw meshes to lie in `[0, H] x [0, W] x [0, D]` corresponding to 
   the chosen resolution `H x D x W`.
2. Voxelize the scaled meshes into SDFs or occupancy grids.
    1. Fill the occupancy grids if necessary.
3. Convert the occupancy grids or SDFs to meshes for visualization.

The first step can be done using `examples/scale_off.py` which takes the
meshes, for example from `examples/raw`, and first normalizes them to
`[-0.5,0.5]^3` to afterwards sale and translate them to `[0, H] x [0, W] x [0, D]`.
This can be accomplished using:

    python ../examples/scale_off.py ../examples/raw/ ../examples/input/

If your file contains color information of faces, set `--color=True`.

The meshes can then be voxelized; for details, check the `--help` option:

    $ ../bin/voxelize --help
    Allowed options:
      --help                produce help message
      --mode arg (=occ)     operation mode, 'occ' or 'sdf'
      --input arg           input, either single OFF file or directory containing 
                            OFF files where the names correspond to integers (zero 
                            padding allowed) and are consecutively numbered 
                            starting with zero
      --height arg (=32)    height of volume, corresponding to y-axis (=up)
      --width arg (=32)     width of volume, corresponding to x-axis (=right
      --depth arg (=32)     depth of volume, corresponding to z-axis (=forward)
      --center              by default, the top-left-front corner is used for SDF 
                            computation; if instead the voxel centers should be 
                            used, set this flag
      --color               by default, color information is not kept after 
                            voxelization, set this flag if you want to voxelize with 
                            colors
      --output arg          output file, will be a HDF5 file containing either a N 
                            x C x height x width x depth tensor or a C x height x 
                            width x depth tensor, where N is the number of files 
                            and C=2 the number of channels, N is discarded if only 
                            a single file is processed; should have the .h5 
                            extension

The mode determines whether occupancy grids or SDFs are computed. For SDFs, `--center`
indicates that the voxel's centers are to be used for SDF computation instead of the
corners (by default); this has influence on the used marching cubes implementation.
The output will be a `N x H x W x D` tensor as HDF5 file containing the occupancy
grids or SDFs per mesh.

When `--color` flag is set, an additional file (or files) `[int]_color.h5` with the original
mesh face colors is created in the input directory. Instead of occupancy grid being filled with
0 or 1 it will consist of 0s and face IDs. When converting voxelized mesh back to OFF file as
in `examples/occ_to_off.py`, these face IDs can be used to point to the corresponding color in the
color file (or files) which is then used for all 12 triangular faces of the voxel. Color file(s)
should be input when calling `examples/occ_to_off.py` with `--color` argument, multiple files can
be listed but ordering is important.

**Note:** The _triangular_ meshes of the input OFF files should be watertight. This can, together
with a simplification of the meshes, be acheived using Andreas Geiger's
[semi-convex hull algorithm](http://www.cvlibs.net/software/semi_convex_hull/)
which, however, imposes a rather crude simplification.
The meshes should additionally be scaled to fit the volume used for voxelization.
For the OFF files, the tool assumes the coordinate system x=right,
y=up and z=forward; this means that the x and y axes are swapped for
voxelization (in the output volume, the height is the first dimension).

## Example

Using example meshes from [ModelNet](http://modelnet.cs.princeton.edu/), two
examples illustrate usage of the tools. In both cases, the meshes
are first scaled:

    python ../examples/scale_off.py ../examples/raw/ ../examples/input/

Here, we use the default resolution of `32^3`.

### SDFs

Given the scaled meshes, voxelization into SDFs is done as follows:

    ../bin/voxelize sdf ../examples/input ../examples/output.h5

In order to visualize the SDFs, marching cubes can be used to derive
triangular meshes. As we used the default parameters, the voxels' corners
were used for SDF computation, therefore, the standard PyMCubes can be used;
alternatively, skimage can also be used:

    python ../examples/marching_cubes.py ../examples/output.h5 ../examples/output/

Note that marching cubes might fail, e.g. with `Surface level must be within volume data range.`,
if the original mesh was not watertight (with significant holes) or structures within
the outer surface prevents SDF computation. In this case, the SDF might not have negative
values and marching cube fails.

Note that in low resolution, the reconstruction might look bad; try the same
procedure with `64^3` to get significantly better results.

### Occupancy Grids

Given the scaled meshes,

    ../bin/voxelize occ ../examples/input ../examples/output.h5

Is used to compute occupancy grids. Note that these are not "filled"; meaning
that only the mesh surfaces are voxelized. Assuming the original shapes to be
mostly watertight, the occupancy grids can be filled using a connected components
algorithm as in `examples/fill_occupancy.py`:

    python ../examples/fill_occupancy.py ../examples/output.h5 ../examples/filled.h5

The occupancy grids can be converted to meshes for visualization using

    python ../examples/occ_to_off.py ../examples/filled.h5 ../examples/output/

## License

License for source code corresponding to:

D. Stutz, A. Geiger. **Learning 3D Shape Completion from Laser Scan Data with Weak Supervision.** IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

Note that the source code is based on the following projects for which separate licenses apply:

* [christopherbatty/SDFGen](https://github.com/christopherbatty/SDFGen)
* [ray-triangle intersection](http://fileadmin.cs.lth.se/cs/personal/tomas_akenine-moller/raytri/)
* [box-triangle intersection](http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/)
* [Tronic/cmake-modules](https://github.com/Tronic/cmake-modules).

Copyright (c) 2018 David Stutz, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Software.
