#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cfloat>

// Boost
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

// Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// HDF5
#include <H5Cpp.h>

// OpenMP
#include <omp.h>

// Point-triangle distance and ray-triangle intersection.
#include "triangle_point/poitri.h"
#include "triangle_ray/raytri.h"
#include "box_triangle/aabb_triangle_overlap.h"

/** \brief Compute triangle point distance and corresponding closest point.
 * \param[in] point point
 * \param[in] v1 first vertex
 * \param[in] v2 second vertex
 * \param[in] v3 third vertex
 * \param[out] ray corresponding closest point
 * \return distance
 */
float triangle_point_distance(const Eigen::Vector3f &point, const Eigen::Vector3f &v1, const Eigen::Vector3f &v2, const Eigen::Vector3f &v3,
    Eigen::Vector3f &closest_point) {

  Vec3f x0(point.data());
  Vec3f x1(v1.data());
  Vec3f x2(v2.data());
  Vec3f x3(v3.data());

  Vec3f r(0);
  float distance = point_triangle_distance(x0, x1, x2, x3, r);

  for (int d = 0; d < 3; d++) {
    closest_point(d) = r[d];
  }

  return distance;
}

/** \brief Test triangle ray intersection.
 * \param[in] origin origin of ray
 * \param[in] dest destination of ray
 * \param[in] v1 first vertex
 * \param[in] v2 second vertex
 * \param[in] v3 third vertex
 * \return intersects
 */
bool triangle_ray_intersection(const Eigen::Vector3f &origin, const Eigen::Vector3f &dest,
    const Eigen::Vector3f &v1, const Eigen::Vector3f &v2, const Eigen::Vector3f &v3, float &t) {

  double _origin[3] = {origin(0), origin(1), origin(2)};
  double _dir[3] = {dest(0) - origin(0), dest(1) - origin(1), dest(2) - origin(2)};
  double _v1[3] = {v1(0), v1(1), v1(2)};
  double _v2[3] = {v2(0), v2(1), v2(2)};
  double _v3[3] = {v3(0), v3(1), v3(2)};

  // t is the distance, u and v are barycentric coordinates
  // http://fileadmin.cs.lth.se/cs/personal/tomas_akenine-moller/code/raytri_tam.pdf
  double _t, u, v;
  int success = intersect_triangle(_origin, _dir, _v1, _v2, _v3, &_t, &u, &v);
  t = _t;

  if (success) {
    return true;
  }

  return false;
}

/** \brief Compute triangle box intersection.
 * \param[in] min defining voxel
 * \param[in] max defining voxel
 * \param[in] v1 first vertex
 * \param[in] v2 second vertex
 * \param[in] v3 third vertex
 * \return intersects
 */
bool triangle_box_intersection(const Eigen::Vector3f &min, Eigen::Vector3f &max, const Eigen::Vector3f &v1, const Eigen::Vector3f &v2, const Eigen::Vector3f &v3) {
  float half_size[3] = {
    (max(0) - min(0))/2.,
    (max(1) - min(1))/2.,
    (max(2) - min(2))/2.
  };

  float center[3] = {
    max(0) - half_size[0],
    max(1) - half_size[1],
    max(2) - half_size[2]
  };

  float vertices[3][3] = {{v1(0), v1(1), v1(2)}, {v2(0), v2(1), v2(2)}, {v3(0), v3(1), v3(2)}};
  return triBoxOverlap(center, half_size, vertices);
}

/** \brief Specifies the voxelization mode, i.e. which point of a voxel to use for SDF computation. */
enum VoxelizationMode {
  CENTER = 0,
  CORNER = 1
};

/** \brief Just encapsulating vertices and faces. */
class Mesh {
public:
  /** \brief Empty constructor. */
  Mesh() {

  }

  /** \brief Reading an off file and returning the vertices x, y, z coordinates and the
   * face indices.
   * \param[in] filepath path to the OFF file
   * \param[out] mesh read mesh with vertices and faces
   * \return success
   */
  static bool from_off(const std::string filepath, Mesh& mesh) {

    std::ifstream* file = new std::ifstream(filepath.c_str());
    std::string line;
    std::stringstream ss;
    int line_nb = 0;

    std::getline(*file, line);
    ++line_nb;

    if (line != "off" && line != "OFF") {
      std::cout << "[Error] Invalid header: \"" << line << "\", " << filepath << std::endl;
      return false;
    }

    size_t n_edges;
    std::getline(*file, line);
    ++line_nb;

    int n_vertices;
    int n_faces;
    ss << line;
    ss >> n_vertices;
    ss >> n_faces;
    ss >> n_edges;

    for (size_t v = 0; v < n_vertices; ++v) {
      std::getline(*file, line);
      ++line_nb;

      ss.clear();
      ss.str("");

      Eigen::Vector3f vertex;
      ss << line;
      ss >> vertex(0);
      ss >> vertex(1);
      ss >> vertex(2);

      mesh.add_vertex(vertex);
    }

    size_t n;
    for (size_t f = 0; f < n_faces; ++f) {
      std::getline(*file, line);
      ++line_nb;

      ss.clear();
      ss.str("");

      size_t n;
      ss << line;
      ss >> n;

      if(n != 3) {
        std::cout << "[Error] Not a triangle (" << n << " points) at " << (line_nb - 1) << std::endl;
        return false;
      }

      Eigen::Vector3i face;
      ss >> face(0);
      ss >> face(1);
      ss >> face(2);

      mesh.add_face(face);
    }

    if (n_vertices != mesh.num_vertices()) {
      std::cout << "[Error] Number of vertices in header differs from actual number of vertices." << std::endl;
      return false;
    }

    if (n_faces != mesh.num_faces()) {
      std::cout << "[Error] Number of faces in header differs from actual number of faces." << std::endl;
      return false;
    }

    file->close();
    delete file;

    return true;
  }

  /** \brief Write mesh to OFF file.
   * \param[in] filepath path to OFF file to write
   * \return success
   */
  bool to_off(const std::string filepath) {
    std::ofstream* out = new std::ofstream(filepath, std::ofstream::out);
    if (!static_cast<bool>(out)) {
      return false;
    }

    (*out) << "OFF" << std::endl;
    (*out) << this->num_vertices() << " " << this->num_faces() << " 0" << std::endl;

    for (unsigned int v = 0; v < this->num_vertices(); v++) {
      (*out) << this->vertices[v](0) << " " << this->vertices[v](1) << " " << this->vertices[v](2) << std::endl;
    }

    for (unsigned int f = 0; f < this->num_faces(); f++) {
      (*out) << "3 " << this->faces[f](0) << " " << this->faces[f](1) << " " << this->faces[f](2) << std::endl;
    }

    out->close();
    delete out;

    return true;
  }

  /** \brief Add a vertex.
   * \param[in] vertex vertex to add
   */
  void add_vertex(Eigen::Vector3f& vertex) {
    this->vertices.push_back(vertex);
  }

  /** \brief Get the number of vertices.
   * \return number of vertices
   */
  int num_vertices() {
    return static_cast<int>(this->vertices.size());
  }

  /** \brief Add a face.
   * \param[in] face face to add
   */
  void add_face(Eigen::Vector3i& face) {
    this->faces.push_back(face);
  }

  /** \brief Get the number of faces.
   * \return number of faces
   */
  int num_faces() {
    return static_cast<int>(this->faces.size());
  }

  /** \brief Translate the mesh.
   * \param[in] translation translation vector
   */
  void translate(const Eigen::Vector3f& translation) {
    for (int v = 0; v < this->num_vertices(); ++v) {
      for (int i = 0; i < 3; ++i) {
        this->vertices[v](i) += translation(i);
      }
    }
  }

  /** \brief Scale the mesh.
   * \param[in] scale scale vector
   */
  void scale(const Eigen::Vector3f& scale) {
    for (int v = 0; v < this->num_vertices(); ++v) {
      for (int i = 0; i < 3; ++i) {
        this->vertices[v](i) *= scale(i);
      }
    }
  }

  /** \brief Voxelize the given mesh into a SDF.
   * \param[out] sdf volume to fill with sdf values
   */
  void voxelize_sdf(Eigen::Tensor<float, 3, Eigen::RowMajor>& sdf, const VoxelizationMode &mode) {

    int height = sdf.dimension(0);
    int width = sdf.dimension(1);
    int depth = sdf.dimension(2);

    #pragma omp parallel
    {
      #pragma omp for
      for (int i = 0; i < height*width*depth; i++) {
        int d = i%depth;
        int w = (i/depth)%width;
        int h = (i/depth)/width;

        sdf(h, w, d) = FLT_MAX;

        // the box corresponding to this voxel
        Eigen::Vector3f min(w, h, d);
        Eigen::Vector3f max(w + 1, h + 1, d + 1);

        Eigen::Vector3f center(w + 0.5f, h + 0.5f, d + 0.5f);
        if (mode == VoxelizationMode::CORNER) {
          center = Eigen::Vector3f(w, h, d);
        }

        // count number of intersections.
        int num_intersect = 0;
        for (unsigned int f = 0; f < this->num_faces(); ++f) {

          Eigen::Vector3f v1 = this->vertices[this->faces[f](0)];
          Eigen::Vector3f v2 = this->vertices[this->faces[f](1)];
          Eigen::Vector3f v3 = this->vertices[this->faces[f](2)];

          Eigen::Vector3f closest_point;
          triangle_point_distance(center, v1, v2, v3, closest_point);
          float distance = (center - closest_point).norm();

          if (distance < sdf(h, w, d)) {
            sdf(h, w, d) = distance;
          }

          bool intersect = triangle_ray_intersection(center, Eigen::Vector3f(0, 0, 0), v1, v2, v3, distance);

          if (intersect && distance >= 0) {
            num_intersect++;
          }
        }

        if (num_intersect%2 == 1) {
          sdf(h, w, d) *= -1;
        }
      }
    }
  }

  /** \brief Voxelize the given mesh into an occupancy grid.
   * \param[out] occ volume to fill
   */
  void voxelize_occ(Eigen::Tensor<int, 3, Eigen::RowMajor>& occ, const VoxelizationMode &mode) {

    int height = occ.dimension(0);
    int width = occ.dimension(1);
    int depth = occ.dimension(2);

    #pragma omp parallel
    {
      #pragma omp for
      for (int i = 0; i < height*width*depth; i++) {
        int d = i%depth;
        int w = (i/depth)%width;
        int h = (i/depth)/width;

        Eigen::Vector3f min(w, h, d);
        Eigen::Vector3f max(w + 1, h + 1, d + 1);

        for (unsigned int f = 0; f < this->num_faces(); ++f) {

          Eigen::Vector3f v1 = this->vertices[this->faces[f](0)];
          Eigen::Vector3f v2 = this->vertices[this->faces[f](1)];
          Eigen::Vector3f v3 = this->vertices[this->faces[f](2)];

          bool overlap = triangle_box_intersection(min, max, v1, v2, v3);
          if (overlap) {
            occ(h, w, d) = 1;
            break;
          }
        }
      }
    }
  }

private:

  /** \brief Vertices as (x,y,z)-vectors. */
  std::vector<Eigen::Vector3f> vertices;

  /** \brief Faces as list of vertex indices. */
  std::vector<Eigen::Vector3i> faces;
};

/** \brief Write the given set of volumes to h5 file.
 * \param[in] filepath h5 file to write
 * \param[in] n number of volumes
 * \param[in] height height of volumes
 * \param[in] width width of volumes
 * \param[in] depth depth of volumes
 * \param[in] dense volume data
 */
template<int RANK>
bool write_float_hdf5(const std::string filepath, Eigen::Tensor<float, RANK, Eigen::RowMajor>& tensor) {

  try {

    /*
     * Turn off the auto-printing when failure occurs so that we can
     * handle the errors appropriately
     */
    H5::Exception::dontPrint();

    /*
     * Create a new file using H5F_ACC_TRUNC access,
     * default file creation properties, and default file
     * access properties.
     */
    H5::H5File file(filepath, H5F_ACC_TRUNC);

    /*
     * Define the size of the array and create the data space for fixed
     * size dataset.
     */
    hsize_t rank = RANK;
    hsize_t dimsf[rank];
    for (int i = 0; i < rank; i++) {
      dimsf[i] = tensor.dimension(i);

    }
    H5::DataSpace dataspace(rank, dimsf);

    /*
     * Define datatype for the data in the file.
     * We will store little endian INT numbers.
     */
    H5::IntType datatype(H5::PredType::NATIVE_FLOAT);
    datatype.setOrder(H5T_ORDER_LE);

    /*
     * Create a new dataset within the file using defined dataspace and
     * datatype and default dataset creation properties.
     */
    H5::DataSet dataset = file.createDataSet("tensor", datatype, dataspace);

    /*
     * Write the data to the dataset using default memory space, file
     * space, and transfer properties.
     */
    float* data = static_cast<float*>(tensor.data());
    dataset.write(data, H5::PredType::NATIVE_FLOAT);
  }  // end of try block

  // catch failure caused by the H5File operations
  catch(H5::FileIException error) {
    error.printErrorStack();
    return false;
  }

  // catch failure caused by the DataSet operations
  catch(H5::DataSetIException error) {
    error.printErrorStack();
    return false;
  }

  // catch failure caused by the DataSpace operations
  catch(H5::DataSpaceIException error) {
    error.printErrorStack();
    return false;
  }

  // catch failure caused by the DataSpace operations
  catch(H5::DataTypeIException error) {
    error.printErrorStack();
    return false;
  }

  return true;
}

/** \brief Write the given set of volumes to h5 file.
 * \param[in] filepath h5 file to write
 * \param[in] n number of volumes
 * \param[in] height height of volumes
 * \param[in] width width of volumes
 * \param[in] depth depth of volumes
 * \param[in] dense volume data
 */
template<int RANK>
bool write_int_hdf5(const std::string filepath, Eigen::Tensor<int, RANK, Eigen::RowMajor>& tensor) {

  try {

    /*
     * Turn off the auto-printing when failure occurs so that we can
     * handle the errors appropriately
     */
    H5::Exception::dontPrint();

    /*
     * Create a new file using H5F_ACC_TRUNC access,
     * default file creation properties, and default file
     * access properties.
     */
    H5::H5File file(filepath, H5F_ACC_TRUNC);

    /*
     * Define the size of the array and create the data space for fixed
     * size dataset.
     */
    hsize_t rank = RANK;
    hsize_t dimsf[rank];
    for (int i = 0; i < rank; i++) {
      dimsf[i] = tensor.dimension(i);

    }
    H5::DataSpace dataspace(rank, dimsf);

    /*
     * Define datatype for the data in the file.
     * We will store little endian INT numbers.
     */
    H5::IntType datatype(H5::PredType::NATIVE_INT);
    datatype.setOrder(H5T_ORDER_LE);

    /*
     * Create a new dataset within the file using defined dataspace and
     * datatype and default dataset creation properties.
     */
    H5::DataSet dataset = file.createDataSet("tensor", datatype, dataspace);

    /*
     * Write the data to the dataset using default memory space, file
     * space, and transfer properties.
     */
    int* data = static_cast<int*>(tensor.data());
    dataset.write(data, H5::PredType::NATIVE_INT);
  }  // end of try block

  // catch failure caused by the H5File operations
  catch(H5::FileIException error) {
    error.printErrorStack();
    return false;
  }

  // catch failure caused by the DataSet operations
  catch(H5::DataSetIException error) {
    error.printErrorStack();
    return false;
  }

  // catch failure caused by the DataSpace operations
  catch(H5::DataSpaceIException error) {
    error.printErrorStack();
    return false;
  }

  // catch failure caused by the DataSpace operations
  catch(H5::DataTypeIException error) {
    error.printErrorStack();
    return false;
  }

  return true;
}

/** \brief Read all files in a directory matching the given extension.
 * \param[in] directory path to directory
 * \param[out] files read file paths
 * \param[in] extension extension to filter for
 */
void read_directory(const boost::filesystem::path directory, std::map<int, boost::filesystem::path>& files, const std::string extension = ".off") {

  files.clear();
  boost::filesystem::directory_iterator end;

  for (boost::filesystem::directory_iterator it(directory); it != end; ++it) {
    if (it->path().extension().string() == extension) {
      int number = std::stoi(it->path().filename().string());
      files.insert(std::pair<int, boost::filesystem::path>(number, it->path()));
    }
  }
}

/** \brief Main entrance point of the script.
 * Expects one parameter, the path to the corresponding config file in config/.
 */
int main(int argc, char** argv) {
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("mode", boost::program_options::value<std::string>()->default_value("occ"), "operation mode, 'occ' or 'sdf'")
      ("input", boost::program_options::value<std::string>(), "input, either single OFF file or directory containing OFF files where the names correspond to integers (zero padding allowed) and are consecutively numbered starting with zero")
      ("height", boost::program_options::value<int>()->default_value(32), "height of volume, corresponding to y-axis (=up)")
      ("width", boost::program_options::value<int>()->default_value(32), "width of volume, corresponding to x-axis (=right")
      ("depth", boost::program_options::value<int>()->default_value(32), "depth of volume, corresponding to z-axis (=forward)")
      ("center", boost::program_options::bool_switch()->default_value(false), "by default, the top-left-front corner is used for SDF computation; if instead the voxel centers should be used, set this flag")
      ("output", boost::program_options::value<std::string>(), "output file, will be a HDF5 file containing either a N x C x height x width x depth tensor or a C x height x width x depth tensor, where N is the number of files and C=2 the number of channels, N is discarded if only a single file is processed; should have the .h5 extension");

  boost::program_options::positional_options_description positionals;
  positionals.add("mode", 1);
  positionals.add("input", 1);
  positionals.add("output", 1);

  boost::program_options::variables_map parameters;
  boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
  boost::program_options::notify(parameters);

  if (parameters.find("help") != parameters.end()) {
    std::cout << desc << std::endl;
    return 0;
  }

  std::string mode = parameters["mode"].as<std::string>();
  if (mode == "occ") {
    std::cout << "Voxelizing occupancy grids." << std::endl;
  }
  else if (mode == "sdf") {
    std::cout << "Voxelizing SDFs." << std::endl;
  }
  else {
    std::cout << "Invalid mode, choose from occ or sdf." << std::endl;
    return 1;
  }

  boost::filesystem::path input(parameters["input"].as<std::string>());
  if (!boost::filesystem::is_directory(input) && !boost::filesystem::is_regular_file(input)) {
    std::cout << "Input is neither directory nor file." << std::endl;
    return 1;
  }

  boost::filesystem::path output(parameters["output"].as<std::string>());
  if (boost::filesystem::is_regular_file(output)) {
    std::cout << "Output file already exists; overwriting." << std::endl;
  }

  VoxelizationMode voxelization_mode;
  if (parameters["center"].as<bool>()) {
    voxelization_mode = VoxelizationMode::CENTER;
    std::cout << "Using the top-left-front voxel corner for voxelization." << std::endl;
  }
  else {
    voxelization_mode = VoxelizationMode::CORNER;
    std::cout << "Using the voxel center for voxelization." << std::endl;
  }

  int height = parameters["height"].as<int>();
  int width = parameters["width"].as<int>();
  int depth = parameters["depth"].as<int>();

  std::cout << "Voxelizing into " << height << " x " << width << " x " << depth << " (height x width x depth)." << std::endl;

  if (boost::filesystem::is_regular_file(input)) {
    Mesh mesh;
    bool success = Mesh::from_off(input.string(), mesh);

    if (!success) {
      std::cout << "Could not read " << input << "." << std::endl;
      return 1;
    }

    std::cout << "Read " << input << "." << std::endl;

    if (mode == "sdf") {
      Eigen::Tensor<float, 3, Eigen::RowMajor> tensor(height, width, depth);

      mesh.voxelize_sdf(tensor, voxelization_mode);
      std::cout << "Voxelized " << input << "." << std::endl;

      bool success = write_float_hdf5<3>(output.string(), tensor);

      if (!success) {
        std::cout << "Could not write " << output << "." << std::endl;
        return 1;
      }
    }
    if (mode == "occ") {
      Eigen::Tensor<int, 3, Eigen::RowMajor> tensor(height, width, depth);
      tensor.setZero();

      mesh.voxelize_occ(tensor, voxelization_mode);
      std::cout << "Voxelized " << input << "." << std::endl;

      bool success = write_int_hdf5<3>(output.string(), tensor);

      if (!success) {
        std::cout << "Could not write " << output << "." << std::endl;
        return 1;
      }
    }

    std::cout << "Wrote " << output << "." << std::endl;
    std::cout << "The output is a " << height << " x " << width << " x " << depth << " tensor." << std::endl;
  }
  else {
    std::map<int, boost::filesystem::path> input_files;
    read_directory(input, input_files);

    if (input_files.size() <= 0) {
      std::cout << "Could not find any OFF files in the input directory." << std::endl;
      return 1;
    }

    std::cout << "Read " << input_files.size() << " files." << std::endl;

    if (mode == "sdf") {
      Eigen::Tensor<float, 4, Eigen::RowMajor> tensor(input_files.size(), height, width, depth);

      int i = 0;
      for (std::map<int, boost::filesystem::path>::iterator it = input_files.begin(); it != input_files.end(); it++) {
        Mesh mesh;
        bool success = Mesh::from_off(it->second.string(), mesh);

        if (!success) {
          std::cout << "Could not read " << it->second << "." << std::endl;
          return 1;
        }

        Eigen::Tensor<float, 3, Eigen::RowMajor> slice(height, width, depth);
        mesh.voxelize_sdf(slice, voxelization_mode);
        tensor.chip(i, 0) = slice;
        std::cout << "Voxelized " << it->second << " (" << (i + 1) << " of " << input_files.size() << ")." << std::endl;

        i++;
      }

      bool success = write_float_hdf5<4>(output.string(), tensor);

      if (!success) {
        std::cout << "Could not write " << output << "." << std::endl;
        return 1;
      }
    }
    if (mode == "occ") {
      Eigen::Tensor<int, 4, Eigen::RowMajor> tensor(input_files.size(), height, width, depth);
      tensor.setZero();

      int i = 0;
      for (std::map<int, boost::filesystem::path>::iterator it = input_files.begin(); it != input_files.end(); it++) {
        Mesh mesh;
        bool success = Mesh::from_off(it->second.string(), mesh);

        if (!success) {
          std::cout << "Could not read " << it->second << "." << std::endl;
          return 1;
        }

        Eigen::Tensor<int, 3, Eigen::RowMajor> slice(height, width, depth);
        slice.setZero();

        mesh.voxelize_occ(slice, voxelization_mode);
        tensor.chip(i, 0) = slice;
        std::cout << "Voxelized " << it->second << " (" << (i + 1) << " of " << input_files.size() << ")." << std::endl;

        i++;
      }

      bool success = write_int_hdf5<4>(output.string(), tensor);

      if (!success) {
        std::cout << "Could not write " << output << "." << std::endl;
        return 1;
      }
    }

    std::cout << "Wrote " << output << "." << std::endl;
    std::cout << "The output is a " << input_files.size() << " x " << height << " x " << width << " x " << depth << " tensor." << std::endl;
  }

  return 0;
}