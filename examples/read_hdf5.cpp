#include <cstdio>
#include <iostream>

// Boost
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

// Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// HDF5
#include <H5Cpp.h>

/** \brief Read a Hdf5 file into an Eigen tensor.
 * \param[in] filepath path to file
 * \param[out] dense Eigen tensor
 * \return success
 */
bool read_hdf5_rank4(const std::string filepath, Eigen::Tensor<float, 4, Eigen::RowMajor>& dense) {

  try {
    H5::H5File file(filepath, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet("tensor");

    /*
     * Get filespace for rank and dimension
     */
    H5::DataSpace filespace = dataset.getSpace();

    /*
     * Get number of dimensions in the file dataspace
     */
    const size_t rank = 4;

    /*
     * Get and print the dimension sizes of the file dataspace
     */
    hsize_t dimsf[rank];
    filespace.getSimpleExtentDims(dimsf);

    /*
     * Setup tensor.
     */
    dense = Eigen::Tensor<float, rank, Eigen::RowMajor>(dimsf[0], dimsf[1], dimsf[2], dimsf[3]);
    float* buffer = static_cast<float*>(dense.data());

    /*
     * Define the memory space to read dataset.
     */
    H5::DataSpace mspace(rank, dimsf);
    dataset.read(buffer, H5::PredType::NATIVE_FLOAT, mspace, filespace);
  }

  // catch failure caused by the H5File operations
  catch(H5::FileIException error) {
    error.printError();
    return false;
  }

  // catch failure caused by the DataSet operations
  catch(H5::DataSetIException error) {
    error.printError();
    return false;
  }

  // catch failure caused by the DataSpace operations
  catch(H5::DataSpaceIException error) {
    error.printError();
    return false;
  }

  return true;
}

/** \brief Read a Hdf5 file into an Eigen tensor.
 * \param[in] filepath path to file
 * \param[out] dense Eigen tensor
 * \return success
 */
bool read_hdf5_rank5(const std::string filepath, Eigen::Tensor<float, 5, Eigen::RowMajor>& dense) {

  try {
    H5::H5File file(filepath, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet("tensor");

    /*
     * Get filespace for rank and dimension
     */
    H5::DataSpace filespace = dataset.getSpace();

    /*
     * Get number of dimensions in the file dataspace
     */
    const size_t rank = 5;

    /*
     * Get and print the dimension sizes of the file dataspace
     */
    hsize_t dimsf[rank];
    filespace.getSimpleExtentDims(dimsf);

    /*
     * Setup tensor.
     */
    dense = Eigen::Tensor<float, rank, Eigen::RowMajor>(dimsf[0], dimsf[1], dimsf[2], dimsf[3], dimsf[4]);
    float* buffer = static_cast<float*>(dense.data());

    /*
     * Define the memory space to read dataset.
     */
    H5::DataSpace mspace(rank, dimsf);
    dataset.read(buffer, H5::PredType::NATIVE_FLOAT, mspace, filespace);
  }

  // catch failure caused by the H5File operations
  catch(H5::FileIException error) {
    error.printError();
    return false;
  }

  // catch failure caused by the DataSet operations
  catch(H5::DataSetIException error) {
    error.printError();
    return false;
  }

  // catch failure caused by the DataSpace operations
  catch(H5::DataSpaceIException error) {
    error.printError();
    return false;
  }

  return true;
}

int main(int argc, char** argv) {
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("input",  boost::program_options::value<std::string>(), "path to input HDF5 file")
      ("rank", boost::program_options::value<int>()->default_value(5), "rank of tensor saved in HDF5 file, '4' or '5' supported");

  boost::program_options::positional_options_description positionals;
  positionals.add("input", 1);

  boost::program_options::variables_map parameters;
  boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
  boost::program_options::notify(parameters);

  if (parameters.find("help") != parameters.end()) {
    std::cout << desc << std::endl;
    return 1;
  }

  boost::filesystem::path input(parameters["input"].as<std::string>());
  if (!boost::filesystem::is_regular_file(input)) {
    std::cout << "Input file does not exist." << std::endl;
    return 1;
  }

  int rank = parameters["rank"].as<int>();
  std::cout << "Trying to read rank-" << rank << " tensor from " << input << "." << std::endl;

  bool success = false;
  std::vector<int> dimensions;

  // The case distinction here is necessary as Eigen does currently not easily allow
  // (let me know if you find out how) to construct a tensor of certain rank by a list
  // of dimensions - the constructor always expects, see
  // https://stackoverflow.com/questions/47475487/eigen-tensor-construction-with-vector-of-dimensions

  if (rank == 4) {
    Eigen::Tensor<float, 4, Eigen::RowMajor> tensor;
    success = read_hdf5_rank4(input.string(), tensor);

    for (int i = 0; i < rank; i++) {
      dimensions.push_back(tensor.dimension(i));
    }
  }
  else if(rank == 5) {
    Eigen::Tensor<float, 5, Eigen::RowMajor> tensor;
    success = read_hdf5_rank5(input.string(), tensor);

    for (int i = 0; i < rank; i++) {
      dimensions.push_back(tensor.dimension(i));
    }
  }
  else {
    std::cout << "Only --rank=4 or --rank=5 supported." << std::endl;
    return 1;
  }

  if (!success) {
    std::cout << "Could not read " << input << "." << std::endl;
    return 1;
  }

  std::cout << "Read tensor of size " << dimensions[0];
  for (unsigned int i = 1; i < dimensions.size(); i++) {
    std::cout << " x " << dimensions[i];
  }
  std::cout << "." << std::endl;

  return 0;
}