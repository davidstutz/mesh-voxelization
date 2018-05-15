# - Try to find EIGEN3
# Once done, this will define
#
#  EIGEN3_FOUND - system has EIGEN3
#  EIGEN3_INCLUDE_DIRS - the EIGEN3 include directories
#  EIGEN3_LIBRARIES - link these to use EIGEN3

include(LibFindMacros)

# Include dir
find_path(EIGEN3_INCLUDE_DIR
  NAMES signature_of_eigen3_matrix_library
  PATHS
  /usr/local/include
  PATH_SUFFIXES eigen3
  NO_CMAKE_SYSTEM_PATH
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this lib depends on.
set(EIGEN3_PROCESS_INCLUDES EIGEN3_INCLUDE_DIR)
libfind_process(EIGEN3)
