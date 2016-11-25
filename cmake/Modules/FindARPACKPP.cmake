find_package(PkgConfig)

find_path(ARPACKPP_INCLUDE_DIR arpackf.h
    PATH_SUFFIXES arpackpp)

find_library(ARPACKPP_LIB NAMES arpack)

find_library(SUPERLU_LIB NAMES superlu)

find_package_handle_standard_args(ARPACKPP  DEFAULT_MSG
                                  ARPACKPP_INCLUDE_DIR ARPACKPP_LIB SUPERLU_LIB)

mark_as_advanced(ARPACKPP_INCLUDE_DIR ARPACKPP_LIB)

set(ARPACKPP_INCLUDE_DIRS ${ARPACKPP_INCLUDE_DIR})
set(ARPACKPP_LIBRARIES ${ARPACKPP_LIB} ${SUPERLU_LIB})
