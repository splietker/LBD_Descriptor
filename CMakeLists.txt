cmake_minimum_required(VERSION 2.8)

PROJECT(LBDDescriptor)

SET(BUILD_SHARED_LIBS ON)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

FIND_PACKAGE(ARPACKPP)
if (ARPACKPP_FOUND)
  INCLUDE_DIRECTORIES(${ARPACKPP_INCLUDE_DIRS})
else (ARPACKPP_FOUND)
  message(FATAL_ERROR, "Could not find ARPACKPP")
endif (ARPACKPP_FOUND)

FIND_PACKAGE(OpenCV)
if (OpenCV_FOUND)
  include_directories(${OpenCV_INCLUDE_DIRS})
else (OpenCV_FOUND)
  message(FATAL_ERROR, "Could not find OpenCV")
endif (OpenCV_FOUND)

# source files of library "LineMatchingLib" to be created
SET(LineMatchingLib_SRCS
    PairwiseLineMatching.cpp
    LineDescriptor.cpp
    EDLineDetector.cpp
    )
# header files to be installed
SET(LineMatchingLib_HEADER
    PairwiseLineMatching.hh
    LineDescriptor.hh
    EDLineDetector.hh
    LineStructure.hh
    )

ADD_LIBRARY(LineMatchingLib
    ${LineMatchingLib_SRCS}
    ${LineMatchingLib_HEADER})
TARGET_LINK_LIBRARIES(LineMatchingLib ${ARPACKPP_LIBRARIES} ${OpenCV_LIBS})

ADD_EXECUTABLE(TestLineMatchingAlgorithm TestLineMatchingAlgorithm.cpp)
TARGET_LINK_LIBRARIES(TestLineMatchingAlgorithm LineMatchingLib)
