cmake_minimum_required(VERSION 2.8)

PROJECT(LBDDescriptor)

SET(BUILD_SHARED_LIBS ON)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")# -Wall -Wextra -Wpedantic")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11")# -Wall -Wextra -Wpedantic")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

FIND_PACKAGE(ARPACKPP REQUIRED)
INCLUDE_DIRECTORIES(${ARPACKPP_INCLUDE_DIRS})

FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)

# source files of library "LineMatchingLib" to be created
SET(LineMatchingLib_SRCS
    src/PairwiseLineMatching.cpp
    src/LineDescriptor.cpp
    src/EDLineDetector.cpp
    )
# header files to be installed
SET(LineMatchingLib_HEADER
    include/lbd_descriptor/PairwiseLineMatching.hh
    include/lbd_descriptor/LineDescriptor.hh
    include/lbd_descriptor/EDLineDetector.hh
    include/lbd_descriptor/LineStructure.hh
    )

ADD_LIBRARY(lbddescriptor
    ${LineMatchingLib_SRCS}
    ${LineMatchingLib_HEADER})
TARGET_LINK_LIBRARIES(lbddescriptor
    ${ARPACKPP_LIBRARIES}
    opencv_core opencv_imgproc opencv_ximgproc
    )

ADD_EXECUTABLE(TestLineMatchingAlgorithm src/TestLineMatchingAlgorithm.cpp)
TARGET_LINK_LIBRARIES(TestLineMatchingAlgorithm lbddescriptor)

install(TARGETS lbddescriptor LIBRARY DESTINATION lib)
install(DIRECTORY include/lbd_descriptor DESTINATION include)
