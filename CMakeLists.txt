cmake_minimum_required( VERSION 2.8 )
project(knn_digitsClassfy)
set(CMAKE_BUILD_TYPE "release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
#####################dependence###################
#opencv
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV not found")
endif(NOT OpenCV_FOUND)
include_directories(${OpenCV_INCLUDE_DIRS})

set(3rdPart_LIBS 
    ${OpenCV_LIBS}
)
######################source and local include######
include_directories(${PROJECT_SOURCE_DIR}/include)
#add subdir
add_subdirectory(src)