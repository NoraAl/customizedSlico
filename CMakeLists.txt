cmake_minimum_required(VERSION 2.8)

project( slic )


find_package( OpenCV 3.3.0 REQUIRED )



include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( slic slicEx.cpp _slicAlgorithm.cpp)

target_link_libraries( slic ${OpenCV_LIBS})


