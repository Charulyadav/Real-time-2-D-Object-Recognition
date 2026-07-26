# Charul - 002535330
# Spring 2026
# CS 5330 Computer Vision
# Project 3: Real-time 2-D Object Recognition

cmake_minimum_required(VERSION 3.16)
project(Project3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(OpenCV_DIR "C:/Users/Charul/Downloads/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

set(SHARED_SOURCES
    src/threshold.cpp
    src/morphology.cpp
    src/regions.cpp
    src/csv_util.cpp
    src/classifier.cpp
    src/embedding.cpp
    src/dashboard.cpp
)

add_executable(orSystem src/orSystem.cpp ${SHARED_SOURCES})
target_link_libraries(orSystem ${OpenCV_LIBS})