cmake_minimum_required(VERSION 3.10)

if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -lstdc++fs")

project(rapidlib C CXX)

# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# The version number
set (rapidmix_VERSION_MAJOR 2)
set (rapidmix_VERSION_MINOR 2)

# Main lib
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/dependencies/bayesfilter/src)

# RapidLib
file(GLOB RAPIDLIB_SRC "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB RAPIDLIB_DEP "${PROJECT_SOURCE_DIR}/dependencies/libsvm/libsvm.cpp")

# Third party
file(GLOB JSON_SRC "${PROJECT_SOURCE_DIR}/dependencies/jsoncpp.cpp")
file(GLOB BAYES_SRC "${PROJECT_SOURCE_DIR}/dependencies/bayesfilter/src/*.cpp")

# Set the source for the main library, using the groups defined above
set(RAPIDLIB_FULL_SRC 
		${RAPIDLIB_SRC}
		${RAPIDLIB_DEP}
		${JSON_SRC}
		${BAYES_SRC}
	)

add_library(RAPIDLIB SHARED ${RAPIDLIB_FULL_SRC})
add_executable(rapidLibTest test/rapidLibTest.cpp)

#include(GenerateExportHeader)
#generate_export_header(RAPIDLIB)

target_link_libraries(rapidLibTest RAPIDLIB)
target_link_libraries(rapidLibTest Threads::Threads)

enable_testing()
add_test(rapidLibTest rapidLibTest)