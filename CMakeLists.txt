cmake_minimum_required(VERSION 3.10)
project(rapidlib)

# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# The version number
set (${PROJECT_NAME}_VERSION_MAJOR 2)
set (${PROJECT_NAME}_VERSION_MINOR 2)

# Main lib
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/dependencies/bayesfilter/src)

# RapidLib
file(GLOB ${PROJECT_NAME}_SRC "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB ${PROJECT_NAME}_DEP "${PROJECT_SOURCE_DIR}/dependencies/libsvm/libsvm.cpp")

# Third party
file(GLOB JSON_SRC "${PROJECT_SOURCE_DIR}/dependencies/jsoncpp.cpp")
file(GLOB BAYES_SRC "${PROJECT_SOURCE_DIR}/dependencies/bayesfilter/src/*.cpp")

# Set the source for the main library, using the groups defined above
set(${PROJECT_NAME}_FULL_SRC 
		${PROJECT_NAME}_SRC
		${PROJECT_NAME}_DEP
		${JSON_SRC}
		${BAYES_SRC}
	)

add_library(${PROJECT_NAME} SHARED ${PROJECT_NAME}_FULL_SRC)
add_executable(rapidLibTest test/rapidLibTest.cpp)

#include(GenerateExportHeader)
#generate_export_header(${PROJECT_NAME})

target_link_libraries(rapidLibTest ${PROJECT_NAME})
target_link_libraries(rapidLibTest Threads::Threads)

enable_testing()
add_test(rapidLibTest rapidLibTest)