cmake_minimum_required(VERSION 3.10)
project(rapidLib)

if (MSVC)
    add_compile_options(/std:c++20)
else()
    add_compile_options(-std=c++20 -O2)   
endif()

# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# The version number
set (${PROJECT_NAME}_VERSION_MAJOR 2)
set (${PROJECT_NAME}_VERSION_MINOR 3)

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
	)

add_library(${PROJECT_NAME} SHARED ${RAPIDLIB_FULL_SRC})
add_executable(rapidLibTest test/rapidLibTest.cpp)

target_sources(${PROJECT_NAME} PRIVATE ${JSON_SRC})
target_sources(${PROJECT_NAME} PRIVATE ${BAYES_SRC})

#include(GenerateExportHeader)
#generate_export_header(${PROJECT_NAME})

target_link_libraries(rapidLibTest ${PROJECT_NAME})
target_link_libraries(rapidLibTest Threads::Threads)

enable_testing()
add_test(rapidLibTest rapidLibTest)
