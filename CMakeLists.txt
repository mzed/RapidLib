cmake_minimum_required(VERSION 3.14)
project(rapidLib
	VERSION 2.5.0
	DESCRIPTION "A lightweight library for interactive machine learning"
	LANGUAGES CXX)

add_library(${PROJECT_NAME})
add_library(rapidLib::rapidLib ALIAS ${PROJECT_NAME})

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME} EXPORT_FILE_NAME include/rapidLib/rapidLib_export.h)
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME RapidLib)
configure_file(include/rapidLib/version.h.in include/rapidLib/version.h @ONLY)

set(PUBLIC_HEADERS
	include/rapidLib/rapidLib.h
	include/rapidLib/classification.h
	include/rapidLib/modelSet.h
	include/rapidLib/regression.h
	include/rapidLib/seriesClassification.h
	include/rapidLib//trainingExample.h)

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
add_subdirectory(src)
add_subdirectory(dependencies)

list(APPEND public_headers
	"include/rapidLib/rapidLib_export.h")
list(APPEND SOURCES
	"include/rapidLib/rapidLib_export.h")

target_sources(${PROJECT_NAME} PRIVATE ${PUBLIC_HEADERS} ${THIRD_PARTY} ${SOURCES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${PUBLIC_HEADERS} ${THIRD_PARTY} ${SOURCES})

list(APPEND PUBLIC_HEADERS
	"${CMAKE_CURRENT_BINARY_DIR}/include/rapidLib/version.h")

target_compile_definitions(${PROJECT_NAME} PUBLIC "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:RAPIDLIB_STATIC_DEFINE>")

target_include_directories(${PROJECT_NAME}
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")

set_target_properties(${PROJECT_NAME} PROPERTIES
    PUBLIC_HEADER "${PUBLIC_HEADERS}"
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

# testing
add_executable(rapidLibTest test/rapidLibTest.cpp)
set_property(TARGET rapidLibTest PROPERTY CXX_STANDARD 20)

target_link_libraries(rapidLibTest ${PROJECT_NAME})
target_link_libraries(rapidLibTest Threads::Threads)

enable_testing()
add_test(rapidLibTest rapidLibTest)
