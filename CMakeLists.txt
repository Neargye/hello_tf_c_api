cmake_minimum_required(VERSION 3.2)
include(CheckCXXCompilerFlag)
project(hello_tf)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    add_compile_options(-Wall -Wextra -pedantic)
    set(CMAKE_VERBOSE_MAKEFILE ON)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4)
    check_cxx_compiler_flag(/permissive HAS_PERMISSIVE_FLAG)
    if(HAS_PERMISSIVE_FLAG)
        add_compile_options(/permissive-)
    endif()
endif()

include_directories(tensorflow/include)

if(UNIX AND NOT APPLE)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/linux)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow/linux)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/linux/lib/libtensorflow.so)
        file(
        DOWNLOAD
            https://github.com/Neargye/tensorflow/releases/download/v1.8.0/libtensorflow-cpu-linux-x86_64-1.8.0.tar.gz ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz
        STATUS
            status
        LOG
            log
        )
        list(GET status 0 status_code)
        list(GET status 1 status_string)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "error downloading tensorflow lib: ${status_string}" "${log}")
        endif()

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorflow/linux)
        file(REMOVE ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz)
    endif()

    link_directories(${CMAKE_SOURCE_DIR}/tensorflow/linux/lib)
elseif(APPLE)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/darwin)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow/darwin)
    endif()
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/darwin/lib/libtensorflow.so)
        file(
        DOWNLOAD
            https://github.com/Neargye/tensorflow/releases/download/v1.8.0/libtensorflow-cpu-darwin-x86_64-1.8.0.tar.gz ${CMAKE_SOURCE_DIR}/tensorflow/darwin.tar.gz
        STATUS
            status
        LOG
            log
        )
        list(GET status 0 status_code)
        list(GET status 1 status_string)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "error downloading tensorflow lib: ${status_string}" "${log}")
        endif()

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_SOURCE_DIR}/tensorflow/darwin.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorflow/darwin)
        file(REMOVE ${CMAKE_SOURCE_DIR}/tensorflow/darwin.tar.gz)
    endif()

    link_directories(tensorflow/darwin/lib)
elseif(WIN32)
    link_directories(tensorflow/windows/lib)
    configure_file(tensorflow/windows/lib/tensorflow.dll ${CMAKE_CURRENT_BINARY_DIR}/tensorflow.dll COPYONLY)
endif()

add_executable(hello_tf src/hello_tf.cpp)
target_link_libraries(hello_tf tensorflow)

add_executable(session_run src/session_run.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(session_run tensorflow)

add_executable(load_graph src/load_graph.cpp)
target_link_libraries(load_graph tensorflow)

add_executable(interface src/interface.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(interface tensorflow)

add_executable(graph_info src/graph_info.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(graph_info tensorflow)

add_executable(creat_tensor src/creat_tensor.cpp)
target_link_libraries(creat_tensor tensorflow)

add_executable(tensor_info src/tensor_info.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(tensor_info tensorflow)

configure_file(models/graph.pb ${CMAKE_CURRENT_BINARY_DIR}/graph.pb COPYONLY)

enable_testing()
add_subdirectory(test)
