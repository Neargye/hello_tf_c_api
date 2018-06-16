cmake_minimum_required(VERSION 3.2)
include(CheckCXXCompilerFlag)
project(hello_tf)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    add_compile_options(-Wall -Wextra -pedantic)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4)
    check_cxx_compiler_flag(/permissive HAS_PERMISSIVE_FLAG)
    if(HAS_PERMISSIVE_FLAG)
        add_compile_options(/permissive-)
    endif()
endif()

include_directories(tensorflow/include)

link_directories(tensorflow/lib)

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

configure_file(models/graph.pb ${CMAKE_CURRENT_BINARY_DIR}/graph.pb COPYONLY)
configure_file(tensorflow/lib/tensorflow.dll ${CMAKE_CURRENT_BINARY_DIR}/tensorflow.dll COPYONLY)
