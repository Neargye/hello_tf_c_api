cmake_minimum_required(VERSION 3.6)

project(hello_tf)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# add an OpenCV dependency for the image processing
set( ENABLE_OPENCV ON )

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_VERBOSE_MAKEFILE ON)
    add_compile_options(-Wall -Wextra -pedantic-errors -Werror)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /WX)
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.11)
        add_compile_options(/permissive-)
    endif()
endif()

if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
    make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
endif()

if(UNIX AND NOT APPLE)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/lib/libtensorflow.so)
        file(
        DOWNLOAD
            https://github.com/Neargye/tensorflow/releases/download/v1.15.0/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz
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

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorflow)
        file(REMOVE ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz)
    endif()

elseif(APPLE)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
    endif()
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/lib/libtensorflow.so)
        file(
        DOWNLOAD
            https://github.com/Neargye/tensorflow/releases/download/v1.15.0/libtensorflow-cpu-darwin-x86_64-1.15.0.tar.gz ${CMAKE_SOURCE_DIR}/tensorflow/darwin.tar.gz
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

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_SOURCE_DIR}/tensorflow/darwin.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorflow)
        file(REMOVE ${CMAKE_SOURCE_DIR}/tensorflow/darwin.tar.gz)
    endif()

elseif(WIN32)
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
    endif()

    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/lib/tensorflow.dll)
        file(
        DOWNLOAD
            https://github.com/Neargye/tensorflow/releases/download/v1.15.0/libtensorflow-cpu-windows-x86_64-1.15.0.zip ${CMAKE_SOURCE_DIR}/tensorflow/windows.zip
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

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_SOURCE_DIR}/tensorflow/windows.zip WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorflow)
        file(REMOVE ${CMAKE_SOURCE_DIR}/tensorflow/windows.zip)
    endif()

    configure_file(tensorflow/lib/tensorflow.dll ${CMAKE_CURRENT_BINARY_DIR}/tensorflow.dll COPYONLY)
    configure_file(tensorflow/lib/tensorflow.dll ${CMAKE_CURRENT_BINARY_DIR}/test/tensorflow.dll COPYONLY)
endif()

if ( ENABLE_OPENCV )
    set(OpenCV_DIR D:/!Projects/opencv-4.1.1)
    set( opencv_components core highgui imgproc imgcodecs )
    find_package( OpenCV REQUIRED COMPONENTS ${opencv_components} )  
    include_directories( ${OpenCV_INCLUDE_DIRS} )    
endif()

include_directories(tensorflow/include)
include_directories(src/3rdparty/scope_guard/include)
link_directories(${CMAKE_SOURCE_DIR}/tensorflow/lib)

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

add_executable(create_tensor src/create_tensor.cpp)
target_link_libraries(create_tensor tensorflow)

add_executable(tensor_info src/tensor_info.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(tensor_info tensorflow)

add_executable(allocate_tensor src/allocate_tensor.cpp)
target_link_libraries(allocate_tensor tensorflow)

add_executable(batch_interface src/batch_interface.cpp src/tf_utils.cpp src/tf_utils.hpp)
target_link_libraries(batch_interface tensorflow)

configure_file(models/graph.pb ${CMAKE_CURRENT_BINARY_DIR}/graph.pb COPYONLY)


if ( ENABLE_OPENCV )
    # add an image processing example, requires OPENCV!
    add_executable(image_example src/image_example.cpp src/tf_utils.cpp src/tf_utils.hpp src/tf_image.cpp src/tf_image.hpp)
    target_link_libraries(image_example tensorflow ${OpenCV_LIBS})

    # add the image processing models:
    # * one for the "image in -> vector out"
    configure_file(models/graph_im2im.pb ${CMAKE_CURRENT_BINARY_DIR}/graph_im2im.pb COPYONLY)
    # * another for "image in -> image out"
    configure_file(models/graph_im2vec.pb ${CMAKE_CURRENT_BINARY_DIR}/graph_im2vec.pb COPYONLY)
    # also copy the sample image:
    configure_file(models/image.jpg ${CMAKE_CURRENT_BINARY_DIR}/image.jpg COPYONLY)    
endif()

enable_testing()
add_subdirectory(test)
