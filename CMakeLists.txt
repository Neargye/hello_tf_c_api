cmake_minimum_required(VERSION 3.20)

project(hello_tf LANGUAGES CXX)

set(TENSORFLOW_VERSION "2.21.0" CACHE STRING "TensorFlow Python wheel version.")
set(HELLO_TF_DEFAULT_TENSORFLOW_ROOT "${CMAKE_BINARY_DIR}/_deps/tensorflow")
set(TENSORFLOW_ROOT "${HELLO_TF_DEFAULT_TENSORFLOW_ROOT}" CACHE PATH "Local TensorFlow wheel extraction directory.")
set(TENSORFLOW_PIP_TARGET "${TENSORFLOW_ROOT}/python")
set(TENSORFLOW_PACKAGE_DIR "${TENSORFLOW_PIP_TARGET}/tensorflow")
option(HELLO_TF_FETCH_TENSORFLOW "Download the TensorFlow Python wheel into TENSORFLOW_ROOT when it is missing." ON)
option(HELLO_TF_BUILD_EXAMPLES "Build TensorFlow C API example executables." ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_VERBOSE_MAKEFILE ON)
    add_compile_options(-Wall -Wextra -pedantic-errors -Werror)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /WX)
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.11)
        add_compile_options(/permissive-)
    endif()
endif()
add_compile_definitions(SCOPE_GUARD_NO_THROW_ACTION)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(TENSORFLOW_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")

get_filename_component(HELLO_TF_DEFAULT_TENSORFLOW_ROOT_ABS "${HELLO_TF_DEFAULT_TENSORFLOW_ROOT}" ABSOLUTE)
get_filename_component(TENSORFLOW_ROOT_ABS "${TENSORFLOW_ROOT}" ABSOLUTE)
set(HELLO_TF_TENSORFLOW_ROOT_IS_BUILD_CACHE OFF)
if(TENSORFLOW_ROOT_ABS STREQUAL HELLO_TF_DEFAULT_TENSORFLOW_ROOT_ABS)
    set(HELLO_TF_TENSORFLOW_ROOT_IS_BUILD_CACHE ON)
endif()

set(TENSORFLOW_INCLUDE_DIR "${TENSORFLOW_PACKAGE_DIR}/include")
set(TENSORFLOW_C_API_HEADER "${TENSORFLOW_INCLUDE_DIR}/tensorflow/c/c_api.h")
set(TENSORFLOW_VERSION_STAMP "${TENSORFLOW_PIP_TARGET}/.tensorflow-${TENSORFLOW_VERSION}")

function(hello_tf_fetch_tensorflow_wheel)
    if(NOT HELLO_TF_FETCH_TENSORFLOW)
        message(FATAL_ERROR "TensorFlow ${TENSORFLOW_VERSION} not found at ${TENSORFLOW_PIP_TARGET}. Configure with -DHELLO_TF_FETCH_TENSORFLOW=ON or provide -DTENSORFLOW_ROOT=<path>.")
    endif()

    if(NOT HELLO_TF_TENSORFLOW_ROOT_IS_BUILD_CACHE)
        message(FATAL_ERROR "Refusing to overwrite external TENSORFLOW_ROOT=${TENSORFLOW_ROOT}. Provide a valid TensorFlow extraction or use the default build-local TENSORFLOW_ROOT for auto-fetch.")
    endif()

    file(REMOVE_RECURSE "${TENSORFLOW_PIP_TARGET}")
    file(MAKE_DIRECTORY "${TENSORFLOW_PIP_TARGET}")

    execute_process(
        COMMAND
            "${TENSORFLOW_PYTHON_EXECUTABLE}" -m pip install
            --disable-pip-version-check
            --no-cache-dir
            --only-binary=:all:
            --no-deps
            --target "${TENSORFLOW_PIP_TARGET}"
            "tensorflow==${TENSORFLOW_VERSION}"
        RESULT_VARIABLE tensorflow_fetch_result
    )
    if(NOT tensorflow_fetch_result EQUAL 0)
        message(FATAL_ERROR "Failed to fetch TensorFlow ${TENSORFLOW_VERSION} wheel into ${TENSORFLOW_PIP_TARGET}")
    endif()

    if(NOT EXISTS "${TENSORFLOW_C_API_HEADER}")
        message(FATAL_ERROR "TensorFlow ${TENSORFLOW_VERSION} wheel fetched, but C API header was not found in ${TENSORFLOW_INCLUDE_DIR}")
    endif()

    file(WRITE "${TENSORFLOW_VERSION_STAMP}" "${TENSORFLOW_VERSION}\n")
endfunction()

set(HELLO_TF_TENSORFLOW_CACHE_NEEDS_FETCH OFF)
if(NOT EXISTS "${TENSORFLOW_VERSION_STAMP}")
    if(EXISTS "${TENSORFLOW_C_API_HEADER}" AND NOT HELLO_TF_TENSORFLOW_ROOT_IS_BUILD_CACHE)
        message(STATUS "Using TensorFlow from external TENSORFLOW_ROOT without version stamp: ${TENSORFLOW_ROOT}")
    else()
        set(HELLO_TF_TENSORFLOW_CACHE_NEEDS_FETCH ON)
    endif()
elseif(NOT EXISTS "${TENSORFLOW_C_API_HEADER}")
    if(HELLO_TF_TENSORFLOW_ROOT_IS_BUILD_CACHE AND HELLO_TF_FETCH_TENSORFLOW)
        message(STATUS "TensorFlow cache stamp exists, but C API header is missing; refreshing build-local cache")
        set(HELLO_TF_TENSORFLOW_CACHE_NEEDS_FETCH ON)
    else()
        message(FATAL_ERROR "TensorFlow cache stamp exists, but C API header was not found in ${TENSORFLOW_INCLUDE_DIR}")
    endif()
endif()

if(HELLO_TF_TENSORFLOW_CACHE_NEEDS_FETCH)
    hello_tf_fetch_tensorflow_wheel()
endif()

if(NOT EXISTS "${TENSORFLOW_C_API_HEADER}")
    message(FATAL_ERROR "TensorFlow C API header not found in ${TENSORFLOW_INCLUDE_DIR}")
endif()

set(TENSORFLOW_LIBRARY_DIRS
    "${TENSORFLOW_PACKAGE_DIR}"
    "${TENSORFLOW_PACKAGE_DIR}/python"
)

function(hello_tf_find_tensorflow_libraries)
    unset(TENSORFLOW_LIBRARY)
    unset(TENSORFLOW_LIBRARY CACHE)
    unset(TENSORFLOW_FRAMEWORK_LIBRARY)
    unset(TENSORFLOW_FRAMEWORK_LIBRARY CACHE)
    unset(TENSORFLOW_RUNTIME_LIBRARY)
    unset(TENSORFLOW_RUNTIME_LIBRARY CACHE)
    unset(TENSORFLOW_FRAMEWORK_RUNTIME_LIBRARY)
    unset(TENSORFLOW_FRAMEWORK_RUNTIME_LIBRARY CACHE)
    unset(PYTHON_RUNTIME_LIBRARY)
    unset(PYTHON_RUNTIME_LIBRARY CACHE)

    if(WIN32)
        find_library(TENSORFLOW_LIBRARY
            NAMES tensorflow _pywrap_tensorflow_common tensorflow_cc
            PATHS ${TENSORFLOW_LIBRARY_DIRS}
            NO_DEFAULT_PATH
        )
        if(NOT TENSORFLOW_LIBRARY)
            find_file(TENSORFLOW_LIBRARY
                NAMES tensorflow.lib _pywrap_tensorflow_common.dll.if.lib tensorflow_cc.lib tensorflow_cc.2.dll.if.lib
                PATHS ${TENSORFLOW_LIBRARY_DIRS}
                NO_DEFAULT_PATH
            )
        endif()
        find_file(TENSORFLOW_FRAMEWORK_LIBRARY
            NAMES tensorflow_framework.lib tensorflow_framework.2.dll.if.lib
            PATHS ${TENSORFLOW_LIBRARY_DIRS}
            NO_DEFAULT_PATH
        )
        find_file(TENSORFLOW_RUNTIME_LIBRARY
            NAMES tensorflow.dll _pywrap_tensorflow_common.dll tensorflow_cc.dll tensorflow_cc.2.dll
            PATHS ${TENSORFLOW_LIBRARY_DIRS}
            NO_DEFAULT_PATH
        )
        find_file(TENSORFLOW_FRAMEWORK_RUNTIME_LIBRARY
            NAMES tensorflow_framework.dll tensorflow_framework.2.dll
            PATHS ${TENSORFLOW_LIBRARY_DIRS}
            NO_DEFAULT_PATH
        )
        get_filename_component(TENSORFLOW_PYTHON_DIR "${TENSORFLOW_PYTHON_EXECUTABLE}" DIRECTORY)
        set(PYTHON_RUNTIME_LIBRARY_NAMES
            "python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.dll"
            python312.dll
            python311.dll
            python310.dll
        )
        list(REMOVE_DUPLICATES PYTHON_RUNTIME_LIBRARY_NAMES)
        find_file(PYTHON_RUNTIME_LIBRARY
            NAMES ${PYTHON_RUNTIME_LIBRARY_NAMES}
            PATHS "${TENSORFLOW_PYTHON_DIR}"
            NO_DEFAULT_PATH
        )
    elseif(APPLE)
        find_library(TENSORFLOW_LIBRARY
            NAMES tensorflow libtensorflow.dylib tensorflow_cc libtensorflow_cc.2.dylib tensorflow_framework libtensorflow_framework.2.dylib
            PATHS ${TENSORFLOW_LIBRARY_DIRS}
            NO_DEFAULT_PATH
        )
        if(NOT TENSORFLOW_LIBRARY)
            find_file(TENSORFLOW_LIBRARY
                NAMES libtensorflow.dylib _pywrap_tensorflow_common.so _pywrap_tensorflow_internal.so libtensorflow_cc.2.dylib libtensorflow_framework.2.dylib
                PATHS ${TENSORFLOW_LIBRARY_DIRS}
                NO_DEFAULT_PATH
            )
        endif()
        find_file(TENSORFLOW_FRAMEWORK_LIBRARY
            NAMES libtensorflow_framework.2.dylib libtensorflow_framework.dylib
            PATHS ${TENSORFLOW_LIBRARY_DIRS}
            NO_DEFAULT_PATH
        )
    else()
        find_library(TENSORFLOW_LIBRARY
            NAMES tensorflow libtensorflow.so tensorflow_cc libtensorflow_cc.so.2 tensorflow_framework libtensorflow_framework.so.2
            PATHS ${TENSORFLOW_LIBRARY_DIRS}
            NO_DEFAULT_PATH
        )
        if(NOT TENSORFLOW_LIBRARY)
            find_file(TENSORFLOW_LIBRARY
                NAMES libtensorflow.so _pywrap_tensorflow_common.so _pywrap_tensorflow_internal.so libtensorflow_cc.so.2 libtensorflow_framework.so.2
                PATHS ${TENSORFLOW_LIBRARY_DIRS}
                NO_DEFAULT_PATH
            )
        endif()
        find_file(TENSORFLOW_FRAMEWORK_LIBRARY
            NAMES libtensorflow_framework.so.2 libtensorflow_framework.so
            PATHS ${TENSORFLOW_LIBRARY_DIRS}
            NO_DEFAULT_PATH
        )
    endif()

    set(TENSORFLOW_LIBRARY "${TENSORFLOW_LIBRARY}" PARENT_SCOPE)
    set(TENSORFLOW_FRAMEWORK_LIBRARY "${TENSORFLOW_FRAMEWORK_LIBRARY}" PARENT_SCOPE)
    set(TENSORFLOW_RUNTIME_LIBRARY "${TENSORFLOW_RUNTIME_LIBRARY}" PARENT_SCOPE)
    set(TENSORFLOW_FRAMEWORK_RUNTIME_LIBRARY "${TENSORFLOW_FRAMEWORK_RUNTIME_LIBRARY}" PARENT_SCOPE)
    set(PYTHON_RUNTIME_LIBRARY "${PYTHON_RUNTIME_LIBRARY}" PARENT_SCOPE)
endfunction()

function(hello_tf_tensorflow_libraries_complete result)
    set(complete ON)
    if(NOT TENSORFLOW_LIBRARY)
        set(complete OFF)
    endif()
    if(WIN32)
        if(NOT TENSORFLOW_RUNTIME_LIBRARY)
            set(complete OFF)
        endif()
        if(TENSORFLOW_FRAMEWORK_LIBRARY AND NOT TENSORFLOW_FRAMEWORK_RUNTIME_LIBRARY)
            set(complete OFF)
        endif()
    endif()

    set(${result} ${complete} PARENT_SCOPE)
endfunction()

hello_tf_find_tensorflow_libraries()
hello_tf_tensorflow_libraries_complete(HELLO_TF_TENSORFLOW_LIBRARIES_COMPLETE)
if(NOT HELLO_TF_TENSORFLOW_LIBRARIES_COMPLETE AND HELLO_TF_TENSORFLOW_ROOT_IS_BUILD_CACHE AND HELLO_TF_FETCH_TENSORFLOW)
    message(STATUS "TensorFlow cache is missing native libraries; refreshing build-local cache")
    hello_tf_fetch_tensorflow_wheel()
    hello_tf_find_tensorflow_libraries()
    hello_tf_tensorflow_libraries_complete(HELLO_TF_TENSORFLOW_LIBRARIES_COMPLETE)
endif()

if(NOT TENSORFLOW_LIBRARY)
    message(FATAL_ERROR "TensorFlow native library not found in ${TENSORFLOW_LIBRARY_DIRS}")
endif()
if(WIN32 AND NOT TENSORFLOW_RUNTIME_LIBRARY)
    message(FATAL_ERROR "TensorFlow runtime DLL not found in ${TENSORFLOW_LIBRARY_DIRS}")
endif()
if(WIN32 AND TENSORFLOW_FRAMEWORK_LIBRARY AND NOT TENSORFLOW_FRAMEWORK_RUNTIME_LIBRARY)
    message(FATAL_ERROR "TensorFlow framework runtime DLL not found in ${TENSORFLOW_LIBRARY_DIRS}")
endif()

set(TENSORFLOW_LINK_LIBRARIES "${TENSORFLOW_LIBRARY}")
if(TENSORFLOW_FRAMEWORK_LIBRARY AND NOT TENSORFLOW_FRAMEWORK_LIBRARY STREQUAL TENSORFLOW_LIBRARY)
    list(APPEND TENSORFLOW_LINK_LIBRARIES "${TENSORFLOW_FRAMEWORK_LIBRARY}")
endif()
list(REMOVE_DUPLICATES TENSORFLOW_LINK_LIBRARIES)

message(STATUS "TensorFlow include dir: ${TENSORFLOW_INCLUDE_DIR}")
message(STATUS "TensorFlow link libraries: ${TENSORFLOW_LINK_LIBRARIES}")

get_filename_component(TENSORFLOW_LIBRARY_DIR "${TENSORFLOW_LIBRARY}" DIRECTORY)
set(CMAKE_BUILD_RPATH "${TENSORFLOW_LIBRARY_DIR}")

add_library(tensorflow SHARED IMPORTED GLOBAL)
set_target_properties(tensorflow PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TENSORFLOW_INCLUDE_DIR}"
)

if(WIN32)
    set_target_properties(tensorflow PROPERTIES IMPORTED_IMPLIB "${TENSORFLOW_LIBRARY}")
    if(TENSORFLOW_FRAMEWORK_LIBRARY)
        set_property(TARGET tensorflow APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${TENSORFLOW_FRAMEWORK_LIBRARY}")
    endif()
    if(TENSORFLOW_RUNTIME_LIBRARY)
        set_target_properties(tensorflow PROPERTIES IMPORTED_LOCATION "${TENSORFLOW_RUNTIME_LIBRARY}")
    else()
        message(FATAL_ERROR "TensorFlow runtime DLL not found in ${TENSORFLOW_LIBRARY_DIRS}")
    endif()
    set(TENSORFLOW_RUNTIME_LIBRARIES "${TENSORFLOW_RUNTIME_LIBRARY}")
    if(TENSORFLOW_FRAMEWORK_RUNTIME_LIBRARY)
        list(APPEND TENSORFLOW_RUNTIME_LIBRARIES "${TENSORFLOW_FRAMEWORK_RUNTIME_LIBRARY}")
    endif()
    if(PYTHON_RUNTIME_LIBRARY)
        list(APPEND TENSORFLOW_RUNTIME_LIBRARIES "${PYTHON_RUNTIME_LIBRARY}")
    endif()
else()
    set_target_properties(tensorflow PROPERTIES IMPORTED_LOCATION "${TENSORFLOW_LIBRARY}")
    if(TENSORFLOW_FRAMEWORK_LIBRARY AND NOT TENSORFLOW_FRAMEWORK_LIBRARY STREQUAL TENSORFLOW_LIBRARY)
        set_property(TARGET tensorflow APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${TENSORFLOW_FRAMEWORK_LIBRARY}")
    endif()
endif()

if(WIN32)
    set(TENSORFLOW_RUNTIME_OUTPUT_DIRS
        "${CMAKE_BINARY_DIR}/$<CONFIG>"
        "${CMAKE_BINARY_DIR}/test/$<CONFIG>"
    )

    set(TENSORFLOW_RUNTIME_COPY_COMMANDS)
    foreach(tensorflow_runtime_output_dir ${TENSORFLOW_RUNTIME_OUTPUT_DIRS})
        list(APPEND TENSORFLOW_RUNTIME_COPY_COMMANDS
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${tensorflow_runtime_output_dir}"
        )
        foreach(tensorflow_runtime_library ${TENSORFLOW_RUNTIME_LIBRARIES})
            list(APPEND TENSORFLOW_RUNTIME_COPY_COMMANDS
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${tensorflow_runtime_library}"
                    "${tensorflow_runtime_output_dir}"
            )
        endforeach()
    endforeach()

    add_custom_target(tensorflow_runtime ${TENSORFLOW_RUNTIME_COPY_COMMANDS})
endif()

function(target_link_tensorflow target)
    target_link_libraries(${target} PRIVATE tensorflow)
    if(WIN32)
        add_dependencies(${target} tensorflow_runtime)
    endif()
endfunction()

function(target_include_scope_guard target)
    target_include_directories(${target} PRIVATE
        "${PROJECT_SOURCE_DIR}/src/3rdparty/scope_guard/include"
    )
endfunction()

function(add_tf_example target)
    add_executable(${target} ${ARGN})
    target_include_scope_guard(${target})
    target_link_tensorflow(${target})
endfunction()

set(TF_UTILS_SOURCES src/tf_utils.cpp src/tf_utils.hpp)

add_library(hello_tf_utils STATIC ${TF_UTILS_SOURCES})
target_include_directories(hello_tf_utils PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
)
target_compile_features(hello_tf_utils PUBLIC cxx_std_17)
target_include_scope_guard(hello_tf_utils)
target_link_libraries(hello_tf_utils PUBLIC tensorflow)
if(WIN32)
    add_dependencies(hello_tf_utils tensorflow_runtime)
endif()

function(add_tf_utils_example target)
    add_executable(${target} ${ARGN})
    target_include_scope_guard(${target})
    target_link_libraries(${target} PRIVATE hello_tf_utils)
endfunction()

if(HELLO_TF_BUILD_EXAMPLES)
    add_tf_example(hello_tf src/hello_tf.cpp)
    add_tf_utils_example(session_run src/session_run.cpp)
    add_tf_utils_example(repeated_inference src/repeated_inference.cpp)
    add_tf_example(load_graph src/load_graph.cpp)
    add_tf_utils_example(interface src/interface.cpp)
    add_tf_utils_example(graph_info src/graph_info.cpp)
    add_tf_example(create_tensor src/create_tensor.cpp)
    add_tf_utils_example(create_string_tensor src/create_string_tensor.cpp)
    add_tf_utils_example(image_example src/image_example.cpp)
    add_tf_utils_example(target_operation src/target_operation.cpp)
    add_tf_utils_example(tensor_info src/tensor_info.cpp)
    add_tf_example(allocate_tensor src/allocate_tensor.cpp)
    add_tf_utils_example(batch_interface src/batch_interface.cpp)

    find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)
    if(OpenCV_FOUND)
        message(STATUS "OpenCV found: ${OpenCV_VERSION}")
        add_tf_utils_example(opencv_image_file_example src/opencv_image_file_example.cpp)
        target_include_directories(opencv_image_file_example PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(opencv_image_file_example PRIVATE ${OpenCV_LIBS})
        set(HELLO_TF_HAS_OPENCV_IMAGE_FILE_EXAMPLE ON)
    else()
        message(STATUS "OpenCV not found; skipping opencv_image_file_example")
        set(HELLO_TF_HAS_OPENCV_IMAGE_FILE_EXAMPLE OFF)
    endif()
else()
    set(HELLO_TF_HAS_OPENCV_IMAGE_FILE_EXAMPLE OFF)
endif()

configure_file(models/graph.pb ${CMAKE_CURRENT_BINARY_DIR}/graph.pb COPYONLY)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(test)
endif()
