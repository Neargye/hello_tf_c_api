cmake_minimum_required(VERSION 3.20)

project(hello_tf)

set(TENSORFLOW_VERSION "2.21.0" CACHE STRING "TensorFlow Python wheel version.")
set(TENSORFLOW_ROOT "${PROJECT_SOURCE_DIR}/tensorflow" CACHE PATH "Local TensorFlow wheel extraction directory.")
set(TENSORFLOW_PIP_TARGET "${TENSORFLOW_ROOT}/python")
set(TENSORFLOW_PACKAGE_DIR "${TENSORFLOW_PIP_TARGET}/tensorflow")

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

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(TENSORFLOW_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")

set(TENSORFLOW_VERSION_STAMP "${TENSORFLOW_PIP_TARGET}/.tensorflow-${TENSORFLOW_VERSION}")
if(NOT EXISTS "${TENSORFLOW_VERSION_STAMP}")
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
        RESULT_VARIABLE tensorflow_install_result
    )
    if(NOT tensorflow_install_result EQUAL 0)
        message(FATAL_ERROR "error installing TensorFlow ${TENSORFLOW_VERSION} wheel into ${TENSORFLOW_PIP_TARGET}")
    endif()

    file(WRITE "${TENSORFLOW_VERSION_STAMP}" "${TENSORFLOW_VERSION}\n")
endif()

set(TENSORFLOW_INCLUDE_DIR "${TENSORFLOW_PACKAGE_DIR}/include")
if(NOT EXISTS "${TENSORFLOW_INCLUDE_DIR}/tensorflow/c/c_api.h")
    message(FATAL_ERROR "TensorFlow C API header not found in ${TENSORFLOW_INCLUDE_DIR}")
endif()

set(TENSORFLOW_LIBRARY_DIRS
    "${TENSORFLOW_PACKAGE_DIR}"
    "${TENSORFLOW_PACKAGE_DIR}/python"
)

unset(TENSORFLOW_LIBRARY CACHE)
unset(TENSORFLOW_FRAMEWORK_LIBRARY CACHE)
unset(TENSORFLOW_RUNTIME_LIBRARY CACHE)
unset(TENSORFLOW_FRAMEWORK_RUNTIME_LIBRARY CACHE)

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
    find_file(PYTHON_RUNTIME_LIBRARY
        NAMES python312.dll python311.dll python310.dll
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

if(NOT TENSORFLOW_LIBRARY)
    message(FATAL_ERROR "TensorFlow native library not found in ${TENSORFLOW_LIBRARY_DIRS}")
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

function(target_link_tensorflow target)
    target_link_libraries(${target} PRIVATE tensorflow ${TENSORFLOW_LINK_LIBRARIES})
    if(WIN32)
        foreach(tensorflow_runtime_library ${TENSORFLOW_RUNTIME_LIBRARIES})
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${tensorflow_runtime_library}"
                    "$<TARGET_FILE_DIR:${target}>"
            )
        endforeach()
    endif()
endfunction()

function(add_tf_example target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE
        "${PROJECT_SOURCE_DIR}/src/3rdparty/scope_guard/include"
    )
    target_link_tensorflow(${target})
endfunction()

set(TF_UTILS_SOURCES src/tf_utils.cpp src/tf_utils.hpp)

add_tf_example(hello_tf src/hello_tf.cpp)
add_tf_example(session_run src/session_run.cpp ${TF_UTILS_SOURCES})
add_tf_example(load_graph src/load_graph.cpp)
add_tf_example(interface src/interface.cpp ${TF_UTILS_SOURCES})
add_tf_example(graph_info src/graph_info.cpp ${TF_UTILS_SOURCES})
add_tf_example(create_tensor src/create_tensor.cpp)
add_tf_example(create_string_tensor src/create_string_tensor.cpp ${TF_UTILS_SOURCES})
add_tf_example(image_example src/image_example.cpp ${TF_UTILS_SOURCES})
add_tf_example(tensor_info src/tensor_info.cpp ${TF_UTILS_SOURCES})
add_tf_example(allocate_tensor src/allocate_tensor.cpp)
add_tf_example(batch_interface src/batch_interface.cpp ${TF_UTILS_SOURCES})

find_package(OpenCV QUIET COMPONENTS core imgcodecs imgproc)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    add_tf_example(opencv_image_file_example src/opencv_image_file_example.cpp ${TF_UTILS_SOURCES})
    target_include_directories(opencv_image_file_example PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(opencv_image_file_example PRIVATE ${OpenCV_LIBS})
    set(HELLO_TF_HAS_OPENCV_IMAGE_FILE_EXAMPLE ON)
else()
    message(STATUS "OpenCV not found; skipping opencv_image_file_example")
    set(HELLO_TF_HAS_OPENCV_IMAGE_FILE_EXAMPLE OFF)
endif()

configure_file(models/graph.pb ${CMAKE_CURRENT_BINARY_DIR}/graph.pb COPYONLY)

enable_testing()
add_subdirectory(test)
