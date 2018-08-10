include_directories(3rdparty/Catch2)

configure_file(${CMAKE_SOURCE_DIR}/models/graph.pb graph.pb COPYONLY)

add_executable(base_test test.cpp)
add_test(NAME base_test COMMAND base_test)
target_link_libraries(base_test tensorflow)

add_test(NAME hello_tf_test COMMAND hello_tf)

add_test(NAME session_run_test COMMAND session_run)

add_test(NAME load_graph_test COMMAND load_graph)

add_test(NAME interface_test COMMAND interface)

add_test(NAME graph_info_test COMMAND graph_info)

add_test(NAME create_tensor_test COMMAND create_tensor)

add_test(NAME tensor_info_test COMMAND tensor_info)

add_test(NAME allocate_tensor_test COMMAND allocate_tensor)
