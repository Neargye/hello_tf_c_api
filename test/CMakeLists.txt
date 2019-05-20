include_directories(3rdparty/Catch2)

configure_file(${CMAKE_SOURCE_DIR}/models/graph.pb graph.pb COPYONLY)

add_executable(base.t test.cpp)
add_test(NAME base.t COMMAND base.t)
target_link_libraries(base.t tensorflow)

add_test(NAME hello_tf.t COMMAND hello_tf)

add_test(NAME session_run.t COMMAND session_run)

add_test(NAME load_graph.t COMMAND load_graph)

add_test(NAME interface.t COMMAND interface)

add_test(NAME graph_info.t COMMAND graph_info)

add_test(NAME create_tensor.t COMMAND create_tensor)

add_test(NAME tensor_info.t COMMAND tensor_info)

add_test(NAME allocate_tensor.t COMMAND allocate_tensor)

add_test(NAME batch_interface.t COMMAND batch_interface)
