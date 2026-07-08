configure_file("${PROJECT_SOURCE_DIR}/models/graph.pb" graph.pb COPYONLY)

add_executable(base.t test.cpp)
target_include_directories(base.t PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/doctest"
)
target_include_scope_guard(base.t)
target_compile_definitions(base.t PRIVATE DOCTEST_CONFIG_USE_STD_HEADERS)
add_test(NAME base.t COMMAND base.t)
target_link_libraries(base.t PRIVATE hello_tf_utils)

if(TARGET hello_tf)
  add_test(NAME hello_tf.t COMMAND hello_tf)
endif()

if(TARGET session_run)
  add_test(NAME session_run.t COMMAND session_run)
endif()

if(TARGET repeated_inference)
  add_test(NAME repeated_inference.t COMMAND repeated_inference)
endif()

if(TARGET load_graph)
  add_test(NAME load_graph.t COMMAND load_graph)
endif()

if(TARGET interface)
  add_test(NAME interface.t COMMAND interface)
endif()

if(TARGET graph_info)
  add_test(NAME graph_info.t COMMAND graph_info)
endif()

if(TARGET create_tensor)
  add_test(NAME create_tensor.t COMMAND create_tensor)
endif()

if(TARGET create_string_tensor)
  add_test(NAME create_string_tensor.t COMMAND create_string_tensor)
endif()

if(TARGET image_example)
  add_test(NAME image_example.t COMMAND image_example)
endif()

if(TARGET target_operation)
  add_test(NAME target_operation.t COMMAND target_operation)
endif()

if(HELLO_TF_HAS_OPENCV_IMAGE_FILE_EXAMPLE)
  add_test(NAME opencv_image_file_example.t COMMAND opencv_image_file_example)
endif()

if(TARGET tensor_info)
  add_test(NAME tensor_info.t COMMAND tensor_info)
endif()

if(TARGET allocate_tensor)
  add_test(NAME allocate_tensor.t COMMAND allocate_tensor)
endif()

if(TARGET batch_interface)
  add_test(NAME batch_interface.t COMMAND batch_interface)
endif()
