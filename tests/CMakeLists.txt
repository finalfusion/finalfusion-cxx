include(CTest)

file(COPY testdata/test.fifu DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)

add_executable(cpp_embedding_lookup embedding_lookup.cc)
target_link_libraries(cpp_embedding_lookup finalfusion_cxx)
add_test(cpp_embedding_lookup cpp_embedding_lookup)
