add_library(finalfusion_cxx Embeddings.cc)
target_include_directories(finalfusion_cxx PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(finalfusion_cxx finalfusion_static dl m pthread)