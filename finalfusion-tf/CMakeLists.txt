execute_process(COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_lib(), end=\"\")" OUTPUT_VARIABLE LIBTENSORFLOW_FRAMEWORK ERROR_QUIET)
execute_process(COMMAND python -c "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()), end=\"\")" OUTPUT_VARIABLE TF_CFLAGS ERROR_QUIET)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${TF_CFLAGS} -DNDEBUG")

set(LIBTENSORFLOW_FRAMEWORK "${LIBTENSORFLOW_FRAMEWORK}/libtensorflow_framework.so.1")

add_library(libtensorflow_framework SHARED IMPORTED GLOBAL)
set_target_properties(libtensorflow_framework PROPERTIES
        IMPORTED_LOCATION ${LIBTENSORFLOW_FRAMEWORK})

add_library(finalfusion_tf lookup/ff_lookup.cc)
target_include_directories(finalfusion_tf PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(finalfusion_tf finalfusion_cxx)

add_library(finalfusion_tf_op SHARED kernel/ff_lookup_kernels.cc ops/ff_lookup_ops.cc)
target_include_directories(finalfusion_tf_op PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(finalfusion_tf_op finalfusion_tf libtensorflow_framework)
