cmake_minimum_required(VERSION 3.16)
project(llm_infer_demo)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${PROJECT_SOURCE_DIR}/infer
    /home/stu/code/llama.cpp/include
    /home/stu/code/llama.cpp/ggml/include
    /home/stu/code/llama.cpp/common
)

link_directories(
    /home/stu/code/llama.cpp/build/bin
    /home/stu/code/llama.cpp/build/common
    /home/stu/code/llama.cpp/build/vendor/cpp-httplib
)

aux_source_directory(infer INFER_SRC)
add_executable(main main.cpp ${INFER_SRC})

target_link_libraries(main
    llama
    ggml
    ggml-base
    ggml-cpu
    llama-common-base
    cpp-httplib
    pthread
    dl
    m
)