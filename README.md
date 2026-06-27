# llama-cpp-cpu-infer
基于 llama.cpp 实现的 C++ 本地大模型 CPU 推理工程

## 项目简介
本项目依托 llama.cpp 框架，使用 C++ 封装可复用的大模型推理类，实现 GGUF 量化格式大模型在 Linux CPU 环境下本地部署，支持流式文本对话输出。
完整记录了从环境搭建、llama.cpp 源码编译、头文件与动态库依赖配置，到模型加载、对话推理的全部踩坑过程，可用于 C++ 后端、大模型工程化方向简历项目展示。

## 技术栈
- 编程语言：C++17
- 构建工具：CMake 4.3.4
- 推理框架：llama.cpp
- 运行系统：Ubuntu Linux
- 编译器：GCC 9.4

## 项目目录结构
├── CMakeLists.txt # 项目编译配置文件

├── main.cpp # 程序入口，推理功能测试

└── infer

├── LlmInfer.h # 大模型推理封装类头文件

└── LlmInfer.cpp # 模型加载、流式推理具体实现

## 前置环境准备
1. 拉取 llama.cpp 源码并编译，生成 `.so` 动态链接库；
2. 将 `llama.cpp/build/bin` 目录配置到系统动态链接器，执行 `sudo ldconfig` 更新动态库缓存；
3. 准备 GGUF 量化大模型，本项目测试使用 `qwen2.5-0.5b-instruct-q4_k_m.gguf`。

## 编译 & 运行指令
```bash
# 创建构建目录
mkdir build && cd build
# CMake 配置项目
cmake ..
# 多线程编译
make -j4
# 运行推理程序
./main

## 已实现功能
1. 封装 `LlmInfer` 推理类，统一管理模型、上下文、采样器生命周期；
2. 支持 GGUF 量化大模型本地加载与 CPU 端推理；
3. 流式回调输出，逐段实时返回模型生成文本；
4. 程序退出时自动释放所有内存资源，避免内存泄漏。

## 待拓展功能
1. 实现多轮对话上下文记忆功能；
2. 基于 JSON 配置文件统一管理模型路径、温度、上下文长度等超参数；
3. 新增日志模块与全局异常捕获，提升程序健壮性；
4. 基于 cpp-httplib 封装 HTTP 接口，实现推理服务化部署。

## 项目踩坑记录
1. 本地重复拷贝 llama 头文件导致多重定义编译报错，仅使用源码官方头文件路径即可解决；
2. 新版 llama.cpp 默认将动态库输出在 `build/bin` 目录，需要正确配置库检索路径；
3. nlohmann json 第三方库存放于 `vendor/nlohmann`，路径配置错误会引发连锁编译报错。
