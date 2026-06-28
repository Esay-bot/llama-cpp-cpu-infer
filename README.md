# llama-cpp-cpu-infer
基于 llama.cpp 实现的 C++ 本地大模型 CPU 推理工程

## 项目简介
本项目基于 llama.cpp 开源框架，使用 C++17 面向对象方式封装大模型离线推理能力，从控制台本地对话程序迭代为兼容 OpenAI 协议的 HTTP 后端推理服务，实现大模型在普通 CPU 设备上的离线部署、多轮上下文对话、工程化日志管理、参数配置、接口化调用能力，适合轻量化私有化大模型部署场景。

## 技术栈
- 编程语言：C++17
- 构建工具：CMake 4.3.4
- 推理框架：llama.cpp
- 运行系统：Ubuntu Linux
- 编译器：GCC 9.4

## 项目目录结构
ai_llm_infer_demo/
├── infer/                # 核心推理封装模块
│   ├── LlmInfer.h        # 推理类头文件：模型加载、多轮对话、参数校验、资源释放
│   ├── LlmInfer.cpp
│   └── Log.hpp           # 三级日志模块：控制台+文件双输出
├── thirdparty/           # 第三方依赖库
│   ├── nlohmann/json.hpp # JSON配置解析库
│   └── httplib.h         # 轻量级HTTP服务库
├── main.cpp              # 控制台本地多轮对话入口
├── http_server.cpp       # HTTP推理服务入口
├── config.json           # 模型、推理超参数配置文件
├── CMakeLists.txt        # 项目编译构建脚本
└── README.md

## 前置环境准备
1. 拉取 llama.cpp 源码并编译，生成 `.so` 动态链接库；
2. 将 `llama.cpp/build/bin` 目录配置到系统动态链接器，执行 `sudo ldconfig` 更新动态库缓存；
3. 准备 GGUF 量化大模型，本项目测试使用 `qwen2.5-0.5b-instruct-q4_k_m.gguf`。

## 编译 & 运行指令
```bash
1. 环境依赖
操作系统：Ubuntu 20.04 / 22.04
C++ 标准：C++17
编译工具：CMake 3.16+、GCC
依赖：llama.cpp 编译库、nlohmann/json、cpp-httplib
2. 编译命令
# 清理旧构建文件
rm -rf build
mkdir build && cd build
cmake ..
make -j4
3. 方式 1：运行控制台本地对话程序
# 回到项目根目录执行
./build/main
4. 方式 2：启动 HTTP 推理服务
./build/http_server
5. 接口测试示例
curl http://127.0.0.1:8090/v1/chat/completions \
-H "Content-Type:application/json" \
-d '{
    "messages": [
        {"role":"user","content":"简单介绍大语言模型"}
    ]
}'

1. 基础大模型推理封装（阶段一）
基于 llama.cpp 新版 C API 封装 LlmInfer 推理类，隔离底层模型加载、Token 编码解码、采样、解码细节；
实现多轮上下文对话，自动拼接 Qwen 模型 Prompt 模板，支持历史对话记忆；
实现中文流式分段输出，按照标点符号缓冲推送回答内容，优化前端展示体验；
统计推理耗时、生成 Token 数量、每秒 Token 生成速率，用于推理性能分析；
解决 llama.cpp KV 缓存上下文复用崩溃问题，每次推理重建上下文规避内存越界异常。
2. 工程化优化（阶段二）
引入 nlohmann/json 库，通过 config.json 统一管理模型路径、上下文长度、线程数、温度、TopK/TopP 等超参数，消除代码硬编码，方便快速切换模型与调参；
自研三级日志组件（INFO/WARN/ERROR），支持控制台实时打印 + 本地日志文件持久化，方便线上问题定位；
所有入参做合法性校验，对非法参数、配置文件缺失、模型加载失败、Token 编码异常做友好异常捕获与日志提示；
封装资源安全释放逻辑，实现模型、上下文、KV 缓存、采样器自动释放，支持运行时动态重载模型；
支持 Valgrind 内存泄漏检测，保证服务长时间稳定运行无内存泄露。
3. HTTP 服务化部署（阶段三）
基于 cpp-httplib 轻量级 HTTP 库搭建后端服务，封装兼容 OpenAI 规范的对话接口 /v1/chat/completions；
使用全局单例管理大模型实例，全局只加载一次模型，避免重复加载大权重文件造成资源浪费；
接口请求 JSON 格式校验，非法请求返回标准错误 JSON，提升接口健壮性；
每次对话自动清空 / 拼接用户多轮会话上下文，接口返回标准 OpenAI 格式结果，可直接对接主流大模型前端调试工具；
记录每一次接口请求的推理耗时并持久化日志，方便服务运维、性能瓶颈排查。
