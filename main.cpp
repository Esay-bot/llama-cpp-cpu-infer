#include <iostream>
#include "LlmInfer.h"
extern "C" void llama_backend_free();

int main()
{
    LlmInfer infer;
    std::string modelPath = "/home/stu/code/llama.cpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    if (!infer.loadModel(modelPath, 2048, 8, 0.7f))
    {
        std::cerr << "[FATAL] 模型加载失败\n";
        return -1;
    }

    std::string q = "你好，请简单介绍一下大语言模型";
    std::cout << "提问：" << q << "\n回答：";

    // 方案B：缓冲输出，遇到标点一次性打印，消除单字蹦的观感
    std::string printBuf;
    std::string fullResult = infer.chat(q, [&printBuf](const std::string& piece) {
        printBuf += piece;
        // 遇到中文标点、换行就刷新输出
        size_t pos = piece.find_first_of("。，！？；：\n");
        if (pos != std::string::npos)
        {
            std::cout << printBuf << std::flush;
            printBuf.clear();
        }
    }, 512);

    // 输出缓冲区剩余文字
    if (!printBuf.empty())
    {
        std::cout << printBuf << std::flush;
    }

    std::cout << "\n推理完成\n";
    llama_backend_free();
    return 0;
}