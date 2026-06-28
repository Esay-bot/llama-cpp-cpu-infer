#include <iostream>
#include "LlmInfer.h"
extern "C" void llama_backend_free();

int main()
{
    LlmInfer infer;
    InferParams params;
    params.ctx_len = 2048;
    params.n_threads = 8;
    params.temp = 0.7f;
    params.top_p = 0.9f;
    params.top_k = 40;
    params.repeat_penalty = 1.1f;
    params.max_gen_tokens = 512;

    std::string modelPath = "/home/stu/code/llama.cpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    if (!infer.loadModel(modelPath, params))
    {
        std::cerr << "[FATAL] 模型加载失败\n";
        return -1;
    }

    // 第一轮对话
    std::string q1 = "你好，请简单介绍一下大语言模型";
    std::cout << "\n===== 第一轮提问 =====\n提问：" << q1 << "\n回答：";
    infer.appendUserMsg(q1);
    std::string buf1;
    auto cb1 = [&buf1](const std::string& piece) {
        buf1 += piece;
        size_t pos = piece.find_first_of("。，！？；：\n");
        if (pos != std::string::npos)
        {
            std::cout << buf1 << std::flush;
            buf1.clear();
        }
    };
    InferResult res1 = infer.chat(cb1);
    if (!buf1.empty()) std::cout << buf1 << std::flush;
    std::cout << "\n[性能] 耗时:" << res1.cost_ms << "ms 生成token:" << res1.total_gen_tokens
              << " token/s:" << res1.token_per_sec << "\n";

    // 第二轮多轮上下文测试
    std::string q2 = "刚才你提到的大模型训练数据来源是什么？";
    std::cout << "\n===== 第二轮提问（携带上文） =====\n提问：" << q2 << "\n回答：";
    infer.appendUserMsg(q2);
    std::string buf2;
    auto cb2 = [&buf2](const std::string& piece) {
        buf2 += piece;
        size_t pos = piece.find_first_of("。，！？；：\n");
        if (pos != std::string::npos)
        {
            std::cout << buf2 << std::flush;
            buf2.clear();
        }
    };
    InferResult res2 = infer.chat(cb2);
    if (!buf2.empty()) std::cout << buf2 << std::flush;
    std::cout << "\n[性能] 耗时:" << res2.cost_ms << "ms 生成token:" << res2.total_gen_tokens
              << " token/s:" << res2.token_per_sec << "\n";

    // 测试清空历史
    infer.clearChatHistory();
    std::cout << "\n已清空对话历史\n";

    llama_backend_free();
    return 0;
}