#include <iostream>
#include "infer/LlmInfer.h"
extern "C" void llama_backend_free();

int main()
{
    Logger::getInstance().setLogFile("./infer.log");
    LlmInfer infer;

    if (!infer.loadConfig("./config.json"))
    {
        LOG_ERROR("加载配置文件失败，程序退出");
        return -1;
    }
    if (!infer.reloadModel())
    {
        LOG_ERROR("模型加载失败");
        return -1;
    }

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

    std::string q2 = "刚才你提到的大模型训练数据来源是什么？";
    std::cout << "\n===== 第二轮提问 =====\n提问：" << q2 << "\n回答：";
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

    infer.clearChatHistory();
    LOG_INFO("程序正常结束");
    llama_backend_free();
    return 0;
}