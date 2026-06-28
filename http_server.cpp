#include <iostream>
#include "infer/LlmInfer.h"
#include "../thirdparty/httplib.h"
#include <sstream>

// 全局单例推理对象
LlmInfer g_infer;

int main()
{
    // 初始化日志文件
    Logger::getInstance().setLogFile("./http_server.log");

    // 加载配置文件
    if (!g_infer.loadConfig("./config.json"))
    {
        LOG_ERROR("加载配置文件失败，HTTP服务程序退出");
        return -1;
    }
    // 加载模型
    if (!g_infer.reloadModel())
    {
        LOG_ERROR("模型加载失败，HTTP服务程序退出");
        return -1;
    }

    httplib::Server server;

    // 对话接口，兼容OpenAI返回格式
    server.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
        nlohmann::json req_body;
        try
        {
            req_body = nlohmann::json::parse(req.body);
        }
        catch (...)
        {
            res.status = 400;
            res.set_content(R"({"error": "请求JSON格式错误"})", "application/json");
            return;
        }

        // 清空历史对话
        g_infer.clearChatHistory();

        // 解析对话消息
        auto& msg_list = req_body["messages"];
        for (auto& msg : msg_list)
        {
            std::string role = msg["role"].get<std::string>();
            std::string content = msg["content"].get<std::string>();
            if (role == "user")
            {
                g_infer.appendUserMsg(content);
            }
            else if (role == "assistant")
            {
                g_infer.appendAssistantMsg(content);
            }
        }

        // 收集完整回答
        std::string answer;
        auto collect_cb = [&answer](const std::string& piece) {
            answer += piece;
        };

        InferResult result = g_infer.chat(collect_cb);

        // 构造标准OpenAI返回JSON
        nlohmann::json resp;
        resp["choices"][0]["message"]["content"] = answer;
        resp["usage"]["completion_tokens"] = result.total_gen_tokens;
        resp["usage"]["total_tokens"] = result.total_gen_tokens;

        res.set_content(resp.dump(), "application/json");

        std::string log_info = "请求处理完成，耗时：" + std::to_string(result.cost_ms) + "ms";
        LOG_INFO(log_info);
    });

    LOG_INFO("HTTP大模型推理服务启动成功，监听 0.0.0.0:8090");
    server.listen("0.0.0.0", 8090);

    return 0;
}