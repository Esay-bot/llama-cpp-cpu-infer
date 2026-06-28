#ifndef LLM_INFER_H
#define LLM_INFER_H
#include <string>
#include <vector>
#include <functional>
#include <chrono>

extern "C" {
#include "llama.h"
}

// 流式回调
using StreamCallback = std::function<void(const std::string&)>;

// 对话角色枚举
enum class ChatRole {
    USER,
    ASSISTANT
};

// 单条对话消息结构体
struct ChatMsg {
    ChatRole role;
    std::string content;
};

// 统一推理参数结构体
struct InferParams {
    int ctx_len = 2048;
    int n_threads = 8;
    float temp = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    int max_gen_tokens = 512;
};

// 推理返回结果结构体（带性能指标）
struct InferResult {
    std::string answer;
    int total_gen_tokens = 0;
    long long cost_ms = 0;
    float token_per_sec = 0.0f;
};

class LlmInfer
{
public:
    LlmInfer();
    ~LlmInfer();

    // 加载模型，使用统一推理参数
    bool loadModel(const std::string& modelPath, const InferParams& params);

    // 多轮对话主接口
    InferResult chat(StreamCallback callback);
    // 兼容旧版单轮对话接口（不缓存历史）
    InferResult chatOnce(const std::string& userMsg, StreamCallback callback, const InferParams& params);

    // 多轮会话操作
    void appendUserMsg(const std::string& msg);
    void appendAssistantMsg(const std::string& msg);
    void clearChatHistory();

private:
    // 构建完整Qwen多轮prompt
    std::string buildFullPrompt();
    // 参数合法性校验
    bool checkInferParams(const InferParams& params);
    // 清空内部推理资源
    void freeAllResources();
    // 采样器重建
    void rebuildSampler(const InferParams& params);

    llama_model*    m_model    = nullptr;
    llama_context*  m_ctx      = nullptr;
    llama_sampler*  m_sampler  = nullptr;

    InferParams m_cur_params;
    std::vector<ChatMsg> m_chat_history;
    static bool g_llama_backend_inited;
};

#endif