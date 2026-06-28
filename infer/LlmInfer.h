#ifndef LLM_INFER_H
#define LLM_INFER_H
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include "Log.hpp"
#include "../thirdparty/nlohmann/json.hpp"

using json = nlohmann::json;

extern "C" {
#include "llama.h"
}

using StreamCallback = std::function<void(const std::string&)>;

enum class ChatRole {
    USER,
    ASSISTANT
};

struct ChatMsg {
    ChatRole role;
    std::string content;
};

struct InferParams {
    int ctx_len = 2048;
    int n_threads = 8;
    float temp = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    int max_gen_tokens = 512;
};

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

    bool loadModel(const std::string& modelPath, const InferParams& params);
    InferResult chat(StreamCallback callback);
    InferResult chatOnce(const std::string& userMsg, StreamCallback callback, const InferParams& params);

    void appendUserMsg(const std::string& msg);
    void appendAssistantMsg(const std::string& msg);
    void clearChatHistory();

    bool loadConfig(const std::string& configPath);
    bool reloadModel();
    InferParams getParams() const { return m_cur_params; }

private:
    std::string buildFullPrompt();
    bool checkInferParams(const InferParams& params);
    void freeAllResources();
    void rebuildSampler(const InferParams& params);

    llama_model*    m_model    = nullptr;
    llama_context*  m_ctx      = nullptr;
    llama_sampler*  m_sampler  = nullptr;

    InferParams m_cur_params;
    std::string m_model_path;
    std::vector<ChatMsg> m_chat_history;
    static bool g_llama_backend_inited;
};

#endif