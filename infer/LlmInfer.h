#ifndef LLM_INFER_H
#define LLM_INFER_H
#include <string>
#include <vector>
#include <functional>

extern "C" {
#include "llama.h"
}

using StreamCallback = std::function<void(const std::string&)>;

class LlmInfer
{
public:
    LlmInfer();
    ~LlmInfer();
    bool loadModel(const std::string& modelPath, int ctxLen = 2048, int nThreads = 8, float temp = 0.7f);
    std::string chat(const std::string& userPrompt, StreamCallback callback, int maxGenToken = 512);
private:
    std::string buildQwenPrompt(const std::string& userMsg);
    llama_model*    m_model    = nullptr;
    llama_context*  m_ctx      = nullptr;
    llama_sampler*  m_sampler  = nullptr;
};
#endif