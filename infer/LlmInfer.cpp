#include "LlmInfer.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>

bool LlmInfer::g_llama_backend_inited = false;

LlmInfer::LlmInfer()
{
    if (!g_llama_backend_inited)
    {
        llama_backend_init();
        g_llama_backend_inited = true;
    }
}

LlmInfer::~LlmInfer()
{
    freeAllResources();
}

void LlmInfer::freeAllResources()
{
    if (m_sampler)
    {
        llama_sampler_free(m_sampler);
        m_sampler = nullptr;
    }
    if (m_ctx)
    {
        llama_free(m_ctx);
        m_ctx = nullptr;
    }
    if (m_model)
    {
        llama_model_free(m_model);
        m_model = nullptr;
    }
}

bool LlmInfer::checkInferParams(const InferParams& params)
{
    if (params.ctx_len <= 0)
    {
        LOG_ERROR(std::string("ctx_len 必须大于0"));
        return false;
    }
    if (params.n_threads <= 0)
    {
        LOG_ERROR(std::string("线程数必须大于0"));
        return false;
    }
    if (params.temp < 0.0f || params.temp > 1.5f)
    {
        LOG_ERROR(std::string("temp 范围建议0~1.5"));
        return false;
    }
    if (params.top_p < 0.0f || params.top_p > 1.0f)
    {
        LOG_ERROR(std::string("top_p 必须0~1"));
        return false;
    }
    if (params.top_k <= 0)
    {
        LOG_ERROR(std::string("top_k 必须大于0"));
        return false;
    }
    if (params.repeat_penalty < 1.0f)
    {
        LOG_ERROR(std::string("重复惩罚不能小于1.0"));
        return false;
    }
    if (params.max_gen_tokens <= 0)
    {
        LOG_ERROR(std::string("max_gen_tokens 必须大于0"));
        return false;
    }
    return true;
}

void LlmInfer::rebuildSampler(const InferParams& params)
{
    if (m_sampler)
    {
        llama_sampler_free(m_sampler);
        m_sampler = nullptr;
    }
    m_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(m_sampler, llama_sampler_init_temp(params.temp));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_top_k(params.top_k));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_top_p(params.top_p, 1));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_penalties(
        params.repeat_penalty,
        0.0f,
        0.0f,
        64
    ));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_dist(1234));
}

bool LlmInfer::loadConfig(const std::string& configPath)
{
    try
    {
        std::ifstream f(configPath);
        if (!f.is_open())
        {
            std::string err = "配置文件打开失败：" + configPath;
            LOG_ERROR(err);
            return false;
        }
        json data = json::parse(f);
        m_model_path = data["model_path"].get<std::string>();
        m_cur_params.ctx_len = data["ctx_len"];
        m_cur_params.n_threads = data["n_threads"];
        m_cur_params.temp = data["temperature"];
        m_cur_params.top_k = data["top_k"];
        m_cur_params.top_p = data["top_p"];
        m_cur_params.repeat_penalty = data["repeat_penalty"];
        m_cur_params.max_gen_tokens = data["max_gen_tokens"];

        if (!checkInferParams(m_cur_params))
        {
            return false;
        }
        LOG_INFO(std::string("配置文件加载成功"));
        return true;
    }
    catch (...)
    {
        LOG_ERROR(std::string("配置文件解析异常"));
        return false;
    }
}

bool LlmInfer::reloadModel()
{
    freeAllResources();
    return loadModel(m_model_path, m_cur_params);
}

bool LlmInfer::loadModel(const std::string& modelPath, const InferParams& params)
{
    if (!checkInferParams(params))
        return false;

    freeAllResources();
    m_cur_params = params;
    m_model_path = modelPath;

    llama_model_params model_params = llama_model_default_params();
    m_model = llama_model_load_from_file(modelPath.c_str(), model_params);
    if (!m_model)
    {
        std::string err = "模型加载失败: " + modelPath;
        LOG_ERROR(err);
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.ctx_len;
    ctx_params.n_threads = params.n_threads;
    m_ctx = llama_init_from_model(m_model, ctx_params);
    if (!m_ctx)
    {
        LOG_ERROR(std::string("创建上下文失败"));
        llama_model_free(m_model);
        m_model = nullptr;
        return false;
    }

    rebuildSampler(params);
    std::stringstream ss;
    ss << "模型加载成功 | ctx_len=" << params.ctx_len << " threads=" << params.n_threads;
    LOG_INFO(ss.str());
    return true;
}

void LlmInfer::appendUserMsg(const std::string& msg)
{
    m_chat_history.push_back({ChatRole::USER, msg});
}

void LlmInfer::appendAssistantMsg(const std::string& msg)
{
    m_chat_history.push_back({ChatRole::ASSISTANT, msg});
}

void LlmInfer::clearChatHistory()
{
    m_chat_history.clear();
    LOG_INFO(std::string("已清空对话历史"));
}

std::string LlmInfer::buildFullPrompt()
{
    std::string prompt;
    for (auto& msg : m_chat_history)
    {
        if (msg.role == ChatRole::USER)
        {
            prompt += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
        }
        else
        {
            prompt += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
        }
    }
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

InferResult LlmInfer::chat(StreamCallback callback)
{
    InferResult res;
    if (!m_model || !m_ctx || !m_sampler)
    {
        LOG_ERROR(std::string("模型未加载"));
        return res;
    }

    const llama_vocab* vocab = llama_model_get_vocab(m_model);
    std::string fullPrompt = buildFullPrompt();

    std::vector<llama_token> tokens(fullPrompt.size() + 512);
    int token_cnt = llama_tokenize(
        vocab,
        fullPrompt.c_str(),
        static_cast<int>(fullPrompt.size()),
        tokens.data(),
        static_cast<int>(tokens.size()),
        true,
        true
    );
    if (token_cnt <= 0)
    {
        LOG_ERROR(std::string("tokenize失败"));
        return res;
    }

    if (m_ctx != nullptr)
    {
        llama_free(m_ctx);
        m_ctx = nullptr;
    }
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = m_cur_params.ctx_len;
    ctx_params.n_threads = m_cur_params.n_threads;
    m_ctx = llama_init_from_model(m_model, ctx_params);
    if (!m_ctx)
    {
        LOG_ERROR(std::string("重建推理上下文失败"));
        return res;
    }

    if (token_cnt >= m_cur_params.ctx_len - 100)
    {
        LOG_WARN(std::string("上下文长度溢出，自动清空对话历史"));
        clearChatHistory();
        return chat(callback);
    }
    tokens.resize(token_cnt);

    auto start_time = std::chrono::steady_clock::now();
    llama_batch batch = llama_batch_init(tokens.size() + m_cur_params.max_gen_tokens, 0, 1);
    batch.n_tokens = 0;
    for (int i = 0; i < token_cnt; i++)
    {
        int idx = batch.n_tokens++;
        batch.token[idx] = tokens[i];
        batch.pos[idx] = i;
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = 0;
        batch.logits[idx] = (i == token_cnt - 1);
    }

    if (llama_decode(m_ctx, batch) != 0)
    {
        LOG_ERROR(std::string("首次decode失败"));
        llama_batch_free(batch);
        return res;
    }

    int cur_pos = batch.n_tokens;
    int gen_token_cnt = 0;
    for (int i = 0; i < m_cur_params.max_gen_tokens; i++)
    {
        llama_token new_tok = llama_sampler_sample(m_sampler, m_ctx, -1);
        if (llama_vocab_is_eog(vocab, new_tok)) break;

        char buf[256] = {0};
        int len = llama_token_to_piece(vocab, new_tok, buf, sizeof(buf) - 1, 0, false);
        if (len > 0)
        {
            std::string piece(buf, len);
            res.answer += piece;
            callback(piece);
            gen_token_cnt++;
        }

        batch.n_tokens = 0;
        int idx = batch.n_tokens++;
        batch.token[idx] = new_tok;
        batch.pos[idx] = cur_pos++;
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = 0;
        batch.logits[idx] = true;

        if (llama_decode(m_ctx, batch) != 0)
        {
            LOG_WARN(std::string("生成中途decode中断"));
            break;
        }
    }
    llama_batch_free(batch);

    auto end_time = std::chrono::steady_clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    res.cost_ms = ms;
    res.total_gen_tokens = gen_token_cnt;
    if (ms > 0)
        res.token_per_sec = gen_token_cnt * 1000.0f / ms;

    appendAssistantMsg(res.answer);
    return res;
}

InferResult LlmInfer::chatOnce(const std::string& userMsg, StreamCallback callback, const InferParams& params)
{
    InferResult res;
    if (!checkInferParams(params)) return res;
    if (!m_model || !m_ctx || !m_sampler)
    {
        LOG_ERROR(std::string("模型未加载"));
        return res;
    }

    const llama_vocab* vocab = llama_model_get_vocab(m_model);
    std::string prompt = "<|im_start|>user\n" + userMsg + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<llama_token> tokens(prompt.size() + 512);
    int token_cnt = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
    if (token_cnt <= 0) return res;

    if (m_ctx != nullptr)
    {
        llama_free(m_ctx);
        m_ctx = nullptr;
    }
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = m_cur_params.ctx_len;
    ctx_params.n_threads = m_cur_params.n_threads;
    m_ctx = llama_init_from_model(m_model, ctx_params);
    if (!m_ctx)
    {
        LOG_ERROR(std::string("chatOnce重建上下文失败"));
        return res;
    }

    auto start = std::chrono::steady_clock::now();
    llama_batch batch = llama_batch_init(token_cnt + params.max_gen_tokens, 0, 1);
    batch.n_tokens = 0;
    for (int i = 0; i < token_cnt; i++)
    {
        int idx = batch.n_tokens++;
        batch.token[idx] = tokens[i];
        batch.pos[idx] = i;
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = 0;
        batch.logits[idx] = (i == token_cnt - 1);
    }
    llama_decode(m_ctx, batch);
    int cur_pos = batch.n_tokens;
    int gen_cnt = 0;
    for (int i = 0; i < params.max_gen_tokens; i++)
    {
        llama_token t = llama_sampler_sample(m_sampler, m_ctx, -1);
        if (llama_vocab_is_eog(vocab, t)) break;
        char buf[256] = {0};
        int len = llama_token_to_piece(vocab, t, buf, 255, 0, false);
        if (len > 0)
        {
            std::string p(buf, len);
            res.answer += p;
            callback(p);
            gen_cnt++;
        }
        batch.n_tokens = 0;
        int idx = batch.n_tokens++;
        batch.token[idx] = t;
        batch.pos[idx] = cur_pos++;
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = 0;
        batch.logits[idx] = true;
        llama_decode(m_ctx, batch);
    }
    llama_batch_free(batch);
    auto end = std::chrono::steady_clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    res.cost_ms = ms;
    res.total_gen_tokens = gen_cnt;
    if (ms > 0) res.token_per_sec = gen_cnt * 1000.0f / ms;
    return res;
}