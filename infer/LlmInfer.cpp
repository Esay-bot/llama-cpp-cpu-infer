#include "LlmInfer.h"
#include <iostream>
#include <cstring>

static bool g_llama_backend_inited = false;

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
    if (m_sampler) {
        llama_sampler_free(m_sampler);
        m_sampler = nullptr;
    }
    if (m_ctx) {
        llama_free(m_ctx);
        m_ctx = nullptr;
    }
    if (m_model) {
        llama_model_free(m_model);
        m_model = nullptr;
    }
}

bool LlmInfer::loadModel(const std::string& modelPath, int ctxLen, int nThreads, float temp)
{
    if (m_sampler) { llama_sampler_free(m_sampler); m_sampler = nullptr; }
    if (m_ctx) { llama_free(m_ctx); m_ctx = nullptr; }
    if (m_model) { llama_model_free(m_model); m_model = nullptr; }

    llama_model_params model_params = llama_model_default_params();
    m_model = llama_model_load_from_file(modelPath.c_str(), model_params);
    if (!m_model)
    {
        std::cerr << "[ERROR] 模型加载失败: " << modelPath << std::endl;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = ctxLen;
    ctx_params.n_threads = nThreads;
    m_ctx = llama_init_from_model(m_model, ctx_params);
    if (!m_ctx)
    {
        std::cerr << "[ERROR] 创建上下文失败" << std::endl;
        llama_model_free(m_model);
        m_model = nullptr;
        return false;
    }

    m_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(m_sampler, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_dist(1234));

    std::cout << "[INFO] 模型加载成功 | ctx_len=" << ctxLen << " threads=" << nThreads << " temp=" << temp << std::endl;
    return true;
}

std::string LlmInfer::buildQwenPrompt(const std::string& userMsg)
{
    return "<|im_start|>user\n" + userMsg + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string LlmInfer::chat(const std::string& userPrompt, StreamCallback callback, int maxGenToken)
{
    if (!m_model || !m_ctx || !m_sampler)
    {
        std::cerr << "[ERROR] 模型未加载" << std::endl;
        return "";
    }
    std::string fullResp;
    const llama_vocab* vocab = llama_model_get_vocab(m_model);

    std::string fullPrompt = buildQwenPrompt(userPrompt);
    std::vector<llama_token> tokens(fullPrompt.size() + 200);
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
        std::cerr << "[ERROR] tokenize失败" << std::endl;
        return "";
    }
    tokens.resize(token_cnt);

    llama_batch batch = llama_batch_init(tokens.size() + maxGenToken, 0, 1);
    batch.n_tokens = 0;
    for (int i = 0; i < token_cnt; i++)
    {
        const int idx = batch.n_tokens++;
        batch.token[idx] = tokens[i];
        batch.pos[idx] = i;
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = 0;
        // 关键修复：只有最后一个token输出logits
        batch.logits[idx] = (i == token_cnt - 1);
    }

    if (llama_decode(m_ctx, batch) != 0)
    {
        std::cerr << "[ERROR] 首次decode失败" << std::endl;
        llama_batch_free(batch);
        return "";
    }

    int cur_pos = batch.n_tokens;
    for (int i = 0; i < maxGenToken; i++)
    {
        llama_token new_tok = llama_sampler_sample(m_sampler, m_ctx, -1);
        if (llama_vocab_is_eog(vocab, new_tok)) break;

        char buf[256] = {0};
        int len = llama_token_to_piece(vocab, new_tok, buf, sizeof(buf)-1, 0, false);
        if (len > 0)
        {
            std::string piece(buf, len);
            fullResp += piece;
            callback(piece);
        }

        // 重置batch，单token，开启logits
        batch.n_tokens = 0;
        const int idx = batch.n_tokens++;
        batch.token[idx] = new_tok;
        batch.pos[idx] = cur_pos++;
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = 0;
        batch.logits[idx] = true;

        if (llama_decode(m_ctx, batch) != 0)
        {
            std::cerr << "[WARN] 生成中途decode中断" << std::endl;
            break;
        }
    }

    llama_batch_free(batch);
    return fullResp;
}