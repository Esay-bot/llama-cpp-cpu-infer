#include "LlmInfer.h"
#include <iostream>
#include <cstring>
#include <algorithm>

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
    // 二次释放防护
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

bool LlmInfer::checkInferParams(const InferParams& params)
{
    if (params.ctx_len <= 0) {
        std::cerr << "[ERROR] ctx_len 必须大于0" << std::endl;
        return false;
    }
    if (params.n_threads <= 0) {
        std::cerr << "[ERROR] 线程数必须大于0" << std::endl;
        return false;
    }
    if (params.temp < 0.0f || params.temp > 1.5f) {
        std::cerr << "[ERROR] temp 范围建议0~1.5" << std::endl;
        return false;
    }
    if (params.top_p < 0.0f || params.top_p > 1.0f) {
        std::cerr << "[ERROR] top_p 必须0~1" << std::endl;
        return false;
    }
    if (params.top_k <= 0) {
        std::cerr << "[ERROR] top_k 必须大于0" << std::endl;
        return false;
    }
    if (params.repeat_penalty < 1.0f) {
        std::cerr << "[ERROR] 重复惩罚不能小于1.0" << std::endl;
        return false;
    }
    if (params.max_gen_tokens <= 0) {
        std::cerr << "[ERROR] max_gen_tokens 必须大于0" << std::endl;
        return false;
    }
    return true;
}

void LlmInfer::rebuildSampler(const InferParams& params)
{
    if (m_sampler) {
        llama_sampler_free(m_sampler);
        m_sampler = nullptr;
    }
    m_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(m_sampler, llama_sampler_init_temp(params.temp));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_top_k(params.top_k));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_top_p(params.top_p, 1));
    // 新版4参数惩罚接口，无编译报错
    llama_sampler_chain_add(m_sampler, llama_sampler_init_penalties(
        params.repeat_penalty,
        0.0f,
        0.0f,
        64
    ));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_dist(1234));
}

bool LlmInfer::loadModel(const std::string& modelPath, const InferParams& params)
{
    if (!checkInferParams(params))
        return false;

    freeAllResources();
    m_cur_params = params;

    llama_model_params model_params = llama_model_default_params();
    m_model = llama_model_load_from_file(modelPath.c_str(), model_params);
    if (!m_model)
    {
        std::cerr << "[ERROR] 模型加载失败: " << modelPath << std::endl;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.ctx_len;
    ctx_params.n_threads = params.n_threads;
    m_ctx = llama_init_from_model(m_model, ctx_params);
    if (!m_ctx)
    {
        std::cerr << "[ERROR] 创建上下文失败" << std::endl;
        llama_model_free(m_model);
        m_model = nullptr;
        return false;
    }

    rebuildSampler(params);
    std::cout << "[INFO] 模型加载成功 | ctx_len=" << params.ctx_len
              << " threads=" << params.n_threads
              << " temp=" << params.temp
              << " top_p=" << params.top_p
              << " top_k=" << params.top_k
              << " repeat_penalty=" << params.repeat_penalty << std::endl;
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
        std::cerr << "[ERROR] 模型未加载" << std::endl;
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
        std::cerr << "[ERROR] tokenize失败" << std::endl;
        return res;
    }

    // 核心修复：销毁重建上下文，清空KV缓存，彻底解决pos不连续报错
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
        std::cerr << "[ERROR] 重建推理上下文失败" << std::endl;
        return res;
    }

    // 上下文溢出自动清空历史
    if (token_cnt >= m_cur_params.ctx_len - 100)
    {
        std::cout << "[WARN] 上下文长度溢出，自动清空对话历史" << std::endl;
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
        std::cerr << "[ERROR] 首次decode失败" << std::endl;
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
        int len = llama_token_to_piece(vocab, new_tok, buf, sizeof(buf)-1, 0, false);
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
            std::cerr << "[WARN] 生成中途decode中断" << std::endl;
            break;
        }
    }
    llama_batch_free(batch);

    // 计算性能指标
    auto end_time = std::chrono::steady_clock::now();
    long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    res.cost_ms = ms;
    res.total_gen_tokens = gen_token_cnt;
    if (ms > 0)
        res.token_per_sec = gen_token_cnt * 1000.0f / ms;

    // 自动保存本轮assistant回答到历史
    appendAssistantMsg(res.answer);
    return res;
}

// 单轮兼容接口，不存入历史
InferResult LlmInfer::chatOnce(const std::string& userMsg, StreamCallback callback, const InferParams& params)
{
    InferResult res;
    if (!checkInferParams(params)) return res;
    if (!m_model || !m_ctx || !m_sampler)
    {
        std::cerr << "[ERROR] 模型未加载" << std::endl;
        return res;
    }

    const llama_vocab* vocab = llama_model_get_vocab(m_model);
    std::string prompt = "<|im_start|>user\n" + userMsg + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<llama_token> tokens(prompt.size() + 512);
    int token_cnt = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), true, true);
    if (token_cnt <= 0) return res;

    // 单轮同样重建上下文
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
        std::cerr << "[ERROR] chatOnce重建上下文失败" << std::endl;
        return res;
    }

    auto start = std::chrono::steady_clock::now();
    llama_batch batch = llama_batch_init(token_cnt + params.max_gen_tokens,0,1);
    batch.n_tokens=0;
    for(int i=0;i<token_cnt;i++){
        int idx=batch.n_tokens++;
        batch.token[idx]=tokens[i];
        batch.pos[idx]=i;
        batch.n_seq_id[idx]=1;
        batch.seq_id[idx][0]=0;
        batch.logits[idx]=(i==token_cnt-1);
    }
    llama_decode(m_ctx,batch);
    int cur_pos=batch.n_tokens;
    int gen_cnt=0;
    for(int i=0;i<params.max_gen_tokens;i++){
        llama_token t=llama_sampler_sample(m_sampler,m_ctx,-1);
        if(llama_vocab_is_eog(vocab,t)) break;
        char buf[256]={0};
        int len=llama_token_to_piece(vocab,t,buf,255,0,false);
        if(len>0){
            std::string p(buf,len);
            res.answer+=p;
            callback(p);
            gen_cnt++;
        }
        batch.n_tokens=0;
        int idx=batch.n_tokens++;
        batch.token[idx]=t;
        batch.pos[idx]=cur_pos++;
        batch.n_seq_id[idx]=1;
        batch.seq_id[idx][0]=0;
        batch.logits[idx]=true;
        llama_decode(m_ctx,batch);
    }
    llama_batch_free(batch);
    auto end=std::chrono::steady_clock::now();
    long long ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    res.cost_ms=ms;
    res.total_gen_tokens=gen_cnt;
    if(ms>0) res.token_per_sec=gen_cnt*1000.0f/ms;
    return res;
}