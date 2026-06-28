#ifndef LOG_HPP
#define LOG_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <mutex>

enum class LogLevel {
    INFO,
    WARN,
    ERROR
};

class Logger {
public:
    static Logger& getInstance() {
        static Logger ins;
        return ins;
    }

    void setLogFile(const std::string& path) {
        std::lock_guard<std::mutex> lock(m_mtx);
        if (m_ofs.is_open())
        {
            m_ofs.close();
        }
        m_ofs.open(path, std::ios::app);
    }

    // 只接收std::string类型，解决模板类型不匹配报错
    void log(LogLevel level, const std::string& msg)
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        std::string timeStr = getTime();
        std::string levelStr;
        switch (level)
        {
        case LogLevel::INFO:
            levelStr = "[INFO]";
            break;
        case LogLevel::WARN:
            levelStr = "[WARN]";
            break;
        case LogLevel::ERROR:
            levelStr = "[ERROR]";
            break;
        default:
            levelStr = "[UNKNOWN]";
            break;
        }
        std::string content = timeStr + " " + levelStr + " " + msg;
        std::cout << content << std::endl;
        if (m_ofs.is_open())
        {
            m_ofs << content << std::endl;
        }
    }

private:
    Logger() = default;
    ~Logger()
    {
        if (m_ofs.is_open())
        {
            m_ofs.close();
        }
    }
    std::mutex m_mtx;
    std::ofstream m_ofs;

    std::string getTime()
    {
        time_t now = time(nullptr);
        tm* t = localtime(&now);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", t);
        return std::string(buf);
    }
};

#define LOG_INFO(x) Logger::getInstance().log(LogLevel::INFO, x)
#define LOG_WARN(x) Logger::getInstance().log(LogLevel::WARN, x)
#define LOG_ERROR(x) Logger::getInstance().log(LogLevel::ERROR, x)

#endif