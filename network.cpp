#include "network.h"
#include <iostream>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>
//初始化Network 
Network create_network(std::size_t input_size, std::size_t hidden_size, std::size_t output_size)
{
    //记录网络尺寸
    Network net{};
    net.input_size = input_size;
    net.hidden_size = hidden_size;
    net.output_size = output_size;
    //初始化权重和偏置为0
    net.w1.assign(hidden_size, std::vector<float>(input_size, 0.0f));
    net.b1.assign(hidden_size, 0.0f);
    net.w2.assign(output_size, std::vector<float>(hidden_size, 0.0f));
    net.b2.assign(output_size, 0.0f);
    //初始化权重和偏置为随机值

    //生成随机数
    std::mt19937 generator(std::random_device{}());
    //把底层引擎产生的随机状态，映射成一个在 [-0.1, 0.1] 附近均匀分布的浮点数。
    std::uniform_real_distribution<float> distribution(-0.1f, 0.1f);

    for (std::size_t i = 0; i < net.w1.size(); ++i)
    {
        for (std::size_t j = 0; j < net.w1[i].size(); ++j)
        {
            net.w1[i][j] = distribution(generator);
        }
    }

    for (std::size_t i = 0; i < net.w2.size(); ++i)
    {
        for (std::size_t j = 0; j < net.w2[i].size(); ++j)
        {
            net.w2[i][j] = distribution(generator);
        }
    }

    return net;
}

// 标量激活函数
float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

// 向量逐元素激活
std::vector<float> apply_sigmoid(const std::vector<float> &z)
{
    std::vector<float> a;
    a.reserve(z.size());
    for (std::size_t i = 0; i < z.size(); ++i)
    {
        a.push_back(sigmoid(z[i]));
    }
    return a;
}

// 矩阵乘向量并加偏置
std::vector<float> linear(const std::vector<std::vector<float>> &w,
                          const std::vector<float> &x,
                          const std::vector<float> &b)
{
    if (w.empty())
    {
        return {};
    }

    if (w.size() != b.size())
    {
        throw std::invalid_argument("linear: w and b size mismatch");
    }

    std::vector<float> z;
    z.reserve(w.size());

    for (std::size_t i = 0; i < w.size(); ++i)
    {
        if (w[i].size() != x.size())
        {
            throw std::invalid_argument("linear: w row size and x size mismatch");
        }

        float result = b[i];
        for (std::size_t j = 0; j < x.size(); ++j)
        {
            result += w[i][j] * x[j];
        }
        z.push_back(result);
    }

    return z;
}

ForwardResult forward(const Network &net, const std::vector<float> &x)
{
    if (x.size() != net.input_size)
    {
        throw std::invalid_argument("forward: input size mismatch");
    }

    ForwardResult result;

    result.z1 = linear(net.w1, x, net.b1);
    result.a1 = apply_sigmoid(result.z1);

    result.z2 = linear(net.w2, result.a1, net.b2);
    result.output = apply_sigmoid(result.z2);

    return result;
}