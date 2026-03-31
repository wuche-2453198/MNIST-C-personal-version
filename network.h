#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct Network
{
    std::size_t input_size = 0;
    std::size_t hidden_size = 0;
    std::size_t output_size = 0;

    std::vector<std::vector<float>> w1;
    std::vector<float> b1;

    std::vector<std::vector<float>> w2;
    std::vector<float> b2;
};

struct ForwardResult
{
    std::vector<float> z1;
    std::vector<float> a1;
    std::vector<float> z2;
    std::vector<float> output;
};

Network create_network(std::size_t input_size, std::size_t hidden_size, std::size_t output_size);
float sigmoid(float x);
std::vector<float> apply_sigmoid(const std::vector<float> &z);
std::vector<float> linear(const std::vector<std::vector<float>> &w,
                          const std::vector<float> &x,
                          const std::vector<float> &b);
ForwardResult forward(const Network &net, const std::vector<float> &x);