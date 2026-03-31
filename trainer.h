#pragma once

#include <vector>
#include "network.h"
#include "mnist_reader.h"
#include "evaluator.h"

// 梯度组合（单层隐藏层）
struct Gradients
{
    std::vector<std::vector<float>> dW1;
    std::vector<float> db1;
    std::vector<std::vector<float>> dW2;
    std::vector<float> db2;
};
float sigmoid_derivative_from_activation(const float a);
std::vector<float> compute_output_delta(const std::vector<float> &output, const std::vector<float> &target);
std::vector<float> compute_hidden_delta(const std::vector<float> &delta2, const std::vector<std::vector<float>> &w2, const std::vector<float> &a1);
Gradients compute_gradients(const std::vector<float> &delta2, const std::vector<float> &delta1, const std::vector<float> &a1, const std::vector<float> &x);
void update_parameters(Network &net, const Gradients &grads, const float learning_rate);
float train_one_sample(Network &net, const float learning_rate, const Sample &sample);
evaluator_result train_epoch(Network &net, const std::vector<Sample> &samples, float learning_rate);