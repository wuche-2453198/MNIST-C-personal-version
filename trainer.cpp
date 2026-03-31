#include <vector>
#include "trainer.h"
#include "evaluator.h"

// 激活函数导数a到z
float sigmoid_derivative_from_activation(const float a)
{
    return a * (1 - a);
}
// 计算输出层delta
std::vector<float> compute_output_delta(const std::vector<float> &output, const std::vector<float> &target)
{
    std::vector<float> delta2;
    delta2.reserve(output.size());
    std::size_t delta_size = output.size();
    for (size_t i = 0; i < delta_size; i++)
    {
        delta2.push_back((output[i] - target[i]) * sigmoid_derivative_from_activation(output[i]));
    }
    return delta2;
}
// 计算隐藏层delta
std::vector<float> compute_hidden_delta(const std::vector<float> &delta2, const std::vector<std::vector<float>> &w2, const std::vector<float> &a1)
{
    std::vector<float> delta1;
    delta1.reserve(a1.size());
    std::size_t delta_size = w2.size();
    for (size_t j = 0; j < w2[0].size(); j++)
    {
        float back = 0;
        for (size_t i = 0; i < delta_size; i++)
        {
            back += delta2[i] * w2[i][j];
        }
        delta1.push_back(back * sigmoid_derivative_from_activation(a1[j]));
    }
    return delta1;
}
// 计算梯度
// dw1[16][784] dw2[10][16] delta1 16 delta2 10 image 784
Gradients compute_gradients(const std::vector<float> &delta2, const std::vector<float> &delta1, const std::vector<float> &a1, const std::vector<float> &x)
{
    Gradients gradients;
    // 初始化gradients
    gradients.db1 = delta1;
    gradients.db2 = delta2;

    size_t dw2_size = delta2.size(); // 10
    size_t dw1_size = delta1.size(); // 16
    size_t x_size = x.size();
    gradients.dW1.assign(dw1_size, std::vector<float>(x_size, 0.0f));
    gradients.dW2.assign(dw2_size, std::vector<float>(dw1_size, 0.0f));

    for (size_t i = 0; i < dw2_size; i++) // 10
    {
        for (size_t j = 0; j < dw1_size; j++) // 16
        {
            gradients.dW2[i][j] = a1[j] * delta2[i];
        }
    }

    for (size_t i = 0; i < dw1_size; i++) // 16
    {
        for (size_t j = 0; j < x_size; j++) // 784
        {
            gradients.dW1[i][j] = x[j] * delta1[i];
        }
    }
    return gradients;
}

// 更新参数
// w = w - lr * dw
// b = b - lr * db
void update_parameters(Network &net, const Gradients &grads, const float learning_rate)
{
    // 修改b
    for (size_t i = 0; i < net.b2.size(); i++)
    {
        net.b2[i] = net.b2[i] - learning_rate * grads.db2[i];
    }
    for (size_t i = 0; i < net.b1.size(); i++)
    {
        net.b1[i] = net.b1[i] - learning_rate * grads.db1[i];
    }

    // 修改w
    for (size_t i = 0; i < net.w2.size(); i++)
    {
        for (size_t j = 0; j < net.w2[i].size(); j++)
        {
            net.w2[i][j] = net.w2[i][j] - learning_rate * grads.dW2[i][j];
        }
    }

    for (size_t i = 0; i < net.w1.size(); i++)
    {
        for (size_t j = 0; j < net.w1[i].size(); j++)
        {
            net.w1[i][j] = net.w1[i][j] - learning_rate * grads.dW1[i][j];
        }
    }

    return;
}
// 训练单个模型 返回当前损失 前向传播+反向传播
float train_one_sample(Network &net, const float learning_rate, const Sample &sample)
{
    ForwardResult forward_result = forward(net, sample.image);

    std::vector<float> delta2 = compute_output_delta(forward_result.output, sample.target);
    std::vector<float> delta1 = compute_hidden_delta(delta2, net.w2, forward_result.a1);

    Gradients gradients = compute_gradients(delta2, delta1, forward_result.a1, sample.image);

    update_parameters(net, gradients, learning_rate);

    forward_result = forward(net, sample.image);
    return mse_loss(forward_result.output, sample.target);
}

// 整体循环用的单样本训练
void train_one_sample_in_epoch(Network &net, const float learning_rate, const Sample &sample)
{
    ForwardResult forward_result = forward(net, sample.image);

    std::vector<float> delta2 = compute_output_delta(forward_result.output, sample.target);
    std::vector<float> delta1 = compute_hidden_delta(delta2, net.w2, forward_result.a1);

    Gradients gradients = compute_gradients(delta2, delta1, forward_result.a1, sample.image);

    update_parameters(net, gradients, learning_rate);

    return;
}
//多样本训练
evaluator_result train_epoch(Network &net, const std::vector<Sample> &samples, float learning_rate)
{
    size_t size_of_samples = samples.size();
    if (size_of_samples == 0)
    {
        return {};
    }

    for (size_t i = 0; i < size_of_samples; i++)
    {
        train_one_sample_in_epoch(net, learning_rate, samples[i]);
    }

    return evaluate_network(net, samples);
}
// loss = 1/2求和(output - target)^2
// delta 是对各层的z求偏导 因为 z=wa+b，求出z偏导，可以一步求出w偏导和b偏导，从而轻易得出w和b的梯度
// loss'(z2) = loss'(output)*output'(z2)=(output - target)*sigmoid_derivative_from_activation

// hidden delta 就是loss对z1求导
// loss -> a2 -> z2 -> a1 -> z1
// z2 = w2a1+b2  a1=sigmoid(z1)
// loss'(z1) = loss'(z2)*sigmoid_derivative_from_activation(a1)*w2