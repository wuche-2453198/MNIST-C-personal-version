#include <vector>
#include <cstdint>
#include <iostream>
#include "evaluator.h"
#include "math_utils.h"

using namespace std;

// 根据前向传输的result 返回结果
size_t predict_label(const vector<float> &output)
{
    if (output.empty())
    {
        return 0;
    }

    float max_value = output[0];
    size_t max_index = 0;
    uint32_t output_size = output.size();
    for (uint32_t i = 0; i < output_size; i++)
    {
        if (output[i] > max_value)
        {
            max_value = output[i];
            max_index = i;
        }
    }
    return max_index;
}

// 单样本的正确判断
bool is_prediction_correct(const vector<float> &output, uint8_t label)
{
    if (predict_label(output) == label)
        return true;
    else
        return false;
}

// 单样本损失函数
float mse_loss(const vector<float> &output, const vector<float> &target)
{
    if (output.size() != target.size())
    {
        cerr << "输出和目标数量不匹配" << endl;
        return 0;
    }
    float loss = 0;
    size_t result_size = target.size();

    for (size_t i = 0; i < result_size; i++)
    {
        loss += 0.5f * squared_difference(output[i], target[i]);
    }
    return loss;
}

evaluator_result evaluate_network(const Network &net,
                                  const vector<Sample> &samples)
{
    vector<vector<float>> outputs;
    outputs.reserve(samples.size());

    for (size_t i = 0; i < samples.size(); i++)
    {
        ForwardResult result = forward(net, samples[i].image);
        outputs.push_back(result.output);
    }

    return evaluate_samples(outputs, samples);
}

evaluator_result evaluate_samples(const vector<vector<float>> &outputs,
                                  const vector<Sample> &samples)
{
    if (outputs.size() != samples.size())
    {
        cerr << "输出和样本数量不匹配" << endl;
        return {};
    }
    evaluator_result result;

    result.sample_count = samples.size();
    float total_loss = 0;
    for (size_t i = 0; i < result.sample_count; i++)
    {
        if (is_prediction_correct(outputs[i], samples[i].label))
        {
            result.correct_count++;
        }
        total_loss += mse_loss(outputs[i], samples[i].target);
    }

    result.accuracy = static_cast<float>(result.correct_count) / result.sample_count;
    result.average_loss = total_loss / result.sample_count;

    return result;
}