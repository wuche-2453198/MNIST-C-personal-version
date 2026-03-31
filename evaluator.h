#pragma once

#include <vector>
#include "mnist_reader.h"
#include "network.h"
#include <cstddef>

struct evaluator_result
{
    std::size_t sample_count = 0;
    std::size_t correct_count = 0;
    float accuracy = 0.0f;
    float average_loss = 0.0f;
};

evaluator_result evaluate_samples(const std::vector<std::vector<float>> &outputs,
                                  const std::vector<Sample> &samples);
evaluator_result evaluate_network(const Network &net,
                                  const std::vector<Sample> &samples);
float mse_loss(const std::vector<float> &output, const std::vector<float> &target);