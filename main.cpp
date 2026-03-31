#include <algorithm>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mnist_reader.h"
#include "network.h"
#include "trainer.h"

using namespace std;

char pixel_to_char(float value)
{
    if (value > 0.85f)
    {
        return '#';
    }
    if (value > 0.60f)
    {
        return 'O';
    }
    if (value > 0.35f)
    {
        return '*';
    }
    if (value > 0.15f)
    {
        return '.';
    }
    return ' ';
}

void print_ascii_image(const vector<float> &image, uint32_t n_rows, uint32_t n_cols)
{
    for (uint32_t r = 0; r < n_rows; ++r)
    {
        for (uint32_t c = 0; c < n_cols; ++c)
        {
            cout << pixel_to_char(image[r * n_cols + c]);
        }
        cout << endl;
    }
}

void print_label_preview(const vector<uint8_t> &labels, size_t preview_count)
{
    for (size_t i = 0; i < preview_count; ++i)
    {
        cout << setw(3) << static_cast<int>(labels[i]);
        if ((i + 1) % 10 == 0 || i + 1 == preview_count)
        {
            cout << endl;
        }
    }
}

void print_float_vector(const vector<float> &values, size_t preview_count)
{
    cout << fixed << setprecision(4);
    for (size_t i = 0; i < preview_count && i < values.size(); ++i)
    {
        cout << setw(8) << values[i];
    }
    cout << endl;
}

string normalize_input_path(const string &raw_path)
{
    const size_t begin = raw_path.find_first_not_of(" \t\r\n");
    if (begin == string::npos)
    {
        return "";
    }

    const size_t end = raw_path.find_last_not_of(" \t\r\n");
    string normalized = raw_path.substr(begin, end - begin + 1);

    if (normalized.size() >= 2 && normalized.front() == '"' && normalized.back() == '"')
    {
        normalized = normalized.substr(1, normalized.size() - 2);
    }

    return normalized;
}

size_t argmax_index(const vector<float> &values)
{
    if (values.empty())
    {
        return 0;
    }

    return static_cast<size_t>(max_element(values.begin(), values.end()) - values.begin());
}

bool read_dataset_paths(int argc, char *argv[], string &image_filename, string &label_filename)
{
    if (argc >= 2)
    {
        image_filename = argv[1];
    }
    else
    {
        cout << "请输入 MNIST 图像文件路径: ";
        if (!getline(cin, image_filename))
        {
            return false;
        }
    }

    if (argc >= 3)
    {
        label_filename = argv[2];
    }
    else
    {
        cout << "请输入 MNIST 标签文件路径: ";
        if (!getline(cin, label_filename))
        {
            return false;
        }
    }

    image_filename = normalize_input_path(image_filename);
    label_filename = normalize_input_path(label_filename);

    return !image_filename.empty() && !label_filename.empty();
}

bool read_train_test_dataset_paths(int argc, char *argv[],
                                   string &train_image_filename,
                                   string &train_label_filename,
                                   string &test_image_filename,
                                   string &test_label_filename)
{
    if (argc >= 5)
    {
        train_image_filename = argv[1];
        train_label_filename = argv[2];
        test_image_filename = argv[3];
        test_label_filename = argv[4];
    }
    else
    {
        cout << "请输入训练集图像文件路径: ";
        if (!getline(cin, train_image_filename))
        {
            return false;
        }

        cout << "请输入训练集标签文件路径: ";
        if (!getline(cin, train_label_filename))
        {
            return false;
        }

        cout << "请输入测试集图像文件路径: ";
        if (!getline(cin, test_image_filename))
        {
            return false;
        }

        cout << "请输入测试集标签文件路径: ";
        if (!getline(cin, test_label_filename))
        {
            return false;
        }
    }

    train_image_filename = normalize_input_path(train_image_filename);
    train_label_filename = normalize_input_path(train_label_filename);
    test_image_filename = normalize_input_path(test_image_filename);
    test_label_filename = normalize_input_path(test_label_filename);

    return !train_image_filename.empty() &&
           !train_label_filename.empty() &&
           !test_image_filename.empty() &&
           !test_label_filename.empty();
}

bool parse_size_t_value(const string &raw_value, size_t &value)
{
    string normalized = normalize_input_path(raw_value);
    if (normalized.empty())
    {
        return false;
    }

    try
    {
        size_t parsed_length = 0;
        unsigned long long parsed_value = stoull(normalized, &parsed_length);
        if (parsed_length != normalized.size())
        {
            return false;
        }
        value = static_cast<size_t>(parsed_value);
        return true;
    }
    catch (const exception &)
    {
        return false;
    }
}

bool read_size_t_input_with_default(const string &prompt, size_t default_value, size_t &value)
{
    cout << prompt << "（直接回车默认 " << default_value << "）: ";

    string input;
    if (!getline(cin, input))
    {
        return false;
    }

    input = normalize_input_path(input);
    if (input.empty())
    {
        value = default_value;
        return true;
    }

    if (!parse_size_t_value(input, value))
    {
        cerr << "请输入有效的非负整数。" << endl;
        return false;
    }

    return true;
}

bool read_train_test_mode_config(int argc, char *argv[],
                                 size_t &train_sample_count,
                                 size_t &test_sample_count,
                                 size_t &epoch_count)
{
    const size_t default_train_sample_count = 1000;
    const size_t default_test_sample_count = 1000;
    const size_t default_epoch_count = 100;

    if (argc >= 9 && string(argv[5]) == "5")
    {
        if (!parse_size_t_value(argv[6], train_sample_count) ||
            !parse_size_t_value(argv[7], test_sample_count) ||
            !parse_size_t_value(argv[8], epoch_count))
        {
            cerr << "mode 5 的训练样本数、测试样本数和 epoch 数必须是有效的非负整数。" << endl;
            return false;
        }
    }
    else if (argc >= 6 && string(argv[5]) == "5")
    {
        train_sample_count = default_train_sample_count;
        test_sample_count = default_test_sample_count;
        epoch_count = default_epoch_count;
    }
    else
    {
        if (!read_size_t_input_with_default("请输入训练样本数", default_train_sample_count, train_sample_count))
        {
            return false;
        }

        if (!read_size_t_input_with_default("请输入测试样本数", default_test_sample_count, test_sample_count))
        {
            return false;
        }

        if (!read_size_t_input_with_default("请输入训练 epoch 数", default_epoch_count, epoch_count))
        {
            return false;
        }
    }

    if (train_sample_count == 0 || test_sample_count == 0 || epoch_count == 0)
    {
        cerr << "训练样本数、测试样本数和 epoch 数必须大于 0。" << endl;
        return false;
    }

    return true;
}

void shuffle_samples(vector<Sample> &samples, mt19937 &generator)
{
    if (samples.size() < 2)
    {
        return;
    }

    shuffle(samples.begin(), samples.end(), generator);
}

void print_sample(const vector<float> &image, uint8_t label, uint32_t n_rows, uint32_t n_cols, size_t index, size_t total_count)
{
    cout << "sample_index: " << index << " / " << total_count - 1 << endl;
    cout << "label: " << static_cast<int>(label) << endl;

    auto minmax = minmax_element(image.begin(), image.end());
    cout << fixed << setprecision(4);
    cout << "min_pixel: " << *minmax.first << endl;
    cout << "max_pixel: " << *minmax.second << endl;

    cout << "ascii_preview:" << endl;
    print_ascii_image(image, n_rows, n_cols);
}

unsigned char debug_reader_mode(int argc, char *argv[])
{
    string image_filename;
    string label_filename;
    if (!read_dataset_paths(argc, argv, image_filename, label_filename))
    {
        return 1;
    }

    uint32_t image_magic_number = 0;
    uint32_t number_of_images = 0;
    uint32_t n_rows = 0;
    uint32_t n_cols = 0;
    if (!read_mnist_image_file_info(image_filename, image_magic_number, number_of_images, n_rows, n_cols))
    {
        return 1;
    }

    uint32_t label_magic_number = 0;
    uint32_t number_of_labels = 0;
    if (!read_mnist_label_file_info(label_filename, label_magic_number, number_of_labels))
    {
        return 1;
    }

    cout << "image_magic_number: " << image_magic_number << endl;
    cout << "number_of_images: " << number_of_images << endl;
    cout << "n_rows: " << n_rows << endl;
    cout << "n_cols: " << n_cols << endl;
    cout << "label_magic_number: " << label_magic_number << endl;
    cout << "number_of_labels: " << number_of_labels << endl;

    if (image_magic_number != 2051)
    {
        cerr << "文件格式错误，图像文件的 magic_number 应为 2051" << endl;
        return 1;
    }
    if (label_magic_number != 2049)
    {
        cerr << "文件格式错误，标签文件的 magic_number 应为 2049" << endl;
        return 1;
    }
    if (number_of_images == 0)
    {
        cerr << "图片数量错误" << endl;
        return 1;
    }
    if (number_of_labels == 0)
    {
        cerr << "标签数量错误" << endl;
        return 1;
    }
    if (n_rows != 28 || n_cols != 28)
    {
        cerr << "图片尺寸错误，当前测试程序预期为 28x28" << endl;
        return 1;
    }
    if (number_of_images != number_of_labels)
    {
        cerr << "图像数量和标签数量不一致" << endl;
        return 1;
    }

    vector<vector<float>> images = read_mnist_images(image_filename);
    if (images.empty())
    {
        return 1;
    }

    vector<uint8_t> labels = read_mnist_labels(label_filename);
    if (labels.empty())
    {
        return 1;
    }

    cout << "loaded_images: " << images.size() << endl;
    cout << "loaded_labels: " << labels.size() << endl;

    if (images.size() != number_of_images)
    {
        cerr << "实际读取到的图片数量与头部信息不一致" << endl;
        return 1;
    }
    if (labels.size() != number_of_labels)
    {
        cerr << "实际读取到的标签数量与头部信息不一致" << endl;
        return 1;
    }
    if (images.size() != labels.size())
    {
        cerr << "图像和标签无法一一对应" << endl;
        return 1;
    }

    const size_t expected_image_size = static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols);
    for (size_t i = 0; i < images.size(); ++i)
    {
        if (images[i].size() != expected_image_size)
        {
            cerr << "第 " << i << " 张图片像素数量错误" << endl;
            return 1;
        }
    }

    cout << "first_10_labels:" << endl;
    print_label_preview(labels, min<size_t>(10, labels.size()));

    size_t current_index = 0;
    while (true)
    {
        cout << endl;
        print_sample(images[current_index], labels[current_index], n_rows, n_cols, current_index, images.size());
        cout << "输入 1 查看下一张，输入 0 退出: ";

        string command;
        if (!getline(cin, command))
        {
            return 0;
        }

        if (command == "0")
        {
            return 0;
        }

        if (command == "1")
        {
            if (current_index + 1 < images.size())
            {
                ++current_index;
            }
            else
            {
                cout << "已经是最后一张图片了。" << endl;
            }
            continue;
        }

        cout << "无效输入，请输入 1 或 0。" << endl;
    }
}

unsigned char debug_forward_mode(int argc, char *argv[])
{
    string image_filename;
    string label_filename;
    if (!read_dataset_paths(argc, argv, image_filename, label_filename))
    {
        return 1;
    }

    uint32_t image_magic_number = 0;
    uint32_t number_of_images = 0;
    uint32_t n_rows = 0;
    uint32_t n_cols = 0;
    if (!read_mnist_image_file_info(image_filename, image_magic_number, number_of_images, n_rows, n_cols))
    {
        return 1;
    }

    vector<Sample> samples = read_mnist_Sample(image_filename, label_filename);
    if (samples.empty())
    {
        cerr << "未能成功读取样本。" << endl;
        return 1;
    }

    const Sample &sample = samples[0];
    if (sample.image.empty())
    {
        cerr << "样本图像为空。" << endl;
        return 1;
    }
    if (sample.target.empty())
    {
        cerr << "样本 target 为空。" << endl;
        return 1;
    }

    const size_t expected_image_size = static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols);
    if (sample.image.size() != expected_image_size)
    {
        cerr << "样本图像尺寸与文件头信息不一致。" << endl;
        return 1;
    }

    Network network = create_network(sample.image.size(), 16, sample.target.size());
    ForwardResult result;

    try
    {
        result = forward(network, sample.image);
    }
    catch (const exception &e)
    {
        cerr << "前向传播失败: " << e.what() << endl;
        return 1;
    }

    cout << "loaded_samples: " << samples.size() << endl;
    print_sample(sample.image, sample.label, n_rows, n_cols, 0, samples.size());
    cout << "network_input_size: " << network.input_size << endl;
    cout << "network_hidden_size: " << network.hidden_size << endl;
    cout << "network_output_size: " << network.output_size << endl;
    cout << "sample_target_size: " << sample.target.size() << endl;
    cout << "z1_size: " << result.z1.size() << endl;
    cout << "a1_size: " << result.a1.size() << endl;
    cout << "z2_size: " << result.z2.size() << endl;
    cout << "output_size: " << result.output.size() << endl;
    cout << "target:" << endl;
    print_float_vector(sample.target, sample.target.size());
    cout << "output:" << endl;
    print_float_vector(result.output, result.output.size());
    cout << "predicted_label: " << argmax_index(result.output) << endl;

    return 0;
}

unsigned char debug_train_mode(int argc, char *argv[])
{
    string image_filename;
    string label_filename;
    if (!read_dataset_paths(argc, argv, image_filename, label_filename))
    {
        return 1;
    }

    vector<Sample> samples = read_mnist_Sample(image_filename, label_filename);
    if (samples.empty())
    {
        cerr << "未能成功读取样本。" << endl;
        return 1;
    }

    const size_t train_sample_count = min<size_t>(1000, samples.size());
    vector<Sample> train_samples(samples.begin(), samples.begin() + train_sample_count);

    if (train_samples[0].image.empty() || train_samples[0].target.empty())
    {
        cerr << "训练样本不完整。" << endl;
        return 1;
    }

    Network network = create_network(train_samples[0].image.size(), 16, train_samples[0].target.size());

    const float learning_rate = 0.1f;
    const size_t epoch_count = 10;
    mt19937 generator(random_device{}());

    cout << "fixed_learning_rate: " << learning_rate << endl;
    cout << "train_sample_count: " << train_samples.size() << endl;
    cout << "epoch_count: " << epoch_count << endl;

    for (size_t epoch = 0; epoch < epoch_count; ++epoch)
    {
        shuffle_samples(train_samples, generator);
        evaluator_result result = train_epoch(network, train_samples, learning_rate);
        cout << "epoch " << (epoch + 1)
             << ": loss=" << result.average_loss
             << ", accuracy=" << result.accuracy << endl;
    }

    return 0;
}

unsigned char debug_train_mode_different_learning_rate(int argc, char *argv[])
{
    string image_filename;
    string label_filename;
    if (!read_dataset_paths(argc, argv, image_filename, label_filename))
    {
        return 1;
    }

    vector<Sample> samples = read_mnist_Sample(image_filename, label_filename);
    if (samples.empty())
    {
        cerr << "未能成功读取样本。" << endl;
        return 1;
    }

    const size_t train_sample_count = min<size_t>(1000, samples.size());
    vector<Sample> train_samples(samples.begin(), samples.begin() + train_sample_count);

    if (train_samples[0].image.empty() || train_samples[0].target.empty())
    {
        cerr << "训练样本不完整。" << endl;
        return 1;
    }
    const size_t epoch_count = 100;

    const Network base_network = create_network(train_samples[0].image.size(), 16, train_samples[0].target.size());

    vector<Sample> train_samples1 = train_samples;
    vector<Sample> train_samples2 = train_samples;
    vector<Sample> train_samples3 = train_samples;

    Network network1 = base_network;
    const float learning_rate1 = 0.1f;
    mt19937 generator1(12345u);

    cout << "fixed_learning_rate: " << learning_rate1 << endl;
    cout << "train_sample_count: " << train_samples.size() << endl;
    cout << "epoch_count: " << epoch_count << endl;

    for (size_t epoch = 0; epoch < epoch_count; ++epoch)
    {
        shuffle_samples(train_samples1, generator1);
        evaluator_result result = train_epoch(network1, train_samples1, learning_rate1);
        cout << "epoch " << (epoch + 1)
             << ": loss=" << result.average_loss
             << ", accuracy=" << result.accuracy << endl;
    }

    Network network2 = base_network;
    const float learning_rate2 = 0.05f;
    mt19937 generator2(12345u);

    cout << "fixed_learning_rate: " << learning_rate2 << endl;
    cout << "train_sample_count: " << train_samples.size() << endl;
    cout << "epoch_count: " << epoch_count << endl;

    for (size_t epoch = 0; epoch < epoch_count; ++epoch)
    {
        shuffle_samples(train_samples2, generator2);
        evaluator_result result = train_epoch(network2, train_samples2, learning_rate2);
        cout << "epoch " << (epoch + 1)
             << ": loss=" << result.average_loss
             << ", accuracy=" << result.accuracy << endl;
    }

    Network network3 = base_network;
    const float learning_rate3 = 0.02f;
    mt19937 generator3(12345u);

    cout << "fixed_learning_rate: " << learning_rate3 << endl;
    cout << "train_sample_count: " << train_samples.size() << endl;
    cout << "epoch_count: " << epoch_count << endl;

    for (size_t epoch = 0; epoch < epoch_count; ++epoch)
    {
        shuffle_samples(train_samples3, generator3);
        evaluator_result result = train_epoch(network3, train_samples3, learning_rate3);
        cout << "epoch " << (epoch + 1)
             << ": loss=" << result.average_loss
             << ", accuracy=" << result.accuracy << endl;
    }

    return 0;
}

unsigned char debug_train_and_test_mode(int argc, char *argv[])
{
    string train_image_filename;
    string train_label_filename;
    string test_image_filename;
    string test_label_filename;
    if (!read_train_test_dataset_paths(argc, argv,
                                       train_image_filename,
                                       train_label_filename,
                                       test_image_filename,
                                       test_label_filename))
    {
        return 1;
    }

    vector<Sample> train_samples_all = read_mnist_Sample(train_image_filename, train_label_filename);
    if (train_samples_all.empty())
    {
        cerr << "未能成功读取训练样本。" << endl;
        return 1;
    }

    vector<Sample> test_samples_all = read_mnist_Sample(test_image_filename, test_label_filename);
    if (test_samples_all.empty())
    {
        cerr << "未能成功读取测试样本。" << endl;
        return 1;
    }

    size_t requested_train_sample_count = 0;
    size_t requested_test_sample_count = 0;
    size_t epoch_count = 0;
    if (!read_train_test_mode_config(argc, argv,
                                     requested_train_sample_count,
                                     requested_test_sample_count,
                                     epoch_count))
    {
        return 1;
    }

    const size_t train_sample_count = min(requested_train_sample_count, train_samples_all.size());
    const size_t test_sample_count = min(requested_test_sample_count, test_samples_all.size());
    vector<Sample> train_samples(train_samples_all.begin(), train_samples_all.begin() + train_sample_count);
    vector<Sample> test_samples(test_samples_all.begin(), test_samples_all.begin() + test_sample_count);

    if (train_samples[0].image.empty() || train_samples[0].target.empty())
    {
        cerr << "训练样本不完整。" << endl;
        return 1;
    }

    if (test_samples[0].image.empty() || test_samples[0].target.empty())
    {
        cerr << "测试样本不完整。" << endl;
        return 1;
    }

    Network network = create_network(train_samples[0].image.size(), 128, train_samples[0].target.size());

    const float learning_rate = 0.1f;
    mt19937 generator(random_device{}());
    evaluator_result best_test_result;
    size_t best_epoch = 0;
    bool has_best_test_result = false;

    cout << "fixed_learning_rate: " << learning_rate << endl;
    cout << "train_sample_count: " << train_samples.size() << endl;
    cout << "test_sample_count: " << test_samples.size() << endl;
    cout << "epoch_count: " << epoch_count << endl;

    for (size_t epoch = 0; epoch < epoch_count; ++epoch)
    {
        shuffle_samples(train_samples, generator);
        evaluator_result train_result = train_epoch(network, train_samples, learning_rate);
        evaluator_result test_result = evaluate_network(network, test_samples);

        if (!has_best_test_result ||
            test_result.accuracy > best_test_result.accuracy ||
            (test_result.accuracy == best_test_result.accuracy &&
             test_result.average_loss < best_test_result.average_loss))
        {
            best_test_result = test_result;
            best_epoch = epoch + 1;
            has_best_test_result = true;
        }

        cout << "epoch " << (epoch + 1)
             << ": train_loss=" << train_result.average_loss
             << ", train_accuracy=" << train_result.accuracy
             << ", test_loss=" << test_result.average_loss
             << ", test_accuracy=" << test_result.accuracy << endl;
    }

    if (has_best_test_result)
    {
        cout << "best_test_epoch: " << best_epoch << endl;
        cout << "best_test_loss: " << best_test_result.average_loss << endl;
        cout << "best_test_accuracy: " << best_test_result.accuracy << endl;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    string mode;
    if (argc >= 6 && string(argv[5]) == "5")
    {
        mode = argv[5];
    }
    else if (argc >= 4)
    {
        mode = argv[3];
    }
    else
    {
        cout << "输入模式：1 为图像浏览，2 为前向传播测试，3 为固定学习率训练测试，4 为不同固定学习率的训练测试效果比较，5 为训练集训练+测试集评估: ";
        if (!getline(cin, mode))
        {
            return 1;
        }
    }

    if (mode == "1")
    {
        unsigned char debug = debug_reader_mode(argc, argv);
        if (debug)
        {
            return 1;
        }
        return 0;
    }

    if (mode == "2")
    {
        unsigned char debug = debug_forward_mode(argc, argv);
        if (debug)
        {
            return 1;
        }
        return 0;
    }

    if (mode == "3")
    {
        unsigned char debug = debug_train_mode(argc, argv);
        if (debug)
        {
            return 1;
        }
        return 0;
    }

    if (mode == "4")
    {
        unsigned char debug = debug_train_mode_different_learning_rate(argc, argv);
        if (debug)
        {
            return 1;
        }
        return 0;
    }

    if (mode == "5")
    {
        unsigned char debug = debug_train_and_test_mode(argc, argv);
        if (debug)
        {
            return 1;
        }
        return 0;
    }

    cerr << "无效模式，请输入 1、2、3、4 或 5。" << endl;
    return 1;
}
