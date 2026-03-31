#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include "mnist_reader.h"

using namespace std;

// 读取4字节
uint32_t read_4bytes(const unsigned char *bytes)
{
    uint32_t result = 0;
    result = ((uint32_t)(bytes[0]) << 24) | ((uint32_t)(bytes[1]) << 16) | ((uint32_t)(bytes[2]) << 8) | ((uint32_t)bytes[3]);
    return result;
};

// 读取文件头
unsigned char read_header(ifstream &file, uint32_t &magic_number, uint32_t &number_of_images, uint32_t &n_rows, uint32_t &n_cols)
{
    unsigned char bytes[4];
    file.read(reinterpret_cast<char *>(bytes), 4);
    if (!file)
    {
        cerr << "读取魔数失败" << endl;
        return 1;
    }
    magic_number = read_4bytes(bytes);

    file.read(reinterpret_cast<char *>(bytes), 4);
    if (!file)
    {
        cerr << "读取图片数量失败" << endl;
        return 2;
    }
    number_of_images = read_4bytes(bytes);

    file.read(reinterpret_cast<char *>(bytes), 4);
    if (!file)
    {
        cerr << "读取行数失败" << endl;
        return 3;
    }
    n_rows = read_4bytes(bytes);

    file.read(reinterpret_cast<char *>(bytes), 4);
    if (!file)
    {
        cerr << "读取列数失败" << endl;
        return 4;
    }
    n_cols = read_4bytes(bytes);

    return 0;
}

// 读取标签文件的header
unsigned char read_label_header(ifstream &file, uint32_t &magic_number, uint32_t &number_of_labels)
{
    unsigned char bytes[4];
    file.read(reinterpret_cast<char *>(bytes), 4);
    if (!file)
    {
        cerr << "读取magic_number失败" << endl;
        return 1;
    }
    magic_number = read_4bytes(bytes);

    file.read(reinterpret_cast<char *>(bytes), 4);
    if (!file)
    {
        cerr << "读取标签数量失败" << endl;
        return 2;
    }
    number_of_labels = read_4bytes(bytes);

    return 0;
}

// 读取单张图片
vector<float> read_image(ifstream &file, uint32_t n_rows, uint32_t n_cols)
{
    uint32_t image_size = n_cols * n_rows;
    vector<float> image;
    image.reserve(image_size);
    vector<unsigned char> pixel(image_size);
    file.read(reinterpret_cast<char *>(pixel.data()), image_size);
    if (!file)
    {
        cerr << "读取图片失败" << endl;
        return {};
    }
    for (uint32_t i = 0; i < image_size; i++)
    {
        image.push_back(static_cast<float>(pixel[i]) / 255.0f);
    }
    return image;
}

// 读取图集
vector<vector<float>> read_mnist_images(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "无法打开文件" << endl;
        return {};
    }

    uint32_t magic_number, number_of_images, n_rows, n_cols;
    unsigned char result = read_header(file, magic_number, number_of_images, n_rows, n_cols);
    if (result != 0)
    {
        return {};
    }

    if (magic_number != 2051)
    {
        cerr << "文件格式错误" << endl;
        return {};
    }

    if (number_of_images == 0)
    {
        cerr << "图片数量错误" << endl;
        return {};
    }

    if (n_rows != 28 || n_cols != 28)
    {
        cerr << "图片尺寸错误" << endl;
        return {};
    }
    vector<vector<float>> images;
    images.reserve(number_of_images);
    for (uint32_t i = 0; i < number_of_images; i++)
    {
        vector<float> image = read_image(file, n_rows, n_cols);
        if (image.empty())
        {
            cerr << "读取图片失败" << endl;
            return {};
        }
        images.push_back(image);
        // 处理每张图片
    }
    return images;
}

// 读取标签
vector<uint8_t> read_labels(ifstream &file, uint32_t number_of_labels)
{
    vector<uint8_t> labels;
    labels.reserve(number_of_labels);
    for (uint32_t i = 0; i < number_of_labels; i++)
    {
        unsigned char result = 0;
        file.read(reinterpret_cast<char *>(&result), 1);
        if (!file)
        {
            cerr << "读取文件标签内容错误:第" << i + 1 << "张" << endl;
            return {};
        }
        if (result > 9)
        {
            cerr << "标签值错误:第" << i + 1 << "个标签超出 0~9 范围" << endl;
            return {};
        }
        labels.push_back((uint8_t)result);
    }
    return labels;
}

bool read_mnist_image_file_info(const string &filename, uint32_t &magic_number, uint32_t &number_of_images, uint32_t &n_rows, uint32_t &n_cols)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "无法打开文件" << endl;
        return false;
    }

    unsigned char result = read_header(file, magic_number, number_of_images, n_rows, n_cols);
    return result == 0;
}

bool read_mnist_label_file_info(const string &filename, uint32_t &magic_number, uint32_t &number_of_labels)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "无法打开文件" << endl;
        return false;
    }

    unsigned char result = read_label_header(file, magic_number, number_of_labels);
    return result == 0;
}

vector<float> read_first_mnist_image(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "无法打开文件" << endl;
        return {};
    }

    uint32_t magic_number = 0;
    uint32_t number_of_images = 0;
    uint32_t n_rows = 0;
    uint32_t n_cols = 0;
    unsigned char result = read_header(file, magic_number, number_of_images, n_rows, n_cols);
    if (result != 0)
    {
        return {};
    }

    if (magic_number != 2051)
    {
        cerr << "文件格式错误" << endl;
        return {};
    }
    if (number_of_images == 0)
    {
        cerr << "图片数量错误" << endl;
        return {};
    }
    if (n_rows != 28 || n_cols != 28)
    {
        cerr << "图片尺寸错误" << endl;
        return {};
    }

    return read_image(file, n_rows, n_cols);
}

vector<uint8_t> read_mnist_labels(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "无法打开文件" << endl;
        return {};
    }

    uint32_t magic_number = 0;
    uint32_t number_of_labels = 0;
    unsigned char result = read_label_header(file, magic_number, number_of_labels);
    if (result != 0)
    {
        return {};
    }

    if (magic_number != 2049)
    {
        cerr << "文件格式错误" << endl;
        return {};
    }
    if (number_of_labels == 0)
    {
        cerr << "标签数量错误" << endl;
        return {};
    }
    return read_labels(file, number_of_labels);
}

vector<Sample> read_mnist_Sample(const string &filename_images, const string &filename_labels)
{
    vector<vector<float>> images = read_mnist_images(filename_images);
    if (images.empty())
    {
        cerr << "读取图像样本失败" << endl;
        return {};
    }

    vector<uint8_t> labels = read_mnist_labels(filename_labels);
    if (labels.empty())
    {
        cerr << "读取标签样本失败" << endl;
        return {};
    }

    size_t number_of_images = images.size();
    size_t number_of_labels = labels.size();
    if (number_of_images != number_of_labels)
    {
        cerr << "图片与标签数量上不匹配，训练集有错误" << endl;
        cerr << "图片数量" << number_of_images << endl;
        cerr << "标签数量" << number_of_labels << endl;
        return {};
    }

    vector<Sample> samples;
    samples.reserve(number_of_images);

    for (size_t index = 0; index < labels.size(); ++index)
    {
        if (labels[index] > 9)
        {
            cerr << "标签值错误:第" << index + 1 << "个标签超出 0~9 范围" << endl;
            return {};
        }

        vector<float> target(10, 0.0f);
        target[labels[index]] = 1.0f;

        Sample sample = {images[index], labels[index], target};
        samples.push_back(sample);
    }

    return samples;
}