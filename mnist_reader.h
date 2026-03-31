#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

// 样本单元
struct Sample
{
    std::vector<float> image;
    std::uint8_t label;
    std::vector<float> target;
};

std::uint32_t read_4bytes(const unsigned char *bytes);
unsigned char read_header(std::ifstream &file, std::uint32_t &magic_number, std::uint32_t &number_of_images, std::uint32_t &n_rows, std::uint32_t &n_cols);
unsigned char read_label_header(std::ifstream &file, std::uint32_t &magic_number, std::uint32_t &number_of_labels);
std::vector<float> read_image(std::ifstream &file, std::uint32_t n_rows, std::uint32_t n_cols);
std::vector<std::vector<float>> read_mnist_images(const std::string &filename);
std::vector<std::uint8_t> read_mnist_labels(const std::string &filename);
bool read_mnist_image_file_info(const std::string &filename, std::uint32_t &magic_number, std::uint32_t &number_of_images, std::uint32_t &n_rows, std::uint32_t &n_cols);
bool read_mnist_label_file_info(const std::string &filename, std::uint32_t &magic_number, std::uint32_t &number_of_labels);
std::vector<float> read_first_mnist_image(const std::string &filename);
std::vector<Sample> read_mnist_Sample(const std::string &filename_images, const std::string &filename_labels);