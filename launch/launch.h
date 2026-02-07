#pragma once

#include <string>

struct Options {
    int M = 1024;
    int K = 256;
    int N = 128;
    int batch = 1;
    int repeats = 10;
    std::string kernel = "naive";
};

void print_usage(const char* prog);
bool parse_args(int argc, char** argv, Options& opt);
int run_naive(const Options& opt);
int run_tiled(const Options& opt);
int run_batch_naive(const Options& opt);
int run_batch_tiled(const Options& opt);
