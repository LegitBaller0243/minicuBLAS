#include "launch/launch.h"

#include <iostream>

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }

    if (opt.kernel == "naive") return run_naive(opt);
    if (opt.kernel == "tiled") return run_tiled(opt);
    if (opt.kernel == "transpose-tiled") return run_transpose_tiled(opt);
    if (opt.kernel == "batch-naive") return run_batch_naive(opt);
    if (opt.kernel == "batch-tiled") return run_batch_tiled(opt);

    std::cerr << "Unknown kernel: " << opt.kernel << "\n";
    print_usage(argv[0]);
    return 1;
}
