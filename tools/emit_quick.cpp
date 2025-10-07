#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/stage0/orbit_ring.h"

using namespace cppfort::stage0;

static std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <source-file>\n";
        return 1;
    }

    std::string source = slurp(argv[1]);

    // Build a single orbit that covers whole file
    OrbitRing ring;
    ring.start_position = 0;
    ring.end_position = source.size();

    // Simple reconstruction: emit source in gaps and orbit spans
    std::string out;
    size_t last = 0;
    // No gaps before first orbit
    out += source.substr(ring.start_position, ring.end_position - ring.start_position);

    std::cout << "Original size: " << source.size() << "\n";
    std::cout << "Reconstructed size: " << out.size() << "\n";
    std::cout << "Match: " << (out == source ? "EXACT" : "DIFFERS") << "\n";
    if (out != source) {
        size_t min_len = std::min(source.size(), out.size());
        size_t diff_pos = min_len;
        for (size_t i = 0; i < min_len; ++i) if (source[i] != out[i]) { diff_pos = i; break; }
        std::cout << "First diff at: " << diff_pos << "\n";
    }

    return 0;
}
