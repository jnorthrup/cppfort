#include <iostream>
#include <cassert>
#include "rabin_karp.h"
#include "orbit_mask.h"

using namespace cppfort::ir;

int main() {
    OrbitContext ctx;
    // Update context with some bracket/brace characters to influence counts
    ctx.update('{');
    ctx.update('{');
    ctx.update('}');
    ctx.update('(');
    ctx.update('(');

    RabinKarp rk;
    auto hashes = rk.processOrbitContext(ctx);

    auto counts = ctx.getCounts();
    // RabinKarp should reflect the same orbit counts
    for (size_t i = 0; i < counts.size(); ++i) {
        assert(rk.orbitCount(i) == counts[i] && "orbit counts must match OrbitContext counts");
        if (counts[i] > 0) {
            assert(rk.hashAt(i) != 0 && "hash must be non-zero for non-empty orbit counts");
        } else {
            assert(rk.hashAt(i) == 0 && "hash should be zero for empty orbit counts");
        }
    }

    std::cout << "test_rabin_karp: OK\n";
    return 0;
}
