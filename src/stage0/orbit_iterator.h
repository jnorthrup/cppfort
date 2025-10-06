#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "orbit_ring.h"
#include "rbcursive.h"

namespace cppfort::stage0 {

class OrbitIterator {
public:
    explicit OrbitIterator(std::size_t combinator_pool_size = 0);
    ~OrbitIterator();

    void add_orbit(Orbit* orbit);

    Orbit* next();
    Orbit* current() const;
    void reset();
    bool has_next() const;

    std::size_t size() const { return orbits_.size(); }

    void clear();

private:
    void release_combinators();

    std::vector<Orbit*> orbits_{};
    std::size_t current_index_ = 0;
    ::cppfort::ir::CombinatorPool pool_;
    std::vector<std::pair<::cppfort::ir::RBCursiveScanner*, Orbit*>> leased_{};
};

} // namespace cppfort::stage0

