#pragma once

#include "orbit_ring.h"
#include "rbcursive.h"
#include "packrat_cache.h"

namespace cppfort::stage0 {

struct PatternData;

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

    // Set patterns for combinators
    void set_patterns(const std::vector<PatternData>& patterns) { patterns_ = &patterns; }

private:
    void release_combinators();

    std::vector<Orbit*> orbits_{};
    std::size_t current_index_ = 0;
    ::cppfort::stage0::CombinatorPool pool_;
    PackratCache packrat_cache_;
    std::vector<std::pair<::cppfort::stage0::RBCursiveScanner*, Orbit*>> leased_{};
    const std::vector<PatternData>* patterns_ = nullptr;
};

} // namespace cppfort::stage0

