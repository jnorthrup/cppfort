#include "orbit_iterator.h"

#include <algorithm>
#include <iostream>

#include "confix_orbit.h"

namespace cppfort::stage0 {

OrbitIterator::OrbitIterator(std::size_t combinator_pool_size)
    : pool_(combinator_pool_size) {}

OrbitIterator::~OrbitIterator() {
    release_combinators();
}

void OrbitIterator::add_orbit(Orbit* orbit) {
    if (orbit) {
        orbits_.push_back(orbit);
    }
}

Orbit* OrbitIterator::next() {
    if (!has_next()) {
        return nullptr;
    }

    Orbit* orbit = orbits_[current_index_++];

    if (auto* confix = dynamic_cast<ConfixOrbit*>(orbit)) {
        // std::cout << "DEBUG: Setting combinator on confix orbit\n";
        if (confix->get_combinator() == nullptr) {
            if (auto* scanner = pool_.allocate()) {
                // std::cout << "DEBUG: Allocated scanner\n";
                if (patterns_) {
                    // std::cout << "DEBUG: Setting patterns on scanner\n";
                    scanner->set_patterns(*patterns_);
                }
                scanner->set_packrat_cache(&packrat_cache_);
                confix->set_combinator(scanner);
                leased_.emplace_back(scanner, orbit);
            } else {
                // std::cout << "DEBUG: Failed to allocate scanner\n";
            }
        }
    }

    return orbit;
}

Orbit* OrbitIterator::current() const {
    if (current_index_ == 0 || current_index_ > orbits_.size()) {
        return nullptr;
    }
    return orbits_[current_index_ - 1];
}

void OrbitIterator::reset() {
    release_combinators();
    current_index_ = 0;
}

bool OrbitIterator::has_next() const {
    return current_index_ < orbits_.size();
}

void OrbitIterator::release_combinators() {
    for (auto& entry : leased_) {
        auto* scanner = entry.first;
        auto* orbit = entry.second;
        if (auto* confix = dynamic_cast<ConfixOrbit*>(orbit)) {
            confix->set_combinator(nullptr);
        }
        pool_.release(scanner);
    }
    leased_.clear();
}


void OrbitIterator::clear() {
    release_combinators();
    orbits_.clear();
    current_index_ = 0;
}

} // namespace cppfort::stage0

