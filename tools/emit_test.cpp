#include <iostream>
#include <string>
#include <memory>

#include "src/stage0/orbit_emitter.h"
#include "src/stage0/orbit_iterator.h"

using namespace cppfort::stage0;

// Provide simple definitions for OrbitIterator methods so we can test OrbitEmitter
OrbitIterator::OrbitIterator(std::size_t combinator_pool_size) : orbits_(), current_index_(0) {}
OrbitIterator::~OrbitIterator() { clear(); }

void OrbitIterator::add_orbit(Orbit* orbit) {
    orbits_.push_back(orbit);
}

Orbit* OrbitIterator::next() {
    if (current_index_ >= orbits_.size()) return nullptr;
    return orbits_[current_index_++];
}

Orbit* OrbitIterator::current() const {
    if (current_index_ == 0 || current_index_ > orbits_.size()) return nullptr;
    return orbits_[current_index_ - 1];
}

void OrbitIterator::reset() { current_index_ = 0; }

bool OrbitIterator::has_next() const { return current_index_ < orbits_.size(); }

void OrbitIterator::clear() {
    for (auto* o : orbits_) delete o;
    orbits_.clear();
    current_index_ = 0;
}

int main() {
    std::string source = "#include <iostream>\nint main() { std::cout << \"hello\" << std::endl; return 0; }\n";

    // Create a single orbit covering the whole source
    Orbit* o = new Orbit(OrbitType::Confix);
    o->start_pos = 0;
    o->end_pos = source.size();
    o->confidence = 1.0;

    OrbitIterator iterator(1);
    iterator.add_orbit(o);

    OrbitEmitter emitter;
    std::string reconstructed = emitter.reconstruct_source(iterator, source);

    std::cout << "Original size: " << source.size() << "\n";
    std::cout << "Reconstructed size: " << reconstructed.size() << "\n";
    std::cout << "Match: " << (reconstructed == source ? "EXACT" : "DIFFERS") << "\n";

    if (reconstructed != source) {
        size_t min_len = std::min(source.size(), reconstructed.size());
        size_t diff_pos = min_len;
        for (size_t i = 0; i < min_len; ++i) {
            if (source[i] != reconstructed[i]) { diff_pos = i; break; }
        }
        std::cout << "First diff at: " << diff_pos << "\n";
        std::cout << "Original fragment: '" << source.substr(diff_pos, 40) << "'\n";
        std::cout << "Reconstructed fragment: '" << reconstructed.substr(diff_pos, 40) << "'\n";
    }

    return 0;
}
