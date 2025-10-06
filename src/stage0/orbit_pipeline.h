#pragma once

#include <memory>
#include <string>
#include <vector>

#include "confix_orbit.h"
#include "orbit_iterator.h"
#include "pattern_loader.h"
#include "correlator.h"
#include "grammar_tree.h"
#include "wide_scanner.h"

namespace cppfort::stage0 {

class OrbitPipeline {
public:
    OrbitPipeline() = default;

    bool load_patterns(const std::string& path);

    void populate_iterator(const std::vector<OrbitFragment>& fragments,
                           OrbitIterator& iterator);

    const std::vector<std::unique_ptr<ConfixOrbit>>& orbits() const { return confix_orbits_; }

private:
    std::unique_ptr<ConfixOrbit> evaluate_fragment(const OrbitFragment& fragment) const;
    std::unique_ptr<ConfixOrbit> make_base_orbit(const OrbitFragment& fragment) const;
    std::pair<char, char> select_confix(const OrbitFragment& fragment) const;
    const PatternData* best_pattern_for(ConfixOrbit& candidate) const;

    PatternLoader loader_;
    FragmentCorrelator correlator_;
    GrammarTree grammar_tree_;
    std::vector<std::unique_ptr<ConfixOrbit>> confix_orbits_;
};

} // namespace cppfort::stage0

