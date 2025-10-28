#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "src/stage0/orbit_pipeline.h"
#include "src/stage0/orbit_iterator.h"
#include "src/stage0/wide_scanner.h"
#include "src/stage0/orbit_ring.h"
#include "src/stage0/correlator.h"

using cppfort::stage0::ConfixOrbit;
using cppfort::stage0::OrbitIterator;
using cppfort::stage0::OrbitPipeline;

namespace {

struct Example {
    std::string id;
    std::string label;
    std::string source;
};

std::string trim(std::string_view view) {
    size_t begin = 0;
    size_t end = view.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(view[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(view[end - 1]))) {
        --end;
    }
    return std::string(view.substr(begin, end - begin));
}

std::vector<Example> parse_corpus(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open corpus " + path);
    }

    std::vector<Example> out;
    Example current;
    bool in_source = false;

    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("- id:", 0) == 0) {
            if (!current.id.empty()) {
                out.push_back(current);
                current = Example{};
            }
            current.id = trim(line.substr(5));
            if (!current.id.empty() && current.id.front() == ' ') {
                current.id.erase(current.id.begin());
            }
            in_source = false;
            continue;
        }
        if (line.rfind("  label:", 0) == 0) {
            current.label = trim(line.substr(8));
            in_source = false;
            continue;
        }
        if (line.rfind("  source:", 0) == 0) {
            current.source.clear();
            in_source = true;
            continue;
        }
        if (in_source) {
            if (line.rfind("    ", 0) == 0) {
                current.source.append(line.substr(4));
                current.source.push_back('\n');
            } else if (!line.empty()) {
                // end of block when indentation stops
                in_source = false;
            }
        }
    }
    if (!current.id.empty()) {
        out.push_back(current);
    }
    return out;
}

std::string grammar_to_string(cppfort::ir::GrammarType g) {
    switch (g) {
        case cppfort::ir::GrammarType::C: return "C";
        case cppfort::ir::GrammarType::CPP: return "CPP";
        case cppfort::ir::GrammarType::CPP2: return "CPP2";
        default: return "UNKNOWN";
    }
}

cppfort::ir::GrammarType string_to_grammar(const std::string& label) {
    if (label == "C") return cppfort::ir::GrammarType::C;
    if (label == "CPP") return cppfort::ir::GrammarType::CPP;
    if (label == "CPP2") return cppfort::ir::GrammarType::CPP2;
    return cppfort::ir::GrammarType::UNKNOWN;
}

struct EvaluationResult {
    cppfort::ir::GrammarType predicted = cppfort::ir::GrammarType::UNKNOWN;
    double confidence = 0.0;
};

EvaluationResult evaluate_source(const std::string& source, OrbitPipeline& pipeline) {
    cppfort::ir::WideScanner scanner;
    auto anchors = cppfort::ir::WideScanner::generateAlternatingAnchors(source);
    (void)scanner.scanAnchorsWithOrbits(source, anchors);

    OrbitIterator iterator(anchors.size());
    pipeline.populate_iterator(scanner.fragments(), iterator, source);

    EvaluationResult result;

    for (auto* orbit = iterator.next(); orbit; orbit = iterator.next()) {
        auto* confix = dynamic_cast<ConfixOrbit*>(orbit);
        if (!confix) continue;
        if (confix->confidence > result.confidence) {
            result.confidence = confix->confidence;
            result.predicted = confix->selected_grammar();
        }
    }

    if (result.confidence <= 0.0 || result.predicted == cppfort::ir::GrammarType::UNKNOWN) {
        cppfort::stage0::FragmentCorrelator correlator;
        cppfort::stage0::OrbitFragment fragment;
        fragment.start_pos = 0;
        fragment.end_pos = source.size();
        correlator.correlate(fragment, source);
        result.predicted = fragment.classified_grammar;
        result.confidence = fragment.confidence > 0.0 ? fragment.confidence : 0.6;
    }
    if (result.confidence <= 0.0) {
        result.confidence = 0.5;
    }

    return result;
}

} // namespace

int main(int argc, char** argv) {
    const std::string corpus_path = (argc >= 2) ? argv[1] : "tests/corpus/grammar_labels.yaml";
    const std::string pattern_path = (argc >= 3) ? argv[2] : "tests/patterns/minimal.yaml";

    std::vector<Example> examples;
    try {
        examples = parse_corpus(corpus_path);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    if (examples.empty()) {
        std::cerr << "corpus is empty" << std::endl;
        return 1;
    }

    OrbitPipeline pipeline;
    if (!pipeline.load_patterns(pattern_path)) {
        std::cerr << "failed to load patterns from " << pattern_path << std::endl;
        return 1;
    }

    std::map<std::string, int> total_per_class;
    std::map<std::string, int> correct_per_class;
    std::map<std::string, int> false_positive;
    std::map<std::string, int> false_negative;

    int total = 0;
    int correct = 0;
    double loss = 0.0;

    for (const auto& example : examples) {
        ++total;
        const auto expected = string_to_grammar(example.label);
        auto predicted = evaluate_source(example.source, pipeline);

        const std::string predicted_label = grammar_to_string(predicted.predicted);
        ++total_per_class[example.label];

        double confidence = std::clamp(predicted.confidence, 1e-6, 1.0 - 1e-6);

        if (predicted.predicted == expected) {
            ++correct;
            ++correct_per_class[example.label];
            loss -= std::log(confidence);
        } else {
            ++false_negative[example.label];
            ++false_positive[predicted_label];
            loss -= std::log(1.0 - confidence);
        }

        std::cout << example.id << ": label=" << example.label
                  << " predicted=" << predicted_label
                  << " confidence=" << predicted.confidence << std::endl;
    }

    const double accuracy = static_cast<double>(correct) / static_cast<double>(total);
    const double avg_loss = loss / static_cast<double>(total);

    std::cout << "\nSummary" << std::endl;
    std::cout << "total=" << total << " correct=" << correct
              << " accuracy=" << accuracy
              << " avg_loss=" << avg_loss << std::endl;

    for (const auto& entry : total_per_class) {
        const std::string& label = entry.first;
        const double tp = static_cast<double>(correct_per_class[label]);
        const double fp = static_cast<double>(false_positive[label]);
        const double fn = static_cast<double>(false_negative[label]);

        const double precision = tp + fp > 0.0 ? tp / (tp + fp) : 0.0;
        const double recall = tp + fn > 0.0 ? tp / (tp + fn) : 0.0;

        std::cout << "class=" << label
                  << " precision=" << precision
                  << " recall=" << recall
                  << std::endl;
    }

    return 0;
}
