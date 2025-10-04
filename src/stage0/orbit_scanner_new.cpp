#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <filesystem>

#include "tblgen_patterns.h"
#include "multi_grammar_loader.h"
#include "orbit_scanner.h"
#include "rabin_karp.h"
#include "wide_scanner.h"
#include "projection_oracle.h"
#include "cpp2_key_resolver.h"

namespace cppfort {
namespace ir {

OrbitScanner::OrbitScanner(const OrbitScannerConfig& config,
                           ::std::unique_ptr<IMultiGrammarLoader> loader)
  : m_config(config),
    m_rabinKarp(::std::make_unique<RabinKarp>()),
    m_context(::std::make_unique<OrbitContext>(m_config.maxDepth)),
    m_loader(::std::move(loader)),
    m_cpp2Resolver(::std::make_unique<cppfort::stage0::CPP2KeyResolver>()) {
  if (!m_loader) {
    m_loader = ::std::unique_ptr<IMultiGrammarLoader>(new MultiGrammarLoader());
  }
}

OrbitScanner::~OrbitScanner() = default;

bool OrbitScanner::initialize() {
  if (!validateConfig()) {
    ::std::cerr << "Invalid scanner configuration" << ::std::endl;
    return false;
  }

  // Load all grammar patterns
  if (!m_loader->loadAllGrammars(m_config.patternsDir)) {
    ::std::cerr << "Failed to load grammar patterns" << ::std::endl;
    return false;
  }

  // Initialize CPP2 key resolver
  m_cpp2Resolver->build_key_database();

  return validateInitialization();
}

DetectionResult OrbitScanner::scan(const ::std::string& code) const {
  // Get all patterns from loaded grammars
  auto allPatterns = m_loader->getAllPatterns();

  return scan(code, allPatterns);
}

DetectionResult OrbitScanner::scan(const ::std::string& code,
                                  const ::std::vector<OrbitPattern>& patterns) const {
  // Find all matches in the code
  auto matches = findMatches(code, patterns);

  // Analyze matches to determine grammar
  return analyzeMatches(matches);
}

const OrbitScannerConfig& OrbitScanner::getConfig() const {
  return m_config;
}

void OrbitScanner::updateConfig(const OrbitScannerConfig& config) {
  m_config = config;
  // Reinitialize components with new config
  m_rabinKarp = ::std::make_unique<RabinKarp>();
  m_context = ::std::make_unique<OrbitContext>(m_config.maxDepth);
  m_cpp2Resolver = ::std::make_unique<cppfort::stage0::CPP2KeyResolver>();
  // Reinitialize CPP2 resolver
  m_cpp2Resolver->build_key_database();
}

size_t OrbitScanner::getPatternCount() const {
  return m_loader->getAllPatterns().size();
}
::std::vector<GrammarType> OrbitScanner::getLoadedGrammars() const {
  return m_loader->getLoadedGrammars();
}

MatchResults OrbitScanner::findMatches(const ::std::string& code,
                                     const ::std::vector<OrbitPattern>& patterns) const {
  MatchResults results;

  // Create a temporary OrbitContext for scanning
  OrbitContext scanContext(m_config.maxDepth);

  // Track potential grammar detections based on orbit patterns
  ::std::unordered_map<GrammarType, double> grammarConfidences;
  ::std::unordered_map<GrammarType, size_t> grammarMatchCounts;

  // Generate alternating anchor points at UTF-8 boundaries (64/32 byte spacing)
  auto anchors = WideScanner::generateAlternatingAnchors(code, 64);

  // SIMD-accelerated boundary detection between anchors
  auto boundaries = WideScanner::scanAnchorsSIMD(code, anchors);

  // Build scan positions: anchor points + boundaries
  ::std::vector<size_t> scanPositions;
  scanPositions.reserve(anchors.size() + boundaries.size());
  for (const auto& anchor : anchors) {
    scanPositions.push_back(anchor.position);
  }
  for (const auto& boundary : boundaries) {
    scanPositions.push_back(boundary.position);
  }
  ::std::sort(scanPositions.begin(), scanPositions.end());
  scanPositions.erase(::std::unique(scanPositions.begin(), scanPositions.end()), scanPositions.end());

  // Scan at each anchor/boundary position
  for (size_t pos : scanPositions) {
    if (pos >= code.length()) continue;
    char ch = code[pos];

    // Update orbit context with current character
    scanContext.update(ch);

    // Get current orbit counts for pattern detection
    auto orbitCounts = scanContext.getCounts();

    // Get current orbit hashes for this position
    auto orbitHashes = m_rabinKarp->processOrbitContext(scanContext);

    // Detect grammar patterns based on orbit structures
    for (const auto& pattern : patterns) {
      GrammarType grammar = static_cast<GrammarType>(pattern.orbit_id);
      bool signatureMatched = pattern.signature_patterns.empty();
      ::std::string matchedSignature;

      if (!signatureMatched) {
        for (const auto& signature : pattern.signature_patterns) {
          if (signature.empty()) continue;

          if (pos + signature.size() <= code.size() &&
              code.compare(pos, signature.size(), signature) == 0) {
            signatureMatched = true;
            matchedSignature = signature;
            break;
          }

          if (pos + 1 >= signature.size() &&
              code.compare(pos + 1 - signature.size(), signature.size(), signature) == 0) {
            signatureMatched = true;
            matchedSignature = signature;
            break;
          }
        }
      }

      if (!signatureMatched) {
        continue;
      }

      // Validate depth context
      int currentDepth = scanContext.getDepth();
      if (pattern.expected_depth >= 0 && currentDepth != pattern.expected_depth) {
        continue;
      }

      // Compute current confix context bitmask using helper
      uint8_t ctxMask = scanContext.confixMask();

      // If the pattern declares a confix_mask, ensure at least one of the active
      // context bits is allowed by the pattern. If no overlap, skip this pattern.
      if ((pattern.confix_mask & ctxMask) == 0) {
        // Legacy behavior: fall back to required_confix check if provided
        if (!pattern.required_confix.empty()) {
          bool confixActive = false;
          if (pattern.required_confix == "{" && scanContext.depth(OrbitType::OpenBrace) > 0) confixActive = true;
          else if (pattern.required_confix == "(" && scanContext.depth(OrbitType::OpenParen) > 0) confixActive = true;
          else if (pattern.required_confix == "[" && scanContext.depth(OrbitType::OpenBracket) > 0) confixActive = true;
          else if (pattern.required_confix == "<" && scanContext.depth(OrbitType::OpenAngle) > 0) confixActive = true;
          else if (pattern.required_confix == "\"" && scanContext.depth(OrbitType::Quote) > 0) confixActive = true;
          if (!confixActive) continue;
        } else {
          continue;
        }
      }

      if (matchedSignature.empty()) {
        matchedSignature = ::std::string(1, ch);
      }

      // Signature match + valid depth/confix = high confidence truth
      double confidence = signatureMatched ? 0.95 : detectOrbitPattern(grammar, orbitCounts, pos, code);

      // Boost non-signature matches slightly if heuristics support them
      if (!signatureMatched && confidence > 0.0 && confidence < 0.25) {
        confidence = 0.25;
      }

      // Report matches that meet threshold (signature matches always do)
      if (confidence >= m_config.patternThreshold) {
        OrbitMatch match;
        match.patternName = pattern.name;
        match.grammarType = grammar;
        match.startPos = pos;
        match.endPos = pos + 1;
        match.signature = matchedSignature;
        match.confidence = confidence * pattern.weight;

        // Add orbit hash information
        match.orbitHashes = orbitHashes;
        match.orbitCounts = orbitCounts;

        results.push_back(match);

        // Update grammar confidence tracking
        grammarConfidences[grammar] += match.confidence;
        grammarMatchCounts[grammar]++;
      }
    }
  }

  // Limit matches to prevent excessive processing
  if (results.size() > m_config.maxMatches) {
    results.resize(m_config.maxMatches);
  }

  // Phase 3-6: CPP2 Key Resolution with Backward Inference
  // Apply CPP2 key-based disambiguation to refine matches
  results = applyCPP2KeyResolution(code, results);

  return results;
}

MatchResults OrbitScanner::applyCPP2KeyResolution(const ::std::string& code,
                                                 const MatchResults& candidates) const {
  if (!m_cpp2Resolver || candidates.empty()) {
    return candidates;
  }

  MatchResults refined_candidates = candidates;

  // Phase 1: Lattice pre-filter (HeuristicGrid) - simplified for now
  // In full implementation, this would use HeuristicGrid for initial filtering

  // Phase 2: Forward pattern matching already completed - candidates are the results

  // Phase 3: CPP2 Key Lookup - Process all candidates together for better context
  std::string full_context = code;  // Use full code for context-aware resolution

  // Get all CPP2 keys that might be relevant to this code segment
  auto relevant_cpp2_keys = m_cpp2Resolver->find_relevant_cpp2_keys(full_context);

  // Phase 4: Peer Activation - Apply similarity-based confidence adjustment
  for (auto& candidate : refined_candidates) {
    double max_confidence_boost = 1.0;  // Track maximum boost from peer activation

    for (const auto& cpp2_key : relevant_cpp2_keys) {
      // Extract local context around this candidate
      size_t local_start = (candidate.startPos > 50) ? candidate.startPos - 50 : 0;
      size_t local_end = std::min(candidate.endPos + 50, code.length());
      std::string local_context = code.substr(local_start, local_end - local_start);

      // Compute similarity score
      double similarity = m_cpp2Resolver->compute_cpp2_similarity(
          candidate.signature, cpp2_key.signature_pattern);

      // Check similarity threshold
      if (similarity >= cpp2_key.similarity_threshold) {
        // Validate scope constraints
        std::string current_scope = determine_scope_type(local_context);
        if (current_scope == cpp2_key.scope_requirement || cpp2_key.scope_requirement == "any") {
          // Validate lattice requirements
          uint16_t current_lattice = determine_lattice_mask(local_context);
          if ((current_lattice & cpp2_key.lattice_filter) != 0) {
            // Apply confidence modifier - accumulate the best boost
            max_confidence_boost = std::max(max_confidence_boost, cpp2_key.confidence_modifier);
          }
        }
      }
    }

    // Apply the maximum confidence boost found
    candidate.confidence *= max_confidence_boost;
    candidate.confidence = std::max(0.0, std::min(1.0, candidate.confidence));
  }

  // Phase 5: Re-rank candidates based on updated confidences
  std::sort(refined_candidates.begin(), refined_candidates.end(),
            [](const OrbitMatch& a, const OrbitMatch& b) {
              return a.confidence > b.confidence;
            });

  // Phase 6: Winner selection (OrbitRing::winner_index)
  // Select top candidates, ensuring we maintain diversity across grammar modes
  MatchResults final_candidates;
  std::unordered_map<GrammarType, size_t> mode_counts;

  for (const auto& candidate : refined_candidates) {
    // Ensure we don't have too many candidates from the same grammar mode
    if (mode_counts[candidate.grammarType] >= 5) {  // Max 5 per mode
      continue;
    }

    if (candidate.confidence >= m_config.patternThreshold) {
      final_candidates.push_back(candidate);
      mode_counts[candidate.grammarType]++;

      if (final_candidates.size() >= m_config.maxMatches) {
        break;
      }
    }
  }

  return final_candidates;
}

double OrbitScanner::detectOrbitPattern(GrammarType grammar, const ::std::array<size_t, 6>& orbitCounts,
                                       size_t pos, const ::std::string& code) const {
  // FALLBACK HEURISTIC: Used only when signature patterns don't provide confidence
  // Returns delimiter-based score as weak signal - NOT ground truth
  // Prefer adding actual signature patterns (keywords) for truth-based detection

  switch (grammar) {
    case GrammarType::C: {
      // C code typically has balanced structures with some nesting
      // Look for moderate use of braces, parentheses, and minimal templates
      size_t totalDelimiters = orbitCounts[0] + orbitCounts[1] + orbitCounts[3];  // braces, brackets, parens
      size_t complexity = totalDelimiters;
      if (complexity > 2) {
        return ::std::min(1.0, static_cast<double>(complexity) / 5.0);
      }
      break;
    }

    case GrammarType::CPP2: {
      // CPP2 might have different patterns - for now, similar to C++ but with different weighting
      size_t cpp2Score = orbitCounts[0] + orbitCounts[3] * 2 + orbitCounts[4];  // braces, parens, quotes
      if (cpp2Score > 1) {
        return ::std::min(1.0, static_cast<double>(cpp2Score) / 3.0);
      }
      break;
    }

    default:
      break;
  }

  return 0.0;
}

DetectionResult OrbitScanner::analyzeMatches(const ::std::vector<OrbitMatch>& matches) const {
  DetectionResult result;

  if (matches.empty()) {
    result.detectedGrammar = GrammarType::UNKNOWN;
    result.confidence = 0.0;
    result.reasoning = "No patterns matched in the code segment";
    return result;
  }

  // Calculate confidence scores for each grammar
  ::std::unordered_map<GrammarType, double> grammarScores;
  ::std::unordered_map<GrammarType, size_t> matchCounts;

  for (const auto& match : matches) {
    grammarScores[match.grammarType] += match.confidence;
    matchCounts[match.grammarType]++;
  }

  // Normalize scores by match count and apply context analysis
  for (auto& [grammar, score] : grammarScores) {
    size_t count = matchCounts[grammar];
    score = (score / count) * ::std::min(1.0, static_cast<double>(count) / 10.0);
    // Clamp to [0.0, 1.0] range
    score = ::std::min(1.0, ::std::max(0.0, score));
  }

  result.grammarScores = grammarScores;
  result.matches = matches;

  // Determine best grammar
  result.detectedGrammar = determineBestGrammar(grammarScores);
  result.confidence = ::std::min(1.0, ::std::max(0.0, grammarScores[result.detectedGrammar]));

  // Generate reasoning
  result.reasoning = generateReasoning(result);

  return result;
}

double OrbitScanner::calculateGrammarConfidence(GrammarType grammar,
                                               const ::std::vector<OrbitMatch>& matches) const {
  double totalScore = 0.0;
  size_t count = 0;

  for (const auto& match : matches) {
    if (match.grammarType == grammar) {
      totalScore += match.confidence;
      count++;
    }
  }

  if (count == 0) return 0.0;

  // Apply diminishing returns for too many matches (potential false positives)
  double densityFactor = ::std::min(1.0, static_cast<double>(count) / 20.0);

  return (totalScore / count) * densityFactor;
}

GrammarType OrbitScanner::determineBestGrammar(const ::std::unordered_map<GrammarType, double>& scores) const {
  GrammarType bestGrammar = GrammarType::UNKNOWN;
  double bestScore = 0.0;

  for (const auto& [grammar, score] : scores) {
    if (score > bestScore) {
      bestScore = score;
      bestGrammar = grammar;
    }
  }

  return bestGrammar;
}

::std::string OrbitScanner::generateReasoning(const DetectionResult& result) const {
  ::std::stringstream ss;

  ss << "Detected grammar: " << static_cast<int>(result.detectedGrammar)
     << " with confidence " << result.confidence << ". ";

  if (!result.grammarScores.empty()) {
    ss << "Grammar scores: ";
    for (const auto& [grammar, score] : result.grammarScores) {
      ss << static_cast<int>(grammar) << "=" << score << " ";
    }
  }

  ss << "Matches found: " << result.matches.size();

  return ss.str();
}

bool OrbitScanner::validateConfig() const {
  return m_config.maxDepth > 0 &&
         m_config.maxMatches > 0 &&
         m_config.patternThreshold >= 0.0 &&
         m_config.patternThreshold <= 1.0 &&
         !m_config.patternsDir.empty();
}

bool OrbitScanner::validateInitialization() const {
  return m_loader && m_loader->getAllPatterns().size() > 0;
}

// Helper methods for CPP2 key resolution
std::string OrbitScanner::determine_scope_type(const std::string& context) const {
  if (context.find("class ") != std::string::npos ||
      context.find("struct ") != std::string::npos) {
    return "class_body";
  } else if (context.find("function") != std::string::npos ||
             context.find("(") != std::string::npos) {
    return "function_body";
  }
  return "global";
}

uint16_t OrbitScanner::determine_lattice_mask(const std::string& context) const {
  // Simplified lattice mask determination
  // In full implementation, this would analyze token types
  uint16_t mask = 0;

  if (context.find(":") != std::string::npos) {
    mask |= (1 << 2);  // PUNCTUATION
  }
  if (context.find_first_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") != std::string::npos) {
    mask |= (1 << 9);  // IDENTIFIER
  }

  return mask > 0 ? mask : 0xFFFF;  // Default: all classes
}

} // namespace ir
} // namespace cppfort