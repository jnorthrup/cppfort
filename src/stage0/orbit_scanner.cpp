#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <filesystem>

#include "multi_grammar_loader.h"
#include "orbit_scanner.h"
#include "rabin_karp.h"
#include "wide_scanner.h"
#include "projection_oracle.h"
#include "rbcursive.h"
#include "rbcursive_combinators.h"

namespace cppfort {
namespace ir {

OrbitScanner::OrbitScanner(const OrbitScannerConfig& config,
                           ::std::unique_ptr<MultiGrammarLoader> loader)
  : m_config(config),
    m_rabinKarp(::std::make_unique<RabinKarp>()),
    m_context(::std::make_unique<OrbitContext>(m_config.maxDepth)),
    m_loader(::std::move(loader)),
    m_unifiedDatabase(::std::make_unique<UnifiedOrbitDatabase>()) {
  if (!m_loader) {
    m_loader = ::std::unique_ptr<MultiGrammarLoader>(new MultiGrammarLoader());
  }
  // Initialize private rbcursive scanner asset
  m_rbcursive = ::std::make_unique<RBCursiveScanner>();
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

  // Load unified orbit patterns
  ::std::filesystem::path unifiedPatternsPath = m_config.patternsDir / "cppfort_core_patterns.yaml";
  if (::std::filesystem::exists(unifiedPatternsPath)) {
    if (!m_unifiedDatabase->loadUnifiedPatterns(unifiedPatternsPath.string())) {
      ::std::cerr << "Warning: Failed to load unified orbit patterns from " << unifiedPatternsPath << ::std::endl;
      // Continue without unified patterns - not a fatal error
    } else {
      auto stats = m_unifiedDatabase->getUnificationStats();
      ::std::cout << "Loaded unified orbit patterns: " << stats.total_patterns
                  << " total, " << stats.unified_patterns << " unified ("
                  << (stats.unification_ratio * 100.0) << "% unification ratio)" << ::std::endl;
    }
  } else {
    ::std::cout << "Unified orbit patterns file not found at " << unifiedPatternsPath
                << " - continuing with legacy patterns only" << ::std::endl;
  }

  return validateInitialization();
}

DetectionResult OrbitScanner::scan(const ::std::string& code) const {
  // Get all patterns from loaded grammars
  auto allPatterns = m_loader->getAllPatterns();

  // Get unified patterns for all grammars
  ::std::vector<UnifiedOrbitPattern> unifiedPatterns;
  if (m_unifiedDatabase) {
    // Get patterns for all supported grammars
    auto cPatterns = m_unifiedDatabase->getPatternsForGrammar(0x01); // C
    auto cppPatterns = m_unifiedDatabase->getPatternsForGrammar(0x02); // CPP
    auto cpp2Patterns = m_unifiedDatabase->getPatternsForGrammar(0x04); // CPP2

    unifiedPatterns.insert(unifiedPatterns.end(), cPatterns.begin(), cPatterns.end());
    unifiedPatterns.insert(unifiedPatterns.end(), cppPatterns.begin(), cppPatterns.end());
    unifiedPatterns.insert(unifiedPatterns.end(), cpp2Patterns.begin(), cpp2Patterns.end());
  }

  return scan(code, allPatterns, unifiedPatterns);
}

DetectionResult OrbitScanner::scan(const ::std::string& code,
                                  const ::std::vector<OrbitPattern>& patterns) const {
  // Convert legacy patterns to unified format for consistency
  ::std::vector<UnifiedOrbitPattern> unifiedPatterns;
  return scan(code, patterns, unifiedPatterns);
}

MatchResults OrbitScanner::findUnifiedMatches(const ::std::string& code,
                                            const ::std::vector<UnifiedOrbitPattern>& patterns) const {
  MatchResults results;

  // Create a temporary OrbitContext for scanning
  OrbitContext scanContext(m_config.maxDepth);

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

    // Detect unified grammar patterns
    for (const auto& pattern : patterns) {
      // Determine grammar type from pattern category
      GrammarType grammar = GrammarType::UNKNOWN;
      if (pattern.category >= UnifiedOrbitCategory::CPP2_Only) {
        if (pattern.category == UnifiedOrbitCategory::CPP2_Only) {
          grammar = GrammarType::CPP2;
        } else if (pattern.category == UnifiedOrbitCategory::CPP_Only) {
          grammar = GrammarType::CPP;
        } else {
          grammar = GrammarType::C;
        }
      } else {
        // Common trunk patterns - determine from context or use multi-grammar
        // For now, default to CPP2 for common patterns (can be refined)
        grammar = GrammarType::CPP2;
      }

      // Get signatures for this grammar
      auto signatures = pattern.getSignaturesForGrammar(static_cast<uint8_t>(grammar));

      bool signatureMatched = signatures.empty();
      ::std::string matchedSignature;

      if (!signatureMatched) {
        for (const auto& signature : signatures) {
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

      // Validate depth context (using pattern's expected depth if specified)
      int currentDepth = scanContext.getDepth();
      // Note: Unified patterns don't have expected_depth field, could be added later

      // Compute current confix context bitmask
      uint8_t ctxMask = scanContext.confixMask();

      // Check confix mask requirements
      if ((pattern.confix_mask & ctxMask) == 0) {
        continue;
      }

      if (matchedSignature.empty()) {
        matchedSignature = ::std::string(1, ch);
      }

      // Signature match + valid confix = high confidence
      double confidence = signatureMatched ? 0.95 : 0.25;

      // Apply pattern weight
      confidence *= pattern.weight;

      // Report matches that meet threshold
      if (confidence >= m_config.patternThreshold) {
        OrbitMatch match;
        match.patternName = pattern.name;
        match.grammarType = grammar;
        match.startPos = pos;
        match.endPos = pos + matchedSignature.size();
        match.signature = matchedSignature;
        match.confidence = confidence;
        match.orbitCounts = orbitCounts;
        match.orbitHashes = orbitHashes;

        results.push_back(match);

        // Limit results to prevent excessive memory usage
        if (results.size() >= m_config.maxMatches) {
          break;
        }
      }
    }

    if (results.size() >= m_config.maxMatches) {
      break;
    }
  }

  return results;
}

/**
 * Scan code with both legacy and unified patterns
 * @param code Source code to scan
 * @param legacyPatterns Legacy orbit patterns
 * @param unifiedPatterns Unified orbit patterns
 * @return Detection result
 */
DetectionResult OrbitScanner::scan(const ::std::string& code,
                                  const ::std::vector<OrbitPattern>& legacyPatterns,
                                  const ::std::vector<UnifiedOrbitPattern>& unifiedPatterns) const {
  // Find all matches using both pattern types
  auto legacyMatches = findMatches(code, legacyPatterns);
  auto unifiedMatches = findUnifiedMatches(code, unifiedPatterns);

  // Combine matches
  MatchResults allMatches;
  allMatches.insert(allMatches.end(), legacyMatches.begin(), legacyMatches.end());
  allMatches.insert(allMatches.end(), unifiedMatches.begin(), unifiedMatches.end());

  // Analyze combined matches to determine grammar
  return analyzeMatches(allMatches);
}

const OrbitScannerConfig& OrbitScanner::getConfig() const {
  return m_config;
}

void OrbitScanner::updateConfig(const OrbitScannerConfig& config) {
  m_config = config;
  // Reinitialize components with new config
  m_rabinKarp = ::std::make_unique<RabinKarp>();
  m_context = ::std::make_unique<OrbitContext>(m_config.maxDepth);
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

  return results;
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
  double bestScore = m_config.patternThreshold;

  for (const auto& [grammar, score] : scores) {
    if (score > bestScore) {
      bestScore = score;
      bestGrammar = grammar;
    }
  }

  return bestGrammar;
}

::std::string OrbitScanner::generateReasoning(const DetectionResult& result) const {
  if (result.detectedGrammar == GrammarType::UNKNOWN) {
    return "No grammar detected with sufficient confidence";
  }

  ::std::string reasoning = "Detected " + grammarTypeToString(result.detectedGrammar) +
                         " with " + ::std::to_string(result.confidence * 100) + "% confidence. ";

  // Add details about matches
  ::std::unordered_map<GrammarType, size_t> matchCounts;
  for (const auto& match : result.matches) {
    matchCounts[match.grammarType]++;
  }

  reasoning += "Found " + ::std::to_string(result.matches.size()) + " pattern matches across " +
              ::std::to_string(matchCounts.size()) + " grammar types.";

  return reasoning;
}

bool OrbitScanner::validateConfig() const {
  if (m_config.patternThreshold < 0.0 || m_config.patternThreshold > 1.0) return false;
  if (m_config.maxDepth == 0) return false;

  return true;
}

bool OrbitScanner::validateInitialization() const {
  if (!m_rabinKarp || !m_context || !m_loader) return false;

  // Check if at least one grammar is loaded
  auto loadedGrammars = m_loader->getLoadedGrammars();
  if (loadedGrammars.empty()) return false;

  // Check if patterns are loaded
  size_t totalPatterns = 0;
  for (auto grammar : loadedGrammars) {
    totalPatterns += m_loader->getPatterns(grammar).size();
  }

  return totalPatterns > 0;
}

::std::string detectionResultToString(const DetectionResult& result) {
  ::std::string output = "Detection Result:\n";
  output += "  Grammar: " + grammarTypeToString(result.detectedGrammar) + "\n";
  output += "  Confidence: " + ::std::to_string(result.confidence * 100) + "%\n";
  output += "  Matches: " + ::std::to_string(result.matches.size()) + "\n";
  output += "  Reasoning: " + result.reasoning + "\n";

  if (!result.grammarScores.empty()) {
    output += "  Grammar Scores:\n";
    for (const auto& [grammar, score] : result.grammarScores) {
      output += "    " + grammarTypeToString(grammar) + ": " +
               ::std::to_string(score * 100) + "%\n";
    }
  }

  return output;
}

void printDetectionResult(const DetectionResult& result) {
  ::std::cout << detectionResultToString(result) << ::std::endl;
}

} // namespace ir
} // namespace cppfort
