#include "rbcursive_regions.h"
#include <stack>
#include <iostream>
#include <algorithm>

namespace cppfort {
namespace ir {

// Helper: Check if a character is an opening confix
bool RBCursiveRegions::isOpeningConfix(char ch) const {
    for (const auto& [open, close] : confixPairs_) {
        if (open == ch) return true;
    }
    return false;
}

// Helper: Check if a character is a closing confix matching an opening
bool RBCursiveRegions::isMatchingClosingConfix(char open, char close) const {
    for (const auto& [o, c] : confixPairs_) {
        if (o == open && c == close) return true;
    }
    return false;
}

// Helper: Get confix depth change for a character
int RBCursiveRegions::getConfixDepthChange(char ch) const {
    for (const auto& [open, close] : confixPairs_) {
        if (open == ch) return 1;  // Opening confix increases depth
        if (close == ch) return -1; // Closing confix decreases depth
    }
    return 0;  // Not a confix character
}

// Wobbling window: Find valid region boundary by perturbing end position
size_t RBCursiveRegions::wobbleFindBoundary(const std::vector<WideScanner::BoundaryEvent>& events,
                                           size_t startIdx,
                                           size_t initialEndIdx,
                                           const std::string& source) const {
    if (!config_.enableWobbling) {
        return initialEndIdx;
    }
    
    size_t bestEndIdx = initialEndIdx;
    double bestConfidence = 0.0;
    int targetDepth = 0;  // Looking for depth to return to 0
    
    // Try perturbing forward and backward within reasonable bounds
    const size_t WOBBLE_RANGE = 50;  // Number of events to wobble in each direction
    
    size_t startPos = events[startIdx].position;
    size_t minIdx = (startIdx > WOBBLE_RANGE) ? startIdx - WOBBLE_RANGE : 0;
    size_t maxIdx = std::min(initialEndIdx + WOBBLE_RANGE, events.size() - 1);
    
    // Calculate initial depth at start position
    int startDepth = 0;
    for (size_t i = 0; i <= startIdx && i < events.size(); ++i) {
        startDepth += getConfixDepthChange(events[i].delimiter);
    }
    
    // Scan through candidate end positions
    int currentDepth = startDepth;
    for (size_t idx = startIdx + 1; idx <= maxIdx && idx < events.size(); ++idx) {
        currentDepth += getConfixDepthChange(events[idx].delimiter);
        
        // Check if depth returns to target (balanced)
        if (currentDepth == targetDepth) {
            // Calculate confidence for this boundary
            double confidence = calculateBoundaryConfidence(events[idx], currentDepth);
            confidence += 0.1 * (1.0 / (1.0 + abs((int)idx - (int)initialEndIdx)));  // Prefer closer to initial
            
            // Update best boundary if this is better
            if (confidence > bestConfidence) {
                bestConfidence = confidence;
                bestEndIdx = idx;
            }
        }
        
        // If we go too deep or invalid, stop
        if (currentDepth < targetDepth || currentDepth > 10) {
            break;
        }
    }
    
    return bestEndIdx;
}

// Determine region type from boundary evidence
mlir::RegionNode::RegionType RBCursiveRegions::inferRegionType(
    const WideScanner::BoundaryEvent& startBoundary,
    const std::vector<WideScanner::BoundaryEvent>& followingEvents) const {
    
    char startChar = startBoundary.delimiter;
    
    // Check for function-related patterns
    if (startChar == '(') {
        // Look backwards for function signature evidence
        if (startBoundary.evidence.total_tokens > 0) {
            // Heuristic: treat parentheses following an identifier or parameters as function
            if (startBoundary.evidence.c_identifiers > 0 || startBoundary.evidence.cpp_identifiers > 0 || startBoundary.evidence.template_ids > 0) {
                return mlir::RegionNode::RegionType::FUNCTION;
            }
        }
            return mlir::RegionNode::RegionType::BLOCK;
    }
    
    // Object literals/scope blocks
    if (startChar == '{') {
        // Check if this looks like a function body
        bool foundReturnType = false;
        bool foundParameters = false;
        
        for (const auto& event : followingEvents) {
            if (event.position > startBoundary.position + 100) break;  // Limit lookahead
            
            if (event.evidence.total_tokens > 0) {
                if (event.evidence.c_keywords > 0 || event.evidence.cpp_keywords > 0 || event.evidence.cpp2_keywords > 0) foundReturnType = true;
                if (event.evidence.template_ids > 0 || event.evidence.cpp_identifiers > 0 || event.evidence.c_identifiers > 0) foundParameters = true;
            }
        }
        
        if (foundReturnType || foundParameters) {
                return mlir::RegionNode::RegionType::FUNCTION;
        }
            return mlir::RegionNode::RegionType::BLOCK;
    }
    
    // Conditional structures
    if (startBoundary.evidence.total_tokens > 0 &&
        (startBoundary.evidence.flow_keywords > 0 || startBoundary.evidence.c_keywords > 0 || startBoundary.evidence.cpp_keywords > 0)) {
        // Heuristic: flow keywords indicate a conditional region
        return mlir::RegionNode::RegionType::CONDITIONAL;
    }
    }
    
        return mlir::RegionNode::RegionType::BLOCK;  // Default to block
}

// Calculate confidence score for a region boundary
double RBCursiveRegions::calculateBoundaryConfidence(
    const WideScanner::BoundaryEvent& event,
    size_t depth) const {
    
    double confidence = config_.minConfidence;
    
    // Boost confidence if this is a structural character
    if (event.is_delimiter) {
        confidence += 0.2;
    }
    
    // Boost if we have high orbit confidence
    if (event.orbit_confidence > 0.5) {
        confidence += event.orbit_confidence * 0.3;
    }
    
    // Boost if depth is reasonable (not too nested)
    if (depth <= 5) {
        confidence += 0.1;
    } else if (depth > 10) {
        confidence -= 0.1;
    }
    
    // If we have TypeEvidence, boost confidence
    if (event.evidence.total_tokens > 0) {
        confidence += 0.15;
    }
    
    return std::min(1.0, confidence);
}

// Recursive region carving
std::unique_ptr<mlir::RegionNode> RBCursiveRegions::carveRegionRecursive(
    const std::vector<WideScanner::BoundaryEvent>& events,
    const std::string& source,
    size_t& currentIdx,
    size_t endIdx,
    mlir::RegionNode::RegionType parentType) {
    
    if (currentIdx >= events.size() || currentIdx >= endIdx) {
        return nullptr;
    }
    
    const auto& startEvent = events[currentIdx];
    
    // Create a new region starting at this boundary
    auto region = std::make_unique<mlir::RegionNode>();
    
    // Infer region type
    std::vector<WideScanner::BoundaryEvent> followingEvents;
    size_t lookahead = std::min(currentIdx + 20, events.size());
    for (size_t i = currentIdx + 1; i < lookahead; ++i) {
        followingEvents.push_back(events[i]);
    }
    
    region->setType(inferRegionType(startEvent, followingEvents));
    region->setSourceLocation(startEvent.position, startEvent.position);
    
    // Calculate initial confidence
    double confidence = calculateBoundaryConfidence(startEvent, currentDepth_);
    region->setOrbitConfidence(confidence);
    
    // Add orbit position
    region->addOrbitPosition(startEvent.position);
    
    // Look for matching closing boundary
    size_t startIdx = currentIdx;
    size_t regionStart = currentIdx;
    size_t regionEnd = endIdx;
    int localDepth = 0;
    
    // Check if this is a structural boundary that might contain nested regions
    if (isOpeningConfix(startEvent.delimiter)) {
        char openChar = startEvent.delimiter;
        localDepth = 1;  // Start with depth 1 for the opening
        
        // Scan forward to find the matching closing confix
        for (size_t idx = currentIdx + 1; idx < endIdx && idx < events.size(); ++idx) {
            char currentChar = events[idx].delimiter;
            int depthChange = getConfixDepthChange(currentChar);
            
            if (depthChange > 0) {
                // Found nested opening - recurse
                size_t nestedIdx = idx;
                auto nestedRegion = carveRegionRecursive(events, source, nestedIdx, endIdx, region->getType());
                if (nestedRegion) {
                    region->addChild(std::move(nestedRegion));
                    idx = nestedIdx;  // Skip past the nested region
                }
                localDepth += depthChange;
            } else if (depthChange < 0) {
                // Found closing
                localDepth += depthChange;
                
                // If depth returns to 0, this is our matching closing boundary
                if (localDepth == 0 && isMatchingClosingConfix(openChar, currentChar)) {
                    regionEnd = idx;
                    region->setSourceLocation(events[regionStart].position, events[idx].position);
                    
                    // Add closing orbit position
                    region->addOrbitPosition(events[idx].position);
                    break;
                }
            }
        }
    }
    
    // Apply wobbling to optimize boundary
    if (config_.enableWobbling && regionEnd > regionStart) {
        size_t optimizedEnd = wobbleFindBoundary(events, regionStart, regionEnd, source);
        if (optimizedEnd != regionEnd) {
            regionEnd = optimizedEnd;
            region->setSourceLocation(events[regionStart].position, events[optimizedEnd].position);
        }
    }
    
    // Process events within this region for operations and values
    for (size_t idx = regionStart + 1; idx < regionEnd && idx < events.size(); ++idx) {
        const auto& event = events[idx];
        
        // Check if this looks like an identifier or literal
        if (event.evidence.total_tokens > 0) {
            
            // Detect identifiers
            // Heuristics: detect identifier-like tokens via identifier counters
            if (event.evidence.c_identifiers > 0 || event.evidence.cpp_identifiers > 0) {
                mlir::RegionNode::Value value;
                value.name = source.substr(event.position, 20);  // Take first 20 chars as name
                value.type = "TODO_infer_type";
                value.defining_op = region->getOperations().size();
                region->addValue(value);
            }
        }
    }
    
    // Update current index to position after this region
    currentIdx = regionEnd + 1;
    
    return region;
}

// Main carving method
RBCursiveRegions::CarveResult RBCursiveRegions::carveRegions(
    const std::vector<WideScanner::BoundaryEvent>& events,
    const std::string& source) {
    
    CarveResult result;
    
    if (events.empty()) {
        result.success = false;
        result.errorMessage = "No boundary events provided";
        return result;
    }
    
    try {
        // Start carving from the beginning
        size_t currentIdx = 0;
        auto rootRegion = carveRegionRecursive(events, source, currentIdx, events.size(), 
                                                  mlir::RegionNode::RegionType::FUNCTION);
        
        if (rootRegion) {
            result.rootRegion = std::move(rootRegion);
            result.success = true;
            
            // Collect all regions for easy access
            std::function<void(const mlir::RegionNode&)> collectRegions = [&](const mlir::RegionNode& node) {
                result.allRegions.push_back(&node);
                for (const auto& child : node.getChildren()) {
                    collectRegions(*child);
                }
            };
            
            collectRegions(*result.rootRegion);
            result.regionCount = result.allRegions.size();
        } else {
            result.success = false;
            result.errorMessage = "Failed to carve any regions";
        }
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Exception during carving: ") + e.what();
    }
    
    return result;
}

// Carve regions from a specific span
RBCursiveRegions::CarveResult RBCursiveRegions::carveRegionsInSpan(
    const std::vector<WideScanner::BoundaryEvent>& events,
    const std::string& source,
    size_t startPos,
    size_t endPos) {
    
    // Find event indices corresponding to character positions
    size_t startIdx = 0;
    size_t endIdx = events.size() - 1;
    
    for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].position >= startPos && startIdx == 0) {
            startIdx = i;
        }
        if (events[i].position >= endPos && endIdx == events.size() - 1) {
            endIdx = i;
            break;
        }
    }
    
    // Adjust endIdx if we couldn't find exact match
    if (endIdx <= startIdx) {
        endIdx = std::min(startIdx + 100, events.size() - 1);  // Try 100 events or until end
    }
    
    CarveResult result;
    size_t currentIdx = startIdx;
    
    auto region = carveRegionRecursive(events, source, currentIdx, endIdx, 
                                      mlir::RegionNode::RegionType::BLOCK);
    
    if (region) {
        result.rootRegion = std::move(region);
        result.success = true;
        result.allRegions.push_back(result.rootRegion.get());
        result.regionCount = 1;
    }
    
    return result;
}

// Debug helper: Print carved region tree
void RBCursiveRegions::printCarvedRegions(const mlir::RegionNode& root, size_t indent) {
    std::string indentStr(indent * 2, ' ');
    
    std::cout << indentStr << "Region(type=";
    switch (root.getType()) {
        case mlir::RegionNode::RegionType::FUNCTION: std::cout << "FUNCTION"; break;
        case mlir::RegionNode::RegionType::BLOCK: std::cout << "BLOCK"; break;
        case mlir::RegionNode::RegionType::CONDITIONAL: std::cout << "CONDITIONAL"; break;
        case mlir::RegionNode::RegionType::LOOP: std::cout << "LOOP"; break;
        default: std::cout << "UNKNOWN"; break;
    }
    
    std::cout << ", confidence=" << root.getOrbitConfidence()
              << ", position=[" << root.getSourceStart() << "-" << root.getSourceEnd() << "])
";
    
    if (!root.getName().empty()) {
        std::cout << indentStr << "  name: \"" << root.getName() << "\"\n";
    }
    
    std::cout << indentStr << "  operations: " << root.getOperations().size()
              << ", values: " << root.getValues().size()
              << ", children: " << root.getChildren().size() << "\n";
    
    for (const auto& child : root.getChildren()) {
        printCarvedRegions(*child, indent + 2);
    }
}

// Helper function to generate boundary events from source
std::vector<WideScanner::BoundaryEvent> generateBoundaryEventsFromSource(
    const std::string& source,
    bool includeStructuralChars) {
    
    std::vector<WideScanner::BoundaryEvent> events;
    
    for (size_t i = 0; i < source.length(); ++i) {
        char ch = source[i];
        
        // Check if this is a structural character we care about
        bool isStructural = (ch == '{' || ch == '}' || ch == '(' || ch == ')' || 
                           ch == '[' || ch == ']' || ch == ';' || ch == ',');
        
        if (isStructural || !includeStructuralChars) {
            WideScanner::BoundaryEvent event;
            event.position = i;
            event.delimiter = ch;
            event.is_delimiter = isStructural;
            
            events.push_back(event);
        }
    }
    
    return events;
}

// Standalone carving function
RBCursiveRegions::CarveResult carveRegionsStandalone(
    const std::string& source,
    const RBCursiveRegions::CarveConfig& config) {
    
    // Generate boundary events from source
    auto events = generateBoundaryEventsFromSource(source, true);
    
    // Create carver and process
    RBCursiveRegions carver(config);
    return carver.carveRegions(events, source);
}

} // namespace ir
} // namespace cppfort
