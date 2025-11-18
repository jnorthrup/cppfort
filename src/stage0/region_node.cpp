#include "region_node.h"
#include <iostream>
#include <sstream>

using namespace cppfort::ir::mlir;

static const char* toRegionTypeString(cppfort::ir::mlir::RegionNode::RegionType t) {
    switch (t) {
        case RegionNode::RegionType::UNKNOWN: return "UNKNOWN";
        case RegionNode::RegionType::FUNCTION: return "FUNCTION";
        case RegionNode::RegionType::BLOCK: return "BLOCK";
        case RegionNode::RegionType::NAMED_REGION: return "NAMED_REGION";
        case RegionNode::RegionType::CONDITIONAL: return "CONDITIONAL";
        case RegionNode::RegionType::LOOP: return "LOOP";
        case RegionNode::RegionType::INITIALIZER: return "INITIALIZER";
        case RegionNode::RegionType::RETURN_REGION: return "RETURN_REGION";
    }
    return "<invalid>";

}

std::string cppfort::ir::mlir::RegionNode::toString(size_t indent) const {
    std::ostringstream out;
    out << std::string(indent, ' ')
        << toRegionTypeString(type_) << "(" << (name_.empty() ? "<anon>" : name_) << ")";
    if (!arguments_.empty()) {
        out << " args=[";
        for (size_t i = 0; i < arguments_.size(); ++i) {
            if (i) out << ", ";
            out << arguments_[i];
        }
        out << "]";
    }
    if (!operations_.empty()) {
        out << " ops=" << operations_.size();
    }
    if (!values_.empty()) {
        out << " values=" << values_.size();
    }
    return out.str();
}

void cppfort::ir::mlir::RegionNode::printTree(size_t indent) const {
    std::cout << toString(indent) << "\n";
    for (const auto& child : children_) {
        if (child) child->printTree(indent + 2);
    }

}

