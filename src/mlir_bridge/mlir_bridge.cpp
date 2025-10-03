#include "mlir_bridge.h"

#include "mlir_abstractions.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser/Parser.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <stdexcept>

using namespace mlir;
using namespace mlir;

namespace son {

std::size_t Graph::addConstant(std::int64_t value) {
  nodes_.push_back(Node{.opcode = Opcode::Constant, .value = value});
  return nodes_.size() - 1;
}

std::size_t Graph::addAdd(std::size_t lhs, std::size_t rhs) {
  assert(lhs < nodes_.size() && rhs < nodes_.size());
  nodes_.push_back(Node{.opcode = Opcode::Add, .lhs = lhs, .rhs = rhs});
  return nodes_.size() - 1;
}

std::size_t Graph::addSubtract(std::size_t lhs, std::size_t rhs) {
  assert(lhs < nodes_.size() && rhs < nodes_.size());
  nodes_.push_back(Node{.opcode = Opcode::Subtract, .lhs = lhs, .rhs = rhs});
  return nodes_.size() - 1;
}

std::size_t Graph::addMultiply(std::size_t lhs, std::size_t rhs) {
  assert(lhs < nodes_.size() && rhs < nodes_.size());
  nodes_.push_back(Node{.opcode = Opcode::Multiply, .lhs = lhs, .rhs = rhs});
  return nodes_.size() - 1;
}

std::size_t Graph::addDivide(std::size_t lhs, std::size_t rhs) {
  assert(lhs < nodes_.size() && rhs < nodes_.size());
  nodes_.push_back(Node{.opcode = Opcode::Divide, .lhs = lhs, .rhs = rhs});
  return nodes_.size() - 1;
}

std::size_t Graph::addCompare(std::size_t lhs, std::size_t rhs, const std::string& predicate) {
  assert(lhs < nodes_.size() && rhs < nodes_.size());
  nodes_.push_back(Node{.opcode = Opcode::Compare, .lhs = lhs, .rhs = rhs, .predicate = predicate});
  return nodes_.size() - 1;
}

} // namespace son

namespace son::bridge {

namespace {
constexpr llvm::StringLiteral kEntryName{"graph_entry"};

Location unknownLoc(MLIRContext &ctx) { return UnknownLoc::get(&ctx); }

void ensureDialectsLoaded(MLIRContext &ctx) {
  // Dialects will be loaded automatically when operations requiring them are created
  // This avoids API compatibility issues with getOrLoadDialect in MLIR 21.1.2
Value materializeNode(const son::Node &node, llvm::SmallVectorImpl<Value> &slot,
                      cppfort::mlir::MLIRBuilder &builder) {
  switch (node.opcode) {
  case son::Opcode::Constant: {
    auto type = builder.getI64Type();
    auto attr = builder.getBuilder().getIntegerAttr(type, node.value);
    auto constant = builder.getBuilder().create<arith::ConstantOp>(builder.getLoc(), type, attr);
    slot.push_back(constant.getResult());
    return constant.getResult();
  }
  case son::Opcode::Add: {
    assert(node.lhs < slot.size() && node.rhs < slot.size());
    auto add = builder.createAdd(slot[node.lhs], slot[node.rhs]);
    slot.push_back(add);
    return add;
  }
  case son::Opcode::Subtract: {
    assert(node.lhs < slot.size() && node.rhs < slot.size());
    auto sub = builder.createSub(slot[node.lhs], slot[node.rhs]);
    slot.push_back(sub);
    return sub;
  }
  case son::Opcode::Multiply: {
    assert(node.lhs < slot.size() && node.rhs < slot.size());
    auto mul = builder.createMul(slot[node.lhs], slot[node.rhs]);
    slot.push_back(mul);
    return mul;
  }
  case son::Opcode::Divide: {
    assert(node.lhs < slot.size() && node.rhs < slot.size());
    auto div = builder.createDiv(slot[node.lhs], slot[node.rhs]);
    slot.push_back(div);
    return div;
  }
  case son::Opcode::Compare: {
    assert(node.lhs < slot.size() && node.rhs < slot.size());
    Value cmp;
    if (node.predicate == "eq") cmp = builder.createCmpEQ(slot[node.lhs], slot[node.rhs]);
    else if (node.predicate == "ne") cmp = builder.createCmpNE(slot[node.lhs], slot[node.rhs]);
    else if (node.predicate == "slt") cmp = builder.createCmpLT(slot[node.lhs], slot[node.rhs]);
    else if (node.predicate == "sle") cmp = builder.createCmpLE(slot[node.lhs], slot[node.rhs]);
    else if (node.predicate == "sgt") cmp = builder.createCmpGT(slot[node.lhs], slot[node.rhs]);
    else if (node.predicate == "sge") cmp = builder.createCmpGE(slot[node.lhs], slot[node.rhs]);
    else {
      throw std::runtime_error("Unsupported comparison predicate: " + node.predicate);
    }
    slot.push_back(cmp);
    return cmp;
  }
  }
  llvm_unreachable("unsupported opcode");
}
  llvm_unreachable("unsupported opcode");
}

} // namespace
ModuleOp exportToMLIR(MLIRContext &context, const Graph &graph) {
::mlir::ModuleOp exportToMLIR(::mlir::MLIRContext &context, const Graph &graph) {
  if (graph.nodes().empty()) {
    throw std::runtime_error("MLIR export failed: Cannot export empty graph. "
ModuleOp exportToMLIR(MLIRContext &context, const Graph &graph) {
  if (graph.nodes().empty()) {
    throw std::runtime_error("MLIR export failed: Cannot export empty graph. "
                           "Graph must contain at least one node.");
  }

  ensureDialectsLoaded(context);

  // Parse valid empty module for MLIR 21.1.2 compatibility
  auto owningOp = mlir::parseSourceString("module {}", &context);
  if (!owningOp) {
    throw std::runtime_error("MLIR export failed: Could not create empty module");
  }
  ModuleOp module = cast<ModuleOp>(owningOp.get());
  OpBuilder rootBuilder(&context);

  // Construct `func.func @graph_entry() -> i64`.
  auto funcType = rootBuilder.getFunctionType({}, rootBuilder.getI64Type());
  auto func = func::FuncOp::create(unknownLoc(context), kEntryName, funcType);
  module.getBody()->push_back(func);
  auto *block = func.addEntryBlock();
  cppfort::mlir::MLIRBuilder builder(rootBuilder, unknownLoc(context));
  builder.getBuilder().setInsertionPointToStart(block);
  llvm::SmallVector<Value> valueSlots;
  valueSlots.reserve(graph.nodes().size());

  unsigned nodeIndex = 0;
  for (const auto &node : graph.nodes()) {
    try {
      materializeNode(node, valueSlots, builder);
      ++nodeIndex;
    } catch (const std::exception& e) {
      throw std::runtime_error("MLIR export failed at node " + std::to_string(nodeIndex) +
                             ": " + e.what());
    }
  }

  if (valueSlots.empty()) {
    throw std::runtime_error("MLIR export failed: No values were materialized from graph nodes. "
                           "This indicates an issue with the graph structure or node materialization.");
  }

  builder.createReturn(valueSlots.back());

  return module;
}

Graph importFromMLIR(::mlir::ModuleOp module) {
  if (!func) {
    throw std::runtime_error("MLIR import failed: Expected func.func named 'graph_entry' in module, but none found. "
                           "Module may not have been created by exportToMLIR or has been modified.");
  }

  auto &block = func.getBody().front();
  llvm::DenseMap<Value, std::size_t> valueToNode;

  unsigned opIndex = 0;
  for (auto &op : block) {
    try {
      if (auto constant = llvm::dyn_cast<arith::ConstantOp>(op)) {
        auto attr = llvm::dyn_cast<IntegerAttr>(constant.getValue());
        if (!attr) {
          throw std::runtime_error("MLIR import failed at operation " + std::to_string(opIndex) + 
                                 ": Unsupported constant type. Only integer constants are supported.");
        }
        if (!attr.getType().isInteger(64)) {
          std::string typeStr;
          llvm::raw_string_ostream typeStream(typeStr);
          attr.getType().print(typeStream);
          throw std::runtime_error("MLIR import failed at operation " + std::to_string(opIndex) + 
                                 ": Constant must be i64 type, got " + typeStr + ".");
        }
        auto index = graph.addConstant(attr.getInt());
        valueToNode.insert({constant.getResult(), index});
        ++opIndex;
        continue;
      }

      if (auto add = llvm::dyn_cast<arith::AddIOp>(op)) {
        auto lhsIt = valueToNode.find(add.getLhs());
        auto rhsIt = valueToNode.find(add.getRhs());
        if (lhsIt == valueToNode.end() || rhsIt == valueToNode.end()) {
          throw std::runtime_error("MLIR import failed at operation " + std::to_string(opIndex) + 
                                 ": AddIOp operands must reference previously defined values.");
        }
        auto index = graph.addAdd(lhsIt->second, rhsIt->second);
        valueToNode.insert({add.getResult(), index});
        ++opIndex;
        continue;
      }

      if (auto sub = llvm::dyn_cast<arith::SubIOp>(op)) {
        auto lhsIt = valueToNode.find(sub.getLhs());
        auto rhsIt = valueToNode.find(sub.getRhs());
        if (lhsIt == valueToNode.end() || rhsIt == valueToNode.end()) {
          throw std::runtime_error("MLIR import failed at operation " + std::to_string(opIndex) + 
                                 ": SubIOp operands must reference previously defined values.");
        }
        auto index = graph.addSubtract(lhsIt->second, rhsIt->second);
        valueToNode.insert({sub.getResult(), index});
        ++opIndex;
        continue;
      }

      if (auto mul = llvm::dyn_cast<arith::MulIOp>(op)) {
        auto lhsIt = valueToNode.find(mul.getLhs());
        auto rhsIt = valueToNode.find(mul.getRhs());
        if (lhsIt == valueToNode.end() || rhsIt == valueToNode.end()) {
          throw std::runtime_error("MLIR import failed at operation " + std::to_string(opIndex) + 
                                 ": MulIOp operands must reference previously defined values.");
        }
        auto index = graph.addMultiply(lhsIt->second, rhsIt->second);
        valueToNode.insert({mul.getResult(), index});
        ++opIndex;
        continue;
      }

      if (auto div = llvm::dyn_cast<arith::DivSIOp>(op)) {
        auto lhsIt = valueToNode.find(div.getLhs());
        auto rhsIt = valueToNode.find(div.getRhs());
        if (lhsIt == valueToNode.end() || rhsIt == valueToNode.end()) {
          throw std::runtime_error("MLIR import failed at operation " + std::to_string(opIndex) + 
                                 ": DivSIOp operands must reference previously defined values.");
        }
        auto index = graph.addDivide(lhsIt->second, rhsIt->second);
        valueToNode.insert({div.getResult(), index});
        ++opIndex;
        continue;
      }

      if (auto cmp = llvm::dyn_cast<arith::CmpIOp>(op)) {
        auto lhsIt = valueToNode.find(cmp.getLhs());
        auto rhsIt = valueToNode.find(cmp.getRhs());
        if (lhsIt == valueToNode.end() || rhsIt == valueToNode.end()) {
          throw std::runtime_error("MLIR import failed at operation " + std::to_string(opIndex) + 
                                 ": CmpIOp operands must reference previously defined values.");
        }
        std::string predicate;
        switch (cmp.getPredicate()) {
        case arith::CmpIPredicate::eq: predicate = "eq"; break;
        case arith::CmpIPredicate::ne: predicate = "ne"; break;
        case arith::CmpIPredicate::slt: predicate = "slt"; break;
        case arith::CmpIPredicate::sle: predicate = "sle"; break;
        case arith::CmpIPredicate::sgt: predicate = "sgt"; break;
        case arith::CmpIPredicate::sge: predicate = "sge"; break;
        case arith::CmpIPredicate::ult: predicate = "ult"; break;
        case arith::CmpIPredicate::ule: predicate = "ule"; break;
        case arith::CmpIPredicate::ugt: predicate = "ugt"; break;
        case arith::CmpIPredicate::uge: predicate = "uge"; break;
        default:
          throw std::runtime_error("MLIR import failed at operation " + std::to_string(opIndex) + 
                                 ": Unsupported comparison predicate.");
        }
        auto index = graph.addCompare(lhsIt->second, rhsIt->second, predicate);
        valueToNode.insert({cmp.getResult(), index});
        ++opIndex;
        continue;
      }

      if (llvm::isa<func::ReturnOp>(op)) {
        break;
      }

      // Enhanced error reporting for unsupported operations
      std::string opName = op.getName().getStringRef().str();
      throw std::runtime_error("MLIR import failed at operation " + std::to_string(opIndex) + 
                             ": Encountered unsupported operation '" + opName + "'. " +
                             "Supported operations: arith.constant, arith.addi, arith.subi, arith.muli, arith.divsi, arith.cmpi, func.return. " +
                             "Module may contain operations not supported by this Sea-of-Nodes bridge.");
    } catch (const std::exception& e) {
      // Re-throw with additional context
      throw std::runtime_error(std::string("MLIR import error: ") + e.what());
    }
  }

  if (graph.nodes().empty()) {
    throw std::runtime_error("MLIR import failed: No operations found in function body. "
                           "Function must contain at least one operation.");
  }

  return graph;
}

} // namespace son::bridge
