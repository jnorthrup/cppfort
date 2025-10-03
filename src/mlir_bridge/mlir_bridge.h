#include "mlir_abstractions.h"
#include <mlir/IR/BuiltinOps.h>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace son {

namespace son {
} // namespace mlir

namespace son {

/// Minimal Sea-of-Nodes opcode set supported by the bridge.
enum class Opcode : std::uint8_t {
  Constant,
  Add,
  Subtract,
  Multiply,
  Divide,
  Compare
};

/// Lightweight node representation used for round-tripping with MLIR.
struct Node {
  Opcode opcode{};
  std::int64_t value = 0;     ///< Valid when opcode == Constant
  std::size_t lhs = 0;        ///< Index of left operand
  std::size_t rhs = 0;        ///< Index of right operand
  std::string predicate;      ///< Comparison predicate (when opcode == Compare)
};

/// Simple graph container that tracks insertion order for deterministic export.
class Graph {
public:
  [[nodiscard]] std::size_t addConstant(std::int64_t value);
  [[nodiscard]] std::size_t addAdd(std::size_t lhs, std::size_t rhs);
  [[nodiscard]] std::size_t addSubtract(std::size_t lhs, std::size_t rhs);
  [[nodiscard]] std::size_t addMultiply(std::size_t lhs, std::size_t rhs);
  [[nodiscard]] std::size_t addDivide(std::size_t lhs, std::size_t rhs);
  [[nodiscard]] std::size_t addCompare(std::size_t lhs, std::size_t rhs, const std::string& predicate);

  [[nodiscard]] const std::vector<Node> &nodes() const noexcept { return nodes_; }

private:
  std::vector<Node> nodes_;
};

} // namespace son

namespace son::bridge {

/// Emit an MLIR module representing the supplied Sea-of-Nodes graph.
///
/// The resulting module contains a single `func.func` named `graph_entry`
/// that returns the value produced by the final node in the graph.
::mlir::ModuleOp exportToMLIR(::mlir::MLIRContext &context, const Graph &graph);

/// Import a previously exported MLIR module back into a Sea-of-Nodes graph.
///
/// The function expects a module produced by `exportToMLIR` and reconstructs
/// the minimal opcode set supported by this bridge.
Graph importFromMLIR(::mlir::ModuleOp module);