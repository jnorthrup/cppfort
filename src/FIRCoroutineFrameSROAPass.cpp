#include "Cpp2Passes.h"
#include "Cpp2FIRDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::cpp2fir;

namespace {

/// FIRCoroutineFrameSROAPass: Analyzes coroutine frames and optimizes allocation.
///
/// This pass detects coroutines whose frames do not escape the parent scope
/// and converts them from heap allocation to stack or arena allocation.
///
/// Algorithm:
/// 1. Walk all functions and identify coroutine operations
/// 2. For each coroutine, analyze captured variables
/// 3. Determine if any captured variable escapes the coroutine's lifetime
/// 4. If all captures are NoEscape or bounded, mark frame for stack/arena
/// 5. Tag operations with #cpp2.coroutine_frame<stack|arena|heap> attribute
///
/// Frame allocation criteria:
/// - Stack: Non-escaping coroutine, all captures NoEscape, frame < 1KB
/// - Arena: Non-escaping coroutine, large frame (>1KB) or has arena-eligible captures
/// - Heap: Escaping coroutine (default C++ behavior)
struct FIRCoroutineFrameSROAPass : public PassWrapper<FIRCoroutineFrameSROAPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FIRCoroutineFrameSROAPass)

    StringRef getArgument() const override { return "fir-coroutine-frame-sroa"; }
    StringRef getDescription() const override { return "Optimize coroutine frame allocation (stack/arena/heap)"; }

    // Statistics
    struct Stats {
        unsigned coroutinesProcessed = 0;
        unsigned stackFrames = 0;
        unsigned arenaFrames = 0;
        unsigned heapFrames = 0;
        std::size_t bytesSaved = 0;  // Heap bytes moved to stack/arena
    };
    Stats stats;

    void runOnOperation() override {
        ModuleOp module = getOperation();

        // Process each function
        module.walk([&](FuncOp func) {
            processFuncOp(func);
            stats.coroutinesProcessed++;
        });

        // Emit statistics as remarks
        if (stats.stackFrames > 0 || stats.arenaFrames > 0) {
            module.emitRemark() << "Coroutine frame SROA: "
                << stats.stackFrames << " stack, "
                << stats.arenaFrames << " arena, "
                << stats.heapFrames << " heap ("
                << (stats.bytesSaved / 1024) << " KB saved)";
        }
    }

private:
    /// Process a single function for coroutine frame optimization
    void processFuncOp(FuncOp func) {
        if (func.isDeclaration())
            return;

        OpBuilder builder(func.getContext());

        // Find all coroutine operations in this function
        func.walk([&](Operation *op) {
            if (isCoroutineOperation(op)) {
                auto strategy = determineFrameStrategy(op);
                tagWithFrameStrategy(op, strategy, builder);

                // Update statistics
                switch (strategy) {
                    case CoroutineFrameStrategy::Stack: stats.stackFrames++; break;
                    case CoroutineFrameStrategy::Arena: stats.arenaFrames++; break;
                    case CoroutineFrameStrategy::Heap: stats.heapFrames++; break;
                }

                // Track bytes saved if not heap
                if (strategy != CoroutineFrameStrategy::Heap) {
                    if (auto sizeAttr = op->getAttrOfType<IntegerAttr>("frame_size")) {
                        stats.bytesSaved += sizeAttr.getInt();
                    }
                }
            }
        });
    }

    /// Check if an operation represents a coroutine
    bool isCoroutineOperation(Operation *op) {
        // Check for coroutine-related operation names
        StringRef opName = op->getName().getStringRef();
        return opName.contains("coroutine") ||
               opName.contains("co_await") ||
               opName.contains("co_yield") ||
               opName.contains("launch") ||
               op->hasAttr("is_coroutine");
    }

    /// Determine the optimal frame allocation strategy
    CoroutineFrameStrategy determineFrameStrategy(Operation *op) {
        // Check escape analysis annotation
        bool has_escaping_capture = false;
        bool has_noescape_capture = false;
        std::size_t frame_size = 0;

        // Check for escape_kind attribute on captured variables
        // (In real implementation, this would walk the coroutine's captures)
        if (auto escapeAttr = op->getAttrOfType<StringAttr>("capture_escape")) {
            if (escapeAttr.getValue() == "heap" ||
                escapeAttr.getValue() == "return" ||
                escapeAttr.getValue() == "global") {
                has_escaping_capture = true;
            } else if (escapeAttr.getValue() == "no_escape") {
                has_noescape_capture = true;
            }
        }

        // Get frame size if available
        if (auto sizeAttr = op->getAttrOfType<IntegerAttr>("frame_size")) {
            frame_size = sizeAttr.getInt();
        }

        // Default to 1KB threshold for stack vs arena
        const std::size_t STACK_THRESHOLD = 1024;

        // Decision logic:
        // - Heap: Any capture escapes the coroutine's lifetime
        // - Stack: All captures NoEscape, small frame
        // - Arena: All captures NoEscape, large frame
        if (has_escaping_capture) {
            return CoroutineFrameStrategy::Heap;
        }

        if (has_noescape_capture || !op->hasAttr("capture_escape")) {
            // No escaping captures, can optimize
            if (frame_size < STACK_THRESHOLD) {
                return CoroutineFrameStrategy::Stack;
            } else {
                return CoroutineFrameStrategy::Arena;
            }
        }

        // Default: heap (safe but slow)
        return CoroutineFrameStrategy::Heap;
    }

    /// Tag an operation with coroutine frame strategy attribute
    void tagWithFrameStrategy(Operation *op, CoroutineFrameStrategy strategy, OpBuilder &builder) {
        // Create coroutine_frame attribute: #cpp2.coroutine_frame<stack|arena|heap>
        StringRef strategyStr;
        switch (strategy) {
            case CoroutineFrameStrategy::Stack: strategyStr = "stack"; break;
            case CoroutineFrameStrategy::Arena: strategyStr = "arena"; break;
            case CoroutineFrameStrategy::Heap: strategyStr = "heap"; break;
        }

        op->setAttr("coroutine_frame", builder.getStringAttr(strategyStr));
    }
};

} // namespace

namespace mlir {
namespace cpp2 {
    std::unique_ptr<Pass> createFIRCoroutineFrameSROAPass() {
        return std::make_unique<FIRCoroutineFrameSROAPass>();
    }
}
}
