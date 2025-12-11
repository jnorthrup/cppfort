#include "Cpp2Dialect.h"
#include "mlir_cpp2_dialect.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include <unordered_map>

namespace cppfort::mlir_son {

// Bridge: Convert CRDT graph to MLIR dialect operations
class CRDTToMLIRConverter {
private:
    mlir::MLIRContext* ctx;
    mlir::OpBuilder builder;
    std::unordered_map<NodeID, mlir::Value> nodeToValue;
    std::unordered_map<uint64_t, mlir::Type> aliasToMemType;

public:
    explicit CRDTToMLIRConverter(mlir::MLIRContext* context)
        : ctx(context), builder(context) {
        ctx->loadDialect<mlir::cpp2::Cpp2Dialect>();
    }

    mlir::ModuleOp convert(const CRDTGraph& graph) {
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
        builder.setInsertionPointToEnd(module.getBody());

        // Convert each node to MLIR operation
        for (const auto& [id, node] : graph.get_nodes()) {
            convertNode(node, graph);
        }

        return module;
    }

private:
    void convertNode(const Node& node, const CRDTGraph& graph) {
        switch (node.kind) {
            case Node::Kind::Start:
                convertStartNode(node);
                break;
            case Node::Kind::Stop:
                // Stop is implicit in function termination
                break;
            case Node::Kind::Constant:
                convertConstantNode(node);
                break;
            case Node::Kind::Add:
            case Node::Kind::Sub:
            case Node::Kind::Mul:
            case Node::Kind::Div:
                convertBinaryOp(node, graph);
                break;
            case Node::Kind::Phi:
                convertPhiNode(node, graph);
                break;
            case Node::Kind::If:
                convertIfNode(node, graph);
                break;
            case Node::Kind::Region:
                convertRegionNode(node, graph);
                break;
            case Node::Kind::Loop:
                convertLoopNode(node, graph);
                break;
            case Node::Kind::New:
                convertNewNode(node, graph);
                break;
            case Node::Kind::Load:
                convertLoadNode(node, graph);
                break;
            case Node::Kind::Store:
                convertStoreNode(node, graph);
                break;
            case Node::Kind::UFCS_Call:
                convertUFCSCall(node, graph);
                break;
            case Node::Kind::Contract:
                convertContract(node, graph);
                break;
            case Node::Kind::Metafunction:
                convertMetafunction(node, graph);
                break;
            case Node::Kind::Return:
                convertReturn(node, graph);
                break;
            default:
                break;
        }
    }

    void convertStartNode(const Node& node) {
        auto startOp = builder.create<mlir::cpp2::StartOp>(
            builder.getUnknownLoc(),
            mlir::TypeRange{} // Results will be memory slices
        );
        nodeToValue[node.id] = startOp.getResult(0);
    }

    void convertConstantNode(const Node& node) {
        mlir::Attribute value;
        mlir::Type type;

        if (std::holds_alternative<int64_t>(node.value)) {
            int64_t v = std::get<int64_t>(node.value);
            type = builder.getI64Type();
            value = builder.getI64IntegerAttr(v);
        } else if (std::holds_alternative<double>(node.value)) {
            double v = std::get<double>(node.value);
            type = builder.getF64Type();
            value = builder.getF64FloatAttr(v);
        } else if (std::holds_alternative<bool>(node.value)) {
            bool v = std::get<bool>(node.value);
            type = builder.getI1Type();
            value = builder.getBoolAttr(v);
        } else {
            return;
        }

        auto constOp = builder.create<mlir::cpp2::ConstantOp>(
            builder.getUnknownLoc(),
            type,
            value
        );
        nodeToValue[node.id] = constOp.getResult();
    }

    void convertBinaryOp(const Node& node, const CRDTGraph& graph) {
        if (node.inputs.size() < 2) return;

        mlir::Value lhs = getValueForNode(node.inputs[0]);
        mlir::Value rhs = getValueForNode(node.inputs[1]);
        if (!lhs || !rhs) return;

        mlir::Value result;
        switch (node.kind) {
            case Node::Kind::Add:
                result = builder.create<mlir::cpp2::AddOp>(
                    builder.getUnknownLoc(), lhs.getType(), lhs, rhs
                ).getResult();
                break;
            case Node::Kind::Sub:
                result = builder.create<mlir::cpp2::SubOp>(
                    builder.getUnknownLoc(), lhs.getType(), lhs, rhs
                ).getResult();
                break;
            case Node::Kind::Mul:
                result = builder.create<mlir::cpp2::MulOp>(
                    builder.getUnknownLoc(), lhs.getType(), lhs, rhs
                ).getResult();
                break;
            case Node::Kind::Div:
                result = builder.create<mlir::cpp2::DivOp>(
                    builder.getUnknownLoc(), lhs.getType(), lhs, rhs
                ).getResult();
                break;
            default:
                return;
        }

        nodeToValue[node.id] = result;
    }

    void convertPhiNode(const Node& node, const CRDTGraph& graph) {
        llvm::SmallVector<mlir::Value> values;
        for (NodeID input : node.inputs) {
            if (auto val = getValueForNode(input)) {
                values.push_back(val);
            }
        }

        if (values.empty()) return;

        auto phiOp = builder.create<mlir::cpp2::PhiOp>(
            builder.getUnknownLoc(),
            values[0].getType(),
            values
        );
        nodeToValue[node.id] = phiOp.getResult();
    }

    void convertIfNode(const Node& node, const CRDTGraph& graph) {
        if (node.inputs.empty()) return;

        mlir::Value condition = getValueForNode(node.inputs[0]);
        if (!condition) return;

        auto ifOp = builder.create<mlir::cpp2::IfOp>(
            builder.getUnknownLoc(),
            mlir::TypeRange{},
            condition
        );

        // Build then/else regions
        builder.createBlock(&ifOp.getThenRegion());
        builder.createBlock(&ifOp.getElseRegion());
    }

    void convertRegionNode(const Node& node, const CRDTGraph& graph) {
        llvm::SmallVector<mlir::Value> inputs;
        for (NodeID inp : node.inputs) {
            if (auto val = getValueForNode(inp)) {
                inputs.push_back(val);
            }
        }

        if (inputs.empty()) return;

        auto regionOp = builder.create<mlir::cpp2::RegionOp>(
            builder.getUnknownLoc(),
            inputs[0].getType(),
            inputs
        );
        nodeToValue[node.id] = regionOp.getResult();
    }

    void convertLoopNode(const Node& node, const CRDTGraph& graph) {
        auto loopOp = builder.create<mlir::cpp2::LoopOp>(
            builder.getUnknownLoc(),
            mlir::TypeRange{}
        );

        builder.createBlock(&loopOp.getBody());
    }

    void convertNewNode(const Node& node, const CRDTGraph& graph) {
        if (node.inputs.size() < 2) return;

        mlir::Value ctrl = getValueForNode(node.inputs[0]);
        if (!ctrl) return;

        // Extract struct type from node metadata
        auto structType = mlir::cpp2::StructType::get(ctx, "UnknownStruct");
        auto ptrType = mlir::cpp2::PtrType::get(structType, false);

        auto newOp = builder.create<mlir::cpp2::NewOp>(
            builder.getUnknownLoc(),
            ptrType,
            ctrl,
            mlir::TypeAttr::get(structType)
        );
        nodeToValue[node.id] = newOp.getPtr();
    }

    void convertLoadNode(const Node& node, const CRDTGraph& graph) {
        if (node.inputs.size() < 2) return;

        mlir::Value mem = getValueForNode(node.inputs[0]);
        mlir::Value ptr = getValueForNode(node.inputs[1]);
        if (!mem || !ptr) return;

        // Extract alias class and field name from node metadata
        uint64_t aliasClass = 0;
        std::string fieldName = "field";

        auto memType = getMemType(aliasClass);
        auto loadOp = builder.create<mlir::cpp2::LoadOp>(
            builder.getUnknownLoc(),
            mlir::TypeRange{builder.getI64Type(), memType},
            mem,
            ptr,
            builder.getStringAttr(fieldName),
            builder.getUI64IntegerAttr(aliasClass)
        );

        nodeToValue[node.id] = loadOp.getValue();
    }

    void convertStoreNode(const Node& node, const CRDTGraph& graph) {
        if (node.inputs.size() < 3) return;

        mlir::Value mem = getValueForNode(node.inputs[0]);
        mlir::Value ptr = getValueForNode(node.inputs[1]);
        mlir::Value val = getValueForNode(node.inputs[2]);
        if (!mem || !ptr || !val) return;

        uint64_t aliasClass = 0;
        std::string fieldName = "field";

        auto memType = getMemType(aliasClass);
        auto storeOp = builder.create<mlir::cpp2::StoreOp>(
            builder.getUnknownLoc(),
            memType,
            mem,
            ptr,
            builder.getStringAttr(fieldName),
            val,
            builder.getUI64IntegerAttr(aliasClass)
        );

        nodeToValue[node.id] = storeOp.getOutMem();
    }

    void convertUFCSCall(const Node& node, const CRDTGraph& graph) {
        llvm::SmallVector<mlir::Value> args;
        for (NodeID input : node.inputs) {
            if (auto val = getValueForNode(input)) {
                args.push_back(val);
            }
        }

        auto callOp = builder.create<mlir::cpp2::UFCSCallOp>(
            builder.getUnknownLoc(),
            mlir::TypeRange{},
            builder.getStringAttr("unknown_method"),
            args
        );
    }

    void convertContract(const Node& node, const CRDTGraph& graph) {
        if (node.inputs.empty()) return;

        mlir::Value condition = getValueForNode(node.inputs[0]);
        if (!condition) return;

        builder.create<mlir::cpp2::ContractOp>(
            builder.getUnknownLoc(),
            condition,
            builder.getStringAttr("assert"),
            builder.getStringAttr("Contract violation")
        );
    }

    void convertMetafunction(const Node& node, const CRDTGraph& graph) {
        llvm::SmallVector<mlir::Value> args;
        for (NodeID input : node.inputs) {
            if (auto val = getValueForNode(input)) {
                args.push_back(val);
            }
        }

        builder.create<mlir::cpp2::MetafunctionOp>(
            builder.getUnknownLoc(),
            mlir::TypeRange{},
            builder.getStringAttr("meta"),
            args
        );
    }

    void convertReturn(const Node& node, const CRDTGraph& graph) {
        llvm::SmallVector<mlir::Value> operands;
        for (NodeID input : node.inputs) {
            if (auto val = getValueForNode(input)) {
                operands.push_back(val);
            }
        }

        builder.create<mlir::cpp2::ReturnOp>(
            builder.getUnknownLoc(),
            operands
        );
    }

    mlir::Value getValueForNode(NodeID id) {
        auto it = nodeToValue.find(id);
        return it != nodeToValue.end() ? it->second : mlir::Value();
    }

    mlir::Type getMemType(uint64_t aliasClass) {
        auto it = aliasToMemType.find(aliasClass);
        if (it != aliasToMemType.end()) {
            return it->second;
        }

        auto memType = mlir::cpp2::MemType::get(ctx, aliasClass);
        aliasToMemType[aliasClass] = memType;
        return memType;
    }
};

// Bridge: Convert MLIR dialect operations back to CRDT graph
class MLIRToCRDTConverter {
private:
    CRDTGraph graph;
    std::unordered_map<mlir::Value, NodeID> valueToNode;
    NodeID nextID = 1;

public:
    CRDTGraph convert(mlir::ModuleOp module) {
        module.walk([this](mlir::Operation* op) {
            convertOperation(op);
        });

        return graph;
    }

private:
    void convertOperation(mlir::Operation* op) {
        if (auto constOp = llvm::dyn_cast<mlir::cpp2::ConstantOp>(op)) {
            convertConstantOp(constOp);
        } else if (auto addOp = llvm::dyn_cast<mlir::cpp2::AddOp>(op)) {
            convertBinaryOp(addOp, Node::Kind::Add);
        } else if (auto subOp = llvm::dyn_cast<mlir::cpp2::SubOp>(op)) {
            convertBinaryOp(subOp, Node::Kind::Sub);
        } else if (auto mulOp = llvm::dyn_cast<mlir::cpp2::MulOp>(op)) {
            convertBinaryOp(mulOp, Node::Kind::Mul);
        } else if (auto divOp = llvm::dyn_cast<mlir::cpp2::DivOp>(op)) {
            convertBinaryOp(divOp, Node::Kind::Div);
        }
        // Add more operation conversions as needed
    }

    void convertConstantOp(mlir::cpp2::ConstantOp op) {
        Node node{Node::Kind::Constant, nextID++};

        auto attr = op.getValue();
        if (auto intAttr = attr.dyn_cast<mlir::IntegerAttr>()) {
            node.value = intAttr.getInt();
        } else if (auto floatAttr = attr.dyn_cast<mlir::FloatAttr>()) {
            node.value = floatAttr.getValueAsDouble();
        }

        Patch patch;
        patch.operation = Patch::Op::AddNode;
        patch.data = node;
        graph.apply_patch(patch);

        valueToNode[op.getResult()] = node.id;
    }

    template<typename OpT>
    void convertBinaryOp(OpT op, Node::Kind kind) {
        Node node{kind, nextID++};

        NodeID lhs = getNodeForValue(op.getLhs());
        NodeID rhs = getNodeForValue(op.getRhs());

        node.inputs = {lhs, rhs};

        Patch patch;
        patch.operation = Patch::Op::AddNode;
        patch.data = node;
        graph.apply_patch(patch);

        valueToNode[op.getResult()] = node.id;
    }

    NodeID getNodeForValue(mlir::Value val) {
        auto it = valueToNode.find(val);
        return it != valueToNode.end() ? it->second : 0;
    }
};

} // namespace cppfort::mlir_son