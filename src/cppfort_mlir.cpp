#include "mlir_semantic_graph.hpp"
#include "token_to_mlir.cpp"
#include <iostream>
#include <fstream>
#include <print>
#include "lexer.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

// MLIR-native transpiler: Tokens → Semantic Ops → Optimized MLIR → C++
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::println("Usage: {} <input.cpp2> <output.cpp>", argv[0]);
        return 1;
    }

    try {
        std::string input_filename = argv[1];
        std::string output_filename = argv[2];

        std::ifstream input_file(input_filename);
        if (!input_file) {
            throw std::runtime_error("Cannot open input file: " + input_filename);
        }

        std::string source_code(std::istreambuf_iterator<char>(input_file),
                                std::istreambuf_iterator<char>());

        // Initialize MLIR context and load Cpp2 dialect
        mlir::MLIRContext ctx;
        ctx.loadDialect<mlir::cpp2::Cpp2Dialect>();

        // Phase 1: Lexical analysis only
        cpp2_transpiler::Lexer lexer(source_code);
        auto tokens = lexer.tokenize();

        // Phase 2: Build MLIR semantic graph directly from tokens
        // No text processing, no AST - just semantic operations
        cppfort::mlir_son::TokenToMLIRBuilder builder(ctx);
        auto semantic_graph = builder.build(tokens);

        // Phase 3: Materialize to MLIR IR
        mlir::OpBuilder op_builder(&ctx);
        auto module = semantic_graph.materialize(op_builder);

        // Phase 4: Verify MLIR IR
        if (failed(mlir::verify(module))) {
            std::println("ERROR: MLIR verification failed");
            return 1;
        }

        // Phase 5: Apply MLIR optimization passes
        mlir::PassManager pm(&ctx);
        pm.addPass(mlir::createCSEPass());           // Common subexpression elimination
        pm.addPass(mlir::createCanonicalizerPass()); // Canonicalization

        if (failed(pm.run(module))) {
            std::println("ERROR: MLIR optimization failed");
            return 1;
        }

        // Phase 6: Generate C++ code from optimized MLIR
        std::ofstream output_file(output_filename);
        if (!output_file) {
            throw std::runtime_error("Cannot open output file: " + output_filename);
        }

        // Walk MLIR operations and emit C++
        module.walk([&](mlir::Operation* op) {
            llvm::StringRef op_name = op->getName().getStringRef();

            if (op_name == "cpp2.constant") {
                auto value_attr = op->getAttr("value");
                if (auto int_attr = value_attr.dyn_cast<mlir::IntegerAttr>()) {
                    // Constants are inlined at use sites
                }
            } else if (op_name == "cpp2.add") {
                // Binary operations handled in expression context
            } else if (op_name == "cpp2.return") {
                output_file << "return ";
                if (op->getNumOperands() > 0) {
                    emit_value(output_file, op->getOperand(0));
                }
                output_file << ";\n";
            }
        });

        std::println("Successfully transpiled {} to {}", input_filename, output_filename);
        std::println("Pipeline: Tokens → MLIR Ops → Optimized → C++");
        return 0;
    }
    catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }
}

// Helper to emit MLIR value as C++ expression
void emit_value(std::ostream& out, mlir::Value value) {
    mlir::Operation* def_op = value.getDefiningOp();
    if (!def_op) {
        out << "/*block_arg*/";
        return;
    }

    llvm::StringRef op_name = def_op->getName().getStringRef();

    if (op_name == "cpp2.constant") {
        auto value_attr = def_op->getAttr("value");
        if (auto int_attr = value_attr.dyn_cast<mlir::IntegerAttr>()) {
            out << int_attr.getInt();
        } else if (auto float_attr = value_attr.dyn_cast<mlir::FloatAttr>()) {
            out << float_attr.getValueAsDouble();
        }
    } else if (op_name == "cpp2.add") {
        out << "(";
        emit_value(out, def_op->getOperand(0));
        out << " + ";
        emit_value(out, def_op->getOperand(1));
        out << ")";
    } else if (op_name == "cpp2.sub") {
        out << "(";
        emit_value(out, def_op->getOperand(0));
        out << " - ";
        emit_value(out, def_op->getOperand(1));
        out << ")";
    } else if (op_name == "cpp2.mul") {
        out << "(";
        emit_value(out, def_op->getOperand(0));
        out << " * ";
        emit_value(out, def_op->getOperand(1));
        out << ")";
    } else if (op_name == "cpp2.div") {
        out << "(";
        emit_value(out, def_op->getOperand(0));
        out << " / ";
        emit_value(out, def_op->getOperand(1));
        out << ")";
    } else {
        out << "/*" << op_name.str() << "*/";
    }
}