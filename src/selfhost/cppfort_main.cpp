// cppfort_main.cpp - Driver for cppfort self-hosted compiler
// Handles file I/O and calls into cppfort.cpp2 core

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <cstdio>

#include CPPFORT_SELFHOST_RBCURSIVE_CPP
#include CPPFORT_SELFHOST_CPPFORT_CPP

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: cppfort <source.cpp2> [output]\n";
        std::cerr << "  Default: parses cpp2 source and outputs canonical AST\n";
        std::cerr << "  With -c: outputs C++ code\n";
        std::cerr << "  With -o <file>: compiles to executable\n";
        std::cerr << "  Post-dogfood: will compile to MLIR SoN → LLVM\n";
        return 1;
    }
    
    // Parse arguments
    bool emit_cpp = false;
    bool compile_exe = false;
    std::string input_path;
    std::string output_path;
    
    int arg_idx = 1;
    while (arg_idx < argc) {
        std::string arg = argv[arg_idx];
        if (arg == "-c") {
            emit_cpp = true;
        } else if (arg == "-o") {
            if (arg_idx + 1 >= argc) {
                std::cerr << "Error: -o requires output file argument\n";
                return 1;
            }
            compile_exe = true;
            output_path = argv[arg_idx + 1];
            arg_idx++;
        } else if (arg[0] != '-') {
            input_path = arg;
        }
        arg_idx++;
    }
    
    if (input_path.empty()) {
        std::cerr << "Error: No input file specified\n";
        return 1;
    }
    
    std::ifstream input_file(input_path);
    
    if (!input_file.is_open()) {
        std::cerr << "Error: Cannot open input file: " << input_path << "\n";
        return 2;
    }
    
    std::stringstream source_stream;
    source_stream << input_file.rdbuf();
    std::string source = source_stream.str();
    input_file.close();
    
    std::cerr << "cppfort: parsing " << input_path << " (" << source.size() << " bytes)\n";
    
    scan_session session{};
    std::cerr << "DEBUG: About to call parse_source\n";
    std::cerr.flush();
    auto ast_result = parse_source(std::string_view(source), session);
    std::cerr << "DEBUG: parse_source returned, has_value=" << ast_result.has_value() << "\n";
    std::cerr << "DEBUG: session.features.size()=" << session.features.size() << "\n";
    if (ast_result.has_value()) {
        std::cerr << "DEBUG: ast_result.value().size()=" << ast_result.value().size() << "\n";
    }
    std::cerr.flush();
    
    if (!ast_result.has_value()) {
        std::cerr << "Error: Parse failed\n";
        return 3;
    }
    
    auto& ast = ast_result.value();
    
    // Debug: test the filter directly
    size_t filter_count = 0;
    for (const auto& feat : session.features) {
        std::string_view sv = feat.semantic;
        bool is_top_level = 
            sv == "translation_unit" ||
            sv == "chart_definition" ||
            sv == "chart_body" ||
            sv == "chart_clause" ||
            sv == "manifold_declaration" ||
            sv == "atlas_literal" ||
            sv == "bootstrap_tag_declaration" ||
            sv == "join_expression" ||
            sv == "alpha_expression" ||
            sv == "transition_expression" ||
            sv == "series_literal" ||
            sv == "struct_declaration" ||
            sv == "lowered_method_call" ||
            sv == "chart_project_expression" ||
            sv == "atlas_locate_expression";
        if (is_top_level) filter_count++;
    }
    std::cerr << "DEBUG: filter would keep " << filter_count << " of " << session.features.size() << " features\n";
    
    // Debug: dump features before returning
    std::cerr << "DEBUG: Dumping " << session.features.size() << " features:\n";
    for (const auto& feat : session.features) {
        std::string_view sv = feat.semantic;
        bool is_bootstrap = (sv == "bootstrap_tag_declaration");
        bool is_join_tag = (sv == "join_tag");
        std::cerr << "  feature: span=" << feat.start << "-" << feat.stop 
                  << " semantic=\"" << feat.semantic << "\""
                  << " is_bootstrap=" << is_bootstrap
                  << " is_join_tag=" << is_join_tag << "\n";
    }
    std::cerr << "cppfort: parsed " << ast.size() << " canonical nodes\n";
    
    for (size_t i = 0; i < ast.size(); ++i) {
        const auto& node = ast[i];
        std::cerr << "  node[" << i << "]: tag=" << node.tag;
        std::cerr << " span=" << node.source_start << "-" << node.source_stop;
        if (!node.semantic.empty()) {
            std::cerr << " semantic=\"" << node.semantic << "\"";
        }
        std::cerr << "\n";
    }
    
    // Emit C++ code
    std::string cpp_code = compile_to_cpp(ast);
    
    if (emit_cpp) {
        // Output C++ code
        if (compile_exe) {
            std::ofstream out_file(output_path);
            if (!out_file.is_open()) {
                std::cerr << "Error: Cannot open output file: " << output_path << "\n";
                return 4;
            }
            out_file << cpp_code;
            out_file.close();
            std::cerr << "cppfort: wrote C++ to " << output_path << "\n";
        } else {
            std::cout << cpp_code;
        }
    } else if (compile_exe) {
        // Compile to executable
        std::string temp_cpp = "/tmp/cppfort_compile.cpp";
        std::ofstream cpp_file(temp_cpp);
        cpp_file << cpp_code;
        cpp_file.close();
        
        std::cerr << "cppfort: compiling to " << output_path << "\n";
        
        std::string compile_cmd = "clang++ -std=c++20 -o " + output_path + " " + temp_cpp + " 2>&1";
        int result = system(compile_cmd.c_str());
        
        if (result != 0) {
            std::cerr << "Error: Compilation failed with code " << result << "\n";
            return 5;
        }
        
        std::cerr << "cppfort: compiled executable to " << output_path << "\n";
    } else {
        // Default: output AST
        if (argc >= 3) {
            std::string output_path = argv[2];
            std::ofstream output_file(output_path);
            
            if (!output_file.is_open()) {
                std::cerr << "Error: Cannot open output file: " << output_path << "\n";
                return 4;
            }
            
            output_file << "// cppfort canonical AST output\n";
            output_file << "// Source: " << input_path << "\n";
            output_file << "// Nodes: " << ast.size() << "\n\n";
            
// Debug: dump features before returning
    std::cerr << "DEBUG: Dumping " << session.features.size() << " features:\n";
    for (const auto& feat : session.features) {
        std::cerr << "  feature: span=" << feat.start << "-" << feat.stop 
                  << " semantic=\"" << feat.semantic << "\"\n";
    }
    std::cerr << "DEBUG: Dumping " << ast.size() << " nodes:\n";
    for (size_t i = 0; i < ast.size(); ++i) {
                const auto& node = ast[i];
                output_file << "node[" << i << "]:\n";
                output_file << "  tag: " << node.tag << "\n";
                output_file << "  span: " << node.source_start << "-" << node.source_stop << "\n";
                output_file << "  semantic: \"" << node.semantic << "\"\n";
                output_file << "  children: " << node.children.size() << "\n\n";
            }
            
            output_file.close();
            std::cerr << "cppfort: wrote canonical AST to " << output_path << "\n";
        }
    }
    
    std::cerr << "cppfort: success\n";
    return 0;
}
