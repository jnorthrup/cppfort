#include "emitter.hpp"
#include <cctype>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace {

using namespace cpp2::ast;

class TreeEmitter {
  const ParseTree &tree_;
  std::span<const cpp2_transpiler::Token> tokens_;
  std::ostringstream out_;
  int indent_ = 0;
  bool in_function_ = false;  // Track if we're emitting inside a function body

  void emit_indent() {
    for (int i = 0; i < indent_; ++i)
      out_ << "    ";
  }

  std::string_view token_text(uint32_t idx) const {
    if (idx < tokens_.size())
      return tokens_[idx].lexeme;
    return "";
  }

  // Check if a node's token range contains a specific token text
  bool node_contains_token(const Node &n, std::string_view target) const {
    for (uint32_t i = n.token_start; i < n.token_end && i < tokens_.size(); ++i) {
      if (tokens_[i].lexeme == target)
        return true;
    }
    return false;
  }

  std::string node_text(const Node &n) const {
    std::string result;
    for (uint32_t i = n.token_start; i < n.token_end && i < tokens_.size();
         ++i) {
      if (!result.empty() && !tokens_[i].lexeme.empty()) {
        char prev = result.back();
        char next = tokens_[i].lexeme[0];
        if (std::isalnum(prev) && std::isalnum(next))
          result += ' ';
      }
      result += tokens_[i].lexeme;
    }
    return result;
  }

  // Process Cpp2 string interpolation: "(expr)$" -> "" + cpp2::to_string(expr) + ""
  // Handles format specifiers: "(expr: fmt)$" -> cpp2::to_string(expr, "{:fmt}")
  // Returns the original string unchanged if no interpolation patterns found.
  std::string process_string_interpolation(const std::string &str) const {
    // Only process quoted strings that contain the )$ interpolation marker
    if (str.size() < 2 || str.front() != '"' || str.back() != '"')
      return str;
    if (str.find(")$") == std::string::npos)
      return str;

    // Work on the content between the outer quotes
    std::string content = str.substr(1, str.size() - 2);
    std::string result;
    bool first_segment = true;
    size_t i = 0;

    while (i < content.size()) {
      // Look for interpolation pattern: (expr)$ or (expr: fmt)$
      if (content[i] == '(' && (i == 0 || content[i - 1] != '\\')) {
        // Find matching close-paren with nesting support
        size_t paren_count = 1;
        size_t j = i + 1;
        while (j < content.size() && paren_count > 0) {
          if (content[j] == '(') paren_count++;
          else if (content[j] == ')') paren_count--;
          j++;
        }
        // j now points one past the matching ')'
        // Check if ')' is followed by '$'
        if (j <= content.size() && j > 0 && content[j - 1] == ')' &&
            j < content.size() && content[j] == '$') {
          // Found interpolation: (expr)$ or (expr: fmt)$
          std::string inner = content.substr(i + 1, j - i - 2);

          // Flush any accumulated literal text as a quoted string segment
          if (!result.empty() || !first_segment) {
            result += "\" + ";
          } else {
            result += "\"\" + ";
          }

          // Check for format specifier - colon separating expr from format
          // But be careful: colons inside nested parens/templates are not format specs
          // The format spec comes after the LAST colon at depth 0
          size_t colon_pos = std::string::npos;
          int depth = 0;
          for (size_t k = 0; k < inner.size(); ++k) {
            if (inner[k] == '(' || inner[k] == '<') depth++;
            else if (inner[k] == ')' || inner[k] == '>') depth--;
            else if (inner[k] == ':' && depth == 0) colon_pos = k;
          }

          if (colon_pos != std::string::npos && colon_pos > 0) {
            // Has format specifier
            std::string expr = inner.substr(0, colon_pos);
            std::string fmt = inner.substr(colon_pos + 1);
            // Trim leading/trailing whitespace from format
            while (!fmt.empty() && fmt.front() == ' ') fmt.erase(fmt.begin());
            while (!fmt.empty() && fmt.back() == ' ') fmt.pop_back();
            result += "cpp2::to_string(" + expr + ", \"{:" + fmt + "}\")";
          } else {
            result += "cpp2::to_string(" + inner + ")";
          }

          result += " + \"";
          i = j + 1;  // Skip past the '$'
          first_segment = false;
          continue;
        }
      }

      // Regular character - accumulate into current string segment
      if (first_segment && result.empty()) {
        result += "\"";
        first_segment = false;
      }
      result += content[i];
      ++i;
    }

    // Close the final string segment
    if (!first_segment) {
      result += "\"";
    } else {
      return str;  // No interpolation found after all
    }

    return result;
  }

  std::string map_type_name(const std::string &name) const {
    if (name == "_")
      return "auto";  // Cpp2 wildcard/deduced type
    if (name == "i32")
      return "int";
    if (name == "u32")
      return "unsigned int";
    if (name == "i16")
      return "int16_t";
    if (name == "u16")
      return "uint16_t";
    if (name == "i8")
      return "int8_t";
    if (name == "u8")
      return "uint8_t";
    if (name == "i64")
      return "int64_t";
    if (name == "u64")
      return "uint64_t";
    if (name == "f32")
      return "float";
    if (name == "f64")
      return "double";
    return name;
  }

  std::string trim(std::string_view s) const {
    size_t first = s.find_first_not_of(" \t\n\r");
    if (first == std::string::npos)
      return "";
    size_t last = s.find_last_not_of(" \t\n\r");
    return std::string(s.substr(first, last - first + 1));
  }

  // Check if a function body contains a return statement with a value
  // or is an expression body (= expr;) which implicitly returns
  // Check if a function has an expression body (= expr; rather than = { ... })
  bool function_is_expression_body(const Node &suffix) const {
    for (const auto &child : tree_.children(suffix)) {
      if (child.kind == NodeKind::FunctionBody) {
        for (const auto &body_child : tree_.children(child)) {
          if (body_child.kind == NodeKind::BlockStatement) {
            return false; // Has a block body
          }
        }
        // No block -> check it's not pure virtual / deleted / empty
        std::string body_text = trim(node_text(child));
        if (body_text == "0" || body_text == "delete" || 
            body_text.empty() || body_text == ";") {
          return false;
        }
        return true; // Expression body
      }
    }
    return false;
  }

  bool function_has_return_value(const Node &suffix) const {
    for (const auto &child : tree_.children(suffix)) {
      if (child.kind == NodeKind::FunctionBody) {
        // Check if this is an expression body (no BlockStatement child)
        bool has_block = false;
        for (const auto &body_child : tree_.children(child)) {
          if (body_child.kind == NodeKind::BlockStatement) {
            has_block = true;
            break;
          }
        }
        if (!has_block) {
          // Check for pure virtual (= 0), deleted (= delete), or empty body markers
          // These are NOT expression bodies
          std::string body_text = trim(node_text(child));
          if (body_text == "0" || body_text == "delete" || 
              body_text.empty() || body_text == ";") {
            return false;
          }
          // Expression body: = expr; → implicitly returns
          return true;
        }
        return body_has_return_value(child);
      }
    }
    return false;
  }

  bool body_has_return_value(const Node &n) const {
    // Check if any token in the body is a 'return' followed by something other than ';'
    for (uint32_t i = n.token_start; i < n.token_end && i < tokens_.size(); ++i) {
      if (tokens_[i].lexeme == "return") {
        // Check if there's a value after return (not just "return;")
        if (i + 1 < n.token_end && i + 1 < tokens_.size()) {
          if (tokens_[i + 1].lexeme != ";") {
            return true;  // Has a return value
          }
        }
      }
    }
    return false;  // No return with value found -> returns void
  }

  std::string format_type(const Node &n) const {
    std::string text = trim(node_text(n));

    // Check if the text contains decltype, sizeof, or typeid
    bool contains_decltype = text.find("decltype(") != std::string::npos ||
                              text.find("sizeof(") != std::string::npos ||
                              text.find("typeid(") != std::string::npos;

    // Check if the text contains template arguments (need to fix pointer prefixes inside)
    bool contains_template = text.find('<') != std::string::npos;

    if (contains_decltype || contains_template) {
      // This type contains expressions or template args that need transformation
      // Reconstruct the type while transforming
      return reconstruct_type_with_expressions(n);
    }

    // Simple heuristic: move leading * and & to the end
    size_t i = 0;
    std::string qualifiers;
    while (i < text.length() && (text[i] == '*' || text[i] == '&')) {
      qualifiers += text[i];
      i++;
    }
    std::string base = trim(text.substr(i));
    return map_type_name(base) + qualifiers;
  }

  // Reconstruct a type node while processing expression children with UFCS
  std::string reconstruct_type_with_expressions(const Node &n) const {
    // Check if this is a decltype/sizeof/typeid node by looking at first token
    std::string_view first_token = token_text(n.token_start);

    // Check for decltype/sizeof/typeid pattern
    if (first_token == "decltype" || first_token == "sizeof" || first_token == "typeid") {
      // Look for expression children
      const Node *expr_child = nullptr;
      for (const auto &child : tree_.children(n)) {
        if (child.kind == NodeKind::Expression || meta::is_expression(child.kind)) {
          expr_child = &child;
          break;
        }
      }

      if (expr_child) {
        std::string transformed = const_cast<TreeEmitter*>(this)->emit_expression_text(*expr_child);
        return std::string(first_token) + "(" + transformed + ")";
      }

      // No expression child - check if we need to recurse into children
      // The expression might be nested deeper in the tree
      std::string text = node_text(n);
      bool has_nested_expression = false;
      for (const auto &child : tree_.children(n)) {
        std::string child_text = node_text(child);
        if (child_text.find(".(") != std::string::npos ||
            child_text.find("().") != std::string::npos) {
          has_nested_expression = true;
          break;
        }
      }

      if (has_nested_expression) {
        // Recurse to find and transform expressions
        std::string result;
        for (const auto &child : tree_.children(n)) {
          result += reconstruct_type_with_expressions(child);
        }
        return result;
      }

      return text;
    }

    // Special handling for TemplateArgs
    // First child is the adoptee (type being templated), remaining are template arguments
    if (n.kind == NodeKind::TemplateArgs) {
      auto children = tree_.children(n);
      auto it = children.begin();
      std::string result;
      
      if (it != children.end()) {
        // First child is the adoptee (the type name being templated)
        if (it->kind == NodeKind::TypeSpecifier || it->kind == NodeKind::Identifier ||
            it->kind == NodeKind::QualifiedType || it->kind == NodeKind::BasicType) {
          result += reconstruct_type_with_expressions(*it);
        } else if (it->has_children()) {
          result += reconstruct_type_with_expressions(*it);
        } else {
          result += node_text(*it);
        }
        ++it;
      }
      
      result += "<";
      bool first = true;
      for (; it != children.end(); ++it) {
        if (!first) result += ", ";
        first = false;

        if (it->kind == NodeKind::Expression) {
          result += const_cast<TreeEmitter*>(this)->emit_expression_text(*it);
        } else if (it->kind == NodeKind::TypeSpecifier) {
          // TypeSpecifier child might contain expressions
          result += reconstruct_type_with_expressions(*it);
        } else if (it->has_children()) {
          result += reconstruct_type_with_expressions(*it);
        } else {
          result += node_text(*it);
        }
      }
      result += ">";
      return result;
    }

    // Special handling for ScopeOp (namespace::type)
    if (n.kind == NodeKind::ScopeOp) {
      auto children = tree_.children(n);
      auto it = children.begin();
      std::string result;
      
      if (it != children.end()) {
        // First child is the LHS (namespace/class)
        result += reconstruct_type_with_expressions(*it);
        ++it;
        result += "::";
        
        // Remaining children are RHS identifiers or template args
        for (; it != children.end(); ++it) {
          result += reconstruct_type_with_expressions(*it);
        }
      }
      return result;
    }

    // For other nodes, check if any descendant has expressions
    bool has_expression_descendant = false;
    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::Expression || meta::is_expression(child.kind)) {
        has_expression_descendant = true;
        break;
      }
      // Check descendants
      for (const auto &grandchild : tree_.children(child)) {
        if (grandchild.kind == NodeKind::Expression || meta::is_expression(grandchild.kind)) {
          has_expression_descendant = true;
          break;
        }
      }
      if (has_expression_descendant) break;
    }

    if (has_expression_descendant) {
      // Reconstruct with recursion
      std::string result;
      for (const auto &child : tree_.children(n)) {
        if (child.kind == NodeKind::Expression) {
          result += const_cast<TreeEmitter*>(this)->emit_expression_text(child);
        } else if (child.kind == NodeKind::TypeSpecifier || child.kind == NodeKind::TemplateArgs) {
          result += reconstruct_type_with_expressions(child);
        } else if (child.has_children()) {
          result += reconstruct_type_with_expressions(child);
        } else {
          result += node_text(child);
        }
      }
      return result;
    }

    // No expressions - fix pointer/reference prefix and return
    std::string text = trim(node_text(n));
    size_t i = 0;
    std::string qualifiers;
    while (i < text.length() && (text[i] == '*' || text[i] == '&')) {
      qualifiers += text[i];
      i++;
    }
    std::string base = trim(text.substr(i));
    return map_type_name(base) + qualifiers;
  }

  // Helper to emit an expression and return it as a string
  std::string emit_expression_text(const Node &n) {
    // Save current output state
    std::ostringstream old_out;
    old_out.swap(out_);

    // Emit the expression
    emit_expression(n);

    // Get the result and restore output
    std::string result = out_.str();
    out_.swap(old_out);

    return result;
  }

public:
  TreeEmitter(const ParseTree &tree,
              std::span<const cpp2_transpiler::Token> tokens)
      : tree_(tree), tokens_(tokens) {}

  std::string emit() {
    out_ << "// Generated by cppfort - slim ParseTree emitter\n";
    out_ << "#include <iostream>\n";
    out_ << "#include <string>\n";
    out_ << "#include <cstdint>\n";
    out_ << "#include <compare>\n";  // For std::strong_ordering, std::weak_ordering
    out_ << "#include <cpp2_runtime.h>\n\n";

    if (tree_.nodes.empty())
      return out_.str();

    const auto &root = tree_[tree_.root];
    
    // First pass: emit forward declarations for functions and types
    for (const auto &child : tree_.children(root)) {
      emit_forward_declaration(child);
    }
    out_ << "\n";
    
    // Second pass: emit type definitions (before functions that use them)
    for (const auto &child : tree_.children(root)) {
      emit_type_definition(child);
    }
    
    // Third pass: emit function definitions and other declarations
    for (const auto &child : tree_.children(root)) {
      emit_non_type_declaration(child);
    }

    return out_.str();
  }

private:
  // Check if a declaration is a type definition
  bool is_type_declaration(const Node &n) const {
    if (n.kind != NodeKind::Declaration)
      return false;
    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::UnifiedDeclaration) {
        for (const auto &grandchild : tree_.children(child)) {
          if (grandchild.kind == NodeKind::TypeSuffix || grandchild.kind == NodeKind::TypeAliasSuffix)
            return true;
        }
      }
    }
    return false;
  }

  // Emit only type definitions
  void emit_type_definition(const Node &n) {
    if (!is_type_declaration(n))
      return;
    emit_declaration(n);
  }

  // Emit declarations that are NOT type definitions
  void emit_non_type_declaration(const Node &n) {
    if (is_type_declaration(n))
      return;
    emit_declaration(n);
  }

  // Emit forward declaration for a function (just the signature, no body)
  void emit_forward_declaration(const Node &n) {
    if (n.kind != NodeKind::Declaration)
      return;

    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::UnifiedDeclaration) {
        emit_forward_unified_decl(child);
      }
    }
  }
  
  void emit_forward_unified_decl(const Node &n) {
    std::string name;
    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::Identifier) {
        name = node_text(child);
        break;
      }
    }
    if (name.empty()) {
      name = std::string(token_text(n.token_start));
    }

    // Look for template parameters and suffix
    const Node *template_params = nullptr;
    const Node *suffix = nullptr;

    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::TemplateArgs && !template_params) {
        template_params = &child;
      } else if (child.kind == NodeKind::TypeSuffix) {
        suffix = &child;
        break;
      } else if (child.kind == NodeKind::FunctionSuffix) {
        suffix = &child;
        break;
      }
    }

    if (suffix) {
      if (suffix->kind == NodeKind::TypeSuffix) {
        // Emit forward declaration for the type
        out_ << "class " << name << ";\n";
        return;
      }
      if (suffix->kind == NodeKind::FunctionSuffix) {
        emit_function_forward(name, *suffix, template_params);
        return;
      }
    }
  }
  
  void emit_function_forward(const std::string &name, const Node &suffix,
                             const Node *template_params = nullptr) {
    // Skip main function (doesn't need forward decl)
    if (name == "main")
      return;

    std::string return_type = "auto";
    std::string params;

    for (const auto &child : tree_.children(suffix)) {
      if (child.kind == NodeKind::ParamList) {
        params = emit_params(child);
      } else if (child.kind == NodeKind::ReturnSpec) {
        return_type = emit_type_spec(child);
      }
    }

    // Check for multi-return or single named return type
    auto multi_returns = parse_named_return_fields(return_type);
    std::string named_return_type = extract_named_return_type(return_type);

    if (!multi_returns.empty()) {
      // Multi-return: emit struct definition
      out_ << "struct " << name << "_ret {\n";
      for (const auto &[field_name, field_type] : multi_returns) {
        out_ << "    " << field_type << " " << field_name << ";\n";
      }
      out_ << "};\n";
      return_type = name + "_ret";
    } else if (!named_return_type.empty()) {
      // Single named return: emit type alias
      out_ << "using " << name << "_ret = " << named_return_type << ";\n";
      return_type = name + "_ret";
    }

    // For deduced return types, check if function returns void (no return value)
    // C++ allows forward-declaring void functions even without explicit return type
    // Exception: main() should return int
    if (return_type == "auto") {
      if (name != "main" && !function_has_return_value(suffix)) {
        return_type = "void";
      } else if (name == "main") {
        return_type = "int";
      } else {
        // Has a return value but type is deduced - can't forward declare in C++
        return;
      }
    }

    // Emit template parameters if present
    if (template_params) {
      out_ << "template";
      emit_template_args(*template_params);
      out_ << "\n";
    }

    out_ << "auto " << name << "(" << params << ") -> " << return_type << ";\n";
  }

  void emit_declaration(const Node &n) {
    if (n.kind != NodeKind::Declaration)
      return;

    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::UnifiedDeclaration) {
        emit_unified_decl(child);
      } else if (child.kind == NodeKind::FunctionSuffix) {
        // C++1 function declaration: auto name() -> type { body }
        // Find the identifier name from inside the FunctionSuffix
        std::string name = "unknown";
        for (const auto &grandchild : tree_.children(child)) {
          if (grandchild.kind == NodeKind::Identifier) {
            name = node_text(grandchild);
            break;
          }
        }
        emit_function(name, child);
      }
    }
  }

  void emit_unified_decl(const Node &n) {
    // Try to find identifier name from children first (for C++1 functions)
    std::string name;
    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::Identifier) {
        name = node_text(child);
        break;
      }
    }
    // Fallback to first token for Cpp2 declarations
    if (name.empty()) {
      auto first = std::string(token_text(n.token_start));
      if (first == "operator" && n.token_start + 1 < n.token_end) {
        // operator=: or operator+: etc.
        auto next = std::string(token_text(n.token_start + 1));
        if (next == "=:") {
          name = "operator=";  // EqualColon is combined =:
        } else {
          name = "operator" + next;  // e.g. operator+, operator[]
        }
      } else {
        name = first;
      }
    }

    // Look for template parameters and function/variable/type suffix
    const Node *template_params = nullptr;
    const Node *suffix = nullptr;

    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::TemplateArgs && !template_params) {
        template_params = &child;
      } else if (child.kind == NodeKind::FunctionSuffix) {
        suffix = &child;
        break;
      } else if (child.kind == NodeKind::VariableSuffix) {
        suffix = &child;
        break;
      } else if (child.kind == NodeKind::TypeSuffix) {
        suffix = &child;
        break;
      } else if (child.kind == NodeKind::NamespaceSuffix) {
        suffix = &child;
        break;
      } else if (child.kind == NodeKind::TypeAliasSuffix) {
        suffix = &child;
        break;
      }
    }

    if (suffix) {
      if (suffix->kind == NodeKind::FunctionSuffix) {
        emit_function(name, *suffix, template_params);
        return;
      }
      if (suffix->kind == NodeKind::VariableSuffix) {
        emit_variable(name, *suffix, template_params);
        return;
      }
      if (suffix->kind == NodeKind::TypeSuffix) {
        emit_type(name, *suffix);
        return;
      }
      if (suffix->kind == NodeKind::NamespaceSuffix) {
        emit_namespace(name, *suffix);
        return;
      }
      if (suffix->kind == NodeKind::TypeAliasSuffix) {
        emit_type_alias(name, *suffix);
        return;
      }
    }

    // Fallback: check for concept declaration
    // Cpp2: name: <T> concept = expr; → template<typename T> concept name = expr;
    if (template_params && node_contains_token(n, "concept")) {
      emit_indent();
      out_ << "template";
      emit_template_args(*template_params);
      out_ << " concept " << name << " = ";
      // Find the expression child (the concept body)
      for (const auto &child : tree_.children(n)) {
        if (meta::is_expression(child.kind) ||
            child.kind == NodeKind::AssignmentExpression) {
          emit_expression(child);
          break;
        }
      }
      out_ << ";\n";
      return;
    }

    // Fallback: simple variable with := initialization
    emit_indent();
    // Check if we have an expression child for initialization
    std::string init;
    for (const auto &child : tree_.children(n)) {
      if (meta::is_expression(child.kind) ||
          child.kind == NodeKind::AssignmentExpression) {
        init = node_text(child);
        break;
      }
    }

    if (!init.empty()) {
      out_ << "auto " << name << " = " << init << ";\n";
    } else {
      out_ << "auto " << name << "; // TODO: Unknown decl\n";
    }
  }

  void emit_function(const std::string &name, const Node &suffix,
                     const Node *template_params = nullptr) {
    // Extract return type (if any) and parameters
    std::string return_type = "auto";
    const Node *param_list = nullptr;
    const Node *body = nullptr;

    for (const auto &child : tree_.children(suffix)) {
      if (child.kind == NodeKind::ParamList) {
        param_list = &child;
      } else if (child.kind == NodeKind::ReturnSpec) {
        return_type = emit_type_spec(child);
      } else if (child.kind == NodeKind::FunctionBody) {
        body = &child;
      }
    }

    // Check for multi-return or single named return
    auto multi_returns = parse_named_return_fields(return_type);
    auto [named_ret_var, named_ret_type] = extract_named_return_info(return_type);
    
    if (!multi_returns.empty()) {
      // Multi-return: use struct as return type
      return_type = name + "_ret";
    } else if (!named_ret_type.empty()) {
      // Single named return: use type alias
      return_type = name + "_ret";
    }

    // Infer void return type when there's no explicit return type and no return value
    // Exception: main() should return int, not void
    if (return_type == "auto" && name != "main" && !function_has_return_value(suffix)) {
      return_type = "void";
    }

    // Determine if this function has a forward declaration (and thus defaults
    // should only appear there, not in the definition)
    bool has_forward_decl = (name != "main") && 
        (return_type != "auto" || !function_has_return_value(suffix));
    
    std::string params;
    if (param_list) {
      params = emit_params(*param_list, /*include_defaults=*/!has_forward_decl);
    }

    // Emit template parameters if present
    if (template_params) {
      out_ << "template";
      emit_template_args(*template_params);
      out_ << "\n";
    }

    // Handle main function specially (only for Cpp2 syntax without explicit
    // return)
    bool is_main = (name == "main");
    if (is_main && return_type == "auto") {
      out_ << "int main(int argc, char* argv[]) {\n";
    } else if (is_main) {
      out_ << "int main(int argc, char* argv[]) {\n";
    } else if (return_type == "auto") {
      // Deduced return type
      // For expression bodies, use decltype(auto) to preserve reference returns.
      bool is_expression_body = function_is_expression_body(suffix);
      if (is_expression_body) {
        out_ << "auto " << name << "(" << params << ") -> decltype(auto) {\n";
      } else {
        out_ << "auto " << name << "(" << params << ") {\n";
      }
    } else {
      out_ << "auto " << name << "(" << params << ") -> " << return_type
           << " {\n";
    }

    ++indent_;
    in_function_ = true;
    // For main(), emit args variable for Cpp2's main(args) parameter
    if (is_main) {
      emit_indent();
      out_ << "auto args = cpp2::make_args(argc, argv);\n";
    }
    if (body)
      emit_function_body(*body, named_ret_var, named_ret_type, multi_returns, return_type);
    in_function_ = false;
    --indent_;

    out_ << "}\n\n";
  }

  void emit_template_args(const Node &n) {
    out_ << "<";

    // Template parameters are stored as direct tokens, not child nodes
    // The node spans from < to >, so we need to look at the tokens inside
    bool first = true;
    for (uint32_t i = n.token_start + 1; i < n.token_end - 1; ++i) {
      const auto &token = tokens_[i];

      // Skip commas and whitespace
      if (token.lexeme == "," || token.lexeme.empty()) {
        continue;
      }

      if (!first) {
        out_ << ", ";
      }
      first = false;

      // Convert _ to typename
      if (token.lexeme == "_") {
        out_ << "typename";
      } else {
        out_ << "typename " << token.lexeme;
      }
    }

    out_ << ">";
  }

  std::string emit_params(const Node &n, bool include_defaults = true) {
    std::string result;
    int wildcard_count = 0;  // Track duplicate '_' parameter names
    for (const auto &param : tree_.children(n)) {
      if (param.kind == NodeKind::Parameter) {
        // Skip 'this' parameter - it becomes implicit in C++ member functions
        std::string param_text = node_text(param);
        if (param_text.find("this") != std::string::npos) {
          continue;
        }
        if (!result.empty())
          result += ", ";
        std::string p = emit_param(param, include_defaults);
        // Rename duplicate '_' parameters to '_2', '_3', etc.
        // C++ doesn't allow multiple parameters with the same name
        if (!p.empty()) {
          // Check if param name is '_' (find it at the end of type portion)
          auto space_pos = p.rfind(' ');
          if (space_pos != std::string::npos) {
            std::string pname = p.substr(space_pos + 1);
            // Check for default value
            auto eq_pos = pname.find('=');
            if (eq_pos != std::string::npos) {
              pname = pname.substr(0, eq_pos);
              // trim
              while (!pname.empty() && pname.back() == ' ') pname.pop_back();
            }
            if (pname == "_") {
              wildcard_count++;
              if (wildcard_count > 1) {
                // Replace the trailing '_ ' with '_N '
                p = p.substr(0, space_pos + 1) + "_" + std::to_string(wildcard_count);
                if (eq_pos != std::string::npos) {
                  // re-append default value portion - but this is complex, skip for now
                }
              }
            }
          }
        }
        result += p;
      }
    }
    return result;
  }

  std::string emit_param(const Node &n, bool include_defaults = true) {
    // Get param name and optional type and default value
    std::string name;
    std::string type = "auto";
    std::string qualifier;     // inout, out, in, etc.
    std::string default_value; // Default argument value

    // Check children for qualifier, type, and default value
    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::ParamQualifier) {
        qualifier = node_text(child);
      } else if (child.kind == NodeKind::Identifier) {
        name = node_text(child);
      } else if (child.kind == NodeKind::TypeSpecifier ||
                 child.kind == NodeKind::BasicType) {
        type = format_type(child);
      } else if (meta::is_expression(child.kind)) {
        // This is the default value expression
        default_value = node_text(child);
      }
    }

    if (name.empty()) {
      // Fallback for cases where identifier might not be wrapped (legacy or
      // specific patterns) But be careful not to pick up qualifier if it exists
      if (!qualifier.empty()) {
        // If we have qualifier, token_start is qualifier. name must be later.
        // We should rely on children. If identifier node is missing, maybe it's
        // implicit or failed parse? Assuming parser always produces Identifier
        // node for name now. If fallback needed, maybe skip qualifier tokens?
        // For now, trust Identifier node.
      } else {
        name = std::string(token_text(n.token_start));
      }
    }

    // Apply qualifiers to type
    // Default for parameters is `in` (const reference).
    std::string result_type = type;
    std::string effective_qualifier = qualifier.empty() ? "in" : qualifier;

    if (effective_qualifier == "inout" || effective_qualifier == "out") {
      // Reference parameter
      if (type == "auto") {
        result_type = "auto&";
      } else {
        result_type = type + "&";
      }
    } else if (effective_qualifier == "move") {
      // Rvalue reference
      if (type == "auto") {
        result_type = "auto&&";
      } else {
        result_type = type + "&&";
      }
    } else if (effective_qualifier == "forward") {
      // Forwarding reference (match cppfront style): always auto&&
      result_type = "auto&&";
    } else if (effective_qualifier == "in") {
      // Const reference
      if (type == "auto") {
        result_type = "auto const&";
      } else {
        // Use postfix const& to preserve pointer types correctly
        result_type = type + " const&";
      }
    } else if (effective_qualifier == "copy") {
      // Value (copy) - no change needed
      result_type = type;
    }

    std::string result = result_type + " " + name;
    if (include_defaults && !default_value.empty()) {
      result += " = " + default_value;
    }
    return result;
  }

  std::string emit_type_spec(const Node &n) {
    // Return spec has arrow -> type
    // DEBUG: Print all children
    std::ostringstream debug;
    debug << "emit_type_spec children:\n";
    for (const auto &child : tree_.children(n)) {
      debug << "  kind=" << static_cast<int>(child.kind) << " text='" << node_text(child) << "'\n";
      if (child.kind == NodeKind::TypeSpecifier ||
          child.kind == NodeKind::BasicType ||
          child.kind == NodeKind::QualifiedType) {
        return format_type(child);
      }
    }
    // Fallback: get text after ->
    std::string text = node_text(n);
    debug << "  fallback, text='" << text << "'\n";
    auto pos = text.find("->");
    if (pos != std::string::npos) {
      std::string result = text.substr(pos + 2);
      // Trim whitespace
      while (!result.empty() && std::isspace(result.front()))
        result.erase(0, 1);
      // Apply format_type to process any decltype expressions
      if (result.find("decltype(") != std::string::npos) {
        // We need to process this through format_type, but we only have text
        // Create a fake type spec from the text
        // For now, apply UFCS transformation manually
        return result; // Just return raw text
      }
      return result.empty() ? "auto" : result;
    }
    return "auto";
  }
  
  // Check if return type is a named return like (i:int) and extract the type
  // Returns empty string if not a named return
  std::string extract_named_return_type(const std::string &return_spec) {
    // Pattern: (name:type) or ( name : type ) or (n1:t1, n2:t2, ...)
    std::string s = return_spec;
    // Trim whitespace
    while (!s.empty() && std::isspace(s.front())) s.erase(0, 1);
    while (!s.empty() && std::isspace(s.back())) s.pop_back();
    
    if (s.empty() || s.front() != '(' || s.back() != ')') 
      return "";
    
    // Remove outer parens
    s = s.substr(1, s.length() - 2);
    
    // Find the colon
    auto colon_pos = s.find(':');
    if (colon_pos == std::string::npos)
      return "";
    
    // Check if there are multiple named returns (contains comma)
    if (s.find(',') != std::string::npos) {
      // Multiple named returns - return a placeholder indicating struct needed
      return "__MULTI_RETURN__";
    }
    
    // Extract type after colon
    std::string type_part = s.substr(colon_pos + 1);
    // Trim whitespace
    while (!type_part.empty() && std::isspace(type_part.front())) 
      type_part.erase(0, 1);
    
    // Strip default value initializer (e.g., std::string = "xyzzy" -> std::string)
    {
      int pd = 0, ad = 0;
      for (size_t i = 0; i < type_part.size(); ++i) {
        if (type_part[i] == '(') pd++;
        else if (type_part[i] == ')') pd--;
        else if (type_part[i] == '<') ad++;
        else if (type_part[i] == '>') ad--;
        else if (type_part[i] == '=' && pd == 0 && ad == 0) {
          type_part = type_part.substr(0, i);
          while (!type_part.empty() && std::isspace(type_part.back())) type_part.pop_back();
          break;
        }
      }
    }
    
    return type_part.empty() ? "" : type_part;
  }

  // Parse named return fields from format like "n1:t1, n2:t2"
  // Returns vector of {name, type} pairs
  std::vector<std::pair<std::string, std::string>> parse_named_return_fields(const std::string &return_spec) {
    std::vector<std::pair<std::string, std::string>> fields;
    std::string s = return_spec;
    
    // Trim whitespace
    while (!s.empty() && std::isspace(s.front())) s.erase(0, 1);
    while (!s.empty() && std::isspace(s.back())) s.pop_back();
    
    if (s.empty() || s.front() != '(' || s.back() != ')') 
      return fields;
    
    // Remove outer parens
    s = s.substr(1, s.length() - 2);
    
    // Split by comma
    std::vector<std::string> parts;
    size_t start = 0;
    int paren_depth = 0;
    int angle_depth = 0;
    for (size_t i = 0; i <= s.length(); ++i) {
      if (i == s.length() || (s[i] == ',' && paren_depth == 0 && angle_depth == 0)) {
        if (i > start) {
          parts.push_back(s.substr(start, i - start));
        }
        start = i + 1;
      } else if (s[i] == '(') {
        paren_depth++;
      } else if (s[i] == ')') {
        paren_depth--;
      } else if (s[i] == '<') {
        angle_depth++;
      } else if (s[i] == '>') {
        angle_depth--;
      }
    }
    
    // Parse each part as "name:type"
    for (const auto &part : parts) {
      auto colon_pos = part.find(':');
      if (colon_pos == std::string::npos) continue;
      
      std::string name = part.substr(0, colon_pos);
      std::string type = part.substr(colon_pos + 1);
      
      // Trim whitespace
      while (!name.empty() && std::isspace(name.front())) name.erase(0, 1);
      while (!name.empty() && std::isspace(name.back())) name.pop_back();
      while (!type.empty() && std::isspace(type.front())) type.erase(0, 1);
      while (!type.empty() && std::isspace(type.back())) type.pop_back();
      
      // Strip default value initializer from type (e.g., std::string = "xyzzy" -> std::string)
      // Look for '=' that's not inside angle brackets or parens
      {
        int pd = 0, ad = 0;
        for (size_t i = 0; i < type.size(); ++i) {
          if (type[i] == '(') pd++;
          else if (type[i] == ')') pd--;
          else if (type[i] == '<') ad++;
          else if (type[i] == '>') ad--;
          else if (type[i] == '=' && pd == 0 && ad == 0) {
            type = type.substr(0, i);
            while (!type.empty() && std::isspace(type.back())) type.pop_back();
            break;
          }
        }
      }
      
      if (!name.empty() && !type.empty()) {
        fields.push_back({name, type});
      }
    }
    
    // Only return non-empty for multi-return case (2+ fields)
    // Single named returns are handled by extract_named_return_info
    if (fields.size() < 2) {
      return {};
    }
    
    return fields;
  }

  // Extract both name and type from named return like (i:int)
  // Returns {name, type} pair, or {"", ""} if not a named return or if multiple
  std::pair<std::string, std::string> extract_named_return_info(const std::string &return_spec) {
    std::string s = return_spec;
    // Trim whitespace
    while (!s.empty() && std::isspace(s.front())) s.erase(0, 1);
    while (!s.empty() && std::isspace(s.back())) s.pop_back();
    
    if (s.empty() || s.front() != '(' || s.back() != ')') 
      return {"", ""};
    
    // Remove outer parens
    s = s.substr(1, s.length() - 2);
    
    // Check if multiple returns (contains comma at top level)
    int paren_depth = 0;
    int angle_depth = 0;
    for (char c : s) {
      if (c == ',' && paren_depth == 0 && angle_depth == 0) {
        // Multiple named returns - handled separately
        return {"", ""};
      }
      if (c == '(') paren_depth++;
      if (c == ')') paren_depth--;
      if (c == '<') angle_depth++;
      if (c == '>') angle_depth--;
    }
    
    // Find the colon
    auto colon_pos = s.find(':');
    if (colon_pos == std::string::npos)
      return {"", ""};
    
    // Extract name before colon
    std::string name_part = s.substr(0, colon_pos);
    // Trim whitespace
    while (!name_part.empty() && std::isspace(name_part.front())) 
      name_part.erase(0, 1);
    while (!name_part.empty() && std::isspace(name_part.back())) 
      name_part.pop_back();
    
    // Extract type after colon
    std::string type_part = s.substr(colon_pos + 1);
    // Trim whitespace
    while (!type_part.empty() && std::isspace(type_part.front())) 
      type_part.erase(0, 1);
    
    // Strip default value initializer (e.g., std::string = "xyzzy" -> std::string)
    {
      int pd = 0, ad = 0;
      for (size_t i = 0; i < type_part.size(); ++i) {
        if (type_part[i] == '(') pd++;
        else if (type_part[i] == ')') pd--;
        else if (type_part[i] == '<') ad++;
        else if (type_part[i] == '>') ad--;
        else if (type_part[i] == '=' && pd == 0 && ad == 0) {
          type_part = type_part.substr(0, i);
          while (!type_part.empty() && std::isspace(type_part.back())) type_part.pop_back();
          break;
        }
      }
    }
    
    if (name_part.empty() || type_part.empty())
      return {"", ""};
    
    return {name_part, type_part};
  }

  void emit_function_body(const Node &n, const std::string &named_ret_var = "", const std::string &named_ret_type = "", 
                          const std::vector<std::pair<std::string, std::string>> &multi_returns = {},
                          const std::string &return_type = "auto") {
    // If we have multiple named return variables, declare them all at the top
    if (!multi_returns.empty()) {
      for (const auto &[name, type] : multi_returns) {
        emit_indent();
        out_ << type << " " << name << ";\n";
      }
    }
    // If we have a single named return variable, declare it at the top
    else if (!named_ret_var.empty() && !named_ret_type.empty()) {
      emit_indent();
      out_ << named_ret_type << " " << named_ret_var << ";\n";
    }
    
    bool ends_with_return = false;
    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::BlockStatement) {
        ends_with_return = block_ends_with_return(child);
        emit_block(child, named_ret_var, multi_returns);
      } else {
        // Expression body: = expr;
        emit_indent();
        
        // Check if this is a discard expression (_ = expr)
        bool is_discard = false;
        for (const auto &ec : tree_.children(child)) {
          if (ec.kind == NodeKind::AssignmentExpression) {
            auto first_tok = token_text(ec.token_start);
            if (first_tok == "_") {
              is_discard = true;
            }
            break;
          }
          // Check direct token
          if (ec.token_start < ec.token_end && token_text(ec.token_start) == "_") {
            is_discard = true;
            break;
          }
        }
        // Also check directly if the expression starts with _
        if (!is_discard) {
          auto first = token_text(child.token_start);
          if (first == "_") {
            is_discard = true;
          }
        }
        
        if (return_type == "void" || is_discard) {
          // Void function or discard expression - don't wrap in return
          emit_expression(child);
          out_ << ";\n";
        } else {
          out_ << "return ";
          emit_expression(child);
          out_ << ";\n";
        }
        return; // Expression body already handled
      }
    }
    
    // Add implicit return at end for named return values (only if not already returning)
    if (!ends_with_return) {
      if (!multi_returns.empty()) {
        emit_indent();
        out_ << "return {";
        for (size_t i = 0; i < multi_returns.size(); ++i) {
          if (i > 0) out_ << ", ";
          out_ << "std::move(" << multi_returns[i].first << ")";
        }
        out_ << "};\n";
      } else if (!named_ret_var.empty()) {
        emit_indent();
        out_ << "return std::move(" << named_ret_var << ");\n";
      }
    }
  }
  
  // Check if a block ends with a return statement
  bool block_ends_with_return(const Node &n) {
    const Node *last_stmt = nullptr;
    for (const auto &child : tree_.children(n)) {
      last_stmt = &child;
    }
    if (!last_stmt) return false;
    
    // Direct return statement
    if (last_stmt->kind == NodeKind::ReturnStatement) {
      return true;
    }
    // Statement wrapper - check inside
    if (last_stmt->kind == NodeKind::Statement) {
      for (const auto &child : tree_.children(*last_stmt)) {
        if (child.kind == NodeKind::ReturnStatement) {
          return true;
        }
      }
    }
    return false;
  }

  void emit_block(const Node &n, const std::string &named_ret_var = "",
                  const std::vector<std::pair<std::string, std::string>> &multi_returns = {}) {
    for (const auto &child : tree_.children(n)) {
      emit_statement(child, named_ret_var, multi_returns);
    }
  }

  void emit_statement(const Node &n, const std::string &named_ret_var = "",
                      const std::vector<std::pair<std::string, std::string>> &multi_returns = {}) {
    emit_indent();

    if (n.kind == NodeKind::ReturnStatement) {
      out_ << "return";
      bool has_value = false;
      for (const auto &child : tree_.children(n)) {
        out_ << " " << emit_expression_text(child);
        has_value = true;
      }
      // For multi-returns with no value, emit struct constructor
      if (!has_value && !multi_returns.empty()) {
        out_ << " {";
        for (size_t i = 0; i < multi_returns.size(); ++i) {
          if (i > 0) out_ << ", ";
          out_ << multi_returns[i].first;
        }
        out_ << "}";
      }
      // For single named return with no value, return the var
      else if (!has_value && !named_ret_var.empty()) {
        out_ << " " << named_ret_var;
      }
      out_ << ";\n";
    } else if (n.kind == NodeKind::IfStatement) {
      emit_if(n, named_ret_var, multi_returns);
    } else if (n.kind == NodeKind::WhileStatement) {
      emit_while(n, named_ret_var, multi_returns);
    } else if (n.kind == NodeKind::ForStatement) {
      emit_for(n, named_ret_var, multi_returns);
    } else if (n.kind == NodeKind::DoWhileStatement) {
      emit_do_while(n, named_ret_var, multi_returns);
    } else if (n.kind == NodeKind::BlockStatement) {
      out_ << "{\n";
      ++indent_;
      emit_block(n, named_ret_var, multi_returns);
      --indent_;
      emit_indent();
      out_ << "}\n";
    } else if (n.kind == NodeKind::ExpressionStatement) {
      // Remove trailing semicolon if present in node_text to avoid double
      // semicolon Actually, we should call emit_expression for consistency But
      // node_text might include the semicolon if it was parsed as
      // ExpressionStatement Let's rely on emit_expression + ";"
      for (const auto &child : tree_.children(n)) {
        if (meta::is_expression(child.kind)) {
          emit_expression(child);
          out_ << ";\n";
          return;
        }
      }
      // Fallback
      out_ << node_text(n) << "\n";
    } else if (n.kind == NodeKind::Statement) {
      // Check the first token for keyword-based statements
      std::string_view first = (n.token_start < tokens_.size()) 
          ? token_text(n.token_start) : "";
      
      if (first == "try") {
        emit_try_catch(n, named_ret_var, multi_returns);
      } else if (first == "throw") {
        out_ << "throw";
        // Emit the expression after 'throw' (if any)
        for (const auto &child : tree_.children(n)) {
          if (meta::is_expression(child.kind)) {
            out_ << " ";
            emit_expression(child);
          }
        }
        out_ << ";\n";
      } else if (first == "break") {
        out_ << "break;\n";
      } else if (first == "continue") {
        out_ << "continue;\n";
      } else if (first == "next") {
        // Cpp2 'next' is like continue but with a clause
        // For now, emit the clause as the loop increment
        // 'next expr;' → the expr is the loop increment, just emit it
        for (const auto &child : tree_.children(n)) {
          emit_statement(child, named_ret_var);
        }
      } else {
        // Generic statement - check children
        for (const auto &child : tree_.children(n)) {
          emit_statement(child, named_ret_var);
        }
      }
    } else if (n.kind == NodeKind::UnifiedDeclaration) {
      // Local variable
      emit_local_var(n);
    } else if (n.kind == NodeKind::AssertStatement) {
      emit_assert_statement(n);
    } else {
      out_ << node_text(n) << ";\n";
    }
  }

  // Emit assert statement as contract check
  void emit_assert_statement(const Node &n) {
    // Format: if (handler.is_active() && !(condition)) { handler.report_violation(CPP2_CONTRACT_MSG("message")); }
    // assert(cond, "msg")          -> cpp2::cpp2_default
    // assert<type>(cond, "msg")    -> cpp2::type
    // assert<type, audit>(cond, msg) -> audit && cpp2::type
    // assert<unevaluated>(cond)    -> no output (completely suppressed)

    // Parse the assert statement structure
    // Token layout: [assert] [<type>] [(] [condition] [,] [message] [)] [;]

    std::string contract_kind = "cpp2_default";  // default
    std::string audit_flag = "";  // optional audit flag

    // Check for template args between 'assert' and '('
    uint32_t open_paren = n.token_start + 1;

    // Check if there's a '<' after 'assert'
    while (open_paren < n.token_end && open_paren < tokens_.size()) {
      if (tokens_[open_paren].lexeme == "<") {
        // Parse template args: <type> or <type, audit>
        uint32_t gt_pos = open_paren + 1;
        int angle_depth = 1;
        while (gt_pos < n.token_end && gt_pos < tokens_.size() && angle_depth > 0) {
          if (tokens_[gt_pos].lexeme == "<") angle_depth++;
          if (tokens_[gt_pos].lexeme == ">") angle_depth--;
          gt_pos++;
        }

        // Extract args between < and >
        std::string args;
        for (uint32_t t = open_paren + 1; t < gt_pos - 1; ++t) {
          args += tokens_[t].lexeme;
        }

        // Parse args: could be "type" or "type, audit"
        auto comma_pos = args.find(',');
        if (comma_pos != std::string::npos) {
          contract_kind = trim(args.substr(0, comma_pos));
          audit_flag = trim(args.substr(comma_pos + 1));
        } else {
          contract_kind = trim(args);
        }

        open_paren = gt_pos;  // Move past '>'
      }
      if (tokens_[open_paren].lexeme == "(") {
        break;
      }
      open_paren++;
    }

    // Special case: unevaluated contracts emit nothing
    if (contract_kind == "unevaluated") {
      return;  // No output
    }

    // Now find condition and message
    // Condition starts after '(' and ends at ',' or ')'
    // Message starts after ',' and ends at ')'

    uint32_t cond_start = open_paren + 1;
    uint32_t cond_end = open_paren + 1;
    uint32_t msg_start = 0;
    uint32_t msg_end = 0;

    // Find the comma separator
    int paren_depth = 0;
    for (uint32_t t = open_paren; t < n.token_end && t < tokens_.size(); ++t) {
      if (tokens_[t].lexeme == "(") paren_depth++;
      if (tokens_[t].lexeme == ")") {
        paren_depth--;
        if (paren_depth == 0) {
          msg_end = t;
          break;
        }
      }
      if (tokens_[t].lexeme == "," && paren_depth == 1) {
        cond_end = t;
        msg_start = t + 1;
      }
    }

    // If no message, condition goes to ')'
    if (msg_start == 0) {
      cond_end = msg_end;
    }

    // Emit the contract check
    out_ << "if (";

    // Emit audit flag if present
    if (!audit_flag.empty()) {
      out_ << audit_flag << " && ";
    }

    // Emit handler check
    out_ << "cpp2::" << contract_kind << ".is_active() && !(";

    // Emit condition
    for (uint32_t t = cond_start; t < cond_end; ++t) {
      out_ << tokens_[t].lexeme;
    }
    out_ << ") ) { cpp2::" << contract_kind << ".report_violation(CPP2_CONTRACT_MSG(";

    // Emit message if present
    if (msg_start > 0 && msg_end > msg_start) {
      for (uint32_t t = msg_start; t < msg_end; ++t) {
        out_ << tokens_[t].lexeme;
      }
    } else {
      out_ << "\"contract violation\"";
    }

    out_ << ")); }\n";
  }

  void emit_if(const Node &n, const std::string &named_ret_var = "",
               const std::vector<std::pair<std::string, std::string>> &multi_returns = {}) {
    out_ << "if (";
    bool emitted_condition = false;
    int block_count = 0;
    for (const auto &child : tree_.children(n)) {
      if (!emitted_condition && child.kind != NodeKind::BlockStatement &&
          child.kind != NodeKind::Statement) {
        emit_expression(child);
        emitted_condition = true;
      } else if (child.kind == NodeKind::BlockStatement) {
        if (block_count == 0) {
          // Then block
          out_ << ") {\n";
          ++indent_;
          emit_block(child, named_ret_var, multi_returns);
          --indent_;
          emit_indent();
          out_ << "}";
        } else {
          // Else block
          out_ << " else {\n";
          ++indent_;
          emit_block(child, named_ret_var, multi_returns);
          --indent_;
          emit_indent();
          out_ << "}";
        }
        block_count++;
      } else if (child.kind == NodeKind::Statement) {
        // Single-statement branch
        if (block_count == 0) {
          out_ << ") ";
          emit_statement(child, named_ret_var);
        } else {
          out_ << " else ";
          emit_statement(child, named_ret_var);
        }
        block_count++;
      }
    }
    out_ << "\n";
  }

  void emit_while(const Node &n, const std::string &named_ret_var = "",
                  const std::vector<std::pair<std::string, std::string>> &multi_returns = {}) {
    out_ << "while (";
    bool first = true;
    for (const auto &child : tree_.children(n)) {
      if (first && child.kind != NodeKind::BlockStatement) {
        emit_expression(child);
        first = false;
      } else if (child.kind == NodeKind::BlockStatement) {
        out_ << ") {\n";
        ++indent_;
        emit_block(child, named_ret_var, multi_returns);
        --indent_;
        emit_indent();
        out_ << "} \n";
      }
    }
  }

  // Emit do-while loop
  // Cpp2: do { body } next clause while (cond);
  // C++:  do { body } while (cond);
  void emit_do_while(const Node &n, const std::string &named_ret_var = "",
                     const std::vector<std::pair<std::string, std::string>> &multi_returns = {}) {
    out_ << "do ";
    const Node *cond = nullptr;
    
    // Find the block (body) and condition expression
    // DoWhileStatement: do block [next clause] while expr ;
    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::BlockStatement) {
        out_ << "{\n";
        ++indent_;
        emit_block(child, named_ret_var, multi_returns);
        --indent_;
        emit_indent();
        out_ << "} ";
      } else if (meta::is_expression(child.kind)) {
        cond = &child;
      }
    }
    
    out_ << "while (";
    if (cond) {
      emit_expression(*cond);
    } else {
      out_ << "true";
    }
    out_ << ");\n";
  }

  void emit_for(const Node &n, const std::string &named_ret_var = "",
                const std::vector<std::pair<std::string, std::string>> &multi_returns = {}) {
    // Cpp2 for: for items do (item) { body }
    // C++1 for: for (item : items) { body }
    // Both emit as: for (auto item : items) { body }
    std::string items, var, qualifier;
    const Node *body = nullptr;

    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::BlockStatement) {
        body = &child;
      } else if (child.kind == NodeKind::ParamQualifier && qualifier.empty()) {
        qualifier = node_text(child);
      } else if (child.kind == NodeKind::Identifier && var.empty()) {
        var = node_text(child);
      } else if (child.kind == NodeKind::Parameter && var.empty()) {
        // Parameter node may contain qualifier + identifier children
        for (const auto &pc : tree_.children(child)) {
          if (pc.kind == NodeKind::ParamQualifier && qualifier.empty()) {
            qualifier = node_text(pc);
          } else if (pc.kind == NodeKind::Identifier && var.empty()) {
            var = node_text(pc);
          }
        }
        if (var.empty()) var = node_text(child);
      } else if (meta::is_expression(child.kind) && items.empty()) {
        // Only use expression nodes for items (skip keywords, punctuation)
        items = emit_expression_text(child);
      }
    }

    // Map Cpp2 qualifiers to C++ binding
    std::string binding = "auto";
    if (qualifier == "inout") binding = "auto&";
    else if (qualifier == "move") binding = "auto&&";
    else if (qualifier == "in") binding = "const auto&";

    out_ << "for (" << binding << " " << var << " : " << items << ") {\n";
    ++indent_;
    if (body)
      emit_block(*body, named_ret_var, multi_returns);
    --indent_;
    emit_indent();
    out_ << "}\n";
  }

  // Emit try/catch statement
  // Cpp2: try { ... } catch (name: type) { ... } catch (_) { ... }
  // C++:  try { ... } catch (type name) { ... } catch (...) { ... }
  void emit_try_catch(const Node &n, const std::string &named_ret_var = "",
                      const std::vector<std::pair<std::string, std::string>> &multi_returns = {}) {
    // Walk through the tokens of this Statement node to reconstruct try/catch
    // The structure is: try { block } catch ( params ) { block } [catch ...]
    
    // Collect blocks and catch info by iterating tokens
    enum class Phase { TryKeyword, TryBody, CatchKeyword, CatchParams, CatchBody };
    Phase phase = Phase::TryKeyword;
    
    // Find block statement children - they alternate: try body, catch body, catch body, ...
    std::vector<const Node *> blocks;
    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::BlockStatement) {
        blocks.push_back(&child);
      }
    }
    
    // Parse catch clauses from tokens
    struct CatchInfo {
      std::string param_name;
      std::string param_type;
      bool is_catch_all = false;
    };
    std::vector<CatchInfo> catches;
    
    // Walk tokens to find catch clause parameters
    for (uint32_t i = n.token_start; i < n.token_end && i < tokens_.size(); ++i) {
      if (tokens_[i].lexeme == "catch") {
        CatchInfo info;
        // Skip to opening paren
        ++i;
        while (i < n.token_end && i < tokens_.size() && tokens_[i].lexeme != "(") ++i;
        if (i >= n.token_end) break;
        ++i; // skip (
        
        // Collect param tokens until )
        std::string param_name;
        std::string param_type;
        bool found_colon = false;
        
        while (i < n.token_end && i < tokens_.size() && tokens_[i].lexeme != ")") {
          if (tokens_[i].lexeme == "_") {
            info.is_catch_all = true;
          } else if (tokens_[i].lexeme == ":") {
            found_colon = true;
          } else if (!found_colon) {
            // Before colon: parameter name (skip qualifiers like 'in')
            auto lex = std::string(tokens_[i].lexeme);
            if (lex != "in" && lex != "inout" && lex != "out" && lex != "move" && lex != "forward" && lex != "copy") {
              param_name = lex;
            }
          } else {
            // After colon: type
            if (!param_type.empty()) param_type += " ";
            param_type += std::string(tokens_[i].lexeme);
          }
          ++i;
        }
        
        if (!info.is_catch_all) {
          info.param_name = param_name;
          info.param_type = param_type.empty() ? "..." : param_type;
        }
        catches.push_back(info);
      }
    }
    
    // Emit try block
    out_ << "try ";
    if (!blocks.empty()) {
      out_ << "{\n";
      ++indent_;
      emit_block(*blocks[0], named_ret_var, multi_returns);
      --indent_;
      emit_indent();
      out_ << "}";
    }
    
    // Emit catch blocks
    for (size_t i = 0; i < catches.size(); ++i) {
      const auto &ci = catches[i];
      if (ci.is_catch_all) {
        out_ << " catch (...) ";
      } else if (ci.param_type == "...") {
        out_ << " catch (...) ";
      } else {
        out_ << " catch (" << ci.param_type << " " << ci.param_name << ") ";
      }
      
      size_t block_idx = i + 1; // blocks[0] is try body
      if (block_idx < blocks.size()) {
        out_ << "{\n";
        ++indent_;
        emit_block(*blocks[block_idx], named_ret_var, multi_returns);
        --indent_;
        emit_indent();
        out_ << "}";
      } else {
        out_ << "{ }";
      }
    }
    out_ << "\n";
  }

  void emit_local_var(const Node &n) {
    std::string name;
    std::string type = "auto";
    const Node *init_expr = nullptr;

    // Look for Identifier child first
    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::Identifier) {
        name = node_text(child);
        break;
      }
    }
    // Fallback to first token if no Identifier child
    if (name.empty()) {
      name = std::string(token_text(n.token_start));
    }

    // Check for children
    for (const auto &child : tree_.children(n)) {
      if (child.kind == NodeKind::VariableSuffix) {
        // Inside VariableSuffix: [TypeSpecifier] [= Expression] ;
        for (const auto &grandchild : tree_.children(child)) {
          if (grandchild.kind == NodeKind::TypeSpecifier ||
              grandchild.kind == NodeKind::BasicType) {
            type = format_type(grandchild);
          } else if (meta::is_expression(grandchild.kind)) {
            init_expr = &grandchild;
          }
        }
      } else if (child.kind == NodeKind::TypeSpecifier ||
                 child.kind == NodeKind::BasicType) {
        // Handle direct type specifier (parser quirk/bug workaround)
        type = format_type(child);
      } else if (meta::is_expression(child.kind)) {
        // Direct expression child means := deduction
        init_expr = &child;
      }
    }

    out_ << type << " " << name;
    if (init_expr) {
      out_ << " = ";
      emit_initializer(*init_expr);
    }
    out_ << ";\n";
  }

  void emit_variable(const std::string &name, const Node &suffix,
                     const Node *template_params = nullptr) {
    // Global variable (from emit_unified_decl)
    emit_indent();

    std::string type = "auto";
    const Node *init_expr = nullptr;

    // Emit template parameters if present
    if (template_params) {
      out_ << "template";
      emit_template_args(*template_params);
      out_ << " ";
    }

    // Suffix might be VariableSuffix, or UnifiedDeclaration logic might have
    // passed the suffix node But emit_unified_decl passes the child node.

    // If suffix is VariableSuffix:
    if (suffix.kind == NodeKind::VariableSuffix) {
      for (const auto &child : tree_.children(suffix)) {
        if (child.kind == NodeKind::TypeSpecifier ||
            child.kind == NodeKind::QualifiedType ||
            child.kind == NodeKind::BasicType) {
          type = format_type(child);
        } else if (meta::is_expression(child.kind)) {
          init_expr = &child;
        }
      }
    } else {
      // Check direct children of the suffix node (whatever it is)
      for (const auto &child : tree_.children(suffix)) {
        if (child.kind == NodeKind::TypeSpecifier ||
            child.kind == NodeKind::QualifiedType ||
            child.kind == NodeKind::BasicType) {
          type = format_type(child);
        } else if (meta::is_expression(child.kind)) {
          init_expr = &child;
        }
      }
    }

    out_ << type << " " << name;
    if (init_expr) {
      out_ << " = ";
      emit_initializer(*init_expr);
    }
    out_ << ";\n";
  }

  // Emit an initializer expression, converting (a,b,c) tuple syntax to {a,b,c}
  void emit_initializer(const Node &n) {
    std::string text = trim(node_text(n));
    // Check if it's empty parens () -> brace init {}
    if (text == "()") {
      out_ << "{}";
      return;
    }
    // Check if it's a parenthesized expression with commas (tuple initializer)
    if (!text.empty() && text.front() == '(' && text.back() == ')' &&
        text.find(',') != std::string::npos) {
      // Replace outer () with {} for brace initialization
      // Text is like "(1, 2, 3)" -> "{1, 2, 3}"
      out_ << "{" << text.substr(1, text.length() - 2) << "}";
      return;
    }
    // Default: emit as regular expression
    emit_expression(n);
  }

  // Extract metafunction names from TypeSuffix
  std::vector<std::string> extract_metafunctions(const Node &suffix) {
    std::vector<std::string> result;
    for (const auto &child : tree_.children(suffix)) {
      if (child.kind == NodeKind::Metafunction) {
        // Extract the metafunction name (skip @ token)
        std::string text = node_text(child);
        // Remove leading @ if present
        if (!text.empty() && text[0] == '@') {
          text = text.substr(1);
        }
        // Remove any template args
        size_t pos = text.find('<');
        if (pos != std::string::npos) {
          text = text.substr(0, pos);
        }
        result.push_back(text);
      }
    }
    return result;
  }

  // Check if a metafunction is present
  bool has_metafunction(const std::vector<std::string> &metafunctions, const std::string &name) {
    for (const auto &mf : metafunctions) {
      if (mf == name) return true;
    }
    return false;
  }

  // Emit special members for @value metafunction
  void emit_value_special_members(const std::string &name) {
    // Default constructor
    emit_indent();
    out_ << "public: explicit " << name << "();\n";
    
    // Spaceship operator for comparison
    emit_indent();
    out_ << "public: [[nodiscard]] auto operator<=>(" << name << " const& that) const& -> std::strong_ordering = default;\n";
    
    // Copy constructor
    emit_indent();
    out_ << "public: " << name << "(" << name << " const& that);\n";
    
    // Copy assignment
    emit_indent();
    out_ << "public: auto operator=(" << name << " const& that) -> " << name << "&;\n";
    
    // Move constructor
    emit_indent();
    out_ << "public: " << name << "(" << name << "&& that) noexcept;\n";
    
    // Move assignment
    emit_indent();
    out_ << "public: auto operator=(" << name << "&& that) noexcept -> " << name << "&;\n";
  }

  // Emit special members for @ordered metafunction
  void emit_ordered_special_members(const std::string &name) {
    // Spaceship operator with weak_ordering
    emit_indent();
    out_ << "public: [[nodiscard]] auto operator<=>(" << name << " const& that) const& -> std::weak_ordering = default;\n";
  }

  // Emit special members for @interface metafunction
  void emit_interface_special_members(const std::string &name, const Node *body) {
    // Explicit default constructor
    emit_indent();
    out_ << "public: explicit " << name << "();\n";
    
    // Protected copy (to prevent slicing)
    emit_indent();
    out_ << "protected: " << name << "([[maybe_unused]] " << name << " const& that);\n";
    emit_indent();
    out_ << "protected: auto operator=([[maybe_unused]] " << name << " const& that) -> " << name << "&;\n";
    
    // Protected move
    emit_indent();
    out_ << "protected: " << name << "([[maybe_unused]] " << name << "&& that) noexcept;\n";
    emit_indent();
    out_ << "protected: auto operator=([[maybe_unused]] " << name << "&& that) noexcept -> " << name << "&;\n";
    
    // Virtual destructor
    emit_indent();
    out_ << "public: virtual ~" << name << "() noexcept;\n";
  }

  // Emit special members for @polymorphic_base metafunction
  void emit_polymorphic_base_special_members(const std::string &name) {
    // Virtual destructor
    emit_indent();
    out_ << "public: virtual ~" << name << "() noexcept;\n";
    
    // Delete copy/move to prevent slicing
    emit_indent();
    out_ << "public: " << name << "(" << name << " const&) = delete;\n";
    emit_indent();
    out_ << "public: auto operator=(" << name << " const&) -> void = delete;\n";
  }

  // Extract base classes from TypeBody by looking for 'this: BaseType' members
  std::vector<std::string> extract_base_classes(const Node &suffix) {
    std::vector<std::string> bases;
    for (const auto &child : tree_.children(suffix)) {
      if (child.kind == NodeKind::TypeBody) {
        for (const auto &member : tree_.children(child)) {
          if (member.kind == NodeKind::Declaration) {
            for (const auto &udc : tree_.children(member)) {
              if (udc.kind == NodeKind::UnifiedDeclaration) {
                std::string member_name;
                for (const auto &gc : tree_.children(udc)) {
                  if (gc.kind == NodeKind::Identifier) {
                    member_name = node_text(gc);
                    break;
                  }
                }
                if (member_name == "this") {
                  // this: Base is an inheritance declaration
                  for (const auto &gc : tree_.children(udc)) {
                    if (gc.kind == NodeKind::VariableSuffix) {
                      for (const auto &type_child : tree_.children(gc)) {
                        if (type_child.kind == NodeKind::TypeSpecifier || type_child.kind == NodeKind::BasicType) {
                          bases.push_back(format_type(type_child));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return bases;
  }

  void emit_type(const std::string &name, const Node &suffix) {
    // Extract metafunctions from the TypeSuffix
    auto metafunctions = extract_metafunctions(suffix);
    bool is_interface = has_metafunction(metafunctions, "interface");
    
    // Extract base classes from 'this: Base' members
    auto base_classes = extract_base_classes(suffix);
    
    // Use 'class' for interface types (for virtual functions)
    if (is_interface) {
      out_ << "class " << name;
    } else {
      out_ << "struct " << name;
    }
    
    // Emit base class list if any
    if (!base_classes.empty()) {
      out_ << " : ";
      for (size_t i = 0; i < base_classes.size(); ++i) {
        if (i > 0) out_ << ", ";
        out_ << "public " << base_classes[i];
      }
    }
    out_ << " {\n";
    ++indent_;
    
    // Find the TypeBody child and emit its declarations
    const Node *body = nullptr;
    for (const auto &child : tree_.children(suffix)) {
      if (child.kind == NodeKind::TypeBody) {
        body = &child;
        emit_type_body_with_interface(child, name, is_interface);
      }
    }
    
    // Emit special members based on metafunctions
    if (has_metafunction(metafunctions, "value") || 
        has_metafunction(metafunctions, "weakly_ordered_value") ||
        has_metafunction(metafunctions, "partially_ordered_value")) {
      emit_value_special_members(name);
    }
    if (has_metafunction(metafunctions, "ordered")) {
      emit_ordered_special_members(name);
    }
    if (is_interface) {
      emit_interface_special_members(name, body);
    }
    if (has_metafunction(metafunctions, "polymorphic_base")) {
      emit_polymorphic_base_special_members(name);
    }
    
    --indent_;
    out_ << "};\n\n";
  }

  void emit_type_body_with_interface(const Node &body, const std::string &type_name, bool is_interface) {
    // TypeBody contains Declaration nodes
    for (const auto &child : tree_.children(body)) {
      if (child.kind == NodeKind::Declaration) {
        emit_type_member_with_interface(child, type_name, is_interface);
      }
    }
  }

  void emit_type_member_with_interface(const Node &decl, const std::string &type_name, bool is_interface) {
    // Get the member name and determine if it's a field or method
    for (const auto &child : tree_.children(decl)) {
      if (child.kind == NodeKind::UnifiedDeclaration) {
        std::string member_name;
        for (const auto &grandchild : tree_.children(child)) {
          if (grandchild.kind == NodeKind::Identifier) {
            member_name = node_text(grandchild);
            break;
          }
        }
        if (member_name.empty()) {
          auto first = std::string(token_text(child.token_start));
          if (first == "operator" && child.token_start + 1 < child.token_end) {
            auto next = std::string(token_text(child.token_start + 1));
            if (next == "=:") {
              member_name = "operator=";
            } else {
              member_name = "operator" + next;
            }
          } else {
            member_name = first;
          }
        }
        
        // Check if it's a function (method), variable (field), or type alias
        for (const auto &grandchild : tree_.children(child)) {
          if (grandchild.kind == NodeKind::FunctionSuffix) {
            emit_method_with_interface(member_name, grandchild, type_name, is_interface);
            return;
          }
          if (grandchild.kind == NodeKind::VariableSuffix) {
            // Skip 'this: Base' members - they become base class inheritance
            if (member_name == "this") {
              return;
            }
            emit_field(member_name, grandchild);
            return;
          }
          if (grandchild.kind == NodeKind::TypeAliasSuffix) {
            emit_type_alias(member_name, grandchild);
            return;
          }
        }
      }
    }
  }

  // Check if function body is empty (just a `;` declaration, no actual content)
  bool is_empty_function_body(const Node *body) {
    if (!body) return true;
    // Check if body has no children or all children are empty
    bool has_content = false;
    for (const auto &child : tree_.children(*body)) {
      has_content = true;
      break;
    }
    if (!has_content) {
      // Also check if the source text is just a semicolon
      std::string text = node_text(*body);
      // Trim whitespace
      size_t start = text.find_first_not_of(" \t\n\r");
      if (start == std::string::npos) return true;
      size_t end = text.find_last_not_of(" \t\n\r");
      text = text.substr(start, end - start + 1);
      return text == ";" || text.empty();
    }
    return false;
  }

  void emit_method_with_interface(const std::string &name, const Node &suffix, 
                                   const std::string &type_name, bool is_interface) {
    // Similar to emit_method but makes pure virtual for interfaces
    std::string return_type = "auto";
    std::string params;
    const Node *body = nullptr;
    bool is_const = false;
    
    for (const auto &child : tree_.children(suffix)) {
      if (child.kind == NodeKind::ParamList) {
        // Check for 'this' parameter to determine const
        for (const auto &param : tree_.children(child)) {
          if (param.kind == NodeKind::Parameter) {
            std::string param_text = node_text(param);
            if (param_text.find("this") != std::string::npos && 
                param_text.find("inout") == std::string::npos &&
                param_text.find("out") == std::string::npos) {
              is_const = true;
            }
          }
        }
        params = emit_params(child);
      } else if (child.kind == NodeKind::ReturnSpec) {
        return_type = emit_type_spec(child);
      } else if (child.kind == NodeKind::FunctionBody) {
        body = &child;
      }
    }
    
    // Check for named return type
    std::string named_return_type = extract_named_return_type(return_type);
    auto [named_ret_var, named_ret_type] = extract_named_return_info(return_type);
    auto multi_returns = parse_named_return_fields(return_type);
    if (!multi_returns.empty()) {
      // Multi-return: emit struct definition (handled by forward decl)
      return_type = name + "_ret";
    } else if (!named_return_type.empty()) {
      return_type = named_return_type;
    }
    
    // Infer void return type when no explicit return
    if (return_type == "auto" && !function_has_return_value(suffix)) {
      return_type = "void";
    }
    
    emit_indent();
    
    // For interfaces with empty body (just `;`), emit as pure virtual
    if (is_interface && is_empty_function_body(body)) {
      out_ << "public: virtual auto " << name << "(" << params << ")";
      if (is_const) out_ << " const";
      out_ << " -> " << return_type << " = 0;\n";
      return;
    }
    
    out_ << "auto " << name << "(" << params << ")";
    if (is_const) out_ << " const";
    out_ << " -> " << return_type << " {\n";
    
    ++indent_;
    if (body)
      emit_function_body(*body, named_ret_var, named_ret_type, multi_returns, return_type);
    --indent_;
    
    emit_indent();
    out_ << "}\n";
  }
  
  void emit_type_body(const Node &body, const std::string &type_name) {
    // TypeBody contains Declaration nodes
    for (const auto &child : tree_.children(body)) {
      if (child.kind == NodeKind::Declaration) {
        emit_type_member(child, type_name);
      }
    }
  }
  
  void emit_type_member(const Node &decl, const std::string &type_name) {
    // Get the member name and determine if it's a field or method
    for (const auto &child : tree_.children(decl)) {
      if (child.kind == NodeKind::UnifiedDeclaration) {
        std::string member_name;
        for (const auto &grandchild : tree_.children(child)) {
          if (grandchild.kind == NodeKind::Identifier) {
            member_name = node_text(grandchild);
            break;
          }
        }
        if (member_name.empty()) {
          auto first = std::string(token_text(child.token_start));
          if (first == "operator" && child.token_start + 1 < child.token_end) {
            auto next = std::string(token_text(child.token_start + 1));
            if (next == "=:") {
              member_name = "operator=";
            } else {
              member_name = "operator" + next;
            }
          } else {
            member_name = first;
          }
        }
        
        // Check if it's a function (method) or variable (field)
        for (const auto &grandchild : tree_.children(child)) {
          if (grandchild.kind == NodeKind::FunctionSuffix) {
            emit_method(member_name, grandchild, type_name);
            return;
          }
          if (grandchild.kind == NodeKind::VariableSuffix) {
            emit_field(member_name, grandchild);
            return;
          }
        }
      }
    }
  }
  
  void emit_method(const std::string &name, const Node &suffix, const std::string &type_name) {
    // Similar to emit_function but for methods
    std::string return_type = "auto";
    std::string params;
    const Node *body = nullptr;
    bool is_static = false;
    
    for (const auto &child : tree_.children(suffix)) {
      if (child.kind == NodeKind::ParamList) {
        // Check for 'this' parameter to determine static vs non-static
        for (const auto &param : tree_.children(child)) {
          if (param.kind == NodeKind::Parameter) {
            std::string param_text = node_text(param);
            if (param_text.find("this") != std::string::npos) {
              is_static = false;
            }
          }
        }
        params = emit_params(child);
      } else if (child.kind == NodeKind::ReturnSpec) {
        return_type = emit_type_spec(child);
      } else if (child.kind == NodeKind::FunctionBody) {
        body = &child;
      }
    }
    
    // Check for named return type
    std::string named_return_type = extract_named_return_type(return_type);
    if (!named_return_type.empty()) {
      return_type = named_return_type;  // Use the actual type for methods
    }
    
    // Infer void return type when no explicit return
    if (return_type == "auto" && !function_has_return_value(suffix)) {
      return_type = "void";
    }
    
    emit_indent();
    if (is_static) {
      out_ << "static ";
    }
    out_ << "auto " << name << "(" << params << ") -> " << return_type << " {\n";
    
    ++indent_;
    if (body)
      emit_function_body(*body);
    --indent_;
    
    emit_indent();
    out_ << "}\n";
  }
  
  void emit_field(const std::string &name, const Node &suffix) {
    std::string type = "auto";
    std::string init;
    
    for (const auto &child : tree_.children(suffix)) {
      if (child.kind == NodeKind::TypeSpecifier || child.kind == NodeKind::BasicType) {
        type = format_type(child);
      } else if (meta::is_expression(child.kind)) {
        init = node_text(child);
      }
    }
    
    emit_indent();
    out_ << type << " " << name;
    if (!init.empty()) {
      out_ << " = " << init;
    }
    out_ << ";\n";
  }

  bool is_infix_expression(NodeKind k) const {
    using K = NodeKind;
    if (k < K::MultiplicativeExpression || k > K::AssignmentExpression)
      return false;
    return k != K::TernaryExpression && k != K::PipelineExpression &&
           k != K::RangeExpression && k != K::AssignmentOp;
  }

  void emit_expression(const Node &n) {
    if (n.kind == NodeKind::Identifier) {
      std::string name = node_text(n);
      // Qualify known cpp2 runtime functions with cpp2:: namespace
      if (name == "unchecked_narrow" || name == "unchecked_cast" ||
          name == "narrow" || name == "narrow_cast") {
        out_ << "cpp2::" << name;
      }
      // Map Cpp2 type aliases to C++ types (use cpp2:: namespace prefix)
      else if (name == "i8" || name == "i16" || name == "i32" || name == "i64" ||
               name == "u8" || name == "u16" || name == "u32" || name == "u64" ||
               name == "f32" || name == "f64") {
        out_ << "cpp2::" << name;
      }
      else {
        out_ << name;
      }
    } else if (n.kind == NodeKind::Literal) {
      // Apply string interpolation processing for string literals
      std::string text = node_text(n);
      if (text.size() >= 2 && text.front() == '"' && text.back() == '"' &&
          text.find(")$") != std::string::npos) {
        out_ << process_string_interpolation(text);
      } else {
        out_ << text;
      }
    } else if (n.kind == NodeKind::Expression) {
      for (const auto &child : tree_.children(n)) {
        emit_expression(child);
      }
    } else if (n.kind == NodeKind::AssignmentExpression) {
      std::vector<const Node *> children;
      for (const auto &child : tree_.children(n))
        children.push_back(&child);

      if (children.size() >= 3 && token_text(children[0]->token_start) == "_") {
        out_ << "(void)(";
        emit_expression(*children[2]);
        out_ << ")";
      } else {
        for (const auto *child : children) {
          if (child->kind == NodeKind::AssignmentOp ||
              child->kind == NodeKind::BinaryOp ||
              (child->token_start < child->token_end &&
               is_operator(token_text(child->token_start)))) {
            auto op = node_text(*child);
            if (op == ",") {
              out_ << ", ";
            } else {
              out_ << " " << op << " ";
            }
          } else {
            emit_expression(*child);
          }
        }
      }
    } else if (n.kind == NodeKind::IsExpression) {
      // is expression: expr is Type -> cpp2::is<Type>(expr)
      auto children = tree_.children(n);
      std::vector<const Node *> parts;
      for (const auto &child : children) {
        if (child.kind != NodeKind::BinaryOp) {
          parts.push_back(&child);
        }
      }
      if (parts.size() >= 2) {
        out_ << "cpp2::is<";
        // Second part is the type
        out_ << format_type(*parts[1]);
        out_ << ">(";
        emit_expression(*parts[0]);
        out_ << ")";
      } else {
        // Fallback
        out_ << node_text(n);
      }
    } else if (n.kind == NodeKind::AsExpression) {
      // as expression: expr as Type -> cpp2::as<Type>(expr)
      auto children = tree_.children(n);
      std::vector<const Node *> parts;
      for (const auto &child : children) {
        if (child.kind != NodeKind::BinaryOp) {
          parts.push_back(&child);
        }
      }
      if (parts.size() >= 2) {
        out_ << "cpp2::as<";
        // Second part is the type
        out_ << format_type(*parts[1]);
        out_ << ">(";
        emit_expression(*parts[0]);
        out_ << ")";
      } else {
        // Fallback
        out_ << node_text(n);
      }
    } else if (is_infix_expression(n.kind)) {
      for (const auto &child : tree_.children(n)) {
        if (child.kind == NodeKind::BinaryOp ||
            child.kind == NodeKind::AssignmentOp ||
            (child.token_start < child.token_end &&
             is_operator(token_text(child.token_start)))) {
          std::string op = node_text(child);
          if (op == ",") {
            out_ << ", "; // comma: no leading space
          } else {
            out_ << " " << op << " ";
          }
        } else {
          emit_expression(child);
        }
      }

    } else if (n.kind == NodeKind::CallOp) {
      auto children = tree_.children(n);
      auto it = children.begin();
      if (it != children.end()) {
        const Node &callee = *it;
        ++it;
        
        // Check if this is a UFCS call (obj.method(args) -> CPP2_UFCS(method)(obj, args))
        // But NOT if it's explicit member access with .. (double dot)
        if (callee.kind == NodeKind::MemberOp) {
          // Check if the MemberOp contains ".." token (explicit member, not UFCS)
          bool is_explicit_member = node_contains_token(callee, "..");
          
          if (!is_explicit_member) {
            // UFCS: obj.method(args) should become CPP2_UFCS(method)(obj, args)
            auto member_children = tree_.children(callee);
            auto member_it = member_children.begin();
            
            if (member_it != member_children.end()) {
              const Node &obj = *member_it;
              ++member_it;
              
              // Get the method name
              std::string method_name;
              for (; member_it != member_children.end(); ++member_it) {
                if (member_it->kind == NodeKind::Identifier) {
                  method_name = node_text(*member_it);
                  break;
                }
              }
              
              if (!method_name.empty()) {
                // Emit: CPP2_UFCS(method)(obj, args...)
                // Use CPP2_UFCS_NONLOCAL when in global scope
                const char *ufcs_macro = in_function_ ? "CPP2_UFCS" : "CPP2_UFCS_NONLOCAL";
                out_ << ufcs_macro << "(" << method_name << ")(";
                emit_expression(obj);
                for (; it != children.end(); ++it) {
                  // CallOp may still contain separator/operator nodes depending on how the
                  // expression was parsed; skip those and emit only argument expressions.
                  if (it->kind == NodeKind::BinaryOp || it->kind == NodeKind::AssignmentOp)
                    continue;
                  if (it->token_start < it->token_end && token_text(it->token_start) == ",")
                    continue;
                  out_ << ", ";
                  emit_expression(*it);
                }
                out_ << ")";
                return;
              }
            }
          }
          // Explicit member call (..): fall through to emit as regular call
        }
        
        // Not UFCS - regular function call
        emit_expression(callee);
        out_ << "(";
        bool first = true;
        for (; it != children.end(); ++it) {
          if (it->kind == NodeKind::BinaryOp || it->kind == NodeKind::AssignmentOp)
            continue;
          if (it->token_start < it->token_end && token_text(it->token_start) == ",")
            continue;
          if (!first)
            out_ << ", ";
          emit_expression(*it);
          first = false;
        }
        out_ << ")";
      } else {
        // Sibling fallback
        out_ << "(";
        bool first = true;
        for (const auto &child : children) {
          if (!first)
            out_ << ", ";
          emit_expression(child);
          first = false;
        }
        out_ << ")";
      }
    } else if (n.kind == NodeKind::MemberOp) {
      auto children = tree_.children(n);
      auto it = children.begin();
      if (it != children.end()) {
        // Check for unique.new<T> / shared.new<T> pattern
        std::string lhs_text = node_text(*it);
        std::string rhs_text;
        auto peek = it;
        ++peek;
        for (auto p = peek; p != children.end(); ++p) {
          rhs_text += node_text(*p);
        }
        if ((lhs_text == "unique" || lhs_text == "shared") && 
            rhs_text.find("new") != std::string::npos) {
          // Transform unique.new<T> → std::make_unique<T>
          //          shared.new<T> → std::make_shared<T>
          if (lhs_text == "unique") {
            // Replace "new" with the full make_unique call
            size_t new_pos = rhs_text.find("new");
            out_ << "std::make_unique" << rhs_text.substr(new_pos + 3);
          } else {
            size_t new_pos = rhs_text.find("new");
            out_ << "std::make_shared" << rhs_text.substr(new_pos + 3);
          }
          return;
        }
        emit_expression(*it); // Adoptee (LHS)
        ++it;
        // In hierarchical mode, start_infix was called at the operator.
        // n.token_start is the operator token.
        std::string op = std::string(token_text(n.token_start));
        // Handle :: (scope), . (member), and .. (explicit member - emit as single .)
        if (op == "::") {
          out_ << "::";
        } else {
          // Both . and .. emit as single . in C++
          out_ << ".";
        }
        for (; it != children.end(); ++it) {
          if (it->kind == NodeKind::Identifier)
            out_ << node_text(*it);
          else
            emit_expression(*it);
        }
      } else {
        // Sibling fallback
        std::string op = std::string(token_text(n.token_start));
        out_ << (op == "::" ? "::" : ".");
        for (const auto &child : children) {
          out_ << node_text(child);
        }
      }
    } else if (n.kind == NodeKind::ScopeOp) {
      // Scope resolution operator :: (e.g., std::cout or ::global)
      auto children = tree_.children(n);
      auto it = children.begin();
      
      // Check if this is a leading :: (global scope)
      // If the first token of this ScopeOp is ::, it's a global scope reference
      bool is_global_scope = (token_text(n.token_start) == "::");
      
      if (is_global_scope) {
        // Leading :: - emit :: first, then all children
        out_ << "::";
        for (; it != children.end(); ++it) {
          if (it->kind == NodeKind::Identifier)
            out_ << node_text(*it);
          else
            emit_expression(*it);
        }
      } else if (it != children.end()) {
        emit_expression(*it); // Adoptee (LHS - namespace or class)
        ++it;
        out_ << "::";
        for (; it != children.end(); ++it) {
          if (it->kind == NodeKind::Identifier)
            out_ << node_text(*it);
          else
            emit_expression(*it);
        }
      } else {
        // Sibling fallback
        out_ << "::";
        for (const auto &child : children) {
          out_ << node_text(child);
        }
      }
    } else if (n.kind == NodeKind::TemplateArgs) {
      auto children = tree_.children(n);
      auto it = children.begin();
      if (it != children.end()) {
        emit_expression(*it); // Adoptee
        ++it;
        out_ << "<";
        bool first = true;
        for (; it != children.end(); ++it) {
          if (!first)
            out_ << ", ";
          if (it->kind == NodeKind::TypeSpecifier ||
              it->kind == NodeKind::QualifiedType ||
              it->kind == NodeKind::BasicType) {
            out_ << format_type(*it);
          } else {
            emit_expression(*it);
          }
          first = false;
        }
        out_ << ">";
      } else {
        out_ << "<";
        bool first = true;
        for (const auto &child : children) {
          if (!first)
            out_ << ", ";
          emit_expression(child);
          first = false;
        }
        out_ << ">";
      }
    } else if (n.kind == NodeKind::TypeSpecifier ||
               n.kind == NodeKind::QualifiedType) {
      out_ << format_type(n);
    } else if (n.kind == NodeKind::SubscriptOp) {
      auto children = tree_.children(n);
      auto it = children.begin();
      if (it != children.end()) {
        emit_expression(*it); // Adoptee
        ++it;
        out_ << "[";
        for (; it != children.end(); ++it) {
          emit_expression(*it);
        }
        out_ << "]";
      } else {
        out_ << "[";
        for (const auto &child : children) {
          emit_expression(child);
        }
        out_ << "]";
      }
    } else if (n.kind == NodeKind::PostfixOp) {
      auto children = tree_.children(n);
      auto it = children.begin();
      if (it != children.end()) {
        const Node &lhs = *it;
        ++it;
        if (it != children.end()) {
          std::string op = node_text(*it);
          if (op == "*") {
            out_ << "*(";
            emit_expression(lhs);
            out_ << ")";
          } else if (op == "&") {
            out_ << "&(";
            emit_expression(lhs);
            out_ << ")";
          } else {
            emit_expression(lhs);
            out_ << op;
          }
        } else {
          out_ << node_text(n);
        }
      } else {
        out_ << node_text(n);
      }
    } else if (n.kind == NodeKind::TernaryExpression) {
      auto children = tree_.children(n);
      std::vector<const Node *> parts;
      for (const auto &child : children)
        parts.push_back(&child);

      if (parts.size() == 3) {
        emit_expression(*parts[0]);
        out_ << " ? ";
        emit_expression(*parts[1]);
        out_ << " : ";
        emit_expression(*parts[2]);
      } else {
        out_ << node_text(n);
      }
    } else if (n.kind == NodeKind::GroupedExpression) {
      out_ << "(";
      for (const auto &child : tree_.children(n)) {
        emit_expression(child);
      }
      out_ << ")";
    } else if (n.kind == NodeKind::PrefixExpression) {
      for (const auto &child : tree_.children(n)) {
        if (child.kind == NodeKind::PrefixOp) {
          out_ << node_text(child);
        } else {
          emit_expression(child);
        }
      }
    } else if (n.kind == NodeKind::InspectExpression) {
      emit_inspect_expression(n);
    } else {
      // Fallback
      out_ << node_text(n);
    }
  }

  void emit_inspect_expression(const Node &n) {
    // Structure: inspect expr -> type { arm1; arm2; ... }
    // Emits as: [&] () -> type { auto&& _expr = expr; if (...) return ...; else return ...; }

    auto children = tree_.children(n);

    // Get the expression being inspected (first child)
    const Node *inspect_expr = nullptr;
    const Node *return_type = nullptr;
    std::vector<const Node*> arms;

    for (const auto &child : children) {
      if (child.kind == NodeKind::Expression || meta::is_expression(child.kind)) {
        if (!inspect_expr) {
          inspect_expr = &child;
        }
      } else if (child.kind == NodeKind::TypeSpecifier) {
        return_type = &child;
      } else if (child.kind == NodeKind::InspectArm) {
        arms.push_back(&child);
      }
    }

    // Emit lambda: [&] () -> type
    out_ << "[&] () -> ";
    if (return_type) {
      out_ << format_type(*return_type);
    } else {
      out_ << "auto";
    }

    out_ << " { ";

    // Emit: auto&& _expr = <inspect_expr>;
    if (inspect_expr) {
      out_ << "auto&& _expr = ";
      emit_expression(*inspect_expr);
      out_ << "; ";
    }

    // Emit each arm as if/else if
    // For generic functions, we need SFINAE guards to prevent instantiation
    // of invalid code paths
    bool first = true;
    std::string return_type_str = return_type ? format_type(*return_type) : "auto";
    
    for (const auto *arm : arms) {
      if (!arm) continue;

      auto arm_children = tree_.children(*arm);

      // Get pattern and body
      const Node *pattern = nullptr;
      const Node *body = nullptr;

      for (const auto &child : arm_children) {
        if (child.kind == NodeKind::Pattern || child.kind == NodeKind::IsPattern) {
          pattern = &child;
        } else if (meta::is_expression(child.kind) || child.kind == NodeKind::BlockStatement) {
          body = &child;
        }
      }

      if (pattern && body) {
        if (first) {
          out_ << "if (";
          first = false;
        } else {
          out_ << "else if (";
        }

        // Emit pattern check: cpp2::is(_expr, pattern) or cpp2::is<type>(_expr)
        emit_pattern_check(*pattern);

        out_ << ") { ";
        
        // Capture body expression as string for SFINAE guards
        std::ostringstream body_stream;
        std::swap(out_, body_stream);
        emit_expression(*body);
        std::swap(out_, body_stream);
        std::string body_str = body_stream.str();
        
        // Emit SFINAE-guarded return:
        // if constexpr(requires{<body>}) 
        //   if constexpr(std::is_convertible_v<decltype((<body>)),<return_type>>) 
        //     return <body>;
        //   else return <return_type>{};
        // else return <return_type>{};
        out_ << "if constexpr( requires{" << body_str << ";} ) ";
        out_ << "if constexpr( std::is_convertible_v<decltype((" << body_str << "))," << return_type_str << "> ) ";
        out_ << "return " << body_str << "; ";
        out_ << "else return " << return_type_str << "{}; ";
        out_ << "else return " << return_type_str << "{}; ";
        out_ << "} ";
      }
    }

    // Default case (wildcard) if no match
    if (!first) {
      out_ << "else return ";
      if (return_type) {
        out_ << format_type(*return_type) << "{}";
      } else {
        out_ << "\"(no match)\"";
      }
      out_ << "; ";
    }

    out_ << "} ()";
  }

  void emit_pattern_check(const Node &pattern) {
    // Pattern can be:
    // - IsPattern: "is type" or "is expr"
    // - AsPattern: "as type = identifier"
    // - Pattern node containing IsPattern/AsPattern/Wildcard/Expression
    // - Wildcard: "_"
    // - Expression

    std::string debug_text = trim(node_text(pattern));

    // First check if this is a Pattern node containing other patterns
    if (pattern.kind == NodeKind::Pattern) {
      auto children = tree_.children(pattern);
      for (const auto &child : children) {
        if (child.kind == NodeKind::IsPattern || child.kind == NodeKind::AsPattern) {
          // Delegate to the inner pattern
          emit_pattern_check(child);
          return;
        }
      }
      
      // No inner pattern found - check if the text starts with "is " (token-based fallback)
      if (debug_text.size() >= 3 && debug_text.substr(0, 3) == "is ") {
        std::string target = trim(debug_text.substr(3));
        if (target == "_" || target.empty()) {
          out_ << "true";
        } else if (is_type_like(target)) {
          out_ << "cpp2::is<" << target << ">(_expr)";
        } else {
          out_ << "cpp2::is(_expr, " << target << ")";
        }
        return;
      }
      
      // Check if it's a wildcard "_"
      if (debug_text == "_") {
        out_ << "true";
        return;
      }
      // It's an expression pattern
      out_ << "cpp2::is(_expr, ";
      emit_expression(pattern);
      out_ << ")";
      return;
    }

    if (pattern.kind == NodeKind::IsPattern) {
      // "is type" or "is expr"
      // IsPattern has children: first token is "is", followed by type or expression
      auto children = tree_.children(pattern);

      // Check for TypeSpecifier child (type pattern)
      for (const auto &child : children) {
        if (child.kind == NodeKind::TypeSpecifier || child.kind == NodeKind::BasicType) {
          std::string type_text = trim(format_type(child));
          if (type_text == "_" || type_text == "auto") {
            out_ << "true";  // Wildcard matches anything
          } else {
            out_ << "cpp2::is<" << type_text << ">(_expr)";
          }
          return;
        }
      }

      // Check for expression child (value pattern)
      for (const auto &child : children) {
        if (meta::is_expression(child.kind)) {
          std::string expr_text = trim(node_text(child));
          if (expr_text == "_") {
            out_ << "true";  // Wildcard matches anything
          } else {
            out_ << "cpp2::is(_expr, ";
            emit_expression(child);
            out_ << ")";
          }
          return;
        }
      }

      // Fallback: parse the text
      std::string text = trim(node_text(pattern));
      // Remove "is " prefix if present
      if (text.substr(0, 3) == "is ") {
        text = trim(text.substr(3));
      }

      if (text == "_" || text.empty()) {
        out_ << "true";
      } else if (is_type_like(text)) {
        out_ << "cpp2::is<" << text << ">(_expr)";
      } else {
        out_ << "cpp2::is(_expr, " << text << ")";
      }
    } else if (pattern.kind == NodeKind::AsPattern) {
      // "as type = identifier" - binds value to identifier
      // For now, just emit true (TODO: implement binding)
      out_ << "true";
    } else {
      // Expression pattern
      std::string text = trim(node_text(pattern));
      if (text == "_") {
        out_ << "true";
      } else {
        out_ << "cpp2::is(_expr, ";
        emit_expression(pattern);
        out_ << ")";
      }
    }
  }

  bool is_type_like(const std::string &s) const {
    // Check if s looks like a type name
    // Types: contain ::, <>, or are known type names
    if (s.find("::") != std::string::npos) return true;
    if (s.find('<') != std::string::npos) return true;
    // Known simple types
    static const std::unordered_set<std::string> known_types = {
        "int", "unsigned", "signed", "char", "short", "long", "float", "double",
        "bool", "void", "auto", "size_t", "ptrdiff_t", "int8_t", "int16_t",
        "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
        "string", "vector", "map", "set", "optional", "variant", "any"
    };
    return known_types.count(s) > 0;
  }

  bool is_operator(std::string_view s) const {
    return s.find_first_not_of("+-*/%&|^<>=!~") == std::string_view::npos ||
           s == "and" || s == "or" || s == "not";
  }

  void emit_namespace(const std::string &name, const Node &suffix) {
    // Check if this is a namespace alias (namespace == Name) or body (namespace = { ... })
    bool has_body = false;
    for (const auto &child : tree_.children(suffix)) {
      if (child.kind == NodeKind::NamespaceBody) {
        has_body = true;
        break;
      }
    }
    
    if (has_body) {
      out_ << "namespace " << name << " {\n\n";
      for (const auto &child : tree_.children(suffix)) {
        if (child.kind == NodeKind::NamespaceBody) {
          for (const auto &decl : tree_.children(child)) {
            emit_declaration(decl);
          }
        }
      }
      out_ << "} // namespace " << name << "\n\n";
    } else {
      // Namespace alias: namespace == QualifiedName
      emit_indent();
      out_ << "namespace " << name << " = ";
      for (const auto &child : tree_.children(suffix)) {
        if (child.kind == NodeKind::TypeSpecifier || child.kind == NodeKind::BasicType) {
          out_ << format_type(child);
          break;
        }
      }
      out_ << ";\n";
    }
  }

  void emit_type_alias(const std::string &name, const Node &suffix) {
    // Cpp2: name: type == OtherType; → using name = OtherType;
    emit_indent();
    out_ << "using " << name << " = ";
    for (const auto &child : tree_.children(suffix)) {
      if (child.kind == NodeKind::TypeSpecifier || child.kind == NodeKind::BasicType) {
        out_ << format_type(child);
        break;
      }
    }
    out_ << ";\n";
  }
};

} // anonymous namespace

std::string generate_from_tree(const cpp2::ast::ParseTree &tree,
                               std::span<const cpp2_transpiler::Token> tokens) {
  TreeEmitter emitter(tree, tokens);
  return emitter.emit();
}
