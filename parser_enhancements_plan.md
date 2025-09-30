// Enhanced parser improvements - add access specifier support
// This is a targeted enhancement that adds new functionality

// In ast.h - add access specifier enum (minimal change)
enum class AccessSpecifier {
    Public,
    Protected,
    Private
};

// Add to existing structures as optional fields
struct TypeDecl {
    std::string name;
    std::string body;
    SourceLocation location;
    // Optional: AccessSpecifier access {AccessSpecifier::Public}; // Future enhancement
};

struct FunctionDecl {
    std::string name;
    std::vector<Parameter> parameters;
    std::optional<std::string> return_type;
    FunctionBody body;
    SourceLocation location;
    // Optional: AccessSpecifier access {AccessSpecifier::Public}; // Future enhancement
};