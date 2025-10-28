// Custom tblgen backend to generate C++ orbit pattern code from semantic_units.td
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <string>
#include <vector>

using namespace llvm;

namespace {

void EmitSemanticUnits(RecordKeeper &Records, raw_ostream &OS) {
  OS << "// Generated from semantic_units.td - DO NOT EDIT\n";
  OS << "#pragma once\n\n";
  OS << "#include <string>\n";
  OS << "#include <vector>\n";
  OS << "#include <map>\n\n";
  OS << "namespace cppfort::stage0::tblgen {\n\n";

  // Get all records that derive from SemanticUnit
  auto Defs = Records.getAllDerivedDefinitions("SemanticUnit");

  for (Record *Def : Defs) {
    std::string Name = Def->getName().str();
    std::string UnitName = Def->getValueAsString("Name").str();

    // Get segments
    auto SegmentsList = Def->getValueAsListOfStrings("Segments");

    // Get patterns
    std::string CPattern = Def->getValueAsString("C_pattern").str();
    std::string CPPPattern = Def->getValueAsString("CPP_pattern").str();
    std::string CPP2Pattern = Def->getValueAsString("CPP2_pattern").str();

    // Generate struct
    OS << "struct " << Name << " {\n";
    OS << "  static constexpr const char* name = \"" << UnitName << "\";\n";
    OS << "  static constexpr std::array<const char*, " << SegmentsList.size()
       << "> segments = {";

    for (size_t i = 0; i < SegmentsList.size(); ++i) {
      OS << "\"" << SegmentsList[i] << "\"";
      if (i + 1 < SegmentsList.size()) OS << ", ";
    }
    OS << "};\n\n";

    OS << "  static constexpr const char* c_pattern = \"" << CPattern << "\";\n";
    OS << "  static constexpr const char* cpp_pattern = \"" << CPPPattern << "\";\n";
    OS << "  static constexpr const char* cpp2_pattern = \"" << CPP2Pattern << "\";\n";
    OS << "};\n\n";
  }

  OS << "} // namespace cppfort::stage0::tblgen\n";
}

} // anonymous namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    errs() << "Usage: " << argv[0] << " <input.td>\n";
    return 1;
  }

  RecordKeeper Records;
  if (TableGenMain(argv[0], &EmitSemanticUnits) != 0) {
    return 1;
  }

  return 0;
}
