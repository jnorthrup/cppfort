#pragma once

namespace cppfort::stage0 {

enum class ParameterKind {
    Default,
    In,
    Out,
    InOut,
    Move,
    Forward
};

} // namespace cppfort::stage0
