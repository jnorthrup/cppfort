#include "cpp2util.h"


struct crash_m0b {
};

struct crash_m0c {
    int name = default;
[[nodiscard]] auto get_name(auto this) -> int     {
        return name;
    }
};

