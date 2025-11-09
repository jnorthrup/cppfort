#pragma once

#include <string>

#include "pijul_patch.h"

namespace cppfort::pijul {

Patch compute_line_patch(const std::string& patch_name,
                         const std::string& source,
                         const std::string& target);

} // namespace cppfort::pijul

