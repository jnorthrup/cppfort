

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto fine() -> int;

[[nodiscard]] auto bad_optional_access() -> int;

[[nodiscard]] auto main() -> int;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto fine() -> int
{
    auto up {CPP2_UFCS_TEMPLATE(cpp2_new<int>)(cpp2::unique, 1)}; 
    auto sp {CPP2_UFCS_TEMPLATE(cpp2_new<int>)(cpp2::shared, 2)}; 
    std::optional<int> op {3}; 

    return *cpp2::impl::assert_not_null(cpp2::move(up)) + *cpp2::impl::assert_not_null(cpp2::move(sp)) + *cpp2::impl::assert_not_null(cpp2::move(op)); 
}

[[nodiscard]] auto bad_optional_access() -> int
{
    std::optional<int> op {std::nullopt}; 
    return *cpp2::impl::assert_not_null(cpp2::move(op)); 
}

[[nodiscard]] auto main() -> int
{
    std::set_terminate(std::abort);
    return fine() + bad_optional_access(); 
}

