

//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


auto f(cpp2::impl::in<cpp2::i32> i, std::string& s) -> void;

auto main() -> int;

//=== Cpp2 function definitions =================================================


auto f(cpp2::impl::in<cpp2::i32> i, std::string& s) -> void{
    // j := i as i16;                     // error, maybe-lossy narrowing
    auto j {cpp2::unchecked_narrow<cpp2::i16>(i)}; // ok, 'unchecked' is explicit

    void* pv {&s}; 
    // pi := pv as *std::string;          // error, unsafe cast
    auto ps {cpp2::unchecked_cast<std::string*>(cpp2::move(pv))}; // ok, 'unchecked' is explicit
    *cpp2::impl::assert_not_null(ps) = "plugh";
}

auto main() -> int{
    std::string str {"xyzzy"}; 
    f(42, str);
    std::cout << cpp2::move(str);
}

