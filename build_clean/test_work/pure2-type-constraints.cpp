

//=== Cpp2 type declarations ====================================================




class irregular;


//=== Cpp2 type definitions and function declarations ===========================


auto print(auto const& r) -> void
CPP2_REQUIRES (std::regular<CPP2_TYPEOF(r)>) ;

auto print([[maybe_unused]] auto const& unnamed_param_1) -> void;

class irregular {
      public: irregular() = default;
      public: irregular(irregular const&) = delete; /* No 'that' constructor, suppress copy */
      public: auto operator=(irregular const&) -> void = delete;
};

auto main() -> int;

//=== Cpp2 function definitions =================================================


auto print(auto const& r) -> void
requires (std::regular<CPP2_TYPEOF(r)>) {
    std::cout << "satisfies std::regular\n";
}

auto print([[maybe_unused]] auto const& unnamed_param_1) -> void{
    std::cout << "fallback\n";
}

auto main() -> int{
    print(42);
    print(irregular());

    std::regular auto ok {42}; 
    //err: _ is std::regular = irregular();
}

