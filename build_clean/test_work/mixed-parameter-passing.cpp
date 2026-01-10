#include <string>
#include <cstdlib>
#include <ctime>


//=== Cpp2 type declarations ====================================================




template<typename T> class container;
    

//=== Cpp2 type definitions and function declarations ===========================


auto copy_from([[maybe_unused]] auto unnamed_param_1) -> void;

auto parameter_styles(
    [[maybe_unused]] cpp2::impl::in<std::string> unnamed_param_1, // "in" is default
    std::string b, 
    [[maybe_unused]] std::string& unnamed_param_3, 
    std::string&& d
    ) -> void;

[[nodiscard]] auto min(auto const& a, auto const& b) -> auto&&;

template<typename T> class container {
    private: std::array<T,10> buf {}; 
    public: [[nodiscard]] auto operator[](cpp2::impl::in<cpp2::i32> idx) const& -> T const&;
    public: [[nodiscard]] auto operator[](cpp2::impl::in<cpp2::i32> idx) & -> T&;
    public: container() = default;
    public: container(container const&) = delete; /* No 'that' constructor, suppress copy */
    public: auto operator=(container const&) -> void = delete;

};

auto main() -> int;


//=== Cpp2 function definitions =================================================


auto copy_from([[maybe_unused]] auto unnamed_param_1) -> void{}

auto parameter_styles(
    [[maybe_unused]] cpp2::impl::in<std::string> unnamed_param_1, 
    std::string b, 
    [[maybe_unused]] std::string& unnamed_param_3, 
    std::string&& d
    ) -> void
{
    int z {12}; 

    ++z;
    b += "plugh";

    if (std::rand() % CPP2_ASSERT_NOT_ZERO_LITERAL(CPP2_TYPEOF(std::rand()),2)) {
        ++z;
        copy_from(cpp2::move(b));// definite last use
    }
    else {
        copy_from(&b);  // NB: better not move from this (why not?)
        copy_from(cpp2::move(d));
        copy_from(++z);
    }

    // std::move(z);

    copy_from(z);

    if (std::time(nullptr) % CPP2_ASSERT_NOT_ZERO_LITERAL(CPP2_TYPEOF(std::time(nullptr)),2) == 0) {
        copy_from(cpp2::move(z));
    }

}

[[nodiscard]] auto min(auto const& a, auto const& b) -> auto&&
    {if (cpp2::impl::cmp_less(b,a)) {return b; }else {return a; }}

    template <typename T> [[nodiscard]] auto container<T>::operator[](cpp2::impl::in<cpp2::i32> idx) const& -> T const& { return CPP2_ASSERT_IN_BOUNDS(buf, idx); }
    template <typename T> [[nodiscard]] auto container<T>::operator[](cpp2::impl::in<cpp2::i32> idx) & -> T& { return CPP2_ASSERT_IN_BOUNDS(buf, idx);  }

auto main() -> int{
    auto x {456}; 
    auto y {123}; 
    std::cout << min(cpp2::move(x), cpp2::move(y)) << '\n';

    container<int> v {}; 
    std::cout << CPP2_ASSERT_IN_BOUNDS_LITERAL(cpp2::move(v), 0) << '\n';
}

