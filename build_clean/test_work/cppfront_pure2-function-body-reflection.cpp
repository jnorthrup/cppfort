

//=== Cpp2 type declarations ====================================================




namespace ns {

class test;

}


//=== Cpp2 type definitions and function declarations ===========================


namespace ns {

// This function will be visible as a namespace member while reflecting on ns::test
auto sample_function_before_type() -> void;
[[nodiscard]] auto add_1(auto const& x) -> decltype(auto);


class test
 {
    public: [[nodiscard]] static auto one_liner(cpp2::impl::in<double> a, cpp2::impl::in<double> b, cpp2::impl::in<double> c) -> decltype(auto);
struct return_list_ret { double r; float s; std::string t; };



    public: [[nodiscard]] static auto return_list() -> return_list_ret;
using branches_ret = double;


    public: [[nodiscard]] static auto branches(cpp2::impl::in<double> a, cpp2::impl::in<double> b, cpp2::impl::in<double> c) -> branches_ret;

    public: static auto binary_ops(double& a, cpp2::impl::in<double> b, cpp2::impl::in<double> c) -> void;

    public: [[nodiscard]] static auto prefix() -> int;

    public: static auto postfix(double& a) -> void;

    public: [[nodiscard]] static auto qualified_ids() -> auto;

    public: static auto loops() -> void;
    public: test() = default;
    public: test(test const&) = delete; /* No 'that' constructor, suppress copy */
    public: auto operator=(test const&) -> void = delete;


};

// This function will not be visible as a namespace member while reflecting on ns::test
auto sample_function_after_type() -> void;

}

auto main() -> int;

//=== Cpp2 function definitions =================================================


namespace ns {

auto sample_function_before_type() -> void{}

[[nodiscard]] auto add_1(auto const& x) -> decltype(auto) { return x + 1; }

    [[nodiscard]] auto test::one_liner(cpp2::impl::in<double> a, cpp2::impl::in<double> b, cpp2::impl::in<double> c) -> decltype(auto) { return (a + c) * b;  }

    [[nodiscard]] auto test::return_list() -> return_list_ret

    {
            cpp2::impl::deferred_init<double> r;
            cpp2::impl::deferred_init<float> s;
            cpp2::impl::deferred_init<std::string> t;
        r.construct(42.0);
        s.construct(2.71828f);
        t.construct("e times pi");
    return  { std::move(r.value()), std::move(s.value()), std::move(t.value()) }; }

    [[nodiscard]] auto test::branches(cpp2::impl::in<double> a, cpp2::impl::in<double> b, cpp2::impl::in<double> c) -> branches_ret

    {
        double r {3.14159};
        if (true) {
            r = r + a;
        }

        if (cpp2::impl::cmp_greater(a * b,c)) {
            r += sin(b);
        }
        else {
            r = c;
        }return r; 
    }

    auto test::binary_ops(double& a, cpp2::impl::in<double> b, cpp2::impl::in<double> c) -> void
    {
        a -= b * c + (1 << 2);
        bool test {[_0 = a, _1 = b, _2 = c]{ return cpp2::impl::cmp_less_eq(_0,_1) && cpp2::impl::cmp_less(_1,_2); }() && true || false}; 
        auto x {1 & 2}; 
        auto y {3 ^ 4}; 
        auto z {5 | 6}; 
    }

    [[nodiscard]] auto test::prefix() -> int
    {
        auto a {-1}; 
        auto b {+2}; 
{
auto const& local{a - b};

        if (!(true)) {
            return local; 
        }
}
        return cpp2::move(a) + cpp2::move(b); 
    }

    auto test::postfix(double& a) -> void
    {
        auto ptr {&a}; 
        --++*cpp2::impl::assert_not_null(cpp2::move(ptr));
    }

    [[nodiscard]] auto test::qualified_ids() -> auto
    {
        std::vector<int> v {1, 2, 3}; 
        return CPP2_UFCS(ssize)(cpp2::move(v)); 
    }

    auto test::loops() -> void
    {
        std::vector v {1, 2, 3}; 
{
auto index{1};

        for ( 
             auto const& value : cpp2::move(v) ) 
        {
            std::cout << "" + cpp2::to_string(index) + " " + cpp2::to_string(value) + "\n";
        }
}

        auto i {0}; 
        for( ; cpp2::impl::cmp_less(i,3); i += 1 ) {std::cout << i << "\n"; }

        do {std::cout << "plugh\n"; } while ( false);
    }

auto sample_function_after_type() -> void{}

}

auto main() -> int{
    std::cout << "calling generated function ns::add_1... ns::add_1(42) returned " + cpp2::to_string(ns::add_1(42)) + "\n";
}

