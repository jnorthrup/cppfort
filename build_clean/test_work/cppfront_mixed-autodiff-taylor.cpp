#include <cpp2taylor.h>


//=== Cpp2 type declarations ====================================================





//=== Cpp2 type definitions and function declarations ===========================


int inline constexpr order{ 6 };
using taylor = cpp2::taylor<double,order>;

struct test_add_ret { double y0; taylor y; };

[[nodiscard]] auto test_add(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_add_ret;
struct test_sub_ret { double y0; taylor y; };



[[nodiscard]] auto test_sub(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_sub_ret;
struct test_mul_ret { double y0; taylor y; };



[[nodiscard]] auto test_mul(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_mul_ret;
struct test_div_ret { double y0; taylor y; };



[[nodiscard]] auto test_div(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_div_ret;
struct test_sqrt_ret { double y0; taylor y; };



[[nodiscard]] auto test_sqrt(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_sqrt_ret;
struct test_log_ret { double y0; taylor y; };



[[nodiscard]] auto test_log(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_log_ret;
struct test_exp_ret { double y0; taylor y; };



[[nodiscard]] auto test_exp(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_exp_ret;
struct test_sin_ret { double y0; taylor y; };



[[nodiscard]] auto test_sin(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_sin_ret;
struct test_cos_ret { double y0; taylor y; };



[[nodiscard]] auto test_cos(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_cos_ret;

auto write_output(cpp2::impl::in<std::string> func, cpp2::impl::in<double> x, cpp2::impl::in<taylor> x_d, auto const& ret) -> void;

auto main() -> int;

//=== Cpp2 function definitions =================================================


[[nodiscard]] auto test_add(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_add_ret{
      cpp2::impl::deferred_init<double> y0;
      cpp2::impl::deferred_init<taylor> y;
  y.construct(CPP2_UFCS(add)(x, x, x0, x0));
  y0.construct(x0 + x0);
return  { std::move(y0.value()), std::move(y.value()) }; }

[[nodiscard]] auto test_sub(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_sub_ret{
      cpp2::impl::deferred_init<double> y0;
      cpp2::impl::deferred_init<taylor> y;
  y0.construct(0.0);
  y.construct(taylor());

  y.value() = CPP2_UFCS(sub)(y.value(), x, y0.value(), x0);
  y0.value() = y0.value() - x0;
return  { std::move(y0.value()), std::move(y.value()) }; }

[[nodiscard]] auto test_mul(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_mul_ret{
      cpp2::impl::deferred_init<double> y0;
      cpp2::impl::deferred_init<taylor> y;
  y0.construct(x0);
  y.construct(x);
{
auto i{0};

  for( ; cpp2::impl::cmp_less(i,6); i += 1 ) {
    y.value() = y.value().mul(x, y0.value(), x0);
    y0.value() *= x0;
  }
}
    return  { std::move(y0.value()), std::move(y.value()) }; 

}

[[nodiscard]] auto test_div(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_div_ret{
      cpp2::impl::deferred_init<double> y0;
      cpp2::impl::deferred_init<taylor> y;
  y0.construct(1.0);
  y.construct(taylor());

  y.value() = CPP2_UFCS(div)(y.value(), x, y0.value(), x0);
  y0.value() /= CPP2_ASSERT_NOT_ZERO(CPP2_TYPEOF(y0.value()),x0);
return  { std::move(y0.value()), std::move(y.value()) }; }

[[nodiscard]] auto test_sqrt(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_sqrt_ret{
      cpp2::impl::deferred_init<double> y0;
      cpp2::impl::deferred_init<taylor> y;
  y0.construct(sqrt(x0));
  y.construct(CPP2_UFCS(sqrt)(x, x0));
return  { std::move(y0.value()), std::move(y.value()) }; }

[[nodiscard]] auto test_log(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_log_ret{
      cpp2::impl::deferred_init<double> y0;
      cpp2::impl::deferred_init<taylor> y;
  y0.construct(log(x0));
  y.construct(CPP2_UFCS(log)(x, x0));
return  { std::move(y0.value()), std::move(y.value()) }; }

[[nodiscard]] auto test_exp(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_exp_ret{
      cpp2::impl::deferred_init<double> y0;
      cpp2::impl::deferred_init<taylor> y;
  y0.construct(exp(x0));
  y.construct(CPP2_UFCS(exp)(x, x0));
return  { std::move(y0.value()), std::move(y.value()) }; }

[[nodiscard]] auto test_sin(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_sin_ret{
      cpp2::impl::deferred_init<double> y0;
      cpp2::impl::deferred_init<taylor> y;
  y0.construct(sin(x0));
  y.construct(CPP2_UFCS(sin)(x, x0));
return  { std::move(y0.value()), std::move(y.value()) }; }

[[nodiscard]] auto test_cos(cpp2::impl::in<double> x0, cpp2::impl::in<taylor> x) -> test_cos_ret{
      cpp2::impl::deferred_init<double> y0;
      cpp2::impl::deferred_init<taylor> y;
  y0.construct(cos(x0));
  y.construct(CPP2_UFCS(cos)(x, x0));
return  { std::move(y0.value()), std::move(y.value()) }; }

auto write_output(cpp2::impl::in<std::string> func, cpp2::impl::in<double> x, cpp2::impl::in<taylor> x_d, auto const& ret) -> void{
    static_cast<void>(x);
    static_cast<void>(x_d);
    std::cout << "" + cpp2::to_string(func) + " = " + cpp2::to_string(ret.y0) + "" << std::endl;
{
auto i{1};

    for( ; cpp2::impl::cmp_less_eq(i,order); i += 1 ) {
       std::cout << "" + cpp2::to_string(func) + " diff order " + cpp2::to_string(i) + " = " + cpp2::to_string(CPP2_ASSERT_IN_BOUNDS(ret.y, i)) + "" << std::endl;
    }
}
}

auto main() -> int{

    double x {2.0}; 
    taylor x_d {1.0}; 

    write_output("x + x", x, x_d, test_add(x, x_d));
    write_output("0 - x", x, x_d, test_sub(x, x_d));
    write_output("x^7", x, x_d, test_mul(x, x_d));
    write_output("1/x", x, x_d, test_div(x, x_d));
    write_output("sqrt(x)", x, x_d, test_sqrt(x, x_d));
    write_output("log(x)", x, x_d, test_log(x, x_d));
    write_output("exp(x)", x, x_d, test_exp(x, x_d));
    write_output("sin(x)", x, x_d, test_sin(x, x_d));
    write_output("cos(x)", x, x_d, test_cos(cpp2::move(x), cpp2::move(x_d)));
}

