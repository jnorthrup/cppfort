

//=== Cpp2 type declarations ====================================================




class crash_m0;
  

//=== Cpp2 type definitions and function declarations ===========================


[[nodiscard]] auto main() -> int;

class crash_m0 {
  public: [[nodiscard]] auto operator-([[maybe_unused]] auto const& unnamed_param_2) const& -> int;
  public: crash_m0() = default;
  public: crash_m0(crash_m0 const&) = delete; /* No 'that' constructor, suppress copy */
  public: auto operator=(crash_m0 const&) -> void = delete;


};


//=== Cpp2 function definitions =================================================


[[nodiscard]] auto main() -> int{
  auto x {crash_m0()}; 
  static_cast<void>(cpp2::move(x));
}

  [[nodiscard]] auto crash_m0::operator-([[maybe_unused]] auto const& unnamed_param_2) const& -> int { return 0;  }/* Comment starts here
And continues here
*/
