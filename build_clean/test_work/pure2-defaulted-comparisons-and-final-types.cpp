

//=== Cpp2 type declarations ====================================================




class widget;


//=== Cpp2 type definitions and function declarations ===========================


class widget final
 {
    private: int v; 

    public: widget(cpp2::impl::in<int> value);
    public: auto operator=(cpp2::impl::in<int> value) -> widget& ;

    public: [[nodiscard]] auto operator==(widget const& that) const& -> bool = default;

    public: [[nodiscard]] auto operator<=>(widget const& that) const& -> std::strong_ordering = default;
    public: widget(widget const&) = delete; /* No 'that' constructor, suppress copy */
    public: auto operator=(widget const&) -> void = delete;

};

auto main() -> int;

//=== Cpp2 function definitions =================================================


    widget::widget(cpp2::impl::in<int> value)
                                          : v{ value }{}
    auto widget::operator=(cpp2::impl::in<int> value) -> widget& {
                                          v = value;
                                          return *this; }

auto main() -> int{
    widget a {1}; 
    widget b {2}; 
    if (cpp2::impl::cmp_less(cpp2::move(a),cpp2::move(b))) {
        std::cout << "less";
    }
    else {
        std::cout << "more";
    }
}

