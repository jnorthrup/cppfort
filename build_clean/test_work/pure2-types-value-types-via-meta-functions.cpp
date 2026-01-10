

//=== Cpp2 type declarations ====================================================




class widget;
    

class w_widget;
    

class p_widget;
    

//=== Cpp2 type definitions and function declarations ===========================


class widget {
    private: int val {0}; 
    public: widget(cpp2::impl::in<int> i);
    public: auto operator=(cpp2::impl::in<int> i) -> widget& ;
    public: [[nodiscard]] auto operator<=>(widget const& that) const& -> std::strong_ordering = default;
public: widget(widget const& that);

public: auto operator=(widget const& that) -> widget& ;
public: widget(widget&& that) noexcept;
public: auto operator=(widget&& that) noexcept -> widget& ;
public: explicit widget();

};

class w_widget {
    private: int val {0}; 
    public: w_widget(cpp2::impl::in<int> i);
    public: auto operator=(cpp2::impl::in<int> i) -> w_widget& ;
    public: [[nodiscard]] auto operator<=>(w_widget const& that) const& -> std::weak_ordering = default;
public: w_widget(w_widget const& that);

public: auto operator=(w_widget const& that) -> w_widget& ;
public: w_widget(w_widget&& that) noexcept;
public: auto operator=(w_widget&& that) noexcept -> w_widget& ;
public: explicit w_widget();

};

class p_widget {
    private: int val {0}; 
    public: p_widget(cpp2::impl::in<int> i);
    public: auto operator=(cpp2::impl::in<int> i) -> p_widget& ;
    public: [[nodiscard]] auto operator<=>(p_widget const& that) const& -> std::partial_ordering = default;
public: p_widget(p_widget const& that);

public: auto operator=(p_widget const& that) -> p_widget& ;
public: p_widget(p_widget&& that) noexcept;
public: auto operator=(p_widget&& that) noexcept -> p_widget& ;
public: explicit p_widget();

};

auto main() -> int;

template<typename T> auto test() -> void;

//=== Cpp2 function definitions =================================================


    widget::widget(cpp2::impl::in<int> i)
                                      : val{ i }{}
    auto widget::operator=(cpp2::impl::in<int> i) -> widget& {
                                      val = i;
                                      return *this; }


    widget::widget(widget const& that)
                                : val{ that.val }{}

auto widget::operator=(widget const& that) -> widget& {
                                val = that.val;
                                return *this;}
widget::widget(widget&& that) noexcept
                                : val{ std::move(that).val }{}
auto widget::operator=(widget&& that) noexcept -> widget& {
                                val = std::move(that).val;
                                return *this;}
widget::widget(){}
    w_widget::w_widget(cpp2::impl::in<int> i)
                                      : val{ i }{}
    auto w_widget::operator=(cpp2::impl::in<int> i) -> w_widget& {
                                      val = i;
                                      return *this; }


    w_widget::w_widget(w_widget const& that)
                                : val{ that.val }{}

auto w_widget::operator=(w_widget const& that) -> w_widget& {
                                val = that.val;
                                return *this;}
w_widget::w_widget(w_widget&& that) noexcept
                                : val{ std::move(that).val }{}
auto w_widget::operator=(w_widget&& that) noexcept -> w_widget& {
                                val = std::move(that).val;
                                return *this;}
w_widget::w_widget(){}
    p_widget::p_widget(cpp2::impl::in<int> i)
                                      : val{ i }{}
    auto p_widget::operator=(cpp2::impl::in<int> i) -> p_widget& {
                                      val = i;
                                      return *this; }


    p_widget::p_widget(p_widget const& that)
                                : val{ that.val }{}

auto p_widget::operator=(p_widget const& that) -> p_widget& {
                                val = that.val;
                                return *this;}
p_widget::p_widget(p_widget&& that) noexcept
                                : val{ std::move(that).val }{}
auto p_widget::operator=(p_widget&& that) noexcept -> p_widget& {
                                val = std::move(that).val;
                                return *this;}
p_widget::p_widget(){}
auto main() -> int{
    test<widget>();
    test<w_widget>();
    test<p_widget>();
}

template<typename T> auto test() -> void{
    //  should be default constructible
    T a {}; 

    //  widget should be comparable
    T b {2}; 
    if ((cpp2::impl::cmp_less(cpp2::move(a),cpp2::move(b)))) {
        std::cout << "less ";
    }
    else {
        std::cout << "more ";
    }
}

