

//=== Cpp2 type declarations ====================================================




class Human;
    

namespace N {
    template<int I> class Machine;
        

}

class Cyborg;


//=== Cpp2 type definitions and function declarations ===========================


class Human {
    public: virtual auto speak() const -> void = 0;
    public: explicit Human();
protected: Human([[maybe_unused]] Human const& that);

protected: auto operator=([[maybe_unused]] Human const& that) -> Human& ;
protected: Human([[maybe_unused]] Human&& that) noexcept;
protected: auto operator=([[maybe_unused]] Human&& that) noexcept -> Human& ;
public: virtual ~Human() noexcept;

};

namespace N {
    template<int I> class Machine {
        public: Machine([[maybe_unused]] cpp2::impl::in<std::string> unnamed_param_2);
        public: virtual auto work() const -> void = 0;
        public: virtual ~Machine() noexcept;

        public: Machine(Machine const&) = delete; /* No 'that' constructor, suppress copy */
        public: auto operator=(Machine const&) -> void = delete;

    };
}

struct Cyborg_name_as_base { std::string name; };
struct Cyborg_address_as_base { std::string address; };
class Cyborg: public Cyborg_name_as_base, public Human, public Cyborg_address_as_base, public N::Machine<99> {

    public: Cyborg(cpp2::impl::in<std::string> n);

    public: auto speak() const -> void override;

    public: auto work() const -> void override;

    public: auto print() const& -> void;

    public: ~Cyborg() noexcept;
    public: Cyborg(Cyborg const&) = delete; /* No 'that' constructor, suppress copy */
    public: auto operator=(Cyborg const&) -> void = delete;


};

auto make_speak(cpp2::impl::in<Human> h) -> void;

auto do_work(cpp2::impl::in<N::Machine<99>> m) -> void;

auto main() -> int;

//=== Cpp2 function definitions =================================================



Human::Human(){}
Human::Human([[maybe_unused]] Human const& that){}
auto Human::operator=([[maybe_unused]] Human const& that) -> Human& {
                                          return *this;}
Human::Human([[maybe_unused]] Human&& that) noexcept{}
auto Human::operator=([[maybe_unused]] Human&& that) noexcept -> Human& {
                                          return *this;}
Human::~Human() noexcept{}
namespace N {

        template <int I> Machine<I>::Machine([[maybe_unused]] cpp2::impl::in<std::string> unnamed_param_2){}

        template <int I> Machine<I>::~Machine() noexcept{}

}

    Cyborg::Cyborg(cpp2::impl::in<std::string> n)
        : Cyborg_name_as_base{ n }
        , Human{  }
        , Cyborg_address_as_base{ "123 Main St." }
        , N::Machine<99>{ "Acme Corp. engineer tech" }{

        std::cout << "" + cpp2::to_string(name) + " checks in for the day's shift\n";
    }

    auto Cyborg::speak() const -> void{
        std::cout << "" + cpp2::to_string(name) + " cracks a few jokes with a coworker\n";
    }

    auto Cyborg::work() const -> void{
        std::cout << "" + cpp2::to_string(name) + " carries some half-tonne crates of Fe2O3 to cold storage\n";
    }

    auto Cyborg::print() const& -> void{
        std::cout << "printing: " + cpp2::to_string(name) + " lives at " + cpp2::to_string(address) + "\n";
    }

    Cyborg::~Cyborg() noexcept { 
        std::cout << "Tired but satisfied after another successful day, " + cpp2::to_string(cpp2::move(*this).name) + " checks out and goes home to their family\n";  }

auto make_speak(cpp2::impl::in<Human> h) -> void{
    std::cout << "-> [vcall: make_speak] ";
    CPP2_UFCS(speak)(h);
}

auto do_work(cpp2::impl::in<N::Machine<99>> m) -> void{
    std::cout << "-> [vcall: do_work] ";
    CPP2_UFCS(work)(m);
}

auto main() -> int{
    Cyborg c {"Parsnip"}; 
    CPP2_UFCS(print)(c);
    CPP2_UFCS(make_speak)(c);
    CPP2_UFCS(do_work)(cpp2::move(c));
}

