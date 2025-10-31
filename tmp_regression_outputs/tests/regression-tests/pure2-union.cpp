#include <iostream>
#include <string>
struct name_or_number {
    using storage_t = std::variant<std::monostate, std::string, int32_t>;
    storage_t data{};

    bool empty() const { return std::holds_alternative<std::monostate>(data); }
    bool is_name() const { return std::holds_alternative<std::string>(data); }
    const std::string& name() const { return std::get<std::string>(data); }
    std::string& name() { return std::get<std::string>(data); }
    template <typename... Args> void set_name(Args&&... args) { data.emplace<std::string>(std::forward<Args>(args)...); }

    bool is_num() const { return std::holds_alternative<int32_t>(data); }
    const int32_t& num() const { return std::get<int32_t>(data); }
    int32_t& num() { return std::get<int32_t>(data); }
    template <typename... Args> void set_num(Args&&... args) { data.emplace<int32_t>(std::forward<Args>(args)...); }

};

template<typename T>
struct name_or_other {
    using storage_t = std::variant<std::monostate, std::string, T>;
    storage_t data{};

    bool empty() const { return std::holds_alternative<std::monostate>(data); }
    bool is_name() const { return std::holds_alternative<std::string>(data); }
    const std::string& name() const { return std::get<std::string>(data); }
    std::string& name() { return std::get<std::string>(data); }
    template <typename... Args> void set_name(Args&&... args) { data.emplace<std::string>(std::forward<Args>(args)...); }

    bool is_other() const { return std::holds_alternative<T>(data); }
    const T& other() const { return std::get<T>(data); }
    T& other() { return std::get<T>(data); }
    template <typename... Args> void set_other(Args&&... args) { data.emplace<T>(std::forward<Args>(args)...); }

    std::string to_string() const { if (is_name())       { return name(); }
            else if (is_other()) { return other() as std::string; }
            else               { return "invalid value"; } }

};


void print_name(const name_or_number& non) { if non.is_name() {
        std::cout << non.name() << "\n";
    }
    else {
        std::cout << "(not a name)\n";
    } }

int main() { name_or_number x = ();
    :cout << "sizeof(x) - alignof(x) std = = max(sizeof(fields))";
              << " is (sizeof(x) - alignof(name_or_number) == std::max(sizeof(i32), sizeof(std::string)))$\n";

    x.print_name();

    x.set_name( "xyzzy", 3 as u8 );

    x.print_name();

    { name_or_other<int> val = ();
        val.set_other(42);
        std::cout << val.to_string(); } }
