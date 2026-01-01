#include "cpp2util.h"

auto test() -> void;

struct widget {
    int val = 0;
widget& =:(, const int& i)     {
        {
            val = i;
        }
        return *this;
    }
auto widget(void& other) -> void    ;
widget& operator=(const widget& param, const const widget&& param)     {
        return *this;
    }
auto widget(void& other) -> void    ;
widget& operator=(const widget& param, const widget&&& param)     {
        return *this;
    }
    
    // @value metafunction: value semantics
    widget(const widget&) = default;
    widget(widget&&) = default;
    widget& operator=(const widget&) = default;
    widget& operator=(widget&&) = default;
    
    bool operator==(const widget& other) const = default;
    bool operator!=(const widget& other) const = default;
};

struct w_widget {
    int val = 0;
w_widget& =:(, const int& i)     {
        {
            val = i;
        }
        return *this;
    }
    
    // @weakly_ordered metafunction: weak ordering operators
    std::weak_ordering operator<=>(const w_widget& other) const = default;
    bool operator==(const w_widget& other) const = default;
};

struct p_widget {
    int val = 0;
p_widget& =:(, const int& i)     {
        {
            val = i;
        }
        return *this;
    }
    
    // @partially_ordered metafunction: partial ordering operators
    std::partial_ordering operator<=>(const p_widget& other) const = default;
    bool operator==(const p_widget& other) const = default;
};

auto main() -> void {
    test();
    test();
    test();
}

template<typename T>
auto test() -> void {
    T a = {};
    T b = 2;
if (a < b)     {
        std::cout << "less ";
    }
 else     {
        std::cout << "more ";
    }
}

