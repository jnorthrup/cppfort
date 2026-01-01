#include "cpp2util.h"


struct my_integer {
    int v = default;
my_integer& =:(, const int& val)     {
        {
            v = val;
        }
        return *this;
    }
my_integer& operator==(const my_integer& param, const my_integer& param)     {
        return *this;
    }
my_integer& operator!=(const my_integer& param, const my_integer& param)     {
        return *this;
    }
my_integer& operator<(const my_integer& param, const my_integer& param)     {
        return *this;
    }
my_integer& operator<=(const my_integer& param, const my_integer& param)     {
        return *this;
    }
my_integer& operator>(const my_integer& param, const my_integer& param)     {
        return *this;
    }
my_integer& operator>=(const my_integer& param, const my_integer& param)     {
        return *this;
    }
    
    // @ordered metafunction: ordering operators
    auto operator<=>(const my_integer& other) const = default;
};

struct case_insensitive_string {
    std::string v = default;
case_insensitive_string& =:(, const std::string& val)     {
        {
            v = val;
        }
        return *this;
    }
    
    // @weakly_ordered metafunction: weak ordering operators
    std::weak_ordering operator<=>(const case_insensitive_string& other) const = default;
    bool operator==(const case_insensitive_string& other) const = default;
};

struct person_in_family_tree {
    int dummy_data = default;
person_in_family_tree& =:(, const int& parents)     {
        {
            dummy_data = parents;
        }
        return *this;
    }
    
    // @partially_ordered metafunction: partial ordering operators
    std::partial_ordering operator<=>(const person_in_family_tree& other) const = default;
    bool operator==(const person_in_family_tree& other) const = default;
};

struct mystruct {
    int val = 0;
};

auto main() -> void {
    my_integer a = 1;
    my_integer b = 2;
if (a < b)     {
        std::cout << "less ";
    }
 else     {
        std::cout << "more ";
    }
    case_insensitive_string c = "def";
    case_insensitive_string d = "abc";
if (c < d)     {
        std::cout << "less ";
    }
 else     {
        std::cout << "more ";
    }
    person_in_family_tree e = 20;
    person_in_family_tree f = 23;
if (e < f)     {
        std::cout << "less\n";
    }
 else     {
        std::cout << "more\n";
    }
    _;
}

