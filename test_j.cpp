

//=== Cpp2 type declarations ====================================================


#include "cpp2util.h"

#line 1 "/tmp/test_j.cpp2"
template<typename A, typename B> class join;
#line 2 "/tmp/test_j.cpp2"
    

//=== Cpp2 type definitions and function declarations ===========================

#line 1 "/tmp/test_j.cpp2"
template<typename A, typename B> class join {
#line 2 "/tmp/test_j.cpp2"
    public: A a {}; 
    public: B b {}; 
    public: join(A&& a_, B&& b_);
public: join();

#line 4 "/tmp/test_j.cpp2"
};

operator"j": <A: type, B: type>(a: A, b: B) -> join<A, B> = {
    result: join<A, B> = ();
    result.a = a;
    result.b = b;
    return result;
}


//=== Cpp2 function definitions =================================================

#line 1 "/tmp/test_j.cpp2"


template <typename A, typename B> join<A,B>::join(A&& a_, B&& b_)
                                                                         : a{ CPP2_FORWARD(a_) }
                                                                         , b{ CPP2_FORWARD(b_) }{}
template <typename A, typename B> join<A,B>::join(){}
