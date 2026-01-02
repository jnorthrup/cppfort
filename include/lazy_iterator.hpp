#ifndef CPP2_LAZY_ITERATOR_HPP
#define CPP2_LAZY_ITERATOR_HPP

#include <vector>
#include <iterator>

namespace cpp2 {

template<typename I, typename F>
struct MapIter {
    I it;
    I end_it;
    F f;

    MapIter(I i, I e, F func) : it(i), end_it(e), f(func) {}

    auto operator*() const { return f(*it); }
    MapIter& operator++() { ++it; return *this; }
    bool operator!=(const MapIter& other) const { return it != other.it; }
    bool operator==(const MapIter& other) const { return it == other.it; }
};

template<typename I, typename P>
struct FilterIter {
    I it;
    I end_it;
    P pred;

    FilterIter(I i, I e, P p) : it(i), end_it(e), pred(p) {
        find_next();
    }

    void find_next() {
        while (it != end_it && !pred(*it)) {
            ++it;
        }
    }

    auto operator*() const { return *it; }
    FilterIter& operator++() {
        if (it != end_it) {
            ++it;
            find_next();
        }
        return *this;
    }
    bool operator!=(const FilterIter& other) const { return it != other.it; }
    bool operator==(const FilterIter& other) const { return it == other.it; }
};

template<typename I>
struct LazyRange {
    I m_begin;
    I m_end;

    LazyRange(I b, I e) : m_begin(b), m_end(e) {}

    auto begin() const { return m_begin; }
    auto end() const { return m_end; }

    template<typename F>
    auto map(F f) const {
        using M = MapIter<I, F>;
        return LazyRange<M>(M(m_begin, m_end, f), M(m_end, m_end, f));
    }

    template<typename P>
    auto filter(P p) const {
        using Fi = FilterIter<I, P>;
        return LazyRange<Fi>(Fi(m_begin, m_end, p), Fi(m_end, m_end, p));
    }

    template<typename Container>
    Container collect(Container c = Container()) const {
        for (auto it = begin(); it != end(); ++it) {
            c.push_back(*it);
        }
        return c;
    }
};

template<typename Container>
auto iter(const Container& c) {
    return LazyRange(c.begin(), c.end());
}

} // namespace cpp2

#endif // CPP2_LAZY_ITERATOR_HPP
