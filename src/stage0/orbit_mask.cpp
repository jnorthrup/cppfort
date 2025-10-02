#include "orbit_mask.h"

namespace cppfort::ir {

bool OrbitContext::processMatch(const OrbitMatch& match) {
    // Update internal match list
    _matches.push_back(match);

    // Speculatively update depths based on signature characters
    // Simplified heuristic: inspect pattern signature for delimiter tokens
    for (char c : match.signature) {
        switch (c) {
            case '{': _braceDepth++; break;
            case '}': _braceDepth = (::std::max)(0, _braceDepth - 1); break;
            case '[': _bracketDepth++; break;
            case ']': _bracketDepth = (::std::max)(0, _bracketDepth - 1); break;
            case '<': _angleDepth++; break;
            case '>': _angleDepth = (::std::max)(0, _angleDepth - 1); break;
            case '(': _parenDepth++; break;
            case ')': _parenDepth = (::std::max)(0, _parenDepth - 1); break;
            case '"': _quoteDepth = (_quoteDepth == 0) ? 1 : 0; break;
            default: break;
        }
    }

    // Keep depths within reasonable bounds
    if (static_cast<size_t>(getDepth()) > _maxDepth) return false;

    return true;
}

bool OrbitContext::isBalanced() const {
    return _braceDepth == 0 && _bracketDepth == 0 && _angleDepth == 0 &&
           _parenDepth == 0 && _quoteDepth == 0;
}

int OrbitContext::depth(OrbitType type) const {
    switch (type) {
        case OrbitType::OpenBrace:
        case OrbitType::CloseBrace: return _braceDepth;
        case OrbitType::OpenBracket:
        case OrbitType::CloseBracket: return _bracketDepth;
        case OrbitType::OpenAngle:
        case OrbitType::CloseAngle: return _angleDepth;
        case OrbitType::OpenParen:
        case OrbitType::CloseParen: return _parenDepth;
        case OrbitType::Quote: return _quoteDepth;
        case OrbitType::NumberStart:
        case OrbitType::NumberEnd: return _numberDepth;
        default: return 0;
    }
}

int OrbitContext::getDepth() const {
    return _braceDepth + _bracketDepth + _angleDepth + _parenDepth + _quoteDepth + _numberDepth;
}

size_t OrbitContext::getMaxDepth() const {
    return _maxDepth;
}

double OrbitContext::calculateConfidence() const {
    // Heuristic: confidence decreases with imbalance magnitude
    int totalDepth = getDepth();
    if (totalDepth == 0) return 1.0;

    // Sum absolute deviations from zero for all tracked depths
    int imbalance = ::std::abs(_braceDepth) + ::std::abs(_bracketDepth) + ::std::abs(_angleDepth) +
                    ::std::abs(_parenDepth) + ::std::abs(_quoteDepth) + ::std::abs(_numberDepth);

    double penalty = static_cast<double>(imbalance) / static_cast<double>(::std::max(1, totalDepth));
    double conf = 1.0 - ::std::min(1.0, penalty);
    return conf;
}

const ::std::vector<OrbitMatch>& OrbitContext::matches() const {
    return _matches;
}

void OrbitContext::reset() {
    _braceDepth = _bracketDepth = _angleDepth = _parenDepth = _quoteDepth = _numberDepth = 0;
    _matches.clear();
}

bool OrbitContext::wouldBeValid(const OrbitMatch& match) const {
    // Speculative check without mutating state: simple heuristic
    int brace = _braceDepth;
    int bracket = _bracketDepth;
    int angle = _angleDepth;
    int paren = _parenDepth;
    int quote = _quoteDepth;

    for (char c : match.signature) {
        switch (c) {
            case '{': brace++; break;
            case '}': brace = (::std::max)(0, brace - 1); break;
            case '[': bracket++; break;
            case ']': bracket = (::std::max)(0, bracket - 1); break;
            case '<': angle++; break;
            case '>': angle = (::std::max)(0, angle - 1); break;
            case '(' : paren++; break;
            case ')' : paren = (::std::max)(0, paren - 1); break;
            case '"': quote = (quote == 0) ? 1 : 0; break;
            default: break;
        }
    }

    int tot = brace + bracket + angle + paren + quote + _numberDepth;
    return static_cast<size_t>(tot) <= _maxDepth;
}

void OrbitContext::update(char ch) {
    switch (ch) {
        case '{': _braceDepth++; break;
        case '}': _braceDepth = (::std::max)(0, _braceDepth - 1); break;
        case '[': _bracketDepth++; break;
        case ']': _bracketDepth = (::std::max)(0, _bracketDepth - 1); break;
        case '<': _angleDepth++; break;
        case '>': _angleDepth = (::std::max)(0, _angleDepth - 1); break;
        case '(': _parenDepth++; break;
        case ')': _parenDepth = (::std::max)(0, _parenDepth - 1); break;
        case '"': _quoteDepth = (_quoteDepth == 0) ? 1 : 0; break;
        // For numbers, increment on any digit
        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
            _numberDepth++;
            break;
        default: break;
    }
}

::std::array<size_t, 6> OrbitContext::getCounts() const {
    return {
        static_cast<size_t>(::std::max(0, _braceDepth)),
        static_cast<size_t>(::std::max(0, _bracketDepth)),
        static_cast<size_t>(::std::max(0, _angleDepth)),
        static_cast<size_t>(::std::max(0, _parenDepth)),
        static_cast<size_t>(::std::max(0, _quoteDepth)),
        static_cast<size_t>(::std::max(0, _numberDepth))
    };
}

} // namespace cppfort::ir
