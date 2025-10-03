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
    _inNumber = false;
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
    // Check if character is a digit
    bool isDigit = (ch >= '0' && ch <= '9');

    // Handle numeric literal span tracking
    if (isDigit) {
        if (!_inNumber) {
            // Start of new numeric literal
            _numberDepth++;
            _inNumber = true;
        }
        // Continue in same numeric literal, no depth change
    } else {
        if (_inNumber) {
            // End of numeric literal
            _numberDepth = (::std::max)(0, _numberDepth - 1);
            _inNumber = false;
        }
    }

    // Handle delimiter tracking
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
 
uint8_t OrbitContext::confixMask() const {
    uint8_t mask = 0;
    // TopLevel: no open delimiters
    if (getDepth() == 0) mask |= (1 << 0);
    if (depth(OrbitType::OpenBrace) > 0)   mask |= (1 << 1);
    if (depth(OrbitType::OpenParen) > 0)   mask |= (1 << 2);
    if (depth(OrbitType::OpenAngle) > 0)   mask |= (1 << 3);
    if (depth(OrbitType::OpenBracket) > 0) mask |= (1 << 4);
    if (depth(OrbitType::Quote) > 0)       mask |= (1 << 5);
    return mask;
}

} // namespace cppfort::ir
