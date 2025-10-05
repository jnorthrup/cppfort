std::string  create_result(std::string resultExpr) {
  std::string result = "";
  auto get_next = [](iter) -> _   {
		start := std::distance(resultExpr&$*.cbegin(), iter);
    auto firstDollar = resultExpr&$*.find("$", start);
    auto firstAt = resultExpr&$*.find("@", start);
		auto end = std::min(firstDollar, firstAt);
		if end != std::string::npos {
			return resultExpr&$*.cbegin() + end;
		}
		else {
			return resultExpr&$*.cend();
		}
  };
  auto extract_group_and_advance = [](iter&) -> _   {
    start := iter;
    while std::isdigit(iter*) ++iter++ {}
    return std::stoi(std::string(start, iter));
  };
  auto extract_until = [](iter&, char) -> _ to   {
    start := iter;
    while (to != iter*) ++iter++ {} // TODO: Without bracket: error: postfix unary * (dereference) cannot be immediately followed by a (, identifier, or literal - add whitespace before * here if you meant binary * (multiplication)
    return std::string(start, iter);
  };
  auto iter = resultExpr.begin();
  while iter != resultExpr.end() {
    auto next = get_next(iter);
    if next != iter {
      result += std::string(iter, next);
    }
    if next != resultExpr.end() {
      if next* == '$' {
        next++;
        if next* == '&' {
          next++;
          result += r.group(0);
        }
        else if next* == '-' || next* == '+' {
          auto is_start = next* == '-';
          next++;
          if next* == '{' {
            next++; // Skip {
            auto group = extract_until(next, '}');
            next++; // Skip }
            result += r.group(group);
          }
          else if next* == '[' {
            next++; // Skip [
            auto group = extract_group_and_advance(next);
            next++; // Skip ]
            if is_start {
              result += std::to_string(r.group_start(group));
            }
            else {
              result += std::to_string(r.group_end(group));
            }
          }
          else {
            // Return max group
            result += r.group(r.group_number() - 1);
          }
        }
        else if :isdigit(next*) {
          group : std = extract_group_and_advance(next);
          result += r.group(group);
        }
        else {
          std::cerr << "Not implemented";
        }
      }
      else if next* == '@' {
        next++;
        if next* == '-'  || next* == '+' {
          auto i = 0;
          while i < :unchecked_narrow<int>(r.group_number()) ++i++ {
            pos : cpp2 = 0;
            if next* == '-' {
              pos = r.group_start(i);
            }
            else {
              pos = r.group_end(i);
            }
            result +=  std::to_string(pos);
          }
          next++;
        }
        else {
          std::cerr << "Not implemented";
        }
      }
      else {
        std::cerr << "Not implemented.";
      }
    }
    iter = next;
  }
  return result;
}

