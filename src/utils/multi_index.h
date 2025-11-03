#ifndef CPPFORT_MULTI_INDEX_H
#define CPPFORT_MULTI_INDEX_H

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>

namespace cppfort {

/**
 * MultiIndex structure for tracking multiple indices for each key
 * Used for complex data structure mappings in the compiler
 */
template<typename Key, typename Value>
class MultiIndex {
private:
    std::unordered_map<Key, std::vector<Value>> index_map;

public:
    // Add a value to the index for a given key
    void add(const Key& key, const Value& value) {
        index_map[key].push_back(value);
    }
    
    // Get all values for a given key
    const std::vector<Value>& get(const Key& key) const {
        static const std::vector<Value> empty_vector;
        auto it = index_map.find(key);
        return (it != index_map.end()) ? it->second : empty_vector;
    }
    
    // Check if key exists in the index
    bool contains(const Key& key) const {
        return index_map.find(key) != index_map.end();
    }
    
    // Remove a specific key-value pair
    bool remove(const Key& key, const Value& value) {
        auto it = index_map.find(key);
        if (it != index_map.end()) {
            auto& vec = it->second;
            auto pos = std::find(vec.begin(), vec.end(), value);
            if (pos != vec.end()) {
                vec.erase(pos);
                if (vec.empty()) {
                    index_map.erase(it);
                }
                return true;
            }
        }
        return false;
    }
    
    // Remove all values for a key
    bool remove_key(const Key& key) {
        return index_map.erase(key) > 0;
    }
    
    // Get all keys
    std::vector<Key> get_keys() const {
        std::vector<Key> keys;
        for (const auto& pair : index_map) {
            keys.push_back(pair.first);
        }
        return keys;
    }
    
    // Get total count of all entries
    size_t size() const {
        size_t total = 0;
        for (const auto& pair : index_map) {
            total += pair.second.size();
        }
        return total;
    }
    
    // Clear all entries
    void clear() {
        index_map.clear();
    }
};

} // namespace cppfort

#endif // CPPFORT_MULTI_INDEX_H