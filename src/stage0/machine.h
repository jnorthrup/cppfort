#pragma once

#include "node.h"
#include "pattern_matcher.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace cppfort::ir {

/**
 * Chapter 19: Machine Abstraction for MLIR Dialects
 *
 * Similar to Simple compiler's Machine class, but adapted for MLIR dialects
 * instead of CPU architectures. Each "machine" represents a target MLIR dialect
 * with its own instruction selection patterns.
 */
class Machine {
public:
    virtual ~Machine() = default;

    /**
     * Get the name of this machine/dialect target.
     */
    virtual ::std::string name() const = 0;

    /**
     * Get the target language this machine emits.
     */
    virtual TargetLanguage targetLanguage() const = 0;

    /**
     * Register all instruction selection patterns for this machine.
     */
    virtual void registerPatterns(PatternMatcher& matcher) = 0;

    /**
     * Check if this machine can handle the given node kind.
     */
    virtual bool canHandle(NodeKind kind) const = 0;
};

/**
 * Chapter 19: MLIR Arith Dialect Machine
 *
 * Handles arithmetic operations: add, sub, mul, div, comparisons.
 */
class MLIRArithMachine : public Machine {
public:
    ::std::string name() const override { return "mlir-arith"; }
    TargetLanguage targetLanguage() const override { return TargetLanguage::MLIR_ARITH; }

    void registerPatterns(PatternMatcher& matcher) override;
    bool canHandle(NodeKind kind) const override;
};

/**
 * Chapter 19: MLIR Control Flow Dialect Machine
 *
 * Handles control flow: branches, jumps, conditional operations.
 */
class MLIRCFMachine : public Machine {
public:
    ::std::string name() const override { return "mlir-cf"; }
    TargetLanguage targetLanguage() const override { return TargetLanguage::MLIR_CF; }

    void registerPatterns(PatternMatcher& matcher) override;
    bool canHandle(NodeKind kind) const override;
};

/**
 * Chapter 19: MLIR Structured Control Flow Dialect Machine
 *
 * Handles structured control flow: if, for, while loops.
 */
class MLIRSCFMachine : public Machine {
public:
    ::std::string name() const override { return "mlir-scf"; }
    TargetLanguage targetLanguage() const override { return TargetLanguage::MLIR_SCF; }

    void registerPatterns(PatternMatcher& matcher) override;
    bool canHandle(NodeKind kind) const override;
};

/**
 * Chapter 19: MLIR MemRef Dialect Machine
 *
 * Handles memory operations: loads, stores, allocations.
 */
class MLIRMemRefMachine : public Machine {
public:
    ::std::string name() const override { return "mlir-memref"; }
    TargetLanguage targetLanguage() const override { return TargetLanguage::MLIR_MEMREF; }

    void registerPatterns(PatternMatcher& matcher) override;
    bool canHandle(NodeKind kind) const override;
};

/**
 * Chapter 19: MLIR Func Dialect Machine
 *
 * Handles function operations: calls, returns, function definitions.
 */
class MLIRFuncMachine : public Machine {
public:
    ::std::string name() const override { return "mlir-func"; }
    TargetLanguage targetLanguage() const override { return TargetLanguage::MLIR_FUNC; }

    void registerPatterns(PatternMatcher& matcher) override;
    bool canHandle(NodeKind kind) const override;
};

/**
 * Chapter 19: Machine Registry
 *
 * Manages available machines and provides lookup by name.
 */
class MachineRegistry {
private:
    ::std::unordered_map<::std::string, ::std::unique_ptr<Machine>> _machines;

public:
    /**
     * Default constructor will register standard MLIR dialect machines
     */
    MachineRegistry();

    /**
     * Get a machine by name.
     * Returns nullptr if no machine with that name exists.
     */
    Machine* getMachine(const ::std::string& name) const;

    /**
     * Register a new machine. Replaces any existing machine with the same name.
     */
    void registerMachine(::std::unique_ptr<Machine> machine);

    /**
     * Register standard MLIR dialect machines.
     * Can be called to reset or re-initialize the machine registry.
     */
    void registerStandardMachines();

    /**
     * Get all available machine names.
     */
    ::std::vector<::std::string> availableMachines() const;

    /**
     * Check if a specific dialect machine is registered.
     */
    bool hasMachine(const ::std::string& name) const;
};

} // namespace cppfort::ir