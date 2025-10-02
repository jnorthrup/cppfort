# Ghidra headless script: Extract CFG and basic blocks for differential tracking
# Run via: analyzeHeadless <project_dir> <project_name> -import <binary> -postScript extract_cfg.py

from ghidra.program.model.block import BasicBlockModel
from ghidra.program.model.listing import CodeUnit
from ghidra.util.task import ConsoleTaskMonitor
import json
import os

def extract_cfg():
    """
    Extract control flow graph with basic blocks from current program.

    Returns JSON structure:
    {
        "functions": [...],
        "basic_blocks": [...],
        "cfg_edges": [...],
        "patterns": [...]
    }
    """
    program = getCurrentProgram()
    if program is None:
        print("ERROR: No program loaded")
        return

    monitor = ConsoleTaskMonitor()
    func_manager = program.getFunctionManager()
    bb_model = BasicBlockModel(program)

    # Output structure
    output = {
        "binary": program.getName(),
        "architecture": program.getLanguage().getProcessor().toString(),
        "functions": [],
        "basic_blocks": [],
        "cfg_edges": [],
        "patterns": [],
        "metrics": {}
    }

    # Extract functions
    functions = func_manager.getFunctions(True)
    total_bb_count = 0
    total_inst_count = 0

    for func in functions:
        func_name = func.getName()
        func_entry = func.getEntryPoint()

        # Get basic blocks for this function
        code_blocks = bb_model.getCodeBlocksContaining(func.getBody(), monitor)
        func_bbs = []

        while code_blocks.hasNext():
            bb = code_blocks.next()
            bb_start = bb.getFirstStartAddress()
            bb_end = bb.getMaxAddress()

            # Extract instructions in this basic block
            instructions = []
            listing = program.getListing()
            inst_iter = listing.getInstructions(bb_start, True)

            while inst_iter.hasNext():
                inst = inst_iter.next()
                if inst.getAddress().compareTo(bb_end) > 0:
                    break

                instructions.append({
                    "addr": inst.getAddress().toString(),
                    "mnemonic": inst.getMnemonicString(),
                    "operands": inst.getDefaultOperandRepresentation(0) if inst.getNumOperands() > 0 else ""
                })
                total_inst_count += 1

            # Basic block info
            bb_info = {
                "id": "bb_{}".format(bb_start.toString()),
                "function": func_name,
                "start_addr": bb_start.toString(),
                "end_addr": bb_end.toString(),
                "size": bb.getNumAddresses(),
                "instruction_count": len(instructions),
                "instructions": instructions
            }

            output["basic_blocks"].append(bb_info)
            func_bbs.append(bb_info["id"])
            total_bb_count += 1

            # Extract CFG edges (successors)
            dest_iter = bb.getDestinations(monitor)
            while dest_iter.hasNext():
                dest = dest_iter.next()
                edge = {
                    "from": bb_info["id"],
                    "to": "bb_{}".format(dest.getDestinationAddress().toString()),
                    "type": dest.getFlowType().toString()
                }
                output["cfg_edges"].append(edge)

        # Function summary
        func_info = {
            "name": func_name,
            "entry": func_entry.toString(),
            "basic_blocks": func_bbs,
            "bb_count": len(func_bbs)
        }
        output["functions"].append(func_info)

    # Metrics
    output["metrics"] = {
        "total_functions": len(output["functions"]),
        "total_basic_blocks": total_bb_count,
        "total_instructions": total_inst_count,
        "avg_bb_size": total_inst_count / total_bb_count if total_bb_count > 0 else 0
    }

    # Write output
    output_path = os.path.join(
        askDirectory("Select output directory", "Choose").toString(),
        "cfg.json"
    )

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("Extracted CFG to: {}".format(output_path))
    print("Functions: {}, Basic Blocks: {}, Instructions: {}".format(
        output["metrics"]["total_functions"],
        output["metrics"]["total_basic_blocks"],
        output["metrics"]["total_instructions"]
    ))

# Run extraction
extract_cfg()
