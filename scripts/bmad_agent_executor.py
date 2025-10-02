#!/usr/bin/env python3
"""
BMAD Agent Executor - Execute agent tasks via GitHub Copilot

Reads a prompt file, calls GitHub Copilot agent, outputs results.
"""
import argparse
import os
import sys
import subprocess
import json

def main():
    parser = argparse.ArgumentParser(description='Execute BMAD agent task')
    parser.add_argument('--agent', required=True, help='Agent type (dev, architect, qa, etc.)')
    parser.add_argument('--prompt', required=True, help='Path to prompt file')
    parser.add_argument('--output', required=True, help='Path to output file')
    args = parser.parse_args()

    # Read prompt
    with open(args.prompt, 'r') as f:
        prompt = f.read()

    # Get GitHub token
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        print("ERROR: GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    # Agent system prompts
    agent_prompts = {
        'dev': "You are a senior software engineer implementing features. Focus on correctness, testing, and clean code.",
        'architect': "You are a system architect designing solutions. Focus on patterns, scalability, and technical decisions.",
        'qa': "You are a QA engineer. Focus on test coverage, edge cases, and quality validation.",
        'analyst': "You are a technical analyst. Focus on performance, metrics, and optimization.",
        'sm': "You are a scrum master. Focus on process, workflow, and team coordination.",
    }

    system_prompt = agent_prompts.get(args.agent, "You are a helpful assistant.")

    print(f"Executing {args.agent} agent via GitHub Copilot...", file=sys.stderr)

    try:
        # Call GitHub Copilot Chat API
        # https://docs.github.com/en/rest/copilot/copilot-chat
        full_prompt = f"{system_prompt}\n\n{prompt}"

        # Use gh CLI to call Copilot API
        result = subprocess.run(
            ['gh', 'copilot', 'suggest', '--prompt', full_prompt],
            capture_output=True,
            text=True,
            check=True
        )

        response_text = result.stdout

        # Write output
        with open(args.output, 'w') as f:
            f.write(response_text)

        print(f"Agent execution complete. Output: {args.output}", file=sys.stderr)
        sys.exit(0)

    except subprocess.CalledProcessError as e:
        print(f"ERROR: GitHub Copilot execution failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Agent execution failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
