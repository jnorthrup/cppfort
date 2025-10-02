#!/usr/bin/env python3
"""
BMAD Agent Executor - Execute agent tasks via Anthropic API

Reads a prompt file, calls the appropriate agent, outputs results.
"""
import argparse
import os
import sys
import anthropic

def main():
    parser = argparse.ArgumentParser(description='Execute BMAD agent task')
    parser.add_argument('--agent', required=True, help='Agent type (dev, architect, qa, etc.)')
    parser.add_argument('--prompt', required=True, help='Path to prompt file')
    parser.add_argument('--output', required=True, help='Path to output file')
    args = parser.parse_args()

    # Read prompt
    with open(args.prompt, 'r') as f:
        prompt = f.read()

    # Get API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Initialize client
    client = anthropic.Anthropic(api_key=api_key)

    # Agent system prompts
    agent_prompts = {
        'dev': "You are a senior software engineer implementing features. Focus on correctness, testing, and clean code.",
        'architect': "You are a system architect designing solutions. Focus on patterns, scalability, and technical decisions.",
        'qa': "You are a QA engineer. Focus on test coverage, edge cases, and quality validation.",
        'analyst': "You are a technical analyst. Focus on performance, metrics, and optimization.",
        'sm': "You are a scrum master. Focus on process, workflow, and team coordination.",
    }

    system_prompt = agent_prompts.get(args.agent, "You are a helpful assistant.")

    print(f"Executing {args.agent} agent...", file=sys.stderr)

    try:
        # Call Claude API
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Extract response
        response_text = message.content[0].text

        # Write output
        with open(args.output, 'w') as f:
            f.write(response_text)

        print(f"Agent execution complete. Output: {args.output}", file=sys.stderr)
        sys.exit(0)

    except Exception as e:
        print(f"ERROR: Agent execution failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
