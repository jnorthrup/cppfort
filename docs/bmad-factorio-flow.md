# BMAD Factorio Flow

Automated pipeline for continuous development via LLM agents.

## Flow Overview

```
TODO.md → Extract → Stories → Dispatch → Agent Execute → Test → PR
   ↓         ↓         ↓          ↓            ↓          ↓     ↓
  Push    APEX     docs/     Executor    Copilot    Build  Auto-merge
         Items    stories/      API        GPT-4     Valid   or Retry
```

## Trigger Mechanisms

### 1. Push to TODO.md (Auto-Fanout)

**Workflow**: `.github/workflows/bmad-fanout.yml`

**Trigger**: Any commit modifying `TODO.md`

**Process**:
1. Extract APEX items marked with `<!-- bmad:apex=true;swimlane=X -->`
2. Generate story files using template at `.bmad-core/templates/apex-story-tmpl.yaml`
3. Commit generated stories to `docs/stories/`
4. Dispatch executor workflow for each story

**Example TODO.md entry**:
```markdown
- [ ] Implement real Sea-of-Nodes IR <!-- bmad:apex=true;swimlane=ir -->
```

**Output**:
- Story file: `docs/stories/implement-real-sea-of-nodes-ir.story.md`
- Automatic dispatch to executor
- Manifest CSV at `.bmad-core/data/apex_manifest.csv`

### 2. Issue with 'apex' Label

**Workflow**: `.github/workflows/bmad-executor.yml`

**Trigger**: Issue opened or labeled with `apex`

**Process**:
1. Extract story path from issue body
2. Route to agent based on swimlane (ir→architect, stage0→dev, etc.)
3. Execute via GitHub Copilot API
4. Auto-retry up to 3 times on failure

### 3. Manual Workflow Dispatch

**Direct execution**:
```bash
gh workflow run bmad-executor.yml -f story_path=docs/stories/my-story.story.md
```

## Agent Routing

**File**: `.github/workflows/bmad-executor.yml` (lines 46-75)

| Swimlane | Agent Type | Specialization |
|----------|-----------|----------------|
| ir, patterns, backend | architect | System design |
| stage0, emitter, parser | dev | Implementation |
| orbit, scanner | architect | Structural design |
| testing, validation | qa | Quality assurance |
| optimization, performance | analyst | Performance tuning |
| build, ci | sm | Process/workflow |

## Executor Pipeline

**Workflow**: `.github/workflows/bmad-executor.yml`

### Step 1: Story Context Loading
- Extract title, acceptance criteria, tasks from story file
- Build agent prompt with full context

### Step 2: Agent Execution
- **Script**: `scripts/bmad_agent_executor.py`
- **API**: GitHub Copilot Chat API (gpt-4o)
- **Mode**: `--mode api` (falls back to local if GITHUB_TOKEN missing)
- **Auth**: Uses `GITHUB_TOKEN` secret

### Step 3: Validation
- Build project with CMake + Ninja
- Run regression tests (`regression-tests/run_regression_stage0.sh`)
- Extract pass rate

### Step 4: PR Creation
- Create feature branch `bmad/auto-impl-{timestamp}`
- Commit with detailed message
- Open PR with agent type, swimlane, test results

### Step 5: Auto-Retry (on failure)
- Max 3 retries (labeled `bmad-retry-1`, `bmad-retry-2`, `bmad-retry-3`)
- Re-dispatch executor with same story
- After 3 failures: label `bmad-manual-required`

## Template System

**Template**: `.bmad-core/templates/apex-story-tmpl.yaml`

**Variables**:
- `{{title}}` - Story title from TODO item
- `{{swimlane}}` - Extracted swimlane (ir, stage0, patterns, etc.)
- `{{slug}}` - URL-safe filename slug

**Generated Story Structure**:
- Status, Story (user story format)
- Context Source (links to TODO.md)
- Acceptance Criteria (functional, integration, quality)
- Dev Technical Guidance (files, constraints, approach)
- Tasks / Subtasks (4-step breakdown)
- Testing (unit, integration, manual)
- Definition of Done (checklist)

## Factorio Characteristics

### Continuous Flow
- Push to TODO.md → immediate extraction
- Story generated → immediate dispatch
- Tests fail → auto-retry (up to 3x)

### Parallel Processing
- Multiple stories dispatch simultaneously
- Each runs in isolated executor job
- No blocking on previous story completion

### Quality Gates
- Build must succeed
- Tests must pass
- Auto-retry on failure
- Manual intervention only after 3 failures

### Feedback Loop
- Test results in PR description
- Retry count visible via labels
- Execution logs in artifacts
- Pass rate tracked per attempt

## Configuration

### Required Secrets
- `GITHUB_TOKEN` - Already available in GitHub Actions
- Optional: `ANTHROPIC_API_KEY` - If using Claude instead of Copilot

### Story Template
Edit `.bmad-core/templates/apex-story-tmpl.yaml` to customize:
- Acceptance criteria structure
- Dev guidance format
- Task breakdown
- Testing requirements

### Agent Routing
Edit `.github/workflows/bmad-executor.yml` lines 46-75 to change swimlane→agent mapping

## Usage Examples

### Add New APEX Task
```markdown
# In TODO.md
- [ ] Add constant folding optimization pass <!-- bmad:apex=true;swimlane=patterns -->
```

Push to GitHub → story auto-generated → executor dispatched → PR created

### Monitor Progress
```bash
# Check running workflows
gh run list --workflow=bmad-executor.yml

# View specific run
gh run view <run-id>

# Check generated stories
ls docs/stories/*.story.md
```

### Manual Trigger
```bash
# Dispatch specific story
gh workflow run bmad-executor.yml \
  -f story_path=docs/stories/add-constant-folding-optimization-pass.story.md

# View workflow runs
gh run list
```

## Metrics

Track via manifest CSV:
```bash
cat .bmad-core/data/apex_manifest.csv
```

Fields:
- `task_id` - Unique hash
- `swimlane` - Task category
- `slug` - Filename-safe identifier
- `story_path` - Full path to story file
- `issue` - GitHub issue number (if created)
- `labels` - Applied labels

## Troubleshooting

### Story Not Dispatching
- Check TODO.md has correct `<!-- bmad:apex=true;swimlane=X -->` annotation
- Verify swimlane name is valid (ir, stage0, patterns, etc.)
- Check `.github/workflows/bmad-fanout.yml` ran successfully

### Executor Failing
- Check `GITHUB_TOKEN` secret exists
- Verify story file exists at specified path
- Review execution logs in workflow run
- Check if retry count exceeded (3 max)

### Tests Not Running
- Verify CMake dependencies installed (cmake, ninja, g++)
- Check regression test script exists and is executable
- Review build logs in workflow artifacts

## Architecture Notes

This is **not** a stub system. The executor:
1. Calls **real** GitHub Copilot API (gpt-4o)
2. Makes **actual** code changes in the repository
3. Runs **real** build and test commands
4. Creates **real** PRs with working code

The flow is designed to be **continuous** (like Factorio):
- No human intervention between stages
- Auto-retry on transient failures
- Parallel execution across stories
- Quality gates prevent broken code from merging

The system is **self-improving**:
- Failed attempts inform retries
- Test results guide next iteration
- Manifest tracks success rate over time
