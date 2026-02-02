# E2B Use Cases for RFSN Controller

## What is E2B?

[E2B](https://e2b.dev/) (Environment to Browser) is an open-source infrastructure that provides **secure, isolated cloud sandboxes** powered by Firecracker microVMs. Unlike Docker containers, E2B sandboxes run in lightweight virtual machines with hardware-level isolation.

---

## When to Use E2B with RFSN Controller

### ✅ Use E2B When

| Use Case | Why E2B Helps |
|----------|---------------|
| **Running untrusted code** | Firecracker microVMs provide stronger isolation than Docker containers |
| **No Docker available** | Works on machines without Docker installed |
| **Multi-tenant systems** | Each user gets a completely isolated VM |
| **Production deployments** | Cloud-hosted = no local resource constraints |
| **Scaling to many concurrent runs** | E2B handles orchestration and resources |
| **Long-running or persistent sandboxes** | Pause/resume capabilities save state |

### ❌ Stick with Docker When

| Use Case | Why Docker is Fine |
|----------|-------------------|
| **Local development** | Lower latency, no API calls |
| **Trusted codebases** | Don't need VM-level isolation |
| **Offline work** | Docker works without internet |
| **Cost-sensitive** | E2B has usage-based pricing |

---

## Use Case Deep Dives

### 1. AI Agent Code Execution

**Scenario**: You're building an AI agent that writes and executes code based on natural language prompts.

**Risk**: The AI might generate malicious or buggy code that could:

- Delete files on your system
- Access sensitive data
- Consume infinite resources
- Make unauthorized network requests

**E2B Solution**:

```python
from rfsn_controller.e2b_sandbox import E2BSandboxAdapter

adapter = E2BSandboxAdapter()
sandbox_id = adapter.create_sandbox()

# AI-generated code runs in isolated microVM
result = adapter.run_command(sandbox_id, ai_generated_code)

# Sandbox destroyed - nothing persists
adapter.destroy_sandbox(sandbox_id)
```

**Why E2B**: Even if the AI generates `rm -rf /`, it only affects the ephemeral VM, not your system.

---

### 2. Continuous Integration for Untrusted PRs

**Scenario**: You accept pull requests from external contributors on an open-source project.

**Risk**: A malicious PR could:

- Steal CI secrets
- Compromise your build infrastructure
- Mine cryptocurrency during builds

**E2B Solution**:

```python
# Each PR gets its own isolated VM
sandbox = adapter.create_sandbox(
    metadata={"pr_number": "123", "author": "external-user"}
)

# Run tests in isolation
adapter.run_command(sandbox, "git clone <pr-branch>")
adapter.run_command(sandbox, "pytest")

# VM destroyed - no persistence
adapter.destroy_sandbox(sandbox)
```

**Why E2B**: The test environment is completely isolated from your CI infrastructure.

---

### 3. Multi-Tenant SaaS Code Execution

**Scenario**: You're building a SaaS platform where users can write and run code (like Replit, CodeSandbox, or a Jupyter notebook service).

**Risk**: Users could:

- Access other users' data
- Break out of their container
- Attack your infrastructure

**E2B Solution**:

```python
# Each user session gets dedicated resources
user_sandbox = adapter.create_sandbox(
    config=E2BSandboxConfig(
        cpu_count=2,
        memory_mb=4096,
        metadata={"user_id": "user_123"}
    )
)

# Users can do anything in their VM safely
adapter.run_command(user_sandbox, user_code)
```

**Why E2B**: Hardware-level isolation means one user cannot affect another, even with kernel exploits.

---

### 4. Security Research and Malware Analysis

**Scenario**: You need to analyze potentially malicious code samples.

**Risk**: Running malware on your system could:

- Infect your machine
- Spread to your network
- Destroy data

**E2B Solution**:

```python
# Isolated environment for malware analysis
analysis_sandbox = adapter.create_sandbox()

# Upload malware sample
adapter.write_file(analysis_sandbox, "/sample/malware.bin", malware_bytes)

# Run analysis tools
result = adapter.run_command(analysis_sandbox, "strings /sample/malware.bin")
result = adapter.run_command(analysis_sandbox, "file /sample/malware.bin")

# Safe - nothing can escape the microVM
adapter.destroy_sandbox(analysis_sandbox)
```

**Why E2B**: Firecracker microVMs are specifically designed for this - AWS uses them for Lambda.

---

### 5. Educational Platforms

**Scenario**: You're building a coding education platform where students submit and run code.

**Risk**: Students (accidentally or intentionally) could:

- Write infinite loops
- Fork bomb the server
- Access instructor solutions

**E2B Solution**:

```python
# Each submission gets isolated execution
submission_sandbox = adapter.create_sandbox(
    config=E2BSandboxConfig(
        cpu_count=1,
        memory_mb=512,  # Limited resources
        timeout_ms=30000  # 30 second timeout
    )
)

# Run student code with resource limits
result = adapter.run_command(submission_sandbox, f"python student_solution.py")

# Grade based on output
grade = check_output(result.stdout, expected_output)
```

**Why E2B**: Resource limits + isolation = safe execution of arbitrary student code.

---

## E2B vs Docker Comparison

| Feature | Docker | E2B (Firecracker) |
|---------|--------|-------------------|
| **Isolation Level** | Process/namespace | Hardware (microVM) |
| **Startup Time** | ~100ms | ~150ms |
| **Memory Overhead** | ~10MB | ~5MB |
| **Escape Difficulty** | Possible (kernel exploits) | Extremely difficult |
| **Multi-tenancy** | Risky | Designed for it |
| **Network Isolation** | Configurable | Default isolated |
| **Persistence** | Volumes | Snapshots/pause |
| **Cost** | Free (local) | Usage-based |

---

## Getting Started with E2B

### 1. Sign Up

Go to [e2b.dev](https://e2b.dev) and create an account.

### 2. Get API Key

Navigate to Dashboard → API Keys → Create Key.

### 3. Set Environment Variable

```bash
export E2B_API_KEY=e2b_your_api_key_here
```

### 4. Install SDK

```bash
pip install e2b-code-interpreter
```

### 5. Use with RFSN Controller

```python
from rfsn_controller.e2b_sandbox import E2BSandboxAdapter, is_e2b_available

if is_e2b_available():
    adapter = E2BSandboxAdapter()
    sandbox_id = adapter.create_sandbox()
    
    # Upload your repo
    adapter.upload_directory(sandbox_id, "./my-project", "/home/user/project")
    
    # Run tests
    result = adapter.run_command(sandbox_id, "cd /home/user/project && pytest")
    
    print(result.stdout)
    adapter.destroy_sandbox(sandbox_id)
else:
    print("E2B not configured, using local Docker instead")
```

---

## Pricing Considerations

E2B uses usage-based pricing:

- **Compute time**: Charged per second of VM runtime
- **Storage**: Charged for persistent snapshots
- **Free tier**: Available for experimentation

For most development and testing, the free tier is sufficient. Production workloads with high volume should factor in the per-second costs.

---

## Summary

| Question | Answer |
|----------|--------|
| Do I need E2B? | No, Docker works fine for most cases |
| When should I use E2B? | Untrusted code, multi-tenant, production |
| Is E2B free? | Free tier available, production is paid |
| Is it hard to set up? | Easy - just set API key and install SDK |
| Can I switch back to Docker? | Yes, the adapter is completely optional |
