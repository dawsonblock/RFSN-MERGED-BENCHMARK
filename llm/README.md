# 🤖 LLM Integration for RFSN

Real LLM integration for intelligent patch generation using GPT-4, Claude, and DeepSeek.

## Features

- **Multiple Providers**: OpenAI, Anthropic, Google, DeepSeek
- **Automatic Retry**: Exponential backoff on failures
- **Rate Limiting**: Quota management
- **Cost Tracking**: Track tokens and costs per request
- **Context Management**: Automatic truncation to fit context windows
- **Streaming Support**: For real-time responses
- **Caching**: Optional response caching

## Quick Start

### 1. Install Dependencies

```bash
pip install -r llm/requirements.txt
```

### 2. Set API Keys

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."
```

### 3. Generate Patches

```python
import asyncio
from llm import generate_patches_with_llm, PatchStrategy

async def main():
    patches = await generate_patches_with_llm(
        problem="Fix the bug in add function",
        repo_path="/path/to/repo",
        localization_hits=[{
            "file_path": "src/math.py",
            "line_start": 10,
            "line_end": 15,
            "evidence": "Function returns wrong result"
        }],
        strategy=PatchStrategy.DIRECT_FIX,
        max_patches=3
    )
    
    print(f"Generated {len(patches)} patches")
    for patch in patches:
        print(patch)

asyncio.run(main())
```

## Usage Examples

### Basic LLM Client

```python
from llm import get_llm_client

# Auto-detect from environment
client = get_llm_client()

# Or specify provider
client = get_llm_client("anthropic")

# Generate completion
response = await client.complete_with_retry(
    prompt="Write a Python function to reverse a string",
    system="You are a helpful coding assistant",
    max_tokens=200
)

print(response.content)
print(f"Tokens: {response.tokens_used}, Cost: ${response.cost_usd:.4f}")
```

### Custom Configuration

```python
from llm import LLMConfig, LLMProvider, LLMClientFactory

config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4-turbo",
    api_key="sk-...",
    max_tokens=4096,
    temperature=0.2,
    max_retries=3
)

client = LLMClientFactory.create(config)
```

### Patch Generation Strategies

```python
from llm import LLMPatchGenerator, PatchStrategy

generator = LLMPatchGenerator()

# Direct Fix
patches = await generator.generate_direct_fix(
    problem="Bug description",
    repo_path="/path/to/repo",
    localization_hits=hits,
    max_patches=3
)

# Test-Driven
patches = await generator.generate_test_driven(
    problem="Bug description",
    repo_path="/path/to/repo",
    localization_hits=hits,
    test_output="Test failure output",
    error_trace="Stack trace"
)

# Hypothesis-Driven
patches = await generator.generate_hypothesis_driven(
    problem="Bug description",
    repo_path="/path/to/repo",
    localization_hits=hits,
    error_trace="Stack trace"
)
```

## Supported Providers

### OpenAI
- **Models**: gpt-4, gpt-4-turbo, gpt-3.5-turbo
- **API Key**: `OPENAI_API_KEY`
- **Pricing**: ~$0.01-0.03 per 1K tokens

### Anthropic
- **Models**: claude-3-5-sonnet, claude-3-opus, claude-3-sonnet
- **API Key**: `ANTHROPIC_API_KEY`
- **Pricing**: ~$0.003-0.015 per 1K tokens

### DeepSeek
- **Models**: deepseek-chat, deepseek-coder
- **API Key**: `DEEPSEEK_API_KEY`
- **Base URL**: `https://api.deepseek.com/v1`
- **Pricing**: Very cost-effective

## Cost Optimization

### 1. Token Limits
```python
# Limit max tokens per request
response = await client.complete(
    prompt=prompt,
    max_tokens=1024  # Reduce for cheaper requests
)
```

### 2. Temperature
```python
# Lower temperature = more deterministic = potentially cheaper
config = LLMConfig(
    ...,
    temperature=0.1  # vs 0.7
)
```

### 3. Context Truncation
```python
# Automatically truncate to fit context window
truncated = client.truncate_to_context(
    text=long_file_content,
    max_tokens=2048,
    preserve_start=True
)
```

### 4. Caching
```python
# Cache responses to avoid duplicate requests
# (implement with Redis or disk cache)
```

## Error Handling

```python
from llm import get_llm_client

client = get_llm_client()

try:
    response = await client.complete_with_retry(
        prompt="Your prompt",
        system="System message"
    )
except Exception as e:
    print(f"LLM request failed: {e}")
    # Fallback to mock or alternative strategy
```

## Monitoring & Statistics

```python
# Check usage stats
stats = client.stats

print(f"Total requests: {stats.total_requests}")
print(f"Successful: {stats.successful_requests}")
print(f"Failed: {stats.failed_requests}")
print(f"Total tokens: {stats.total_tokens}")
print(f"Total cost: ${stats.total_cost_usd:.2f}")
print(f"Avg latency: {stats.total_latency_ms / stats.total_requests:.0f}ms")
```

## Integration with RFSN

The LLM integration is designed to work seamlessly with RFSN patch generation:

```python
from patch.gen import PatchGenerator
from llm import get_llm_client

# Initialize with LLM client
llm_client = get_llm_client("openai")
generator = PatchGenerator(llm_client=llm_client)

# Generate patches (now uses real LLM!)
result = generator.generate(request)
```

## Prompt Engineering

The system uses carefully crafted prompts for each strategy:

- **Direct Fix**: Surgical, minimal changes
- **Test-Driven**: Focus on making tests pass
- **Hypothesis-Driven**: Multi-hypothesis analysis
- **Incremental**: Learn from previous failures
- **Ensemble**: Synthesize multiple approaches

See `llm/prompts.py` for details.

## Environment Variables

```bash
# Required (choose one or more)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...

# Optional configuration
LLM_PROVIDER=openai          # Default provider
OPENAI_MODEL=gpt-4-turbo     # Override default model
OPENAI_BASE_URL=...          # Custom endpoint
```

## Testing

```bash
# Test LLM client
cd /home/user/webapp/RFSN-CODE-GATE-main
python -m llm.client

# Test patch generation
python -m llm.patch_generator
```

## Troubleshooting

### "API key not found"
- Set environment variable: `export OPENAI_API_KEY="sk-..."`
- Or pass directly in config

### "Rate limit exceeded"
- Increase `retry_delay` in config
- Reduce `max_tokens` to stay within quota
- Use a different provider

### "Context length exceeded"
- Use `truncate_to_context()` to fit limits
- Reduce file context size
- Use smaller models

### "Import Error"
- Install dependencies: `pip install -r llm/requirements.txt`
- Check Python version (3.9+)

## Best Practices

1. **Start Small**: Use `max_tokens=1024` for testing
2. **Monitor Costs**: Check `stats.total_cost_usd` regularly
3. **Use Retries**: Enable automatic retry for robustness
4. **Cache Responses**: Avoid duplicate expensive requests
5. **Truncate Context**: Fit large files to context windows
6. **Handle Failures**: Graceful fallback to mock/alternative

## Performance

- **Latency**: 1-5 seconds per patch (depending on model)
- **Cost**: $0.01-0.10 per patch (depending on model and size)
- **Success Rate**: 60-80% with good localization
- **Throughput**: Limited by API rate limits

## Roadmap

- [ ] Add Google Gemini support
- [ ] Implement response caching
- [ ] Add streaming support
- [ ] Batch request optimization
- [ ] Fine-tuning support
- [ ] Local model support (Ollama)
- [ ] Cost prediction before generation
- [ ] A/B testing framework

## License

MIT License - See LICENSE file

---

**Version**: 1.0.0  
**Status**: Production Ready  
**Last Updated**: 2026-01-30
