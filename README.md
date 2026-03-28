# sgai-lite

**Goal-driven single-file code generator powered by AI.**

Give sgai a goal in plain English — it generates a complete, working code file for you.

## Install

```bash
pip install sgai-lite
```

Requires Python 3.8+ and an [OpenAI API key](https://platform.openai.com/api-keys).

```bash
export OPENAI_API_KEY=sk-your-key-here
```

## Usage

```bash
# Generate a Python script
sgai "a Python script that fetches weather for a city from wttr.in and prints it"

# Specify language
sgai --lang ts "a TypeScript function that validates email addresses"

# Custom output file
sgai --output server.py "a Python HTTP server that handles GET /hello"

# Use a different model
sgai --model gpt-4o-mini "a rust program that prints fibonacci numbers"

# Dry run (show detected language without generating)
sgai --dry-run "a rust program that prints fibonacci numbers"

# Open generated file automatically
sgai --open "a simple HTML page with a gradient background"

# Auto-install detected dependencies
sgai --install "a Python script that downloads images from URLs"

# Commit to git automatically
sgai --git-commit "a Python CLI tool for managing todos"

# Verbose mode (model info, dependencies, timing)
sgai --verbose "a Python script that parses CSV files and makes charts"

# Machine-readable JSON output with token usage and cost
sgai --json "a Python function that reverses a string"
```

## Options

| Flag | Description |
|------|-------------|
| `goal` | Natural language description of what to build (required) |
| `-l, --lang` | Target language (auto-detected if not specified) |
| `-o, --output` | Output file path (auto-generated if not specified) |
| `-m, --model` | OpenAI model (default: `gpt-4o`) |
| `--temp` | Sampling temperature 0.0–2.0 (default: 0.3) |
| `--dry-run` | Show detected language without generating |
| `--formatter` | Auto-format code (black, ruff, autopep8) |
| `--no-validate` | Skip syntax validation |
| `--verbose` | Show model, language, dependencies, timing |
| `--open` | Open generated file in default application |
| `--install` | Auto-install detected Python dependencies |
| `--git-commit` | Commit generated file to git |
| `--json` | Machine-readable JSON output with metadata, tokens, and cost |
| `--history` | Show generation history |
| `--rerun N` | Regenerate from history entry N |
| `--refine` | Refine an existing file with instructions |
| `--list-langs` | List all supported languages |
| `--version` | Show version |

## Configuration

Copy `config.example.yaml` to `~/.sgai-lite/config.yaml` for persistent settings:

```yaml
default_model: gpt-4o
temperature: 0.3
validate: true
# formatter: black
```

See `config.example.yaml` for all available options.

## Architecture

```
sgai-lite/
├── cli.py             # CLI parsing, user interaction, streaming display
├── generator.py       # OpenAI API, streaming, validation, retry logic
├── history.py         # JSONL-based generation history (~/.sgai-lite/)
├── config.py          # Config file loading (JSON/YAML)
├── languages.py       # Language detection, extension mapping
├── prompts.py         # Intent detection, language-specific tips
└── completions/       # Shell completions (bash, zsh)

# Flow:
#   CLI → detect intent → build prompt → OpenAI streaming → validate → save → history
```

## How It Works

1. **Intent Detection** — Scans your goal for keywords (cli, web, data, gui, etc.) to select language-specific best-practice tips
2. **Streaming Generation** — Sends goal to OpenAI with a crafted system prompt; code streams token-by-token to your terminal
3. **Validation** — Python uses `compile()`, Bash uses `bash -n`, Go uses `gofmt`, Rust uses `rustfmt`, Ruby uses `ruby -c`, PHP uses `php -l`, Lua uses `lua -p`. Falls back gracefully if tools aren't installed.
4. **Retry Logic** — Transient API errors (rate limits, timeouts) automatically retry with exponential backoff (up to 3 attempts)
5. **History** — Every generation is saved to `~/.sgai-lite/history.jsonl` with metadata

## Common Use Cases

```bash
# Data processing
sgai "a Python script that reads a CSV file, filters rows, and outputs a bar chart"

# Web servers
sgai --lang py "a FastAPI server with GET /health and POST /tasks endpoints"

# CLI tools
sgai --lang py "a CLI tool that recursively searches for files by name"

# Automation scripts
sgai --lang bash "a script that backs up a MySQL database and uploads it to S3"

# API integrations
sgai --install "a Python script that fetches GitHub repos and displays their stars"
```

## Supported Languages

Python, JavaScript, TypeScript, Bash/Shell, Go, Rust, Ruby, PHP, Java, C, C++, C#, Swift, Kotlin, Scala, R, Lua, Perl, Haskell, Elixir, Clojure, Dart, Vue, Svelte, HTML, CSS, SQL, YAML, JSON, TOML, Dockerfile, and more.

## Shell Completions

Install bash or zsh completions for a better experience:

**Bash:**
```bash
# Add to ~/.bashrc
source /path/to/sgai-lite/completions/sgai.bash
```

**Zsh:**
```bash
# Add to ~/.zshrc
source /path/to/sgai-lite/completions/sgai.zsh
# Or copy to your fpath
```

## License

MIT
