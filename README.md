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
sgai --model gpt-4o-mini "a bash script to find files larger than 100MB"

# Dry run (show detected language without generating)
sgai --dry-run "a rust program that prints fibonacci numbers"
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
| `--list-langs` | List all supported languages |
| `--version` | Show version |

## Supported Languages

Python, JavaScript, TypeScript, Bash/Shell, Go, Rust, Ruby, PHP, Java, C, C++, C#, Swift, Kotlin, Scala, R, Lua, Perl, Haskell, Elixir, Clojure, Dart, Vue, Svelte, HTML, CSS, SQL, YAML, JSON, TOML, Dockerfile, and more.

## How It Works

sgai-lite sends your goal to OpenAI's API with a carefully crafted system prompt that instructs the model to generate complete, production-ready code in a single file. The output streams to both the terminal and the output file in real time.

## License

MIT
