#!/usr/bin/env bash
# sgai-lite bash completion
# Install: source this file or add to ~/.bash_completion

_sgai_completion() {
    local cur prev words cword
    _init_completion || return

    # Global flags
    local flags="-l --lang -o --output -m --model --temp -h --help
                --no-validate --formatter --dry-run --history
                --history-count --rerun --refine --input --clear-history
                --list-langs --version --verbose --open --install
                --git-commit --json"

    # Options that take an argument
    local opts_with_args="-l --lang -o --output -m --model --temp
                          --history-count --rerun --refine --input --formatter"

    # Languages
    local languages="python py javascript js typescript ts bash sh shell
                     go golang rust ruby php java c cpp csharp swift kotlin
                     scala r lua perl haskell elixir clojure dart vue svelte
                     html css sql yaml json toml dockerfile docker"

    # Formatters
    local formatters="black ruff autopep8 none"

    # Subcommands
    local subcommands="--history --rerun --refine --clear-history --list-langs"

    # Handle flags that need arguments
    if [[ "${prev}" == -* ]] && [[ " ${opts_with_args} " =~ " ${prev} " ]]; then
        case "${prev}" in
            -l|--lang)
                COMPREPLY=($(compgen -W "${languages}" -- "${cur}"))
                return
                ;;
            --formatter)
                COMPREPLY=($(compgen -W "${formatters}" -- "${cur}"))
                return
                ;;
            -m|--model)
                COMPREPLY=($(compgen -W "gpt-4o gpt-4o-mini gpt-4-turbo gpt-3.5-turbo" -- "${cur}"))
                return
                ;;
            -o|--output)
                COMPREPLY=($(compgen -f -- "${cur}"))
                return
                ;;
            --temp)
                COMPREPLY=($(compgen -W "0.0 0.3 0.5 0.7 1.0 1.5" -- "${cur}"))
                return
                ;;
            --rerun|--history-count)
                COMPREPLY=($(compgen -W "1 2 3 4 5 6 7 8 9 10" -- "${cur}"))
                return
                ;;
            --refine|--input)
                COMPREPLY=($(compgen -f -- "${cur}"))
                return
                ;;
        esac
    fi

    # Complete flags
    if [[ "${cur}" == -* ]]; then
        COMPREPLY=($(compgen -W "${flags}" -- "${cur}"))
        return
    fi

    # Complete file paths for completion that look like file args
    if [[ "${prev}" == "--input" ]] || [[ "${prev}" == "-o" ]] || [[ "${prev}" == "--output" ]]; then
        COMPREPLY=($(compgen -f -- "${cur}"))
        return
    fi

    # Default: complete with flags
    COMPREPLY=($(compgen -W "${flags}" -- "${cur}"))
}

complete -F _sgai_completion sgai
