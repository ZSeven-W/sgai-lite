#compdef sgai
# sgai-lite zsh completion

_sgai() {
    local -a commands flags lang_opts model_opts formatter_opts

    commands=(
        '--history:Show generation history'
        '--rerun:Regenerate from history entry'
        '--refine:Refine an existing file'
        '--clear-history:Clear all history'
        '--list-langs:List supported languages'
    )

    flags=(
        '-l[Target language]: :->lang'
        '--lang[Target language]: :->lang'
        '-o[Output file path]: :_files'
        '--output[Output file path]: :_files'
        '-m[OpenAI model]: :->model'
        '--model[OpenAI model]: :->model'
        '--temp[Sampling temperature (0.0-2.0)]'
        '--no-validate[Skip syntax validation]'
        '--formatter[Auto-format code]:(black ruff autopep8 none)'
        '--dry-run[Show detected language and exit]'
        '--history[Show generation history]'
        '--history-count[Number of history entries]: :->number'
        '--rerun[Regenerate from history entry]: :->number'
        '--refine[Refinement instruction]:'
        '--input[Input file for refine mode]: :_files'
        '--clear-history[Clear all history]'
        '--list-langs[List supported languages]'
        '--version[Show version number]'
        '--verbose[Show model and token info]'
        '--open[Open generated file]'
        '--install[Auto-install dependencies]'
        '--git-commit[Commit to git]'
        '--json[Machine-readable JSON output]'
    )

    local languages=(python py javascript js typescript ts bash sh shell go golang
                     rust ruby php java c cpp csharp swift kotlin scala r lua perl
                     haskell elixir clojure dart vue svelte html css sql yaml json
                     toml dockerfile docker)
    local models=(gpt-4o gpt-4o-mini gpt-4-turbo gpt-3.5-turbo)

    case "$state" in
        lang)
            _describe 'language' languages
            ;;
        model)
            _describe 'model' models
            ;;
        number)
            _describe 'number' '(1 2 3 4 5 6 7 8 9 10)'
            ;;
    esac

    _describe 'options' flags
    _describe 'commands' commands

    _arguments -s $flags $commands '1:goal:_SgaiGoal' && return

    if [[ ${words[CURRENT-1]} != -* ]]; then
        _SgaiGoal() {
            _message 'goal description'
        }
    fi
}

_sgai "$@"
