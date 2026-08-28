"""
Translate German radiological descriptions to English via DIZ LLM API.
Uses curl subprocess — no HTTP.jl dependency required.
"""
module TranslateReports

using JSON

function _load_env(path::String)::Dict{String,String}
    d = Dict{String,String}()
    if !isfile(path)
        return d
    end
    for line in readlines(path)
        line = strip(line)
        isempty(line) && continue
        startswith(line, "#") && continue
        parts = split(line, "="; limit=2)
        if length(parts) == 2
            d[strip(parts[1])] = strip(parts[2])
        end
    end
    return d
end

function _find_and_load_env(data_dir::String, env_path::String)::Dict{String,String}
    candidates = [
        env_path,
        joinpath(data_dir, ".env"),
        "/mnt/big/project_ssd/project_ssd/slicer_lesion_text_extension/data/.env",
        joinpath(dirname(dirname(data_dir)), ".env")
    ]
    for c in candidates
        if !isempty(c) && isfile(c)
            env_map = _load_env(c)
            if haskey(env_map, "DIZ_API_KEY")
                return env_map
            end
        end
    end
    return Dict{String,String}()
end

function _call_diz_llm(text::String, api_key::String, api_base::String, model::String)::String
    url = rstrip(api_base, '/') * "/chat/completions"
    body_dict = Dict(
        "model" => model,
        "messages" => [
            Dict("role" => "system",
                 "content" => "You are an expert radiologist and medical translator."),
            Dict("role" => "user",
                 "content" => "Translate the following German radiological dictation to English. Only output the English translation and nothing else:\n\n$text")
        ]
    )
    body_json = JSON.json(body_dict)
    
    try
        cmd = `curl -s -X POST $url -H "Authorization: Bearer $api_key" -H "Content-Type: application/json" -d $body_json`
        output = read(cmd, String)
        resp = JSON.parse(output)
        choices = get(resp, "choices", [])
        if isempty(choices)
            return "Error: no choices in response: $output"
        end
        msg = get(choices[1], "message", Dict())
        return strip(get(msg, "content", "Error: no message content"))
    catch e
        return "Error: $e"
    end
end

"""
    translate_descriptions!(data_dir::String; env_path::String="")

Iterates through all studies in metadata.json within `data_dir`. For any study with a `Description`
(German) and missing/empty `EnglishDescription`, calls the DIZ LLM translation endpoint and saves
the translation back into metadata.json.
"""
function translate_descriptions!(data_dir::String; env_path::String="")
    meta_path = joinpath(data_dir, "metadata.json")
    if !isfile(meta_path)
        println("translate_descriptions!: metadata.json not found in $data_dir")
        return
    end

    config = _find_and_load_env(data_dir, env_path)
    api_key = get(config, "DIZ_API_KEY", "")
    if isempty(api_key)
        println("  No DIZ_API_KEY found, skipping translation")
        return
    end

    api_base = get(config, "DIZ_API_BASE", "https://ki-plattform.diz-ag.med.ovgu.de/api/")
    model = get(config, "DIZ_MODEL", "gpt-oss:120b")

    meta = JSON.parsefile(meta_path)
    modified = false

    for item in meta
        if !(item isa Dict)
            continue
        end
        for (date_key, info) in item
            if !(info isa Dict)
                continue
            end
            german = get(info, "Description", "")
            if isempty(german)
                continue
            end

            existing_en = get(info, "EnglishDescription", "")
            if !isempty(existing_en)
                println("  $date_key: English translation already present ($(length(existing_en)) chars)")
                continue
            end

            println("  Translating $date_key ($(length(german)) chars)...")
            english = _call_diz_llm(german, api_key, api_base, model)
            if !isempty(english) && !startswith(english, "Error")
                info["EnglishDescription"] = english
                modified = true
                println("  ✓ $date_key: translated to $(length(english)) chars")
            else
                println("  ✗ $date_key: translation failed — $english")
            end
        end
    end

    if modified
        open(meta_path, "w") do f
            JSON.print(f, meta, 2)
        end
        println("  Saved translations to metadata.json")
    end
end

export translate_descriptions!

end # module
