content = read("src/display/LesionMetadataWindow.jl", String)

# Replace const definitions with function definitions
content = replace(content, "const DEF_JSON_PATH       = _find_metadata_data_file(\"def.json\")" => "def_json_path() = _find_metadata_data_file(\"def.json\")")
content = replace(content, "const RADLEX_CSV_PATH     = _find_metadata_data_file(\"RadLex.csv\")" => "radlex_csv_path() = _find_metadata_data_file(\"RadLex.csv\")")
content = replace(content, "const ANATOMY_CSV_PATH    = _find_metadata_data_file(\"FoundationalAnatomy.csv\")" => "anatomy_csv_path() = _find_metadata_data_file(\"FoundationalAnatomy.csv\")")
content = replace(content, "const CUSTOM_OPTS_PATH    = _find_metadata_data_file(\"custom_options.json\")" => "custom_opts_path() = _find_metadata_data_file(\"custom_options.json\")")
content = replace(content, "const ANATOMY_MAPPING_PATH= _find_metadata_data_file(\"max_anatomy_to_ontology.json\")" => "anatomy_mapping_path() = _find_metadata_data_file(\"max_anatomy_to_ontology.json\")")
content = replace(content, "const GLOBAL_CUSTOM_OPTS_PATH = joinpath(homedir(), \".medeye3d_custom_options.json\")" => "global_custom_opts_path() = joinpath(homedir(), \".medeye3d_custom_options.json\")")

# Replace usages
content = replace(content, "DEF_JSON_PATH" => "def_json_path()")
content = replace(content, "RADLEX_CSV_PATH" => "radlex_csv_path()")
content = replace(content, "ANATOMY_CSV_PATH" => "anatomy_csv_path()")
content = replace(content, "CUSTOM_OPTS_PATH" => "custom_opts_path()")
content = replace(content, "ANATOMY_MAPPING_PATH" => "anatomy_mapping_path()")
content = replace(content, "GLOBAL_CUSTOM_OPTS_PATH" => "global_custom_opts_path()")

write("src/display/LesionMetadataWindow.jl", content)
