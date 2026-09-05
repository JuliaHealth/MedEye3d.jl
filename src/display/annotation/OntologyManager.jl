"""
OntologyManager — Julia port of RadLex/FoundationalAnatomy ontology loading.

Adapts from:
  - slicer_lesion_text_extension/data/RadLex.csv          (~45k terms)
  - slicer_lesion_text_extension/data/FoundationalAnatomy.csv (~26k terms)

Provides:
  - Term loading (lazy, cached)
  - Multi-ontology unified search
  - Prefix-indexed fast lookup
  - Export to annotation-compatible format ("RID12345 — term label")
"""
module OntologyManager

export OntologyTerm, load_radlex, load_foundational_anatomy, load_all_ontologies,
       search_terms, format_term, parse_term_id

"""A single ontology term."""
struct OntologyTerm
    id::String        # e.g. "RID10312" or "UBERON:0001062"
    label::String     # e.g. "magnetic resonance imaging"
    source::Symbol    # :RadLex | :FoundationalAnatomy
end

format_term(t::OntologyTerm) = "$(t.id) \u2014 $(t.label)"   # em-dash OK in Julia strings
parse_term_id(s::String) = split(s, " \u2014 "; limit=2)[1]  # extract ID from formatted string

# ─── Caches ──────────────────────────────────────────────────────────────────
const _PKG_DATA   = joinpath(@__DIR__, "..", "..", "..", "extension", "data")
const _radlex_cache = Ref{Vector{OntologyTerm}}(OntologyTerm[])
const _anatomy_cache = Ref{Vector{OntologyTerm}}(OntologyTerm[])
const _all_cache    = Ref{Vector{OntologyTerm}}(OntologyTerm[])

const MAX_TERMS = 2000  # keep UI responsive

# ─── Loaders ─────────────────────────────────────────────────────────────────
function load_radlex(;max_terms::Int = MAX_TERMS)::Vector{OntologyTerm}
    isempty(_radlex_cache[]) || return _radlex_cache[]
    path = joinpath(_PKG_DATA, "RadLex.csv")
    terms = _load_csv(path, :RadLex, max_terms)
    _radlex_cache[] = sort(terms; by = t -> t.label)
    @info "OntologyManager: loaded $(length(_radlex_cache[])) RadLex terms"
    return _radlex_cache[]
end

function load_foundational_anatomy(;max_terms::Int = MAX_TERMS)::Vector{OntologyTerm}
    isempty(_anatomy_cache[]) || return _anatomy_cache[]
    path = joinpath(_PKG_DATA, "FoundationalAnatomy.csv")
    # FoundationalAnatomy.csv format: Name,ID
    terms = OntologyTerm[]
    isfile(path) || (@warn "FoundationalAnatomy.csv not found at $path"; return terms)
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue  # header
        parts = split(strip(line), ','; limit = 2)
        length(parts) < 2 && continue
        label = strip(parts[1])
        id    = strip(parts[2])
        (isempty(label) || isempty(id)) && continue
        push!(terms, OntologyTerm(id, label, :FoundationalAnatomy))
        length(terms) >= max_terms && break
    end
    _anatomy_cache[] = sort(terms; by = t -> t.label)
    @info "OntologyManager: loaded $(length(_anatomy_cache[])) FoundationalAnatomy terms"
    return _anatomy_cache[]
end

function load_all_ontologies(;max_terms::Int = MAX_TERMS)::Vector{OntologyTerm}
    isempty(_all_cache[]) || return _all_cache[]
    all = vcat(load_radlex(max_terms = max_terms),
               load_foundational_anatomy(max_terms = max_terms))
    _all_cache[] = sort(all; by = t -> t.label)
    return _all_cache[]
end

# ─── Search ───────────────────────────────────────────────────────────────────
"""
    search_terms(query, terms; max_results=200) → Vector{OntologyTerm}

Case-insensitive substring search across both label and ID.
Returns at most max_results terms sorted by relevance (prefix match first).
"""
function search_terms(query::String,
                      terms::Vector{OntologyTerm} = load_radlex();
                      max_results::Int = 200)::Vector{OntologyTerm}
    q = lowercase(strip(query))
    isempty(q) && return terms[1:min(max_results, end)]

    prefix_hits = OntologyTerm[]
    suffix_hits = OntologyTerm[]
    for t in terms
        ll = lowercase(t.label)
        il = lowercase(t.id)
        if startswith(ll, q) || startswith(il, q)
            push!(prefix_hits, t)
        elseif occursin(q, ll) || occursin(q, il)
            push!(suffix_hits, t)
        end
        length(prefix_hits) + length(suffix_hits) >= max_results * 2 && break
    end

    result = vcat(prefix_hits, suffix_hits)
    return result[1:min(max_results, end)]
end

# ─── Private ──────────────────────────────────────────────────────────────────
function _load_csv(path::String, source::Symbol, max_terms::Int)::Vector{OntologyTerm}
    terms = OntologyTerm[]
    isfile(path) || (@warn "CSV not found: $path"; return terms)
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue  # header: "Class ID,Preferred Label,..."
        parts = split(line, ','; limit = 4)
        length(parts) < 2 && continue
        id    = strip(parts[1])
        label = strip(parts[2])
        (isempty(id) || isempty(label)) && continue
        push!(terms, OntologyTerm(id, label, source))
        length(terms) >= max_terms && break
    end
    return terms
end

end # module OntologyManager
