"""
    XnatAuth

Module for XNAT authentication and 24-hour rotating Alias/Secret token lifecycle.
Implements the "verify-first" policy: checks existing cached tokens automatically
and only prompts the user if tokens are expired, missing, or invalid.
"""
module XnatAuth

using JSON
using Base64
using Downloads
using Dates

export XnatCredentials,
       get_credentials_path,
       load_cached_credentials,
       save_cached_credentials,
       clear_cached_credentials,
       is_token_valid_locally,
       get_auth_headers,
       validate_credentials,
       issue_token_from_userpass,
       auto_verify_credentials

"""
    XnatCredentials

Holds active XNAT server connection parameters and 24-hour rotating token credentials.
"""
struct XnatCredentials
    server_url::String
    alias::String
    secret::String
    issued_at::Float64
    expires_at::Float64
    username::String

    function XnatCredentials(
        server_url::AbstractString,
        alias::AbstractString,
        secret::AbstractString,
        issued_at::Real,
        expires_at::Real,
        username::AbstractString
    )
        clean_url = String(rstrip(strip(server_url), '/'))
        new(
            clean_url,
            String(strip(alias)),
            String(strip(secret)),
            Float64(issued_at),
            Float64(expires_at),
            String(strip(username))
        )
    end
end

function XnatCredentials(;
    server_url::AbstractString="",
    alias::AbstractString="",
    secret::AbstractString="",
    issued_at::Real=time(),
    expires_at::Real=time() + 86400.0,
    username::AbstractString=""
)
    return XnatCredentials(
        server_url,
        alias,
        secret,
        issued_at,
        expires_at,
        username
    )
end

"""
    get_credentials_path() -> String

Returns the absolute path to the local credentials cache file.
Respects MEDEYE3D_XNAT_CREDS environment variable if defined.
"""
function get_credentials_path()::String
    if haskey(ENV, "MEDEYE3D_XNAT_CREDS") && !isempty(ENV["MEDEYE3D_XNAT_CREDS"])
        return ENV["MEDEYE3D_XNAT_CREDS"]
    end
    if Sys.iswindows()
        appdata = get(ENV, "APPDATA", "")
        base_dir = !isempty(appdata) ? joinpath(appdata, "MedEye3D") : joinpath(homedir(), "AppData", "Roaming", "MedEye3D")
    else
        base_dir = joinpath(homedir(), ".medeye3d")
    end
    return joinpath(base_dir, "xnat_credentials.json")
end

"""
    load_cached_credentials() -> Union{XnatCredentials, Nothing}

Loads saved credentials from disk if present and parseable.
"""
function load_cached_credentials()::Union{XnatCredentials, Nothing}
    path = get_credentials_path()
    !isfile(path) && return nothing
    try
        raw = JSON.parse(read(path, String))
        server_url = get(raw, "server_url", "")
        alias = get(raw, "alias", "")
        secret = get(raw, "secret", "")
        issued_at = Float64(get(raw, "issued_at", time()))
        expires_at = Float64(get(raw, "expires_at", issued_at + 86400.0))
        username = get(raw, "username", "")

        # Check if more than 24 hours elapsed since token was first provided
        now_sec = time()
        if (now_sec - issued_at) >= 86400.0 || now_sec >= expires_at
            @info "[XNAT] 24 hours elapsed since token was first provided. Clearing cached secret and credentials."
            clear_cached_credentials()
            return nothing
        end

        if isempty(server_url) || isempty(alias) || isempty(secret)
            return nothing
        end
        return XnatCredentials(
            server_url = server_url,
            alias = alias,
            secret = secret,
            issued_at = issued_at,
            expires_at = expires_at,
            username = username
        )
    catch e
        @warn "Failed to parse cached XNAT credentials: $e"
        return nothing
    end
end

"""
    save_cached_credentials(creds::XnatCredentials) -> Bool

Saves credentials to local disk cache with restrictive permissions.
"""
function save_cached_credentials(creds::XnatCredentials)::Bool
    path = get_credentials_path()
    try
        mkpath(dirname(path))
        dict = Dict(
            "server_url" => creds.server_url,
            "alias" => creds.alias,
            "secret" => creds.secret,
            "issued_at" => creds.issued_at,
            "expires_at" => creds.expires_at,
            "username" => creds.username,
            "last_saved" => string(now())
        )
        open(path, "w") do io
            JSON.print(io, dict, 2)
        end
        if !Sys.iswindows()
            try chmod(path, 0o600) catch end
        end
        return true
    catch e
        @warn "Failed to write XNAT credentials to $path: $e"
        return false
    end
end

"""
    clear_cached_credentials() -> Bool

Deletes cached credentials file.
"""
function clear_cached_credentials()::Bool
    path = get_credentials_path()
    if isfile(path)
        try
            rm(path, force=true)
            return true
        catch; return false; end
    end
    return true
end

"""
    is_token_valid_locally(creds::XnatCredentials) -> Bool

Checks if token expiration time has not elapsed (with 15 minute safety margin)
and has not exceeded 24 hours since it was first provided.
"""
function is_token_valid_locally(creds::XnatCredentials)::Bool
    if isempty(creds.alias) || isempty(creds.secret) || isempty(creds.server_url)
        return false
    end
    now_sec = time()
    if (now_sec - creds.issued_at) >= 86400.0 || now_sec >= creds.expires_at
        return false
    end
    return now_sec < (creds.expires_at - 900.0)
end

"""
    get_auth_headers(creds::XnatCredentials) -> Vector{Pair{String, String}}

Generates HTTP Basic Auth headers using `base64(alias:secret)`.
"""
function get_auth_headers(creds::XnatCredentials)::Vector{Pair{String, String}}
    token = Base64.base64encode("$(creds.alias):$(creds.secret)")
    return [
        "Authorization" => "Basic $token",
        "Accept" => "application/json, text/plain, */*"
    ]
end

"""
    validate_credentials(creds::XnatCredentials; timeout::Float64=5.0) -> Tuple{Bool, String}

Directly pings the XNAT server to verify the token is active.
Tries `/data/JSESSION` first (accepting `text/plain, */*`), falling back to `/data/projects?format=json`.
"""
function validate_credentials(creds::XnatCredentials; timeout::Float64=5.0)::Tuple{Bool, String}
    if isempty(creds.server_url) || isempty(creds.alias) || isempty(creds.secret)
        return (false, "Missing server URL or token credentials")
    end

    auth_token = Base64.base64encode("$(creds.alias):$(creds.secret)")

    # 1. Primary check: /data/JSESSION (standard XNAT session endpoint, returns 200 + JSESSIONID)
    jsession_url = "$(creds.server_url)/data/JSESSION"
    try
        buf = IOBuffer()
        js_headers = [
            "Authorization" => "Basic $auth_token",
            "Accept" => "text/plain, */*"
        ]
        resp = Downloads.request(jsession_url; method="GET", headers=js_headers, output=buf, timeout=timeout)
        if resp.status == 200
            return (true, "Token valid")
        elseif resp.status == 401
            return (false, "Token expired or unauthorized (HTTP 401)")
        end
    catch e
    end

    # 2. Secondary check: /data/projects?format=json
    proj_url = "$(creds.server_url)/data/projects?format=json"
    try
        buf = IOBuffer()
        proj_headers = [
            "Authorization" => "Basic $auth_token",
            "Accept" => "application/json, */*"
        ]
        resp = Downloads.request(proj_url; method="GET", headers=proj_headers, output=buf, timeout=timeout)
        if resp.status == 200
            return (true, "Token valid")
        elseif resp.status == 401
            return (false, "Token expired or unauthorized (HTTP 401)")
        elseif resp.status == 403
            return (false, "Access forbidden (HTTP 403)")
        else
            return (false, "Server returned HTTP $(resp.status)")
        end
    catch e
        return (false, "Connection error: $(sprint(showerror, e))")
    end
end

"""
    issue_token_from_userpass(server_url, username, password; timeout::Float64=10.0) -> Tuple{Union{XnatCredentials, Nothing}, String}

Contacts XNAT `POST /data/services/tokens/issue` using standard credentials to request
a fresh 24-hour Alias and Secret Token. Automatically persists the resulting credentials.
"""
function issue_token_from_userpass(
    server_url::AbstractString,
    username::AbstractString,
    password::AbstractString;
    timeout::Float64=10.0
)::Tuple{Union{XnatCredentials, Nothing}, String}
    clean_url = String(rstrip(strip(server_url), '/'))
    user_str = String(strip(username))
    pass_str = String(strip(password))
    if isempty(clean_url) || isempty(user_str) || isempty(pass_str)
        return (nothing, "Server URL, username, and password are required")
    end

    issue_url = "$clean_url/data/services/tokens/issue"
    basic_token = Base64.base64encode("$user_str:$pass_str")
    headers = [
        "Authorization" => "Basic $basic_token",
        "Accept" => "application/json"
    ]

    try
        buf = IOBuffer()
        resp = Downloads.request(issue_url; method="POST", headers=headers, output=buf, timeout=timeout)
        if resp.status != 200
            return (nothing, "Failed to issue token: HTTP $(resp.status)")
        end

        body = String(take!(buf))
        data = JSON.parse(body)
        alias = get(data, "alias", "")
        secret = get(data, "secret", "")
        # estimatedExpirationTime is in milliseconds from issue
        exp_ms = get(data, "estimatedExpirationTime", 86400000.0)
        exp_sec = Float64(exp_ms) / 1000.0

        if isempty(alias) || isempty(secret)
            return (nothing, "Server did not return alias and secret tokens")
        end

        now_t = time()
        creds = XnatCredentials(
            server_url = clean_url,
            alias = String(alias),
            secret = String(secret),
            issued_at = now_t,
            expires_at = now_t + exp_sec,
            username = user_str
        )
        save_cached_credentials(creds)
        return (creds, "Token issued successfully (valid for $(round(exp_sec/3600, digits=1)) hours)")
    catch e
        return (nothing, "Error issuing token: $(sprint(showerror, e))")
    end
end

"""
    auto_verify_credentials(; timeout::Float64=5.0) -> Tuple{Bool, Union{XnatCredentials, Nothing}, String}

Executes the "verify-first" check:
1. Loads cached credentials from disk if present.
2. If present, tests them against XNAT.
3. If valid, returns `(true, creds, "Valid")`.
4. If missing or invalid, returns `(false, creds_or_nothing, reason)`.
"""
function auto_verify_credentials(; timeout::Float64=5.0)::Tuple{Bool, Union{XnatCredentials, Nothing}, String}
    creds = load_cached_credentials()
    if creds === nothing
        return (false, nothing, "No stored credentials found")
    end

    # Check local expiration first
    if !is_token_valid_locally(creds)
        clear_cached_credentials()
        return (false, nothing, "Token has expired locally (>24h since first provided)")
    end

    ok, msg = validate_credentials(creds; timeout=timeout)
    if ok
        return (true, creds, "Authenticated")
    else
        return (false, creds, msg)
    end
end

end # module XnatAuth
