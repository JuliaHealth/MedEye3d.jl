module Telemetry

using Downloads
using JSON
using Dates
using Sockets

# Setup the spreadsheet URL. Note: this needs a Google Apps Script Web App URL to actually accept POST requests.
# But for now, we will log to the provided spreadsheet link or an expected web app URL.
const SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1fH0b4v9K62jJpNXnnTlowQWF54qSQCt4aK0cX19dCs0/edit?usp=sharing"

# This should be replaced by the user with their Google Apps Script Web App URL
const WEB_APP_URL = Ref{String}("")

function set_web_app_url!(url::String)
    WEB_APP_URL[] = url
end

function get_telemetry_url()::String
    if !isempty(WEB_APP_URL[])
        return WEB_APP_URL[]
    end
    return get(ENV, "MEDEYE3D_TELEMETRY_URL", "")
end

function log_action(action::AbstractString, details=Dict{String, Any}())
    try
        url = get_telemetry_url()
        if isempty(url)
            return
        end

        payload = Dict(
            "timestamp" => string(Dates.now()),
            "action" => string(action),
            "details" => JSON.json(details),
            "os" => string(Sys.MACHINE),
            "version" => "0.5.9"
        )

        body_str = JSON.json(payload)

        Threads.@spawn begin
            try
                Downloads.request(url, 
                                  method="POST", 
                                  input=IOBuffer(body_str), 
                                  headers=["Content-Type" => "application/json"])
            catch e
                @warn "Telemetry HTTP POST failed" exception=e
            end
        end
    catch e
        @warn "Telemetry logging failed" exception=e
    end
end

end # module
