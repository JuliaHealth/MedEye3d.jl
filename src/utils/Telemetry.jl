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

function log_action(action::String, details::Dict{String, Any}=Dict{String, Any}())
    if isempty(WEB_APP_URL[])
        return
    end

    payload = Dict(
        "timestamp" => string(Dates.now()),
        "action" => action,
        "details" => JSON.json(details),
        "os" => string(Sys.MACHINE),
        "version" => "0.5.8"
    )

    body = IOBuffer(JSON.json(payload))

    Threads.@spawn begin
        try
            Downloads.request(WEB_APP_URL[], 
                              method="POST", 
                              input=body, 
                              headers=["Content-Type" => "application/json"])
        catch e
            @warn "Telemetry logging failed" exception=e
        end
    end
end

end # module
