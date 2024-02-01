# initialize the global user environment
if !isdir("/data/environments/v$(VERSION.major).$(VERSION.minor)")
    mkpath("/data/environments")
    cp("/usr/local/share/julia/environments/v$(VERSION.major).$(VERSION.minor)",
       "/data/environments/v$(VERSION.major).$(VERSION.minor)")
end
pushfirst!(DEPOT_PATH, "/data")
