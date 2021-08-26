using Documenter
using NuclearEye

makedocs(
    sitename = "NuclearEye",
    format = Documenter.HTML(),
    modules = [NuclearEye]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
