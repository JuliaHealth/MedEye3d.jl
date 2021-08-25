using Documenter
using Julia Med 3d

makedocs(
    sitename = "Julia Med 3d",
    format = Documenter.HTML(),
    modules = [Julia Med 3d]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
