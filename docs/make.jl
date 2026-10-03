using ViewStructArrays
using Documenter

DocMeta.setdocmeta!(ViewStructArrays, :DocTestSetup, :(using ViewStructArrays); recursive = true)

makedocs(;
    modules = [ViewStructArrays],
    authors = "Paul Tiede <ptiede91@gmail.com> and contributors",
    sitename = "ViewStructArrays.jl",
    format = Documenter.HTML(;
        canonical = "https://ptiede.github.io/ViewStructArrays.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/ptiede/ViewStructArrays.jl",
    devbranch = "main",
)
