using FieldDimArrays
using Documenter

DocMeta.setdocmeta!(FieldDimArrays, :DocTestSetup, :(using FieldDimArrays); recursive = true)

makedocs(;
    modules = [FieldDimArrays],
    authors = "Paul Tiede <ptiede91@gmail.com> and contributors",
    sitename = "FieldDimArrays.jl",
    format = Documenter.HTML(;
        canonical = "https://ptiede.github.io/FieldDimArrays.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/ptiede/FieldDimArrays.jl",
    devbranch = "main",
)
