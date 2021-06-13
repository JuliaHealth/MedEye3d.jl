using DrWatson
@quickactivate "Probabilistic medical segmentation"

module FileStructs
export fileStructs

struct filePathAndModuleName
    filePath::String
    moduleName::String
    end

end # module