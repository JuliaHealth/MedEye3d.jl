module A
    module B
        function test()
            println(@__MODULE__)
            println(parentmodule(@__MODULE__))
            println(parentmodule(parentmodule(@__MODULE__)))
        end
    end
end
A.B.test()
