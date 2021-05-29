using DrWatson
@quickactivate "Probabilistic medical segmentation"
DrWatson.greet()


using Observables
observable = Observable(0)

obs_func = on(observable) do val
    println("Got an update: ", val)
end


observable[] = 42

off(obs_func)