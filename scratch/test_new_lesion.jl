using Test
using GLMakie
using MedEye3d
import MedEye3d.LesionMetadataWindow as LMW_mod
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH

fig = Figure()
active_id = Observable("1: Liver")
lesion_ids = Observable(["1: Liver"])

MEH.current_tp_index[] = 1
MEH.tp_organ_mapping = Dict(1 => Dict(1 => "Liver"))
MEH.global_organ_mapping[] = Dict(1 => "Liver")

LMW = LMW_mod.create_metadata_window(active_id, lesion_ids, Channel{Any}(32))

println("Setting initial state for Lesion 1")
LMW.lesion_db[] = Dict{String, Any}("1" => Dict("ObservationState" => "ACCEPTED", "Anatomic Location" => "Liver", "SUV max" => "10.0"))

sleep(1)

obs = LMW_mod._lmw_observables
println("Initial active lesion: ", active_id[])
println("Initial observation state: ", obs[:menu_obs_state].selection[])
println("Initial Anatomic Location: ", obs[:field_widgets]["Anatomic Location"].selection[])
if haskey(obs[:field_widgets], "SUV max")
    println("Initial SUV max: ", obs[:field_widgets]["SUV max"].stored_string[])
end

println("\nClicking New Lesion")
obs[:btn_new_lesion].clicks[] = obs[:btn_new_lesion].clicks[] + 1
sleep(1)

println("Active Lesion: ", active_id[])
println("New observation state: ", obs[:menu_obs_state].selection[])
println("New Anatomic Location: ", obs[:field_widgets]["Anatomic Location"].selection[])
if haskey(obs[:field_widgets], "SUV max")
    println("New SUV max: ", obs[:field_widgets]["SUV max"].stored_string[])
end
