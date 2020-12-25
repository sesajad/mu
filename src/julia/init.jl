
push!(LOAD_PATH, "/home/sloth/Files/Works/Personal/mu/src/julia/lang")
push!(LOAD_PATH, "/home/sloth/Files/Works/Personal/mu/src/julia/machine")
push!(LOAD_PATH, "/home/sloth/Files/Works/Personal/mu/src/julia/spaces")
push!(LOAD_PATH, "/home/sloth/Files/Works/Personal/mu/src/julia/types")

include("./machine/SimulatorMachine.jl")
using .SimuatorMachine
