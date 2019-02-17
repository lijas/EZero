using BenchmarkTools
include("src/EZero.jl")

game = Connect4()
move = Connect4Move(4)

@btime begin
    make_move!(game, move)
    take_move!(game, move)
end

e0 = EGo()
reset!(game)
@btime begin 
    visited = visited = Dict{Int, Vector{Any}}()
    ego_search(game, e0.mm, visited)
end

