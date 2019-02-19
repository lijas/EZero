using BenchmarkTools
include("src/EZero.jl")

game = Connect4()
move = Connect4Move(4)

@benchmark begin
    make_move!(game, move)
    take_move!(game, move)
end

e0 = EGo()
reset!(game)
@benchmark begin 
    visited = Dict{Int, Vector{Any}}()
    ego_search(game, e0.mm, visited)
end

#Check how long a game between Montecarlo players take
#They should always be picking the same move...
reset!(game)
p1 = MonteCarloPlayer(10)
p2 = MonteCarloPlayer(10)
@benchmark pit(game,p1,p2; should_print = false)
