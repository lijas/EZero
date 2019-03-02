using BenchmarkTools
include("src/EZero.jl")

game = Connect4()
move = Connect4Move(4)

@btime begin
    make_move!(game, move)
    take_move!(game, move)
end

e0 = EZero1()
reset!(game)
@btime begin 
    visited = Dict{Int, Vector{Any}}()
    reset!(game)
    for _ in 1:50
        ego_search(game, e0.mm, visited)
    end
end

#Check how long a game between Montecarlo players take
#There are some randomness to montecarlo, so probobly
#not the best way to benchmark
#=
reset!(game)
p1 = MonteCarloPlayer(100)
p2 = MonteCarloPlayer(10)
@btime begin
    reset!(game)
    pit(game,p1,p2; should_print = false)
end
=#

reset!(game)
make_move!(game, Connect4Move(4));make_move!(game, Connect4Move(5))
make_move!(game, Connect4Move(4));make_move!(game, Connect4Move(4))
make_move!(game, Connect4Move(3));make_move!(game, Connect4Move(1))
make_move!(game, Connect4Move(6));make_move!(game, Connect4Move(4))
make_move!(game, Connect4Move(3));make_move!(game, Connect4Move(7))
make_move!(game, Connect4Move(3));make_move!(game, Connect4Move(2))
@btime was_last_move_a_win(game)
make_move!(game, Connect4Move(1));make_move!(game, Connect4Move(3))
@btime was_last_move_a_win(game);