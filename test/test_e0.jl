
@testset "EZero search" begin
	
	e0 = EGo()
	#Create a position where player is one move
	#from winning. See if E0 finds that move
	game = Connect4()
	make_move!(game, Connect4Move(1))
	make_move!(game, Connect4Move(2))
	make_move!(game, Connect4Move(1))
	make_move!(game, Connect4Move(2))
	make_move!(game, Connect4Move(1))
	make_move!(game, Connect4Move(2))

	visited = Dict{Int, Vector{Any}}()
	bestmove = search(e0, game)
	@test bestmove == Connect4Move(1)

end