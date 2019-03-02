
@testset "EZero search" begin
	fake_neural_network(x) = vcat(ones(7)*1/7, 0.0)
	e0 = EZero(fake_neural_network,fake_neural_network)
	#Create a position where player is one move
	#from winning. See if E0 finds that move
	game = Connect4()
	make_move!(game, Connect4Move(4))
	make_move!(game, Connect4Move(5))
	make_move!(game, Connect4Move(4))
	make_move!(game, Connect4Move(5))
	make_move!(game, Connect4Move(4))
	make_move!(game, Connect4Move(5))

	visited = Dict{Int, MCTSVectors}()
	bestmove = search(e0, game)
	@test bestmove == Connect4Move(4)

    #Create a position where player can force a win in two moves
    #Test if E0 finds that move

    #Make a neural network which values all positions equally
    #Otherwise it does not really work....
    e0 = EZero(fake_neural_network,fake_neural_network)

    game = Connect4()
    make_move!(game, Connect4Move(3))
    make_move!(game, Connect4Move(3))
    make_move!(game, Connect4Move(5))
    make_move!(game, Connect4Move(5))
    
    visited = Dict{Int, MCTSVectors}()
    bestmove = search(e0, game)
    print_board(game)
    @test bestmove == Connect4Move(4)


    #-------
    game = Connect4()
    make_move!(game, Connect4Move(1))
    make_move!(game, Connect4Move(1))
    make_move!(game, Connect4Move(1))
    make_move!(game, Connect4Move(4))
    make_move!(game, Connect4Move(3))
    make_move!(game, Connect4Move(1))
    make_move!(game, Connect4Move(1))
    make_move!(game, Connect4Move(4))
    make_move!(game, Connect4Move(4))
    make_move!(game, Connect4Move(1))
    make_move!(game, Connect4Move(4))
    make_move!(game, Connect4Move(5))
    make_move!(game, Connect4Move(5))
    make_move!(game, Connect4Move(2))
    make_move!(game, Connect4Move(4))
    make_move!(game, Connect4Move(7))
    print_board(game)

end