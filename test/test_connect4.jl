#test_connect4

@testset "Connect 4" begin
	game = Connect4()
    
    #
    @test is_move_legal(game, Connect4Move(1)) == true

    #check if reset!
    reset!(game)
    @test all(game.board .== 0) == true
    @test is_draw(game) == false
    
    #check if is_draw works
    for i in 1:42
        moves = generate_moves(game)
        make_move!(game, moves[1])
    end
    @test is_draw(game) == true
    
    #Check if poskey work with making and taking moves
    reset!(game)
    first_poskey = game.poskey
    make_move!(game, Connect4Move(1))
    make_move!(game, Connect4Move(2))
    take_move!(game, Connect4Move(2))
    take_move!(game, Connect4Move(1))
    @test game.poskey == first_poskey

    #Is the next move a winning move
    reset!(game)
    make_move!(game, Connect4Move(1))
    make_move!(game, Connect4Move(2))
    make_move!(game, Connect4Move(1))
    make_move!(game, Connect4Move(2))
    make_move!(game, Connect4Move(1))
    make_move!(game, Connect4Move(2))
    @test is_move_winning(game, Connect4Move(2)) == false
    @test is_move_winning(game, Connect4Move(1)) == true
end