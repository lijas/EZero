

function pit(game, player1, player2; should_print = true)

    @assert was_last_move_a_win(game) == false

    first_player_to_move = game.current_player
    second_player_to_move = game.current_player == PLAYER1 ? PLAYER2 : PLAYER1
	winning_player = 0.0
    while true
		move = search(player1, game)
		make_move!(game, move)
		
        should_print && print_board(game)
        should_print && println("")

        if was_last_move_a_win(game)
            winning_player = first_player_to_move
            break
        elseif is_draw(game)
            break
        end

		move = search(player2, game)
		make_move!(game, move)
		
        should_print && print_board(game)
        should_print && println("")

        if was_last_move_a_win(game)
            winning_player = second_player_to_move
            break
        elseif is_draw(game)
            break
        end

	end

    return winning_player



end