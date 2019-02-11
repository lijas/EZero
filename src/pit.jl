

function pit(game, player1, player2; should_print = true)

    first_player_to_move = game.current_player
    second_player_to_move = game.current_player == PLAYER1 ? PLAYER2 : PLAYER1
	winning_player = 0.0
    while true
		move = search(player1, game)
		make_move(game, move)
		
        should_print && print_board(game)
        should_print && println("")

        if _is_player_winning(game, first_player_to_move)
            winning_player = first_player_to_move
            break
        elseif is_draw(game)
            break
        end

		move = search(player2, game)
		make_move(game, move)
		
        should_print && print_board(game)
        should_print && println("")

        if _is_player_winning(game, second_player_to_move)
            winning_player = second_player_to_move
            break
        elseif is_draw(game)
            break
        end

	end

    return winning_player



end