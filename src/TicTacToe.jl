mutable struct TicTacToe <: AbstractGame
    board::Matrix{Int}
    current_player::Int
    
    poskey::Int
    piecekeys::Array{Int,3}
    sidekey::Int
end

function TicTacToe()
    poskey = 0
    piecekeys = rand(Int,2,3,3)
    sidekey = rand(Int)
    poskey ⊻= sidekey

    return TicTacToe(zeros(Int,3,3), PLAYER1,
            poskey, piecekeys, sidekey)
end

struct TicTacToeMove <: AbstractMove
    r::Int
    c::Int
end

getmovetype(::TicTacToe) = return TicTacToeMove

function parse_human_input(str::String)
    intvec = parse.(Int, split(str,' '))
    return TicTacToeMove(intvec[1], intvec[2])
end

function is_position_terminal(game::TicTacToe)
    if is_draw(game)
        return true
    elseif _is_player_winning(game, game.current_player)
        return true
    elseif _is_player_winning(game, (game.current_player == PLAYER1) ? PLAYER2 : PLAYER1)
        return true
    end
    return false
end

function generate_moves(game::TicTacToe)
    moves = TicTacToeMove[]
    for i in 1:3
        for j in 1:3
            if game.board[i,j] == 0
                push!(moves, TicTacToeMove(i,j))
            end
        end
    end
    return movesf
end

function make_move!(game::TicTacToe, move)
    game.board[move.r,move.c] = game.current_player

    game.poskey ⊻= game.piecekeys[game.current_player, move.r, move.c]
    game.poskey ⊻= game.sidekey

    game.current_player =  (game.current_player == PLAYER1) ? PLAYER2 : PLAYER1
end

function take_move!(game::TicTacToe, move)
    game.board[move.r,move.c] = 0

    game.current_player =  (game.current_player == PLAYER1) ? PLAYER2 : PLAYER1

    game.poskey ⊻= game.piecekeys[game.current_player, move.r, move.c]
    game.poskey ⊻= game.sidekey
end

function _is_player_winning(game::TicTacToe, PLAYER::Int)
    #rows
    for i in 1:3
        
        if all(game.board[1:3,i] .== PLAYER)
            return true
        end

        #col
        if all(game.board[i,1:3] .== PLAYER)
            return true
        end
    end

    if all(game.board[[1,5,9]] .== PLAYER)
        return true
    end

    if all(game.board[[3,5,6]] .== PLAYER)
        return true
    end

    return false
end

function print_board(game::TicTacToe)
    #@show game.board
    for i in 1:3
        for j in 1:3
            print(game.board[i,j])
        end
        print('\n')
    end
end

function rollout(game::TicTacToe)
    #if _is_player_winning(game, (game.current_player == PLAYER1) ? PLAYER2 : PLAYER1)
    #    println("In rollout, Player1 wins")
    #    return -1
    #elseif is_player2_winning(game)
    #    println("In rollout, Player2 wins")
    #    return 1
    #elseif is_draw(game)
    #    println("In rollout, draw")
    #    return 0.0
    #end
    first_player = game.current_player
    moves_taken = []
    println("Rollout from $(game.board) and curent player is Player $(game.current_player)")
    if _is_player_winning(game, (game.current_player == PLAYER1) ? PLAYER2 : PLAYER1)
        println("Player $((game.current_player == PLAYER1) ? PLAYER2 : PLAYER1) had already made a game wining move ")
        return Inf
    end
    while true
        if _is_player_winning(game, (game.current_player == PLAYER1) ? PLAYER2 : PLAYER1)
            println("In rollout, Player $((game.current_player == PLAYER1) ? PLAYER2 : PLAYER1) wins with board $(game.board)")

            returnvalue = 0.0
            if first_player == game.current_player
                returnvalue = 1
            else
                returnvalue = -1
            end
            while length(moves_taken) > 0
                take_move!(game, pop!(moves_taken))
            end
            return returnvalue

        elseif is_draw(game)
            println("In rollout, it is a draw")
            while length(moves_taken) > 0
                take_move!(game, pop!(moves_taken))
            end
            return 0.0
        end

        moves = generate_moves(game)
        move = rand(moves)
        make_move!(game, move)
        push!(moves_taken, move)
    end
    
    #return outcome
end

is_player1_winning(game::TicTacToe) = _is_player_winning(game,PLAYER1)
is_player2_winning(game::TicTacToe) = _is_player_winning(game,PLAYER2)
is_draw(game::TicTacToe) = all(game.board[:] .!= 0)
