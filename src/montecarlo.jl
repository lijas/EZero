
abstract type AbstractGame end 

const PLAYER1 = 1
const PLAYER2 = 2

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

struct TicTacToeMove 
    r::Int
    c::Int
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
    return moves
end

function make_move(game::TicTacToe, move)
    game.board[move.r,move.c] = game.current_player

    game.poskey ⊻= game.piecekeys[game.current_player, move.r, move.c]
    game.poskey ⊻= game.sidekey

    game.current_player =  (game.current_player == PLAYER1) ? PLAYER2 : PLAYER1
end

function take_move(game::TicTacToe, move)
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
    @show game.board
end

function rollout(game::TicTacToe)
    if is_player1_winning(game)
        return 1.0
    elseif is_player2_winning(game)
        return 0.0
    else
        return 0.5
    end
    moves = generate_moves(game)
    move = rand(move[1])
    make_move(game, move)
    outcome = rollout(game)
    take_move(game, move)
    return outcome
end

is_player1_winning(game::TicTacToe) = _is_player_winning(game,PLAYER1)
is_player2_winning(game::TicTacToe) = _is_player_winning(game,PLAYER2)
is_draw(game::TicTacToe) = all(game.board[:] .!= 0)

function search(game::AbstractGame, visited, Ntot)


    if is_player1_winning(game)
        return 1.0
    elseif is_player2_winning(game)
        return 0.0
    elseif is_draw(game)
        return 0.5
    end

    #print_board(game)

    #
    moves = generate_moves(game)

    #
    maxu = -Inf
    bestmove = nothing
    this_Ntot = 0
    for move in moves

        make_move(game, move)
        U = 0.0
        wins = 0
        nsim = 0
        if !haskey(visited, game.poskey)
            outcome = rollout(game)
            visited[game.poskey] = [outcome,1.0]
            take_move(game, move)
            return outcome
        end
        wins, nsim = visited[game.poskey]
        U = wins/nsim + sqrt(2 * log(Ntot)/nsim)
        #@show wins,nsim, Ntot, U  
          
        take_move(game, move)
        this_Ntot += nsim
        
        if U > maxu
            maxu = U
            bestmove = move
        end
    end

    #
	make_move(game, bestmove)
	outcome = search(game, visited, this_Ntot)
	visited[game.poskey] += [outcome, 1]
    take_move(game,bestmove)

    visited[game.poskey] += [outcome, 1]
    #
    return outcome

end

function gogogo()
    game = TicTacToe()

    make_move(game, TicTacToeMove(2,2))
    make_move(game, TicTacToeMove(1,1))
    make_move(game, TicTacToeMove(2,1))
    make_move(game, TicTacToeMove(1,2))

    visited = Dict{Int, Vector{Float64}}()
    visited[game.poskey] = [0.5,1.0]
    for i in 1:200
        Ntot = visited[game.poskey][2]
        search(game, visited, Ntot)
    end

    moves = generate_moves(game)
    for move in moves
        make_move(game, move)
        println("r: $(move.r), c: $(move.c) with $(visited[game.poskey])")
        take_move(game,move)
    end



end