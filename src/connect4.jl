struct Connect4Move <: AbstractMove
    c::Int
end

mutable struct Connect4 <: AbstractGame
    ROWS::Int
    COLS::Int
    board::Matrix{Int}
    current_player::Int
        
    nmovescol::Vector{Int}

    poskey::Int
    piecekeys::Array{Int,3}
    sidekey::Int

    #Maybe have a movehistory vector instead
    lastmove::Connect4Move
    tempmove::Connect4Move
end

function Connect4()
    rows = 6
    cols = 7
    poskey = 0
    piecekeys = rand(Int,2,rows,cols)
    sidekey = rand(Int)
    poskey ⊻= sidekey

    return Connect4(rows, cols, zeros(Int,rows,cols), PLAYER1, zeros(Int,cols),
            poskey, piecekeys, sidekey,
            Connect4Move(-1), Connect4Move(-1))
end


getmovetype(::Connect4) = return Connect4Move

function generate_random_position!(game::Connect4, depth::Int)

    for i in 1:depth
        moves = generate_moves(game)
        random_move = rand(moves)
        make_move!(game,random_move)
    end

end

function reset!(game::Connect4)
    fill!(game.board, 0.0)
    game.current_player = 1
    fill!(game.nmovescol, 0.0)
    game.poskey = 0
    game.poskey ⊻= game.sidekey
    game.lastmove = Connect4Move(-1)
    game.tempmove = Connect4Move(-1)
end

function parse_human_input(str::String)
    myint = parse(Int, str)
    return Connect4Move(myint[1])
end

function is_position_terminal(game::Connect4)
    if is_draw(game)
        return true
    elseif _is_player_winning(game, game.current_player)
        return true
    elseif _is_player_winning(game, (game.current_player == PLAYER1) ? PLAYER2 : PLAYER1)
        return true
    end
    return false
end

function generate_moves(game::Connect4)
    moves = Connect4Move[]
    for i in 1:game.COLS
        toprow = game.board[game.ROWS,i]
        if toprow == 0
            push!(moves, Connect4Move(i))
        end
    end
    return moves
end

function make_move!(game::Connect4, move)
    r = game.nmovescol[move.c]+1
    @assert !(r > game.ROWS)

    game.board[r,move.c] = game.current_player

    game.poskey ⊻= game.piecekeys[game.current_player, r, move.c]
    game.poskey ⊻= game.sidekey

    game.current_player =  (game.current_player == PLAYER1) ? PLAYER2 : PLAYER1

    game.nmovescol[move.c] = r
    game.tempmove = game.lastmove
    game.lastmove = move
end

function take_move!(game::Connect4, move)
    r = game.nmovescol[move.c]
    game.board[r,move.c] = 0

    game.current_player =  (game.current_player == PLAYER1) ? PLAYER2 : PLAYER1

    game.poskey ⊻= game.piecekeys[game.current_player, r, move.c]
    game.poskey ⊻= game.sidekey

    game.nmovescol[move.c] -= 1
    game.lastmove = game.tempmove
    game.tempmove = Connect4Move(-1)
end

function _is_player_winning(game::Connect4, PLAYER::Int)
    
    #rows
    for ir in 1:game.ROWS
        if all(game.board[ir,1:4] .== PLAYER); return true; end;
        if all(game.board[ir,2:5] .== PLAYER); return true; end;
        if all(game.board[ir,3:6] .== PLAYER); return true; end;
        if all(game.board[ir,4:7] .== PLAYER); 
            return true
        end;
    end

    #cols

    for ic in 1:game.COLS
        if all(game.board[1:4, ic] .== PLAYER)
            return true
        end;
        if all(game.board[2:5, ic] .== PLAYER); return true; end;
        if all(game.board[3:6, ic] .== PLAYER); return true; end;
    end

    #diag /
    
    for ic in [0, 6, 12, 18]
    if all(game.board[[1, 8, 15, 22] .+ 0 .+ ic] .== PLAYER); return true; end;
    if all(game.board[[1, 8, 15, 22] .+ 1 .+ ic] .== PLAYER); return true; end;
    if all(game.board[[1, 8, 15, 22] .+ 2 .+ ic] .== PLAYER); return true; end;
    #if all(game.board[[1, 8, 15, 22] .+ 3 .+ ic] .== PLAYER); return true; end;
    end

    #diag \
    for ic in [0, 6, 12, 18]
    if all(game.board[[19, 14, 9, 4] .+ 0 .+ ic] .== PLAYER); return true; end;
    if all(game.board[[19, 14, 9, 4] .+ 1 .+ ic] .== PLAYER); return true; end;
    if all(game.board[[19, 14, 9, 4] .+ 2 .+ ic] .== PLAYER); return true; end;
    #if all(game.board[[19, 14, 9, 4] .+ 3 .+ ic] .== PLAYER); return true; end;
    end
    
    return false


end

function print_board(game::Connect4)
    #@show game.board
    print("\n+-+-+-+-+-+-+-+\n")
    _sign = Dict()
    _sign[1] = "x"
    _sign[2] = "o"
    _sign[0] = "."
    for i in game.ROWS:-1:1
        print("|")
        for j in 1:game.COLS
            print(_sign[game.board[i,j]])
            print("|")
        end
        print("\n+-+-+-+-+-+-+-+\n")
    end
end

function is_move_legal(game::Connect4, move::Connect4Move)
    return (game.nmovescol[move.c]+1) <= game.ROWS
end

function rollout(game::Connect4)
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
    #println("Rollout from $(game.board) and curent player is Player $(game.current_player)")
    if _is_player_winning(game, (game.current_player == PLAYER1) ? PLAYER2 : PLAYER1)
        #println("Player $((game.current_player == PLAYER1) ? PLAYER2 : PLAYER1) had already made a game wining move ")
        return Inf
    end
    while true
        if _is_player_winning(game, (game.current_player == PLAYER1) ? PLAYER2 : PLAYER1)
            #println("In rollout, Player $((game.current_player == PLAYER1) ? PLAYER2 : PLAYER1) wins with board $(game.board)")

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
            #println("In rollout, it is a draw")
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

in_bounds(row::Int, col::Int) = 0<row<(6+1) && 0<col<(7+1)
function is_move_winning(game::Connect4, move::Connect4Move)

    PLAYER = game.current_player
    col = move.c
    row = game.nmovescol[move.c]+1
    @assert !(row > game.ROWS)

    horizontal_offset = [-3,-2,-1,0,1,2,3]*(6)
    vertical_offset = [-3,-2,-1,0,1,2,3]
    rdiag_offset = [-21,-14,-7,0,7,14,21] #/
    ldiag_offset = [-15,-10,-5,0,5,10,15] #\
    
    ii = (game.ROWS*(col-1)) + row

    #Function that ches if 4 pieces are connected
    _check(board, offset) = begin
        connected_pieces = 0
        counter = 1
        while true
            #@show counter
            ind = ii + offset[counter]
            if 0 <  ind < (game.ROWS*game.COLS)
                if offset[counter] == 0
                    connected_pieces += 1
                else
                    if board[ind] == PLAYER
                        connected_pieces +=1
                    else
                        connected_pieces = 0
                    end
                end
            end

            if connected_pieces == 4
                return true
            end

            if counter == 7
                break
            end

            counter +=1
        end
        return false
    end

    if _check(game.board, horizontal_offset) 
        return true
    elseif _check(game.board, vertical_offset) 
        return true
    elseif _check(game.board, ldiag_offset) 
        return true
    elseif _check(game.board, rdiag_offset) 
        return true
    else
        return false
    end


end

function was_last_move_a_win(game::Connect4)


    PLAYER = game.current_player == PLAYER1 ? PLAYER2 : PLAYER1
    col = game.lastmove.c
    if col == -1
        return false
    end
    row = game.nmovescol[col]

    horizontal_offset = [(0,-3),(0,-2),(0,-1),(0,0),(0,1),(0,2),(0,3)]
    #horizontal_offset = [-3,-2,-1,0,1,2,3]*(6)
    vertical_offset = [(-3,0),(-2,0),(-1,0),(0,0),(1,0),(2,0),(3,0)]
    #vertical_offset = [-3,-2,-1,0,1,2,3]
    rdiag_offset = [(-3,-3),(-2,-2),(-1,-1),(0,0),(1,1),(2,2),(3,3)]
    #rdiag_offset = [-21,-14,-7,0,7,14,21] #/
    ldiag_offset = [(-3,3),(-2,2),(-1,1),(0,0),(1,-1),(2,-2),(3,-3)]
    #ldiag_offset = [-15,-10,-5,0,5,10,15] #\
    
    #ii = (game.ROWS*(col-1)) + row
    ir = row
    ic = col

    #Function that checks if 4 pieces are connected
    _check(board, offset) = begin
        connected_pieces = 0
        counter = 1
        while true
            #@show counter
            rr = ir + offset[counter][1]
            cc = ic + offset[counter][2]
            if (0 < cc <= game.COLS) && (0 < rr <= game.ROWS)
                if board[rr,cc] == PLAYER
                    connected_pieces += 1
                else
                    connected_pieces = 0
                end
            end

            if connected_pieces == 4
                return true
            end

            if counter == 7
                break
            end

            counter +=1
        end
        return false
    end

    if _check(game.board, horizontal_offset) 
        return true
    elseif _check(game.board, vertical_offset) 
        return true
    elseif _check(game.board, ldiag_offset) 
        return true
    elseif _check(game.board, rdiag_offset) 
        return true
    else
        return false
    end


end

is_player1_winning(game::Connect4) = _is_player_winning(game,PLAYER1)
is_player2_winning(game::Connect4) = _is_player_winning(game,PLAYER2)
is_draw(game::Connect4) = all(game.board[:] .!= 0)
