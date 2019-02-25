

struct MonteCarloPlayer <: AbstractPlayer 
    nsearches::Int
end

struct HumanPlayer <: AbstractPlayer 
    
end

function search(player::HumanPlayer, game::AbstractGame)
    str = readline(stdin)
    move = parse_human_input(str)
    return move
end

function search(player::MonteCarloPlayer, game::AbstractGame)

    visited = Dict{Int, Vector{Float64}}()
    visited[game.poskey] = [0.0,1.0]
    for i in 1:player.nsearches
        mct_search(game, visited)
    end

    moves = generate_moves(game)
    bestmove = nothing
    max_visits = -Inf
    for move in moves
        make_move!(game, move)
        
        nwins, nvisits = visited[game.poskey]
        if(nwins == Inf)
            bestmove = move
            take_move!(game,move)
            #println("r: $(move.r), c: $(move.c) with INF")
            break
        end
        if(nvisits > max_visits)
            max_visits = nvisits
            bestmove = move
        end 
       # println("r: $(move.r), c: $(move.c) with $((nwins, nvisits))")
        take_move!(game,move)
    end
    return bestmove
end

function mct_search(game::AbstractGame, visited)
    
    moves_taken = []
    outcome = 0.0
    wins, Ntot = visited[game.poskey]
    while true
        #print_board(game)
        moves = generate_moves(game)

        #Should check if this move is terminal maybe
        if length(moves) == 0
            #if there are no legal moves,
            #it is a draw
            outcome = 0.0
            backprop = true
            break
        end
        backprop = false
        maxU = -Inf
        Ntot_bestmove = 0.0
        bestmove = nothing
        for move in moves
            make_move!(game,move)

            if !haskey(visited, game.poskey)
                #We hit a leafnode
                #Perform a rollout and backpropagate the result
                backprop = true
                outcome = rollout(game)
                #println("Found a leafnode with outcome: $outcome")
                visited[game.poskey] = [outcome,1]
                if outcome == Inf; outcome = 1;  end
                take_move!(game,move)
                break
            end
            nwins, ntot = visited[game.poskey]
            if nwins == Inf || nwins == -Inf
                outcome = 1
                backprop = true
                take_move!(game, move)
                break
            end
            U = (nwins/ntot) + sqrt(2)*sqrt(log(Ntot)/ntot)
            #println("Checking move $move:, $nwins, $ntot, $U")
            if U>maxU
                maxU = U
                bestmove = move
                Ntot_bestmove = ntot

            end
            take_move!(game, move)
        end
        
        if backprop == true
            break
        else
            #println("Player $(game.current_player) choose moved r: $(bestmove.r), c: $(bestmove.c)")
            Ntot = Ntot_bestmove
            make_move!(game, bestmove)
            push!(moves_taken, bestmove)
        end
    end
    
    #Backprop
    while length(moves_taken) > 0
        move = pop!(moves_taken)
        outcome *= -1
        #println("Backproping move $move")
        visited[game.poskey] += [outcome,1] 
        take_move!(game, move)
        #@show game.board
        
    end
    visited[game.poskey] += [outcome,1] 

end
