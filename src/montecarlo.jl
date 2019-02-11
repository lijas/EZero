
abstract type AbstractGame end 
abstract type AbstractPlayer end
abstract type AbstractMove end

const PLAYER1 = 1
const PLAYER2 = 2

include("TicTacToe.jl")
include("Connect4.jl")
include("pit.jl")
include("EGo.jl")

struct MonteCarloPlayer <: AbstractPlayer 
    search_time::Float64
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
    for i in 1:50
        search2(game, visited)
    end

    moves = generate_moves(game)
    bestmove = nothing
    max_visits = -Inf
    for move in moves
        make_move(game, move)
        
        nwins, nvisits = visited[game.poskey]
        if(nwins == Inf)
            bestmove = move
            take_move(game,move)
            #println("r: $(move.r), c: $(move.c) with INF")
            break
        end
        if(nvisits > max_visits)
            max_visits = nvisits
            bestmove = move
        end 
       # println("r: $(move.r), c: $(move.c) with $((nwins, nvisits))")
        take_move(game,move)
    end
    return bestmove
end

function search(game::AbstractGame, visited, Ntot)

    if is_player1_winning(game)
        if game.current_player == PLAYER1
            return +1.0
        else
            return +1.0
        end
    elseif is_player2_winning(game)
        if game.current_player == PLAYER2
            return -1.0
        else
            return -1.0
        end
    elseif is_draw(game)
        return 0.0
    end

    print_board(game)

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
            println("Player $((game.current_player == PLAYER1) ? PLAYER2 : PLAYER1) choosed $move, outcome from rollout: $(-outcome)")
            visited[game.poskey] = [-outcome,1.0]
            take_move(game, move)
            return -outcome
        end
        Q, nsim = visited[game.poskey]
        #U = wins/nsim + sqrt(2 * log(Ntot)/nsim)
        U = Q/nsim + (sqrt(2) * sqrt(Ntot)/nsim)
        println("Move r: $(move.r), c: $(move.c) ->  $Q, $nsim, $Ntot, $U")  
          
        take_move(game, move)
        this_Ntot += nsim
        
        if U > maxu
            maxu = U
            bestmove = move
        end
    end
    println("Player $(game.current_player) choose moved r: $(bestmove.r), c: $(bestmove.c)")
    #
	make_move(game, bestmove)
	outcome = search(game, visited, this_Ntot)
	#visited[game.poskey] += [-outcome, 1.0]
    Q, nsim = visited[game.poskey]
    visited[game.poskey][1] -= outcome
    visited[game.poskey][2] += 1
    take_move(game,bestmove)

    #visited[game.poskey] += [outcome, 1.0]
    #Q, nsim = visited[game.poskey] 
    #visited[game.poskey][1] = (nsim*Q + -outcome)/nsim
    #visited[game.poskey][2] += 1
    #
    return outcome

end

function search2(game::AbstractGame, visited)
    
    moves_taken = []
    outcome = 0.0
    wins, Ntot = visited[game.poskey]
    while true
        #print_board(game)
        moves = generate_moves(game)

        #Should check if this move is terminal maybe
        if length(moves) == 0
            #if there are no legal moves,
            #it is probaebly a draw
            outcome = 0.0
            backprop = true
            break
        end
        backprop = false
        maxU = -Inf
        Ntot_bestmove = 0.0
        bestmove = nothing
        for move in moves
            make_move(game,move)

            ##########
            ##########
            #=
            if haskey(visited, game.poskey)
                if _is_player_winning(game, (game.current_player == PLAYER1) ? PLAYER2 : PLAYER1)
                    println("In this move, Player $((game.current_player == PLAYER1) ? PLAYER2 : PLAYER1) wins  with board $(game.board)")
                    outcome = -1
                    backprop = true
                    visited[game.poskey] = [outcome,1]
                    take_move(game,move)
                    break

                elseif is_draw(game)
                    println("In this move, it is a draw")
                    outcome = 0.0
                    backprop = true
                    visited[game.poskey] = [outcome,1]
                    take_move(game,move)
                    break
                end
            end
            =#
            ##########
            ##########

            if !haskey(visited, game.poskey)
                #We hit a leafnode
                #Perform a rollout and backpropagate the result
                backprop = true
                outcome = rollout(game)
                #println("Found a leafnode with outcome: $outcome")
                visited[game.poskey] = [outcome,1]
                if outcome == Inf; outcome = 1;  end
                take_move(game,move)
                break
            end
            nwins, ntot = visited[game.poskey]
            if nwins == Inf || nwins == -Inf
                outcome = 1
                backprop = true
                take_move(game, move)
                break
            end
            U = (nwins/ntot) + sqrt(2)*sqrt(log(Ntot)/ntot)
            #println("Checking move $move:, $nwins, $ntot, $U")
            if U>maxU
                maxU = U
                bestmove = move
                Ntot_bestmove = ntot

            end
            take_move(game, move)
        end
        
        if backprop == true
            break
        else
            #println("Player $(game.current_player) choose moved r: $(bestmove.r), c: $(bestmove.c)")
            Ntot = Ntot_bestmove
            make_move(game, bestmove)
            push!(moves_taken, bestmove)
        end
    end
    @show moves_taken
    #Backprop
    while length(moves_taken) > 0
        move = pop!(moves_taken)
        outcome *= -1
        println("Backproping move $move")
        visited[game.poskey] += [outcome,1] 
        take_move(game, move)
        #@show game.board
        
    end
    visited[game.poskey] += [outcome,1] 

end

function gogogo()
    game = TicTacToe()

    make_move(game, TicTacToeMove(2,2))
    make_move(game, TicTacToeMove(1,1))
    make_move(game, TicTacToeMove(1,2))
    make_move(game, TicTacToeMove(2,1))
    make_move(game, TicTacToeMove(3,3))

    visited = Dict{Int, Vector{Float64}}()
    visited[game.poskey] = [0.0,1.0]
    for i in 1:1000
        search2(game, visited)
    end

    moves = generate_moves(game)
    print_board(game)
    for move in moves
        make_move(game, move)
        println("r: $(move.r), c: $(move.c) with $(visited[game.poskey])")
        take_move(game,move)
    end

end