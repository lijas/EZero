#From https://web.stanford.edu/~surag/posts/alphazero.html
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using LinearAlgebra: dot
import Random: shuffle!
using BSON
using Dates: now, format
using StatsBase: Weights, sample

mutable struct EZero{M,M2} <: AbstractPlayer 
    nn::M
    mm::M2
end

mutable struct TrainingExample
    input::Vector{Int}
    P::Vector{Float64}
    pi::Vector{Float64}
    v::Float64
    outcome::Int
end

function policy_iter(ego, game)

    num_iters = 10000
    num_eps = 400
    threshold = 0.53
    
    #Make sure game is reset
    reset!(game)

    thread_games = [Connect4() for _ in 1:Threads.nthreads()]

    for i in 1:num_iters
        
        #Multithreaded self_play
        thread_examples = [TrainingExample[] for _ in 1:Threads.nthreads()]
        #
        Threads.@threads for _ in 1:num_eps
            #Thread id
            tid = Threads.threadid()

            e = execute_episode(ego, thread_games[tid])

            append!(thread_examples[tid], e)
            reset!(thread_games[tid])
        end 

        #Flatten TrainingExamples
        examples = [x for examples in thread_examples for x in examples]

        #Symmetrize input if possible (In Connect4 it is possible)
        symmetry_examples = symmetrize_example.(examples)
        append!(examples, symmetry_examples)

        println("Number of training examples: $(length(examples))")

        old_ego = deepcopy(ego)     
        train_nn!(ego, examples)                  
        frac_win = pit_ego(ego, old_ego, deepcopy(game))                      # compare new net with previous net

        println("Fraction of wins: $(frac_win)")

        if frac_win > threshold
            savefile = "src/saved_models/model_" * format(now(), "yyyy_mm_dd_HH_MM_SS") * ".bson"
            println("Saving current nn to $savefile")
            nn = ego.nn
            BSON.@save savefile nn
        else
            ego = old_ego
        end
    end                             # replace with new net            
end

function board_representation(game::Connect4)
    board_rep = zeros(Float64, game.ROWS*game.COLS)

    counter = 1
    for j in 1:game.COLS#game.ROWS
        for i in 1:game.ROWS#game.COLS
            if 0 == game.board[i,j]
                board_rep[counter] = 0.0
            elseif game.board[i,j] == game.current_player
                board_rep[counter] = 1.0
            else
                board_rep[counter] = -1.0
            end
            counter +=1 
        end
    end
    return board_rep
    #return [game.board[:]; game.current_player]
end

function symmetrize_example(ex::TrainingExample)
    symmetry_input = reverse(reshape(ex.input,6,7), dims = 2)[:]
    symmetry_pi = reverse(ex.pi)
    symmetry_P = reverse(ex.P)
    newex = TrainingExample(symmetry_input, symmetry_P, symmetry_pi, ex.v ,ex.outcome)
    return newex
end

function pit_ego(new_ego, old_ego, game, npitgames = 100)
    #npitgames = 100
    nwins = 0

    for i in 1:(npitgames/2)
        reset!(game)
        generate_random_position!(game, 6)
        
        new_ego_player = game.current_player
        winner = pit(deepcopy(game), new_ego, old_ego; should_print = false)
        nwins += winner == new_ego_player ? 1 : 0
        #If it is a draw, winner = 0.0
        npitgames -= winner == 0.0 ? 1 : 0

        new_ego_player = game.current_player == PLAYER1 ? PLAYER2 : PLAYER1
        winner = pit(deepcopy(game), old_ego, new_ego; should_print = false)
        nwins += winner == new_ego_player ? 1 : 0
        #If it is a draw, winner = 0.0
        npitgames -= winner == 0.0 ? 1 : 0

    end
    #all games are a draw
    if npitgames == 0
        return .5
    end
    println("$nwins / $(npitgames)")
    return nwins/(npitgames)
end

function execute_episode(ego, game)

    examples = TrainingExample[]
    n_mcts_sim = 200
    
    local outcome::Int
    while true

        visited = Dict{Int, MCTSVectors}()
        for i in 1:n_mcts_sim
            ego_search(game, ego.mm, visited)
        end

        mcts_vectors = visited[game.poskey]
        P = mcts_vectors.P
        v = mcts_vectors.v
        Q = mcts_vectors.Q 
        N = mcts_vectors.N
        pi = N./sum(N)

        probabilities = Weights(pi)

        N2 = copy(N)
        best_i = sample(1:7,probabilities)#findmax(N2)[2]
        move = index_2_move(game,best_i)
        while !is_move_legal(game, move)
            probabilities[best_i] = 0.0
            best_i = sample(1:7,probabilities)
            move = index_2_move(game,best_i)
        end

        #move = random.choice(len(mcts.pi(s)), p=mcts.pi(s))
        push!(examples, TrainingExample(board_representation(game), P, pi, v,-100))

        make_move!(game,move)
        
        if is_draw(game)
            outcome = 0.0
            break
        elseif was_last_move_a_win(game)
            outcome = 1.0
            break
        end
    end

    #update training examples with outcome
    for e in reverse(examples)
        e.outcome = outcome
        outcome *= -1
    end

    return examples

end

function train_nn!(ego, examples)
    
    #X = zeros(Float64,6*7,length(examples))
    X = zeros(Float64,6,7,1,length(examples))
    Y = zeros(Float64,7+1,length(examples))
    for i in 1:length(examples)
        counter = 1
        for c in 1:7
            for r in 1:6
                X[r,c,1,i] = examples[i].input[counter]
                counter += 1
            end
        end
        Y[:,i] = vcat(examples[i].pi, Float64(examples[i].outcome))
    end

    c = 0.1 # what value to use for this
    loss = function ffff(x,y)
        P_and_v = ego.nn(x)
        P = P_and_v[1:7]
        v = P_and_v[end]
        _pi = y[1:7]
        z = y[end]
        ppp = params(ego.nn)
        return (z-v)^2 - dot(_pi,log.(P)) + c*dot(ppp,ppp)
    end

    dataset = repeated((X, Y),1)
    evalcb = () -> @show(loss(X, Y))
    opt = ADAM()

    Flux.train!(loss, params(ego.nn), dataset, opt, cb = throttle(evalcb, 10))

end

function EZero(model::T) where T
    model2 = Flux.mapleaves(Flux.data, model)
    return EZero(model,model2)
end

function train_ego()


end

function index_2_move(game::Connect4, i::Int)
    return Connect4Move(i)
end


function search(ego::EZero, game::AbstractGame)
    
    visited = Dict{Int, MCTSVectors}()
    for i in 1:300
        ego_search(game, ego.mm, visited)
    end

    #
    mcts_vectors = visited[game.poskey]
    P = mcts_vectors.P
    v = mcts_vectors.v
    Q = mcts_vectors.Q 
    N = mcts_vectors.N
    @show N
    N2 = copy(N)
    best_i = findmax(N2)[2]
    move = index_2_move(game,best_i)
    while !is_move_legal(game, move)
        N2[best_i] = -Inf
        best_i = findmax(N2)[2]
        move = index_2_move(game,best_i)
    end

    return move

end

struct MCTSVectors
    P::Vector{Float64}
    v::Float64
    Q::Vector{Float64}
    N::Vector{Int}
end

function ego_search(game::AbstractGame, nn, visited)::Float64

    if was_last_move_a_win(game)
        return 1.0
    else
        if is_draw(game)
           return 0.0
        end
    end

    if !haskey(visited, game.poskey)

            boardrep = board_representation(game)
            P_and_v = nn(reshape(boardrep,(6,7,1,1)))

            P = P_and_v[1:7] #hardcode connect4
            v = P_and_v[end]
            Q = zeros(Float64,7)
            N = zeros(Int,7)

            visited[game.poskey] = MCTSVectors(P,v,Q,N)
        
        return -v
    end
  
  	moves = generate_moves(game)
    
    mcts_vectors  = visited[game.poskey]
    P = mcts_vectors.P
    v = mcts_vectors.v
    Q = mcts_vectors.Q 
    N = mcts_vectors.N

    local u::Float64
    local best_move::getmovetype(game)
    u_best, Q_best, N_best::Float64, i_best = -Inf, 0.0, 0.0, 0

    for move in moves
        
        i  = move.c #Hardcode connect4, must be changed for other games

        u = Q[i] + sqrt(2)*P[i]*sqrt(sum(N))/(1+N[i]) 
 
        if u>u_best
            u_best = u
            best_move = move
            N_best = Float64(N[i])
            Q_best = Q[i]
            i_best = i
        end
    end

    make_move!(game, best_move)
    v = ego_search(game, nn, visited)
    take_move!(game, best_move)

    visited[game.poskey].Q[i_best] = (N_best*Q_best + v)/(N_best+1)
    visited[game.poskey].N[i_best] += 1
    
    return -v
end