#From https://web.stanford.edu/~surag/posts/alphazero.html
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using LinearAlgebra: dot
#using BSON: @save
import Dates: now

mutable struct EGo{M,M2} <: AbstractPlayer 
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

    num_iters = 10
    num_eps = 100
    threshold = 0.55
    
    game = Connect4()            

    for i in 1:num_iters
        
        examples = TrainingExample[]
        for _ in 1:num_eps
            e = execute_episode(ego, game)
            append!(examples, e)
            reset!(game)
        end 
        #Symmetrize input if possible (In Connect4 it is possible)
        symmetry_examples = symmetrize_example.(examples)
        append!(examples, symmetry_examples)

        println("Number of training examples: $(length(examples))")

        old_ego = deepcopy(ego)     
        train_nn!(ego, examples)                  
        frac_win = pit_ego(ego, old_ego, deepcopy(game))                      # compare new net with previous net

        println("Fraction of wins: $(frac_win)")

        if !(frac_win > threshold)
            ego = old_ego #reset to old nn since in was bad  

            #savefile = "saved_models/mode_"*Dates.now()
            #@show savefile
            #@show nn = ego.nn
            #@save savefile nn
        end
    end                             # replace with new net            
end

#=function board_representation(game::Connect4)
    board_rep = zeros(Float64, game.ROWS,game.COLS,1,1)

    counter = 1
    for j in 1:game.COLS#game.ROWS
        for i in 1:game.ROWS#game.COLS
            if 0 == game.board[i,j]
                board_rep[i,j,1,1] = 0.0
            elseif game.board[i,j,1] == game.current_player
                board_rep[i,j,1,1] = 1.0
            else
                board_rep[i,j,1,1] = -1.0
            end
            counter +=1 
        end
    end
    return board_rep
    #return [game.board[:]; game.current_player]
end=#

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
    newex = TrainingExample(symmetry_input, ex.pi, ex.P ,ex.v ,ex.outcome)
    return newex
end

function pit_ego(new_ego::T, old_ego::T, game) where T
    npitgames = 100
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
    n_mcts_sim = 100
    visited = Dict{Int, Vector{Any}}()
    while true

        for i in 1:n_mcts_sim
            ego_search(game, ego.mm, visited)
        end

        P,v,Q,N = visited[game.poskey]
        pi = N./sum(N)

        N2 = copy(N)
        best_i = findmax(N2)[2]
        move = index_2_move(game,best_i)
        while !is_move_legal(game, move)
            N2[best_i] = -Inf
            best_i = findmax(N2)[2]
            move = index_2_move(game,best_i)
        end

        #move = random.choice(len(mcts.pi(s)), p=mcts.pi(s))
        make_move!(game,move)
        outcome = 0.0
        if is_draw(game)
            outcome = 0.0
        elseif _is_player_winning(game, game.current_player == PLAYER1 ? PLAYER2 : PLAYER1 )
            outcome = 1.0
        else
            push!(examples, TrainingExample(board_representation(game), P, pi, v,-100))
            continue
        end

        #update training examples with outcome
        for e in reverse(examples)
            e.outcome = outcome
            outcome *= -1
        end

        return examples

    end
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

    c = 1
    loss = function ffff(x,y)
        P_and_v = ego.nn(x)
        P = P_and_v[1:7]
        v = P_and_v[end]
        _pi = y[1:7]
        z = y[end]
        ppp = params(ego.nn)
        return (v-z)^2 - dot(_pi,log.(P)) + c*dot(ppp,ppp)
    end

    dataset = [(X, Y)]
    evalcb = () -> @show(loss(X, Y))
    opt = ADAM()

    Flux.train!(loss, params(ego.nn), dataset, opt, cb = throttle(evalcb, 10))

end

function train_nn2!()
    X = rand(6*7)
    Y = ones(8)*(1/7)
    
    mask1 = [0., 0., 0., 0., 0., 0., 0., -1.0e10]
    mask2 = [false,false,false,false,false,false,false,true]

    mysoftmax = (x) -> softmax(x.+mask1) + mask2 .* x

    #=
    model = Chain(
        Dense(6*7, 10),
        Dense(10, 7+1),
        mysoftmax)
    =#
    m = Chain(
        # First convolution, operating upon a 28x28 image
        Conv((3, 3), 1=>16, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),

        # Second convolution, operating upon a 14x14 image
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),

        # Third convolution, operating upon a 7x7 image
        Conv((3, 3), 32=>32, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),

        # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
        # which is where we get the 288 in the `Dense` layer below:
        x -> reshape(x, :, size(x, 4)),
        Dense(288, 8),
        mysoftmax)
      #softmax)
      #x -> [softmax(x[1:7]); x[8]])
    m2 = Flux.mapleaves(Flux.data, m)

    @show m2(rand(28,28,1))
    error("hej")
    loss(x, y) = crossentropy(m(x), y)

    dataset = [(X,Y)]
    evalcb = () -> @show(loss(X, Y))
    opt = ADAM()

    Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10))
    @show m2(X)
end

function EGo()

    mask1 = [0., 0., 0., 0., 0., 0., 0., -1.0e10]
    mask2 = [false,false,false,false,false,false,false,true]

    mysoftmax = (x) -> softmax(x.+mask1) + mask2 .* x

    #=
    model = Chain(
        Dense(6*7, 10),
        Dense(10, 7+1),
        mysoftmax)
    =#

    model = Chain(
        # First convolution, operating upon a 28x28 image
        Conv((3, 3), 1=>10, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),
        # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
        # which is where we get the 288 in the `Dense` layer below:
        x -> reshape(x, :, size(x, 4)),
        Dense(90, 8),
        mysoftmax)
        #(x) -> [softmax(x[1:7]); x[8]])

    model2 = Flux.mapleaves(Flux.data, model)

    return EGo(model,model2)
end

function train_ego()


end

function index_2_move(game::Connect4, i::Int)
    return Connect4Move(i)
end


function search(ego::EGo, game::AbstractGame)
    
    visited = Dict{Int, Vector{Any}}()
    #visited[game.poskey] = [0.0,0.0,0.0,1]

    for i in 1:100
        #Ntot = visited[game.poskey][4]
        ego_search(game, ego.mm, visited)
    end

    #
    P, v,Q, N = visited[game.poskey]
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

function ego_search(game::AbstractGame, nn, visited)::Float64

    if !haskey(visited, game.poskey)
            boardrep = board_representation(game)

            P_and_v = nn(reshape(boardrep,(6,7,1,1)))
            #mapleaves(Flux.data, ego.mm(board_representation(game)))
            #P_and_v = Float64[]
            #append!(P_and_v, ones(Float64, 7) * 1/7)
            #push!(P_and_v, Float64(0.0))

            P = P_and_v[1:7] #hardcode connect4
            v = P_and_v[end]
            Q = zeros(Float64,7)
            N = zeros(Int,7)

            aa = []
            push!(aa, P)
            push!(aa, v)
            push!(aa, Q)
            push!(aa, N)
            visited[game.poskey] = aa
        
        return -v
    end
  
  	moves = generate_moves(game)
    max_u::Float64 = -Inf
    local best_move::getmovetype(game)
    Q_best, N_best::Float64, i_best = 0.0, 0.0, 0
    P, v::Float64, Q::Vector{Float64}, N::Vector{Int} = visited[game.poskey]
    local u::Float64
    for move in moves
        
        i  = move.c #Hardcode connect4

        #If Q[i] == inf, this move is a win
        #if Q[i] == Inf
        #    u = max_u+1.0
        #elseif Q[i] == -Inf
        #    u = -Inf
        #else
            u = Q[i] + sqrt(2)*P[i]*sqrt(sum(N))/(1+N[i]) 
        #end
        
        if u>=max_u
            max_u = u
            best_move = move
            N_best = Float64(N[i])
            Q_best = Q[i]
            i_best = i
            if Q[i] == Inf
                break
            end
        end
        #take_move!(game,move)
    end
    
    if is_move_winning(game,best_move)
        v = 1.0
        visited[game.poskey][3][i_best] = (N_best*Q_best + v)/(N_best+1)
        visited[game.poskey][4][i_best] += 1
        return -v
    else
        if is_draw(game)
           v = 0.0
           visited[game.poskey][3][i_best] = (N_best*Q_best + v)/(N_best+1)
           visited[game.poskey][4][i_best] += 1
           return v
        end
    end

    make_move!(game, best_move)
    v = ego_search(game, nn, visited)
    take_move!(game, best_move)

    #@show N_best,Q_best
    visited[game.poskey][3][i_best] = (N_best*Q_best + v)/(N_best+1)
    visited[game.poskey][4][i_best] += 1

    #if v == -Inf
    #    return 1.0
    #else
    return -v
    #end
end


using InteractiveUtils
function egogo2()

    game = Connect4()
    ego = EGo()

    visited = Dict{Int, Vector{Any}}()
    
    ego_search(game, ego.mm,visited)

    ego_search(game, ego.mm,visited)
end

function egogo()
    game = Connect4()
    ego = EGo()

    make_move!(game, Connect4Move(4))
    make_move!(game, Connect4Move(4))
    make_move!(game, Connect4Move(5))
    #make_move!(game, Connect4Move(5))

    print_board(game)
    visited = Dict{Int, Vector{Any}}()
    #visited[game.poskey] = [0.0,0.0,0.0,1]
    for i in 1:1000
        #
        ego_search(game, ego.mm, visited)
        #Ntot = visited[game.poskey][4]
    end

    println("--EVAL--")
    best_p = -Inf
    best_move = nothing
    #for (i, move) in enumerate(moves)
        #i  = move.c #Hardcode connect4
        P, v, Q, N = visited[game.poskey]
        println("N: $N, P: $P")
    #end

end