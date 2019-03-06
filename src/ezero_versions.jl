
struct BranchLayer{T1,T2}
    head1::T1
    head2::T2
end
(b::BranchLayer)(x) = vcat(b.head1(x), b.head2(x))

struct IdentitySkip{T}
    inner::T
end
(m::IdentitySkip)(x) = m.inner(x) + x

Flux.@treelike BranchLayer
Flux.@treelike IdentitySkip

function EZero4()

    residual_block() = Chain(
        IdentitySkip(Dense(84,84,σ)))

    first_block() = Chain(
        Dense(42,84, σ),
        residual_block(),
        residual_block(),
        residual_block())

    policy_block() = Chain(
        residual_block(),
        Dense(84,7,σ),
        softmax)

    value_block() = Chain(
        residual_block(),
        Dense(84,1,tanh))

    model = Chain(first_block(), BranchLayer(policy_block(),
                                             value_block()))

    model2 = Flux.mapleaves(Flux.data, model)

    return EZero(model, model2)
end

function EZero3()

    conv_block(x;inputsize=42) = Chain(
        Conv((4,4), inputsize=>42), 
        BatchNorm(42),
        (x)->leakyrelu.(x))(x)

    conv_block2(x;inputsize=42) = Chain(
        Conv((4,4), inputsize=>42), 
        BatchNorm(42))(x)

    residual_block(x) = Chain(
        conv_block,
        IdentitySkip(conv_block2),
        (x)->leakyrelu.(x))(x)

    first_block(x) = Chain(
        residual_block,
        residual_block,
        residual_block,
        residual_block)(x)

    value_block(x) = Chain(
        Conv((1,1), 42=>1),
        BatchNorm(1),
        (x)->leakyrelu.(x),
        x -> reshape(x, :, size(x, 4)),
        Dense(20,20),
        (x)->leakyrelu.(x),
        Dense(20,1, tanh)
        )(x)

    policy_block(x) = Chain(
        Conv((1,1), 42=>2),
        BatchNorm(2),
        (x)->leakyrelu.(x),
        x -> reshape(x, :, size(x, 4)),
        Dense(42,8),
        softmax
        )(x)

        #@show size(conv_block(rand(6,7,42,1)))

    main_block = IdentitySkip( Conv((4,4), 1=>42, (x)->x ))

end


function EZero1()

    mask1 = [0., 0., 0., 0., 0., 0., 0., -1.0e10]
    mask2 = [false,false,false,false,false,false,false,true]

    mysoftmax = (x) -> softmax(x.+mask1) + mask2 .* x

    model = Chain(
        Conv((2, 2), 1=>10, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),

        Conv((2, 2), 10=>10, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),

        Conv((2, 2), 10=>10, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),

        Conv((2, 2), 10=>10, pad=(1,1), relu),
        x -> maxpool(x, (2,2)),
        # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
        # which is where we get the 288 in the `Dense` layer below:
        x -> reshape(x, :, size(x, 4)),
        Dense(10, 8),
        mysoftmax)
        #(x) -> [softmax(x[1:7]); x[8]])

    model2 = Flux.mapleaves(Flux.data, model)

    return EZero(model,model2)
end

function EZero2()

    mask1 = [0., 0., 0., 0., 0., 0., 0., -1.0e10]
    mask2 = [false,false,false,false,false,false,false,true]

    mysoftmax = (x) -> softmax(x.+mask1) + mask2 .* x

    model = Chain(
        #The board comes in as an image so it can be 
        #the first layer can be a Conv-layer, but in this 
        #the first layer is no conv-layer, so we must reahspe it
        x -> reshape(x, :, size(x, 4)),
        Dense(6*7,50),
        Dense(50,50),
        Dense(50,40),
        Dense(40,30),
        Dense(30,20),
        Dense(20,8),
        mysoftmax)
        #(x) -> [softmax(x[1:7]); x[8]])
    return EZero(model,model2)
end