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