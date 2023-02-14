using Flux
using Flux: chunk, batchseq, logitcrossentropy, onecold, onehot
using Base.Iterators: partition
using ProgressBars
using Statistics
using Printf
using BSON
using Random: shuffle


function set_data(test_percent, sequence_size, batch_size)

    @info("Loading data set")

    #We want to transofrm the text into separated sequences. So we want a 
    text = String(read("input.txt"));
    alphabet=[[x for x in Set(text)]..., '_'];

    L=length(alphabet);

    #The chunk function divides the dataset into batch_size vectors of dimension N, which is way longer then the length of the sequence we want
    #The batchseq function just creates a vector of N elements each of size batch_size, adding '_' at the end of the last one.
    #Partition simply divides it into batches of dimension sequence_size each made of batch_size elements: this is our sequence to be processed.
    Xs = partition(batchseq(chunk(text, batch_size), '_'), sequence_size);
    Ys = partition(batchseq(chunk(text[2:end], batch_size), '_'), sequence_size);

    #Now we want a onehotbatch encoding where each batch is of size sequence_size x length(alphabet) x batch_size
    Xs = [Flux.onehotbatch.(bs, (alphabet,)) for bs in Xs];
    Ys = [Flux.onehotbatch.(bs, (alphabet,)) for bs in Ys];

    perm = shuffle(1:length(Xs));
    split = floor(Int, (1-test_percent) * length(Xs));

    trainX, trainY = Xs[perm[1:split]], Ys[perm[1:split]];
    testX,  testY =  Xs[perm[(split+1):end]], Ys[perm[(split+1):end]]; 

    return L,alphabet,trainX,trainY,testX,testY
end

function build_model(L) 
    @info("Building model")

    model=Chain(
        LSTM(L => 128),
        LSTM(128 => 128),
        Dense(128=>L),
    )
    return model
    
end

function train(; learning_rate=1e-2, nepochs=3,test_percentage=0.05, sequence_size=50, batch_size=50)

    L,alphabet,trainX,trainY,testX,testY=set_data(test_percentage,sequence_size,batch_size);
    model=build_model(L)
    
    function loss(m,sx,ys)
        Flux.reset!(m)
        return sum(logitcrossentropy.([m(x) for x in sx], ys))
    end

    function accuracy(m,x,y)
        Flux.reset!(m)
        return sum([sum(onecold(m(x[i])) .== onecold(y[i])) for i in 1:length(x)])
    end

    @info("Beginning training loop...")

    best_acc=0
    last_improvement = 0
    opt_state=Flux.setup(Adam(learning_rate),model)

    for epoch_id in 1:nepochs
        global best_acc, last_improvement

        @info "Training, epoch $(epoch_id) / $(nepochs)"
        Flux.train!(
            loss,
            model,
            zip(trainX, trainY),
            opt_state
        )

        #acc=400
        acc= sum(accuracy.(Ref(model), testX, testY))/ (batch_size * sequence_size * length(testX))
        @info(@sprintf("[%d]: Accuracy: %.4f", epoch_id, acc))

        # loss_test=(sum(loss.(Ref(model), testX, testY)) / (batch_size * sequence_size * length(testX)))
        #@info(@sprintf("[%d]: Test loss: %.4f", epoch_id, loss_test))

        @info "Human readable test"
        Flux.reset!(model)
        start="DUKE"
        print(start)
        char='D'
        for i in start[2:end]
            temp=Vector{Float32}(onehot(char,alphabet))
            model(temp)
        end
        for i in 1:50
            temp=Vector{Float32}(onehot(char,alphabet))
            char=alphabet[onecold(model(temp))]
            print(char)
        end
        print("\n")
        Flux.reset!(model)

        
        if acc >= 0.999
            @info(" Super accurate network")
            break
        end

        if acc>best_acc
            @info("New best!")
            BSON.@save "model.bson" model epoch_id acc
            best_acc=acc
            last_improvement=epoch_id
        end

        if epoch_id - last_improvement >= 5 && opt.eta > 1e-6
	    opt.eta /= 10.0
	    @warn(" -> Dropping learning rate to try and improve")	
	    last_improvement = epoch_id
        end
        if epoch_id - last_improvement >= 10
            @warn(" Hasn't improved in over 10 epochs, exiting")
            break
        end
    end

    return model, alphabet
end

train()
