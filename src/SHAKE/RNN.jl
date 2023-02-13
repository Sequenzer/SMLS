using Flux
using Flux: chunk, batchseq
using Base.Iterators: partition
using ProgressMeter
using Statistics
using Printf
using BSON
using Random: shuffle

#Length of a sequence
sequence_size=50;
#Size of batch_size
batch_size=250;

learning_rate=0.02;
test_percent=0.7

@info("Loading training data set")

#We want to transofrm the text into separated sequences. So we want a 
text = String(read(dirname(@__FILE__)*"/input.txt"));
alphabet=[[x for x in Set(text)]..., '_'];
stop = '_'

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

@info("Building model")

model=Chain(
    LSTM(L => 128),
    Dense(128=>L),
    softmax
)


@info("Start training")

function loss(m,x,y)
    Flux.reset!(m)
    y_hat = m(x)
    return crossentropy(y_hat, y)
end

#one cold find the index of the largest element of each column
accuracy(x,y)=mean(onecold(model(x)) .== onecold(y));


@info("Beginning training loop...")

best_acc = 0.0
last_improvement = 0

opt = Adam(learning_rate);

for epoch_id in 1:100
    global best_acc, last_improvement

    @showprogress for (batch,label) in zip(trainX, trainY)
        grad = gradient(model) do m 
            loss(m,batch, label);
        end
        Flux.Optimise.update!(Flux.setup(opt,model), model, grad[1])
    end

    acc = accuracy(testX,testY)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_id, acc))

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
