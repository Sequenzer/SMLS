using MLDatasets: MNIST

using Flux
using Logging
using TensorBoardLogger
using ProgressMeter
using MLJ

using BSON: @save,@load

using Plots
##using CUDA ##For Nvidia gpu
##using AMDGPU ##For AMD gpu

rawtrain = MNIST(:train)
rawtest = MNIST(:test)

trainY = Flux.onehotbatch(rawtrain.targets,0:9)
trainX = reshape(Flux.unsqueeze(rawtrain.features,dims=2),(28,28,1,60000)) 

testY = Flux.onehotbatch(rawtest.targets,0:9)
testX =reshape(Flux.unsqueeze(rawtest.features,dims=2),(28,28,1,10000)) 

train = Flux.DataLoader((trainX,trainY),batchsize=50)
test = Flux.DataLoader((testX,testY),batchsize=50)

x1,y1 = first(train)
x1 |> size 

model = Chain(
    Conv((4,4),1=>25, sigmoid),
    MaxPool((4,4),),
    Conv((3,3),25=>15, sigmoid),
    MaxPool((3,3),),
    Flux.flatten,
    Dense(15=>10),
    softmax)

model(x1)

#Initialize TensorBoardLogger

lg = TBLogger("src/MNIST/tensorboard_logs/run",min_level=Logging.Info)

using Random

struct sample_struct first_field; other_field; end

with_logger(lg) do
    for i=1:100
        x0          = 0.5+i/30; s0 = 0.5/(i/20);
        edges       = collect(-5:0.1:5)
        centers     = collect(edges[1:end-1] .+0.05)
        histvals    = [exp(-((c-x0)/s0)^2) for c=centers]
        data_tuple  = (edges, histvals)
        data_struct = sample_struct(i^2, i^1.5-0.3*i)


        @info "test" i=i j=i^2 dd=rand(10).+0.1*i hh=data_tuple
        @info "test_2" i=i j=2^i hh=data_tuple log_step_increment=0
        @info "" my_weird_struct=data_struct   log_step_increment=0
        @debug "debug_msg" this_wont_show_up=i
    end
end











#Training Section uncomment for futur
optimiser = Flux.setup(Flux.Adam(0.01),model)

loss(y_hat,y) = Flux.Losses.crossentropy(y_hat,y) 

losses=[]
@showprogress for epoch in 1:100
    for (x,y) in train
        currloss,grads = Flux.withgradient(model) do m
            y_hat = m(x)
            loss(y_hat,y)
        end
        Flux.update!(optimiser,model,grads[1])
        push!(losses,currloss)
    end
end

#@save "MNIST_CNN_100ep.bson" model

#@load "MNIST_CNN_100ep.bson" model


#accuracy(Flux.onecold(model(testX)),Flux.onecold(testY))
