using MLDatasets: MNIST

using Flux
using ProgressMeter
using MLJ

using BSON: @save,@load

using Plots
##using CUDA ##For Nvidia gpu
##using AMDGPU ##For AMD gpu

rawtrain = MNIST(:train)
rawtest = MNIST(:test)

trainY = Flux.onehotbatch(rawtrain.targets,0:9)
trainX = Flux.flatten(rawtrain.features)

testY = Flux.onehotbatch(rawtest.targets,0:9)
testX = Flux.flatten(rawtest.features)

train = Flux.DataLoader((trainX,trainY),batchsize=50)
test = Flux.DataLoader((testX,testY),batchsize=50)

x1,y1 = first(train)


model = Chain(
    Dense(prod((28,28))=> 100, sigmoid),
    Dense(100=> 50, sigmoid),
    Dense(50=>10),
    softmax)

#Training Section uncomment for futuru
#optimiser = Flux.setup(Flux.Adam(0.01),model)
#
#loss(y_hat,y) = Flux.Losses.crossentropy(y_hat,y) 
#
#losses=[]
#@showprogress for epoch in 1:100
#    for (x,y) in train
#        currloss,grads = Flux.withgradient(model) do m
#            y_hat = m(x)
#            loss(y_hat,y)
#        end
#        Flux.update!(optimiser,model,grads[1])
#        push!(losses,currloss)
#    end
#end
#
#@save "MNIST_BASIC_100ep.bson" model
#Training Statistics
@load "MNIST_BASIC_100ep.bson" model

#comes from the MLJ package
accuracy(Flux.onecold(model(testX)),Flux.onecold(testY))
