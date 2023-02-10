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

train = Flux.DataLoader((trainX,trainY),batchsize=50,shuffle=true)
test = Flux.DataLoader((testX,testY),batchsize=50,shuffle=true)

x1,y1 = first(train)
x1 |> size 


model = Chain(
    LSTM(prod((28,28))=>150),
    RNN(150=>75),
    Dense(75=>10),
    softmax)


model(x1)
    
