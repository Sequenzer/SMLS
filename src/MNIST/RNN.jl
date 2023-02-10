using MLDatasets: MNIST

using Flux

using CUDA ##For Nvidia gpu
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
    softmax) |> gpu


model(x1)

x1 |> gpu
