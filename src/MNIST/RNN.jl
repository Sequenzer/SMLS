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
trainX = reshape(Flux.unsqueeze(rawtrain.features,dims=2),(28,28,1,60000)) 

testY = Flux.onehotbatch(rawtest.targets,0:9)
testX =reshape(Flux.unsqueeze(rawtest.features,dims=2),(28,28,1,10000)) 

train = Flux.DataLoader((trainX,trainY),batchsize=50)
test = Flux.DataLoader((testX,testY),batchsize=50)

x1,y1 = first(train)
x1 |> size 
    
#Input: image patch + location
#Output: Convolution of location and Glimpse


G_image=Chain(
    Conv((3,3),1=>3),
    Conv((3,3),3=>3),
    Conv((3,3),3=>1),
    Flux.flatten,
    Dense(484=>5)
)
G_loc=Chain(
    Dense(2=>5)
)

G_loc([1,2])



struct mixIn 
    image
    location
    mixIn(image,location) = new(image,location)
end
(m::mixIn)(x,l) = m.image(x).*m.location(l)
Flux.@functor mixIn


glimpse = mixIn(G_image,G_loc)
gl=glimpse(x1,[2,2])


recc = Chain(
    LSTM(5=>10),
    LSTM(10=>15)
)

emission=Dense(15=>2)

optimiser = Flux.setup(Flux.Adam(0.01),model)

for (i,data) in enumerate(train)
    input,label = data

    val, grads = Flux.withgradient(context) do m
        initial location 
end

    
