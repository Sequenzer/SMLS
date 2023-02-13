using MLDatasets: SVHN2, convert2image
using Images

using Plots
using Statistics
using Flux


dataset = SVHN2(:train)
testset = SVHN2(:test)

convert2image(dataset, 3)

dataset.targets
function toCategory(x)
    sol = zeros(10)
    sol[x]=1
    return sol
end


data = Flux.DataLoader((Flux.flatten(dataset.features),Flux.stack(toCategory.(dataset.targets))),batchsize=32,shuffle = true)
test = Flux.DataLoader((Flux.flatten(testset.features),Flux.stack(toCategory.(testset.targets))),shuffle = true)
x1, y1 = first(data)
x1

model = Chain(
Dense( 3072 => 1000,relu),
Dense( 1000 =>10),
softmax)

model(x1)
loss(x,y) = Flux.crossentropy(x,y)
loss(model(x1),y1)

acc(x,y) = model(x) .== y 

optimiser = Flux.setup(Flux.Adam(0.01),model)

for epoch in 1:10 
    train!(model,dataset,optimiser) do m, x, y
        loss(m(x),y)
    end
    println(accuracy(testset...,model))
end
@show(accuracy(testset...,model))
