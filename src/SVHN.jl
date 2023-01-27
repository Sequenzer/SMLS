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
    sol = zeros(11)
    sol[x+1]=1
    return sol
end

toCategory.(dataset.targets)

data = Flux.DataLoader((Flux.flatten(dataset.features),Flux.stack(toCategory.(dataset.targets))),batchsize=32,shuffle = true)
x1, y1 = first(data)
x1