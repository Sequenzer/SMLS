using MLDatasets: MNIST, convert2image
using Images

dataset = MNIST(:train)
testset = MNIST(:test)

##convert2image(dataset, 1)

using Plots
using Statistics
using Flux


##Back to MNIST
Flux.onehotbatch(dataset.targets,0:9)
datazip= zip(eachslice(dataset.features,dims=3),eachcol(dataset.targets))

function toCategory(x)
    sol = zeros(10)
    sol[x+1]=1
    return sol
end



Flux.DataLoader((Flux.flatten(dataset.features),Flux.onehotbatch(dataset.targets,0:9)),:
data = Flux.DataLoader((Flux.flatten(dataset.features),Flux.stack(toCategory.(dataset.targets))),batchsize=32,shuffle = true)

x1, y1 = first(data)


test = Flux.DataLoader((Flux.flatten(testset.features),Flux.stack(toCategory.(testset.targets))),shuffle = true)


size(x1)
size(y1)
y1


toCategory(4)
    
reshape(x1,(28*28,32))

model = Chain(
    Conv((3, 3), 1 => 10, pad=(1, 1), relu))

    #x -> maxpool(x, (2, 2)),

    #x -> reshape(x, :, size(x, 3)), 
    #Dense(14 * 14 => 10),
    #softmax)

model(x1)


opti = Flux.setup(Flux.Adam(0.02),model)

Flux.train!(model,data,opti) do m,x,y
    Flux.crossentropy(m(x),y)
end

Flux.crossentropy(model(x1),y1)

loss=[]
for epoch in 1:10
    Flux.train!(model,data,opti) do m,x,y
        Flux.crossentropy(m(x),y)
    end
    push!(loss,mean([Flux.crossentropy(model(x),y) for (x,y) in test]))
end


loss

xt,yt = first(test)
xt
yt

convert2image(testset, 1)


using Flux

Flux.s
