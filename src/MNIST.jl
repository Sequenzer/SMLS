using MLDatasets: MNIST, convert2image
using Images

dataset = MNIST(:train)
testset = MNIST(:test)

##convert2image(dataset, 1)

using Plots
using Statistics
using Flux

##Gradients


f(x)=x^2

df(x)=gradient(f,x)[1]

nt = (a=[2,1],b=[2,0],c=tanh)


##Calculate 
g(x::NamedTuple) = sum(abs2,x.a .- x.b)

g(nt)

dg_nt = gradient(g,nt)[1]

## Build model

layers= [Dense(10=>5, σ),Dense(5=>2),softmax]

#How its stacked under the hood
models(x)= foldl((x,m)->m(x),layers,init=x)


model2 = Chain(
    Dense(10=>5, σ),
    Dense(5=>2),
    softmax)

model2(rand(10))

opt_state= Flux.setup(Adam(),model2)

X = rand(28, 28, 60_000);  # many images, each 28 × 28
Y = rand(10, 60_000)
data2 = zip(eachslice(X; dims=3), eachcol(Y))

data2 = Flux.DataLoader((X, Y), batchsize=32)

x1, y1 = first(data2)
size(x1)

##Back to MNIST

datazip= zip(eachslice(dataset.features,dims=3),eachcol(dataset.targets))

function toCategory(x)
    sol = zeros(10)
    sol[x+1]=1
    return sol
end

data = Flux.DataLoader((Flux.flatten(dataset.features),Flux.stack(toCategory.(dataset.targets))),batchsize=32,shuffle = true)
x1, y1 = first(data)
test = Flux.DataLoader((Flux.flatten(testset.features),Flux.stack(toCategory.(testset.targets))),shuffle = true)


size(x1)
size(y1)
y1




toCategory(4)
    
reshape(x1,(28*28,32))

model = Chain(
    Dense(784=>20,σ),
    Dense(20,10),
    softmax)

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