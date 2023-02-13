using MLDatasets: PTBLM

using Flux
##using CUDA ##For Nvidia gpu
##using AMDGPU ##For AMD gpu

using MLDatasets: UD_English 

rawtrain = UD_English(:train)

func = arr->arr[2]
train = reduce(vcat,map(x->func.(x),rawtrain))
Set(train)

ngram(join(train[1:100]," "),2)


#Thanks Internet

ngram(s,n)=[view(s,i:nextind(s,i,1)) for i in eachindex(s)]

SubString("héllo",2:3)
ngram("h😄llo ",2)
collect.("hello")
