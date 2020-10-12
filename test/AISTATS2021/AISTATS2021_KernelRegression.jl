using CompressedSensing
using CompressedSensing: sbl, rmp, rmps

using Kernel


################# Import Data

################## Relevance Vector Machine ####################
l = .1 # length scale
k = Kernel.NeuralNetwork(l)
K = Kernel.gramian(k, x)
K = Matrix(K)

wrmp = rmp(K, y) # run RelevanceMatchingPursuit
wsbl = sbl(K, y)
