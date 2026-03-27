module test_bytepairencoding_learn

using Test
using BytePairEncoding: BytePairEncoding as BPE, TextEncodeBase as TEB
using .BPE: BPETokenization, NoBPE, Merge, GPT2Tokenization, learn, rank2list

# from BytePairEncoding.jl/test/test_learn.jl
#                               test_bpe.jl
#                               test_bbpe.jl

BPE.BPETokenization
BPE.NoBPE()
BPE.Merge("a")
BPE.learn
BPE.rank2list

sentence = TEB.Sentence("hello world")
@test sentence.x == "hello world"
@test sentence.meta === nothing

end # module test_bytepairencoding_learn
