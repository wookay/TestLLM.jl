using Jive
@useinside Main module test_bytepairencoding_learn

# BytePairEncoding.jl/test/test_learn.jl
#                          test_bpe.jl
#                          test_bbpe.jl
using BytePairEncoding: BytePairEncoding, TextEncodeBase # modules
using .BytePairEncoding: BPETokenization, NoBPE, Merge, GPT2Tokenization, learn, rank2list
using .TextEncodeBase: FlatTokenizer, CodeNormalizer, Sentence

BPETokenization
NoBPE()
Merge("a")
learn
rank2list

Sentence("hello world")

end # module test_bytepairencoding_learn
