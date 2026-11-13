open BinNums
open FlambdaModel
open FlambdaPrimitiveModel

val same_answer : equality_comparison -> const

val different_answer : equality_comparison -> const

val fold_same_comparison : named -> named

val fold_same_comparisons : expr -> expr
