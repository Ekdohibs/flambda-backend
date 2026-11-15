open Datatypes
open FlambdaCseSpec
open FlambdaModel

val lengths_knowing : computed -> expr -> expr

val lengths_in_handler :
  computed -> continuation_handler -> continuation_handler

val fold_made_lengths : expr -> expr
