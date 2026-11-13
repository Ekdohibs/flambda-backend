open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaUnaliasParamsSpec
open List

val same_simple : simple -> simple -> bool

val is_plain : simple -> bool

val apply_cont_gives : continuation -> nat -> simple -> apply_cont -> bool

val all_give : continuation -> nat -> simple -> expr -> bool

val handler_all_give :
  continuation -> nat -> simple -> continuation_handler -> bool

val given_by : continuation -> nat -> apply_cont -> simple option option

val first_given : continuation -> nat -> expr -> simple option option

val handler_first_given :
  continuation -> nat -> continuation_handler -> simple option option

val bind_const_param :
  nat ->
  continuation ->
  continuation_handler ->
  expr ->
  (continuation_handler * expr) option

val bind_const_from :
  nat ->
  continuation ->
  continuation_handler ->
  expr ->
  continuation_handler * expr

val bind_const_params : expr -> expr
