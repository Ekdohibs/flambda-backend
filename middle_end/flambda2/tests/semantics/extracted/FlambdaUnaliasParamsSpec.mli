open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaSpineWalkSpec
open List
open ListDef

val drop_nth : nat -> 'a1 list -> 'a1 list

val first_answer : ('a1 -> 'a2 option) -> 'a1 list -> 'a2 option

val apply_cont_gives_var :
  continuation -> nat -> variable -> bool -> apply_cont -> bool

val all_give_var : continuation -> nat -> variable -> bool -> expr -> bool

val handler_all_give_var :
  continuation -> nat -> variable -> bool -> continuation_handler -> bool

val given_var_by : continuation -> nat -> apply_cont -> variable option option

val first_given_var : continuation -> nat -> expr -> variable option option

val handler_first_given_var :
  continuation -> nat -> continuation_handler -> variable option option

val drop_arg_of : continuation -> nat -> apply_cont -> apply_cont

val drop_arg : continuation -> nat -> expr -> expr

val drop_arg_in_handler :
  continuation -> nat -> continuation_handler -> continuation_handler

val unalias_param :
  nat ->
  continuation ->
  continuation_handler ->
  expr ->
  (continuation_handler * expr) option

val unalias_from :
  nat ->
  continuation ->
  continuation_handler ->
  expr ->
  continuation_handler * expr

val unalias_node : expr -> expr

val unalias_params : expr -> expr
