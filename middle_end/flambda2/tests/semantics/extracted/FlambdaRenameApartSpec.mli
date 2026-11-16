open BinNums
open Datatypes
open FlambdaBindersSpec
open FlambdaCompareSpec
open FlambdaFreshQuantifierSpec
open FlambdaModel
open FlambdaNominalSpec
open List
open ListDef
open Nat

val binders : expr -> nat

val binders_in_handler : continuation_handler -> nat

val binders_in_binding : static_binding -> nat

val binders_in_code : code -> nat

val bind_variable : renaming -> variable -> variable -> renaming

val bind_continuation : renaming -> continuation -> continuation -> renaming

val new_variable : renaming -> positive list -> variable -> variable

val new_continuation : renaming -> positive list -> continuation -> continuation

val bind_variables :
  renaming ->
  positive list ->
  variable list ->
  (positive list * variable list) * renaming

val bind_continuations :
  renaming ->
  positive list ->
  continuation list ->
  (positive list * continuation list) * renaming

val thread :
  (positive list -> 'a1 -> positive list * 'a2) ->
  positive list ->
  'a1 list ->
  positive list * 'a2 list

val apart : renaming -> positive list -> expr -> positive list * expr

val apart_handler :
  renaming ->
  positive list ->
  continuation_handler ->
  positive list * continuation_handler

val apart_binding :
  renaming -> positive list -> static_binding -> positive list * static_binding

val apart_code : renaming -> positive list -> code -> positive list * code

val canonical_avoiding : continuation list -> expr -> expr

val rename_apart_avoiding : continuation list -> expr -> expr
