open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open List
open ListDef

val shareable : prim -> bool

type computed = (variable * (prim * simple list)) list

val forget_computed : variable -> computed -> computed

val forget_computed_vars : variable list -> computed -> computed

val tagged_again : variable -> prim -> simple list -> computed

val computed_under_let : computed -> variable -> named -> computed

val array_made_under_let : variable -> named -> computed

val computed_or_made_under_let : computed -> variable -> named -> computed

val same_simples : simple list -> simple list -> bool

val computed_by : computed -> prim -> simple list -> variable option

val made_length : computed -> variable -> coq_Z option

val made_length_named : computed -> named -> named option

val share_named : computed -> named -> named

val share_primitives_knowing : computed -> expr -> expr

val share_primitives_in_handler :
  computed -> continuation_handler -> continuation_handler

val share_known_primitives : expr -> expr
