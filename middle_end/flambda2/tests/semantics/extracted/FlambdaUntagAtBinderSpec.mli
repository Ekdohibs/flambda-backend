open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaSubkindOrderSpec
open FlambdaUntagParamsSpec
open List
open ListDef

val states_tagged : param_subkind -> bool

val tagged_params : variable list -> param_subkind list -> variable list

val untags : variable -> flambda_kind -> named -> bool

val untaggings : variable -> expr -> variable list

val untaggings_in_handler : variable -> continuation_handler -> variable list

val first_untagging : variable -> expr -> variable option

val drop_untagging : variable -> variable -> expr -> expr

val drop_untagging_in_handler :
  variable -> variable -> continuation_handler -> continuation_handler

val untagging_at_head : variable list -> variable -> expr -> expr

val returns_to : continuation -> expr -> bool

val returns_to_in_handler : continuation -> continuation_handler -> bool

val bind_untaggings : expr -> expr

val bind_untaggings_in_handler :
  bool -> continuation_handler -> continuation_handler
