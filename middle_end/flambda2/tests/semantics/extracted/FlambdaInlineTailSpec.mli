open Datatypes
open FlambdaKindModel
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaStatedScopeSpec
open PeanoNat

val tail_args_stated : stated_in_scope -> param_subkind list -> expr -> bool

val inline_tail_conts_kinded : expr -> expr
