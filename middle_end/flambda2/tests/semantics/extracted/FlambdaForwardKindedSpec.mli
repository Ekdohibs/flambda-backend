open Datatypes
open FlambdaCodeLetsSpec
open FlambdaForwardSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open ListDef
open PeanoNat

type cont_kinds = (continuation * flambda_kind list) list

val have_kinds : kinds_in_scope -> simple list -> flambda_kind list -> bool

val kinded_entry :
  continuation ->
  variable list ->
  handler_metadata ->
  expr ->
  expr ->
  (variable list * apply_cont) option

val forward_kinded_apply_cont :
  kinds_in_scope -> cont_kinds -> forwards -> apply_cont -> apply_cont

val forward_kinded_arm :
  kinds_in_scope -> cont_kinds -> forwards -> apply_cont -> apply_cont

val forward_kinded_under :
  kinds_in_scope -> cont_kinds -> forwards -> expr -> expr

val forward_kinded_in_handler :
  kinds_in_scope ->
  cont_kinds ->
  forwards ->
  continuation_handler ->
  continuation_handler

val forward_kinded : expr -> expr

val forward_kinded_in_binding : static_binding -> static_binding

val forward_kinded_in_code : expr -> expr

val forward_kinded_in_code_handler :
  continuation_handler -> continuation_handler
