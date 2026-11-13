open BinNums
open BinPos
open Datatypes
open FlambdaCodeLetsSpec
open FlambdaForwardKindedSpec
open FlambdaForwardSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaSemanticsSpec
open List
open ListDef
open PeanoNat

val position_of : variable -> variable list -> nat option

type arm_entry =
  (((variable list * handler_metadata) * nat) * (coq_Z * apply_cont) list)
  * expr

type arm_forwards = (continuation * arm_entry option) list

val arm_entry_of :
  bool -> variable list -> handler_metadata -> expr -> expr -> arm_entry option

val arm_of_apply_cont :
  arm_forwards ->
  apply_cont ->
  ((variable list * flambda_kind list) * apply_cont) option

val forward_to_arm_apply_cont :
  bool -> kinds_in_scope -> arm_forwards -> apply_cont -> apply_cont

val forward_to_arms_under :
  bool -> kinds_in_scope -> arm_forwards -> expr -> expr

val forward_to_arms_in_handler :
  bool ->
  kinds_in_scope ->
  arm_forwards ->
  continuation_handler ->
  continuation_handler

val forward_to_arms : expr -> expr

val forward_to_arms_kinded : expr -> expr

val forward_to_arms_in_binding : static_binding -> static_binding

val forward_to_arms_in_code : expr -> expr

val forward_to_arms_in_code_handler :
  continuation_handler -> continuation_handler
