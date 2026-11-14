open Datatypes
open FlambdaInlineCallsAtSpec
open FlambdaLiftSpec
open FlambdaLoopifySpec
open FlambdaModel
open FlambdaRenameApartSpec
open FlambdaUnusedStaticsSpec
open FlambdaValidateKnowingSpec
open FlambdaValidateSpec
open ListDef

type inlining_choice = path list * calls_in_code

(** val inlined_in_turn : continuation list -> inlining_choice list -> expr ->
    expr **)

let rec inlined_in_turn avoid choices e =
  match choices with
  | [] -> e
  | i :: choices0 ->
    let sites, k = i in
    inlined_in_turn avoid choices0
      (rename_apart_avoiding avoid (inline_calls_at sites k e))

(** val before_inlined : inlining_choice list -> loopified -> lifting list ->
    expr -> expr **)

let before_inlined choices g ls before =
  inlined_in_turn (app (unit_continuations ls) (map snd g)) choices before

(** val validate_unit_inlining : inlining_choice list -> nat -> dropped ->
    value_slot list -> loopified -> lifting list -> code_id list -> expr -> expr
    -> validation **)

let validate_unit_inlining choices rounds unused slots g ls kept before after =
  validate_unit rounds unused slots g ls kept
    (before_inlined choices g ls before)
    after

(** val validate_unit_knowing_inlining : inlining_choice list -> nat -> dropped
    -> value_slot list -> loopified -> lifting list -> code_id list -> expr ->
    expr -> validation **)

let validate_unit_knowing_inlining choices rounds unused slots g ls kept before
    after =
  validate_unit_knowing rounds unused slots g ls kept
    (before_inlined choices g ls before)
    after
