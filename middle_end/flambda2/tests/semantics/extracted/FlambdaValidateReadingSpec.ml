open Datatypes
open FlambdaArrayLoadSpec
open FlambdaCompareSpec
open FlambdaGuardScopeSpec
open FlambdaLiftSpec
open FlambdaLoopifySpec
open FlambdaModel
open FlambdaNeverReturnsSpec
open FlambdaNormaliseSpec
open FlambdaRecHandlersSpec
open FlambdaSimplifyCarriedSpec
open FlambdaStubCallsSpec
open FlambdaUnusedStaticsSpec
open FlambdaValidateKnowingSpec
open FlambdaValidateSpec
open ListDef

(** val read_arrays_in_code : expr -> expr **)

let read_arrays_in_code body =
  rewrite_rec_handlers load_known_arrays (load_known_arrays body)

(** val read_arrays : expr -> expr **)

let read_arrays e = rewrite_code read_arrays_in_code (read_arrays_in_code e)

(** val normalise_before_avoiding : continuation list -> nat -> code_id list ->
    expr -> expr **)

let normalise_before_avoiding avoid rounds kept e =
  demote_and_tidy rounds
    (iterate rounds
       (normalise_before_round_avoiding avoid kept)
       (guard_scope simplify_by_types_carried
          (iterate rounds
             (normalise_before_round_avoiding avoid kept)
             (iterate rounds normalise_params_round e))))

(** val reading_round : continuation list -> nat -> loopified -> code_id list ->
    expr -> expr **)

let reading_round avoid rounds g kept e =
  let normalised =
    normalise_before_avoiding avoid rounds kept
      (loopify_unit g (cut_returns [] (demote_stub_calls e)))
  in
  let read = read_arrays normalised in
  if expr_eqb [] [] normalised read then e else read

(** val normalise_reading : nat -> continuation list -> loopified -> code_id
    list -> expr -> expr **)

let normalise_reading rounds avoid g kept e =
  iterate rounds (reading_round avoid rounds g kept) e

(** val before_read : nat -> loopified -> lifting list -> code_id list -> expr
    -> expr **)

let before_read rounds g ls kept before =
  normalise_reading rounds
    (app (unit_continuations ls) (map snd g))
    g kept before

(** val validate_unit_reading : nat -> dropped -> value_slot list -> loopified
    -> lifting list -> code_id list -> expr -> expr -> validation **)

let validate_unit_reading rounds unused slots g ls kept before after =
  validate_unit rounds unused slots g ls kept
    (before_read rounds g ls kept before)
    after

(** val validate_unit_knowing_reading : nat -> dropped -> value_slot list ->
    loopified -> lifting list -> code_id list -> expr -> expr -> validation **)

let validate_unit_knowing_reading rounds unused slots g ls kept before after =
  validate_unit_knowing rounds unused slots g ls kept
    (before_read rounds g ls kept before)
    after
