open Datatypes
open FlambdaBindersSpec
open FlambdaCompareSpec
open FlambdaFloatCodeSpec
open FlambdaImportsSpec
open FlambdaKnownSlotsSpec
open FlambdaLiftObservedSpec
open FlambdaLiftSpec
open FlambdaLoopifySpec
open FlambdaModel
open FlambdaNeverReturnsSpec
open FlambdaStaticsOrderSpec
open FlambdaStubCallsSpec
open FlambdaUnusedStaticsSpec
open FlambdaValidateImportsSpec
open FlambdaValidateSpec
open FlambdaValueSpec

(** val lifted_in_turn_observed : nat -> continuation list -> lifting list ->
    code_id list -> expr -> expr **)

let rec lifted_in_turn_observed rounds avoid ls kept e =
  match ls with
  | [] -> demote_and_tidy_known rounds (read_known_slots_anywhere e)
  | l :: ls0 ->
    lifted_in_turn_observed rounds avoid ls0 kept
      (normalise_lifted_own l.made_closures_called l.own_closures_known rounds
         avoid kept
         (with_lifting_observed l e))

(** val lifting_observed_free_in : static_env -> lifting -> expr -> bool **)

let lifting_observed_free_in statics l e =
  let at_head = to_head_observed l e in
  let out_of_code = out_of_code_observed l at_head in
  let lifted = lets_observed l out_of_code in
  ((static_names_free_in statics at_head
   && static_names_free_in statics out_of_code)
  && static_names_free_in statics lifted)
  && static_names_free_in statics (with_lifting_observed l e)

(** val lifted_observed_free_in : static_env -> nat -> continuation list ->
    lifting list -> code_id list -> expr -> bool **)

let rec lifted_observed_free_in statics rounds avoid ls kept e =
  match ls with
  | [] ->
    demote_and_tidy_known_free_in statics rounds (read_known_slots_anywhere e)
  | l :: ls0 ->
    (lifting_observed_free_in statics l e
    && normalise_lifted_own_free_in statics l.made_closures_called
         l.own_closures_known rounds avoid kept
         (with_lifting_observed l e))
    && lifted_observed_free_in statics rounds avoid ls0 kept
         (normalise_lifted_own l.made_closures_called l.own_closures_known
            rounds avoid kept
            (with_lifting_observed l e))

type related_validation =
  | Related_once_lifted
  | Not_related

(** val validate_unit_related : (symbol * import_class) list -> nat -> dropped
    -> value_slot list -> loopified -> lifting list -> code_id list -> expr ->
    expr -> related_validation **)

let validate_unit_related imports rounds unused slots g ls kept before after =
  let statics = imports_statics imports in
  let source = loopify_unit g (cut_returns [] (demote_stub_calls before)) in
  let tidied = tidied_after rounds (normalise_after rounds kept after) in
  let taken = negb (keeps_a_return tidied) in
  let target = without_head_statics unused tidied in
  let lifted =
    without_statics unused
      (without_slots slots
         (tidied_before rounds
            (with_returns taken
               (lifted_in_turn_observed rounds (unit_continuations ls) ls kept
                  source))))
  in
  if
    (((code_closed before && code_closed after)
     && head_removal_free_in statics unused tidied)
    && lifted_observed_free_in statics rounds (unit_continuations ls) ls kept
         source)
    && (expr_eqb [] []
          (order_statics_as
             (float_code (with_code_first target))
             (float_code lifted))
          (float_code (with_code_first target))
       || (statics_first_free_in statics ls target
          && statics_first_free_in statics ls lifted)
          && expr_eqb [] []
               (order_statics_as
                  (with_statics_first ls target)
                  (with_statics_first ls lifted))
               (with_statics_first ls target))
  then Related_once_lifted
  else Not_related
