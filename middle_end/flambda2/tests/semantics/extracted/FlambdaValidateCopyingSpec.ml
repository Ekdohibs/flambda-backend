open Datatypes
open FlambdaCopyLetsSpec
open FlambdaLiftSpec
open FlambdaLoopifySpec
open FlambdaModel
open FlambdaNeverReturnsSpec
open FlambdaStubCallsSpec
open FlambdaUnusedStaticsSpec
open FlambdaValidateImportsSpec
open FlambdaValidateKnowingSpec
open FlambdaValidateSpec
open FlambdaValueSpec
open ListDef

(** val lifted_steps : nat -> continuation list -> lifting list -> code_id list
    -> expr -> expr **)

let rec lifted_steps rounds avoid ls kept e =
  match ls with
  | [] -> e
  | l :: ls0 ->
    lifted_steps rounds avoid ls0 kept
      (normalise_lifted_own l.made_closures_called l.own_closures_known rounds
         avoid kept (with_lifting l e))

(** val copied_avoiding : loopified -> lifting list -> lifting -> lifting list
    -> continuation list **)

let copied_avoiding g ls l ls' =
  app (unit_continuations (app ls (l :: ls'))) (map snd g)

(** val before_copied : nat -> loopified -> lifting list -> lifting -> let_copy
    list -> lifting list -> code_id list -> expr -> expr **)

let before_copied rounds g ls l copies ls' kept before =
  with_let_copies copies
    (with_lifting l
       (lifted_steps rounds
          (copied_avoiding g ls l ls')
          ls kept
          (loopify_unit g (cut_returns [] (demote_stub_calls before)))))

(** val validate_unit_copying : nat -> dropped -> value_slot list -> loopified
    -> lifting list -> lifting -> let_copy list -> lifting list -> code_id list
    -> expr -> expr -> validation **)

let validate_unit_copying rounds unused slots g ls l copies ls' kept before
    after =
  validate_unit rounds unused slots g ls' kept
    (before_copied rounds g ls l copies ls' kept before)
    after

(** val validate_unit_knowing_copying : nat -> dropped -> value_slot list ->
    loopified -> lifting list -> lifting -> let_copy list -> lifting list ->
    code_id list -> expr -> expr -> validation **)

let validate_unit_knowing_copying rounds unused slots g ls l copies ls' kept
    before after =
  validate_unit_knowing rounds unused slots g ls' kept
    (before_copied rounds g ls l copies ls' kept before)
    after

(** val lifted_steps_free_in : static_env -> nat -> continuation list -> lifting
    list -> code_id list -> expr -> bool **)

let rec lifted_steps_free_in statics rounds avoid ls kept e =
  match ls with
  | [] -> true
  | l :: ls0 ->
    (lifting_free_in statics l e
    && normalise_lifted_own_free_in statics l.made_closures_called
         l.own_closures_known rounds avoid kept (with_lifting l e))
    && lifted_steps_free_in statics rounds avoid ls0 kept
         (normalise_lifted_own l.made_closures_called l.own_closures_known
            rounds avoid kept (with_lifting l e))

(** val before_copied_holds_from : static_env -> nat -> loopified -> lifting
    list -> lifting -> lifting list -> code_id list -> expr -> bool **)

let before_copied_holds_from statics rounds g ls l ls' kept before =
  let avoid = copied_avoiding g ls l ls' in
  let looped = loopify_unit g (cut_returns [] (demote_stub_calls before)) in
  lifted_steps_free_in statics rounds avoid ls kept looped
  && lifting_free_in statics l (lifted_steps rounds avoid ls kept looped)
