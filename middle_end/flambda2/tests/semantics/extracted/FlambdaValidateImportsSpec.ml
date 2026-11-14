open Datatypes
open FlambdaCompareSpec
open FlambdaFloatCodeSpec
open FlambdaGuardScopeSpec
open FlambdaHoistStaticsSpec
open FlambdaImportsSpec
open FlambdaKnownClosuresSpec
open FlambdaKnownSlotsSpec
open FlambdaKnownSymbolsSpec
open FlambdaLiftAroundSpec
open FlambdaLiftFromArmsSpec
open FlambdaLiftFromCodeSpec
open FlambdaLiftSpec
open FlambdaLoopifySpec
open FlambdaModel
open FlambdaNeverReturnsSpec
open FlambdaSharedReadsSpec
open FlambdaSimplifyCarriedSpec
open FlambdaStaticsOrderSpec
open FlambdaStubCallsSpec
open FlambdaUnusedHeadStaticsSpec
open FlambdaUnusedStaticsSpec
open FlambdaValidateKnowingSpec
open FlambdaValidateSpec
open FlambdaValueSpec
open ListDef

(** val holds_from : static_env -> nat -> code_id list -> expr -> validation ->
    bool **)

let holds_from statics rounds kept after = function
  | Same_once_head_statics_removed ->
    head_free_in statics
      (tidied_after rounds (normalise_after rounds kept after))
  | Same_once_lifted -> false
  | Not_validated -> false
  | _ -> true

(** val made_free_in : static_env -> expr option -> bool **)

let made_free_in statics = function
  | Some e' -> static_names_free_in statics e'
  | None -> true

(** val tables_free_in : static_env -> lifting -> expr -> bool **)

let tables_free_in statics l e =
  match l.tables_put with
  | [] -> true
  | _ :: _ -> head_free_in statics (add_head_tables l.tables_put e)

(** val lifting_free_in : static_env -> lifting -> expr -> bool **)

let lifting_free_in statics l e =
  let to_head = lift_to_head_unit l.lifting_unit l.lifted_to_head e in
  let e0 = match to_head with Some e' -> e' | None -> e in
  let from_code = lift_from_code_unit l.lifting_unit l.lifted_out_of_code e0 in
  let e1 = match from_code with Some e' -> e' | None -> e0 in
  let where_bound = lift_unit l e1 in
  let e2 = match where_bound with Some e' -> e' | None -> e1 in
  let around =
    lift_around_unit l.lifting_unit l.lifted_around l.groups_put e2
  in
  let e3 = match around with Some e' -> e' | None -> e2 in
  (((made_free_in statics to_head && made_free_in statics from_code)
   && made_free_in statics where_bound)
  && made_free_in statics around)
  && tables_free_in statics l e3

(** val known_free_in : static_env -> nat -> continuation list -> code_id list
    -> expr -> bool **)

let rec known_free_in statics rounds avoid kept e =
  match rounds with
  | O -> true
  | S rounds0 -> (
    let given =
      use_known_closures (normalise_before_round_avoiding avoid kept e)
    in
    let rest =
     fun e' ->
      if expr_eqb [] [] e e'
      then true
      else known_free_in statics rounds0 avoid kept e'
    in
    match fold_known_unit given with
    | Some folded ->
      static_names_free_in statics given
      && rest (share_static_reads (read_known_slots folded))
    | None -> rest (share_static_reads (read_known_slots given)))

(** val known_calling_free_in : static_env -> bool -> nat -> continuation list
    -> code_id list -> expr -> bool **)

let rec known_calling_free_in statics calling rounds avoid kept e =
  match rounds with
  | O -> true
  | S rounds0 ->
    known_free_in statics (S O) avoid kept e
    &&
    let e' = normalise_lifted_round_calling calling avoid kept e in
    if expr_eqb [] [] e e'
    then true
    else known_calling_free_in statics calling rounds0 avoid kept e'

(** val normalise_lifted_calling_free_in : static_env -> bool -> nat ->
    continuation list -> code_id list -> expr -> bool **)

let normalise_lifted_calling_free_in statics calling rounds avoid kept e =
  let e0 = iterate rounds first_phase_round e in
  known_calling_free_in statics calling rounds avoid kept e0
  && known_calling_free_in statics calling rounds avoid kept
       (guard_scope simplify_by_types_carried
          (iterate rounds
             (normalise_lifted_round_calling calling avoid kept)
             e0))

(** val tidy_known_free_in : static_env -> nat -> nat -> expr -> bool **)

let rec tidy_known_free_in statics inner rounds e =
  match rounds with
  | O -> true
  | S rounds0 -> (
    let given = use_known_closures e in
    let rest =
     fun e' ->
      if expr_eqb [] [] e e'
      then true
      else tidy_known_free_in statics inner rounds0 e'
    in
    match fold_known_unit given with
    | Some folded ->
      static_names_free_in statics given
      && rest
           (demote_and_tidy inner
              (share_static_reads (read_known_slots folded)))
    | None ->
      rest (demote_and_tidy inner (share_static_reads (read_known_slots given)))
    )

(** val demote_and_tidy_known_free_in : static_env -> nat -> expr -> bool **)

let demote_and_tidy_known_free_in statics rounds e =
  tidy_known_free_in statics rounds rounds (demote_and_tidy rounds e)

(** val lifted_free_in : static_env -> nat -> continuation list -> lifting list
    -> code_id list -> expr -> bool **)

let rec lifted_free_in statics rounds avoid ls kept e =
  match ls with
  | [] ->
    demote_and_tidy_known_free_in statics rounds (read_known_slots_anywhere e)
  | l :: ls0 ->
    (lifting_free_in statics l e
    && normalise_lifted_calling_free_in statics l.made_closures_called rounds
         avoid kept (with_lifting l e))
    && lifted_free_in statics rounds avoid ls0 kept
         (normalise_lifted_calling l.made_closures_called rounds avoid kept
            (with_lifting l e))

(** val statics_first_free_in : static_env -> lifting list -> expr -> bool **)

let statics_first_free_in statics ls e =
  match ls with
  | [] -> true
  | l :: _ -> (
    match hoist_statics l.lifting_unit e with
    | Some _ -> static_names_free_in statics e
    | None -> true)

(** val head_removal_free_in : static_env -> dropped -> expr -> bool **)

let head_removal_free_in statics d e =
  match drop_head_statics d e with
  | Some _ -> head_free_in statics e
  | None -> true

(** val lifted_holds_from : static_env -> nat -> dropped -> value_slot list ->
    loopified -> lifting list -> code_id list -> expr -> expr -> bool **)

let lifted_holds_from statics rounds unused slots g ls kept before after =
  let before0 = loopify_unit g (cut_returns [] (demote_stub_calls before)) in
  let tidied = tidied_after rounds (normalise_after rounds kept after) in
  let taken = negb (keeps_a_return tidied) in
  let after0 = without_head_statics unused tidied in
  let lifted =
    without_statics unused
      (without_slots slots
         (tidied_before rounds
            (with_returns taken
               (lifted_in_turn rounds (unit_continuations ls) ls kept before0))))
  in
  (head_removal_free_in statics unused tidied
  && lifted_free_in statics rounds (unit_continuations ls) ls kept before0)
  && (expr_eqb [] []
        (order_statics_as
           (float_code (with_code_first after0))
           (float_code lifted))
        (float_code (with_code_first after0))
     || statics_first_free_in statics ls after0
        && statics_first_free_in statics ls lifted)

(** val verdict_holds_from : static_env -> nat -> dropped -> value_slot list ->
    loopified -> lifting list -> code_id list -> expr -> expr -> validation ->
    bool **)

let verdict_holds_from statics rounds unused slots g ls kept before after v =
  match v with
  | Same_once_lifted ->
    lifted_holds_from statics rounds unused slots g ls kept before after
  | _ -> holds_from statics rounds kept after v

(** val verdict_knowing_holds_from : static_env -> nat -> dropped -> value_slot
    list -> loopified -> lifting list -> code_id list -> expr -> expr ->
    validation -> bool **)

let verdict_knowing_holds_from statics rounds unused slots g ls kept before
    after v =
  verdict_holds_from statics rounds unused slots g ls kept
    (normalise_knowing
       (normalise_before_round_avoiding
          (app (unit_continuations ls) (map snd g))
          kept)
       rounds before)
    (normalise_knowing (normalise_after_round kept) rounds after)
    v
