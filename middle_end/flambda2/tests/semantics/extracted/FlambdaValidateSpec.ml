open Datatypes
open FlambdaAliasLetsSpec
open FlambdaAliasesSpec
open FlambdaArmNumbersSpec
open FlambdaArrayLoadSpec
open FlambdaBlockLoadSpec
open FlambdaCodeLetsSpec
open FlambdaCompareSpec
open FlambdaCseSpec
open FlambdaDeadParamsSpec
open FlambdaDemoteSpec
open FlambdaFloatCodeSpec
open FlambdaForwardKindedSpec
open FlambdaForwardSpec
open FlambdaGiveScrutineeSpec
open FlambdaGroupsUpSpec
open FlambdaGuardScopeSpec
open FlambdaHoistCodeSpec
open FlambdaHoistStaticsSpec
open FlambdaIdentitiesSpec
open FlambdaInlineSiblingsSpec
open FlambdaInlineTailSpec
open FlambdaKnownClosuresSpec
open FlambdaKnownSlotsSpec
open FlambdaKnownSymbolsSpec
open FlambdaLiftAroundSpec
open FlambdaLiftFromArmsSpec
open FlambdaLiftFromCodeSpec
open FlambdaLiftSpec
open FlambdaLoopAliasesSpec
open FlambdaLoopLetsSpec
open FlambdaLoopifySpec
open FlambdaMergeSwitchesSpec
open FlambdaMergeTablesSpec
open FlambdaModel
open FlambdaNeverReturnsSpec
open FlambdaNormaliseSpec
open FlambdaParamAliasesSpec
open FlambdaRecHandlersSpec
open FlambdaRecoverComparisonSpec
open FlambdaRenameApartSpec
open FlambdaSameComparisonSpec
open FlambdaSameHandlersSpec
open FlambdaShadowParamsSpec
open FlambdaShareProjectionsSpec
open FlambdaSharedReadsSpec
open FlambdaShortcutSpec
open FlambdaSimplifyCarriedSpec
open FlambdaSinkContsSpec
open FlambdaSinkUsesSpec
open FlambdaStaticsOrderSpec
open FlambdaStubCallsSpec
open FlambdaTwinParamsSpec
open FlambdaUnaliasParamsSpec
open FlambdaUnboxInScopeSpec
open FlambdaUnboxLoopsSpec
open FlambdaUnboxParamsSpec
open FlambdaUnboxSpec
open FlambdaUnboxTogetherSpec
open FlambdaUnboxVariantsSpec
open FlambdaUntagParamsSpec
open FlambdaUnusedContsSpec
open FlambdaUnusedHeadStaticsSpec
open FlambdaUnusedParamsSpec
open FlambdaUnusedProjectionsSpec
open FlambdaUnusedSlotsSpec
open FlambdaUnusedStaticsSpec
open FlambdaWellFormedSpec
open FlambdaWrapReturnsSpec
open List

(** val iterate : nat -> (expr -> expr) -> expr -> expr **)

let rec iterate rounds f e =
  match rounds with
  | O -> e
  | S rounds0 ->
    let e' = f e in
    if expr_eqb [] [] e e' then e' else iterate rounds0 f e'

(** val normalise_expr_before : expr -> expr **)

let normalise_expr_before e =
  let e0 =
    shortcut_switches
      (share_handlers
         (thread_switches
            (inline_tail_conts
               (sink_let_conts
                  (remove_unused_conts
                     (merge_switches (collapse_switches (fold_constants e))))))))
  in
  let e1 =
    apply_identities
      (follow_aliases_in_loops []
         (follow_aliases []
            (share_known_projections
               (share_known_primitives
                  (recover_known_comparisons
                     (fold_same_comparisons
                        (unbox_known_numbers
                           (load_known_arrays (load_known_blocks e0)))))))))
  in
  wrap_returns
    (drop_unused_params
       (remove_unused_defined_lets [] (remove_unused_lets (fold_constants e1))))

(** val normalise_code_before : expr -> expr **)

let normalise_code_before body =
  rewrite_rec_handlers normalise_expr_before (normalise_expr_before body)

(** val normalise_before_inlined : expr -> expr **)

let normalise_before_inlined e =
  rewrite_code normalise_code_before
    (merge_statics (drop_deleted_code (normalise_code_before e)))

(** val with_twins : continuation list -> expr -> expr **)

let with_twins avoid e =
  let shadowed =
    rename_apart_avoiding avoid (rewrite_code shadow_params (shadow_params e))
  in
  let twinned = rewrite_code twin_params (twin_params shadowed) in
  if expr_eqb [] [] shadowed twinned
  then e
  else normalise_before_inlined twinned

(** val normalise_before_round_avoiding : continuation list -> code_id list ->
    expr -> expr **)

let normalise_before_round_avoiding avoid kept e =
  let e' =
    normalise_before_inlined
      (rename_apart_avoiding avoid
         (inline_sibling_calls kept (inline_calls kept [] e)))
  in
  if expr_eqb [] [] e e' then with_twins avoid e' else e'

(** val normalise_before_round : code_id list -> expr -> expr **)

let normalise_before_round kept e = normalise_before_round_avoiding [] kept e

(** val normalise_expr_params_knowing : facts -> expr -> expr **)

let normalise_expr_params_knowing known e =
  let e0 =
    shortcut_switches
      (share_handlers
         (thread_switches
            (inline_tail_conts
               (sink_let_conts
                  (remove_unused_conts
                     (merge_switches (collapse_switches (fold_constants e))))))))
  in
  let e1 = remove_unused_conts (forward_applications e0) in
  let e2 =
    unbox_together_knowing known
      (untag_params
         (unbox_loops_knowing known
            (unbox_variants_knowing known
               (follow_aliases_in_loops []
                  (follow_aliases []
                     (unbox_known_numbers
                        (load_known_arrays (load_known_blocks e1))))))))
  in
  drop_dead_params
    (bind_aliased_params
       (remove_unused_defined_lets [] (remove_unused_lets (fold_constants e2))))

(** val normalise_code_params_knowing : facts -> expr -> expr **)

let normalise_code_params_knowing known body =
  normalise_expr_params_knowing known
    (rewrite_rec_handlers_in_scope normalise_expr_params_knowing known body)

(** val normalise_code_params : expr -> expr **)

let normalise_code_params body = normalise_code_params_knowing no_facts body

(** val normalise_params_round : expr -> expr **)

let normalise_params_round e =
  rewrite_code_in_scope normalise_code_params_knowing no_facts
    (normalise_code_params (merge_tables e))

(** val demote_and_tidy : nat -> expr -> expr **)

let demote_and_tidy rounds e =
  iterate rounds normalise_before_inlined
    (iterate rounds normalise_params_round (demote_handlers [] e))

(** val normalise_before : nat -> code_id list -> expr -> expr **)

let normalise_before rounds kept e =
  demote_and_tidy rounds
    (iterate rounds
       (normalise_before_round kept)
       (guard_scope simplify_by_types_carried
          (iterate rounds
             (normalise_before_round kept)
             (iterate rounds normalise_params_round e))))

(** val inline_once_applied_after : expr -> expr **)

let inline_once_applied_after e =
  remove_unused_conts
    (thread_switches
       (forward_kinded
          (inline_tail_conts_kinded (sink_let_conts (remove_unused_conts e)))))

(** val normalise_expr_after : expr -> expr **)

let normalise_expr_after e =
  remove_unused_defined_lets_in_loops []
    (remove_unused_projections
       (fold_constants
          (follow_aliases_in_loops []
             (follow_aliases []
                (merge_alias_lets (inline_once_applied_after e))))))

(** val normalise_code_after : expr -> expr **)

let normalise_code_after body =
  rewrite_rec_handlers normalise_expr_after (normalise_expr_after body)

(** val normalise_after_round : code_id list -> expr -> expr **)

let normalise_after_round kept e =
  remove_unused_defined_lets_in_code
    (forward_kinded_in_code
       (rewrite_code normalise_code_after
          (merge_statics
             (drop_deleted_code
                (rewrite_rec_handlers normalise_expr_after
                   (remove_unused_defined_lets_in_loops []
                      (remove_unused_projections
                         (fold_constants
                            (inline_known_calls kept [] []
                               (follow_aliases_in_loops []
                                  (follow_aliases []
                                     (merge_alias_lets
                                        (inline_once_applied_after e)))))))))))))

(** val normalise_after : nat -> code_id list -> expr -> expr **)

let normalise_after rounds kept e =
  iterate rounds (normalise_after_round kept) e

(** val without_statics : dropped -> expr -> expr **)

let without_statics d e =
  match drop_statics d e with
  | Some e' -> merge_statics (drop_deleted_code e')
  | None -> e

(** val without_head_statics : dropped -> expr -> expr **)

let without_head_statics d e =
  if head_statics_removable d e then without_statics d e else e

(** val with_lifting : lifting -> expr -> expr **)

let with_lifting l e =
  let e0 =
    match lift_to_head_unit l.lifting_unit l.lifted_to_head e with
    | Some e' -> e'
    | None -> e
  in
  let e1 =
    match lift_from_code_unit l.lifting_unit l.lifted_out_of_code e0 with
    | Some e' -> e'
    | None -> e0
  in
  let e2 = match lift_unit l e1 with Some e' -> e' | None -> e1 in
  let e3 =
    match lift_around_unit l.lifting_unit l.lifted_around l.groups_put e2 with
    | Some e' -> e'
    | None -> e2
  in
  add_head_tables l.tables_put e3

(** val with_known : expr -> expr **)

let with_known e = match fold_known_unit e with Some e' -> e' | None -> e

(** val normalise_lifted_round : continuation list -> code_id list -> expr ->
    expr **)

let normalise_lifted_round avoid kept e =
  share_static_reads
    (read_known_slots
       (with_known
          (use_known_closures (normalise_before_round_avoiding avoid kept e))))

(** val normalise_lifted : nat -> continuation list -> code_id list -> expr ->
    expr **)

let normalise_lifted rounds avoid kept e =
  iterate rounds
    (normalise_lifted_round avoid kept)
    (guard_scope simplify_by_types_carried
       (iterate rounds
          (normalise_lifted_round avoid kept)
          (iterate rounds normalise_params_round e)))

(** val unit_continuations : lifting list -> continuation list **)

let unit_continuations ls =
  flat_map (fun l -> [l.lifting_unit.unit_return; l.lifting_unit.unit_exn]) ls

(** val demote_and_tidy_known_round : nat -> expr -> expr **)

let demote_and_tidy_known_round rounds e =
  demote_and_tidy rounds
    (share_static_reads (read_known_slots (with_known (use_known_closures e))))

(** val demote_and_tidy_known : nat -> expr -> expr **)

let demote_and_tidy_known rounds e =
  iterate rounds (demote_and_tidy_known_round rounds) (demote_and_tidy rounds e)

(** val lifted_in_turn : nat -> continuation list -> lifting list -> code_id
    list -> expr -> expr **)

let rec lifted_in_turn rounds avoid ls kept e =
  match ls with
  | [] -> demote_and_tidy_known rounds (read_known_slots_anywhere e)
  | l :: ls0 ->
    lifted_in_turn rounds avoid ls0 kept
      (normalise_lifted rounds avoid kept (with_lifting l e))

(** val without_slots : value_slot list -> expr -> expr **)

let without_slots slots e =
  match drop_unused_slots slots e with
  | Some e' -> rewrite_code remove_unused_lets (remove_unused_lets e')
  | None -> e

(** val with_code_first : expr -> expr **)

let with_code_first e =
  match hoist_code e with
  | Some e' -> merge_statics (drop_deleted_code e')
  | None -> e

(** val with_statics_first : lifting list -> expr -> expr **)

let with_statics_first ls e =
  match ls with
  | [] -> e
  | l :: _ -> (
    match hoist_statics l.lifting_unit e with
    | Some e' -> merge_statics (drop_deleted_code e')
    | None -> e)

(** val place_conts : expr -> expr **)

let place_conts e =
  rewrite_rec_handlers sink_conts_to_uses (sink_conts_to_uses e)

(** val placed : expr -> expr **)

let placed e = rewrite_code place_conts (place_conts e)

(** val tidy_expr_before : expr -> expr **)

let tidy_expr_before e =
  remove_unused_defined_lets []
    (remove_unused_lets
       (follow_aliases_in_loops []
          (follow_aliases []
             (share_known_primitives
                (inline_tail_conts
                   (sink_let_conts
                      (remove_unused_conts
                         (merge_switches
                            (collapse_switches
                               (fold_constants
                                  (numbers_in_arms
                                     (unalias_params (give_scrutinees e)))))))))))))

(** val tidy_code_before : expr -> expr **)

let tidy_code_before body =
  rewrite_rec_handlers tidy_expr_before (tidy_expr_before body)

(** val tidy_before_round : expr -> expr **)

let tidy_before_round e = rewrite_code tidy_code_before (tidy_code_before e)

(** val tidied_before : nat -> expr -> expr **)

let tidied_before rounds e =
  groups_up (iterate rounds tidy_before_round (placed e))

(** val tidy_expr_after : expr -> expr **)

let tidy_expr_after e =
  remove_unused_defined_lets_in_loops []
    (follow_aliases_in_loops []
       (follow_aliases []
          (share_known_primitives
             (inline_tail_conts_kinded
                (sink_let_conts
                   (remove_unused_conts
                      (fold_constants
                         (numbers_in_arms (unalias_params (give_scrutinees e))))))))))

(** val tidy_code_after : expr -> expr **)

let tidy_code_after body =
  rewrite_rec_handlers tidy_expr_after (tidy_expr_after body)

(** val tidy_after_round : expr -> expr **)

let tidy_after_round e =
  remove_unused_defined_lets_in_code
    (rewrite_code tidy_code_after (tidy_code_after e))

(** val tidied_after : nat -> expr -> expr **)

let tidied_after rounds e =
  groups_up (iterate rounds tidy_after_round (placed e))

(** val take_returns_on : expr -> expr **)

let take_returns_on e =
  rewrite_rec_handlers remove_unused_conts
    (remove_unused_conts (forward_applications_with true e))

(** val returns_taken_on : expr -> expr **)

let returns_taken_on e = rewrite_code take_returns_on (take_returns_on e)

(** val forwarded_in_unit : bool -> expr -> expr **)

let forwarded_in_unit calls e =
  rewrite_code
    (forward_applications_with calls)
    (forward_applications_with calls e)

(** val keeps_a_return : expr -> bool **)

let keeps_a_return e =
  negb (expr_eqb [] [] (forwarded_in_unit true e) (forwarded_in_unit false e))

(** val with_returns : bool -> expr -> expr **)

let with_returns taken e = if taken then returns_taken_on e else e

type validation =
  | Same_up_to_bound_names
  | Same_once_normalised
  | Same_once_unused_statics_removed
  | Same_once_head_statics_removed
  | Same_once_lifted
  | Not_validated

(** val validate_unit : nat -> dropped -> value_slot list -> loopified ->
    lifting list -> code_id list -> expr -> expr -> validation **)

let validate_unit rounds unused slots g ls kept before after =
  if expr_eqb [] [] before after
  then Same_up_to_bound_names
  else
    let before0 = loopify_unit g (cut_returns [] (demote_stub_calls before)) in
    let after0 = tidied_after rounds (normalise_after rounds kept after) in
    let taken = negb (keeps_a_return after0) in
    let normalised =
      tidied_before rounds
        (with_returns taken (normalise_before rounds kept before0))
    in
    if expr_eqb [] [] (order_statics_as after0 normalised) after0
    then Same_once_normalised
    else if
      expr_eqb [] []
        (order_statics_as after0 (without_statics unused normalised))
        after0
    then Same_once_unused_statics_removed
    else
      let after1 = without_head_statics unused after0 in
      if
        expr_eqb [] []
          (order_statics_as after1 (without_statics unused normalised))
          after1
      then Same_once_head_statics_removed
      else if
        expr_eqb [] []
          (order_statics_as
             (float_code (with_code_first after1))
             (float_code
                (without_statics unused
                   (without_slots slots
                      (tidied_before rounds
                         (with_returns taken
                            (lifted_in_turn rounds (unit_continuations ls) ls
                               kept before0)))))))
          (float_code (with_code_first after1))
      then Same_once_lifted
      else if
        expr_eqb [] []
          (order_statics_as
             (with_statics_first ls after1)
             (with_statics_first ls
                (without_statics unused
                   (without_slots slots
                      (tidied_before rounds
                         (with_returns taken
                            (lifted_in_turn rounds (unit_continuations ls) ls
                               kept before0)))))))
          (with_statics_first ls after1)
      then Same_once_lifted
      else Not_validated
