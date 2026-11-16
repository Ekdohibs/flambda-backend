(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Proofs of the statements of FlambdaValidateCopyingSpec.v. Nothing
   here needs review.

   What is accepted: each step of
   FlambdaValidateCopyingSpec.before_copied refines the unit run on
   its own, by the law of that step (the loops; a lifting,
   FlambdaValidateProofs.with_lifting_program_refines; the rounds; the
   copies, FlambdaCopyLetsProofs.v), and from the imports where its
   test passes; the function asked is then sound of the term that it
   is given.

   Under renaming: the steps keep scope, and commute with renaming up
   to the names that the rounds make, as those of
   FlambdaValidateSpec.lifted_in_turn do
   (FlambdaValidatorEquivarianceProofs.v); the copies commute with
   renaming exactly. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaCompareSpec
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaUnusedStaticsSpec
  FlambdaLoopifySpec FlambdaLoopifyProofs FlambdaStubCallsSpec
  FlambdaStubCallsProofs FlambdaNeverReturnsSpec FlambdaNeverReturnsProofs
  FlambdaScopedProofs FlambdaStaticsScopeSpec FlambdaStaticsScopeProofs
  FlambdaLiftSpec FlambdaGuardScopeSpec FlambdaGuardScopeProofs
  FlambdaHeapFreeSpec FlambdaCopyCodeSpec FlambdaCopyCodeProofs
  FlambdaCopyLetsSpec FlambdaCopyLetsProofs FlambdaValidateSpec
  FlambdaValidateProofs FlambdaValidatorEquivarianceSpec
  FlambdaValidatorEquivarianceProofs FlambdaValidateKnowingSpec
  FlambdaValidateKnowingProofs FlambdaImportsSpec FlambdaImportsProofs
  FlambdaValidateImportsSpec FlambdaValidateImportsProofs
  FlambdaValidateImportsRenameSpec FlambdaValidateImportsRenameProofs
  FlambdaValidateReadingRenameProofs FlambdaValidateCopyingSpec.
From OxCamlFlambda2Types Require FlambdaSimplifyCarriedSpec
  FlambdaSimplifyCarriedProofs.
Import ListNotations.

(* ======================================================================== *)
(* What is accepted                                                         *)
(* ======================================================================== *)

(* The loops of the term made, as the validation makes them first,
   refine the term. *)
Lemma made_loops_refines (G : loopified) (e : expr) :
  expr_refines e (made_loops G e).
Proof.
  unfold made_loops.
  apply (expr_refines_trans _ _ _ (demote_full_stub_calls_refines e)).
  apply (expr_refines_trans _ _ _
    (cut_returns_refines (demote_full_stub_calls e))).
  apply loopify_unit_refines.
Qed.

Theorem lifted_steps_program_refines : lifted_steps_program_refines_statement.
Proof.
  intros rounds avoid Ls kept.
  induction Ls as [ | L Ls IH]; intros e; cbn [lifted_steps];
    [ apply refines_refl | ].
  apply (refines_trans _ _ _ (with_lifting_program_refines L _)).
  apply (refines_trans _ _ _
    (normalise_lifted_own_program_refines (made_closures_called L)
       (own_closures_known L) rounds avoid kept _)).
  apply IH.
Qed.

Theorem before_copied_program_refines :
  before_copied_program_refines_statement.
Proof.
  intros rounds G Ls L copies Ls' kept before. unfold before_copied.
  apply (refines_trans _ _ _
    (expr_refines_program _ _ (made_loops_refines G before))).
  apply (refines_trans _ _ _
    (lifted_steps_program_refines rounds (copied_avoiding G Ls L Ls') Ls kept
       _)).
  apply (refines_trans _ _ _ (with_lifting_program_refines L _)).
  apply with_let_copies_program_refines.
Qed.

Theorem validate_unit_copying_sound : validate_unit_copying_sound_statement.
Proof.
  intros rounds unused slots G Ls L copies Ls' kept before after Hvalidated.
  apply (refines_trans _ _ _
    (before_copied_program_refines rounds G Ls L copies Ls' kept before)).
  exact (validate_unit_sound rounds unused slots G Ls' kept _ after
    Hvalidated).
Qed.

Theorem validate_unit_knowing_copying_sound :
  validate_unit_knowing_copying_sound_statement.
Proof.
  intros rounds unused slots G Ls L copies Ls' kept before after Hvalidated.
  apply (refines_trans _ _ _
    (before_copied_program_refines rounds G Ls L copies Ls' kept before)).
  exact (validate_unit_knowing_sound rounds unused slots G Ls' kept _ after
    Hvalidated).
Qed.

(* ======================================================================== *)
(* From the imports                                                         *)
(* ======================================================================== *)

(* The rounds that follow a lifting, from the imports, where their
   test passes: the steps of
   FlambdaValidateImportsProofs.lifted_in_turn_unit_refines. *)
Lemma normalise_lifted_own_unit_refines
    (imports : list (symbol * import_class)) (calling own : bool)
    (rounds : nat) (avoid : list continuation) (kept : list code_id)
    (e : expr) :
  normalise_lifted_own_free_in (imports_statics imports) calling own rounds
    avoid kept e
  = true ->
  unit_refines imports e (normalise_lifted_own calling own rounds avoid kept e).
Proof.
  intros Hknown. unfold unit_refines, normalise_lifted_own.
  unfold normalise_lifted_own_free_in in Hknown. cbv zeta in Hknown.
  apply andb_true_iff in Hknown as [Hfirst Hsecond].
  apply (refines_trans _ _ _
    (iterate_refines _ first_phase_round_refines rounds _ [] []
       (imports_statics imports) empty_heap rd_)).
  apply (refines_trans _ _ _
    (iterate_lifted_own_unit_refines imports _ _ rounds avoid kept _ Hfirst)).
  apply (refines_trans _ _ _
    (guard_scope_refines _ _
       (FlambdaSimplifyCarriedProofs.simplify_by_types_carried_refines _)
       [] [] (imports_statics imports) empty_heap rd_)).
  exact (iterate_lifted_own_unit_refines imports _ _ rounds avoid kept _
           Hsecond).
Qed.

Lemma lifted_steps_unit_refines (imports : list (symbol * import_class))
    (rounds : nat) (avoid : list continuation) (Ls : list lifting)
    (kept : list code_id) :
  ∀ e : expr,
    lifted_steps_free_in (imports_statics imports) rounds avoid Ls kept e
    = true ->
    unit_refines imports e (lifted_steps rounds avoid Ls kept e).
Proof.
  induction Ls as [ | L Ls IH]; intros e;
    cbn [lifted_steps_free_in lifted_steps].
  - intros _. apply refines_refl.
  - intros Hfree.
    apply andb_true_iff in Hfree as [Hfree Hrest].
    apply andb_true_iff in Hfree as [Hlifting Hknown].
    apply (refines_trans _ _ _
      (with_lifting_unit_refines imports L e Hlifting)).
    apply (refines_trans _ _ _
      (normalise_lifted_own_unit_refines imports _ _ rounds avoid kept _
         Hknown)).
    exact (IH _ Hrest).
Qed.

Theorem before_copied_refines_imports :
  before_copied_refines_imports_statement.
Proof.
  intros imports rounds G Ls L copies Ls' kept before Hholds.
  unfold before_copied_holds_from in Hholds. cbv zeta in Hholds.
  apply andb_true_iff in Hholds as [Hsteps Hlifting].
  unfold before_copied.
  apply (refines_trans _ _ _
    (expr_refines_unit_refines imports _ _ (made_loops_refines G before))).
  apply (refines_trans _ _ _
    (lifted_steps_unit_refines imports rounds _ Ls kept _ Hsteps)).
  apply (refines_trans _ _ _
    (with_lifting_unit_refines imports L _ Hlifting)).
  apply with_let_copies_refines_imports.
Qed.

Theorem validate_unit_copying_verdict_imports :
  validate_unit_copying_verdict_imports_statement.
Proof.
  intros imports rounds unused slots G Ls L copies Ls' kept before after
    Hfrom Hholds.
  apply (refines_trans _ _ _
    (before_copied_refines_imports imports rounds G Ls L copies Ls' kept
       before Hfrom)).
  exact (validate_unit_sound_verdict_imports imports rounds unused slots G
    Ls' kept _ after Hholds).
Qed.

Theorem validate_unit_knowing_copying_verdict_imports :
  validate_unit_knowing_copying_verdict_imports_statement.
Proof.
  intros imports rounds unused slots G Ls L copies Ls' kept before after
    Hfrom Hholds.
  apply (refines_trans _ _ _
    (before_copied_refines_imports imports rounds G Ls L copies Ls' kept
       before Hfrom)).
  exact (validate_unit_knowing_sound_verdict_imports imports rounds unused
    slots G Ls' kept _ after Hholds).
Qed.

(* ======================================================================== *)
(* The copies keep scope                                                    *)
(* ======================================================================== *)

Lemma copy_named_free_variables (ids : list (code_id * code_id))
    (d : named) :
  free_variables_of_named (copy_named ids d) = free_variables_of_named d.
Proof. destruct d; reflexivity. Qed.

Fixpoint copy_lets_expr_scoped (ids : list (code_id * code_id))
    (lets : list path) (at_path : path) (e : expr) {struct e} :
  scoped e (copy_lets_expr ids lets at_path e)
with copy_lets_handler_scoped (ids : list (code_id * code_id))
    (lets : list path) (at_path : path) (h : continuation_handler)
    {struct h} :
  scoped_handler h (copy_lets_handler ids lets at_path h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [copy_lets_expr]; try apply scoped_refl.
    + apply scoped_let; [ | apply copy_lets_expr_scoped ].
      destruct (among at_path lets);
        [ rewrite copy_named_free_variables | ]; apply incl_refl.
    + apply scoped_let_static;
        [ apply scoped_bindings_refl | apply copy_lets_expr_scoped ].
    + apply scoped_let_cont;
        [ apply copy_lets_handler_scoped | apply copy_lets_expr_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply copy_lets_expr_scoped ].
      * generalize (xI at_path) as q.
        induction handlers as [ | [k h] handlers IH]; intros q;
          cbn [rec_handlers_map map fst]; [ reflexivity | ].
        f_equal. apply IH.
      * generalize (xI at_path) as q. revert handlers. fix IH 1.
        intros [ | [k h] handlers] q; cbn [rec_handlers_map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply copy_lets_handler_scoped | apply IH ].
  - destruct h as [params metadata handler]. cbn [copy_lets_handler].
    apply scoped_handler_intro, copy_lets_expr_scoped.
Qed.

Lemma copy_lets_binding_scoped_code (ids : list (code_id * code_id))
    (in_code : list (code_id * list path)) (id : code_id) (c : code) :
  ∃ c' : code,
    copy_lets_binding ids in_code (Bind_code id c) = Bind_code id c'
    ∧ scoped_code c c'.
Proof.
  destruct c as [params my_closure ret exn metadata body]. simpl.
  destruct (lookup id in_code) as [lets | ]; eexists;
    (split; [ reflexivity | ]).
  - apply scoped_code_intro, copy_lets_expr_scoped.
  - intros H. exact H.
Qed.

Fixpoint copy_lets_in_code_scoped (ids : list (code_id * code_id))
    (in_code : list (code_id * list path)) (e : expr) {struct e} :
  scoped e (copy_lets_in_code ids in_code e)
with copy_lets_in_code_handler_scoped (ids : list (code_id * code_id))
    (in_code : list (code_id * list path)) (h : continuation_handler)
    {struct h} :
  scoped_handler h (copy_lets_in_code_handler ids in_code h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [copy_lets_in_code]; try apply scoped_refl.
    + apply scoped_let; [ apply incl_refl | apply copy_lets_in_code_scoped ].
    + apply scoped_let_static; [ | apply copy_lets_in_code_scoped ].
      induction bindings as [ | binding bindings IH]; cbn [map];
        [ apply scoped_bindings_nil | ].
      destruct binding as [id c | id | sym const].
      * destruct (copy_lets_binding_scoped_code ids in_code id c)
          as (c' & -> & Hc).
        apply scoped_bindings_cons_code; assumption.
      * cbn [copy_lets_binding]. apply scoped_bindings_cons_same, IH.
      * cbn [copy_lets_binding]. apply scoped_bindings_cons_same, IH.
    + apply scoped_let_cont;
        [ apply copy_lets_in_code_handler_scoped
        | apply copy_lets_in_code_scoped ].
    + apply scoped_let_cont_recursive;
        [ | | apply copy_lets_in_code_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply copy_lets_in_code_handler_scoped | exact IH ].
  - destruct h as [params metadata handler]. cbn [copy_lets_in_code_handler].
    apply scoped_handler_intro, copy_lets_in_code_scoped.
Qed.

Lemma copy_let_code_scoped (copy : let_copy) (e : expr) :
  scoped e (copy_let_code copy e).
Proof.
  destruct copy as [[ids symbols] [lets in_code]]. unfold copy_let_code.
  destruct (exits_plain_knowing [] [] e && expr_free_of_heap e
            && copies_apart ids);
    [ | apply scoped_refl ].
  unfold copy_static_code. destruct (copy_ok ids e); [ | apply scoped_refl ].
  apply (scoped_trans _ _ _ (copy_expr_scoped ids symbols e)).
  apply (scoped_trans _ _ _ (copy_lets_expr_scoped ids lets root _)).
  apply copy_lets_in_code_scoped.
Qed.

Theorem with_let_copies_keeps_scope : with_let_copies_keeps_scope_statement.
Proof.
  intros copies. apply keeps_scope_intro.
  induction copies as [ | copy copies IH]; intros e; cbn [with_let_copies];
    [ apply scoped_refl | ].
  exact (scoped_trans _ _ _ (copy_let_code_scoped copy e) (IH _)).
Qed.

(* ======================================================================== *)
(* The liftings in turn, each with its rounds                               *)
(* ======================================================================== *)

Lemma lifted_steps_scope (rounds : nat) (avoid : list continuation)
    (kept : list code_id) (Ls : list lifting) :
  keeps_scope (lifted_steps rounds avoid Ls kept).
Proof.
  induction Ls as [ | L Ls IH]; intros e Hclosed; cbn [lifted_steps].
  - split; [ exact Hclosed | split; intros; assumption ].
  - destruct (with_lifting_scope L e Hclosed) as (Hclosed1 & Hv1 & Hk1).
    destruct
      (normalise_lifted_own_scope (made_closures_called L)
         (own_closures_known L) rounds avoid kept (with_lifting L e)
         Hclosed1)
      as (Hclosed2 & Hv2 & Hk2).
    destruct (IH _ Hclosed2) as (Hclosed3 & Hv3 & Hk3).
    split; [ exact Hclosed3 | split ].
    + intros x Hx. apply Hv1, Hv2, Hv3, Hx.
    + intros k Hk. apply Hk1, Hk2, Hk3, Hk.
Qed.

Theorem lifted_steps_keeps_scope : lifted_steps_keeps_scope_statement.
Proof. exact lifted_steps_scope. Qed.

(* The steps of FlambdaValidatorEquivarianceProofs.lifted_in_turn_up_to;
   after the last lifting the term is left as it is. *)
Lemma lifted_steps_up_to (r : renaming) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (Ls : list lifting) :
  (∀ L : lifting,
    In L Ls ->
    among (unit_return (lifting_unit L)) avoid = true
    ∧ among (unit_exn (lifting_unit L)) avoid = true) ->
  ∀ (r1 : renaming) (e : expr),
    code_closed e = true ->
    renames_alike r r1 e ->
    rename_conts_alike r r1 avoid ->
    ∃ r' : renaming,
      renames_alike r r' (lifted_steps rounds avoid Ls kept e)
      ∧ rename_conts_alike r r' avoid
      ∧ lifted_steps rounds (map (rename_continuation r) avoid)
          (map (rename_lifting r) Ls) kept (rename_expr r1 e)
        = rename_expr r' (lifted_steps rounds avoid Ls kept e).
Proof.
  induction Ls as [ | L Ls IH]; intros Hunits r1 e Hclosed Halike Havoid;
    cbn [map lifted_steps].
  - exists r1. split; [ exact Halike | split; [ exact Havoid | reflexivity ] ].
  - destruct (Hunits L (or_introl eq_refl)) as [Hret Hexn].
    rewrite (rename_lifting_alike r r1 L avoid Havoid Hret Hexn),
      with_lifting_renamed.
    destruct
      (renames_alike_kept (with_lifting L) r r1 e (with_lifting_scope L)
         Hclosed Halike)
      as (Hclosed1 & Halike1).
    change (made_closures_called (rename_lifting r1 L))
      with (made_closures_called L).
    change (own_closures_known (rename_lifting r1 L))
      with (own_closures_known L).
    destruct
      (normalise_lifted_own_up_to (made_closures_called L)
         (own_closures_known L) rounds kept r1 avoid (with_lifting L e)
         Hclosed1)
      as (r2 & Halike2 & Havoid2 & Heq).
    cbv beta in Heq, Halike2.
    rewrite (map_rename_conts_alike r r1 avoid Havoid) in Heq.
    rewrite Heq.
    destruct
      (normalise_lifted_own_scope (made_closures_called L)
         (own_closures_known L) rounds avoid kept (with_lifting L e)
         Hclosed1)
      as (Hclosed2 & _ & _).
    apply (IH (fun L' HL' => Hunits L' (or_intror HL')) r2).
    + exact Hclosed2.
    + exact
        (renames_alike_then
           (normalise_lifted_own (made_closures_called L)
              (own_closures_known L) rounds avoid kept)
           r r1 r2 (with_lifting L e)
           (normalise_lifted_own_scope (made_closures_called L)
              (own_closures_known L) rounds avoid kept)
           Hclosed1 Halike1 Halike2).
    + intros k Hk. rewrite (Havoid2 k Hk). apply Havoid, Hk.
Qed.

Theorem lifted_steps_equivariant_up_to :
  lifted_steps_equivariant_up_to_statement.
Proof. exact lifted_steps_up_to. Qed.

(* ======================================================================== *)
(* The term that is given the copies                                        *)
(* ======================================================================== *)

Theorem before_copied_keeps_scope : before_copied_keeps_scope_statement.
Proof.
  intros rounds G Ls L copies Ls' kept.
  exact
    (keeps_scope_compose _ (with_let_copies copies)
       (keeps_scope_compose _ (with_lifting L)
          (keeps_scope_compose (made_loops G)
             (lifted_steps rounds (copied_avoiding G Ls L Ls') Ls kept)
             (made_loops_scope G)
             (lifted_steps_scope rounds (copied_avoiding G Ls L Ls') kept Ls))
          (with_lifting_scope L))
       (with_let_copies_keeps_scope copies)).
Qed.

(* The continuations to avoid, of the loops and the liftings
   renamed. *)
Lemma copied_avoiding_renamed (r : renaming) (G : loopified)
    (Ls : list lifting) (L : lifting) (Ls' : list lifting) :
  copied_avoiding (rename_loopified r G) (map (rename_lifting r) Ls)
    (rename_lifting r L) (map (rename_lifting r) Ls')
  = map (rename_continuation r) (copied_avoiding G Ls L Ls').
Proof.
  unfold copied_avoiding.
  change (rename_lifting r L :: map (rename_lifting r) Ls')
    with (map (rename_lifting r) (L :: Ls')).
  rewrite <- map_app, unit_continuations_renamed, loop_continuations_renamed,
    <- map_app.
  reflexivity.
Qed.

(* The continuations of the unit of a lifting among those given are
   among those to avoid. *)
Lemma copied_avoiding_has (G : loopified) (Ls : list lifting) (L : lifting)
    (Ls' : list lifting) (L0 : lifting) :
  In L0 (Ls ++ L :: Ls') ->
  among (unit_return (lifting_unit L0)) (copied_avoiding G Ls L Ls') = true
  ∧ among (unit_exn (lifting_unit L0)) (copied_avoiding G Ls L Ls') = true.
Proof.
  intros Hin. unfold copied_avoiding.
  destruct (unit_continuations_has (Ls ++ L :: Ls') L0 Hin) as [Hret Hexn].
  rewrite !among_app, Hret, Hexn. split; reflexivity.
Qed.

Theorem before_copied_renamed : before_copied_renamed_statement.
Proof.
  intros r rounds G Ls L copies Ls' kept before Hclosed.
  unfold before_copied. rewrite copied_avoiding_renamed.
  set (avoid := copied_avoiding G Ls L Ls').
  change
    (loopify_unit (rename_loopified r G)
       (cut_returns [] (demote_full_stub_calls (rename_expr r before))))
    with (made_loops (rename_loopified r G) (rename_expr r before)).
  change (loopify_unit G (cut_returns [] (demote_full_stub_calls before)))
    with (made_loops G before).
  rewrite made_loops_renamed.
  destruct (made_loops_scope G before Hclosed) as (Hclosed0 & _ & _).
  destruct
    (lifted_steps_up_to r rounds avoid kept Ls
       (fun L0 HL0 =>
          copied_avoiding_has G Ls L Ls' L0 (in_or_app _ _ _ (or_introl HL0)))
       r (made_loops G before) Hclosed0
       (conj (fun x _ => eq_refl) (fun k _ => eq_refl)) (fun k _ => eq_refl))
    as (r1 & Halike1 & Havoid1 & Heq1).
  rewrite Heq1.
  destruct
    (copied_avoiding_has G Ls L Ls' L
       (in_or_app Ls (L :: Ls') L (or_intror (in_eq L Ls'))))
    as [Hret Hexn].
  rewrite (rename_lifting_alike r r1 L avoid Havoid1 Hret Hexn),
    with_lifting_renamed.
  destruct
    (lifted_steps_scope rounds avoid kept Ls (made_loops G before) Hclosed0)
    as (Hclosed1 & _ & _).
  destruct
    (renames_alike_kept (with_lifting L) r r1 _ (with_lifting_scope L)
       Hclosed1 Halike1)
    as (Hclosed2 & Halike2).
  exists r1. split; [ | split; [ exact Havoid1 | ] ].
  - exact
      (proj2
         (renames_alike_kept (with_let_copies copies) r r1 _
            (with_let_copies_keeps_scope copies) Hclosed2 Halike2)).
  - symmetry. apply with_let_copies_equivariant.
Qed.

(* ======================================================================== *)
(* The two functions asked                                                  *)
(* ======================================================================== *)

(* The second renaming does to the continuations that the function
   asked reads what the first does: they are among those that the
   rounds of the term avoided. *)
Lemma read_conts_alike (r r1 : renaming) (G : loopified) (Ls : list lifting)
    (L : lifting) (Ls' : list lifting) :
  rename_conts_alike r r1 (copied_avoiding G Ls L Ls') ->
  rename_conts_alike r r1 (unit_continuations Ls' ++ map snd G).
Proof.
  intros Havoid k Hk. apply Havoid. unfold copied_avoiding.
  rewrite among_app in Hk |- *. apply orb_true_iff in Hk as [Hk | Hk];
    [ | rewrite Hk; apply orb_true_r ].
  apply orb_true_iff. left. unfold unit_continuations in Hk |- *.
  rewrite flat_map_app, among_app. cbn [flat_map]. rewrite among_app, Hk.
  rewrite !orb_true_r. reflexivity.
Qed.

Theorem validate_unit_copying_equivariant :
  validate_unit_copying_equivariant_statement.
Proof.
  intros r rounds unused slots G Ls L copies Ls' kept before after Hclosed.
  unfold validate_unit_copying.
  destruct (before_copied_renamed r rounds G Ls L copies Ls' kept before
              Hclosed)
    as (r1 & Halike & Havoid & Heq).
  rewrite Heq.
  apply validate_unit_renamed_alike.
  - exact
      (proj1
         (before_copied_keeps_scope rounds G Ls L copies Ls' kept before
            Hclosed)).
  - exact Halike.
  - exact (read_conts_alike r r1 G Ls L Ls' Havoid).
Qed.

Theorem validate_unit_knowing_copying_equivariant :
  validate_unit_knowing_copying_equivariant_statement.
Proof.
  intros r rounds unused slots G Ls L copies Ls' kept before after Hclosed.
  unfold validate_unit_knowing_copying.
  destruct (before_copied_renamed r rounds G Ls L copies Ls' kept before
              Hclosed)
    as (r1 & Halike & Havoid & Heq).
  rewrite Heq.
  apply validate_unit_knowing_renamed_alike.
  - exact
      (proj1
         (before_copied_keeps_scope rounds G Ls L copies Ls' kept before
            Hclosed)).
  - exact Halike.
  - exact (read_conts_alike r r1 G Ls L Ls' Havoid).
Qed.

(* ======================================================================== *)
(* The tests from the imports, under renaming                               *)
(* ======================================================================== *)

(* The tests of the liftings in turn, of a term under a second
   renaming that does to the continuations to avoid what the first
   does: FlambdaValidateImportsRenameProofs.lifted_free_in_renamed_alike,
   of [lifted_steps_free_in]. *)
Lemma lifted_steps_free_in_renamed_alike (r : renaming)
    (statics : static_env) (rounds : nat) (avoid : list continuation)
    (kept : list code_id) (Ls : list lifting) :
  (∀ L : lifting,
    In L Ls ->
    among (unit_return (lifting_unit L)) avoid = true
    ∧ among (unit_exn (lifting_unit L)) avoid = true) ->
  ∀ (r1 : renaming) (e : expr),
    code_closed e = true ->
    rename_conts_alike r r1 avoid ->
    lifted_steps_free_in statics rounds (map (rename_continuation r) avoid)
      (map (rename_lifting r) Ls) kept (rename_expr r1 e)
    = lifted_steps_free_in statics rounds avoid Ls kept e.
Proof.
  induction Ls as [ | L Ls IH]; intros Hunits r1 e Hclosed Havoid;
    cbn [map lifted_steps_free_in]; [ reflexivity | ].
  destruct (Hunits L (or_introl eq_refl)) as [Hret Hexn].
  rewrite (rename_lifting_alike r r1 L avoid Havoid Hret Hexn),
    lifting_free_in_renamed, with_lifting_renamed.
  destruct (with_lifting_scope L e Hclosed) as (Hclosed1 & _ & _).
  change (made_closures_called (rename_lifting r1 L))
    with (made_closures_called L).
  change (own_closures_known (rename_lifting r1 L))
    with (own_closures_known L).
  pose proof
    (normalise_lifted_own_free_in_equivariant r1 statics
       (made_closures_called L) (own_closures_known L) rounds avoid kept
       (with_lifting L e) Hclosed1)
    as Htest.
  rewrite (map_rename_conts_alike r r1 avoid Havoid) in Htest.
  rewrite Htest.
  destruct
    (normalise_lifted_own_up_to (made_closures_called L)
       (own_closures_known L) rounds kept r1 avoid (with_lifting L e)
       Hclosed1)
    as (r2 & _ & Havoid2 & Heq).
  cbv beta in Heq.
  rewrite (map_rename_conts_alike r r1 avoid Havoid) in Heq.
  rewrite Heq.
  destruct
    (normalise_lifted_own_scope (made_closures_called L)
       (own_closures_known L) rounds avoid kept (with_lifting L e) Hclosed1)
    as (Hclosed2 & _ & _).
  rewrite (IH (fun L' HL' => Hunits L' (or_intror HL')) r2 _ Hclosed2);
    [ reflexivity | ].
  intros k Hk. rewrite (Havoid2 k Hk). apply Havoid, Hk.
Qed.

Theorem before_copied_holds_from_equivariant :
  before_copied_holds_from_equivariant_statement.
Proof.
  intros r statics rounds G Ls L Ls' kept before Hclosed.
  unfold before_copied_holds_from. cbv zeta.
  rewrite copied_avoiding_renamed.
  set (avoid := copied_avoiding G Ls L Ls').
  change
    (loopify_unit (rename_loopified r G)
       (cut_returns [] (demote_full_stub_calls (rename_expr r before))))
    with (made_loops (rename_loopified r G) (rename_expr r before)).
  change (loopify_unit G (cut_returns [] (demote_full_stub_calls before)))
    with (made_loops G before).
  rewrite made_loops_renamed.
  destruct (made_loops_scope G before Hclosed) as (Hclosed0 & _ & _).
  assert (Hunits :
    ∀ L0 : lifting,
      In L0 Ls ->
      among (unit_return (lifting_unit L0)) avoid = true
      ∧ among (unit_exn (lifting_unit L0)) avoid = true).
  { intros L0 HL0.
    exact
      (copied_avoiding_has G Ls L Ls' L0 (in_or_app _ _ _ (or_introl HL0))). }
  rewrite
    (lifted_steps_free_in_renamed_alike r statics rounds avoid kept Ls Hunits
       r (made_loops G before) Hclosed0 (fun k _ => eq_refl)).
  f_equal.
  destruct
    (lifted_steps_up_to r rounds avoid kept Ls Hunits r (made_loops G before)
       Hclosed0 (conj (fun x _ => eq_refl) (fun k _ => eq_refl))
       (fun k _ => eq_refl))
    as (r1 & _ & Havoid1 & Heq1).
  rewrite Heq1.
  destruct
    (copied_avoiding_has G Ls L Ls' L
       (in_or_app Ls (L :: Ls') L (or_intror (in_eq L Ls'))))
    as [Hret Hexn].
  rewrite (rename_lifting_alike r r1 L avoid Havoid1 Hret Hexn).
  apply lifting_free_in_renamed.
Qed.

Theorem validate_unit_copying_holds_from_equivariant :
  validate_unit_copying_holds_from_equivariant_statement.
Proof.
  intros r statics rounds unused slots G Ls L copies Ls' kept before after
    Hclosed.
  rewrite
    (before_copied_holds_from_equivariant r statics rounds G Ls L Ls' kept
       before Hclosed),
    (validate_unit_copying_equivariant r rounds unused slots G Ls L copies
       Ls' kept before after Hclosed).
  f_equal.
  destruct (before_copied_renamed r rounds G Ls L copies Ls' kept before
              Hclosed)
    as (r1 & Halike & Havoid & Heq).
  rewrite Heq.
  apply verdict_holds_from_renamed_alike.
  - exact
      (proj1
         (before_copied_keeps_scope rounds G Ls L copies Ls' kept before
            Hclosed)).
  - exact Halike.
  - exact (read_conts_alike r r1 G Ls L Ls' Havoid).
Qed.

Theorem validate_unit_knowing_copying_holds_from_equivariant :
  validate_unit_knowing_copying_holds_from_equivariant_statement.
Proof.
  intros r statics rounds unused slots G Ls L copies Ls' kept before after
    Hclosed.
  rewrite
    (before_copied_holds_from_equivariant r statics rounds G Ls L Ls' kept
       before Hclosed),
    (validate_unit_knowing_copying_equivariant r rounds unused slots G Ls L
       copies Ls' kept before after Hclosed).
  f_equal.
  destruct (before_copied_renamed r rounds G Ls L copies Ls' kept before
              Hclosed)
    as (r1 & Halike & Havoid & Heq).
  rewrite Heq.
  apply verdict_knowing_holds_from_renamed_alike.
  - exact
      (proj1
         (before_copied_keeps_scope rounds G Ls L copies Ls' kept before
            Hclosed)).
  - exact Halike.
  - exact (read_conts_alike r r1 G Ls L Ls' Havoid).
Qed.
