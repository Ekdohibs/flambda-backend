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

(* Proofs of the statements of FlambdaValidateImportsSpec.v. Nothing here
   needs review: the steps are those of FlambdaValidateProofs.v, each
   taken at a static environment in whose code and values the names
   removed do not occur; that of the imports is one. For the verdict
   that lifts, each step is the law of its pass from the imports, and
   the test gives its premise. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaWellFormedSpec FlambdaCompareSpec
  FlambdaCompareProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaStaticsOrderSpec
  FlambdaStaticsOrderProofs FlambdaUnusedStaticsSpec
  FlambdaUnusedStaticsProofs FlambdaUnusedHeadStaticsSpec
  FlambdaUnusedHeadStaticsProofs FlambdaLoopifySpec FlambdaLoopifyProofs
  FlambdaLiftSpec FlambdaLiftProofs
  FlambdaLiftFromCodeSpec FlambdaLiftFromCodeProofs
  FlambdaLiftFromArmsSpec FlambdaLiftFromArmsProofs
  FlambdaLiftAroundSpec FlambdaLiftAroundProofs
  FlambdaUnusedSlotsSpec FlambdaUnusedSlotsProofs FlambdaKnownSymbolsSpec
  FlambdaKnownSymbolsProofs FlambdaKnownClosuresSpec
  FlambdaKnownClosuresProofs FlambdaKnownSlotsSpec FlambdaKnownSlotsProofs
  FlambdaFloatCodeSpec FlambdaFloatCodeProofs FlambdaHoistCodeSpec
  FlambdaHoistCodeProofs FlambdaHoistStaticsSpec
  FlambdaGuardScopeSpec FlambdaGuardScopeProofs
  FlambdaValidateSpec FlambdaValidateProofs
  FlambdaValidateKnowingSpec FlambdaValidateKnowingProofs
  FlambdaImportsSpec FlambdaImportsProofs FlambdaValidateImportsSpec.
From OxCamlFlambda2 Require FlambdaSharedReadsProofs.
From OxCamlFlambda2 Require FlambdaHoistStaticsProofs.
From OxCamlFlambda2Types Require FlambdaSimplifyCarriedSpec
  FlambdaSimplifyCarriedProofs.
From OxCamlFlambda2 Require FlambdaCopyCodeProofs.
From OxCamlFlambda2 Require Import FlambdaStubCallsSpec
  FlambdaStubCallsProofs.
From OxCamlFlambda2 Require Import FlambdaNeverReturnsSpec
  FlambdaNeverReturnsProofs.
Import ListNotations.

Lemma envs_free_of_statics (D : dropped) (statics : static_env) :
  statics_free_of D statics = true -> envs_free_of D [] [] statics = true.
Proof. intros Hfree. unfold envs_free_of. simpl. exact Hfree. Qed.

Lemma without_statics_refines_from (statics : static_env) (hp : heap)
    (D : dropped) (e : expr) :
  statics_free_of D statics = true ->
  refines (State e [] [] statics hp)
    (State (without_statics D e) [] [] statics hp).
Proof.
  intros Hfree. unfold without_statics.
  destruct (drop_statics D e) as [e' | ] eqn:Hdrop; [ | apply refines_refl ].
  apply (refines_trans _ _ _
    (drop_statics_refines D e e' Hdrop [] [] statics hp
       (envs_free_of_statics D statics Hfree))).
  apply (expr_equivalent_refines _ _
    (expr_equivalent_trans _ _ _ (drop_deleted_code_equivalent e')
       (merge_statics_equivalent _))).
Qed.

Theorem validate_unit_sound_from : validate_unit_sound_from_statement.
Proof.
  intros statics hp rounds unused slots G Ls kept before after.
  unfold validate_unit.
  destruct (expr_eqb [] [] before after) eqn:Hsame.
  - intros _ _.
    apply (expr_equivalent_refines _ _ (expr_eqb_sound _ _ Hsame)).
  - cbv zeta. intros Hvalidated Hfree.
    apply (refines_trans _ _ _
      (demote_stub_calls_refines before [] [] statics hp)).
    apply (refines_trans _ _ _
      (cut_returns_refines (demote_stub_calls before) [] [] statics hp)).
    apply (refines_trans _ _ _
      (loopify_unit_refines G (cut_returns [] (demote_stub_calls before))
         [] [] statics hp)).
    revert Hvalidated. clear Hsame.
    generalize (loopify_unit G (cut_returns [] (demote_stub_calls before))).
    clear before. intros before.
    set (a := tidied_after rounds (normalise_after rounds kept after)).
    set (taken := negb (keeps_a_return a)).
    destruct (expr_eqb [] []
                (order_statics_as a
                   (tidied_before rounds
                      (with_returns taken
                         (normalise_before rounds kept before))))
                a)
      eqn:Hnormalised.
    + intros _.
      exact (normalised_refines rounds kept taken before after Hnormalised
               [] [] statics hp).
    + destruct (expr_eqb [] []
                  (order_statics_as a
                     (without_statics unused
                        (tidied_before rounds
                           (with_returns taken
                              (normalise_before rounds kept before)))))
                  a) eqn:Hremoved.
      * intros _.
        apply (refines_trans _
          (State
             (tidied_before rounds
             (with_returns taken (normalise_before rounds kept before)))
             [] [] statics hp)).
        { apply (expr_refines_trans _ _ _
            (normalise_before_refines rounds kept before)).
          apply (expr_refines_trans _ _ _ (with_returns_refines taken _)
            (tidied_before_refines rounds _)). }
        apply (refines_trans _ _ _
          (without_statics_refines_from statics hp unused _ Hfree)).
        apply (expr_equivalent_refines _ _
          (ordered_same_equivalent rounds kept _ after Hremoved)).
      * (* What is left removes static bindings from the head of the
           term after, lifts, or is not validated. *)
        repeat
          match goal with
          | |- context [if ?c then _ else _] => destruct c
          end;
          discriminate.
Qed.

Theorem validate_unit_knowing_sound_from
    : validate_unit_knowing_sound_from_statement.
Proof.
  intros statics hp rounds unused slots G Ls kept before after Hvalidated
    Hfree.
  apply (refines_trans _ _ _
    (normalise_knowing_refines
       (normalise_before_round_avoiding
          (unit_continuations Ls ++ map snd G) kept)
       rounds before
       (normalise_before_round_avoiding_refines
          (unit_continuations Ls ++ map snd G) kept)
       [] [] statics hp)).
  apply (refines_trans _ _ _
    (validate_unit_sound_from statics hp rounds unused slots G Ls kept _ _
       Hvalidated Hfree)).
  apply (expr_equivalent_refines _ _
    (expr_equivalent_sym _ _
       (normalise_knowing_equivalent (normalise_after_round kept) rounds
          after (normalise_after_round_equivalent kept)))).
Qed.

(* The static environment of the imports holds no code and binds each
   symbol to something that holds no value: the names removed occur in
   none of its code and in none of its values. *)
Lemma imports_statics_free (D : dropped)
    (imports : list (symbol * import_class)) :
  statics_free_of D (imports_statics imports) = true.
Proof.
  pose proof (imports_statics_free_of D imports) as Hfree.
  unfold envs_free_of in Hfree. simpl in Hfree. exact Hfree.
Qed.

Theorem validate_unit_sound_imports : validate_unit_sound_imports_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hvalidated.
  exact (validate_unit_sound_from (imports_statics imports) empty_heap rounds
           unused slots G Ls kept before after Hvalidated
           (imports_statics_free unused imports)).
Qed.

Theorem validate_unit_knowing_sound_imports
    : validate_unit_knowing_sound_imports_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hvalidated.
  exact (validate_unit_knowing_sound_from (imports_statics imports)
           empty_heap rounds unused slots G Ls kept before after Hvalidated
           (imports_statics_free unused imports)).
Qed.

(* ======================================================================== *)
(* The verdict that removes static bindings from the head of the term after *)
(* ======================================================================== *)

(* The term refines the term without the unused static bindings of its
   head, from a static environment that binds none of the names at its
   head and in which the names removed do not occur: the steps of
   FlambdaValidateProofs.without_head_statics_program_refines. *)
Lemma without_head_statics_refines_from (statics : static_env) (hp : heap)
    (D : dropped) (e : expr) :
  statics_free_of D statics = true ->
  head_free_in statics e = true ->
  refines (State (without_head_statics D e) [] [] statics hp)
    (State e [] [] statics hp).
Proof.
  intros Hfree Hhead. unfold without_head_statics.
  destruct (head_statics_removable D e) eqn:Hchecks; [ | apply refines_refl ].
  unfold without_statics.
  destruct (drop_statics D e) as [e' | ] eqn:Hdrop; [ | apply refines_refl ].
  assert (Hdropped : drop_head_statics D e = Some e')
    by (unfold drop_head_statics; rewrite Hchecks; exact Hdrop).
  apply (refines_trans _ (State e' [] [] statics hp));
    [ | exact (drop_head_statics_refines_from D e e' statics hp Hdropped
                 Hfree Hhead) ].
  apply (expr_equivalent_refines _ _
    (expr_equivalent_sym _ _
       (expr_equivalent_trans _ _ _ (drop_deleted_code_equivalent e')
          (merge_statics_equivalent _)))).
Qed.

Theorem static_names_free_in_head : static_names_free_in_head_statement.
Proof.
  intros statics e. unfold static_names_free_in, head_free_in.
  assert (Hsub : ∀ (names_of : static_binding -> list positive)
                   (bound : positive -> bool),
            forallb bound (static_names names_of e) = true ->
            forallb bound (flat_map names_of (head_bindings e)) = true).
  { intros names_of bound.
    induction e as [ | bindings body IH | | | | | | ];
      try (intros _; reflexivity).
    cbn [static_names head_bindings]. rewrite flat_map_app.
    rewrite !forallb_app. intros H.
    apply andb_true_iff in H as [Hbindings Hbody].
    rewrite Hbindings. exact (IH Hbody). }
  intros H. apply andb_true_iff in H as [Hsymbols Hcode].
  apply andb_true_iff. split.
  - exact (Hsub symbols_of_binding _ Hsymbols).
  - exact (Hsub code_ids_of_binding _ Hcode).
Qed.

Theorem validate_unit_sound_head_from
    : validate_unit_sound_head_from_statement.
Proof.
  intros statics hp rounds unused slots G Ls kept before after Hvalidated
    Hfree.
  destruct (validate_unit rounds unused slots G Ls kept before after)
    eqn:Hverdict;
    cbn [holds_from] in Hvalidated; try discriminate;
    try (apply (validate_unit_sound_from statics hp rounds unused slots G Ls
                  kept before after);
         [ rewrite Hverdict; reflexivity | exact Hfree ]).
  revert Hverdict. unfold validate_unit.
  destruct (expr_eqb [] [] before after) eqn:Hsame; [ discriminate | ].
  cbv zeta. intros Hverdict.
  apply (refines_trans _ _ _
    (demote_stub_calls_refines before [] [] statics hp)).
  apply (refines_trans _ _ _
    (cut_returns_refines (demote_stub_calls before) [] [] statics hp)).
  apply (refines_trans _ _ _
    (loopify_unit_refines G (cut_returns [] (demote_stub_calls before))
       [] [] statics hp)).
  revert Hverdict. clear Hsame.
  generalize (loopify_unit G (cut_returns [] (demote_stub_calls before))).
  clear before. intros before.
  set (a := tidied_after rounds (normalise_after rounds kept after)) in *.
  set (taken := negb (keeps_a_return a)) in *.
  destruct (expr_eqb [] []
              (order_statics_as a
                 (tidied_before rounds
                    (with_returns taken
                       (normalise_before rounds kept before))))
              a);
    [ discriminate | ].
  destruct (expr_eqb [] []
              (order_statics_as a
                 (without_statics unused
                    (tidied_before rounds
                       (with_returns taken
                          (normalise_before rounds kept before)))))
              a);
    [ discriminate | ].
  set (c := without_head_statics unused a).
  destruct (expr_eqb [] []
              (order_statics_as c
                 (without_statics unused
                    (tidied_before rounds
                       (with_returns taken
                          (normalise_before rounds kept before)))))
              c) eqn:Hremoved_head.
  - intros _.
    apply (refines_trans _
      (State (tidied_before rounds
             (with_returns taken (normalise_before rounds kept before)))
         [] [] statics hp)).
    { apply (expr_refines_trans _ _ _
        (normalise_before_refines rounds kept before)).
      apply (expr_refines_trans _ _ _ (with_returns_refines taken _)
        (tidied_before_refines rounds _)). }
    apply (refines_trans _ _ _
      (without_statics_refines_from statics hp unused _ Hfree)).
    apply (refines_trans _ (State c [] [] statics hp)).
    { apply (expr_equivalent_refines _ _
        (expr_equivalent_trans _ _ _ (order_statics_as_equivalent c _)
           (expr_eqb_sound _ _ Hremoved_head))). }
    apply (refines_trans _ _ _
      (without_head_statics_refines_from statics hp unused a Hfree
         Hvalidated)).
    apply equivalent_refines, equivalent_sym.
    exact (tidied_after_normalised_equivalent rounds kept after [] []
             statics hp).
  - (* What is left lifts, or is not validated. *)
    repeat
      match goal with
      | |- context [if ?c then _ else _] => destruct c
      end;
      discriminate.
Qed.

Theorem validate_unit_knowing_sound_head_from
    : validate_unit_knowing_sound_head_from_statement.
Proof.
  intros statics hp rounds unused slots G Ls kept before after Hvalidated
    Hfree.
  apply (refines_trans _ _ _
    (normalise_knowing_refines
       (normalise_before_round_avoiding
          (unit_continuations Ls ++ map snd G) kept)
       rounds before
       (normalise_before_round_avoiding_refines
          (unit_continuations Ls ++ map snd G) kept)
       [] [] statics hp)).
  apply (refines_trans _ _ _
    (validate_unit_sound_head_from statics hp rounds unused slots G Ls kept
       _ _ Hvalidated Hfree)).
  apply (expr_equivalent_refines _ _
    (expr_equivalent_sym _ _
       (normalise_knowing_equivalent (normalise_after_round kept) rounds
          after (normalise_after_round_equivalent kept)))).
Qed.

Theorem validate_unit_sound_head_imports
    : validate_unit_sound_head_imports_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hvalidated.
  exact (validate_unit_sound_head_from (imports_statics imports) empty_heap
           rounds unused slots G Ls kept before after Hvalidated
           (imports_statics_free unused imports)).
Qed.

Theorem validate_unit_knowing_sound_head_imports
    : validate_unit_knowing_sound_head_imports_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hvalidated.
  exact (validate_unit_knowing_sound_head_from (imports_statics imports)
           empty_heap rounds unused slots G Ls kept before after Hvalidated
           (imports_statics_free unused imports)).
Qed.

Lemma with_lifting_unit_refines (imports : list (symbol * import_class))
    (L : lifting) (e : expr) :
  lifting_free_in (imports_statics imports) L e = true ->
  unit_refines imports e (with_lifting L e).
Proof.
  unfold lifting_free_in, with_lifting, unit_refines. cbv zeta.
  set (statics := imports_statics imports).
  set (at_head :=
    match lift_to_head_unit (lifting_unit L) (lifted_to_head L) e with
    | Some e' => e'
    | None => e
    end).
  set (out_of_code :=
    match
      lift_from_code_unit (lifting_unit L) (lifted_out_of_code L) at_head
    with
    | Some e' => e'
    | None => at_head
    end).
  set (lifted :=
    match lift_unit L out_of_code with
    | Some e' => e'
    | None => out_of_code
    end).
  set (around :=
    match
      lift_around_unit (lifting_unit L) (lifted_around L) (groups_put L)
        lifted
    with
    | Some e' => e'
    | None => lifted
    end).
  intros Hfree.
  apply andb_true_iff in Hfree as [Hfree Htables].
  apply andb_true_iff in Hfree as [Hfree Haround_free].
  apply andb_true_iff in Hfree as [Hfree Hlift_free].
  apply andb_true_iff in Hfree as [Hhead_free Hout_free].
  apply
    (refines_trans _
       (State (add_head_tables (tables_put L) around) [] [] statics
          empty_heap));
    [ | exact (FlambdaCopyCodeProofs.with_copies_refines_imports imports
                 (code_copied L) _) ].
  apply (refines_trans _ (State around [] [] statics empty_heap));
    [ | revert Htables; unfold tables_free_in;
        destruct (tables_put L) as [ | table tables];
          [ intros _; apply refines_refl
          | exact (add_head_tables_refines_imports imports (table :: tables)
                     around) ] ].
  apply (refines_trans _ (State at_head [] [] statics empty_heap)).
  - subst at_head.
    destruct (lift_to_head_unit (lifting_unit L) (lifted_to_head L) e)
      as [e' | ] eqn:Hhead;
      [ | apply refines_refl ].
    exact (lift_to_head_unit_refines_imports imports _ _ e e' Hhead
             Hhead_free).
  - apply (refines_trans _ (State out_of_code [] [] statics empty_heap)).
    + subst out_of_code.
      destruct
        (lift_from_code_unit (lifting_unit L) (lifted_out_of_code L) at_head)
        as [e' | ] eqn:Hout;
        [ | apply refines_refl ].
      exact (lift_from_code_unit_refines_imports imports _ _ at_head e' Hout
               Hout_free).
    + apply (refines_trans _ (State lifted [] [] statics empty_heap)).
      * subst lifted.
        destruct (lift_unit L out_of_code) as [e' | ] eqn:Hlift;
          [ | apply refines_refl ].
        exact (lift_unit_refines_imports imports L out_of_code e' Hlift
                 Hlift_free).
      * subst around.
        destruct
          (lift_around_unit (lifting_unit L) (lifted_around L) (groups_put L)
             lifted)
          as [e' | ] eqn:Haround;
          [ | apply refines_refl ].
        exact (lift_around_unit_refines_imports imports _ _ _ lifted e'
                 Haround Haround_free).
Qed.

Lemma iterate_lifted_unit_refines (imports : list (symbol * import_class))
    (rounds : nat) (avoid : list continuation) (kept : list code_id) :
  ∀ e : expr,
    known_free_in (imports_statics imports) rounds avoid kept e = true ->
    unit_refines imports e
      (iterate rounds (normalise_lifted_round avoid kept) e).
Proof.
  induction rounds as [ | rounds IH]; intros e;
    cbn [known_free_in iterate];
    [ intros _; apply refines_refl | ].
  cbv zeta.
  set (given :=
    use_known_closures (normalise_before_round_avoiding avoid kept e)).
  intros Hfree.
  (* One round, and what the test says after it: nothing where the
     round gives back the term it was given, which is where [iterate]
     stops, and the test of the other rounds otherwise. *)
  assert (Hround :
    unit_refines imports e (normalise_lifted_round avoid kept e)
    ∧ (if expr_eqb [] [] e (normalise_lifted_round avoid kept e) then true
       else
         known_free_in (imports_statics imports) rounds avoid kept
           (normalise_lifted_round avoid kept e))
      = true).
  { assert (Hgiven_refines :
      refines (State e [] [] (imports_statics imports) empty_heap)
        (State given [] [] (imports_statics imports) empty_heap)).
    { apply (refines_trans _ _ _
        (normalise_before_round_avoiding_refines avoid kept e [] []
           (imports_statics imports) empty_heap)).
      exact (use_known_closures_refines _ [] [] (imports_statics imports)
               empty_heap). }
    unfold normalise_lifted_round. fold given. unfold with_known.
    destruct (fold_known_unit given) as [folded | ] eqn:Hfold.
    - apply andb_true_iff in Hfree as [Hgiven Hfree].
      split; [ | exact Hfree ].
      apply (refines_trans _ _ _ Hgiven_refines).
      apply (refines_trans _ _ _
        (fold_known_unit_refines_imports imports given folded Hfold Hgiven)).
      apply (refines_trans _ _ _
        (FlambdaKnownSlotsProofs.read_known_slots_refines_imports imports
           folded)).
      exact (FlambdaSharedReadsProofs.share_static_reads_refines_imports
               imports _).
    - split; [ | exact Hfree ].
      apply (refines_trans _ _ _ Hgiven_refines).
      apply (refines_trans _ _ _
        (FlambdaKnownSlotsProofs.read_known_slots_refines_imports imports
           given)).
      exact (FlambdaSharedReadsProofs.share_static_reads_refines_imports
               imports _). }
  destruct Hround as [Hround Hrest].
  destruct (expr_eqb [] [] e (normalise_lifted_round avoid kept e));
    [ exact Hround | ].
  exact (refines_trans _ _ _ Hround (IH _ Hrest)).
Qed.

(* One round of [normalise_lifted], from what is proved of its rounds
   in turn: [iterate] of one round is that round. *)
Lemma normalise_lifted_round_unit_refines
    (imports : list (symbol * import_class)) (avoid : list continuation)
    (kept : list code_id) (e : expr) :
  known_free_in (imports_statics imports) 1 avoid kept e = true ->
  unit_refines imports e (normalise_lifted_round avoid kept e).
Proof.
  intros Hfree.
  generalize (iterate_lifted_unit_refines imports 1 avoid kept e Hfree).
  cbn [iterate]. cbv zeta.
  destruct (expr_eqb [] [] e (normalise_lifted_round avoid kept e));
    exact (fun H => H).
Qed.

(* The same of the rounds that are followed by the calls of the
   closures that a [Let] makes made direct, where they are asked to. *)
Lemma iterate_lifted_calling_unit_refines
    (imports : list (symbol * import_class)) (calling : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) :
  ∀ e : expr,
    known_calling_free_in (imports_statics imports) calling rounds avoid kept
      e
    = true ->
    unit_refines imports e
      (iterate rounds (normalise_lifted_round_calling calling avoid kept) e).
Proof.
  induction rounds as [ | rounds IH]; intros e;
    cbn [known_calling_free_in iterate];
    [ intros _; apply refines_refl | ].
  cbv zeta. intros Hfree.
  apply andb_true_iff in Hfree as [Hone Hrest].
  assert (Hround :
    unit_refines imports e
      (normalise_lifted_round_calling calling avoid kept e)).
  { unfold normalise_lifted_round_calling. cbv zeta.
    apply (refines_trans _ _ _
      (normalise_lifted_round_unit_refines imports avoid kept e Hone)).
    destruct calling; [ | apply refines_refl ].
    exact (with_made_closures_called_refines _ [] []
             (imports_statics imports) empty_heap). }
  destruct
    (expr_eqb [] [] e (normalise_lifted_round_calling calling avoid kept e));
    [ exact Hround | ].
  exact (refines_trans _ _ _ Hround (IH _ Hrest)).
Qed.

(* The test of the rounds of [normalise_lifted], a round at a time:
   that of its first round, and that of the others of what the first
   gives. *)
Lemma known_free_in_by_rounds (statics : static_env) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr) :
  known_free_in statics (S rounds) avoid kept e
  = known_free_in statics 1 avoid kept e
    && (let e' := normalise_lifted_round avoid kept e in
        if expr_eqb [] [] e e' then true
        else known_free_in statics rounds avoid kept e').
Proof.
  cbn [known_free_in]. cbv zeta. unfold normalise_lifted_round, with_known.
  destruct
    (fold_known_unit
       (use_known_closures (normalise_before_round_avoiding avoid kept e)))
    as [folded | ];
    match goal with
    | |- context [expr_eqb [] [] e ?e'] => destruct (expr_eqb [] [] e e')
    end;
    rewrite ?andb_true_r; reflexivity.
Qed.

(* With nothing asked, the test is the one that there was. *)
Theorem known_calling_free_in_unset : known_calling_free_in_unset_statement.
Proof.
  intros statics rounds avoid kept.
  induction rounds as [ | rounds IH]; intros e; [ reflexivity | ].
  rewrite known_free_in_by_rounds.
  cbn [known_calling_free_in]. cbv zeta.
  change (normalise_lifted_round_calling false avoid kept e)
    with (normalise_lifted_round avoid kept e).
  rewrite IH. reflexivity.
Qed.

Theorem normalise_lifted_calling_free_in_unset :
  normalise_lifted_calling_free_in_unset_statement.
Proof.
  intros statics rounds avoid kept e.
  unfold normalise_lifted_calling_free_in, normalise_lifted_free_in.
  cbv zeta. rewrite !known_calling_free_in_unset. reflexivity.
Qed.

(* The rounds that end the normalisation of a term in which something
   was lifted, where the test passes: each use of what the unit binds
   its symbols to is one of a term that binds no name of the imports. *)
Lemma tidy_known_unit_refines (imports : list (symbol * import_class))
    (inner : nat) :
  ∀ (rounds : nat) (e : expr),
    tidy_known_free_in (imports_statics imports) inner rounds e = true ->
    unit_refines imports e
      (iterate rounds (demote_and_tidy_known_round inner) e).
Proof.
  induction rounds as [ | rounds IH]; intros e;
    cbn [tidy_known_free_in iterate];
    [ intros _; apply refines_refl | ].
  cbv zeta.
  set (given := use_known_closures e).
  intros Hfree.
  assert (Hround :
    unit_refines imports e (demote_and_tidy_known_round inner e)
    ∧ (if expr_eqb [] [] e (demote_and_tidy_known_round inner e) then true
       else
         tidy_known_free_in (imports_statics imports) inner rounds
           (demote_and_tidy_known_round inner e))
      = true).
  { assert (Hgiven_refines :
      refines (State e [] [] (imports_statics imports) empty_heap)
        (State given [] [] (imports_statics imports) empty_heap)).
    { exact (use_known_closures_refines _ [] [] (imports_statics imports)
               empty_heap). }
    unfold demote_and_tidy_known_round. fold given. unfold with_known.
    destruct (fold_known_unit given) as [folded | ] eqn:Hfold.
    - apply andb_true_iff in Hfree as [Hgiven Hfree].
      split; [ | exact Hfree ].
      apply (refines_trans _ _ _ Hgiven_refines).
      apply (refines_trans _ _ _
        (fold_known_unit_refines_imports imports given folded Hfold Hgiven)).
      apply (refines_trans _ _ _
        (FlambdaKnownSlotsProofs.read_known_slots_refines_imports imports
           folded)).
      apply (refines_trans _ _ _
        (FlambdaSharedReadsProofs.share_static_reads_refines_imports imports
           _)).
      exact (demote_and_tidy_refines inner _ [] [] (imports_statics imports)
               empty_heap).
    - split; [ | exact Hfree ].
      apply (refines_trans _ _ _ Hgiven_refines).
      apply (refines_trans _ _ _
        (FlambdaKnownSlotsProofs.read_known_slots_refines_imports imports
           given)).
      apply (refines_trans _ _ _
        (FlambdaSharedReadsProofs.share_static_reads_refines_imports imports
           _)).
      exact (demote_and_tidy_refines inner _ [] [] (imports_statics imports)
               empty_heap). }
  destruct Hround as [Hround Hrest].
  destruct (expr_eqb [] [] e (demote_and_tidy_known_round inner e));
    [ exact Hround | ].
  exact (refines_trans _ _ _ Hround (IH _ Hrest)).
Qed.

Lemma demote_and_tidy_known_unit_refines
    (imports : list (symbol * import_class)) (rounds : nat) (e : expr) :
  demote_and_tidy_known_free_in (imports_statics imports) rounds e = true ->
  unit_refines imports e (demote_and_tidy_known rounds e).
Proof.
  unfold demote_and_tidy_known_free_in, demote_and_tidy_known. intros Hfree.
  apply (refines_trans _ _ _
    (demote_and_tidy_refines rounds e [] [] (imports_statics imports)
       empty_heap)).
  exact (tidy_known_unit_refines imports rounds rounds _ Hfree).
Qed.

Lemma lifted_in_turn_unit_refines (imports : list (symbol * import_class))
    (rounds : nat) (avoid : list continuation) (Ls : list lifting)
    (kept : list code_id) :
  ∀ e : expr,
    lifted_free_in (imports_statics imports) rounds avoid Ls kept e = true ->
    unit_refines imports e (lifted_in_turn rounds avoid Ls kept e).
Proof.
  induction Ls as [ | L Ls IH]; intros e;
    cbn [lifted_free_in lifted_in_turn].
  - intros Hfree.
    apply (refines_trans _ _ _
      (FlambdaKnownSlotsProofs.read_known_slots_anywhere_refines_imports
         imports e)).
    exact (demote_and_tidy_known_unit_refines imports rounds _ Hfree).
  - intros Hfree.
    apply andb_true_iff in Hfree as [Hfree Hrest].
    apply andb_true_iff in Hfree as [Hlifting Hknown].
    apply (refines_trans _ _ _
      (with_lifting_unit_refines imports L e Hlifting)).
    apply (refines_trans _
      (State
         (normalise_lifted_calling (made_closures_called L) rounds avoid kept
            (with_lifting L e))
         [] [] (imports_statics imports) empty_heap)).
    + unfold normalise_lifted_calling.
      unfold normalise_lifted_calling_free_in in Hknown.
      cbv zeta in Hknown.
      apply andb_true_iff in Hknown as [Hfirst Hsecond].
      apply (refines_trans _ _ _
        (iterate_refines _ first_phase_round_refines rounds _ [] []
           (imports_statics imports) empty_heap)).
      apply (refines_trans _ _ _
        (iterate_lifted_calling_unit_refines imports _ rounds avoid kept _
           Hfirst)).
      apply (refines_trans _ _ _
        (guard_scope_refines _ _
           (FlambdaSimplifyCarriedProofs.simplify_by_types_carried_refines
              _)
           [] []
           (imports_statics imports) empty_heap)).
      exact (iterate_lifted_calling_unit_refines imports _ rounds avoid kept
               _ Hsecond).
    + exact (IH _ Hrest).
Qed.

Lemma without_slots_unit_refines (imports : list (symbol * import_class))
    (slots : list value_slot) (e : expr) :
  unit_refines imports e (without_slots slots e).
Proof.
  unfold without_slots.
  destruct (drop_unused_slots slots e) as [e' | ] eqn:Hdrop;
    [ | apply refines_refl ].
  apply (refines_trans _ _ _
    (drop_unused_slots_refines_imports imports slots e e' Hdrop)).
  assert (H : expr_refines e'
                (rewrite_code remove_unused_lets (remove_unused_lets e'))).
  { eapply expr_refines_trans; [ apply remove_unused_lets_refines | ].
    apply rewrite_code_refines, remove_unused_lets_refines. }
  exact (H [] [] (imports_statics imports) empty_heap).
Qed.

Lemma with_code_first_equivalent_imports
    (imports : list (symbol * import_class)) (e : expr) :
  equivalent (State e [] [] (imports_statics imports) empty_heap)
    (State (with_code_first e) [] [] (imports_statics imports) empty_heap).
Proof.
  unfold with_code_first.
  destruct (hoist_code e) as [e' | ] eqn:Hhoist; [ | apply equivalent_refl ].
  apply (equivalent_trans _ _ _
    (hoist_code_equivalent_imports imports e e' Hhoist)).
  apply (equivalent_trans _ _ _ (drop_deleted_code_equivalent e' [] [] _ _)).
  apply merge_statics_equivalent.
Qed.

Lemma with_statics_first_equivalent_imports
    (imports : list (symbol * import_class)) (Ls : list lifting) (e : expr) :
  statics_first_free_in (imports_statics imports) Ls e = true ->
  equivalent (State e [] [] (imports_statics imports) empty_heap)
    (State (with_statics_first Ls e) [] [] (imports_statics imports)
       empty_heap).
Proof.
  unfold with_statics_first, statics_first_free_in.
  destruct Ls as [ | L Ls]; [ intros _; apply equivalent_refl | ].
  destruct (hoist_statics (lifting_unit L) e) as [e' | ] eqn:Hhoist;
    [ | intros _; apply equivalent_refl ].
  intros Hfree.
  apply (equivalent_trans _ _ _
    (FlambdaHoistStaticsProofs.hoist_statics_equivalent_imports imports _ e
       e' Hhoist Hfree)).
  apply (equivalent_trans _ _ _ (drop_deleted_code_equivalent e' [] [] _ _)).
  apply merge_statics_equivalent.
Qed.

Lemma without_head_statics_refines_where_made (statics : static_env)
    (hp : heap) (D : dropped) (e : expr) :
  statics_free_of D statics = true ->
  head_removal_free_in statics D e = true ->
  refines (State (without_head_statics D e) [] [] statics hp)
    (State e [] [] statics hp).
Proof.
  intros Hfree Hhead. unfold without_head_statics.
  destruct (head_statics_removable D e) eqn:Hchecks; [ | apply refines_refl ].
  unfold without_statics.
  destruct (drop_statics D e) as [e' | ] eqn:Hdrop; [ | apply refines_refl ].
  assert (Hdropped : drop_head_statics D e = Some e')
    by (unfold drop_head_statics; rewrite Hchecks; exact Hdrop).
  unfold head_removal_free_in in Hhead. rewrite Hdropped in Hhead.
  apply (refines_trans _ (State e' [] [] statics hp));
    [ | exact (drop_head_statics_refines_from D e e' statics hp Hdropped
                 Hfree Hhead) ].
  apply (expr_equivalent_refines _ _
    (expr_equivalent_sym _ _
       (expr_equivalent_trans _ _ _ (drop_deleted_code_equivalent e')
          (merge_statics_equivalent _)))).
Qed.

Lemma validate_unit_sound_lifted_imports
    (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr) :
  validate_unit rounds unused slots G Ls kept before after
  = Same_once_lifted ->
  lifted_holds_from (imports_statics imports) rounds unused slots G Ls kept
    before after
  = true ->
  unit_refines imports before after.
Proof.
  intros Hverdict Hholds.
  unfold unit_refines.
  set (statics := imports_statics imports) in *.
  apply (refines_trans _ _ _
    (demote_stub_calls_refines before [] [] statics empty_heap)).
  apply (refines_trans _ _ _
    (cut_returns_refines (demote_stub_calls before) [] [] statics
       empty_heap)).
  apply (refines_trans _ _ _
    (loopify_unit_refines G (cut_returns [] (demote_stub_calls before))
       [] [] statics empty_heap)).
  revert Hverdict Hholds. unfold validate_unit, lifted_holds_from.
  destruct (expr_eqb [] [] before after); [ discriminate | ].
  cbv zeta.
  generalize (loopify_unit G (cut_returns [] (demote_stub_calls before))).
  clear before. intros before.
  set (a := tidied_after rounds (normalise_after rounds kept after)).
  set (taken := negb (keeps_a_return a)).
  destruct (expr_eqb [] []
              (order_statics_as a
                 (tidied_before rounds
                    (with_returns taken
                       (normalise_before rounds kept before))))
              a);
    [ discriminate | ].
  destruct (expr_eqb [] []
              (order_statics_as a
                 (without_statics unused
                    (tidied_before rounds
                       (with_returns taken
                          (normalise_before rounds kept before)))))
              a);
    [ discriminate | ].
  set (c := without_head_statics unused a).
  destruct (expr_eqb [] []
              (order_statics_as c
                 (without_statics unused
                    (tidied_before rounds
                       (with_returns taken
                          (normalise_before rounds kept before)))))
              c);
    [ discriminate | ].
  set (lifted :=
    without_statics unused
      (without_slots slots
         (tidied_before rounds
            (with_returns taken
               (lifted_in_turn rounds (unit_continuations Ls) Ls kept
                  before))))).
  intros Hverdict Hholds.
  apply andb_true_iff in Hholds as [Hholds Hlast].
  apply andb_true_iff in Hholds as [Hhead Hlifted_free].
  assert (Hcleaned :
    refines (State c [] [] statics empty_heap)
      (State after [] [] statics empty_heap)).
  { apply (refines_trans _ _ _
      (without_head_statics_refines_where_made statics empty_heap unused a
         (imports_statics_free unused imports) Hhead)).
    apply equivalent_refines, equivalent_sym.
    exact (tidied_after_normalised_equivalent rounds kept after [] []
             statics empty_heap). }
  assert (Hlifting :
    refines (State before [] [] statics empty_heap)
      (State lifted [] [] statics empty_heap)).
  { apply (refines_trans _ _ _
      (lifted_in_turn_unit_refines imports rounds (unit_continuations Ls) Ls
         kept _ Hlifted_free)).
    apply (refines_trans _ _ _
      (with_returns_refines taken _ [] [] statics empty_heap)).
    apply (refines_trans _ _ _
      (tidied_before_refines rounds _ [] [] statics empty_heap)).
    apply (refines_trans _ _ _ (without_slots_unit_refines imports slots _)).
    exact (without_statics_refines_from statics empty_heap unused _
             (imports_statics_free unused imports)). }
  revert Hverdict Hlast.
  destruct
    (expr_eqb [] []
       (order_statics_as (float_code (with_code_first c))
          (float_code lifted))
       (float_code (with_code_first c))) eqn:Hlifted.
  - intros _ _. apply (refines_trans _ _ _ Hlifting).
    apply (refines_trans _
      (State (with_code_first c) [] [] statics empty_heap)).
    { apply expr_equivalent_refines.
      apply (expr_equivalent_trans _ _ _ (float_code_equivalent _)).
      apply (expr_equivalent_trans _ _ _
        (order_statics_as_equivalent (float_code (with_code_first c)) _)).
      apply (expr_equivalent_trans _ _ _ (expr_eqb_sound _ _ Hlifted)).
      apply expr_equivalent_sym, float_code_equivalent. }
    apply (refines_trans _ (State c [] [] statics empty_heap));
      [ | exact Hcleaned ].
    apply equivalent_refines, equivalent_sym.
    apply with_code_first_equivalent_imports.
  - destruct
      (expr_eqb [] []
         (order_statics_as (with_statics_first Ls c)
            (with_statics_first Ls lifted))
         (with_statics_first Ls c)) eqn:Hfirst;
      [ | discriminate ].
    intros _ Hlast. cbn [orb] in Hlast.
    apply andb_true_iff in Hlast as [Hfirst_after Hfirst_lifted].
    apply (refines_trans _ _ _ Hlifting).
    apply (refines_trans _ _ _
      (equivalent_refines _ _
         (with_statics_first_equivalent_imports imports Ls lifted
            Hfirst_lifted))).
    apply (refines_trans _
      (State (with_statics_first Ls c) [] [] statics empty_heap)).
    { apply expr_equivalent_refines.
      apply (expr_equivalent_trans _ _ _
        (order_statics_as_equivalent (with_statics_first Ls c) _)).
      exact (expr_eqb_sound _ _ Hfirst). }
    apply (refines_trans _ (State c [] [] statics empty_heap));
      [ | exact Hcleaned ].
    apply equivalent_refines, equivalent_sym.
    exact (with_statics_first_equivalent_imports imports Ls c Hfirst_after).
Qed.

Theorem validate_unit_sound_verdict_imports
    : validate_unit_sound_verdict_imports_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hholds.
  destruct (validate_unit rounds unused slots G Ls kept before after)
    eqn:Hverdict;
    cbn [verdict_holds_from] in Hholds;
    try (apply (validate_unit_sound_head_imports imports rounds unused slots
                  G Ls kept before after);
         rewrite Hverdict; exact Hholds).
  exact (validate_unit_sound_lifted_imports imports rounds unused slots G Ls
           kept before after Hverdict Hholds).
Qed.

Theorem validate_unit_knowing_sound_verdict_imports
    : validate_unit_knowing_sound_verdict_imports_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hholds.
  unfold verdict_knowing_holds_from, validate_unit_knowing in Hholds.
  unfold unit_refines.
  apply (refines_trans _ _ _
    (normalise_knowing_refines
       (normalise_before_round_avoiding
          (unit_continuations Ls ++ map snd G) kept)
       rounds before
       (normalise_before_round_avoiding_refines
          (unit_continuations Ls ++ map snd G) kept)
       [] [] (imports_statics imports) empty_heap)).
  apply (refines_trans _ _ _
    (validate_unit_sound_verdict_imports imports rounds unused slots G Ls
       kept _ _ Hholds)).
  apply (expr_equivalent_refines _ _
    (expr_equivalent_sym _ _
       (normalise_knowing_equivalent (normalise_after_round kept) rounds
          after (normalise_after_round_equivalent kept)))).
Qed.
