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

(* Proofs of the statements of FlambdaValidateImportsRenameSpec.v. This
   file needs no review.

   A test that reads the static names of a term reads the same of the
   term under any renaming. A test that computes again terms of the
   validation follows the proof that the validation commutes with
   renaming up to the names that it makes
   (FlambdaValidatorEquivarianceProofs.v): at each step the renamed
   computation has the term of the computation under some renaming,
   and that is all that the tests on static names ask. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaBindersSpec
  FlambdaWellFormedSpec FlambdaCompareSpec FlambdaStaticsOrderSpec
  FlambdaStaticsOrderProofs
  FlambdaUnusedStaticsSpec FlambdaUnusedHeadStaticsSpec FlambdaLiftSpec
  FlambdaLiftFromCodeSpec FlambdaLiftFromArmsSpec FlambdaLiftAroundSpec
  FlambdaKnownSymbolsSpec FlambdaKnownSymbolsProofs FlambdaKnownClosuresSpec
  FlambdaKnownSlotsSpec
  FlambdaKnownSlotsProofs FlambdaHoistCodeSpec FlambdaHoistStaticsSpec
  FlambdaFloatCodeSpec FlambdaGuardScopeSpec FlambdaLoopifySpec
  FlambdaStubCallsSpec
  FlambdaStubCallsProofs FlambdaNeverReturnsSpec FlambdaNeverReturnsProofs
  FlambdaNeverReturnsScopeProofs FlambdaLoopsScopeProofs
  FlambdaStaticsScopeProofs FlambdaValidateSpec
  FlambdaValidatorEquivarianceSpec FlambdaValidatorEquivarianceProofs
  FlambdaValidateKnowingSpec FlambdaValidateKnowingProofs FlambdaImportsSpec
  FlambdaValidateImportsSpec FlambdaValidateImportsRenameSpec.
From OxCamlFlambda2Types Require FlambdaSimplifyCarriedSpec.
Import ListNotations.

(* ======================================================================== *)
(* The tests that read the static names of a term                           *)
(* ======================================================================== *)

Theorem static_names_free_in_equivariant :
  static_names_free_in_equivariant_statement.
Proof.
  intros r statics e. unfold static_names_free_in.
  rewrite
    (static_names_equivariant r symbols_of_binding e
       (symbols_of_binding_renamed r)),
    (static_names_equivariant r code_ids_of_binding e
       (code_ids_of_binding_renamed r)).
  reflexivity.
Qed.

Theorem head_free_in_equivariant : head_free_in_equivariant_statement.
Proof.
  intros r statics e. unfold head_free_in.
  rewrite head_bindings_renamed, symbols_of_bindings_renamed,
    code_ids_of_bindings_renamed.
  reflexivity.
Qed.

Theorem head_removal_free_in_equivariant :
  head_removal_free_in_equivariant_statement.
Proof.
  intros r statics D e. unfold head_removal_free_in, drop_head_statics.
  rewrite head_statics_removable_renamed, drop_statics_equivariant,
    head_free_in_equivariant.
  destruct (head_statics_removable D e); [ | reflexivity ].
  destruct (drop_statics D e); reflexivity.
Qed.

(* ======================================================================== *)
(* What the unit binds its symbols to, of a renamed unit                    *)
(* ======================================================================== *)

(* The test answers the same, and the rewrite commutes with the
   renaming. *)
Lemma fold_known_unit_renamed (r : renaming) (e : expr) :
  fold_known_unit (rename_expr r e)
  = option_map (rename_expr r) (fold_known_unit e).
Proof.
  unfold fold_known_unit. cbv zeta.
  rewrite known_bindings_renamed, known_sets_renamed, bound_as_renamed.
  destruct (bound_as (known_bindings e) _ e); [ | reflexivity ].
  cbn [option_map]. f_equal. symmetry. apply fold_known_equivariant.
Qed.

(* What a round that uses what is known gives, by the cases of the
   test. *)
Lemma use_known_cases (e : expr) :
  use_known e
  = FlambdaSharedReadsSpec.share_static_reads
      (read_known_slots
         match fold_known_unit (use_known_closures e) with
         | Some folded => folded
         | None => use_known_closures e
         end).
Proof. reflexivity. Qed.

(* ======================================================================== *)
(* The tests along the rounds                                               *)
(* ======================================================================== *)

Lemma tidy_known_free_in_renamed (r : renaming) (statics : static_env)
    (inner rounds : nat) :
  ∀ e : expr,
    tidy_known_free_in statics inner rounds (rename_expr r e)
    = tidy_known_free_in statics inner rounds e.
Proof.
  induction rounds as [ | rounds IH]; intros e; [ reflexivity | ].
  cbn [tidy_known_free_in]. cbv zeta.
  rewrite use_known_closures_rewrite_equivariant, fold_known_unit_renamed.
  destruct (fold_known_unit (use_known_closures e)) as [folded | ];
    cbn [option_map];
    rewrite ?static_names_free_in_equivariant,
      read_known_slots_rewrite_equivariant,
      share_static_reads_rewrite_equivariant, demote_and_tidy_equivariant,
      expr_eqb_renamed, IH;
    reflexivity.
Qed.

Theorem demote_and_tidy_known_free_in_equivariant :
  demote_and_tidy_known_free_in_equivariant_statement.
Proof.
  intros r statics rounds e. unfold demote_and_tidy_known_free_in.
  rewrite demote_and_tidy_equivariant. apply tidy_known_free_in_renamed.
Qed.

Theorem known_free_in_equivariant : known_free_in_equivariant_statement.
Proof.
  intros r statics rounds avoid kept. revert r.
  induction rounds as [ | rounds IH]; intros r e Hclosed; [ reflexivity | ].
  cbn [known_free_in]. cbv zeta.
  destruct (normalise_before_round_avoiding_up_to kept r avoid e Hclosed)
    as (r1 & Halike1 & Havoid1 & Heq1).
  cbv beta in Halike1, Heq1. rewrite Heq1.
  destruct (normalise_before_round_avoiding_scope avoid kept e Hclosed)
    as (Hclosed1 & _ & _).
  destruct
    (renames_alike_kept use_known r r1
       (normalise_before_round_avoiding avoid kept e) use_known_scope
       Hclosed1 Halike1)
    as (Hclosed2 & Halike2).
  rewrite use_known_cases in Hclosed2, Halike2.
  rewrite use_known_closures_rewrite_equivariant, fold_known_unit_renamed.
  rewrite <- (map_rename_conts_alike r r1 avoid Havoid1).
  destruct
    (fold_known_unit
       (use_known_closures (normalise_before_round_avoiding avoid kept e)))
    as [folded | ];
    cbn [option_map];
    rewrite ?static_names_free_in_equivariant,
      read_known_slots_rewrite_equivariant,
      share_static_reads_rewrite_equivariant,
      (same_after_renamed r r1 e _ Hclosed2 Halike2),
      (IH r1 _ Hclosed2);
    reflexivity.
Qed.

Theorem normalise_lifted_free_in_equivariant :
  normalise_lifted_free_in_equivariant_statement.
Proof.
  intros r statics rounds avoid kept e Hclosed.
  unfold normalise_lifted_free_in. cbv zeta.
  rewrite (params_phase_equivariant rounds r e).
  destruct (params_phase_scope rounds e Hclosed) as (Hclosed1 & _ & _).
  rewrite (known_free_in_equivariant r statics rounds avoid kept _ Hclosed1).
  f_equal.
  destruct (lifted_rounds_up_to rounds kept r avoid _ Hclosed1)
    as (r1 & _ & Havoid1 & Heq1).
  cbv beta in Heq1. rewrite Heq1.
  destruct (lifted_rounds_scope rounds avoid kept _ Hclosed1)
    as (Hclosed2 & _ & _).
  destruct (by_types_scope _ Hclosed2) as (Hclosed3 & _ & _).
  change
    (guard_scope FlambdaSimplifyCarriedSpec.simplify_by_types_carried)
    with by_types.
  rewrite by_types_equivariant,
    <- (map_rename_conts_alike r r1 avoid Havoid1).
  exact (known_free_in_equivariant r1 statics rounds avoid kept _ Hclosed3).
Qed.

(* ======================================================================== *)
(* The test of one lifting                                                  *)
(* ======================================================================== *)

Lemma tables_free_in_renamed (r : renaming) (statics : static_env)
    (L : lifting) (e : expr) :
  tables_free_in statics (rename_lifting r L) (rename_expr r e)
  = tables_free_in statics L e.
Proof.
  unfold tables_free_in.
  change (tables_put (rename_lifting r L)) with (tables_put L).
  destruct (tables_put L); [ reflexivity | ].
  rewrite add_head_tables_renamed. apply head_free_in_equivariant.
Qed.

Lemma lifting_free_in_renamed (r : renaming) (statics : static_env)
    (L : lifting) (e : expr) :
  lifting_free_in statics (rename_lifting r L) (rename_expr r e)
  = lifting_free_in statics L e.
Proof.
  unfold lifting_free_in. cbv zeta.
  change (lifting_unit (rename_lifting r L))
    with (rename_unit_interface r (lifting_unit L)).
  change (lifted_out_of_code (rename_lifting r L))
    with (lifted_out_of_code L).
  change (lifted_to_head (rename_lifting r L)) with (lifted_to_head L).
  change (lifted_around (rename_lifting r L)) with (lifted_around L).
  change (groups_put (rename_lifting r L)) with (groups_put L).
  rewrite lift_to_head_unit_renamed.
  destruct (lift_to_head_unit (lifting_unit L) (lifted_to_head L) e)
    as [at_head | ];
    cbn [option_map made_free_in]; rewrite lift_from_code_unit_renamed;
    destruct (lift_from_code_unit (lifting_unit L) (lifted_out_of_code L) _)
      as [e' | ];
    cbn [option_map made_free_in]; rewrite lift_unit_renamed;
    destruct (lift_unit L _) as [lifted | ];
    cbn [option_map made_free_in]; rewrite lift_around_unit_renamed;
    destruct
      (lift_around_unit (lifting_unit L) (lifted_around L) (groups_put L) _);
    cbn [option_map made_free_in];
    rewrite ?static_names_free_in_equivariant, tables_free_in_renamed;
    reflexivity.
Qed.

(* ======================================================================== *)
(* The liftings made in turn                                                *)
(* ======================================================================== *)

Theorem lifted_free_in_renamed_alike : lifted_free_in_renamed_alike_statement.
Proof.
  intros r r1 statics rounds avoid Ls kept. revert r1.
  induction Ls as [ | L Ls IH]; intros r1 e Hunits Hclosed Havoid;
    cbn [map lifted_free_in].
  - rewrite read_known_slots_anywhere_equivariant.
    apply demote_and_tidy_known_free_in_equivariant.
  - destruct (Hunits L (or_introl eq_refl)) as [Hret Hexn].
    rewrite (rename_lifting_alike r r1 L avoid Havoid Hret Hexn),
      lifting_free_in_renamed, with_lifting_renamed.
    destruct (with_lifting_scope L e Hclosed) as (Hclosed1 & _ & _).
    pose proof
      (normalise_lifted_free_in_equivariant r1 statics rounds avoid kept
         (with_lifting L e) Hclosed1)
      as Htest.
    rewrite (map_rename_conts_alike r r1 avoid Havoid) in Htest.
    rewrite Htest.
    destruct
      (normalise_lifted_equivariant_up_to rounds kept r1 avoid
         (with_lifting L e) Hclosed1)
      as (r2 & _ & Havoid2 & Heq).
    cbv beta in Heq.
    rewrite (map_rename_conts_alike r r1 avoid Havoid) in Heq.
    rewrite Heq.
    destruct
      (normalise_lifted_keeps_scope rounds avoid kept (with_lifting L e)
         Hclosed1)
      as (Hclosed2 & _ & _).
    rewrite (IH r2 _ (fun L' HL' => Hunits L' (or_intror HL')) Hclosed2);
      [ reflexivity | ].
    intros k Hk. rewrite (Havoid2 k Hk). apply Havoid, Hk.
Qed.

Theorem statics_first_free_in_renamed_alike :
  statics_first_free_in_renamed_alike_statement.
Proof.
  intros r r1 statics Ls e Halike.
  destruct Ls as [ | L Ls]; [ reflexivity | ].
  cbn [map statics_first_free_in].
  assert (Hunit :
    lifting_unit (rename_lifting r L)
    = rename_unit_interface r1 (lifting_unit L)).
  { unfold rename_lifting, rename_unit_interface. cbn [lifting_unit].
    rewrite <- (Halike (unit_return (lifting_unit L))),
      <- (Halike (unit_exn (lifting_unit L)));
      [ reflexivity | | ];
      unfold unit_continuations; cbn [flat_map]; rewrite among_app;
      apply orb_true_iff; left; rewrite !among_cons, Pos.eqb_refl;
      [ apply orb_true_r | reflexivity ]. }
  rewrite Hunit, hoist_statics_renamed.
  destruct (hoist_statics (lifting_unit L) e); cbn [option_map];
    [ apply static_names_free_in_equivariant | reflexivity ].
Qed.

(* ======================================================================== *)
(* The tests of a verdict                                                   *)
(* ======================================================================== *)

Theorem holds_from_equivariant : holds_from_equivariant_statement.
Proof.
  intros r statics rounds kept after v.
  destruct v; cbn [holds_from]; try reflexivity.
  rewrite normalise_after_equivariant, tidied_after_equivariant.
  apply head_free_in_equivariant.
Qed.

(* As FlambdaValidatorEquivarianceProofs.validate_unit_renamed_alike
   for the terms that the two comparisons of lifted terms compare: the
   term before, with the liftings made in turn, is under a renaming
   that the rounds chose, which renames its free names and the
   continuations of the units as the first does. *)
Theorem lifted_holds_from_renamed_alike :
  lifted_holds_from_renamed_alike_statement.
Proof.
  intros r r1 statics rounds unused slots G Ls kept before after Hclosed
    Halike1 Hheld.
  unfold lifted_holds_from. cbv zeta.
  assert (Hunits : rename_conts_alike r r1 (unit_continuations Ls)).
  { intros k Hk. apply Hheld. rewrite among_app, Hk. reflexivity. }
  assert (Hloops : rename_conts_alike r r1 (map snd G)).
  { intros k Hk. apply Hheld. rewrite among_app, Hk. apply orb_true_r. }
  rewrite FlambdaStubCallsProofs.demote_stub_calls_equivariant,
    FlambdaNeverReturnsProofs.cut_returns_unit_equivariant,
    (rename_loopified_alike r r1 G Hloops), loopify_unit_equivariant.
  destruct
    (renames_alike_kept demote_stub_calls r r1 before
       FlambdaStaticsScopeProofs.demote_stub_calls_keeps_scope Hclosed
       Halike1)
    as (Hclosed0 & Halike0).
  destruct
    (renames_alike_kept (cut_returns []) r r1 (demote_stub_calls before)
       (FlambdaNeverReturnsScopeProofs.cut_returns_keeps_scope [])
       Hclosed0 Halike0)
    as (Hclosed00 & Halike00).
  destruct
    (renames_alike_kept (loopify_unit G) r r1
       (cut_returns [] (demote_stub_calls before))
       (FlambdaLoopsScopeProofs.loopify_unit_keeps_scope G) Hclosed00
       Halike00)
    as (Hclosed' & Halike1').
  clear Hclosed Halike1 Hheld Hloops Hclosed0 Halike0 Hclosed00 Halike00.
  revert Hclosed' Halike1'.
  generalize (loopify_unit G (cut_returns [] (demote_stub_calls before))).
  clear before.
  intros before Hclosed Halike1.
  destruct
    (lifted_in_turn_up_to r rounds (unit_continuations Ls) kept Ls
       (unit_continuations_has Ls) r1 before Hclosed Halike1 Hunits)
    as (r2 & Halike2 & Havoid2 & Heq2).
  destruct
    (lifted_in_turn_scope rounds (unit_continuations Ls) kept Ls before
       Hclosed)
    as (Hclosed_l0 & _ & _).
  rewrite unit_continuations_renamed.
  rewrite Heq2, normalise_after_equivariant, tidied_after_equivariant,
    keeps_a_return_equivariant, with_returns_equivariant,
    tidied_before_equivariant, without_head_statics_equivariant,
    without_slots_equivariant, without_statics_equivariant,
    with_code_first_equivariant, !float_code_rewrite_equivariant,
    order_statics_as_equivariant, head_removal_free_in_equivariant,
    (lifted_free_in_renamed_alike r r1 statics rounds
       (unit_continuations Ls) Ls kept before (unit_continuations_has Ls)
       Hclosed Hunits),
    (statics_first_free_in_renamed_alike r r statics Ls _
       (fun k _ => eq_refl)),
    (statics_first_free_in_renamed_alike r r2 statics Ls _ Havoid2).
  set (a := tidied_after rounds (normalise_after rounds kept after)).
  generalize (negb (keeps_a_return a)). intros taken.
  destruct
    (renames_alike_kept (with_returns taken) r r2
       (lifted_in_turn rounds (unit_continuations Ls) Ls kept before)
       (with_returns_keeps_scope taken) Hclosed_l0 Halike2)
    as (Hclosed_lt & Halike_lt).
  destruct
    (renames_alike_kept (tidied_before rounds) r r2
       (with_returns taken
          (lifted_in_turn rounds (unit_continuations Ls) Ls kept before))
       (tidied_before_keeps_scope rounds) Hclosed_lt Halike_lt)
    as (Hclosed_l & Halike_l).
  set
    (l :=
      tidied_before rounds
        (with_returns taken
           (lifted_in_turn rounds (unit_continuations Ls) Ls kept before)))
    in *.
  set (c := float_code (with_code_first (without_head_statics unused a))).
  destruct
    (renames_alike_kept (without_slots slots) r r2 l
       (without_slots_scope slots) Hclosed_l Halike_l)
    as (Hclosed_slots & Halike_slots).
  destruct
    (renames_alike_kept (without_statics unused) r r2
       (without_slots slots l) (without_statics_scope unused) Hclosed_slots
       Halike_slots)
    as (Hclosed_without & Halike_without).
  destruct
    (renames_alike_kept float_code r r2
       (without_statics unused (without_slots slots l)) float_code_scope
       Hclosed_without Halike_without)
    as (Hclosed_floated & Halike_floated).
  destruct
    (renames_alike_kept (order_statics_as c) r r2
       (float_code (without_statics unused (without_slots slots l)))
       (order_statics_as_scope c) Hclosed_floated Halike_floated)
    as (Hclosed_last & Halike_last).
  rewrite
    (expr_eqb_renamed_alike r r2
       (order_statics_as c
          (float_code (without_statics unused (without_slots slots l))))
       c Hclosed_last Halike_last).
  reflexivity.
Qed.

Theorem verdict_holds_from_renamed_alike :
  verdict_holds_from_renamed_alike_statement.
Proof.
  intros r r1 statics rounds unused slots G Ls kept before after v Hclosed
    Halike Hheld.
  destruct v; cbn [verdict_holds_from];
    first
      [ exact
          (lifted_holds_from_renamed_alike r r1 statics rounds unused slots
             G Ls kept before after Hclosed Halike Hheld)
      | apply holds_from_equivariant ].
Qed.

Theorem verdict_holds_from_equivariant :
  verdict_holds_from_equivariant_statement.
Proof.
  intros r statics rounds unused slots G Ls kept before after v Hclosed.
  apply verdict_holds_from_renamed_alike; [ exact Hclosed | | ].
  - split; intros name _; reflexivity.
  - intros k _. reflexivity.
Qed.

(* The second asking: as
   FlambdaValidateKnowingProofs.validate_unit_knowing_equivariant. The
   round of the term before gives its result under a second renaming,
   which does to the free names of the result and to the continuations
   that the round was given what the first does. *)
Theorem verdict_knowing_holds_from_equivariant :
  verdict_knowing_holds_from_equivariant_statement.
Proof.
  intros r statics rounds unused slots G Ls kept before after v Hclosed.
  unfold verdict_knowing_holds_from.
  rewrite unit_continuations_renamed, loop_continuations_renamed, <- map_app.
  set (avoid := unit_continuations Ls ++ map snd G).
  destruct
    (normalise_knowing_equivariant_up_to_avoiding
       (fun avoid => normalise_before_round_avoiding avoid kept) rounds
       (fun avoid => normalise_before_round_avoiding_scope avoid kept)
       (normalise_before_round_avoiding_up_to kept) r avoid before Hclosed)
    as (r1 & Halike & Havoid & Heq).
  cbv beta in Halike, Heq. rewrite Heq.
  rewrite
    (normalise_knowing_equivariant (normalise_after_round kept) rounds
       (normalise_after_round_equivariant kept) r after).
  apply verdict_holds_from_renamed_alike.
  - exact
      (proj1
         (normalise_knowing_keeps_scope
            (normalise_before_round_avoiding avoid kept) rounds
            (normalise_before_round_avoiding_scope avoid kept) before
            Hclosed)).
  - exact Halike.
  - exact Havoid.
Qed.

(* ======================================================================== *)
(* The verdict that the validation gives                                    *)
(* ======================================================================== *)

Theorem validate_unit_holds_from_equivariant :
  validate_unit_holds_from_equivariant_statement.
Proof.
  intros r statics rounds unused slots G Ls kept before after Hclosed.
  rewrite
    (validate_unit_equivariant r rounds unused slots G Ls kept before after
       Hclosed).
  apply verdict_holds_from_equivariant, Hclosed.
Qed.

Theorem validate_unit_knowing_holds_from_equivariant :
  validate_unit_knowing_holds_from_equivariant_statement.
Proof.
  intros r statics rounds unused slots G Ls kept before after Hclosed.
  rewrite
    (validate_unit_knowing_equivariant r rounds unused slots G Ls kept before
       after Hclosed).
  apply verdict_knowing_holds_from_equivariant, Hclosed.
Qed.
