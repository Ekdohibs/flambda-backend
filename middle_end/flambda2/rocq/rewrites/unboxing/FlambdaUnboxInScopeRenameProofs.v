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


(* Proofs of the statements of FlambdaUnboxInScopeRenameSpec.v. Nothing
   here needs review.

   The validation runs the two rewrites:
   FlambdaValidatorEquivarianceProofs.v and FlambdaValidateScopeProofs.v
   need what is proved here for the round of FlambdaValidateSpec.v, so
   this file imports neither. That each rewrite commutes with renaming
   is proved by the tactic of FlambdaEquivarianceTactic.v. That it
   keeps scope is proved as for FlambdaNormaliseSpec.rewrite_code and
   for FlambdaRecHandlersSpec.rewrite_rec_handlers, with what is known
   of symbols alone given to the rewrite. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaNormaliseSpec FlambdaBlockLoadSpec FlambdaUnboxSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaKeepsScopeProofs FlambdaUnboxParamsSpec FlambdaUnboxParamsRenameSpec
  FlambdaUnboxVariantsSpec FlambdaUnboxInScopeSpec FlambdaUnboxInScopeProofs
  FlambdaUnboxInScopeRenameSpec FlambdaUnboxLoopsSpec
  FlambdaUnboxTogetherSpec.
From OxCamlFlambda2 Require FlambdaUnboxParamsNamesProofs.
From OxCamlFlambda2 Require FlambdaUnboxTogetherNamesProofs.
From OxCamlFlambda2 Require Import FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* What is known of symbols, and renaming                                   *)
(* ======================================================================== *)

Theorem symbols_only_renamed : symbols_only_renamed_statement.
Proof.
  intros r known Honly. rewrite <- Honly.
  unfold rename_facts, facts_of_symbols, rename_known_blocks,
    rename_known_numbers, map_values.
  cbn [facts_blocks facts_numbers known_vars known_symbols
       known_wrapped_vars known_boxed_symbols map].
  do 2 f_equal. rewrite map_map. apply map_ext. intros [s [tag fields]].
  unfold rename_known_block. cbn [fst snd]. rewrite map_map.
  apply f_equal, f_equal, map_ext.
  intros [[x | s' | c] | ]; reflexivity.
Qed.

(* What is known of the symbols alone has the fields that hold symbols
   and constants, and no other: so the answer is the same, though the
   blocks that the bindings make are renamed. The definitions are
   unfolded here, and the two equations of lists are proved by
   [equivariance] (FlambdaEquivarianceTactic.v). *)
Theorem symbols_under_statics_renamed :
  symbols_under_statics_renamed_statement.
Proof.
  intros r known bindings.
  unfold symbols_under_statics, facts_under_statics, known_under_statics,
    numbers_under_statics, facts_of_symbols.
  cbn [facts_blocks facts_numbers known_symbols known_boxed_symbols].
  rewrite !map_app. do 3 f_equal; equivariance.
Qed.

(* ======================================================================== *)
(* The two rewrites commute with renaming                                   *)
(* ======================================================================== *)

(* The two laws are proved from the definitions by [equivariance]. Two
   things are given to it:
   - the law above, to rewrite with;
   - a step: the rewrite given, and the hypothesis of the induction,
     ask that what is known be of symbols alone, and under a group of
     static bindings it is ([facts_of_symbols_only]). *)
#[local] Hint Rewrite symbols_under_statics_renamed : equivariance.

#[local] Ltac eqv_extra ::=
  match goal with
  | H : _ |- _ =>
    eqv_not_true H;
    rewrite H by first [ assumption | apply facts_of_symbols_only ]
  end.

Theorem rewrite_code_in_scope_equivariant :
  rewrite_code_in_scope_equivariant_statement.
Proof. equivariance. Qed.

Theorem rewrite_rec_handlers_in_scope_equivariant :
  rewrite_rec_handlers_in_scope_equivariant_statement.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* The two rewrites keep scope                                              *)
(* ======================================================================== *)

Section Scoped.

Variable f : facts -> expr -> expr.

Hypothesis f_keeps_scope : keeps_scope_where_known f.

Fixpoint rewrite_code_in_scope_scoped (e : expr) :
  ∀ known : facts, scoped e (rewrite_code_in_scope f known e)
with rewrite_code_in_scope_in_handler_scoped (h : continuation_handler) :
  ∀ known : facts,
    scoped_handler h (rewrite_code_in_scope_in_handler f known h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known;
      cbn [rewrite_code_in_scope rewrite_code_in_scope_in_handler];
      cbv zeta.
    + apply scoped_let;
        [ apply incl_refl | apply rewrite_code_in_scope_scoped ].
    + apply scoped_let_static;
        [ apply rewrite_code_in_bindings_scoped, f_keeps_scope,
            facts_of_symbols_only
        | apply rewrite_code_in_scope_scoped ].
    + apply scoped_let_cont;
        [ apply rewrite_code_in_scope_in_handler_scoped
        | apply rewrite_code_in_scope_scoped ].
    + apply scoped_let_cont_recursive;
        [ | | apply rewrite_code_in_scope_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply rewrite_code_in_scope_in_handler_scoped | exact IH ].
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
  - destruct h as [params metadata handler]. intros known.
    cbn [rewrite_code_in_scope rewrite_code_in_scope_in_handler].
    apply scoped_handler_intro, rewrite_code_in_scope_scoped.
Qed.

Fixpoint rewrite_rec_handlers_in_scope_scoped (e : expr) :
  ∀ known : facts,
    symbols_only known ->
    scoped e (rewrite_rec_handlers_in_scope f known e)
with rewrite_rec_handlers_in_scope_in_handler_scoped
    (h : continuation_handler) :
  ∀ known : facts,
    symbols_only known ->
    scoped_handler h (rewrite_rec_handlers_in_scope_in_handler f known h)
with rewrite_rec_handler_in_scope_scoped (h : continuation_handler) :
  ∀ known : facts,
    symbols_only known ->
    scoped_handler h (rewrite_rec_handler_in_scope f known h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known Honly;
      cbn [rewrite_rec_handlers_in_scope
           rewrite_rec_handlers_in_scope_in_handler
           rewrite_rec_handler_in_scope].
    + apply scoped_let;
        [ apply incl_refl
        | apply rewrite_rec_handlers_in_scope_scoped, Honly ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl
        | apply rewrite_rec_handlers_in_scope_scoped, facts_of_symbols_only ].
    + apply scoped_let_cont;
        [ apply rewrite_rec_handlers_in_scope_in_handler_scoped, Honly
        | apply rewrite_rec_handlers_in_scope_scoped, Honly ].
    + apply scoped_let_cont_recursive;
        [ | | apply rewrite_rec_handlers_in_scope_scoped, Honly ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply rewrite_rec_handler_in_scope_scoped, Honly | exact IH ].
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
  - destruct h as [params metadata handler]. intros known Honly.
    cbn [rewrite_rec_handlers_in_scope
         rewrite_rec_handlers_in_scope_in_handler
         rewrite_rec_handler_in_scope].
    apply scoped_handler_intro, rewrite_rec_handlers_in_scope_scoped, Honly.
  - destruct h as [params metadata handler]. intros known Honly.
    cbn [rewrite_rec_handlers_in_scope
         rewrite_rec_handlers_in_scope_in_handler
         rewrite_rec_handler_in_scope].
    apply scoped_handler_intro.
    eapply scoped_trans;
      [ apply rewrite_rec_handlers_in_scope_scoped, Honly
      | apply rewrite_code_body_scoped, f_keeps_scope, Honly ].
Qed.

End Scoped.

Theorem rewrite_code_in_scope_keeps_scope :
  rewrite_code_in_scope_keeps_scope_statement.
Proof.
  intros f Hf known _. apply keeps_scope_intro. intros e.
  exact (rewrite_code_in_scope_scoped f Hf e known).
Qed.

Theorem rewrite_rec_handlers_in_scope_keeps_scope :
  rewrite_rec_handlers_in_scope_keeps_scope_statement.
Proof.
  intros f Hf known Honly. apply keeps_scope_intro. intros e.
  exact (rewrite_rec_handlers_in_scope_scoped f Hf e known Honly).
Qed.

(* ======================================================================== *)
(* The unboxing given what is known of symbols                              *)
(* ======================================================================== *)

(* What is known of symbols alone holds no variable. *)
Lemma symbols_only_within (known : facts) :
  symbols_only known ->
  FlambdaUnboxParamsNamesProofs.facts_within [] known.
Proof.
  intros Honly. rewrite <- Honly. split; [ split | split ].
  - intros entry [].
  - intros entry Hentry s Hs.
    cbn [facts_of_symbols facts_blocks known_symbols] in Hentry.
    apply in_map_iff in Hentry as ([s' [tag fields]] & <- & _).
    cbn [fst snd] in Hs. apply in_map_iff in Hs as (field & Hfield & _).
    destruct field as [[x | s'' | c] | ]; try discriminate Hfield;
      injection Hfield as <-; apply incl_refl.
  - intros entry [].
  - intros entry [].
Qed.

Theorem unbox_variants_knowing_symbols_keeps_scope :
  unbox_variants_knowing_symbols_keeps_scope_statement.
Proof.
  intros known Honly. apply keeps_scope_intro. intros e.
  apply scoped_under_none.
  apply FlambdaUnboxParamsNamesProofs.unbox_variants_knowing_scoped.
  exact (symbols_only_within known Honly).
Qed.

Theorem unbox_loops_knowing_symbols_keeps_scope :
  unbox_loops_knowing_symbols_keeps_scope_statement.
Proof.
  intros known Honly. apply keeps_scope_intro. intros e.
  apply scoped_under_none.
  apply FlambdaUnboxParamsNamesProofs.unbox_loops_knowing_scoped.
  exact (symbols_only_within known Honly).
Qed.

Theorem unbox_together_knowing_symbols_keeps_scope :
  unbox_together_knowing_symbols_keeps_scope_statement.
Proof.
  intros known Honly. apply keeps_scope_intro. intros e.
  apply scoped_under_none.
  apply FlambdaUnboxTogetherNamesProofs.unbox_together_knowing_scoped.
  exact (symbols_only_within known Honly).
Qed.
