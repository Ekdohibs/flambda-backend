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


(* Proofs of the statements of FlambdaValidateInliningRenameSpec.v.
   Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaNominalSpec FlambdaBindersSpec
  FlambdaNormaliseSpec FlambdaUnusedStaticsSpec FlambdaLoopifySpec
  FlambdaLiftSpec FlambdaRenameApartSpec FlambdaScopedProofs
  FlambdaKeepsScopeSpec FlambdaKeepsScopeProofs FlambdaValidateScopeSpec
  FlambdaValidateScopeProofs FlambdaInlineCallsAtSpec
  FlambdaInlineCallsAtProofs FlambdaValidateSpec
  FlambdaValidatorEquivarianceSpec FlambdaValidatorEquivarianceProofs
  FlambdaValidateKnowingSpec FlambdaValidateKnowingProofs
  FlambdaValidateReadingRenameProofs FlambdaValidateInliningSpec
  FlambdaValidateInliningRenameSpec.
Import ListNotations.

(* ======================================================================== *)
(* The inlining at chosen places                                            *)
(* ======================================================================== *)

(* As FlambdaValidateScopeProofs.inline_calls_scoped: a call is given
   to [inline_apply] or left. *)
Fixpoint inline_calls_at_paths_scoped (K : calls_in_code)
    (sites : list path) (known : list (code_id * code)) (at_path : path)
    (e : expr) {struct e} :
  scoped e (inline_calls_at_paths K sites known at_path e)
with handler_inline_calls_at_paths_scoped (K : calls_in_code)
    (sites : list path) (known : list (code_id * code)) (at_path : path)
    (h : continuation_handler) {struct h} :
  scoped_handler h (handler_inline_calls_at_paths K sites known at_path h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [inline_calls_at_paths]; try apply scoped_refl.
    + apply scoped_let;
        [ apply incl_refl | apply inline_calls_at_paths_scoped ].
    + apply scoped_let_static; [ | apply inline_calls_at_paths_scoped ].
      induction bindings as [ | binding bindings IH]; cbn [map];
        [ apply scoped_bindings_nil | ].
      destruct binding
        as [id [params my_closure ret exn metadata body'] | id | sym const];
        cbn [binding_inline_calls_at_paths];
        [ apply scoped_bindings_cons_code;
          [ apply scoped_code_intro, inline_calls_at_paths_scoped
          | exact IH ]
        | apply scoped_bindings_cons_same, IH
        | apply scoped_bindings_cons_same, IH ].
    + apply scoped_let_cont;
        [ apply handler_inline_calls_at_paths_scoped
        | apply inline_calls_at_paths_scoped ].
    + apply scoped_let_cont_recursive;
        [ | | apply inline_calls_at_paths_scoped ].
      * generalize (xI at_path).
        induction handlers as [ | [k1 h1] handlers IH]; intros q;
          [ reflexivity | ].
        cbn [rec_handlers_map map fst]. rewrite IH. reflexivity.
      * generalize (xI at_path).
        induction handlers as [ | [k1 h1] handlers IH]; intros q;
          cbn [rec_handlers_map]; [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply handler_inline_calls_at_paths_scoped | apply IH ].
    + destruct (among at_path sites);
        [ apply inline_apply_scoped | apply scoped_refl ].
  - destruct h as [params metadata handler].
    cbn [handler_inline_calls_at_paths].
    apply scoped_handler_intro, inline_calls_at_paths_scoped.
Qed.

Theorem inline_calls_at_keeps_scope : inline_calls_at_keeps_scope_statement.
Proof.
  intros sites K. apply keeps_scope_intro. intros e.
  apply inline_calls_at_paths_scoped.
Qed.

Lemma inline_calls_at_commutes (sites : list path) (K : calls_in_code) :
  rewrite_equivariant (inline_calls_at sites K).
Proof. intros r e. apply inline_calls_at_equivariant. Qed.

(* ======================================================================== *)
(* The choices in turn                                                      *)
(* ======================================================================== *)

(* One choice: the calls inlined, then the term renamed apart. *)
Lemma choice_inlined_scope (avoid : list continuation) (sites : list path)
    (K : calls_in_code) :
  keeps_scope
    (fun e => rename_apart_avoiding avoid (inline_calls_at sites K e)).
Proof.
  exact
    (keeps_scope_compose (inline_calls_at sites K)
       (rename_apart_avoiding avoid) (inline_calls_at_keeps_scope sites K)
       (rename_apart_avoiding_scope avoid)).
Qed.

Theorem inlined_in_turn_keeps_scope : inlined_in_turn_keeps_scope_statement.
Proof.
  intros avoid choices.
  induction choices as [ | [sites K] choices IH].
  - intros e Hclosed. split; [ exact Hclosed | ].
    split; intros name Hname; exact Hname.
  - exact
      (keeps_scope_compose
         (fun e => rename_apart_avoiding avoid (inline_calls_at sites K e))
         (inlined_in_turn avoid choices)
         (choice_inlined_scope avoid sites K) IH).
Qed.

Theorem inlined_in_turn_equivariant_up_to :
  inlined_in_turn_equivariant_up_to_statement.
Proof.
  intros choices.
  induction choices as [ | [sites K] choices IH].
  - intros r avoid e _. exists r. split; [ split; intros; reflexivity | ].
    split; [ intros k _; reflexivity | reflexivity ].
  - exact
      (equivariant_up_to_avoiding_then
         (fun avoid e =>
            rename_apart_avoiding avoid (inline_calls_at sites K e))
         (fun avoid => inlined_in_turn avoid choices)
         (fun avoid => choice_inlined_scope avoid sites K)
         (fun avoid => inlined_in_turn_keeps_scope avoid choices)
         (equivariant_up_to_avoiding_before rename_apart_avoiding
            (inline_calls_at sites K) (inline_calls_at_keeps_scope sites K)
            (inline_calls_at_commutes sites K)
            rename_apart_avoiding_commutes_up_to)
         IH).
Qed.

(* ======================================================================== *)
(* The term that the checks give                                            *)
(* ======================================================================== *)

Theorem before_inlined_keeps_scope : before_inlined_keeps_scope_statement.
Proof.
  intros choices G Ls.
  exact
    (inlined_in_turn_keeps_scope (unit_continuations Ls ++ map snd G)
       choices).
Qed.

Theorem before_inlined_renamed : before_inlined_renamed_statement.
Proof.
  intros r choices G Ls before Hclosed. unfold before_inlined.
  rewrite unit_continuations_renamed, loop_continuations_renamed, <- map_app.
  exact
    (inlined_in_turn_equivariant_up_to choices r
       (unit_continuations Ls ++ map snd G) before Hclosed).
Qed.

(* ======================================================================== *)
(* The checks                                                               *)
(* ======================================================================== *)

Theorem validate_unit_inlining_equivariant :
  validate_unit_inlining_equivariant_statement.
Proof.
  intros r choices rounds unused slots G Ls kept before after Hclosed.
  unfold validate_unit_inlining.
  destruct (before_inlined_renamed r choices G Ls before Hclosed)
    as (r1 & Halike & Havoid & Heq).
  rewrite Heq.
  apply validate_unit_renamed_alike.
  - exact (proj1 (before_inlined_keeps_scope choices G Ls before Hclosed)).
  - exact Halike.
  - exact Havoid.
Qed.

Theorem validate_unit_knowing_inlining_equivariant :
  validate_unit_knowing_inlining_equivariant_statement.
Proof.
  intros r choices rounds unused slots G Ls kept before after Hclosed.
  unfold validate_unit_knowing_inlining.
  destruct (before_inlined_renamed r choices G Ls before Hclosed)
    as (r1 & Halike & Havoid & Heq).
  rewrite Heq.
  apply validate_unit_knowing_renamed_alike.
  - exact (proj1 (before_inlined_keeps_scope choices G Ls before Hclosed)).
  - exact Halike.
  - exact Havoid.
Qed.
