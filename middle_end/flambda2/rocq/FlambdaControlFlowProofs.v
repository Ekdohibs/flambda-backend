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

(* Proofs of the statements of FlambdaControlFlowSpec.v. Each equivalence is
   proved by stepping both sides until they come to the same thing, with
   the step lemmas of FlambdaSemanticsProofs.v ([step_equivalent] for a
   step of one side, [step_eq_equivalent] where the two sides step
   alike). Nothing here needs review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaControlFlowSpec.
Import ListNotations.

(* ======================================================================== *)
(* Continuations                                                            *)
(* ======================================================================== *)

(* Two steps of the left-hand side reach the right-hand side. *)
Theorem let_cont_apply_cont : let_cont_apply_cont_statement.
Proof.
  intros k params metadata handler args vars handler_vars conts statics hp rd
    vs Hargs Hkinds Hbind.
  eapply equivalent_trans;
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  apply step_equivalent.
  apply (step_apply_cont_handler _ _ _ _ _ rd_ vs params metadata handler vars
    conts []);
    [ exact Hargs | apply resolve_head
    | rewrite Hkinds; apply kinds_eqb_refl | exact Hbind ].
Qed.

(* Both sides evaluate the argument first; if it has a value, both reach the
   handler with the parameter bound, and if not, both are undefined. *)
Theorem let_cont_apply_cont_one_param :
  let_cont_apply_cont_one_param_statement.
Proof.
  intros k x metadata kind handler arg Hkinds vars conts statics hp rd.
  eapply equivalent_trans;
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  apply step_eq_equivalent. rewrite step_apply_cont, step_let_simple. simpl.
  destruct (eval_simple vars arg) as [v | ]; simpl; [ | reflexivity ].
  unfold apply_continuation. rewrite resolve_head. simpl. rewrite Hkinds.
  rewrite (kinds_eqb_singleton (kind_of_value v) kind).
  destruct (kind_eqb (kind_of_value v) kind); reflexivity.
Qed.

Theorem let_cont_recursive_apply_cont :
  let_cont_recursive_apply_cont_statement.
Proof.
  intros k params metadata handler handlers args vars handler_vars conts
    statics hp rd vs Hstatics Hlookup Hargs Hkinds Hbind.
  eapply equivalent_trans;
    [ apply step_equivalent; rewrite step_let_cont_recursive, Hstatics;
      reflexivity
    | ].
  apply step_equivalent.
  apply (step_apply_cont_handler _ _ _ _ _ rd_ vs params metadata handler vars
    conts handlers);
    [ exact Hargs | | rewrite Hkinds; apply kinds_eqb_refl | exact Hbind ].
  simpl. rewrite resolve_rec_closures, Hlookup. reflexivity.
Qed.

(* After its first step the loop is in a state that steps to itself. *)
Theorem loop_forever_diverges : loop_forever_diverges_statement.
Proof.
  intros k vars conts statics hp rd.
  set (handlers :=
    [(k,
      Handler [] (make_handler_metadata [])
        (Apply_cont (Apply_cont_expr k [] None)))]).
  set (looping :=
    State (Apply_cont (Apply_cont_expr k [] None)) vars
      (rec_closures handlers vars conts ++ conts) statics hp rd).
  assert (Hfirst :
    step (State (loop_forever k) vars conts statics hp rd) = Step looping).
  { unfold loop_forever. rewrite step_let_cont_recursive. reflexivity. }
  assert (Hstep : step looping = Step looping).
  { apply
   (step_apply_cont_handler _ _ _ _ _ rd_ [] [] (make_handler_metadata [])
      _ vars conts handlers);
      [ reflexivity | | reflexivity | reflexivity ].
    rewrite resolve_rec_closures. simpl. rewrite Pos.eqb_refl.
    reflexivity. }
  apply (step_behaves _ Diverges). rewrite Hfirst. simpl.
  clearbody looping. intros n. exists looping.
  induction n as [ | n IH]; simpl; [ reflexivity | ].
  rewrite Hstep. exact IH.
Qed.

(* ======================================================================== *)
(* Switches                                                                 *)
(* ======================================================================== *)

Theorem switch_const : switch_const_statement.
Proof.
  intros i arms ac Hfind vars conts statics hp rd.
  apply step_equivalent. rewrite step_switch. simpl. rewrite Hfind.
  reflexivity.
Qed.

(* Either the [Switch] steps to one of its arms, which is [ac], or it is
   undefined. *)
Theorem switch_same_arms : switch_same_arms_statement.
Proof.
  intros scrutinee arms ac Hsame vars conts statics hp rd.
  destruct (step_switch_cases scrutinee arms vars conts statics hp rd)
    as [Hwrong | (v & i & arm & _ & _ & Hfind & Hstep)];
    [ apply step_undefined_refines, Hwrong | ].
  apply find_arm_in, Hsame in Hfind. subst arm.
  apply equivalent_refines, step_equivalent, Hstep.
Qed.
