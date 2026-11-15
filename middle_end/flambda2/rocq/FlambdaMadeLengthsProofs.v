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

(* Proofs of the statements of FlambdaMadeLengthsSpec.v. Nothing here
   needs review.

   The recursion is that of FlambdaCseProofs.v and of
   FlambdaCseEquivalentProofs.v for the sharing of primitives, with
   what is known holding where the term runs
   (FlambdaCseSpec.all_computed_or_made). The lemmas of
   FlambdaCseProofs.v keep it, and say that the constant which takes
   the place of the length of an array made runs as the length does
   ([made_length_named_same]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextProofs FlambdaContextMemoryProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaCseSpec
  FlambdaCseProofs FlambdaMadeLengthsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The result is equivalent to the term                                     *)
(* ======================================================================== *)

(* Where what is known holds, what the rule puts for a defining
   expression runs as the expression does. *)
Lemma made_length_folded_same (known : computed) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record) (d : named) :
  all_computed_or_made known vars statics hp rd ->
  exec_named statics hp rd vars
    match made_length_named known d with
    | Some d' => d'
    | None => d
    end
  = exec_named statics hp rd vars d.
Proof.
  intros Hhold.
  destruct (made_length_named known d) as [d' | ] eqn:Hlength;
    [ | reflexivity ].
  exact (made_length_named_same known vars statics hp rd d d' Hhold Hlength).
Qed.

(* Under a [Let], at the heap and the record that it leaves: the array
   that the [Let] makes is known, and what was known holds there
   still. *)
Lemma all_made_under_let (known : computed) (x : variable) (d : named)
    (vars : env) (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (v : value) :
  all_computed_or_made known vars statics hp rd ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  all_computed_or_made (array_made_under_let x d ++ forget_computed x known)
    ((x, v) :: vars) statics hp' rd'.
Proof.
  intros Hhold Hd entry Hin. apply in_app_or in Hin as [Hmade | Hold].
  - right.
    exact (array_made_new statics hp hp' rd rd' vars x d v entry Hd Hmade).
  - exact
      (all_computed_or_made_forget known vars statics hp' rd' x v
         (all_computed_or_made_evolves known vars statics hp hp' rd rd'
            Hhold
            (exec_named_memory_evolves statics hp hp' rd rd' vars d v Hd))
         entry Hold).
Qed.

(* The result refines the term. *)
Fixpoint lengths_knowing_refines (e : expr) :
  ∀ (known : computed) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    all_computed_or_made known vars statics hp rd ->
    refines (State e vars conts statics hp rd)
      (State (lengths_knowing known e) vars conts statics hp rd)
with lengths_in_handler_refines (h : continuation_handler) :
  ∀ (known : computed) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    all_computed_or_made known vars statics hp rd ->
    refines (State (Let_cont_non_recursive k h body) vars conts statics hp rd)
      (State (Let_cont_non_recursive k (lengths_in_handler known h) body)
         vars conts statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known vars conts statics hp rd Hhold; simpl.
    + (* Let *)
      pose proof (made_length_folded_same known vars statics hp rd d Hhold)
        as Hsame.
      apply let_refines_step_exec; [ exact Hsame | ].
      intros v hp' rd' Hd _.
      apply lengths_knowing_refines.
      exact
        (all_made_under_let known x d vars statics hp hp' rd rd' v Hhold Hd).
    + (* Let_static *)
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply lengths_knowing_refines.
      exact
        (all_computed_or_made_extend known vars statics extended hp rd Hhold
           (eval_static_bindings_extend vars bindings _ _ Hbindings)).
    + (* Let_cont_non_recursive *)
      apply (refines_trans _
        (State (Let_cont_non_recursive k (lengths_in_handler known h) body)
           vars conts statics hp rd)).
      * apply lengths_in_handler_refines, Hhold.
      * apply let_cont_non_recursive_refines_step,
          lengths_knowing_refines, Hhold.
    + (* Let_cont_recursive *)
      apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply lengths_knowing_refines, Hhold.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros known k body vars conts statics hp rd Hhold. simpl.
    apply
      (let_cont_handler_refines_where_memory
         (all_computed_or_made known vars)
         (all_computed_or_made_extend known vars)
         (all_computed_or_made_evolves known vars));
      [ exact Hhold | ].
    intros vs handler_vars conts' statics' hp' rd' Hhold' _ Hbind.
    apply lengths_knowing_refines.
    exact
      (all_computed_or_made_bind_params statics' hp' rd' params known vs
         vars handler_vars Hhold' Hbind).
Qed.

(* The term refines the result. *)
Fixpoint lengths_knowing_refined (e : expr) :
  ∀ (known : computed) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    all_computed_or_made known vars statics hp rd ->
    refines (State (lengths_knowing known e) vars conts statics hp rd)
      (State e vars conts statics hp rd)
with lengths_in_handler_refined (h : continuation_handler) :
  ∀ (known : computed) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    all_computed_or_made known vars statics hp rd ->
    refines
      (State (Let_cont_non_recursive k (lengths_in_handler known h) body)
         vars conts statics hp rd)
      (State (Let_cont_non_recursive k h body) vars conts statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known vars conts statics hp rd Hhold; simpl.
    + (* Let *)
      pose proof (made_length_folded_same known vars statics hp rd d Hhold)
        as Hsame.
      apply let_refines_step_exec; [ symmetry; exact Hsame | ].
      intros v hp' rd' Hd _. rewrite Hsame in Hd.
      apply lengths_knowing_refined.
      exact
        (all_made_under_let known x d vars statics hp hp' rd rd' v Hhold Hd).
    + (* Let_static *)
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply lengths_knowing_refined.
      exact
        (all_computed_or_made_extend known vars statics extended hp rd Hhold
           (eval_static_bindings_extend vars bindings _ _ Hbindings)).
    + (* Let_cont_non_recursive *)
      apply (refines_trans _
        (State (Let_cont_non_recursive k (lengths_in_handler known h) body)
           vars conts statics hp rd)).
      * apply let_cont_non_recursive_refines_step,
          lengths_knowing_refined, Hhold.
      * apply lengths_in_handler_refined, Hhold.
    + (* Let_cont_recursive *)
      apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply lengths_knowing_refined, Hhold.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros known k body vars conts statics hp rd Hhold. simpl.
    apply
      (let_cont_handler_refines_where_memory
         (all_computed_or_made known vars)
         (all_computed_or_made_extend known vars)
         (all_computed_or_made_evolves known vars));
      [ exact Hhold | ].
    intros vs handler_vars conts' statics' hp' rd' Hhold' _ Hbind.
    apply lengths_knowing_refined.
    exact
      (all_computed_or_made_bind_params statics' hp' rd' params known vs
         vars handler_vars Hhold' Hbind).
Qed.

Theorem fold_made_lengths_equivalent :
  fold_made_lengths_equivalent_statement.
Proof.
  intros e vars conts statics hp rd. apply refines_antisymmetric.
  - apply lengths_knowing_refines. intros entry [].
  - apply lengths_knowing_refined. intros entry [].
Qed.

(* ======================================================================== *)
(* The rule commutes with renaming                                          *)
(* ======================================================================== *)

Lemma rename_computed_app (r : renaming) (known known' : computed) :
  rename_computed r (known ++ known')
  = rename_computed r known ++ rename_computed r known'.
Proof. unfold rename_computed. apply map_app. Qed.

Fixpoint lengths_knowing_equivariant_aux (r : renaming) (e : expr) :
  ∀ known : computed,
    lengths_knowing (rename_computed r known) (rename_expr r e)
    = rename_expr r (lengths_knowing known e)
with lengths_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ known : computed,
    lengths_in_handler (rename_computed r known) (rename_handler r h)
    = rename_handler r (lengths_in_handler known h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known; simpl; try reflexivity.
    + rewrite made_length_named_renamed, array_made_under_let_renamed,
        forget_computed_renamed, <- rename_computed_app,
        lengths_knowing_equivariant_aux.
      destruct (made_length_named known d); reflexivity.
    + rewrite lengths_knowing_equivariant_aux. reflexivity.
    + rewrite lengths_in_handler_equivariant,
        lengths_knowing_equivariant_aux.
      reflexivity.
    + rewrite lengths_knowing_equivariant_aux. reflexivity.
  - destruct h as [params metadata handler]. intros known. simpl.
    rewrite forget_computed_vars_renamed, lengths_knowing_equivariant_aux.
    reflexivity.
Qed.

Theorem lengths_knowing_equivariant : lengths_knowing_equivariant_statement.
Proof.
  intros r known e. apply lengths_knowing_equivariant_aux.
Qed.

Theorem fold_made_lengths_equivariant :
  fold_made_lengths_equivariant_statement.
Proof.
  intros r e. exact (lengths_knowing_equivariant_aux r e []).
Qed.
