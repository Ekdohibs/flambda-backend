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

(* Proofs of the statements of FlambdaSameComparisonSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextSpec FlambdaContextProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaHeapSpec FlambdaCseSpec FlambdaCseProofs
  FlambdaSameComparisonSpec.
Import ListNotations.

(* ======================================================================== *)
(* The value                                                                *)
(* ======================================================================== *)

Lemma same_answer_naked_bool (op : equality_comparison) :
  Value_const (same_answer op)
  = naked_bool
      match op with
      | Phys_eq => true
      | Phys_neq => false
      end.
Proof. destruct op; reflexivity. Qed.

Lemma different_answer_naked_bool (op : equality_comparison) :
  Value_const (different_answer op)
  = naked_bool
      match op with
      | Phys_eq => false
      | Phys_neq => true
      end.
Proof. destruct op; reflexivity. Qed.

Theorem fold_same_comparison_sound : fold_same_comparison_sound_statement.
Proof.
  intros statics vars d v Hd.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hd.
  destruct p as [p | p | p | p | rp]; try exact Hd.
  destruct p as [k lk m | w | op | k o | k o | k b]; try exact Hd.
  destruct args as [ | a [ | b [ | c args]]]; try exact Hd.
  cbn [fold_same_comparison].
  destruct (FlambdaModel.simple_eq_dec a b) as [<- | Hne].
  - cbn [eval_named eval_simples] in Hd. cbn [eval_named eval_simple].
    destruct (eval_simple vars a) as [va | ]; [ | discriminate Hd ].
    assert (Hprim
      : eval_binary_primitive statics (Phys_equal op) va va = Some v)
      by exact Hd.
    rewrite same_answer_naked_bool.
    rewrite (phys_equal_same statics op va v Hprim). reflexivity.
  - destruct a as [x1 | s1 | c1]; try exact Hd.
    destruct b as [x2 | s2 | c2]; try exact Hd.
    assert (Hsym : s1 <> s2) by (intros ->; apply Hne; reflexivity).
    assert (Hprim
      : eval_binary_primitive statics (Phys_equal op) (Value_symbol s1)
          (Value_symbol s2)
        = Some v)
      by exact Hd.
    cbn [eval_named eval_simple]. rewrite different_answer_naked_bool.
    rewrite (phys_equal_symbols statics op s1 s2 v Hsym Hprim). reflexivity.
Qed.

(* ======================================================================== *)
(* The machine with a heap                                                  *)
(* ======================================================================== *)

(* A location against itself is the same location, and no other value
   is compared as a location is. *)
Lemma eval_phys_equal_location_same (statics : static_env) (v : value)
    (same : bool) :
  eval_phys_equal_location statics v v = Some same -> same = true.
Proof.
  destruct v; simpl; try discriminate.
  rewrite Pos.eqb_refl. intros [= <-]. reflexivity.
Qed.

(* What [exec_bound_prim] is of [Phys_equal], whatever its two arguments
   are. *)
Lemma exec_bound_phys_equal (statics : static_env) (h : heap)
    (op : equality_comparison) (v1 v2 : value) :
  exec_bound_prim statics h (Binary (Phys_equal op)) [v1; v2]
  = if is_location v1 || is_location v2
    then
      let* same := eval_phys_equal_location statics v1 v2 in
      Some
        (naked_bool
           match op with
           | Phys_eq => same
           | Phys_neq => negb same
           end,
         h)
    else
      let* v := eval_prim statics (Binary (Phys_equal op)) [v1; v2] in
      Some (v, h).
Proof. destruct v1; reflexivity. Qed.

Theorem phys_equal_same_heap : phys_equal_same_heap_statement.
Proof.
  intros statics h h' op v result. unfold exec_prim.
  destruct (forallb (argument_bound statics h) [v; v]); [ | discriminate ].
  rewrite exec_bound_phys_equal.
  destruct (is_location v || is_location v).
  - destruct (eval_phys_equal_location statics v v) as [same | ] eqn:Hsame;
      [ | discriminate ].
    apply eval_phys_equal_location_same in Hsame. subst same.
    cbv beta iota. intros [= <- <-]. split; [ | reflexivity ].
    destruct op; reflexivity.
  - destruct (eval_prim statics (Binary (Phys_equal op)) [v; v]) as [r | ]
      eqn:Hprim; [ | discriminate ].
    cbv beta iota. intros [= <- <-]. split; [ | reflexivity ].
    rewrite same_answer_naked_bool.
    exact (phys_equal_same statics op v r Hprim).
Qed.

Theorem phys_equal_symbols_heap : phys_equal_symbols_heap_statement.
Proof.
  intros statics h h' op s1 s2 result Hne. unfold exec_prim.
  destruct (forallb (argument_bound statics h)
              [Value_symbol s1; Value_symbol s2]);
    [ | discriminate ].
  rewrite exec_bound_phys_equal. cbn [is_location orb].
  destruct (eval_prim statics (Binary (Phys_equal op))
              [Value_symbol s1; Value_symbol s2]) as [r | ] eqn:Hprim;
    [ | discriminate ].
  cbv beta iota. intros [= <- <-]. split; [ | reflexivity ].
  rewrite different_answer_naked_bool.
  exact (phys_equal_symbols statics op s1 s2 r Hne Hprim).
Qed.

(* The same for the run of the machine: where the value is a location
   the comparison reads the heap, its answer is the same, and the heap
   is left as it was. *)
Lemma fold_same_comparison_exec (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record)
    (vars : env) (d : named) (v : value) :
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  exec_named statics hp rd vars (fold_same_comparison d) = Some (v, hp', rd').
Proof.
  intros Hd.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hd.
  destruct p as [p | p | p | p | rp]; try exact Hd.
  destruct p as [k lk m | w | op | k o | k o | k b]; try exact Hd.
  destruct args as [ | a [ | b [ | c args]]]; try exact Hd.
  cbn [fold_same_comparison].
  destruct (FlambdaModel.simple_eq_dec a b) as [<- | Hne].
  - apply exec_named_prim_some in Hd as (vs & Hvs & Hprim & ->).
    cbn [eval_simples] in Hvs.
    destruct (eval_simple vars a) as [va | ]; [ | discriminate Hvs ].
    injection Hvs as <-.
    destruct (phys_equal_same_heap statics hp hp' op va v Hprim) as [-> ->].
    reflexivity.
  - destruct a as [x1 | s1 | c1]; try exact Hd.
    destruct b as [x2 | s2 | c2]; try exact Hd.
    assert (Hsym : s1 <> s2) by (intros ->; apply Hne; reflexivity).
    apply exec_named_prim_some in Hd as (vs & Hvs & Hprim & ->).
    injection Hvs as <-.
    destruct (phys_equal_symbols_heap statics hp hp' op s1 s2 v Hsym Hprim)
      as [-> ->].
    reflexivity.
Qed.

(* ======================================================================== *)
(* The refinement                                                           *)
(* ======================================================================== *)

Lemma let_fold_same_comparison_refines (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  expr_refines (Let x kind d body) (Let x kind (fold_same_comparison d) body).
Proof.
  intros vars conts statics hp rd. apply let_refines_step_forward.
  - intros v hp' rd' Hd.
    exact (fold_same_comparison_exec statics hp hp' rd rd' vars d v Hd).
  - intros v hp' rd' _ _. apply refines_refl.
Qed.

Fixpoint fold_same_comparisons_refines_aux (e : expr) :
  expr_refines e (fold_same_comparisons e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [fold_same_comparisons]; try apply expr_refines_refl.
  - apply (expr_refines_trans _ (Let x kind (fold_same_comparison d) body)).
    + apply let_fold_same_comparison_refines.
    + exact (plug_refines (Let_body x kind (fold_same_comparison d) Hole) _ _
        (not_many _) (fold_same_comparisons_refines_aux body)).
  - exact (plug_refines (Let_static_body bindings Hole) _ _ (not_many _)
      (fold_same_comparisons_refines_aux body)).
  - apply (expr_refines_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (fold_same_comparisons handler)) body)).
    + exact (plug_refines
        (Let_cont_non_recursive_handler k params metadata Hole body)
        _ _ (not_many _)
        (fold_same_comparisons_refines_aux handler)).
    + exact (plug_refines
        (Let_cont_non_recursive_body k
           (Handler params metadata (fold_same_comparisons handler)) Hole)
        _ _ (not_many _) (fold_same_comparisons_refines_aux body)).
  - exact (plug_refines (Let_cont_recursive_body handlers Hole) _ _
      (not_many _) (fold_same_comparisons_refines_aux body)).
Qed.

Theorem fold_same_comparisons_refines
    : fold_same_comparisons_refines_statement.
Proof. exact fold_same_comparisons_refines_aux. Qed.

(* ======================================================================== *)
(* The renamings                                                            *)
(* ======================================================================== *)

Lemma fold_same_comparison_equivariant (r : renaming) (d : named) :
  fold_same_comparison (rename_named r d)
  = rename_named r (fold_same_comparison d).
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; try reflexivity.
  destruct p as [p | p | p | p | rp]; try reflexivity.
  destruct p as [k lk m | w | op | k o | k o | k b]; try reflexivity.
  destruct args as [ | a [ | b [ | c args]]]; try reflexivity.
  cbn [rename_named map fold_same_comparison].
  destruct (FlambdaModel.simple_eq_dec a b) as [<- | Hne].
  - destruct
      (FlambdaModel.simple_eq_dec (rename_simple r a) (rename_simple r a))
      as [_ | Hne]; [ reflexivity | elim Hne; reflexivity ].
  - destruct
      (FlambdaModel.simple_eq_dec (rename_simple r a) (rename_simple r b))
      as [Heq | _];
      [ elim Hne; exact (rename_simple_injective r a b Heq) | ].
    destruct a as [x1 | s1 | c1], b as [x2 | s2 | c2]; reflexivity.
Qed.

Fixpoint fold_same_comparisons_equivariant_aux (r : renaming) (e : expr) :
  fold_same_comparisons (rename_expr r e)
  = rename_expr r (fold_same_comparisons e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler fold_same_comparisons]; try reflexivity.
  - rewrite fold_same_comparison_equivariant,
      fold_same_comparisons_equivariant_aux.
    reflexivity.
  - rewrite fold_same_comparisons_equivariant_aux. reflexivity.
  - rewrite !fold_same_comparisons_equivariant_aux. reflexivity.
  - rewrite fold_same_comparisons_equivariant_aux. reflexivity.
Qed.

Theorem fold_same_comparisons_equivariant
    : fold_same_comparisons_equivariant_statement.
Proof. exact fold_same_comparisons_equivariant_aux. Qed.
