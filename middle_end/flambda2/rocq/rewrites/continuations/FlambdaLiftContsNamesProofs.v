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

(* FlambdaLiftContsSpec.lift_handler_conts commutes with renaming: the
   lemma from which FlambdaLiftContsProofs.v proves
   [lift_handler_conts_equivariant_statement]. There is no statement
   here, and nothing here needs review.

   The pass compares names by equality only and makes none, so each of
   its functions gives, on renamed arguments, the renamed result or
   the same answer. The lemma is proved from the definitions
   ([equivariance], of FlambdaEquivarianceTactic.v), which derives the
   laws of most of the functions of the pass by itself. What is stated
   below is what the tactic is given, for this file only.

   The functions that carry a list along the term (what a handler
   reads, the continuations moved, the scrutinees known) have their
   laws stated, with the renaming of that list, which this file
   defines. The tactic proves them, but for two that are proved by
   hand: [lift_spine_equivariant], where the list of the continuations
   moved grows at each step by a pair whose second part is computed
   from the term, and [keep_known_equivariant], where a test on a pair
   stands under [existsb] and the renaming of the list is written with
   a pattern.

   The laws of [lift_spine] and of [still_known] are given also at the
   empty list as it is written, where the pass begins with it.
   Computation is kept from unfolding [lift_spine] and [lift_conts],
   and the laws of [var_free_b] and [cont_free_b], which are proved
   elsewhere, are given as hints: without either the lemma is proved
   more slowly or not at all. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaNormaliseSpec FlambdaUnusedContsSpec FlambdaEquivarianceSpec
  FlambdaEquivarianceProofs FlambdaValidatorEquivarianceBaseProofs
  FlambdaUnaliasParamsSpec FlambdaGiveScrutineeSpec FlambdaLiftContsSpec
  FlambdaEquivarianceTactic.
Import ListNotations.

#[local] Hint Rewrite var_free_b_equivariant_aux cont_free_b_equivariant_aux
  : equivariance.

(* ======================================================================== *)
(* Lists of names                                                           *)
(* ======================================================================== *)

Lemma distinct_variables_renamed (r : renaming) (names : list variable) :
  distinct (map (rename_variable r) names) = distinct names.
Proof. equivariance. Qed.
#[local] Hint Rewrite distinct_variables_renamed : equivariance.

Lemma any_among_renamed (r : renaming) (params ps : list variable) :
  existsb (fun p => among p (map (rename_variable r) params))
    (map (rename_variable r) ps)
  = existsb (fun p => among p params) ps.
Proof. equivariance. Qed.
#[local] Hint Rewrite any_among_renamed : equivariance.

(* ======================================================================== *)
(* The applications given more arguments                                    *)
(* ======================================================================== *)

Lemma only_applied_equivariant (r : renaming) (j : continuation)
    (ps : list variable) (ok : bool) (e : expr) :
  only_applied (rename_continuation r j) (map (rename_variable r) ps) ok
    (rename_expr r e)
  = only_applied j ps ok e.
Proof. equivariance. Qed.
#[local] Hint Rewrite only_applied_equivariant : equivariance.

(* ======================================================================== *)
(* The continuations of a handler, moved out                                *)
(* ======================================================================== *)

Definition rename_read (r : renaming)
    (read : list (variable * flambda_kind)) : list (variable * flambda_kind) :=
  map (fun '(p, kind) => (rename_variable r p, kind)) read.

Lemma map_fst_rename_read (r : renaming)
    (read : list (variable * flambda_kind)) :
  map fst (rename_read r read) = map (rename_variable r) (map fst read).
Proof. equivariance. Qed.
#[local] Hint Rewrite map_fst_rename_read : equivariance.

Lemma map_snd_rename_read (r : renaming)
    (read : list (variable * flambda_kind)) :
  map snd (rename_read r read) = map snd read.
Proof. equivariance. Qed.
#[local] Hint Rewrite map_snd_rename_read : equivariance.

Lemma read_params_equivariant (r : renaming) (e : expr)
    (params : list variable) :
  ∀ kinds : list flambda_kind,
    read_params (map (rename_variable r) params) kinds (rename_expr r e)
    = rename_read r (read_params params kinds e).
Proof. equivariance. Qed.
#[local] Hint Rewrite read_params_equivariant : equivariance.

Definition rename_moved (r : renaming)
    (moved : list (continuation * list variable))
    : list (continuation * list variable) :=
  map (fun '(j, ps) => (rename_continuation r j, map (rename_variable r) ps))
    moved.

Lemma give_all_equivariant (r : renaming)
    (moved : list (continuation * list variable)) (e : expr) :
  give_all (rename_moved r moved) (rename_expr r e)
  = rename_expr r (give_all moved e).
Proof. equivariance. Qed.
#[local] Hint Rewrite give_all_equivariant : equivariance.

Definition rename_lifted (r : renaming)
    (lifted : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(j, h) => (rename_continuation r j, rename_handler r h)) lifted.

Definition rename_spine (r : renaming)
    (result : option (list (continuation * continuation_handler) * expr))
    : option (list (continuation * continuation_handler) * expr) :=
  option_map
    (fun '(lifted, switch) => (rename_lifted r lifted, rename_expr r switch))
    result.

Lemma lift_spine_equivariant (r : renaming) (body : expr)
    (params : list variable) (kinds : list flambda_kind) (handler : expr) :
  ∀ moved : list (continuation * list variable),
    lift_spine (rename_expr r body) (map (rename_variable r) params) kinds
      (rename_moved r moved) (rename_expr r handler)
    = rename_spine r (lift_spine body params kinds moved handler).
Proof.
  induction handler
    as [ y kind d rest _ | bindings rest _ | j [q mdj h] rest IH
       | handlers rest _ | ap | ac | scrutinee arms | ];
    intros moved; cbn [rename_expr rename_handler lift_spine];
    try reflexivity.
  - (* a continuation of the handler *)
    rewrite !give_all_equivariant, !read_params_equivariant,
      !map_fst_rename_read, map_snd_rename_read, only_applied_equivariant,
      !cont_free_b_equivariant_aux, <- map_app, distinct_variables_renamed.
    set (h' := give_all moved h).
    set (read := read_params params kinds h').
    destruct
      (negb (hm_is_exn_handler mdj) && distinct (params ++ q)
       && negb (cont_free_b j body) && negb (cont_free_b j h')
       && only_applied j (map fst read) true (give_all moved rest));
      [ | reflexivity ].
    pose proof (IH ((j, map fst read) :: moved)) as Hrest.
    cbn [rename_moved map] in Hrest. fold (rename_moved r moved) in Hrest.
    rewrite Hrest.
    destruct (lift_spine body params kinds ((j, map fst read) :: moved) rest)
      as [[others switch] | ];
      [ | reflexivity ].
    cbn [rename_spine option_map rename_lifted map rename_handler].
    rewrite map_app. reflexivity.
  - (* the [Switch] *)
    destruct scrutinee as [p | sym | c]; cbn [rename_simple];
      try reflexivity.
    rewrite among_rename_variable.
    destruct (among p params); [ | reflexivity ].
    cbn [rename_spine option_map rename_lifted map].
    exact
      (f_equal (fun e => Some ([], e))
         (give_all_equivariant r moved (Switch (Var p) arms))).
Qed.
#[local] Hint Rewrite lift_spine_equivariant : equivariance.

(* ======================================================================== *)
(* Whether the tested parameter is a scrutinee under another name           *)
(* ======================================================================== *)

Lemma filter_under_map {A B : Type} (keep' : B -> bool) (g : A -> B)
    (keep : A -> bool) (l : list A) :
  (∀ a : A, keep' (g a) = keep a) ->
  filter keep' (map g l) = map g (filter keep l).
Proof.
  intros Hkeep. induction l as [ | a l IH]; cbn [map filter];
    [ reflexivity | ].
  rewrite Hkeep, IH. destruct (keep a); reflexivity.
Qed.

Definition rename_known (r : renaming) (known : list (variable * Z))
    : list (variable * Z) :=
  map (fun '(x, c) => (rename_variable r x, c)) known.

Lemma facts_in_equivariant (r : renaming) (k' : continuation) (body : expr)
    (known : list (variable * Z)) :
  facts_in (rename_continuation r k') (rename_expr r body)
    (rename_known r known)
  = rename_known r (facts_in k' body known).
Proof. equivariance. Qed.
#[local] Hint Rewrite facts_in_equivariant : equivariance.

Lemma forgetting_equivariant (r : renaming) (ys : list variable)
    (known : list (variable * Z)) :
  forgetting (map (rename_variable r) ys) (rename_known r known)
  = rename_known r (forgetting ys known).
Proof. equivariance. Qed.
#[local] Hint Rewrite forgetting_equivariant : equivariance.

Lemma in_arm_equivariant (r : renaming) (scrutinee : simple) (d : Z)
    (known : list (variable * Z)) :
  in_arm (rename_simple r scrutinee) d (rename_known r known)
  = rename_known r (in_arm scrutinee d known).
Proof. equivariance. Qed.
#[local] Hint Rewrite in_arm_equivariant : equivariance.

Lemma keep_known_equivariant (r : renaming) (k : continuation) (i : nat)
    (known : list (variable * Z)) (ac : apply_cont) (cands : list variable) :
  keep_known (rename_continuation r k) i (rename_known r known)
    (rename_apply_cont r ac) (map (rename_variable r) cands)
  = map (rename_variable r) (keep_known k i known ac cands).
Proof.
  unfold keep_known.
  replace (ac_continuation (rename_apply_cont r ac))
    with (rename_continuation r (ac_continuation ac)) by reflexivity.
  replace (ac_args (rename_apply_cont r ac))
    with (map (rename_simple r) (ac_args ac)) by reflexivity.
  rewrite rename_continuation_eqb, nth_error_map.
  destruct (Pos.eqb (ac_continuation ac) k); [ | reflexivity ].
  destruct (nth_error (ac_args ac) i) as [[y | sym | cst] | ];
    try reflexivity.
  destruct cst; try reflexivity.
  cbn [option_map rename_simple].
  apply filter_under_map. intros x.
  unfold rename_known. apply existsb_map_with. intros [y d].
  rewrite rename_variable_eqb. reflexivity.
Qed.
#[local] Hint Rewrite keep_known_equivariant : equivariance.

Lemma still_known_equivariant (r : renaming) (k : continuation) (i : nat)
    (e : expr) :
  ∀ (known : list (variable * Z)) (cands : list variable),
    still_known (rename_continuation r k) i (rename_known r known)
      (rename_expr r e) (map (rename_variable r) cands)
    = map (rename_variable r) (still_known k i known e cands).
Proof. equivariance. Qed.
#[local] Hint Rewrite still_known_equivariant : equivariance.

Lemma scrutinees_equivariant (r : renaming) (e : expr) :
  scrutinees (rename_expr r e) = map (rename_variable r) (scrutinees e).
Proof. equivariance. Qed.
#[local] Hint Rewrite scrutinees_equivariant : equivariance.

Lemma still_known_at_nil_equivariant (r : renaming) (k : continuation)
    (i : nat) (e : expr) (cands : list variable) :
  still_known (rename_continuation r k) i [] (rename_expr r e)
    (map (rename_variable r) cands)
  = map (rename_variable r) (still_known k i [] e cands).
Proof. exact (still_known_equivariant r k i e [] cands). Qed.
#[local] Hint Rewrite still_known_at_nil_equivariant : equivariance.

(* ======================================================================== *)
(* The [Let_cont] with the continuations of its handler moved out           *)
(* ======================================================================== *)

Lemma lift_spine_at_nil_equivariant (r : renaming) (body : expr)
    (params : list variable) (kinds : list flambda_kind) (handler : expr) :
  lift_spine (rename_expr r body) (map (rename_variable r) params) kinds []
    (rename_expr r handler)
  = rename_spine r (lift_spine body params kinds [] handler).
Proof. exact (lift_spine_equivariant r body params kinds handler []). Qed.
#[local] Hint Rewrite lift_spine_at_nil_equivariant : equivariance.
#[local] Arguments lift_spine : simpl never.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

#[local] Arguments lift_conts : simpl never.

Lemma lift_handler_conts_renamed : lift_handler_conts_equivariant_statement.
Proof. equivariance. Qed.
