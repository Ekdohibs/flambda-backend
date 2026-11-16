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

(* Proofs of the statements of FlambdaParamOrderNamesSpec.v. Nothing
   here needs review.

   Renaming: the law of the rewrite of one node is proved from the
   definitions by the tactic of FlambdaEquivarianceTactic.v, and the
   pass has it by the walk (FlambdaSpineWalkNamesProofs.v). Three
   laws are given to the tactic, which needs each of them: that of
   [in_order] under [map], which exists; that renaming keeps a list
   of names distinct, by the tactic; and the law of [use_entries], by
   hand, on which the tactic gives up: the function carries a
   position and a sorted list down the parameters.

   Scope: an application with its arguments in an order mentions the
   variables that it mentioned; a handler whose parameters are in an
   order of their positions binds the names that it bound. *)

From Stdlib Require Import ZArith List Bool Arith.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaAlphaProofs FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaForwardArmSpec FlambdaParamOrderSpec FlambdaParamOrderProofs
  FlambdaTermInductionProofs FlambdaEquivarianceTactic
  FlambdaScopedInductionProofs FlambdaScopeTactic
  FlambdaSpineWalkSpec FlambdaSpineWalkNamesProofs FlambdaParamOrderNamesSpec.
Import ListNotations.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

#[local] Hint Rewrite @in_order_map : equivariance.

Lemma distinct_variables_renamed (r : renaming) (names : list variable) :
  distinct (map (rename_variable r) names) = distinct names.
Proof. equivariance. Qed.
#[local] Hint Rewrite distinct_variables_renamed : equivariance.

Lemma param_use_key_renamed : equivariance_of param_use_key.
Proof. equivariance. Qed.

(* The entries are of positions: a renaming of the names leaves them
   as they are. *)
Lemma use_entries_renamed (r : renaming) (free : list variable) :
  ∀ (params : list variable) (i : nat) (sorted : list (nat * nat)),
    use_entries (map (rename_variable r) free) i
      (map (rename_variable r) params) sorted
    = use_entries free i params sorted.
Proof.
  induction params as [ | x params IH]; intros i sorted; simpl;
    [ reflexivity | ].
  rewrite param_use_key_renamed, IH. reflexivity.
Qed.
#[local] Hint Rewrite use_entries_renamed : equivariance.

Lemma order_node_equivariant : rewrite_equivariant order_node.
Proof. equivariance. Qed.

Theorem order_params_equivariant : order_params_equivariant_statement.
Proof. exact (walk_spine_equivariant _ order_node_equivariant). Qed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

Lemma order_args_of_free_variables (k : continuation) (order : list nat)
    (ac : apply_cont) :
  incl (free_variables_of_apply_cont (order_args_of k order ac))
    (free_variables_of_apply_cont ac).
Proof.
  unfold order_args_of.
  destruct (Pos.eqb (ac_continuation ac) k); [ | apply incl_refl ].
  destruct (in_order order (ac_args ac)) as [args | ] eqn:Hargs;
    [ | apply incl_refl ].
  intros a Ha. apply in_flat_map in Ha as (s & Hs & Ha).
  apply in_flat_map. exists s. split; [ | exact Ha ].
  apply In_nth_error in Hs as [j Hj].
  destruct (in_order_from order (ac_args ac) args j s Hargs Hj)
    as (i & _ & Hi).
  exact (nth_error_In _ i Hi).
Qed.

Lemma order_args_of_continuations (k : continuation) (order : list nat)
    (ac : apply_cont) :
  apply_cont_continuations (order_args_of k order ac)
  = apply_cont_continuations ac.
Proof.
  unfold order_args_of.
  destruct (Pos.eqb (ac_continuation ac) k); [ | reflexivity ].
  destruct (in_order order (ac_args ac)); reflexivity.
Qed.

Lemma order_args_scoped (k : continuation) (order : list nat) (e : expr) :
  scoped e (order_args k order e).
Proof.
  induction e using expr_deep_ind; cbn [order_args order_args_in_handler];
    scoped_congruence.
  - apply scoped_apply_cont; [ apply order_args_of_free_variables | ].
    rewrite order_args_of_continuations. apply incl_refl.
  - apply scoped_switch_map; intros ac';
      [ apply order_args_of_free_variables | ].
    rewrite order_args_of_continuations. apply incl_refl.
Qed.

(* A handler with its parameters in an order of their positions. *)
Lemma ordered_handler_scoped (order : list nat)
    (params params' : list variable) (metadata metadata' : handler_metadata)
    (handler : expr) :
  is_order (length params) order = true ->
  in_order order params = Some params' ->
  scoped_handler (Handler params metadata handler)
    (Handler params' metadata' handler).
Proof.
  intros Horder Hparams' Hclosed.
  destruct (is_order_facts (length params) order Horder) as (_ & _ & Hall).
  split; [ exact Hclosed | split; [ | apply incl_refl ] ].
  cbn [free_variables_of_handler]. unfold without.
  intros a Ha. apply filter_In in Ha as [Hin Hout]. apply filter_In.
  split; [ exact Hin | ].
  apply negb_true_iff in Hout. apply negb_true_iff.
  destruct (among a params) eqn:Hamong; [ | reflexivity ].
  rewrite <- Hout. symmetry. apply among_In. apply among_In in Hamong.
  destruct (In_nth_error params a Hamong) as [i Hi].
  assert (Hlt : i < length params)
    by (apply nth_error_Some; rewrite Hi; discriminate).
  exact (in_order_in order params params' i a Hparams' (Hall i Hlt) Hi).
Qed.

Lemma order_node_scoped (e : expr) : scoped e (order_node e).
Proof.
  destruct e as [ | | k [params metadata handler] body | | | | | ];
    try apply scoped_refl.
  cbn [order_node].
  destruct
    (negb (hm_is_exn_handler metadata) && distinct params
     && Nat.eqb (length (hm_params_kinds metadata)) (length params)
     && is_order (length params) (use_order params handler)
     && all_apply_with k (length params) body)
    eqn:Hcond; [ | apply scoped_refl ].
  destruct (in_order (use_order params handler) params) as [params' | ]
    eqn:Hparams'; [ | apply scoped_refl ].
  destruct (in_order (use_order params handler) (hm_params_kinds metadata))
    as [kinds' | ]; [ | apply scoped_refl ].
  apply andb_true_iff in Hcond as [Hcond _].
  apply andb_true_iff in Hcond as [_ Horder].
  apply scoped_let_cont; [ | apply order_args_scoped ].
  exact (ordered_handler_scoped _ params params' _ _ handler Horder Hparams').
Qed.

Theorem order_params_keeps_scope : order_params_keeps_scope_statement.
Proof.
  apply keeps_scope_intro. exact (walk_spine_scoped _ order_node_scoped).
Qed.
