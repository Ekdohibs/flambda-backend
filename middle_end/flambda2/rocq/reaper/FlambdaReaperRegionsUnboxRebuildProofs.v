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

(* The commutation of the rebuild with unboxing
   (FlambdaReaperUnboxing.rebuild_expr_u, in the mode of the branch, with
   no change of representation) with the erasure of the regions, on the
   units of FlambdaReaperRegionsUnboxSpec.regions_local: no closure,
   projection, call or code, and the regions as the plain rebuild has
   them. That of FlambdaReaperRegionsRebuildProofs.v, case for case: the
   [Let]s of the variables of the fields that the unboxing makes are
   [Let]s of simples, which the erasure keeps; the [Local_allocation] of
   an unboxed block goes with the block; the decisions of a handler are
   the same in both, and make no parameter of the kind [Region] where no
   unboxed field is of that kind.

   This file needs no review: it proves lemmas for
   FlambdaReaperRegionsUnboxProofs.v. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaBindersSpec
  FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperRebuild
  FlambdaReaperRebuildSpec FlambdaReaperCallingConventions
  FlambdaReaperUnboxing FlambdaRegionsSpec FlambdaReaperRegionsSpec
  FlambdaReaperRegionsUnboxSpec FlambdaReaperRegionsRebuildProofs.
From OxCamlFlambda2 Require FlambdaPrimitiveProofs FlambdaReaperFlowProofs
  FlambdaReaperUnboxingRefinesProofs.
Import ListNotations.

(* ======================================================================== *)
(* The erasure of the [Let]s that the unboxing makes                        *)
(* ======================================================================== *)

Section Erase.

Variable c : region_conts.

Lemma erase_lets_simple (pairs : list (variable * variable * flambda_kind)) :
  ∀ body : expr,
    erase c
      (List.fold_left
         (fun body '(x, x', kind) => Let x kind (Simple (Var x')) body)
         pairs body)
    = List.fold_left
        (fun body '(x, x', kind) => Let x kind (Simple (Var x')) body)
        pairs (erase c body).
Proof.
  induction pairs as [ | [[x x'] k] pairs IH]; intros body; [ reflexivity | ].
  cbn [List.fold_left]. rewrite IH. reflexivity.
Qed.

Lemma bind_fields_erase (u u' : unboxed_field) (body : expr) :
  option_map (erase c) (bind_fields u u' body)
  = bind_fields u u' (erase c body).
Proof.
  unfold bind_fields. destruct (unboxed_field_pairs Value u u');
    [ | reflexivity ].
  cbn [option_map]. rewrite erase_lets_simple. reflexivity.
Qed.

Lemma bind_field_to_simple_erase (u : unboxed_field) (kind : flambda_kind)
    (s : simple) (body : expr) :
  option_map (erase c) (bind_field_to_simple u kind s body)
  = bind_field_to_simple u kind s (erase c body).
Proof. destruct u; reflexivity. Qed.

Lemma bind_block_fields_erase (ub : list (node * unboxed_fields))
    (en : rebuild_env) (tag : Z) (args : list simple)
    (to_bind : unboxed_fields) (body : expr) :
  option_map (erase c) (bind_block_fields ub en tag args to_bind body)
  = bind_block_fields ub en tag args to_bind (erase c body).
Proof.
  unfold bind_block_fields.
  change (Some (erase c body)) with (option_map (erase c) (Some body)).
  generalize (Some body) as acc.
  induction to_bind as [ | [f u] to_bind IH]; intros acc; [ reflexivity | ].
  cbn [List.fold_left]. rewrite IH. f_equal.
  destruct acc as [b0 | ]; [ | reflexivity ]. cbn [option_map].
  destruct f; try reflexivity;
    try apply bind_field_to_simple_erase.
  match goal with
  | |- context [unboxed_of ?l ?a] => destruct (unboxed_of l a)
  end; [ apply bind_fields_erase | apply bind_field_to_simple_erase ].
Qed.

Lemma bind_boxed_number_field_erase (bn : boxable_number) (contents : simple)
    (to_bind : unboxed_fields) (body : expr) :
  option_map (erase c) (bind_boxed_number_field bn contents to_bind body)
  = bind_boxed_number_field bn contents to_bind (erase c body).
Proof.
  unfold bind_boxed_number_field.
  change (Some (erase c body)) with (option_map (erase c) (Some body)).
  generalize (Some body) as acc.
  induction to_bind as [ | [f u] to_bind IH]; intros acc; [ reflexivity | ].
  cbn [List.fold_left]. rewrite IH. f_equal.
  destruct acc as [b0 | ]; [ | reflexivity ]. cbn [option_map].
  destruct f; try reflexivity.
  destruct (Nat.eqb _ _); [ apply bind_field_to_simple_erase | reflexivity ].
Qed.

Lemma load_unboxed_field_erase (ub : list (node * unboxed_fields))
    (f : field) (arg : simple) (to_bind : unboxed_fields) (body : expr) :
  option_map (erase c) (load_unboxed_field ub [] f arg to_bind body)
  = load_unboxed_field ub [] f arg to_bind (erase c body).
Proof.
  unfold load_unboxed_field. destruct (unboxed_of ub arg) as [af | ].
  - destruct (Keyed.lookup field_eqb f af); [ apply bind_fields_erase | ].
    reflexivity.
  - destruct arg; reflexivity.
Qed.

Lemma erase_opt (o o' : option expr) :
  option_map (erase c) o = o' ->
  erase c (match o with Some e => e | None => Invalid end)
  = match o' with Some e => e | None => Invalid end.
Proof. destruct o; cbn; intros <-; reflexivity. Qed.

Lemma rebuild_unboxed_binding_erase (ub : list (node * unboxed_fields))
    (en : rebuild_env) (to_bind : unboxed_fields) (d : named)
    (body : expr) :
  region_prim_of d = None ->
  erase c (rebuild_unboxed_binding ub [] en to_bind d body)
  = rebuild_unboxed_binding ub [] en to_bind d (erase c body).
Proof.
  intros Hd. unfold rebuild_unboxed_binding. apply erase_opt.
  destruct d as [s | p args | | | w arg | mf w arg | mf mt arg];
    try reflexivity.
  - destruct (unboxed_of ub s); [ apply bind_fields_erase | reflexivity ].
  - destruct p as [p | p | p | p | p]; try reflexivity.
    + destruct p; try reflexivity;
        destruct args as [ | a [ | ]]; try reflexivity;
        first [ apply load_unboxed_field_erase
              | apply bind_boxed_number_field_erase ].
    + destruct p as [bk | ak]; [ | reflexivity ].
      destruct bk; try reflexivity. apply bind_block_fields_erase.
    + destruct p; try reflexivity. apply bind_block_fields_erase.
    + discriminate Hd.
  - apply load_unboxed_field_erase.
  - apply load_unboxed_field_erase.
  - destruct (unboxed_of ub arg); [ apply bind_fields_erase | reflexivity ].
Qed.

Lemma rebuild_let_u_erase (sol : solution) (ub : list (node * unboxed_fields))
    (en : rebuild_env) (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  region_prim_of d = None ->
  erase c (rebuild_let_u sol ub [] en x kind d body)
  = rebuild_let_u sol ub [] en x kind d (erase c body).
Proof.
  intros Hd. unfold rebuild_let_u.
  destruct (field_read d) as [[f arg] | ].
  - destruct (unboxed_of ub arg) as [af | ].
    + destruct (Keyed.lookup field_eqb f af) as [[v | n] | ]; reflexivity.
    + destruct arg; cbn [changed_of changed_of_node Keyed.lookup];
        apply erase_let_plain, rebuild_named_plain, Hd.
  - apply erase_let_plain, rebuild_named_plain, Hd.
Qed.

End Erase.

(* ======================================================================== *)
(* The environments of the two rebuilds                                     *)
(* ======================================================================== *)

(* That of the rebuild of the erased term ([en']) is that of the term
   ([en]) without the variables of the regions. *)
Definition uenv_rel (en en' : unbox_env) : Prop :=
  env_rel (ue_env en) (ue_env en')
  ∧ ue_decisions en' = ue_decisions en
  ∧ ue_drop_projections en' = ue_drop_projections en.

(* No projection is pending, and none is to be dropped. *)
Definition uenv_plain (en : unbox_env) : Prop :=
  env_projections (ue_env en) = [] ∧ ue_drop_projections en = false.

Lemma uenv_rel_refl (en : unbox_env) : uenv_rel en en.
Proof.
  split; [ apply env_rel_refl | split; reflexivity ].
Qed.

Lemma uenv_rel_bind (en en' : unbox_env)
    (bound : list (variable * flambda_kind)) :
  uenv_rel en en' ->
  uenv_rel (ubind_variables en bound) (ubind_variables en' bound).
Proof.
  intros (He & Hd & _). split; [ apply (env_rel_bind_then _ _ _ [] He) | ].
  split; [ exact Hd | reflexivity ].
Qed.

Lemma uenv_plain_bind (en : unbox_env)
    (bound : list (variable * flambda_kind)) :
  uenv_plain (ubind_variables en bound).
Proof. split; reflexivity. Qed.

Lemma uenv_rel_region (en en' : unbox_env) (r : variable) :
  uenv_rel en en' -> uenv_plain en ->
  uenv_rel (ubind_variables en [(r, Region)]) en'.
Proof.
  intros (He & Hd & Hp) (Hproj & Hdrop). split;
    [ exact (env_rel_bind_region _ _ r He Hproj) | ].
  split; [ exact Hd | rewrite Hp, Hdrop; reflexivity ].
Qed.

Lemma uenv_rel_twice (en en' : unbox_env) (x : variable)
    (kind : flambda_kind) :
  uenv_rel en en' ->
  uenv_rel (ubind_variables (ubind_variables en [(x, kind)]) [(x, kind)])
    (ubind_variables en' [(x, kind)]).
Proof.
  intros (He & Hd & _). split; [ exact (env_rel_bind_twice _ _ x kind He) | ].
  split; [ exact Hd | reflexivity ].
Qed.

Lemma uenv_rel_conts (en en' : unbox_env)
    (bound : list (continuation * list decision)) :
  uenv_rel en en' ->
  uenv_rel (ubind_continuations en bound) (ubind_continuations en' bound).
Proof.
  intros (He & Hd & _). split; [ apply env_rel_bind_continuations, He | ].
  split; [ cbn; rewrite Hd; reflexivity | reflexivity ].
Qed.

Lemma uenv_plain_conts (en : unbox_env)
    (bound : list (continuation * list decision)) :
  uenv_plain (ubind_continuations en bound).
Proof. split; reflexivity. Qed.

Lemma pending_plain (en : unbox_env) (x : variable) (d : named) :
  uenv_plain en -> is_pending_projection (ue_env en) x d = false.
Proof.
  intros (Hp & _). unfold is_pending_projection. rewrite Hp. reflexivity.
Qed.

Lemma pending_plain_rel (en en' : unbox_env) (x : variable) (d : named) :
  uenv_rel en en' -> uenv_plain en ->
  is_pending_projection (ue_env en') x d = false.
Proof.
  intros ((_ & Hp & _) & _) (Hproj & _). unfold is_pending_projection.
  rewrite Hp, Hproj. reflexivity.
Qed.

(* ======================================================================== *)
(* The decisions make no parameter of the kind [Region]                     *)
(* ======================================================================== *)

Definition decision_plain (d : decision) : bool :=
  forallb kind_not_region (decision_kinds d).

Lemma decision_params_kinds (x : variable) (d : decision) :
  map snd (decision_params x d) = decision_kinds d.
Proof. destruct d as [[k | ] | fields | k]; reflexivity. Qed.

Lemma params_of_decisions_plain (bound : list (variable * flambda_kind)) :
  ∀ decisions : list decision,
    forallb kind_not_region (map snd bound) = true ->
    forallb decision_plain decisions = true ->
    forallb kind_not_region (map snd (params_of_decisions bound decisions))
    = true.
Proof.
  induction bound as [ | [x kind] bound IH]; intros ds Hb Hds;
    [ reflexivity | ].
  cbn [map forallb snd] in Hb. apply andb_true_iff in Hb as [Hk Hb].
  destruct ds as [ | d ds]; cbn [params_of_decisions map forallb snd].
  - rewrite Hk. exact (IH [] Hb eq_refl).
  - cbn [forallb] in Hds. apply andb_true_iff in Hds as [Hd Hds].
    rewrite map_app, forallb_app, decision_params_kinds. unfold decision_plain
      in Hd. rewrite Hd. exact (IH ds Hb Hds).
Qed.

Lemma with_kinds_plain (params : list variable) :
  ∀ kinds : list flambda_kind,
    forallb kind_not_region kinds = true ->
    forallb kind_not_region (map snd (with_kinds params kinds)) = true.
Proof.
  induction params as [ | x params IH]; intros kinds Hk; [ reflexivity | ].
  destruct kinds as [ | k kinds]; cbn [with_kinds map forallb snd].
  - exact (IH [] eq_refl).
  - cbn [forallb] in Hk. apply andb_true_iff in Hk as [Hk Hks].
    rewrite Hk. exact (IH kinds Hks).
Qed.

Section Decisions.

Variable ub : list (node * unboxed_fields).
Hypothesis Hleaves : ∀ (n : node) (fields : unboxed_fields),
  (n, fields) ∈ ub -> forallb kind_not_region (map snd (fields_leaves fields))
                      = true.

Lemma handler_decisions_plain (fx : bool) (s0 : solution) (is_exn : bool)
    (h : continuation_handler) :
  forallb kind_not_region (handler_kinds h) = true ->
  forallb decision_plain (handler_decisions fx s0 ub is_exn h) = true.
Proof.
  destruct h as [params md hbody]. cbn [handler_kinds handler_decisions].
  intros Hk. pose proof (with_kinds_plain params _ Hk) as Hb.
  apply forallb_forall. intros d Hd. apply in_map_iff in Hd
    as ([[x kind] keep] & <- & Hin).
  apply in_combine_l in Hin.
  assert (Hkind : kind_not_region kind = true).
  { rewrite forallb_forall in Hb. apply Hb.
    exact (in_map snd _ (x, kind) Hin). }
  destruct (unboxed_of_node ub (Node_variable x)) as [fields | ] eqn:Hu.
  - unfold decision_plain. cbn [decision_kinds decision_params].
    apply (Hleaves (Node_variable x)).
    exact (keyed_lookup_in node_eqb
             FlambdaReaperUnboxingRefinesProofs.node_eqb_decides _ _ _ Hu).
  - destruct (fx && keep && negb (is_var_used s0 x kind));
      [ unfold decision_plain; cbn; rewrite Hkind; reflexivity | ].
    destruct keep; unfold decision_plain; cbn; rewrite ?Hkind; reflexivity.
Qed.

End Decisions.

(* ======================================================================== *)
(* The parts of the two rebuilds                                            *)
(* ======================================================================== *)

Section Parts.

Variables sol sol' : solution.
Variable ub : list (node * unboxed_fields).

Hypothesis Hu : ∀ n : node,
  n <> Node_all_constants -> s_has_usage sol' n = s_has_usage sol n.
Hypothesis Hsrc : s_has_source sol' = s_has_source sol.
Hypothesis Hfu : s_field_used sol' = s_field_used sol.

Lemma hu_u : ∀ n : node,
  n <> Node_all_constants ->
  s_has_usage (solution_without_unboxed sol' ub) n
  = s_has_usage (solution_without_unboxed sol ub) n.
Proof.
  intros n Hn. cbn [solution_without_unboxed s_has_usage].
  rewrite (Hu n Hn). reflexivity.
Qed.

Lemma hfu_u :
  s_field_used (solution_without_unboxed sol' ub)
  = s_field_used (solution_without_unboxed sol ub).
Proof. exact Hfu. Qed.

Lemma handler_decisions_eq (fx is_exn : bool) (h : continuation_handler) :
  handler_decisions fx sol' ub is_exn h
  = handler_decisions fx sol ub is_exn h.
Proof.
  destruct h as [params md hbody]. cbn [handler_decisions].
  rewrite (params_to_keep_eq sol sol' Hu). apply map_ext.
  intros [[x kind] keep]. rewrite (var_used_eq sol sol' Hu). reflexivity.
Qed.

Section Env.

Variables en en' : rebuild_env.
Hypothesis Hen : env_rel en en'.

Lemma simple_kind_eq (s : simple) :
  simple_not_region (env_kinds en) s = true ->
  FlambdaReaperUnboxing.simple_kind en' s
  = FlambdaReaperUnboxing.simple_kind en s.
Proof.
  intros H. destruct s as [x | | ]; [ | reflexivity | reflexivity ].
  cbn [FlambdaReaperUnboxing.simple_kind].
  apply (kind_of_variable_rel _ _ _ Hen). apply negb_true_iff in H. exact H.
Qed.

Lemma bind_block_fields_env (tag : Z) (args : list simple)
    (to_bind : unboxed_fields) (body : expr) :
  forallb (simple_not_region (env_kinds en)) args = true ->
  bind_block_fields ub en' tag args to_bind body
  = bind_block_fields ub en tag args to_bind body.
Proof.
  intros Hargs. unfold bind_block_fields. generalize (Some body) as acc.
  induction to_bind as [ | [f u] to_bind IH]; intros acc; [ reflexivity | ].
  cbn [List.fold_left]. rewrite IH. f_equal.
  destruct acc as [b0 | ]; [ | reflexivity ].
  destruct f as [nth kind | | | | | | | | ]; try reflexivity.
  destruct (nth_error args nth) as [a | ] eqn:Hn; [ | reflexivity ].
  rewrite forallb_forall in Hargs.
  rewrite (simple_kind_eq a (Hargs a (nth_error_In _ _ Hn))).
  reflexivity.
Qed.

Lemma rebuild_unboxed_binding_env (to_bind : unboxed_fields) (d : named)
    (body : expr) :
  forallb (simple_not_region (env_kinds en)) (simples_of_named d) = true ->
  rebuild_unboxed_binding ub [] en' to_bind d body
  = rebuild_unboxed_binding ub [] en to_bind d body.
Proof.
  intros Hd. unfold rebuild_unboxed_binding.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  destruct p as [p | p | p | p | p]; try reflexivity.
  - destruct p as [bk | ak]; [ | reflexivity ].
    destruct bk; try reflexivity.
    rewrite (bind_block_fields_env _ _ _ _ Hd). reflexivity.
  - destruct p; try reflexivity.
    rewrite (bind_block_fields_env _ _ _ _ Hd). reflexivity.
Qed.

Lemma rebuild_let_u_env (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  forallb (simple_not_region (env_kinds en)) (simples_of_named d) = true ->
  rebuild_let_u sol' ub [] en' x kind d body
  = rebuild_let_u sol ub [] en x kind d body.
Proof.
  intros Hd. unfold rebuild_let_u.
  rewrite (rebuild_named_eq _ _ hu_u hfu_u en en' Hen x d Hd).
  reflexivity.
Qed.

Lemma get_args_eq (decisions : list decision) (args : list simple) :
  forallb (simple_not_region (env_kinds en)) args = true ->
  get_args sol' ub en' decisions args = get_args sol ub en decisions args.
Proof.
  revert decisions. induction args as [ | a args IH]; intros ds Hargs;
    [ destruct ds; reflexivity | ].
  cbn [forallb] in Hargs. apply andb_true_iff in Hargs as [Ha Hargs].
  pose proof (rewrite_simple_eq _ _ hu_u en en' Hen a Ha) as Hr.
  destruct ds as [ | d ds]; cbn [get_args].
  - rewrite (IH [] Hargs), Hr. reflexivity.
  - rewrite (IH ds Hargs), Hr. reflexivity.
Qed.

Lemma has_no_source_eq (args : list simple) :
  existsb (simple_has_no_source sol') args
  = existsb (simple_has_no_source sol) args.
Proof. unfold simple_has_no_source. rewrite Hsrc. reflexivity. Qed.

End Env.

Lemma rewrite_apply_cont_u_eq (en en' : unbox_env) (ac : apply_cont) :
  uenv_rel en en' ->
  forallb (simple_not_region (env_kinds (ue_env en))) (ac_args ac) = true ->
  rewrite_apply_cont_u sol' ub en' ac = rewrite_apply_cont_u sol ub en ac.
Proof.
  intros (He & Hd & _) Hargs. unfold rewrite_apply_cont_u.
  rewrite has_no_source_eq, Hd.
  destruct (lookup (ac_continuation ac) (ue_decisions en)).
  - rewrite (get_args_eq _ _ He _ _ Hargs). reflexivity.
  - rewrite (map_rewrite_eq _ _ hu_u _ _ He _ Hargs). reflexivity.
Qed.

Lemma rebuild_apply_cont_u_eq (en en' : unbox_env) (ac : apply_cont) :
  uenv_rel en en' ->
  forallb (simple_not_region (env_kinds (ue_env en))) (ac_args ac) = true ->
  rebuild_apply_cont_u sol' ub en' ac = rebuild_apply_cont_u sol ub en ac.
Proof.
  intros He Hargs. unfold rebuild_apply_cont_u.
  rewrite (rewrite_apply_cont_u_eq _ _ _ He Hargs). reflexivity.
Qed.

Lemma rebuild_switch_u_eq (en en' : unbox_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  uenv_rel en en' ->
  simple_not_region (env_kinds (ue_env en)) scrutinee = true ->
  forallb (fun '(_, ac) => forallb (simple_not_region (env_kinds (ue_env en)))
                             (ac_args ac)) arms = true ->
  rebuild_switch_u sol' ub en' scrutinee arms
  = rebuild_switch_u sol ub en scrutinee arms.
Proof.
  intros He Hs Harms. unfold rebuild_switch_u.
  rewrite (rewrite_simple_eq _ _ hu_u _ _ (proj1 He) _ Hs).
  rewrite (flat_map_ext_in
             (fun '(d, ac) =>
                match rewrite_apply_cont_u sol' ub en' ac with
                | Some ac0 => [(d, ac0)]
                | None => []
                end)
             (fun '(d, ac) =>
                match rewrite_apply_cont_u sol ub en ac with
                | Some ac0 => [(d, ac0)]
                | None => []
                end)); [ reflexivity | ].
  intros [d ac] Hin. apply in_compact_arms in Hin.
  rewrite forallb_forall in Harms.
  rewrite (rewrite_apply_cont_u_eq _ _ ac He (Harms _ Hin)). reflexivity.
Qed.

End Parts.

(* ======================================================================== *)
(* The steps of the rebuild with unboxing                                   *)
(* ======================================================================== *)

Section Unfold.

Variable fx : bool.
Variables (F0 : facts) (s0 : solution) (cc0 : node -> bool)
  (cds0 : finite_map code_dep) (b0 : bool) (u0 : list code_id)
  (c0 : finite_map code_metadata) (ub0 : list (node * unboxed_fields))
  (n0 : positive).

Abbreviation RU0 := (rebuild_expr_u fx F0 s0 cc0 cds0 b0 u0 c0 ub0 n0 []
                       None).
Abbreviation RH0 := (rebuild_handler_u fx F0 s0 cc0 cds0 b0 u0 c0 ub0 n0
                       [] None).

Lemma ru_let_other (en : unbox_env) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  is_pending_projection (ue_env en) x d = false -> named_is_set d = false ->
  RU0 en (Let x kind d body)
  = if is_dead_var s0 x kind then Invalid
    else if is_var_used s0 x kind
    then
      match unboxed_of_node ub0 (Node_variable x) with
      | Some to_bind =>
        rebuild_unboxed_binding ub0 [] (ue_env en) to_bind d
          (RU0 (ubind_variables en [(x, kind)]) body)
      | None =>
        rebuild_let_u s0 ub0 [] (ue_env en) x kind d
          (RU0 (ubind_variables en [(x, kind)]) body)
      end
    else RU0 (ubind_variables en [(x, kind)]) body.
Proof.
  intros H Hs. cbn [rebuild_expr_u]. rewrite H.
  destruct d; try discriminate Hs; reflexivity.
Qed.

Lemma ru_let_static (en : unbox_env) (bindings : list static_binding)
    (body : expr) :
  RU0 en (Let_static bindings body)
  = Let_static
      (flat_map (rebuild_binding_u fx F0 s0 cc0 cds0 b0 u0 c0 ub0 n0 []
                   None en) bindings)
      (RU0 (ubind_variables en []) body).
Proof. reflexivity. Qed.

Lemma ru_let_cont (en : unbox_env) (k : continuation)
    (h : continuation_handler) (body : expr) :
  RU0 en (Let_cont_non_recursive k h body)
  = if among k
         (free_continuations_of_expr
            (RU0 (ubind_continuations en
                    [(k, handler_decisions fx s0 ub0
                           (handler_is_exn_handler h) h)])
               body))
    then
      Let_cont_non_recursive k
        (RH0 en true
           (handler_decisions fx s0 ub0 (handler_is_exn_handler h) h) h)
        (RU0 (ubind_continuations en
                [(k, handler_decisions fx s0 ub0
                       (handler_is_exn_handler h) h)])
           body)
    else
      RU0 (ubind_continuations en
             [(k, handler_decisions fx s0 ub0
                    (handler_is_exn_handler h) h)])
        body.
Proof. reflexivity. Qed.

Lemma ru_let_cont_rec (en : unbox_env)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  RU0 en (Let_cont_recursive handlers body)
  = Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k, RH0 (ubind_continuations en
                       (map (fun '(k, h) =>
                               (k, handler_decisions fx s0 ub0 false h))
                          handlers))
                  false (handler_decisions fx s0 ub0 false h) h))
         handlers)
      (RU0 (ubind_continuations en
              (map (fun '(k, h) => (k, handler_decisions fx s0 ub0 false h))
                 handlers))
         body).
Proof. reflexivity. Qed.

Lemma rh_unfold (en : unbox_env) (dc : bool) (ds : list decision)
    (params : list variable) (md : handler_metadata) (hbody : expr) :
  RH0 en dc ds (Handler params md hbody)
  = Handler
      (map fst (params_of_decisions (with_kinds params (hm_params_kinds md))
                  ds))
      (with_params_kinds md
         (map snd (params_of_decisions
                     (with_kinds params (hm_params_kinds md)) ds)))
      (if dc
          && existsb (fun '(x, kind) => is_dead_var s0 x kind)
               (with_kinds params (hm_params_kinds md))
       then Invalid
       else RU0 (ubind_variables en (with_kinds params (hm_params_kinds md)))
              hbody).
Proof. reflexivity. Qed.

End Unfold.

Lemma with_params_kinds_plain (md : handler_metadata)
    (kinds : list flambda_kind) :
  forallb kind_not_region kinds = true ->
  forallb kind_not_region (hm_params_kinds (with_params_kinds md kinds))
  = true.
Proof.
  intros H. unfold with_params_kinds.
  destruct (hm_is_exn_handler md); exact H.
Qed.

(* ======================================================================== *)
(* Helpers of the commutation                                               *)
(* ======================================================================== *)

Lemma region_binders_let (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  ∀ r : variable,
    r ∈ region_binders body -> r ∈ region_binders (Let x kind d body).
Proof.
  intros r Hr.
  destruct d as [ | [ | | | | [ | | | | ]] | | | | | ]; cbn [region_binders];
    first [ exact Hr | right; exact Hr ].
Qed.

Lemma handler_decisions_erase (fx : bool) (s0 : solution)
    (ub : list (node * unboxed_fields)) (is_exn : bool)
    (h : continuation_handler) (c : region_conts) :
  forallb kind_not_region (handler_kinds h) = true ->
  handler_decisions fx s0 ub is_exn (erase_handler c h)
  = handler_decisions fx s0 ub is_exn h.
Proof.
  destruct h as [params md hbody]. intros H. cbn [handler_kinds] in H.
  rewrite (erase_handler_no_region c params md hbody H). reflexivity.
Qed.

Lemma erase_rebuild_apply_cont_u (s0 : solution)
    (ub : list (node * unboxed_fields)) (en : unbox_env) (ac : apply_cont) :
  erase [] (rebuild_apply_cont_u s0 ub en ac)
  = rebuild_apply_cont_u s0 ub en ac.
Proof.
  unfold rebuild_apply_cont_u. destruct (rewrite_apply_cont_u s0 ub en ac);
    reflexivity.
Qed.

Lemma erase_rebuild_switch_u (s0 : solution)
    (ub : list (node * unboxed_fields)) (en : unbox_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  erase [] (rebuild_switch_u s0 ub en scrutinee arms)
  = rebuild_switch_u s0 ub en scrutinee arms.
Proof.
  unfold rebuild_switch_u.
  match goal with
  |- context [ Keyed.is_empty ?l ] => destruct (Keyed.is_empty l)
  end; [ reflexivity | ].
  cbn [erase]. rewrite map_arms_id. reflexivity.
Qed.

(* A variable used and not unboxed is kept where it is bound. *)
Lemma rewrite_bound_used (s0 : solution) (ub : list (node * unboxed_fields))
    (en : rebuild_env) (x : variable) (kind : flambda_kind) :
  is_var_used s0 x kind = true ->
  unboxed_of_node ub (Node_variable x) = None ->
  rewrite_simple (solution_without_unboxed s0 ub)
    (bind_variables en [(x, kind)]) (Var x)
  = Var x.
Proof.
  intros Hused Hnu. cbn [rewrite_simple]. rewrite kind_of_bound.
  assert (Hb : is_unboxed_name ub (Node_variable x) = false).
  { unfold is_unboxed_name, Keyed.is_bound. unfold unboxed_of_node in Hnu.
    rewrite Hnu. reflexivity. }
  destruct kind; cbn [is_var_used] in Hused |- *; try reflexivity;
    cbn [solution_without_unboxed s_has_usage]; rewrite Hused, Hb;
    reflexivity.
Qed.

(* The places of the regions of a handler that the rebuild with unboxing
   makes. *)
Lemma rebuilt_u_handler_places (fx : bool) (F0 : facts) (s0 : solution)
    (cc0 : node -> bool) (cds0 : finite_map code_dep) (b0 : bool)
    (u0 : list code_id) (c0 : finite_map code_metadata)
    (ub0 : list (node * unboxed_fields)) (n0 : positive) (en : unbox_env)
    (dc : bool) (ds : list decision) (h : continuation_handler) :
  forallb kind_not_region (handler_kinds h) = true ->
  forallb decision_plain ds = true ->
  forallb negb
    (region_params
       (rebuild_handler_u fx F0 s0 cc0 cds0 b0 u0 c0 ub0 n0 [] None en dc
          ds h))
  = true.
Proof.
  destruct h as [params md hbody]. intros Hk Hds.
  rewrite rh_unfold. unfold region_params. cbn [handler_kinds].
  apply region_places_plain, with_params_kinds_plain.
  apply params_of_decisions_plain; [ | exact Hds ].
  apply with_kinds_plain. exact Hk.
Qed.

Lemma rebuilt_u_handler_plain (fx : bool) (F0 : facts) (s0 : solution)
    (cc0 : node -> bool) (cds0 : finite_map code_dep) (b0 : bool)
    (u0 : list code_id) (c0 : finite_map code_metadata)
    (ub0 : list (node * unboxed_fields)) (n0 : positive) (en : unbox_env)
    (dc : bool) (ds : list decision) (h : continuation_handler) :
  forallb kind_not_region (handler_kinds h) = true ->
  forallb decision_plain ds = true ->
  forallb kind_not_region
    (handler_kinds
       (rebuild_handler_u fx F0 s0 cc0 cds0 b0 u0 c0 ub0 n0 [] None en dc
          ds h))
  = true.
Proof.
  destruct h as [params md hbody]. intros Hk Hds.
  rewrite rh_unfold. cbn [handler_kinds].
  apply with_params_kinds_plain.
  apply params_of_decisions_plain; [ | exact Hds ].
  apply with_kinds_plain. exact Hk.
Qed.

(* ======================================================================== *)
(* The commutation                                                          *)
(* ======================================================================== *)

Section Commute.

(* The facts, the solution, the code ids and the metadata of the code of
   the term ([F], [sol], ...) and of the erased term ([F'], [sol'], ...):
   the rebuild of a term with no closure, call or code reads none of the
   facts, code ids or metadata, and the solutions differ by the usages of
   the constants at most. *)
Variable fx : bool.
Variables F F' : facts.
Variables sol sol' : solution.
Variable cc : node -> bool.
Variable cds : finite_map code_dep.
Variable b : bool.
Variables ucids ucids' : list code_id.
Variables codes codes' : finite_map code_metadata.
Variable ub : list (node * unboxed_fields).
Variable next : positive.

Hypothesis Hu : ∀ n : node,
  n <> Node_all_constants -> s_has_usage sol' n = s_has_usage sol n.
Hypothesis Hsrc : s_has_source sol' = s_has_source sol.
Hypothesis Hfu : s_field_used sol' = s_field_used sol.
(* No variable of an unboxed field is of the kind [Region]. *)
Hypothesis Hleaves : ∀ (n : node) (fields : unboxed_fields),
  (n, fields) ∈ ub -> forallb kind_not_region (map snd (fields_leaves fields))
                      = true.

Abbreviation RU := (rebuild_expr_u fx F sol cc cds b ucids codes ub next []
                      None).
Abbreviation RU' := (rebuild_expr_u fx F' sol' cc cds b ucids' codes' ub
                       next [] None).
Abbreviation RH := (rebuild_handler_u fx F sol cc cds b ucids codes ub next
                      [] None).
Abbreviation RH' := (rebuild_handler_u fx F' sol' cc cds b ucids' codes' ub
                       next [] None).

(* The variables of the regions are not unboxed. *)
Definition begins_kept (l : list variable) : Prop :=
  ∀ r : variable, r ∈ l -> unboxed_of_node ub (Node_variable r) = None.

Lemma rebuild_u_erase_commutes :
  (∀ e : expr, ∀ en en' : unbox_env,
     regions_plain (env_kinds (ue_env en)) e = true ->
     closure_free e = true ->
     begins_kept (region_binders e) ->
     uenv_rel en en' -> uenv_plain en ->
     erase [] (RU en e) = RU' en' (erase [] e))
  ∧ (∀ h : continuation_handler,
       ∀ (en en' : unbox_env) (dc : bool) (ds : list decision),
         handler_regions_plain (env_kinds (ue_env en)) h = true ->
         handler_closure_free h = true ->
         begins_kept (handler_region_binders h) ->
         forallb decision_plain ds = true ->
         uenv_rel en en' ->
         erase_handler [] (RH en dc ds h)
         = RH' en' dc ds (erase_handler [] h)).
Proof.
  apply regions_ind.
  - (* Let *)
    intros x kind d body IH IH2 en en' Hpl Hcf Hbk Hen Hpe.
    cbn [closure_free] in Hcf. apply andb_true_iff in Hcf as [Hcd Hcb].
    assert (Hset : named_is_set d = false)
      by (destruct d; try discriminate Hcd; reflexivity).
    rewrite (ru_let_other fx F sol cc cds b ucids codes ub next en x kind d body
               (pending_plain en x d Hpe) Hset).
    destruct (region_prim_of d) as [[p args] | ] eqn:Hrp.
    + apply region_prim_of_some in Hrp. subst d.
      destruct p; destruct args as [ | s [ | s' args]];
        cbn [regions_plain] in Hpl; try discriminate Hpl.
      * (* Begin_region *)
        apply andb_true_iff in Hpl as [Hk Hb].
        apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk. subst kind.
        assert (Hx : unboxed_of_node ub (Node_variable x) = None)
          by (apply Hbk; cbn [region_binders]; left; reflexivity).
        cbn [is_dead_var is_var_used]. rewrite Hx.
        unfold rebuild_let_u.
        cbn [field_read rebuild_named map erase erase_region_primitive].
        apply IH; [ exact Hb | exact Hcb
                  | intros r Hr; apply Hbk; cbn [region_binders]; right;
                    exact Hr
                  | apply uenv_rel_region; assumption
                  | apply uenv_plain_bind ].
      * (* Begin_try_region *)
        apply andb_true_iff in Hpl as [Hk Hb].
        apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk. subst kind.
        assert (Hx : unboxed_of_node ub (Node_variable x) = None)
          by (apply Hbk; cbn [region_binders]; left; reflexivity).
        cbn [is_dead_var is_var_used]. rewrite Hx.
        unfold rebuild_let_u.
        cbn [field_read rebuild_named map erase erase_region_primitive].
        apply IH; [ exact Hb | exact Hcb
                  | intros r Hr; apply Hbk; cbn [region_binders]; right;
                    exact Hr
                  | apply uenv_rel_region; assumption
                  | apply uenv_plain_bind ].
      * (* End_region *)
        apply andb_true_iff in Hpl as [_ Hb].
        assert (Hbk' : begins_kept (region_binders body))
          by (intros r Hr; apply Hbk, region_binders_let, Hr).
        cbn [erase erase_region_primitive].
        rewrite (ru_let_other fx F' sol' cc cds b ucids' codes' ub next en' x
                   kind (Simple unit_simple) (erase [] body)
                   (pending_plain_rel en en' x _ Hen Hpe) eq_refl).
        rewrite (dead_var_eq sol sol' Hsrc), (var_used_eq sol sol' Hu).
        destruct (is_dead_var sol x kind); [ reflexivity | ].
        destruct (is_var_used sol x kind).
        -- destruct (unboxed_of_node ub (Node_variable x)) as [tb | ];
             [ reflexivity | ].
           unfold rebuild_let_u.
           cbn [field_read rebuild_named map erase erase_region_primitive
                  rewrite_simple unit_simple].
           f_equal. apply IH; [ exact Hb | exact Hcb | exact Hbk'
                              | apply uenv_rel_bind, Hen
                              | apply uenv_plain_bind ].
        -- apply IH; [ exact Hb | exact Hcb | exact Hbk'
                     | apply uenv_rel_bind, Hen | apply uenv_plain_bind ].
      * (* End_try_region *)
        apply andb_true_iff in Hpl as [_ Hb].
        assert (Hbk' : begins_kept (region_binders body))
          by (intros r Hr; apply Hbk, region_binders_let, Hr).
        cbn [erase erase_region_primitive].
        rewrite (ru_let_other fx F' sol' cc cds b ucids' codes' ub next en' x
                   kind (Prim (Unary (Opaque_identity Value)) [unit_simple])
                   (erase [] body) (pending_plain_rel en en' x _ Hen Hpe)
                   eq_refl).
        rewrite (dead_var_eq sol sol' Hsrc), (var_used_eq sol sol' Hu).
        destruct (is_dead_var sol x kind); [ reflexivity | ].
        destruct (is_var_used sol x kind).
        -- destruct (unboxed_of_node ub (Node_variable x)) as [tb | ];
             [ reflexivity | ].
           unfold rebuild_let_u.
           cbn [field_read rebuild_named map erase erase_region_primitive
                  rewrite_simple unit_simple].
           f_equal. apply IH; [ exact Hb | exact Hcb | exact Hbk'
                              | apply uenv_rel_bind, Hen
                              | apply uenv_plain_bind ].
        -- apply IH; [ exact Hb | exact Hcb | exact Hbk'
                     | apply uenv_rel_bind, Hen | apply uenv_plain_bind ].
    + (* Not a primitive of regions. *)
      rewrite (regions_plain_let _ _ _ _ _ Hrp) in Hpl.
      apply andb_true_iff in Hpl as [Hpl Hla].
      apply andb_true_iff in Hpl as [Hpl _].
      apply andb_true_iff in Hpl as [Hk Hd].
      assert (Hbk' : begins_kept (region_binders body))
        by (intros r Hr; apply Hbk, region_binders_let, Hr).
      rewrite (erase_let_plain _ _ _ _ _ Hrp).
      rewrite (ru_let_other fx F' sol' cc cds b ucids' codes' ub next en' x kind
                 d (erase [] body) (pending_plain_rel en en' x d Hen Hpe)
                 Hset).
      rewrite (dead_var_eq sol sol' Hsrc), (var_used_eq sol sol' Hu).
      destruct (is_dead_var sol x kind) eqn:Hdead; [ reflexivity | ].
      unfold la_check in Hla.
      destruct (la_body body) as [[[[y k'] args] body'] | ] eqn:Hlab.
      * (* A local allocation follows. *)
        apply la_body_some in Hlab. subst body.
        apply andb_true_iff in Hla as [Hla Hb'].
        apply andb_true_iff in Hla as [Hla Hargs].
        apply andb_true_iff in Hla as [Hla Hk'].
        apply andb_true_iff in Hla as [_ Hy].
        apply Pos.eqb_eq in Hy. subst y.
        apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk'. subst k'.
        destruct args as [ | [x' | | ] rs]; try discriminate Hargs.
        apply Pos.eqb_eq in Hargs. subst x'.
        cbn [closure_free named_closure_free andb] in Hcb.
        assert (Hbk'' : begins_kept (region_binders body'))
          by (intros r Hr; apply Hbk'; exact Hr).
        cbn [erase erase_region_primitive]. rewrite Pos.eqb_refl.
        set (en1 := ubind_variables en [(x, kind)]).
        assert (IHb : erase [] (RU (ubind_variables en1 [(x, kind)]) body')
                      = RU' (ubind_variables en' [(x, kind)]) (erase [] body')).
        { apply IH2 with (y := x) (kind' := kind)
            (d' := Prim (Region_primitive Local_allocation) (Var x :: rs));
            [ reflexivity | exact Hb' | exact Hcb | exact Hbk''
            | apply uenv_rel_twice, Hen | apply uenv_plain_bind ]. }
        rewrite (ru_let_other fx F sol cc cds b ucids codes ub next en1 x kind
                   (Prim (Region_primitive Local_allocation) (Var x :: rs))
                   body' (pending_plain en1 x _ (uenv_plain_bind _ _))
                   eq_refl).
        rewrite Hdead.
        destruct (is_var_used sol x kind) eqn:Hused; [ | exact IHb ].
        destruct (unboxed_of_node ub (Node_variable x)) as [tb | ] eqn:Hub.
        -- cbn [rebuild_unboxed_binding].
           rewrite (rebuild_unboxed_binding_erase _ _ _ _ _ _ Hrp), IHb.
           symmetry.
           apply (rebuild_unboxed_binding_env _ _ _ (proj1 Hen) _ _ _ Hd).
        -- rewrite (rebuild_let_u_erase _ _ _ _ _ _ _ _ Hrp).
           assert (Hinner :
             erase []
               (rebuild_let_u sol ub [] (ue_env en1) x kind
                  (Prim (Region_primitive Local_allocation) (Var x :: rs))
                  (RU (ubind_variables en1 [(x, kind)]) body'))
             = erase [] (RU (ubind_variables en1 [(x, kind)]) body')).
           { unfold rebuild_let_u. cbn [field_read rebuild_named map].
             change (ue_env en1) with (bind_variables (ue_env en) [(x, kind)]).
             rewrite (rewrite_bound_used sol ub (ue_env en) x kind Hused Hub).
             cbn [erase erase_region_primitive]. rewrite Pos.eqb_refl.
             reflexivity. }
           rewrite Hinner, IHb.
           symmetry.
           apply (rebuild_let_u_env sol sol' ub Hu Hfu _ _ (proj1 Hen) _ _ _
                    _ Hd).
      * (* No local allocation follows. *)
        assert (IHb : erase [] (RU (ubind_variables en [(x, kind)]) body)
                      = RU' (ubind_variables en' [(x, kind)]) (erase [] body))
          by (apply IH; [ exact Hla | exact Hcb | exact Hbk'
                        | apply uenv_rel_bind, Hen | apply uenv_plain_bind ]).
        destruct (is_var_used sol x kind); [ | exact IHb ].
        destruct (unboxed_of_node ub (Node_variable x)) as [tb | ].
        -- rewrite (rebuild_unboxed_binding_erase _ _ _ _ _ _ Hrp), IHb.
           symmetry.
           apply (rebuild_unboxed_binding_env _ _ _ (proj1 Hen) _ _ _ Hd).
        -- rewrite (rebuild_let_u_erase _ _ _ _ _ _ _ _ Hrp), IHb.
           symmetry.
           apply (rebuild_let_u_env sol sol' ub Hu Hfu _ _ (proj1 Hen) _ _ _
                    _ Hd).
  - (* Let_static *)
    intros bindings body _ IH en en' Hpl Hcf Hbk Hen Hpe.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hbs Hb].
    cbn [closure_free] in Hcf. apply andb_true_iff in Hcf as [Hbc Hcb].
    rewrite ru_let_static. cbn [erase]. rewrite ru_let_static. f_equal.
    + clear IH Hb Hcb Hbk Hpe.
      induction bindings as [ | bd bds IHb]; [ reflexivity | ].
      cbn [forallb] in Hbs, Hbc. apply andb_true_iff in Hbs as [Hbd Hbs].
      apply andb_true_iff in Hbc as [Hbdc Hbc].
      cbn [flat_map map]. rewrite map_app. f_equal; [ | exact (IHb Hbs Hbc) ].
      destruct bd as [ | | sym [ | | tag fields | | | ]]; try discriminate Hbdc.
      cbn [erase_binding rebuild_binding_u static_set_repr].
      rewrite (static_const_is_used_eq sol sol' Hu).
      destruct (static_const_is_used sol sym _); [ | reflexivity ].
      cbn [map erase_binding]. cbn [binding_regions_plain] in Hbd.
      rewrite (rebuild_static_const_eq _ _ (hu_u sol sol' ub Hu)
                 (hfu_u sol sol' ub Hfu) _ _ (proj1 Hen) _ _ Hbd).
      reflexivity.
    + apply IH; [ exact Hb | exact Hcb
                | intros r Hr; apply Hbk; cbn [region_binders];
                  apply in_or_app; right; exact Hr
                | apply uenv_rel_bind, Hen | apply uenv_plain_bind ].
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH en en' Hpl Hcf Hbk Hen Hpe.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hh Hb].
    cbn [closure_free] in Hcf. apply andb_true_iff in Hcf as [Hhc Hcb].
    pose proof (handler_kinds_plain _ _ Hh) as Hhk.
    cbn [erase].
    rewrite (erase_plain _ body
               (conts_plain_cons _ _ _ (region_places_plain _ Hhk)
                  conts_plain_nil)).
    rewrite !ru_let_cont.
    rewrite (handler_exn_erase h [] Hhk),
      (handler_decisions_erase _ _ _ _ h [] Hhk),
      (handler_decisions_eq sol sol' ub Hu).
    set (ds := handler_decisions fx sol ub (handler_is_exn_handler h) h).
    assert (Hds : forallb decision_plain ds = true)
      by exact (handler_decisions_plain ub Hleaves fx sol _ h Hhk).
    assert (IHb : erase [] (RU (ubind_continuations en [(k, ds)]) body)
                  = RU' (ubind_continuations en' [(k, ds)]) (erase [] body))
      by (apply IH; [ exact Hb | exact Hcb
                    | intros r Hr; apply Hbk; cbn [region_binders];
                      apply in_or_app; right; exact Hr
                    | apply uenv_rel_conts, Hen | apply uenv_plain_conts ]).
    rewrite <- IHb, (proj1 free_continuations_erase).
    destruct (among k (free_continuations_of_expr
                         (RU (ubind_continuations en [(k, ds)]) body)));
      [ | reflexivity ].
    cbn [erase]. f_equal.
    + apply IHh; [ exact Hh | exact Hhc
                 | intros r Hr; apply Hbk; cbn [region_binders];
                   apply in_or_app; left; exact Hr
                 | exact Hds | exact Hen ].
    + apply erase_plain, conts_plain_cons; [ | apply conts_plain_nil ].
      apply rebuilt_u_handler_places; assumption.
  - (* Let_cont_recursive *)
    intros handlers body IHh IH en en' Hpl Hcf Hbk Hen Hpe.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hhs Hb].
    cbn [closure_free] in Hcf. apply andb_true_iff in Hcf as [Hhc Hcb].
    rewrite forallb_forall in Hhs, Hhc.
    assert (Hplain : ∀ k h, (k, h) ∈ handlers ->
                       forallb kind_not_region (handler_kinds h) = true)
      by (intros k h Hin; exact (handler_kinds_plain _ _ (Hhs (k, h) Hin))).
    rewrite ru_let_cont_rec. cbn [erase]. rewrite ru_let_cont_rec.
    assert (Hds_eq :
      map (fun '(k, h) => (k, handler_decisions fx sol' ub false h))
        (map (fun '(k, h) =>
                (k, erase_handler
                      (map (fun '(k0, h0) => (k0, region_params h0)) handlers
                       ++ []) h)) handlers)
      = map (fun '(k, h) => (k, handler_decisions fx sol ub false h))
          handlers).
    { rewrite map_map. apply map_ext_in. intros [k h] Hin. cbn beta iota.
      rewrite (handler_decisions_erase _ _ _ _ h _ (Hplain k h Hin)),
        (handler_decisions_eq sol sol' ub Hu). reflexivity. }
    rewrite Hds_eq.
    set (en1 := ubind_continuations en
                  (map (fun '(k, h) =>
                          (k, handler_decisions fx sol ub false h))
                     handlers)).
    set (en1' := ubind_continuations en'
                   (map (fun '(k, h) =>
                           (k, handler_decisions fx sol ub false h))
                      handlers)).
    assert (Hen1 : uenv_rel en1 en1') by (apply uenv_rel_conts, Hen).
    f_equal.
    + rewrite !map_map. apply map_ext_in.
      intros [k h] Hin. cbn beta iota.
      assert (Hds : forallb decision_plain
                      (handler_decisions fx sol ub false h) = true)
        by exact (handler_decisions_plain ub Hleaves fx sol _ h
                    (Hplain k h Hin)).
      rewrite (erase_handler_plain _ h)
        by (apply conts_plain_map; intros [k0 h0] Hin0; cbn;
            apply region_places_plain; exact (Hplain k0 h0 Hin0)).
      rewrite (erase_handler_plain _ (rebuild_handler_u _ _ _ _ _ _ _ _ _ _ _
                                        _ _ _ _ _))
        by (apply conts_plain_map; intros [k0 h0] Hin0; cbn;
            apply rebuilt_u_handler_places;
            [ exact (Hplain k0 h0 Hin0)
            | exact (handler_decisions_plain ub Hleaves fx sol _ h0
                       (Hplain k0 h0 Hin0)) ]).
      rewrite (handler_decisions_erase _ _ _ _ h [] (Hplain k h Hin)),
        (handler_decisions_eq sol sol' ub Hu).
      f_equal. apply (IHh k h Hin); [ exact (Hhs (k, h) Hin)
                                     | exact (Hhc (k, h) Hin)
                                     | | exact Hds | exact Hen1 ].
      intros r Hr. apply Hbk. cbn [region_binders]. apply in_or_app. left.
      apply in_flat_map. exists (k, h). split; [ exact Hin | exact Hr ].
    + rewrite (erase_plain _ body).
      2: { apply conts_plain_app; [ | apply conts_plain_nil ].
           intros k p Hin. apply in_map_iff in Hin as [[k' h] [Heq Hin]].
           injection Heq as <- <-. apply region_places_plain.
           exact (Hplain k' h Hin). }
      rewrite (erase_plain _ (RU en1 body)).
      2: { apply conts_plain_app; [ | apply conts_plain_nil ].
           intros k p Hin. apply in_map_iff in Hin as [[k' h'] [Heq Hin]].
           injection Heq as <- <-.
           apply in_map_iff in Hin as [[k0 h0] [Heq Hin]].
           injection Heq as <- <-. apply rebuilt_u_handler_places;
             [ exact (Hplain k0 h0 Hin)
             | exact (handler_decisions_plain ub Hleaves fx sol _ h0
                        (Hplain k0 h0 Hin)) ]. }
      apply IH; [ exact Hb | exact Hcb
                | intros r Hr; apply Hbk; cbn [region_binders];
                  apply in_or_app; right; exact Hr
                | exact Hen1 | apply uenv_plain_conts ].
  - (* Apply *)
    intros ap en en' Hpl Hcf. discriminate Hcf.
  - (* Apply_cont *)
    intros ac en en' Hpl Hcf Hbk Hen Hpe. cbn [regions_plain] in Hpl.
    cbn [rebuild_expr_u erase]. rewrite erase_rebuild_apply_cont_u,
      erase_apply_cont_nil.
    symmetry. exact (rebuild_apply_cont_u_eq sol sol' ub Hu Hsrc _ _ ac Hen
                       Hpl).
  - (* Switch *)
    intros scrutinee arms en en' Hpl Hcf Hbk Hen Hpe.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hs Harms].
    cbn [rebuild_expr_u erase]. rewrite erase_rebuild_switch_u, map_arms_id.
    symmetry.
    exact (rebuild_switch_u_eq sol sol' ub Hu Hsrc _ _ scrutinee arms Hen Hs
             Harms).
  - (* Invalid *)
    reflexivity.
  - (* Handler *)
    intros params md hbody IH en en' dc ds Hpl Hcf Hbk Hds Hen.
    cbn [handler_regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hk Hb].
    rewrite (erase_handler_no_region [] params md hbody Hk).
    rewrite !rh_unfold.
    rewrite erase_handler_no_region.
    2: { apply with_params_kinds_plain.
         apply params_of_decisions_plain; [ | exact Hds ].
         apply with_kinds_plain, Hk. }
    f_equal.
    rewrite (existsb_ext_in (fun '(x, kind) => is_dead_var sol' x kind)
               (fun '(x, kind) => is_dead_var sol x kind))
      by (intros [x k] _; apply (dead_var_eq sol sol' Hsrc)).
    destruct (dc && _); [ reflexivity | ].
    apply IH; [ exact Hb | exact Hcf | exact Hbk
              | apply uenv_rel_bind, Hen | apply uenv_plain_bind ].
Qed.

End Commute.
