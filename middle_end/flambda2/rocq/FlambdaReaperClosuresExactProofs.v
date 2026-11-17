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

(* The closures of a unit that binds each variable once are made where the
   graph of the unit has only the facts of their sets
   (FlambdaReaperRebuildProofs.closures_exact), which the relation of the
   two runs at their exit asks (FlambdaReaperTypesSpec.
   rebuild_exit_relation_holds_statement): [closures_exact_of_bound_once].

   Only the facts of the sets of closures, which
   [record_set_of_closures_deps] adds, are constructor facts at a function
   slot or a value slot ([slot_fact]); every other step of the traversal
   adds other facts ([ns_ok]). A call of [record_set_of_closures_deps]
   adds the facts of a group ([sgroup]): at each of its allocations
   ([sg_bases]) the same function slots and value slots. The invariant of
   the traversal ([inv]) is that the slot facts are those of groups whose
   allocations are disjoint, each made by a [Let] of a closure or a
   binding of a static closure of the unit ([sg_ok]): the allocations of a
   [Let] of [x] are [x], the variables that project the other closures of
   its set from [x], and nodes made by the traversal; that of a static
   closure is its symbol. Where each variable and each symbol is bound
   once, the group that a [Let] or a binding makes has allocations that no
   earlier group has.

   This file needs no review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs FlambdaReaperRebuildProofs.
Import ListNotations.

Module RP := FlambdaReaperRebuildProofs.

(* ======================================================================== *)
(* The facts of the sets of closures                                       *)
(* ======================================================================== *)

Definition is_slot (f : field) : bool :=
  match f with
  | Field_function_slot _ | Field_value_slot _ => true
  | Field_block _ _ | Field_is_int | Field_get_tag | Field_boxed_number _
  | Field_call_witness _ | Field_code_id_of_call_witness
  | Field_return_of_call _ =>
    false
  end.

(* A constructor fact at a function slot or a value slot. *)
Definition slot_fact (fct : fact) : Prop :=
  match fct with
  | Fact_constructor _ f _ => is_slot f = true
  | Fact_alias _ _ | Fact_use _ _ | Fact_accessor _ _ _
  | Fact_argument _ _ _ | Fact_parameter _ _ _ | Fact_propagate _ _ _
  | Fact_alias_if_any_source _ _ _ | Fact_any_usage _ | Fact_any_source _
  | Fact_any_source_if_used _ _ =>
    False
  end.

(* From [a] to [a'], facts are added that are not slot facts, and nodes
   are made. *)
Definition ns_ok (a a' : acc) : Prop :=
  (∀ fct : fact, fct ∈ acc_facts a -> fct ∈ acc_facts a')
  ∧ (∀ fct : fact,
       fct ∈ acc_facts a' -> slot_fact fct -> fct ∈ acc_facts a)
  ∧ (acc_next a <= acc_next a')%positive.

Lemma ns_refl (a : acc) : ns_ok a a.
Proof. split; [ auto | split; [ auto | lia ] ]. Qed.

Lemma ns_trans (a a1 a2 : acc) : ns_ok a a1 -> ns_ok a1 a2 -> ns_ok a a2.
Proof.
  intros (H1 & H2 & H3) (H1' & H2' & H3').
  split; [ auto | split; [ auto | lia ] ].
Qed.

Lemma ns_same (a a' : acc) :
  acc_facts a' = acc_facts a -> (acc_next a <= acc_next a')%positive ->
  ns_ok a a'.
Proof.
  intros E Hn. split; [ rewrite E; auto | split; [ rewrite E; auto | lia ] ].
Qed.

Lemma ns_add_facts (L : list fact) (a : acc) :
  (∀ fct : fact, fct ∈ L -> ~ slot_fact fct) -> ns_ok a (add_facts L a).
Proof.
  intros HL. unfold ns_ok. cbn [add_facts acc_facts acc_next].
  split; [ intros fct Hin; apply in_or_app; right; exact Hin | ].
  split; [ | lia ].
  intros fct Hin Hs. apply in_app_or in Hin as [Hin | Hin];
    [ destruct (HL fct Hin Hs) | exact Hin ].
Qed.

Lemma ns_add_fact (fct : fact) (a : acc) :
  ~ slot_fact fct -> ns_ok a (add_fact fct a).
Proof.
  intros H. apply ns_add_facts. intros fct' [<- | []]. exact H.
Qed.

(* Shows that the facts of a list are not slot facts. *)
Ltac slot_free :=
  let fct := fresh "fct" in let Hin := fresh "Hin" in
  let Hs := fresh "Hs" in
  intros fct Hin Hs;
  repeat match goal with
  | H : In _ (_ ++ _) |- _ => apply in_app_or in H; destruct H as [H | H]
  | H : In _ (map _ _) |- _ =>
    let y := fresh "y" in let Heq := fresh "Heq" in
    apply in_map_iff in H; destruct H as [y [Heq _]]; subst fct;
    repeat match goal with
           | p : (_ * _)%type |- _ => destruct p
           end
  | H : In _ (_ :: _) |- _ => destruct H as [H | H]; [ subst fct | ]
  | H : In _ [] |- _ => destruct H
  end;
  cbn [slot_fact is_slot] in *; try discriminate; try contradiction.

Lemma ns_fresh (a : acc) : ns_ok a (snd (fresh a)).
Proof. apply ns_same; cbn; [ reflexivity | lia ]. Qed.

Lemma ns_fresh_list (n : nat) : ∀ a : acc, ns_ok a (snd (fresh_list n a)).
Proof.
  induction n as [ | n IH]; intros a; [ apply ns_refl | ].
  cbn [fresh_list]. destruct (fresh a) as [x a1] eqn:E1.
  destruct (fresh_list n a1) as [xs a2] eqn:E2. cbn [snd].
  apply (ns_trans _ a1).
  - replace a1 with (snd (fresh a)) by (rewrite E1; reflexivity).
    apply ns_fresh.
  - replace a2 with (snd (fresh_list n a1)) by (rewrite E2; reflexivity).
    apply IH.
Qed.

Lemma ns_note_simple (s : simple) (a : acc) : ns_ok a (note_simple s a).
Proof. destruct s; apply ns_same; cbn; (reflexivity || lia). Qed.

Lemma ns_fold {A : Type} (f : acc -> A -> acc) (l : list A) :
  (∀ (a : acc) (x : A), x ∈ l -> ns_ok a (f a x)) ->
  ∀ a : acc, ns_ok a (List.fold_left f l a).
Proof.
  induction l as [ | x l IH]; intros Hf a; [ apply ns_refl | ].
  cbn [List.fold_left].
  apply (ns_trans _ (f a x)); [ apply Hf; left; reflexivity | ].
  apply IH. intros a' y Hy. apply Hf. right. exact Hy.
Qed.

Lemma ns_note_simples (l : list simple) (a : acc) :
  ns_ok a (note_simples l a).
Proof. apply ns_fold. intros a' s _. apply ns_note_simple. Qed.

Lemma ns_push_apply_dep (d : apply_dep) (a : acc) :
  ns_ok a (push_apply_dep d a).
Proof. apply ns_same; cbn; (reflexivity || lia). Qed.

Lemma ns_push_closure_dep (d : closure_dep) (a : acc) :
  ns_ok a (push_closure_dep d a).
Proof. apply ns_same; cbn; (reflexivity || lia). Qed.

Lemma ns_add_code_dep (id : code_id) (cd : code_dep) (a : acc) :
  ns_ok a (add_code_dep id cd a).
Proof. apply ns_same; cbn; (reflexivity || lia). Qed.

Lemma ns_cond_any_usage (env : traverse_env) (s : simple) (a : acc) :
  ns_ok a (cond_any_usage env s a).
Proof.
  unfold cond_any_usage. apply (ns_trans _ (note_simple s a));
    [ apply ns_note_simple | ].
  destruct (env_current_code_id env); apply ns_add_fact; cbn; auto.
Qed.

Lemma ns_cond_alias (env : traverse_env) (from to_ : node) (a : acc) :
  ns_ok a (cond_alias env from to_ a).
Proof.
  unfold cond_alias.
  destruct (env_current_code_id env); apply ns_add_fact; cbn; auto.
Qed.

(* Steps whose result is a node and an accumulator. *)
Ltac ns_let H lem :=
  match type of H with
  | ?f = (?x, ?a1) =>
    let Hn := fresh "Hns" in
    assert (Hn : ns_ok _ a1)
      by (change a1 with (snd (x, a1)); rewrite <- H; apply lem)
  end.

(* ======================================================================== *)
(* The steps that make no slot fact                                         *)
(* ======================================================================== *)

Lemma ns_find_cont (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  ns_ok a (snd (find_cont env k n a)).
Proof.
  unfold find_cont. destruct (lookup k (env_conts env)); [ apply ns_refl | ].
  destruct (fresh_list n a) as [params a1] eqn:E. ns_let E ns_fresh_list.
  apply (ns_trans _ _ _ Hns). apply ns_add_facts. slot_free.
Qed.

Lemma ns_block_deps (base : node) (fields : list simple) (a : acc) :
  ns_ok a (block_deps base fields a).
Proof.
  unfold block_deps. apply (ns_trans _ _ _ (ns_note_simples fields a)).
  apply ns_add_facts. slot_free.
Qed.

Lemma ns_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  ns_ok a (traverse_prim env x p args a).
Proof.
  assert (Hacc : ∀ (f : field) (s : simple) (a0 : acc),
             ns_ok a0 (add_fact (Fact_accessor (Node_variable x) f
                                   (node_of_simple s)) (note_simple s a0))).
  { intros f s a0. apply (ns_trans _ _ _ (ns_note_simple s a0)).
    apply ns_add_fact. cbn. auto. }
  assert (Hdefault :
            ns_ok a
              (add_facts
                 (Fact_any_source (Node_variable x)
                  :: map (fun from => Fact_use (Node_variable x) from)
                       (free_names_of_simples args))
                 (if prim_has_arbitrary_effects p
                  then cond_any_usage env (Var x) a else a))).
  { apply (ns_trans _ (if prim_has_arbitrary_effects p
                        then cond_any_usage env (Var x) a else a)).
    - destruct (prim_has_arbitrary_effects p);
        [ apply ns_cond_any_usage | apply ns_refl ].
    - apply ns_add_facts. slot_free. }
  unfold traverse_prim; cbv zeta.
  destruct p as [u0 | b0 | v0 | h0 | r0];
    [ destruct u0 | | destruct v0 | destruct h0 | destruct r0 ];
    try (destruct args as [ | arg [ | arg2 args]]);
    cbv beta iota;
    try first [ exact Hdefault | apply ns_block_deps | apply ns_refl
              | exact (Hacc _ _ _) ].
  all: try (destruct variant_only;
            first [ exact Hdefault | exact (Hacc _ _ _) ]).
  - destruct mut; [ | exact (Hacc _ _ _) | exact (Hacc _ _ _) ].
    eapply ns_trans; [ | apply ns_add_fact; cbn; auto ]. apply Hacc.
  - apply (ns_trans _ _ _ (ns_note_simple arg a)).
    apply ns_add_fact. cbn. auto.
Qed.

Lemma ns_create_known (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  ns_ok a (snd (create_known_arity_call_witness id params returns exn a)).
Proof.
  unfold create_known_arity_call_witness.
  destruct (fresh a) as [w a1] eqn:E. ns_let E ns_fresh.
  apply (ns_trans _ _ _ Hns). apply ns_add_facts. slot_free.
Qed.

Lemma ns_witness_deps (id : code_id) (returns : list node) (exn : node) :
  ∀ (l : list (list node * node)) (a : acc),
    ns_ok a (unknown_arity_witness_deps id returns exn l a).
Proof.
  induction l as [ | [first w] rest IH]; intros a; [ apply ns_refl | ].
  cbn [unknown_arity_witness_deps].
  match goal with
  | |- context [add_facts ?L a] =>
    apply (ns_trans _ (add_facts L a)); [ apply ns_add_facts; slot_free | ]
  end.
  destruct rest as [ | [? next] rest'].
  - apply ns_add_facts. slot_free.
  - match goal with
    | |- context [fresh ?b] => destruct (fresh b) as [v a1] eqn:E
    end.
    ns_let E ns_fresh. apply (ns_trans _ _ _ Hns).
    match goal with
    | |- context [add_facts ?L a1] =>
      apply (ns_trans _ (add_facts L a1)); [ apply ns_add_facts; slot_free | ]
    end.
    apply IH.
Qed.

Lemma ns_create_unknown (id : code_id) (is_tupled : bool) (ar : arity)
    (params returns : list node) (exn : node) (a : acc) :
  ns_ok a
    (snd (create_unknown_arity_call_witnesses id is_tupled ar params returns
            exn a)).
Proof.
  unfold create_unknown_arity_call_witnesses. destruct is_tupled.
  - unfold create_unknown_arity_tupled_call_witnesses.
    destruct (fresh a) as [w a1] eqn:E1. ns_let E1 ns_fresh.
    destruct (fresh a1) as [u a2] eqn:E2.
    assert (Hns2 : ns_ok a1 a2)
      by (change a2 with (snd (u, a2)); rewrite <- E2; apply ns_fresh).
    destruct (fresh a2) as [k a3] eqn:E3.
    assert (Hns3 : ns_ok a2 a3)
      by (change a3 with (snd (k, a3)); rewrite <- E3; apply ns_fresh).
    cbn [snd]. apply (ns_trans _ _ _ Hns), (ns_trans _ _ _ Hns2),
      (ns_trans _ _ _ Hns3).
    apply ns_add_facts. slot_free.
  - unfold create_unknown_arity_non_tupled_call_witnesses.
    match goal with
    | |- context [fresh_list ?n a] => destruct (fresh_list n a) as [ws a1] eqn:E
    end.
    ns_let E ns_fresh_list. cbn [snd]. apply (ns_trans _ _ _ Hns).
    apply ns_witness_deps.
Qed.

Lemma ns_prepare_code (id : code_id) (c : code) (a : acc) :
  ns_ok a (prepare_code id c a).
Proof.
  unfold prepare_code.
  destruct (fresh_list _ a) as [returns a1] eqn:E1. ns_let E1 ns_fresh_list.
  destruct (fresh a1) as [exn a2] eqn:E2.
  assert (Hns2 : ns_ok a1 a2)
    by (change a2 with (snd (exn, a2)); rewrite <- E2; apply ns_fresh).
  destruct (fresh a2) as [mc a3] eqn:E3.
  assert (Hns3 : ns_ok a2 a3)
    by (change a3 with (snd (mc, a3)); rewrite <- E3; apply ns_fresh).
  destruct (fresh_list _ a3) as [params a4] eqn:E4.
  assert (Hns4 : ns_ok a3 a4)
    by (change a4 with (snd (params, a4)); rewrite <- E4;
        apply ns_fresh_list).
  destruct (create_known_arity_call_witness id params returns exn a4)
    as [known a5] eqn:E5.
  assert (Hns5 : ns_ok a4 a5)
    by (change a5 with (snd (known, a5)); rewrite <- E5;
        apply ns_create_known).
  destruct (create_unknown_arity_call_witnesses _ _ _ _ _ _ a5)
    as [unknown a6] eqn:E6.
  assert (Hns6 : ns_ok a5 a6)
    by (change a6 with (snd (unknown, a6)); rewrite <- E6;
        apply ns_create_unknown).
  apply (ns_trans _ _ _ Hns), (ns_trans _ _ _ Hns2), (ns_trans _ _ _ Hns3),
    (ns_trans _ _ _ Hns4), (ns_trans _ _ _ Hns5), (ns_trans _ _ _ Hns6).
  match goal with
  | |- context [add_fact ?f a6] =>
    apply (ns_trans _ (add_fact f a6)); [ apply ns_add_fact; cbn; auto | ]
  end.
  apply ns_add_code_dep.
Qed.

Lemma ns_prepare_binding (b : static_binding) (a : acc) :
  ns_ok a (prepare_binding b a).
Proof.
  destruct b; [ apply ns_prepare_code | apply ns_refl | apply ns_refl ].
Qed.

Lemma ns_make_known (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  ns_ok a (snd (make_known_arity_apply_widget env ap returns exn a)).
Proof.
  unfold make_known_arity_apply_widget.
  destruct (fresh a) as [w a1] eqn:E1. ns_let E1 ns_fresh.
  destruct (fresh a1) as [called a2] eqn:E2.
  assert (Hns2 : ns_ok a1 a2)
    by (change a2 with (snd (called, a2)); rewrite <- E2; apply ns_fresh).
  destruct (fresh a2) as [an a3] eqn:E3.
  assert (Hns3 : ns_ok a2 a3)
    by (change a3 with (snd (an, a3)); rewrite <- E3; apply ns_fresh).
  cbn [snd].
  apply (ns_trans _ _ _ Hns), (ns_trans _ _ _ Hns2), (ns_trans _ _ _ Hns3),
    (ns_trans _ _ _ (ns_note_simples (ap_args ap) a3)).
  match goal with
  | |- context [add_facts ?L ?b] =>
    apply (ns_trans _ (add_facts L b)); [ apply ns_add_facts; slot_free | ]
  end.
  apply ns_cond_alias.
Qed.

Lemma ns_apply_deps (returns : list node) (exn called : node) :
  ∀ (l : list (list simple * node)) (a : acc),
    ns_ok a (unknown_arity_apply_deps returns exn called l a).
Proof.
  induction l as [ | [first w] rest IH]; intros a; [ apply ns_refl | ].
  cbn [unknown_arity_apply_deps].
  match goal with
  | |- context [add_facts ?L a] =>
    apply (ns_trans _ (add_facts L a)); [ apply ns_add_facts; slot_free | ]
  end.
  destruct rest as [ | [? next] rest'].
  - apply ns_add_facts. slot_free.
  - match goal with
    | |- context [fresh ?b] => destruct (fresh b) as [v a1] eqn:E
    end.
    ns_let E ns_fresh. apply (ns_trans _ _ _ Hns).
    match goal with
    | |- context [add_facts ?L a1] =>
      apply (ns_trans _ (add_facts L a1)); [ apply ns_add_facts; slot_free | ]
    end.
    apply IH.
Qed.

Lemma ns_make_unknown (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  ns_ok a (snd (make_unknown_arity_apply_widget env ap returns exn a)).
Proof.
  unfold make_unknown_arity_apply_widget.
  destruct (fresh a) as [called a1] eqn:E1. ns_let E1 ns_fresh.
  apply (ns_trans _ _ _ Hns).
  apply (ns_trans _ (add_fact (Fact_any_usage called) a1));
    [ apply ns_add_fact; cbn; auto | ].
  match goal with
  | |- context [fresh_list ?n ?b] => destruct (fresh_list n b) as [ws a2] eqn:E2
  end.
  ns_let E2 ns_fresh_list. apply (ns_trans _ _ _ Hns0).
  apply (ns_trans _ _ _ (ns_note_simples (ap_args ap) a2)).
  match goal with
  | |- context [unknown_arity_apply_deps ?r ?x ?c ?l ?b] =>
    apply (ns_trans _ (unknown_arity_apply_deps r x c l b));
      [ apply ns_apply_deps | ];
    destruct (fresh (unknown_arity_apply_deps r x c l b)) as [an a3] eqn:E3
  end.
  ns_let E3 ns_fresh. apply (ns_trans _ _ _ Hns1).
  destruct ws; [ apply ns_refl | apply ns_cond_alias ].
Qed.

Lemma ns_add_apply (env : traverse_env) (id : code_id) (callee widget : node)
    (only : bool) (a : acc) :
  ns_ok a (add_apply env id callee widget only a).
Proof.
  unfold add_apply. destruct only; [ | apply ns_push_apply_dep ].
  destruct (fresh a) as [c1 a1] eqn:E1. ns_let E1 ns_fresh.
  destruct (fresh a1) as [w1 a2] eqn:E2.
  assert (Hns2 : ns_ok a1 a2)
    by (change a2 with (snd (w1, a2)); rewrite <- E2; apply ns_fresh).
  apply (ns_trans _ _ _ Hns), (ns_trans _ _ _ Hns2).
  match goal with
  | |- context [add_facts ?L a2] =>
    apply (ns_trans _ (add_facts L a2)); [ apply ns_add_facts; slot_free | ]
  end.
  apply ns_push_apply_dep.
Qed.

Lemma ns_traverse_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  ns_ok a (traverse_apply env ap a).
Proof.
  unfold traverse_apply.
  destruct (find_cont env (ap_continuation ap) _ a) as [ra a1] eqn:E1.
  ns_let E1 ns_find_cont.
  destruct (find_cont env (ap_exn_continuation ap) _ a1) as [ep a2] eqn:E2.
  assert (Hns2 : ns_ok a1 a2)
    by (change a2 with (snd (ep, a2)); rewrite <- E2; apply ns_find_cont).
  apply (ns_trans _ _ _ Hns), (ns_trans _ _ _ Hns2).
  destruct (match ep with
            | [] => fresh a2
            | exn_param :: extra_params =>
              (exn_param,
               add_facts
                 (map (fun '(p, s) => Fact_alias p (node_of_simple s))
                    (combine extra_params (exn_extra_simples ap)))
                 (note_simples (exn_extra_simples ap) a2))
            end)
    as [exn_arg a3] eqn:E3.
  assert (Hns3 : ns_ok a2 a3).
  { destruct ep as [ | p ps].
    - change a3 with (snd (exn_arg, a3)). rewrite <- E3. apply ns_fresh.
    - injection E3 as _ <-.
      apply (ns_trans _ _ _ (ns_note_simples (exn_extra_simples ap) a2)).
      apply ns_add_facts. slot_free. }
  apply (ns_trans _ _ _ Hns3).
  unfold traverse_call_kind.
  apply (ns_trans _ _ _ (ns_note_simple (ap_callee ap) a3)).
  destruct (ap_call_kind ap) as [id | ].
  - destruct (make_known_arity_apply_widget env ap ra exn_arg
                (note_simple (ap_callee ap) a3))
      as [cw a4] eqn:E4.
    assert (Hns4 : ns_ok (note_simple (ap_callee ap) a3) a4)
      by (change a4 with (snd (cw, a4)); rewrite <- E4; apply ns_make_known).
    apply (ns_trans _ _ _ Hns4).
    match goal with
    | |- context [add_fact ?f a4] =>
      apply (ns_trans _ (add_fact f a4)); [ apply ns_add_fact; cbn; auto | ]
    end.
    destruct (env_should_preserve_direct_calls env);
      try apply ns_add_apply.
    destruct (negb _); [ apply ns_add_apply | apply ns_refl ].
  - destruct (make_unknown_arity_apply_widget env ap ra exn_arg
                (note_simple (ap_callee ap) a3))
      as [cw a4] eqn:E4.
    assert (Hns4 : ns_ok (note_simple (ap_callee ap) a3) a4)
      by (change a4 with (snd (cw, a4)); rewrite <- E4;
          apply ns_make_unknown).
    apply (ns_trans _ _ _ Hns4). apply ns_add_fact. cbn. auto.
Qed.

Lemma ns_apply_cont_deps (env : traverse_env) (ac : apply_cont) (a : acc) :
  ns_ok a (apply_cont_deps env ac a).
Proof.
  unfold apply_cont_deps.
  destruct (find_cont env (ac_continuation ac) _ a) as [ps a1] eqn:E1.
  ns_let E1 ns_find_cont. apply (ns_trans _ _ _ Hns).
  apply (ns_trans _ _ _ (ns_note_simples (ac_args ac) a1)).
  apply ns_add_facts. slot_free.
Qed.

Lemma ns_traverse_switch (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  ns_ok a (traverse_switch env scrutinee arms a).
Proof.
  unfold traverse_switch.
  apply (ns_trans _ _ _ (ns_cond_any_usage env scrutinee a)).
  apply ns_fold. intros a' [i ac] _. apply ns_apply_cont_deps.
Qed.

Lemma ns_exn_handler_deps (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  ns_ok a (exn_handler_deps env h a).
Proof.
  unfold exn_handler_deps. destruct h as [[ | p ps] md body];
    [ apply ns_refl | ].
  destruct (hm_is_exn_handler md); [ | apply ns_refl ].
  eapply ns_trans; [ | apply ns_cond_any_usage ].
  apply ns_add_fact. cbn. auto.
Qed.

Lemma ns_traverse_static_binding (env : traverse_env) (b : static_binding)
    (a : acc) :
  (∀ (s : symbol) (slot : function_slot)
     (cs : list (function_slot * symbol))
     (decls : list (function_slot * code_id))
     (vs : list (value_slot * simple)),
     b <> Bind_symbol s (Static_const_closure_of_set slot cs decls vs)) ->
  (∀ (s : symbol) (id : code_id) (vs : list (value_slot * simple)),
     b <> Bind_symbol s (Static_const_set_of_closures id vs)) ->
  ns_ok a (traverse_static_binding env b a).
Proof.
  intros Hcl Hset. destruct b as [id c | id | s const];
    cbn [traverse_static_binding]; try apply ns_refl.
  destruct const as [id vs | slot cs decls vs | tag fields | contents
                    | bytes | [ | el els]].
  - destruct (Hset s id vs eq_refl).
  - destruct (Hcl s slot cs decls vs eq_refl).
  - apply ns_block_deps.
  - apply ns_add_facts. unfold boxed_number_facts.
    destruct (boxed_number_kind contents); [ slot_free | ].
    destruct contents; slot_free.
  - apply ns_add_fact. cbn. auto.
  - apply ns_add_fact. cbn. auto.
  - apply ns_block_deps.
Qed.

Lemma ns_code_aliases (my_closure : variable) (params : list variable)
    (cd : code_dep) (a : acc) :
  ns_ok a
    (add_facts
       (Fact_alias (Node_variable my_closure) (cd_my_closure cd)
        :: map (fun '(p, arg) => Fact_alias (Node_variable p) arg)
             (combine params (cd_params cd)))
       a).
Proof. apply ns_add_facts. slot_free. Qed.

Lemma ns_add_closure_dep (code_deps : finite_map code_dep) (d : closure_dep)
    (a : acc) :
  ns_ok a (add_closure_dep code_deps d a).
Proof.
  unfold add_closure_dep.
  destruct (lookup (cld_closure_code_id d) code_deps) as [cd | ].
  - apply ns_add_facts.
    destruct (cd_unknown_arity_call_witnesses cd); slot_free.
  - destruct (fresh a) as [w a1] eqn:E. ns_let E ns_fresh.
    apply (ns_trans _ _ _ Hns). apply ns_add_facts. slot_free.
Qed.

Lemma ns_resolve (unit_symbols : list symbol) (a : acc) :
  ns_ok a (resolve_delayed_deps unit_symbols a).
Proof.
  unfold resolve_delayed_deps. cbv zeta.
  match goal with
  | |- context [add_facts (flat_map ?f ?l) a] =>
    apply (ns_trans _ (add_facts (flat_map f l) a))
  end.
  { apply ns_add_facts. intros fct Hin Hs.
    apply in_flat_map in Hin as (d & _ & Hin). unfold apply_dep_facts in Hin.
    destruct (lookup _ _); [ | destruct (ad_function_containing_apply_expr d) ];
      revert fct Hin Hs; try (destruct (ad_function_containing_apply_expr d));
      slot_free. }
  cbv zeta.
  match goal with
  | |- context [List.fold_left ?f ?l ?b] =>
    apply (ns_trans _ (List.fold_left f l b));
      [ apply ns_fold; intros; apply ns_add_closure_dep | ]
  end.
  apply ns_add_facts. slot_free.
Qed.

(* ======================================================================== *)
(* The groups                                                               *)
(* ======================================================================== *)

(* The symbols that the static bindings of an expression bind, with what
   they bind them to, the bodies of code included. *)
Fixpoint sym_bindings (e : expr) : list (symbol * static_const) :=
  match e with
  | Let _ _ _ body => sym_bindings body
  | Let_static bindings body =>
    flat_map binding_sym_bindings bindings ++ sym_bindings body
  | Let_cont_non_recursive _ h body =>
    handler_sym_bindings h ++ sym_bindings body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_sym_bindings h) handlers
    ++ sym_bindings body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_sym_bindings (h : continuation_handler)
    : list (symbol * static_const) :=
  match h with
  | Handler _ _ handler => sym_bindings handler
  end
with binding_sym_bindings (binding : static_binding)
    : list (symbol * static_const) :=
  match binding with
  | Bind_symbol s const => [(s, const)]
  | Bind_code _ c => code_sym_bindings c
  | Bind_deleted_code _ => []
  end
with code_sym_bindings (c : code) : list (symbol * static_const) :=
  match c with
  | Code _ _ _ _ _ body => sym_bindings body
  end.

(* The facts that a call of [record_set_of_closures_deps] adds: at each of
   its allocations [sg_bases], the function slots [sg_fslots] and the
   value slots [sg_vslots]; [sg_origin] is the variable of its [Let] or
   the symbol of its binding. *)
Record sgroup : Type := SGroup {
  sg_origin : node;
  sg_bases : list node;
  sg_fslots : list (function_slot * node);
  sg_vslots : list (value_slot * node)
}.

Definition sg_fact (γ : sgroup) (fct : fact) : Prop :=
  ∃ N : node,
    N ∈ sg_bases γ
    ∧ ((∃ (f : function_slot) (b : node),
          fct = Fact_constructor N (Field_function_slot f) b
          ∧ (f, b) ∈ sg_fslots γ)
       ∨ ∃ (w : value_slot) (b : node),
           fct = Fact_constructor N (Field_value_slot w) b
           ∧ (w, b) ∈ sg_vslots γ).

(* The closures of [e] are made where [origin_in] says. *)
Definition origin_in (e : expr) (N : node) : Prop :=
  (∃ (x : variable) (slot : function_slot)
     (decls : list (function_slot * code_id))
     (vs : list (value_slot * simple)),
     N = Node_variable x ∧ (x, Closure_of_set slot decls vs) ∈ let_bindings e)
  ∨ ∃ (t : symbol) (slot : function_slot)
      (cs : list (function_slot * symbol))
      (decls : list (function_slot * code_id))
      (vs : list (value_slot * simple)),
      N = Node_symbol t
      ∧ (t, Static_const_closure_of_set slot cs decls vs) ∈ sym_bindings e.

Section Groups.

Variable e0 : expr.
Variable keep : value_slot -> bool.

(* A group made by the traversal of [e0]: its slots are functional and its
   value slots kept. One made by a [Let] of [x] in [e0] has the function
   slots of the set, at its allocations, which are [x], variables bound
   in [e0] to projections from [x] and made nodes. One made by a binding
   of a static closure has the symbol for its allocation and the
   function slots and the value slots of the binding. *)
Definition sg_ok (γ : sgroup) : Prop :=
  NoDup (map fst (sg_fslots γ)) ∧ NoDup (map fst (sg_vslots γ))
  ∧ (∀ (w : value_slot) (b : node), (w, b) ∈ sg_vslots γ -> keep w = true)
  ∧ ((∃ (x : variable) (slot : function_slot)
        (decls : list (function_slot * code_id))
        (vs : list (value_slot * simple)),
        sg_origin γ = Node_variable x
        ∧ (x, Closure_of_set slot decls vs) ∈ let_bindings e0
        ∧ Node_variable x ∈ sg_bases γ
        ∧ (∀ g : function_slot, is_bound g (sg_fslots γ) = is_bound g decls)
        ∧ (∀ (f : function_slot) (M : node),
             (f, M) ∈ sg_fslots γ -> M ∈ sg_bases γ)
        ∧ ∀ N : node,
            N ∈ sg_bases γ ->
            N = Node_variable x
            ∨ (∃ (y : variable) (s o : function_slot),
                 N = Node_variable y
                 ∧ (y, Project_function_slot s o (Var x)) ∈ let_bindings e0)
            ∨ ∃ k : positive, N = Node_made k)
     ∨ ∃ (t : symbol) (slot : function_slot)
         (cs : list (function_slot * symbol))
         (decls : list (function_slot * code_id))
         (vs : list (value_slot * simple)),
         sg_origin γ = Node_symbol t
         ∧ (t, Static_const_closure_of_set slot cs decls vs) ∈ sym_bindings e0
         ∧ sg_bases γ = [Node_symbol t]
         ∧ sg_fslots γ
           = map (fun '(f, sym) => (f, Node_symbol sym)) (compact cs)
         ∧ sg_vslots γ
           = map (fun '(w, s) => (w, node_of_simple s))
               (filter (fun '(w, _) => keep w) (compact vs))).

(* The slot facts of [a] are those of the groups [Gs], whose allocations
   are disjoint and whose made nodes are made. *)
Definition inv (Gs : list sgroup) (a : acc) : Prop :=
  (∀ fct : fact,
     fct ∈ acc_facts a -> slot_fact fct ->
     ∃ γ : sgroup, γ ∈ Gs ∧ sg_fact γ fct)
  ∧ (∀ (γ : sgroup) (fct : fact),
       γ ∈ Gs -> sg_fact γ fct -> fct ∈ acc_facts a)
  ∧ (∀ (γ γ' : sgroup) (N : node),
       γ ∈ Gs -> γ' ∈ Gs -> N ∈ sg_bases γ -> N ∈ sg_bases γ' -> γ = γ')
  ∧ (∀ γ : sgroup, γ ∈ Gs -> sg_ok γ)
  ∧ ∀ (γ : sgroup) (k : positive),
      γ ∈ Gs -> Node_made k ∈ sg_bases γ -> (k < acc_next a)%positive.

Lemma inv_ns (Gs : list sgroup) (a a' : acc) :
  inv Gs a -> ns_ok a a' -> inv Gs a'.
Proof.
  intros (HA & HB & HC & HD & HM) (H1 & H2 & H3).
  split; [ intros fct Hin Hs; exact (HA fct (H2 fct Hin Hs) Hs) | ].
  split; [ intros γ fct Hγ Hf; exact (H1 fct (HB γ fct Hγ Hf)) | ].
  split; [ exact HC | split; [ exact HD | ] ].
  intros γ k Hγ Hk. pose proof (HM γ k Hγ Hk). lia.
Qed.


Lemma inv_add (Gs : list sgroup) (γ : sgroup) (a a' : acc) :
  inv Gs a -> sg_ok γ ->
  (∀ (γ' : sgroup) (N : node),
     γ' ∈ Gs -> N ∈ sg_bases γ -> N ∈ sg_bases γ' -> False) ->
  (∃ L : list fact,
     acc_facts a' = L ++ acc_facts a
     ∧ ∀ fct : fact, fct ∈ L <-> sg_fact γ fct) ->
  (acc_next a <= acc_next a')%positive ->
  (∀ k : positive, Node_made k ∈ sg_bases γ -> (k < acc_next a')%positive) ->
  inv (γ :: Gs) a'.
Proof.
  intros (HA & HB & HC & HD & HM) Hok Hfresh (L & HL & HLf) Hn Hk.
  split; [ | split; [ | split; [ | split ] ] ].
  - intros fct Hin Hs. rewrite HL in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + exists γ. split; [ left; reflexivity | exact (proj1 (HLf fct) Hin) ].
    + destruct (HA fct Hin Hs) as (γ' & Hγ' & Hf).
      exists γ'. split; [ right; exact Hγ' | exact Hf ].
  - intros γ' fct [<- | Hγ'] Hf; rewrite HL; apply in_or_app;
      [ left; exact (proj2 (HLf fct) Hf) | right; exact (HB γ' fct Hγ' Hf) ].
  - intros γ1 γ2 N [<- | H1] [<- | H2] HN1 HN2; try reflexivity.
    + destruct (Hfresh γ2 N H2 HN1 HN2).
    + destruct (Hfresh γ1 N H1 HN2 HN1).
    + exact (HC γ1 γ2 N H1 H2 HN1 HN2).
  - intros γ' [<- | Hγ']; [ exact Hok | exact (HD γ' Hγ') ].
  - intros γ' k [<- | Hγ'] Hkb; [ exact (Hk k Hkb) | ].
    pose proof (HM γ' k Hγ' Hkb). lia.
Qed.

Lemma nodup_same {A : Type} (l : list (positive * A)) (x : positive) (a b : A) :
  NoDup (map fst l) -> (x, a) ∈ l -> (x, b) ∈ l -> a = b.
Proof.
  induction l as [ | [y c] l IH]; intros Hnd Ha Hb; [ destruct Ha | ].
  inversion Hnd as [ | y' l' Hy Hnd' ]; subst.
  destruct Ha as [Ha | Ha]; destruct Hb as [Hb | Hb].
  - congruence.
  - injection Ha as -> ->. exfalso. apply Hy.
    exact (in_map fst l (x, b) Hb).
  - injection Hb as -> ->. exfalso. apply Hy.
    exact (in_map fst l (x, a) Ha).
  - exact (IH Hnd' Ha Hb).
Qed.

(* A group whose origin no group of [Gs] has, and whose made nodes are
   new, has allocations that no group of [Gs] has. *)
Lemma group_fresh (Gs : list sgroup) (γ : sgroup) (a : acc) :
  NoDup (map fst (let_bindings e0)) -> NoDup (map fst (sym_bindings e0)) ->
  inv Gs a -> sg_ok γ ->
  (∀ γ' : sgroup, γ' ∈ Gs -> sg_origin γ' <> sg_origin γ) ->
  (∀ k : positive, Node_made k ∈ sg_bases γ -> (acc_next a <= k)%positive) ->
  ∀ (γ' : sgroup) (N : node),
    γ' ∈ Gs -> N ∈ sg_bases γ -> N ∈ sg_bases γ' -> False.
Proof.
  intros Hnl Hns (_ & _ & _ & HD & HM) Hok Horig Hmade γ' N Hγ' HN HN'.
  destruct N as [v | sym | c | | k].
  - destruct Hok
      as (_ & _ & _ & [(x & slot & decls & vs & Ho & Hx & _ & _ & _ & Hb)
                      | (t & _ & _ & _ & _ & _ & _ & Hb & _)]);
      [ | rewrite Hb in HN; destruct HN as [HN | []]; discriminate HN ].
    destruct (HD γ' Hγ') as (_ & _ & _ & [(x' & slot' & decls' & vs' & Ho'
                                            & Hx' & _ & _ & _ & Hb')
                                         | (t' & _ & _ & _ & _ & _ & _ & Hb'
                                            & _)]);
      [ | rewrite Hb' in HN'; destruct HN' as [HN' | []];
          discriminate HN' ].
    pose proof (Horig γ' Hγ') as Hne. rewrite Ho, Ho' in Hne.
    destruct (Hb _ HN) as [[= ->] | [(y & s & o & [= <-] & Hy) | (k & Hk)]];
      [ | | discriminate Hk ];
      destruct (Hb' _ HN')
        as [[= Heq] | [(y' & s' & o' & [= <-] & Hy') | (k' & Hk')]];
      try discriminate Hk'.
    + subst x'. apply Hne. reflexivity.
    + pose proof (nodup_same _ _ _ _ Hnl Hx Hy') as E. discriminate E.
    + subst x'. pose proof (nodup_same _ _ _ _ Hnl Hx' Hy) as E.
      discriminate E.
    + pose proof (nodup_same _ _ _ _ Hnl Hy Hy') as E.
      injection E as _ _ <-. apply Hne. reflexivity.
  - destruct Hok as (_ & _ & _ & [(x & _ & _ & _ & Ho & _ & _ & _ & _ & Hb)
                                 | (t & _ & _ & _ & _ & Ho & _ & Hb & _)]).
    + destruct (Hb _ HN) as [E | [(y & _ & _ & E & _) | (k & E)]];
        discriminate E.
    + destruct (HD γ' Hγ')
        as (_ & _ & _ & [(x' & _ & _ & _ & _ & _ & _ & _ & _ & Hb')
                        | (t' & _ & _ & _ & _ & Ho' & _ & Hb' & _)]).
      * destruct (Hb' _ HN') as [E | [(y & _ & _ & E & _) | (k & E)]];
          discriminate E.
      * rewrite Hb in HN. rewrite Hb' in HN'.
        destruct HN as [[= <-] | []]. destruct HN' as [[= <-] | []].
        apply (Horig γ' Hγ'). rewrite Ho, Ho'. reflexivity.
  - destruct Hok as (_ & _ & _ & [(x & _ & _ & _ & _ & _ & _ & _ & _ & Hb)
                                 | (t & _ & _ & _ & _ & _ & _ & Hb & _)]).
    + destruct (Hb _ HN) as [E | [(y & _ & _ & E & _) | (k & E)]];
        discriminate E.
    + rewrite Hb in HN. destruct HN as [E | []]. discriminate E.
  - destruct Hok as (_ & _ & _ & [(x & _ & _ & _ & _ & _ & _ & _ & _ & Hb)
                                 | (t & _ & _ & _ & _ & _ & _ & Hb & _)]).
    + destruct (Hb _ HN) as [E | [(y & _ & _ & E & _) | (k & E)]];
        discriminate E.
    + rewrite Hb in HN. destruct HN as [E | []]. discriminate E.
  - pose proof (Hmade k HN). pose proof (HM γ' k Hγ' HN'). lia.
Qed.

End Groups.

(* ======================================================================== *)
(* The steps that make a group                                              *)
(* ======================================================================== *)

Lemma note_simples_facts (l : list simple) :
  ∀ a : acc, acc_facts (note_simples l a) = acc_facts a.
Proof.
  unfold note_simples. induction l as [ | s0 l IH]; intros a;
    [ reflexivity | ].
  cbn [List.fold_left]. rewrite IH. destruct s0; reflexivity.
Qed.

Lemma note_simples_next (l : list simple) :
  ∀ a : acc, acc_next (note_simples l a) = acc_next a.
Proof.
  unfold note_simples. induction l as [ | s0 l IH]; intros a;
    [ reflexivity | ].
  cbn [List.fold_left]. rewrite IH. destruct s0; reflexivity.
Qed.

(* The facts that [record_set_of_closures_deps] adds. *)
Lemma record_set_facts (env : traverse_env)
    (closures : list (node * option code_id))
    (fs : list (function_slot * node))
    (value_slots : list (value_slot * simple))
    (a : acc) :
  (∃ L : list fact,
     acc_facts (record_set_of_closures_deps env closures fs value_slots a)
     = L ++ acc_facts a
     ∧ ∀ fct : fact,
         fct ∈ L
         <-> ∃ (base : node) (code : option code_id),
               (base, code) ∈ closures
               ∧ ((∃ (w : value_slot) (s : simple),
                     (w, s)
                     ∈ filter (fun '(w, _) => should_keep_value_slot env w)
                         (compact value_slots)
                     ∧ fct
                       = Fact_constructor base (Field_value_slot w)
                           (node_of_simple s))
                  ∨ ∃ (f : function_slot) (c : node),
                      (f, c) ∈ fs
                      ∧ fct = Fact_constructor base (Field_function_slot f) c))
  ∧ acc_next (record_set_of_closures_deps env closures fs value_slots a)
    = acc_next a.
Proof.
  unfold record_set_of_closures_deps.
  set (vsk := filter (fun '(w, _) => should_keep_value_slot env w)
                (compact value_slots)).
  revert a. induction closures as [ | [base code] closures IH]; intros a.
  - split; [ exists []; split; [ reflexivity | ] | reflexivity ].
    intros fct. split; [ intros [] | intros (b & c & [] & _) ].
  - cbn [List.fold_left].
    set (a1 := add_facts _ (note_simples _ _)).
    destruct (IH a1) as [(L & HL & HLf) Hn].
    split.
    + exists (L ++ map (fun '(w, s) =>
                          Fact_constructor base (Field_value_slot w)
                            (node_of_simple s)) vsk
              ++ map (fun '(f, closure) =>
                        Fact_constructor base (Field_function_slot f) closure)
                   fs).
      split.
      * rewrite HL. unfold a1. cbn [add_facts acc_facts].
        rewrite <- !app_assoc.
        rewrite note_simples_facts. destruct code; reflexivity.
      * intros fct. rewrite in_app_iff, HLf, in_app_iff.
        split.
        -- intros [(b & c & Hin & Hx) | [Hin | Hin]].
           ++ exists b, c. split; [ right; exact Hin | exact Hx ].
           ++ apply in_map_iff in Hin as ([w s] & <- & Hin).
              exists base, code. split; [ left; reflexivity | left ].
              exists w, s. auto.
           ++ apply in_map_iff in Hin as ([f c] & <- & Hin).
              exists base, code. split; [ left; reflexivity | right ].
              exists f, c. auto.
        -- intros (b & c & [[= <- <-] | Hin] & Hx).
           ++ right.
              destruct Hx as [(w & s & Hin & ->) | (f & c' & Hin & ->)].
              ** left. apply in_map_iff. exists (w, s). auto.
              ** right. apply in_map_iff. exists (f, c'). auto.
           ++ left. exists b, c. auto.
    + rewrite Hn. unfold a1. cbn [add_facts acc_next].
      rewrite note_simples_next. destruct code; reflexivity.
Qed.

(* The closures of a [Closure_of_set] bound to [x] at [slot]: [x] for
   that slot, the variable of its projection for a slot projected, a node
   made for each other. *)
Lemma closures_of_set_spec (x : variable) (slot : function_slot)
    (others : list (function_slot * variable)) :
  ∀ (decls : list (function_slot * code_id)) (a : acc)
    (cl : list (function_slot * node * code_id)) (a' : acc),
    closures_of_set x slot others decls a = (cl, a') ->
    acc_facts a' = acc_facts a
    ∧ (acc_next a <= acc_next a')%positive
    ∧ map (fun '(f, _, _) => f) cl = map fst decls
    ∧ ∀ (f : function_slot) (N : node) (id : code_id),
        (f, N, id) ∈ cl ->
        (f = slot -> N = Node_variable x)
        ∧ (N = Node_variable x
           ∨ (∃ y : variable, lookup f others = Some y ∧ N = Node_variable y)
           ∨ ∃ k : positive,
               N = Node_made k ∧ (acc_next a <= k)%positive
               ∧ (k < acc_next a')%positive).
Proof.
  induction decls as [ | [f id] decls IH]; intros a cl a' E;
    cbn [closures_of_set] in E.
  - injection E as <- <-.
    split; [ reflexivity | split; [ lia | split; [ reflexivity | ] ] ].
    intros f N id [].
  - destruct (if Pos.eqb f slot then (Node_variable x, a)
              else match lookup f others with
                   | Some y => (Node_variable y, a)
                   | None => fresh a
                   end)
      as [name a1] eqn:E1.
    assert (H1 : acc_facts a1 = acc_facts a
                 ∧ (acc_next a <= acc_next a1)%positive
                 ∧ (f = slot -> name = Node_variable x)
                 ∧ (name = Node_variable x
                    ∨ (∃ y : variable,
                         lookup f others = Some y ∧ name = Node_variable y)
                    ∨ ∃ k : positive,
                        name = Node_made k ∧ (acc_next a <= k)%positive
                        ∧ (k < acc_next a1)%positive)).
    { destruct (Pos.eqb_spec f slot) as [Hfs | Hfs].
      - injection E1 as <- <-. split; [ reflexivity | split; [ lia | ] ].
        split; [ reflexivity | ]. left. reflexivity.
      - destruct (lookup f others) as [y | ] eqn:Hy.
        + injection E1 as <- <-. split; [ reflexivity | split; [ lia | ] ].
          split; [ intros Hc; destruct (Hfs Hc) | ].
          right. left. exists y. auto.
        + injection E1 as <- <-. cbn.
          split; [ reflexivity | split; [ lia | ] ].
          split; [ intros Hc; destruct (Hfs Hc) | ].
          right. right. exists (acc_next a). split; [ reflexivity | lia ]. }
    destruct H1 as (Hf1 & Hn1 & Hslot & Hname).
    destruct (closures_of_set x slot others decls a1) as [rest a2] eqn:E2.
    injection E as <- <-.
    destruct (IH a1 rest a2 E2) as (Hf2 & Hn2 & Hmap & Hrest).
    split; [ rewrite Hf2; exact Hf1 | split; [ lia | split ] ].
    + cbn [map]. rewrite Hmap. reflexivity.
    + intros g N id' [[= <- <- <-] | Hin].
      * split; [ exact Hslot | ].
        destruct Hname as [? | [? | (k & -> & Hk1 & Hk2)]]; auto.
        right. right. exists k. split; [ reflexivity | lia ].
      * destruct (Hrest g N id' Hin)
          as [Hs' [? | [? | (k & -> & Hk1 & Hk2)]]];
          split; auto.
        right. right. exists k. split; [ reflexivity | lia ].
Qed.

(* The variables that project the closures of a set from its first are
   bound to such projections. *)
Lemma let_projections_in (x : variable) (slot : function_slot) :
  ∀ (others : list function_slot) (e : expr) (o : function_slot)
    (y : variable),
    (o, y) ∈ let_projections_of x slot others e ->
    ∃ s : function_slot,
      (y, Project_function_slot s o (Var x)) ∈ let_bindings e.
Proof.
  induction others as [ | other others IH]; intros e o y Hin;
    [ destruct e; destruct Hin | ].
  destruct e as [y' k d body | | | | | | | ]; cbn in Hin; try contradiction.
  destruct d as [ | | | | | | mf mt [x' | | ]]; try contradiction.
  destruct (Pos.eqb x' x && Pos.eqb mf slot && Pos.eqb mt other
            && negb (Pos.eqb y' x)) eqn:Hc; [ | destruct Hin ].
  apply andb_true_iff in Hc as [Hc _]. apply andb_true_iff in Hc as [Hc Hmt].
  apply andb_true_iff in Hc as [Hx _].
  apply Pos.eqb_eq in Hx. apply Pos.eqb_eq in Hmt. subst x' mt.
  destruct Hin as [[= <- <-] | Hin].
  - exists mf. left. reflexivity.
  - destruct (IH body o y Hin) as (s & Hs). exists s. right. exact Hs.
Qed.

Lemma after_local_allocation_lets (x : variable) (e : expr) :
  ∀ b, b ∈ let_bindings (after_local_allocation x e) -> b ∈ let_bindings e.
Proof.
  intros b Hb. unfold after_local_allocation in Hb.
  destruct e as [y k d body | | | | | | | ]; try exact Hb.
  destruct d as [ | p args | | | | | ]; try exact Hb.
  destruct p as [ | | | | r]; try exact Hb.
  destruct r; try exact Hb.
  destruct args as [ | [x' | | ] rest]; try exact Hb.
  destruct (Pos.eqb y x && Pos.eqb x' x); [ right; exact Hb | exact Hb ].
Qed.

Lemma set_closures_projs :
  (∀ e : expr,
     ∀ (x : variable) (projs : list (function_slot * variable)),
       (x, projs) ∈ set_closures e ->
       ∀ (o : function_slot) (y : variable),
         (o, y) ∈ projs ->
         ∃ s : function_slot,
           (y, Project_function_slot s o (Var x)) ∈ let_bindings e)
  ∧ (∀ h : continuation_handler,
       ∀ (x : variable) (projs : list (function_slot * variable)),
         (x, projs) ∈ handler_set_closures h ->
         ∀ (o : function_slot) (y : variable),
           (o, y) ∈ projs ->
           ∃ s : function_slot,
             (y, Project_function_slot s o (Var x))
             ∈ handler_let_bindings h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind; cbn [set_closures
    handler_set_closures let_bindings handler_let_bindings].
  - intros x0 kind d body IH x projs Hin o y Hy.
    apply in_app_or in Hin as [Hin | Hin].
    + destruct d as [ | | | slot decls vs | | | ];
        try (exact (False_ind _ Hin)).
      destruct Hin as [[= <- <-] | []].
      unfold projected_set_closures in Hy.
      destruct decls as [ | [first id] others]; [ destruct Hy | ].
      destruct (Pos.eqb first slot); [ | destruct Hy ].
      destruct (let_projections_in x0 slot _ _ o y Hy) as (s & Hs).
      exists s. right. exact (after_local_allocation_lets x0 body _ Hs).
    + destruct (IH x projs Hin o y Hy) as (s & Hs). exists s. right. exact Hs.
  - intros bindings body Hcodes IH x projs Hin o y Hy.
    apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as (b & Hb & Hin).
      destruct b as [id c | id | sym const]; cbn in Hin;
        try (exact (False_ind _ Hin)).
      destruct c as [params mc ret exn md cbody]. cbn in Hin.
      destruct (Hcodes id _ Hb x projs Hin o y Hy) as (s & Hs).
      exists s. apply in_or_app. left. apply in_flat_map.
      exists (Bind_code id (Code params mc ret exn md cbody)).
      split; [ exact Hb | exact Hs ].
    + destruct (IH x projs Hin o y Hy) as (s & Hs).
      exists s. apply in_or_app. right. exact Hs.
  - intros k h body IHh IH x projs Hin o y Hy.
    apply in_app_or in Hin as [Hin | Hin].
    + destruct (IHh x projs Hin o y Hy) as (s & Hs).
      exists s. apply in_or_app. left. exact Hs.
    + destruct (IH x projs Hin o y Hy) as (s & Hs).
      exists s. apply in_or_app. right. exact Hs.
  - intros handlers body IHh IH x projs Hin o y Hy.
    apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as ([k h] & Hkh & Hin).
      destruct (IHh k h Hkh x projs Hin o y Hy) as (s & Hs).
      exists s. apply in_or_app. left. apply in_flat_map.
      exists (k, h). split; [ exact Hkh | exact Hs ].
    + destruct (IH x projs Hin o y Hy) as (s & Hs).
      exists s. apply in_or_app. right. exact Hs.
  - intros ap x projs [].
  - intros ac x projs [].
  - intros scrutinee arms x projs [].
  - intros x projs [].
  - intros params metadata handler IH. exact IH.
Qed.

Lemma nodup_map_fst_filter {A B : Type} (p : A * B -> bool) :
  ∀ l : list (A * B), NoDup (map fst l) -> NoDup (map fst (filter p l)).
Proof.
  induction l as [ | [x y] l IH]; intros Hnd; [ constructor | ].
  inversion Hnd as [ | x' l' Hx Hnd' ]; subst. cbn [filter].
  destruct (p (x, y)); [ | exact (IH Hnd') ].
  cbn [map fst]. constructor; [ | exact (IH Hnd') ].
  intros Hin. apply Hx. apply in_map_iff in Hin as ([x2 y2] & Hx2 & Hin).
  cbn in Hx2. subst x2. apply filter_In in Hin as [Hin _].
  exact (in_map fst l (x, y2) Hin).
Qed.

Lemma is_bound_compact {A : Type} (g : positive) (l : finite_map A) :
  is_bound g (compact l) = is_bound g l.
Proof. unfold is_bound. rewrite lookup_compact. reflexivity. Qed.

Lemma nodup_compact {A : Type} (l : finite_map A) :
  NoDup (map fst (compact l)).
Proof. apply distinct_NoDup. apply compact_distinct. Qed.

(* The facts of [sgroup] that [record_set_of_closures_deps] adds. *)
Lemma record_set_group (env : traverse_env) (origin : node)
    (closures : list (node * option code_id))
    (fs : list (function_slot * node))
    (value_slots : list (value_slot * simple))
    (a : acc) :
  ∃ L : list fact,
    acc_facts (record_set_of_closures_deps env closures fs value_slots a)
    = L ++ acc_facts a
    ∧ ∀ fct : fact,
        fct ∈ L
        <-> sg_fact
              (SGroup origin (map fst closures) fs
                 (map (fun '(w, s) => (w, node_of_simple s))
                    (filter (fun '(w, _) => should_keep_value_slot env w)
                       (compact value_slots))))
              fct.
Proof.
  destruct (record_set_facts env closures fs value_slots a)
    as [(L & HL & HLf) _].
  exists L. split; [ exact HL | ]. intros fct. rewrite HLf.
  unfold sg_fact. cbn [sg_bases sg_fslots sg_vslots]. split.
  - intros (base & code & Hin & [(w & s & Hw & ->) | (f & c & Hf & ->)]).
    + exists base. split; [ exact (in_map fst _ (base, code) Hin) | right ].
      exists w, (node_of_simple s). split; [ reflexivity | ].
      apply in_map_iff. exists (w, s). auto.
    + exists base. split; [ exact (in_map fst _ (base, code) Hin) | left ].
      exists f, c. auto.
  - intros (N & HN & [(f & b & -> & Hf) | (w & b & -> & Hw)]).
    + apply in_map_iff in HN as ([base code] & Hb & HN). cbn in Hb. subst N.
      exists base, code. split; [ exact HN | right ]. exists f, b. auto.
    + apply in_map_iff in HN as ([base code] & Hb & HN). cbn in Hb. subst N.
      apply in_map_iff in Hw as ([w' s] & Heq & Hw). injection Heq as -> <-.
      exists base, code. split; [ exact HN | left ]. exists w, s. auto.
Qed.

Lemma map_fst_pairs {A B C : Type} (f : B -> C) (l : list (A * B)) :
  map fst (map (fun '(a, b) => (a, f b)) l) = map fst l.
Proof.
  rewrite map_map. apply map_ext. intros [a b]. reflexivity.
Qed.

Section Steps.

Variable e0 : expr.
Variable keep : value_slot -> bool.
Hypothesis Hnl : NoDup (map fst (let_bindings e0)).
Hypothesis Hns : NoDup (map fst (sym_bindings e0)).

(* The [Let] of a closure of a set adds the group of its set, whose origin
   is its variable. *)
Lemma closure_let_step (env : traverse_env) (x : variable)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (a : acc) (Gs : list sgroup) :
  env_set_closures env = set_closures e0 ->
  (∀ w : value_slot, should_keep_value_slot env w = keep w) ->
  (x, Closure_of_set slot decls vs) ∈ let_bindings e0 ->
  inv e0 keep Gs a ->
  (∀ γ : sgroup, γ ∈ Gs -> sg_origin γ <> Node_variable x) ->
  ∃ Gs' : list sgroup,
    inv e0 keep Gs' (traverse_closure_of_set env x slot decls vs a)
    ∧ (∀ γ : sgroup, γ ∈ Gs' -> γ ∈ Gs ∨ sg_origin γ = Node_variable x)
    ∧ ∀ γ : sgroup, γ ∈ Gs -> γ ∈ Gs'.
Proof.
  intros Hsc Hkeep Hx Hinv Horig. unfold traverse_closure_of_set.
  destruct (among slot (map fst (compact decls))) eqn:Hslot;
    [ | exists Gs; split; [ exact Hinv | split; auto ] ].
  destruct (closures_of_set x slot (set_closures_of env x) (compact decls) a)
    as [cl a1] eqn:Ecl.
  destruct (closures_of_set_spec x slot _ _ a cl a1 Ecl)
    as (Hf1 & Hn1 & Hmap & Hcl).
  set (closures := map (fun '(_, name, id) => (name, Some id)) cl).
  set (fs := map (fun '(f, name, _) => (f, name)) cl).
  set (vsk := map (fun '(w, s) => (w, node_of_simple s))
                (filter (fun '(w, _) => should_keep_value_slot env w)
                   (compact vs))).
  set (γ := SGroup (Node_variable x) (map fst closures) fs vsk).
  assert (Hbases : ∀ N : node,
             N ∈ map fst closures <->
             ∃ (f : function_slot) (id : code_id), (f, N, id) ∈ cl).
  { intros N. unfold closures. rewrite map_map. split.
    - intros Hin. apply in_map_iff in Hin as ([[f N'] id] & Heq & Hin).
      cbn in Heq. subst N'. exists f, id. exact Hin.
    - intros (f & id & Hin). apply in_map_iff. exists (f, N, id). auto. }
  assert (Hfs : ∀ (f : function_slot) (M : node),
             (f, M) ∈ fs <-> ∃ id : code_id, (f, M, id) ∈ cl).
  { intros f M. unfold fs. split.
    - intros Hin. apply in_map_iff in Hin as ([[f' M'] id] & Heq & Hin).
      injection Heq as -> ->. exists id. exact Hin.
    - intros (id & Hin). apply in_map_iff. exists (f, M, id). auto. }
  assert (Hfs_keys : map fst fs = map fst (compact decls)).
  { transitivity (map (fun '(f, _, _) => f) cl); [ | exact Hmap ].
    unfold fs. rewrite map_map.
    apply map_ext. intros [[f M] id]. reflexivity. }
  exists (γ :: Gs).
  split;
    [ | split; [ intros γ' [<- | Hγ']; auto | intros γ' Hγ'; right; auto ] ].
  assert (Hok : sg_ok e0 keep γ).
  { split; [ cbn; rewrite Hfs_keys; apply nodup_compact | ].
    split.
    { cbn. unfold vsk. rewrite map_map.
      replace (map (fun x0 => fst (let '(w, s) := x0 in (w, node_of_simple s)))
                 (filter (fun '(w, _) => should_keep_value_slot env w)
                    (compact vs)))
        with (map fst (filter (fun '(w, _) => should_keep_value_slot env w)
                         (compact vs)))
        by (apply map_ext; intros [w s]; reflexivity).
      apply nodup_map_fst_filter. apply nodup_compact. }
    split.
    { intros w b Hin. cbn in Hin. unfold vsk in Hin.
      apply in_map_iff in Hin as ([w' s] & Heq & Hin). injection Heq as -> _.
      apply filter_In in Hin as [_ Hk]. rewrite <- Hkeep. exact Hk. }
    left. exists x, slot, decls, vs.
    split; [ reflexivity | split; [ exact Hx | split ] ].
    { cbn. apply Hbases. apply among_In in Hslot.
      assert (Hslot' : slot ∈ map (fun '(f, _, _) => f) cl)
        by (rewrite Hmap; exact Hslot).
      apply in_map_iff in Hslot' as ([[f N] id] & Heq & Hin). cbn in Heq.
      subst f. exists slot, id.
      rewrite <- (proj1 (Hcl slot N id Hin) eq_refl). exact Hin. }
    split.
    { intros g. cbn [sg_fslots γ].
      transitivity (among g (map fst fs)); [ apply is_bound_among | ].
      rewrite Hfs_keys, <- is_bound_among. apply is_bound_compact. }
    split.
    { intros f M Hin. cbn. apply Hbases. apply Hfs in Hin as (id & Hin).
      exists f, id. exact Hin. }
    intros N HN. cbn in HN. apply Hbases in HN as (f & id & Hin).
    destruct (proj2 (Hcl f N id Hin)) as [-> | [(y & Hy & ->) | (k & -> & _)]];
      [ left; reflexivity | right; left | right; right; exists k; reflexivity ].
    unfold set_closures_of in Hy. rewrite Hsc in Hy.
    destruct (lookup x (set_closures e0)) as [projs | ] eqn:Hp;
      [ | discriminate Hy ].
    destruct (proj1 set_closures_projs e0 x projs (lookup_in _ _ _ Hp) f y
                (lookup_in _ _ _ Hy))
      as (s & Hs).
    exists y, s, f. split; [ reflexivity | exact Hs ]. }
  apply (inv_add e0 keep Gs γ a1).
  - exact (inv_ns e0 keep Gs a a1 Hinv (ns_same a a1 Hf1 Hn1)).
  - exact Hok.
  - apply (group_fresh e0 keep Gs γ a Hnl Hns Hinv Hok).
    + intros γ' Hγ'. exact (Horig γ' Hγ').
    + intros k Hk. cbn in Hk. apply Hbases in Hk as (f & id & Hin).
      destruct (proj2 (Hcl f _ id Hin))
        as [E | [(y & _ & E) | (k' & [= <-] & Hk1 & _)]];
        [ discriminate E | discriminate E | exact Hk1 ].
  - exact (record_set_group env (Node_variable x) closures fs vs a1).
  - destruct (record_set_facts env closures fs vs a1) as [_ ->]. lia.
  - intros k Hk. cbn in Hk. apply Hbases in Hk as (f & id & Hin).
    destruct (proj2 (Hcl f _ id Hin))
      as [E | [(y & _ & E) | (k' & [= <-] & _ & Hk2)]];
      [ discriminate E | discriminate E | ].
    destruct (record_set_facts env closures fs vs a1) as [_ ->]. exact Hk2.
Qed.

(* The binding of a static closure adds the group of its symbol. *)
Lemma static_closure_step (env : traverse_env) (s : symbol)
    (slot : function_slot) (cs : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (a : acc) (Gs : list sgroup) :
  (∀ w : value_slot, should_keep_value_slot env w = keep w) ->
  (s, Static_const_closure_of_set slot cs decls vs) ∈ sym_bindings e0 ->
  inv e0 keep Gs a ->
  (∀ γ : sgroup, γ ∈ Gs -> sg_origin γ <> Node_symbol s) ->
  ∃ Gs' : list sgroup,
    inv e0 keep Gs'
      (traverse_static_binding env
         (Bind_symbol s (Static_const_closure_of_set slot cs decls vs)) a)
    ∧ (∀ γ : sgroup, γ ∈ Gs' -> γ ∈ Gs ∨ sg_origin γ = Node_symbol s)
    ∧ (∀ γ : sgroup, γ ∈ Gs -> γ ∈ Gs')
    ∧ ∃ γ : sgroup, γ ∈ Gs' ∧ sg_origin γ = Node_symbol s.
Proof.
  intros Hkeep Hs Hinv Horig. cbn [traverse_static_binding].
  set (fs := map (fun '(f, sym) => (f, Node_symbol sym)) (compact cs)).
  set (closures := [(Node_symbol s, lookup slot decls)]).
  set (vsk := map (fun '(w, s0) => (w, node_of_simple s0))
                (filter (fun '(w, _) => should_keep_value_slot env w)
                   (compact vs))).
  set (γ := SGroup (Node_symbol s) (map fst closures) fs vsk).
  exists (γ :: Gs).
  split; [ | split; [ intros γ' [<- | Hγ']; auto | split ] ];
    [ | intros γ' Hγ'; right; exact Hγ'
      | exists γ; split; [ left | ]; reflexivity ].
  assert (Hvsk : vsk = map (fun '(w, s0) => (w, node_of_simple s0))
                         (filter (fun '(w, _) => keep w) (compact vs))).
  { unfold vsk. f_equal. apply filter_ext. intros [w s0]. apply Hkeep. }
  assert (Hok : sg_ok e0 keep γ).
  { split.
    { cbn [sg_fslots γ]. unfold fs.
      rewrite (map_fst_pairs Node_symbol). apply nodup_compact. }
    split.
    { cbn [sg_vslots γ]. unfold vsk.
      rewrite (map_fst_pairs node_of_simple).
      apply nodup_map_fst_filter. apply nodup_compact. }
    split.
    { intros w b Hin. cbn in Hin. unfold vsk in Hin.
      apply in_map_iff in Hin as ([w' s0] & Heq & Hin). injection Heq as -> _.
      apply filter_In in Hin as [_ Hk]. rewrite <- Hkeep. exact Hk. }
    right. exists s, slot, cs, decls, vs.
    split; [ reflexivity | split; [ exact Hs | split; [ reflexivity | ] ] ].
    split; [ reflexivity | exact Hvsk ]. }
  apply (inv_add e0 keep Gs γ a).
  - exact Hinv.
  - exact Hok.
  - apply (group_fresh e0 keep Gs γ a Hnl Hns Hinv Hok).
    + intros γ' Hγ'. exact (Horig γ' Hγ').
    + intros k [E | []]. discriminate E.
  - exact (record_set_group env (Node_symbol s) closures fs vs a).
  - destruct (record_set_facts env closures fs vs a) as [_ ->]. lia.
  - intros k [E | []]. discriminate E.
Qed.

End Steps.

(* ======================================================================== *)
(* The traversal                                                            *)
(* ======================================================================== *)

(* The origins that bindings give: the variables of the [Let]s of closures
   among [L], the symbols of the static closures among [S]. *)
Definition orig (L : list (variable * named))
    (S : list (symbol * static_const)) (N : node) : Prop :=
  (∃ (x : variable) (slot : function_slot)
     (decls : list (function_slot * code_id))
     (vs : list (value_slot * simple)),
     N = Node_variable x ∧ (x, Closure_of_set slot decls vs) ∈ L)
  ∨ ∃ (t : symbol) (slot : function_slot)
      (cs : list (function_slot * symbol))
      (decls : list (function_slot * code_id))
      (vs : list (value_slot * simple)),
      N = Node_symbol t ∧ (t, Static_const_closure_of_set slot cs decls vs) ∈ S.


Lemma orig_app (L1 L2 : list (variable * named))
    (S1 S2 : list (symbol * static_const)) (N : node) :
  orig (L1 ++ L2) (S1 ++ S2) N <-> orig L1 S1 N ∨ orig L2 S2 N.
Proof.
  unfold orig. split.
  - intros [(x & sl & d & v & -> & Hin) | (t & sl & cs & d & v & -> & Hin)];
      apply in_app_or in Hin as [Hin | Hin].
    + left. left. exists x, sl, d, v. auto.
    + right. left. exists x, sl, d, v. auto.
    + left. right. exists t, sl, cs, d, v. auto.
    + right. right. exists t, sl, cs, d, v. auto.
  - intros [[(x & sl & d & v & -> & Hin) | (t & sl & cs & d & v & -> & Hin)]
           | [(x & sl & d & v & -> & Hin) | (t & sl & cs & d & v & -> & Hin)]].
    + left. exists x, sl, d, v. split; [ reflexivity | apply in_or_app; auto ].
    + right. exists t, sl, cs, d, v.
      split; [ reflexivity | apply in_or_app; auto ].
    + left. exists x, sl, d, v. split; [ reflexivity | apply in_or_app; auto ].
    + right. exists t, sl, cs, d, v.
      split; [ reflexivity | apply in_or_app; auto ].
Qed.

Lemma nodup_app_fst {A B : Type} (l1 l2 : list (A * B)) (x : A) (v1 v2 : B) :
  NoDup (map fst (l1 ++ l2)) -> (x, v1) ∈ l1 -> (x, v2) ∈ l2 -> False.
Proof.
  intros Hnd H1 H2. rewrite map_app in Hnd.
  apply (NoDup_app_remove_l (map fst l1) (map fst l2)) in Hnd as Hnd2.
  revert Hnd H1. induction l1 as [ | [y w] l1 IH]; intros Hnd H1;
    [ destruct H1 | ].
  cbn [map app fst] in Hnd. inversion Hnd as [ | y' l' Hy Hnd' ]; subst.
  destruct H1 as [[= -> ->] | H1].
  - apply Hy. apply in_or_app. right. exact (in_map fst l2 (x, v2) H2).
  - exact (IH Hnd' H1).
Qed.

Lemma orig_disjoint (L1 L2 : list (variable * named))
    (S1 S2 : list (symbol * static_const)) (N : node) :
  NoDup (map fst (L1 ++ L2)) -> NoDup (map fst (S1 ++ S2)) ->
  orig L1 S1 N -> orig L2 S2 N -> False.
Proof.
  intros HL HS H1 H2.
  destruct H1 as [(x & s1 & d1 & v1 & E1 & H1)
                 | (t & s1 & c1 & d1 & v1 & E1 & H1)];
    destruct H2 as [(x' & s2 & d2 & v2 & E2 & H2)
                   | (t' & s2 & c2 & d2 & v2 & E2 & H2)];
    rewrite E1 in E2; try discriminate E2; injection E2 as E2; subst.
  - exact (nodup_app_fst _ _ _ _ _ HL H1 H2).
  - exact (nodup_app_fst _ _ _ _ _ HS H1 H2).
Qed.

Section Composition.

Variable e0 : expr.
Variable keep : value_slot -> bool.

(* A step that adds groups of origins of [O] only, if the groups that it
   is given have none of them, keeps those, and adds one of each origin of
   [Os]. *)
Definition step_ok (O Os : node -> Prop) (f : acc -> acc) : Prop :=
  ∀ (a : acc) (Gs : list sgroup),
    inv e0 keep Gs a ->
    (∀ γ : sgroup, γ ∈ Gs -> ~ O (sg_origin γ)) ->
    ∃ Gs' : list sgroup,
      inv e0 keep Gs' (f a)
      ∧ (∀ γ : sgroup, γ ∈ Gs' -> γ ∈ Gs ∨ O (sg_origin γ))
      ∧ (∀ γ : sgroup, γ ∈ Gs -> γ ∈ Gs')
      ∧ ∀ N : node, Os N -> ∃ γ : sgroup, γ ∈ Gs' ∧ sg_origin γ = N.

Definition no_origin (N : node) : Prop := False.

Lemma step_ok_ext (O Os : node -> Prop) (f g : acc -> acc) :
  (∀ a : acc, g a = f a) -> step_ok O Os f -> step_ok O Os g.
Proof. intros He Hf a Gs. rewrite He. apply Hf. Qed.

Lemma step_ok_ns (O : node -> Prop) (f : acc -> acc) :
  (∀ a : acc, ns_ok a (f a)) -> step_ok O no_origin f.
Proof.
  intros Hf a Gs Hinv _. exists Gs.
  split; [ exact (inv_ns e0 keep Gs a (f a) Hinv (Hf a)) | ].
  split; [ auto | split; [ auto | intros N [] ] ].
Qed.

Lemma step_ok_mono (O O' Os Os' : node -> Prop) (f : acc -> acc) :
  (∀ N : node, O N -> O' N) -> (∀ N : node, Os' N -> Os N) ->
  step_ok O Os f -> step_ok O' Os' f.
Proof.
  intros Hsub Hsub' Hf a Gs Hinv Hfresh.
  destruct (Hf a Gs Hinv (fun γ Hγ HO => Hfresh γ Hγ (Hsub _ HO)))
    as (Gs' & Hinv' & HGs' & Hkeep' & Hpres).
  exists Gs'. split; [ exact Hinv' | split; [ | split; [ exact Hkeep' | ] ] ].
  - intros γ Hγ. destruct (HGs' γ Hγ) as [H | H]; [ left | right ]; auto.
  - intros N HN. exact (Hpres N (Hsub' N HN)).
Qed.

Lemma step_ok_seq (O1 O2 Os1 Os2 : node -> Prop) (f1 f2 : acc -> acc) :
  (∀ N : node, O1 N -> O2 N -> False) ->
  step_ok O1 Os1 f1 -> step_ok O2 Os2 f2 ->
  step_ok (fun N => O1 N ∨ O2 N) (fun N => Os1 N ∨ Os2 N)
    (fun a => f2 (f1 a)).
Proof.
  intros Hdis H1 H2 a Gs Hinv Hfresh.
  destruct (H1 a Gs Hinv (fun γ Hγ HO => Hfresh γ Hγ (or_introl HO)))
    as (Gs1 & Hinv1 & HGs1 & Hk1 & Hp1).
  destruct (H2 (f1 a) Gs1 Hinv1) as (Gs2 & Hinv2 & HGs2 & Hk2 & Hp2).
  { intros γ Hγ HO. destruct (HGs1 γ Hγ) as [Hold | Hnew].
    - exact (Hfresh γ Hold (or_intror HO)).
    - exact (Hdis _ Hnew HO). }
  exists Gs2. split; [ exact Hinv2 | split; [ | split ] ].
  - intros γ Hγ. destruct (HGs2 γ Hγ) as [H | H]; [ | right; right; exact H ].
    destruct (HGs1 γ H) as [H' | H'];
      [ left; exact H' | right; left; exact H' ].
  - intros γ Hγ. exact (Hk2 γ (Hk1 γ Hγ)).
  - intros N [HN | HN].
    + destruct (Hp1 N HN) as (γ & Hγ & Ho). exists γ. auto.
    + exact (Hp2 N HN).
Qed.

(* Each element of a list has origins that no later element has. *)
Fixpoint disjoint_family {A : Type} (O : A -> node -> Prop) (l : list A)
    : Prop :=
  match l with
  | [] => True
  | x :: l' =>
    (∀ N : node, O x N -> (∃ y : A, y ∈ l' ∧ O y N) -> False)
    ∧ disjoint_family O l'
  end.

Lemma step_ok_fold {A : Type} (g : acc -> A -> acc)
    (O Os : A -> node -> Prop) (l : list A) :
  (∀ x : A, x ∈ l -> step_ok (O x) (Os x) (fun a => g a x)) ->
  disjoint_family O l ->
  step_ok (fun N => ∃ x : A, x ∈ l ∧ O x N)
    (fun N => ∃ x : A, x ∈ l ∧ Os x N) (fun a => List.fold_left g l a).
Proof.
  induction l as [ | x l IH]; intros Hsteps Hdis.
  - apply (step_ok_mono no_origin _ no_origin);
      [ intros N [] | intros N (x & [] & _) | ].
    apply step_ok_ns. intros a. apply ns_refl.
  - destruct Hdis as [Hx Hdis].
    apply (step_ok_mono (fun N => O x N ∨ (∃ y : A, y ∈ l ∧ O y N)) _
             (fun N => Os x N ∨ (∃ y : A, y ∈ l ∧ Os y N))).
    { intros N [H | (y & Hy & H)].
      - exists x. split; [ left; reflexivity | exact H ].
      - exists y. split; [ right; exact Hy | exact H ]. }
    { intros N (y & [<- | Hy] & H); [ left; exact H | right ].
      exists y. auto. }
    apply (step_ok_seq (O x) (fun N => ∃ y : A, y ∈ l ∧ O y N) (Os x)
             (fun N => ∃ y : A, y ∈ l ∧ Os y N)
             (fun a => g a x) (fun a => List.fold_left g l a)).
    + exact Hx.
    + apply Hsteps. left. reflexivity.
    + apply IH; [ | exact Hdis ]. intros y Hy. apply Hsteps. right. exact Hy.
Qed.

Lemma disjoint_family_mono {A : Type} (O O' : A -> node -> Prop)
    (l : list A) :
  (∀ (x : A) (N : node), O x N -> O' x N) ->
  disjoint_family O' l -> disjoint_family O l.
Proof.
  intros Hsub. induction l as [ | x l IH]; cbn; [ auto | ].
  intros [Hx Hrest]. split; [ | exact (IH Hrest) ].
  intros N HO (y & Hy & HOy). exact (Hx N (Hsub _ _ HO) (ex_intro _ y
    (conj Hy (Hsub _ _ HOy)))).
Qed.

(* The bindings of a list of elements, each with its own: an element's
   origins are not those of a later one. *)
Lemma disjoint_family_bindings {A : Type}
    (gL : A -> list (variable * named)) (gS : A -> list (symbol * static_const))
    (l : list A) :
  NoDup (map fst (flat_map gL l)) -> NoDup (map fst (flat_map gS l)) ->
  disjoint_family (fun x => orig (gL x) (gS x)) l.
Proof.
  induction l as [ | x l IH]; intros HL HS; cbn; [ exact I | ].
  cbn [flat_map] in HL, HS. split.
  - intros N Hx (y & Hy & HOy).
    apply (orig_disjoint (gL x) (flat_map gL l) (gS x) (flat_map gS l) N HL
             HS Hx).
    destruct HOy as [(v & sl & d & vs & -> & H)
                    | (t & sl & cs & d & vs & -> & H)].
    + left. exists v, sl, d, vs. split; [ reflexivity | ].
      apply in_flat_map. exists y. auto.
    + right. exists t, sl, cs, d, vs. split; [ reflexivity | ].
      apply in_flat_map. exists y. auto.
  - rewrite map_app in HL, HS.
    exact (IH (NoDup_app_remove_l _ _ HL) (NoDup_app_remove_l _ _ HS)).
Qed.

(* Two different elements of a list have no key in common, if the keys of
   the list are distinct. *)
Lemma nodup_flat_map_distinct {A B C : Type} (g : A -> list (B * C))
    (l : list A) (x y : A) (k : B) (v w : C) :
  NoDup (map fst (flat_map g l)) -> x ∈ l -> y ∈ l -> x <> y ->
  (k, v) ∈ g x -> (k, w) ∈ g y -> False.
Proof.
  induction l as [ | z l IH]; intros Hnd Hx Hy Hne Hv Hw; [ destruct Hx | ].
  cbn [flat_map] in Hnd.
  destruct Hx as [Ex | Hx]; destruct Hy as [Ey | Hy].
  - apply Hne. rewrite <- Ex, <- Ey. reflexivity.
  - subst z. apply (nodup_app_fst (g x) (flat_map g l) k v w Hnd Hv).
    apply in_flat_map. exists y. auto.
  - subst z. apply (nodup_app_fst (g y) (flat_map g l) k w v Hnd Hw).
    apply in_flat_map. exists x. auto.
  - rewrite map_app in Hnd.
    exact (IH (NoDup_app_remove_l _ _ Hnd) Hx Hy Hne Hv Hw).
Qed.

End Composition.

(* No old form of set of closures: those of the fragment. *)
Fixpoint no_old_sets (e : expr) : Prop :=
  match e with
  | Let _ _ d body =>
    (∀ (id : code_id) (vs : list (value_slot * simple)),
       d <> Set_of_closures id vs)
    ∧ no_old_sets body
  | Let_static bindings body =>
    every binding_no_old_sets bindings ∧ no_old_sets body
  | Let_cont_non_recursive _ h body =>
    handler_no_old_sets h ∧ no_old_sets body
  | Let_cont_recursive handlers body =>
    every (fun '(_, h) => handler_no_old_sets h) handlers ∧ no_old_sets body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => True
  end
with handler_no_old_sets (h : continuation_handler) : Prop :=
  match h with
  | Handler _ _ handler => no_old_sets handler
  end
with binding_no_old_sets (binding : static_binding) : Prop :=
  match binding with
  | Bind_code _ c => code_no_old_sets c ∧ code_sym_bindings c = []
  | Bind_deleted_code _ => True
  | Bind_symbol _ const =>
    ∀ (id : code_id) (vs : list (value_slot * simple)),
      const <> Static_const_set_of_closures id vs
  end
with code_no_old_sets (c : code) : Prop :=
  match c with
  | Code _ _ _ _ _ body => no_old_sets body
  end.

Lemma ns_traverse_named (env : traverse_env) (x : variable) (d : named)
    (a : acc) :
  (∀ (slot : function_slot) (decls : list (function_slot * code_id))
     (vs : list (value_slot * simple)), d <> Closure_of_set slot decls vs) ->
  (∀ (id : code_id) (vs : list (value_slot * simple)),
     d <> Set_of_closures id vs) ->
  ns_ok a (traverse_named env x d a).
Proof.
  intros Hcl Hset.
  destruct d as [s | p args | id vs | slot decls vs | w c | pf w c | mf mt c];
    cbn [traverse_named]; cbv zeta.
  - apply (ns_trans _ _ _ (ns_note_simple s a)). apply ns_add_fact. cbn. auto.
  - apply ns_traverse_prim.
  - destruct (Hset id vs eq_refl).
  - destruct (Hcl slot decls vs eq_refl).
  - apply (ns_trans _ _ _ (ns_note_simple c a)). apply ns_add_fact. cbn. auto.
  - apply (ns_trans _ _ _ (ns_note_simple c a)). apply ns_add_fact. cbn. auto.
  - apply (ns_trans _ _ _ (ns_note_simple c a)). apply ns_add_fact. cbn. auto.
Qed.

(* The origins of the static closures of [e]. *)
Definition static_origin_in (e : expr) (N : node) : Prop :=
  ∃ (t : symbol) (slot : function_slot)
    (cs : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)),
    N = Node_symbol t
    ∧ (t, Static_const_closure_of_set slot cs decls vs) ∈ sym_bindings e.

Section Main.

Variable e0 : expr.
Variable keep : value_slot -> bool.
Hypothesis Hnl : NoDup (map fst (let_bindings e0)).
Hypothesis Hns : NoDup (map fst (sym_bindings e0)).

Definition env_good (env : traverse_env) : Prop :=
  env_set_closures env = set_closures e0
  ∧ ∀ w : value_slot, should_keep_value_slot env w = keep w.

Lemma env_good_add_cont (env : traverse_env) (k : continuation)
    (params : list node) :
  env_good env -> env_good (add_cont env k params).
Proof. intros H. exact H. Qed.

Lemma env_good_add_rec_conts (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) :
  env_good env -> env_good (add_rec_conts env handlers).
Proof.
  intros H. unfold add_rec_conts. induction handlers as [ | [k h] hs IH];
    [ exact H | exact IH ].
Qed.

Lemma env_good_code_env (env : traverse_env) (id : code_id)
    (ret exn : continuation) (cd : code_dep) :
  env_good env -> env_good (code_env env id ret exn cd).
Proof. intros H. exact H. Qed.

(* The bindings of a part of [e0]. *)
Definition pre (L : list (variable * named))
    (S : list (symbol * static_const)) : Prop :=
  (∀ b, b ∈ L -> b ∈ let_bindings e0) ∧ (∀ b, b ∈ S -> b ∈ sym_bindings e0)
  ∧ NoDup (map fst L) ∧ NoDup (map fst S).

Lemma pre_app_l (L1 L2 : list (variable * named))
    (S1 S2 : list (symbol * static_const)) :
  pre (L1 ++ L2) (S1 ++ S2) -> pre L1 S1.
Proof.
  intros (HL & HS & HnL & HnS). rewrite map_app in HnL, HnS.
  split; [ intros b Hb; apply HL; apply in_or_app; left; exact Hb | ].
  split; [ intros b Hb; apply HS; apply in_or_app; left; exact Hb | ].
  split; [ exact (NoDup_app_remove_r _ _ HnL)
         | exact (NoDup_app_remove_r _ _ HnS) ].
Qed.

Lemma pre_app_r (L1 L2 : list (variable * named))
    (S1 S2 : list (symbol * static_const)) :
  pre (L1 ++ L2) (S1 ++ S2) -> pre L2 S2.
Proof.
  intros (HL & HS & HnL & HnS). rewrite map_app in HnL, HnS.
  split; [ intros b Hb; apply HL; apply in_or_app; right; exact Hb | ].
  split; [ intros b Hb; apply HS; apply in_or_app; right; exact Hb | ].
  split; [ exact (NoDup_app_remove_l _ _ HnL)
         | exact (NoDup_app_remove_l _ _ HnS) ].
Qed.

Lemma pre_flat_map {A : Type} (gL : A -> list (variable * named))
    (gS : A -> list (symbol * static_const)) (l : list A) (x : A) :
  pre (flat_map gL l) (flat_map gS l) -> x ∈ l -> pre (gL x) (gS x).
Proof.
  induction l as [ | y l IH]; intros Hp Hx; [ destruct Hx | ].
  cbn [flat_map] in Hp. destruct Hx as [<- | Hx].
  - exact (pre_app_l _ _ _ _ Hp).
  - exact (IH (pre_app_r _ _ _ _ Hp) Hx).
Qed.

Abbreviation step_ok := (step_ok e0 keep).

(* The [Let] of a closure of a set. *)
Lemma closure_let_ok (env : traverse_env) (x : variable)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) :
  env_good env ->
  (x, Closure_of_set slot decls vs) ∈ let_bindings e0 ->
  step_ok (orig [(x, Closure_of_set slot decls vs)] []) no_origin
    (traverse_named env x (Closure_of_set slot decls vs)).
Proof.
  intros [Hsc Hkeep] Hx a Gs Hinv Hfresh.
  assert (Ho : orig [(x, Closure_of_set slot decls vs)] [] (Node_variable x)).
  { left. exists x, slot, decls, vs.
    split; [ reflexivity | left; reflexivity ]. }
  destruct (closure_let_step e0 keep Hnl Hns env x slot decls vs a Gs Hsc
              Hkeep Hx Hinv)
    as (Gs' & Hinv' & HGs' & Hk').
  { intros γ Hγ Ho'. apply (Hfresh γ Hγ). rewrite Ho'. exact Ho. }
  exists Gs'. split; [ exact Hinv' | split; [ | split; [ exact Hk' | ] ] ].
  - intros γ Hγ. destruct (HGs' γ Hγ) as [H | H]; [ left; exact H | right ].
    rewrite H. exact Ho.
  - intros N [].
Qed.

(* The binding of a static closure. *)
Lemma static_closure_ok (env : traverse_env) (s : symbol)
    (slot : function_slot) (cs : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) :
  env_good env ->
  (s, Static_const_closure_of_set slot cs decls vs) ∈ sym_bindings e0 ->
  step_ok (orig [] [(s, Static_const_closure_of_set slot cs decls vs)])
    (orig [] [(s, Static_const_closure_of_set slot cs decls vs)])
    (fun a =>
       traverse_static_binding env
         (Bind_symbol s (Static_const_closure_of_set slot cs decls vs)) a).
Proof.
  intros [Hsc Hkeep] Hs a Gs Hinv Hfresh.
  assert (Ho : orig [] [(s, Static_const_closure_of_set slot cs decls vs)]
                 (Node_symbol s)).
  { right. exists s, slot, cs, decls, vs.
    split; [ reflexivity | left; reflexivity ]. }
  destruct (static_closure_step e0 keep Hnl Hns env s slot cs decls vs a Gs
              Hkeep Hs Hinv)
    as (Gs' & Hinv' & HGs' & Hk' & (γ0 & Hγ0 & Ho0)).
  { intros γ Hγ Ho'. apply (Hfresh γ Hγ). rewrite Ho'. exact Ho. }
  exists Gs'. split; [ exact Hinv' | split; [ | split; [ exact Hk' | ] ] ].
  - intros γ Hγ. destruct (HGs' γ Hγ) as [H | H]; [ left; exact H | right ].
    rewrite H. exact Ho.
  - intros N [(x & _ & _ & _ & _ & [])
             | (t & sl & cs' & d & v & -> & [[= <- _] | []])].
    exists γ0. split; [ exact Hγ0 | exact Ho0 ].
Qed.

End Main.

Lemma orig_flat_map {A : Type} (gL : A -> list (variable * named))
    (gS : A -> list (symbol * static_const)) (l : list A) (N : node) :
  orig (flat_map gL l) (flat_map gS l) N
  <-> ∃ x : A, x ∈ l ∧ orig (gL x) (gS x) N.
Proof.
  split.
  - intros [(v & sl & d & vs & -> & Hin) | (t & sl & cs & d & vs & -> & Hin)];
      apply in_flat_map in Hin as (x & Hx & Hin); exists x; split; auto.
    + left. exists v, sl, d, vs. auto.
    + right. exists t, sl, cs, d, vs. auto.
  - intros (x & Hx & [(v & sl & d & vs & -> & Hin)
                     | (t & sl & cs & d & vs & -> & Hin)]).
    + left. exists v, sl, d, vs. split; [ reflexivity | ].
      apply in_flat_map. exists x. auto.
    + right. exists t, sl, cs, d, vs. split; [ reflexivity | ].
      apply in_flat_map. exists x. auto.
Qed.

(* The origins of the static closures among [S]. *)
Definition static_orig (S : list (symbol * static_const)) (N : node) : Prop :=
  ∃ (t : symbol) (slot : function_slot)
    (cs : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)),
    N = Node_symbol t ∧ (t, Static_const_closure_of_set slot cs decls vs) ∈ S.

Lemma static_orig_app (S1 S2 : list (symbol * static_const)) (N : node) :
  static_orig (S1 ++ S2) N -> static_orig S1 N ∨ static_orig S2 N.
Proof.
  intros (t & sl & cs & d & vs & -> & Hin).
  apply in_app_or in Hin as [Hin | Hin]; [ left | right ];
    exists t, sl, cs, d, vs; auto.
Qed.

Lemma static_orig_flat_map {A : Type} (gS : A -> list (symbol * static_const))
    (l : list A) (N : node) :
  static_orig (flat_map gS l) N -> ∃ x : A, x ∈ l ∧ static_orig (gS x) N.
Proof.
  intros (t & sl & cs & d & vs & -> & Hin).
  apply in_flat_map in Hin as (x & Hx & Hin). exists x.
  split; [ exact Hx | ]. exists t, sl, cs, d, vs. auto.
Qed.

Section Traverse.

Variable e0 : expr.
Variable keep : value_slot -> bool.
Hypothesis Hnl : NoDup (map fst (let_bindings e0)).
Hypothesis Hns : NoDup (map fst (sym_bindings e0)).

Abbreviation step_ok := (step_ok e0 keep).
Abbreviation env_good := (env_good e0 keep).
Abbreviation pre := (pre e0).

Definition P_expr (e : expr) : Prop :=
  ∀ env : traverse_env,
    env_good env -> pre (let_bindings e) (sym_bindings e) -> no_old_sets e ->
    step_ok (origin_in e) (static_origin_in e) (traverse env e).

Lemma ns_ok_step (O Os : node -> Prop) (f : acc -> acc) :
  (∀ a : acc, ns_ok a (f a)) -> step_ok O (fun N => no_origin N ∧ Os N) f.
Proof.
  intros Hf. apply (step_ok_mono e0 keep no_origin _ no_origin);
    [ intros N [] | intros N [H _]; exact H | ].
  apply step_ok_ns. exact Hf.
Qed.

Lemma traverse_ok :
  (∀ e : expr, P_expr e)
  ∧ (∀ h : continuation_handler,
       match h with Handler _ _ handler => P_expr handler end).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x k d body IH env Henv Hpre [Hd Hold].
    apply (step_ok_ext e0 keep _ _
             (fun a => traverse env body (traverse_named env x d a)));
      [ reflexivity | ].
    apply (step_ok_mono e0 keep
             (fun N => orig [(x, d)] [] N ∨ origin_in body N) _
             (fun N => no_origin N ∨ static_origin_in body N)).
    { intros N HN. unfold origin_in. cbn [let_bindings sym_bindings].
      change ((x, d) :: let_bindings body) with ([(x, d)] ++ let_bindings body).
      change (sym_bindings body) with ([] ++ sym_bindings body).
      apply orig_app. exact HN. }
    { intros N HN. right. exact HN. }
    cbn [let_bindings sym_bindings] in Hpre.
    change ((x, d) :: let_bindings body) with ([(x, d)] ++ let_bindings body)
      in Hpre.
    change (sym_bindings body) with ([] ++ sym_bindings body) in Hpre.
    apply step_ok_seq.
    + intros N H1 H2. destruct Hpre as (_ & _ & HnL & HnS).
      exact (orig_disjoint _ _ _ _ N HnL HnS H1 H2).
    + destruct d as [s | p args | id vs | slot decls vs | w c | pf w c
                    | mf mt c];
        try (apply (step_ok_mono e0 keep no_origin _ no_origin);
             [ intros N [] | intros N HN; exact HN | ];
             apply step_ok_ns; intros a; apply ns_traverse_named;
             intros; discriminate).
      * destruct (Hd id vs eq_refl).
      * apply (closure_let_ok e0 keep Hnl Hns); [ exact Henv | ].
        destruct Hpre as (HL & _). apply HL. left. reflexivity.
    + apply IH; [ exact Henv | exact (pre_app_r e0 _ _ _ _ Hpre) | exact Hold ].
  - (* Let_static *)
    intros bindings body Hcodes IH env Henv Hpre [Hbs Hold].
    set (gL := binding_let_bindings). set (gS := binding_sym_bindings).
    set (Ostat := fun (b : static_binding) (N : node) =>
                    match b with
                    | Bind_symbol s (Static_const_closure_of_set sl cs d v) =>
                      orig [] [(s, Static_const_closure_of_set sl cs d v)] N
                    | _ => False
                    end).
    set (Ocode := fun (b : static_binding) (N : node) =>
                    match b with
                    | Bind_code _ c => orig (code_let_bindings c)
                                         (code_sym_bindings c) N
                    | _ => False
                    end).
    apply (step_ok_ext e0 keep _ _
             (fun a =>
                traverse env body
                  (List.fold_left (fun a b => traverse_binding env b a)
                     bindings
                     (List.fold_left
                        (fun a b => traverse_static_binding env b a) bindings
                        (List.fold_left (fun a b => prepare_binding b a)
                           bindings a)))));
      [ reflexivity | ].
    cbn [let_bindings sym_bindings] in Hpre.
    change (flat_map binding_let_bindings bindings)
      with (flat_map gL bindings) in Hpre.
    change (flat_map binding_sym_bindings bindings)
      with (flat_map gS bindings) in Hpre.
    assert (Hpre_b : pre (flat_map gL bindings) (flat_map gS bindings))
      by exact (pre_app_l e0 _ _ _ _ Hpre).
    assert (Hsub_stat : ∀ (b : static_binding) (N : node),
               Ostat b N -> orig (gL b) (gS b) N).
    { intros b N Hb. destruct b as [ | | s0 [ | sl cs d v | | | | ]];
        try contradiction. exact Hb. }
    assert (Hsub_code : ∀ (b : static_binding) (N : node),
               Ocode b N -> orig (gL b) (gS b) N).
    { intros b N Hb. destruct b; try contradiction. exact Hb. }
    apply (step_ok_mono e0 keep
             (fun N => (((no_origin N ∨ (∃ b, b ∈ bindings ∧ Ostat b N))
                         ∨ (∃ b, b ∈ bindings ∧ Ocode b N))
                        ∨ origin_in body N)) _
             (fun N => (((no_origin N ∨ (∃ b, b ∈ bindings ∧ Ostat b N))
                         ∨ (∃ b, b ∈ bindings ∧ (fun _ _ => False) b N))
                        ∨ static_origin_in body N))).
    { intros N HN. unfold origin_in. cbn [let_bindings sym_bindings].
      change (flat_map binding_let_bindings bindings)
        with (flat_map gL bindings).
      change (flat_map binding_sym_bindings bindings)
        with (flat_map gS bindings).
      apply orig_app.
      destruct HN as [[[[] | (b & Hb & H)] | (b & Hb & H)] | H];
        [ left | left | right; exact H ];
        apply orig_flat_map; exists b; split; auto. }
    { intros N HN. unfold static_origin_in in HN.
      cbn [sym_bindings] in HN.
      destruct (static_orig_app _ _ N HN) as [H | H]; [ left | right; exact H ].
      destruct (static_orig_flat_map binding_sym_bindings bindings N H)
        as (b & Hb & Hs). left. right. exists b. split; [ exact Hb | ].
      destruct Hs as (t & sl & cs & d & v & -> & Hin).
      destruct b as [id c | id | s0 const]; cbn in Hin.
      - pose proof (proj1 (FlambdaMapProofs.every_in _ _ _) Hbs _ Hb)
          as [_ Hempty].
        rewrite Hempty in Hin. destruct Hin.
      - destruct Hin.
      - destruct Hin as [Heq | []]. injection Heq as Es Ec. subst s0 const.
        unfold Ostat. cbv beta iota. right.
        exists t, sl, cs, d, v. split; [ reflexivity | left; reflexivity ]. }
    destruct Hpre as (HL & HS & HnL & HnS).
    apply step_ok_seq; [ | apply step_ok_seq; [ | apply step_ok_seq | ] | ].
    + intros N H1 H2.
      apply (orig_disjoint _ _ _ _ N HnL HnS); [ | exact H2 ].
      apply orig_flat_map.
      destruct H1 as [[[] | (b & Hb & H)] | (b & Hb & H)]; exists b;
        split; auto.
    + intros N [[] | (b1 & Hb1 & H1)] (b2 & Hb2 & H2).
      destruct b1 as [ | | s0 [ | sl cs d v | | | | ]]; try contradiction.
      destruct b2 as [id c | | ]; try contradiction.
      destruct H1 as [(x & _ & _ & _ & _ & [])
                     | (t & sl1 & cs1 & d1 & v1 & E1 & Ht)].
      destruct H2 as [(x & _ & _ & _ & E & _)
                     | (t' & sl' & cs' & d' & v' & E & Ht')];
        rewrite E1 in E; [ discriminate E | injection E as E ].
      subst t'. rewrite map_app in HnS.
      apply (nodup_flat_map_distinct gS bindings _ _ t _ _
               (NoDup_app_remove_r _ _ HnS) Hb1 Hb2 ltac:(discriminate)
               Ht Ht').
    + intros N [] _.
    + apply step_ok_ns. intros a. apply ns_fold. intros a' b _.
      apply ns_prepare_binding.
    + apply (step_ok_fold e0 keep
               (fun a b => traverse_static_binding env b a) Ostat Ostat).
      * intros b Hb.
        destruct b as [id c | id | s0 const];
          [ | | destruct const as [id vs | sl cs d v | | | | ] ];
          try (apply (step_ok_mono e0 keep no_origin _ no_origin);
               [ intros N [] | intros N [] | ];
               apply step_ok_ns; intros a;
               apply ns_traverse_static_binding; intros; discriminate).
        -- pose proof (proj1 (FlambdaMapProofs.every_in _ _ _) Hbs _ Hb)
             as Hno.
           destruct (Hno id vs eq_refl).
        -- apply (static_closure_ok e0 keep Hnl Hns env s0 sl cs d v Henv).
           apply HS. apply in_or_app. left. apply in_flat_map.
           exists (Bind_symbol s0
             (Static_const_closure_of_set sl cs d v)).
           split; [ exact Hb | left; reflexivity ].
      * rewrite map_app in HnL, HnS.
        apply (disjoint_family_mono Ostat (fun b => orig (gL b) (gS b)));
          [ exact Hsub_stat | ].
        exact (disjoint_family_bindings gL gS bindings
                 (NoDup_app_remove_r _ _ HnL) (NoDup_app_remove_r _ _ HnS)).
    + apply (step_ok_fold e0 keep (fun a b => traverse_binding env b a) Ocode
               (fun _ _ => False)).
      * intros b Hb. destruct b as [id c | id | s0 const];
          [ | apply (step_ok_mono e0 keep no_origin _ no_origin);
              [ intros N [] | intros N [] | ];
              apply step_ok_ns; intros a; apply ns_refl
            | apply (step_ok_mono e0 keep no_origin _ no_origin);
              [ intros N [] | intros N [] | ];
              apply step_ok_ns; intros a; apply ns_refl ].
        pose proof (proj1 (FlambdaMapProofs.every_in _ _ _) Hbs _ Hb)
          as [Hno _].
        pose proof (Hcodes id c Hb) as Hc.
        destruct c as [params mc ret exn md cbody]. cbn in Hno, Hc.
        intros a Gs Hinv Hfresh. cbn [traverse_binding traverse_code].
        destruct (lookup id (acc_code_deps a)) as [cd | ].
        -- pose proof (inv_ns e0 keep Gs a _ Hinv
                         (ns_code_aliases mc params cd a)) as Hinv1.
           destruct (Hc (code_env env id ret exn cd)
                       (env_good_code_env e0 keep env id ret exn cd Henv)
                       (pre_flat_map e0 gL gS bindings _ Hpre_b Hb)
                       Hno _ Gs Hinv1 Hfresh)
             as (Gs' & Hinv' & HGs' & Hk' & _).
           exists Gs'. split; [ exact Hinv' | split; [ exact HGs' | ] ].
           split; [ exact Hk' | intros N [] ].
        -- exists Gs. split; [ exact Hinv | split; [ auto | ] ].
           split; [ auto | intros N [] ].
      * rewrite map_app in HnL, HnS.
        apply (disjoint_family_mono Ocode (fun b => orig (gL b) (gS b)));
          [ exact Hsub_code | ].
        exact (disjoint_family_bindings gL gS bindings
                 (NoDup_app_remove_r _ _ HnL) (NoDup_app_remove_r _ _ HnS)).
    + apply IH; [ exact Henv | | exact Hold ].
      exact (pre_app_r e0 _ _ _ _ (conj HL (conj HS (conj HnL HnS)))).
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH env Henv Hpre [Hh Hold].
    destruct h as [params md handler]. cbn in IHh, Hh.
    apply (step_ok_ext e0 keep _ _
             (fun a => traverse (add_cont env k (handler_params
                                                  (Handler params md handler)))
                         body
                         (exn_handler_deps env (Handler params md handler)
                            (traverse env handler a))));
      [ reflexivity | ].
    cbn [let_bindings sym_bindings handler_let_bindings
         handler_sym_bindings] in Hpre.
    apply (step_ok_mono e0 keep
             (fun N => (origin_in handler N ∨ no_origin N) ∨ origin_in body N) _
             (fun N => (static_origin_in handler N ∨ no_origin N)
                       ∨ static_origin_in body N)).
    { intros N HN. unfold origin_in. cbn [let_bindings sym_bindings
        handler_let_bindings handler_sym_bindings].
      apply orig_app.
      destruct HN as [[H | []] | H]; [ left | right ]; exact H. }
    { intros N HN. unfold static_origin_in in HN |- *.
      cbn [sym_bindings handler_sym_bindings] in HN.
      destruct HN as (t & sl & cs & dc & v & -> & Hin).
      apply in_app_or in Hin as [Hin | Hin]; [ left; left | right ];
        exists t, sl, cs, dc, v; auto. }
    apply step_ok_seq; [ | apply step_ok_seq | ].
    + intros N [H1 | []] H2. destruct Hpre as (_ & _ & HnL & HnS).
      exact (orig_disjoint _ _ _ _ N HnL HnS H1 H2).
    + intros N H1 [].
    + apply IHh; [ exact Henv | exact (pre_app_l e0 _ _ _ _ Hpre) | exact Hh ].
    + apply step_ok_ns. intros a. apply ns_exn_handler_deps.
    + apply IH; [ exact (env_good_add_cont e0 keep env k _ Henv)
                | exact (pre_app_r e0 _ _ _ _ Hpre) | exact Hold ].
  - (* Let_cont_recursive *)
    intros handlers body IHh IH env Henv Hpre [Hhs Hold].
    set (env' := add_rec_conts env handlers).
    set (gL := fun kh : continuation * continuation_handler =>
                 let '(_, h) := kh in handler_let_bindings h).
    set (gS := fun kh : continuation * continuation_handler =>
                 let '(_, h) := kh in handler_sym_bindings h).
    apply (step_ok_ext e0 keep _ _
             (fun a => traverse env' body
                         (List.fold_left
                            (fun a '(_, h) => traverse_handler env' h a)
                            handlers a)));
      [ reflexivity | ].
    cbn [let_bindings sym_bindings] in Hpre.
    change (flat_map (fun '(_, h) => handler_let_bindings h) handlers)
      with (flat_map gL handlers) in Hpre.
    change (flat_map (fun '(_, h) => handler_sym_bindings h) handlers)
      with (flat_map gS handlers) in Hpre.
    apply (step_ok_mono e0 keep
             (fun N => (∃ kh, kh ∈ handlers ∧ orig (gL kh) (gS kh) N)
                       ∨ origin_in body N) _
             (fun N => (∃ kh, kh ∈ handlers ∧ static_orig (gS kh) N)
                       ∨ static_origin_in body N)).
    { intros N HN. unfold origin_in. cbn [let_bindings sym_bindings].
      change (flat_map (fun '(_, h) => handler_let_bindings h) handlers)
        with (flat_map gL handlers).
      change (flat_map (fun '(_, h) => handler_sym_bindings h) handlers)
        with (flat_map gS handlers).
      apply orig_app. destruct HN as [HN | HN]; [ left | right; exact HN ].
      apply orig_flat_map. exact HN. }
    { intros N HN. unfold static_origin_in in HN.
      cbn [sym_bindings] in HN.
      change (flat_map (fun '(_, h) => handler_sym_bindings h) handlers)
        with (flat_map gS handlers) in HN.
      destruct (static_orig_app _ _ N HN) as [H | H]; [ left | right; exact H ].
      exact (static_orig_flat_map gS handlers N H). }
    apply step_ok_seq.
    + intros N H1 H2. destruct Hpre as (_ & _ & HnL & HnS).
      apply (orig_disjoint _ _ _ _ N HnL HnS); [ | exact H2 ].
      apply orig_flat_map. exact H1.
    + apply (step_ok_fold e0 keep
               (fun a kh => let '(_, h) := kh in traverse_handler env' h a)
               (fun kh => orig (gL kh) (gS kh))
               (fun kh => static_orig (gS kh))).
      * intros [k h] Hkh. destruct h as [params md handler].
        pose proof (IHh k _ Hkh) as Hh. cbn in Hh.
        apply (step_ok_ext e0 keep _ _ (traverse env' handler));
          [ reflexivity | ].
        apply Hh.
        -- exact (env_good_add_rec_conts e0 keep env handlers Henv).
        -- exact (pre_flat_map e0 gL gS handlers (k, Handler params md handler)
                    (pre_app_l e0 _ _ _ _ Hpre) Hkh).
        -- exact (proj1 (FlambdaMapProofs.every_in _ _ _) Hhs _ Hkh).
      * destruct Hpre as (_ & _ & HnL & HnS). rewrite map_app in HnL, HnS.
        exact (disjoint_family_bindings gL gS handlers
                 (NoDup_app_remove_r _ _ HnL) (NoDup_app_remove_r _ _ HnS)).
    + apply IH; [ exact (env_good_add_rec_conts e0 keep env handlers Henv)
                | exact (pre_app_r e0 _ _ _ _ Hpre) | exact Hold ].
  - (* Apply *)
    intros ap env _ _ _.
    apply (step_ok_mono e0 keep no_origin _ no_origin);
      [ intros N [] | intros N (t & _ & _ & _ & _ & _ & []) | ].
    apply step_ok_ns. intros a. apply ns_traverse_apply.
  - (* Apply_cont *)
    intros ac env _ _ _.
    apply (step_ok_mono e0 keep no_origin _ no_origin);
      [ intros N [] | intros N (t & _ & _ & _ & _ & _ & []) | ].
    apply step_ok_ns. intros a. apply ns_apply_cont_deps.
  - (* Switch *)
    intros scrutinee arms env _ _ _.
    apply (step_ok_mono e0 keep no_origin _ no_origin);
      [ intros N [] | intros N (t & _ & _ & _ & _ & _ & []) | ].
    apply step_ok_ns. intros a. apply ns_traverse_switch.
  - (* Invalid *)
    intros env _ _ _.
    apply (step_ok_mono e0 keep no_origin _ no_origin);
      [ intros N [] | intros N (t & _ & _ & _ & _ & _ & []) | ].
    apply step_ok_ns. intros a. apply ns_refl.
  - (* Handler *)
    intros params metadata handler IH. exact IH.
Qed.

End Traverse.

(* ======================================================================== *)
(* The unit                                                                 *)
(* ======================================================================== *)

Lemma constructor_of_list (l : list fact) (N b : node) (f : field) :
  g_constructor (graph_of_facts (facts_of_list l)) N f b
  <-> Fact_constructor N f b ∈ l.
Proof.
  cbn [graph_of_facts facts_of_list g_constructor f_constructor].
  rewrite in_flat_map. split.
  - intros (fct & Hin & H). destruct fct; cbn in H; try contradiction.
    destruct H as [[= -> -> ->] | []]. exact Hin.
  - intros Hin. exists (Fact_constructor N f b). split; [ exact Hin | ].
    left. reflexivity.
Qed.

(* The traversal of a unit: its slot facts are those of groups, and each
   of its static closures has a group. *)
Lemma traverse_unit_groups (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  NoDup (map fst (let_bindings e)) -> NoDup (map fst (sym_bindings e)) ->
  no_old_sets e ->
  ∃ (Gs : list sgroup) (a : acc),
    traverse_unit flag u e = facts_of_list (acc_facts a)
    ∧ inv e (unit_kept e) Gs a
    ∧ ∀ N : node,
        static_origin_in e N ->
        ∃ γ : sgroup, γ ∈ Gs ∧ sg_origin γ = N.
Proof.
  intros Hnl Hns Hold. unfold traverse_unit.
  set (a0 := add_fact (Fact_any_source Node_all_constants) empty_acc).
  destruct (fresh a0) as [r a1] eqn:E1.
  destruct (fresh a1) as [x a2] eqn:E2.
  set (a3 := add_fact (Fact_any_usage x) a2).
  set (env := {| env_conts := [(unit_exn u, [x]); (unit_return u, [r])];
                 env_current_code_id := None;
                 env_should_preserve_direct_calls :=
                   should_preserve_of_flag flag;
                 env_value_slots_to_keep := projected_value_slots e;
                 env_unit_code_ids := static_names all_code_ids_of_binding e;
                 env_set_closures := set_closures e |}).
  assert (Hinv3 : inv e (unit_kept e) [] a3).
  { assert (Hf : acc_facts a3
                 = [Fact_any_usage x; Fact_any_source Node_all_constants]).
    { unfold a3. cbn [add_fact add_facts acc_facts app].
      assert (Hf2 : acc_facts a2 = acc_facts a1)
        by (change a2 with (snd (x, a2)); rewrite <- E2; reflexivity).
      assert (Hf1 : acc_facts a1 = acc_facts a0)
        by (change a1 with (snd (r, a1)); rewrite <- E1; reflexivity).
      rewrite Hf2, Hf1. reflexivity. }
    split; [ | split; [ | split; [ | split ] ] ];
      try (intros; contradiction).
    intros fct Hin Hs. rewrite Hf in Hin.
    destruct Hin as [<- | [<- | []]]; destruct Hs. }
  destruct (proj1 (traverse_ok e (unit_kept e) Hnl Hns) e env
              (conj eq_refl (fun w => eq_refl))
              (conj (fun b Hb => Hb) (conj (fun b Hb => Hb) (conj Hnl Hns)))
              Hold a3 [] Hinv3 (fun γ Hγ => match Hγ with end))
    as (Gs & Hinv4 & _ & _ & Hpres).
  exists Gs. eexists. split; [ reflexivity | split; [ | exact Hpres ] ].
  apply (inv_ns e (unit_kept e) Gs _ _ Hinv4).
  eapply ns_trans; [ | apply ns_resolve ]. apply ns_add_fact. cbn. auto.
Qed.

Section Final.

Variable e : expr.
Abbreviation keep := (unit_kept e).
Hypothesis Hnl : NoDup (map fst (let_bindings e)).
Hypothesis Hns : NoDup (map fst (sym_bindings e)).
Variable Gs : list sgroup.
Variable a : acc.
Hypothesis Hinv : inv e keep Gs a.
Abbreviation G := (graph_of_facts (facts_of_list (acc_facts a))).

(* The facts at an allocation of a group are those of the group. *)
Lemma facts_at (γ : sgroup) (N : node) :
  γ ∈ Gs -> N ∈ sg_bases γ ->
  (∀ (f : function_slot) (b : node),
     g_constructor G N (Field_function_slot f) b <-> (f, b) ∈ sg_fslots γ)
  ∧ ∀ (w : value_slot) (b : node),
      g_constructor G N (Field_value_slot w) b <-> (w, b) ∈ sg_vslots γ.
Proof.
  intros Hγ HN. destruct Hinv as (HA & HB & HC & _).
  split; intros k b; rewrite constructor_of_list; split.
  - intros Hin. destruct (HA _ Hin eq_refl) as (γ' & Hγ' & N' & HN' & Hf).
    destruct Hf as [(f' & b' & [= <- <- <-] & Hf) | (w & b' & E & _)];
      [ | discriminate E ].
    rewrite (HC γ γ' N Hγ Hγ' HN HN'). exact Hf.
  - intros Hf. apply (HB γ _ Hγ). exists N. split; [ exact HN | left ].
    exists k, b. auto.
  - intros Hin. destruct (HA _ Hin eq_refl) as (γ' & Hγ' & N' & HN' & Hf).
    destruct Hf as [(f' & b' & E & _) | (w & b' & [= <- <- <-] & Hf)];
      [ discriminate E | ].
    rewrite (HC γ γ' N Hγ Hγ' HN HN'). exact Hf.
  - intros Hf. apply (HB γ _ Hγ). exists N. split; [ exact HN | right ].
    exists k, b. auto.
Qed.

(* A slot fact at [N] is of a group of which [N] is an allocation. *)
Lemma group_at (N b : node) (f : field) :
  g_constructor G N f b -> is_slot f = true ->
  ∃ γ : sgroup, γ ∈ Gs ∧ N ∈ sg_bases γ.
Proof.
  intros Hc Hs. apply constructor_of_list in Hc.
  destruct Hinv as (HA & _).
  destruct (HA _ Hc Hs) as (γ & Hγ & N' & HN' & Hf). exists γ.
  split; [ exact Hγ | ].
  destruct Hf as [(f' & b' & [= <- _ _] & _) | (w & b' & [= <- _ _] & _)];
    exact HN'.
Qed.

Lemma in_is_bound {A : Type} (k : positive) (v : A) (l : finite_map A) :
  (k, v) ∈ l -> is_bound k l = true.
Proof.
  intros Hin. rewrite is_bound_among. apply among_In.
  exact (in_map fst l (k, v) Hin).
Qed.

(* The group of an allocation that is the variable of a [Let] of a
   closure is that of the [Let]. *)
Lemma closure_var_group (γ : sgroup) (x : variable) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) :
  γ ∈ Gs -> Node_variable x ∈ sg_bases γ ->
  (x, Closure_of_set slot decls vs) ∈ let_bindings e ->
  (∀ g : function_slot, is_bound g (sg_fslots γ) = is_bound g decls)
  ∧ ∀ (f : function_slot) (M : node), (f, M) ∈ sg_fslots γ -> M ∈ sg_bases γ.
Proof.
  intros Hγ Hx Hlet.
  destruct (proj1 (proj2 (proj2 (proj2 Hinv))) γ Hγ)
    as (_ & _ & _ & [(x' & slot' & decls' & vs' & _ & Hlet' & _ & Hdom & Htgt
                      & Hb)
                    | (t & _ & _ & _ & _ & _ & _ & Hb & _)]).
  - destruct (Hb _ Hx) as [E | [(y & s0 & o & E & Hy) | (k & E)]];
      [ | | discriminate E ]; injection E as E; subst.
    + pose proof (nodup_same _ _ _ _ Hnl Hlet Hlet') as E.
      injection E as _ <- _. split; assumption.
    + pose proof (nodup_same _ _ _ _ Hnl Hlet Hy) as E. discriminate E.
  - rewrite Hb in Hx. destruct Hx as [E | []]. discriminate E.
Qed.

Lemma group_nodup (γ : sgroup) :
  γ ∈ Gs -> NoDup (map fst (sg_fslots γ)) ∧ NoDup (map fst (sg_vslots γ))
  ∧ ∀ (w : value_slot) (b : node), (w, b) ∈ sg_vslots γ -> keep w = true.
Proof.
  intros Hγ.
  destruct (proj1 (proj2 (proj2 (proj2 Hinv))) γ Hγ) as (H1 & H2 & H3 & _).
  auto.
Qed.

(* The allocation of the first closure of a [Let] holds exactly the facts
   of its set. *)
Lemma closure_node_exact (x : variable) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) :
  (x, Closure_of_set slot decls vs) ∈ let_bindings e ->
  RP.node_exact G keep (Node_variable x) decls.
Proof.
  intros Hlet. split; [ | split; [ | split ] ].
  - intros g b Hc. destruct (group_at _ _ _ Hc eq_refl) as (γ & Hγ & Hx).
    rewrite <- (proj1 (closure_var_group γ x slot decls vs Hγ Hx Hlet) g).
    exact (in_is_bound g b _ (proj1 (proj1 (facts_at γ _ Hγ Hx) g b) Hc)).
  - intros g b b' Hc Hc'.
    destruct (group_at _ _ _ Hc eq_refl) as (γ & Hγ & Hx).
    exact (nodup_same _ g b' b (proj1 (group_nodup γ Hγ))
             (proj1 (proj1 (facts_at γ _ Hγ Hx) g b') Hc')
             (proj1 (proj1 (facts_at γ _ Hγ Hx) g b) Hc)).
  - intros g M Hc. destruct (group_at _ _ _ Hc eq_refl) as (γ & Hγ & Hx).
    pose proof (proj1 (proj1 (facts_at γ _ Hγ Hx) g M) Hc) as HM.
    pose proof (proj2 (closure_var_group γ x slot decls vs Hγ Hx Hlet) g M HM)
      as HMb.
    split; intros k b.
    + rewrite (proj1 (facts_at γ M Hγ HMb) k b),
        (proj1 (facts_at γ _ Hγ Hx) k b).
      reflexivity.
    + rewrite (proj2 (facts_at γ M Hγ HMb) k b),
        (proj2 (facts_at γ _ Hγ Hx) k b).
      reflexivity.
  - intros w b Hc. destruct (group_at _ _ _ Hc eq_refl) as (γ & Hγ & Hx).
    pose proof (proj1 (proj2 (facts_at γ _ Hγ Hx) w b) Hc) as Hw.
    destruct (group_nodup γ Hγ) as (_ & Hnd & Hk).
    split; [ exact (Hk w b Hw) | ].
    intros b' Hc'.
    exact (nodup_same _ w b' b Hnd (proj1 (proj2 (facts_at γ _ Hγ Hx) w b') Hc')
             Hw).
Qed.

(* The group of the symbol of a static closure is that of its binding. *)
Lemma static_symbol_group (γ : sgroup) (t : symbol) (slot : function_slot)
    (cs : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) :
  γ ∈ Gs -> Node_symbol t ∈ sg_bases γ ->
  (t, Static_const_closure_of_set slot cs decls vs) ∈ sym_bindings e ->
  sg_fslots γ = map (fun '(f, sym) => (f, Node_symbol sym)) (compact cs)
  ∧ sg_vslots γ
    = map (fun '(w, s) => (w, node_of_simple s))
        (filter (fun '(w, _) => keep w) (compact vs)).
Proof.
  intros Hγ Ht Hs.
  destruct (proj1 (proj2 (proj2 (proj2 Hinv))) γ Hγ)
    as (_ & _ & _ & [(x & _ & _ & _ & _ & _ & _ & _ & _ & Hb)
                    | (t' & slot' & cs' & decls' & vs' & _ & Hs' & Hb & Hf
                       & Hv)]).
  - destruct (Hb _ Ht) as [E | [(y & _ & _ & E & _) | (k & E)]];
      discriminate E.
  - rewrite Hb in Ht. destruct Ht as [E | []]. injection E as E. subst t'.
    pose proof (nodup_same _ _ _ _ Hns Hs Hs') as E.
    injection E as _ <- _ <-. split; assumption.
Qed.

(* The symbol of a static closure, whose set is bound whole, holds exactly
   the facts of its set, if every static closure has a group. *)
Lemma static_node_exact (t : symbol) (slot : function_slot)
    (cs : list (function_slot * symbol))
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) :
  (t, Static_const_closure_of_set slot cs decls vs) ∈ sym_bindings e ->
  map fst cs = map fst decls ->
  (∀ (g : function_slot) (sym : symbol),
     (g, sym) ∈ cs ->
     (sym, Static_const_closure_of_set g cs decls vs) ∈ sym_bindings e) ->
  (∀ N : node,
     static_origin_in e N -> ∃ γ : sgroup, γ ∈ Gs ∧ sg_origin γ = N) ->
  RP.node_exact G keep (Node_symbol t) decls.
Proof.
  intros Hs Hdom Hwhole Hpres.
  assert (Hbase : ∀ (sym : symbol) (g : function_slot),
             (sym, Static_const_closure_of_set g cs decls vs)
             ∈ sym_bindings e ->
             ∃ γ : sgroup,
               γ ∈ Gs ∧ Node_symbol sym ∈ sg_bases γ
               ∧ sg_fslots γ
                 = map (fun '(f, sym) => (f, Node_symbol sym)) (compact cs)
               ∧ sg_vslots γ
                 = map (fun '(w, s) => (w, node_of_simple s))
                     (filter (fun '(w, _) => keep w) (compact vs))).
  { intros sym g Hsym.
    destruct (Hpres (Node_symbol sym)) as (γ & Hγ & Ho).
    { exists sym, g, cs, decls, vs. auto. }
    assert (Hb : Node_symbol sym ∈ sg_bases γ).
    { destruct (proj1 (proj2 (proj2 (proj2 Hinv))) γ Hγ)
        as (_ & _ & _ & [(x & _ & _ & _ & Ox & _) | (t' & _ & _ & _ & _ & Ot
                                                      & _ & Hb & _)]).
      - rewrite Ox in Ho. discriminate Ho.
      - rewrite Ot in Ho. injection Ho as <-. rewrite Hb. left.
        reflexivity. }
    exists γ. split; [ exact Hγ | split; [ exact Hb | ] ].
    exact (static_symbol_group γ sym g cs decls vs Hγ Hb Hsym). }
  destruct (Hbase t slot Hs) as (γt & Hγt & Ht & Hft & Hvt).
  assert (Hat : ∀ (N b : node) (f : field),
             g_constructor G (Node_symbol t) f b -> is_slot f = true ->
             (∀ γ : sgroup, γ ∈ Gs -> Node_symbol t ∈ sg_bases γ -> γ = γt)).
  { intros N b f _ _ γ Hγ Hb.
    exact (proj1 (proj2 (proj2 Hinv)) γ γt _ Hγ Hγt Hb Ht). }
  split; [ | split; [ | split ] ].
  - intros g b Hc.
    pose proof (proj1 (proj1 (facts_at γt _ Hγt Ht) g b) Hc) as Hin.
    rewrite Hft in Hin. apply in_map_iff in Hin as ([g' sym] & Heq & Hin).
    injection Heq as -> _.
    assert (E : is_bound g cs = is_bound g decls).
    { rewrite !is_bound_among. f_equal. exact Hdom. }
    rewrite <- E, <- is_bound_compact. exact (in_is_bound g sym _ Hin).
  - intros g b b' Hc Hc'.
    exact (nodup_same _ g b' b (proj1 (group_nodup γt Hγt))
             (proj1 (proj1 (facts_at γt _ Hγt Ht) g b') Hc')
             (proj1 (proj1 (facts_at γt _ Hγt Ht) g b) Hc)).
  - intros g M Hc.
    pose proof (proj1 (proj1 (facts_at γt _ Hγt Ht) g M) Hc) as Hin.
    rewrite Hft in Hin. apply in_map_iff in Hin as ([g' sym] & Heq & Hin).
    injection Heq as E1 E2. subst g' M.
    apply in_compact in Hin.
    destruct (Hbase sym g (Hwhole g sym (lookup_in _ _ _ Hin)))
      as (γM & HγM & HM & HfM & HvM).
    split; intros k b.
    + rewrite (proj1 (facts_at γM _ HγM HM) k b),
        (proj1 (facts_at γt _ Hγt Ht) k b), HfM, Hft.
      reflexivity.
    + rewrite (proj2 (facts_at γM _ HγM HM) k b),
        (proj2 (facts_at γt _ Hγt Ht) k b), HvM, Hvt.
      reflexivity.
  - intros w b Hc.
    pose proof (proj1 (proj2 (facts_at γt _ Hγt Ht) w b) Hc) as Hw.
    destruct (group_nodup γt Hγt) as (_ & Hnd & Hk).
    split; [ exact (Hk w b Hw) | ].
    intros b' Hc'.
    exact (nodup_same _ w b' b Hnd
             (proj1 (proj2 (facts_at γt _ Hγt Ht) w b') Hc') Hw).
Qed.

End Final.

(* The predicate of the unit, from what is asked of each of its bindings. *)
Lemma closures_exact_struct (G : graph) (keep : value_slot -> bool) :
  (∀ e : expr,
     (∀ (x : variable) (d : named),
        (x, d) ∈ let_bindings e -> RP.named_exact G keep x d) ->
     (∀ (t : symbol) (c : static_const),
        (t, c) ∈ sym_bindings e -> RP.binding_exact G keep (Bind_symbol t c)) ->
     RP.closures_exact G keep e)
  ∧ ∀ h : continuation_handler,
      match h with
      | Handler _ _ handler =>
        (∀ (x : variable) (d : named),
           (x, d) ∈ let_bindings handler -> RP.named_exact G keep x d) ->
        (∀ (t : symbol) (c : static_const),
           (t, c) ∈ sym_bindings handler ->
           RP.binding_exact G keep (Bind_symbol t c)) ->
        RP.closures_exact G keep handler
      end.
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind; cbn [RP.closures_exact
    let_bindings sym_bindings].
  - intros x k d body IH Hl Hs.
    split; [ apply Hl; left; reflexivity | ].
    apply IH; [ intros y d' H; apply Hl; right; exact H | exact Hs ].
  - intros bindings body Hcodes IH Hl Hs. split.
    + apply (FlambdaMapProofs.every_in _ _ _). intros b Hb.
      destruct b as [id c | id | t c]; cbn [RP.binding_exact].
      * destruct c as [params mc ret exn md cbody].
        cbn [RP.code_exact]. pose proof (Hcodes id _ Hb) as Hc. cbn in Hc.
        apply Hc.
        -- intros y d H. apply Hl. apply in_or_app. left.
           apply in_flat_map. exists (Bind_code id (Code params mc ret exn md
             cbody)). split; [ exact Hb | exact H ].
        -- intros t c H. apply Hs. apply in_or_app. left.
           apply in_flat_map. exists (Bind_code id (Code params mc ret exn md
             cbody)). split; [ exact Hb | exact H ].
      * exact I.
      * apply Hs. apply in_or_app. left. apply in_flat_map.
        exists (Bind_symbol t c). split; [ exact Hb | left; reflexivity ].
    + apply IH.
      * intros y d H. apply Hl. apply in_or_app. right. exact H.
      * intros t c H. apply Hs. apply in_or_app. right. exact H.
  - intros k h body IHh IH Hl Hs. destruct h as [params md handler].
    cbn [RP.handler_exact handler_let_bindings handler_sym_bindings] in *.
    split.
    + apply IHh.
      * intros y d H. apply Hl. apply in_or_app. left. exact H.
      * intros t c H. apply Hs. apply in_or_app. left. exact H.
    + apply IH.
      * intros y d H. apply Hl. apply in_or_app. right. exact H.
      * intros t c H. apply Hs. apply in_or_app. right. exact H.
  - intros handlers body IHh IH Hl Hs. split.
    + apply (FlambdaMapProofs.every_in _ _ _). intros [k h] Hkh.
      destruct h as [params md handler]. cbn [RP.handler_exact].
      pose proof (IHh k _ Hkh) as Hh. cbn in Hh. apply Hh.
      * intros y d H. apply Hl. apply in_or_app. left. apply in_flat_map.
        exists (k, Handler params md handler). split; [ exact Hkh | exact H ].
      * intros t c H. apply Hs. apply in_or_app. left. apply in_flat_map.
        exists (k, Handler params md handler). split; [ exact Hkh | exact H ].
    + apply IH.
      * intros y d H. apply Hl. apply in_or_app. right. exact H.
      * intros t c H. apply Hs. apply in_or_app. right. exact H.
  - intros. exact I.
  - intros. exact I.
  - intros. exact I.
  - intros. exact I.
  - intros params metadata handler IH. exact IH.
Qed.

(* A static set of a unit whose sets are well formed is bound whole: a
   symbol for each function of the set, each bound to the closure of the
   set at its slot. *)
Lemma static_sets_whole :
  (∀ e : expr,
     sets_well_formed e = true ->
     ∀ (t : symbol) (slot : function_slot)
       (cs : list (function_slot * symbol))
       (decls : list (function_slot * code_id))
       (vs : list (value_slot * simple)),
       (t, Static_const_closure_of_set slot cs decls vs) ∈ sym_bindings e ->
       map fst cs = map fst decls
       ∧ ∀ (g : function_slot) (sym : symbol),
           (g, sym) ∈ cs ->
           (sym, Static_const_closure_of_set g cs decls vs) ∈ sym_bindings e)
  ∧ ∀ h : continuation_handler,
      match h with
      | Handler _ _ handler =>
        sets_well_formed handler = true ->
        ∀ (t : symbol) (slot : function_slot)
          (cs : list (function_slot * symbol))
          (decls : list (function_slot * code_id))
          (vs : list (value_slot * simple)),
          (t, Static_const_closure_of_set slot cs decls vs)
          ∈ sym_bindings handler ->
          map fst cs = map fst decls
          ∧ ∀ (g : function_slot) (sym : symbol),
              (g, sym) ∈ cs ->
              (sym, Static_const_closure_of_set g cs decls vs)
              ∈ sym_bindings handler
      end.
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [sets_well_formed sym_bindings].
  - intros x k d body IH Hwf. apply andb_true_iff in Hwf as [_ Hwf].
    exact (IH Hwf).
  - intros bindings body Hcodes IH Hwf t slot cs decls vs Hin.
    apply andb_true_iff in Hwf as [Hwf Hbody].
    apply andb_true_iff in Hwf as [Hwf Hcodes_wf].
    apply andb_true_iff in Hwf as [Hwhole _].
    rewrite forallb_forall in Hwhole, Hcodes_wf.
    apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as (b & Hb & Hin).
      destruct b as [id c | id | t' c']; cbn in Hin.
      * destruct c as [params mc ret exn md cbody]. cbn in Hin.
        pose proof (Hcodes_wf _ Hb) as Hc. cbn in Hc.
        destruct (Hcodes id _ Hb Hc t slot cs decls vs Hin) as [Hdom Hsym].
        split; [ exact Hdom | ]. intros g sym Hgs. apply in_or_app. left.
        apply in_flat_map.
        exists (Bind_code id (Code params mc ret exn md cbody)).
        split; [ exact Hb | exact (Hsym g sym Hgs) ].
      * destruct Hin.
      * destruct Hin as [Heq | []]. injection Heq as E1 E2. subst t' c'.
        pose proof (Hwhole _ Hb) as Hw. cbn [static_set_whole] in Hw.
        apply andb_true_iff in Hw as [Hw Hall].
        apply andb_true_iff in Hw as [Hw _].
        apply andb_true_iff in Hw as [_ Hdom].
        destruct (list_eq_dec Pos.eq_dec (map fst cs) (map fst decls))
          as [Hdom' | ]; [ | discriminate Hdom ].
        split; [ exact Hdom' | ]. intros g sym Hgs.
        rewrite forallb_forall in Hall. pose proof (Hall _ Hgs) as Hex.
        cbn in Hex. apply existsb_exists in Hex as (b' & Hb' & Hbt).
        destruct b' as [ | | sym' c']; try discriminate Hbt.
        cbn [binds_symbol_to] in Hbt. apply andb_true_iff in Hbt as [Hs Hc].
        apply Pos.eqb_eq in Hs. subst sym'.
        destruct (static_const_eq_dec _ c') as [<- | ]; [ | discriminate Hc ].
        apply in_or_app. left. apply in_flat_map.
        exists (Bind_symbol sym (Static_const_closure_of_set g cs decls vs)).
        split; [ exact Hb' | left; reflexivity ].
    + destruct (IH Hbody t slot cs decls vs Hin) as [Hdom Hsym].
      split; [ exact Hdom | ]. intros g sym Hgs. apply in_or_app. right.
      exact (Hsym g sym Hgs).
  - intros k h body IHh IH Hwf t slot cs decls vs Hin.
    destruct h as [params md handler].
    cbn [handler_sets_well_formed handler_sym_bindings] in Hwf, Hin.
    apply andb_true_iff in Hwf as [Hh Hb].
    apply in_app_or in Hin as [Hin | Hin].
    + destruct (IHh Hh t slot cs decls vs Hin) as [Hdom Hsym].
      split; [ exact Hdom | ]. intros g sym Hgs. apply in_or_app. left.
      exact (Hsym g sym Hgs).
    + destruct (IH Hb t slot cs decls vs Hin) as [Hdom Hsym].
      split; [ exact Hdom | ]. intros g sym Hgs. apply in_or_app. right.
      exact (Hsym g sym Hgs).
  - intros handlers body IHh IH Hwf t slot cs decls vs Hin.
    apply andb_true_iff in Hwf as [Hhs Hb]. rewrite forallb_forall in Hhs.
    apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as ([k h] & Hkh & Hin).
      destruct h as [params md handler]. cbn in Hin.
      pose proof (Hhs _ Hkh) as Hh. cbn in Hh.
      destruct (IHh k _ Hkh Hh t slot cs decls vs Hin) as [Hdom Hsym].
      split; [ exact Hdom | ]. intros g sym Hgs. apply in_or_app. left.
      apply in_flat_map. exists (k, Handler params md handler).
      split; [ exact Hkh | exact (Hsym g sym Hgs) ].
    + destruct (IH Hb t slot cs decls vs Hin) as [Hdom Hsym].
      split; [ exact Hdom | ]. intros g sym Hgs. apply in_or_app. right.
      exact (Hsym g sym Hgs).
  - intros ap _ t slot cs decls vs [].
  - intros ac _ t slot cs decls vs [].
  - intros scrutinee arms _ t slot cs decls vs [].
  - intros _ t slot cs decls vs [].
  - intros params metadata handler IH. exact IH.
Qed.

(* What binds nothing static binds no symbol. *)
Lemma no_binds_no_syms :
  (∀ e : expr, binds_statics e = false -> sym_bindings e = [])
  ∧ ∀ h : continuation_handler,
      match h with
      | Handler _ _ handler =>
        binds_statics handler = false -> sym_bindings handler = []
      end.
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [binds_statics sym_bindings handler_binds_statics
         handler_sym_bindings].
  - intros x k d body IH H. exact (IH H).
  - intros bindings body _ IH H. apply orb_false_iff in H as [Hb H].
    rewrite (IH H), app_nil_r.
    induction bindings as [ | b bs IHb]; [ reflexivity | ].
    cbn [existsb] in Hb. apply orb_false_iff in Hb as [Hb Hbs].
    destruct b; cbn in Hb; try discriminate Hb. exact (IHb Hbs).
  - intros k h body IHh IH H. destruct h as [params md handler].
    apply orb_false_iff in H as [Hh H]. cbn in Hh, IHh |- *.
    rewrite (IHh Hh), (IH H). reflexivity.
  - intros handlers body IHh IH H. apply orb_false_iff in H as [Hh H].
    rewrite (IH H), app_nil_r.
    induction handlers as [ | [k h] hs IHhs]; [ reflexivity | ].
    cbn [existsb] in Hh. apply orb_false_iff in Hh as [Hh Hhs].
    destruct h as [params md handler].
    cbn [flat_map handler_sym_bindings handler_binds_statics] in Hh |- *.
    rewrite (IHh k _ (or_introl eq_refl) Hh). cbn [app].
    apply IHhs; [ | exact Hhs ].
    intros k' h' Hin. exact (IHh k' h' (or_intror Hin)).
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH. exact IH.
Qed.

(* The units of the fragment have no old form of set of closures, nor a
   symbol bound in code. *)
Lemma fragment_no_old_sets (heap : bool) (codes : list (code_id * code))
    (symbols : list symbol) :
  (∀ e : expr, flow_fragment heap codes symbols e = true -> no_old_sets e)
  ∧ ∀ h : continuation_handler,
      match h with
      | Handler _ _ handler =>
        flow_fragment heap codes symbols handler = true ->
        no_old_sets handler
      end.
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [flow_fragment no_old_sets handler_flow_fragment].
  - intros x k d body IH H. apply andb_true_iff in H as [Hd H].
    split; [ | exact (IH H) ].
    intros id vs ->. discriminate Hd.
  - intros bindings body Hcodes IH H. apply andb_true_iff in H as [Hb H].
    split; [ | exact (IH H) ].
    apply (FlambdaMapProofs.every_in _ _ _). intros b Hin.
    rewrite forallb_forall in Hb. pose proof (Hb b Hin) as Hbf.
    destruct b as [id c | id | s const]; cbn [binding_no_old_sets].
    + destruct c as [params mc ret exn md cbody].
      cbn [binding_flow_fragment code_flow_fragment] in Hbf.
      apply andb_true_iff in Hbf as [Hbf Hc].
      apply andb_true_iff in Hbf as [Hbf _].
      apply andb_true_iff in Hbf as [_ Hbs]. apply negb_true_iff in Hbs.
      cbn [code_no_old_sets code_sym_bindings].
      split; [ exact (Hcodes id _ Hin Hc) | ].
      exact (proj1 no_binds_no_syms _ Hbs).
    + exact I.
    + intros id vs ->. cbn in Hbf. discriminate Hbf.
  - intros k h body IHh IH H. destruct h as [params md handler].
    apply andb_true_iff in H as [Hh H]. cbn in Hh, IHh |- *.
    split; [ exact (IHh Hh) | exact (IH H) ].
  - intros handlers body IHh IH H. apply andb_true_iff in H as [Hh H].
    split; [ | exact (IH H) ].
    apply (FlambdaMapProofs.every_in _ _ _). intros [k h] Hin.
    rewrite forallb_forall in Hh. pose proof (Hh _ Hin) as Hhf.
    destruct h as [params md handler]. cbn in Hhf |- *.
    exact (IHh k _ Hin Hhf).
  - intros. exact I.
  - intros. exact I.
  - intros. exact I.
  - intros. exact I.
  - intros params metadata handler IH. exact IH.
Qed.

(* Without symbols bound in code, the symbols bound are the static names
   of the term. *)
Lemma sym_bindings_static_names :
  (∀ e : expr,
     no_old_sets e ->
     map fst (sym_bindings e) = static_names symbols_of_binding e)
  ∧ ∀ h : continuation_handler,
      match h with
      | Handler _ _ handler =>
        no_old_sets handler ->
        map fst (sym_bindings handler) = static_names symbols_of_binding handler
      end.
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [no_old_sets sym_bindings static_names].
  - intros x k d body IH [_ H]. exact (IH H).
  - intros bindings body _ IH [Hb H]. rewrite map_app, (IH H). f_equal.
    induction bindings as [ | b bs IHb]; [ reflexivity | ].
    destruct Hb as [Hb Hbs]. cbn [flat_map]. rewrite map_app, (IHb Hbs).
    f_equal. destruct b as [id c | id | s const]; cbn in Hb |- *;
      [ | reflexivity | reflexivity ].
    destruct c. cbn in Hb |- *. rewrite (proj2 Hb). reflexivity.
  - intros k h body IHh IH [Hh H]. destruct h as [params md handler].
    cbn in Hh, IHh |- *. rewrite map_app, (IHh Hh), (IH H). reflexivity.
  - intros handlers body IHh IH [Hh H]. rewrite map_app, (IH H). f_equal.
    induction handlers as [ | [k h] hs IHhs]; [ reflexivity | ].
    destruct Hh as [Hh Hhs]. destruct h as [params md handler].
    cbn [flat_map handler_sym_bindings handler_static_names] in Hh |- *.
    rewrite map_app, (IHh k _ (or_introl eq_refl) Hh). f_equal.
    apply IHhs; [ | exact Hhs ].
    intros k' h' Hin. exact (IHh k' h' (or_intror Hin)).
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH. exact IH.
Qed.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

(* The closures of a unit of the fragment that is well formed and binds
   each variable once are made where the graph of the unit has only the
   facts of their sets. *)
Theorem closures_exact_of_bound_once (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  in_flow_fragment e = true -> well_formed_unit u e = true ->
  variables_bound_once e = true ->
  RP.closures_exact (graph_of_facts (traverse_unit flag u e)) (unit_kept e)
    e.
Proof.
  intros Hfrag Hwf Honce.
  assert (Hold : no_old_sets e).
  { unfold in_flow_fragment in Hfrag. apply andb_true_iff in Hfrag as [Hf _].
    exact (proj1 (fragment_no_old_sets _ _ _) e Hf). }
  assert (Hnl : NoDup (map fst (let_bindings e))).
  { unfold variables_bound_once in Honce. apply distinct_NoDup in Honce.
    exact (NoDup_app_remove_r _ _ Honce). }
  assert (Hns : NoDup (map fst (sym_bindings e))).
  { rewrite (proj1 sym_bindings_static_names e Hold).
    apply distinct_NoDup. unfold well_formed_unit in Hwf.
    repeat (apply andb_true_iff in Hwf as [Hwf ?]).
    assumption. }
  assert (Hsets : sets_well_formed e = true).
  { unfold well_formed_unit in Hwf.
    repeat (apply andb_true_iff in Hwf as [Hwf ?]).
    assumption. }
  destruct (traverse_unit_groups flag u e Hnl Hns Hold)
    as (Gs & a & -> & Hinv & Hpres).
  apply (proj1 (closures_exact_struct _ _) e).
  - intros x d Hin. destruct d as [ | | | slot decls vs | | | ];
      cbn [RP.named_exact]; try exact I.
    exact (closure_node_exact e Hnl Gs a Hinv x slot decls vs Hin).
  - intros t c Hin. destruct c as [ | slot cs decls vs | | | | ];
      cbn [RP.binding_exact]; try exact I.
    destruct (proj1 static_sets_whole e Hsets t slot cs decls vs Hin)
      as [Hdom Hwhole].
    split.
    + exact (static_node_exact e Hns Gs a Hinv t slot cs decls vs Hin
               Hdom Hwhole Hpres).
    + intros g. rewrite !is_bound_among. f_equal. exact Hdom.
Qed.
