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

(* The traversal of the reaper on a term of
   FlambdaReaperRegionsSpec.regions_plain and on the term with its
   regions dropped: the two graphs have the same constructors,
   accessors, parameters and arguments, and every other fact of the
   second is one of the first, or makes a variable that has any source
   an alias of the constants. And of the first: no fact has the
   constants where something is put.

   This file needs no review: it proves lemmas for
   FlambdaReaperRegionsProofs.v. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaRegionsSpec
  FlambdaReaperRegionsSpec FlambdaReaperRegionsRebuildProofs.
Import ListNotations.

(* ======================================================================== *)
(* The traversal does not read the facts                                    *)
(* ======================================================================== *)

(* The facts [l] put after those of [a]. *)
Definition app_facts (l : list fact) (a : acc) : acc :=
  {| acc_next := acc_next a; acc_facts := acc_facts a ++ l;
     acc_code_deps := acc_code_deps a; acc_apply_deps := acc_apply_deps a;
     acc_closure_deps := acc_closure_deps a; acc_symbols := acc_symbols a |}.

Definition clear (a : acc) : acc :=
  {| acc_next := acc_next a; acc_facts := [];
     acc_code_deps := acc_code_deps a; acc_apply_deps := acc_apply_deps a;
     acc_closure_deps := acc_closure_deps a; acc_symbols := acc_symbols a |}.

Lemma app_facts_clear (a : acc) : a = app_facts (acc_facts a) (clear a).
Proof. destruct a. reflexivity. Qed.

Lemma app_facts_app (l l' : list fact) (a : acc) :
  app_facts l (app_facts l' a) = app_facts (l' ++ l) a.
Proof. unfold app_facts. cbn. rewrite app_assoc. reflexivity. Qed.

Lemma fr_add_facts (L l : list fact) (a : acc) :
  add_facts L (app_facts l a) = app_facts l (add_facts L a).
Proof. unfold add_facts, app_facts. cbn. rewrite app_assoc. reflexivity.
Qed.

Lemma fr_add_fact (f : fact) (l : list fact) (a : acc) :
  add_fact f (app_facts l a) = app_facts l (add_fact f a).
Proof. apply fr_add_facts. Qed.

Lemma fr_fresh (l : list fact) (a : acc) :
  fresh (app_facts l a) = (fst (fresh a), app_facts l (snd (fresh a))).
Proof. reflexivity. Qed.

Lemma fr_fresh_list (n : nat) (l : list fact) :
  ∀ a : acc,
    fresh_list n (app_facts l a)
    = (fst (fresh_list n a), app_facts l (snd (fresh_list n a))).
Proof.
  induction n as [ | n IH]; intros a; [ reflexivity | ].
  cbn [fresh_list]. rewrite fr_fresh. destruct (fresh a) as [x a1].
  cbn [fst snd]. rewrite IH. destruct (fresh_list n a1). reflexivity.
Qed.

Lemma fr_note_simple (s : simple) (l : list fact) (a : acc) :
  note_simple s (app_facts l a) = app_facts l (note_simple s a).
Proof. destruct s; reflexivity. Qed.

Lemma fr_fold {A : Type} (f : acc -> A -> acc) (xs : list A) (l : list fact) :
  (∀ a x, x ∈ xs -> f (app_facts l a) x = app_facts l (f a x)) ->
  ∀ a,
    List.fold_left f xs (app_facts l a) = app_facts l (List.fold_left f xs a).
Proof.
  induction xs as [ | x xs IH]; intros H a; [ reflexivity | ].
  cbn [List.fold_left]. rewrite (H a x (or_introl eq_refl)). apply IH.
  intros a' x' Hx'. apply H. right. exact Hx'.
Qed.

Lemma fr_note_simples (ss : list simple) (l : list fact) (a : acc) :
  note_simples ss (app_facts l a) = app_facts l (note_simples ss a).
Proof.
  unfold note_simples. apply fr_fold. intros a' s _. apply fr_note_simple.
Qed.

Lemma fr_add_code_dep (id : code_id) (cd : code_dep) (l : list fact)
    (a : acc) :
  add_code_dep id cd (app_facts l a) = app_facts l (add_code_dep id cd a).
Proof. reflexivity. Qed.

Lemma fr_push_apply_dep (d : apply_dep) (l : list fact) (a : acc) :
  push_apply_dep d (app_facts l a) = app_facts l (push_apply_dep d a).
Proof. reflexivity. Qed.

Lemma fr_push_closure_dep (d : closure_dep) (l : list fact) (a : acc) :
  push_closure_dep d (app_facts l a) = app_facts l (push_closure_dep d a).
Proof. reflexivity. Qed.

Lemma fr_cond_any_usage (env : traverse_env) (s : simple) (l : list fact)
    (a : acc) :
  cond_any_usage env s (app_facts l a) = app_facts l (cond_any_usage env s a).
Proof.
  unfold cond_any_usage. rewrite fr_note_simple.
  destruct (env_current_code_id env); apply fr_add_fact.
Qed.

Lemma fr_cond_alias (env : traverse_env) (from to_ : node) (l : list fact)
    (a : acc) :
  cond_alias env from to_ (app_facts l a)
  = app_facts l (cond_alias env from to_ a).
Proof.
  unfold cond_alias. destruct (env_current_code_id env); apply fr_add_fact.
Qed.

Lemma fr_find_cont (env : traverse_env) (k : continuation) (n : nat)
    (l : list fact) (a : acc) :
  find_cont env k n (app_facts l a)
  = (fst (find_cont env k n a), app_facts l (snd (find_cont env k n a))).
Proof.
  unfold find_cont. destruct (lookup k (env_conts env)); [ reflexivity | ].
  rewrite fr_fresh_list. destruct (fresh_list n a) as [ps a1].
  cbn [fst snd]. rewrite fr_add_facts. reflexivity.
Qed.

Ltac fr :=
  repeat first
    [ rewrite fr_add_facts | rewrite fr_add_fact | rewrite fr_note_simple
    | rewrite fr_note_simples | rewrite fr_add_code_dep
    | rewrite fr_push_apply_dep | rewrite fr_push_closure_dep
    | rewrite fr_cond_any_usage | rewrite fr_cond_alias ].

Lemma fr_block_deps (base : node) (fields : list simple) (l : list fact)
    (a : acc) :
  block_deps base fields (app_facts l a)
  = app_facts l (block_deps base fields a).
Proof. unfold block_deps. fr. reflexivity. Qed.

Lemma fr_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (l : list fact) (a : acc) :
  traverse_prim env x p args (app_facts l a)
  = app_facts l (traverse_prim env x p args a).
Proof.
  unfold traverse_prim.
  repeat match goal with
  | |- context [ match ?t with _ => _ end ] => destruct t
  end; rewrite ?fr_block_deps; fr; reflexivity.
Qed.

Lemma fr_record_set (env : traverse_env)
    (closures : list (node * option code_id))
    (function_slots : list (function_slot * node))
    (value_slots : list (value_slot * simple)) (l : list fact) (a : acc) :
  record_set_of_closures_deps env closures function_slots value_slots
    (app_facts l a)
  = app_facts l
      (record_set_of_closures_deps env closures function_slots value_slots a).
Proof.
  unfold record_set_of_closures_deps. apply fr_fold.
  intros a' [base code] _. destruct code; fr; reflexivity.
Qed.

Lemma fr_closures_of_set (x : variable) (slot : function_slot)
    (others : list (function_slot * variable)) (l : list fact) :
  ∀ (decls : list (function_slot * code_id)) (a : acc),
    closures_of_set x slot others decls (app_facts l a)
    = (fst (closures_of_set x slot others decls a),
       app_facts l (snd (closures_of_set x slot others decls a))).
Proof.
  induction decls as [ | [f id] decls IH]; intros a; [ reflexivity | ].
  cbn [closures_of_set].
  destruct (Pos.eqb f slot).
  - rewrite IH. destruct (closures_of_set x slot others decls a).
    reflexivity.
  - destruct (lookup f others).
    + rewrite IH. destruct (closures_of_set x slot others decls a).
      reflexivity.
    + rewrite fr_fresh. destruct (fresh a) as [n a1]. cbn [fst snd].
      rewrite IH. destruct (closures_of_set x slot others decls a1).
      reflexivity.
Qed.

Lemma fr_traverse_closure_of_set (env : traverse_env) (x : variable)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (l : list fact) (a : acc) :
  traverse_closure_of_set env x slot decls value_slots (app_facts l a)
  = app_facts l (traverse_closure_of_set env x slot decls value_slots a).
Proof.
  unfold traverse_closure_of_set.
  destruct (among slot (map fst (compact decls))); [ | reflexivity ].
  rewrite fr_closures_of_set.
  destruct (closures_of_set x slot (set_closures_of env x) (compact decls) a)
    as [cls a1].
  cbn [fst snd]. apply fr_record_set.
Qed.

Lemma fr_traverse_named (env : traverse_env) (x : variable) (d : named)
    (l : list fact) (a : acc) :
  traverse_named env x d (app_facts l a)
  = app_facts l (traverse_named env x d a).
Proof.
  destruct d; cbn [traverse_named]; fr;
    try reflexivity;
    first [ apply fr_traverse_prim | apply fr_record_set
          | apply fr_traverse_closure_of_set ].
Qed.

Lemma fr_traverse_static_binding (env : traverse_env) (b : static_binding)
    (l : list fact) (a : acc) :
  traverse_static_binding env b (app_facts l a)
  = app_facts l (traverse_static_binding env b a).
Proof.
  destruct b as [ | | s const]; try reflexivity. cbn [traverse_static_binding].
  destruct const as [ | | | | | elements];
    try apply fr_record_set; try apply fr_block_deps; fr; try reflexivity.
  destruct elements; [ fr; reflexivity | apply fr_block_deps ].
Qed.

Lemma fr_create_known (id : code_id) (params returns : list node)
    (exn : node) (l : list fact) (a : acc) :
  create_known_arity_call_witness id params returns exn (app_facts l a)
  = (fst (create_known_arity_call_witness id params returns exn a),
     app_facts l
       (snd (create_known_arity_call_witness id params returns exn a))).
Proof.
  unfold create_known_arity_call_witness. rewrite fr_fresh.
  destruct (fresh a) as [w a1]. cbn [fst snd]. fr. reflexivity.
Qed.

Lemma fr_witness_deps (id : code_id) (returns : list node) (exn : node)
    (l : list fact) :
  ∀ (pws : list (list node * node)) (a : acc),
    unknown_arity_witness_deps id returns exn pws (app_facts l a)
    = app_facts l (unknown_arity_witness_deps id returns exn pws a).
Proof.
  induction pws as [ | [first w] pws IH]; intros a; [ reflexivity | ].
  cbn [unknown_arity_witness_deps]. fr.
  destruct pws as [ | [first' w'] pws']; [ fr; reflexivity | ].
  rewrite fr_fresh. destruct (fresh _) as [v a1]. cbn [fst snd]. fr.
  apply IH.
Qed.

Lemma fr_create_unknown (id : code_id) (is_tupled : bool)
    (params_arity : arity) (params returns : list node) (exn : node)
    (l : list fact) (a : acc) :
  create_unknown_arity_call_witnesses id is_tupled params_arity params
    returns exn (app_facts l a)
  = (fst (create_unknown_arity_call_witnesses id is_tupled params_arity
            params returns exn a),
     app_facts l
       (snd (create_unknown_arity_call_witnesses id is_tupled params_arity
               params returns exn a))).
Proof.
  unfold create_unknown_arity_call_witnesses. destruct is_tupled.
  - unfold create_unknown_arity_tupled_call_witnesses. rewrite fr_fresh.
    destruct (fresh a) as [w a1]. cbn [fst snd]. rewrite fr_fresh.
    destruct (fresh a1) as [u a2]. cbn [fst snd]. rewrite fr_fresh.
    destruct (fresh a2) as [k a3]. cbn [fst snd]. fr. reflexivity.
  - unfold create_unknown_arity_non_tupled_call_witnesses.
    rewrite fr_fresh_list.
    destruct (fresh_list _ a) as [ws a1]. cbn [fst snd].
    rewrite fr_witness_deps. reflexivity.
Qed.

Lemma fr_prepare_code (id : code_id) (c : code) (l : list fact) (a : acc) :
  prepare_code id c (app_facts l a) = app_facts l (prepare_code id c a).
Proof.
  unfold prepare_code.
  rewrite fr_fresh_list. destruct (fresh_list _ a) as [rs a1].
  cbn [fst snd]. rewrite fr_fresh. destruct (fresh a1) as [ex a2].
  cbn [fst snd]. rewrite fr_fresh. destruct (fresh a2) as [mc a3].
  cbn [fst snd]. rewrite fr_fresh_list.
  destruct (fresh_list _ a3) as [ps a4]. cbn [fst snd].
  rewrite fr_create_known.
  destruct (create_known_arity_call_witness id ps rs ex a4) as [kw a5].
  cbn [fst snd]. rewrite fr_create_unknown.
  destruct (create_unknown_arity_call_witnesses _ _ _ _ _ _ a5) as [uw a6].
  cbn [fst snd]. fr. reflexivity.
Qed.

Lemma fr_prepare_binding (b : static_binding) (l : list fact) (a : acc) :
  prepare_binding b (app_facts l a) = app_facts l (prepare_binding b a).
Proof. destruct b; [ apply fr_prepare_code | reflexivity | reflexivity ].
Qed.

Lemma fr_make_known (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (l : list fact) (a : acc) :
  make_known_arity_apply_widget env ap returns exn (app_facts l a)
  = (fst (make_known_arity_apply_widget env ap returns exn a),
     app_facts l (snd (make_known_arity_apply_widget env ap returns exn a))).
Proof.
  unfold make_known_arity_apply_widget.
  rewrite fr_fresh. destruct (fresh a) as [w a1]. cbn [fst snd].
  rewrite fr_fresh. destruct (fresh a1) as [c a2]. cbn [fst snd].
  rewrite fr_fresh. destruct (fresh a2) as [n a3]. cbn [fst snd].
  fr. reflexivity.
Qed.

Lemma fr_apply_deps (returns : list node) (exn called : node)
    (l : list fact) :
  ∀ (aws : list (list simple * node)) (a : acc),
    unknown_arity_apply_deps returns exn called aws (app_facts l a)
    = app_facts l (unknown_arity_apply_deps returns exn called aws a).
Proof.
  induction aws as [ | [first w] aws IH]; intros a; [ reflexivity | ].
  cbn [unknown_arity_apply_deps]. fr.
  destruct aws as [ | [first' w'] aws']; [ fr; reflexivity | ].
  rewrite fr_fresh. destruct (fresh _) as [v a1]. cbn [fst snd]. fr.
  apply IH.
Qed.

Lemma fr_make_unknown (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (l : list fact) (a : acc) :
  make_unknown_arity_apply_widget env ap returns exn (app_facts l a)
  = (fst (make_unknown_arity_apply_widget env ap returns exn a),
     app_facts l
       (snd (make_unknown_arity_apply_widget env ap returns exn a))).
Proof.
  unfold make_unknown_arity_apply_widget.
  rewrite fr_fresh. destruct (fresh a) as [c a1]. cbn [fst snd]. fr.
  rewrite fr_fresh_list. destruct (fresh_list _ _) as [ws a2].
  cbn [fst snd]. fr. rewrite fr_apply_deps.
  rewrite fr_fresh. destruct (fresh _) as [n a3]. cbn [fst snd].
  destruct ws; fr; reflexivity.
Qed.

Lemma fr_add_apply (env : traverse_env) (id : code_id) (callee widget : node)
    (only : bool) (l : list fact) (a : acc) :
  add_apply env id callee widget only (app_facts l a)
  = app_facts l (add_apply env id callee widget only a).
Proof.
  unfold add_apply. destruct only; [ | fr; reflexivity ].
  rewrite fr_fresh. destruct (fresh a) as [c a1]. cbn [fst snd].
  rewrite fr_fresh. destruct (fresh a1) as [w a2]. cbn [fst snd].
  fr. reflexivity.
Qed.

Lemma fr_traverse_call_kind (env : traverse_env) (ap : apply_expr)
    (exn : node) (returns : list node) (l : list fact) (a : acc) :
  traverse_call_kind env ap exn returns (app_facts l a)
  = app_facts l (traverse_call_kind env ap exn returns a).
Proof.
  unfold traverse_call_kind. fr.
  destruct (ap_call_kind ap).
  - rewrite fr_make_known. destruct (make_known_arity_apply_widget _ _ _ _ _)
      as [w a1]. cbn [fst snd]. fr.
    destruct (env_should_preserve_direct_calls env);
      try destruct (negb _); try rewrite fr_add_apply; reflexivity.
  - rewrite fr_make_unknown.
    destruct (make_unknown_arity_apply_widget _ _ _ _ _) as [w a1].
    cbn [fst snd]. fr. reflexivity.
  - rewrite fr_make_known. destruct (make_known_arity_apply_widget _ _ _ _ _)
      as [w a1]. cbn [fst snd]. fr. reflexivity.
Qed.

Lemma fr_traverse_apply (env : traverse_env) (ap : apply_expr)
    (l : list fact) (a : acc) :
  traverse_apply env ap (app_facts l a)
  = app_facts l (traverse_apply env ap a).
Proof.
  unfold traverse_apply.
  rewrite fr_find_cont. destruct (find_cont env _ _ a) as [rs a1].
  cbn [fst snd]. rewrite fr_find_cont.
  destruct (find_cont env _ _ a1) as [ps a2]. cbn [fst snd].
  destruct ps as [ | p ps].
  - rewrite fr_fresh. destruct (fresh a2) as [x a3]. cbn [fst snd].
    apply fr_traverse_call_kind.
  - fr. apply fr_traverse_call_kind.
Qed.

Lemma fr_apply_cont_deps (env : traverse_env) (ac : apply_cont)
    (l : list fact) (a : acc) :
  apply_cont_deps env ac (app_facts l a)
  = app_facts l (apply_cont_deps env ac a).
Proof.
  unfold apply_cont_deps. rewrite fr_find_cont.
  destruct (find_cont env _ _ a) as [ps a1]. cbn [fst snd]. fr.
  reflexivity.
Qed.

Lemma fr_traverse_switch (env : traverse_env) (s : simple)
    (arms : list (Z * apply_cont)) (l : list fact) (a : acc) :
  traverse_switch env s arms (app_facts l a)
  = app_facts l (traverse_switch env s arms a).
Proof.
  unfold traverse_switch. fr. apply fr_fold.
  intros a' [i ac] _. apply fr_apply_cont_deps.
Qed.

Lemma fr_exn_handler_deps (env : traverse_env) (h : continuation_handler)
    (l : list fact) (a : acc) :
  exn_handler_deps env h (app_facts l a)
  = app_facts l (exn_handler_deps env h a).
Proof.
  destruct h as [[ | p ps] md hbody]; cbn [exn_handler_deps];
    [ reflexivity | ].
  destruct (hm_is_exn_handler md); [ fr; reflexivity | reflexivity ].
Qed.

Lemma fr_traverse :
  (∀ e : expr, ∀ env l a,
     traverse env e (app_facts l a) = app_facts l (traverse env e a))
  ∧ (∀ h : continuation_handler, ∀ env l a,
       traverse_handler env h (app_facts l a)
       = app_facts l (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH env l a. cbn [traverse].
    rewrite fr_traverse_named. apply IH.
  - intros bindings body Hcodes IH env l a. cbn [traverse].
    rewrite (fr_fold (fun a b => prepare_binding b a))
      by (intros; apply fr_prepare_binding).
    rewrite (fr_fold (fun a b => traverse_static_binding env b a))
      by (intros; apply fr_traverse_static_binding).
    rewrite (fr_fold (fun a b => traverse_binding env b a)); [ apply IH | ].
    intros a' b Hb. destruct b as [id c | | ]; try reflexivity.
    cbn [traverse_binding]. destruct c as [params mc ret exn md cbody].
    cbn [traverse_code acc_code_deps app_facts].
    destruct (lookup id (acc_code_deps a')); [ | reflexivity ].
    fr. apply (Hcodes id _ Hb).
  - intros k h body IHh IH env l a. cbn [traverse].
    rewrite IHh, fr_exn_handler_deps. apply IH.
  - intros handlers body IHh IH env l a. cbn [traverse].
    rewrite (fr_fold (fun a '(_, h) => traverse_handler _ h a)); [ apply IH | ].
    intros a' [k h] Hin. apply (IHh k h Hin).
  - intros ap env l a. apply fr_traverse_apply.
  - intros ac env l a. apply fr_apply_cont_deps.
  - intros s arms env l a. apply fr_traverse_switch.
  - reflexivity.
  - intros params md hbody IH env l a. apply IH.
Qed.

Lemma fr_add_closure_dep (cds : finite_map code_dep) (d : closure_dep)
    (l : list fact) (a : acc) :
  add_closure_dep cds d (app_facts l a) = app_facts l (add_closure_dep cds d a).
Proof.
  unfold add_closure_dep. destruct (lookup _ cds); [ fr; reflexivity | ].
  rewrite fr_fresh. destruct (fresh a) as [w a1]. cbn [fst snd].
  fr. reflexivity.
Qed.

Lemma fr_resolve (syms : list symbol) (l : list fact) (a : acc) :
  resolve_delayed_deps syms (app_facts l a)
  = app_facts l (resolve_delayed_deps syms a).
Proof.
  unfold resolve_delayed_deps. cbn [acc_code_deps acc_apply_deps
    acc_closure_deps app_facts]. fr.
  rewrite (fr_fold (fun a d => add_closure_dep _ d a))
    by (intros; apply fr_add_closure_dep).
  fr. reflexivity.
Qed.

(* ======================================================================== *)
(* The facts of the two traversals                                          *)
(* ======================================================================== *)

(* A fact that is not a constructor, an accessor, a parameter or an
   argument. *)
Definition capa_free (fct : fact) : Prop :=
  match fct with
  | Fact_constructor _ _ _ | Fact_accessor _ _ _ | Fact_parameter _ _ _
  | Fact_argument _ _ _ => False
  | _ => True
  end.

Definition capa_eq (l l' : list fact) : Prop :=
  f_constructor (facts_of_list l') = f_constructor (facts_of_list l)
  ∧ f_accessor (facts_of_list l') = f_accessor (facts_of_list l)
  ∧ f_parameter (facts_of_list l') = f_parameter (facts_of_list l)
  ∧ f_argument (facts_of_list l') = f_argument (facts_of_list l).

(* An alias of a variable that has any source to the constants. *)
Definition allowed (l : list fact) (fct : fact) : Prop :=
  ∃ x : variable,
    fct = Fact_alias (Node_variable x) Node_all_constants
    ∧ Fact_any_source (Node_variable x) ∈ l.

Definition facts_rel (l l' : list fact) : Prop :=
  capa_eq l l' ∧ ∀ fct, fct ∈ l' -> fct ∈ l ∨ allowed l fct.

Definition acc_rel (a a' : acc) : Prop :=
  clear a' = clear a ∧ facts_rel (acc_facts a) (acc_facts a').

Lemma acc_rel_refl (a : acc) : acc_rel a a.
Proof.
  split; [ reflexivity | ]. split; [ repeat split | ].
  intros fct H. left. exact H.
Qed.

Lemma capa_eq_app (L l l' : list fact) :
  capa_eq l l' -> capa_eq (L ++ l) (L ++ l').
Proof.
  intros (H1 & H2 & H3 & H4). unfold capa_eq. cbn [facts_of_list
    f_constructor f_accessor f_parameter f_argument] in *.
  rewrite !flat_map_app, H1, H2, H3, H4. repeat split.
Qed.

Lemma capa_free_nil (L : list fact) :
  (∀ fct, fct ∈ L -> capa_free fct) ->
  f_constructor (facts_of_list L) = [] ∧ f_accessor (facts_of_list L) = []
  ∧ f_parameter (facts_of_list L) = [] ∧ f_argument (facts_of_list L) = [].
Proof.
  induction L as [ | fct L IH]; intros H; [ repeat split | ].
  destruct IH as (H1 & H2 & H3 & H4);
    [ intros f Hf; apply H; right; exact Hf | ].
  specialize (H fct (or_introl eq_refl)).
  cbn [facts_of_list f_constructor f_accessor f_parameter f_argument
         flat_map] in *.
  rewrite H1, H2, H3, H4.
  destruct fct; cbn in H; try contradiction; repeat split.
Qed.

Lemma facts_rel_app (L l l' : list fact) :
  facts_rel l l' -> facts_rel (L ++ l) (L ++ l').
Proof.
  intros [Hc Hi]. split; [ exact (capa_eq_app L l l' Hc) | ].
  intros fct Hin. apply in_app_or in Hin as [Hin | Hin];
    [ left; apply in_or_app; left; exact Hin | ].
  destruct (Hi fct Hin) as [H | (x & -> & Hx)];
    [ left; apply in_or_app; right; exact H | ].
  right. exists x. split; [ reflexivity | ]. apply in_or_app. right.
  exact Hx.
Qed.

Lemma facts_rel_left (L l l' : list fact) :
  (∀ fct, fct ∈ L -> capa_free fct) -> facts_rel l l' ->
  facts_rel (L ++ l) l'.
Proof.
  intros HL [(H1 & H2 & H3 & H4) Hi].
  destruct (capa_free_nil L HL) as (G1 & G2 & G3 & G4). split.
  - unfold capa_eq. cbn [facts_of_list f_constructor f_accessor f_parameter
      f_argument] in *.
    rewrite !flat_map_app, G1, G2, G3, G4. cbn [app].
    repeat split; assumption.
  - intros fct Hin. destruct (Hi fct Hin) as [H | (x & -> & Hx)];
      [ left; apply in_or_app; right; exact H | ].
    right. exists x. split; [ reflexivity | ]. apply in_or_app. right.
    exact Hx.
Qed.

Lemma facts_rel_alias (x : variable) (l l' : list fact) :
  Fact_any_source (Node_variable x) ∈ l -> facts_rel l l' ->
  facts_rel l (Fact_alias (Node_variable x) Node_all_constants :: l').
Proof.
  intros Hx [(H1 & H2 & H3 & H4) Hi]. split.
  - unfold capa_eq in *. cbn [facts_of_list f_constructor f_accessor
      f_parameter f_argument flat_map app] in *.
    repeat split; assumption.
  - intros fct [<- | Hin]; [ right; exists x; split; [ reflexivity | ] | ];
      [ exact Hx | exact (Hi fct Hin) ].
Qed.

Lemma clear_app_facts (l : list fact) (a : acc) :
  clear (app_facts l a) = clear a.
Proof. reflexivity. Qed.

Lemma acc_rel_frame (f : acc -> acc) :
  (∀ l a, f (app_facts l a) = app_facts l (f a)) ->
  ∀ a a', acc_rel a a' -> acc_rel (f a) (f a').
Proof.
  intros Hf a a' [Hcl Hr].
  rewrite (app_facts_clear a), (app_facts_clear a'), Hcl, !Hf.
  split; [ reflexivity | ]. cbn [app_facts acc_facts].
  apply facts_rel_app, Hr.
Qed.

Lemma acc_rel_fold {A : Type} (f g : acc -> A -> acc) (xs : list A) :
  (∀ x a a', x ∈ xs -> acc_rel a a' -> acc_rel (f a x) (g a' x)) ->
  ∀ a a', acc_rel a a' ->
    acc_rel (List.fold_left f xs a) (List.fold_left g xs a').
Proof.
  induction xs as [ | x xs IH]; intros H a a' Hr; [ exact Hr | ].
  cbn [List.fold_left]. apply IH; [ | apply H; [ left; reflexivity | ] ];
    [ intros y b b' Hy; apply H; right; exact Hy | exact Hr ].
Qed.

Lemma fold_left_map {A B C : Type} (f : A -> B -> A) (g : C -> B)
    (l : list C) (a : A) :
  List.fold_left f (map g l) a = List.fold_left (fun a x => f a (g x)) l a.
Proof.
  revert a. induction l as [ | x l IH]; intros a; [ reflexivity | ].
  cbn. apply IH.
Qed.

Lemma acc_rel_extra (L : list fact) (a a' : acc) :
  (∀ fct, fct ∈ L -> capa_free fct) -> acc_rel a a' ->
  acc_rel (add_facts L a) a'.
Proof.
  intros HL [Hcl Hr]. split; [ exact Hcl | ].
  apply facts_rel_left; assumption.
Qed.

Lemma acc_rel_cond (env : traverse_env) (x : variable) (a a' : acc) :
  acc_rel a a' -> acc_rel (cond_any_usage env (Var x) a) a'.
Proof.
  intros H. unfold cond_any_usage. cbn [note_simple].
  destruct (env_current_code_id env); apply acc_rel_extra;
    try exact H; intros fct [<- | []]; exact I.
Qed.

Lemma prepare_binding_erase (b : static_binding) (a : acc) :
  prepare_binding (erase_binding b) a = prepare_binding b a.
Proof. destruct b as [id c | | ]; [ destruct c | | ]; reflexivity. Qed.

Lemma traverse_static_binding_erase (env : traverse_env)
    (b : static_binding) (a : acc) :
  traverse_static_binding env (erase_binding b) a
  = traverse_static_binding env b a.
Proof. destruct b; reflexivity. Qed.

Lemma handler_params_erase (h : continuation_handler) (c : region_conts) :
  forallb kind_not_region (handler_kinds h) = true ->
  handler_params (erase_handler c h) = handler_params h.
Proof.
  destruct h as [params md hbody]. intros H. cbn [handler_kinds] in H.
  rewrite (erase_handler_no_region c params md hbody H). reflexivity.
Qed.

Lemma exn_handler_deps_erase (env : traverse_env) (h : continuation_handler)
    (c : region_conts) (a : acc) :
  forallb kind_not_region (handler_kinds h) = true ->
  exn_handler_deps env (erase_handler c h) a = exn_handler_deps env h a.
Proof.
  destruct h as [params md hbody]. intros H. cbn [handler_kinds] in H.
  rewrite (erase_handler_no_region c params md hbody H).
  destruct params; reflexivity.
Qed.

Lemma add_rec_conts_erase (env : traverse_env)
    (hs : list (continuation * continuation_handler)) (c : region_conts) :
  (∀ k h, (k, h) ∈ hs -> forallb kind_not_region (handler_kinds h) = true) ->
  add_rec_conts env (map (fun '(k, h) => (k, erase_handler c h)) hs)
  = add_rec_conts env hs.
Proof.
  induction hs as [ | [k h] hs IH]; intros H; [ reflexivity | ].
  cbn [map add_rec_conts fold_right].
  rewrite (handler_params_erase h c (H k h (or_introl eq_refl))).
  unfold add_rec_conts in IH. rewrite IH; [ reflexivity | ].
  intros k' h' Hin. apply (H k' h'). right. exact Hin.
Qed.

Lemma uses_capa_free (x : variable) (ns : list node) :
  ∀ fct, fct ∈ map (fun from => Fact_use (Node_variable x) from) ns ->
    capa_free fct.
Proof. intros fct Hin. apply in_map_iff in Hin as [n [<- _]]. exact I. Qed.

Lemma acc_rel_end (env : traverse_env) (x : variable) (U : list fact)
    (a a' : acc) :
  (∀ fct, fct ∈ U -> capa_free fct) -> acc_rel a a' ->
  acc_rel (add_facts (Fact_any_source (Node_variable x) :: U)
             (cond_any_usage env (Var x) a))
    (add_fact (Fact_alias (Node_variable x) Node_all_constants) a').
Proof.
  intros HU Hr.
  pose proof (acc_rel_cond env x a a' Hr) as H1.
  assert (H2 : acc_rel (add_facts (Fact_any_source (Node_variable x) :: U)
                          (cond_any_usage env (Var x) a)) a').
  { apply acc_rel_extra; [ | exact H1 ].
    intros fct [<- | Hin]; [ exact I | exact (HU fct Hin) ]. }
  destruct H2 as [Hcl Hrel]. split; [ exact Hcl | ].
  apply facts_rel_alias; [ left; reflexivity | exact Hrel ].
Qed.

Lemma acc_rel_end_try (f : fact) (U : list fact) (b b' : acc) :
  (∀ fct, fct ∈ U -> capa_free fct) -> acc_rel b b' ->
  acc_rel (add_facts (f :: U) b) (add_facts [f] b').
Proof.
  intros HU [Hcl Hr]. split; [ exact Hcl | ].
  cbn [add_facts acc_facts app].
  apply (facts_rel_app [f]). apply facts_rel_left; assumption.
Qed.

Lemma traverse_rel :
  (∀ e : expr, ∀ kinds env a a',
     regions_plain kinds e = true -> acc_rel a a' ->
     acc_rel (traverse env e a) (traverse env (erase [] e) a'))
  ∧ (∀ h : continuation_handler, ∀ kinds env a a',
       handler_regions_plain kinds h = true -> acc_rel a a' ->
       acc_rel (traverse_handler env h a)
         (traverse_handler env (erase_handler [] h) a')).
Proof.
  apply regions_ind.
  - (* Let *)
    intros x kind d body IH IH2 kinds env a a' Hpl Hr.
    destruct (region_prim_of d) as [[p args] | ] eqn:Hrp.
    + apply region_prim_of_some in Hrp. subst d.
      destruct p; destruct args as [ | s [ | s' args]];
        cbn [regions_plain] in Hpl; try discriminate Hpl;
        apply andb_true_iff in Hpl as [_ Hb];
        cbn [erase erase_region_primitive traverse].
      * apply (IH _ env _ _ Hb). apply acc_rel_extra; [ | exact Hr ].
        intros fct [<- | []]. exact I.
      * apply (IH _ env _ _ Hb). apply acc_rel_extra; [ | exact Hr ].
        intros fct [<- | []]. exact I.
      * apply (IH _ env _ _ Hb).
        cbn [traverse_named traverse_prim prim_has_arbitrary_effects].
        apply acc_rel_end; [ apply uses_capa_free | exact Hr ].
      * apply (IH _ env _ _ Hb).
        cbn [traverse_named traverse_prim prim_has_arbitrary_effects].
        apply acc_rel_end_try; [ apply uses_capa_free | ].
        apply (acc_rel_frame _ (fr_cond_any_usage env (Var x))), Hr.
    + rewrite (regions_plain_let _ _ _ _ _ Hrp) in Hpl.
      apply andb_true_iff in Hpl as [_ Hla].
      unfold la_check in Hla.
      destruct (la_body body) as [[[[y k'] args] body'] | ] eqn:Hlab.
      * apply la_body_some in Hlab. subst body.
        apply andb_true_iff in Hla as [Hla Hb'].
        apply andb_true_iff in Hla as [Hla Hargs].
        apply andb_true_iff in Hla as [Hla Hk'].
        apply andb_true_iff in Hla as [_ Hy]. apply Pos.eqb_eq in Hy.
        subst y.
        destruct args as [ | [x' | | ] rs]; try discriminate Hargs.
        apply Pos.eqb_eq in Hargs. subst x'.
        rewrite (erase_let_plain _ _ _ _ _ Hrp).
        cbn [erase erase_region_primitive]. rewrite Pos.eqb_refl.
        cbn [traverse traverse_named traverse_prim].
        apply (IH2 x k' (Prim (Region_primitive Local_allocation)
                           (Var x :: rs)) body' eq_refl _ env _ _ Hb').
        apply (acc_rel_frame _ (fr_traverse_named env x d)), Hr.
      * rewrite (erase_let_plain _ _ _ _ _ Hrp). cbn [traverse].
        apply (IH _ env _ _ Hla).
        apply (acc_rel_frame _ (fr_traverse_named env x d)), Hr.
  - (* Let_static *)
    intros bindings body Hcodes IH kinds env a a' Hpl Hr.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hbs Hb].
    rewrite forallb_forall in Hbs.
    cbn [erase traverse]. rewrite !fold_left_map.
    apply (IH _ env _ _ Hb).
    apply acc_rel_fold.
    + intros bd b0 b0' Hin Hr0.
      destruct bd as [id c | id | sym const]; try exact Hr0.
      destruct c as [params mc ret exn md cbody].
      cbn [erase_binding erase_code traverse_binding traverse_code].
      destruct Hr0 as [Hcl Hrel].
      assert (Hcd : acc_code_deps b0' = acc_code_deps b0)
        by (apply (f_equal acc_code_deps) in Hcl; exact Hcl).
      rewrite Hcd.
      destruct (lookup id (acc_code_deps b0)) as [cd | ];
        [ | split; assumption ].
      specialize (Hbs _ Hin).
      cbn [binding_regions_plain code_regions_plain] in Hbs.
      apply andb_true_iff in Hbs as [_ Hcb].
      apply (Hcodes id _ Hin _ _ _ _ Hcb).
      apply (acc_rel_frame _ (fr_add_facts _)). split; assumption.
    + apply acc_rel_fold.
      * intros bd b0 b0' _ Hr0. rewrite traverse_static_binding_erase.
        apply (acc_rel_frame _ (fr_traverse_static_binding env bd)), Hr0.
      * apply acc_rel_fold; [ | exact Hr ].
        intros bd b0 b0' _ Hr0. rewrite prepare_binding_erase.
        apply (acc_rel_frame _ (fr_prepare_binding bd)), Hr0.
  - (* Let_cont_non_recursive *)
    intros k h body IHh IH kinds env a a' Hpl Hr.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hh Hb].
    pose proof (handler_kinds_plain _ _ Hh) as Hhk.
    cbn [erase].
    rewrite (erase_plain _ body
               (conts_plain_cons _ _ _ (region_places_plain _ Hhk)
                  conts_plain_nil)).
    cbn [traverse].
    rewrite (handler_params_erase h [] Hhk),
      (exn_handler_deps_erase env h [] _ Hhk).
    apply (IH _ _ _ _ Hb).
    apply (acc_rel_frame _ (fr_exn_handler_deps env h)).
    exact (IHh _ env _ _ Hh Hr).
  - (* Let_cont_recursive *)
    intros handlers body IHh IH kinds env a a' Hpl Hr.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hhs Hb].
    rewrite forallb_forall in Hhs.
    assert (Hplain : ∀ k h, (k, h) ∈ handlers ->
                       forallb kind_not_region (handler_kinds h) = true)
      by (intros k h Hin; exact (handler_kinds_plain _ _ (Hhs (k, h) Hin))).
    cbn [erase traverse].
    rewrite (erase_plain _ body (rec_conts_plain handlers Hplain)).
    rewrite (add_rec_conts_erase env handlers _ Hplain).
    rewrite fold_left_map.
    apply (IH _ _ _ _ Hb).
    apply acc_rel_fold; [ | exact Hr ].
    intros [k h] b0 b0' Hin Hr0. cbn beta iota.
    rewrite (erase_handler_plain _ h (rec_conts_plain handlers Hplain)).
    exact (IHh k h Hin _ _ _ _ (Hhs (k, h) Hin) Hr0).
  - (* Apply *)
    intros ap kinds env a a' _ Hr. cbn [erase traverse].
    apply (acc_rel_frame _ (fr_traverse_apply env ap)), Hr.
  - (* Apply_cont *)
    intros ac kinds env a a' _ Hr. cbn [erase traverse].
    rewrite erase_apply_cont_nil.
    apply (acc_rel_frame _ (fr_apply_cont_deps env ac)), Hr.
  - (* Switch *)
    intros sc arms kinds env a a' _ Hr. cbn [erase traverse].
    rewrite map_arms_id.
    apply (acc_rel_frame _ (fr_traverse_switch env sc arms)), Hr.
  - (* Invalid *)
    intros kinds env a a' _ Hr. exact Hr.
  - (* Handler *)
    intros params md hbody IH kinds env a a' Hpl Hr.
    cbn [handler_regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hk Hb].
    rewrite (erase_handler_no_region [] params md hbody Hk).
    cbn [traverse_handler]. exact (IH _ env _ _ Hb Hr).
Qed.

(* ======================================================================== *)
(* What the traversal reads off the whole unit                              *)
(* ======================================================================== *)

Lemma erase_region_primitive_named (x : variable) (p : region_primitive)
    (args : list simple) (d : named) :
  erase_region_primitive x p args = Some d ->
  projected_value_slots_of_named d = [] ∧
  match d with Closure_of_set _ _ _ => False | _ => True end.
Proof.
  intros H. unfold erase_region_primitive in H.
  repeat match type of H with
  | context [ match ?t with _ => _ end ] => destruct t
  end; try discriminate H; injection H as <-;
    split; first [ reflexivity | exact I ].
Qed.

Lemma projected_value_slots_erase :
  (∀ e : expr, ∀ c, projected_value_slots (erase c e) = projected_value_slots e)
  ∧ (∀ h : continuation_handler, ∀ c,
       handler_projected_value_slots (erase_handler c h)
       = handler_projected_value_slots h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH c.
    destruct d as [s | p args | | | | | ]; cbn [erase];
      try (cbn [projected_value_slots]; rewrite IH; reflexivity).
    destruct p as [ | | | | r]; try (cbn [projected_value_slots];
                                     rewrite IH; reflexivity).
    destruct (erase_region_primitive x r args) as [d' | ] eqn:Hd;
      cbn [projected_value_slots]; rewrite IH; [ | reflexivity ].
    destruct (erase_region_primitive_named _ _ _ _ Hd) as [-> _].
    reflexivity.
  - intros bindings body Hcodes IH c. cbn [erase projected_value_slots].
    rewrite IH. f_equal. rewrite flat_map_concat_map, map_map,
      flat_map_concat_map. f_equal. apply map_ext_in.
    intros b Hin. destruct b as [id cd | | ]; try reflexivity.
    destruct cd. cbn [erase_binding erase_code binding_projected_value_slots
      code_projected_value_slots].
    exact (Hcodes id _ Hin []).
  - intros k h body IHh IH c. cbn [erase projected_value_slots].
    rewrite IHh, IH. reflexivity.
  - intros handlers body IHh IH c. cbn [erase projected_value_slots].
    rewrite IH. f_equal. rewrite flat_map_concat_map, map_map,
      flat_map_concat_map. f_equal. apply map_ext_in.
    intros [k h] Hin. exact (IHh k h Hin _).
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params md hbody IH c. cbn [erase_handler
      handler_projected_value_slots]. apply IH.
Qed.

Lemma static_names_erase (names_of : static_binding -> list positive) :
  (∀ b, names_of (erase_binding b) = names_of b) ->
  (∀ e : expr, ∀ c, static_names names_of (erase c e)
                    = static_names names_of e)
  ∧ (∀ h : continuation_handler, ∀ c,
       handler_static_names names_of (erase_handler c h)
       = handler_static_names names_of h).
Proof.
  intros Hn. apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH c.
    destruct d as [s | p args | | | | | ]; cbn [erase];
      try (cbn [static_names]; apply IH).
    destruct p as [ | | | | r]; try (cbn [static_names]; apply IH).
    destruct (erase_region_primitive x r args);
      cbn [static_names]; apply IH.
  - intros bindings body _ IH c. cbn [erase static_names].
    rewrite IH, flat_map_concat_map, map_map, flat_map_concat_map.
    f_equal. f_equal. apply map_ext. exact Hn.
  - intros k h body IHh IH c. cbn [erase static_names].
    rewrite IHh, IH. reflexivity.
  - intros handlers body IHh IH c. cbn [erase static_names].
    rewrite IH. f_equal. rewrite flat_map_concat_map, map_map,
      flat_map_concat_map. f_equal. apply map_ext_in.
    intros [k h] Hin. exact (IHh k h Hin _).
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params md hbody IH c. cbn [erase_handler handler_static_names].
    apply IH.
Qed.

Lemma let_projections_of_nil (x : variable) (slot : function_slot)
    (e : expr) :
  let_projections_of x slot [] e = [].
Proof. destruct e; reflexivity. Qed.

Lemma let_projections_of_erase (x : variable) (slot : function_slot) :
  ∀ (others : list function_slot) (body : expr),
    projections_plain x slot others body = true ->
    let_projections_of x slot others (erase [] body)
    = let_projections_of x slot others body.
Proof.
  intros others. induction others as [ | o os IH]; intros body H;
    [ rewrite !let_projections_of_nil; reflexivity | ].
  destruct body as [y k d b0 | | | | | | | ]; try reflexivity.
  destruct (region_prim_of d) as [[p args] | ] eqn:Hrp.
  - apply region_prim_of_some in Hrp. subst d. discriminate H.
  - rewrite (erase_let_plain _ _ _ _ _ Hrp).
    destruct d as [ | | | | | | mf mt c]; try reflexivity.
    destruct c as [x' | | ]; try reflexivity.
    cbn [projections_plain] in H. cbn [let_projections_of].
    destruct (Pos.eqb x' x && Pos.eqb mf slot && Pos.eqb mt o
              && negb (Pos.eqb y x));
      [ rewrite (IH b0 H); reflexivity | reflexivity ].
Qed.

Lemma projected_set_closures_other (x : variable) (slot : function_slot)
    (decls : list (function_slot * code_id)) (e : expr) :
  projected_set_closures x slot decls e
  = let_projections_of x slot (FlambdaReaperRebuild.other_slots slot decls)
      (after_local_allocation x e).
Proof.
  destruct decls as [ | [first id] others]; [ reflexivity | ].
  cbn [projected_set_closures FlambdaReaperRebuild.other_slots].
  destruct (Pos.eqb first slot); [ reflexivity | ].
  rewrite let_projections_of_nil. reflexivity.
Qed.

Lemma after_local_allocation_plain (x : variable) (slot : function_slot)
    (o : function_slot) (os : list function_slot) (e : expr) :
  projections_plain x slot (o :: os) e = true ->
  after_local_allocation x e = e.
Proof.
  intros H. destruct e as [y k d b | | | | | | | ]; try reflexivity.
  destruct d as [ | [ | | | | [ | | | | ]] | | | | | ]; try reflexivity;
    discriminate H.
Qed.

Lemma projections_plain_erase_head (x : variable) (slot : function_slot)
    (o : function_slot) (os : list function_slot) (e : expr) :
  projections_plain x slot (o :: os) e = true ->
  after_local_allocation x (erase [] e) = erase [] e.
Proof.
  intros H. destruct e as [y k d b | | | | | | | ]; try reflexivity.
  destruct (region_prim_of d) as [[p args] | ] eqn:Hrp.
  - apply region_prim_of_some in Hrp. subst d. discriminate H.
  - rewrite (erase_let_plain _ _ _ _ _ Hrp).
    destruct d as [ | [ | | | | ] | | | | | ]; try reflexivity.
    discriminate Hrp.
Qed.

Lemma set_entry_erase (x : variable) (d : named) (body : expr) :
  set_plain x d body = true ->
  match d with
  | Closure_of_set slot decls _ =>
    [(x, projected_set_closures x slot decls (erase [] body))]
  | _ => []
  end
  = match d with
    | Closure_of_set slot decls _ =>
      [(x, projected_set_closures x slot decls body)]
    | _ => []
    end.
Proof.
  intros H. destruct d; try reflexivity. cbn [set_plain] in H.
  rewrite !projected_set_closures_other.
  destruct (FlambdaReaperRebuild.other_slots slot function_decls)
    as [ | o os] eqn:Ho; [ rewrite !let_projections_of_nil; reflexivity | ].
  rewrite (after_local_allocation_plain _ _ _ _ _ H),
    (projections_plain_erase_head _ _ _ _ _ H).
  rewrite (let_projections_of_erase _ _ _ _ H). reflexivity.
Qed.

Lemma set_closures_erase :
  (∀ e : expr, ∀ kinds,
     regions_plain kinds e = true -> set_closures (erase [] e) = set_closures e)
  ∧ (∀ h : continuation_handler, ∀ kinds,
       handler_regions_plain kinds h = true ->
       handler_set_closures (erase_handler [] h) = handler_set_closures h).
Proof.
  apply regions_ind.
  - intros x kind d body IH IH2 kinds Hpl.
    destruct (region_prim_of d) as [[p args] | ] eqn:Hrp.
    + apply region_prim_of_some in Hrp. subst d.
      destruct p; destruct args as [ | s [ | s' args]];
        cbn [regions_plain] in Hpl; try discriminate Hpl;
        apply andb_true_iff in Hpl as [_ Hb];
        cbn [erase erase_region_primitive set_closures app];
        rewrite (IH _ Hb); reflexivity.
    + rewrite (regions_plain_let _ _ _ _ _ Hrp) in Hpl.
      apply andb_true_iff in Hpl as [Hpl Hla].
      apply andb_true_iff in Hpl as [_ Hset].
      rewrite (erase_let_plain _ _ _ _ _ Hrp).
      unfold la_check in Hla.
      destruct (la_body body) as [[[[y k'] args] body'] | ] eqn:Hlab.
      * apply la_body_some in Hlab. subst body.
        apply andb_true_iff in Hla as [Hla Hb'].
        apply andb_true_iff in Hla as [Hla Hargs].
        apply andb_true_iff in Hla as [Hla _].
        apply andb_true_iff in Hla as [_ Hy]. apply Pos.eqb_eq in Hy.
        subst y.
        destruct args as [ | [x' | | ] rs]; try discriminate Hargs.
        apply Pos.eqb_eq in Hargs. subst x'.
        cbn [erase erase_region_primitive]. rewrite Pos.eqb_refl.
        cbn [set_closures app].
        rewrite (IH2 _ _ _ _ eq_refl _ Hb').
        destruct d; try reflexivity. cbn [set_plain] in Hset.
        rewrite !projected_set_closures_other.
        destruct (FlambdaReaperRebuild.other_slots slot function_decls);
          [ rewrite !let_projections_of_nil; reflexivity
          | discriminate Hset ].
      * cbn [set_closures]. rewrite (IH _ Hla).
        f_equal. exact (set_entry_erase x d body Hset).
  - intros bindings body Hcodes IH kinds Hpl.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hbs Hb].
    rewrite forallb_forall in Hbs.
    cbn [erase set_closures]. rewrite (IH _ Hb). f_equal.
    rewrite flat_map_concat_map, map_map, flat_map_concat_map. f_equal.
    apply map_ext_in. intros bd Hin. destruct bd as [id c | | ];
      try reflexivity.
    destruct c as [params mc ret exn md cbody].
    cbn [erase_binding erase_code binding_set_closures code_set_closures].
    specialize (Hbs _ Hin).
    cbn [binding_regions_plain code_regions_plain] in Hbs.
    apply andb_true_iff in Hbs as [_ Hcb].
    exact (Hcodes id _ Hin _ Hcb).
  - intros k h body IHh IH kinds Hpl.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hh Hb].
    pose proof (handler_kinds_plain _ _ Hh) as Hhk.
    cbn [erase set_closures].
    rewrite (erase_plain _ body
               (conts_plain_cons _ _ _ (region_places_plain _ Hhk)
                  conts_plain_nil)).
    rewrite (IHh _ Hh), (IH _ Hb). reflexivity.
  - intros handlers body IHh IH kinds Hpl.
    cbn [regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hhs Hb].
    rewrite forallb_forall in Hhs.
    assert (Hplain : ∀ k h, (k, h) ∈ handlers ->
                       forallb kind_not_region (handler_kinds h) = true)
      by (intros k h Hin; exact (handler_kinds_plain _ _ (Hhs (k, h) Hin))).
    cbn [erase set_closures].
    rewrite (erase_plain _ body (rec_conts_plain handlers Hplain)),
      (IH _ Hb).
    f_equal. rewrite flat_map_concat_map, map_map, flat_map_concat_map.
    f_equal. apply map_ext_in. intros [k h] Hin. cbn beta iota.
    rewrite (erase_handler_plain _ h (rec_conts_plain handlers Hplain)).
    exact (IHh k h Hin _ (Hhs (k, h) Hin)).
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params md hbody IH kinds Hpl.
    cbn [handler_regions_plain] in Hpl. apply andb_true_iff in Hpl as [Hk Hb].
    rewrite (erase_handler_no_region [] params md hbody Hk).
    cbn [handler_set_closures]. exact (IH _ Hb).
Qed.

(* The facts of the two traversals of a unit. *)
Lemma traverse_unit_rel (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  regions_plain [] e = true ->
  ∃ l l' : list fact,
    traverse_unit flag u e = facts_of_list l
    ∧ traverse_unit flag u (erase [] e) = facts_of_list l'
    ∧ facts_rel l l'.
Proof.
  intros Hpl. unfold traverse_unit.
  rewrite (proj1 projected_value_slots_erase e []).
  rewrite (proj1 (static_names_erase all_code_ids_of_binding
                    ltac:(intros [ | | ]; reflexivity)) e []).
  rewrite (proj1 (static_names_erase symbols_of_binding
                    ltac:(intros [ | | ]; reflexivity)) e []).
  rewrite (proj1 set_closures_erase e [] Hpl).
  cbn [fresh fst snd].
  eexists _, _. split; [ reflexivity | ]. split; [ reflexivity | ].
  apply (acc_rel_frame _ (fr_resolve _)).
  apply (acc_rel_frame _ (fr_add_fact _)).
  apply (proj1 traverse_rel e [] _ _ _ Hpl), acc_rel_refl.
Qed.

(* ======================================================================== *)
(* Nothing is put into the constants                                        *)
(* ======================================================================== *)

Definition not_consts (n : node) : Prop := n <> Node_all_constants.

(* The places of a fact where something is put, or that a rule reads
   with [any_usage] or [has_usage] to put something elsewhere, are not
   the constants. *)
Definition pos_ok (fct : fact) : Prop :=
  match fct with
  | Fact_alias to_ _ | Fact_use to_ _ | Fact_accessor to_ _ _
  | Fact_alias_if_any_source _ to_ _ => not_consts to_
  | Fact_constructor base _ _ => not_consts base
  | Fact_parameter base _ to_ => not_consts base ∧ not_consts to_
  | Fact_propagate if_used to_ _ => not_consts if_used ∧ not_consts to_
  | Fact_any_source_if_used if_used _ => not_consts if_used
  | Fact_argument _ _ _ | Fact_any_usage _ | Fact_any_source _ => True
  end.

Definition cd_ok (cd : code_dep) : Prop :=
  Forall not_consts (cd_params cd) ∧ not_consts (cd_my_closure cd)
  ∧ Forall not_consts (cd_return cd) ∧ not_consts (cd_exn cd)
  ∧ not_consts (cd_known_arity_call_witness cd)
  ∧ Forall not_consts (cd_unknown_arity_call_witnesses cd).

Definition pinv (a : acc) : Prop :=
  (∀ fct, fct ∈ acc_facts a -> pos_ok fct)
  ∧ (∀ id cd, (id, cd) ∈ acc_code_deps a -> cd_ok cd)
  ∧ (∀ d, d ∈ acc_apply_deps a -> not_consts (ad_apply_call_witness d))
  ∧ (∀ d, d ∈ acc_closure_deps a ->
       not_consts (cld_let_bound_name_of_the_closure d)).

Definition penv (env : traverse_env) : Prop :=
  ∀ k ps, (k, ps) ∈ env_conts env -> Forall not_consts ps.

Lemma nc_var (x : variable) : not_consts (Node_variable x).
Proof. discriminate. Qed.

Lemma nc_sym (s : symbol) : not_consts (Node_symbol s).
Proof. discriminate. Qed.

Lemma nc_code (id : code_id) : not_consts (Node_code_id id).
Proof. discriminate. Qed.

Lemma nc_made (k : positive) : not_consts (Node_made k).
Proof. discriminate. Qed.

Lemma pinv_add_facts (L : list fact) (a : acc) :
  (∀ fct, fct ∈ L -> pos_ok fct) -> pinv a -> pinv (add_facts L a).
Proof.
  intros HL (H1 & H2 & H3 & H4). split; [ | split; [ exact H2 | ] ];
    [ | split; assumption ].
  intros fct Hin. apply in_app_or in Hin as [Hin | Hin];
    [ exact (HL _ Hin) | exact (H1 _ Hin) ].
Qed.

Lemma pinv_add_fact (fct : fact) (a : acc) :
  pos_ok fct -> pinv a -> pinv (add_fact fct a).
Proof. intros H. apply pinv_add_facts. intros f [<- | []]. exact H. Qed.

Lemma pinv_fresh (a : acc) :
  pinv a -> pinv (snd (fresh a)) ∧ not_consts (fst (fresh a)).
Proof. intros H. split; [ exact H | apply nc_made ]. Qed.

Lemma pinv_fresh_list (n : nat) :
  ∀ a, pinv a ->
    pinv (snd (fresh_list n a)) ∧ Forall not_consts (fst (fresh_list n a)).
Proof.
  induction n as [ | n IH]; intros a H; [ split; [ exact H | constructor ] | ].
  cbn [fresh_list]. destruct (fresh a) as [x a1] eqn:E.
  destruct (pinv_fresh a H) as [H1 Hx]. rewrite E in H1, Hx. cbn in H1, Hx.
  destruct (IH a1 H1) as [H2 Hxs].
  destruct (fresh_list n a1) as [xs a2]. cbn in *. split; [ exact H2 | ].
  constructor; assumption.
Qed.

Lemma pinv_note_simple (s : simple) (a : acc) :
  pinv a -> pinv (note_simple s a).
Proof. destruct s; intros H; exact H. Qed.

Lemma pinv_note_simples (ss : list simple) :
  ∀ a, pinv a -> pinv (note_simples ss a).
Proof.
  unfold note_simples. induction ss as [ | s ss IH]; intros a H;
    [ exact H | ].
  cbn [List.fold_left]. apply IH, pinv_note_simple, H.
Qed.

Lemma pinv_cond_any_usage (env : traverse_env) (s : simple) (a : acc) :
  pinv a -> pinv (cond_any_usage env s a).
Proof.
  intros H. unfold cond_any_usage.
  destruct (env_current_code_id env); apply pinv_add_fact;
    try exact I; try apply nc_code; apply pinv_note_simple, H.
Qed.

Lemma pinv_cond_alias (env : traverse_env) (from to_ : node) (a : acc) :
  not_consts to_ -> pinv a -> pinv (cond_alias env from to_ a).
Proof.
  intros Ht H. unfold cond_alias.
  destruct (env_current_code_id env); apply pinv_add_fact; try exact H;
    [ split; [ apply nc_code | exact Ht ] | exact Ht ].
Qed.

Lemma pinv_find_cont (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  penv env -> pinv a ->
  pinv (snd (find_cont env k n a))
  ∧ Forall not_consts (fst (find_cont env k n a)).
Proof.
  intros He H. unfold find_cont.
  destruct (lookup k (env_conts env)) as [ps | ] eqn:Hl.
  - split; [ exact H | ]. apply (He k). exact (lookup_in _ _ _ Hl).
  - destruct (pinv_fresh_list n a H) as [H1 Hps].
    destruct (fresh_list n a) as [ps a1]. cbn in *.
    split; [ | exact Hps ]. apply pinv_add_facts; [ | exact H1 ].
    intros fct Hin. apply in_map_iff in Hin as [p [<- _]]. exact I.
Qed.

(* Facts from lists, broken into their elements. *)
Ltac into_facts :=
  let fct := fresh "fct" in let Hin := fresh "Hin" in
  intros fct Hin;
  repeat match goal with
  | H : In _ (_ ++ _) |- _ => apply in_app_or in H; destruct H as [H | H]
  | H : In _ (map _ _) |- _ =>
    let y := fresh "y" in let Heq := fresh "Heq" in let Hy := fresh "Hy" in
    apply in_map_iff in H; destruct H as [y [Heq Hy]]; subst fct;
    repeat match goal with p : (_ * _)%type |- _ => destruct p end
  | H : In _ (_ :: _) |- _ => destruct H as [H | H]; [ subst fct | ]
  | H : In _ [] |- _ => destruct H
  end;
  cbn [pos_ok];
  repeat split; try exact I; try apply nc_var; try apply nc_sym;
  try apply nc_code; try apply nc_made; try assumption.

Lemma Forall_in {A : Type} (P : A -> Prop) (l : list A) (x : A) :
  Forall P l -> x ∈ l -> P x.
Proof. intros H Hx. rewrite Forall_forall in H. exact (H x Hx). Qed.

Lemma in_indexed {A : Type} (i : nat) (x : A) (l : list A) :
  (i, x) ∈ indexed l -> x ∈ l.
Proof. unfold indexed. intros H. exact (in_combine_r _ _ _ _ H). Qed.

Lemma pinv_block_deps (base : node) (fields : list simple) (a : acc) :
  not_consts base -> pinv a -> pinv (block_deps base fields a).
Proof.
  intros Hb H. unfold block_deps. apply pinv_add_facts;
    [ into_facts | apply pinv_note_simples, H ].
Qed.

Lemma pinv_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  pinv a -> pinv (traverse_prim env x p args a).
Proof.
  intros H. unfold traverse_prim.
  repeat match goal with
  | |- context [ match ?t with _ => _ end ] => destruct t
  end;
    repeat first
      [ apply pinv_block_deps; [ apply nc_var | ]
      | apply pinv_add_fact; [ cbn [pos_ok]; apply nc_var | ]
      | apply pinv_add_fact; [ cbn [pos_ok]; exact I | ]
      | apply pinv_add_facts; [ into_facts | ]
      | apply pinv_note_simple
      | apply pinv_cond_any_usage
      | exact H ].
Qed.

Lemma pinv_record_set (env : traverse_env)
    (closures : list (node * option code_id))
    (function_slots : list (function_slot * node))
    (value_slots : list (value_slot * simple)) (a : acc) :
  (∀ b c, (b, c) ∈ closures -> not_consts b) -> pinv a ->
  pinv (record_set_of_closures_deps env closures function_slots value_slots
          a).
Proof.
  intros Hc. unfold record_set_of_closures_deps. revert a.
  induction closures as [ | [b c] closures IH]; intros a H; [ exact H | ].
  cbn [List.fold_left]. apply IH;
    [ intros b' c' Hin; apply (Hc b' c'); right; exact Hin | ].
  pose proof (Hc b c (or_introl eq_refl)) as Hb.
  apply pinv_add_facts; [ into_facts | apply pinv_note_simples ].
  destruct c; [ | exact H ].
  destruct H as (H1 & H2 & H3 & H4). split; [ exact H1 | ].
  split; [ exact H2 | ]. split; [ exact H3 | ].
  intros d [<- | Hd]; [ exact Hb | exact (H4 d Hd) ].
Qed.

Lemma pinv_closures_of_set (x : variable) (slot : function_slot)
    (others : list (function_slot * variable)) :
  ∀ (decls : list (function_slot * code_id)) (a : acc),
    pinv a ->
    pinv (snd (closures_of_set x slot others decls a))
    ∧ ∀ f n id, (f, n, id) ∈ fst (closures_of_set x slot others decls a) ->
        not_consts n.
Proof.
  induction decls as [ | [f id] decls IH]; intros a H;
    [ split; [ exact H | intros ? ? ? [] ] | ].
  cbn [closures_of_set].
  assert (Hstep : ∀ (n : node) (a1 : acc),
            not_consts n -> pinv a1 ->
            pinv (snd (let '(rest, a2) := closures_of_set x slot others
                                            decls a1 in
                       ((f, n, id) :: rest, a2)))
            ∧ ∀ f' n' id',
                (f', n', id') ∈ fst (let '(rest, a2) :=
                                       closures_of_set x slot others decls
                                         a1 in
                                     ((f, n, id) :: rest, a2)) ->
                not_consts n').
  { intros n a1 Hn H1. destruct (IH a1 H1) as [H2 H3].
    destruct (closures_of_set x slot others decls a1) as [rest a2].
    cbn in *. split; [ exact H2 | ].
    intros f' n' id' [Heq | Hin];
      [ injection Heq as _ <- _; exact Hn | exact (H3 _ _ _ Hin) ]. }
  destruct (Pos.eqb f slot); [ apply Hstep; [ apply nc_var | exact H ] | ].
  destruct (lookup f others); [ apply Hstep; [ apply nc_var | exact H ] | ].
  destruct (pinv_fresh a H) as [H1 Hn]. destruct (fresh a) as [n a1].
  apply Hstep; assumption.
Qed.

Lemma pinv_traverse_closure_of_set (env : traverse_env) (x : variable)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (a : acc) :
  pinv a -> pinv (traverse_closure_of_set env x slot decls value_slots a).
Proof.
  intros H. unfold traverse_closure_of_set.
  destruct (among slot (map fst (compact decls))); [ | exact H ].
  destruct (pinv_closures_of_set x slot (set_closures_of env x)
              (compact decls) a H) as [H1 Hn].
  destruct (closures_of_set _ _ _ _ a) as [cls a1]. cbn in *.
  apply pinv_record_set; [ | exact H1 ].
  intros b c Hin. apply in_map_iff in Hin as [[[f n] id] [Heq Hin]].
  injection Heq as <- _. exact (Hn _ _ _ Hin).
Qed.

Lemma pinv_traverse_named (env : traverse_env) (x : variable) (d : named)
    (a : acc) :
  pinv a -> pinv (traverse_named env x d a).
Proof.
  intros H. destruct d; cbn [traverse_named].
  - apply pinv_add_fact; [ apply nc_var | apply pinv_note_simple, H ].
  - apply pinv_traverse_prim, H.
  - apply pinv_record_set; [ | exact H ].
    intros b c [Heq | []]. injection Heq as <- _. apply nc_var.
  - apply pinv_traverse_closure_of_set, H.
  - apply pinv_add_fact; [ apply nc_var | apply pinv_note_simple, H ].
  - apply pinv_add_fact; [ apply nc_var | apply pinv_note_simple, H ].
  - apply pinv_add_fact; [ apply nc_var | apply pinv_note_simple, H ].
Qed.

Lemma pinv_traverse_static_binding (env : traverse_env) (b : static_binding)
    (a : acc) :
  pinv a -> pinv (traverse_static_binding env b a).
Proof.
  intros H. destruct b as [ | | s const]; try exact H.
  cbn [traverse_static_binding].
  destruct const as [ | slot syms decls vs | | contents | | elements].
  - apply pinv_record_set; [ | exact H ].
    intros b c [Heq | []]. injection Heq as <- _. apply nc_sym.
  - apply pinv_record_set; [ | exact H ].
    intros b c [Heq | []]. injection Heq as <- _. apply nc_sym.
  - apply pinv_block_deps; [ apply nc_sym | exact H ].
  - apply pinv_add_facts; [ | exact H ]. unfold boxed_number_facts.
    destruct (boxed_number_kind contents); [ into_facts | ].
    destruct contents; into_facts.
  - apply pinv_add_fact; [ apply nc_sym | exact H ].
  - destruct elements;
      [ apply pinv_add_fact; [ apply nc_sym | exact H ]
      | apply pinv_block_deps; [ apply nc_sym | exact H ] ].
Qed.

Lemma group_by_sizes_in {A : Type} (sizes : list nat) :
  ∀ (l g : list A) (x : A),
    g ∈ group_by_sizes sizes l -> x ∈ g -> x ∈ l.
Proof.
  induction sizes as [ | n sizes IH]; intros l g x Hg Hx; [ destruct Hg | ].
  cbn [group_by_sizes] in Hg. rewrite <- (firstn_skipn n l).
  apply in_or_app. destruct Hg as [<- | Hg]; [ left; exact Hx | ].
  right. exact (IH _ _ _ Hg Hx).
Qed.

Lemma groups_in {A : Type} (a : arity) (l g : list A) (x : A) :
  g ∈ at_least_one_group (group_by_parameter a l) -> x ∈ g -> x ∈ l.
Proof.
  unfold at_least_one_group, group_by_parameter. intros Hg Hx.
  destruct (group_by_sizes _ l) as [ | g0 gs] eqn:E.
  - destruct Hg as [<- | []]. destruct Hx.
  - rewrite <- E in Hg. exact (group_by_sizes_in _ _ _ _ Hg Hx).
Qed.

Lemma pinv_create_known (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  Forall not_consts params -> pinv a ->
  pinv (snd (create_known_arity_call_witness id params returns exn a))
  ∧ not_consts (fst (create_known_arity_call_witness id params returns exn a)).
Proof.
  intros Hp H. unfold create_known_arity_call_witness.
  destruct (fresh a) as [w a1] eqn:E.
  assert (Hw : not_consts w) by (injection E as <- _; apply nc_made).
  assert (H1 : pinv a1) by (injection E as _ <-; exact H).
  cbn [fst snd]. split; [ | exact Hw ].
  apply pinv_add_facts; [ | exact H1 ]. into_facts.
  apply (Forall_in _ _ _ Hp). exact (in_indexed _ _ _ Hy).
Qed.

Lemma pinv_witness_deps (id : code_id) (returns : list node) (exn : node) :
  ∀ (pws : list (list node * node)) (a : acc),
    (∀ g w, (g, w) ∈ pws -> not_consts w ∧ Forall not_consts g) ->
    pinv a -> pinv (unknown_arity_witness_deps id returns exn pws a).
Proof.
  induction pws as [ | [first w] pws IH]; intros a Hpws H; [ exact H | ].
  destruct (Hpws first w (or_introl eq_refl)) as [Hw Hfirst].
  cbn [unknown_arity_witness_deps].
  assert (H1 : pinv (add_facts
                       (map (fun '(i, p) => Fact_parameter w i p)
                          (indexed first)
                        ++ [Fact_constructor w Field_code_id_of_call_witness
                              (Node_code_id id)]) a)).
  { apply pinv_add_facts; [ | exact H ]. into_facts.
    apply (Forall_in _ _ _ Hfirst). exact (in_indexed _ _ _ Hy). }
  destruct pws as [ | [first' w'] pws'].
  - apply pinv_add_facts; [ into_facts | exact H1 ].
  - destruct (fresh _) as [v a1] eqn:E.
    assert (Hv : not_consts v) by (injection E as <- _; apply nc_made).
    assert (H2 : pinv a1) by (injection E as _ <-; exact H1).
    apply IH; [ intros g w0 Hin; apply (Hpws g w0); right; exact Hin | ].
    apply pinv_add_facts; [ into_facts | exact H2 ].
Qed.

Lemma pinv_create_unknown (id : code_id) (is_tupled : bool)
    (params_arity : arity) (params returns : list node) (exn : node)
    (a : acc) :
  Forall not_consts params -> pinv a ->
  pinv (snd (create_unknown_arity_call_witnesses id is_tupled params_arity
               params returns exn a))
  ∧ Forall not_consts
      (fst (create_unknown_arity_call_witnesses id is_tupled params_arity
              params returns exn a)).
Proof.
  intros Hp H. unfold create_unknown_arity_call_witnesses. destruct is_tupled.
  - unfold create_unknown_arity_tupled_call_witnesses.
    destruct (fresh a) as [w a1] eqn:E1.
    destruct (fresh a1) as [u a2] eqn:E2.
    destruct (fresh a2) as [k a3] eqn:E3.
    assert (Hw : not_consts w) by (injection E1 as <- _; apply nc_made).
    assert (Hu : not_consts u) by (injection E2 as <- _; apply nc_made).
    assert (Hk : not_consts k) by (injection E3 as <- _; apply nc_made).
    assert (H3 : pinv a3)
      by (injection E1 as _ <-; injection E2 as _ <-; injection E3 as _ <-;
          exact H).
    cbn [fst snd]. split; [ | constructor; [ exact Hw | constructor ] ].
    apply pinv_add_facts; [ | exact H3 ]. into_facts.
    apply (Forall_in _ _ _ Hp). exact (in_indexed _ _ _ Hy).
  - unfold create_unknown_arity_non_tupled_call_witnesses.
    destruct (pinv_fresh_list (length (at_least_one_group
                                        (group_by_parameter params_arity
                                           params))) a H) as [H1 Hws].
    destruct (fresh_list _ a) as [ws a1]. cbn [fst snd] in *.
    split; [ | exact Hws ].
    apply pinv_witness_deps; [ | exact H1 ].
    intros g w Hin. split.
    + exact (Forall_in _ _ _ Hws (in_combine_r _ _ _ _ Hin)).
    + apply Forall_forall. intros x Hx.
      apply (Forall_in _ _ _ Hp).
      exact (groups_in _ _ _ _ (in_combine_l _ _ _ _ Hin) Hx).
Qed.

Lemma pinv_prepare_code (id : code_id) (c : code) (a : acc) :
  pinv a -> pinv (prepare_code id c a).
Proof.
  intros H. unfold prepare_code.
  destruct (pinv_fresh_list (length (code_result_kinds c)) a H) as [H1 Hrs].
  destruct (fresh_list _ a) as [rs a1]. cbn [fst snd] in *.
  destruct (pinv_fresh a1 H1) as [H2 Hex].
  destruct (fresh a1) as [ex a2]. cbn [fst snd] in *.
  destruct (pinv_fresh a2 H2) as [H3 Hmc].
  destruct (fresh a2) as [mc a3]. cbn [fst snd] in *.
  destruct (pinv_fresh_list (length (code_params_kinds c)) a3 H3)
    as [H4 Hps].
  destruct (fresh_list _ a3) as [ps a4]. cbn [fst snd] in *.
  destruct (pinv_create_known id ps rs ex a4 Hps H4) as [H5 Hkw].
  destruct (create_known_arity_call_witness id ps rs ex a4) as [kw a5].
  cbn [fst snd] in *.
  destruct (pinv_create_unknown id (code_is_tupled c) (code_params_arity c)
              ps rs ex a5 Hps H5) as [H6 Huw].
  destruct (create_unknown_arity_call_witnesses _ _ _ _ _ _ a5) as [uw a6].
  cbn [fst snd] in *.
  assert (H7 : pinv (add_fact (Fact_any_source (Node_code_id id)) a6))
    by (apply pinv_add_fact; [ exact I | exact H6 ]).
  destruct H7 as (G1 & G2 & G3 & G4). split; [ exact G1 | ].
  split; [ | split; assumption ].
  intros id' cd [Heq | Hin]; [ | exact (G2 _ _ Hin) ].
  injection Heq as _ <-. repeat split; assumption.
Qed.

Lemma pinv_prepare_binding (b : static_binding) (a : acc) :
  pinv a -> pinv (prepare_binding b a).
Proof.
  destruct b; intros H; [ apply pinv_prepare_code, H | exact H | exact H ].
Qed.

Lemma pinv_make_known (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  Forall not_consts returns -> not_consts exn -> pinv a ->
  pinv (snd (make_known_arity_apply_widget env ap returns exn a))
  ∧ not_consts (fst (make_known_arity_apply_widget env ap returns exn a)).
Proof.
  intros Hr He H. unfold make_known_arity_apply_widget.
  destruct (fresh a) as [w a1] eqn:E1.
  destruct (fresh a1) as [c a2] eqn:E2.
  destruct (fresh a2) as [n a3] eqn:E3.
  assert (Hw : not_consts w) by (injection E1 as <- _; apply nc_made).
  assert (Hc : not_consts c) by (injection E2 as <- _; apply nc_made).
  assert (Hn : not_consts n) by (injection E3 as <- _; apply nc_made).
  assert (H3 : pinv a3)
    by (injection E1 as _ <-; injection E2 as _ <-; injection E3 as _ <-;
        exact H).
  cbn [fst snd]. split; [ | exact Hn ].
  apply pinv_cond_alias; [ exact Hw | ].
  apply pinv_add_facts; [ | apply pinv_note_simples, H3 ]. into_facts.
  apply (Forall_in _ _ _ Hr). exact (in_indexed _ _ _ Hy).
Qed.

Lemma pinv_apply_deps (returns : list node) (exn called : node) :
  ∀ (aws : list (list simple * node)) (a : acc),
    Forall not_consts returns -> not_consts exn -> not_consts called ->
    (∀ g w, (g, w) ∈ aws -> not_consts w) ->
    pinv a -> pinv (unknown_arity_apply_deps returns exn called aws a).
Proof.
  induction aws as [ | [first w] aws IH]; intros a Hr He Hc Haws H;
    [ exact H | ].
  pose proof (Haws first w (or_introl eq_refl)) as Hw.
  cbn [unknown_arity_apply_deps].
  assert (H1 : pinv (add_facts
                       (map (fun '(i, s) => Fact_argument (node_of_simple s)
                                              i w) (indexed first)
                        ++ [Fact_accessor exn (Field_return_of_call Exn) w;
                            Fact_accessor called Field_code_id_of_call_witness
                              w]) a))
    by (apply pinv_add_facts; [ into_facts | exact H ]).
  destruct aws as [ | [first' w'] aws'].
  - apply pinv_add_facts; [ | exact H1 ]. into_facts.
    apply (Forall_in _ _ _ Hr). exact (in_indexed _ _ _ Hy).
  - destruct (fresh _) as [v a1] eqn:E.
    assert (Hv : not_consts v) by (injection E as <- _; apply nc_made).
    assert (H2 : pinv a1) by (injection E as _ <-; exact H1).
    apply IH; try assumption;
      [ intros g w0 Hin; apply (Haws g w0); right; exact Hin | ].
    apply pinv_add_facts; [ | exact H2 ]. into_facts.
    exact (Haws first' w' (or_intror (or_introl eq_refl))).
Qed.

Lemma pinv_make_unknown (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  Forall not_consts returns -> not_consts exn -> pinv a ->
  pinv (snd (make_unknown_arity_apply_widget env ap returns exn a))
  ∧ not_consts (fst (make_unknown_arity_apply_widget env ap returns exn a)).
Proof.
  intros Hr He H. unfold make_unknown_arity_apply_widget.
  destruct (fresh a) as [c a1] eqn:E1.
  assert (Hc : not_consts c) by (injection E1 as <- _; apply nc_made).
  assert (H1 : pinv (add_fact (Fact_any_usage c) a1))
    by (injection E1 as _ <-; apply pinv_add_fact; [ exact I | exact H ]).
  destruct (pinv_fresh_list (length (at_least_one_group
                                      (group_by_parameter (ap_args_arity ap)
                                         (ap_args ap))))
              _ H1) as [H2 Hws].
  destruct (fresh_list _ _) as [ws a2]. cbn [fst snd] in *.
  assert (H3 : pinv (unknown_arity_apply_deps returns exn c
                       (combine (at_least_one_group
                                   (group_by_parameter (ap_args_arity ap)
                                      (ap_args ap))) ws)
                       (note_simples (ap_args ap) a2))).
  { apply pinv_apply_deps; try assumption;
      [ | apply pinv_note_simples, H2 ].
    intros g w Hin. exact (Forall_in _ _ _ Hws (in_combine_r _ _ _ _ Hin)). }
  revert H3.
  generalize (unknown_arity_apply_deps returns exn c
                (combine (at_least_one_group
                            (group_by_parameter (ap_args_arity ap)
                               (ap_args ap))) ws)
                (note_simples (ap_args ap) a2)).
  intros a2' H3.
  destruct (fresh a2') as [n a3] eqn:E3.
  assert (Hn : not_consts n) by (injection E3 as <- _; apply nc_made).
  assert (H4 : pinv a3) by (injection E3 as _ <-; exact H3).
  destruct ws as [ | w ws']; cbn [fst snd]; split; try exact Hn;
    [ exact H4 | ].
  apply pinv_cond_alias; [ | exact H4 ].
  exact (Forall_in _ _ _ Hws (or_introl eq_refl)).
Qed.

Lemma pinv_add_apply (env : traverse_env) (id : code_id) (callee widget : node)
    (only : bool) (a : acc) :
  not_consts widget -> pinv a -> pinv (add_apply env id callee widget only a).
Proof.
  intros Hw H. unfold add_apply. destruct only.
  - destruct (fresh a) as [c a1] eqn:E1.
    destruct (fresh a1) as [w a2] eqn:E2.
    assert (Hc : not_consts c) by (injection E1 as <- _; apply nc_made).
    assert (Hw' : not_consts w) by (injection E2 as <- _; apply nc_made).
    assert (H2 : pinv a2)
      by (injection E1 as _ <-; injection E2 as _ <-; exact H).
    assert (H3 : pinv (add_facts
                         [Fact_alias_if_any_source callee c callee;
                          Fact_alias_if_any_source callee widget w] a2))
      by (apply pinv_add_facts; [ into_facts | exact H2 ]).
    destruct H3 as (G1 & G2 & G3 & G4). split; [ exact G1 | ].
    split; [ exact G2 | ]. split; [ | exact G4 ].
    intros d [<- | Hd]; [ exact Hw' | exact (G3 _ Hd) ].
  - destruct H as (G1 & G2 & G3 & G4). split; [ exact G1 | ].
    split; [ exact G2 | ]. split; [ | exact G4 ].
    intros d [<- | Hd]; [ exact Hw | exact (G3 _ Hd) ].
Qed.

Lemma pinv_traverse_call_kind (env : traverse_env) (ap : apply_expr)
    (exn : node) (returns : list node) (a : acc) :
  Forall not_consts returns -> not_consts exn -> pinv a ->
  pinv (traverse_call_kind env ap exn returns a).
Proof.
  intros Hr He H. unfold traverse_call_kind.
  pose proof (pinv_note_simple (ap_callee ap) a H) as H0.
  destruct (ap_call_kind ap).
  - destruct (pinv_make_known env ap returns exn _ Hr He H0) as [H1 Hw].
    destruct (make_known_arity_apply_widget _ _ _ _ _) as [w a1].
    cbn [fst snd] in *.
    assert (H2 : pinv (add_fact (Fact_accessor w
                         (Field_call_witness Known_arity_code_pointer)
                         (node_of_simple (ap_callee ap))) a1))
      by (apply pinv_add_fact; [ exact Hw | exact H1 ]).
    destruct (env_should_preserve_direct_calls env);
      try destruct (negb _); try (apply pinv_add_apply; assumption);
      exact H2.
  - destruct (pinv_make_unknown env ap returns exn _ Hr He H0) as [H1 Hw].
    destruct (make_unknown_arity_apply_widget _ _ _ _ _) as [w a1].
    cbn [fst snd] in *. apply pinv_add_fact; [ exact Hw | exact H1 ].
  - destruct (pinv_make_known env ap returns exn _ Hr He H0) as [H1 Hw].
    destruct (make_known_arity_apply_widget _ _ _ _ _) as [w a1].
    cbn [fst snd] in *. apply pinv_add_fact; [ exact Hw | exact H1 ].
Qed.

Lemma pinv_traverse_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  penv env -> pinv a -> pinv (traverse_apply env ap a).
Proof.
  intros He H. unfold traverse_apply.
  destruct (pinv_find_cont env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a He H) as [H1 Hrs].
  destruct (find_cont env _ _ a) as [rs a1]. cbn [fst snd] in *.
  destruct (pinv_find_cont env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) a1 He H1) as [H2 Hps].
  destruct (find_cont env _ _ a1) as [ps a2]. cbn [fst snd] in *.
  destruct ps as [ | p ps].
  - destruct (pinv_fresh a2 H2) as [H3 Hx].
    destruct (fresh a2) as [x a3]. cbn [fst snd] in *.
    apply pinv_traverse_call_kind; assumption.
  - inversion Hps as [ | ? ? Hp Hps']; subst.
    apply pinv_traverse_call_kind; try assumption.
    apply pinv_add_facts; [ | apply pinv_note_simples, H2 ]. into_facts.
    apply (Forall_in _ _ _ Hps'). exact (in_combine_l _ _ _ _ Hy).
Qed.

Lemma pinv_apply_cont_deps (env : traverse_env) (ac : apply_cont) (a : acc) :
  penv env -> pinv a -> pinv (apply_cont_deps env ac a).
Proof.
  intros He H. unfold apply_cont_deps.
  destruct (pinv_find_cont env (ac_continuation ac) (length (ac_args ac)) a
              He H) as [H1 Hps].
  destruct (find_cont env _ _ a) as [ps a1]. cbn [fst snd] in *.
  apply pinv_add_facts; [ | apply pinv_note_simples, H1 ]. into_facts.
  apply (Forall_in _ _ _ Hps). exact (in_combine_l _ _ _ _ Hy).
Qed.

Lemma pinv_traverse_switch (env : traverse_env) (s : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  penv env -> pinv a -> pinv (traverse_switch env s arms a).
Proof.
  intros He H. unfold traverse_switch.
  generalize (pinv_cond_any_usage env s a H).
  generalize (cond_any_usage env s a).
  induction (Keyed.compact Z.eqb arms) as [ | [i ac] arms' IH];
    intros a' H'; [ exact H' | ].
  cbn [List.fold_left]. apply IH, pinv_apply_cont_deps; assumption.
Qed.

Lemma pinv_exn_handler_deps (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  pinv a -> pinv (exn_handler_deps env h a).
Proof.
  intros H. destruct h as [[ | p ps] md hbody]; cbn [exn_handler_deps];
    [ exact H | ].
  destruct (hm_is_exn_handler md); [ | exact H ].
  apply pinv_cond_any_usage, pinv_add_fact; [ exact I | exact H ].
Qed.

Lemma penv_add_cont (env : traverse_env) (k : continuation)
    (params : list variable) :
  penv env -> penv (add_cont env k (map Node_variable params)).
Proof.
  intros He k' ps [Heq | Hin]; [ | exact (He k' ps Hin) ].
  injection Heq as _ <-. apply Forall_forall. intros x Hx.
  apply in_map_iff in Hx as [v [<- _]]. apply nc_var.
Qed.

Lemma penv_add_rec_conts (env : traverse_env)
    (hs : list (continuation * continuation_handler)) :
  penv env -> penv (add_rec_conts env hs).
Proof.
  induction hs as [ | [k h] hs IH]; intros He; [ exact He | ].
  cbn [add_rec_conts fold_right]. destruct h as [params md hbody].
  apply penv_add_cont. exact (IH He).
Qed.

Lemma penv_code_env (env : traverse_env) (id : code_id)
    (ret exn : continuation) (cd : code_dep) :
  cd_ok cd -> penv (code_env env id ret exn cd).
Proof.
  intros (H1 & H2 & H3 & H4 & H5 & H6) k ps Hin.
  cbn [code_env env_conts] in Hin.
  destruct Hin as [Heq | [Heq | []]]; injection Heq as _ <-.
  - constructor; [ exact H4 | constructor ].
  - exact H3.
Qed.

Lemma pinv_fold {A : Type} (f : acc -> A -> acc) (xs : list A) :
  (∀ x a, x ∈ xs -> pinv a -> pinv (f a x)) ->
  ∀ a, pinv a -> pinv (List.fold_left f xs a).
Proof.
  induction xs as [ | x xs IH]; intros H a Ha; [ exact Ha | ].
  cbn [List.fold_left]. apply IH; [ | apply H; [ left; reflexivity | ] ];
    [ intros y b Hy; apply H; right; exact Hy | exact Ha ].
Qed.

Lemma pinv_traverse :
  (∀ e : expr, ∀ env a, penv env -> pinv a -> pinv (traverse env e a))
  ∧ (∀ h : continuation_handler, ∀ env a,
       penv env -> pinv a -> pinv (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH env a He H. cbn [traverse].
    apply IH, pinv_traverse_named; assumption.
  - intros bindings body Hcodes IH env a He H. cbn [traverse].
    apply IH; [ exact He | ].
    assert (H1 : pinv (List.fold_left (fun a b => prepare_binding b a)
                         bindings a)).
    { clear Hcodes. revert a H. induction bindings as [ | b bs IHb];
        intros a H; [ exact H | ].
      cbn [List.fold_left]. apply IHb, pinv_prepare_binding, H. }
    assert (H2 : ∀ a0, pinv a0 ->
               pinv (List.fold_left
                       (fun a b => traverse_static_binding env b a)
                       bindings a0)).
    { clear Hcodes H1. induction bindings as [ | b bs IHb];
        intros a0 H0; [ exact H0 | ].
      cbn [List.fold_left]. apply IHb, pinv_traverse_static_binding, H0. }
    specialize (H2 _ H1). revert H2.
    generalize (List.fold_left (fun a b => traverse_static_binding env b a)
                  bindings
                  (List.fold_left (fun a b => prepare_binding b a)
                     bindings a)).
    clear H1.
    induction bindings as [ | b bs IHb]; intros a0 H0; [ exact H0 | ].
    cbn [List.fold_left]. apply IHb;
      [ intros id c Hin; apply (Hcodes id c); right; exact Hin | ].
    destruct b as [id c | | ]; try exact H0.
    destruct c as [params mc ret exn md cbody].
    cbn [traverse_binding traverse_code].
    destruct (lookup id (acc_code_deps a0)) as [cd | ] eqn:Hl;
      [ | exact H0 ].
    destruct H0 as (G1 & G2 & G3 & G4).
    assert (Hcd : cd_ok cd) by exact (G2 id cd (lookup_in _ _ _ Hl)).
    apply (Hcodes id _ (or_introl eq_refl)).
    + apply penv_code_env, Hcd.
    + apply pinv_add_facts; [ | exact (conj G1 (conj G2 (conj G3 G4))) ].
      into_facts.
  - intros k h body IHh IH env a He H. cbn [traverse].
    destruct h as [params md hbody]. cbn [handler_params].
    apply IH; [ apply penv_add_cont, He | ].
    apply pinv_exn_handler_deps, IHh; assumption.
  - intros handlers body IHh IH env a He H. cbn [traverse].
    pose proof (penv_add_rec_conts env handlers He) as He'.
    apply IH; [ exact He' | ].
    apply pinv_fold; [ | exact H ].
    intros [k h] a' Hin Ha'. exact (IHh k h Hin _ _ He' Ha').
  - intros ap env a He H. apply pinv_traverse_apply; assumption.
  - intros ac env a He H. apply pinv_apply_cont_deps; assumption.
  - intros s arms env a He H. apply pinv_traverse_switch; assumption.
  - intros env a _ H. exact H.
  - intros params md hbody IH env a He H. apply IH; assumption.
Qed.

Lemma pinv_add_closure_dep (cds : finite_map code_dep) (d : closure_dep)
    (a : acc) :
  (∀ id cd, (id, cd) ∈ cds -> cd_ok cd) ->
  not_consts (cld_let_bound_name_of_the_closure d) -> pinv a ->
  pinv (add_closure_dep cds d a).
Proof.
  intros Hcds Hd H. unfold add_closure_dep.
  destruct (lookup _ cds) as [cd | ] eqn:Hl.
  - destruct (Hcds _ _ (lookup_in _ _ _ Hl)) as (C1 & C2 & C3 & C4 & C5 & C6).
    apply pinv_add_facts; [ | exact H ].
    destruct (cd_unknown_arity_call_witnesses cd); into_facts.
  - destruct (fresh a) as [w a1] eqn:E.
    assert (Hw : not_consts w) by (injection E as <- _; apply nc_made).
    assert (H1 : pinv a1) by (injection E as _ <-; exact H).
    apply pinv_add_facts; [ into_facts | exact H1 ].
Qed.

Lemma pinv_resolve (syms : list symbol) (a : acc) :
  pinv a -> pinv (resolve_delayed_deps syms a).
Proof.
  intros H. pose proof H as (H1 & H2 & H3 & H4). unfold resolve_delayed_deps.
  apply pinv_add_facts; [ into_facts | ].
  assert (Ha : pinv (add_facts
                       (flat_map (apply_dep_facts (acc_code_deps a))
                          (acc_apply_deps a)) a)).
  { apply pinv_add_facts; [ | exact H ].
    intros fct Hin. apply in_flat_map in Hin as [d [Hd Hin]].
    pose proof (H3 d Hd) as Hw. unfold apply_dep_facts in Hin.
    destruct (lookup (ad_apply_code_id d) (acc_code_deps a)) as [cd | ]
      eqn:Hl.
    - destruct (H2 _ _ (lookup_in _ _ _ Hl))
        as (C1 & C2 & C3 & C4 & C5 & C6).
      unfold alias_for_caller in Hin.
      destruct (ad_function_containing_apply_expr d);
        destruct Hin as [<- | [<- | []]]; cbn [pos_ok];
        repeat split; first [ apply nc_code | assumption ].
    - destruct (ad_function_containing_apply_expr d);
        destruct Hin as [<- | [<- | []]]; cbn [pos_ok];
        first [ exact I | apply nc_code ]. }
  apply pinv_fold; [ | exact Ha ].
  intros d a' Hd Ha'. apply pinv_add_closure_dep;
    [ exact H2 | exact (H4 d Hd) | exact Ha' ].
Qed.

(* No fact of the traversal of a unit has the constants where something
   is put. *)
Lemma traverse_unit_positions (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  ∃ l : list fact,
    traverse_unit flag u e = facts_of_list l
    ∧ ∀ fct, fct ∈ l -> pos_ok fct.
Proof.
  unfold traverse_unit. cbn [fresh fst snd].
  eexists. split; [ reflexivity | ].
  match goal with
  | |- ∀ fct, fct ∈ acc_facts (resolve_delayed_deps ?syms (add_fact ?f
                (traverse ?env e ?a))) -> _ =>
    assert (He : penv env); [ | assert (Ha : pinv a) ]
  end.
  - intros k ps Hin. cbn [env_conts] in Hin.
    destruct Hin as [Heq | [Heq | []]]; injection Heq as _ <-;
      constructor; [ apply nc_made | constructor | apply nc_made
                   | constructor ].
  - split; [ | split; [ | split ] ].
    + intros fct Hin. cbn in Hin.
      destruct Hin as [<- | [<- | []]]; exact I.
    + intros id cd [].
    + intros d [].
    + intros d [].
  - apply pinv_resolve, pinv_add_fact; [ exact I | ].
    exact (proj1 pinv_traverse e _ _ He Ha).
Qed.
