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

(* Proofs of the statements of FlambdaStaticsOrderSpec.v. Nothing here
   needs review.

   Two orders of a group give static environments with the same lookups,
   or are both undefined: two bindings next to each other may change
   places, and the pass is made of such moves.

   A state runs alike under two static environments with the same
   lookups. That is the simulation of FlambdaUnusedStaticsProofs.v, of a
   state and the state with some static bindings removed, where nothing
   is removed: the relation there between the two static environments is
   one of lookups. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaContextSpec FlambdaContextProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaUnusedStaticsSpec
  FlambdaUnusedStaticsProofs FlambdaStaticsOrderSpec.
Import ListNotations.

(* ======================================================================== *)
(* Static environments with the same lookups                                *)
(* ======================================================================== *)

Definition same_lookups (statics statics' : static_env) : Prop :=
  (∀ id : code_id,
    lookup id (static_code statics) = lookup id (static_code statics'))
  ∧ (∀ s : symbol,
      lookup s (static_values statics) = lookup s (static_values statics')).

Lemma same_lookups_refl (statics : static_env) : same_lookups statics statics.
Proof. split; reflexivity. Qed.

Lemma same_lookups_sym (statics statics' : static_env) :
  same_lookups statics statics' -> same_lookups statics' statics.
Proof.
  intros [Hcode Hvalues].
  split;
    [ intros id; symmetry; apply Hcode | intros s; symmetry; apply Hvalues ].
Qed.

Lemma same_lookups_trans (statics statics' statics'' : static_env) :
  same_lookups statics statics' -> same_lookups statics' statics'' ->
  same_lookups statics statics''.
Proof.
  intros [Hcode Hvalues] [Hcode' Hvalues'].
  split;
    [ intros id; rewrite (Hcode id); apply Hcode'
    | intros s; rewrite (Hvalues s); apply Hvalues' ].
Qed.

(* What two groups give: static environments with the same lookups, or
   nothing for either. *)
Definition results_alike (result result' : option static_env) : Prop :=
  match result, result' with
  | Some statics, Some statics' => same_lookups statics statics'
  | None, None => True
  | _, _ => False
  end.

Lemma results_alike_refl (result : option static_env) :
  results_alike result result.
Proof. destruct result; simpl; [ apply same_lookups_refl | exact I ]. Qed.

Lemma results_alike_trans (result result' result'' : option static_env) :
  results_alike result result' -> results_alike result' result'' ->
  results_alike result result''.
Proof.
  destruct result, result', result''; simpl; try contradiction; auto.
  apply same_lookups_trans.
Qed.

(* A group gives the same from two static environments with the same
   lookups. *)
Lemma eval_static_bindings_same_lookups (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    same_lookups statics statics' ->
    results_alike (eval_static_bindings vars bindings statics)
      (eval_static_bindings vars bindings statics').
Proof.
  induction bindings as [ | binding bindings IH]; intros statics statics' Hsame;
    [ exact Hsame | ].
  destruct Hsame as [Hcode Hvalues].
  destruct binding as [id c | id | s const]; simpl.
  - rewrite <- (Hcode id).
    destruct (lookup id (static_code statics)); [ exact I | ].
    apply IH. split; simpl; [ | exact Hvalues ].
    intros id'. rewrite (Hcode id'). reflexivity.
  - apply IH. split; assumption.
  - rewrite <- (Hvalues s).
    destruct (lookup s (static_values statics)); [ exact I | ].
    destruct (eval_static_const vars const) as [allocated | ]; simpl;
      [ | exact I ].
    apply IH. split; simpl; [ exact Hcode | ].
    intros s'. rewrite (Hvalues s'). reflexivity.
Qed.

(* Two bindings next to each other may change places. *)
Lemma eval_static_bindings_swap (vars : env)
    (binding binding' : static_binding) (rest : list static_binding)
    (statics : static_env) :
  results_alike
    (eval_static_bindings vars (binding :: binding' :: rest) statics)
    (eval_static_bindings vars (binding' :: binding :: rest) statics).
Proof.
  destruct binding as [id c | id | s const],
    binding' as [id' c' | id' | s' const'];
    simpl; try apply results_alike_refl.
  - (* two bindings of code *)
    rewrite (Pos.eqb_sym id' id).
    destruct (Pos.eqb id id') eqn:Hid.
    + apply Pos.eqb_eq in Hid. subst id'.
      destruct (lookup id (static_code statics)); exact I.
    + destruct (lookup id (static_code statics)),
        (lookup id' (static_code statics)); try exact I.
      apply eval_static_bindings_same_lookups. split; simpl; [ | reflexivity ].
      intros other.
      destruct (Pos.eqb other id') eqn:Hother', (Pos.eqb other id) eqn:Hother;
        try reflexivity.
      apply Pos.eqb_eq in Hother', Hother. subst id id'.
      rewrite Pos.eqb_refl in Hid. discriminate Hid.
  - (* code, then a symbol *)
    destruct (lookup id (static_code statics)),
      (lookup s' (static_values statics)),
      (eval_static_const vars const');
      simpl; try exact I; apply results_alike_refl.
  - (* a symbol, then code *)
    destruct (lookup id' (static_code statics)),
      (lookup s (static_values statics)),
      (eval_static_const vars const);
      simpl; try exact I; apply results_alike_refl.
  - (* two bindings of symbols *)
    rewrite (Pos.eqb_sym s' s).
    destruct (Pos.eqb s s') eqn:Hs.
    + apply Pos.eqb_eq in Hs. subst s'.
      destruct (lookup s (static_values statics)); [ exact I | ].
      destruct (eval_static_const vars const),
        (eval_static_const vars const');
        simpl; exact I.
    + destruct (lookup s (static_values statics)),
        (lookup s' (static_values statics)),
        (eval_static_const vars const),
        (eval_static_const vars const');
        simpl; try exact I.
      apply eval_static_bindings_same_lookups. split; simpl; [ reflexivity | ].
      intros sym.
      destruct (Pos.eqb sym s') eqn:Hsym', (Pos.eqb sym s) eqn:Hsym;
        try reflexivity.
      apply Pos.eqb_eq in Hsym', Hsym. subst s s'.
      rewrite Pos.eqb_refl in Hs. discriminate Hs.
Qed.

(* The same binding before two groups that give the same. *)
Lemma eval_static_bindings_cons_alike (vars : env) (binding : static_binding)
    (bindings bindings' : list static_binding) :
  (∀ statics : static_env,
    results_alike (eval_static_bindings vars bindings statics)
      (eval_static_bindings vars bindings' statics)) ->
  ∀ statics : static_env,
    results_alike (eval_static_bindings vars (binding :: bindings) statics)
      (eval_static_bindings vars (binding :: bindings') statics).
Proof.
  intros Halike statics. destruct binding as [id c | id | s const]; simpl.
  - destruct (lookup id (static_code statics)); [ exact I | apply Halike ].
  - apply Halike.
  - destruct (lookup s (static_values statics)); [ exact I | ].
    destruct (eval_static_const vars const); simpl;
      [ apply Halike | exact I ].
Qed.

Lemma to_front_alike (vars : env) (is_it : static_binding -> bool)
    (bindings : list static_binding) :
  ∀ statics : static_env,
    results_alike (eval_static_bindings vars bindings statics)
      (eval_static_bindings vars (to_front is_it bindings) statics).
Proof.
  induction bindings as [ | binding rest IH]; intros statics;
    [ apply results_alike_refl | ].
  cbn [to_front].
  destruct (is_it binding); [ apply results_alike_refl | ].
  destruct (to_front is_it rest) as [ | found rest'];
    [ apply results_alike_refl | ].
  destruct (is_it found); [ | apply results_alike_refl ].
  apply (results_alike_trans _
    (eval_static_bindings vars (binding :: found :: rest') statics)).
  - apply eval_static_bindings_cons_alike. exact IH.
  - apply eval_static_bindings_swap.
Qed.

Lemma order_as_alike (vars : env) (pattern bindings : list static_binding) :
  ∀ statics : static_env,
    results_alike (eval_static_bindings vars bindings statics)
      (eval_static_bindings vars (order_as pattern bindings) statics).
Proof.
  unfold order_as.
  induction pattern as [ | named pattern IH]; intros statics;
    [ apply results_alike_refl | ].
  cbn [fold_right].
  exact (results_alike_trans _ _ _ (IH statics)
    (to_front_alike vars _ _ statics)).
Qed.

(* ======================================================================== *)
(* Where no static binding is removed                                       *)
(* ======================================================================== *)

(* The names to remove of FlambdaUnusedStaticsSpec.v, when there are
   none: everything avoids them, and removing their bindings changes
   nothing. *)
Definition nothing : dropped := Dropped [] [].

Lemma forallb_always {A : Type} (f : A -> bool) (l : list A) :
  (∀ a : A, f a = true) -> forallb f l = true.
Proof. intros Hf. apply forallb_forall. intros a _. apply Hf. Qed.

Fixpoint value_avoids_nothing (v : value) : value_avoids nothing v = true.
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | elements | contents | loc | callee args args_arity];
    simpl; try reflexivity.
  - induction slots as [ | [slot v] slots IH]; simpl; [ reflexivity | ].
    rewrite (value_avoids_nothing v). exact IH.
  - replace (function_decls_avoid nothing decls) with true
      by (symmetry; apply forallb_always; reflexivity).
    induction slots as [ | [slot' v] slots IH]; simpl; [ reflexivity | ].
    rewrite (value_avoids_nothing v). exact IH.
  - induction fields as [ | v fields IH]; simpl; [ reflexivity | ].
    rewrite (value_avoids_nothing v). exact IH.
  - induction elements as [ | v elements IH]; simpl; [ reflexivity | ].
    rewrite (value_avoids_nothing v). exact IH.
  - rewrite (value_avoids_nothing callee). simpl.
    induction args as [ | v args IH]; simpl; [ reflexivity | ].
    rewrite (value_avoids_nothing v). exact IH.
Qed.

Lemma simple_avoids_nothing (s : simple) : simple_avoids nothing s = true.
Proof. destruct s; reflexivity. Qed.

Lemma simples_avoid_nothing (args : list simple) :
  simples_avoid nothing args = true.
Proof. apply forallb_always, simple_avoids_nothing. Qed.

Lemma value_slots_avoid_nothing (slots : list (value_slot * simple)) :
  value_slots_avoid nothing slots = true.
Proof. apply forallb_always. intros slot. apply simple_avoids_nothing. Qed.

Lemma function_decls_avoid_nothing (decls : list (function_slot * code_id)) :
  function_decls_avoid nothing decls = true.
Proof. apply forallb_always. reflexivity. Qed.

Lemma closure_symbols_avoid_nothing
    (symbols : list (function_slot * symbol)) :
  closure_symbols_avoid nothing symbols = true.
Proof. apply forallb_always. reflexivity. Qed.

Lemma named_avoids_nothing (d : named) : named_avoids nothing d = true.
Proof.
  destruct d; simpl;
    rewrite ?function_decls_avoid_nothing, ?value_slots_avoid_nothing;
    try reflexivity;
    first [ apply simple_avoids_nothing | apply simples_avoid_nothing ].
Qed.

Lemma static_const_avoids_nothing (const : static_const) :
  static_const_avoids nothing const = true.
Proof.
  destruct const; simpl;
    rewrite ?closure_symbols_avoid_nothing, ?function_decls_avoid_nothing,
      ?value_slots_avoid_nothing;
    try reflexivity;
    first [ apply simple_avoids_nothing | apply simples_avoid_nothing ].
Qed.

Lemma apply_avoids_nothing (ap : apply_expr) : apply_avoids nothing ap = true.
Proof.
  unfold apply_avoids.
  rewrite simple_avoids_nothing, !simples_avoid_nothing.
  destruct (ap_call_kind ap) as [code | | [code_ids | ]]; try reflexivity.
  rewrite !andb_true_r. apply forallb_forall. intros id _. reflexivity.
Qed.

Fixpoint expr_avoids_nothing (e : expr) : expr_avoids nothing e = true
with handler_avoids_nothing (h : continuation_handler) :
  handler_avoids nothing h = true
with binding_avoids_nothing (binding : static_binding) :
  binding_avoids nothing binding = true
with code_avoids_nothing (c : code) : code_avoids nothing c = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl.
    + rewrite named_avoids_nothing. apply expr_avoids_nothing.
    + rewrite (expr_avoids_nothing body), andb_true_r.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite (binding_avoids_nothing binding). exact IH.
    + rewrite (handler_avoids_nothing h). apply expr_avoids_nothing.
    + rewrite (expr_avoids_nothing body), andb_true_r.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_avoids_nothing h). exact IH.
    + apply apply_avoids_nothing.
    + apply simples_avoid_nothing.
    + rewrite simple_avoids_nothing. apply forallb_always.
      intros arm. apply simples_avoid_nothing.
    + reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply expr_avoids_nothing.
  - destruct binding as [id c | id | s const]; simpl.
    + apply code_avoids_nothing.
    + reflexivity.
    + apply static_const_avoids_nothing.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    apply expr_avoids_nothing.
Qed.

Fixpoint binds_only_kept_nothing (e : expr) : binds_only_kept nothing e = true
with handler_binds_only_kept_nothing (h : continuation_handler) :
  handler_binds_only_kept nothing h = true
with binding_binds_only_kept_nothing (binding : static_binding) :
  binding_binds_only_kept nothing binding = true
with code_binds_only_kept_nothing (c : code) :
  code_binds_only_kept nothing c = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply binds_only_kept_nothing.
    + rewrite (binds_only_kept_nothing body), andb_true_r.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite (binding_binds_only_kept_nothing binding). exact IH.
    + rewrite (handler_binds_only_kept_nothing h).
      apply binds_only_kept_nothing.
    + rewrite (binds_only_kept_nothing body), andb_true_r.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_binds_only_kept_nothing h). exact IH.
  - destruct h as [params metadata handler]. simpl.
    apply binds_only_kept_nothing.
  - destruct binding as [id c | id | s const]; simpl; try reflexivity.
    apply code_binds_only_kept_nothing.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    apply binds_only_kept_nothing.
Qed.

Lemma handler_free_of_nothing (h : continuation_handler) :
  handler_free_of nothing h = true.
Proof.
  unfold handler_free_of.
  rewrite handler_avoids_nothing, handler_binds_only_kept_nothing.
  reflexivity.
Qed.

Lemma env_free_of_nothing (vars : env) : env_free_of nothing vars = true.
Proof.
  apply forallb_always. intros [x v]. apply value_avoids_nothing.
Qed.

Fixpoint cont_free_of_nothing (c : cont_value) :
  cont_free_of nothing c = true.
Proof.
  destruct c as [h vars conts recursive | k | c extra
    | remaining remaining_arity return_arity return_to exn_to];
    simpl;
    [ | reflexivity
    | rewrite (cont_free_of_nothing c); simpl;
      apply forallb_always, value_avoids_nothing
    | rewrite (cont_free_of_nothing return_to),
        (cont_free_of_nothing exn_to), !andb_true_r;
      apply forallb_always, value_avoids_nothing ].
  rewrite handler_free_of_nothing, env_free_of_nothing. simpl.
  replace
    (forallb (fun '(_, h) => handler_free_of nothing h) recursive)
    with true
    by (symmetry; apply forallb_always; intros [k' h'];
        apply handler_free_of_nothing).
  rewrite andb_true_r.
  induction conts as [ | [k c] conts IH]; simpl; [ reflexivity | ].
  rewrite (cont_free_of_nothing c). exact IH.
Qed.

Lemma conts_erased_nothing (conts : cont_env) :
  conts_erased nothing conts conts.
Proof.
  induction conts as [ | [k c] conts IH]; [ apply Forall2_nil | ].
  apply Forall2_cons; [ | exact IH ].
  split; [ reflexivity | ].
  apply cont_free_of_erased, cont_free_of_nothing.
Qed.

Lemma allocated_avoids_nothing (allocated : statically_allocated_value) :
  allocated_avoids nothing allocated = true.
Proof.
  assert (Hslots :
    ∀ slots : list (value_slot * value),
      slot_values_avoid nothing slots = true).
  { intros slots. apply forallb_always. intros [slot v].
    apply value_avoids_nothing. }
  destruct allocated; simpl;
    rewrite ?closure_symbols_avoid_nothing, ?function_decls_avoid_nothing,
      ?Hslots;
    try reflexivity;
    first [ apply value_avoids_nothing
          | apply forallb_always, value_avoids_nothing ].
Qed.

Lemma same_lookups_erased (statics statics' : static_env) :
  same_lookups statics statics' -> statics_erased nothing statics statics'.
Proof.
  intros [Hcode Hvalues]. split; [ | split; [ | split ] ].
  - intros id _. rewrite <- (Hcode id).
    destruct (lookup id (static_code statics)) as [c | ]; simpl;
      [ | reflexivity ].
    rewrite (erase_code_binds_only_kept nothing c
      (code_binds_only_kept_nothing c)).
    reflexivity.
  - intros id c _ _. apply code_avoids_nothing.
  - intros s _. symmetry. apply Hvalues.
  - intros s allocated _ _. apply allocated_avoids_nothing.
Qed.

(* No value of a cell of a heap refers to a name removed, when there
   are none. *)
Lemma heap_free_of_nothing (hp : heap) : heap_free_of nothing hp = true.
Proof.
  unfold heap_free_of. apply forallb_always. intros [l c].
  apply forallb_always, value_avoids_nothing.
Qed.

(* A state runs alike under two static environments with the same
   lookups. *)
Lemma same_lookups_refines (e : expr) (vars : env) (conts : cont_env)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  same_lookups statics statics' ->
  refines (State e vars conts statics hp rd)
    (State e vars conts statics' hp rd).
Proof.
  intros Hsame.
  assert (Hrelated :
    state_erased_free nothing (State e vars conts statics hp rd)
      (State (erase_expr nothing e) vars conts statics' hp rd)).
  { split; [ | apply heap_free_of_nothing ].
    apply State_erased;
      [ apply expr_avoids_nothing
      | intros x v _; apply value_avoids_nothing
      | apply conts_erased_nothing
      | apply same_lookups_erased, Hsame ]. }
  rewrite (erase_binds_only_kept nothing e (binds_only_kept_nothing e))
    in Hrelated.
  exact (simulation_refines (state_erased_free nothing)
    (erased_free_step nothing) _ _ Hrelated).
Qed.

(* ======================================================================== *)
(* The normalisation                                                        *)
(* ======================================================================== *)

Lemma order_as_equivalent (pattern bindings : list static_binding)
    (body : expr) :
  expr_equivalent (Let_static bindings body)
    (Let_static (order_as pattern bindings) body).
Proof.
  intros vars conts statics hp rd.
  pose proof (order_as_alike vars pattern bindings statics) as Halike.
  destruct (eval_static_bindings vars bindings statics)
    as [extended | ] eqn:Hextended,
    (eval_static_bindings vars (order_as pattern bindings) statics)
    as [extended' | ] eqn:Hextended';
    simpl in Halike; try contradiction.
  - apply refines_antisymmetric.
    + apply (refines_step _ (State body vars conts extended hp rd) _
        (State body vars conts extended' hp rd));
        [ rewrite step_let_static, Hextended; reflexivity
        | rewrite step_let_static, Hextended'; reflexivity
        | apply same_lookups_refines, Halike ].
    + apply (refines_step _ (State body vars conts extended' hp rd) _
        (State body vars conts extended hp rd));
        [ rewrite step_let_static, Hextended'; reflexivity
        | rewrite step_let_static, Hextended; reflexivity
        | apply same_lookups_refines, same_lookups_sym, Halike ].
  - apply step_eq_equivalent.
    rewrite !step_let_static, Hextended, Hextended'. reflexivity.
Qed.

Fixpoint order_statics_as_equivalent_aux (e : expr) :
  ∀ pattern : expr, expr_equivalent e (order_statics_as pattern e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros pattern;
    destruct pattern
      as [ px pkind pd pbody | pbindings pbody
         | pk [pparams pmetadata phandler] pbody | phandlers pbody | pap | pac
         | pscrutinee parms | ];
    simpl; try (intros vars conts statics hp rd; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (order_statics_as_equivalent_aux body pbody)).
  - apply (expr_equivalent_trans _
      (Let_static bindings (order_statics_as pbody body))).
    + exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
        (order_statics_as_equivalent_aux body pbody)).
    + apply order_as_equivalent.
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (order_statics_as phandler handler)) body)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole body)
        _ _ (not_many _)
        (order_statics_as_equivalent_aux handler phandler)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_body k
           (Handler params metadata (order_statics_as phandler handler)) Hole)
        _ _ (not_many _) (order_statics_as_equivalent_aux body pbody)).
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole)
        _ _ (not_many _)
      (order_statics_as_equivalent_aux body pbody)).
Qed.

Theorem order_statics_as_equivalent : order_statics_as_equivalent_statement.
Proof. intros pattern e. apply order_statics_as_equivalent_aux. Qed.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

Lemma binds_same_name_renamed (r r' : renaming)
    (named binding : static_binding) :
  binds_same_name (rename_static_binding r' named)
    (rename_static_binding r binding)
  = binds_same_name named binding.
Proof. destruct named, binding; reflexivity. Qed.

Lemma to_front_map (f : static_binding -> static_binding)
    (is_it is_it' : static_binding -> bool) (bindings : list static_binding) :
  (∀ binding : static_binding, is_it' (f binding) = is_it binding) ->
  to_front is_it' (map f bindings) = map f (to_front is_it bindings).
Proof.
  intros Hsame.
  induction bindings as [ | binding rest IH]; [ reflexivity | ].
  cbn [map to_front]. rewrite Hsame.
  destruct (is_it binding); [ reflexivity | ].
  rewrite IH.
  destruct (to_front is_it rest) as [ | found rest']; [ reflexivity | ].
  cbn [map]. rewrite Hsame.
  destruct (is_it found); reflexivity.
Qed.

Lemma order_as_renamed (r r' : renaming)
    (pattern bindings : list static_binding) :
  order_as (map (rename_static_binding r') pattern)
    (map (rename_static_binding r) bindings)
  = map (rename_static_binding r) (order_as pattern bindings).
Proof.
  unfold order_as.
  induction pattern as [ | named pattern IH]; [ reflexivity | ].
  cbn [map fold_right]. rewrite IH.
  apply to_front_map. intros binding. apply binds_same_name_renamed.
Qed.

Fixpoint order_statics_as_equivariant_aux (r r' : renaming) (e : expr) :
  ∀ pattern : expr,
    order_statics_as (rename_expr r' pattern) (rename_expr r e)
    = rename_expr r (order_statics_as pattern e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros pattern;
    destruct pattern
      as [ px pkind pd pbody | pbindings pbody
         | pk [pparams pmetadata phandler] pbody | phandlers pbody | pap | pac
         | pscrutinee parms | ];
    simpl; try reflexivity.
  - rewrite order_statics_as_equivariant_aux. reflexivity.
  - rewrite order_as_renamed, order_statics_as_equivariant_aux. reflexivity.
  - rewrite !order_statics_as_equivariant_aux. reflexivity.
  - rewrite order_statics_as_equivariant_aux. reflexivity.
Qed.

Theorem order_statics_as_equivariant : order_statics_as_equivariant_statement.
Proof.
  intros r r' pattern e. apply order_statics_as_equivariant_aux.
Qed.
