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

(* Proofs of the statements of FlambdaStateSupportSpec.v. This file needs
   no review.

   [supported act t]: there are finitely many names such that a renaming
   that fixes them fixes [t] under [act]. A name supports itself; what
   is built from supported parts is supported by the names of the
   parts ([supported1] to [supported5], [supported_map]). The rest goes
   through the terms and the parts of a state. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaAlphaSpec FlambdaNameAbstractionSpec FlambdaEquivarianceSpec
  FlambdaTrapSpec FlambdaStateSupportSpec.
Import ListNotations.

Definition supported {T : Type} (act : renaming -> T -> T) (t : T) : Prop :=
  ∃ (vs : list variable) (ks : list continuation),
    ∀ r : renaming,
      fixes (variable_permutation r) vs ->
      fixes (continuation_permutation r) ks ->
      act r t = t.

Lemma fixes_app_l (p : permutation) (names names' : list positive) :
  fixes p (names ++ names') -> fixes p names.
Proof.
  intros Hfixes name Hname. apply Hfixes. rewrite among_app, Hname.
  reflexivity.
Qed.

Lemma fixes_app_r (p : permutation) (names names' : list positive) :
  fixes p (names ++ names') -> fixes p names'.
Proof.
  intros Hfixes name Hname. apply Hfixes. rewrite among_app, Hname.
  apply orb_true_r.
Qed.

Lemma supported_none {T : Type} (act : renaming -> T -> T) (t : T) :
  (∀ r : renaming, act r t = t) -> supported act t.
Proof. intros Hsame. exists [], []. intros r _ _. apply Hsame. Qed.

Lemma supported_variable (x : variable) : supported rename_variable x.
Proof.
  exists [x], []. intros r Hvars _. apply Hvars.
  rewrite among_cons, Pos.eqb_refl. reflexivity.
Qed.

Lemma supported_continuation (k : continuation) :
  supported rename_continuation k.
Proof.
  exists [], [k]. intros r _ Hconts. apply Hconts.
  rewrite among_cons, Pos.eqb_refl. reflexivity.
Qed.

(* What is built from supported parts. *)
Lemma supported1 {A T : Type} (actA : renaming -> A -> A)
    (actT : renaming -> T -> T) (f : A -> T) (a : A) :
  (∀ r : renaming, actT r (f a) = f (actA r a)) ->
  supported actA a -> supported actT (f a).
Proof.
  intros Hf (vs & ks & Ha). exists vs, ks. intros r Hvars Hconts.
  rewrite Hf, Ha by assumption. reflexivity.
Qed.

Lemma supported2 {A B T : Type} (actA : renaming -> A -> A)
    (actB : renaming -> B -> B) (actT : renaming -> T -> T)
    (f : A -> B -> T) (a : A) (b : B) :
  (∀ r : renaming, actT r (f a b) = f (actA r a) (actB r b)) ->
  supported actA a -> supported actB b -> supported actT (f a b).
Proof.
  intros Hf (vs & ks & Ha) (vs' & ks' & Hb).
  exists (vs ++ vs'), (ks ++ ks'). intros r Hvars Hconts.
  rewrite Hf, Ha, Hb; eauto using fixes_app_l, fixes_app_r.
Qed.

Lemma supported3 {A B C T : Type} (actA : renaming -> A -> A)
    (actB : renaming -> B -> B) (actC : renaming -> C -> C)
    (actT : renaming -> T -> T) (f : A -> B -> C -> T) (a : A) (b : B)
    (c : C) :
  (∀ r : renaming,
     actT r (f a b c) = f (actA r a) (actB r b) (actC r c)) ->
  supported actA a -> supported actB b -> supported actC c ->
  supported actT (f a b c).
Proof.
  intros Hf Ha Hb Hc.
  apply (supported2 (fun r ab => (actA r (fst ab), actB r (snd ab))) actC
           actT (fun ab c => f (fst ab) (snd ab) c) (a, b) c);
    [ exact Hf | | exact Hc ].
  apply (supported2 actA actB _ pair a b);
    [ reflexivity | exact Ha | exact Hb ].
Qed.

Lemma supported4 {A B C D T : Type} (actA : renaming -> A -> A)
    (actB : renaming -> B -> B) (actC : renaming -> C -> C)
    (actD : renaming -> D -> D) (actT : renaming -> T -> T)
    (f : A -> B -> C -> D -> T) (a : A) (b : B) (c : C) (d : D) :
  (∀ r : renaming,
     actT r (f a b c d) = f (actA r a) (actB r b) (actC r c) (actD r d)) ->
  supported actA a -> supported actB b -> supported actC c ->
  supported actD d -> supported actT (f a b c d).
Proof.
  intros Hf Ha Hb Hc Hd.
  apply (supported3 (fun r ab => (actA r (fst ab), actB r (snd ab))) actC
           actD actT (fun ab c d => f (fst ab) (snd ab) c d) (a, b) c d);
    [ exact Hf | | exact Hc | exact Hd ].
  apply (supported2 actA actB _ pair a b);
    [ reflexivity | exact Ha | exact Hb ].
Qed.

Lemma supported5 {A B C D E T : Type} (actA : renaming -> A -> A)
    (actB : renaming -> B -> B) (actC : renaming -> C -> C)
    (actD : renaming -> D -> D) (actE : renaming -> E -> E)
    (actT : renaming -> T -> T) (f : A -> B -> C -> D -> E -> T) (a : A)
    (b : B) (c : C) (d : D) (e : E) :
  (∀ r : renaming,
     actT r (f a b c d e)
     = f (actA r a) (actB r b) (actC r c) (actD r d) (actE r e)) ->
  supported actA a -> supported actB b -> supported actC c ->
  supported actD d -> supported actE e -> supported actT (f a b c d e).
Proof.
  intros Hf Ha Hb Hc Hd He.
  apply (supported4 (fun r ab => (actA r (fst ab), actB r (snd ab))) actC
           actD actE actT (fun ab c d e => f (fst ab) (snd ab) c d e) (a, b)
           c d e);
    [ exact Hf | | exact Hc | exact Hd | exact He ].
  apply (supported2 actA actB _ pair a b);
    [ reflexivity | exact Ha | exact Hb ].
Qed.

Lemma supported_map {A : Type} (act : renaming -> A -> A) (l : list A) :
  Forall (supported act) l -> supported (fun r => map (act r)) l.
Proof.
  induction 1 as [ | a l Ha _ IH]; [ apply supported_none; reflexivity | ].
  exact (supported2 act (fun r => map (act r)) (fun r => map (act r)) cons a
           l (fun r => eq_refl) Ha IH).
Qed.

(* ======================================================================== *)
(* The parts of a term                                                      *)
(* ======================================================================== *)

Lemma simple_supported (s : simple) : supported rename_simple s.
Proof.
  destruct s as [x | sym | c]; [ | apply supported_none; reflexivity .. ].
  exact (supported1 rename_variable rename_simple Var x (fun r => eq_refl)
           (supported_variable x)).
Qed.

Lemma simples_supported (args : list simple) :
  supported (fun r => map (rename_simple r)) args.
Proof.
  apply supported_map. induction args as [ | arg args IH]; constructor;
    [ apply simple_supported | exact IH ].
Qed.

Lemma variables_supported (xs : list variable) :
  supported (fun r => map (rename_variable r)) xs.
Proof.
  apply supported_map. induction xs as [ | x xs IH]; constructor;
    [ apply supported_variable | exact IH ].
Qed.

Lemma value_slots_supported (slots : list (value_slot * simple)) :
  supported
    (fun r => map (fun slot => (fst slot, rename_simple r (snd slot))))
    slots.
Proof.
  apply
    (supported_map (fun r slot => (fst slot, rename_simple r (snd slot)))).
  induction slots as [ | [slot s] slots IH]; constructor; [ | exact IH ].
  exact (supported1 rename_simple
           (fun r slot' => (fst slot', rename_simple r (snd slot')))
           (pair slot) s (fun r => eq_refl) (simple_supported s)).
Qed.

Lemma named_supported (d : named) : supported rename_named d.
Proof.
  destruct d
    as [ s | p args | code slots | slot decls slots | slot closure
       | from slot closure | move_from move_to closure ].
  - exact (supported1 _ rename_named Simple s (fun r => eq_refl)
             (simple_supported s)).
  - exact (supported1 _ rename_named (Prim p) args (fun r => eq_refl)
             (simples_supported args)).
  - exact (supported1 _ rename_named (Set_of_closures code) slots
             (fun r => eq_refl) (value_slots_supported slots)).
  - exact (supported1 _ rename_named (Closure_of_set slot decls) slots
             (fun r => eq_refl) (value_slots_supported slots)).
  - exact (supported1 _ rename_named (Project_value_slot slot) closure
             (fun r => eq_refl) (simple_supported closure)).
  - exact (supported1 _ rename_named (Project_value_slot_from from slot)
             closure (fun r => eq_refl) (simple_supported closure)).
  - exact (supported1 _ rename_named
             (Project_function_slot move_from move_to) closure
             (fun r => eq_refl) (simple_supported closure)).
Qed.

Lemma static_const_supported (const : static_const) :
  supported rename_static_const const.
Proof.
  destruct const
    as [ code slots | slot symbols decls slots | tag fields | contents
       | bytes | elements ].
  - exact (supported1 _ rename_static_const
             (Static_const_set_of_closures code) slots (fun r => eq_refl)
             (value_slots_supported slots)).
  - exact (supported1 _ rename_static_const
             (Static_const_closure_of_set slot symbols decls) slots
             (fun r => eq_refl) (value_slots_supported slots)).
  - exact (supported1 _ rename_static_const (Static_const_block tag) fields
             (fun r => eq_refl) (simples_supported fields)).
  - exact (supported1 _ rename_static_const Static_const_boxed_number
             contents (fun r => eq_refl) (simple_supported contents)).
  - apply supported_none. reflexivity.
  - exact (supported1 _ rename_static_const
             Static_const_immutable_value_array elements
             (fun r => eq_refl) (simples_supported elements)).
Qed.

Lemma trap_action_supported (action : option trap_action) :
  supported (fun r => option_map (rename_trap_action r)) action.
Proof.
  destruct action as [[k | k kind] | ];
    [ | | apply supported_none; reflexivity ].
  - exact (supported1 rename_continuation
             (fun r => option_map (rename_trap_action r))
             (fun k => Some (Push k)) k (fun r => eq_refl)
             (supported_continuation k)).
  - exact (supported1 rename_continuation
             (fun r => option_map (rename_trap_action r))
             (fun k => Some (Pop k kind)) k (fun r => eq_refl)
             (supported_continuation k)).
Qed.

Lemma apply_cont_supported (ac : apply_cont) :
  supported rename_apply_cont ac.
Proof.
  destruct ac as [k args action].
  exact (supported3 _ _ _ rename_apply_cont Apply_cont_expr k args action
           (fun r => eq_refl) (supported_continuation k)
           (simples_supported args) (trap_action_supported action)).
Qed.

Lemma apply_supported (ap : apply_expr) : supported rename_apply ap.
Proof.
  destruct ap as [callee kind args k k_exn extra args_arity return_arity].
  refine (supported5 _ _ _ _
            (fun r => map (fun arg => (rename_simple r (fst arg), snd arg)))
            rename_apply
            (fun callee args k k_exn extra =>
               Apply_expr callee kind args k k_exn extra args_arity
                 return_arity)
            callee args k k_exn extra (fun r => eq_refl)
            (simple_supported callee) (simples_supported args)
            (supported_continuation k) (supported_continuation k_exn) _).
  apply
    (supported_map (fun r arg => (rename_simple r (fst arg), snd arg))).
  induction extra as [ | [arg kind'] extra IH]; constructor; [ | exact IH ].
  exact (supported1 rename_simple
           (fun r arg' => (rename_simple r (fst arg'), snd arg'))
           (fun arg' => (arg', kind')) arg (fun r => eq_refl)
           (simple_supported arg)).
Qed.

Lemma arms_supported (arms : list (Z * apply_cont)) :
  supported
    (fun r => map (fun arm => (fst arm, rename_apply_cont r (snd arm))))
    arms.
Proof.
  apply
    (supported_map
       (fun r arm => (fst arm, rename_apply_cont r (snd arm)))).
  induction arms as [ | [tag ac] arms IH]; constructor; [ | exact IH ].
  exact (supported1 rename_apply_cont
           (fun r arm => (fst arm, rename_apply_cont r (snd arm))) (pair tag)
           ac (fun r => eq_refl) (apply_cont_supported ac)).
Qed.

(* ======================================================================== *)
(* Expressions                                                              *)
(* ======================================================================== *)

Fixpoint expr_supported (e : expr) {struct e} : supported rename_expr e
with handler_supported (h : continuation_handler) {struct h} :
  supported rename_handler h
with binding_supported (binding : static_binding) {struct binding} :
  supported rename_static_binding binding
with code_supported (c : code) {struct c} : supported rename_code c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ].
    + exact (supported3 _ _ _ rename_expr (fun x d body => Let x kind d body)
               x d body (fun r => eq_refl) (supported_variable x)
               (named_supported d) (expr_supported body)).
    + refine (supported2 (fun r => map (rename_static_binding r)) _
                rename_expr Let_static bindings body (fun r => eq_refl) _
                (expr_supported body)).
      apply supported_map.
      induction bindings as [ | binding bindings IH]; constructor;
        [ apply binding_supported | exact IH ].
    + exact (supported3 _ _ _ rename_expr Let_cont_non_recursive k h body
               (fun r => eq_refl) (supported_continuation k)
               (handler_supported h) (expr_supported body)).
    + refine (supported2
                (fun r =>
                   map
                     (fun '(k, h) =>
                        (rename_continuation r k, rename_handler r h)))
                _ rename_expr Let_cont_recursive handlers body
                (fun r => eq_refl) _ (expr_supported body)).
      apply (supported_map
               (fun r '(k, h) =>
                  (rename_continuation r k, rename_handler r h))).
      induction handlers as [ | [k h] handlers IH]; constructor;
        [ | exact IH ].
      exact (supported2 rename_continuation rename_handler
               (fun r '(k, h) =>
                  (rename_continuation r k, rename_handler r h))
               pair k h (fun r => eq_refl) (supported_continuation k)
               (handler_supported h)).
    + exact (supported1 _ rename_expr Apply ap (fun r => eq_refl)
               (apply_supported ap)).
    + exact (supported1 _ rename_expr Apply_cont ac (fun r => eq_refl)
               (apply_cont_supported ac)).
    + exact (supported2 _ _ rename_expr Switch scrutinee arms
               (fun r => eq_refl) (simple_supported scrutinee)
               (arms_supported arms)).
    + apply supported_none. reflexivity.
  - destruct h as [params metadata handler].
    exact (supported2 _ _ rename_handler
             (fun params handler => Handler params metadata handler) params
             handler (fun r => eq_refl) (variables_supported params)
             (expr_supported handler)).
  - destruct binding as [id c | id | sym const].
    + exact (supported1 _ rename_static_binding (Bind_code id) c
               (fun r => eq_refl) (code_supported c)).
    + apply supported_none. reflexivity.
    + exact (supported1 _ rename_static_binding (Bind_symbol sym) const
               (fun r => eq_refl) (static_const_supported const)).
  - destruct c as [params my_closure ret exn metadata body].
    exact (supported5 _ _ _ _ _ rename_code
             (fun params my_closure ret exn body =>
                Code params my_closure ret exn metadata body)
             params my_closure ret exn body (fun r => eq_refl)
             (variables_supported params) (supported_variable my_closure)
             (supported_continuation ret) (supported_continuation exn)
             (expr_supported body)).
Qed.

Theorem expr_finitely_supported : expr_finitely_supported_statement.
Proof. exact expr_supported. Qed.

(* ======================================================================== *)
(* States                                                                   *)
(* ======================================================================== *)

Lemma env_supported (vars : env) : supported rename_env vars.
Proof.
  apply (supported_map (fun r '(x, v) => (rename_variable r x, v))).
  induction vars as [ | [x v] vars IH]; constructor; [ | exact IH ].
  exact (supported1 rename_variable
           (fun r '(x, v) => (rename_variable r x, v))
           (fun x => (x, v)) x (fun r => eq_refl) (supported_variable x)).
Qed.

Lemma handlers_supported
    (handlers : list (continuation * continuation_handler)) :
  supported rename_handlers handlers.
Proof.
  apply (supported_map
           (fun r '(k, h) => (rename_continuation r k, rename_handler r h))).
  induction handlers as [ | [k h] handlers IH]; constructor; [ | exact IH ].
  exact (supported2 rename_continuation rename_handler
           (fun r '(k, h) => (rename_continuation r k, rename_handler r h))
           pair k h (fun r => eq_refl) (supported_continuation k)
           (handler_supported h)).
Qed.

Fixpoint cont_value_supported (c : cont_value) {struct c} :
  supported rename_cont_value c.
Proof.
  destruct c
    as [h vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to].
  - refine (supported4 rename_handler rename_env
              (fun r =>
                 map
                   (fun '(k, c) =>
                      (rename_continuation r k, rename_cont_value r c)))
              rename_handlers rename_cont_value Closure h vars conts recursive
              (fun r => eq_refl) (handler_supported h) (env_supported vars) _
              (handlers_supported recursive)).
    apply (supported_map
             (fun r '(k, c) =>
                (rename_continuation r k, rename_cont_value r c))).
    induction conts as [ | [k c] conts IH]; constructor; [ | exact IH ].
    exact (supported2 rename_continuation rename_cont_value
             (fun r '(k, c) =>
                (rename_continuation r k, rename_cont_value r c))
             pair k c (fun r => eq_refl) (supported_continuation k)
             (cont_value_supported c)).
  - exact (supported1 _ rename_cont_value Outer k (fun r => eq_refl)
             (supported_continuation k)).
  - exact (supported1 rename_cont_value rename_cont_value
             (fun c => Extra_args c extra) c (fun r => eq_refl)
             (cont_value_supported c)).
  - exact (supported2 rename_cont_value rename_cont_value rename_cont_value
             (Over_application remaining remaining_arity return_arity)
             return_to exn_to (fun r => eq_refl)
             (cont_value_supported return_to)
             (cont_value_supported exn_to)).
Qed.

Lemma cont_env_supported (conts : cont_env) :
  supported rename_cont_env conts.
Proof.
  apply (supported_map
           (fun r '(k, c) =>
              (rename_continuation r k, rename_cont_value r c))).
  induction conts as [ | [k c] conts IH]; constructor; [ | exact IH ].
  exact (supported2 rename_continuation rename_cont_value
           (fun r '(k, c) =>
              (rename_continuation r k, rename_cont_value r c))
           pair k c (fun r => eq_refl) (supported_continuation k)
           (cont_value_supported c)).
Qed.

Lemma static_env_supported (statics : static_env) :
  supported rename_static_env statics.
Proof.
  destruct statics as [codes values].
  refine (supported1
            (fun r =>
               map
                 (fun binding => (fst binding, rename_code r (snd binding))))
            rename_static_env (fun codes => Static_env codes values) codes
            (fun r => eq_refl) _).
  apply (supported_map
           (fun r binding => (fst binding, rename_code r (snd binding)))).
  induction codes as [ | [id c] codes IH]; constructor; [ | exact IH ].
  exact (supported1 rename_code
           (fun r binding => (fst binding, rename_code r (snd binding)))
           (pair id) c (fun r => eq_refl) (code_supported c)).
Qed.

Theorem state_finitely_supported : state_finitely_supported_statement.
Proof.
  intros [e vars conts statics hp rd].
  exact (supported4 rename_expr rename_env rename_cont_env rename_static_env
           rename_state
           (fun e vars conts statics => State e vars conts statics hp rd)
           e vars conts statics (fun r => eq_refl)
           (expr_supported e) (env_supported vars) (cont_env_supported conts)
           (static_env_supported statics)).
Qed.

(* ======================================================================== *)
(* A state with a stack                                                     *)
(* ======================================================================== *)

Lemma stack_entry_supported (entry : stack_entry) :
  supported rename_stack_entry entry.
Proof.
  destruct entry
    as [c | ret exn return_to extra
       | ret exn remaining remaining_arity return_arity return_to extra].
  - exact (supported1 rename_cont_value rename_stack_entry Trap c
             (fun r => eq_refl) (cont_value_supported c)).
  - exact (supported3 rename_continuation rename_continuation
             rename_cont_value rename_stack_entry
             (fun ret exn return_to => Call ret exn return_to extra) ret exn
             return_to (fun r => eq_refl) (supported_continuation ret)
             (supported_continuation exn)
             (cont_value_supported return_to)).
  - exact (supported3 rename_continuation rename_continuation
             rename_cont_value rename_stack_entry
             (fun ret exn return_to =>
                Call_then ret exn remaining remaining_arity return_arity
                  return_to extra)
             ret exn return_to (fun r => eq_refl) (supported_continuation ret)
             (supported_continuation exn)
             (cont_value_supported return_to)).
Qed.

Lemma stack_supported (st : stack) : supported rename_stack st.
Proof.
  apply (supported_map rename_stack_entry).
  induction st as [ | entry st IH]; constructor;
    [ apply stack_entry_supported | exact IH ].
Qed.

Theorem stack_configuration_finitely_supported :
  stack_configuration_finitely_supported_statement.
Proof.
  intros s st.
  destruct (state_finitely_supported s) as (vs & ks & Hstate).
  destruct (stack_supported st) as (vs' & ks' & Hstack).
  exists (vs ++ vs'), (ks ++ ks'). intros r Hvars Hconts. split.
  - apply Hstate; eauto using fixes_app_l.
  - apply Hstack; eauto using fixes_app_r.
Qed.
