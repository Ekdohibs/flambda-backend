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

(* Proofs of the statements of FlambdaClosureOfCodeSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Arith Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaNominalSpec
  FlambdaWellFormedSpec FlambdaWellFormedProofs FlambdaStaticsInScopeSpec
  FlambdaUnusedStaticsSpec FlambdaKnownSymbolsSpec FlambdaClosuresSpec
  FlambdaClosuresProofs FlambdaContextWhereSpec FlambdaContextWhereProofs
  FlambdaKnownSlotsSpec FlambdaKnownSlotsProofs FlambdaSetsOfCodeSpec
  FlambdaSetsOfCodeProofs FlambdaImportsSpec FlambdaObserversSpec
  FlambdaUnitFromProofs FlambdaSharedReadsProofs FlambdaClosureOfCodeSpec.
From OxCamlFlambda2 Require Import FlambdaBindersSpec FlambdaNominalProofs
  FlambdaEquivarianceProofs FlambdaScopedProofs.
Import ListNotations.

(* ======================================================================== *)
(* What the rewrite keeps                                                   *)
(* ======================================================================== *)

Lemma forallb_same {A : Type} (p q : A -> bool) (l : list A) :
  (∀ a : A, p a = q a) -> forallb p l = forallb q l.
Proof.
  intros H. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite H, IH. reflexivity.
Qed.

(* It makes no set of closures and changes none. *)
Lemma read_code_slot_sets_made (sets : list set_made) (syms : list symbol)
    (known : closure_codes) (d : named) :
  named_sets_made (read_code_slot sets syms known d) = named_sets_made d.
Proof.
  destruct d as [a | p args | code slots | slot decls slots | slot closure
    | project_from slot closure | move_from move_to closure];
    cbn [read_code_slot]; try reflexivity.
  destruct closure as [x | sym | k]; try reflexivity.
  destruct (lookup x known) as [c | ]; [ | reflexivity ].
  destruct (slot_held sets c project_from slot) as [a | ]; [ | reflexivity ].
  destruct (simple_statics_in_scope syms a); reflexivity.
Qed.

Fixpoint know_in_body_sets_made (sets : list set_made) (e : expr)
    {struct e} :
  ∀ (syms : list symbol) (known : closure_codes),
    sets_made (know_in_body sets syms known e) = sets_made e
with know_in_body_handler_sets_made (sets : list set_made)
    (h : continuation_handler) {struct h} :
  ∀ (syms : list symbol) (known : closure_codes),
    handler_sets_made (know_in_body_handler sets syms known h)
    = handler_sets_made h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros syms known; cbn [know_in_body sets_made]; try reflexivity.
    + rewrite read_code_slot_sets_made,
        (know_in_body_sets_made sets body).
      reflexivity.
    + rewrite (know_in_body_sets_made sets body). reflexivity.
    + rewrite (know_in_body_handler_sets_made sets h),
        (know_in_body_sets_made sets body).
      reflexivity.
    + rewrite (know_in_body_sets_made sets body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
        [ reflexivity | ].
      rewrite (know_in_body_handler_sets_made sets h), IH. reflexivity.
  - destruct h as [params metadata handler]. intros syms known.
    cbn [know_in_body_handler handler_sets_made].
    apply know_in_body_sets_made.
Qed.

Fixpoint know_in_body_groups_whole (sets : list set_made) (e : expr)
    {struct e} :
  ∀ (syms : list symbol) (known : closure_codes),
    groups_whole (know_in_body sets syms known e) = groups_whole e
with know_in_body_handler_groups_whole (sets : list set_made)
    (h : continuation_handler) {struct h} :
  ∀ (syms : list symbol) (known : closure_codes),
    handler_groups_whole (know_in_body_handler sets syms known h)
    = handler_groups_whole h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros syms known; cbn [know_in_body groups_whole]; try reflexivity.
    + apply know_in_body_groups_whole.
    + rewrite (know_in_body_groups_whole sets body). reflexivity.
    + rewrite (know_in_body_handler_groups_whole sets h),
        (know_in_body_groups_whole sets body).
      reflexivity.
    + rewrite (know_in_body_groups_whole sets body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (know_in_body_handler_groups_whole sets h), IH. reflexivity.
  - destruct h as [params metadata handler]. intros syms known.
    cbn [know_in_body_handler handler_groups_whole].
    apply know_in_body_groups_whole.
Qed.

(* It changes no binding, so what binds something static is as it
   was. *)
Fixpoint know_in_body_binds_statics (sets : list set_made) (e : expr) :
  ∀ (syms : list symbol) (known : closure_codes),
    binds_statics (know_in_body sets syms known e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros syms known; simpl; try reflexivity.
  - apply know_in_body_binds_statics.
  - rewrite know_in_body_binds_statics. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite !know_in_body_binds_statics. reflexivity.
  - rewrite know_in_body_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite know_in_body_binds_statics, IH. reflexivity.
Qed.

(* The same of the rewrite of the code of a unit. *)
Lemma know_in_code_sets_made (sets : list set_made) (syms : list symbol)
    (id : code_id) (c : code) :
  code_sets_made (know_in_code sets syms id c) = code_sets_made c.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  cbn [know_in_code].
  destruct (cm_is_stub metadata || among my_closure params);
    [ reflexivity | ].
  cbn [code_sets_made]. apply know_in_body_sets_made.
Qed.

Lemma know_code_in_bindings_sets_made (sets : list set_made)
    (syms : list symbol) (bindings : list static_binding) :
  flat_map binding_sets_made (map (know_code_in_binding sets syms) bindings)
  = flat_map binding_sets_made bindings.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    cbn [map flat_map know_code_in_binding binding_sets_made];
    [ reflexivity | | | ]; rewrite IH; [ | reflexivity | reflexivity ].
  rewrite know_in_code_sets_made. reflexivity.
Qed.

Fixpoint know_code_closures_sets_made (sets : list set_made) (e : expr)
    {struct e} :
  ∀ syms : list symbol,
    sets_made (know_code_closures sets syms e) = sets_made e
with know_code_in_handler_sets_made (sets : list set_made)
    (h : continuation_handler) {struct h} :
  ∀ syms : list symbol,
    handler_sets_made (know_code_in_handler sets syms h)
    = handler_sets_made h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros syms; cbn [know_code_closures sets_made]; try reflexivity.
    + rewrite (know_code_closures_sets_made sets body). reflexivity.
    + rewrite (know_code_closures_sets_made sets body),
        know_code_in_bindings_sets_made.
      reflexivity.
    + rewrite (know_code_in_handler_sets_made sets h),
        (know_code_closures_sets_made sets body).
      reflexivity.
    + rewrite (know_code_closures_sets_made sets body). f_equal.
      generalize (static_names symbols_of_binding body ++ syms).
      intros all.
      induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
        [ reflexivity | ].
      rewrite (know_code_in_handler_sets_made sets h), IH. reflexivity.
  - destruct h as [params metadata handler]. intros syms.
    cbn [know_code_in_handler handler_sets_made].
    apply know_code_closures_sets_made.
Qed.

Lemma know_in_code_groups_whole (sets : list set_made) (syms : list symbol)
    (id : code_id) (c : code) :
  code_groups_whole (know_in_code sets syms id c) = code_groups_whole c.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  cbn [know_in_code].
  destruct (cm_is_stub metadata || among my_closure params);
    [ reflexivity | ].
  cbn [code_groups_whole]. apply know_in_body_groups_whole.
Qed.

Lemma know_code_in_bindings_groups_whole (sets : list set_made)
    (syms : list symbol) (bindings : list static_binding) :
  forallb binding_groups_whole
    (map (know_code_in_binding sets syms) bindings)
  = forallb binding_groups_whole bindings.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    cbn [map forallb know_code_in_binding binding_groups_whole];
    [ reflexivity | | | ]; rewrite IH; [ | reflexivity | reflexivity ].
  rewrite know_in_code_groups_whole. reflexivity.
Qed.

(* A group binds its static sets whole whatever its code is. *)
Lemma binds_symbol_to_know_code (sets : list set_made) (syms : list symbol)
    (s : symbol) (const : static_const) (binding : static_binding) :
  binds_symbol_to s const (know_code_in_binding sets syms binding)
  = binds_symbol_to s const binding.
Proof. destruct binding; reflexivity. Qed.

Lemma existsb_binds_symbol_to_know_code (sets : list set_made)
    (syms : list symbol) (s : symbol) (const : static_const)
    (bindings : list static_binding) :
  existsb (binds_symbol_to s const)
    (map (know_code_in_binding sets syms) bindings)
  = existsb (binds_symbol_to s const) bindings.
Proof.
  induction bindings as [ | binding bindings IH]; cbn [map existsb];
    [ reflexivity | ].
  rewrite binds_symbol_to_know_code, IH. reflexivity.
Qed.

Lemma static_set_whole_know_code (sets : list set_made) (syms : list symbol)
    (bindings : list static_binding) (binding : static_binding) :
  static_set_whole (map (know_code_in_binding sets syms) bindings)
    (know_code_in_binding sets syms binding)
  = static_set_whole bindings binding.
Proof.
  destruct binding as [id c | id | sym const];
    cbn [know_code_in_binding static_set_whole]; try reflexivity.
  destruct const
    as [code slots | slot closure_symbols decls slots | tag fields
       | contents | contents | elements];
    try reflexivity.
  cbn [static_set_whole]. f_equal. apply forallb_same. intros [slot' s'].
  apply existsb_binds_symbol_to_know_code.
Qed.

Lemma know_code_in_bindings_whole (sets : list set_made) (syms : list symbol)
    (bindings : list static_binding) :
  forallb
    (static_set_whole (map (know_code_in_binding sets syms) bindings))
    (map (know_code_in_binding sets syms) bindings)
  = forallb (static_set_whole bindings) bindings.
Proof.
  generalize bindings at 1 3. intros all.
  induction bindings as [ | binding bindings IH]; cbn [map forallb];
    [ reflexivity | ].
  rewrite static_set_whole_know_code, IH. reflexivity.
Qed.

Fixpoint know_code_closures_groups_whole (sets : list set_made) (e : expr)
    {struct e} :
  ∀ syms : list symbol,
    groups_whole (know_code_closures sets syms e) = groups_whole e
with know_code_in_handler_groups_whole (sets : list set_made)
    (h : continuation_handler) {struct h} :
  ∀ syms : list symbol,
    handler_groups_whole (know_code_in_handler sets syms h)
    = handler_groups_whole h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros syms; cbn [know_code_closures groups_whole]; try reflexivity.
    + apply know_code_closures_groups_whole.
    + rewrite (know_code_closures_groups_whole sets body),
        know_code_in_bindings_whole, know_code_in_bindings_groups_whole.
      reflexivity.
    + rewrite (know_code_in_handler_groups_whole sets h),
        (know_code_closures_groups_whole sets body).
      reflexivity.
    + rewrite (know_code_closures_groups_whole sets body). f_equal.
      generalize (static_names symbols_of_binding body ++ syms).
      intros all.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (know_code_in_handler_groups_whole sets h), IH. reflexivity.
  - destruct h as [params metadata handler]. intros syms.
    cbn [know_code_in_handler handler_groups_whole].
    apply know_code_closures_groups_whole.
Qed.

Fixpoint know_code_closures_binds_statics (sets : list set_made) (e : expr) :
  ∀ syms : list symbol,
    binds_statics (know_code_closures sets syms e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros syms; cbn [know_code_closures binds_statics]; try reflexivity.
  - apply know_code_closures_binds_statics.
  - rewrite (know_code_closures_binds_statics sets body). f_equal.
    generalize (flat_map symbols_of_binding bindings ++ syms). intros all.
    induction bindings as [ | [id c | id | sym const] bindings IH];
      cbn [map existsb know_code_in_binding binding_binds]; congruence.
  - destruct h as [params metadata handler].
    cbn [know_code_in_handler handler_binds_statics].
    rewrite !know_code_closures_binds_statics. reflexivity.
  - rewrite (know_code_closures_binds_statics sets body). f_equal.
    generalize (static_names symbols_of_binding body ++ syms). intros all.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      cbn [map existsb know_code_in_handler handler_binds_statics];
      [ reflexivity | ].
    rewrite know_code_closures_binds_statics, IH. reflexivity.
Qed.

(* A group in which the body of one code is another binds its static
   sets whole as it did. *)
Lemma static_set_whole_code_changed (done after : list static_binding)
    (id : code_id) (c c' : code) (binding : static_binding) :
  static_set_whole (done ++ Bind_code id c' :: after) binding
  = static_set_whole (done ++ Bind_code id c :: after) binding.
Proof.
  destruct binding as [id0 c0 | id0 | sym const]; try reflexivity.
  destruct const
    as [code slots | slot closure_symbols decls slots | tag fields
       | contents | contents | elements];
    try reflexivity.
  cbn [static_set_whole]. f_equal. apply forallb_same. intros [slot' s'].
  rewrite !existsb_app. reflexivity.
Qed.

Lemma groups_whole_let_static_code (done after : list static_binding)
    (id : code_id) (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (code_body code_body' body : expr) :
  groups_whole code_body' = groups_whole code_body ->
  groups_whole
    (Let_static
       (done
        ++ Bind_code id (Code params my_closure ret exn metadata code_body')
        :: after)
       body)
  = groups_whole
      (Let_static
         (done
          ++ Bind_code id (Code params my_closure ret exn metadata code_body)
          :: after)
         body).
Proof.
  intros Hbody. cbn [groups_whole]. rewrite !forallb_app.
  cbn [forallb static_set_whole binding_groups_whole code_groups_whole].
  rewrite Hbody.
  rewrite
    (forallb_same _ _ done
       (static_set_whole_code_changed done after id
          (Code params my_closure ret exn metadata code_body)
          (Code params my_closure ret exn metadata code_body'))),
    (forallb_same _ _ after
       (static_set_whole_code_changed done after id
          (Code params my_closure ret exn metadata code_body)
          (Code params my_closure ret exn metadata code_body'))).
  reflexivity.
Qed.

(* ======================================================================== *)
(* The invariant                                                            *)
(* ======================================================================== *)

(* The groups of the terms of a state bind their static sets whole, and
   the static sets are bound. *)
Definition whole_inv (s : state) : Prop :=
  terms_inv groups_whole handler_groups_whole code_groups_whole s
  ∧ static_sets_bound (state_statics s).

Lemma groups_whole_terms_inv_step (s s' : state) :
  terms_inv groups_whole handler_groups_whole code_groups_whole s ->
  step s = Step s' ->
  terms_inv groups_whole handler_groups_whole code_groups_whole s'.
Proof.
  apply terms_inv_step.
  - intros x kind d body H. exact H.
  - intros bindings body H. cbn [groups_whole] in H.
    apply andb_true_iff in H as [H Hbody].
    apply andb_true_iff in H as [_ Hbindings].
    split; [ exact Hbody | ]. intros id c Hin.
    rewrite forallb_forall in Hbindings. exact (Hbindings _ Hin).
  - intros k h body H. cbn [groups_whole] in H.
    apply andb_true_iff in H. exact H.
  - intros handlers body H. cbn [groups_whole] in H.
    apply andb_true_iff in H. exact H.
  - intros ac. reflexivity.
  - intros params metadata handler H. exact H.
  - intros [params my_closure ret exn metadata body] H. exact H.
Qed.

Lemma whole_inv_kept : kept_by_step whole_inv.
Proof.
  intros s s' [Hterms Hbound] Hstep. split.
  - exact (groups_whole_terms_inv_step s s' Hterms Hstep).
  - destruct (step_statics s s' Hstep)
      as [Hsame | (bindings & body & Hexpr & Heval)].
    + rewrite Hsame. exact Hbound.
    + destruct Hterms as (Hwhole & _). rewrite Hexpr in Hwhole.
      cbn [groups_whole] in Hwhole.
      apply andb_true_iff in Hwhole as [Hwhole _].
      apply andb_true_iff in Hwhole as [Hwhole _].
      exact (eval_static_bindings_sets_bound (state_vars s) bindings
               (state_statics s) (state_statics s') Hwhole Hbound Heval).
Qed.

Section Known.

Variable sets : list set_made.

(* Every fact that the sets given pass the test of, the state has. *)
Definition facts_hold (s : state) : Prop :=
  ∀ (c : code_id) (f : function_slot) (fact : set_fact),
    fact_closed fact = true ->
    sets_made_have c f fact sets = true ->
    state_sets_have c f fact s.

Definition code_inv (s : state) : Prop := facts_hold s ∧ whole_inv s.

Lemma code_inv_kept : kept_by_step code_inv.
Proof.
  intros s s' [Hfacts Hwhole] Hstep. split.
  - intros c f fact Hclosed Htest.
    exact (state_sets_have_step c f fact Hclosed s s'
             (Hfacts c f fact Hclosed Htest) Hstep).
  - exact (whole_inv_kept s s' Hwhole Hstep).
Qed.

(* The invariant looks at the expression of a state only to ask the two
   tests of it. *)
Lemma code_inv_expr (e e' : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) :
  sets_made e' = sets_made e ->
  groups_whole e' = groups_whole e ->
  code_inv (State e vars conts statics hp)
  <-> code_inv (State e' vars conts statics hp).
Proof.
  intros Hsets Hwhole.
  unfold code_inv, facts_hold, whole_inv, state_sets_have, terms_inv,
    state_values_all.
  simpl_state. rewrite Hsets, Hwhole. reflexivity.
Qed.

(* What the invariant gives of the closures that the variables are
   bound to. *)
Definition facts_read (vars : env) (statics : static_env) : Prop :=
  static_sets_bound statics
  ∧ ∀ (c : code_id) (f : function_slot) (fact : set_fact) (x : variable)
      (v : value) (decls : list (function_slot * code_id))
      (slot_values : list (value_slot * value)),
      fact_closed fact = true ->
      sets_made_have c f fact sets = true ->
      lookup x vars = Some v ->
      closure_in_set statics v = Some (f, decls, slot_values) ->
      lookup f decls = Some c ->
      set_held_has fact decls slot_values = true.

Lemma code_inv_facts_read (s : state) :
  code_inv s -> facts_read (state_vars s) (state_statics s).
Proof.
  intros [Hfacts [_ Hbound]]. split; [ exact Hbound | ].
  intros c f fact x v decls slot_values Hclosed Htest Hx Hin_set Hcode.
  exact (closure_set_has c f fact s x v decls slot_values
           (Hfacts c f fact Hclosed Htest) Hx Hin_set Hcode).
Qed.

(* ======================================================================== *)
(* What is known holds                                                      *)
(* ======================================================================== *)

(* A variable known to be a closure of some code is bound to a closure
   of that code. *)
Definition codes_hold (known : closure_codes) (vars : env)
    (statics : static_env) : Prop :=
  ∀ (x : variable) (c : code_id),
    lookup x known = Some c ->
    ∃ (v : value) (slots : list (value_slot * value)),
      lookup x vars = Some v ∧ closure_of statics v = Some (c, slots).

Lemma codes_hold_unbind (known : closure_codes) (vars : env)
    (statics : static_env) (x : variable) (v : value) :
  codes_hold known vars statics ->
  codes_hold (unbind x known) ((x, v) :: vars) statics.
Proof.
  intros Hhold y c Hy. rewrite lookup_unbind in Hy. cbn [lookup].
  destruct (Pos.eqb y x); [ discriminate | ].
  exact (Hhold y c Hy).
Qed.

Lemma codes_hold_bind_params (statics : static_env)
    (params : list variable) :
  ∀ (known : closure_codes) (vs : list value) (vars handler_vars : env),
    codes_hold known vars statics ->
    bind_params params vs vars = Some handler_vars ->
    codes_hold (forget_code_vars params known) handler_vars statics.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hhold; simpl; try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, codes_hold_unbind, Hhold.
Qed.

Lemma codes_hold_extend (known : closure_codes) (vars : env)
    (statics statics' : static_env) :
  statics_extend statics statics' ->
  codes_hold known vars statics ->
  codes_hold known vars statics'.
Proof.
  intros [_ Hvalues] Hhold x c Hx.
  destruct (Hhold x c Hx) as (v & slots & Hv & Hclosure).
  exists v, slots. split; [ exact Hv | ].
  exact (closure_of_statics_extend statics statics' v
           (fun s allocated _ H => Hvalues s allocated H) (c, slots)
           Hclosure).
Qed.

(* The tests that [sibling_code] and [slot_held] have passed. *)
Lemma sibling_code_test (c : code_id) (f g : function_slot) (d : code_id) :
  sibling_code sets c f g = Some d ->
  sets_made_have c f (Code_at g d) sets = true.
Proof.
  unfold sibling_code.
  destruct (set_with_code sets c f) as [set | ]; [ | discriminate ].
  destruct (lookup g (fst set)) as [d' | ]; [ | discriminate ].
  destruct (sets_made_have c f (Code_at g d') sets) eqn:Htest;
    [ | discriminate ].
  intros [= <-]. exact Htest.
Qed.

Lemma slot_held_test (c : code_id) (f : function_slot) (slot : value_slot)
    (a : simple) :
  slot_held sets c f slot = Some a ->
  closed_simple a = true
  ∧ sets_made_have c f (Slot_holds slot a) sets = true.
Proof.
  unfold slot_held.
  destruct (set_with_code sets c f) as [set | ]; [ | discriminate ].
  destruct (lookup slot (snd set)) as [a' | ]; [ | discriminate ].
  destruct (closed_simple a' && sets_made_have c f (Slot_holds slot a') sets)
    eqn:Htest;
    [ | discriminate ].
  intros [= <-]. apply andb_true_iff in Htest. exact Htest.
Qed.

(* The set of a closure that is known by its code. *)
Lemma closure_of_in_set (statics : static_env) (v : value) (c : code_id)
    (slots : list (value_slot * value)) (f : function_slot)
    (decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) :
  closure_of statics v = Some (c, slots) ->
  closure_in_set statics v = Some (f, decls, slot_values) ->
  lookup f decls = Some c.
Proof.
  intros Hclosure Hin_set.
  destruct (proj1 (closure_in_set_of statics v c slots) Hclosure)
    as (slot & decls' & Hin_set' & Hcode).
  rewrite Hin_set in Hin_set'. injection Hin_set' as <- <- _.
  exact Hcode.
Qed.

(* The projection of a function slot from a closure of known code is a
   closure of the code that [sibling_code] gives. *)
Lemma sibling_closure (vars : env) (statics : static_env) (x : variable)
    (v w : value) (c d : code_id) (f g : function_slot)
    (slots : list (value_slot * value)) :
  facts_read vars statics ->
  lookup x vars = Some v ->
  closure_of statics v = Some (c, slots) ->
  sibling_code sets c f g = Some d ->
  project_function_slot statics f g v = Some w ->
  ∃ slots' : list (value_slot * value),
    closure_of statics w = Some (d, slots').
Proof.
  intros [Hbound Hread] Hx Hclosure Hsibling Hproject.
  destruct (project_function_slot_bound statics f g v w Hbound Hproject)
    as (decls & slot_values & Hv & Hw).
  pose proof (closure_of_in_set statics v c slots f decls slot_values
                Hclosure Hv) as Hcode.
  pose proof (Hread c f (Code_at g d) x v decls slot_values eq_refl
                (sibling_code_test c f g d Hsibling) Hx Hv Hcode) as Hfact.
  cbn [set_held_has] in Hfact. unfold has_code_at in Hfact.
  destruct (lookup g decls) as [d' | ] eqn:Hg; [ | discriminate ].
  apply Pos.eqb_eq in Hfact. subst d'.
  exists slot_values.
  apply (proj2 (closure_in_set_of statics w d slot_values)).
  exists g, decls. split; [ exact Hw | exact Hg ].
Qed.

(* The projection of a value slot. *)
Lemma read_code_slot_sound (syms : list symbol) (known : closure_codes)
    (vars : env) (statics : static_env) (d : named) (v : value) :
  facts_read vars statics ->
  codes_hold known vars statics ->
  eval_named statics vars d = Some v ->
  eval_named statics vars (read_code_slot sets syms known d) = Some v.
Proof.
  intros [Hbound Hread] Hhold.
  destruct d as [a | p args | code slots | slot decls slots | slot closure
    | project_from slot closure | move_from move_to closure];
    cbn [read_code_slot]; try (intros H; exact H).
  destruct closure as [x | sym | k]; try (intros H; exact H).
  destruct (lookup x known) as [c | ] eqn:Hx; [ | intros H; exact H ].
  destruct (slot_held sets c project_from slot) as [a | ] eqn:Hheld;
    [ | intros H; exact H ].
  destruct (simple_statics_in_scope syms a); [ | intros H; exact H ].
  destruct (Hhold x c Hx) as (k & kslots & Hk & Hclosure).
  destruct (slot_held_test c project_from slot a Hheld) as [Hclosed Htest].
  cbn [eval_named eval_simple]. rewrite Hk. simpl.
  unfold project_value_slot.
  destruct (closure_in_set statics k) as [[[at_slot decls] slot_values] | ]
    eqn:Hin_set;
    [ | discriminate ].
  simpl. destruct (Pos.eqb_spec at_slot project_from) as [-> | _];
    [ | discriminate ].
  intros Hlookup.
  pose proof (closure_of_in_set statics k c kslots project_from decls
                slot_values Hclosure Hin_set) as Hcode.
  pose proof (Hread c project_from (Slot_holds slot a) x k decls slot_values
                Hclosed Htest Hk Hin_set Hcode) as Hfact.
  cbn [set_held_has] in Hfact. rewrite Hlookup in Hfact.
  exact (value_is_eval a v vars Hfact).
Qed.

(* Under a [Let]. *)
Lemma codes_hold_under_let (known : closure_codes) (x : variable)
    (d : named) (vars : env) (statics : static_env) (v : value) :
  facts_read vars statics ->
  codes_hold known vars statics ->
  eval_named statics vars d = Some v ->
  codes_hold (codes_under_let sets known x d) ((x, v) :: vars) statics.
Proof.
  intros Hread Hhold Hd. unfold codes_under_let.
  pose proof (codes_hold_unbind known vars statics x v Hhold) as Hforgot.
  destruct (closure_code_made sets known d) as [made | ] eqn:Hmade;
    [ | exact Hforgot ].
  assert (Hv :
    ∃ slots : list (value_slot * value),
      closure_of statics v = Some (made, slots)).
  { destruct d as [a | p args | code slots | slot decls slots | slot closure
      | project_from slot closure | move_from move_to closure];
      cbn [closure_code_made] in Hmade; try discriminate.
    destruct closure as [y | sym | k]; try discriminate.
    destruct (lookup y known) as [c | ] eqn:Hy; [ | discriminate ].
    destruct (Hhold y c Hy) as (w & wslots & Hw & Hclosure).
    cbn [eval_named eval_simple] in Hd. rewrite Hw in Hd. simpl in Hd.
    exact (sibling_closure vars statics y w v c made move_from move_to
             wslots Hread Hw Hclosure Hmade Hd). }
  destruct Hv as (slots & Hv).
  intros y c Hy. specialize (Hforgot y c).
  cbn [lookup] in Hy, Hforgot |- *.
  destruct (Pos.eqb y x); [ | exact (Hforgot Hy) ].
  injection Hy as <-. exists v, slots. split; [ reflexivity | exact Hv ].
Qed.

(* A call. *)
Lemma call_code_closure_step (known : closure_codes) (statics : static_env)
    (hp : heap) (vars : env) (conts : cont_env) (ap : apply_expr) :
  codes_hold known vars statics ->
  step (State (Apply (call_code_closure known ap)) vars conts statics hp)
  = step (State (Apply ap) vars conts statics hp).
Proof.
  intros Hhold. unfold call_code_closure.
  destruct (ap_call_kind ap) as [named | ] eqn:Hkind;
    [ destruct (ap_callee ap); reflexivity | ].
  destruct (ap_callee ap) as [x | sym | k] eqn:Hcallee; try reflexivity.
  destruct (lookup x known) as [c | ] eqn:Hx; [ | reflexivity ].
  destruct (Hhold x c Hx) as (v & slots & Hv & Hclosure).
  rewrite !step_apply. unfold exn_extra_simples.
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity].
  rewrite Hkind, Hcallee. cbn [eval_simple]. rewrite Hv.
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (map fst (ap_exn_extra_args ap)))
    as [extra | ];
    [ | reflexivity ].
  unfold call_function.
  rewrite !(shape_of_call_closure statics v _ vs _ _
              (closure_not_partial statics v _ Hclosure)),
    (shape_of_closure_call_kind statics v (Direct c) vs _ _ c slots
       Hclosure (or_intror eq_refl)).
  reflexivity.
Qed.

(* ======================================================================== *)
(* The body of code                                                         *)
(* ======================================================================== *)

(* The invariant, in a static environment that extends a given one: what
   a handler is run from. *)
Definition inv_at (statics0 : static_env) (s : state) : Prop :=
  code_inv s ∧ statics_extend statics0 (state_statics s).

Lemma inv_at_kept (statics0 : static_env) : kept_by_step (inv_at statics0).
Proof.
  intros s s' [Hinv Hextend] Hstep.
  split; [ exact (code_inv_kept s s' Hinv Hstep) | ].
  exact (statics_extend_trans _ _ _ Hextend (step_extends_statics s s' Hstep)).
Qed.

Lemma inv_at_expr (statics0 : static_env) (e e' : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) :
  sets_made e' = sets_made e ->
  groups_whole e' = groups_whole e ->
  inv_at statics0 (State e vars conts statics hp)
  <-> inv_at statics0 (State e' vars conts statics hp).
Proof.
  intros Hsets Hwhole.
  pose proof (code_inv_expr e e' vars conts statics hp Hsets Hwhole) as Hiff.
  unfold inv_at. simpl_state.
  split; intros [H1 H2]; (split; [ apply Hiff, H1 | exact H2 ]).
Qed.

(* At a [Let], what is known under it holds once it has run; a handler
   runs under the variables of its definition with its parameters bound
   and from a static environment that extends that of its definition,
   where what is known less the parameters holds; at a call, the call
   made direct steps as the call does. *)
Fixpoint know_in_body_refines (e : expr) {struct e} :
  ∀ (syms : list symbol) (known : closure_codes) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap),
    code_inv (State e vars conts statics hp) ->
    codes_hold known vars statics ->
    refines (State e vars conts statics hp)
      (State (know_in_body sets syms known e) vars conts statics hp).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros syms known vars conts statics hp Hinv Hhold; cbn [know_in_body];
    try apply refines_refl.
  - (* Let *)
    pose proof (code_inv_facts_read _ Hinv) as Hread. simpl_state.
    destruct (eval_named statics vars d) as [v | ] eqn:Hd;
      [ | apply step_undefined_refines; rewrite step_let, Hd; reflexivity ].
    apply let_refines_step.
    + rewrite Hd.
      exact (read_code_slot_sound syms known vars statics d v Hread Hhold Hd).
    + intros v' Hv' Hkind. apply know_in_body_refines.
      * apply (code_inv_kept _ _ Hinv).
        rewrite step_let, Hv', Hkind, kind_eqb_refl. reflexivity.
      * exact (codes_hold_under_let known x d vars statics v' Hread Hhold
                 Hv').
  - (* Let_static *)
    apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Heval;
      [ | exact I ].
    apply know_in_body_refines.
    + apply (code_inv_kept _ _ Hinv).
      rewrite step_let_static, Heval. reflexivity.
    + exact (codes_hold_extend known vars statics extended
               (eval_static_bindings_extend vars bindings statics extended
                  Heval)
               Hhold).
  - (* Let_cont_non_recursive *)
    destruct h as [params metadata handler]. cbn [know_in_body_handler].
    set (handler' :=
           know_in_body sets syms (forget_code_vars params known) handler).
    assert (Hinv' :
      code_inv
        (State
           (Let_cont_non_recursive k (Handler params metadata handler') body)
           vars conts statics hp)).
    { revert Hinv. apply code_inv_expr.
      - cbn [sets_made handler_sets_made]. unfold handler'.
        rewrite know_in_body_sets_made. reflexivity.
      - cbn [groups_whole handler_groups_whole]. unfold handler'.
        rewrite know_in_body_groups_whole. reflexivity. }
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k (Handler params metadata handler') body)
         vars conts statics hp)).
    + apply (let_cont_handler_refines_from (inv_at statics) k params metadata
               handler handler' body vars conts statics hp
               (inv_at_kept statics)).
      * exact (conj Hinv (statics_extend_refl statics)).
      * exact (conj Hinv' (statics_extend_refl statics)).
      * intros vs handler_vars conts' statics' hp' _ Hbind. split.
        -- apply inv_at_expr; unfold handler';
             [ apply know_in_body_sets_made
             | apply know_in_body_groups_whole ].
        -- intros [Hentry Hextend]. simpl_state.
           apply know_in_body_refines; [ exact Hentry | ].
           exact (codes_hold_bind_params statics' params known vs vars
                    handler_vars
                    (codes_hold_extend known vars statics statics' Hextend
                       Hhold)
                    Hbind).
    + apply let_cont_non_recursive_refines_step.
      apply know_in_body_refines; [ | exact Hhold ].
      apply (code_inv_kept _ _ Hinv'). apply step_let_cont_non_recursive.
  - (* Let_cont_recursive *)
    set (know :=
           fun kh : continuation * continuation_handler =>
             let '(k, h) := kh in
             (k, know_in_body_handler sets syms known h)).
    apply (refines_trans _
      (State (Let_cont_recursive (map know handlers) body) vars conts
         statics hp)).
    + assert (Hhandlers :
        ∀ done : list (continuation * continuation_handler),
          code_inv
            (State (Let_cont_recursive (done ++ handlers) body) vars conts
               statics hp)
          ->
          refines
            (State (Let_cont_recursive (done ++ handlers) body) vars conts
               statics hp)
            (State (Let_cont_recursive (done ++ map know handlers) body)
               vars conts statics hp)).
      { clear Hinv.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          intros done Hdone; cbn [map]; [ apply refines_refl | ].
        unfold know at 1. cbn [know_in_body_handler].
        set (handler' :=
               know_in_body sets syms (forget_code_vars params known)
                 handler).
        assert (Hdone' :
          code_inv
            (State
               (Let_cont_recursive
                  (done ++ (k, Handler params metadata handler') :: handlers)
                  body)
               vars conts statics hp)).
        { revert Hdone. apply code_inv_expr.
          - cbn [sets_made]. rewrite !flat_map_app.
            cbn [flat_map handler_sets_made]. unfold handler'.
            rewrite know_in_body_sets_made. reflexivity.
          - cbn [groups_whole]. rewrite !forallb_app.
            cbn [forallb handler_groups_whole]. unfold handler'.
            rewrite know_in_body_groups_whole. reflexivity. }
        apply (refines_trans _
          (State
             (Let_cont_recursive
                (done ++ (k, Handler params metadata handler') :: handlers)
                body)
             vars conts statics hp)).
        - apply (let_cont_recursive_handler_refines_from (inv_at statics)
                   done k params metadata handler handler' handlers body
                   vars conts statics hp (inv_at_kept statics)).
          + exact (conj Hdone (statics_extend_refl statics)).
          + exact (conj Hdone' (statics_extend_refl statics)).
          + symmetry. unfold handler'. apply know_in_body_binds_statics.
          + intros vs handler_vars conts' statics' hp' _ Hbind. split.
            * apply inv_at_expr; unfold handler';
                [ apply know_in_body_sets_made
                | apply know_in_body_groups_whole ].
            * intros [Hentry Hextend]. simpl_state.
              apply know_in_body_refines; [ exact Hentry | ].
              exact (codes_hold_bind_params statics' params known vs vars
                       handler_vars
                       (codes_hold_extend known vars statics statics'
                          Hextend Hhold)
                       Hbind).
        - specialize
            (IH (done ++ [(k, Handler params metadata handler')])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone'). }
      exact (Hhandlers [] Hinv).
    + assert (Hinv' :
        code_inv
          (State (Let_cont_recursive (map know handlers) body) vars conts
             statics hp)).
      { revert Hinv. apply code_inv_expr.
        - cbn [sets_made]. f_equal. clear.
          induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
            [ reflexivity | ].
          unfold know at 1. rewrite know_in_body_handler_sets_made, IH.
          reflexivity.
        - cbn [groups_whole]. f_equal. clear.
          induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
            [ reflexivity | ].
          unfold know at 1. rewrite know_in_body_handler_groups_whole, IH.
          reflexivity. }
      apply let_cont_recursive_refines_step; [ reflexivity | ].
      intros Hbinds. apply know_in_body_refines; [ | exact Hhold ].
      apply (code_inv_kept _ _ Hinv').
      rewrite step_let_cont_recursive, Hbinds. reflexivity.
  - (* Apply *)
    apply equivalent_refines, step_eq_equivalent. symmetry.
    apply call_code_closure_step, Hhold.
Qed.

(* ======================================================================== *)
(* The code of a unit                                                       *)
(* ======================================================================== *)

(* Down the term. The code of a group and the handlers of a recursive
   [Let_cont] are rewritten one at a time, each from the states that
   have the invariant; the body of code, from those in which the callee
   is a closure of that code
   (FlambdaContextWhereSpec.let_static_code_refines_from_called_statement). *)
Fixpoint know_code_closures_refines_aux (e : expr) {struct e} :
  ∀ (syms : list symbol) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap),
    code_inv (State e vars conts statics hp) ->
    refines (State e vars conts statics hp)
      (State (know_code_closures sets syms e) vars conts statics hp).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros syms vars conts statics hp Hinv; cbn [know_code_closures];
    try apply refines_refl.
  - (* Let *)
    apply let_refines_step; [ reflexivity | ].
    intros v Hv Hkind. apply know_code_closures_refines_aux.
    apply (code_inv_kept _ _ Hinv).
    rewrite step_let, Hv, Hkind, kind_eqb_refl. reflexivity.
  - (* Let_static *)
    generalize (flat_map symbols_of_binding bindings ++ syms). intros all.
    apply (refines_trans _
      (State
         (Let_static (map (know_code_in_binding sets all) bindings) body)
         vars conts statics hp)).
    + assert (Hcode :
        ∀ done : list static_binding,
          code_inv
            (State (Let_static (done ++ bindings) body) vars conts statics hp)
          ->
          refines
            (State (Let_static (done ++ bindings) body) vars conts statics hp)
            (State
               (Let_static
                  (done ++ map (know_code_in_binding sets all) bindings)
                  body)
               vars conts statics hp)).
      { clear Hinv.
        induction bindings
          as [ | [id [params my_closure ret exn metadata code_body] | id
                 | sym const]
                 bindings IH];
          intros done Hdone; cbn [map]; [ apply refines_refl | | | ].
        - cbn [know_code_in_binding know_in_code].
          destruct (cm_is_stub metadata || among my_closure params)
            eqn:Hleft.
          { (* The code is left. *)
            specialize
              (IH
                 (done
                  ++ [Bind_code id
                        (Code params my_closure ret exn metadata code_body)])).
            rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone). }
          apply orb_false_iff in Hleft as [_ Hmine].
          set (code_body' :=
                 know_in_body sets all [(my_closure, id)] code_body).
          assert (Hdone' :
            code_inv
              (State
                 (Let_static
                    (done
                     ++ Bind_code id
                          (Code params my_closure ret exn metadata code_body')
                     :: bindings)
                    body)
                 vars conts statics hp)).
          { revert Hdone. apply code_inv_expr.
            - cbn [sets_made]. rewrite !flat_map_app.
              cbn [flat_map binding_sets_made code_sets_made].
              unfold code_body'. rewrite know_in_body_sets_made.
              reflexivity.
            - apply groups_whole_let_static_code. unfold code_body'.
              symmetry. apply know_in_body_groups_whole. }
          apply (refines_trans _
            (State
               (Let_static
                  (done
                   ++ Bind_code id
                        (Code params my_closure ret exn metadata code_body')
                   :: bindings)
                  body)
               vars conts statics hp)).
          + apply (let_static_code_refines_from_called code_inv done id
                     params my_closure ret exn metadata code_body code_body'
                     bindings body vars conts statics hp code_inv_kept Hdone
                     Hdone').
            * symmetry. unfold code_body'. apply know_in_body_binds_statics.
            * intros vs callee callee_vars return_to exn_to statics' hp'
                value_slots Hbind _ _ Hclosure.
              split.
              -- apply code_inv_expr; unfold code_body';
                   [ apply know_in_body_sets_made
                   | apply know_in_body_groups_whole ].
              -- intros Hentry.
                 apply know_in_body_refines; [ exact Hentry | ].
                 intros x c Hx. cbn [lookup] in Hx.
                 destruct (Pos.eqb_spec x my_closure) as [-> | _];
                   [ | discriminate ].
                 injection Hx as <-. exists callee, value_slots.
                 split; [ | exact Hclosure ].
                 rewrite (bind_params_lookup_other my_closure params vs _
                            callee_vars Hmine Hbind).
                 cbn [lookup]. rewrite Pos.eqb_refl. reflexivity.
          + specialize
              (IH
                 (done
                  ++ [Bind_code id
                        (Code params my_closure ret exn metadata code_body')])).
            rewrite <- !app_assoc in IH. cbn [app] in IH.
            exact (IH Hdone').
        - cbn [know_code_in_binding].
          specialize (IH (done ++ [Bind_deleted_code id])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone).
        - cbn [know_code_in_binding].
          specialize (IH (done ++ [Bind_symbol sym const])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone). }
      exact (Hcode [] Hinv).
    + assert (Hinv' :
        code_inv
          (State
             (Let_static (map (know_code_in_binding sets all) bindings) body)
             vars conts statics hp)).
      { revert Hinv. apply code_inv_expr.
        - cbn [sets_made]. rewrite know_code_in_bindings_sets_made.
          reflexivity.
        - cbn [groups_whole].
          rewrite know_code_in_bindings_whole,
            know_code_in_bindings_groups_whole.
          reflexivity. }
      apply let_static_refines_step.
      destruct
        (eval_static_bindings vars
           (map (know_code_in_binding sets all) bindings) statics)
        as [extended | ] eqn:Heval;
        [ | exact I ].
      apply know_code_closures_refines_aux. apply (code_inv_kept _ _ Hinv').
      rewrite step_let_static, Heval. reflexivity.
  - (* Let_cont_non_recursive *)
    destruct h as [params metadata handler]. cbn [know_code_in_handler].
    generalize (static_names symbols_of_binding body ++ syms). intros all.
    assert (Hinv' :
      code_inv
        (State
           (Let_cont_non_recursive k
              (Handler params metadata (know_code_closures sets all handler))
              body)
           vars conts statics hp)).
    { revert Hinv. apply code_inv_expr.
      - cbn [sets_made handler_sets_made].
        rewrite know_code_closures_sets_made. reflexivity.
      - cbn [groups_whole handler_groups_whole].
        rewrite know_code_closures_groups_whole. reflexivity. }
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata (know_code_closures sets all handler))
            body)
         vars conts statics hp)).
    + apply (let_cont_handler_refines_from code_inv k params metadata handler
               (know_code_closures sets all handler) body vars conts statics
               hp code_inv_kept Hinv Hinv').
      intros vs handler_vars conts' statics' hp' _ _. split.
      * apply code_inv_expr;
          [ apply know_code_closures_sets_made
          | apply know_code_closures_groups_whole ].
      * intros Hentry. apply know_code_closures_refines_aux. exact Hentry.
    + apply let_cont_non_recursive_refines_step.
      apply know_code_closures_refines_aux. apply (code_inv_kept _ _ Hinv').
      apply step_let_cont_non_recursive.
  - (* Let_cont_recursive *)
    generalize (static_names symbols_of_binding body ++ syms). intros all.
    set (know :=
           fun kh : continuation * continuation_handler =>
             let '(k, h) := kh in (k, know_code_in_handler sets all h)).
    apply (refines_trans _
      (State (Let_cont_recursive (map know handlers) body) vars conts
         statics hp)).
    + assert (Hhandlers :
        ∀ done : list (continuation * continuation_handler),
          code_inv
            (State (Let_cont_recursive (done ++ handlers) body) vars conts
               statics hp)
          ->
          refines
            (State (Let_cont_recursive (done ++ handlers) body) vars conts
               statics hp)
            (State (Let_cont_recursive (done ++ map know handlers) body)
               vars conts statics hp)).
      { clear Hinv.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          intros done Hdone; cbn [map]; [ apply refines_refl | ].
        unfold know at 1. cbn [know_code_in_handler].
        assert (Hdone' :
          code_inv
            (State
               (Let_cont_recursive
                  (done
                   ++ (k,
                       Handler params metadata
                         (know_code_closures sets all handler))
                   :: handlers)
                  body)
               vars conts statics hp)).
        { revert Hdone. apply code_inv_expr.
          - cbn [sets_made]. rewrite !flat_map_app.
            cbn [flat_map handler_sets_made].
            rewrite know_code_closures_sets_made. reflexivity.
          - cbn [groups_whole]. rewrite !forallb_app.
            cbn [forallb handler_groups_whole].
            rewrite know_code_closures_groups_whole. reflexivity. }
        apply (refines_trans _
          (State
             (Let_cont_recursive
                (done
                 ++ (k,
                     Handler params metadata
                       (know_code_closures sets all handler))
                 :: handlers)
                body)
             vars conts statics hp)).
        - apply (let_cont_recursive_handler_refines_from code_inv done k
                   params metadata handler
                   (know_code_closures sets all handler) handlers body vars
                   conts statics hp code_inv_kept Hdone Hdone').
          + symmetry. apply know_code_closures_binds_statics.
          + intros vs handler_vars conts' statics' hp' _ _. split.
            * apply code_inv_expr;
                [ apply know_code_closures_sets_made
                | apply know_code_closures_groups_whole ].
            * intros Hentry. apply know_code_closures_refines_aux.
              exact Hentry.
        - specialize
            (IH
               (done
                ++ [(k,
                     Handler params metadata
                       (know_code_closures sets all handler))])).
          rewrite <- !app_assoc in IH. cbn [app] in IH. exact (IH Hdone'). }
      exact (Hhandlers [] Hinv).
    + assert (Hinv' :
        code_inv
          (State (Let_cont_recursive (map know handlers) body) vars conts
             statics hp)).
      { revert Hinv. apply code_inv_expr.
        - cbn [sets_made]. f_equal. clear.
          induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
            [ reflexivity | ].
          unfold know at 1. rewrite know_code_in_handler_sets_made, IH.
          reflexivity.
        - cbn [groups_whole]. f_equal. clear.
          induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
            [ reflexivity | ].
          unfold know at 1. rewrite know_code_in_handler_groups_whole, IH.
          reflexivity. }
      apply let_cont_recursive_refines_step; [ reflexivity | ].
      intros Hbinds. apply know_code_closures_refines_aux.
      apply (code_inv_kept _ _ Hinv').
      rewrite step_let_cont_recursive, Hbinds. reflexivity.
Qed.

(* The states that a unit starts in. *)
Lemma facts_hold_from (e : expr) (conts : cont_env) (statics : static_env) :
  incl (sets_made e) sets ->
  (∀ (c : code_id) (f : function_slot) (fact : set_fact),
    conts_sets_have c f fact conts
    ∧ conts_values_all (fun v => value_sets_have c f fact v = true) conts
    ∧ code_sets_have c f fact statics
    ∧ Forall
        (fun binding => allocated_sets_have c f fact (snd binding) = true)
        (static_values statics)) ->
  facts_hold (State e [] conts statics empty_heap).
Proof.
  intros Hincl Hrest c f fact _ Htest.
  destruct (Hrest c f fact) as (Hconts & Hvalues & Hcode & Hstatics).
  apply state_sets_have_from; try assumption.
  unfold sets_made_have in Htest |- *. rewrite forallb_forall in Htest |- *.
  intros set Hin. exact (Htest set (Hincl set Hin)).
Qed.

End Known.

(* ======================================================================== *)
(* The static environment of the imports, and what may follow a unit        *)
(* ======================================================================== *)

(* What the static environment of the imports binds a symbol to. *)
Lemma imports_statics_lookup (imports : list (symbol * import_class))
    (s : symbol) (allocated : statically_allocated_value) :
  lookup s (static_values (imports_statics imports)) = Some allocated ->
  ∃ c : import_class, allocated = Static_imported c.
Proof.
  unfold imports_statics. cbn [static_values]. intros Hlookup.
  apply lookup_in, in_map_iff in Hlookup as ([s' c] & Heq & _).
  injection Heq as _ <-. exists c. reflexivity.
Qed.

Lemma imports_sets_bound (imports : list (symbol * import_class)) :
  static_sets_bound (imports_statics imports).
Proof.
  intros s slot symbols decls slot_values Hs.
  destruct (imports_statics_lookup imports s _ Hs) as (c & Heq).
  discriminate Heq.
Qed.

Lemma imports_sets_have (imports : list (symbol * import_class))
    (c : code_id) (f : function_slot) (fact : set_fact) :
  code_sets_have c f fact (imports_statics imports)
  ∧ Forall
      (fun binding => allocated_sets_have c f fact (snd binding) = true)
      (static_values (imports_statics imports)).
Proof.
  split.
  - intros id code Hlookup. discriminate Hlookup.
  - unfold imports_statics. cbn [static_values]. apply Forall_forall.
    intros [s allocated] Hin.
    apply in_map_iff in Hin as ([s' c'] & Heq & _).
    injection Heq as _ <-. reflexivity.
Qed.

(* What may follow a unit names no code id, so it makes no set that has
   code; and it binds nothing static. *)
Lemma named_no_code_sets_have (c : code_id) (f : function_slot)
    (fact : set_fact) (syms : list symbol) (d : named) :
  named_statics_in_scope syms [] d = true ->
  sets_made_have c f fact (named_sets_made d) = true.
Proof.
  destruct d as [ | | code slots | slot decls slots | | | ]; simpl;
    try reflexivity.
  - discriminate.
  - intros H. apply andb_true_iff in H as [Hdecls _].
    destruct decls as [ | decl decls]; [ reflexivity | ].
    simpl in Hdecls. discriminate Hdecls.
Qed.

Fixpoint observer_sets_have (c : code_id) (f : function_slot)
    (fact : set_fact) (e : expr) {struct e} :
  binds_no_static e = true ->
  expr_statics_in_scope [] [] e = true ->
  sets_made_have c f fact (sets_made e) = true.
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    cbn [sets_made]; try (intros _ _; reflexivity).
  - intros Hno H. simpl in Hno, H. apply andb_true_iff in H as [Hd Hbody].
    rewrite sets_made_have_app, (named_no_code_sets_have c f fact [] d Hd).
    exact (observer_sets_have c f fact body Hno Hbody).
  - discriminate.
  - destruct h as [params metadata handler]. cbn [handler_sets_made].
    intros Hno H. simpl in Hno, H.
    apply andb_true_iff in Hno as [Hno_h Hno_b].
    apply andb_true_iff in H as [Hh Hb].
    rewrite (no_static_names symbols_of_binding body Hno_b),
      (no_static_names declared_code_ids_of_binding body Hno_b)
      in Hh.
    simpl in Hh.
    rewrite sets_made_have_app,
      (observer_sets_have c f fact handler Hno_h Hh),
      (observer_sets_have c f fact body Hno_b Hb).
    reflexivity.
  - intros Hno H. simpl in Hno, H.
    apply andb_true_iff in Hno as [Hno_hs Hno_b].
    apply andb_true_iff in H as [Hhs Hb].
    rewrite sets_made_have_app, (observer_sets_have c f fact body Hno_b Hb),
      andb_true_r.
    rewrite (no_static_names symbols_of_binding body Hno_b),
      (no_static_names declared_code_ids_of_binding body Hno_b)
      in Hhs.
    simpl in Hhs.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      [ reflexivity | ].
    simpl in Hno_hs, Hhs.
    apply andb_true_iff in Hno_hs as [Hno_h Hno_rest].
    apply andb_true_iff in Hhs as [Hh Hrest].
    cbn [flat_map handler_sets_made].
    rewrite sets_made_have_app,
      (observer_sets_have c f fact handler Hno_h Hh).
    exact (IH Hno_rest Hrest).
Qed.

Fixpoint binds_no_static_groups_whole (e : expr) {struct e} :
  binds_no_static e = true -> groups_whole e = true.
Proof.
  destruct e
    as [ x xkind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try (intros _; reflexivity).
  - apply binds_no_static_groups_whole.
  - discriminate.
  - destruct h as [params metadata handler]. simpl. intros Hno.
    apply andb_true_iff in Hno as [Hh Hb].
    rewrite (binds_no_static_groups_whole handler Hh),
      (binds_no_static_groups_whole body Hb).
    reflexivity.
  - intros Hno. apply andb_true_iff in Hno as [Hhs Hb].
    rewrite (binds_no_static_groups_whole body Hb), andb_true_r.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    simpl in Hhs. apply andb_true_iff in Hhs as [Hh Hrest].
    rewrite (binds_no_static_groups_whole handler Hh), (IH Hrest).
    reflexivity.
Qed.

Lemma observers_sets_have (c : code_id) (f : function_slot)
    (fact : set_fact) (slots : list value_slot) (conts : cont_env) :
  observers slots conts -> conts_sets_have c f fact conts.
Proof.
  unfold observers, conts_sets_have.
  induction conts as [ | [k cont] conts IH]; simpl;
    [ intros _; exact I | ].
  intros H. inversion H as [ | kc rest Hc Hrest]; subst.
  split; [ | exact (IH Hrest) ].
  revert Hc.
  destruct cont
    as [ [params metadata body] vars kept recursive | k0 | c0 extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction; [ | intros _; exact I ].
  destruct vars; [ | contradiction ].
  destruct kept; [ | contradiction ].
  destruct recursive; [ | contradiction ].
  intros (Hno & Hscope & _).
  split; [ exact (observer_sets_have c f fact body Hno Hscope) | ].
  split; [ reflexivity | exact I ].
Qed.

Lemma observers_groups_whole (slots : list value_slot) (conts : cont_env) :
  observers slots conts -> conts_terms_ok handler_groups_whole conts.
Proof.
  unfold observers, conts_terms_ok.
  induction conts as [ | [k cont] conts IH]; simpl;
    [ intros _; exact I | ].
  intros H. inversion H as [ | kc rest Hc Hrest]; subst.
  split; [ | exact (IH Hrest) ].
  revert Hc.
  destruct cont
    as [ [params metadata body] vars kept recursive | k0 | c0 extra
       | remaining remaining_arity return_arity return_to exn_to ];
    simpl; try contradiction; [ | intros _; exact I ].
  destruct vars; [ | contradiction ].
  destruct kept; [ | contradiction ].
  destruct recursive; [ | contradiction ].
  intros (Hno & _ & _).
  split; [ exact (binds_no_static_groups_whole body Hno) | ].
  split; [ reflexivity | exact I ].
Qed.

(* The invariant of the state that a unit starts in, from its imports
   and with what may follow it. *)
Lemma code_inv_observed (sets : list set_made)
    (imports : list (symbol * import_class)) (slots : list value_slot)
    (conts : cont_env) (e : expr) :
  incl (sets_made e) sets ->
  groups_whole e = true ->
  observers slots conts ->
  code_inv sets (State e [] conts (imports_statics imports) empty_heap).
Proof.
  intros Hincl Hwhole Hobservers. split.
  - apply facts_hold_from; [ exact Hincl | ].
    intros c f fact.
    destruct (imports_sets_have imports c f fact) as [Hcode Hstatics].
    exact (conj (observers_sets_have c f fact slots conts Hobservers)
             (conj (observers_values_all _ slots conts Hobservers)
                (conj Hcode Hstatics))).
  - split; [ | apply imports_sets_bound ].
    unfold terms_inv. simpl_state.
    refine (conj Hwhole (conj (observers_groups_whole slots conts Hobservers)
                           _)).
    intros id c Hlookup. discriminate Hlookup.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem know_code_closures_program_refines :
  know_code_closures_program_refines_statement.
Proof.
  intros sets syms e Hincl Hwhole. unfold program_refines.
  apply know_code_closures_refines_aux.
  exact (code_inv_observed sets [] [] [] e Hincl Hwhole (Forall_nil _)).
Qed.

Theorem know_own_closures_program_refines :
  know_own_closures_program_refines_statement.
Proof.
  intros e. unfold know_own_closures.
  destruct (groups_whole e) eqn:Hwhole; [ | apply refines_refl ].
  exact (know_code_closures_program_refines (sets_made e) [] e
           (incl_refl _) Hwhole).
Qed.

Theorem know_own_closures_refines_imports :
  know_own_closures_refines_imports_statement.
Proof.
  intros imports e. unfold unit_refines, know_own_closures.
  destruct (groups_whole e) eqn:Hwhole; [ | apply refines_refl ].
  apply know_code_closures_refines_aux.
  exact (code_inv_observed (sets_made e) imports [] [] e (incl_refl _) Hwhole
           (Forall_nil _)).
Qed.

Theorem know_own_closures_refines_observed :
  know_own_closures_refines_observed_statement.
Proof.
  intros imports slots conts e Hobservers. unfold know_own_closures.
  destruct (groups_whole e) eqn:Hwhole; [ | apply refines_refl ].
  apply know_code_closures_refines_aux.
  exact (code_inv_observed (sets_made e) imports slots conts e (incl_refl _)
           Hwhole Hobservers).
Qed.

(* ======================================================================== *)
(* The names of the result                                                  *)
(* ======================================================================== *)

(* What is put in the place of a projection has no variable. *)
Lemma read_code_slot_free (sets : list set_made) (syms : list symbol)
    (known : closure_codes) (d : named) :
  incl (free_variables_of_named (read_code_slot sets syms known d))
    (free_variables_of_named d).
Proof.
  destruct d as [s | p args | code slots | slot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    cbn [read_code_slot]; try apply incl_refl.
  destruct closure as [x | sym | k]; try apply incl_refl.
  destruct (lookup x known) as [c | ]; [ | apply incl_refl ].
  destruct (slot_held sets c pfrom slot) as [a | ] eqn:Hheld;
    [ | apply incl_refl ].
  destruct (simple_statics_in_scope syms a); [ | apply incl_refl ].
  destruct (slot_held_test sets c pfrom slot a Hheld) as [Hclosed _].
  destruct a as [y | sym | k]; [ discriminate | | ]; apply incl_nil_l.
Qed.

(* A call is left, or is the same call made direct. *)
Lemma call_code_closure_shape (known : closure_codes) (ap : apply_expr) :
  call_code_closure known ap = ap
  ∨ ∃ c : code_id,
      call_code_closure known ap
      = Apply_expr (ap_callee ap) (Direct c) (ap_args ap)
          (ap_continuation ap) (ap_exn_continuation ap)
          (ap_exn_extra_args ap) (ap_args_arity ap) (ap_return_arity ap).
Proof.
  unfold call_code_closure.
  destruct (ap_call_kind ap);
    [ left; destruct (ap_callee ap); reflexivity | ].
  destruct (ap_callee ap) as [x | sym | k]; try (left; reflexivity).
  destruct (lookup x known) as [c | ]; [ | left; reflexivity ].
  right. exists c. reflexivity.
Qed.

Fixpoint know_in_body_scoped (sets : list set_made) (e : expr) {struct e} :
  ∀ (syms : list symbol) (known : closure_codes),
    scoped e (know_in_body sets syms known e)
with know_in_body_handler_scoped (sets : list set_made)
    (h : continuation_handler) {struct h} :
  ∀ (syms : list symbol) (known : closure_codes),
    scoped_handler h (know_in_body_handler sets syms known h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      intros syms known; cbn [know_in_body]; try apply scoped_refl.
    + apply scoped_let;
        [ apply read_code_slot_free | apply know_in_body_scoped ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl | apply know_in_body_scoped ].
    + apply scoped_let_cont;
        [ apply know_in_body_handler_scoped | apply know_in_body_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply know_in_body_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply know_in_body_handler_scoped | exact IH ].
    + intros _.
      destruct (call_code_closure_shape known ap) as [-> | (c & ->)];
        (split; [ reflexivity | split; apply incl_refl ]).
  - destruct h as [params metadata handler]. intros syms known.
    cbn [know_in_body_handler].
    apply scoped_handler_intro, know_in_body_scoped.
Qed.

Fixpoint know_code_closures_scoped (sets : list set_made) (e : expr)
    {struct e} :
  ∀ syms : list symbol, scoped e (know_code_closures sets syms e)
with know_code_in_handler_scoped (sets : list set_made)
    (h : continuation_handler) {struct h} :
  ∀ syms : list symbol,
    scoped_handler h (know_code_in_handler sets syms h).
Proof.
  - destruct e as [x kind d body | bindings body | k h body | handlers body
                  | ap | ac | scrutinee arms | ];
      intros syms; cbn [know_code_closures]; try apply scoped_refl.
    + apply scoped_let;
        [ apply incl_refl | apply know_code_closures_scoped ].
    + apply scoped_let_static; [ | apply know_code_closures_scoped ].
      generalize (flat_map symbols_of_binding bindings ++ syms).
      intros all.
      induction bindings
        as [ | [id [params my_closure ret exn metadata code_body] | id
               | sym const]
               bindings IH];
        cbn [map]; [ apply scoped_bindings_nil | | | ].
      * cbn [know_code_in_binding know_in_code].
        destruct (cm_is_stub metadata || among my_closure params);
          [ apply scoped_bindings_cons_same, IH | ].
        apply scoped_bindings_cons_code; [ | exact IH ].
        apply scoped_code_intro, know_in_body_scoped.
      * apply scoped_bindings_cons_same, IH.
      * apply scoped_bindings_cons_same, IH.
    + apply scoped_let_cont;
        [ apply know_code_in_handler_scoped
        | apply know_code_closures_scoped ].
    + apply scoped_let_cont_recursive;
        [ | | apply know_code_closures_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * generalize (static_names symbols_of_binding body ++ syms).
        intros all.
        induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply know_code_in_handler_scoped | exact IH ].
  - destruct h as [params metadata handler]. intros syms.
    cbn [know_code_in_handler].
    apply scoped_handler_intro, know_code_closures_scoped.
Qed.

Theorem know_own_closures_names : know_own_closures_names_statement.
Proof.
  intros e. unfold know_own_closures.
  destruct (groups_whole e);
    [ apply know_code_closures_scoped | apply scoped_refl ].
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* A set that a term makes, renamed: the variables in its value
   slots. *)
Definition rename_set_made (r : renaming) (set : set_made) : set_made :=
  (fst set, map (rename_slot r) (snd set)).

Lemma named_sets_made_renamed (r : renaming) (d : named) :
  named_sets_made (rename_named r d)
  = map (rename_set_made r) (named_sets_made d).
Proof. destruct d; reflexivity. Qed.

Lemma static_const_sets_made_renamed (r : renaming) (const : static_const) :
  static_const_sets_made (rename_static_const r const)
  = map (rename_set_made r) (static_const_sets_made const).
Proof. destruct const; reflexivity. Qed.

Fixpoint sets_made_renamed (r : renaming) (e : expr) {struct e} :
  sets_made (rename_expr r e) = map (rename_set_made r) (sets_made e)
with handler_sets_made_renamed (r : renaming) (h : continuation_handler)
    {struct h} :
  handler_sets_made (rename_handler r h)
  = map (rename_set_made r) (handler_sets_made h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rename_expr sets_made]; try reflexivity.
    + rewrite named_sets_made_renamed, (sets_made_renamed r body), map_app.
      reflexivity.
    + rewrite (sets_made_renamed r body), map_app. f_equal.
      induction bindings
        as [ | [id [params my_closure ret exn metadata code_body] | id
               | sym const]
               bindings IH];
        cbn [map flat_map]; [ reflexivity | | | ]; rewrite map_app, IH;
        f_equal.
      * cbn [rename_static_binding rename_code binding_sets_made
               code_sets_made].
        apply sets_made_renamed.
      * cbn [rename_static_binding binding_sets_made].
        apply static_const_sets_made_renamed.
    + rewrite (handler_sets_made_renamed r h), (sets_made_renamed r body),
        map_app.
      reflexivity.
    + rewrite (sets_made_renamed r body), map_app. f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
        [ reflexivity | ].
      rewrite map_app, (handler_sets_made_renamed r h), IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_sets_made]. apply sets_made_renamed.
Qed.

(* The tests answer the same of the sets renamed. *)
Lemma lookup_slots_renamed (r : renaming) (slot : value_slot)
    (slots : list (value_slot * simple)) :
  lookup slot (map (rename_slot r) slots)
  = option_map (rename_simple r) (lookup slot slots).
Proof.
  induction slots as [ | [slot' a] slots IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb slot slot'); [ reflexivity | exact IH ].
Qed.

Lemma set_with_code_renamed (r : renaming) (sets : list set_made)
    (c : code_id) (f : function_slot) :
  set_with_code (map (rename_set_made r) sets) c f
  = option_map (rename_set_made r) (set_with_code sets c f).
Proof.
  unfold set_with_code.
  induction sets as [ | set sets IH]; cbn [map find]; [ reflexivity | ].
  cbn [rename_set_made fst].
  destruct (has_code_at c f (fst set)); [ reflexivity | exact IH ].
Qed.

Lemma set_made_has_renamed (r : renaming) (fact : set_fact)
    (set : set_made) :
  fact_closed fact = true ->
  set_made_has fact (rename_set_made r set) = set_made_has fact set.
Proof.
  destruct fact as [slot code | slot held]; [ reflexivity | ].
  cbn [fact_closed set_made_has rename_set_made snd]. intros Hclosed.
  rewrite lookup_slots_renamed.
  destruct (lookup slot (snd set)) as [a | ]; cbn [option_map];
    [ | reflexivity ].
  transitivity (same_simple (rename_simple r a) (rename_simple r held));
    [ rewrite (rename_closed_simple r held Hclosed); reflexivity
    | apply same_simple_renamed ].
Qed.

Lemma sets_made_have_renamed (r : renaming) (c : code_id)
    (f : function_slot) (fact : set_fact) (sets : list set_made) :
  fact_closed fact = true ->
  sets_made_have c f fact (map (rename_set_made r) sets)
  = sets_made_have c f fact sets.
Proof.
  intros Hclosed. unfold sets_made_have.
  induction sets as [ | set sets IH]; cbn [map forallb]; [ reflexivity | ].
  rewrite (set_made_has_renamed r fact set Hclosed), IH. reflexivity.
Qed.

Lemma sibling_code_renamed (r : renaming) (sets : list set_made)
    (c : code_id) (f g : function_slot) :
  sibling_code (map (rename_set_made r) sets) c f g
  = sibling_code sets c f g.
Proof.
  unfold sibling_code. rewrite set_with_code_renamed.
  destruct (set_with_code sets c f) as [set | ]; cbn [option_map];
    [ | reflexivity ].
  cbn [rename_set_made fst].
  destruct (lookup g (fst set)) as [d | ]; [ | reflexivity ].
  rewrite (sets_made_have_renamed r c f (Code_at g d) sets eq_refl).
  reflexivity.
Qed.

Lemma slot_held_renamed (r : renaming) (sets : list set_made) (c : code_id)
    (f : function_slot) (slot : value_slot) :
  slot_held (map (rename_set_made r) sets) c f slot
  = slot_held sets c f slot.
Proof.
  unfold slot_held. rewrite set_with_code_renamed.
  destruct (set_with_code sets c f) as [set | ]; cbn [option_map];
    [ | reflexivity ].
  cbn [rename_set_made snd]. rewrite lookup_slots_renamed.
  destruct (lookup slot (snd set)) as [a | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite closed_simple_renamed.
  destruct (closed_simple a) eqn:Ha; cbn [andb]; [ | reflexivity ].
  rewrite (rename_closed_simple r a Ha),
    (sets_made_have_renamed r c f (Slot_holds slot a) sets Ha).
  reflexivity.
Qed.

(* What is known, renamed: the variables; code ids are not renamed. *)
Definition rename_codes (r : renaming) (known : closure_codes)
    : closure_codes :=
  map (fun entry => (rename_variable r (fst entry), snd entry)) known.

Lemma lookup_codes_renamed (r : renaming) (x : variable)
    (known : closure_codes) :
  lookup (rename_variable r x) (rename_codes r known) = lookup x known.
Proof.
  unfold rename_codes.
  induction known as [ | [y c] known IH]; simpl; [ reflexivity | ].
  rewrite rename_variable_eqb. destruct (Pos.eqb x y); [ reflexivity | ].
  exact IH.
Qed.

Lemma unbind_codes_renamed (r : renaming) (x : variable)
    (known : closure_codes) :
  unbind (rename_variable r x) (rename_codes r known)
  = rename_codes r (unbind x known).
Proof.
  unfold unbind, rename_codes.
  induction known as [ | [y c] known IH]; simpl; [ reflexivity | ].
  rewrite rename_variable_eqb, IH. destruct (Pos.eqb x y); reflexivity.
Qed.

Lemma forget_code_vars_renamed (r : renaming) (params : list variable) :
  ∀ known : closure_codes,
    forget_code_vars (map (rename_variable r) params) (rename_codes r known)
    = rename_codes r (forget_code_vars params known).
Proof.
  induction params as [ | param params IH]; intros known; simpl;
    [ reflexivity | ].
  rewrite unbind_codes_renamed. apply IH.
Qed.

Lemma closure_code_made_renamed (r : renaming) (sets : list set_made)
    (known : closure_codes) (d : named) :
  closure_code_made (map (rename_set_made r) sets) (rename_codes r known)
    (rename_named r d)
  = closure_code_made sets known d.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | vslot closure
    | pfrom vslot closure | move_from move_to closure];
    try reflexivity.
  destruct closure as [y | sym | c]; try reflexivity.
  cbn [rename_named rename_simple closure_code_made].
  rewrite lookup_codes_renamed.
  destruct (lookup y known) as [c | ]; [ | reflexivity ].
  apply sibling_code_renamed.
Qed.

Lemma codes_under_let_renamed (r : renaming) (sets : list set_made)
    (known : closure_codes) (x : variable) (d : named) :
  codes_under_let (map (rename_set_made r) sets) (rename_codes r known)
    (rename_variable r x) (rename_named r d)
  = rename_codes r (codes_under_let sets known x d).
Proof.
  unfold codes_under_let.
  rewrite closure_code_made_renamed, unbind_codes_renamed.
  destruct (closure_code_made sets known d); reflexivity.
Qed.

Lemma read_code_slot_renamed (r : renaming) (sets : list set_made)
    (syms : list symbol) (known : closure_codes) (d : named) :
  read_code_slot (map (rename_set_made r) sets) syms (rename_codes r known)
    (rename_named r d)
  = rename_named r (read_code_slot sets syms known d).
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | vslot closure
    | pfrom vslot closure | move_from move_to closure];
    try reflexivity.
  destruct closure as [y | sym | c]; try reflexivity.
  cbn [rename_named rename_simple read_code_slot].
  rewrite lookup_codes_renamed.
  destruct (lookup y known) as [c | ]; [ | reflexivity ].
  rewrite slot_held_renamed.
  destruct (slot_held sets c pfrom vslot) as [a | ] eqn:Hheld;
    [ | reflexivity ].
  destruct (simple_statics_in_scope syms a); [ | reflexivity ].
  destruct (slot_held_test sets c pfrom vslot a Hheld) as [Hclosed _].
  cbn [rename_named]. rewrite (rename_closed_simple r a Hclosed).
  reflexivity.
Qed.

Lemma call_code_closure_renamed (r : renaming) (known : closure_codes)
    (ap : apply_expr) :
  call_code_closure (rename_codes r known) (rename_apply r ap)
  = rename_apply r (call_code_closure known ap).
Proof.
  unfold call_code_closure.
  cbn [rename_apply ap_call_kind ap_callee].
  destruct (ap_call_kind ap); [ destruct (ap_callee ap); reflexivity | ].
  destruct (ap_callee ap) as [x | sym | c]; try reflexivity.
  cbn [rename_simple]. rewrite lookup_codes_renamed.
  destruct (lookup x known); reflexivity.
Qed.

Fixpoint know_in_body_renamed (r : renaming) (sets : list set_made)
    (e : expr) {struct e} :
  ∀ (syms : list symbol) (known : closure_codes),
    know_in_body (map (rename_set_made r) sets) syms (rename_codes r known)
      (rename_expr r e)
    = rename_expr r (know_in_body sets syms known e)
with know_in_body_handler_renamed (r : renaming) (sets : list set_made)
    (h : continuation_handler) {struct h} :
  ∀ (syms : list symbol) (known : closure_codes),
    know_in_body_handler (map (rename_set_made r) sets) syms
      (rename_codes r known) (rename_handler r h)
    = rename_handler r (know_in_body_handler sets syms known h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros syms known; try reflexivity;
      cbn [rename_expr know_in_body].
    + rewrite read_code_slot_renamed, codes_under_let_renamed,
        (know_in_body_renamed r sets body).
      reflexivity.
    + rewrite (know_in_body_renamed r sets body). reflexivity.
    + rewrite (know_in_body_handler_renamed r sets h),
        (know_in_body_renamed r sets body).
      reflexivity.
    + rewrite (know_in_body_renamed r sets body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map];
        [ reflexivity | ].
      rewrite (know_in_body_handler_renamed r sets h), IH. reflexivity.
    + rewrite call_code_closure_renamed. reflexivity.
  - destruct h as [params metadata handler]. intros syms known.
    cbn [rename_handler know_in_body_handler].
    rewrite forget_code_vars_renamed, (know_in_body_renamed r sets handler).
    reflexivity.
Qed.

Lemma know_code_in_binding_renamed (r : renaming) (sets : list set_made)
    (syms : list symbol) (binding : static_binding) :
  know_code_in_binding (map (rename_set_made r) sets) syms
    (rename_static_binding r binding)
  = rename_static_binding r (know_code_in_binding sets syms binding).
Proof.
  destruct binding
    as [id [params my_closure ret exn metadata body] | id | sym const];
    try reflexivity.
  cbn [rename_static_binding rename_code know_code_in_binding know_in_code].
  rewrite among_rename_variable.
  destruct (cm_is_stub metadata || among my_closure params);
    [ reflexivity | ].
  cbn [rename_static_binding rename_code].
  rewrite <- (know_in_body_renamed r sets body syms [(my_closure, id)]).
  reflexivity.
Qed.

Fixpoint know_code_closures_renamed (r : renaming) (sets : list set_made)
    (e : expr) {struct e} :
  ∀ syms : list symbol,
    know_code_closures (map (rename_set_made r) sets) syms (rename_expr r e)
    = rename_expr r (know_code_closures sets syms e)
with know_code_in_handler_renamed (r : renaming) (sets : list set_made)
    (h : continuation_handler) {struct h} :
  ∀ syms : list symbol,
    know_code_in_handler (map (rename_set_made r) sets) syms
      (rename_handler r h)
    = rename_handler r (know_code_in_handler sets syms h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros syms; cbn [rename_expr know_code_closures]; try reflexivity.
    + rewrite (know_code_closures_renamed r sets body). reflexivity.
    + assert (Hbindings :
        ∀ all : list symbol,
          map (know_code_in_binding (map (rename_set_made r) sets) all)
            (map (rename_static_binding r) bindings)
          = map (rename_static_binding r)
              (map (know_code_in_binding sets all) bindings)).
      { intros all.
        induction bindings as [ | binding bindings IH]; cbn [map];
          [ reflexivity | ].
        rewrite know_code_in_binding_renamed, IH. reflexivity. }
      rewrite
        (bindings_names_renamed r symbols_of_binding
           (symbols_of_binding_renamed r)),
        (know_code_closures_renamed r sets body), Hbindings.
      reflexivity.
    + rewrite
        (static_names_of_renamed r symbols_of_binding
           (symbols_of_binding_renamed r)),
        (know_code_in_handler_renamed r sets h),
        (know_code_closures_renamed r sets body).
      reflexivity.
    + assert (Hhandlers :
        ∀ all : list symbol,
          map
            (fun '(k, h) =>
               (k,
                know_code_in_handler (map (rename_set_made r) sets) all h))
            (map
               (fun '(k, h) => (rename_continuation r k, rename_handler r h))
               handlers)
          = map
              (fun '(k, h) => (rename_continuation r k, rename_handler r h))
              (map (fun '(k, h) => (k, know_code_in_handler sets all h))
                 handlers)).
      { intros all.
        induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ reflexivity | ].
        rewrite (know_code_in_handler_renamed r sets h), IH. reflexivity. }
      rewrite
        (static_names_of_renamed r symbols_of_binding
           (symbols_of_binding_renamed r)),
        (know_code_closures_renamed r sets body), Hhandlers.
      reflexivity.
  - destruct h as [params metadata handler]. intros syms.
    cbn [rename_handler know_code_in_handler].
    rewrite (know_code_closures_renamed r sets handler). reflexivity.
Qed.

(* The test of the groups is the same of a term renamed: a renaming is
   injective, so the test of equality of static constants answers the
   same. (FlambdaValidatorEquivarianceProofs.v has the like of
   FlambdaWellFormedSpec.static_set_whole; that file imports this
   one.) *)
Lemma whole_rename_simple_injective (r : renaming) (a a' : simple) :
  rename_simple r a = rename_simple r a' -> a = a'.
Proof.
  destruct a as [x | sym | k], a' as [x' | sym' | k']; simpl; intros Heq;
    try discriminate; try exact Heq.
  injection Heq as Heq. f_equal.
  exact (permute_injective (variable_permutation r) x x' Heq).
Qed.

Lemma whole_rename_simples_injective (r : renaming)
    (args args' : list simple) :
  map (rename_simple r) args = map (rename_simple r) args' -> args = args'.
Proof.
  revert args'.
  induction args as [ | s args IH]; intros [ | s' args']; simpl; intros Heq;
    try discriminate; [ reflexivity | ].
  injection Heq as Hs Hargs.
  apply whole_rename_simple_injective in Hs. apply IH in Hargs. congruence.
Qed.

Lemma whole_rename_value_slots_injective (r : renaming)
    (slots slots' : list (value_slot * simple)) :
  map (fun slot => (fst slot, rename_simple r (snd slot))) slots
  = map (fun slot => (fst slot, rename_simple r (snd slot))) slots' ->
  slots = slots'.
Proof.
  revert slots'.
  induction slots as [ | [slot s] slots IH]; intros [ | [slot' s'] slots'];
    simpl; intros Heq; try discriminate; [ reflexivity | ].
  injection Heq as Hslot Hs Hslots.
  apply whole_rename_simple_injective in Hs. apply IH in Hslots. congruence.
Qed.

Lemma whole_rename_static_const_injective (r : renaming)
    (const const' : static_const) :
  rename_static_const r const = rename_static_const r const' ->
  const = const'.
Proof.
  destruct const, const'; simpl; intros Heq; try discriminate;
    injection Heq; intros; subst; try reflexivity; f_equal;
    eauto using whole_rename_value_slots_injective,
      whole_rename_simples_injective, whole_rename_simple_injective.
Qed.

Lemma whole_binds_symbol_to_renamed (r : renaming) (s : symbol)
    (const : static_const) (binding : static_binding) :
  binds_symbol_to s (rename_static_const r const)
    (rename_static_binding r binding)
  = binds_symbol_to s const binding.
Proof.
  destruct binding as [id c | id | s' const']; try reflexivity.
  cbn [rename_static_binding binds_symbol_to]. f_equal.
  destruct (static_const_eq_dec const const') as [-> | Hne],
    (static_const_eq_dec (rename_static_const r _)
       (rename_static_const r _))
      as [Heq | Hne'];
    try reflexivity; [ contradiction | ].
  elim Hne. exact (whole_rename_static_const_injective r _ _ Heq).
Qed.

Lemma whole_static_set_whole_renamed (r : renaming)
    (bindings : list static_binding) (binding : static_binding) :
  static_set_whole (map (rename_static_binding r) bindings)
    (rename_static_binding r binding)
  = static_set_whole bindings binding.
Proof.
  destruct binding as [id c | id | s const]; try reflexivity.
  destruct const
    as [ | slot closure_symbols function_decls value_slots | | | | ];
    try reflexivity.
  cbn [rename_static_binding rename_static_const static_set_whole]. f_equal.
  apply forallb_same. intros [slot' s'].
  induction bindings as [ | binding' bindings IH]; cbn [map existsb];
    [ reflexivity | ].
  rewrite IH. f_equal.
  exact
    (whole_binds_symbol_to_renamed r s'
       (Static_const_closure_of_set slot' closure_symbols function_decls
          value_slots)
       binding').
Qed.

Fixpoint groups_whole_renamed (r : renaming) (e : expr) {struct e} :
  groups_whole (rename_expr r e) = groups_whole e
with handler_groups_whole_renamed (r : renaming) (h : continuation_handler)
    {struct h} :
  handler_groups_whole (rename_handler r h) = handler_groups_whole h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rename_expr groups_whole]; try reflexivity.
    + apply groups_whole_renamed.
    + rewrite (groups_whole_renamed r body). f_equal. f_equal.
      * generalize bindings at 1 3. intros all.
        induction bindings as [ | binding bindings IH]; cbn [map forallb];
          [ reflexivity | ].
        rewrite whole_static_set_whole_renamed, IH. reflexivity.
      * induction bindings
          as [ | [id [params my_closure ret exn metadata code_body] | id
                 | sym const]
                 bindings IH];
          cbn [map forallb rename_static_binding rename_code
                 binding_groups_whole code_groups_whole];
          [ reflexivity | | | ]; rewrite IH; [ | reflexivity | reflexivity ].
        rewrite (groups_whole_renamed r code_body). reflexivity.
    + rewrite (handler_groups_whole_renamed r h),
        (groups_whole_renamed r body).
      reflexivity.
    + rewrite (groups_whole_renamed r body). f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (handler_groups_whole_renamed r h), IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_groups_whole]. apply groups_whole_renamed.
Qed.

Theorem know_own_closures_equivariant :
  know_own_closures_equivariant_statement.
Proof.
  intros r e. unfold know_own_closures.
  rewrite groups_whole_renamed.
  destruct (groups_whole e); [ | reflexivity ].
  rewrite sets_made_renamed. apply know_code_closures_renamed.
Qed.
