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

(* Proofs of the statements of FlambdaContextSpec.v. Nothing here needs
   review.

   A hole in the body of a binder is one step away from the premise. A hole
   in a handler is not: the handler goes into a closure, which is applied
   later, any number of times. So the proof relates continuation
   environments whose closures have related handlers ([cont_env_related]),
   and shows that one expression under two related environments ends alike
   ([transfer]), by induction on the length of the run: where a closure is
   applied, the induction gives the first handler under the second
   environment, and the relation of the handlers then gives the second
   handler there. Divergence needs no argument of its own
   (FlambdaSemanticsProofs.equivalent_of_terminating).

   A hole in code is the same again for the static environment: the proof
   relates static environments whose code has related bodies
   ([statics_related]), and where the code is called the induction gives
   the first body under the second environments. Values hold code ids and
   not code, so the values of the two runs stay equal.

   The relations may rely on a predicate on states that a step keeps
   ([inv]): corresponding handlers, and corresponding code, need only be
   related from the states that have it, where they have it together.
   Corresponding code need only be related, besides, where the callee is
   a closure of the code id that the code is bound to, since a call
   finds its code by the code id of its callee. The lemmas named [_from]
   are that. The lemmas named [_where] are the case of a property of the
   static environment that, once it holds, holds of every extension:
   what a name is bound to never changes (FlambdaWellFormedSpec.v), so a
   handler may be rewritten using the bindings made before its
   definition. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaContextSpec.
Import ListNotations.

(* ======================================================================== *)
(* Related continuation environments                                        *)
(* ======================================================================== *)

Section Transfer.

(* The relation between corresponding handlers, given their parameters,
   the kinds of those, and the variables of the closure they are in: a
   handler runs under those variables with its parameters bound to values
   of those kinds, so the relation may depend on them. *)
Variable related
  : list variable -> list flambda_kind -> env -> expr -> expr -> Prop.

Inductive handler_related (vars : env)
    : continuation_handler -> continuation_handler -> Prop :=
  | Handler_related (params : list variable) (metadata : handler_metadata)
      (e e' : expr) :
    related params (hm_params_kinds metadata) vars e e' ->
    handler_related vars (Handler params metadata e)
      (Handler params metadata e')
  | Handler_same (h : continuation_handler) : handler_related vars h h.

Definition handlers_related (vars : env)
    (handlers handlers' : list (continuation * continuation_handler))
    : Prop :=
  Forall2
    (fun kh kh' =>
       fst kh = fst kh' ∧ handler_related vars (snd kh) (snd kh'))
    handlers handlers'.

(* Closures with related handlers and, in turn, related environments;
   and what is left of the same over-application, with related
   continuations. (The warning silenced is about induction schemes,
   which nothing uses.) *)
#[local] Set Warnings "-register-all".
Inductive closure_related : cont_value -> cont_value -> Prop :=
  | Closure_related (h h' : continuation_handler) (vars : env)
      (conts conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    handler_related vars h h' ->
    Forall2
      (fun kc kc' => fst kc = fst kc' ∧ closure_related (snd kc) (snd kc'))
      conts conts' ->
    handlers_related vars recursive recursive' ->
    closure_related (Closure h vars conts recursive)
      (Closure h' vars conts' recursive')
  | Extra_args_related (c c' : cont_value) (extra : list value) :
    closure_related c c' ->
    closure_related (Extra_args c extra) (Extra_args c' extra)
  | Closure_same (c : cont_value) : closure_related c c
  | Over_related (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    closure_related return_to return_to' ->
    closure_related exn_to exn_to' ->
    closure_related
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

Definition cont_env_related (conts conts' : cont_env) : Prop :=
  Forall2
    (fun kc kc' => fst kc = fst kc' ∧ closure_related (snd kc) (snd kc'))
    conts conts'.

Lemma cont_env_related_refl (conts : cont_env) : cont_env_related conts conts.
Proof.
  induction conts as [ | kc conts IH]; [ apply Forall2_nil | ].
  apply Forall2_cons; [ | exact IH ].
  split; [ reflexivity | apply Closure_same ].
Qed.

Lemma handlers_related_refl (vars : env)
    (handlers : list (continuation * continuation_handler)) :
  handlers_related vars handlers handlers.
Proof.
  induction handlers as [ | kh handlers IH]; [ apply Forall2_nil | ].
  apply Forall2_cons; [ | exact IH ].
  split; [ reflexivity | apply Handler_same ].
Qed.

Lemma closure_related_inv (c c' : cont_value) :
  closure_related c c' ->
  match c, c' with
  | Closure h vars conts recursive, Closure h' vars' conts' recursive' =>
    vars = vars' ∧ handler_related vars h h' ∧ cont_env_related conts conts'
    ∧ handlers_related vars recursive recursive'
  | Outer k, Outer k' => k = k'
  | Extra_args c extra, Extra_args c' extra' =>
    extra = extra' ∧ closure_related c c'
  | Over_application remaining remaining_arity return_arity return_to exn_to,
    Over_application remaining' remaining_arity' return_arity' return_to'
      exn_to' =>
    remaining = remaining' ∧ remaining_arity = remaining_arity'
    ∧ return_arity = return_arity' ∧ closure_related return_to return_to'
    ∧ closure_related exn_to exn_to'
  | _, _ => False
  end.
Proof.
  intros Hrel. inversion Hrel; subst.
  - auto.
  - auto.
  - destruct c' as [h vars conts recursive | k | c extra | ];
      [ | reflexivity | split; [ reflexivity | apply Closure_same ] | ].
    + split; [ reflexivity | ]. split; [ apply Handler_same | ].
      split; [ apply cont_env_related_refl | apply handlers_related_refl ].
    + repeat split; apply Closure_same.
  - auto 6.
Qed.

Lemma with_extra_args_related (c c' : cont_value) (extra : list value) :
  closure_related c c' ->
  closure_related (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hrel. destruct extra; simpl;
    [ exact Hrel | apply Extra_args_related, Hrel ].
Qed.

Lemma handler_related_inv (vars : env) (params params' : list variable)
    (metadata metadata' : handler_metadata) (e e' : expr) :
  handler_related vars (Handler params metadata e)
    (Handler params' metadata' e') ->
  params = params' ∧ metadata = metadata'
  ∧ (related params (hm_params_kinds metadata) vars e e' ∨ e = e').
Proof. intros Hrel. inversion Hrel; subst; auto. Qed.

(* A continuation stands for related things under related environments. *)
Lemma resolve_related (k : continuation) (conts conts' : cont_env) :
  cont_env_related conts conts' ->
  closure_related (resolve k conts) (resolve k conts').
Proof.
  unfold resolve.
  induction 1 as [ | [k1 c] [k1' c'] conts conts' [Hk Hc] _ IH]; simpl;
    [ apply Closure_same | ].
  simpl in Hk. subst k1'. destruct (Pos.eqb k k1); [ exact Hc | exact IH ].
Qed.

(* [all] and [all'] are the whole lists of handlers, kept apart from the
   ones still to be gone through. *)
Lemma rec_closures_related_aux
    (handlers handlers' all all' : list (continuation * continuation_handler))
    (vars : env) (conts conts' : cont_env) :
  handlers_related vars handlers handlers' ->
  handlers_related vars all all' ->
  cont_env_related conts conts' ->
  cont_env_related
    (map (fun '(k, h) => (k, Closure h vars conts all)) handlers)
    (map (fun '(k, h) => (k, Closure h vars conts' all')) handlers').
Proof.
  intros Hhandlers Hall Hconts.
  induction Hhandlers as [ | [k h] [k' h'] handlers handlers' [Hk Hh] _ IH];
    simpl; [ apply Forall2_nil | ].
  apply Forall2_cons; [ | exact IH ].
  simpl in Hk, Hh. subst k'. split; [ reflexivity | ].
  apply Closure_related; assumption.
Qed.

Lemma rec_closures_related
    (handlers handlers' : list (continuation * continuation_handler))
    (vars : env) (conts conts' : cont_env) :
  handlers_related vars handlers handlers' ->
  cont_env_related conts conts' ->
  cont_env_related (rec_closures handlers vars conts)
    (rec_closures handlers' vars conts').
Proof.
  intros Hhandlers Hconts.
  apply rec_closures_related_aux; assumption.
Qed.

(* ======================================================================== *)
(* One expression under related environments                                *)
(* ======================================================================== *)

(* ======================================================================== *)
(* Related static environments                                              *)
(* ======================================================================== *)

(* The relation between the bodies of corresponding code, given the code
   id that the code is bound to, what the code binds and its metadata.
   Code runs at a call: under its parameters and [my_closure] bound and
   its two continuations, and under a static environment that the call
   decides. So the relation may depend on the first, and is not relative
   to the last. A call finds the code by the code id of its callee, a
   closure: so the relation may depend on the code id, and where it is
   used the callee is a closure of that code id. *)
Variable code_related :
  code_id -> list variable -> variable -> continuation -> continuation ->
  code_metadata -> expr -> expr -> Prop.

Inductive codes_related_where (id : code_id) : code -> code -> Prop :=
  | Codes_related_where (params : list variable) (my_closure : variable)
      (ret exn : continuation) (metadata : code_metadata)
      (body body' : expr) :
    binds_statics body = binds_statics body' ->
    code_related id params my_closure ret exn metadata body body' ->
    codes_related_where id (Code params my_closure ret exn metadata body)
      (Code params my_closure ret exn metadata body')
  | Codes_same_where (c : code) : codes_related_where id c c.

Inductive binding_related_where : static_binding -> static_binding -> Prop :=
  | Binding_code_where (id : code_id) (c c' : code) :
    codes_related_where id c c' ->
    binding_related_where (Bind_code id c) (Bind_code id c')
  | Binding_same_where (binding : static_binding) :
    binding_related_where binding binding.

(* Static environments with the same statically allocated values and, for
   each code id, code related at that code id. Values hold code ids and
   not code, so they need no relation. *)
Definition statics_related_where (statics statics' : static_env) : Prop :=
  Forall2
    (fun ic ic' =>
       fst ic = fst ic'
       ∧ codes_related_where (fst ic) (snd ic) (snd ic'))
    (static_code statics) (static_code statics')
  ∧ static_values statics = static_values statics'.

Lemma statics_related_where_refl (statics : static_env) :
  statics_related_where statics statics.
Proof.
  split; [ | reflexivity ].
  induction (static_code statics) as [ | ic code IH];
    [ apply Forall2_nil | ].
  apply Forall2_cons; [ | exact IH ].
  split; [ reflexivity | apply Codes_same_where ].
Qed.

Lemma bindings_related_where_refl (bindings : list static_binding) :
  Forall2 binding_related_where bindings bindings.
Proof.
  induction bindings as [ | binding bindings IH]; [ apply Forall2_nil | ].
  apply Forall2_cons; [ apply Binding_same_where | exact IH ].
Qed.

Lemma lookup_code_related (id : code_id) (statics statics' : static_env) :
  statics_related_where statics statics' ->
  match lookup id (static_code statics), lookup id (static_code statics') with
  | Some c, Some c' => codes_related_where id c c'
  | None, None => True
  | _, _ => False
  end.
Proof.
  intros [Hcode _].
  induction Hcode as [ | [id1 c] [id1' c'] code code' [Hid Hc] _ IH]; simpl;
    [ exact I | ].
  simpl in Hid, Hc. subst id1'.
  destruct (Pos.eqb id id1) eqn:Hsame; [ | exact IH ].
  apply Pos.eqb_eq in Hsame. subst id1. exact Hc.
Qed.

Lemma closure_of_related (statics statics' : static_env) (v : value) :
  statics_related_where statics statics' ->
  closure_of statics v = closure_of statics' v.
Proof.
  intros [_ Hvalues]. unfold closure_of. rewrite Hvalues. reflexivity.
Qed.

Lemma project_value_slot_related (statics statics' : static_env)
    (project_from : function_slot) (slot : value_slot) (v : value) :
  statics_related_where statics statics' ->
  project_value_slot statics project_from slot v
  = project_value_slot statics' project_from slot v.
Proof.
  intros [_ Hvalues]. unfold project_value_slot, closure_in_set.
  rewrite Hvalues. reflexivity.
Qed.

Lemma project_function_slot_related (statics statics' : static_env)
    (move_from move_to : function_slot) (v : value) :
  statics_related_where statics statics' ->
  project_function_slot statics move_from move_to v
  = project_function_slot statics' move_from move_to v.
Proof.
  intros [_ Hvalues]. unfold project_function_slot. rewrite Hvalues.
  reflexivity.
Qed.

Lemma eval_named_related (statics statics' : static_env) (vars : env)
    (d : named) :
  statics_related_where statics statics' ->
  eval_named statics vars d = eval_named statics' vars d.
Proof.
  intros Hstatics.
  destruct d as
    [s | p args | code slots | slot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    simpl; try reflexivity.
  - destruct (eval_simples vars args) as [vs | ]; simpl; [ | reflexivity ].
    apply eval_prim_same_static_values, Hstatics.
  - destruct (eval_simple vars closure) as [v | ]; simpl; [ | reflexivity ].
    rewrite (closure_of_related statics statics' v Hstatics). reflexivity.
  - destruct (eval_simple vars closure) as [v | ]; simpl; [ | reflexivity ].
    apply project_value_slot_related, Hstatics.
  - destruct (eval_simple vars closure) as [v | ]; simpl; [ | reflexivity ].
    apply project_function_slot_related, Hstatics.
Qed.

(* Related groups of bindings added to related static environments: both
   fail, or the results are related. *)
Lemma eval_static_bindings_related_where (vars : env)
    (bindings bindings' : list static_binding) :
  Forall2 binding_related_where bindings bindings' ->
  ∀ statics statics' : static_env,
    statics_related_where statics statics' ->
    match
      eval_static_bindings vars bindings statics,
      eval_static_bindings vars bindings' statics'
    with
    | Some extended, Some extended' => statics_related_where extended extended'
    | None, None => True
    | _, _ => False
    end.
Proof.
  induction 1 as [ | binding binding' bindings bindings' Hbinding _ IH];
    intros statics statics' Hstatics; simpl; [ exact Hstatics | ].
  pose proof Hstatics as [Hcode Hvalues].
  destruct Hbinding as [id c c' Hc | [id c | id | sym const]]; simpl.
  - pose proof (lookup_code_related id statics statics' Hstatics) as Hid.
    destruct (lookup id (static_code statics)),
      (lookup id (static_code statics')); try contradiction; [ exact I | ].
    apply IH. split; [ | exact Hvalues ].
    apply Forall2_cons; [ split; [ reflexivity | exact Hc ] | exact Hcode ].
  - pose proof (lookup_code_related id statics statics' Hstatics) as Hid.
    destruct (lookup id (static_code statics)),
      (lookup id (static_code statics')); try contradiction; [ exact I | ].
    apply IH. split; [ | exact Hvalues ].
    apply Forall2_cons; [ | exact Hcode ].
    split; [ reflexivity | apply Codes_same_where ].
  - apply IH. split; assumption.
  - rewrite <- Hvalues.
    destruct (lookup sym (static_values statics)); [ exact I | ].
    destruct (eval_static_const vars const) as [allocated | ]; simpl;
      [ | exact I ].
    apply IH. split; [ exact Hcode | ]. simpl. rewrite Hvalues. reflexivity.
Qed.

(* What a call tests of the kinds of its arguments gives the kinds of the
   arity of the code. *)
Lemma unarize_per_parameter_unarize (a a' : arity) :
  unarize_per_parameter a = unarize_per_parameter a' ->
  unarize a = unarize a'.
Proof.
  unfold unarize, unarize_per_parameter. intros Heq.
  rewrite !flat_map_concat_map, Heq. reflexivity.
Qed.

(* A closure is a value of kind [Value]. *)
Lemma closure_of_kind (statics : static_env) (v : value) (called : code_id)
    (slots : list (value_slot * value)) :
  closure_of statics v = Some (called, slots) -> kind_of_value v = Value.
Proof. destruct v; simpl; try discriminate; intros _; reflexivity. Qed.

(* Related static environments bind a code id alike. The relation
   between the two codes is not said here, as it depends on the code id:
   [called_code_related] below gives it. *)
Definition any_codes (c c' : code) : Prop := True.

Lemma codes_agree_related (statics statics' : static_env) (called : code_id) :
  statics_related_where statics statics' ->
  codes_agree any_codes statics statics' called.
Proof.
  intros Hstatics. unfold codes_agree.
  pose proof (lookup_code_related called statics statics' Hstatics) as Hc.
  destruct (lookup called (static_code statics)) as [c | ],
    (lookup called (static_code statics')) as [c' | ];
    try contradiction; [ | exact I ].
  split; [ exact I | ]. destruct Hc; repeat split.
Qed.

(* A closure is one of the same code id under related static
   environments, and what the two bind that code id to is related at
   it. *)
Lemma called_code_related (statics statics' : static_env) (closure : value)
    (called called' : code_id) (slots slots' : list (value_slot * value))
    (c c' : code) :
  statics_related_where statics statics' ->
  closure_of statics closure = Some (called, slots) ->
  lookup called (static_code statics) = Some c ->
  closure_of statics' closure = Some (called', slots') ->
  lookup called' (static_code statics') = Some c' ->
  called' = called ∧ slots' = slots ∧ codes_related_where called c c'.
Proof.
  intros Hstatics Hclosure Hcode Hclosure' Hcode'.
  rewrite (closure_of_related statics statics' closure Hstatics), Hclosure'
    in Hclosure.
  injection Hclosure as -> ->.
  split; [ reflexivity | ]. split; [ reflexivity | ].
  pose proof (lookup_code_related called statics statics' Hstatics) as Hc.
  rewrite Hcode, Hcode' in Hc. exact Hc.
Qed.

(* ======================================================================== *)
(* One expression under related environments                                *)
(* ======================================================================== *)

(* What is to be carried from a run under the first environments to the
   state under the second: a property of a state and a behaviour that holds
   of a state when it holds of the state it steps to, that holds of how a
   state ends at once, and that [related] carries from a handler to the
   corresponding one, under the variables of their closures with the
   parameters bound, and [code_related] from a body to the corresponding
   one, where a call runs it: the callee is then a closure, under the
   static environment of the state, of the code id that the two bodies
   are related at.

   The relations need only carry it from the states that have [inv], a
   predicate on states that a step keeps. The state under the second
   environments has [inv] all along. Where a closure is applied or code is
   called, that state steps to the second handler or body, and the
   induction is about the first under the same environments, which no run
   need reach. So the relations are also asked to give [inv] of the first
   there from [inv] of the second. *)
Variable inv : state -> Prop.

Hypothesis inv_step :
  ∀ s s' : state, inv s -> step s = Step s' -> inv s'.

Variable carried : state -> behaviour -> Prop.

Hypothesis carried_step :
  ∀ (s s' : state) (b : behaviour),
    step s = Step s' -> carried s' b -> carried s b.

Hypothesis carried_exit :
  ∀ (s : state) (k : continuation) (vs : list value) (hp : heap),
    step s = Exit k vs hp -> carried s (Exits k vs hp).

Hypothesis carried_undefined :
  ∀ s : state, step s = Undefined -> carried s Goes_wrong.

Hypothesis inv_related :
  ∀ (params : list variable) (kinds : list flambda_kind) (def_vars : env)
    (e e' : expr) (vs : list value) (handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap),
    related params kinds def_vars e e' ->
    map kind_of_value vs = kinds ->
    bind_params params vs def_vars = Some handler_vars ->
    inv (State e' handler_vars conts statics hp) ->
    inv (State e handler_vars conts statics hp).

Hypothesis carried_related :
  ∀ (params : list variable) (kinds : list flambda_kind) (def_vars : env)
    (e e' : expr) (vs : list value) (handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (b : behaviour),
    related params kinds def_vars e e' ->
    inv (State e handler_vars conts statics hp) ->
    map kind_of_value vs = kinds ->
    bind_params params vs def_vars = Some handler_vars ->
    carried (State e handler_vars conts statics hp) b ->
    carried (State e' handler_vars conts statics hp) b.

Hypothesis inv_code_related :
  ∀ (id : code_id) (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata) (e e' : expr)
    (vs : list value) (callee : value) (callee_vars : env)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (slots : list (value_slot * value)),
    code_related id params my_closure ret exn metadata e e' ->
    bind_params params vs [(my_closure, callee)] = Some callee_vars ->
    map kind_of_value vs = unarize (cm_params_arity metadata) ->
    kind_of_value callee = Value ->
    closure_of statics callee = Some (id, slots) ->
    inv (State e' callee_vars [(ret, return_to); (exn, exn_to)] statics hp) ->
    inv (State e callee_vars [(ret, return_to); (exn, exn_to)] statics hp).

Hypothesis carried_code_related :
  ∀ (id : code_id) (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata) (e e' : expr)
    (vs : list value) (callee : value) (callee_vars : env)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (slots : list (value_slot * value)) (b : behaviour),
    code_related id params my_closure ret exn metadata e e' ->
    inv (State e callee_vars [(ret, return_to); (exn, exn_to)] statics hp) ->
    bind_params params vs [(my_closure, callee)] = Some callee_vars ->
    map kind_of_value vs = unarize (cm_params_arity metadata) ->
    kind_of_value callee = Value ->
    closure_of statics callee = Some (id, slots) ->
    carried (State e callee_vars [(ret, return_to); (exn, exn_to)] statics hp)
      b ->
    carried (State e' callee_vars [(ret, return_to); (exn, exn_to)] statics hp)
      b.

(* What is carried to a state, given what its step is. *)
Definition result_carried (result' : step_result) (b : behaviour) : Prop :=
  match result' with
  | Step s' => carried s' b
  | Exit k vs hp => b = Exits k vs hp
  | Undefined => b = Goes_wrong
  end.

Lemma carried_of_result (s : state) (b : behaviour) :
  result_carried (step s) b -> carried s b.
Proof.
  destruct (step s) as [s' | k vs | ] eqn:Hstep; simpl.
  - exact (carried_step s s' b Hstep).
  - intros ->. exact (carried_exit s k vs hp Hstep).
  - intros ->. exact (carried_undefined s Hstep).
Qed.

(* A run of [n] steps from where a step leads. *)
Definition run_on (n : nat) (result : step_result) : step_result :=
  match result with
  | Step s' => run n s'
  | Exit k vs hp => Exit k vs hp
  | Undefined => Undefined
  end.

(* What [transfer] below says of runs of [n] steps. *)
Definition transfer_at (n : nat) : Prop :=
  ∀ (e : expr) (vars : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp : heap) (b : behaviour),
    cont_env_related conts conts' ->
    statics_related_where statics statics' ->
    inv (State e vars conts' statics' hp) ->
    ended (run n (State e vars conts statics hp)) b ->
    carried (State e vars conts' statics' hp) b.

(* [inv] of the state that a step leads to, where it leads to one. *)
Definition result_inv (result : step_result) : Prop :=
  match result with
  | Step s' => inv s'
  | Exit _ _ _ | Undefined => True
  end.

(* The same full call of code related at a code id, of a closure of
   that code id under the second static environment, returning to
   related continuations. Where the call under the second environments
   runs a body, that state has [inv]. *)
Lemma call_code_transfer (n : nat) (closure : value) (called : code_id)
    (slots : list (value_slot * value)) (c c' : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) (b : behaviour) :
  transfer_at n ->
  codes_related_where called c c' ->
  closure_of statics' closure = Some (called, slots) ->
  closure_related return_to return_to' ->
  closure_related exn_to exn_to' ->
  statics_related_where statics statics' ->
  result_inv
    (call_code closure c' vs args_arity return_arity return_to' exn_to'
       statics' hp) ->
  ended
    (run_on n
       (call_code closure c vs args_arity return_arity return_to exn_to
          statics hp))
    b ->
  result_carried
    (call_code closure c' vs args_arity return_arity return_to' exn_to'
       statics' hp)
    b.
Proof.
  intros IH Hc Hclosure Hreturn Hexn Hstatics Hinv Hended.
  pose proof (closure_of_kind _ _ _ _ Hclosure) as Hvalue.
  (* Related code, or the same code: in both cases the two differ in
     their bodies at most. *)
  assert (Hcodes :
    ∃ (params : list variable) (my_closure : variable)
      (ret exn : continuation) (metadata : code_metadata)
      (body body' : expr),
      c = Code params my_closure ret exn metadata body
      ∧ c' = Code params my_closure ret exn metadata body'
      ∧ binds_statics body = binds_statics body'
      ∧ (code_related called params my_closure ret exn metadata body body'
         ∨ body = body')).
  { destruct Hc
      as [params my_closure ret exn metadata body body' Hbinds Hbody
         | [params my_closure ret exn metadata body]].
    - exists params, my_closure, ret, exn, metadata, body, body'. auto.
    - exists params, my_closure, ret, exn, metadata, body, body. auto. }
  destruct Hcodes
    as (params & my_closure & ret & exn & metadata & body & body' & -> & ->
        & Hbinds & Hbody).
  destruct (call_code_alike closure
              (Code params my_closure ret exn metadata body)
              (Code params my_closure ret exn metadata body') vs args_arity
              return_arity statics statics' hp eq_refl eq_refl eq_refl eq_refl
              eq_refl eq_refl (eq_sym Hbinds))
    as [Hundefined | [callee_vars Hcall]].
  - destruct (Hundefined return_to exn_to return_to' exn_to')
      as [Hsource Htarget].
    rewrite Hsource in Hended. rewrite Htarget.
    exact (ended_undefined b Hended).
  - destruct (Hcall return_to exn_to return_to' exn_to')
      as [Hsource Htarget].
    (* What the call tested, and what it bound. *)
    destruct (call_code_step _ _ _ _ _ _ _ _ _ _ Hsource)
      as (callee_vars' & _ & Hkinds & Hper & _ & Hbind & Hstate).
    assert (Hsame : callee_vars' = callee_vars) by congruence.
    subst callee_vars'. clear Hstate. simpl in Hper, Hbind.
    assert (Hparams : map kind_of_value vs = unarize (cm_params_arity metadata))
      by (rewrite Hkinds; apply unarize_per_parameter_unarize, Hper).
    rewrite Hsource in Hended. rewrite Htarget in Hinv |- *.
    simpl in Hended, Hinv |- *.
    assert (Hconts :
      cont_env_related [(ret, return_to); (exn, exn_to)]
        [(ret, return_to'); (exn, exn_to')]).
    { apply Forall2_cons; [ split; [ reflexivity | exact Hreturn ] | ].
      apply Forall2_cons; [ split; [ reflexivity | exact Hexn ] | ].
      apply Forall2_nil. }
    destruct Hbody as [Hbody | <-];
      [ | exact (IH _ _ _ _ _ _ _ _ Hconts Hstatics Hinv Hended) ].
    (* The state of the second body has [inv]; so has that of the
       first, which the induction is about. *)
    pose proof
      (inv_code_related _ _ _ _ _ _ _ _ _ _ _ _ _ _ hp _ Hbody Hbind Hparams
         Hvalue Hclosure Hinv)
      as Hfirst.
    exact
      (carried_code_related _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Hbody Hfirst
         Hbind Hparams Hvalue Hclosure
         (IH _ _ _ _ _ _ _ _ Hconts Hstatics Hfirst Hended)).
Qed.

(* Calls of the same shape, given that what is done with a partial
   application is carried. Each of the two, where it calls something,
   calls the code that its static environment binds to the code id of
   the closure (FlambdaSemanticsProofs.shape_calls): so the two codes
   are related at that code id. *)
Lemma run_call_transfer (n : nat) (kind : call_kind)
    (return_partial return_partial' : value -> step_result)
    (shape shape' : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) (b : behaviour) :
  transfer_at n ->
  shapes_related any_codes shape shape' ->
  shape_calls statics kind shape ->
  shape_calls statics' kind shape' ->
  closure_related return_to return_to' ->
  closure_related exn_to exn_to' ->
  statics_related_where statics statics' ->
  result_inv
    (run_call return_partial' shape' return_arity return_to' exn_to'
       statics' hp) ->
  (∀ partial : value,
    result_inv (return_partial' partial) ->
    ended (run_on n (return_partial partial)) b ->
    result_carried (return_partial' partial) b) ->
  ended
    (run_on n
       (run_call return_partial shape return_arity return_to exn_to statics hp))
    b ->
  result_carried
    (run_call return_partial' shape' return_arity return_to' exn_to' statics'
       hp)
    b.
Proof.
  intros IH Hshapes Hcalls Hcalls' Hreturn Hexn Hstatics Hinv Hpartial.
  destruct Hshapes
    as [ | closure c c' vs args_arity _
       | closure c c' vs remaining args_arity remaining_arity _ | partial];
    simpl in Hcalls, Hcalls', Hinv |- *.
  - exact (ended_undefined b).
  - destruct Hcalls as (called & slots & Hclosure & Hcode & _).
    destruct Hcalls' as (called' & slots' & Hclosure' & Hcode' & _).
    destruct (called_code_related _ _ _ _ _ _ _ _ _ Hstatics Hclosure Hcode
                Hclosure' Hcode')
      as (-> & -> & Hc).
    exact (call_code_transfer n closure called slots c c' vs _ _ _ _ _ _
             statics statics' _ b IH Hc Hclosure' Hreturn Hexn Hstatics Hinv).
  - destruct Hcalls as (called & slots & Hclosure & Hcode & _).
    destruct Hcalls' as (called' & slots' & Hclosure' & Hcode' & _).
    destruct (called_code_related _ _ _ _ _ _ _ _ _ Hstatics Hclosure Hcode
                Hclosure' Hcode')
      as (-> & -> & Hc).
    replace (code_result_arity c') with (code_result_arity c) in Hinv |- *
      by (destruct Hc; reflexivity).
    apply (call_code_transfer n closure called slots c c' vs _ _ _ _ _ _
             statics statics' _ b IH Hc Hclosure');
      [ apply Over_related; assumption | exact Hexn | exact Hstatics
      | exact Hinv ].
  - exact (Hpartial partial Hinv).
Qed.

(* Related continuations applied to the same values. Where the second
   steps, the state it steps to has [inv]. *)
Lemma apply_cont_value_transfer (n : nat) (c : cont_value) :
  transfer_at n ->
  ∀ (c' : cont_value) (vs : list value) (statics statics' : static_env)
    (hp : heap)
    (b : behaviour),
    closure_related c c' ->
    statics_related_where statics statics' ->
    result_inv (apply_cont_value c' vs statics' hp) ->
    ended (run_on n (apply_cont_value c vs statics hp)) b ->
    result_carried (apply_cont_value c' vs statics' hp) b.
Proof.
  intros IH.
  induction c
    as [[params metadata body] def_vars def_conts recursive | exit
       | c IHc extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros c' vs statics statics' hp b Hc Hstatics Hinv Hended;
    apply closure_related_inv in Hc.
  - destruct c'
      as [[params' metadata' body'] def_vars' def_conts' recursive' | | | ];
      try contradiction.
    destruct Hc as (<- & Hh & Hconts & Hrec).
    apply handler_related_inv in Hh as (<- & <- & Hbody).
    simpl in Hended, Hinv |- *.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
      eqn:Hkinds;
      simpl in Hended, Hinv |- *; [ | exact (ended_undefined b Hended) ].
    destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
      simpl in Hended, Hinv |- *; [ | exact (ended_undefined b Hended) ].
    assert (Hconts' :
      cont_env_related
        (rec_closures recursive def_vars def_conts ++ def_conts)
        (rec_closures recursive' def_vars def_conts' ++ def_conts')).
    { apply Forall2_app; [ | exact Hconts ].
      apply rec_closures_related; assumption. }
    destruct Hbody as [Hbody | <-];
      [ | exact (IH _ _ _ _ _ _ _ _ Hconts' Hstatics Hinv Hended) ].
    (* The state of the second handler has [inv]; so has that of the
       first, which the induction is about. *)
    pose proof (proj1 (kinds_eqb_eq _ _) Hkinds) as Hkinds_eq.
    pose proof (inv_related _ _ _ _ _ _ _ _ _ hp Hbody Hkinds_eq Hbind Hinv)
      as Hfirst.
    exact
      (carried_related _ _ _ _ _ _ _ _ _ _ _ Hbody Hfirst Hkinds_eq Hbind
         (IH _ _ _ _ _ _ _ _ Hconts' Hstatics Hfirst Hended)).
  - destruct c' as [ | exit' | | ]; try contradiction. subst exit'.
    exact (ended_exit exit vs _ b Hended).
  - destruct c' as [ | | c' extra' | ]; try contradiction.
    destruct Hc as [<- Hc].
    exact (IHc c' (vs ++ extra) statics statics' hp b Hc Hstatics Hinv Hended).
  - destruct c'
      as [ | |
         | remaining' remaining_arity' return_arity' return_to' exn_to'];
      try contradiction.
    destruct Hc as (<- & <- & <- & Hreturn & Hexn).
    rewrite apply_cont_value_over_application in Hended, Hinv |- *.
    destruct vs as [ | v [ | v' vs]];
      [ exact (ended_undefined b Hended) | | exact (ended_undefined b Hended) ].
    unfold call_function in Hended, Hinv |- *.
    refine
      (run_call_transfer n _ _ _ _ _ _ _ _ _ _ statics statics' _ b IH _
         (shape_of_call_calls _ _ _ _ _ _) (shape_of_call_calls _ _ _ _ _ _)
         Hreturn Hexn Hstatics Hinv _ Hended).
    + apply shape_of_call_related;
        [ symmetry; apply closure_of_related; exact Hstatics | | ].
      * intros called slots _. apply codes_agree_related. exact Hstatics.
      * intros tuple rest p _. symmetry.
        exact (eval_prim_same_static_values _ _ p [tuple] (proj2 Hstatics)).
    + intros partial Hinv' Hpartial.
      exact (IHreturn return_to' [partial] statics statics' hp b Hreturn
               Hstatics Hinv' Hpartial).
Qed.

(* The same call under related static environments, returning to related
   continuations. *)
Lemma call_function_transfer (n : nat) (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics statics' : static_env) (hp : heap) (b : behaviour) :
  transfer_at n ->
  closure_related return_to return_to' ->
  closure_related exn_to exn_to' ->
  statics_related_where statics statics' ->
  result_inv
    (call_function callee kind vs args_arity return_arity return_to' exn_to'
       statics' hp) ->
  ended
    (run_on n
       (call_function callee kind vs args_arity return_arity return_to
          exn_to statics hp))
    b ->
  result_carried
    (call_function callee kind vs args_arity return_arity return_to' exn_to'
       statics' hp)
    b.
Proof.
  intros IH Hreturn Hexn Hstatics Hinv Hended.
  unfold call_function in Hended, Hinv |- *.
  refine
    (run_call_transfer n _ _ _ _ _ _ _ _ _ _ statics statics' _ b IH _
       (shape_of_call_calls _ _ _ _ _ _) (shape_of_call_calls _ _ _ _ _ _)
       Hreturn Hexn Hstatics Hinv _ Hended).
  - apply shape_of_call_related;
      [ symmetry; apply closure_of_related; exact Hstatics | | ].
    + intros called slots _. apply codes_agree_related. exact Hstatics.
    + intros tuple rest p _. symmetry.
      exact (eval_prim_same_static_values _ _ p [tuple] (proj2 Hstatics)).
  - intros partial Hinv' Hpartial.
    exact (apply_cont_value_transfer n return_to IH return_to' [partial]
             statics statics' hp b Hreturn Hstatics Hinv' Hpartial).
Qed.

Lemma transfer (n : nat) : transfer_at n.
Proof.
  unfold transfer_at.
  induction n as [ | n IH];
    intros e vars conts conts' statics statics' hp b Hrel Hstatics Hinv Hended;
    [ destruct b; contradiction | ].
  rewrite run_succ in Hended.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hended; simpl in Hended.
  - (* Let *)
    destruct (eval_named statics vars d) as [v | ] eqn:Hd; simpl in Hended;
      [ destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind;
        simpl in Hended
      | ].
    + assert (Hstep :
        step (State (Let x kind d body) vars conts' statics' hp)
        = Step (State body ((x, v) :: vars) conts' statics' hp)).
      { rewrite step_let,
          <- (eval_named_related statics statics' vars d Hstatics), Hd,
          Hkind.
        reflexivity. }
      apply (carried_step _ _ _ Hstep).
      exact
        (IH _ _ _ _ _ _ _ _ Hrel Hstatics (inv_step _ _ Hinv Hstep) Hended).
    + apply ended_undefined in Hended. subst b.
      apply carried_undefined.
      rewrite step_let,
        <- (eval_named_related statics statics' vars d Hstatics), Hd, Hkind.
      reflexivity.
    + apply ended_undefined in Hended. subst b.
      apply carried_undefined.
      rewrite step_let,
        <- (eval_named_related statics statics' vars d Hstatics), Hd.
      reflexivity.
  - (* Let_static: the same bindings, added to related environments *)
    pose proof (eval_static_bindings_related_where vars bindings bindings
      (bindings_related_where_refl bindings) statics statics' Hstatics)
      as Hextended.
    destruct (eval_static_bindings vars bindings statics) as [extended | ],
      (eval_static_bindings vars bindings statics') as [extended' | ]
        eqn:Hbindings';
      try contradiction; simpl in Hended.
    + assert (Hstep :
        step (State (Let_static bindings body) vars conts' statics' hp)
        = Step (State body vars conts' extended' hp)).
      { rewrite step_let_static, Hbindings'. reflexivity. }
      apply (carried_step _ _ _ Hstep).
      exact
        (IH _ _ _ _ _ _ _ _ Hrel Hextended (inv_step _ _ Hinv Hstep) Hended).
    + apply ended_undefined in Hended. subst b.
      apply carried_undefined. rewrite step_let_static, Hbindings'.
      reflexivity.
  - (* Let_cont_non_recursive *)
    pose proof (step_let_cont_non_recursive k h body vars conts' statics')
      as Hstep.
    apply (carried_step _ _ _ (Hstep hp)).
    eapply IH;
      [ | exact Hstatics | exact (inv_step _ _ Hinv (Hstep hp)) | exact Hended
       ].
    apply Forall2_cons; [ | exact Hrel ].
    split; [ reflexivity | ].
    apply Closure_related;
      [ apply Handler_same | exact Hrel | apply Forall2_nil ].
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers) eqn:Hbinds; simpl in Hended;
      [ apply ended_undefined in Hended; subst b; apply carried_undefined;
        rewrite step_let_cont_recursive, Hbinds; reflexivity
      | ].
    assert (Hstep :
      step (State (Let_cont_recursive handlers body) vars conts' statics' hp)
      = Step
          (State body vars (rec_closures handlers vars conts' ++ conts')
             statics' hp)).
    { rewrite step_let_cont_recursive, Hbinds. reflexivity. }
    apply (carried_step _ _ _ Hstep).
    eapply IH;
      [ | exact Hstatics | exact (inv_step _ _ Hinv Hstep) | exact Hended ].
    apply Forall2_app; [ | exact Hrel ].
    apply rec_closures_related; [ apply handlers_related_refl | exact Hrel ].
  - (* Apply: the same call of related code, returning to related
       continuations *)
    destruct (eval_simple vars (ap_callee ap)) as [closure | ] eqn:Hcallee;
      simpl in Hended;
      [ destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
        simpl in Hended;
        [ destruct (eval_simples vars (exn_extra_simples ap))
            as [extra | ] eqn:Hextra;
          simpl in Hended
        | ]
      | ].
    + assert (Hinv' :
        result_inv (step (State (Apply ap) vars conts' statics' hp))).
      { destruct (step (State (Apply ap) vars conts' statics' hp))
          as [s' | | ] eqn:Hstep; simpl; [ | exact I | exact I ].
        exact (inv_step _ _ Hinv Hstep). }
      apply carried_of_result.
      rewrite step_apply, Hcallee, Hargs, Hextra in Hinv' |- *.
      exact (call_function_transfer n closure (ap_call_kind ap) vs _ _ _ _ _
               _ statics statics' _ b IH
               (resolve_related (ap_continuation ap) _ _ Hrel)
               (with_extra_args_related _ _ extra
                  (resolve_related (ap_exn_continuation ap) _ _ Hrel))
               Hstatics Hinv' Hended).
    + apply ended_undefined in Hended. subst b.
      apply carried_undefined.
      rewrite step_apply, Hcallee, Hargs, Hextra. reflexivity.
    + apply ended_undefined in Hended. subst b.
      apply carried_undefined. rewrite step_apply, Hcallee, Hargs. reflexivity.
    + apply ended_undefined in Hended. subst b.
      apply carried_undefined. rewrite step_apply, Hcallee. reflexivity.
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      simpl in Hended.
    + unfold apply_continuation in Hended.
      assert (Hinv' :
        result_inv (step (State (Apply_cont ac) vars conts' statics' hp))).
      { destruct (step (State (Apply_cont ac) vars conts' statics' hp))
          as [s' | | ] eqn:Hstep; simpl; [ | exact I | exact I ].
        exact (inv_step _ _ Hinv Hstep). }
      apply carried_of_result.
      rewrite step_apply_cont, Hargs in Hinv' |- *.
      unfold apply_continuation in Hinv' |- *.
      exact (apply_cont_value_transfer n _ IH _ vs statics statics' hp b
               (resolve_related (ac_continuation ac) _ _ Hrel) Hstatics Hinv'
               Hended).
    + apply ended_undefined in Hended. subst b.
      apply carried_undefined. rewrite step_apply_cont, Hargs. reflexivity.
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ] eqn:Hscrutinee;
      simpl in Hended;
      [ destruct (naked_immediate_of_value v) as [i | ] eqn:Himmediate;
        simpl in Hended;
        [ destruct (find_arm i arms) as [ac | ] eqn:Hfind; simpl in Hended
        | ]
      | ].
    + assert (Hstep :
        step (State (Switch scrutinee arms) vars conts' statics' hp)
        = Step (State (Apply_cont ac) vars conts' statics' hp)).
      { rewrite step_switch, Hscrutinee, Himmediate, Hfind. reflexivity. }
      apply (carried_step _ _ _ Hstep).
      exact
        (IH _ _ _ _ _ _ _ _ Hrel Hstatics (inv_step _ _ Hinv Hstep) Hended).
    + apply ended_undefined in Hended. subst b.
      apply carried_undefined.
      rewrite step_switch, Hscrutinee, Himmediate, Hfind. reflexivity.
    + apply ended_undefined in Hended. subst b.
      apply carried_undefined. rewrite step_switch, Hscrutinee, Himmediate.
      reflexivity.
    + apply ended_undefined in Hended. subst b.
      apply carried_undefined. rewrite step_switch, Hscrutinee. reflexivity.
  - (* Invalid *)
    apply ended_undefined in Hended. subst b.
    apply carried_undefined, step_invalid.
Qed.

End Transfer.

(* ======================================================================== *)
(* Relations that do not look at what the code binds                        *)
(* ======================================================================== *)

(* The relations on code, bindings and static environments for a relation
   between the bodies alone: the instance of the ones above at a relation
   that ignores the code id, what the code binds and its metadata. Two
   codes are related when they are at every code id. *)
Section Bodies.

Variable code_related : expr -> expr -> Prop.

Definition ignoring_binders (id : code_id) (params : list variable)
    (my_closure : variable) (ret exn : continuation)
    (metadata : code_metadata)
    : expr -> expr -> Prop :=
  code_related.

Definition codes_related (c c' : code) : Prop :=
  ∀ id : code_id, codes_related_where ignoring_binders id c c'.

Definition binding_related : static_binding -> static_binding -> Prop :=
  binding_related_where ignoring_binders.

Definition statics_related : static_env -> static_env -> Prop :=
  statics_related_where ignoring_binders.

Lemma Codes_related (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata) (body body' : expr) :
  binds_statics body = binds_statics body' ->
  code_related body body' ->
  codes_related (Code params my_closure ret exn metadata body)
    (Code params my_closure ret exn metadata body').
Proof. intros Hbinds Hbody id. apply Codes_related_where; assumption. Qed.

Lemma Codes_same (c : code) : codes_related c c.
Proof. intros id. apply Codes_same_where. Qed.

Lemma Binding_code (id : code_id) (c c' : code) :
  codes_related c c' -> binding_related (Bind_code id c) (Bind_code id c').
Proof. intros Hc. apply Binding_code_where, Hc. Qed.

Lemma Binding_same (binding : static_binding) :
  binding_related binding binding.
Proof. apply Binding_same_where. Qed.

Lemma statics_related_refl (statics : static_env) :
  statics_related statics statics.
Proof. apply statics_related_where_refl. Qed.

Lemma bindings_related_refl (bindings : list static_binding) :
  Forall2 binding_related bindings bindings.
Proof. apply bindings_related_where_refl. Qed.

Lemma eval_static_bindings_related (vars : env)
    (bindings bindings' : list static_binding) :
  Forall2 binding_related bindings bindings' ->
  ∀ statics statics' : static_env,
    statics_related statics statics' ->
    match
      eval_static_bindings vars bindings statics,
      eval_static_bindings vars bindings' statics'
    with
    | Some extended, Some extended' => statics_related extended extended'
    | None, None => True
    | _, _ => False
    end.
Proof. apply eval_static_bindings_related_where. Qed.

End Bodies.

(* ======================================================================== *)
(* Refinement under related environments                                    *)
(* ======================================================================== *)

(* The two halves of [refines], as properties to carry: forwards an exit,
   backwards the way the target ends or that the source goes wrong. *)
Definition exits_carried (s : state) (b : behaviour) : Prop :=
  match b with
  | Exits _ _ _ => behaves s b
  | Goes_wrong | Diverges => True
  end.

Definition ends_or_wrong (s : state) (b : behaviour) : Prop :=
  b <> Diverges -> behaves s b ∨ behaves s Goes_wrong.

(* ======================================================================== *)
(* Refinement from the states that have a predicate                         *)
(* ======================================================================== *)

(* The two states have [inv] together, and the first, if it has it,
   refines the second. *)
Definition refines_from (inv : state -> Prop) (source target : state)
    : Prop :=
  (inv source <-> inv target) ∧ (inv source -> refines source target).

Section From.

(* A predicate on states that a step keeps. *)
Variable inv : state -> Prop.

Hypothesis inv_step :
  ∀ s s' : state, inv s -> step s = Step s' -> inv s'.

(* Relations between handlers, and between the bodies of code, from the
   source to the target and back. Related expressions refine where they
   run, from the states that have [inv], and have [inv] together there: a
   handler under the variables of its closure with its parameters bound,
   any continuations and any static environment; the body of code at a
   call of a closure of the code id that the two bodies are related at.
   The second is for [transfer], which goes through states that no run
   of the two whole states need reach: an expression of one side under
   the environments of the other. A step keeps [inv] along a run, which
   says nothing of such a state; [inv] of it comes from [inv] of the
   state with the corresponding expression, which a run does reach. *)
Variable forwards backwards
  : list variable -> list flambda_kind -> env -> expr -> expr -> Prop.

Variable code_forwards code_backwards :
  code_id -> list variable -> variable -> continuation -> continuation ->
  code_metadata -> expr -> expr -> Prop.

Hypothesis forwards_refines :
  ∀ (params : list variable) (kinds : list flambda_kind) (def_vars : env)
    (source target : expr) (vs : list value) (handler_vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap),
    forwards params kinds def_vars source target ->
    map kind_of_value vs = kinds ->
    bind_params params vs def_vars = Some handler_vars ->
    refines_from inv (State source handler_vars conts statics hp)
      (State target handler_vars conts statics hp).

Hypothesis backwards_refines :
  ∀ (params : list variable) (kinds : list flambda_kind) (def_vars : env)
    (target source : expr) (vs : list value) (handler_vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap),
    backwards params kinds def_vars target source ->
    map kind_of_value vs = kinds ->
    bind_params params vs def_vars = Some handler_vars ->
    refines_from inv (State source handler_vars conts statics hp)
      (State target handler_vars conts statics hp).

Hypothesis code_forwards_refines :
  ∀ (id : code_id) (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (source target : expr) (vs : list value) (callee : value)
    (callee_vars : env) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (slots : list (value_slot * value)),
    code_forwards id params my_closure ret exn metadata source target ->
    bind_params params vs [(my_closure, callee)] = Some callee_vars ->
    map kind_of_value vs = unarize (cm_params_arity metadata) ->
    kind_of_value callee = Value ->
    closure_of statics callee = Some (id, slots) ->
    refines_from inv
      (State source callee_vars [(ret, return_to); (exn, exn_to)] statics hp)
      (State target callee_vars [(ret, return_to); (exn, exn_to)] statics hp).

Hypothesis code_backwards_refines :
  ∀ (id : code_id) (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (target source : expr) (vs : list value) (callee : value)
    (callee_vars : env) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (slots : list (value_slot * value)),
    code_backwards id params my_closure ret exn metadata target source ->
    bind_params params vs [(my_closure, callee)] = Some callee_vars ->
    map kind_of_value vs = unarize (cm_params_arity metadata) ->
    kind_of_value callee = Value ->
    closure_of statics callee = Some (id, slots) ->
    refines_from inv
      (State source callee_vars [(ret, return_to); (exn, exn_to)] statics hp)
      (State target callee_vars [(ret, return_to); (exn, exn_to)] statics hp).

(* An expression under environments refines it under ones whose closures
   have refining handlers and whose code has refining bodies, when both
   states have [inv]. The relations are needed in both directions, as the
   two halves of [refines] run different sides. *)
Lemma related_states_refines_from (e : expr) (vars : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp : heap) :
  inv (State e vars conts statics hp) ->
  inv (State e vars conts' statics' hp) ->
  cont_env_related forwards conts conts' ->
  cont_env_related backwards conts' conts ->
  statics_related_where code_forwards statics statics' ->
  statics_related_where code_backwards statics' statics ->
  refines (State e vars conts statics hp) (State e vars conts' statics' hp).
Proof.
  intros Hinv Hinv' Hrel Hrel' Hstatics Hstatics'. split.
  - intros k vs hp' [n Hrun].
    refine (transfer forwards code_forwards inv inv_step exits_carried
      _ _ _ _ _ _ _ n e vars conts conts' statics statics' hp (Exits k vs hp')
      Hrel Hstatics Hinv' _).
    + intros s s' b Hstep.
      destruct b as [k' vs' hp'' | | ]; unfold exits_carried; auto.
      intros Hb. apply (step_behaves s (Exits k' vs' hp'')). rewrite Hstep.
      exact Hb.
    + intros s k' vs' hp'' Hstep. apply (step_behaves s (Exits k' vs' hp'')).
      rewrite Hstep. reflexivity.
    + intros s _. exact I.
    + intros params kinds def_vars source target vs' handler_vars conts''
        statics'' hp'' Hrelated Hkinds Hbind.
      exact
        (proj2
           (proj1
              (forwards_refines _ _ _ _ _ _ _ _ _ _ Hrelated Hkinds Hbind))).
    + intros params kinds def_vars source target vs' handler_vars conts''
        statics'' hp'' b Hrelated Hsource Hkinds Hbind.
      destruct b; unfold exits_carried; auto.
      apply
        (proj1
           (proj2
              (forwards_refines _ _ _ _ _ _ _ _ _ _ Hrelated Hkinds Hbind)
              Hsource)).
    + intros id params my_closure ret exn metadata source target vs' callee
        callee_vars return_to exn_to statics'' hp'' slots Hrelated Hbind
        Hkinds Hvalue Hclosure.
      exact
        (proj2
           (proj1
              (code_forwards_refines _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
                 Hrelated Hbind Hkinds Hvalue Hclosure))).
    + intros id params my_closure ret exn metadata source target vs' callee
        callee_vars return_to exn_to statics'' hp'' slots b Hrelated Hsource
        Hbind Hkinds Hvalue Hclosure.
      destruct b; unfold exits_carried; auto.
      apply
        (proj1
           (proj2
              (code_forwards_refines _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
                 Hrelated Hbind Hkinds Hvalue Hclosure)
              Hsource)).
    + rewrite Hrun. simpl. auto.
  - intros b Hterm Hb. destruct (behaves_ended _ _ Hterm Hb) as [n Hended].
    refine (transfer backwards code_backwards inv inv_step ends_or_wrong
      _ _ _ _ _ _ _ n e vars conts' conts statics' statics hp b Hrel'
      Hstatics' Hinv Hended Hterm).
    + intros s s' b' Hstep Hcarried Hterm'.
      destruct (Hcarried Hterm') as [Hb' | Hwrong]; [ left | right ];
        [ apply (step_behaves s b') | apply (step_behaves s Goes_wrong) ];
        rewrite Hstep; assumption.
    + intros s k vs hp'' Hstep _. left.
      apply (step_behaves s (Exits k vs hp'')).
      rewrite Hstep. reflexivity.
    + intros s Hstep _. left. apply (step_behaves s Goes_wrong).
      rewrite Hstep. reflexivity.
    + intros params kinds def_vars target source vs handler_vars conts''
        statics'' hp'' Hrelated Hkinds Hbind.
      exact
        (proj1
           (proj1
              (backwards_refines _ _ _ _ _ _ _ _ _ _ Hrelated Hkinds Hbind))).
    + intros params kinds def_vars target source vs handler_vars conts''
        statics'' hp'' b' Hrelated Htarget Hkinds Hbind Hcarried Hterm'.
      destruct
        (backwards_refines _ _ _ _ _ _ _ conts'' statics'' hp'' Hrelated
           Hkinds Hbind)
        as [Htogether Hrefines].
      destruct (Hrefines (proj2 Htogether Htarget)) as [_ Hends].
      destruct (Hcarried Hterm') as [Hb' | Hwrong];
        [ exact (Hends b' Hterm' Hb') | ].
      destruct (Hends Goes_wrong ltac:(discriminate) Hwrong); right;
        assumption.
    + intros id params my_closure ret exn metadata target source vs callee
        callee_vars return_to exn_to statics'' hp'' slots Hrelated Hbind
        Hkinds Hvalue Hclosure.
      exact
        (proj1
           (proj1
              (code_backwards_refines _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
                 Hrelated Hbind Hkinds Hvalue Hclosure))).
    + intros id params my_closure ret exn metadata target source vs callee
        callee_vars return_to exn_to statics'' hp'' slots b' Hrelated
        Htarget Hbind Hkinds Hvalue Hclosure Hcarried Hterm'.
      destruct
        (code_backwards_refines _ _ _ _ _ _ _ _ _ _ _ return_to exn_to
           statics'' hp'' _ Hrelated Hbind Hkinds Hvalue Hclosure)
        as [Htogether Hrefines].
      destruct (Hrefines (proj2 Htogether Htarget)) as [_ Hends].
      destruct (Hcarried Hterm') as [Hb' | Hwrong];
        [ exact (Hends b' Hterm' Hb') | ].
      destruct (Hends Goes_wrong ltac:(discriminate) Hwrong); right;
        assumption.
Qed.

(* The case of one static environment. *)
Lemma related_envs_refines_from (e : expr) (vars : env)
    (conts conts' : cont_env) (statics : static_env) (hp : heap) :
  inv (State e vars conts statics hp) ->
  inv (State e vars conts' statics hp) ->
  cont_env_related forwards conts conts' ->
  cont_env_related backwards conts' conts ->
  refines (State e vars conts statics hp) (State e vars conts' statics hp).
Proof.
  intros Hinv Hinv' Hrel Hrel'.
  apply related_states_refines_from; try assumption;
    apply statics_related_where_refl.
Qed.

(* A [Let_cont] refines the same with a handler replaced by a related
   one, when both states have [inv]. *)
Lemma let_cont_handler_refines_from (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) :
  inv
    (State
       (Let_cont_non_recursive k (Handler params metadata handler) body)
       vars conts statics hp) ->
  inv
    (State
       (Let_cont_non_recursive k (Handler params metadata handler') body)
       vars conts statics hp) ->
  forwards params (hm_params_kinds metadata) vars handler handler' ->
  backwards params (hm_params_kinds metadata) vars handler' handler ->
  refines
    (State
       (Let_cont_non_recursive k (Handler params metadata handler) body)
       vars conts statics hp)
    (State
       (Let_cont_non_recursive k (Handler params metadata handler') body)
       vars conts statics hp).
Proof.
  intros Hinv Hinv' Hforwards Hbackwards.
  apply let_cont_non_recursive_refines_step.
  apply related_envs_refines_from;
    [ exact
        (inv_step _ _ Hinv
           (step_let_cont_non_recursive k (Handler params metadata handler)
              body vars conts statics hp))
    | exact
        (inv_step _ _ Hinv'
           (step_let_cont_non_recursive k (Handler params metadata handler')
              body vars conts statics hp))
    | | ];
    (apply Forall2_cons; [ | apply cont_env_related_refl ]);
    (split; [ reflexivity | ]);
    (apply Closure_related;
      [ apply Handler_related; assumption | apply cont_env_related_refl
      | apply Forall2_nil ]).
Qed.

(* The same for recursive handlers, any number of them replaced. *)
Lemma let_cont_recursive_handlers_refine_from
    (handlers handlers' : list (continuation * continuation_handler))
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
      (hp : heap) :
  inv (State (Let_cont_recursive handlers body) vars conts statics hp) ->
  inv (State (Let_cont_recursive handlers' body) vars conts statics hp) ->
  handlers_bind_statics handlers = handlers_bind_statics handlers' ->
  handlers_related forwards vars handlers handlers' ->
  handlers_related backwards vars handlers' handlers ->
  refines (State (Let_cont_recursive handlers body) vars conts statics hp)
    (State (Let_cont_recursive handlers' body) vars conts statics hp).
Proof.
  intros Hinv Hinv' Hbinds Hhandlers Hhandlers'.
  apply let_cont_recursive_refines_step; [ exact Hbinds | intros Hnone ].
  assert (Hstep :
    ∀ group : list (continuation * continuation_handler),
      handlers_bind_statics group = false ->
      step (State (Let_cont_recursive group body) vars conts statics hp)
      = Step
          (State body vars (rec_closures group vars conts ++ conts) statics
             hp)).
  { intros group Hgroup. rewrite step_let_cont_recursive, Hgroup.
    reflexivity. }
  apply related_envs_refines_from;
    [ exact (inv_step _ _ Hinv (Hstep handlers Hnone))
    | exact
        (inv_step _ _ Hinv'
           (Hstep handlers' (eq_trans (eq_sym Hbinds) Hnone)))
    | | ];
    (apply Forall2_app; [ | apply cont_env_related_refl ]);
    (apply rec_closures_related; [ assumption | apply cont_env_related_refl ]).
Qed.

(* A [Let_static] refines the same with the body of a piece of code
   replaced by one related at the code id that the code is bound to,
   when both states have [inv]. *)
Lemma let_static_code_refines_from (before : list static_binding)
    (id : code_id) (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (code_body code_body' : expr) (after : list static_binding) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap) :
  inv
    (State
       (Let_static
          (before
           ++ Bind_code id
                (Code params my_closure ret exn metadata code_body)
           :: after)
          body)
       vars conts statics hp) ->
  inv
    (State
       (Let_static
          (before
           ++ Bind_code id
                (Code params my_closure ret exn metadata code_body')
           :: after)
          body)
       vars conts statics hp) ->
  binds_statics code_body = binds_statics code_body' ->
  code_forwards id params my_closure ret exn metadata code_body code_body' ->
  code_backwards id params my_closure ret exn metadata code_body'
    code_body ->
  refines
    (State
       (Let_static
          (before
           ++ Bind_code id
                (Code params my_closure ret exn metadata code_body)
           :: after)
          body)
       vars conts statics hp)
    (State
       (Let_static
          (before
           ++ Bind_code id
                (Code params my_closure ret exn metadata code_body')
           :: after)
          body)
       vars conts statics hp).
Proof.
  intros Hinv Hinv' Hbinds Hforwards Hbackwards.
  set (c := Code params my_closure ret exn metadata code_body) in *.
  set (c' := Code params my_closure ret exn metadata code_body') in *.
  assert (Hbindings :
    ∀ (R :
         code_id -> list variable -> variable -> continuation ->
         continuation -> code_metadata -> expr -> expr -> Prop)
      (c1 c2 : code),
      codes_related_where R id c1 c2 ->
      Forall2 (binding_related_where R) (before ++ Bind_code id c1 :: after)
        (before ++ Bind_code id c2 :: after)).
  { intros R c1 c2 Hc.
    apply Forall2_app; [ apply bindings_related_where_refl | ].
    apply Forall2_cons; [ | apply bindings_related_where_refl ].
    apply Binding_code_where, Hc. }
  pose proof (eval_static_bindings_related_where code_forwards vars _ _
    (Hbindings code_forwards c c'
       (Codes_related_where _ _ _ _ _ _ _ _ _ Hbinds Hforwards))
    statics statics (statics_related_where_refl _ statics)) as Hthere.
  pose proof (eval_static_bindings_related_where code_backwards vars _ _
    (Hbindings code_backwards c' c
       (Codes_related_where code_backwards id params my_closure ret exn
          metadata code_body' code_body (eq_sym Hbinds) Hbackwards))
    statics statics (statics_related_where_refl _ statics)) as Hback.
  apply let_static_refines_step.
  destruct
    (eval_static_bindings vars (before ++ Bind_code id c :: after) statics)
    as [extended | ] eqn:Hextended,
    (eval_static_bindings vars (before ++ Bind_code id c' :: after) statics)
    as [extended' | ] eqn:Hextended';
    try contradiction; [ | exact I ].
  assert (Hstep :
    ∀ (bindings : list static_binding) (extended'' : static_env),
      eval_static_bindings vars bindings statics = Some extended'' ->
      step (State (Let_static bindings body) vars conts statics hp)
      = Step (State body vars conts extended'' hp)).
  { intros bindings extended'' Hbound.
    rewrite step_let_static, Hbound. reflexivity. }
  apply related_states_refines_from;
    [ exact (inv_step _ _ Hinv (Hstep _ _ Hextended))
    | exact (inv_step _ _ Hinv' (Hstep _ _ Hextended'))
    | apply cont_env_related_refl | apply cont_env_related_refl
    | exact Hthere | exact Hback ].
Qed.

End From.

(* A handler refines another where it runs: under the variables of its
   closure with its parameters bound, any continuations, and any static
   environment of which [ok] holds. *)
Definition handler_refines_where (ok : static_env -> Prop)
    (params : list variable) (kinds : list flambda_kind) (def_vars : env)
    (source target : expr) : Prop :=
  ∀ (vs : list value) (handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap),
    ok statics ->
    map kind_of_value vs = kinds ->
    bind_params params vs def_vars = Some handler_vars ->
    refines (State source handler_vars conts statics hp)
      (State target handler_vars conts statics hp).

Definition expr_refines_where (ok : static_env -> Prop)
    (source target : expr) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap),
    ok statics ->
    refines (State source vars conts statics hp)
      (State target vars conts statics hp).

(* The body of code refines another where it runs: at a call, under its
   parameters bound to values of the kinds of its arity and [my_closure] to
   a value of kind [Value], any continuations, and any static environment
   of which [ok] holds. The kinds are those of the values; nothing is
   asked of the kinds that a term gives. *)
Definition code_refines_where (ok : static_env -> Prop)
    (params : list variable) (my_closure : variable)
    (metadata : code_metadata) (source target : expr) : Prop :=
  ∀ (vs : list value) (callee : value) (callee_vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap),
    ok statics ->
    map kind_of_value vs = unarize (cm_params_arity metadata) ->
    kind_of_value callee = Value ->
    bind_params params vs [(my_closure, callee)] = Some callee_vars ->
    refines (State source callee_vars conts statics hp)
      (State target callee_vars conts statics hp).

(* The same under any static environment. *)
Definition handler_refines_at (params : list variable)
    (kinds : list flambda_kind) (def_vars : env) (source target : expr)
    : Prop :=
  ∀ (vs : list value) (handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap),
    map kind_of_value vs = kinds ->
    bind_params params vs def_vars = Some handler_vars ->
    refines (State source handler_vars conts statics hp)
      (State target handler_vars conts statics hp).

Definition always (statics : static_env) : Prop := True.

Lemma always_extend (statics statics' : static_env) :
  always statics -> statics_extend statics statics' -> always statics'.
Proof. intros _ _. exact I. Qed.

Section Where.

Variable ok : static_env -> Prop.

Hypothesis ok_extend :
  ∀ statics statics' : static_env,
    ok statics -> statics_extend statics statics' -> ok statics'.

(* [ok] of the static environment of a state: a predicate on states that
   a step keeps, since a step extends the static environment, and that
   does not look at the expression. The lemmas of this section are those
   of the section above at this predicate. *)
Definition statics_ok (s : state) : Prop := ok (state_statics s).

Lemma statics_ok_step (s s' : state) :
  statics_ok s -> step s = Step s' -> statics_ok s'.
Proof.
  intros Hok Hstep.
  exact (ok_extend _ _ Hok (step_extends_statics s s' Hstep)).
Qed.

Lemma statics_ok_refines_from (source target : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) :
  (ok statics ->
   refines (State source vars conts statics hp)
     (State target vars conts statics hp)) ->
  refines_from statics_ok (State source vars conts statics hp)
    (State target vars conts statics hp).
Proof.
  intros Hrefines. split; [ split; intros Hok; exact Hok | exact Hrefines ].
Qed.

Lemma handler_refines_where_from (params : list variable)
    (kinds : list flambda_kind) (def_vars : env) (source target : expr)
    (vs : list value) (handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) :
  handler_refines_where ok params kinds def_vars source target ->
  map kind_of_value vs = kinds ->
  bind_params params vs def_vars = Some handler_vars ->
  refines_from statics_ok (State source handler_vars conts statics hp)
    (State target handler_vars conts statics hp).
Proof.
  intros Hrefines Hkinds Hbind. apply statics_ok_refines_from. intros Hok.
  exact (Hrefines vs handler_vars conts statics hp Hok Hkinds Hbind).
Qed.

Lemma expr_refines_where_from (source target : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) :
  expr_refines_where ok source target ->
  refines_from statics_ok (State source vars conts statics hp)
    (State target vars conts statics hp).
Proof.
  intros Hrefines. apply statics_ok_refines_from. intros Hok.
  exact (Hrefines vars conts statics hp Hok).
Qed.

Lemma code_refines_where_from (params : list variable)
    (my_closure : variable) (metadata : code_metadata)
    (source target : expr) (vs : list value) (callee : value)
    (callee_vars : env) (conts : cont_env) (statics : static_env) (hp : heap) :
  code_refines_where ok params my_closure metadata source target ->
  bind_params params vs [(my_closure, callee)] = Some callee_vars ->
  map kind_of_value vs = unarize (cm_params_arity metadata) ->
  kind_of_value callee = Value ->
  refines_from statics_ok (State source callee_vars conts statics hp)
    (State target callee_vars conts statics hp).
Proof.
  intros Hrefines Hbind Hkinds Hvalue. apply statics_ok_refines_from.
  intros Hok.
  exact
    (Hrefines vs callee callee_vars conts statics hp Hok Hkinds Hvalue Hbind).
Qed.

(* An expression under environments refines it under ones whose closures
   have refining handlers and whose code has refining bodies. The
   relations are needed in both directions, as the two halves of [refines]
   run different sides. *)
Lemma related_states_refines (e : expr) (vars : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp : heap) :
  ok statics -> ok statics' ->
  cont_env_related (handler_refines_where ok) conts conts' ->
  cont_env_related
    (fun params kinds def_vars target source =>
       handler_refines_where ok params kinds def_vars source target)
    conts' conts ->
  statics_related (expr_refines_where ok) statics statics' ->
  statics_related (fun target source => expr_refines_where ok source target)
    statics' statics ->
  refines (State e vars conts statics hp) (State e vars conts' statics' hp).
Proof.
  intros Hok Hok' Hrel Hrel' Hstatics Hstatics'.
  exact
    (related_states_refines_from statics_ok statics_ok_step
       (handler_refines_where ok)
       (fun params kinds def_vars target source =>
          handler_refines_where ok params kinds def_vars source target)
       (ignoring_binders (expr_refines_where ok))
       (ignoring_binders
          (fun target source => expr_refines_where ok source target))
       handler_refines_where_from
       (fun params kinds def_vars target source =>
          handler_refines_where_from params kinds def_vars source target)
       (fun id params my_closure ret exn metadata source target vs callee
          callee_vars return_to exn_to statics'' hp'' slots Hrefines _ _ _
          _ =>
          expr_refines_where_from source target callee_vars
            [(ret, return_to); (exn, exn_to)] statics'' hp'' Hrefines)
       (fun id params my_closure ret exn metadata target source vs callee
          callee_vars return_to exn_to statics'' hp'' slots Hrefines _ _ _
          _ =>
          expr_refines_where_from source target callee_vars
            [(ret, return_to); (exn, exn_to)] statics'' hp'' Hrefines)
       e vars conts conts' statics statics' _ Hok Hok' Hrel Hrel' Hstatics
       Hstatics').
Qed.

(* The case of one static environment. *)
Lemma related_envs_refines (e : expr) (vars : env) (conts conts' : cont_env)
    (statics : static_env) (hp : heap) :
  ok statics ->
  cont_env_related (handler_refines_where ok) conts conts' ->
  cont_env_related
    (fun params kinds def_vars target source =>
       handler_refines_where ok params kinds def_vars source target)
    conts' conts ->
  refines (State e vars conts statics hp) (State e vars conts' statics hp).
Proof.
  intros Hok Hrel Hrel'.
  apply related_states_refines; try assumption; apply statics_related_refl.
Qed.

(* A [Let_cont] refines the same with a handler replaced by one that
   refines it where it runs. *)
Lemma let_cont_handler_refines_where (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) :
  ok statics ->
  handler_refines_where ok params (hm_params_kinds metadata) vars handler
    handler' ->
  refines
    (State
       (Let_cont_non_recursive k (Handler params metadata handler) body)
       vars conts statics hp)
    (State
       (Let_cont_non_recursive k (Handler params metadata handler') body)
       vars conts statics hp).
Proof.
  intros Hok Hhandler. apply let_cont_non_recursive_refines_step.
  apply related_envs_refines; [ exact Hok | | ];
    (apply Forall2_cons; [ | apply cont_env_related_refl ]);
    (split; [ reflexivity | ]);
    (apply Closure_related;
      [ apply Handler_related; exact Hhandler | apply cont_env_related_refl
      | apply Forall2_nil ]).
Qed.

(* The same for recursive handlers, any number of them replaced. *)
Lemma let_cont_recursive_handlers_refine_where
    (handlers handlers' : list (continuation * continuation_handler))
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
      (hp : heap) :
  ok statics ->
  handlers_bind_statics handlers = handlers_bind_statics handlers' ->
  handlers_related (handler_refines_where ok) vars handlers handlers' ->
  handlers_related
    (fun params kinds def_vars target source =>
       handler_refines_where ok params kinds def_vars source target)
    vars handlers' handlers ->
  refines (State (Let_cont_recursive handlers body) vars conts statics hp)
    (State (Let_cont_recursive handlers' body) vars conts statics hp).
Proof.
  intros Hok Hbinds Hhandlers Hhandlers'.
  apply let_cont_recursive_refines_step; [ exact Hbinds | intros _ ].
  apply related_envs_refines; [ exact Hok | | ];
    (apply Forall2_app; [ | apply cont_env_related_refl ]);
    (apply rec_closures_related; [ assumption | apply cont_env_related_refl ]).
Qed.

(* A [Let_static] refines the same with the body of a piece of code
   replaced by one that refines it where it runs. *)
Lemma let_static_code_refines_where (before : list static_binding)
    (id : code_id) (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (code_body code_body' : expr) (after : list static_binding) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap) :
  ok statics ->
  binds_statics code_body = binds_statics code_body' ->
  code_refines_where ok params my_closure metadata code_body code_body' ->
  refines
    (State
       (Let_static
          (before
           ++ Bind_code id
                (Code params my_closure ret exn metadata code_body)
           :: after)
          body)
       vars conts statics hp)
    (State
       (Let_static
          (before
           ++ Bind_code id
                (Code params my_closure ret exn metadata code_body')
           :: after)
          body)
       vars conts statics hp).
Proof.
  intros Hok Hbinds Hcode.
  exact
    (let_static_code_refines_from statics_ok statics_ok_step
       (handler_refines_where ok)
       (fun params kinds def_vars target source =>
          handler_refines_where ok params kinds def_vars source target)
       (fun _ params my_closure _ _ metadata =>
          code_refines_where ok params my_closure metadata)
       (fun _ params my_closure _ _ metadata target source =>
          code_refines_where ok params my_closure metadata source target)
       handler_refines_where_from
       (fun params kinds def_vars target source =>
          handler_refines_where_from params kinds def_vars source target)
       (fun _ params my_closure ret exn metadata source target vs callee
          callee_vars return_to exn_to statics' hp' _ Hrefines Hbind Hkinds
          Hvalue _ =>
          code_refines_where_from params my_closure metadata source target
            vs callee callee_vars [(ret, return_to); (exn, exn_to)]
            statics' hp' Hrefines Hbind Hkinds Hvalue)
       (fun _ params my_closure ret exn metadata target source vs callee
          callee_vars return_to exn_to statics' hp' _ Hrefines Hbind Hkinds
          Hvalue _ =>
          code_refines_where_from params my_closure metadata source target
            vs callee callee_vars [(ret, return_to); (exn, exn_to)]
            statics' hp' Hrefines Hbind Hkinds Hvalue)
       before id params my_closure ret exn metadata code_body code_body'
       after body vars conts statics hp Hok Hok Hbinds Hcode Hcode).
Qed.

End Where.

Lemma handler_refines_at_where (params : list variable)
    (kinds : list flambda_kind) (def_vars : env) (source target : expr) :
  handler_refines_at params kinds def_vars source target ->
  handler_refines_where always params kinds def_vars source target.
Proof. intros H vs handler_vars conts statics hp _. apply H. Qed.

Lemma let_cont_handler_refines_at (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) :
  handler_refines_at params (hm_params_kinds metadata) vars handler
    handler' ->
  refines
    (State
       (Let_cont_non_recursive k (Handler params metadata handler) body)
       vars conts statics hp)
    (State
       (Let_cont_non_recursive k (Handler params metadata handler') body)
       vars conts statics hp).
Proof.
  intros Hhandler.
  apply (let_cont_handler_refines_where always always_extend);
    [ exact I | apply handler_refines_at_where, Hhandler ].
Qed.

Lemma let_cont_recursive_handler_refines_at
    (before : list (continuation * continuation_handler)) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' : expr)
    (after : list (continuation * continuation_handler)) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap) :
  binds_statics handler = binds_statics handler' ->
  handler_refines_at params (hm_params_kinds metadata) vars handler
    handler' ->
  refines
    (State
       (Let_cont_recursive
          (before ++ (k, Handler params metadata handler) :: after) body)
       vars conts statics hp)
    (State
       (Let_cont_recursive
          (before ++ (k, Handler params metadata handler') :: after) body)
       vars conts statics hp).
Proof.
  intros Hbinds Hhandler.
  apply (let_cont_recursive_handlers_refine_where always always_extend);
    [ exact I
    | unfold handlers_bind_statics; rewrite !existsb_app; simpl;
      rewrite Hbinds; reflexivity
    | | ];
    (apply Forall2_app; [ apply handlers_related_refl | ]);
    (apply Forall2_cons; [ | apply handlers_related_refl ]);
    (split; [ reflexivity | ]); apply Handler_related;
    apply handler_refines_at_where, Hhandler.
Qed.

(* A [Let_static] refines the same with the body of a piece of code
   replaced by one that refines it. *)
Lemma let_static_code_refines (before : list static_binding) (id : code_id)
    (params : list variable) (my_closure : variable)
    (ret exn : continuation) (metadata : code_metadata)
    (code_body code_body' : expr) (after : list static_binding) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap) :
  binds_statics code_body = binds_statics code_body' ->
  expr_refines code_body code_body' ->
  refines
    (State
       (Let_static
          (before
           ++ Bind_code id
                (Code params my_closure ret exn metadata code_body)
           :: after)
          body)
       vars conts statics hp)
    (State
       (Let_static
          (before
           ++ Bind_code id
                (Code params my_closure ret exn metadata code_body')
           :: after)
          body)
       vars conts statics hp).
Proof.
  intros Hbinds Hcode.
  apply (let_static_code_refines_where always always_extend);
    [ exact I | exact Hbinds | ].
  intros vs callee callee_vars conts' statics' hp' _ _ _ _. apply Hcode.
Qed.

(* ======================================================================== *)
(* Contexts                                                                 *)
(* ======================================================================== *)

(* For a context whose hole is not in code or in a recursive handler. *)
Lemma not_many (P : Prop) : false = true -> P.
Proof. discriminate. Qed.

Lemma plug_binds_statics (c : context) (e e' : expr) :
  binds_statics e = binds_statics e' ->
  binds_statics (plug c e) = binds_statics (plug c e').
Proof.
  intros Hbinds.
  induction c
    as [ | x kind d c IH | bindings c IH
       | before id params my_closure ret exn metadata c IH
           after body
       | k h c IH | k params metadata c IH body | handlers c IH
       | before k params metadata c IH after body ];
    simpl; try congruence.
  - rewrite !existsb_app. reflexivity.
  - rewrite !existsb_app. simpl. rewrite IH. reflexivity.
Qed.

(* By induction on the context. A hole in a body: one step, then the
   induction. A hole in a handler: the handler refines where it runs, as
   it does everywhere. A hole in code: the body refines. *)
Theorem plug_refines : plug_refines_statement.
Proof.
  intros c source target Hbinds Hrefines.
  induction c
    as [ | x kind d c IH | bindings c IH
       | before id params my_closure ret exn metadata c IH
           after body
       | k h c IH | k params metadata c IH body | handlers c IH
       | before k params metadata c IH after body ];
    simpl in Hbinds |- *; [ exact Hrefines | | | | | | | ];
    intros vars conts statics hp.
  - apply let_refines_step; [ reflexivity | intros v _ _ ]. apply IH, Hbinds.
  - apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [statics' | ];
      [ apply IH, Hbinds | exact I ].
  - apply let_static_code_refines;
      [ apply plug_binds_statics, Hbinds, eq_refl
      | apply IH; intros _; apply Hbinds, eq_refl ].
  - apply let_cont_non_recursive_refines_step, IH, Hbinds.
  - apply let_cont_handler_refines_at.
    intros vs handler_vars conts' statics' hp' _ _. apply IH, Hbinds.
  - apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
    apply IH, Hbinds.
  - apply let_cont_recursive_handler_refines_at;
      [ apply plug_binds_statics, Hbinds, eq_refl | ].
    intros vs handler_vars conts' statics' hp' _ _. apply IH.
    intros _. apply Hbinds, eq_refl.
Qed.

Theorem plug_equivalent : plug_equivalent_statement.
Proof.
  intros c e e' Hbinds Hequiv vars conts statics hp.
  apply refines_antisymmetric; apply plug_refines;
    try (apply expr_equivalent_refines; exact Hequiv);
    try exact Hbinds.
  - intros Hmany. symmetry. apply Hbinds, Hmany.
  - intros vars' conts' statics' hp'. apply equivalent_refines, equivalent_sym.
    apply Hequiv.
Qed.
