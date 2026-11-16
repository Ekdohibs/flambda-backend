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

(* Proofs for FlambdaUnusedHeadStaticsSpec.v. This file needs no review.

   The term without the bindings is the source and the term with them
   the target: where the first has a step, the second has the same one.
   Two things make it so.

   - The machine does not lose a step when its static environment binds
     more ([step_more]): a lookup that succeeded succeeds, and no step
     but that of a [Let_static] looks at what is not bound.
   - The two runs are related by the erasure of
     FlambdaUnusedStaticsProofs.v, read the other way, with what it
     does not say: the static environment of the run with the bindings
     has everything that the other has, and what it has more is of the
     names removed ([only_more]). While the groups at the head run, the
     run with the bindings binds a name removed where the other binds
     nothing, and that is defined by the checks of
     [head_statics_removable] ([head_group_evaluates]). Under the head
     the two expressions are the same one, and so are the
     continuations; a [Let_static] there binds no name removed, as
     erasing it changes nothing, so what it binds is fresh in both
     static environments or in neither.

   The two runs may start under continuations, the same in both, in
   which the names removed do not occur: such a continuation is its
   own erasure (FlambdaUnusedStaticsProofs.cont_free_of_erased), and
   the groups at the head do not look at the continuations. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaPrimitiveSpec FlambdaPrimitiveProofs
  FlambdaWellFormedSpec FlambdaWellFormedProofs FlambdaNormaliseSpec
  FlambdaUnusedStaticsSpec FlambdaUnusedStaticsProofs FlambdaLiftSpec
  FlambdaShareProjectionsSpec FlambdaShareProjectionsProofs
  FlambdaImportsSpec FlambdaImportsProofs FlambdaUnusedHeadStaticsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The machine under a static environment that binds more                   *)
(* ======================================================================== *)

Section More.

Variables statics' statics : static_env.

Hypothesis Hextend : statics_extend statics' statics.

Lemma symbols_extend (s : symbol) (allocated : statically_allocated_value) :
  lookup s (static_values statics') = Some allocated ->
  lookup s (static_values statics) = Some allocated.
Proof. exact (proj2 Hextend s allocated). Qed.

Lemma project_function_slot_more (move_from move_to : function_slot)
    (v w : value) :
  project_function_slot statics' move_from move_to v = Some w ->
  project_function_slot statics move_from move_to v = Some w.
Proof.
  destruct v; simpl; try exact (fun H => H).
  match goal with
  | |- context [lookup ?s (static_values statics')] =>
    destruct (lookup s (static_values statics')) eqn:Hlookup;
      [ rewrite (symbols_extend _ _ Hlookup); exact (fun H => H)
      | discriminate ]
  end.
Qed.

Lemma eval_named_more (vars : env) (d : named) (v : value) :
  eval_named statics' vars d = Some v -> eval_named statics vars d = Some v.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure].
  - exact (fun H => H).
  - simpl. destruct (eval_simples vars args) as [vs | ]; [ | discriminate ].
    apply eval_prim_statics_extend.
    intros s allocated _. apply symbols_extend.
  - exact (fun H => H).
  - exact (fun H => H).
  - exact (eval_projection_statics_extend statics' statics vars
             (Project_value_slot slot closure) v Hextend eq_refl).
  - exact (eval_projection_statics_extend statics' statics vars
             (Project_value_slot_from pfrom slot closure) v Hextend eq_refl).
  - simpl. destruct (eval_simple vars closure) as [c | ]; [ | discriminate ].
    apply project_function_slot_more.
Qed.

Lemma tuple_fields_from_more (size : nat) (tuple : value) (count : nat) :
  ∀ (field : nat) (fields : list value),
    tuple_fields_from statics' size tuple field count = Some fields ->
    tuple_fields_from statics size tuple field count = Some fields.
Proof.
  induction count as [ | count IH]; intros field fields; simpl;
    [ exact (fun H => H) | ].
  destruct (eval_prim statics' _ [tuple]) as [v | ] eqn:Hv; [ | discriminate ].
  rewrite (eval_prim_statics_extend statics' statics _ _ v
             (fun s allocated _ => symbols_extend s allocated) Hv).
  destruct (tuple_fields_from statics' size tuple (S field) count)
    as [vs | ] eqn:Hvs; [ | discriminate ].
  rewrite (IH _ _ Hvs). exact (fun H => H).
Qed.

Lemma arguments_for_code_more (c : code) (vs : list value)
    (args_arity : arity) (result : list value * arity) :
  arguments_for_code statics' c vs args_arity = Some result ->
  arguments_for_code statics c vs args_arity = Some result.
Proof.
  unfold arguments_for_code.
  destruct (code_is_tupled c); [ | exact (fun H => H) ].
  unfold detuple.
  destruct vs as [ | tuple rest]; [ exact (fun H => H) | ].
  destruct args_arity as [ | first rest_arity]; [ exact (fun H => H) | ].
  destruct (negb _); [ exact (fun H => H) | ].
  unfold tuple_fields.
  destruct (tuple_fields_from statics' _ tuple 0 _) as [fields | ] eqn:Hfields;
    [ | discriminate ].
  rewrite (tuple_fields_from_more _ _ _ _ _ Hfields). exact (fun H => H).
Qed.

Lemma shape_of_closure_call_more (closure : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity) (shape : call_shape) :
  shape_of_closure_call statics' closure kind vs args_arity return_arity
  = Some shape ->
  shape_of_closure_call statics closure kind vs args_arity return_arity
  = Some shape.
Proof.
  unfold shape_of_closure_call.
  destruct (closure_of statics' closure) as [[called slots] | ] eqn:Hclosure;
    [ | discriminate ].
  rewrite (closure_of_statics_extend statics' statics closure
             (fun s allocated _ => symbols_extend s allocated) _ Hclosure).
  destruct (lookup called (static_code statics')) as [c | ] eqn:Hcode.
  - rewrite (proj1 Hextend called c Hcode).
    rewrite (known_arity_full_code statics' statics called args_arity)
      by (rewrite Hcode; exact (proj1 Hextend called c Hcode)).
    destruct kind as [named | | code_ids];
      [ destruct (Pos.eqb called named) |
      | destruct
          (code_ids_allow code_ids called
           && known_arity_full statics' called args_arity) ];
      try discriminate;
      (destruct (arguments_for_code statics' c vs args_arity)
         as [[vs' args_arity'] | ] eqn:Hargs; [ | discriminate ];
       rewrite (arguments_for_code_more _ _ _ _ Hargs); exact (fun H => H)).
  - destruct kind as [named | | code_ids];
      [ destruct (Pos.eqb called named) |
      | destruct
          (code_ids_allow code_ids called
           && known_arity_full statics' called args_arity) ];
      discriminate.
Qed.

Lemma shape_of_call_more (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity) (shape : call_shape) :
  shape_of_call statics' callee kind vs args_arity return_arity = Some shape ->
  shape_of_call statics callee kind vs args_arity return_arity = Some shape.
Proof.
  unfold shape_of_call.
  destruct kind as [named | | code_ids];
    destruct (partial_application_of callee)
      as [[[closure applied] applied_arity] | ];
    try apply shape_of_closure_call_more.
  destruct args_arity as [ | component args_arity]; [ exact (fun H => H) | ].
  destruct (kinds_eqb _ _); [ | exact (fun H => H) ].
  apply shape_of_closure_call_more.
Qed.

(* What a step under the smaller static environment comes to, and the
   same step under the larger: the same state but for the static
   environment, which the step has not changed; or the same exit. *)
Definition result_more (result' result : step_result) : Prop :=
  match result' with
  | Step (State e vars conts extended' hp rd) =>
    extended' = statics' ∧ result = Step (State e vars conts statics hp rd)
  | Exit k vs hp rd => result = Exit k vs hp rd
  | Undefined => True
  end.

Lemma call_code_more (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (hp : heap) (rd : read_record) :
  result_more
    (call_code closure c vs args_arity return_arity return_to exn_to statics'
       hp rd)
    (call_code closure c vs args_arity return_arity return_to exn_to statics
       hp rd).
Proof.
  unfold call_code.
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct (negb _); [ exact I | ].
  destruct (bind_params _ vs _); simpl; auto.
Qed.

Lemma run_call_more (return_partial' return_partial : value -> step_result)
    (shape' shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (hp : heap) (rd : read_record) :
  (∀ partial : value,
    result_more (return_partial' partial) (return_partial partial)) ->
  (∀ sh : call_shape, shape' = Some sh -> shape = Some sh) ->
  result_more
    (run_call return_partial' shape' return_arity return_to exn_to statics'
       hp rd)
    (run_call return_partial shape return_arity return_to exn_to statics hp rd).
Proof.
  intros Hpartial Hshape.
  destruct shape' as [sh | ]; [ | exact I ].
  rewrite (Hshape sh eq_refl).
  destruct sh; simpl;
    [ apply call_code_more | apply call_code_more | apply Hpartial ].
Qed.

Fixpoint apply_cont_value_more (c : cont_value) :
  ∀ (vs : list value) (hp : heap) (rd : read_record),
    result_more (apply_cont_value c vs statics' hp rd)
      (apply_cont_value c vs statics hp rd).
Proof.
  destruct c
    as [h def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    intros vs hp rd; simpl.
  - destruct h as [params metadata handler].
    destruct (kinds_eqb _ _); [ | exact I ].
    destruct (bind_params params vs def_vars); simpl; auto.
  - reflexivity.
  - apply apply_cont_value_more.
  - destruct vs as [ | v [ | v' vs]]; try exact I.
    apply (run_call_more _ _
             (shape_of_call statics' v Indirect_unknown_arity remaining
                remaining_arity return_arity)
             (shape_of_call statics v Indirect_unknown_arity remaining
                remaining_arity return_arity));
      [ intros partial; apply apply_cont_value_more
      | intros sh; apply shape_of_call_more ].
Qed.

Lemma call_function_more (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (hp : heap) (rd : read_record) :
  result_more
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics' hp rd)
    (call_function callee kind vs args_arity return_arity return_to exn_to
       statics hp rd).
Proof.
  unfold call_function.
  apply run_call_more;
    [ intros partial; apply apply_cont_value_more
    | intros sh; apply shape_of_call_more ].
Qed.

(* The run of a named expression by the machine: the same value and the
   same heap after. *)
Lemma exec_named_more (hp hp' : heap) (rd rd' : read_record) (vars : env)
    (d : named) (v : value) :
  exec_named statics' hp rd vars d = Some (v, hp', rd') ->
  exec_named statics hp rd vars d = Some (v, hp', rd').
Proof.
  apply exec_named_extend.
  - intros _. apply eval_named_more.
  - intros p args vs s allocated _ _ _. apply symbols_extend.
Qed.

(* A step that is not that of a [Let_static]. *)
Lemma step_more (e : expr) (vars : env) (conts : cont_env) (hp : heap)
    (rd : read_record) :
  (∀ (bindings : list static_binding) (body : expr),
    e <> Let_static bindings body) ->
  result_more (step (State e vars conts statics' hp rd))
    (step (State e vars conts statics hp rd)).
Proof.
  intros Hnot.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step.
  - destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ] eqn:Hd;
    [ | exact I ].
    rewrite (exec_named_more hp hp' rd rd' vars d v Hd).
    destruct (kind_eqb _ _); simpl; auto.
  - exfalso. exact (Hnot bindings body eq_refl).
  - simpl. auto.
  - destruct (handlers_bind_statics handlers); simpl; auto.
  - destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    apply call_function_more.
  - destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
    unfold apply_continuation. apply apply_cont_value_more.
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ]; simpl; auto.
  - exact I.
Qed.

End More.

(* ======================================================================== *)
(* Lists of names and of bindings                                           *)
(* ======================================================================== *)

Lemma distinct_app (names names' : list positive) :
  distinct (names ++ names') = true ->
  distinct names = true
  ∧ distinct names' = true
  ∧ ∀ name : positive, among name names' = true -> among name names = false.
Proof.
  induction names as [ | first names IH]; cbn [app distinct];
    intros Hdistinct.
  - auto.
  - apply andb_true_iff in Hdistinct as [Hfirst Hrest].
    apply negb_true_iff in Hfirst. rewrite among_app in Hfirst.
    apply orb_false_iff in Hfirst as [Hnames Hnames'].
    destruct (IH Hrest) as (Hdistinct & Hdistinct' & Hapart).
    split; [ rewrite Hnames, Hdistinct; reflexivity | ].
    split; [ exact Hdistinct' | ].
    intros name Hname. rewrite among_cons, (Hapart name Hname), orb_false_r.
    destruct (Pos.eqb name first) eqn:Heq; [ | reflexivity ].
    apply Pos.eqb_eq in Heq. subst name. rewrite Hname in Hnames'.
    discriminate.
Qed.

(* The symbols, and the code ids, that a list of bindings binds, written
   so that a binding at the head of the list is added or not:
   [flat_map] of FlambdaWellFormedSpec.symbols_of_binding and of
   [code_ids_of_binding]. *)
Fixpoint symbols_bound_by (bindings : list static_binding) : list symbol :=
  match bindings with
  | [] => []
  | Bind_symbol s _ :: bindings => s :: symbols_bound_by bindings
  | (Bind_code _ _ | Bind_deleted_code _) :: bindings =>
    symbols_bound_by bindings
  end.

Fixpoint code_bound_by (bindings : list static_binding) : list code_id :=
  match bindings with
  | [] => []
  | Bind_code id _ :: bindings => id :: code_bound_by bindings
  | (Bind_symbol _ _ | Bind_deleted_code _) :: bindings =>
    code_bound_by bindings
  end.

Lemma symbols_bound_by_flat_map (bindings : list static_binding) :
  flat_map symbols_of_binding bindings = symbols_bound_by bindings.
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH]; simpl;
    [ reflexivity | exact IH | exact IH | rewrite IH; reflexivity ].
Qed.

Lemma code_bound_by_flat_map (bindings : list static_binding) :
  flat_map code_ids_of_binding bindings = code_bound_by bindings.
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH]; simpl;
    [ reflexivity | rewrite IH; reflexivity | exact IH | exact IH ].
Qed.

Lemma symbols_bound_by_app (bindings bindings' : list static_binding) :
  symbols_bound_by (bindings ++ bindings')
  = symbols_bound_by bindings ++ symbols_bound_by bindings'.
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH]; simpl;
    [ reflexivity | exact IH | exact IH | rewrite IH; reflexivity ].
Qed.

Lemma code_bound_by_app (bindings bindings' : list static_binding) :
  code_bound_by (bindings ++ bindings')
  = code_bound_by bindings ++ code_bound_by bindings'.
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH]; simpl;
    [ reflexivity | rewrite IH; reflexivity | exact IH | exact IH ].
Qed.

Lemma keep_map_length {A : Type} (keep : A -> bool) (f : A -> A)
    (l : list A) :
  length (keep_map keep f l) <= length l.
Proof.
  induction l as [ | a l IH]; simpl; [ lia | ].
  destruct (keep a); simpl; lia.
Qed.

(* A list that is its own [keep_map] keeps all its elements. *)
Lemma keep_map_same {A : Type} (keep : A -> bool) (f : A -> A) (l : list A) :
  keep_map keep f l = l -> forallb keep l = true.
Proof.
  induction l as [ | a l IH]; simpl; [ reflexivity | ].
  destruct (keep a) eqn:Hkeep.
  - intros [= _ Hrest]. exact (IH Hrest).
  - intros Hsame. pose proof (keep_map_length keep f l) as Hlength.
    rewrite Hsame in Hlength. simpl in Hlength. lia.
Qed.

(* ======================================================================== *)
(* Constants that have a value whatever the environment                     *)
(* ======================================================================== *)

Lemma plain_simple_evaluates (vars : env) (s : simple) :
  plain_simple s = true -> ∃ v : value, eval_simple vars s = Some v.
Proof. destruct s as [x | sym | c]; [ discriminate | | ]; simpl; eauto. Qed.

Lemma plain_simples_evaluate (vars : env) (args : list simple) :
  forallb plain_simple args = true ->
  ∃ vs : list value, eval_simples vars args = Some vs.
Proof.
  induction args as [ | s args IH]; simpl; [ eauto | ].
  intros Hplain. apply andb_true_iff in Hplain as [Hs Hargs].
  destruct (plain_simple_evaluates vars s Hs) as [v ->].
  destruct (IH Hargs) as [vs ->]. simpl. eauto.
Qed.

Lemma plain_value_slots_evaluate (vars : env)
    (slots : list (value_slot * simple)) :
  value_slots_plain slots = true ->
  ∃ vs : list (value_slot * value), eval_value_slots vars slots = Some vs.
Proof.
  unfold value_slots_plain.
  induction slots as [ | [slot s] slots IH]; simpl; [ eauto | ].
  intros Hplain. apply andb_true_iff in Hplain as [Hs Hslots].
  destruct (plain_simple_evaluates vars s Hs) as [v ->].
  destruct (IH Hslots) as [vs ->]. simpl. eauto.
Qed.

Lemma static_const_total_evaluates (vars : env) (const : static_const) :
  static_const_total const = true ->
  ∃ allocated : statically_allocated_value,
    eval_static_const vars const = Some allocated.
Proof.
  destruct const as [code slots | slot symbols decls slots | tag fields
    | contents | bytes | elements]; simpl.
  - intros Hplain.
    destruct (plain_value_slots_evaluate vars slots Hplain) as [vs ->].
    simpl. eauto.
  - destruct (lookup slot decls); [ | discriminate ]. intros Hplain.
    destruct (plain_value_slots_evaluate vars slots Hplain) as [vs ->].
    simpl. eauto.
  - intros Hplain.
    destruct (plain_simples_evaluate vars fields Hplain) as [vs ->].
    simpl. eauto.
  - intros Hplain.
    destruct (plain_simple_evaluates vars contents Hplain) as [v ->].
    simpl. eauto.
  - eauto.
  - intros Hplain.
    destruct (plain_simples_evaluate vars elements Hplain) as [vs ->].
    simpl. eauto.
Qed.

(* ======================================================================== *)
(* The two static environments                                              *)
(* ======================================================================== *)

Section Head.

Variable D : dropped.

(* The second static environment has what the first has, and what it
   has more is of the names removed. *)
Definition only_more (statics' statics : static_env) : Prop :=
  statics_extend statics' statics
  ∧ (∀ id : code_id,
      code_kept D id = true ->
      lookup id (static_code statics') = None ->
      lookup id (static_code statics) = None)
  ∧ (∀ s : symbol,
      symbol_kept D s = true ->
      lookup s (static_values statics') = None ->
      lookup s (static_values statics) = None).

Lemma only_more_add_code (statics' statics : static_env) (id : code_id)
    (c : code) :
  only_more statics' statics ->
  only_more
    (Static_env ((id, c) :: static_code statics') (static_values statics'))
    (Static_env ((id, c) :: static_code statics) (static_values statics)).
Proof.
  intros [[Hcode Hvalues] [Hcode_none Hvalues_none]].
  split; [ split | split ]; cbn [static_code static_values].
  - intros id' c'. rewrite !lookup_cons.
    destruct (Pos.eqb id' id); [ exact (fun H => H) | apply Hcode ].
  - exact Hvalues.
  - intros id' Hkept. rewrite !lookup_cons.
    destruct (Pos.eqb id' id); [ discriminate | exact (Hcode_none id' Hkept) ].
  - exact Hvalues_none.
Qed.

Lemma only_more_add_code_right (statics' statics : static_env)
    (id : code_id) (c : code) :
  only_more statics' statics ->
  code_kept D id = false ->
  lookup id (static_code statics) = None ->
  only_more statics'
    (Static_env ((id, c) :: static_code statics) (static_values statics)).
Proof.
  intros [[Hcode Hvalues] [Hcode_none Hvalues_none]] Hremoved Hfresh.
  split; [ split | split ]; cbn [static_code static_values].
  - intros id' c' Hid'. rewrite lookup_cons.
    destruct (Pos.eqb id' id) eqn:Heq; [ | exact (Hcode id' c' Hid') ].
    apply Pos.eqb_eq in Heq. subst id'.
    rewrite (Hcode id c' Hid') in Hfresh. discriminate.
  - exact Hvalues.
  - intros id' Hkept Hnone. rewrite lookup_cons.
    destruct (Pos.eqb id' id) eqn:Heq; [ | exact (Hcode_none id' Hkept Hnone) ].
    apply Pos.eqb_eq in Heq. subst id'. rewrite Hkept in Hremoved.
    discriminate.
  - exact Hvalues_none.
Qed.

Lemma only_more_add_value (statics' statics : static_env) (s : symbol)
    (allocated : statically_allocated_value) :
  only_more statics' statics ->
  only_more
    (Static_env (static_code statics')
       ((s, allocated) :: static_values statics'))
    (Static_env (static_code statics)
       ((s, allocated) :: static_values statics)).
Proof.
  intros [[Hcode Hvalues] [Hcode_none Hvalues_none]].
  split; [ split | split ]; cbn [static_code static_values].
  - exact Hcode.
  - intros s' allocated'. rewrite !lookup_cons.
    destruct (Pos.eqb s' s); [ exact (fun H => H) | apply Hvalues ].
  - exact Hcode_none.
  - intros s' Hkept. rewrite !lookup_cons.
    destruct (Pos.eqb s' s); [ discriminate | exact (Hvalues_none s' Hkept) ].
Qed.

Lemma only_more_add_value_right (statics' statics : static_env) (s : symbol)
    (allocated : statically_allocated_value) :
  only_more statics' statics ->
  symbol_kept D s = false ->
  lookup s (static_values statics) = None ->
  only_more statics'
    (Static_env (static_code statics)
       ((s, allocated) :: static_values statics)).
Proof.
  intros [[Hcode Hvalues] [Hcode_none Hvalues_none]] Hremoved Hfresh.
  split; [ split | split ]; cbn [static_code static_values].
  - exact Hcode.
  - intros s' allocated' Hs'. rewrite lookup_cons.
    destruct (Pos.eqb s' s) eqn:Heq; [ | exact (Hvalues s' allocated' Hs') ].
    apply Pos.eqb_eq in Heq. subst s'.
    rewrite (Hvalues s allocated' Hs') in Hfresh. discriminate.
  - exact Hcode_none.
  - intros s' Hkept Hnone. rewrite lookup_cons.
    destruct (Pos.eqb s' s) eqn:Heq;
      [ | exact (Hvalues_none s' Hkept Hnone) ].
    apply Pos.eqb_eq in Heq. subst s'. rewrite Hkept in Hremoved.
    discriminate.
Qed.

(* A group that binds none of the names removed, under the head: where
   it has a result in the smaller static environment it has one in the
   larger. *)
Lemma eval_static_bindings_more (vars : env)
    (bindings : list static_binding) :
  ∀ statics' statics extended' : static_env,
    forallb (binding_kept D) bindings = true ->
    only_more statics' statics ->
    eval_static_bindings vars bindings statics' = Some extended' ->
    ∃ extended : static_env,
      eval_static_bindings vars bindings statics = Some extended
      ∧ only_more extended' extended.
Proof.
  induction bindings as [ | binding bindings IH];
    intros statics' statics extended' Hkept Hmore; simpl.
  - intros [= <-]. exists statics. auto.
  - simpl in Hkept. apply andb_true_iff in Hkept as [Hkept1 Hkept].
    destruct binding as [id c | id | s const]; simpl in Hkept1.
    + destruct (lookup id (static_code statics')) eqn:Hnone';
        [ discriminate | ].
      rewrite (proj1 (proj2 Hmore) id Hkept1 Hnone').
      apply IH; [ exact Hkept | apply only_more_add_code; exact Hmore ].
    + apply IH; assumption.
    + destruct (lookup s (static_values statics')) eqn:Hnone';
        [ discriminate | ].
      rewrite (proj2 (proj2 Hmore) s Hkept1 Hnone').
      destruct (eval_static_const vars const) as [allocated | ]; simpl;
        [ | discriminate ].
      apply IH; [ exact Hkept | apply only_more_add_value; exact Hmore ].
Qed.

(* A group of the head: where the group erased has a result in the
   smaller static environment, the group has one in the larger. What it
   binds was not bound; a name it does not bind, and that was not bound,
   is not bound after it. *)
Lemma head_group_evaluates (bindings : list static_binding) :
  ∀ statics' statics extended' : static_env,
    forallb (kept_code_binds_only_kept D) bindings = true ->
    forallb (removed_binding_total D) bindings = true ->
    distinct (symbols_bound_by bindings) = true ->
    distinct (code_bound_by bindings) = true ->
    (∀ s : symbol,
      among s (symbols_bound_by bindings) = true ->
      lookup s (static_values statics) = None) ->
    (∀ id : code_id,
      among id (code_bound_by bindings) = true ->
      lookup id (static_code statics) = None) ->
    only_more statics' statics ->
    eval_static_bindings []
      (keep_map (binding_kept D) (erase_binding D) bindings) statics'
    = Some extended' ->
    ∃ extended : static_env,
      eval_static_bindings [] bindings statics = Some extended
      ∧ only_more extended' extended
      ∧ (∀ s : symbol,
          among s (symbols_bound_by bindings) = false ->
          lookup s (static_values statics) = None ->
          lookup s (static_values extended) = None)
      ∧ (∀ id : code_id,
          among id (code_bound_by bindings) = false ->
          lookup id (static_code statics) = None ->
          lookup id (static_code extended) = None).
Proof.
  induction bindings as [ | binding bindings IH];
    intros statics' statics extended' Hcode Htotal Hsyms Hcodes Hfresh_s
      Hfresh_c Hmore.
  - simpl. intros [= <-]. exists statics. auto.
  - simpl in Hcode, Htotal.
    apply andb_true_iff in Hcode as [Hcode1 Hcode].
    apply andb_true_iff in Htotal as [Htotal1 Htotal].
    destruct binding as [id c | id | s const].
    + simpl in Hcodes.
      apply andb_true_iff in Hcodes as [Hid Hcodes].
      apply negb_true_iff in Hid.
      assert (Hnone : lookup id (static_code statics) = None).
      { apply Hfresh_c. cbn [code_bound_by].
        rewrite among_cons, Pos.eqb_refl. reflexivity. }
      assert (Hrest :
        ∀ id0 : code_id,
          among id0 (code_bound_by bindings) = true ->
          lookup id0 ((id, c) :: static_code statics) = None).
      { intros id0 Hid0. rewrite lookup_cons.
        destruct (Pos.eqb id0 id) eqn:Heq.
        - apply Pos.eqb_eq in Heq. subst id0. rewrite Hid0 in Hid.
          discriminate.
        - apply Hfresh_c. cbn [code_bound_by]. rewrite among_cons, Heq.
          exact Hid0. }
      simpl. rewrite Hnone.
      destruct (code_kept D id) eqn:Hkept; simpl.
      * simpl in Hcode1. rewrite Hkept in Hcode1. simpl in Hcode1.
        rewrite (erase_code_binds_only_kept D c Hcode1).
        destruct (lookup id (static_code statics')) eqn:Hnone';
          [ discriminate | ].
        intros Hextended'.
        destruct (IH
                    (Static_env ((id, c) :: static_code statics')
                       (static_values statics'))
                    (Static_env ((id, c) :: static_code statics)
                       (static_values statics))
                    extended' Hcode Htotal Hsyms Hcodes Hfresh_s Hrest
                    (only_more_add_code _ _ id c Hmore) Hextended')
          as (extended & Heval & Hmore' & Hs & Hc).
        exists extended.
        split; [ exact Heval | split; [ exact Hmore' | split ] ];
          [ exact Hs | ].
        intros id0 Hid0 Hnone0.
        apply orb_false_iff in Hid0 as [Hneq Hid0].
        apply Hc; [ exact Hid0 | ]. cbn [static_code].
        rewrite lookup_cons, Hneq. exact Hnone0.
      * intros Hextended'.
        destruct (IH statics'
                    (Static_env ((id, c) :: static_code statics)
                       (static_values statics))
                    extended' Hcode Htotal Hsyms Hcodes Hfresh_s Hrest
                    (only_more_add_code_right _ _ id c Hmore Hkept Hnone)
                    Hextended')
          as (extended & Heval & Hmore' & Hs & Hc).
        exists extended.
        split; [ exact Heval | split; [ exact Hmore' | split ] ];
          [ exact Hs | ].
        intros id0 Hid0 Hnone0.
        apply orb_false_iff in Hid0 as [Hneq Hid0].
        apply Hc; [ exact Hid0 | ]. cbn [static_code].
        rewrite lookup_cons, Hneq. exact Hnone0.
    + simpl. intros Hextended'.
      exact (IH _ _ _ Hcode Htotal Hsyms Hcodes Hfresh_s Hfresh_c Hmore
               Hextended').
    + simpl in Hsyms.
      apply andb_true_iff in Hsyms as [Hs0 Hsyms].
      apply negb_true_iff in Hs0.
      assert (Hnone : lookup s (static_values statics) = None).
      { apply Hfresh_s. cbn [symbols_bound_by].
        rewrite among_cons, Pos.eqb_refl. reflexivity. }
      assert (Hrest :
        ∀ (allocated : statically_allocated_value) (s0 : symbol),
          among s0 (symbols_bound_by bindings) = true ->
          lookup s0 ((s, allocated) :: static_values statics) = None).
      { intros allocated s0 Hs1. rewrite lookup_cons.
        destruct (Pos.eqb s0 s) eqn:Heq.
        - apply Pos.eqb_eq in Heq. subst s0. rewrite Hs1 in Hs0.
          discriminate.
        - apply Hfresh_s. cbn [symbols_bound_by]. rewrite among_cons, Heq.
          exact Hs1. }
      simpl. rewrite Hnone.
      destruct (symbol_kept D s) eqn:Hkept; simpl.
      * destruct (lookup s (static_values statics')) eqn:Hnone';
          [ discriminate | ].
        destruct (eval_static_const [] const) as [allocated | ]; simpl;
          [ | discriminate ].
        intros Hextended'.
        destruct (IH
                    (Static_env (static_code statics')
                       ((s, allocated) :: static_values statics'))
                    (Static_env (static_code statics)
                       ((s, allocated) :: static_values statics))
                    extended' Hcode Htotal Hsyms Hcodes (Hrest allocated)
                    Hfresh_c (only_more_add_value _ _ s allocated Hmore)
                    Hextended')
          as (extended & Heval & Hmore' & Hs & Hc).
        exists extended.
        split; [ exact Heval | split; [ exact Hmore' | split ] ];
          [ | exact Hc ].
        intros s0 Hs1 Hnone0.
        apply orb_false_iff in Hs1 as [Hneq Hs1].
        apply Hs; [ exact Hs1 | ]. cbn [static_values].
        rewrite lookup_cons, Hneq. exact Hnone0.
      * simpl in Htotal1. rewrite Hkept in Htotal1. simpl in Htotal1.
        destruct (static_const_total_evaluates [] const Htotal1)
          as [allocated Hconst].
        rewrite Hconst. simpl. intros Hextended'.
        destruct (IH statics'
                    (Static_env (static_code statics)
                       ((s, allocated) :: static_values statics))
                    extended' Hcode Htotal Hsyms Hcodes (Hrest allocated)
                    Hfresh_c
                    (only_more_add_value_right _ _ s allocated Hmore Hkept
                       Hnone)
                    Hextended')
          as (extended & Heval & Hmore' & Hs & Hc).
        exists extended.
        split; [ exact Heval | split; [ exact Hmore' | split ] ];
          [ | exact Hc ].
        intros s0 Hs1 Hnone0.
        apply orb_false_iff in Hs1 as [Hneq Hs1].
        apply Hs; [ exact Hs1 | ]. cbn [static_values].
        rewrite lookup_cons, Hneq. exact Hnone0.
Qed.

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

(* What the checks say of a term that is still in the groups of the
   head, with the static environment of the run that has the bindings:
   nothing that the groups to come bind is bound. *)
Definition head_ok (e : expr) (statics : static_env) : Prop :=
  binds_only_kept D (under_head e) = true
  ∧ forallb (kept_code_binds_only_kept D) (head_bindings e) = true
  ∧ forallb (removed_binding_total D) (head_bindings e) = true
  ∧ distinct (symbols_bound_by (head_bindings e)) = true
  ∧ distinct (code_bound_by (head_bindings e)) = true
  ∧ (∀ s : symbol,
      among s (symbols_bound_by (head_bindings e)) = true ->
      lookup s (static_values statics) = None)
  ∧ (∀ id : code_id,
      among id (code_bound_by (head_bindings e)) = true ->
      lookup id (static_code statics) = None).

(* The state of the run without the bindings, and that of the run with
   them: in the head, or under it, where the two differ by their static
   environments only. *)
Inductive related : state -> state -> Prop :=
  | Related_head (e : expr) (conts : cont_env)
      (statics' statics : static_env) (hp : heap) (rd : read_record) :
    expr_avoids D e = true ->
    head_ok e statics ->
    conts_erased D conts conts ->
    statics_erased D statics statics' ->
    only_more statics' statics ->
    heap_free_of D hp = true ->
    related (State (erase_expr D e) [] conts statics' hp rd)
      (State e [] conts statics hp rd)
  | Related_under (e : expr) (vars : env) (conts : cont_env)
      (statics' statics : static_env) (hp : heap) (rd : read_record) :
    state_erased D (State e vars conts statics hp rd)
      (State e vars conts statics' hp rd) ->
    only_more statics' statics ->
    heap_free_of D hp = true ->
    related (State e vars conts statics' hp rd)
      (State e vars conts statics hp rd).

Lemma env_avoids_nil : env_avoids D [].
Proof. intros x v. discriminate. Qed.

Lemma state_erased_expr (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) (t : state) :
  state_erased D (State e vars conts statics hp rd) t ->
  state_expr t = erase_expr D e.
Proof. inversion 1. reflexivity. Qed.

Lemma let_static_or_not (e : expr) :
  (∃ (bindings : list static_binding) (body : expr),
    e = Let_static bindings body)
  ∨ (∀ (bindings : list static_binding) (body : expr),
      e <> Let_static bindings body).
Proof.
  destruct e; try (right; intros; discriminate). left. eauto.
Qed.

(* Under the head. *)
Lemma under_step (e : expr) (vars : env) (conts : cont_env)
    (statics' statics : static_env) (hp : heap) (rd : read_record) :
  state_erased D (State e vars conts statics hp rd)
    (State e vars conts statics' hp rd) ->
  only_more statics' statics ->
  heap_free_of D hp = true ->
  result_simulated related (step (State e vars conts statics' hp rd))
    (step (State e vars conts statics hp rd)).
Proof.
  intros Herased Hmore Hfree.
  pose proof (erased_free_step D _ _ (conj Herased Hfree)) as Hsim.
  destruct (let_static_or_not e) as [(bindings & body & ->) | Hnot].
  - pose proof (state_erased_expr _ _ _ _ _ rd_ _ Herased) as Hsame.
    simpl in Hsame. injection Hsame as Hbindings _.
    pose proof (keep_map_same _ _ _ (eq_sym Hbindings)) as Hkept.
    rewrite !step_let_static in Hsim |- *.
    destruct (eval_static_bindings vars bindings statics')
      as [extended' | ] eqn:Heval'; [ | exact I ].
    destruct (eval_static_bindings_more vars bindings _ _ _ Hkept Hmore
                Heval')
      as (extended & Heval & Hmore').
    rewrite Heval in Hsim |- *. simpl in Hsim |- *.
    destruct Hsim as [Hsim Hfree'].
    apply Related_under; assumption.
  - pose proof (step_more statics' statics (proj1 Hmore) e vars conts hp
                  rd Hnot)
      as Hstep.
    destruct (step (State e vars conts statics' hp rd))
      as [[e1 vars1 conts1 extended' hp1 rd1] | k vs hp1 rd1 | ] eqn:Hstep';
      simpl in Hstep.
    + destruct Hstep as [-> Hstep]. rewrite Hstep in Hsim |- *.
      simpl in Hsim |- *. destruct Hsim as [Hsim Hfree'].
      apply Related_under; assumption.
    + rewrite Hstep. simpl. auto.
    + exact I.
Qed.

Lemma related_step (s' s : state) :
  related s' s -> result_simulated related (step s') (step s).
Proof.
  intros [e conts statics' statics hp rd Havoids Hok Hconts Hstatics Hmore
           Hfree
         | e vars conts statics' statics hp rd Herased Hmore Hfree];
    [ | exact (under_step e vars conts statics' statics hp rd Herased Hmore
                 Hfree) ].
  destruct (let_static_or_not e) as [(bindings & body & ->) | Hnot].
  - (* A group of the head. *)
    simpl. rewrite !step_let_static.
    destruct (eval_static_bindings []
                (keep_map (binding_kept D) (erase_binding D) bindings)
                statics')
      as [extended' | ] eqn:Heval'; [ | exact I ].
    destruct Hok as (Hunder & Hcode & Htotal & Hsyms & Hcodes & Hfresh_s
      & Hfresh_c).
    cbn [head_bindings under_head] in Hunder, Hcode, Htotal, Hsyms, Hcodes,
      Hfresh_s, Hfresh_c.
    rewrite forallb_app in Hcode, Htotal.
    apply andb_true_iff in Hcode as [Hcode Hcode_rest].
    apply andb_true_iff in Htotal as [Htotal Htotal_rest].
    rewrite symbols_bound_by_app in Hsyms.
    rewrite code_bound_by_app in Hcodes.
    destruct (distinct_app _ _ Hsyms) as (Hsyms1 & Hsyms_rest & Hsyms_apart).
    destruct (distinct_app _ _ Hcodes)
      as (Hcodes1 & Hcodes_rest & Hcodes_apart).
    assert (Hfresh_s1 :
      ∀ sym : symbol,
        among sym (symbols_bound_by bindings) = true ->
        lookup sym (static_values statics) = None).
    { intros sym Hsym. apply Hfresh_s.
      rewrite symbols_bound_by_app, among_app, Hsym. reflexivity. }
    assert (Hfresh_c1 :
      ∀ id : code_id,
        among id (code_bound_by bindings) = true ->
        lookup id (static_code statics) = None).
    { intros id Hid. apply Hfresh_c.
      rewrite code_bound_by_app, among_app, Hid. reflexivity. }
    destruct (head_group_evaluates bindings statics' statics extended' Hcode
                Htotal Hsyms1 Hcodes1 Hfresh_s1 Hfresh_c1 Hmore Heval')
      as (extended & Heval & Hmore' & Hs & Hc).
    rewrite Heval. simpl.
    simpl in Havoids. apply andb_true_iff in Havoids as [Hbindings Hbody].
    apply Related_head;
      [ exact Hbody | | exact Hconts | | exact Hmore' | exact Hfree ].
    + repeat split; try assumption.
      * intros sym Hsym. apply Hs; [ exact (Hsyms_apart sym Hsym) | ].
        apply Hfresh_s.
        rewrite symbols_bound_by_app, among_app, Hsym, orb_true_r. reflexivity.
      * intros id Hid0. apply Hc; [ exact (Hcodes_apart id Hid0) | ].
        apply Hfresh_c.
        rewrite code_bound_by_app, among_app, Hid0, orb_true_r. reflexivity.
    + pose proof (eval_static_bindings_erased D [] env_avoids_nil bindings
        statics statics' Hbindings Hstatics) as Herased.
      rewrite Heval in Herased.
      destruct Herased as (extended'' & Heval'' & Herased).
      rewrite Heval' in Heval''. injection Heval'' as <-. exact Herased.
  - (* The head is over. *)
    assert (Hunder : under_head e = e)
      by (destruct e; try reflexivity; exfalso; eapply Hnot; reflexivity).
    destruct Hok as [Hkept _]. rewrite Hunder in Hkept.
    pose proof (State_erased D e [] conts conts statics statics' hp rd Havoids
      env_avoids_nil Hconts Hstatics) as Herased.
    rewrite (erase_binds_only_kept D e Hkept) in Herased |- *.
    exact (under_step e [] conts statics' statics hp rd Herased Hmore Hfree).
Qed.

End Head.

(* The names of the head are not bound, as the first state of the
   simulation asks. *)
Lemma head_free_in_symbols (statics : static_env) (e : expr) (s : symbol) :
  head_free_in statics e = true ->
  among s (symbols_bound_by (head_bindings e)) = true ->
  lookup s (static_values statics) = None.
Proof.
  unfold head_free_in. intros Hfree Hs.
  apply andb_true_iff in Hfree as [Hsyms _].
  rewrite symbols_bound_by_flat_map in Hsyms.
  rewrite forallb_forall in Hsyms.
  specialize (Hsyms s (proj1 (among_In s _) Hs)).
  unfold is_bound in Hsyms.
  destruct (lookup s (static_values statics)); [ discriminate | reflexivity ].
Qed.

Lemma head_free_in_code (statics : static_env) (e : expr) (id : code_id) :
  head_free_in statics e = true ->
  among id (code_bound_by (head_bindings e)) = true ->
  lookup id (static_code statics) = None.
Proof.
  unfold head_free_in. intros Hfree Hid.
  apply andb_true_iff in Hfree as [_ Hcodes].
  rewrite code_bound_by_flat_map in Hcodes.
  rewrite forallb_forall in Hcodes.
  specialize (Hcodes id (proj1 (among_In id _) Hid)).
  unfold is_bound in Hcodes.
  destruct (lookup id (static_code statics)); [ discriminate | reflexivity ].
Qed.

Lemma head_free_in_empty (e : expr) :
  head_free_in empty_static_env e = true.
Proof.
  unfold head_free_in. apply andb_true_iff.
  split; apply forallb_forall; intros name _; reflexivity.
Qed.

(* Continuations in which the names removed do not occur are their own
   erasure. *)
Lemma conts_free_of_erased (D : dropped) (conts : cont_env) :
  forallb (fun '(_, c) => cont_free_of D c) conts = true ->
  conts_erased D conts conts.
Proof.
  induction conts as [ | [k c] conts IH]; simpl;
    [ intros _; apply Forall2_nil | ].
  intros Hconts. apply andb_true_iff in Hconts as [Hc Hrest].
  apply Forall2_cons; [ | exact (IH Hrest) ].
  split; [ reflexivity | exact (cont_free_of_erased D c Hc) ].
Qed.

Theorem drop_head_statics_refines_under :
  drop_head_statics_refines_under_statement.
Proof.
  intros D e e' conts statics hp rd. unfold drop_head_statics.
  destruct (head_statics_removable D e) eqn:Hchecks; [ | discriminate ].
  unfold head_statics_removable in Hchecks.
  apply andb_true_iff in Hchecks as [Hchecks Htotal].
  apply andb_true_iff in Hchecks as [Hchecks Hcodes].
  apply andb_true_iff in Hchecks as [Hchecks Hsyms].
  apply andb_true_iff in Hchecks as [Hchecks Hcode].
  apply andb_true_iff in Hchecks as [Havoids Hunder].
  rewrite symbols_bound_by_flat_map in Hsyms.
  rewrite code_bound_by_flat_map in Hcodes.
  unfold drop_statics. rewrite Havoids. intros [= <-] Hfree Hheap Hhead.
  unfold envs_free_of in Hfree.
  apply andb_true_iff in Hfree as [Hfree Hstatics].
  apply andb_true_iff in Hfree as [_ Hconts].
  apply (simulation_refines (related D) (related_step D)).
  apply Related_head.
  - exact Havoids.
  - repeat split; try assumption.
    + intros s. apply head_free_in_symbols, Hhead.
    + intros id. apply head_free_in_code, Hhead.
  - exact (conts_free_of_erased D conts Hconts).
  - exact (statics_free_of_erased D statics Hstatics).
  - split; [ apply statics_extend_refl | split; intros; assumption ].
  - exact Hheap.
Qed.

Theorem drop_head_statics_refines_from :
  drop_head_statics_refines_from_statement.
Proof.
  intros D e e' statics hp rd Hdrop Hfree Hheap Hhead.
  exact (drop_head_statics_refines_under D e e' [] statics hp rd Hdrop Hfree
           Hheap Hhead).
Qed.

Theorem drop_head_statics_program_refines :
  drop_head_statics_program_refines_statement.
Proof.
  intros D e e' Hdrop.
  exact
    (drop_head_statics_refines_from D e e' empty_static_env empty_heap no_reads
           Hdrop eq_refl eq_refl (head_free_in_empty e)).
Qed.

(* ======================================================================== *)
(* Tables bound at the head                                                 *)
(* ======================================================================== *)

(* The binding of a table whose symbol is among the names removed is
   removed. *)
Lemma tables_all_removed (names : list symbol)
    (tables : list (symbol * list simple)) :
  incl (map fst tables) names ->
  keep_map (binding_kept (Dropped names []))
    (erase_binding (Dropped names [])) (map table_binding tables)
  = [].
Proof.
  induction tables as [ | [s elements] tables IH]; intros Hincl;
    [ reflexivity | ].
  cbn [map table_binding fst snd keep_map binding_kept].
  unfold symbol_kept. cbn [dropped_symbols].
  assert (Hamong : among s names = true)
    by (apply among_In, Hincl; left; reflexivity).
  rewrite Hamong. cbn [negb].
  apply IH. intros s' Hs'. apply Hincl. right. exact Hs'.
Qed.

(* A group of no binding runs as what is under it. *)
Lemma empty_group_equivalent (body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  equivalent (State (Let_static [] body) vars conts statics hp rd)
    (State body vars conts statics hp rd).
Proof. apply step_equivalent. reflexivity. Qed.

(* The term without the tables is the removal of their symbols from the
   term with them, up to a group of no binding; the removal is a
   refinement the other way by [drop_head_statics_refines_under], and
   removing those symbols from the term itself, which binds none of
   them and mentions none, is a refinement by
   FlambdaUnusedStaticsProofs.drop_statics_refines. Both are from one
   static environment and under one continuation environment. *)
Theorem add_head_tables_refines_under :
  add_head_tables_refines_under_statement.
Proof.
  intros tables e conts statics hp rd. unfold add_head_tables.
  destruct tables as [ | table tables];
    [ intros _ _ _; apply refines_refl | ].
  cbv zeta.
  set (all := table :: tables).
  set (D := tables_dropped all).
  set (e' := Let_static (map table_binding all) e).
  destruct (tables_plain all && head_statics_removable D e') eqn:Hchecks;
    [ | intros _ _ _; apply refines_refl ].
  intros Hfree Hheap Hfresh.
  apply andb_true_iff in Hchecks as [_ Hchecks].
  assert (Havoids' : expr_avoids D e' = true).
  { unfold head_statics_removable in Hchecks.
    do 5 (apply andb_true_iff in Hchecks as [Hchecks _]). exact Hchecks. }
  assert (Havoids : expr_avoids D e = true).
  { subst e'. simpl in Havoids'.
    apply andb_true_iff in Havoids' as [_ Havoids']. exact Havoids'. }
  assert (Herased : erase_expr D e' = Let_static [] (erase_expr D e)).
  { subst e'.
    change (erase_expr D (Let_static (map table_binding all) e))
      with
        (Let_static
           (keep_map (binding_kept D) (erase_binding D)
              (map table_binding all))
           (erase_expr D e)).
    subst D. unfold tables_dropped.
    rewrite (tables_all_removed (map fst all) all (incl_refl _)).
    reflexivity. }
  assert
    (Hhead : drop_head_statics D e' = Some (Let_static [] (erase_expr D e))).
  { unfold drop_head_statics. rewrite Hchecks. unfold drop_statics.
    rewrite Havoids', Herased. reflexivity. }
  apply (refines_trans _ (State (erase_expr D e) [] conts statics hp rd)).
  - apply (drop_statics_refines D e (erase_expr D e));
      [ unfold drop_statics; rewrite Havoids; reflexivity | exact Hfree
      | exact Hheap ].
  - apply
      (refines_trans _
         (State (Let_static [] (erase_expr D e)) [] conts statics hp rd)).
    + apply equivalent_refines, equivalent_sym, empty_group_equivalent.
    + exact (drop_head_statics_refines_under D e' _ conts statics hp rd Hhead
               Hfree Hheap Hfresh).
Qed.

Theorem add_head_tables_program_refines :
  add_head_tables_program_refines_statement.
Proof.
  intros tables e.
  exact (add_head_tables_refines_under tables e [] empty_static_env
           empty_heap no_reads eq_refl eq_refl (head_free_in_empty _)).
Qed.

(* The static environment of the imports holds nothing in which a name
   removed could occur (FlambdaImportsProofs.imports_statics_free_of). *)
Theorem add_head_tables_refines_imports :
  add_head_tables_refines_imports_statement.
Proof.
  intros imports tables e Hfresh.
  exact (add_head_tables_refines_under tables e [] (imports_statics imports)
           empty_heap no_reads (imports_statics_free_of _ imports) eq_refl
             Hfresh).
Qed.
