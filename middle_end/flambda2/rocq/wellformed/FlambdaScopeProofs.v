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

(* Proofs of the statements of FlambdaScopeSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaBindersSpec FlambdaBindersProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaScopeSpec.
Import ListNotations.

(* ======================================================================== *)
(* The names that environments bind                                         *)
(* ======================================================================== *)

Lemma bind_params_domain (params : list variable) :
  ∀ (vs : list value) (vars vars' : env),
    bind_params params vs vars = Some vars' ->
    map fst vars' = rev params ++ map fst vars.
Proof.
  induction params as [ | x params IH]; intros [ | v vs] vars vars'; simpl;
    try discriminate.
  - intros [= <-]. reflexivity.
  - intros Hbind. rewrite (IH vs _ _ Hbind). simpl.
    rewrite <- app_assoc. reflexivity.
Qed.

(* A parameter, and a variable bound before, are bound after. *)
Lemma among_bind_params (params : list variable) (vs : list value)
    (vars vars' : env) (x : variable) :
  bind_params params vs vars = Some vars' ->
  among x (params ++ map fst vars) = true ->
  among x (map fst vars') = true.
Proof.
  intros Hbind. rewrite (bind_params_domain _ _ _ _ Hbind), !among_app.
  rewrite among_rev. auto.
Qed.

Lemma rec_closures_domain
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  map fst (rec_closures handlers vars conts) = map fst handlers.
Proof.
  unfold rec_closures. rewrite map_map. apply map_ext.
  intros [k h]. reflexivity.
Qed.

(* ======================================================================== *)
(* Continuations in scope                                                   *)
(* ======================================================================== *)

Lemma conts_in_scope_lookup (outer : list continuation) (conts : cont_env)
    (k : continuation) (c : cont_value) :
  conts_in_scope outer conts -> lookup k conts = Some c ->
  cont_in_scope outer c.
Proof.
  unfold conts_in_scope.
  induction conts as [ | [k' c'] conts IH]; simpl; [ discriminate | ].
  intros [Hc Hrest].
  destruct (Pos.eqb k k'); [ intros [= <-]; exact Hc | exact (IH Hrest) ].
Qed.

(* The exception continuation that a call gives its callee. *)
Lemma with_extra_args_in_scope (outer : list continuation) (c : cont_value)
    (extra : list value) :
  cont_in_scope outer c -> cont_in_scope outer (with_extra_args c extra).
Proof. intros Hc. destruct extra; exact Hc. Qed.

(* What a continuation that is bound or is a way out stands for. *)
Lemma resolve_in_scope (outer : list continuation) (k : continuation)
    (conts : cont_env) :
  conts_in_scope outer conts ->
  among k (map fst conts ++ outer) = true ->
  cont_in_scope outer (resolve k conts).
Proof.
  intros Hconts Hk. unfold resolve.
  destruct (lookup k conts) as [c | ] eqn:Hlookup.
  - exact (conts_in_scope_lookup outer conts k c Hconts Hlookup).
  - simpl. rewrite among_app, <- is_bound_among in Hk.
    unfold is_bound in Hk. rewrite Hlookup in Hk. exact Hk.
Qed.

Lemma conts_in_scope_app (outer : list continuation)
    (conts conts' : cont_env) :
  conts_in_scope outer conts -> conts_in_scope outer conts' ->
  conts_in_scope outer (conts ++ conts').
Proof.
  unfold conts_in_scope.
  induction conts as [ | binding conts IH]; simpl; [ auto | ].
  intros [Hc Hrest] Hconts'. split; [ exact Hc | exact (IH Hrest Hconts') ].
Qed.

(* The closures of a recursive group whose handlers are scoped by the
   group. [all] is the whole group, kept apart from the handlers still to
   be gone through. *)
Lemma rec_closures_in_scope (outer : list continuation) (vars : env)
    (conts : cont_env)
    (all handlers : list (continuation * continuation_handler)) :
  let scoped :=
    fun binding : continuation * continuation_handler =>
      handler_scoped (map fst vars)
        (map fst all ++ map fst conts ++ outer) (snd binding)
  in
  Forall scoped all ->
  Forall scoped handlers ->
  conts_in_scope outer conts ->
  conts_in_scope outer
    (map (fun '(k, h) => (k, Closure h vars conts all)) handlers).
Proof.
  intros scoped Hall Hhandlers Hconts. unfold conts_in_scope.
  induction handlers as [ | [k h] handlers IH]; simpl; [ exact I | ].
  split; [ | exact (IH (Forall_inv_tail Hhandlers)) ].
  split; [ exact (Forall_inv Hhandlers) | ].
  split; [ exact Hall | exact Hconts ].
Qed.

(* ======================================================================== *)
(* The code of the static environment                                       *)
(* ======================================================================== *)

Definition code_all_closed (statics : static_env) : Prop :=
  Forall (fun binding => code_is_closed (snd binding) = true)
    (static_code statics).

Lemma eval_static_bindings_closed (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    forallb binding_code_closed bindings = true ->
    code_all_closed statics ->
    eval_static_bindings vars bindings statics = Some statics' ->
    code_all_closed statics'.
Proof.
  induction bindings as [ | binding bindings IH]; intros statics statics';
    simpl.
  - intros _ Hstatics [= <-]. exact Hstatics.
  - rewrite andb_true_iff. intros [Hbinding Hbindings] Hstatics.
    destruct binding as [id c | id | sym const]; simpl in Hbinding.
    + destruct (lookup id (static_code statics)); [ discriminate | ].
      apply (IH _ _ Hbindings).
      apply Forall_cons; [ exact Hbinding | exact Hstatics ].
    + exact (IH _ _ Hbindings Hstatics).
    + destruct (lookup sym (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ]; simpl;
        [ | discriminate ].
      apply (IH _ _ Hbindings). exact Hstatics.
Qed.

Lemma lookup_code_closed (statics : static_env) (id : code_id) (c : code) :
  code_all_closed statics -> lookup id (static_code statics) = Some c ->
  code_is_closed c = true.
Proof.
  intros Hstatics Hlookup. apply lookup_in in Hlookup.
  unfold code_all_closed in Hstatics. rewrite Forall_forall in Hstatics.
  exact (Hstatics (id, c) Hlookup).
Qed.

(* ======================================================================== *)
(* A step keeps a state in scope                                            *)
(* ======================================================================== *)

Lemma state_in_scope_intro (outer : list continuation) (e : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  names_among (map fst vars) (map fst conts ++ outer) e = true ->
  code_closed e = true ->
  conts_in_scope outer conts ->
  code_all_closed statics ->
  state_in_scope outer (State e vars conts statics hp rd).
Proof. intros Hnames Hcode Hconts Hstatics. repeat split; assumption. Qed.

Theorem unit_in_scope : unit_in_scope_statement.
Proof.
  intros unit_return unit_exn e Hnames Hcode.
  apply state_in_scope_intro;
    [ exact Hnames | exact Hcode | exact I | apply Forall_nil ].
Qed.

(* A full call of code that the static environment binds, returning to
   continuations in scope. *)
Lemma call_code_in_scope (outer : list continuation) (closure : value)
    (called : code_id) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  code_all_closed statics ->
  lookup called (static_code statics) = Some c ->
  cont_in_scope outer return_to ->
  cont_in_scope outer exn_to ->
  match
    call_code closure c vs args_arity return_arity return_to exn_to statics hp
      rd
  with
  | Step s' => state_in_scope outer s'
  | Exit k _ _ _ => among k outer = true
  | Undefined => True
  end.
Proof.
  intros Hstatics Hc Hreturn Hexn. unfold call_code.
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (code_params_arity c))
        && arity_eqb return_arity (code_result_arity c)));
    [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [callee_vars | ] eqn:Hbind; [ | exact I ].
  destruct
    (proj1 (closed_code c) (lookup_code_closed statics called c Hstatics Hc))
    as [Hbody Hbody_closed].
  apply state_in_scope_intro; [ | exact Hbody_closed | | exact Hstatics ].
  - refine (names_among_mono _ _ _ _ _ _ _ Hbody).
    + intros x Hx. apply (among_bind_params _ _ _ _ x Hbind).
      rewrite among_app. rewrite among_cons in Hx. simpl.
      destruct (Pos.eqb x (code_my_closure c));
        destruct (among x (code_params c)); simpl in *; auto.
    + intros k' Hk'. simpl in *.
      destruct (Pos.eqb k' (code_return_continuation c));
        destruct (Pos.eqb k' (code_exn_continuation c)); simpl in *;
        auto; discriminate.
  - split; [ exact Hreturn | ]. split; [ exact Hexn | exact I ].
Qed.

Lemma run_call_in_scope (outer : list continuation)
    (return_partial : value -> step_result) (kind : call_kind)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  code_all_closed statics ->
  shape_calls statics kind shape ->
  cont_in_scope outer return_to ->
  cont_in_scope outer exn_to ->
  (∀ partial : value,
    match return_partial partial with
    | Step s' => state_in_scope outer s'
    | Exit k _ _ _ => among k outer = true
    | Undefined => True
    end) ->
  match run_call return_partial shape return_arity return_to exn_to statics hp
    rd
    with
  | Step s' => state_in_scope outer s'
  | Exit k _ _ _ => among k outer = true
  | Undefined => True
  end.
Proof.
  intros Hstatics Hcalls Hreturn Hexn Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls |- *.
  - destruct Hcalls as (called & slots & _ & Hc & _).
    exact (call_code_in_scope outer closure called c vs _ _ _ _ statics _ rd_
             Hstatics Hc Hreturn Hexn).
  - destruct Hcalls as (called & slots & _ & Hc & _).
    apply (call_code_in_scope outer closure called c vs _ _ _ _ statics _ rd_
             Hstatics Hc);
      [ exact (conj Hreturn Hexn) | exact Hexn ].
  - apply Hpartial.
  - exact I.
Qed.

(* Applying a continuation in scope. *)
Lemma apply_cont_value_in_scope (outer : list continuation)
    (c : cont_value) :
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record),
    code_all_closed statics ->
    cont_in_scope outer c ->
    match apply_cont_value c vs statics hp rd with
    | Step s' => state_in_scope outer s'
    | Exit k _ _ _ => among k outer = true
    | Undefined => True
    end.
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k'
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs statics hp rd Hstatics Hc.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
      [ | exact I ].
    destruct Hc as ([Hh Hh_closed] & Hrecursive & Hdef_conts).
    rewrite handler_names_among_handler in Hh.
    apply state_in_scope_intro; [ | exact Hh_closed | | exact Hstatics ].
    + rewrite map_app, rec_closures_domain, <- app_assoc.
      refine (names_among_mono _ _ _ _ _ _ (fun k'' Hk'' => Hk'') Hh).
      intros x. exact (among_bind_params _ _ _ _ x Hbind).
    + apply conts_in_scope_app; [ | exact Hdef_conts ].
      apply rec_closures_in_scope; assumption.
  - exact Hc.
  - exact (IH (vs ++ extra) statics hp rd_ Hstatics Hc).
  - rewrite apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    destruct Hc as [Hreturn Hexn]. unfold call_function.
    apply (run_call_in_scope outer _ Indirect); try assumption.
    + apply shape_of_call_calls.
    + intros partial. exact
      (IHreturn [partial] statics hp rd_ Hstatics Hreturn).
Qed.

Lemma call_function_in_scope (outer : list continuation) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  code_all_closed statics ->
  cont_in_scope outer return_to ->
  cont_in_scope outer exn_to ->
  match
    call_function callee kind vs args_arity return_arity return_to exn_to
      statics hp rd
  with
  | Step s' => state_in_scope outer s'
  | Exit k _ _ _ => among k outer = true
  | Undefined => True
  end.
Proof.
  intros Hstatics Hreturn Hexn. unfold call_function.
  apply (run_call_in_scope outer _ kind); try assumption.
  - apply shape_of_call_calls.
  - intros partial.
    exact (apply_cont_value_in_scope outer return_to [partial] statics _ rd_
             Hstatics Hreturn).
Qed.

Theorem step_in_scope : step_in_scope_statement.
Proof.
  intros outer [e vars conts statics] ([Hnames Hcode] & Hconts & Hstatics).
  simpl state_expr in *. simpl state_vars in *. simpl state_conts in *.
  simpl state_statics in *. fold (code_all_closed statics) in Hstatics.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step.
  - (* Let *)
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); [ | exact I ].
    destruct (names_among_let _ _ _ _ _ _ Hnames) as [_ Hbody].
    apply state_in_scope_intro; assumption.
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Hbindings; [ | exact I ].
    simpl in Hcode. apply andb_true_iff in Hcode.
    destruct Hcode as [Hbindings_closed Hbody_closed].
    apply state_in_scope_intro;
      [ exact (names_among_let_static _ _ _ _ Hnames) | exact Hbody_closed
      | exact Hconts | ].
    exact (eval_static_bindings_closed vars bindings statics statics'
             Hbindings_closed Hstatics Hbindings).
  - (* Let_cont_non_recursive *)
    destruct (names_among_let_cont _ _ _ _ _ Hnames) as [Hh Hbody].
    simpl in Hcode. apply andb_true_iff in Hcode.
    destruct Hcode as [Hh_closed Hbody_closed].
    apply state_in_scope_intro;
      [ exact Hbody | exact Hbody_closed | | exact Hstatics ].
    split; [ | exact Hconts ].
    split; [ split; [ exact Hh | exact Hh_closed ] | ].
    split; [ apply Forall_nil | exact Hconts ].
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ exact I | ].
    destruct (names_among_let_cont_recursive _ _ _ _ Hnames)
      as [Hhandlers Hbody].
    simpl in Hcode. apply andb_true_iff in Hcode.
    destruct Hcode as [Hhandlers_closed Hbody_closed].
    assert (Hscoped :
      Forall
        (fun binding : continuation * continuation_handler =>
           handler_scoped (map fst vars)
             (map fst handlers ++ map fst conts ++ outer) (snd binding))
        handlers).
    { apply Forall_forall. intros [k h] Hin. split.
      - rewrite Forall_forall in Hhandlers. exact (Hhandlers (k, h) Hin).
      - rewrite forallb_forall in Hhandlers_closed.
        exact (Hhandlers_closed (k, h) Hin). }
    apply state_in_scope_intro; [ | exact Hbody_closed | | exact Hstatics ].
    + rewrite map_app, rec_closures_domain, <- app_assoc. exact Hbody.
    + apply conts_in_scope_app; [ | exact Hconts ].
      apply rec_closures_in_scope; assumption.
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    destruct (names_among_apply _ _ _ Hnames) as [Hreturn Hexn].
    apply call_function_in_scope;
      [ exact Hstatics | exact (resolve_in_scope _ _ _ Hconts Hreturn)
      | apply with_extra_args_in_scope;
        exact (resolve_in_scope _ _ _ Hconts Hexn) ].
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
    unfold apply_continuation.
    apply apply_cont_value_in_scope; [ exact Hstatics | ].
    exact (resolve_in_scope outer (ac_continuation ac) conts Hconts
             (names_among_apply_cont _ _ _ Hnames)).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact I ].
    apply find_arm_in in Harm.
    apply state_in_scope_intro;
      [ | reflexivity | exact Hconts | exact Hstatics ].
    exact (names_among_switch_arm _ _ _ _ _ _ Hnames Harm).
  - (* Invalid *)
    exact I.
Qed.

Lemma run_in_scope (outer : list continuation) (n : nat) :
  ∀ s : state,
    state_in_scope outer s ->
    match run n s with
    | Step s' => state_in_scope outer s'
    | Exit k _ _ _ => among k outer = true
    | Undefined => True
    end.
Proof.
  induction n as [ | n IH]; intros s Hs; simpl; [ exact Hs | ].
  pose proof (step_in_scope outer s Hs) as Hstep.
  destruct (step s) as [s' | k vs | ];
    [ exact (IH s' Hstep) | exact Hstep | exact I ].
Qed.

(* ======================================================================== *)
(* A state in scope reads no variable that has no value                     *)
(* ======================================================================== *)

Lemma variables_read_free (e : expr) (x : variable) :
  x ∈ variables_read e -> x ∈ free_variables_of_expr e.
Proof.
  destruct e
    as [ y kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try (intros []; fail); try (intros Hx; exact Hx).
  - intros Hx. apply in_or_app. left. exact Hx.
  - intros Hx. apply in_or_app. left.
    apply in_flat_map in Hx. destruct Hx as [binding [Hin Hx]].
    apply in_flat_map. exists binding. split; [ exact Hin | ].
    destruct binding as [id c | id | s const]; simpl in *;
      [ destruct Hx | destruct Hx | exact Hx ].
  - intros Hx. apply in_or_app. left. exact Hx.
Qed.

Theorem in_scope_reads_bound : in_scope_reads_bound_statement.
Proof.
  intros outer [e vars conts statics] ([Hnames _] & _).
  unfold reads_unbound_variable. simpl in *.
  apply not_true_is_false. intros Hreads.
  apply existsb_exists in Hreads. destruct Hreads as [x [Hin Hunbound]].
  apply variables_read_free, among_In in Hin.
  unfold names_among in Hnames. apply andb_true_iff in Hnames.
  destruct Hnames as [Hvars _].
  assert (Hbound : is_bound x vars = true).
  { rewrite is_bound_among. exact (all_among_among _ _ x Hvars Hin). }
  rewrite Hbound in Hunbound. discriminate.
Qed.

(* ======================================================================== *)
(* The run of a unit                                                        *)
(* ======================================================================== *)

Theorem unit_run : unit_run_statement.
Proof.
  intros unit_return unit_exn e n Hnames Hcode.
  pose proof
    (run_in_scope [unit_return; unit_exn] n _
       (unit_in_scope unit_return unit_exn e Hnames Hcode))
    as Hrun.
  destruct (run n (State e [] [] empty_static_env empty_heap no_reads)) as
    [s | k vs | ];
    [ exact (in_scope_reads_bound _ s Hrun) | | exact I ].
  apply among_In in Hrun.
  destruct Hrun as [<- | [<- | []]]; [ left | right ]; reflexivity.
Qed.
