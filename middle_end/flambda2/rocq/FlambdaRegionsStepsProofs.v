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

(* Lemmas for the proofs of the statements of
   FlambdaRegionsInvariantSpec.v: what a local allocation draws and
   binds, what a step of the machine of regions does to the values and
   to the terms of a state, and one traversal of the forms of a step.
   Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaClosuresProofs FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsNeedsHeapProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsReadsProofs.
Import ListNotations.

(* ======================================================================== *)
(* The symbols that a local allocation draws                                *)
(* ======================================================================== *)

Lemma advance_S (next : symbol) (count : nat) :
  advance next (S count) = Pos.succ (advance next count).
Proof. reflexivity. Qed.

Lemma advance_succ (next : symbol) (count : nat) :
  advance (Pos.succ next) count = Pos.succ (advance next count).
Proof.
  induction count as [ | count IH]; [ reflexivity | ].
  rewrite !advance_S, IH. reflexivity.
Qed.

Lemma number_slots_length (function_decls : list (function_slot * code_id)) :
  ∀ next : symbol,
    length (number_slots function_decls next) = length function_decls.
Proof.
  induction function_decls as [ | [slot code] function_decls IH];
    intros next; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma number_slots_below (function_decls : list (function_slot * code_id)) :
  ∀ (next : symbol) (slot : function_slot) (s : symbol),
    In (slot, s) (number_slots function_decls next) ->
    (s < advance next (length function_decls))%positive.
Proof.
  induction function_decls as [ | [slot0 code] function_decls IH];
    intros next slot s; cbn [number_slots length In]; [ contradiction | ].
  rewrite advance_S. intros [[= _ <-] | Hin].
  - pose proof (advance_le next (length function_decls)). lia.
  - specialize (IH _ _ _ Hin). rewrite advance_succ in IH. exact IH.
Qed.

(* What [localise] binds: the symbols are from [next] on and below the
   counter after; each is bound to something that holds directly what
   the value holds; the symbols that a set gives its slots are among
   those bound; and the symbol that is the value is bound. *)
Lemma localise_makes (next : symbol) (v : value) (sym : symbol)
    (bindings : list (symbol * statically_allocated_value)) :
  localise next v = Some (sym, bindings) ->
  is_made v = true
  ∧ (∃ allocated : statically_allocated_value, In (sym, allocated) bindings)
  ∧ ∀ (s : symbol) (allocated : statically_allocated_value),
      In (s, allocated) bindings ->
      (next <= s)%positive
      ∧ (s < advance next (length bindings))%positive
      ∧ allocated_values allocated = value_parts v
      ∧ ∀ s' : symbol,
          In s' (allocated_symbols allocated) ->
          ∃ allocated' : statically_allocated_value,
            In (s', allocated') bindings.
Proof.
  destruct v as [c | code value_slots | slot function_decls value_slots | s0
                | tag fields | elements | c | l | callee args arity];
    simpl; try discriminate;
    try (intros [= <- <-]; split; [ reflexivity | ];
         split; [ eexists; left; reflexivity | ];
         intros s allocated [[= <- <-] | []];
         split; [ lia | ];
         split; [ simpl; lia | ];
         split; [ reflexivity | ]; intros s' []).
  destruct (lookup slot (number_slots function_decls next)) as [s1 | ]
    eqn:Hs1; simpl; [ | discriminate ].
  intros [= <- <-]. split; [ reflexivity | ]. split.
  - eexists. apply in_map_iff. exists (slot, s1).
    split; [ reflexivity | exact (lookup_in _ _ _ Hs1) ].
  - intros s allocated Hin.
    apply in_map_iff in Hin as ([slot' s'] & [= <- <-] & Hin).
    split; [ exact (number_slots_from _ _ _ _ Hin) | ].
    split;
      [ rewrite length_map, number_slots_length;
        exact (number_slots_below _ _ _ _ Hin)
      | ].
    split; [ reflexivity | ].
    simpl. intros s'' Hs''.
    apply in_map_iff in Hs'' as ([slot'' s3] & <- & Hin'').
    eexists. apply in_map_iff. exists (slot'', s3).
    split; [ reflexivity | exact Hin'' ].
Qed.

(* ======================================================================== *)
(* What a group of static bindings adds                                     *)
(* ======================================================================== *)

Lemma eval_static_bindings_parts (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    eval_static_bindings vars bindings statics = Some statics' ->
    ∃ (new_code : list (code_id * code))
      (new_values : list (symbol * statically_allocated_value)),
      static_code statics' = new_code ++ static_code statics
      ∧ static_values statics' = new_values ++ static_values statics
      ∧ (∀ (id : code_id) (c : code),
          In (id, c) new_code -> In (Bind_code id c) bindings)
      ∧ (∀ (s : symbol) (allocated : statically_allocated_value),
          In (s, allocated) new_values ->
          ∃ const : static_const,
            In (Bind_symbol s const) bindings
            ∧ eval_static_const vars const = Some allocated).
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH];
    intros statics statics'; simpl.
  - intros [= <-]. exists [], [].
    repeat split; try reflexivity; intros ? ? [].
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    intros Heval.
    destruct (IH _ _ Heval)
      as (new_code & new_values & Hcode & Hvalues & Hc & Hv).
    simpl in Hcode, Hvalues.
    exists (new_code ++ [(id, c)]), new_values.
    split; [ rewrite <- app_assoc; exact Hcode | ].
    split; [ exact Hvalues | ]. split.
    + intros id' c' Hin.
      apply in_app_iff in Hin as [Hin | [[= <- <-] | []]];
        [ right; exact (Hc _ _ Hin) | left; reflexivity ].
    + intros s a Hin. destruct (Hv s a Hin) as (const & Hin' & Hconst).
      exists const. split; [ right; exact Hin' | exact Hconst ].
  - intros Heval.
    destruct (IH _ _ Heval)
      as (new_code & new_values & Hcode & Hvalues & Hc & Hv).
    exists new_code, new_values.
    split; [ exact Hcode | ]. split; [ exact Hvalues | ]. split.
    + intros id' c' Hin. right. exact (Hc _ _ Hin).
    + intros s a Hin. destruct (Hv s a Hin) as (const & Hin' & Hconst).
      exists const. split; [ right; exact Hin' | exact Hconst ].
  - destruct (lookup s (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
      simpl; [ | discriminate ].
    intros Heval.
    destruct (IH _ _ Heval)
      as (new_code & new_values & Hcode & Hvalues & Hc & Hv).
    simpl in Hcode, Hvalues.
    exists new_code, (new_values ++ [(s, allocated)]).
    split; [ exact Hcode | ].
    split; [ rewrite <- app_assoc; exact Hvalues | ]. split.
    + intros id' c' Hin. right. exact (Hc _ _ Hin).
    + intros s' a Hin.
      apply in_app_iff in Hin as [Hin | [[= <- <-] | []]].
      * destruct (Hv s' a Hin) as (const' & Hin' & Hconst').
        exists const'. split; [ right; exact Hin' | exact Hconst' ].
      * exists const. split; [ left; reflexivity | exact Hconst ].
Qed.

(* ======================================================================== *)
(* The values of a state after a step                                       *)
(* ======================================================================== *)

(* The heap of the state is not asked anything, so this is not
   FlambdaSemanticsProofs.step_values_all, which asks the property of
   the cells: the invariants of regions say nothing of the heap. It is
   that proof, on the two lemmas under it, of a state that is not at a
   [Let] that needs the heap (decision D53): such a [Let] binds what
   [eval_named] gives, and reads no cell. *)
Section Step_values.

Variable P : value -> Prop.

Hypothesis P_partial :
  ∀ (callee : value) (args : list value) (args_arity : arity),
    P (Value_partial_application callee args args_arity)
    <-> P callee ∧ Forall P args.

Definition vals_ok (s : state) : Prop :=
  env_values_all P (state_vars s) ∧ conts_values_all P (state_conts s).

Definition result_vals_ok (result : step_result) : Prop :=
  match result with
  | Step s' => vals_ok s'
  | Exit _ vs _ _ => Forall P vs
  | Undefined => True
  end.

Lemma step_values (s : state) :
  P_fields P (state_statics s) ->
  vals_ok s ->
  needs_no_heap s ->
  (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr)
     (v : value),
    state_expr s = Let x kind d body ->
    eval_named (state_statics s) (state_vars s) d = Some v ->
    P v) ->
  (∀ (sim : simple) (v : value),
    In sim (evaluated_simples (state_expr s)) ->
    eval_simple (state_vars s) sim = Some v ->
    P v) ->
  result_vals_ok (step s).
Proof using P_partial.
  destruct s as [e vars conts statics hp rd].
  cbn [state_expr state_vars state_conts state_statics].
  intros Hfields [Hvars Hconts] Hneeds Hlet Hsimples.
  cbn [state_vars state_conts] in Hvars, Hconts.
  assert (Hstate :
    ∀ (e' : expr) (vars' : env) (conts' : cont_env),
      env_values_all P vars' ->
      conts_values_all P conts' ->
      vals_ok (State e' vars' conts' statics hp rd))
    by (intros e' vars' conts' Hv Hc; exact (conj Hv Hc)).
  unfold result_vals_ok.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step.
  - destruct (exec_named _ _ _ _ d) as [[[v hp'] rd'] | ] eqn:Hd;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); [ | exact I ].
    destruct
      (exec_named_needs_no_heap_some _ _ _ _ _ _ _ _
         (Hneeds x kind d body eq_refl) Hd)
      as (Hv & _ & _).
    split; [ | exact Hconts ].
    apply Forall_cons; [ exact (Hlet x kind d body v eq_refl Hv) | ].
    exact Hvars.
  - destruct (eval_static_bindings vars bindings statics); [ | exact I ].
    exact (conj Hvars Hconts).
  - split; [ exact Hvars | ].
    split; [ split; [ exact Hvars | exact Hconts ] | exact Hconts ].
  - destruct (handlers_bind_statics handlers); [ exact I | ].
    split; [ exact Hvars | ].
    apply conts_values_all_app; [ | exact Hconts ].
    apply rec_closures_values_all; assumption.
  - destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextra; [ | exact I ].
    simpl in Hsimples.
    apply (call_function_values_all P P_partial vals_ok);
      [ exact Hfields
      | exact (Hsimples _ _ (or_introl eq_refl) Hcallee)
      | apply (eval_simples_forall P vars (ap_args ap) vs); [ | exact Hargs ];
        intros sim w Hin; apply Hsimples; right; apply in_or_app; left;
        exact Hin
      | apply resolve_values_all, Hconts
      | apply with_extra_args_values_all;
        [ apply resolve_values_all, Hconts
        | apply (eval_simples_forall P vars (exn_extra_simples ap) extra);
          [ | exact Hextra ];
          intros sim w Hin; apply Hsimples; right; apply in_or_app; right;
          exact Hin ]
      | exact Hstate ].
  - destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    unfold apply_continuation.
    apply (apply_cont_value_values_all P P_partial vals_ok);
      [ exact Hfields | apply resolve_values_all, Hconts | | exact Hstate ].
    exact (eval_simples_forall P vars (ac_args ac) vs Hsimples Hargs).
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ]; [ | exact I ].
    exact (conj Hvars Hconts).
  - exact I.
Qed.

Definition region_result_vals_ok (result : region_step_result) : Prop :=
  match result with
  | Region_step rs' => vals_ok (rs_state rs')
  | Region_exit _ vs _ => Forall P vs
  | Region_undefined _ | Region_let_not_run => True
  end.

Lemma plain_step_values (rs : region_state) :
  P_fields P (live_statics rs) ->
  vals_ok (rs_state rs) ->
  needs_no_heap (rs_state rs) ->
  (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr)
     (v : value),
    state_expr (rs_state rs) = Let x kind d body ->
    eval_named (live_statics rs) (state_vars (rs_state rs)) d = Some v ->
    P v) ->
  (∀ (sim : simple) (v : value),
    In sim (evaluated_simples (state_expr (rs_state rs))) ->
    eval_simple (state_vars (rs_state rs)) sim = Some v ->
    P v) ->
  region_result_vals_ok (plain_step rs).
Proof.
  intros Hfields Hvals Hneeds Hlet Hsimples. unfold plain_step.
  pose proof
    (step_values (with_statics (rs_state rs) (live_statics rs)) Hfields Hvals
       Hneeds Hlet Hsimples)
    as Hstep.
  destruct (step (with_statics (rs_state rs) (live_statics rs)))
    as [s' | k vs hp rd | ];
    [ exact Hstep | exact Hstep | exact I ].
Qed.

Lemma region_apply_cont_values (rs : region_state) (ac : apply_cont) :
  state_expr (rs_state rs) = Apply_cont ac ->
  P_fields P (live_statics rs) ->
  vals_ok (rs_state rs) ->
  (∀ (sim : simple) (v : value),
    In sim (ac_args ac) ->
    eval_simple (state_vars (rs_state rs)) sim = Some v ->
    P v) ->
  region_result_vals_ok (region_apply_cont rs ac).
Proof.
  intros He Hfields Hvals Hsimples. unfold region_apply_cont.
  pose proof (resolve_values_all P (ac_continuation ac) _ (proj2 Hvals))
    as Hc.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [[params metadata handler] def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    try (apply plain_step_values;
         [ exact Hfields | exact Hvals
         | intros x kind d body Hlet; rewrite He in Hlet; discriminate
         | intros x kind d body v Hlet; rewrite He in Hlet; discriminate
         | rewrite He; exact Hsimples ]).
  destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac)) as [vs | ]
    eqn:Hargs; [ | exact I ].
  destruct (kinds_eqb _ _); [ | exact I ].
  destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
    [ | exact I ].
  destruct Hc as [Hdef_vars Hdef_conts].
  split; cbn [rs_state state_vars state_conts].
  - exact
      (bind_params_values_all P params vs def_vars handler_vars
         (eval_simples_forall P _ _ vs Hsimples Hargs) Hdef_vars Hbind).
  - apply conts_values_all_app; [ | exact Hdef_conts ].
    apply rec_closures_values_all; assumption.
Qed.

End Step_values.

(* ======================================================================== *)
(* The terms of a state after a step                                        *)
(* ======================================================================== *)

(* FlambdaClosuresProofs.terms_inv_step asks that the test holds of
   every [Apply_cont]. The tests here are of the simples of the term,
   so that is asked of the arm that a [Switch] takes instead. *)
Section Step_terms.

Variable Pe : expr -> bool.
Variable Ph : continuation_handler -> bool.
Variable Pc : code -> bool.

Hypothesis Pe_let :
  ∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
    Pe (Let x kind d body) = true -> Pe body = true.
Hypothesis Pe_let_static :
  ∀ (bindings : list static_binding) (body : expr),
    Pe (Let_static bindings body) = true ->
    Pe body = true
    ∧ ∀ (id : code_id) (c : code), In (Bind_code id c) bindings -> Pc c = true.
Hypothesis Pe_let_cont :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    Pe (Let_cont_non_recursive k h body) = true ->
    Ph h = true ∧ Pe body = true.
Hypothesis Pe_let_cont_rec :
  ∀ (handlers : list (continuation * continuation_handler)) (body : expr),
    Pe (Let_cont_recursive handlers body) = true ->
    forallb (fun '(_, h) => Ph h) handlers = true ∧ Pe body = true.
Hypothesis Pe_switch :
  ∀ (scrutinee : simple) (arms : list (Z * apply_cont)) (i : Z)
    (ac : apply_cont),
    Pe (Switch scrutinee arms) = true ->
    find_arm i arms = Some ac ->
    Pe (Apply_cont ac) = true.
Hypothesis Ph_handler :
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr),
    Ph (Handler params metadata handler) = true -> Pe handler = true.
Hypothesis Pc_body : ∀ c : code, Pc c = true -> Pe (code_body c) = true.

Lemma step_terms (s s' : state) :
  terms_inv Pe Ph Pc s -> step s = Step s' -> terms_inv Pe Ph Pc s'.
Proof.
  intros (P & C & S) Hstep. rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hstep.
  - destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    exact (conj (Pe_let x kind d body P) (conj C S)).
  - destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    destruct (Pe_let_static bindings body P) as [Pbody Pcode].
    refine (conj Pbody (conj C _)).
    exact (eval_static_bindings_code_terms_ok Pc vars bindings statics
             statics' Pcode S Heval).
  - injection Hstep as <-. unfold terms_inv. simpl_state.
    destruct (Pe_let_cont k h body P) as [Hh Pbody].
    refine (conj Pbody (conj (conj _ C) S)).
    simpl. exact (conj Hh (conj eq_refl C)).
  - destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    destruct (Pe_let_cont_rec handlers body P) as [Phandlers Pbody].
    refine (conj Pbody (conj (conts_terms_ok_app Ph _ _ _ C) S)).
    exact (rec_closures_terms_ok Ph vars conts handlers handlers Phandlers C
             Phandlers).
  - destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    refine
      (call_function_lands statics _ rd_ (cont_terms_ok Ph) (terms_inv Pe Ph Pc)
         (fun c extra H => H) (terms_inv_over Ph)
         (terms_inv_code Pe Ph Pc Pc_body statics _ rd_ S)
         (terms_inv_handler Pe Ph Pc Ph_handler statics _ rd_ S) _ _ _ _ _ _ _
           s'
         (resolve_terms_ok Ph _ conts C) _ Hstep).
    apply with_extra_args_terms_ok, resolve_terms_ok, C.
  - destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep.
    exact
      (apply_cont_value_lands statics _ rd_ (cont_terms_ok Ph)
         (terms_inv Pe Ph Pc) (fun c extra H => H) (terms_inv_over Ph)
         (terms_inv_code Pe Ph Pc Pc_body statics _ rd_ S)
         (terms_inv_handler Pe Ph Pc Ph_handler statics _ rd_ S) _ vs s'
         (resolve_terms_ok Ph _ conts C) Hstep).
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | discriminate ].
    injection Hstep as <-. unfold terms_inv. simpl_state.
    exact (conj (Pe_switch scrutinee arms i ac P Harm) (conj C S)).
  - discriminate.
Qed.

(* [live_statics] has the code of the static environment of the state,
   so [terms_inv] of the state that [plain_step] gives to [step] is
   that of the state. *)
Lemma plain_step_terms (rs rs' : region_state) :
  terms_inv Pe Ph Pc (rs_state rs) ->
  plain_step rs = Region_step rs' ->
  terms_inv Pe Ph Pc (rs_state rs').
Proof.
  intros (He & Hconts & Hcode). unfold plain_step.
  destruct (step (with_statics (rs_state rs) (live_statics rs)))
    as [s' | k vs hp rd | ] eqn:Hstep; [ | discriminate | discriminate ].
  intros [= <-].
  destruct
    (step_terms (with_statics (rs_state rs) (live_statics rs)) s'
       (conj He (conj Hconts Hcode)) Hstep)
    as (He' & Hconts' & _).
  exact (conj He' (conj Hconts' Hcode)).
Qed.

Lemma static_step_terms (rs rs' : region_state) :
  terms_inv Pe Ph Pc (rs_state rs) ->
  static_step rs = Region_step rs' ->
  terms_inv Pe Ph Pc (rs_state rs').
Proof.
  intros Hinv. unfold static_step.
  destruct (step (rs_state rs)) as [s' | k vs hp rd | ] eqn:Hstep;
    [ | discriminate | discriminate ].
  intros [= <-]. exact (step_terms _ _ Hinv Hstep).
Qed.

Lemma region_apply_cont_terms (rs rs' : region_state) (ac : apply_cont) :
  terms_inv Pe Ph Pc (rs_state rs) ->
  region_apply_cont rs ac = Region_step rs' ->
  terms_inv Pe Ph Pc (rs_state rs').
Proof.
  intros Hinv. pose proof Hinv as (He & Hconts & Hcode).
  unfold region_apply_cont.
  pose proof (resolve_terms_ok Ph (ac_continuation ac) _ Hconts) as Hc.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [[params metadata handler] def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    try (apply plain_step_terms; exact Hinv).
  destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac)) as [vs | ];
    [ | discriminate ].
  destruct (kinds_eqb _ _); [ | discriminate ].
  destruct (bind_params params vs def_vars) as [handler_vars | ];
    [ | discriminate ].
  intros [= <-].
  exact
    (terms_inv_handler Pe Ph Pc Ph_handler (state_statics (rs_state rs))
       (state_heap (rs_state rs)) rd_ Hcode params metadata handler def_vars
       def_conts recursive handler_vars Hc).
Qed.

End Step_terms.

(* ======================================================================== *)
(* The tests of a continuation, as a test of terms and one of values        *)
(* ======================================================================== *)

Lemma cont_value_deep_ind (K : cont_value -> Prop) :
  (∀ (h : continuation_handler) (vars : env)
     (conts : list (continuation * cont_value))
     (recursive : list (continuation * continuation_handler)),
    Forall (fun kc => K (snd kc)) conts ->
    K (Closure h vars conts recursive)) ->
  (∀ k : continuation, K (Outer k)) ->
  (∀ (c : cont_value) (extra : list value), K c -> K (Extra_args c extra)) ->
  (∀ (remaining : list value) (remaining_arity return_arity : arity)
     (return_to exn_to : cont_value),
    K return_to ->
    K exn_to ->
    K (Over_application remaining remaining_arity return_arity return_to
         exn_to)) ->
  ∀ c : cont_value, K c.
Proof.
  intros Hclosure Houter Hextra Hover. fix IH 1.
  intros [h vars conts recursive | k | c extra
         | remaining remaining_arity return_arity return_to exn_to].
  - apply Hclosure.
    exact
      ((fix go (conts : list (continuation * cont_value))
          : Forall (fun kc => K (snd kc)) conts :=
          match conts with
          | [] => Forall_nil _
          | kc :: conts => Forall_cons _ (IH (snd kc)) (go conts)
          end)
         conts).
  - apply Houter.
  - apply Hextra. apply IH.
  - apply Hover; apply IH.
Qed.

Lemma forallb_pairs_Forall {A B : Type} (f : B -> bool) (l : list (A * B)) :
  forallb (fun '(_, b) => f b) l = true
  <-> Forall (fun ab => f (snd ab) = true) l.
Proof.
  induction l as [ | [a b] l [IH1 IH2]]; simpl.
  - split; [ intros _; apply Forall_nil | reflexivity ].
  - split.
    + intros H. apply andb_true_iff in H as [Hb Hl].
      apply Forall_cons; [ exact Hb | exact (IH1 Hl) ].
    + intros H. rewrite (Forall_inv H : f b = true).
      exact (IH2 (Forall_inv_tail H)).
Qed.

Lemma conts_split (f : cont_value -> bool) (K : cont_value -> Prop)
    (conts : cont_env) :
  Forall (fun kc => f (snd kc) = true <-> K (snd kc)) conts ->
  (forallb (fun '(_, c) => f c) conts = true
   <-> fold_right (fun kc acc => K (snd kc) ∧ acc) True conts).
Proof.
  induction conts as [ | [k c] conts IH]; simpl; intros Hall.
  - split; [ intros _; exact I | reflexivity ].
  - destruct (Forall_inv Hall) as [H1 H2]. simpl in H1, H2.
    destruct (IH (Forall_inv_tail Hall)) as [H3 H4]. split.
    + intros H. apply andb_true_iff in H as [Hc Hrest].
      exact (conj (H1 Hc) (H3 Hrest)).
    + intros [Hc Hrest]. rewrite (H2 Hc). exact (H4 Hrest).
Qed.

Lemma fold_right_conj_split (A B : cont_value -> Prop) (conts : cont_env) :
  fold_right (fun kc acc => (A (snd kc) ∧ B (snd kc)) ∧ acc) True conts
  <-> fold_right (fun kc acc => A (snd kc) ∧ acc) True conts
      ∧ fold_right (fun kc acc => B (snd kc) ∧ acc) True conts.
Proof.
  induction conts as [ | kc conts [IH1 IH2]]; simpl.
  - split; [ intros _; exact (conj I I) | intros _; exact I ].
  - split.
    + intros [[Ha Hb] Hrest]. destruct (IH1 Hrest) as [HA HB].
      exact (conj (conj Ha HA) (conj Hb HB)).
    + intros [[Ha HA] [Hb HB]].
      exact (conj (conj Ha Hb) (IH2 (conj HA HB))).
Qed.

(* A test of the handlers [fh] and one of the values [fv], as
   FlambdaRegionsInvariantSpec.cont_below and [cont_no_region_poison]
   are made. *)
Section Cont_split.

Variable fh : continuation_handler -> bool.
Variable fv : value -> bool.
Variable f : cont_value -> bool.

Hypothesis f_closure :
  ∀ (h : continuation_handler) (vars : env)
    (conts : list (continuation * cont_value))
    (recursive : list (continuation * continuation_handler)),
    f (Closure h vars conts recursive)
    = fh h
      && forallb (fun '(_, v) => fv v) vars
      && forallb (fun '(_, c) => f c) conts
      && forallb (fun '(_, h) => fh h) recursive.
Hypothesis f_outer : ∀ k : continuation, f (Outer k) = true.
Hypothesis f_extra :
  ∀ (c : cont_value) (extra : list value),
    f (Extra_args c extra) = f c && forallb fv extra.
Hypothesis f_over :
  ∀ (remaining : list value) (remaining_arity return_arity : arity)
    (return_to exn_to : cont_value),
    f (Over_application remaining remaining_arity return_arity return_to
         exn_to)
    = forallb fv remaining && f return_to && f exn_to.

Lemma cont_split (c : cont_value) :
  f c = true
  <-> cont_terms_ok fh c ∧ cont_values_all (fun v => fv v = true) c.
Proof.
  induction c
    as [h vars conts recursive IH | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_deep_ind.
  - rewrite f_closure. simpl.
    destruct
      (conts_split f
         (fun c =>
            cont_terms_ok fh c ∧ cont_values_all (fun v => fv v = true) c)
         conts IH)
      as [Hc1 Hc2].
    destruct
      (fold_right_conj_split (cont_terms_ok fh)
         (cont_values_all (fun v => fv v = true)) conts)
      as [Hs1 Hs2].
    destruct (forallb_pairs_Forall fv vars) as [Hv1 Hv2].
    split.
    + intros H. apply andb_true_iff in H as [H Hrec].
      apply andb_true_iff in H as [H Hconts].
      apply andb_true_iff in H as [Hh Hvars].
      destruct (Hs1 (Hc1 Hconts)) as [Ht Hv].
      exact (conj (conj Hh (conj Hrec Ht)) (conj (Hv1 Hvars) Hv)).
    + intros [(Hh & Hrec & Ht) [Hvars Hv]].
      rewrite Hh, (Hv2 Hvars), (Hc2 (Hs2 (conj Ht Hv))), Hrec. reflexivity.
  - rewrite f_outer. simpl.
    split; [ intros _; exact (conj I I) | reflexivity ].
  - rewrite f_extra. simpl. destruct IH as [IH1 IH2]. split.
    + intros H. apply andb_true_iff in H as [Hc Hextra].
      destruct (IH1 Hc) as [Ht Hv].
      exact (conj Ht (conj (proj1 (forallb_Forall fv extra) Hextra) Hv)).
    + intros [Ht [Hextra Hv]].
      rewrite (IH2 (conj Ht Hv)), (proj2 (forallb_Forall fv extra) Hextra).
      reflexivity.
  - rewrite f_over. simpl.
    destruct IHreturn as [IHr1 IHr2]. destruct IHexn as [IHe1 IHe2]. split.
    + intros H. apply andb_true_iff in H as [H Hexn].
      apply andb_true_iff in H as [Hremaining Hreturn].
      destruct (IHr1 Hreturn) as [Htr Hvr].
      destruct (IHe1 Hexn) as [Hte Hve].
      exact
        (conj (conj Htr Hte)
           (conj (proj1 (forallb_Forall fv remaining) Hremaining)
              (conj Hvr Hve))).
    + intros [[Htr Hte] (Hremaining & Hvr & Hve)].
      rewrite (proj2 (forallb_Forall fv remaining) Hremaining),
        (IHr2 (conj Htr Hvr)), (IHe2 (conj Hte Hve)).
      reflexivity.
Qed.

Lemma conts_split_all (conts : cont_env) :
  forallb (fun '(_, c) => f c) conts = true
  <-> conts_terms_ok fh conts
      ∧ conts_values_all (fun v => fv v = true) conts.
Proof.
  destruct
    (conts_split f
       (fun c =>
          cont_terms_ok fh c ∧ cont_values_all (fun v => fv v = true) c)
       conts)
    as [Hc1 Hc2].
  { apply Forall_forall. intros kc _. apply cont_split. }
  destruct
    (fold_right_conj_split (cont_terms_ok fh)
       (cont_values_all (fun v => fv v = true)) conts)
    as [Hs1 Hs2].
  split; [ intros H; exact (Hs1 (Hc1 H)) | intros H; exact (Hc2 (Hs2 H)) ].
Qed.

End Cont_split.

Lemma conts_below_split (base next : symbol) (conts : cont_env) :
  forallb (fun '(_, c) => cont_below base next c) conts = true
  <-> conts_terms_ok (handler_below base) conts
      ∧ conts_values_all (fun v => value_below next v = true) conts.
Proof.
  apply (conts_split_all (handler_below base) (value_below next)
           (cont_below base next));
    reflexivity.
Qed.

Lemma conts_no_region_poison_split (conts : cont_env) :
  forallb (fun '(_, c) => cont_no_region_poison c) conts = true
  <-> conts_terms_ok handler_no_region_poison conts
      ∧ conts_values_all (fun v => value_no_region_poison v = true) conts.
Proof.
  apply (conts_split_all handler_no_region_poison value_no_region_poison
           cont_no_region_poison);
    reflexivity.
Qed.

Lemma forallb_pairs_true {A B : Type} (l : list (A * B)) :
  forallb (fun '(_, _) => true) l = true.
Proof. induction l as [ | [a b] l IH]; [ reflexivity | exact IH ]. Qed.

Lemma cont_terms_ok_true (c : cont_value) : cont_terms_ok (fun _ => true) c.
Proof.
  induction c
    as [h vars conts recursive IH | k | c extra IH
       | remaining remaining_arity return_arity return_to exn_to IHreturn
           IHexn]
    using cont_value_deep_ind;
    simpl.
  - split; [ reflexivity | ]. split; [ apply forallb_pairs_true | ].
    induction IH as [ | kc conts Hkc _ IHconts]; simpl;
      [ exact I | exact (conj Hkc IHconts) ].
  - exact I.
  - exact IH.
  - exact (conj IHreturn IHexn).
Qed.

Lemma conts_terms_ok_true (conts : cont_env) :
  conts_terms_ok (fun _ => true) conts.
Proof.
  unfold conts_terms_ok.
  induction conts as [ | kc conts IH]; simpl; [ exact I | ].
  exact (conj (cont_terms_ok_true (snd kc)) IH).
Qed.

(* [cont_kept] has no test of the handlers: it is the test of the
   values with one of the handlers that every handler passes. *)
Lemma conts_kept_split (ok : symbol -> bool) (conts : cont_env) :
  forallb (fun '(_, c) => cont_kept ok c) conts = true
  <-> conts_values_all (fun v => value_kept ok v = true) conts.
Proof.
  assert (Hsplit :
    forallb (fun '(_, c) => cont_kept ok c) conts = true
    <-> conts_terms_ok (fun _ => true) conts
        ∧ conts_values_all (fun v => value_kept ok v = true) conts).
  { apply (conts_split_all (fun _ => true) (value_kept ok) (cont_kept ok));
      try reflexivity.
    intros h vars conts' recursive. simpl.
    rewrite forallb_pairs_true, andb_true_r. reflexivity. }
  destruct Hsplit as [H1 H2]. split.
  - intros H. exact (proj2 (H1 H)).
  - intros H. exact (H2 (conj (conts_terms_ok_true conts) H)).
Qed.

(* ======================================================================== *)
(* The forms of a step                                                      *)
(* ======================================================================== *)

Lemma begin_region_step (rs rs' : region_state) (x : variable)
    (kind : flambda_kind) (mark_kind : mark_kind) (body : expr) :
  begin_region rs x kind mark_kind body = Region_step rs' ->
  rs'
  = bind_let rs x (Value_symbol (rs_next rs)) body
      (state_statics (rs_state rs))
      (Frame (rs_next rs) mark_kind [] :: rs_frames rs)
      (Pos.succ (rs_next rs)) (rs_counts rs).
Proof.
  unfold begin_region. destruct (kind_eqb kind Region); [ | discriminate ].
  intros [= <-]. reflexivity.
Qed.

Lemma end_region_step (rs rs' : region_state) (x : variable)
    (kind : flambda_kind) (ended_by : mark_kind) (region : simple)
    (body : expr) :
  end_region rs x kind ended_by region body = Region_step rs' ->
  ∃ (m : mark) (above : list frame) (f : frame) (below : list frame)
    (counts : region_counts),
    eval_simple (state_vars (rs_state rs)) region = Some (Value_symbol m)
    ∧ rs_frames rs = above ++ f :: below
    ∧ rs'
      = bind_let rs x unit_value body (state_statics (rs_state rs)) below
          (rs_next rs) counts.
Proof.
  unfold end_region. cbv zeta.
  destruct (kind_eqb kind Value); [ | discriminate ].
  destruct (eval_simple (state_vars (rs_state rs)) region) as [v | ];
    [ | discriminate ].
  destruct v as [ | | | m | | | | | ]; try discriminate.
  destruct (cut m (rs_frames rs)) as [[above below] | ] eqn:Hcut;
    [ | discriminate ].
  destruct (cut_app _ _ _ _ Hcut) as (f & Hframes).
  intros [= <-]. exists m, above, f, below. eexists.
  split; [ reflexivity | ]. split; [ exact Hframes | reflexivity ].
Qed.

Lemma local_allocation_step (rs rs' : region_state) (x y : variable)
    (kind : flambda_kind) (d : named) (region : option simple)
    (body : expr) :
  local_allocation rs x y kind d region body = Region_step rs' ->
  ∃ (v : value) (sym : symbol)
    (bindings : list (symbol * statically_allocated_value)),
    is_allocation d = true
    ∧ eval_named (live_statics rs) (state_vars (rs_state rs)) d = Some v
    ∧ localise (rs_next rs) v = Some (sym, bindings)
    ∧ state_expr (rs_state rs') = body
    ∧ state_vars (rs_state rs')
      = (y, Value_symbol sym) :: (x, Value_symbol sym)
        :: state_vars (rs_state rs)
    ∧ state_conts (rs_state rs') = state_conts (rs_state rs)
    ∧ rs_base rs' = rs_base rs
    ∧ rs_next rs' = advance (rs_next rs) (length bindings)
    ∧ static_code (state_statics (rs_state rs'))
      = static_code (state_statics (rs_state rs))
    ∧ match rs_frames rs with
      | [] =>
        rs_frames rs' = []
        ∧ static_values (state_statics (rs_state rs'))
          = bindings ++ static_values (state_statics (rs_state rs))
      | Frame m mark_kind locals :: frames =>
        rs_frames rs' = Frame m mark_kind (bindings ++ locals) :: frames
        ∧ state_statics (rs_state rs') = state_statics (rs_state rs)
      end.
Proof.
  unfold local_allocation. cbv zeta.
  destruct (is_allocation d && kind_eqb kind Value) eqn:Hcond;
    [ | discriminate ].
  apply andb_true_iff in Hcond as [His _].
  destruct (named_region rs region) as [other | ]; [ | discriminate ].
  destruct (eval_named (live_statics rs) (state_vars (rs_state rs)) d)
    as [v | ] eqn:Hnamed; [ | discriminate ].
  destruct (localise (rs_next rs) v) as [[sym bindings] | ] eqn:Hlocalise;
    [ | discriminate ].
  destruct (rs_frames rs) as [ | [m mark_kind locals] frames];
    intros [= <-]; exists v, sym, bindings;
    repeat split; try reflexivity; assumption.
Qed.

Lemma plain_step_step (rs rs' : region_state) :
  plain_step rs = Region_step rs' ->
  ∃ s' : state,
    step (with_statics (rs_state rs) (live_statics rs)) = Step s'
    ∧ rs'
      = Region_state (with_statics s' (state_statics (rs_state rs)))
          (rs_frames rs) (rs_base rs) (rs_next rs) (rs_counts rs).
Proof.
  unfold plain_step.
  destruct (step (with_statics (rs_state rs) (live_statics rs)))
    as [s' | k vs hp rd | ]; [ | discriminate | discriminate ].
  intros [= <-]. exists s'. split; reflexivity.
Qed.

Lemma static_step_step (rs rs' : region_state)
    (bindings : list static_binding) (body : expr) :
  state_expr (rs_state rs) = Let_static bindings body ->
  static_step rs = Region_step rs' ->
  ∃ statics' : static_env,
    eval_static_bindings (state_vars (rs_state rs)) bindings
      (state_statics (rs_state rs))
    = Some statics'
    ∧ rs'
      = Region_state
          (State body (state_vars (rs_state rs)) (state_conts (rs_state rs))
             statics' (state_heap (rs_state rs)) (state_reads (rs_state rs)))
          (rs_frames rs) (rs_base rs) (rs_next rs) (rs_counts rs).
Proof.
  intros He. unfold static_step.
  destruct (rs_state rs) as [e vars conts statics hp rd]. simpl in He. subst e.
  rewrite step_let_static. simpl.
  destruct (eval_static_bindings vars bindings statics) as [statics' | ];
    [ | discriminate ].
  intros [= <-]. exists statics'. split; reflexivity.
Qed.

(* The test of the simples, where none is at the place of a region. *)
Lemma simples_fault_no_place (rs : region_state) (simples : list simple) :
  simples_fault rs simples [] = None ->
  forallb (simple_usable rs) simples = true.
Proof.
  induction simples as [ | s simples IH]; simpl; [ reflexivity | ].
  destruct (simple_usable rs s); [ exact IH | ].
  destruct (simple_open_mark rs s); discriminate.
Qed.

(* One traversal of [region_step]: what is to be shown of each form of
   a step, to have it of the step. The state that the helpers are given
   is [count_uses rs], which differs from [rs] by its counts. *)
Section Region_step_cases.

Variable rs : region_state.
Variable Q : region_step_result -> Prop.

Hypothesis Q_undefined : ∀ fault : region_fault, Q (Region_undefined fault).
Hypothesis Q_let_not_run :
  ∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
    state_expr (rs_state rs) = Let x kind d body ->
    named_needs_heap (state_vars (rs_state rs)) d = true ->
    forallb (simple_usable rs) (simples_of_named d) = true ->
    Q Region_let_not_run.
Hypothesis Q_begin :
  ∀ (x : variable) (kind : flambda_kind) (d : named) (mark_kind : mark_kind)
    (body : expr),
    state_expr (rs_state rs) = Let x kind d body ->
    Q (begin_region (count_uses rs) x kind mark_kind body).
Hypothesis Q_end :
  ∀ (x : variable) (kind : flambda_kind) (d : named) (ended_by : mark_kind)
    (region : simple) (body : expr),
    state_expr (rs_state rs) = Let x kind d body ->
    Q (end_region (count_uses rs) x kind ended_by region body).
Hypothesis Q_local :
  ∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr)
    (y : variable) (region : option simple) (body' : expr),
    state_expr (rs_state rs) = Let x kind d body ->
    named_needs_heap (state_vars (rs_state rs)) d = false ->
    marker_of x body = Marker y region body' ->
    forallb (simple_usable rs) (simples_of_named d) = true ->
    Q (local_allocation (count_uses rs) x y kind d region body').
Hypothesis Q_let :
  ∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
    state_expr (rs_state rs) = Let x kind d body ->
    named_needs_heap (state_vars (rs_state rs)) d = false ->
    forallb (simple_usable rs) (simples_of_named d) = true ->
    is_allocation d && existsb (simple_local rs) (simples_of_named d)
    = false ->
    Q (plain_step (count_uses rs)).
Hypothesis Q_static :
  ∀ (bindings : list static_binding) (body : expr),
    state_expr (rs_state rs) = Let_static bindings body ->
    forallb (simple_usable rs) (flat_map simples_of_binding bindings)
    = true ->
    existsb (simple_local rs) (flat_map simples_of_binding bindings)
    = false ->
    Q (static_step (count_uses rs)).
Hypothesis Q_region_apply_cont :
  ∀ ac : apply_cont,
    state_expr (rs_state rs) = Apply_cont ac ->
    Q (region_apply_cont (count_uses rs) ac).
Hypothesis Q_plain :
  (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
    state_expr (rs_state rs) <> Let x kind d body) ->
  (∀ (bindings : list static_binding) (body : expr),
    state_expr (rs_state rs) <> Let_static bindings body) ->
  Q (plain_step (count_uses rs)).

Lemma region_step_cases : Q (region_step rs).
Proof.
  unfold region_step. cbv zeta.
  destruct
    (simples_fault rs (evaluated_simples (state_expr (rs_state rs)))
       (region_places rs))
    eqn:Hfault; [ apply Q_undefined | ].
  unfold region_places in Hfault.
  destruct (state_expr (rs_state rs))
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ] eqn:He;
    try (apply Q_plain; intros; discriminate).
  - assert (Hrest :
              forallb (simple_usable rs) (simples_of_named d) = true ->
              Q (if named_needs_heap (state_vars (rs_state (count_uses rs))) d
                 then Region_let_not_run
                 else
                   match marker_of x body with
                   | Marker y region body' =>
                     local_allocation (count_uses rs) x y kind d region body'
                   | Bad_marker => Region_undefined Not_of_the_conversion
                   | No_marker =>
                     if is_allocation d
                        && existsb (simple_local (count_uses rs))
                             (simples_of_named d)
                     then Region_undefined Heap_holds_local
                     else plain_step (count_uses rs)
                   end)).
    { intros Husable.
      destruct (named_needs_heap (state_vars (rs_state (count_uses rs))) d)
        eqn:Hneeds;
        [ exact (Q_let_not_run x kind d body eq_refl Hneeds Husable) | ].
      destruct (marker_of x body) as [ | y region body' | ] eqn:Hmarker.
      - destruct
          (is_allocation d
           && existsb (simple_local (count_uses rs)) (simples_of_named d))
          eqn:Hheap; [ apply Q_undefined | ].
        exact (Q_let x kind d body eq_refl Hneeds Husable Hheap).
      - exact
          (Q_local x kind d body y region body' eq_refl Hneeds Hmarker
             Husable).
      - apply Q_undefined. }
    destruct d as [s | [u | b | v | hp | rp] args | | | | | ];
      try (apply Hrest; exact (simples_fault_no_place rs _ Hfault)).
    destruct rp, args as [ | arg [ | arg' args]];
      first
        [ apply Q_undefined
        | exact (Q_begin x kind _ _ body eq_refl)
        | exact (Q_end x kind _ _ _ body eq_refl) ].
  - destruct
      (existsb (simple_local (count_uses rs))
         (evaluated_simples (Let_static bindings body)))
      eqn:Hlocal; [ apply Q_undefined | ].
    exact
      (Q_static bindings body eq_refl (simples_fault_no_place rs _ Hfault)
         Hlocal).
  - destruct (existsb (simple_open_mark (count_uses rs)) (ac_args ac));
      [ exact (Q_region_apply_cont ac eq_refl) | ].
    apply Q_plain; intros; discriminate.
Qed.

End Region_step_cases.

(* ======================================================================== *)
(* What the forms of a step leave, and the state of an exit                 *)
(* ======================================================================== *)

(* The simples that the step of a [Let] evaluates are among those of its
   defining expression. *)
Lemma evaluated_simples_let (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (sim : simple) :
  In sim (evaluated_simples (Let x kind d body)) -> In sim (simples_of_named d).
Proof.
  destruct d as [ | [ | | | | ] args | | | | | ]; simpl; auto; contradiction.
Qed.

(* The arm that a [Switch] takes is one of its arms. *)
Lemma find_arm_some_in (i : Z) (arms : list (Z * apply_cont))
    (ac : apply_cont) :
  find_arm i arms = Some ac -> ∃ i' : Z, In (i', ac) arms.
Proof.
  induction arms as [ | [i' ac'] arms IH]; simpl; [ discriminate | ].
  destruct (Z.eqb i i').
  - intros [= <-]. exists i'. left. reflexivity.
  - intros H. destruct (IH H) as (i0 & Hin). exists i0. right. exact Hin.
Qed.

(* What follows the [Let] of a local allocation. *)
Lemma marker_of_marker (x y : variable) (region : option simple)
    (body body' : expr) :
  marker_of x body = Marker y region body' ->
  ∃ (kind : flambda_kind) (args : list simple),
    body = Let y kind (Prim (Region_primitive Local_allocation) args) body'.
Proof.
  unfold marker_of.
  destruct body as [y0 kind d body0 | | | | | | | ]; try discriminate.
  destruct d as [ | p args | | | | | ]; try discriminate.
  destruct p as [ | | | | rp]; try discriminate.
  destruct rp; try discriminate.
  destruct args as [ | [x' | | ] rest]; try discriminate.
  destruct (Pos.eqb x x' && kind_eqb kind Value); [ | discriminate ].
  destruct rest as [ | r [ | r' rest]].
  - intros [= <- <- <-]. eexists _, _. reflexivity.
  - destruct r as [z | sym | c];
      [ destruct (Pos.eqb z x); [ discriminate | ] | | ];
      intros [= <- <- <-]; eexists _, _; reflexivity.
  - destruct r; discriminate.
Qed.

(* The same, with the arguments of the [Local_allocation] and the kind
   of its [Let]. *)
Lemma marker_of_marker_args (x y : variable) (region : option simple)
    (body body' : expr) :
  marker_of x body = Marker y region body' ->
  ∃ kind : flambda_kind,
    body
    = Let y kind
        (Prim (Region_primitive Local_allocation)
           (Var x :: match region with Some r => [r] | None => [] end))
        body'
    ∧ kind_eqb kind Value = true.
Proof.
  unfold marker_of.
  destruct body as [y0 kind d body0 | | | | | | | ]; try discriminate.
  destruct d as [ | p args | | | | | ]; try discriminate.
  destruct p as [ | | | | rp]; try discriminate.
  destruct rp; try discriminate.
  destruct args as [ | [x' | | ] rest]; try discriminate.
  destruct (Pos.eqb x x' && kind_eqb kind Value) eqn:Hcond;
    [ | discriminate ].
  apply andb_true_iff in Hcond as [Hx Hkind]. apply Pos.eqb_eq in Hx.
  subst x'. destruct rest as [ | r [ | r' rest]].
  - intros [= <- <- <-]. exists kind. split; [ reflexivity | exact Hkind ].
  - destruct r as [z | sym | c];
      [ destruct (Pos.eqb z x); [ discriminate | ] | | ];
      intros [= <- <- <-]; exists kind;
      (split; [ reflexivity | exact Hkind ]).
  - destruct r; discriminate.
Qed.

Lemma marker_of_marker_var (x y : variable) (region : option simple)
    (body body' : expr) :
  marker_of x body = Marker y region body' ->
  ∃ (kind : flambda_kind) (rest : list simple),
    body
    = Let y kind (Prim (Region_primitive Local_allocation) (Var x :: rest))
        body'
    ∧ kind_eqb kind Value = true.
Proof.
  intros Hmarker.
  destruct (marker_of_marker_args x y region body body' Hmarker)
    as (kind & Hbody & Hkind).
  exact (ex_intro _ kind (ex_intro _ _ (conj Hbody Hkind))).
Qed.

(* A local allocation is of the kind [Value], and leaves the heap of
   the state. *)
Lemma local_allocation_kind_heap (rs rs' : region_state) (x y : variable)
    (kind : flambda_kind) (d : named) (region : option simple)
    (body : expr) :
  local_allocation rs x y kind d region body = Region_step rs' ->
  kind_eqb kind Value = true
  ∧ state_heap (rs_state rs') = state_heap (rs_state rs).
Proof.
  unfold local_allocation. cbv zeta.
  destruct (is_allocation d && kind_eqb kind Value) eqn:Hcond;
    [ | discriminate ].
  apply andb_true_iff in Hcond as [_ Hkind].
  destruct (named_region rs region) as [other | ]; [ | discriminate ].
  destruct (eval_named (live_statics rs) (state_vars (rs_state rs)) d)
    as [v | ]; [ | discriminate ].
  destruct (localise (rs_next rs) v) as [[sym bindings] | ];
    [ | discriminate ].
  destruct (rs_frames rs) as [ | [m mark_kind locals] frames];
    intros [= <-]; (split; [ exact Hkind | reflexivity ]).
Qed.

(* And the read record. *)
Lemma local_allocation_reads (rs rs' : region_state) (x y : variable)
    (kind : flambda_kind) (d : named) (region : option simple)
    (body : expr) :
  local_allocation rs x y kind d region body = Region_step rs' ->
  state_reads (rs_state rs') = state_reads (rs_state rs).
Proof.
  unfold local_allocation. cbv zeta.
  destruct (is_allocation d && kind_eqb kind Value) eqn:Hcond;
    [ | discriminate ].
  destruct (named_region rs region) as [other | ]; [ | discriminate ].
  destruct (eval_named (live_statics rs) (state_vars (rs_state rs)) d)
    as [v | ]; [ | discriminate ].
  destruct (localise (rs_next rs) v) as [[sym bindings] | ];
    [ | discriminate ].
  destruct (rs_frames rs) as [ | [m mark_kind locals] frames];
    intros [= <-]; reflexivity.
Qed.

Lemma region_apply_cont_step (rs rs' : region_state) (ac : apply_cont) :
  region_apply_cont rs ac = Region_step rs' ->
  rs_frames rs' = rs_frames rs
  ∧ rs_base rs' = rs_base rs
  ∧ rs_next rs' = rs_next rs
  ∧ state_statics (rs_state rs') = state_statics (rs_state rs).
Proof.
  assert (Hplain :
    plain_step rs = Region_step rs' ->
    rs_frames rs' = rs_frames rs
    ∧ rs_base rs' = rs_base rs
    ∧ rs_next rs' = rs_next rs
    ∧ state_statics (rs_state rs') = state_statics (rs_state rs)).
  { intros Hstep. destruct (plain_step_step rs rs' Hstep) as (s' & _ & ->).
    repeat split; reflexivity. }
  unfold region_apply_cont.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [[params metadata handler] def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    try exact Hplain.
  destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac)) as [vs | ];
    [ | discriminate ].
  destruct (kinds_eqb _ _); [ | discriminate ].
  destruct (bind_params params vs def_vars) as [handler_vars | ];
    [ | discriminate ].
  intros [= <-]. repeat split; reflexivity.
Qed.

Lemma plain_step_exit (rs rs' : region_state) (k : continuation)
    (vs : list value) :
  plain_step rs = Region_exit k vs rs' -> rs' = rs.
Proof.
  unfold plain_step.
  destruct (step (with_statics (rs_state rs) (live_statics rs)))
    as [s' | k' vs' hp rd | ]; [ discriminate | | discriminate ].
  intros [= _ _ <-]. reflexivity.
Qed.

Lemma static_step_exit (rs rs' : region_state) (k : continuation)
    (vs : list value) :
  static_step rs = Region_exit k vs rs' -> rs' = rs.
Proof.
  unfold static_step.
  destruct (step (rs_state rs)) as [s' | k' vs' hp rd | ];
    [ discriminate | | discriminate ].
  intros [= _ _ <-]. reflexivity.
Qed.

(* The step of a [Let_static] is no exit. *)
Lemma static_step_no_exit (rs : region_state)
    (bindings : list static_binding) (body : expr) (k : continuation)
    (vs : list value) (rs' : region_state) :
  state_expr (rs_state rs) = Let_static bindings body ->
  static_step rs <> Region_exit k vs rs'.
Proof.
  intros He. unfold static_step.
  destruct (rs_state rs) as [e vars conts statics hp rd]. simpl in He. subst e.
  rewrite step_let_static.
  destruct (eval_static_bindings vars bindings statics); discriminate.
Qed.

Lemma region_apply_cont_exit (rs rs' : region_state) (ac : apply_cont)
    (k : continuation) (vs : list value) :
  region_apply_cont rs ac = Region_exit k vs rs' -> rs' = rs.
Proof.
  unfold region_apply_cont.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [[params metadata handler] def_vars def_conts recursive | k0 | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    try apply plain_step_exit.
  destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac)) as [vs0 | ];
    [ | discriminate ].
  destruct (kinds_eqb _ _); [ | discriminate ].
  destruct (bind_params params vs0 def_vars); discriminate.
Qed.

Lemma local_allocation_no_exit (rs : region_state) (x y : variable)
    (kind : flambda_kind) (d : named) (region : option simple) (body : expr)
    (k : continuation) (vs : list value) (rs' : region_state) :
  local_allocation rs x y kind d region body <> Region_exit k vs rs'.
Proof.
  unfold local_allocation. cbv zeta.
  destruct (is_allocation d && kind_eqb kind Value); [ | discriminate ].
  destruct (named_region rs region); [ | discriminate ].
  destruct (eval_named (live_statics rs) (state_vars (rs_state rs)) d)
    as [v | ]; [ | discriminate ].
  destruct (localise (rs_next rs) v) as [[sym bindings] | ];
    [ | discriminate ].
  destruct (rs_frames rs) as [ | [m mark_kind locals] frames]; discriminate.
Qed.

(* The state of an exit is the state that was left, with its uses
   counted. *)
Lemma region_step_exit (rs rs' : region_state) (k : continuation)
    (vs : list value) :
  region_step rs = Region_exit k vs rs' -> rs' = count_uses rs.
Proof.
  apply
    (region_step_cases rs
       (fun result => result = Region_exit k vs rs' -> rs' = count_uses rs)).
  - discriminate.
  - discriminate.
  - intros x kind d mark_kind body _. unfold begin_region.
    destruct (kind_eqb kind Region); discriminate.
  - intros x kind d ended_by region body _. unfold end_region. cbv zeta.
    destruct (kind_eqb kind Value); [ | discriminate ].
    destruct (eval_simple _ region) as [v | ]; [ | discriminate ].
    destruct v; try discriminate.
    destruct (cut _ _) as [[above below] | ]; discriminate.
  - intros x kind d body y region body' _ _ _ _ Hexit.
    contradiction (local_allocation_no_exit _ _ _ _ _ _ _ _ _ _ Hexit).
  - intros x kind d body _ _ _ _. apply plain_step_exit.
  - intros bindings body _ _ _. apply static_step_exit.
  - intros ac _. apply region_apply_cont_exit.
  - intros _ _. apply plain_step_exit.
Qed.

(* Where the machine does not run a step, the state is at a [Let]
   that needs the heap, whose simples may be evaluated (decision
   D53). *)
Lemma region_step_let_not_run (rs : region_state) :
  region_step rs = Region_let_not_run ->
  ∃ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
    state_expr (rs_state rs) = Let x kind d body
    ∧ named_needs_heap (state_vars (rs_state rs)) d = true
    ∧ forallb (simple_usable rs) (simples_of_named d) = true.
Proof.
  apply
    (region_step_cases rs
       (fun result =>
          result = Region_let_not_run ->
          ∃ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
            state_expr (rs_state rs) = Let x kind d body
            ∧ named_needs_heap (state_vars (rs_state rs)) d = true
            ∧ forallb (simple_usable rs) (simples_of_named d) = true)).
  - discriminate.
  - intros x kind d body He Hneeds Husable _. exists x, kind, d, body. auto.
  - intros x kind d mark_kind body _. unfold begin_region.
    destruct (kind_eqb kind Region); discriminate.
  - intros x kind d ended_by region body _. unfold end_region. cbv zeta.
    destruct (kind_eqb kind Value); [ | discriminate ].
    destruct (eval_simple _ region) as [v | ]; [ | discriminate ].
    destruct v; try discriminate.
    destruct (cut _ _) as [[above below] | ]; discriminate.
  - intros x kind d body y region body' _ _ _ _. unfold local_allocation.
    cbv zeta.
    destruct (is_allocation d && kind_eqb kind Value); [ | discriminate ].
    destruct (named_region _ region); [ | discriminate ].
    destruct (eval_named _ _ d); [ | discriminate ].
    destruct (localise _ _) as [[sym bindings] | ]; [ | discriminate ].
    destruct (rs_frames _) as [ | [m mark_kind locals] frames]; discriminate.
  - intros x kind d body _ _ _ _. unfold plain_step.
    destruct (step _); discriminate.
  - intros bindings body _ _ _. unfold static_step.
    destruct (step _); discriminate.
  - intros ac _. unfold region_apply_cont, plain_step.
    destruct (resolve _ _) as [[params metadata handler] ? ? ? | | | ];
      try (destruct (step _); discriminate).
    destruct (eval_simples _ _); [ | discriminate ].
    destruct (kinds_eqb _ _); [ | discriminate ].
    destruct (bind_params _ _ _); discriminate.
  - intros _ _. unfold plain_step. destruct (step _); discriminate.
Qed.

(* ======================================================================== *)
(* The terms of a state after a step of the machine of regions              *)
(* ======================================================================== *)

(* The tests of FlambdaRegionsInvariantSpec.v ask a test of all the code
   of the static environment, shadowed or not, where
   FlambdaClosuresProofs.code_terms_ok asks it of the code that a lookup
   finds. *)
Section Region_step_written.

Variable Pe : expr -> bool.
Variable Ph : continuation_handler -> bool.
Variable Pc : code -> bool.

Hypothesis Pe_let :
  ∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
    Pe (Let x kind d body) = true -> Pe body = true.
Hypothesis Pe_let_static :
  ∀ (bindings : list static_binding) (body : expr),
    Pe (Let_static bindings body) = true ->
    Pe body = true
    ∧ ∀ (id : code_id) (c : code), In (Bind_code id c) bindings -> Pc c = true.
Hypothesis Pe_let_cont :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    Pe (Let_cont_non_recursive k h body) = true ->
    Ph h = true ∧ Pe body = true.
Hypothesis Pe_let_cont_rec :
  ∀ (handlers : list (continuation * continuation_handler)) (body : expr),
    Pe (Let_cont_recursive handlers body) = true ->
    forallb (fun '(_, h) => Ph h) handlers = true ∧ Pe body = true.
Hypothesis Pe_switch :
  ∀ (scrutinee : simple) (arms : list (Z * apply_cont)) (i : Z)
    (ac : apply_cont),
    Pe (Switch scrutinee arms) = true ->
    find_arm i arms = Some ac ->
    Pe (Apply_cont ac) = true.
Hypothesis Ph_handler :
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr),
    Ph (Handler params metadata handler) = true -> Pe handler = true.
Hypothesis Pc_body : ∀ c : code, Pc c = true -> Pe (code_body c) = true.

(* What a state has written passes the tests: its expression, the
   handlers that its continuations hold, the code of its static
   environment. *)
Definition written_ok (s : state) : Prop :=
  Pe (state_expr s) = true
  ∧ conts_terms_ok Ph (state_conts s)
  ∧ forallb (fun '(_, c) => Pc c) (static_code (state_statics s)) = true.

Lemma written_terms_inv (s : state) : written_ok s -> terms_inv Pe Ph Pc s.
Proof.
  intros (He & Hconts & Hcode). refine (conj He (conj Hconts _)).
  intros id c Hlookup.
  exact (forallb_in _ _ (id, c) Hcode (lookup_in _ _ _ Hlookup)).
Qed.

Lemma plain_step_written (rs rs' : region_state) :
  written_ok (rs_state rs) ->
  plain_step rs = Region_step rs' ->
  written_ok (rs_state rs').
Proof.
  intros Hw Hstep.
  destruct
    (plain_step_terms Pe Ph Pc Pe_let Pe_let_static Pe_let_cont
       Pe_let_cont_rec Pe_switch Ph_handler Pc_body rs rs'
       (written_terms_inv _ Hw) Hstep)
    as (He & Hconts & _).
  destruct (plain_step_step rs rs' Hstep) as (s' & _ & ->).
  exact (conj He (conj Hconts (proj2 (proj2 Hw)))).
Qed.

Lemma static_step_written (rs rs' : region_state)
    (bindings : list static_binding) (body : expr) :
  state_expr (rs_state rs) = Let_static bindings body ->
  written_ok (rs_state rs) ->
  static_step rs = Region_step rs' ->
  written_ok (rs_state rs').
Proof.
  intros He Hw Hstep.
  destruct
    (static_step_terms Pe Ph Pc Pe_let Pe_let_static Pe_let_cont
       Pe_let_cont_rec Pe_switch Ph_handler Pc_body rs rs'
       (written_terms_inv _ Hw) Hstep)
    as (He' & Hconts & _).
  refine (conj He' (conj Hconts _)).
  destruct (static_step_step rs rs' bindings body He Hstep)
    as (statics' & Heval & ->).
  destruct (eval_static_bindings_parts _ _ _ _ Heval)
    as (new_code & new_values & Hcode & _ & Hnew & _).
  cbn [rs_state state_statics]. rewrite Hcode, forallb_app.
  destruct Hw as (HPe & _ & Hold). rewrite Hold, andb_true_r.
  rewrite He in HPe. destruct (Pe_let_static bindings body HPe) as [_ Hbound].
  apply forallb_forall. intros [id c] Hin. exact (Hbound id c (Hnew id c Hin)).
Qed.

Lemma region_apply_cont_written (rs rs' : region_state) (ac : apply_cont) :
  written_ok (rs_state rs) ->
  region_apply_cont rs ac = Region_step rs' ->
  written_ok (rs_state rs').
Proof.
  intros Hw Hstep.
  destruct
    (region_apply_cont_terms Pe Ph Pc Pe_let Pe_let_static Pe_let_cont
       Pe_let_cont_rec Pe_switch Ph_handler Pc_body rs rs' ac
       (written_terms_inv _ Hw) Hstep)
    as (He & Hconts & _).
  destruct (region_apply_cont_step rs rs' ac Hstep) as (_ & _ & _ & Hstatics).
  refine (conj He (conj Hconts _)). rewrite Hstatics.
  exact (proj2 (proj2 Hw)).
Qed.

(* A step keeps the tests of what is written. *)
Lemma region_step_written (rs rs' : region_state) :
  written_ok (rs_state rs) ->
  region_step rs = Region_step rs' ->
  written_ok (rs_state rs').
Proof.
  intros Hw. pose proof Hw as (HPe & Hconts & Hcode).
  apply
    (region_step_cases rs
       (fun result => result = Region_step rs' -> written_ok (rs_state rs'))).
  - discriminate.
  - discriminate.
  - intros x kind d mark_kind body He Hstep.
    rewrite (begin_region_step _ _ _ _ _ _ Hstep). rewrite He in HPe.
    exact (conj (Pe_let x kind d body HPe) (conj Hconts Hcode)).
  - intros x kind d ended_by region body He Hstep.
    destruct (end_region_step _ _ _ _ _ _ _ Hstep)
      as (m & above & f & below & counts & _ & _ & ->).
    rewrite He in HPe.
    exact (conj (Pe_let x kind d body HPe) (conj Hconts Hcode)).
  - intros x kind d body y region body' He _ Hmarker _ Hstep.
    destruct (local_allocation_step _ _ _ _ _ _ _ _ Hstep)
      as (v & sym & bindings & _ & _ & _ & Hexpr & _ & Hconts' & _ & _
          & Hcode' & _).
    destruct (marker_of_marker x y region body body' Hmarker)
      as (kind' & args & ->).
    rewrite He in HPe. unfold written_ok. rewrite Hexpr, Hconts', Hcode'.
    exact (conj (Pe_let _ _ _ _ (Pe_let _ _ _ _ HPe)) (conj Hconts Hcode)).
  - intros x kind d body _ _ _ _.
    exact (plain_step_written (count_uses rs) rs' Hw).
  - intros bindings body He _ _.
    exact (static_step_written (count_uses rs) rs' bindings body He Hw).
  - intros ac _. exact (region_apply_cont_written (count_uses rs) rs' ac Hw).
  - intros _ _. exact (plain_step_written (count_uses rs) rs' Hw).
Qed.

End Region_step_written.

(* ======================================================================== *)
(* The values of a state after a step of the machine of regions             *)
(* ======================================================================== *)

Section Region_step_values.

Variable P : value -> Prop.

Hypothesis P_partial :
  ∀ (callee : value) (args : list value) (args_arity : arity),
    P (Value_partial_application callee args args_arity)
    <-> P callee ∧ Forall P args.

(* What is shown of a step: the values of the state that it comes to
   have the property, if the symbols that it drew have it; and so have
   the values of an exit. *)
Definition region_step_vals_ok (rs : region_state)
    (result : region_step_result) : Prop :=
  match result with
  | Region_step rs' =>
    (∀ s : symbol,
      (rs_next rs <= s)%positive ->
      (s < rs_next rs')%positive ->
      P (Value_symbol s)) ->
    vals_ok P (rs_state rs')
  | Region_exit _ vs _ => Forall P vs
  | Region_undefined _ | Region_let_not_run => True
  end.

(* The value of a [Let] is asked the property where the [Let] is run by
   [step]: it is not a local allocation, its simples may be evaluated,
   and if it allocates none of them is local (decision D46). *)
Lemma region_step_values (rs : region_state) :
  P_fields P (live_statics rs) ->
  vals_ok P (rs_state rs) ->
  (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr)
     (v : value),
    state_expr (rs_state rs) = Let x kind d body ->
    forallb (simple_usable rs) (simples_of_named d) = true ->
    is_allocation d && existsb (simple_local rs) (simples_of_named d)
    = false ->
    eval_named (live_statics rs) (state_vars (rs_state rs)) d = Some v ->
    P v) ->
  (∀ (sim : simple) (v : value),
    In sim (evaluated_simples (state_expr (rs_state rs))) ->
    eval_simple (state_vars (rs_state rs)) sim = Some v ->
    P v) ->
  P unit_value ->
  region_step_vals_ok rs (region_step rs).
Proof.
  intros Hfields Hvals Hlet Hsimples Hunit.
  pose proof Hvals as [Hvars Hconts].
  apply (region_step_cases rs (region_step_vals_ok rs)).
  - intros fault. exact I.
  - intros x kind d body _ _ _. exact I.
  - intros x kind d mark_kind body _. unfold begin_region.
    destruct (kind_eqb kind Region); [ | exact I ].
    intros Hdrawn.
    assert (Hmark : P (Value_symbol (rs_next rs))).
    { apply Hdrawn; cbn [bind_let rs_next count_uses with_counts]; lia. }
    exact (conj (Forall_cons (x, Value_symbol (rs_next rs)) Hmark Hvars)
             Hconts).
  - intros x kind d ended_by region body _. unfold end_region. cbv zeta.
    destruct (kind_eqb kind Value); [ | exact I ].
    destruct (eval_simple _ region) as [v | ]; [ | exact I ].
    destruct v; try exact I.
    destruct (cut _ _) as [[above below] | ]; [ | exact I ].
    intros _. exact (conj (Forall_cons (x, unit_value) Hunit Hvars) Hconts).
  - intros x kind d body y region body' _ _ _ _.
    destruct (local_allocation (count_uses rs) x y kind d region body')
      as [rs' | k vs rs' | fault | ] eqn:Hlocal; [ | | exact I | exact I ].
    + destruct (local_allocation_step _ _ _ _ _ _ _ _ Hlocal)
        as (v & sym & bindings & _ & _ & Hlocalise & _ & Hvars' & Hconts'
            & _ & Hnext & _).
      destruct (localise_makes _ _ _ _ Hlocalise)
        as (_ & [allocated Hin] & Hbound).
      destruct (Hbound sym allocated Hin) as (Hfrom & Hbelow & _).
      intros Hdrawn.
      assert (Hsym : P (Value_symbol sym)).
      { apply Hdrawn; [ exact Hfrom | rewrite Hnext; exact Hbelow ]. }
      split.
      * rewrite Hvars'.
        exact
          (Forall_cons (y, Value_symbol sym) Hsym
             (Forall_cons (x, Value_symbol sym) Hsym Hvars)).
      * rewrite Hconts'. exact Hconts.
    + contradiction (local_allocation_no_exit _ _ _ _ _ _ _ _ _ _ Hlocal).
  - intros x kind d body He Hneeds Husable Hheap.
    assert (Hplain : region_result_vals_ok P (plain_step (count_uses rs))).
    { apply (plain_step_values P P_partial (count_uses rs) Hfields Hvals);
        [ | | exact Hsimples ].
      - intros x' kind' d' body0 He'.
        change (state_expr (rs_state (count_uses rs)))
          with (state_expr (rs_state rs)) in He'.
        rewrite He in He'. injection He' as <- <- <- <-. exact Hneeds.
      - intros x' kind' d' body0 v He' Hnamed.
        change (state_expr (rs_state (count_uses rs)))
          with (state_expr (rs_state rs)) in He'.
        rewrite He in He'. injection He' as <- <- <- <-.
        exact (Hlet x kind d body v He Husable Hheap Hnamed). }
    destruct (plain_step (count_uses rs)) as [rs' | k vs rs' | fault | ];
      [ intros _; exact Hplain | exact Hplain | exact I | exact I ].
  - intros bindings body He _ _.
    destruct (static_step (count_uses rs)) as [rs' | k vs rs' | fault | ]
      eqn:Hstatic; [ | | exact I | exact I ].
    + destruct (static_step_step (count_uses rs) _ bindings body He Hstatic)
        as (statics' & _ & ->).
      intros _. exact (conj Hvars Hconts).
    + contradiction
        (static_step_no_exit (count_uses rs) bindings body _ _ _ He Hstatic).
  - intros ac He.
    assert
      (Hac : region_result_vals_ok P (region_apply_cont (count_uses rs) ac)).
    { apply
        (region_apply_cont_values P P_partial (count_uses rs) ac He Hfields
           Hvals).
      intros sim v Hin. apply Hsimples. rewrite He. exact Hin. }
    destruct (region_apply_cont (count_uses rs) ac)
      as [rs' | k vs rs' | fault | ];
      [ intros _; exact Hac | exact Hac | exact I | exact I ].
  - intros Hnot_let _.
    assert (Hplain : region_result_vals_ok P (plain_step (count_uses rs))).
    { apply (plain_step_values P P_partial (count_uses rs) Hfields Hvals);
        [ | | exact Hsimples ].
      - intros x kind d body He. contradiction (Hnot_let x kind d body He).
      - intros x kind d body v He.
        contradiction (Hnot_let x kind d body He). }
    destruct (plain_step (count_uses rs)) as [rs' | k vs rs' | fault | ];
      [ intros _; exact Hplain | exact Hplain | exact I | exact I ].
Qed.

End Region_step_values.
