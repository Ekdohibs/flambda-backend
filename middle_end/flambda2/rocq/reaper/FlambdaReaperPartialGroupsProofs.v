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

(* A run invariant: every partial application that a state holds was
   made by the machine, so it holds fewer groups of arguments than the
   code of its closure has parameters ([pa_bound]). Statics only grow,
   with fresh keys, so the bound is stable ([pa_ok_extend]).
   FlambdaSemanticsProofs.step_values_all asks [P] of a partial
   application to be exactly [P] of its closure and its arguments; its
   call lemmas are copied here with the bound given where
   [shape_of_code_call] makes a [Partial_call] ([step_values_bound]).
   So every state of a run satisfies [pa_inv] ([pa_inv_run]), and a call
   with one group of arguments of a value of such a state, of code that
   has a parameter, is no over-application ([one_group_no_over]).
   What a step over-applies, by a call or by what is left of an
   over-application, a partial application that it makes going to a
   continuation that may over-apply in turn ([step_over_ok]), and the code
   that what is left of an over-application calls ([step_rem_ok]), are said
   here for the simulations; FlambdaReaperFlowProofs.step_over_used and
   step_rem_used prove them of the run of a unit. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaHeapSpec FlambdaWellFormedSpec.
From OxCamlFlambda2 Require FlambdaWellFormedProofs FlambdaPrimitiveProofs
  FlambdaHeapProofs.
Import ListNotations.

(* ======================================================================== *)
(* The call lemmas, with the bound at a partial application                 *)
(* ======================================================================== *)

Section Values_bound.

Variable P : value -> Prop.
Variable statics : static_env.

Hypothesis P_partial :
  ∀ (callee : value) (args : list value) (args_arity : arity),
    P (Value_partial_application callee args args_arity) ->
    P callee ∧ Forall P args.

(* A partial application that the machine makes: of a closure of bound
   code that has more parameters than the groups given. *)
Hypothesis P_partial_made :
  ∀ (callee : value) (called : code_id) (slots : list (value_slot * value))
    (c : code) (args : list value) (args_arity : arity),
    closure_of statics callee = Some (called, slots) ->
    lookup called (static_code statics) = Some c ->
    length args_arity < length (code_params_arity c) ->
    P callee -> Forall P args ->
    P (Value_partial_application callee args args_arity).

Lemma shape_of_closure_call_values_bound (closure : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity) :
  P_fields P statics ->
  P closure -> Forall P vs ->
  shape_values_all P
    (shape_of_closure_call statics closure kind vs args_arity return_arity).
Proof.
  intros Hfields Hclosure Hvs.
  destruct
    (shape_of_closure_call statics closure kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape; [ | exact I ].
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
    as (called & slots & c & vs' & args_arity' & Hcl & Hc & _ & Hargs
        & Hcall).
  pose proof (arguments_for_code_forall P _ _ _ _ _ _ Hfields Hvs Hargs)
    as Hvs'.
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
    as [[_ ->] | [(_ & _ & _ & ->) | (Hlt & _ & _ & _ & ->)]];
    simpl.
  - split; assumption.
  - rewrite <- (firstn_skipn (cardinal_unarized (code_params_arity c)) vs')
      in Hvs'.
    apply Forall_app in Hvs'. split; [ exact Hclosure | exact Hvs' ].
  - exact (P_partial_made closure called slots c vs' args_arity' Hcl Hc Hlt
             Hclosure Hvs').
Qed.

Lemma shape_of_call_values_bound (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity) :
  P_fields P statics ->
  P callee -> Forall P vs ->
  shape_values_all P
    (shape_of_call statics callee kind vs args_arity return_arity).
Proof.
  intros Hfields Hcallee Hvs.
  destruct (shape_of_call statics callee kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape; [ | exact I ].
  destruct (shape_of_call_cases _ _ _ _ _ _ _ Hshape)
    as [H | (closure & applied & applied_arity & _ & -> & _ & H)];
    rewrite <- H.
  - apply shape_of_closure_call_values_bound; assumption.
  - apply P_partial in Hcallee as [Hclosure Happlied].
    apply shape_of_closure_call_values_bound;
      [ exact Hfields | exact Hclosure | ].
    apply Forall_app. split; assumption.
Qed.

Lemma apply_cont_value_values_bound (Q : state -> Prop) (c : cont_value)
    (vs : list value) (hp : heap) (rd : read_record) :
  P_fields P statics ->
  cont_values_all P c ->
  Forall P vs ->
  (∀ (e : expr) (vars : env) (conts : cont_env),
    env_values_all P vars -> conts_values_all P conts ->
    Q (State e vars conts statics hp rd)) ->
  match apply_cont_value c vs statics hp rd with
  | Step s' => Q s'
  | Exit _ vs' _ _ => Forall P vs'
  | Undefined => True
  end.
Proof.
  intros Hfields Hc Hvs Hstate. revert vs Hc Hvs.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs Hc Hvs.
  - simpl. destruct Hc as [Hdef_vars Hdef_conts].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (values_of_subkinds statics vs (hm_params_subkinds metadata));
      [ | exact I ].
    destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
      [ | exact I ].
    apply Hstate.
    + exact (bind_params_values_all P params vs def_vars handler_vars Hvs
               Hdef_vars Hbind).
    + apply conts_values_all_app; [ | exact Hdef_conts ].
      apply rec_closures_values_all; assumption.
  - exact Hvs.
  - simpl. destruct Hc as [Hextra Hc]. apply (IH (vs ++ extra) Hc).
    apply Forall_app. split; assumption.
  - rewrite apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    destruct Hc as (Hremaining & Hreturn & Hexn).
    apply run_call_values_all; try assumption.
    + apply shape_of_call_values_bound;
        [ exact Hfields | exact (Forall_inv Hvs) | exact Hremaining ].
    + intros partial Hpartial.
      apply IHreturn; [ exact Hreturn | ].
      apply Forall_cons; [ exact Hpartial | apply Forall_nil ].
Qed.

Lemma call_function_values_bound (Q : state -> Prop) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (hp : heap) (rd : read_record) :
  P_fields P statics ->
  P callee ->
  Forall P vs ->
  cont_values_all P return_to ->
  cont_values_all P exn_to ->
  (∀ (e : expr) (vars : env) (conts : cont_env),
    env_values_all P vars -> conts_values_all P conts ->
    Q (State e vars conts statics hp rd)) ->
  match
    call_function callee kind vs args_arity return_arity return_to exn_to
      statics hp rd
  with
  | Step s' => Q s'
  | Exit _ vs' _ _ => Forall P vs'
  | Undefined => True
  end.
Proof.
  intros Hfields Hcallee Hvs Hreturn Hexn Hstate. unfold call_function.
  apply run_call_values_all; try assumption.
  - apply shape_of_call_values_bound; assumption.
  - intros partial Hpartial.
    apply apply_cont_value_values_bound;
      [ exact Hfields | exact Hreturn | | exact Hstate ].
    apply Forall_cons; [ exact Hpartial | apply Forall_nil ].
Qed.

End Values_bound.

(* [FlambdaSemanticsProofs.step_values_all], with the bound at a partial
   application that the machine makes. *)
Lemma step_values_bound (P : static_env -> value -> Prop)
    (Q : static_env -> statically_allocated_value -> Prop) (s : state) :
  (∀ (statics : static_env) (callee : value) (args : list value)
     (args_arity : arity),
    P statics (Value_partial_application callee args args_arity) ->
    P statics callee ∧ Forall (P statics) args) ->
  (∀ (statics : static_env) (callee : value) (called : code_id)
     (slots : list (value_slot * value)) (c : code) (args : list value)
     (args_arity : arity),
    closure_of statics callee = Some (called, slots) ->
    lookup called (static_code statics) = Some c ->
    length args_arity < length (code_params_arity c) ->
    P statics callee -> Forall (P statics) args ->
    P statics (Value_partial_application callee args args_arity)) ->
  state_values_all P Q s ->
  (∀ (size : nat) (tuple : value) (fields : list value),
    P (state_statics s) tuple ->
    tuple_fields (state_statics s) size tuple = Some fields ->
    Forall (P (state_statics s)) fields) ->
  (∀ (x : variable) (kind : flambda_kind) (defining_expr : named)
     (body : expr) (v : value) (hp' : heap) (rd' : read_record),
    state_expr s = Let x kind defining_expr body ->
    exec_named (state_statics s) (state_heap s) (state_reads s) (state_vars s)
      defining_expr
    = Some (v, hp', rd') ->
    P (state_statics s) v ∧ heap_values_all (P (state_statics s)) hp') ->
  (∀ (bindings : list static_binding) (body : expr)
     (statics' : static_env),
    state_expr s = Let_static bindings body ->
    eval_static_bindings (state_vars s) bindings (state_statics s)
    = Some statics' ->
    (∀ v : value, P (state_statics s) v -> P statics' v)
    ∧ Forall (fun binding => Q statics' (snd binding))
        (static_values statics')) ->
  (∀ (ap : apply_expr) (callee : value) (vs extra : list value),
    state_expr s = Apply ap ->
    eval_simple (state_vars s) (ap_callee ap) = Some callee ->
    eval_simples (state_vars s) (ap_args ap) = Some vs ->
    eval_simples (state_vars s) (exn_extra_simples ap) = Some extra ->
    P (state_statics s) callee
    ∧ Forall (P (state_statics s)) vs
    ∧ Forall (P (state_statics s)) extra) ->
  (∀ (ac : apply_cont) (vs : list value),
    state_expr s = Apply_cont ac ->
    eval_simples (state_vars s) (ac_args ac) = Some vs ->
    Forall (P (state_statics s)) vs) ->
  match step s with
  | Step s' => state_values_all P Q s'
  | Exit _ vs _ _ => Forall (P (state_statics s)) vs
  | Undefined => True
  end.
Proof.
  intros Hinv Hmade.
  destruct s as [e vars conts statics hp rd].
  intros (Hvars & Hconts & Hstatics & Hheap) Hfields Hlet Hlet_static Happly
    Happly_cont.
  simpl in *.
  assert (Hstate :
    ∀ (e' : expr) (vars' : env) (conts' : cont_env),
      env_values_all (P statics) vars' ->
      conts_values_all (P statics) conts' ->
      state_values_all P Q (State e' vars' conts' statics hp rd))
    by (intros e' vars' conts' Hvars' Hconts'; repeat split; assumption).
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    [ rewrite step_let_exec | autorewrite with step .. ].
  - (* Let *)
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hd;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); [ | exact I ].
    destruct (Hlet x kind d body v hp' rd' eq_refl Hd) as [Hv Hheap'].
    split; [ | split; [ | split ] ]; simpl;
      [ apply Forall_cons; [ exact Hv | exact Hvars ] | exact Hconts
      | exact Hstatics | exact Hheap' ].
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Hbindings; [ | exact I ].
    destruct (Hlet_static bindings body statics' eq_refl Hbindings)
      as [Hkept Hstatics'].
    split; [ | split; [ | split; [ exact Hstatics' | ] ] ]; simpl.
    + exact (Forall_impl _ (fun binding => Hkept (snd binding)) Hvars).
    + exact (conts_values_all_impl _ _ conts Hkept Hconts).
    + intros l c v Hin Hv. exact (Hkept v (Hheap l c v Hin Hv)).
  - (* Let_cont_non_recursive *)
    apply Hstate; [ exact Hvars | ].
    split; [ split; [ exact Hvars | exact Hconts ] | exact Hconts ].
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ exact I | ].
    apply Hstate; [ exact Hvars | ].
    apply conts_values_all_app; [ | exact Hconts ].
    apply rec_closures_values_all; assumption.
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextra; [ | exact I ].
    destruct (Happly ap callee vs extra eq_refl Hcallee Hargs Hextra)
      as (Hcallee_ok & Hvs & Hextra_ok).
    apply (call_function_values_bound (P statics) statics (Hinv statics)
             (Hmade statics) (state_values_all P Q));
      [ exact Hfields | exact Hcallee_ok | exact Hvs
      | apply resolve_values_all, Hconts
      | apply with_extra_args_values_all;
        [ apply resolve_values_all, Hconts | exact Hextra_ok ]
      | exact Hstate ].
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    unfold apply_continuation.
    apply (apply_cont_value_values_bound (P statics) statics (Hinv statics)
             (Hmade statics) (state_values_all P Q));
      [ exact Hfields | apply resolve_values_all, Hconts | | exact Hstate ].
    exact (Happly_cont ac vs eq_refl Hargs).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ]; [ | exact I ].
    apply Hstate; assumption.
  - (* Invalid *)
    exact I.
Qed.

(* ======================================================================== *)
(* Partial applications made by the machine                                 *)
(* ======================================================================== *)

(* The partial applications in a value: their closure, and the groups of
   arguments that they hold. *)
Fixpoint partials_of_value (v : value) : list (value * arity) :=
  match v with
  | Value_const _ | Value_symbol _ | Value_boxed_number _
  | Value_location _ =>
    []
  | Value_closure _ slots | Value_closure_of_set _ _ slots =>
    flat_map (fun slot => partials_of_value (snd slot)) slots
  | Value_block _ fields | Value_immutable_array fields =>
    flat_map partials_of_value fields
  | Value_partial_application callee args args_arity =>
    (callee, args_arity)
    :: partials_of_value callee ++ flat_map partials_of_value args
  end.

(* A closure of bound code, which has more parameters than [args_arity]
   has groups. *)
Definition pa_bound (statics : static_env) (cl : value) (args_arity : arity)
    : Prop :=
  ∃ (called : code_id) (slots : list (value_slot * value)) (c : code),
    closure_of statics cl = Some (called, slots)
    ∧ lookup called (static_code statics) = Some c
    ∧ length args_arity < length (code_params_arity c).

Definition pa_ok (statics : static_env) (v : value) : Prop :=
  ∀ (cl : value) (args_arity : arity),
    (cl, args_arity) ∈ partials_of_value v -> pa_bound statics cl args_arity.

Definition static_pa_ok (statics : static_env)
    (a : statically_allocated_value) : Prop :=
  ∀ v : value, v ∈ values_of_static a -> pa_ok statics v.

(* The invariant of a state. *)
Definition pa_inv (s : state) : Prop := state_values_all pa_ok static_pa_ok s.

Lemma pa_ok_nil (statics : static_env) (v : value) :
  partials_of_value v = [] -> pa_ok statics v.
Proof. intros H cl aa Hin. rewrite H in Hin. destruct Hin. Qed.

Lemma pa_ok_list (statics : static_env) (vs : list value) :
  (∀ (cl : value) (aa : arity),
     (cl, aa) ∈ flat_map partials_of_value vs -> pa_bound statics cl aa)
  <-> ∀ v : value, v ∈ vs -> pa_ok statics v.
Proof.
  split.
  - intros H v Hv cl aa Hin. apply H. apply in_flat_map. exists v.
    split; assumption.
  - intros H cl aa Hin. apply in_flat_map in Hin as (v & Hv & Hin).
    exact (H v Hv cl aa Hin).
Qed.

Lemma pa_ok_slots (statics : static_env)
    (slots : list (value_slot * value)) :
  (∀ (cl : value) (aa : arity),
     (cl, aa) ∈ flat_map (fun slot => partials_of_value (snd slot)) slots ->
     pa_bound statics cl aa)
  <-> ∀ (w : value_slot) (v : value), (w, v) ∈ slots -> pa_ok statics v.
Proof.
  split.
  - intros H w v Hv cl aa Hin. apply H. apply in_flat_map. exists (w, v).
    split; assumption.
  - intros H cl aa Hin. apply in_flat_map in Hin as ([w v] & Hv & Hin).
    exact (H w v Hv cl aa Hin).
Qed.

Lemma pa_ok_partial (statics : static_env) (callee : value)
    (args : list value) (args_arity : arity) :
  pa_ok statics (Value_partial_application callee args args_arity)
  <-> pa_bound statics callee args_arity
      ∧ pa_ok statics callee ∧ Forall (pa_ok statics) args.
Proof.
  unfold pa_ok at 1. cbn [partials_of_value]. rewrite Forall_forall.
  split.
  - intros H. split; [ apply H; left; reflexivity | ]. split.
    + intros cl aa Hin. apply H. right. apply in_or_app. left. exact Hin.
    + apply pa_ok_list. intros cl aa Hin. apply H. right. apply in_or_app.
      right. exact Hin.
  - intros (Hb & Hc & Ha) cl aa [E | Hin]; [ injection E as <- <-; exact Hb | ].
    apply in_app_or in Hin as [Hin | Hin]; [ exact (Hc cl aa Hin) | ].
    exact (proj2 (pa_ok_list statics args) Ha cl aa Hin).
Qed.

Lemma pa_bound_extend (statics statics' : static_env) (cl : value)
    (aa : arity) :
  statics_extend statics statics' ->
  pa_bound statics cl aa -> pa_bound statics' cl aa.
Proof.
  intros [Hcode Hvalues] (called & slots & c & Hcl & Hc & Hlt).
  exists called, slots, c. split; [ | split; [ exact (Hcode _ _ Hc) | ] ];
    [ | exact Hlt ].
  apply (FlambdaPrimitiveProofs.closure_of_statics_extend statics statics' cl);
    [ | exact Hcl ].
  intros s a _ Hs. exact (Hvalues s a Hs).
Qed.

Lemma pa_ok_extend (statics statics' : static_env) (v : value) :
  statics_extend statics statics' -> pa_ok statics v -> pa_ok statics' v.
Proof.
  intros Hext H cl aa Hin. exact (pa_bound_extend _ _ cl aa Hext (H cl aa Hin)).
Qed.

(* What a static environment binds satisfies [pa_ok]. *)
Definition statics_pa_ok (statics : static_env) : Prop :=
  ∀ (s : symbol) (a : statically_allocated_value) (v : value),
    lookup s (static_values statics) = Some a -> v ∈ values_of_static a ->
    pa_ok statics v.

Definition env_pa_ok (statics : static_env) (vars : env) : Prop :=
  ∀ (x : variable) (v : value), lookup x vars = Some v -> pa_ok statics v.

Lemma env_pa_ok_of (statics : static_env) (vars : env) :
  env_values_all (pa_ok statics) vars -> env_pa_ok statics vars.
Proof.
  intros H x v Hx. unfold env_values_all in H. rewrite Forall_forall in H.
  exact (H (x, v) (lookup_in _ _ _ Hx)).
Qed.

Lemma statics_pa_ok_of (statics : static_env) :
  Forall (fun binding => static_pa_ok statics (snd binding))
    (static_values statics) ->
  statics_pa_ok statics.
Proof.
  intros H s a v Hs Hv. rewrite Forall_forall in H.
  exact (H (s, a) (lookup_in _ _ _ Hs) v Hv).
Qed.

Lemma eval_simple_pa_ok (statics : static_env) (vars : env) (s : simple)
    (v : value) :
  env_pa_ok statics vars -> eval_simple vars s = Some v -> pa_ok statics v.
Proof.
  intros Hvars Hs. destruct s as [x | sym | c]; cbn [eval_simple] in Hs.
  - exact (Hvars x v Hs).
  - injection Hs as <-. apply pa_ok_nil. reflexivity.
  - injection Hs as <-. apply pa_ok_nil. reflexivity.
Qed.

Lemma eval_simples_pa_ok (statics : static_env) (vars : env) :
  env_pa_ok statics vars ->
  ∀ (args : list simple) (vs : list value),
    eval_simples vars args = Some vs -> ∀ v : value, v ∈ vs -> pa_ok statics v.
Proof.
  intros Hvars. induction args as [ | s args IH]; intros vs Hvs v Hv;
    cbn [eval_simples] in Hvs.
  - injection Hvs as <-. destruct Hv.
  - destruct (eval_simple vars s) as [w | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [ws | ] eqn:Hws; [ | discriminate ].
    injection Hvs as <-. destruct Hv as [<- | Hv];
      [ exact (eval_simple_pa_ok statics vars s _ Hvars Hs)
      | exact (IH ws eq_refl v Hv) ].
Qed.

Lemma eval_simples_pa_ok_forall (statics : static_env) (vars : env)
    (args : list simple) (vs : list value) :
  env_pa_ok statics vars ->
  eval_simples vars args = Some vs -> Forall (pa_ok statics) vs.
Proof.
  intros Hvars Hvs. apply Forall_forall.
  exact (eval_simples_pa_ok statics vars Hvars args vs Hvs).
Qed.

Lemma eval_value_slots_pa_ok (statics : static_env) (vars : env) :
  env_pa_ok statics vars ->
  ∀ (slots : list (value_slot * simple)) (vs : list (value_slot * value)),
    eval_value_slots vars slots = Some vs ->
    ∀ (w : value_slot) (v : value), (w, v) ∈ vs -> pa_ok statics v.
Proof.
  intros Hvars. induction slots as [ | [w0 s] slots IH]; intros vs Hvs w v Hv;
    cbn [eval_value_slots] in Hvs.
  - injection Hvs as <-. destruct Hv.
  - destruct (eval_simple vars s) as [x | ] eqn:Hs; [ | discriminate ].
    destruct (eval_value_slots vars slots) as [xs | ] eqn:Hxs;
      [ | discriminate ].
    injection Hvs as <-. destruct Hv as [E | Hv].
    + injection E as <- <-. exact (eval_simple_pa_ok statics vars s _ Hvars Hs).
    + exact (IH xs eq_refl w v Hv).
Qed.

Lemma closure_in_set_pa_ok (statics : static_env) (v : value)
    (slot : function_slot) (decls : list (function_slot * code_id))
    (slots : list (value_slot * value)) :
  statics_pa_ok statics -> pa_ok statics v ->
  closure_in_set statics v = Some (slot, decls, slots) ->
  ∀ (w : value_slot) (x : value), (w, x) ∈ slots -> pa_ok statics x.
Proof.
  intros Hst Hv Hcl w x Hin.
  destruct v as [ | code vs | sl ds vs | s | | | | | ];
    cbn [closure_in_set] in Hcl; try discriminate Hcl.
  - injection Hcl as _ _ <-. unfold pa_ok in Hv. cbn [partials_of_value] in Hv.
    exact (proj1 (pa_ok_slots statics vs) Hv w x Hin).
  - destruct (lookup sl ds); [ | discriminate Hcl ].
    injection Hcl as _ _ <-. unfold pa_ok in Hv. cbn [partials_of_value] in Hv.
    exact (proj1 (pa_ok_slots statics vs) Hv w x Hin).
  - destruct (lookup s (static_values statics)) as [a | ] eqn:Hs;
      [ | discriminate Hcl ].
    destruct a as [code vs | sl syms ds vs | | | | | ]; try discriminate Hcl.
    + injection Hcl as _ _ <-.
      apply (Hst s _ x Hs). cbn [values_of_static].
      exact (in_map snd _ (w, x) Hin).
    + destruct (lookup sl ds); [ | discriminate Hcl ].
      injection Hcl as _ _ <-.
      apply (Hst s _ x Hs). cbn [values_of_static].
      exact (in_map snd _ (w, x) Hin).
Qed.

Lemma closure_of_pa_ok (statics : static_env) (v : value)
    (code : code_id) (slots : list (value_slot * value)) :
  statics_pa_ok statics -> pa_ok statics v ->
  closure_of statics v = Some (code, slots) ->
  ∀ (w : value_slot) (x : value), (w, x) ∈ slots -> pa_ok statics x.
Proof.
  intros Hst Hv Hcl w x Hin.
  destruct v as [ | c vs | sl ds vs | s | | | | | ];
    cbn [closure_of] in Hcl; try discriminate Hcl.
  - injection Hcl as _ <-. unfold pa_ok in Hv. cbn [partials_of_value] in Hv.
    exact (proj1 (pa_ok_slots statics vs) Hv w x Hin).
  - destruct (lookup sl ds); [ | discriminate Hcl ].
    injection Hcl as _ <-. unfold pa_ok in Hv. cbn [partials_of_value] in Hv.
    exact (proj1 (pa_ok_slots statics vs) Hv w x Hin).
  - destruct (lookup s (static_values statics)) as [a | ] eqn:Hs;
      [ | discriminate Hcl ].
    destruct a as [c vs | sl syms ds vs | | | | | ]; try discriminate Hcl.
    + injection Hcl as _ <-.
      apply (Hst s _ x Hs). cbn [values_of_static].
      exact (in_map snd _ (w, x) Hin).
    + destruct (lookup sl ds); [ | discriminate Hcl ].
      injection Hcl as _ <-.
      apply (Hst s _ x Hs). cbn [values_of_static].
      exact (in_map snd _ (w, x) Hin).
Qed.

(* A named expression other than a primitive. *)
Lemma eval_named_pa_ok (statics : static_env) (vars : env) (d : named)
    (v : value) :
  statics_pa_ok statics -> env_pa_ok statics vars ->
  (∀ (p : prim) (args : list simple), d <> Prim p args) ->
  eval_named statics vars d = Some v -> pa_ok statics v.
Proof.
  intros Hst Hvars Hnp Hev.
  destruct d as [s | p args | code vsl | slot decls vsl | w cl | from w cl
                | mf mt cl ];
    cbn [eval_named] in Hev.
  - exact (eval_simple_pa_ok statics vars s v Hvars Hev).
  - destruct (Hnp p args eq_refl).
  - destruct (eval_value_slots vars vsl) as [vs | ] eqn:Hvs;
      [ | discriminate Hev ].
    injection Hev as <-. unfold pa_ok. cbn [partials_of_value].
    apply pa_ok_slots.
    exact (eval_value_slots_pa_ok statics vars Hvars vsl vs Hvs).
  - destruct (lookup slot decls); [ | discriminate Hev ].
    destruct (eval_value_slots vars vsl) as [vs | ] eqn:Hvs;
      [ | discriminate Hev ].
    injection Hev as <-. unfold pa_ok. cbn [partials_of_value].
    apply pa_ok_slots.
    exact (eval_value_slots_pa_ok statics vars Hvars vsl vs Hvs).
  - destruct (eval_simple vars cl) as [c | ] eqn:Hc; [ | discriminate Hev ].
    destruct (closure_of statics c) as [[code slots] | ] eqn:Hcl;
      [ | discriminate Hev ].
    exact (closure_of_pa_ok statics c code slots Hst
             (eval_simple_pa_ok statics vars cl c Hvars Hc) Hcl w v
             (lookup_in w v slots Hev)).
  - destruct (eval_simple vars cl) as [c | ] eqn:Hc; [ | discriminate Hev ].
    unfold project_value_slot in Hev.
    destruct (closure_in_set statics c) as [[[atf decls] slots] | ] eqn:Hcl;
      [ | discriminate Hev ].
    destruct (Pos.eqb atf from); [ | discriminate Hev ].
    exact (closure_in_set_pa_ok statics c atf decls slots Hst
             (eval_simple_pa_ok statics vars cl c Hvars Hc) Hcl w v
             (lookup_in w v slots Hev)).
  - destruct (eval_simple vars cl) as [c | ] eqn:Hc; [ | discriminate Hev ].
    pose proof (eval_simple_pa_ok statics vars cl c Hvars Hc) as Hcn.
    unfold project_function_slot in Hev.
    destruct c as [ | code vs | sl ds vs | s | | | | | ]; try discriminate Hev.
    + destruct (Pos.eqb mf unnamed_function_slot);
        [ | discriminate Hev ].
      destruct (Pos.eqb mt unnamed_function_slot); [ | discriminate Hev ].
      injection Hev as <-. exact Hcn.
    + destruct (Pos.eqb sl mf); [ | discriminate Hev ].
      destruct (lookup sl ds); [ | discriminate Hev ].
      destruct (lookup mt ds); [ | discriminate Hev ].
      injection Hev as <-. exact Hcn.
    + destruct (lookup s (static_values statics)) as [a | ];
        [ | discriminate Hev ].
      destruct a as [ | sl syms ds vs | | | | | ]; try discriminate Hev.
      * destruct (Pos.eqb mf unnamed_function_slot);
          [ | discriminate Hev ].
        destruct (Pos.eqb mt unnamed_function_slot); [ | discriminate Hev ].
        injection Hev as <-. exact Hcn.
      * destruct (Pos.eqb sl mf); [ | discriminate Hev ].
        destruct (lookup sl ds); [ | discriminate Hev ].
        destruct (lookup mt ds); [ | discriminate Hev ].
        destruct (lookup mt syms); [ | discriminate Hev ].
        injection Hev as <-. apply pa_ok_nil. reflexivity.
Qed.

(* What static bindings add satisfies [pa_ok] under the statics they are
   evaluated in. *)
Lemma eval_static_bindings_pa_ok (statics0 : static_env) (vars : env) :
  env_pa_ok statics0 vars ->
  ∀ (bindings : list static_binding) (statics statics' : static_env),
    (∀ (s : symbol) (a : statically_allocated_value) (v : value),
       (s, a) ∈ static_values statics -> v ∈ values_of_static a ->
       pa_ok statics0 v) ->
    eval_static_bindings vars bindings statics = Some statics' ->
    ∀ (s : symbol) (a : statically_allocated_value) (v : value),
      (s, a) ∈ static_values statics' -> v ∈ values_of_static a ->
      pa_ok statics0 v.
Proof.
  intros Hvars. induction bindings as [ | b bindings IH];
    intros statics statics' Hst Hev; cbn [eval_static_bindings] in Hev.
  - injection Hev as <-. exact Hst.
  - destruct b as [id c | id | sym const].
    + destruct (lookup id (static_code statics)); [ discriminate Hev | ].
      exact (IH (Static_env ((id, c) :: static_code statics)
                   (static_values statics)) statics' Hst Hev).
    + exact (IH statics statics' Hst Hev).
    + destruct (lookup sym (static_values statics)); [ discriminate Hev | ].
      destruct (eval_static_const vars const) as [a | ] eqn:Ha;
        [ | discriminate Hev ].
      refine (IH _ statics' _ Hev).
      intros s a' v [E | Hin] Hv; [ | exact (Hst s a' v Hin Hv) ].
      injection E as <- <-.
      destruct const as [code vsl | slot syms decls vsl | tag fields
                        | contents | bytes | elements ];
        cbn [eval_static_const] in Ha.
      * destruct (eval_value_slots vars vsl) as [vs | ] eqn:Hvs;
          [ | discriminate Ha ].
        injection Ha as <-. cbn [values_of_static] in Hv.
        apply in_map_iff in Hv as ([w x] & <- & Hin).
        exact (eval_value_slots_pa_ok statics0 vars Hvars vsl vs Hvs w x Hin).
      * destruct (lookup slot decls); [ | discriminate Ha ].
        destruct (eval_value_slots vars vsl) as [vs | ] eqn:Hvs;
          [ | discriminate Ha ].
        injection Ha as <-. cbn [values_of_static] in Hv.
        apply in_map_iff in Hv as ([w x] & <- & Hin).
        exact (eval_value_slots_pa_ok statics0 vars Hvars vsl vs Hvs w x Hin).
      * destruct (eval_simples vars fields) as [vs | ] eqn:Hvs;
          [ | discriminate Ha ].
        injection Ha as <-.
        exact (eval_simples_pa_ok statics0 vars Hvars fields vs Hvs v Hv).
      * destruct (eval_simple vars contents) as [w | ] eqn:Hw;
          [ | discriminate Ha ].
        injection Ha as <-. destruct Hv as [<- | []].
        exact (eval_simple_pa_ok statics0 vars contents _ Hvars Hw).
      * injection Ha as <-. destruct Hv.
      * destruct (eval_simples vars elements) as [vs | ] eqn:Hvs;
          [ | discriminate Ha ].
        injection Ha as <-.
        exact (eval_simples_pa_ok statics0 vars Hvars elements vs Hvs v Hv).
Qed.

Lemma pa_ok_holds_of (statics : static_env) :
  statics_pa_ok statics ->
  holds_of_numbers (pa_ok statics) ∧ holds_of_blocks (pa_ok statics)
  ∧ holds_of_arrays (pa_ok statics)
  ∧ holds_in_static_blocks statics (pa_ok statics)
  ∧ holds_in_static_arrays statics (pa_ok statics).
Proof.
  intros Hst. split; [ | split; [ | split; [ | split ] ] ].
  - intros c. split; apply pa_ok_nil; reflexivity.
  - intros tag fields. unfold pa_ok at 1. cbn [partials_of_value].
    exact (pa_ok_list statics fields).
  - intros elements. unfold pa_ok at 1. cbn [partials_of_value].
    exact (pa_ok_list statics elements).
  - intros s tag fields _ Hs v Hv. exact (Hst s _ v Hs Hv).
  - intros s elements _ Hs v Hv. exact (Hst s _ v Hs Hv).
Qed.

(* A step keeps the invariant. *)
Lemma pa_inv_step (s s' : state) :
  pa_inv s -> step s = Step s' -> pa_inv s'.
Proof.
  intros Hall Hstep.
  pose proof Hall as (Hvars & Hconts & Hstatics & Hhv).
  assert (Hvn : env_pa_ok (state_statics s) (state_vars s))
    by exact (env_pa_ok_of _ _ Hvars).
  assert (Hsn : statics_pa_ok (state_statics s))
    by exact (statics_pa_ok_of _ Hstatics).
  destruct (pa_ok_holds_of _ Hsn) as (Hn & Hb & Ha & Hsb & Hsa).
  assert (Hgen :
    match step s with
    | Step s' => pa_inv s'
    | Exit _ vs _ _ => Forall (pa_ok (state_statics s)) vs
    | Undefined => True
    end).
  { apply (step_values_bound pa_ok static_pa_ok s).
    - intros statics callee args aa H. apply pa_ok_partial in H.
      exact (proj2 H).
    - intros statics callee called slots c args aa Hcl Hc Hlt Hcallee Hargs.
      apply pa_ok_partial.
      split; [ exists called, slots, c; split; [ exact Hcl | ] | ];
        [ split; [ exact Hc | exact Hlt ] | split; assumption ].
    - exact Hall.
    - intros size tuple fields Ht Hf.
      apply (tuple_fields_forall _ (state_statics s) size tuple fields);
        [ | exact Hf ].
      intros field v Hprim.
      refine (FlambdaPrimitiveProofs.eval_prim_preserves (pa_ok _)
                (state_statics s) _ [tuple] v Hn Hb Ha Hsb Hsa _ Hprim).
      intros w [<- | []]. exact Ht.
    - intros x kind d body v hp' rd' _ Hexec.
      apply (exec_named_values_all (pa_ok (state_statics s))
               (state_statics s) (state_heap s) hp' (state_reads s) rd'
               (state_vars s) d v Hn Hb Ha Hsb Hsa); try assumption.
      + intros l. apply pa_ok_nil. reflexivity.
      + intros p args vs _ Hvs arg Harg.
        exact (eval_simples_pa_ok _ _ Hvn args vs Hvs arg Harg).
      + intros Hnp w Hw. exact (eval_named_pa_ok _ _ d w Hsn Hvn Hnp Hw).
    - intros bindings body statics' _ Hev.
      pose proof (FlambdaWellFormedProofs.eval_static_bindings_extend
                    (state_vars s) bindings (state_statics s) statics' Hev)
        as Hext.
      split; [ intros v Hv; exact (pa_ok_extend _ _ v Hext Hv) | ].
      apply Forall_forall. intros [s0 a] Hin v Hv.
      apply (pa_ok_extend _ _ v Hext).
      refine (eval_static_bindings_pa_ok _ _ Hvn bindings _ statics' _
                Hev s0 a v Hin Hv).
      intros s1 a1 w Hin1 Hw. rewrite Forall_forall in Hstatics.
      exact (Hstatics (s1, a1) Hin1 w Hw).
    - intros ap callee vs extra _ Hc Hvs Hx.
      split; [ exact (eval_simple_pa_ok _ _ _ _ Hvn Hc) | ].
      split; [ exact (eval_simples_pa_ok_forall _ _ _ _ Hvn Hvs)
             | exact (eval_simples_pa_ok_forall _ _ _ _ Hvn Hx) ].
    - intros ac vs _ Hvs. exact (eval_simples_pa_ok_forall _ _ _ _ Hvn Hvs). }
  rewrite Hstep in Hgen. exact Hgen.
Qed.

Lemma pa_inv_initial (e : expr) :
  pa_inv (State e [] [] empty_static_env empty_heap no_reads).
Proof.
  split; [ apply Forall_nil | split; [ exact I | split ] ];
    [ apply Forall_nil | apply heap_values_all_empty ].
Qed.

Lemma pa_inv_run (n : nat) :
  ∀ s s' : state, pa_inv s -> run n s = Step s' -> pa_inv s'.
Proof.
  induction n as [ | n IH]; intros s s' Hinv; simpl.
  - intros [= <-]. exact Hinv.
  - destruct (step s) as [next | k vs | ] eqn:Hstep; try discriminate.
    exact (IH next s' (pa_inv_step s next Hinv Hstep)).
Qed.

(* ======================================================================== *)
(* A call with one group of arguments                                       *)
(* ======================================================================== *)

(* The value of a simple in a state of the invariant. *)
Lemma pa_inv_eval_simple (s : state) (x : simple) (v : value) :
  pa_inv s -> eval_simple (state_vars s) x = Some v ->
  pa_ok (state_statics s) v.
Proof.
  intros (Hvars & _) Hx. exact (eval_simple_pa_ok _ _ x v
                                  (env_pa_ok_of _ _ Hvars) Hx).
Qed.

(* A call with one group of arguments, of a value that satisfies [pa_ok],
   is no over-application where the code that it may call is not tupled
   and has a parameter. *)
Lemma one_group_no_over (statics : static_env) (v : value)
    (kind : call_kind) (vs : list value) (g : arity_component)
    (rta : arity) (cl0 : value) (c0 : code) (ws rest : list value)
    (aa ra1 : arity) :
  pa_ok statics v ->
  (∀ (id : code_id) (c : code),
     lookup id (static_code statics) = Some c ->
     code_is_tupled c = false ∧ code_params_arity c <> []) ->
  shape_of_call statics v kind vs [g] rta
  <> Some (Over_call cl0 c0 ws rest aa ra1).
Proof.
  intros Hv Hcodes Hshape.
  destruct (shape_of_call_cases _ _ _ _ _ _ _ Hshape)
    as [H | (closure & applied & applied_arity & _ & Hpa & _ & H)].
  - destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ H)
      as (called & slots & c & vs' & aa' & _ & Hc & _ & Hargs & Hcall).
    destruct (Hcodes called c Hc) as [Htup Hne].
    unfold arguments_for_code in Hargs. rewrite Htup in Hargs.
    injection Hargs as <- <-.
    destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
      as [[_ E] | [(Hlt & _ & _ & _) | (_ & _ & _ & _ & E)]];
      try discriminate E.
    cbn [length] in Hlt.
    destruct (code_params_arity c); [ exact (Hne eq_refl) | ].
    cbn [length] in Hlt. lia.
  - subst v. apply pa_ok_partial in Hv as [Hb _].
    destruct Hb as (called' & slots' & c' & Hcl' & Hc' & Hlt').
    destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ H)
      as (called & slots & c & vs' & aa' & Hcl & Hc & _ & Hargs & Hcall).
    rewrite Hcl' in Hcl. injection Hcl as <- <-.
    rewrite Hc' in Hc. injection Hc as <-.
    destruct (Hcodes called' c' Hc') as [Htup _].
    unfold arguments_for_code in Hargs. rewrite Htup in Hargs.
    injection Hargs as <- <-.
    destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
      as [[_ E] | [(Hlt & _ & _ & _) | (_ & _ & _ & _ & E)]];
      try discriminate E.
    rewrite length_app in Hlt. cbn [length] in Hlt. lia.
Qed.

(* ======================================================================== *)
(* The code that a step over-applies                                        *)
(* ======================================================================== *)

(* A call of the shape [shp] over-applies only code of [OV], and gives a
   partial application that it makes to a continuation of which [ret]
   holds. *)
Definition shape_over_ok (OV : code_id -> Prop) (statics : static_env)
    (ret : list value -> Prop) (shp : option call_shape) : Prop :=
  match shp with
  | Some (Over_call cl _ _ _ _ _) =>
    ∀ (id : code_id) (slots : list (value_slot * value)),
      closure_of statics cl = Some (id, slots) -> OV id
  | Some (Partial_call p) => ret [p]
  | Some (Full_call _ _ _ _) | None => True
  end.

(* A continuation given [vs] over-applies only code of [OV]: what is left
   of an over-application calls the one value it is given with the
   arguments that remain, and a partial application that this call makes
   goes to the continuation of the whole call. *)
Fixpoint cont_over_ok (OV : code_id -> Prop) (statics : static_env)
    (c : cont_value) (vs : list value) {struct c} : Prop :=
  match c with
  | Over_application remaining remaining_arity return_arity return_to _ =>
    match vs with
    | [v] =>
      shape_over_ok OV statics (cont_over_ok OV statics return_to)
        (shape_of_call statics v Indirect_unknown_arity remaining
           remaining_arity return_arity)
    | _ => True
    end
  | Extra_args c extra => cont_over_ok OV statics c (vs ++ extra)
  | Closure _ _ _ _ | Outer _ => True
  end.

(* The step of a state over-applies only code of [OV]: a call, or the
   continuation that an [Apply_cont] applies. *)
Definition step_over_ok (OV : code_id -> Prop) (s : state) : Prop :=
  match state_expr s with
  | Apply ap =>
    ∀ (callee : value) (vs : list value),
      eval_simple (state_vars s) (ap_callee ap) = Some callee ->
      eval_simples (state_vars s) (ap_args ap) = Some vs ->
      shape_over_ok OV (state_statics s)
        (cont_over_ok OV (state_statics s)
           (resolve (ap_continuation ap) (state_conts s)))
        (shape_of_call (state_statics s) callee (ap_call_kind ap) vs
           (ap_args_arity ap) (ap_return_arity ap))
  | Apply_cont ac =>
    ∀ vs : list value,
      eval_simples (state_vars s) (ac_args ac) = Some vs ->
      cont_over_ok OV (state_statics s)
        (resolve (ac_continuation ac) (state_conts s)) vs
  | Let _ _ _ _ | Let_static _ _ | Let_cont_non_recursive _ _ _
  | Let_cont_recursive _ _ | Switch _ _ | Invalid =>
    True
  end.

(* Where [OV] holds of every code, every step is as it says. *)
Lemma cont_over_ok_total (OV : code_id -> Prop) (statics : static_env) :
  (∀ id : code_id, OV id) ->
  ∀ (c : cont_value) (vs : list value), cont_over_ok OV statics c vs.
Proof.
  intros Hall c.
  induction c as [h vars conts recursive | k | c IH extra
                 | remaining remaining_arity return_arity return_to exn_to
                     IHret _]
    using cont_value_over_ind;
    intros vs; cbn [cont_over_ok]; [ exact I | exact I | apply IH | ].
  destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
  destruct (shape_of_call statics v Indirect_unknown_arity remaining
              remaining_arity return_arity)
    as [[cl c vs' aa | cl c vs' rest aa ra | p] | ];
    cbn [shape_over_ok]; [ exact I | | apply IHret | exact I ].
  intros id slots _. exact (Hall id).
Qed.

Lemma step_over_ok_total (OV : code_id -> Prop) (s : state) :
  (∀ id : code_id, OV id) -> step_over_ok OV s.
Proof.
  intros Hall. unfold step_over_ok.
  destruct (state_expr s); try exact I.
  - intros callee vs _ _.
    destruct (shape_of_call _ _ _ _ _ _)
      as [[cl c vs' aa | cl c vs' rest aa ra | p] | ];
      cbn [shape_over_ok];
      [ exact I | intros id slots _; exact (Hall id)
      | apply cont_over_ok_total; exact Hall | exact I ].
  - intros vs _. apply cont_over_ok_total. exact Hall.
Qed.

(* The exception continuation of a call, with its extra arguments. *)
Lemma cont_over_ok_extra (OV : code_id -> Prop) (statics : static_env)
    (c : cont_value) (extra vs : list value) :
  cont_over_ok OV statics (with_extra_args c extra) vs ->
  cont_over_ok OV statics c (vs ++ extra).
Proof.
  destruct extra as [ | x extra]; cbn [with_extra_args].
  - rewrite app_nil_r. exact (fun H => H).
  - exact (fun H => H).
Qed.

(* The closure that a call of a value calls: that of a partial
   application, or the value. *)
Definition closure_called (v : value) : value :=
  match partial_application_of v with
  | Some (cl, _, _) => cl
  | None => v
  end.

(* What is left of an over-application, given [vs], calls only a closure
   of code of [REM], and a partial application that it makes goes to a
   continuation that does too. *)
Fixpoint cont_rem_ok (REM : code_id -> Prop) (statics : static_env)
    (c : cont_value) (vs : list value) {struct c} : Prop :=
  match c with
  | Over_application remaining remaining_arity return_arity return_to _ =>
    match vs with
    | [v] =>
      (∀ (id : code_id) (slots : list (value_slot * value)),
         closure_of statics (closure_called v) = Some (id, slots) -> REM id)
      ∧ shape_over_ok (fun _ => True) statics
          (cont_rem_ok REM statics return_to)
          (shape_of_call statics v Indirect_unknown_arity remaining
             remaining_arity return_arity)
    | _ => True
    end
  | Extra_args c extra => cont_rem_ok REM statics c (vs ++ extra)
  | Closure _ _ _ _ | Outer _ => True
  end.

(* The step of a state calls by what is left of an over-application only
   closures of code of [REM]: a call gives a partial application that it
   makes, and an [Apply_cont] its values, to a continuation that does. *)
Definition step_rem_ok (REM : code_id -> Prop) (s : state) : Prop :=
  match state_expr s with
  | Apply ap =>
    ∀ (callee : value) (vs : list value),
      eval_simple (state_vars s) (ap_callee ap) = Some callee ->
      eval_simples (state_vars s) (ap_args ap) = Some vs ->
      shape_over_ok (fun _ => True) (state_statics s)
        (cont_rem_ok REM (state_statics s)
           (resolve (ap_continuation ap) (state_conts s)))
        (shape_of_call (state_statics s) callee (ap_call_kind ap) vs
           (ap_args_arity ap) (ap_return_arity ap))
  | Apply_cont ac =>
    ∀ vs : list value,
      eval_simples (state_vars s) (ac_args ac) = Some vs ->
      cont_rem_ok REM (state_statics s)
        (resolve (ac_continuation ac) (state_conts s)) vs
  | Let _ _ _ _ | Let_static _ _ | Let_cont_non_recursive _ _ _
  | Let_cont_recursive _ _ | Switch _ _ | Invalid =>
    True
  end.

Lemma cont_rem_ok_total (REM : code_id -> Prop) (statics : static_env) :
  (∀ id : code_id, REM id) ->
  ∀ (c : cont_value) (vs : list value), cont_rem_ok REM statics c vs.
Proof.
  intros Hall c.
  induction c as [h vars conts recursive | k | c IH extra
                 | remaining remaining_arity return_arity return_to exn_to
                     IHret _]
    using cont_value_over_ind;
    intros vs; cbn [cont_rem_ok]; [ exact I | exact I | apply IH | ].
  destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
  split; [ intros id slots _; exact (Hall id) | ].
  destruct (shape_of_call statics v Indirect_unknown_arity remaining
              remaining_arity return_arity)
    as [[cl c vs' aa | cl c vs' rest aa ra | p] | ];
    cbn [shape_over_ok]; [ exact I | intros; exact I | apply IHret
                         | exact I ].
Qed.

Lemma step_rem_ok_total (REM : code_id -> Prop) (s : state) :
  (∀ id : code_id, REM id) -> step_rem_ok REM s.
Proof.
  intros Hall. unfold step_rem_ok.
  destruct (state_expr s); try exact I.
  - intros callee vs _ _.
    destruct (shape_of_call _ _ _ _ _ _)
      as [[cl c vs' aa | cl c vs' rest aa ra | p] | ];
      cbn [shape_over_ok];
      [ exact I | intros; exact I
      | apply cont_rem_ok_total; exact Hall | exact I ].
  - intros vs _. apply cont_rem_ok_total. exact Hall.
Qed.

Lemma cont_rem_ok_extra (REM : code_id -> Prop) (statics : static_env)
    (c : cont_value) (extra vs : list value) :
  cont_rem_ok REM statics (with_extra_args c extra) vs ->
  cont_rem_ok REM statics c (vs ++ extra).
Proof.
  destruct extra as [ | x extra]; cbn [with_extra_args].
  - rewrite app_nil_r. exact (fun H => H).
  - exact (fun H => H).
Qed.

(* [shape_over_ok], for wider [OV] and [ret]. *)
Lemma shape_over_ok_mono (OV OV' : code_id -> Prop) (statics : static_env)
    (ret ret' : list value -> Prop) (shp : option call_shape) :
  (∀ id : code_id, OV id -> OV' id) ->
  (∀ ws : list value, ret ws -> ret' ws) ->
  shape_over_ok OV statics ret shp -> shape_over_ok OV' statics ret' shp.
Proof.
  intros Hov Hret.
  destruct shp as [[cl c vs aa | cl c vs rest aa ra | p] | ];
    cbn [shape_over_ok]; [ exact (fun H => H) | | apply Hret
                         | exact (fun H => H) ].
  intros H id slots Hclo. exact (Hov id (H id slots Hclo)).
Qed.
