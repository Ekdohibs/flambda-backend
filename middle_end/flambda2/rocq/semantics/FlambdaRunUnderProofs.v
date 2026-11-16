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

(* Proofs of the statements of FlambdaRunUnderSpec.v. Nothing here needs
   review.

   The two runs are related step by step. Their states are the same but
   for their continuation environments, and what a free continuation of
   the term stands for in the one is [under] what it stands for in the
   other: a way out, and what [base] binds it to; two closures of one
   handler whose environments are so related at the continuations that
   the handler and its group leave free; or the same with extra
   arguments, or with what is left of an over-application. The relation
   is asked only at the free continuations, so that the body of closed
   code, which is run under the two continuations of the call, is
   related to itself whatever [base] binds. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaBindersSpec FlambdaBindersProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaScopeProofs
  FlambdaExitWorldSpec FlambdaRunUnderSpec.
Import ListNotations.

(* ======================================================================== *)
(* The free continuations of a term, by its head                            *)
(* ======================================================================== *)

(* The free continuations of the handlers of a group. *)
Definition group_continuations
    (recursive : list (continuation * continuation_handler))
    : list continuation :=
  flat_map (fun '(_, h) => free_continuations_of_handler h) recursive.

Lemma among_group (recursive : list (continuation * continuation_handler))
    (k0 : continuation) (h0 : continuation_handler) (k : continuation) :
  (k0, h0) ∈ recursive ->
  among k (free_continuations_of_handler h0) = true ->
  among k (group_continuations recursive) = true.
Proof.
  intros Hin Hk. apply among_In. apply among_In in Hk.
  unfold group_continuations. apply in_flat_map. exists (k0, h0).
  split; [ exact Hin | exact Hk ].
Qed.

Lemma among_let_static (bindings : list static_binding) (body : expr)
    (k : continuation) :
  among k (free_continuations_of_expr body) = true ->
  among k (free_continuations_of_expr (Let_static bindings body)) = true.
Proof.
  intros Hk.
  change (free_continuations_of_expr (Let_static bindings body))
    with (flat_map free_continuations_of_binding bindings
          ++ free_continuations_of_expr body).
  rewrite among_app, Hk. apply orb_true_r.
Qed.

Lemma among_let_cont (k0 : continuation) (h : continuation_handler)
    (body : expr) (k : continuation) :
  among k (free_continuations_of_expr (Let_cont_non_recursive k0 h body))
  = among k (free_continuations_of_handler h)
    || among k (free_continuations_of_expr body) && negb (Pos.eqb k k0).
Proof.
  change (free_continuations_of_expr (Let_cont_non_recursive k0 h body))
    with (free_continuations_of_handler h
          ++ without [k0] (free_continuations_of_expr body)).
  rewrite among_app, among_without, among_cons.
  change (among k []) with false. rewrite orb_false_r. reflexivity.
Qed.

Lemma among_let_cont_recursive
    (handlers : list (continuation * continuation_handler)) (body : expr)
    (k : continuation) :
  among k (free_continuations_of_expr (Let_cont_recursive handlers body))
  = (among k (group_continuations handlers)
     || among k (free_continuations_of_expr body))
    && negb (among k (map fst handlers)).
Proof.
  change (free_continuations_of_expr (Let_cont_recursive handlers body))
    with (without (map fst handlers)
            (group_continuations handlers
             ++ free_continuations_of_expr body)).
  rewrite among_without, among_app. reflexivity.
Qed.

Lemma lookup_none_among {A : Type} (k : positive)
    (bindings : list (positive * A)) :
  lookup k bindings = None -> among k (map fst bindings) = false.
Proof.
  intros Hlookup. rewrite <- is_bound_among. unfold is_bound.
  rewrite Hlookup. reflexivity.
Qed.

(* ======================================================================== *)
(* The relation between the two runs                                        *)
(* ======================================================================== *)

Section Under.

Variable base : cont_env.

(* What a continuation of the unit on its own stands for, and what it
   stands for in the run from [base]. *)
Inductive under : cont_value -> cont_value -> Prop :=
  | Under_outer (k : continuation) : under (Outer k) (resolve k base)
  | Under_closure (h : continuation_handler) (vars : env)
      (conts conts' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    handler_code_closed h = true ->
    forallb (fun '(_, h') => handler_code_closed h') recursive = true ->
    (∀ k : continuation,
      among k
        (free_continuations_of_handler h ++ group_continuations recursive)
      = true ->
      among k (map fst recursive) = false ->
      under (resolve k conts) (resolve k conts')) ->
    under (Closure h vars conts recursive) (Closure h vars conts' recursive)
  | Under_extra_args (c c' : cont_value) (extra : list value) :
    under c c' -> under (Extra_args c extra) (Extra_args c' extra)
  | Under_over_application (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    under return_to return_to' ->
    under exn_to exn_to' ->
    under
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

(* The two environments, at the free continuations of a term. *)
Definition conts_under (e : expr) (conts conts' : cont_env) : Prop :=
  ∀ k : continuation,
    among k (free_continuations_of_expr e) = true ->
    under (resolve k conts) (resolve k conts').

(* A state of the unit on its own and the state of the run from [base].
   The code of the term and of the static environment is closed. *)
Definition state_under (s s' : state) : Prop :=
  same_but_conts s s'
  ∧ conts_under (state_expr s) (state_conts s) (state_conts s')
  ∧ code_closed (state_expr s) = true
  ∧ code_all_closed (state_statics s).

(* What a step of the one comes to, and what the step of the other
   does: an exit of the unit on its own is, in the run from [base], the
   application of what [base] binds its continuation to. *)
Definition result_under (statics : static_env) (r r' : step_result)
    : Prop :=
  match r with
  | Step s => ∃ s' : state, r' = Step s' ∧ state_under s s'
  | Exit k vs hp rd => r' = apply_cont_value (resolve k base) vs statics hp rd
  | Undefined => r' = Undefined
  end.

Lemma state_under_intro (e : expr) (vars : env) (conts conts' : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  conts_under e conts conts' ->
  code_closed e = true ->
  code_all_closed statics ->
  state_under (State e vars conts statics hp rd)
    (State e vars conts' statics hp rd).
Proof.
  intros Hconts Hcode Hstatics.
  split; [ repeat split | ]. split; [ exact Hconts | ].
  split; [ exact Hcode | exact Hstatics ].
Qed.

Lemma result_under_step (statics : static_env) (e : expr) (vars : env)
    (conts conts' : cont_env) (statics' : static_env) (hp : heap)
      (rd : read_record) :
  conts_under e conts conts' ->
  code_closed e = true ->
  code_all_closed statics' ->
  result_under statics (Step (State e vars conts statics' hp rd))
    (Step (State e vars conts' statics' hp rd)).
Proof.
  intros Hconts Hcode Hstatics. eexists. split; [ reflexivity | ].
  apply state_under_intro; assumption.
Qed.

Lemma with_extra_args_under (c c' : cont_value) (extra : list value) :
  under c c' -> under (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hc. destruct extra; [ exact Hc | ]. apply Under_extra_args, Hc.
Qed.

(* The environments under which a closure runs its handler. *)
Lemma closure_conts_under (h : continuation_handler) (vars : env)
    (conts conts' : cont_env)
    (recursive : list (continuation * continuation_handler)) :
  forallb (fun '(_, h') => handler_code_closed h') recursive = true ->
  (∀ k : continuation,
    among k
      (free_continuations_of_handler h ++ group_continuations recursive)
    = true ->
    among k (map fst recursive) = false ->
    under (resolve k conts) (resolve k conts')) ->
  ∀ k : continuation,
    among k (free_continuations_of_handler h) = true ->
    under (resolve k (rec_closures recursive vars conts ++ conts))
      (resolve k (rec_closures recursive vars conts' ++ conts')).
Proof.
  intros Hrecursive Hconts k Hk. rewrite !resolve_rec_closures.
  destruct (lookup k recursive) as [h_k | ] eqn:Hlookup.
  - pose proof (lookup_in _ _ _ Hlookup) as Hin.
    apply Under_closure; [ | exact Hrecursive | ].
    + rewrite forallb_forall in Hrecursive. exact (Hrecursive (k, h_k) Hin).
    + intros k' Hk' Hfree. apply Hconts; [ | exact Hfree ].
      rewrite among_app in Hk' |- *. apply orb_true_iff. right.
      apply orb_true_iff in Hk'. destruct Hk' as [Hk' | Hk']; [ | exact Hk' ].
      exact (among_group recursive k h_k k' Hin Hk').
  - apply Hconts; [ | exact (lookup_none_among k recursive Hlookup) ].
    rewrite among_app, Hk. reflexivity.
Qed.

(* A full call of closed code. *)
Lemma call_code_under (closure : value) (called : code_id) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  code_all_closed statics ->
  lookup called (static_code statics) = Some c ->
  under return_to return_to' ->
  under exn_to exn_to' ->
  result_under statics
    (call_code closure c vs args_arity return_arity return_to exn_to statics
       hp rd)
    (call_code closure c vs args_arity return_arity return_to' exn_to'
       statics hp rd).
Proof.
  intros Hstatics Hc Hreturn Hexn.
  destruct (call_code_cases closure c vs args_arity return_arity statics hp rd)
    as [H | [callee_vars H]]; rewrite !H; [ exact eq_refl | ].
  destruct
    (proj1 (closed_code c) (lookup_code_closed statics called c Hstatics Hc))
    as [Hbody Hbody_closed].
  apply result_under_step; [ | exact Hbody_closed | exact Hstatics ].
  intros k Hk. unfold names_among in Hbody.
  apply andb_true_iff in Hbody. destruct Hbody as [_ Hbody].
  pose proof (all_among_among _ _ k Hbody Hk) as Hamong.
  rewrite !among_cons in Hamong. change (among k []) with false in Hamong.
  destruct (Pos.eqb_spec k (code_return_continuation c)) as [-> | Hne].
  - rewrite !resolve_head. exact Hreturn.
  - rewrite !(resolve_other _ _ _ _ Hne).
    destruct (Pos.eqb_spec k (code_exn_continuation c)) as [-> | Hne'].
    + rewrite !resolve_head. exact Hexn.
    + discriminate Hamong.
Qed.

Lemma run_call_under (return_partial return_partial' : value -> step_result)
    (kind : call_kind) (shape : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  code_all_closed statics ->
  shape_calls statics kind shape ->
  under return_to return_to' ->
  under exn_to exn_to' ->
  (∀ partial : value,
    result_under statics (return_partial partial)
      (return_partial' partial)) ->
  result_under statics
    (run_call return_partial shape return_arity return_to exn_to statics hp rd)
    (run_call return_partial' shape return_arity return_to' exn_to' statics
       hp rd).
Proof.
  intros Hstatics Hcalls Hreturn Hexn Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls; cbn [run_call].
  - destruct Hcalls as (called & slots & _ & Hc & _).
    exact (call_code_under closure called c vs _ _ _ _ _ _ statics hp rd
             Hstatics Hc Hreturn Hexn).
  - destruct Hcalls as (called & slots & _ & Hc & _).
    apply (call_code_under closure called c vs _ _ _ _ _ _ statics hp rd
             Hstatics Hc);
      [ apply Under_over_application; assumption | exact Hexn ].
  - apply Hpartial.
  - exact eq_refl.
Qed.

(* Applying two continuations so related. *)
Lemma apply_cont_value_under (c c' : cont_value) :
  under c c' ->
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record),
    code_all_closed statics ->
    result_under statics (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' vs statics hp rd).
Proof.
  induction 1
    as [ k
       | [params metadata handler] vars conts conts' recursive Hh Hrecursive
           Hconts _
       | c c' extra _ IH
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hexn _ ];
    intros vs statics hp rd Hstatics.
  - exact eq_refl.
  - cbn [apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact eq_refl ].
    destruct (bind_params params vs vars) as [handler_vars | ];
      [ | exact eq_refl ].
    apply result_under_step; [ | exact Hh | exact Hstatics ].
    exact (closure_conts_under _ vars conts conts' recursive Hrecursive
             Hconts).
  - cbn [apply_cont_value]. apply IH. exact Hstatics.
  - rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact eq_refl | | exact eq_refl ].
    unfold call_function.
    apply (run_call_under _ _ Indirect_unknown_arity); try assumption.
    + apply shape_of_call_calls.
    + intros partial. apply IHreturn. exact Hstatics.
Qed.

(* A step of the two states. *)
Lemma step_under (s s' : state) :
  state_under s s' -> result_under (state_statics s) (step s) (step s').
Proof.
  destruct s as [e vars conts statics hp rd],
    s' as [e' vars' conts' statics' hp' rd'].
  intros ((He & Hvars & Hstatics_eq & Hhp & Hrd) & Hconts & Hcode & Hstatics).
  simpl in He, Hvars, Hstatics_eq, Hhp, Hrd. subst e' vars' statics' hp' rd'.
  simpl state_expr in *. simpl state_conts in *. simpl state_statics in *.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ].
  - (* Let *)
    rewrite !step_let_exec.
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ];
      [ | exact eq_refl ].
    destruct (kind_eqb (kind_of_value v) kind); [ | exact eq_refl ].
    apply result_under_step; [ exact Hconts | exact Hcode | exact Hstatics ].
  - (* Let_static *)
    rewrite !step_let_static.
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Hbindings; [ | exact eq_refl ].
    simpl in Hcode. apply andb_true_iff in Hcode.
    destruct Hcode as [Hbindings_closed Hbody_closed].
    apply result_under_step; [ | exact Hbody_closed | ].
    + intros k Hk. apply Hconts. apply among_let_static, Hk.
    + exact (eval_static_bindings_closed vars bindings statics statics'
               Hbindings_closed Hstatics Hbindings).
  - (* Let_cont_non_recursive *)
    rewrite !step_let_cont_non_recursive.
    simpl in Hcode. apply andb_true_iff in Hcode.
    destruct Hcode as [Hh_closed Hbody_closed].
    apply result_under_step; [ | exact Hbody_closed | exact Hstatics ].
    intros k' Hk'.
    destruct (Pos.eqb_spec k' k) as [-> | Hne].
    + rewrite !resolve_head.
      apply Under_closure; [ exact Hh_closed | reflexivity | ].
      intros k'' Hk'' _. apply Hconts. rewrite among_let_cont.
      rewrite among_app in Hk''. change (among k'' []) with false in Hk''.
      rewrite orb_false_r in Hk''. rewrite Hk''. reflexivity.
    + rewrite !(resolve_other _ _ _ _ Hne). apply Hconts.
      rewrite among_let_cont, Hk'. apply Pos.eqb_neq in Hne. rewrite Hne.
      apply orb_true_r.
  - (* Let_cont_recursive *)
    rewrite !step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); [ exact eq_refl | ].
    simpl in Hcode. apply andb_true_iff in Hcode.
    destruct Hcode as [Hhandlers_closed Hbody_closed].
    apply result_under_step; [ | exact Hbody_closed | exact Hstatics ].
    intros k' Hk'. rewrite !resolve_rec_closures.
    destruct (lookup k' handlers) as [h | ] eqn:Hlookup.
    + pose proof (lookup_in _ _ _ Hlookup) as Hin.
      apply Under_closure; [ | exact Hhandlers_closed | ].
      * rewrite forallb_forall in Hhandlers_closed.
        exact (Hhandlers_closed (k', h) Hin).
      * intros k'' Hk'' Hfree. apply Hconts.
        rewrite among_let_cont_recursive, Hfree.
        rewrite among_app in Hk''. apply orb_true_iff in Hk''.
        destruct Hk'' as [Hk'' | Hk''].
        -- rewrite (among_group handlers k' h k'' Hin Hk''). reflexivity.
        -- rewrite Hk''. reflexivity.
    + assert (Hfree : among k' (map fst handlers) = false)
        by exact (lookup_none_among k' handlers Hlookup).
      apply Hconts. rewrite among_let_cont_recursive, Hk', Hfree.
      rewrite orb_true_r. reflexivity.
  - (* Apply *)
    rewrite !step_apply.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | exact eq_refl ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact eq_refl ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact eq_refl ].
    assert (Hreturn :
      under (resolve (ap_continuation ap) conts)
        (resolve (ap_continuation ap) conts')).
    { apply Hconts. apply among_In. left. reflexivity. }
    assert (Hexn :
      under (resolve (ap_exn_continuation ap) conts)
        (resolve (ap_exn_continuation ap) conts')).
    { apply Hconts. apply among_In. right. left. reflexivity. }
    unfold call_function.
    apply (run_call_under _ _ (ap_call_kind ap));
      [ exact Hstatics | apply shape_of_call_calls | exact Hreturn
      | apply with_extra_args_under; exact Hexn | ].
    intros partial.
    exact
      (apply_cont_value_under _ _ Hreturn [partial] statics hp rd Hstatics).
  - (* Apply_cont *)
    rewrite !step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact eq_refl ].
    unfold apply_continuation.
    apply apply_cont_value_under; [ | exact Hstatics ].
    apply Hconts. apply among_In. left. reflexivity.
  - (* Switch *)
    rewrite !step_switch.
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact eq_refl ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact eq_refl ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | exact eq_refl ].
    apply result_under_step; [ | reflexivity | exact Hstatics ].
    intros k Hk. apply Hconts. apply among_In.
    change (free_continuations_of_expr (Switch scrutinee arms))
      with (flat_map (fun arm => apply_cont_continuations (snd arm)) arms).
    apply in_flat_map. exists (i, ac).
    split; [ exact (find_arm_in i arms ac Harm) | ].
    apply among_In. exact Hk.
  - (* Invalid *)
    rewrite !step_invalid. exact eq_refl.
Qed.

(* A run of the two states. *)
Lemma run_under_steps (n : nat) :
  ∀ s s' t : state,
    state_under s s' ->
    run n s = Step t ->
    ∃ t' : state, run n s' = Step t' ∧ state_under t t'.
Proof.
  induction n as [ | n IH]; intros s s' t Hunder; simpl.
  - intros [= <-]. exists s'. split; [ reflexivity | exact Hunder ].
  - generalize (step_under s s' Hunder).
    destruct (step s) as [s1 | k vs hp rd | ]; unfold result_under;
      [ | discriminate | discriminate ].
    intros (s1' & Hstep' & Hunder1) Hrun. rewrite Hstep'.
    exact (IH s1 s1' t Hunder1 Hrun).
Qed.

Lemma run_under_wrong (n : nat) :
  ∀ s s' : state,
    state_under s s' -> run n s = Undefined -> run n s' = Undefined.
Proof.
  induction n as [ | n IH]; intros s s' Hunder; simpl; [ discriminate | ].
  generalize (step_under s s' Hunder).
  destruct (step s) as [s1 | k vs hp rd | ]; unfold result_under.
  - intros (s1' & Hstep' & Hunder1) Hrun. rewrite Hstep'.
    exact (IH s1 s1' Hunder1 Hrun).
  - discriminate.
  - intros Hstep' _. rewrite Hstep'. reflexivity.
Qed.

(* The unit on its own and the unit from [base], before a step. *)
Lemma initial_under (e : expr) (vars : env) (statics : static_env)
    (hp : heap) (rd : read_record) :
  code_closed e = true ->
  static_code_closed statics ->
  state_under (State e vars [] statics hp rd) (State e vars base statics hp rd).
Proof.
  intros Hcode Hstatics.
  apply state_under_intro; [ | exact Hcode | exact Hstatics ].
  intros k _. exact (Under_outer k).
Qed.

End Under.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem run_under : run_under_statement.
Proof.
  intros base e vars statics hp rd n s Hcode Hstatics Hrun.
  destruct
    (run_under_steps base n _ _ s
       (initial_under base e vars statics hp rd_ Hcode Hstatics) Hrun)
    as (s' & Hrun' & Hunder).
  exists s'. split; [ exact Hrun' | exact (proj1 Hunder) ].
Qed.

Theorem exit_under : exit_under_statement.
Proof.
  intros base e vars statics hp rd n last k vs hp' rd' Hcode Hstatics Hrun
    Hstep.
  destruct
    (run_under_steps base n _ _ last
       (initial_under base e vars statics hp rd_ Hcode Hstatics) Hrun)
    as (last' & Hrun' & Hunder).
  exists last'. split; [ exact Hrun' | ].
  split; [ exact (proj1 Hunder) | ].
  pose proof (step_under base last last' Hunder) as H.
  rewrite Hstep in H. exact H.
Qed.

Theorem wrong_under : wrong_under_statement.
Proof.
  intros base e vars statics hp rd n Hcode Hstatics.
  exact (run_under_wrong base n _ _
           (initial_under base e vars statics hp rd_ Hcode Hstatics)).
Qed.

Theorem behaves_under : behaves_under_statement.
Proof.
  intros base e vars statics hp rd Hcode Hstatics. cbv zeta.
  split; [ | split ].
  - intros w b (n & last & Hrun & Hstep & Hlast).
    destruct
      (exit_under base e vars statics hp rd n last _ _ _ rd_ Hcode Hstatics Hrun
         Hstep)
      as (last' & Hrun' & _ & Hstep').
    rewrite Hlast in Hstep'.
    rewrite (run_behaves n _ b), Hrun'. simpl result_behaves.
    rewrite (step_behaves last' b), Hstep'. reflexivity.
  - intros [n Hrun]. exists n.
    exact (wrong_under base e vars statics hp rd n Hcode Hstatics Hrun).
  - intros Hdiv n. destruct (Hdiv n) as [s Hrun].
    destruct (run_under base e vars statics hp rd n s Hcode Hstatics Hrun)
      as (s' & Hrun' & _).
    exists s'. exact Hrun'.
Qed.
