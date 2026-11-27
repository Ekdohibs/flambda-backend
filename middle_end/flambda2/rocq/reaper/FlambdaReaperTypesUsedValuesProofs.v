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

(* Proofs of FlambdaReaperTypesUsedValuesSpec.v. This file needs no
   review.

   The invariant of the run of FlambdaReaperTypesTermProofs.v
   ([state_of_values]), with the check of [used]
   ([state_of_values_with]): a step keeps it as it keeps that one, and a
   [Let_static] binds a symbol that [used] holds of to a static value of
   the fragment, by the check of its constant. The simulation of the
   rebuild relates the static environments where the unit exits by
   [statics_rebuilt] from the static values of the symbols with a use
   alone ([statics_rel_rebuilt_used], the proof of
   FlambdaReaperTypesProofs.statics_rel_rebuilt, which reads the fragment
   of the static values of those symbols only); the statements follow as
   those of FlambdaReaperTypesTermProofs.v. The counterexample is
   computed; the rebuilt unit exits from one state only
   ([run_exit_unique]). *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaHeapSpec
  FlambdaPrimitiveProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaWellFormedSpec
  FlambdaWellKindedSpec FlambdaWellKindedProofs FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperRebuild FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperLocalFields FlambdaReaperTypes FlambdaReaperTypesSpec
  FlambdaReaperTypesTermSpec FlambdaReaperTypesUsedValuesSpec
  FlambdaReaperSolutionProofs.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec FlambdaReaperRebuildProofs
  FlambdaReaperFlowProofs FlambdaReaperLeastProofs
  FlambdaReaperLocalFieldsProofs
  FlambdaReaperClosuresExactProofs FlambdaClosuresProofs
  FlambdaReaperTypesProofs FlambdaReaperTypesTermProofs FlambdaReaperUnboxing.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaEnvAliases.
From OxCamlFlambda2Types Require FlambdaTypingEnvProofs
  FlambdaTypeGrammarProofs FlambdaTypeMeaningBoolProofs
  FlambdaEnvAliasesProofs FlambdaBindingTime.
Import ListNotations.

Module RP := FlambdaReaperRebuildProofs.
Module TP := FlambdaReaperTypesProofs.
Module TermP := FlambdaReaperTypesTermProofs.

(* ======================================================================== *)
(* The invariant of the run                                                 *)
(* ======================================================================== *)

Section Used.

Variable used : symbol -> bool.

Lemma unbound_of_values_with (e : expr) :
  ∀ kinds : kind_env,
    binds_statics e = false -> expr_of_values_with used kinds e = true.
Proof.
  revert e. fix IH 1.
  intros [x kind d body | bindings body | k [params metadata handler] body
    | handlers body | ap | ac | scrutinee arms | ] kinds; simpl; intros Hb;
    try reflexivity.
  - exact (IH body _ Hb).
  - apply orb_false_iff in Hb as [Hbinds Hbody].
    rewrite (IH body _ Hbody), andb_true_r.
    apply forallb_forall. intros [id c | id | s const] Hin; try reflexivity.
    exfalso. apply (proj2 (not_true_iff_false _) Hbinds).
    apply existsb_exists. exists (Bind_symbol s const). split; [exact Hin |].
    reflexivity.
  - apply orb_false_iff in Hb as [Hh Hbody].
    rewrite (IH body _ Hbody), andb_true_r.
    destruct (bind_param_kinds params (hm_params_kinds metadata) kinds) as
      [handler_kinds | ]; [ | reflexivity ].
    exact (IH handler _ Hh).
  - apply orb_false_iff in Hb as [_ Hbody]. exact (IH body _ Hbody).
Qed.

Lemma handler_unbound_of_values_with (h : continuation_handler)
    (kinds : kind_env) :
  handler_binds_statics h = false -> handler_of_values_with used kinds h = true.
Proof.
  destruct h as [params metadata handler]. simpl. intros Hh.
  destruct (bind_param_kinds params (hm_params_kinds metadata) kinds) as
    [handler_kinds | ]; [ | reflexivity ].
  exact (unbound_of_values_with handler _ Hh).
Qed.


Lemma statics_of_used_values_empty :
  statics_of_used_values used empty_static_env.
Proof. intros s allocated Hl. discriminate Hl. Qed.

(* A symbol bound to a constant that passes the check. *)
Lemma statics_of_used_values_bind (statics : static_env)
    (code : list (code_id * code)) (vars : env) (s : symbol)
    (const : static_const) (allocated : statically_allocated_value) :
  statics_of_used_values used statics ->
  binding_of_values_with used (kinds_of_env vars) (Bind_symbol s const)
  = true ->
  eval_static_const vars const = Some allocated ->
  statics_of_used_values used
    (Static_env code ((s, allocated) :: static_values statics)).
Proof.
  intros Hstatics Hb Heval s' allocated'. simpl.
  destruct (Pos.eqb s' s) eqn:Hs; [ | exact (Hstatics s' allocated') ].
  apply Pos.eqb_eq in Hs. subst s'.
  intros [= <-] Hu. cbn [binding_of_values_with] in Hb. rewrite Hu in Hb.
  cbn [negb orb] in Hb.
  apply (TermP.statics_of_values_bind empty_static_env code vars s const
           allocated TermP.statics_of_values_empty Hb Heval s allocated).
  simpl. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma eval_static_bindings_of_values_with (vars : env)
    (bindings : list static_binding) :
  ∀ statics statics' : static_env,
    statics_of_used_values used statics ->
    forallb (binding_of_values_with used (kinds_of_env vars)) bindings
    = true ->
    eval_static_bindings vars bindings statics = Some statics' ->
    statics_of_used_values used statics'.
Proof.
  induction bindings as [ | binding bindings IH]; simpl;
    intros statics statics' Hstatics Hall Heval.
  - injection Heval as <-. exact Hstatics.
  - apply andb_true_iff in Hall as [Hbinding Hall].
    destruct binding as [id c | id | s const].
    + destruct (lookup id (static_code statics)); [ discriminate | ].
      refine (IH _ _ _ Hall Heval). exact Hstatics.
    + exact (IH _ _ Hstatics Hall Heval).
    + destruct (lookup s (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
        [ | discriminate ].
      exact (IH _ _
               (statics_of_used_values_bind statics _ vars s const allocated
                  Hstatics Hbinding Hconst)
               Hall Heval).
Qed.


(* ======================================================================== *)

(* The handlers that a continuation keeps pass the check under the kinds
   of the values it keeps. *)
Fixpoint cont_of_values_with (c : cont_value) : bool :=
  match c with
  | Closure h vars conts recursive =>
    handler_of_values_with used (kinds_of_env vars) h
    && forallb
         (fun '(_, h) => handler_of_values_with used (kinds_of_env vars) h)
         recursive
    && forallb (fun '(_, c) => cont_of_values_with c) conts
  | Outer _ => true
  | Extra_args c _ => cont_of_values_with c
  | Over_application _ _ _ return_to exn_to =>
    cont_of_values_with return_to && cont_of_values_with exn_to
  end.

Definition conts_of_values_with (conts : cont_env) : bool :=
  forallb (fun '(_, c) => cont_of_values_with c) conts.

Definition state_of_values_with (s : state) : Prop :=
  expr_of_values_with used (kinds_of_env (state_vars s)) (state_expr s) = true
  ∧ conts_of_values_with (state_conts s) = true
  ∧ statics_of_used_values used (state_statics s).

Lemma resolve_of_values_with (k : continuation) (conts : cont_env) :
  conts_of_values_with conts = true ->
  cont_of_values_with (resolve k conts) = true.
Proof.
  unfold resolve, conts_of_values_with.
  induction conts as [ | [k' c] conts IH]; simpl; [ reflexivity | ].
  intros Hall. apply andb_true_iff in Hall as [Hc Hall].
  destruct (Pos.eqb k k'); [ exact Hc | exact (IH Hall) ].
Qed.

Lemma with_extra_args_of_values_with (c : cont_value) (extra : list value) :
  cont_of_values_with c = true ->
  cont_of_values_with (with_extra_args c extra) = true.
Proof. destruct extra; simpl; intros Hc; exact Hc. Qed.

Lemma rec_closures_of_values_with
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  forallb (fun '(_, h) => handler_of_values_with used (kinds_of_env vars) h)
    handlers
  = true ->
  conts_of_values_with conts = true ->
  conts_of_values_with (rec_closures handlers vars conts) = true.
Proof.
  intros Hall Hconts. unfold rec_closures, conts_of_values_with.
  apply forallb_forall. intros [k c] Hin.
  apply in_map_iff in Hin as [[k' h] [Heq Hin]]. injection Heq as <- <-.
  simpl. rewrite Hall. unfold conts_of_values_with in Hconts. rewrite Hconts.
  rewrite andb_true_r, andb_true_r.
  exact (proj1 (forallb_forall _ _) Hall (k', h) Hin).
Qed.

Lemma conts_of_values_app_with (conts conts' : cont_env) :
  conts_of_values_with conts = true -> conts_of_values_with conts' = true ->
  conts_of_values_with (conts ++ conts') = true.
Proof.
  unfold conts_of_values_with. rewrite forallb_app. intros -> ->. reflexivity.
Qed.

Lemma call_code_of_values_with (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) (s' : state) :
  cont_of_values_with return_to = true -> cont_of_values_with exn_to = true ->
  statics_of_used_values used statics ->
  call_code closure c vs args_arity return_arity return_to exn_to statics hp
    rd
  = Step s' ->
  state_of_values_with s'.
Proof.
  intros Hreturn Hexn Hstatics. unfold call_code.
  destruct (binds_statics (code_body c)) eqn:Hb; [ discriminate | ].
  match goal with
  | |- (if ?test then _ else _) = _ -> _ => destruct test
  end; [ discriminate | ].
  destruct (negb (values_of_subkinds _ _ _)); [ discriminate | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)]);
    [ | discriminate ].
  intros [= <-]. split; [ | split ]; simpl.
  - exact (unbound_of_values_with _ _ Hb).
  - rewrite Hreturn, Hexn. reflexivity.
  - exact Hstatics.
Qed.

Lemma run_call_of_values_with (return_partial : value -> step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (rd : read_record) (s' : state) :
  (∀ (partial : value) (s'' : state),
     return_partial partial = Step s'' -> state_of_values_with s'') ->
  cont_of_values_with return_to = true -> cont_of_values_with exn_to = true ->
  statics_of_used_values used statics ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  = Step s' ->
  state_of_values_with s'.
Proof.
  intros Hpartial Hreturn Hexn Hstatics.
  destruct shape as [[closure c vs args_arity
    | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - exact (call_code_of_values_with _ _ _ _ _ _ _ _ _ _ _ Hreturn Hexn
             Hstatics).
  - apply (call_code_of_values_with _ _ _ _ _ _ _ _ _ _ _); [ | exact Hexn |
      exact Hstatics ].
    simpl. rewrite Hreturn, Hexn. reflexivity.
  - exact (Hpartial partial s').
  - discriminate.
Qed.

Lemma apply_cont_value_of_values_with (c : cont_value) :
  cont_of_values_with c = true ->
  ∀ (vs : list value) (statics : static_env) (hp : heap)
    (rd : read_record) (s' : state),
    statics_of_used_values used statics ->
    apply_cont_value c vs statics hp rd = Step s' ->
    state_of_values_with s'.
Proof.
  induction c as [h def_vars def_conts recursive | k | c IH extra
    | remaining remaining_arity return_arity return_to IHreturn exn_to
      IHexn];
    simpl; intros Hc vs statics hp rd s' Hstatics.
  - destruct h as [params metadata handler]. simpl in Hc |- *.
    apply andb_true_iff in Hc as [Hc Hconts].
    apply andb_true_iff in Hc as [Hh Hrecursive].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
      eqn:Hkinds; [ | discriminate ].
    destruct (values_of_subkinds _ vs (hm_params_subkinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
      [ | discriminate ].
    intros [= <-]. split; [ | split ]; simpl.
    + apply kinds_eqb_eq in Hkinds. rewrite <- Hkinds in Hh.
      rewrite (bind_params_kinds params vs def_vars handler_vars Hbind) in Hh.
      exact Hh.
    + apply conts_of_values_app_with; [ | exact Hconts ].
      exact (rec_closures_of_values_with recursive def_vars def_conts Hrecursive
               Hconts).
    + exact Hstatics.
  - discriminate.
  - exact (IH Hc _ _ _ _ _ Hstatics).
  - apply andb_true_iff in Hc as [Hreturn Hexn].
    destruct vs as [ | v [ | ? ?]]; try discriminate.
    apply (run_call_of_values_with _ _ _ _ _ _ _ _ _); try assumption.
    intros partial s'' Hs''.
    exact (IHreturn Hreturn _ _ _ _ _ Hstatics Hs'').
Qed.

Lemma step_of_values_with (s s' : state) :
  state_of_values_with s -> step s = Step s' -> state_of_values_with s'.
Proof.
  destruct s as [e vars conts statics hp rd].
  intros [He [Hconts Hstatics]]. simpl in He, Hconts, Hstatics.
  destruct e as [x kind d body | bindings body | k h body | handlers body
    | ap | ac | scrutinee arms | ]; simpl in He; cbn [step].
  - destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ];
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind; [ | discriminate ].
    intros [= <-]. split; [ | split ]; simpl; [ | exact Hconts |
      exact Hstatics ].
    apply kind_eqb_eq in Hkind.
    change (kinds_of_env ((x, v) :: vars))
      with ((x, kind_of_value v) :: kinds_of_env vars).
    rewrite Hkind. exact He.
  - apply andb_true_iff in He as [Hbindings Hbody].
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Heval; [ | discriminate ].
    intros [= <-]. split; [ exact Hbody | split; [ exact Hconts | ] ].
    exact (eval_static_bindings_of_values_with vars bindings statics statics'
             Hstatics Hbindings Heval).
  - apply andb_true_iff in He as [Hh Hbody].
    intros [= <-]. split; [ exact Hbody | split; [ | exact Hstatics ] ].
    unfold conts_of_values_with in *. simpl. rewrite Hh, Hconts. reflexivity.
  - destruct (handlers_bind_statics handlers) eqn:Hhb; [ discriminate | ].
    intros [= <-]. split; [ exact He | split; [ | exact Hstatics ] ].
    apply conts_of_values_app_with; [ | exact Hconts ].
    apply rec_closures_of_values_with; [ | exact Hconts ].
    apply forallb_forall. intros [k h] Hin.
    apply handler_unbound_of_values_with.
    destruct (handler_binds_statics h) eqn:Hh; [ | reflexivity ].
    exfalso. apply (proj2 (not_true_iff_false _) Hhb).
    apply existsb_exists. exists (k, h). split; [ exact Hin | exact Hh ].
  - destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    unfold call_function.
    apply (run_call_of_values_with _ _ _ _ _ _ _ _ _).
    + intros partial s'' Hs''.
      exact (apply_cont_value_of_values_with _
               (resolve_of_values_with _ _ Hconts) _ _ _ _ _ Hstatics Hs'').
    + exact (resolve_of_values_with _ _ Hconts).
    + exact (with_extra_args_of_values_with _ _
               (resolve_of_values_with _ _ Hconts)).
    + exact Hstatics.
  - destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    exact (apply_cont_value_of_values_with _
             (resolve_of_values_with _ _ Hconts) _ _
             _ _ _ Hstatics).
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    intros [= <-]. split; [ reflexivity | split; assumption ].
  - discriminate.
Qed.

Lemma run_of_values_with (n : nat) :
  ∀ s s' : state,
    state_of_values_with s -> run n s = Step s' -> state_of_values_with s'.
Proof.
  induction n as [ | n IH]; simpl; intros s s' Hs.
  - intros [= <-]. exact Hs.
  - destruct (step s) as [s'' | | ] eqn:Hstep; try discriminate.
    exact (IH s'' s' (step_of_values_with s s'' Hs Hstep)).
Qed.


End Used.

Theorem unit_of_values_with_runs : unit_of_values_with_runs_statement.
Proof.
  intros used e n s He Hrun.
  assert (Hinit :
            state_of_values_with used
              (State e [] [] empty_static_env empty_heap no_reads)).
  { split; [ exact He | split; [ reflexivity | ] ].
    exact (statics_of_used_values_empty used). }
  exact (proj2 (proj2 (run_of_values_with used n _ _ Hinit Hrun))).
Qed.

(* The check of FlambdaReaperTypesTermSpec.v is the instance where [used]
   holds of every symbol. *)
Lemma expr_of_values_with_of (used : symbol -> bool) (e : expr) :
  ∀ kinds : kind_env,
    expr_of_values kinds e = true -> expr_of_values_with used kinds e = true.
Proof.
  revert e. fix IH 1.
  intros [x kind d body | bindings body | k [params metadata handler] body
    | handlers body | ap | ac | scrutinee arms | ] kinds; simpl; intros He;
    try reflexivity.
  - exact (IH body _ He).
  - apply andb_true_iff in He as [Hb Hbody].
    rewrite (IH body _ Hbody), andb_true_r.
    apply forallb_forall. intros binding Hin.
    pose proof (proj1 (forallb_forall _ _) Hb binding Hin) as H.
    destruct binding as [id c | id | s const]; simpl in H |- *;
      try reflexivity.
    rewrite H, orb_true_r. reflexivity.
  - apply andb_true_iff in He as [Hh Hbody].
    rewrite (IH body _ Hbody), andb_true_r.
    destruct (bind_param_kinds params (hm_params_kinds metadata) kinds) as
      [handler_kinds | ]; [ | reflexivity ].
    exact (IH handler _ Hh).
  - exact (IH body _ He).
Qed.

Theorem unit_of_values_with_instance :
  unit_of_values_with_instance_statement.
Proof. intros used e He. exact (expr_of_values_with_of used e [] He). Qed.

(* ======================================================================== *)
(* The static environments where the unit exits                             *)
(* ======================================================================== *)

Lemma statics_rel_rebuilt_used (is_local : field -> bool) (G : graph)
    (sol : solution) (keep : value_slot -> bool)
    (statics statics' : static_env) :
  closed is_local G sol -> field_used_exact is_local G sol ->
  RP.exit_statics_rel G sol keep statics statics' ->
  RP.statics_no_location statics ->
  statics_of_used_values (fun s => s_has_usage sol (Node_symbol s)) statics ->
  statics_rebuilt G sol statics statics'.
Proof.
  intros Hc Hexact (cds & F & b & ids & ff & cc & cdeps & cmds & ty & located
                    & Hnone & Hblocks & Honly & _)
    Hnl Hvals s allocated Hl Hhu.
  pose proof (Hvals s allocated Hl Hhu) as Hval.
  assert (Hloc : ∀ v : value, v ∈ FlambdaHeapSpec.values_of_static allocated ->
                 locations_of_value v = []).
  { intros v Hv. apply RP.no_location_iff. exact (Hnl s allocated v Hl Hv). }
  destruct (Honly s allocated Hl)
    as [(tag & fields & ->)
       | [(slot & syms & decls & vs & -> & Htarget & Hbound)
         | (Ho & Hbound)]].
  - destruct (Hblocks s tag fields Hl Hhu) as (fields' & Hl' & Hf).
    exists fields'. split; [ exact Hl' | ].
    cbn [FlambdaHeapSpec.values_of_static] in Hloc.
    clear Hl Hl' Hblocks Honly. revert fields' Hf. generalize 0 as i.
    induction fields as [ | field fields IH]; intros i [ | field' fields']
      Hf; cbn [RP.fields_related fields_rebuilt] in Hf |- *;
      try contradiction; [ exact I | ].
    destruct Hf as [(from & Hcons & Hused & Hunused) Hrest].
    assert (Hkf : kind_of_value field = Value)
      by exact (Hval field (or_introl eq_refl)).
    split.
    + unfold field_rebuilt. rewrite Hkf. exists from.
      split; [ exact Hcons | split; [ | exact Hunused ] ].
      intros Hu.
      split; [ exact (TP.used_field_hu is_local G sol Hc Hexact _ _ _ Hu Hcons)
             | ].
      apply (TP.rel_rebuilt_value is_local G sol Hc Hexact _ _ _ _ _ _ field
               from field' (Hused Hu));
        [ exact (Hloc field (or_introl eq_refl)) | ].
      intros c ->. pose proof (Hused Hu) as Hr. cbn [RP.rel] in Hr.
      cbn [kind_of_value] in Hkf. rewrite Hkf in Hr.
      exact (proj1 Hr eq_refl).
    + exact (IH (fun v Hv => Hloc v (or_intror Hv))
               (fun v Hv => Hval v (or_intror Hv)) (S i) fields' Hrest).
  - destruct Htarget as [Hn | (vs' & Hls' & _ & _ & Hsites)];
      [ destruct (Hbound Hhu Hn) | ].
    destruct (Hsites I) as (sites & Hsite & Hse & Hall).
    exists sites, vs'.
    split; [ exact Hls' | split; [ exact Hsite | ] ].
    apply (TP.closure_rebuilt is_local G sol Hc Hexact _ _ _ _ _ _
             (rebuilt_value G sol) decls sites vs vs' Hse Hall).
    intros w x from x' Hlw Hrel Has.
    apply (TP.rel_rebuilt_value is_local G sol Hc Hexact _ _ _ _ _ _ x from x'
             Hrel); [ | exact Has ].
    apply Hloc. cbn [FlambdaHeapSpec.values_of_static].
    apply in_map_iff. exists (w, x). split; [ reflexivity | ].
    exact (lookup_in _ _ _ Hlw).
  - destruct allocated as [ | | | contents | bytes | elements | ];
      cbn [static_rebuilt]; cbn [RP.static_other_rel] in Ho;
      try contradiction.
    + destruct Ho as [Hn | (v' & Hls' & Hv')];
        [ destruct (Hbound Hhu Hn) | ].
      rewrite Hls'. f_equal. f_equal.
      destruct Hv' as [Hv' | Hsim]; [ exact Hv' | ].
      destruct contents as [c | | | | | | | | ];
        cbn [kind_of_value boxable_number_of_kind] in Hval;
        try (exfalso; exact (Hval eq_refl)).
      exact Hsim.
    + destruct Ho as [Hn | Hls']; [ destruct (Hbound Hhu Hn) | exact Hls' ].
    + destruct Ho as [Hn | (elements' & Hls' & _)];
        [ destruct (Hbound Hhu Hn) | ].
      exists elements'. exact Hls'.
Qed.

(* Where the unit exits, the static blocks of the symbols with a use hold
   values. *)
Lemma used_values_exits (T : tables) (e : expr) (n : nat) (s : state)
    (k : continuation) (vs : list value) (hp : heap) (rd : read_record) :
  used_values T e = true ->
  run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
  step s = Exit k vs hp rd ->
  statics_of_used_values
    (fun s => s_has_usage (solution_of_tables T) (Node_symbol s))
    (state_statics s).
Proof.
  intros He Hrun _. exact (unit_of_values_with_runs _ e n s He Hrun).
Qed.

(* FlambdaReaperTypesProofs.exit_relation_of_exact_local with the wider
   check. *)
Lemma exit_relation_used_local (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
  FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
  well_formed_unit u e = true ->
  FlambdaReaperRebuildSpec.exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_local_reads_used is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  variables_bound_once e = true ->
  used_values T e = true ->
  rebuild_exit_relation (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) e
    (rebuild_unit_from_tables (traverse_unit flag u e) T
       preserve_direct_calls e)
  ∧ rebuild_exit_relation_whole (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).
Proof.
  intros Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce Hvals.
  pose proof (FlambdaReaperClosuresExactProofs.closures_exact_of_bound_once
                flag u e Hfrag Hwf Honce) as Hxe.
  assert (H : ∀ (n : nat) (s : state) (k : continuation) (vs : list value)
                (hp : heap) (rd : read_record),
             run n (State e [] [] empty_static_env empty_heap no_reads)
             = Step s ->
             step s = Exit k vs hp rd ->
             statics_allocated (solution_of_tables T) (state_statics s)
             ∧ ∃ (m : nat) (t : state) (vs' : list value)
                 (rd' : read_record),
                 run m
                   (State
                      (rebuild_unit_from_tables (traverse_unit flag u e) T
                         preserve_direct_calls e)
                      [] [] empty_static_env empty_heap no_reads)
                 = Step t
                 ∧ step t = Exit k vs' hp rd'
                 ∧ statics_rebuilt (graph_of_facts (traverse_unit flag u e))
                     (solution_of_tables T) (state_statics s)
                     (state_statics t)
                 ∧ static_sets_whole (solution_of_tables T)
                     (state_statics s) (state_statics t)).
  { intros n s k vs hp rd Hrun Hstep.
    split.
    - exact (proj1 (TP.rebuild_exits_local flag is_local u e T
                      preserve_direct_calls n s k vs hp rd Hlocal Hfrag Hdc
                      Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Hrun Hstep)).
    - destruct (RP.rebuild_exit_statics_local flag is_local u e T
                  preserve_direct_calls Hlocal Hfrag Hdc Hrk Hwf Hexits
                  Hcheck Hlrs Hcheck_exact Hxe n s k vs hp rd Hrun Hstep)
        as (m & t & vs' & rd' & Hm & Ht & Hst & Hnl).
      exists m, t, vs', rd'.
      split; [ exact Hm | split; [ exact Ht | split ] ].
      + exact (statics_rel_rebuilt_used is_local _ _ _ _ _
                 (check_closed_sound is_local _ _ Hcheck)
                 (check_field_used_exact_sound is_local _ _ Hcheck_exact) Hst
                 Hnl (used_values_exits T e n s k vs hp rd Hvals Hrun Hstep)).
      + apply (TP.statics_rel_whole _ _ _ _ _ Hst).
        * exact (FlambdaClosuresProofs.well_formed_sets_bound u e n s Hwf
                   Hrun).
        * exact (TP.rebuilt_sets_bound _ T preserve_direct_calls u e m t Hwf
                   Hm). }
  split.
  - intros n s k vs hp rd Hrun Hstep.
    destruct (H n s k vs hp rd Hrun Hstep)
      as [Halloc (m & t & vs' & rd' & Hm & Ht & Hst & _)].
    split; [ exact Halloc | ]. exists m, t, vs', rd'.
    split; [ exact Hm | split; [ exact Ht | exact Hst ] ].
  - exact H.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem exit_relation_used_term_local :
  exit_relation_used_term_local_statement.
Proof.
  intros flag is_local u e T b Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
    Hcheck_exact Honce Hvals.
  exact (proj1 (exit_relation_used_local flag is_local u e T b Hlocal Hfrag
                  Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce Hvals)).
Qed.

Theorem exit_relation_whole_used_term_local :
  exit_relation_whole_used_term_local_statement.
Proof.
  intros flag is_local u e T b Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
    Hcheck_exact Honce Hvals.
  exact (proj2 (exit_relation_used_local flag is_local u e T b Hlocal Hfrag
                  Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce Hvals)).
Qed.

Theorem exit_relation_used_term : exit_relation_used_term_statement.
Proof.
  intros flag is_local u e T b Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hcheck
    Hcheck_exact.
  exact (exit_relation_used_term_local flag is_local u e T b Hlocal Hfrag Hdc
           Hrk Hwf Hexits Hcheck
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal Hslots)
           Hcheck_exact).
Qed.

Theorem exit_relation_whole_used_term :
  exit_relation_whole_used_term_statement.
Proof.
  intros flag is_local u e T b Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hcheck
    Hcheck_exact.
  exact (exit_relation_whole_used_term_local flag is_local u e T b Hlocal
           Hfrag Hdc Hrk Hwf Hexits Hcheck
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal Hslots)
           Hcheck_exact).
Qed.

Theorem types_end_to_end_used_term_local :
  types_end_to_end_used_term_local_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce Hvals
    Hsets Hrun Hstep Henv Hfr Hcopies Hl.
  destruct (FlambdaReaperLeastProofs.least_local_checks is_local _ T Hleast)
    as [Hcheck Hlrs].
  apply (TP.exported_types_sound_local flag is_local u e T b sets tenv canon
           valuation valuation' n s k vs hp rd sym allocated Hlocal Hleast
           Hsets);
    try assumption.
  exact (exit_relation_used_term_local flag is_local u e T b Hlocal Hfrag Hdc
           Hrk Hwf Hexits Hcheck Hlrs
           (FlambdaReaperLeastProofs.least_field_used_exact_local is_local _ T
              Hleast)
           Honce Hvals).
Qed.

Theorem types_end_to_end_used_term_local_landed :
  types_end_to_end_used_term_local_landed_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce Hvals
    Hsets Hrun Hstep Henv Hfr Hcopies Hl.
  destruct (FlambdaReaperLeastProofs.least_local_checks is_local _ T Hleast)
    as [Hcheck Hlrs].
  apply (TP.exported_types_sound_local_landed flag is_local u e T b sets tenv
           canon valuation valuation' n s k vs hp rd sym allocated Hlocal
           Hleast Hsets);
    try assumption.
  exact (exit_relation_used_term_local flag is_local u e T b Hlocal Hfrag Hdc
           Hrk Hwf Hexits Hcheck Hlrs
           (FlambdaReaperLeastProofs.least_field_used_exact_local is_local _ T
              Hleast)
           Honce Hvals).
Qed.

Theorem types_end_to_end_used_term : types_end_to_end_used_term_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast.
  exact (types_end_to_end_used_term_local flag is_local u e T b sets tenv
           canon valuation valuation' n s k vs hp rd sym allocated Hlocal Hfrag
           Hdc Hrk Hwf Hexits
           (FlambdaReaperLeastProofs.least_local_of_least is_local _ T Hleast
              (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
                 is_local u e T Hlocal Hslots))).
Qed.

Theorem types_end_to_end_used_term_landed :
  types_end_to_end_used_term_landed_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast.
  exact (types_end_to_end_used_term_local_landed flag is_local u e T b sets
           tenv canon valuation valuation' n s k vs hp rd sym allocated Hlocal
           Hfrag Hdc Hrk Hwf Hexits
           (FlambdaReaperLeastProofs.least_local_of_least is_local _ T Hleast
              (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
                 is_local u e T Hlocal Hslots))).
Qed.

Theorem unaliased_types_end_to_end_used_term_local :
  unaliased_types_end_to_end_used_term_local_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce Hvals
    Hsets Hrun Hstep Henv Hfr Hcopies Hl.
  destruct (FlambdaReaperLeastProofs.least_local_checks is_local _ T Hleast)
    as [Hcheck Hlrs].
  apply (TP.exported_unaliased_types_sound_local flag is_local u e T b sets
           tenv canon valuation valuation' n s k vs hp rd sym allocated Hlocal
           Hleast Hsets);
    try assumption.
  exact (exit_relation_whole_used_term_local flag is_local u e T b Hlocal
           Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
           (FlambdaReaperLeastProofs.least_field_used_exact_local is_local _ T
              Hleast)
           Honce Hvals).
Qed.

Theorem unaliased_types_end_to_end_used_term_local_landed :
  unaliased_types_end_to_end_used_term_local_landed_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce Hvals
    Hsets Hrun Hstep Henv Hfr Hcopies Hl.
  destruct (FlambdaReaperLeastProofs.least_local_checks is_local _ T Hleast)
    as [Hcheck Hlrs].
  apply (TP.exported_unaliased_types_sound_local_landed flag is_local u e T b
           sets tenv canon valuation valuation' n s k vs hp rd sym allocated
           Hlocal Hleast Hsets);
    try assumption.
  exact (exit_relation_whole_used_term_local flag is_local u e T b Hlocal
           Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
           (FlambdaReaperLeastProofs.least_field_used_exact_local is_local _ T
              Hleast)
           Honce Hvals).
Qed.

Theorem unaliased_types_end_to_end_used_term :
  unaliased_types_end_to_end_used_term_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast.
  exact (unaliased_types_end_to_end_used_term_local flag is_local u e T b sets
           tenv canon valuation valuation' n s k vs hp rd sym allocated Hlocal
           Hfrag Hdc Hrk Hwf Hexits
           (FlambdaReaperLeastProofs.least_local_of_least is_local _ T Hleast
              (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
                 is_local u e T Hlocal Hslots))).
Qed.

Theorem unaliased_types_end_to_end_used_term_landed :
  unaliased_types_end_to_end_used_term_landed_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast.
  exact (unaliased_types_end_to_end_used_term_local_landed flag is_local u e T
           b sets tenv canon valuation valuation' n s k vs hp rd sym allocated
           Hlocal Hfrag Hdc Hrk Hwf Hexits
           (FlambdaReaperLeastProofs.least_local_of_least is_local _ T Hleast
              (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
                 is_local u e T Hlocal Hslots))).
Qed.

(* ======================================================================== *)
(* The example and the counterexample                                       *)
(* ======================================================================== *)

Theorem dead_naked_block_example : dead_naked_block_example_statement.
Proof.
  split; [ apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 20);
           vm_compute; reflexivity | ].
  split; [ intros f Hf; discriminate Hf | ].
  split; [ intros f; reflexivity | ].
  do 10 (split; [ vm_compute; reflexivity | ]).
  intros b. destruct b; vm_compute; reflexivity.
Qed.

(* A run exits from one state only. *)
Lemma run_exit_unique (n : nat) :
  ∀ (s0 : state) (n' : nat) (t t' : state) (k k' : continuation)
    (vs vs' : list value) (hp hp' : heap) (rd rd' : read_record),
    run n s0 = Step t -> step t = Exit k vs hp rd ->
    run n' s0 = Step t' -> step t' = Exit k' vs' hp' rd' ->
    t = t'.
Proof.
  induction n as [ | n IH];
    intros s0 [ | n'] t t' k k' vs vs' hp hp' rd rd' Hn Ht Hn' Ht'.
  - cbn in Hn, Hn'. congruence.
  - cbn in Hn. injection Hn as <-. cbn in Hn'. rewrite Ht in Hn'.
    discriminate Hn'.
  - cbn in Hn'. injection Hn' as <-. cbn in Hn. rewrite Ht' in Hn.
    discriminate Hn.
  - cbn in Hn, Hn'.
    destruct (step s0) as [s1 | | ]; try discriminate Hn.
    exact (IH s1 n' t t' k k' vs vs' hp hp' rd rd' Hn Ht Hn' Ht').
Qed.

Module NakedUnreadRuns.

Import NakedUnreadExample.
Local Open Scope positive_scope.

Lemma lookup_tenv (n : name) (ty : flambda_type)
    (bt : FlambdaBindingTime.Binding_time.With_name_mode.t) :
  Keyed.lookup name_eqb n (Typing_env.names_to_types tenv) = Some (ty, bt) ->
  n = Name_symbol 50 ∧ (ty = block_type ∨ ty = unknown Value).
Proof.
  intros Hl.
  apply (FlambdaMapProofs.keyed_lookup_in name_eqb
           FlambdaTypeGrammarProofs.name_eqb_eq) in Hl.
  vm_compute in Hl.
  repeat (destruct Hl as [Hl | Hl];
          [ injection Hl as <- <- _; split; [ reflexivity | auto ] | ]).
  destruct Hl.
Qed.

Lemma tenv_satisfied :
  typing_env_satisfied_strictly (state_statics exit_state) [] tenv.
Proof.
  split; [ vm_compute; reflexivity | split ].
  - intros n ty bt w Hl Hw. right.
    destruct (lookup_tenv n ty bt Hl) as [-> [-> | ->]];
      vm_compute in Hw; injection Hw as <-;
      apply FlambdaTypeMeaningBoolProofs.type_satisfied_b_iff;
      vm_compute; reflexivity.
  - change (Typing_env.aliases tenv) with Aliases.empty.
    apply FlambdaEnvAliasesProofs.empty_respected.
Qed.

(* The rebuilt unit, which does not depend on the flag of direct calls. *)
Definition rebuilt : expr :=
  rebuild_unit_from_tables facts tables true unit_expr.

Lemma rebuilt_any (b : bool) :
  rebuild_unit_from_tables facts tables b unit_expr = rebuilt.
Proof. destruct b; vm_compute; reflexivity. Qed.

(* The state from which the rebuilt unit exits. *)
Definition rebuilt_exit : state :=
  match run 2 (State rebuilt [] [] empty_static_env empty_heap no_reads) with
  | Step t => t
  | _ => exit_state
  end.

Lemma rebuilt_run :
  run 2 (State rebuilt [] [] empty_static_env empty_heap no_reads)
  = Step rebuilt_exit.
Proof. vm_compute. reflexivity. Qed.

Lemma rebuilt_step :
  step rebuilt_exit
  = Exit 1 [Value_const (Tagged_immediate 7)] empty_heap no_reads.
Proof. vm_compute. reflexivity. Qed.

End NakedUnreadRuns.

Lemma naked_unread_gen (fx : fixes) :
  rewrite_symbol_type fx
    (plain_context FlambdaReaperUnboxing.Examples.no_local
       NakedUnreadExample.facts NakedUnreadExample.tables [])
    NakedUnreadExample.tenv NakedUnreadExample.canon 50%positive
  = NakedUnreadExample.block_type ->
  naked_unread_counterexample_with fx.
Proof.
  intros Hrw.
  split; [ intros f Hf; discriminate Hf | ].
  split; [ intros f; reflexivity | ].
  do 5 (split; [ vm_compute; reflexivity | ]).
  split; [ apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 20);
           vm_compute; reflexivity | ].
  do 3 (split; [ vm_compute; reflexivity | ]).
  split.
  { (* The graph has no constructor at a function slot. *)
    intros decls sites slot id Hl Hsites. exfalso.
    assert (Hb : is_bound slot decls = true)
      by (unfold is_bound; rewrite Hl; reflexivity).
    pose proof (proj2 (Hsites slot id Hl slot (sites slot))
                  (conj Hb eq_refl)) as Hcons.
    vm_compute in Hcons.
    repeat (destruct Hcons as [Hcons | Hcons]; [ discriminate Hcons | ]).
    exact Hcons. }
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact NakedUnreadRuns.tenv_satisfied | ].
  split.
  { intros n ty bt Hl.
    destruct (NakedUnreadRuns.lookup_tenv n ty bt Hl) as [_ [-> | ->]];
      [ | apply TP.unknown_fragment ].
    split; [ vm_compute; reflexivity | ].
    intros s k w Hin Hw. vm_compute in Hin.
    repeat (destruct Hin as [Hin | Hin];
            [ injection Hin as <- _; cbn in Hw; injection Hw as <-;
              left; eexists; reflexivity | ]).
    destruct Hin. }
  split.
  { intros s x k m Hcan. unfold value_of_name.
    exact (FlambdaTypingEnvProofs.get_canonical_simple_satisfied
             _ _ _ _ _ _ _ _ NakedUnreadRuns.tenv_satisfied Hcan). }
  split; [ reflexivity | ].
  split; [ exact Hrw | ].
  intros b m t vs' rd' Hm Ht Hsat.
  rewrite NakedUnreadRuns.rebuilt_any in Hm.
  rewrite (run_exit_unique m _ 2 t NakedUnreadRuns.rebuilt_exit _ _ _ _ _ _
             _ _ Hm Ht NakedUnreadRuns.rebuilt_run NakedUnreadRuns.rebuilt_step)
    in Hsat.
  rewrite Hrw in Hsat.
  apply FlambdaTypeMeaningBoolProofs.type_satisfied_b_iff in Hsat.
  vm_compute in Hsat. discriminate Hsat.
Qed.

Theorem naked_unread_counterexample : naked_unread_counterexample_statement.
Proof. apply naked_unread_gen. vm_compute. reflexivity. Qed.

Theorem naked_unread_counterexample_landed :
  naked_unread_counterexample_landed_statement.
Proof. apply naked_unread_gen. vm_compute. reflexivity. Qed.

Theorem naked_unread_counterexample_compiler :
  naked_unread_counterexample_compiler_statement.
Proof. apply naked_unread_gen. vm_compute. reflexivity. Qed.
