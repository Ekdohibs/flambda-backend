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

(* Proofs of the statements of FlambdaApplySpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaCompareProofs
  FlambdaApplySpec.
Import ListNotations.

Theorem apply_indirect_known : apply_indirect_known_statement.
Proof.
  intros callee args extra k exn args_arity return_arity v called slots vars
    conts statics hp rd Hcallee Hclosure.
  apply step_eq_equivalent. rewrite !step_apply.
  unfold exn_extra_simples. simpl. rewrite Hcallee.
  destruct (eval_simples vars args) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (map fst extra)) as [extra_vs | ];
    [ | reflexivity ].
  unfold call_function.
  rewrite !(shape_of_call_closure statics v _ _ _ _
              (closure_not_partial statics v _ Hclosure)).
  unfold shape_of_closure_call. rewrite Hclosure, Pos.eqb_refl. reflexivity.
Qed.

(* ======================================================================== *)
(* Binding the parameters                                                   *)
(* ======================================================================== *)

Lemma eval_simple_extend (vars : env) (x : variable) (v : value) (s : simple) :
  ~ var_free_in_simple x s ->
  eval_simple ((x, v) :: vars) s = eval_simple vars s.
Proof.
  destruct s as [y | sym | c]; simpl; try reflexivity.
  intros Hnotfree. destruct (Pos.eqb_spec y x) as [-> | _]; [ | reflexivity ].
  exfalso. apply Hnotfree. reflexivity.
Qed.

Lemma eval_simples_extend (vars : env) (x : variable) (v : value)
    (args : list simple) :
  ~ var_free_in_simples x args ->
  eval_simples ((x, v) :: vars) args = eval_simples vars args.
Proof.
  unfold var_free_in_simples.
  induction args as [ | s args IH]; intros Hnotfree; simpl; [ reflexivity | ].
  rewrite eval_simple_extend, IH; [ reflexivity | | ];
    intros Hfree; apply Hnotfree;
    [ apply Exists_cons_tl | apply Exists_cons_hd ]; exact Hfree.
Qed.

(* The [Let]s of [let_params] run to the body, with the parameters bound as
   a call binds them, provided no parameter is free in an argument and the
   values of the arguments are of the kinds given. *)
Lemma let_params_run (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : list simple) (vs : list value)
    (vars bound : env) (body : expr) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    eval_simples vars args = Some vs ->
    map kind_of_value vs = kinds ->
    (∀ param : variable,
      param ∈ params -> ~ var_free_in_simples param args) ->
    bind_params params vs vars = Some bound ->
    ∃ n : nat,
      run n (State (let_params params kinds args body) vars conts statics hp rd)
      = Step (State body bound conts statics hp rd).
Proof.
  induction params as [ | param params IH];
    intros kinds args vs vars bound body conts statics hp rd Hargs Hkinds Hfresh
      Hbind.
  - destruct vs; [ | discriminate ]. injection Hbind as <-.
    exists 0. reflexivity.
  - destruct vs as [ | v vs]; [ discriminate | ].
    destruct args as [ | arg args]; [ discriminate | ].
    simpl in Hargs, Hbind.
    destruct (eval_simple vars arg) as [v' | ] eqn:Harg; [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ] eqn:Hrest;
      [ | discriminate ].
    simpl in Hargs. injection Hargs as -> ->. subst kinds.
    assert (Hnotfree : ~ var_free_in_simples param args).
    { intros Hfree. apply (Hfresh param (or_introl eq_refl)).
      apply Exists_cons_tl. exact Hfree. }
    destruct
      (IH (map kind_of_value vs) args vs ((param, v) :: vars) bound body
         conts statics hp rd)
      as [n Hrun].
    + rewrite eval_simples_extend; assumption.
    + reflexivity.
    + intros p Hin Hfree. apply (Hfresh p (or_intror Hin)).
      apply Exists_cons_tl. exact Hfree.
    + exact Hbind.
    + exists (S n). simpl. rewrite step_let_simple. simpl.
      rewrite Harg, kind_eqb_refl. exact Hrun.
Qed.

(* ======================================================================== *)
(* Inlining                                                                 *)
(* ======================================================================== *)

(* The step of a direct call, with as many arguments as the code has
   parameters, of a closure of the code named. *)
Lemma step_apply_direct (ap : apply_expr) (callee : value) (called : code_id)
    (slots : list (value_slot * value)) (vs : list value) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
      (c : code) :
  eval_simple vars (ap_callee ap) = Some callee ->
  closure_of statics callee = Some (called, slots) ->
  eval_simples vars (ap_args ap) = Some vs ->
  ap_call_kind ap = Direct called ->
  ap_exn_extra_args ap = [] ->
  lookup called (static_code statics) = Some c ->
  code_is_tupled c = false ->
  length (ap_args_arity ap) = length (code_params_arity c) ->
  step (State (Apply ap) vars conts statics hp rd)
  = if binds_statics (code_body c) then Undefined
    else if
      negb
        (kinds_eqb (map kind_of_value vs) (unarize (ap_args_arity ap))
         && kinds_per_parameter_eqb
              (unarize_per_parameter (ap_args_arity ap))
              (unarize_per_parameter (code_params_arity c))
         && arity_eqb (ap_return_arity ap) (code_result_arity c))
    then Undefined
    else
      match bind_params (code_params c) vs [(code_my_closure c, callee)] with
      | Some callee_vars =>
        Step (State (code_body c) callee_vars
          [(code_return_continuation c,
            resolve (ap_continuation ap) conts);
           (code_exn_continuation c,
            resolve (ap_exn_continuation ap) conts)]
          statics hp rd)
      | None => Undefined
      end.
Proof.
  intros Hcallee Hclosure Hargs Hkind Hextra Hcode Htupled Hfull.
  rewrite (step_apply_no_extra_args _ _ _ _ _ rd_ Hextra), Hcallee, Hargs,
    Hkind.
  unfold call_function. rewrite shape_of_call_direct.
  unfold shape_of_closure_call.
  rewrite Hclosure, Pos.eqb_refl, Hcode,
    (arguments_for_code_not_tupled _ _ _ _ Htupled).
  cbv beta iota. unfold shape_of_code_call.
  rewrite Hfull, Nat.eqb_refl. reflexivity.
Qed.

(* The call goes wrong, or it steps to the body under the callee's
   variables. The inlined body runs its [Let]s to the renamed body under
   those variables followed by the caller's. The two are a state and its
   renaming: the body's free variables are bound alike, and its return and
   exception continuations stand for those of the call. *)
Theorem inline : inline_statement.
Proof.
  intros ap called c r vars conts statics hp rd Hkind Hcode
    (Hclosed_vars & Hclosed_conts & Hvars & Hret & Hexn & Hextra
     & Hmy_closure & Hparams & Harity & Hfull & Hstatic & Htupled).
  pose proof
    (step_undefined_refines (State (Apply ap) vars conts statics hp rd)
       (State (inlined_body c r ap) vars conts statics hp rd))
    as Hwrong.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
    [ | apply Hwrong;
        rewrite (step_apply_no_extra_args _ _ _ _ _ rd_ Hextra), Hcallee;
        reflexivity ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
    [ | apply Hwrong;
        rewrite (step_apply_no_extra_args _ _ _ _ _ rd_ Hextra), Hcallee, Hargs;
        reflexivity ].
  destruct (closure_of statics callee) as [[called' slots] | ] eqn:Hclosure;
    [ | apply Hwrong;
        rewrite (step_apply_no_extra_args _ _ _ _ _ rd_ Hextra), Hcallee, Hargs,
          Hkind;
        unfold call_function; rewrite shape_of_call_direct;
        unfold shape_of_closure_call; rewrite Hclosure; reflexivity ].
  destruct (Pos.eqb_spec called' called) as [-> | Hother];
    [ | apply Hwrong;
        rewrite (step_apply_no_extra_args _ _ _ _ _ rd_ Hextra), Hcallee, Hargs,
          Hkind;
        unfold call_function; rewrite shape_of_call_direct;
        unfold shape_of_closure_call; rewrite Hclosure;
        destruct (Pos.eqb_spec called' called);
        [ contradiction | reflexivity ] ].
  pose proof
    (step_apply_direct ap callee called slots vs vars conts statics hp rd
    c Hcallee Hclosure Hargs Hkind Hextra Hcode Htupled Hfull) as Hstep.
  rewrite Hstatic in Hstep.
  match type of Hstep with
  | context [negb ?agrees] => destruct agrees eqn:Hagrees
  end;
    [ | exact (Hwrong Hstep) ].
  apply Bool.andb_true_iff in Hagrees as [Hagrees _].
  apply Bool.andb_true_iff in Hagrees as [Hkinds Hper_parameter].
  apply kinds_eqb_eq in Hkinds.
  apply kinds_per_parameter_eqb_eq, unarize_per_parameter_eq
    in Hper_parameter.
  destruct (bind_params (code_params c) vs [(code_my_closure c, callee)])
    as [callee_vars | ] eqn:Hbind; [ | exact (Hwrong Hstep) ].
  apply equivalent_refines.
  (* the call *)
  eapply equivalent_trans; [ exact (step_equivalent _ _ Hstep) | ].
  (* the inlined body *)
  destruct (let_params_run (code_params c) (code_params_kinds c)
              (ap_args ap) vs ((code_my_closure c, callee) :: vars)
              (callee_vars ++ vars) (rename_expr r (code_body c)) conts
              statics hp rd)
    as [n Hrun].
  { rewrite eval_simples_extend; assumption. }
  { unfold code_params_kinds. rewrite <- Hper_parameter. exact Hkinds. }
  { exact Hparams. }
  { exact (bind_params_app (code_params c) vs
      [(code_my_closure c, callee)] callee_vars vars Hbind). }
  apply equivalent_sym.
  eapply equivalent_trans.
  { apply (equivalent_of_run (S n) 0). unfold inlined_body.
    rewrite run_succ, step_let_simple. simpl.
    rewrite Hcallee, (closure_of_kind _ _ _ _ Hclosure). exact Hrun. }
  apply equivalent_sym, renamed_equivalent.
  - apply binds_statics_code_closed, Hstatic.
  - intros x Hx. rewrite Hvars, lookup_app.
    destruct (bind_params_bound (code_params c) vs
                [(code_my_closure c, callee)] callee_vars x Hbind)
      as [v Hlookup]; [ | rewrite Hlookup; reflexivity ].
    destruct (Hclosed_vars x Hx) as [-> | Hin]; [ right | left; exact Hin ].
    exists callee. simpl. rewrite Pos.eqb_refl. reflexivity.
  - intros k Hk. unfold conts_agree_at.
    destruct (Hclosed_conts k Hk) as [-> | ->].
    + rewrite Hret. unfold resolve at 1. simpl. rewrite Pos.eqb_refl.
      apply Same_matches.
    + rewrite Hexn. unfold resolve at 1. simpl.
      destruct (Pos.eqb_spec (code_exn_continuation c)
                  (code_return_continuation c)) as [Heq | _].
      * rewrite <- Hexn, Heq, Hret. apply Same_matches.
      * rewrite Pos.eqb_refl. apply Same_matches.
Qed.

(* One step binds the code; then [inline]. *)
Theorem inline_let_code : inline_let_code_statement.
Proof.
  intros ap called c r Hkind Hinlinable vars conts statics hp rd.
  apply let_static_refines_step. simpl.
  destruct (lookup called (static_code statics)) as [bound | ];
    [ exact I | ].
  apply (inline ap called c r); [ exact Hkind | | exact Hinlinable ].
  simpl. rewrite Pos.eqb_refl. reflexivity.
Qed.

(* Swap the return continuation with the call's, then what the exception
   continuation has become with the call's. *)
Theorem inline_renaming_exists : inline_renaming_exists_statement.
Proof.
  intros ret exn k k_exn Hcode Hcall.
  exists (Renaming [] [(swap ret k exn, k_exn); (ret, k)]).
  split; [ reflexivity | ].
  unfold rename_continuation. simpl. split.
  - assert (Hk : swap ret k ret = k)
      by (unfold swap; rewrite Pos.eqb_refl; reflexivity).
    rewrite Hk. unfold swap at 1.
    destruct (Pos.eqb_spec k (swap ret k exn)) as [Heq | _].
    + rewrite <- Hk in Heq at 1. apply swap_injective in Heq.
      contradiction.
    + destruct (Pos.eqb_spec k k_exn); [ contradiction | reflexivity ].
  - unfold swap at 1. rewrite Pos.eqb_refl. reflexivity.
Qed.

(* ======================================================================== *)
(* Inlining as an equivalence                                               *)
(* ======================================================================== *)

Lemma eval_simples_length (vars : env) (args : list simple) :
  ∀ vs : list value,
    eval_simples vars args = Some vs -> length vs = length args.
Proof.
  induction args as [ | s args IH]; intros vs; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s); [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ]; [ | discriminate ].
    simpl. intros [= <-]. simpl. rewrite (IH vs' eq_refl). reflexivity.
Qed.

Lemma bind_params_some (params : list variable) :
  ∀ (vs : list value) (vars : env),
    length params = length vs ->
    ∃ bound : env, bind_params params vs vars = Some bound.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars; simpl;
    try discriminate; [ eauto | ].
  intros [= Hlength]. apply IH, Hlength.
Qed.

(* Where an argument has no value, the [Let]s of [let_params] go wrong. *)
Lemma let_params_wrong (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : list simple) (vars : env)
    (body : expr) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record),
    length params = length args ->
    eval_simples vars args = None ->
    (∀ param : variable,
      param ∈ params -> ~ var_free_in_simples param args) ->
    behaves (State (let_params params kinds args body) vars conts statics hp rd)
      Goes_wrong.
Proof.
  induction params as [ | param params IH];
    intros kinds [ | arg args] vars body conts statics; simpl;
    try discriminate.
  intros hp rd [= Hlength] Hargs Hfresh.
  destruct (eval_simple vars arg) as [v | ] eqn:Harg.
  - destruct (eval_simples vars args) as [vs | ] eqn:Hrest;
      [ discriminate | ].
    destruct (kind_eqb (kind_of_value v) (hd Value kinds)) eqn:Hkind;
      [ | apply step_undefined_wrong; rewrite step_let_simple; simpl;
          rewrite Harg, Hkind; reflexivity ].
    apply (step_behaves _ Goes_wrong). rewrite step_let_simple. simpl.
    rewrite Harg, Hkind.
    apply (IH (tl kinds) args _ body conts statics hp rd_ Hlength).
    + rewrite eval_simples_extend; [ exact Hrest | ].
      intros Hfree. apply (Hfresh param (or_introl eq_refl)).
      apply Exists_cons_tl. exact Hfree.
    + intros p Hin Hfree. apply (Hfresh p (or_intror Hin)).
      apply Exists_cons_tl. exact Hfree.
  - apply step_undefined_wrong. rewrite step_let_simple. simpl. rewrite Harg.
    reflexivity.
Qed.

(* Where the arguments have values that are not of the kinds given, one
   for each parameter, the [Let]s of [let_params] go wrong: one of them
   tests a value against another kind. *)
Lemma let_params_wrong_kinds (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : list simple) (vs : list value)
    (vars : env) (body : expr) (conts : cont_env) (statics : static_env)
      (hp : heap) (rd : read_record),
    length params = length args ->
    length params = length kinds ->
    eval_simples vars args = Some vs ->
    map kind_of_value vs <> kinds ->
    (∀ param : variable,
      param ∈ params -> ~ var_free_in_simples param args) ->
    behaves (State (let_params params kinds args body) vars conts statics hp rd)
      Goes_wrong.
Proof.
  induction params as [ | param params IH];
    intros [ | kind kinds] [ | arg args] vs vars body conts statics; simpl;
    try discriminate.
  - intros hp rd _ _ [= <-] Hkinds. destruct Hkinds. reflexivity.
  - intros hp rd [= Hlength] [= Hlength'] Hargs Hkinds Hfresh.
    destruct (eval_simple vars arg) as [v | ] eqn:Harg; [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ] eqn:Hrest;
      [ | discriminate ].
    simpl in Hargs. injection Hargs as <-.
    destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind;
      [ | apply step_undefined_wrong; rewrite step_let_simple; simpl;
          rewrite Harg, Hkind; reflexivity ].
    apply (step_behaves _ Goes_wrong). rewrite step_let_simple. simpl.
    rewrite Harg, Hkind.
    apply (IH kinds args vs' _ body conts statics hp rd_ Hlength Hlength').
    + rewrite eval_simples_extend; [ exact Hrest | ].
      intros Hfree. apply (Hfresh param (or_introl eq_refl)).
      apply Exists_cons_tl. exact Hfree.
    + intros Heq. apply Hkinds. simpl.
      rewrite (proj1 (kind_eqb_eq _ _) Hkind), Heq. reflexivity.
    + intros p Hin Hfree. apply (Hfresh p (or_intror Hin)).
      apply Exists_cons_tl. exact Hfree.
Qed.

(* Two states that go wrong are equivalent. *)
Lemma both_wrong_equivalent (s s' : state) :
  behaves s Goes_wrong -> behaves s' Goes_wrong -> equivalent s s'.
Proof.
  intros Hs Hs' b. split; intros Hb.
  - rewrite (behaviour_unique _ _ _ Hb Hs). exact Hs'.
  - rewrite (behaviour_unique _ _ _ Hb Hs'). exact Hs.
Qed.

(* Where the callee is a closure of the code called, the call is made at
   the arities of the code, and the code gives one kind for each
   parameter, the call cannot be undefined for being of something else or
   for its arities, and inlining is an equivalence: if the arguments have
   values of the kinds of the call's arity the two meet as in [inline];
   if they have values of other kinds the call is undefined and so is a
   [Let] of the inlined body; and if they have no values both go
   wrong. *)
Lemma inline_equivalent (ap : apply_expr) (called : code_id) (c : code)
    (r : renaming) (callee : value) (slots : list (value_slot * value))
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  ap_call_kind ap = Direct called ->
  eval_simple vars (ap_callee ap) = Some callee ->
  closure_of statics callee = Some (called, slots) ->
  lookup called (static_code statics) = Some c ->
  inlinable ap c r ->
  unarize_per_parameter (ap_args_arity ap)
  = unarize_per_parameter (code_params_arity c) ->
  ap_return_arity ap = code_result_arity c ->
  length (code_params c) = length (code_params_kinds c) ->
  equivalent (State (Apply ap) vars conts statics hp rd)
    (State (inlined_body c r ap) vars conts statics hp rd).
Proof.
  intros Hkind Hcallee Hclosure Hcode
    (Hclosed_vars & Hclosed_conts & Hvars & Hret & Hexn & Hextra
     & Hmy_closure & Hparams & Harity & Hfull & Hstatic & Htupled)
    Hargs_arity Hreturn_arity Hparams_kinds.
  destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
    [ destruct
        (list_eq_dec kind_eq_dec (map kind_of_value vs)
           (unarize (ap_args_arity ap)))
        as [Hkinds | Hkinds]
    | ].
  - pose proof (step_apply_direct ap callee called slots vs vars conts
      statics hp rd c Hcallee Hclosure Hargs Hkind Hextra Hcode Htupled Hfull)
      as Hstep.
    rewrite Hstatic in Hstep.
    rewrite Hkinds, Hargs_arity, Hreturn_arity, kinds_eqb_refl,
      arity_eqb_refl in Hstep.
    rewrite (proj2 (kinds_per_parameter_eqb_eq _ _) eq_refl) in Hstep.
    cbn [andb negb] in Hstep.
    destruct (bind_params_some (code_params c) vs
                [(code_my_closure c, callee)])
      as [callee_vars Hbind].
    { rewrite (eval_simples_length vars _ vs Hargs). exact Harity. }
    rewrite Hbind in Hstep.
    eapply equivalent_trans; [ exact (step_equivalent _ _ Hstep) | ].
    destruct (let_params_run (code_params c) (code_params_kinds c)
                (ap_args ap) vs ((code_my_closure c, callee) :: vars)
                (callee_vars ++ vars) (rename_expr r (code_body c)) conts
                statics hp rd)
      as [n Hrun].
    { rewrite eval_simples_extend; assumption. }
    { unfold code_params_kinds.
      rewrite <- (unarize_per_parameter_eq _ _ Hargs_arity).
      exact Hkinds. }
    { exact Hparams. }
    { exact (bind_params_app (code_params c) vs
        [(code_my_closure c, callee)] callee_vars vars Hbind). }
    apply equivalent_sym.
    eapply equivalent_trans.
    { apply (equivalent_of_run (S n) 0). unfold inlined_body.
      rewrite run_succ, step_let_simple. simpl.
      rewrite Hcallee, (closure_of_kind _ _ _ _ Hclosure). exact Hrun. }
    apply equivalent_sym, renamed_equivalent.
    + apply binds_statics_code_closed, Hstatic.
    + intros x Hx. rewrite Hvars, lookup_app.
      destruct (bind_params_bound (code_params c) vs
                  [(code_my_closure c, callee)] callee_vars x Hbind)
        as [v Hlookup]; [ | rewrite Hlookup; reflexivity ].
      destruct (Hclosed_vars x Hx) as [-> | Hin]; [ right | left; exact Hin ].
      exists callee. simpl. rewrite Pos.eqb_refl. reflexivity.
    + intros k Hk. unfold conts_agree_at.
      destruct (Hclosed_conts k Hk) as [-> | ->].
      * rewrite Hret. unfold resolve at 1. simpl. rewrite Pos.eqb_refl.
        apply Same_matches.
      * rewrite Hexn. unfold resolve at 1. simpl.
        destruct (Pos.eqb_spec (code_exn_continuation c)
                    (code_return_continuation c)) as [Heq | _].
        -- rewrite <- Hexn, Heq, Hret. apply Same_matches.
        -- rewrite Pos.eqb_refl. apply Same_matches.
  - apply both_wrong_equivalent.
    + apply step_undefined_wrong.
      rewrite (step_apply_direct ap callee called slots vs vars conts
        statics _ rd_ c Hcallee Hclosure Hargs Hkind Hextra Hcode Htupled
          Hfull),
        Hstatic.
      destruct
        (kinds_eqb (map kind_of_value vs) (unarize (ap_args_arity ap)))
        eqn:Heq; [ | reflexivity ].
      apply kinds_eqb_eq in Heq. contradiction.
    + apply (step_behaves _ Goes_wrong). unfold inlined_body.
      rewrite step_let_simple. simpl.
      rewrite Hcallee, (closure_of_kind _ _ _ _ Hclosure).
      apply (let_params_wrong_kinds _ _ _ vs);
        [ exact Harity | exact Hparams_kinds | | | exact Hparams ].
      * rewrite eval_simples_extend; assumption.
      * unfold code_params_kinds.
        rewrite <- (unarize_per_parameter_eq _ _ Hargs_arity).
        exact Hkinds.
  - apply both_wrong_equivalent.
    + apply step_undefined_wrong.
      rewrite (step_apply_no_extra_args _ _ _ _ _ rd_ Hextra), Hcallee, Hargs.
      reflexivity.
    + apply (step_behaves _ Goes_wrong). unfold inlined_body.
      rewrite step_let_simple. simpl.
      rewrite Hcallee, (closure_of_kind _ _ _ _ Hclosure).
      apply let_params_wrong; [ exact Harity | | exact Hparams ].
      rewrite eval_simples_extend; assumption.
Qed.
