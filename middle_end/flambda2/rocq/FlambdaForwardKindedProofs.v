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

(* Proofs of the statements of FlambdaForwardKindedSpec.v. Nothing here
   needs review.

   The relation between a term and its result is that of
   FlambdaForwardProofs.v, taken strict, with the kinds that the pass
   keeps: an application that goes on has arguments that are known to
   have the kinds of the parameters, and its entry is for a handler
   that gives as many kinds as it has parameters. The first run then
   takes its step where the application is: the arguments have values
   of those kinds (FlambdaNormaliseProofs.kinds_hold,
   [have_kinds_sound]), the test of kinds passes
   and the parameters are bound. So the first run is undefined only
   where the second is, the rounds of FlambdaForwardProofs.v are read
   both ways ([fwd_round], [fwd_round_back]), and each term refines
   the other. That holds under any variables that have the kinds the
   pass starts from ([forward_kinded_under_equivalent_bound]).

   In the body of code the pass starts from the kinds of what a call
   binds, and a call binds the variables of
   FlambdaCodeLetsSpec.code_in_scope so
   (FlambdaCodeLetsProofs.code_in_scope_hold). So the new body
   refines the old one where it runs, and the old one the new, and a
   group of static bindings may have the bodies of its code replaced,
   one piece of code at a time
   (FlambdaContextProofs.let_static_code_refines_where), as in
   FlambdaCodeLetsProofs.v. The rest is the induction of
   FlambdaNormaliseProofs.rewrite_code_equivalent. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaRoundsProofs FlambdaUnusedParamsSpec FlambdaUnusedParamsProofs
  FlambdaParamAliasesSpec FlambdaParamAliasesProofs FlambdaLoopLetsSpec
  FlambdaLoopLetsProofs FlambdaCodeLetsSpec FlambdaCodeLetsProofs
  FlambdaForwardSpec FlambdaForwardKindedSpec FlambdaForwardProofs.
Import ListNotations.

Fixpoint forward_kinded_binds_statics (e : expr) :
  ∀ (G : kinds_in_scope) (K : cont_kinds) (F : forwards),
    binds_statics (forward_kinded_under G K F e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros G K F; simpl; try reflexivity.
  - apply forward_kinded_binds_statics.
  - rewrite forward_kinded_binds_statics. reflexivity.
  - rewrite !forward_kinded_binds_statics. reflexivity.
  - rewrite forward_kinded_binds_statics. f_equal.
    generalize (map (fun '(k, _) => (k, @None (list variable * apply_cont)))
                  handlers
                ++ F).
    intros F'.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite forward_kinded_binds_statics, IH. reflexivity.
Qed.

(* Arguments known to have some kinds have values of those kinds. *)
Lemma have_kinds_sound (G : kinds_in_scope) (args : list simple) :
  ∀ kinds : list flambda_kind,
    have_kinds G args kinds = true -> args_have_kinds G args kinds.
Proof.
  induction args as [ | s args IH]; intros [ | kind kinds] Hhave;
    try discriminate Hhave; intros vars Hhold.
  - exists []. split; reflexivity.
  - simpl in Hhave.
    destruct (kind_of_simple_in_scope G s) as [kind' | ] eqn:Hs;
      [ | discriminate Hhave ].
    apply andb_true_iff in Hhave as [Hkind Hrest].
    apply kind_eqb_eq in Hkind. subst kind'.
    destruct (IH kinds Hrest vars Hhold) as (vs & Hvs & Hkinds).
    destruct (kind_of_simple_in_scope_sound G vars s kind Hhold Hs)
      as (v & Hv & Hkind).
    exists (v :: vs). simpl. rewrite Hv, Hvs, Hkind, Hkinds.
    split; reflexivity.
Qed.

(* The entry that the pass makes is a strict one for the handler, which
   the body leaves alone. *)
Lemma kinded_entry_fits (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler' body : expr) :
  entry_fits true k params metadata handler'
    (kinded_entry k params metadata handler' body)
  ∧ (∀ x : variable,
      frozen_var (kinded_entry k params metadata handler' body) x ->
      binds_var x body = false)
  ∧ (∀ k0 : continuation,
      frozen_cont (kinded_entry k params metadata handler' body) k0 ->
      binds_cont k0 body = false).
Proof.
  unfold kinded_entry.
  destruct (Nat.eqb (length params) (length (hm_params_kinds metadata)))
    eqn:Hlength.
  - apply forwarding_fits. intros _. apply Nat.eqb_eq, Hlength.
  - split; [ exact I | ].
    split; [ intros x (params0 & target & Hsome & _)
           | intros k0 (params0 & target & Hsome & _) ];
      discriminate Hsome.
Qed.

(* An application that the pass forwards has arguments of the kinds of
   the parameters. *)
Lemma goes_forward_kinded (G : kinds_in_scope) (K : cont_kinds) (F : forwards)
    (ac : apply_cont) :
  goes true G K F ac (forward_kinded_apply_cont G K F ac).
Proof.
  unfold forward_kinded_apply_cont.
  destruct (forward_of F (ac_continuation ac)) as [[params target] | ] eqn:Hk;
    [ | apply Goes_same ].
  destruct (lookup (ac_continuation ac) K) as [kinds | ] eqn:Hkinds;
    [ | apply Goes_same ].
  destruct
    (Nat.eqb (length (ac_args ac)) (length params) && no_trap_action target
     && have_kinds G (ac_args ac) kinds)
    eqn:Hcheck; [ | apply Goes_same ].
  apply andb_true_iff in Hcheck as [Hcheck Hhave].
  apply andb_true_iff in Hcheck as [Hlength _].
  apply Nat.eqb_eq in Hlength.
  apply Goes_keeping; [ exact Hk | exact Hlength | ].
  intros _. exists kinds.
  split; [ exact Hkinds | exact (have_kinds_sound G _ kinds Hhave) ].
Qed.

(* The same of an arm. Its case more is an application of no arguments
   to a continuation of no parameters whose handler gives no kinds:
   there is nothing to know of kinds, and it goes on with the trap
   action of the handler. *)
Lemma goes_forward_kinded_arm (G : kinds_in_scope) (K : cont_kinds)
    (F : forwards) (ac : apply_cont) :
  goes true G K F ac (forward_kinded_arm G K F ac).
Proof.
  unfold forward_kinded_arm.
  destruct (forward_of F (ac_continuation ac))
    as [[[ | p params] target] | ] eqn:Hk;
    try apply goes_forward_kinded.
  destruct (lookup (ac_continuation ac) K) as [[ | kind kinds] | ] eqn:Hkinds;
    try apply goes_forward_kinded.
  destruct (ac_args ac) as [ | arg args] eqn:Hargs;
    try apply goes_forward_kinded.
  destruct (no_trap_action ac); [ | apply goes_forward_kinded ].
  replace (forwarded [] [] target) with (forwarded [] (ac_args ac) target)
    by (rewrite Hargs; reflexivity).
  apply Goes_on; [ exact Hk | rewrite Hargs; reflexivity | ].
  intros _. exists []. split; [ exact Hkinds | ].
  rewrite Hargs. exact (have_kinds_sound G [] [] eq_refl).
Qed.

Fixpoint forward_kinded_fwd (e : expr) :
  ∀ (G : kinds_in_scope) (K : cont_kinds) (F : forwards),
    safe F e -> fwd true G K F e (forward_kinded_under G K F e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros G K F Hsafe; simpl.
  - destruct (safe_let F x kind d body Hsafe) as [Hbody Hx].
    apply Fwd_let; [ exact Hx | apply forward_kinded_fwd, Hbody ].
  - apply Fwd_let_static, forward_kinded_fwd. exact Hsafe.
  - destruct (safe_let_cont F k params metadata handler body Hsafe)
      as (Hhandler & Hbody & Hparams & Hk).
    set (handler' :=
      forward_kinded_under
        (params_in_scope params (hm_params_kinds metadata) G) K F handler).
    destruct (kinded_entry_fits k params metadata handler' body)
      as (Hfits & Hvars & Hconts).
    apply (Fwd_let_cont true G K F k params metadata metadata handler handler'
      (kinded_entry k params metadata handler' body));
      [ apply forward_kinded_fwd, Hhandler | reflexivity
      | exact Hparams | exact Hk | exact Hfits | ].
    apply forward_kinded_fwd, safe_cons; assumption.
  - destruct (safe_let_cont_recursive F handlers body Hsafe)
      as [Hbody Hhandlers].
    change (map (fun '(k, _) => (k, @None (list variable * apply_cont)))
              handlers)
      with (plain_forwards handlers).
    assert (Hall :
      Forall
        (fun kh =>
           safe (plain_forwards handlers ++ F) (handler_expr (snd kh))
           ∧ (∀ p : variable, p ∈ handler_params (snd kh) -> var_unfrozen F p)
           ∧ cont_unfrozen F (fst kh))
        handlers).
    { apply Forall_forall. intros [k [params metadata handler]] Hin.
      destruct (Hhandlers k params metadata handler Hin)
        as (Hhandler & Hparams & Hk).
      split; [ apply safe_plain, Hhandler | split; assumption ]. }
    apply Fwd_let_cont_recursive.
    + clear Hsafe Hhandlers. revert Hall.
      generalize (plain_forwards handlers ++ F). intros F' Hall.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        [ apply Forall2_nil | ].
      inversion Hall as [ | kh rest (Hhandler & Hparams & Hk) Hrest].
      subst kh rest.
      apply Forall2_cons; [ | exact (IH Hrest) ].
      simpl. split; [ reflexivity | ]. split; [ reflexivity | ].
      split; [ reflexivity | ].
      split; [ apply forward_kinded_fwd, Hhandler | ].
      split; assumption.
    + clear. unfold handlers_bind_statics.
      generalize (plain_forwards handlers ++ F). intros F'.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        simpl; [ reflexivity | ].
      rewrite forward_kinded_binds_statics, IH. reflexivity.
    + apply forward_kinded_fwd, safe_plain, Hbody.
  - apply Fwd_apply, Returns_same.
  - apply Fwd_apply_cont, goes_forward_kinded.
  - apply Fwd_switch. clear Hsafe.
    induction arms as [ | arm arms IH]; simpl; [ apply Forall2_nil | ].
    apply Forall2_cons; [ | exact IH ].
    split; [ reflexivity | apply goes_forward_kinded_arm ].
  - apply Fwd_invalid.
Qed.

(* The pass that starts from some kinds is an equivalence under the
   variables that have them. *)
Lemma forward_kinded_under_equivalent_bound (G : kinds_in_scope) (e : expr) :
  expr_equivalent_bound G e (forward_kinded_under G [] [] e).
Proof.
  intros vars conts statics hp rd Hhold.
  assert (Hrel :
    state_fwd true (State e vars conts statics hp rd)
      (State (forward_kinded_under G [] [] e) vars conts statics hp rd)).
  { apply (State_fwd true G [] []);
      [ apply forward_kinded_fwd, safe_none; reflexivity
      | intros k; apply Same_fwd | apply held_nil | exact Hhold ]. }
  apply refines_antisymmetric.
  - exact (rounds_refines (state_fwd true) (fwd_round true) _ _ Hrel).
  - exact
      (rounds_refines (fun t' t => state_fwd true t t') fwd_round_back _ _
         Hrel).
Qed.

Theorem forward_kinded_equivalent : forward_kinded_equivalent_statement.
Proof.
  intros e vars conts statics hp rd.
  exact
    (forward_kinded_under_equivalent_bound [] e vars conts statics hp
       rd_ (kinds_hold_nil vars)).
Qed.

(* ======================================================================== *)
(* The pass in the bodies of code                                           *)
(* ======================================================================== *)

Fixpoint forward_kinded_in_code_binds_statics (e : expr) :
  binds_statics (forward_kinded_in_code e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - apply forward_kinded_in_code_binds_statics.
  - rewrite forward_kinded_in_code_binds_statics. f_equal.
    induction bindings
      as [ | [id [params my_closure ret exn metadata code_body] | id
             | sym const]
             bindings IH];
      simpl; congruence.
  - destruct h as [params metadata handler]. simpl.
    rewrite !forward_kinded_in_code_binds_statics. reflexivity.
  - rewrite forward_kinded_in_code_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite forward_kinded_in_code_binds_statics, IH. reflexivity.
Qed.

(* The body of code and the body as rewritten refine each other where
   the code runs. *)
Lemma code_body_forwarded_refines (params : list variable)
    (my_closure : variable) (metadata : code_metadata) (body : expr) :
  code_refines_where always params my_closure metadata body
    (forward_kinded_under (code_in_scope params my_closure metadata) [] []
       body)
  ∧ code_refines_where always params my_closure metadata
      (forward_kinded_under (code_in_scope params my_closure metadata) [] []
         body)
      body.
Proof.
  split; intros vs callee callee_vars conts statics hp rd _ Hkinds Hvalue Hbind;
    apply equivalent_refines; [ | apply equivalent_sym ];
    apply forward_kinded_under_equivalent_bound,
      (code_in_scope_hold params my_closure metadata vs callee callee_vars
         Hkinds Hvalue Hbind).
Qed.

(* A group of static bindings with the bodies of its code rewritten, one
   piece of code at a time: [done] is what has been rewritten. *)
Lemma bindings_forwarded_equivalent (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
      (todo : list static_binding) :
  ∀ done : list static_binding,
    equivalent (State (Let_static (done ++ todo) body) vars conts statics hp rd)
      (State
         (Let_static (done ++ map forward_kinded_in_binding todo) body)
         vars conts statics hp rd).
Proof.
  induction todo as [ | binding todo IH]; intros done; simpl;
    [ apply equivalent_refl | ].
  apply (equivalent_trans _
    (State
       (Let_static (done ++ forward_kinded_in_binding binding :: todo) body)
       vars conts statics hp rd)).
  - destruct binding
      as [id [params my_closure ret exn metadata code_body] | id
         | sym const];
      simpl; try apply equivalent_refl.
    destruct
      (code_body_forwarded_refines params my_closure metadata code_body)
      as [Hthere Hback].
    apply refines_antisymmetric;
      apply (let_static_code_refines_where always always_extend);
      first
        [ exact I | exact Hthere | exact Hback
        | apply forward_kinded_binds_statics
        | symmetry; apply forward_kinded_binds_statics ].
  - specialize (IH (done ++ [forward_kinded_in_binding binding])).
    rewrite <- !app_assoc in IH. exact IH.
Qed.

Fixpoint forward_kinded_in_code_equivalent_aux (e : expr) :
  expr_equivalent e (forward_kinded_in_code e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; intros vars conts statics hp rd; try apply equivalent_refl.
  - apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
    apply forward_kinded_in_code_equivalent_aux.
  - (* down the body, then the code of the group *)
    apply (equivalent_trans _
      (State (Let_static bindings (forward_kinded_in_code body)) vars conts
         statics hp rd)).
    + apply let_static_equivalent_step.
      destruct (eval_static_bindings vars bindings statics) as [extended | ];
        [ | exact I ].
      apply forward_kinded_in_code_equivalent_aux.
    + exact
      (bindings_forwarded_equivalent _ vars conts statics _ rd_ bindings []).
  - apply (equivalent_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata (forward_kinded_in_code handler)) body)
         vars conts statics hp rd)).
    + apply
        (let_cont_handler_equivalent_bound [] k params metadata handler _
           body);
        [ | intros x kind [=] ].
      intros vars' conts' statics' hp' rd' _.
      apply forward_kinded_in_code_equivalent_aux.
    + apply let_cont_non_recursive_equivalent_step.
      apply forward_kinded_in_code_equivalent_aux.
  - apply (equivalent_trans _
      (State
         (Let_cont_recursive
            (map (fun '(k, h) => (k, forward_kinded_in_code_handler h))
               handlers)
            body)
         vars conts statics hp rd)).
    + apply (let_cont_recursive_handlers_equivalent_bound [] handlers _ body);
        [ | | intros x kind [=] ].
      * induction handlers as [ | [k [params metadata handler]] handlers IH];
          simpl; [ apply Forall2_nil | ].
        apply Forall2_cons; [ | exact IH ].
        split; [ reflexivity | ]. simpl.
        exists params, metadata, handler, (forward_kinded_in_code handler).
        split; [ reflexivity | ]. split; [ reflexivity | ].
        intros vars' conts' statics' hp' rd' _.
        apply forward_kinded_in_code_equivalent_aux.
      * unfold handlers_bind_statics.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          simpl; [ reflexivity | ].
        rewrite forward_kinded_in_code_binds_statics, IH. reflexivity.
    + apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
      apply forward_kinded_in_code_equivalent_aux.
Qed.

Theorem forward_kinded_in_code_equivalent :
  forward_kinded_in_code_equivalent_statement.
Proof. exact forward_kinded_in_code_equivalent_aux. Qed.
