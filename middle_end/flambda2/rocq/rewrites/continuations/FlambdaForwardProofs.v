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

(* Proofs of the statements of FlambdaForwardSpec.v. Nothing here needs
   review.

   The pass changes the applications of a continuation on the strength
   of its handler, so it is not a rewrite of two expressions on their
   own; and where the first term applies a continuation whose handler
   applies another, the second applies that other at once. The term and
   its result are run side by side by rounds
   (FlambdaRoundsProofs.rounds_refines): a round is one step of each,
   but for such an application, where the first run takes the steps
   through the handlers and the second one step. The relation between
   the two states ([state_fwd]) is:

   - the second expression is the first with some applications replaced
     by the [Apply_cont] of the handler of their continuation, the
     arguments put for the parameters ([fwd], [goes]), as the entries of
     some [forwards] say; no binder in scope binds a name that an entry
     depends on;
   - the variables of the two states are the same;
   - each continuation name stands, in the two states, for the same
     thing or for closures whose handlers are so related, under the same
     variables and continuations in turn so related ([cont_fwd]);
   - for each entry, the name stands in the second state for a closure
     whose handler is the [Apply_cont] of the entry; the variables of the
     state agree with those of that closure on the variables that the
     [Apply_cont] reads, and the continuation that it applies is the same
     in the state as in the closure ([held]).

   The round at an application that goes on ([cont_fwd_rounds]) is by
   induction on how the closures are related: the handler of the first
   closure is an application that the handler of the second is related
   to, itself the same or one that goes on.

   A call of the second term may return to the continuation that the
   handler of the return continuation of the first passes its
   parameters on to ([returns]). What the callee is given to return to
   is then, in the first run, a continuation related to the closure of
   an entry and, in the second, what that closure applies
   ([Alias_fwd]). Results given to the first reach, after the steps
   through the handlers, what answers the same results given to the
   second: it is the round of an application that goes on, with the
   parameters themselves for arguments. So a round at a call or at an
   application of related continuations is some steps of the first
   run to one of the second. The relation between continuations has a
   flag for whether such a pair is allowed; a name that has an entry
   never stands for one, as the round of an application that goes on
   needs the closure of the entry.

   The relation is taken with a flag ([strict]). Without it, the second
   run answers the first where the first is defined, which is the
   refinement. With it, an application that goes on has arguments
   that are known, by the binders around it, to have the kinds that
   the handler of its continuation gives its parameters, as many as
   the handler has parameters; the first run then takes its step there
   too, the second run is undefined where the first is, and a round is
   read the other way too ([fwd_round_back]).
   FlambdaForwardJumpsProofs.v and FlambdaForwardKindedProofs.v make
   equivalences of that. For it the relation carries the kinds of the
   variables bound around a point and those that the handlers of the
   continuations bound around it give their parameters
   (FlambdaNormaliseSpec.kinds_in_scope, [kinds_of_conts]), and the
   two states have variables of those kinds
   (FlambdaNormaliseProofs.kinds_hold). *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaCompareSpec FlambdaCompareProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaRoundsProofs
  FlambdaSinkContsProofs FlambdaUnusedParamsSpec FlambdaUnusedParamsProofs
  FlambdaParamAliasesSpec FlambdaParamAliasesProofs FlambdaForwardSpec.
From OxCamlFlambda2 Require Import FlambdaValueSpec FlambdaPrimitiveProofs.
Import ListNotations.

(* ======================================================================== *)
(* What is known of the continuations in scope                              *)
(* ======================================================================== *)

Definition entry : Type := option (list variable * apply_cont).

Lemma forward_of_head (F : forwards) (k : continuation) (e : entry) :
  forward_of ((k, e) :: F) k = e.
Proof. unfold forward_of. simpl. rewrite Pos.eqb_refl. reflexivity. Qed.

Lemma forward_of_other (F : forwards) (k k' : continuation) (e : entry) :
  k' <> k -> forward_of ((k, e) :: F) k' = forward_of F k'.
Proof.
  intros Hne. unfold forward_of. simpl.
  destruct (Pos.eqb_spec k' k); [ contradiction | reflexivity ].
Qed.

(* The entries that the handlers of a recursive [Let_cont] add. *)
Definition plain_forwards
    (handlers : list (continuation * continuation_handler)) : forwards :=
  map (fun '(k, _) => (k, None)) handlers.

Lemma forward_of_plain (handlers : list (continuation * continuation_handler))
    (F : forwards) (k : continuation) :
  forward_of (plain_forwards handlers ++ F) k
  = match lookup k handlers with
    | Some _ => None
    | None => forward_of F k
    end.
Proof.
  unfold forward_of.
  induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

(* The names that an entry depends on: the variables that its
   [Apply_cont] reads and that are not its parameters, and the
   continuation that it applies. *)
Definition frozen_var (e : entry) (x : variable) : Prop :=
  ∃ (params : list variable) (target : apply_cont),
    e = Some (params, target)
    ∧ x ∉ params
    ∧ var_free_in_simples x (ac_args target).

Definition frozen_cont (e : entry) (k : continuation) : Prop :=
  ∃ (params : list variable) (target : apply_cont),
    e = Some (params, target) ∧ ac_continuation target = k.

Definition var_unfrozen (F : forwards) (x : variable) : Prop :=
  ∀ k : continuation, ~ frozen_var (forward_of F k) x.

Definition cont_unfrozen (F : forwards) (k : continuation) : Prop :=
  ∀ k' : continuation, ~ frozen_cont (forward_of F k') k.

Lemma var_unfrozen_none (F : forwards) (x : variable) :
  (∀ k : continuation, forward_of F k = None) -> var_unfrozen F x.
Proof.
  intros Hnone k (params & target & Hsome & _). rewrite Hnone in Hsome.
  discriminate Hsome.
Qed.

Lemma cont_unfrozen_none (F : forwards) (k : continuation) :
  (∀ k' : continuation, forward_of F k' = None) -> cont_unfrozen F k.
Proof.
  intros Hnone k' (params & target & Hsome & _). rewrite Hnone in Hsome.
  discriminate Hsome.
Qed.

(* ======================================================================== *)
(* Putting the arguments for the parameters                                 *)
(* ======================================================================== *)

(* What is put for a variable has, under the variables of the
   application, the value that the variable has once the parameters are
   bound; a variable for which nothing is put is not a parameter. *)
Lemma lookup_combine_params (params : list variable) :
  ∀ (args : list simple) (vs : list value) (vars def_vars bound : env)
    (x : variable),
    NoDup params ->
    eval_simples vars args = Some vs ->
    bind_params params vs def_vars = Some bound ->
    match lookup x (combine params args) with
    | Some arg => eval_simple vars arg = lookup x bound
    | None => x ∉ params
    end.
Proof.
  induction params as [ | p params IH];
    intros args vs vars def_vars bound x Hnodup Hargs Hbind;
    [ simpl; auto | ].
  inversion Hnodup as [ | p0 rest Hnotin Hnodup']. subst p0 rest.
  destruct args as [ | a args].
  { simpl in Hargs. injection Hargs as <-. discriminate Hbind. }
  simpl in Hargs.
  destruct (eval_simple vars a) as [v | ] eqn:Ha; [ | discriminate ].
  destruct (eval_simples vars args) as [rest | ] eqn:Hrest;
    [ | discriminate ].
  injection Hargs as <-. simpl in Hbind. simpl.
  destruct (Pos.eqb_spec x p) as [-> | Hne].
  - rewrite (lookup_bind_params_other params rest _ bound p Hbind Hnotin),
      lookup_head.
    exact Ha.
  - specialize (IH args rest vars ((p, v) :: def_vars) bound x Hnodup' Hrest
      Hbind).
    destruct (lookup x (combine params args)); [ exact IH | ].
    intros [Heq | Hin]; [ congruence | exact (IH Hin) ].
Qed.

Lemma eval_subst_simple (params : list variable) (args : list simple)
    (vs : list value) (vars def_vars bound : env) (s : simple) :
  NoDup params ->
  eval_simples vars args = Some vs ->
  bind_params params vs def_vars = Some bound ->
  (∀ x : variable,
    x ∉ params -> var_free_in_simple x s -> lookup x vars = lookup x def_vars)
  ->
  eval_simple vars (subst_simple params args s) = eval_simple bound s.
Proof.
  intros Hnodup Hargs Hbind Hfrozen.
  destruct s as [x | sym | c]; simpl; try reflexivity.
  pose proof (lookup_combine_params params args vs vars def_vars bound x
    Hnodup Hargs Hbind) as Hlookup.
  destruct (lookup x (combine params args)) as [arg | ]; [ exact Hlookup | ].
  simpl. rewrite (Hfrozen x Hlookup eq_refl). symmetry.
  exact (lookup_bind_params_other params vs def_vars bound x Hbind Hlookup).
Qed.

Lemma eval_subst_simples (params : list variable) (args : list simple)
    (vs : list value) (vars def_vars bound : env) (l : list simple) :
  NoDup params ->
  eval_simples vars args = Some vs ->
  bind_params params vs def_vars = Some bound ->
  (∀ x : variable,
    x ∉ params -> var_free_in_simples x l -> lookup x vars = lookup x def_vars)
  ->
  eval_simples vars (map (subst_simple params args) l) = eval_simples bound l.
Proof.
  intros Hnodup Hargs Hbind. unfold var_free_in_simples.
  induction l as [ | s l IH]; intros Hfrozen; simpl; [ reflexivity | ].
  rewrite (eval_subst_simple params args vs vars def_vars bound s Hnodup Hargs
    Hbind), IH;
    [ reflexivity | | ].
  - intros x Hnotin Hx. apply Hfrozen; [ exact Hnotin | ].
    apply Exists_cons_tl, Hx.
  - intros x Hnotin Hx. apply Hfrozen; [ exact Hnotin | ].
    apply Exists_cons_hd, Hx.
Qed.

(* ======================================================================== *)
(* Variables of the kinds of their binders                                  *)
(* ======================================================================== *)

(* The variables that a term binds have values of the kinds of their
   binders: FlambdaNormaliseProofs.kinds_hold, with [kinds_hold_cons]
   for a [Let] and [params_in_scope_hold] for the parameters of a
   handler. *)
Lemma kinds_hold_nil (vars : env) : kinds_hold [] vars.
Proof. intros x kind Hx. discriminate Hx. Qed.

Lemma bind_params_length (params : list variable) :
  ∀ (vs : list value) (vars : env),
    length params = length vs ->
    ∃ bound : env, bind_params params vs vars = Some bound.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars Hlength;
    try discriminate Hlength; simpl; [ eauto | ].
  injection Hlength as Hlength. exact (IH vs _ Hlength).
Qed.

(* The kinds that the handlers of the continuations bound around a
   point give their parameters. *)
Definition kinds_of_conts : Type := list (continuation * list flambda_kind).

(* The arguments have values of these kinds, one each, wherever the
   variables in scope have values of theirs. *)
Definition args_have_kinds (G : kinds_in_scope) (args : list simple)
    (kinds : list flambda_kind) : Prop :=
  ∀ vars : env,
    kinds_hold G vars ->
    ∃ vs : list value,
      eval_simples vars args = Some vs ∧ map kind_of_value vs = kinds.

(* ======================================================================== *)
(* An expression with applications that go on                               *)
(* ======================================================================== *)

(* The relation below is used twice. As it stands ([strict] false) the
   second run answers the first where the first is defined, which is
   what a refinement asks. Where the arguments of every application
   that goes on are known to have the kinds of the parameters
   ([strict] true), the second run is undefined where the first is,
   and each run answers the other (FlambdaForwardJumpsProofs.v,
   FlambdaForwardKindedProofs.v). *)
Section Strict.

Variable strict : bool.

(* The second application is the first, or the [Apply_cont] of the
   handler of its continuation, as the entry of that says, with the
   trap action of the handler or with that of the first. Where
   [strict], the arguments of the first are known, by the binders
   around it, to have the kinds that the handler of its continuation
   gives its parameters. *)
Inductive goes (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
    : apply_cont -> apply_cont -> Prop :=
  | Goes_same (ac : apply_cont) : goes G K F ac ac
  | Goes_on (ac : apply_cont) (params : list variable) (target : apply_cont) :
    forward_of F (ac_continuation ac) = Some (params, target) ->
    length (ac_args ac) = length params ->
    (strict = true ->
     ∃ kinds : list flambda_kind,
       lookup (ac_continuation ac) K = Some kinds
       ∧ args_have_kinds G (ac_args ac) kinds) ->
    goes G K F ac (forwarded params (ac_args ac) target)
  | Goes_keeping (ac : apply_cont) (params : list variable)
      (target : apply_cont) :
    forward_of F (ac_continuation ac) = Some (params, target) ->
    length (ac_args ac) = length params ->
    (strict = true ->
     ∃ kinds : list flambda_kind,
       lookup (ac_continuation ac) K = Some kinds
       ∧ args_have_kinds G (ac_args ac) kinds) ->
    goes G K F ac
      (forwarded_keeping (ac_trap_action ac) params (ac_args ac) target).

(* What [strict] asks of an entry: a handler that gives as many kinds
   as it has parameters, so that it runs on values of those kinds. *)
Definition strict_entry (params : list variable)
    (metadata' : handler_metadata) : Prop :=
  strict = true -> length params = length (hm_params_kinds metadata').

(* What an entry for [k] says of a handler with these parameters, as the
   second term has it. *)
Definition entry_fits (k : continuation) (params : list variable)
    (metadata' : handler_metadata) (handler' : expr) (e : entry) : Prop :=
  match e with
  | None => True
  | Some (params0, target) =>
    params0 = params
    ∧ handler' = Apply_cont target
    ∧ NoDup params
    ∧ ac_continuation target <> k
    ∧ strict_entry params metadata'
  end.

(* The second call is the first, or returns to the continuation that
   the handler of the return continuation of the first passes its
   parameters on to, as the entry of that says. Not where [strict]. *)
Inductive returns (F : forwards) : apply_expr -> apply_expr -> Prop :=
  | Returns_same (ap : apply_expr) : returns F ap ap
  | Returns_on (ap : apply_expr) (params : list variable)
      (target : apply_cont) :
    strict = false ->
    forward_of F (ap_continuation ap) = Some (params, target) ->
    ac_args target = map Var params ->
    returns F ap
      (Apply_expr (ap_callee ap) (ap_call_kind ap) (ap_args ap)
         (ac_continuation target) (ap_exn_continuation ap)
         (ap_exn_extra_args ap) (ap_args_arity ap) (ap_return_arity ap)).

(* The second expression is the first with applications that go on,
   and calls that return further on. No binder binds a name that an
   entry in scope depends on. (The warning silenced is about induction
   schemes, which nothing uses.) *)
#[local] Set Warnings "-register-all".
Inductive fwd
    : kinds_in_scope -> kinds_of_conts -> forwards -> expr -> expr -> Prop :=
  | Fwd_let (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
      (x : variable)
      (kind : flambda_kind) (d : named) (body body' : expr) :
    var_unfrozen F x ->
    fwd ((x, kind) :: G) K F body body' ->
    fwd G K F (Let x kind d body) (Let x kind d body')
  | Fwd_let_static (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
      (bindings : list static_binding) (body body' : expr) :
    fwd G K F body body' ->
    fwd G K F (Let_static bindings body) (Let_static bindings body')
  | Fwd_let_cont (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
      (k : continuation) (params : list variable)
      (metadata metadata' : handler_metadata) (handler handler' : expr)
      (e : entry) (body body' : expr) :
    fwd (params_in_scope params (hm_params_kinds metadata) G) K F handler
      handler' ->
    hm_params_kinds metadata' = hm_params_kinds metadata ->
    (∀ p : variable, p ∈ params -> var_unfrozen F p) ->
    cont_unfrozen F k ->
    entry_fits k params metadata' handler' e ->
    fwd G ((k, hm_params_kinds metadata') :: K) ((k, e) :: F) body body' ->
    fwd G K F
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata' handler') body')
  | Fwd_let_cont_recursive (G : kinds_in_scope) (K : kinds_of_conts)
      (F : forwards)
      (handlers handlers' : list (continuation * continuation_handler))
      (body body' : expr) :
    Forall2
      (fun kh kh' =>
         fst kh = fst kh'
         ∧ handler_params (snd kh) = handler_params (snd kh')
         ∧ handler_kinds (snd kh') = handler_kinds (snd kh)
         ∧ fwd
             (params_in_scope (handler_params (snd kh))
                (handler_kinds (snd kh)) G)
             K (plain_forwards handlers ++ F) (handler_expr (snd kh))
             (handler_expr (snd kh'))
         ∧ (∀ p : variable, p ∈ handler_params (snd kh) -> var_unfrozen F p)
         ∧ cont_unfrozen F (fst kh))
      handlers handlers' ->
    handlers_bind_statics handlers' = handlers_bind_statics handlers ->
    fwd G K (plain_forwards handlers ++ F) body body' ->
    fwd G K F (Let_cont_recursive handlers body)
      (Let_cont_recursive handlers' body')
  | Fwd_apply (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
      (ap ap' : apply_expr) :
    returns F ap ap' -> fwd G K F (Apply ap) (Apply ap')
  | Fwd_apply_cont (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
      (ac ac' : apply_cont) :
    goes G K F ac ac' -> fwd G K F (Apply_cont ac) (Apply_cont ac')
  | Fwd_switch (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
      (scrutinee : simple) (arms arms' : list (Z * apply_cont)) :
    Forall2
      (fun arm arm' => fst arm = fst arm' ∧ goes G K F (snd arm) (snd arm'))
      arms arms' ->
    fwd G K F (Switch scrutinee arms) (Switch scrutinee arms')
  | Fwd_invalid (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards) :
    fwd G K F Invalid Invalid.

(* The handlers of a recursive [Let_cont], one with one. *)
Definition handlers_fwd (G : kinds_in_scope) (K : kinds_of_conts)
    (inside F : forwards)
    (handlers handlers' : list (continuation * continuation_handler)) : Prop :=
  Forall2
    (fun kh kh' =>
       fst kh = fst kh'
       ∧ handler_params (snd kh) = handler_params (snd kh')
       ∧ handler_kinds (snd kh') = handler_kinds (snd kh)
       ∧ fwd
           (params_in_scope (handler_params (snd kh))
              (handler_kinds (snd kh)) G)
           K inside (handler_expr (snd kh)) (handler_expr (snd kh'))
       ∧ (∀ p : variable, p ∈ handler_params (snd kh) -> var_unfrozen F p)
       ∧ cont_unfrozen F (fst kh))
    handlers handlers'.

Lemma handlers_fwd_lookup (G : kinds_in_scope) (K : kinds_of_conts)
    (inside F : forwards) (k : continuation)
    (handlers handlers' : list (continuation * continuation_handler)) :
  handlers_fwd G K inside F handlers handlers' ->
  match lookup k handlers, lookup k handlers' with
  | Some h, Some h' =>
    handler_params h = handler_params h'
    ∧ handler_kinds h' = handler_kinds h
    ∧ fwd (params_in_scope (handler_params h) (handler_kinds h) G) K inside
        (handler_expr h) (handler_expr h')
    ∧ (∀ p : variable, p ∈ handler_params h -> var_unfrozen F p)
  | None, None => True
  | _, _ => False
  end.
Proof.
  induction 1 as [ | [k1 h] [k1' h'] handlers handlers' (Hk & Hh) _ IH];
    simpl; [ exact I | ].
  simpl in Hk, Hh. subst k1'.
  destruct (Pos.eqb k k1); [ | exact IH ].
  destruct Hh as (Hparams & Hmeta & Hexpr & Hunfrozen & _). auto.
Qed.

(* The names of related handlers are not names that an entry depends
   on, and are the same. *)
Lemma handlers_fwd_names (G : kinds_in_scope) (K : kinds_of_conts)
    (inside F : forwards) (k : continuation)
    (handlers handlers' : list (continuation * continuation_handler)) :
  handlers_fwd G K inside F handlers handlers' ->
  lookup k handlers <> None ->
  cont_unfrozen F k.
Proof.
  induction 1 as [ | [k1 h] [k1' h'] handlers handlers' (Hk & Hh) _ IH];
    simpl; [ congruence | ].
  destruct (Pos.eqb_spec k k1) as [-> | _]; [ intros _ | exact IH ].
  simpl in Hh. apply Hh.
Qed.

Lemma find_arm_goes (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
    (i : Z) (arms arms' : list (Z * apply_cont)) :
  Forall2
    (fun arm arm' => fst arm = fst arm' ∧ goes G K F (snd arm) (snd arm'))
    arms arms' ->
  match find_arm i arms, find_arm i arms' with
  | Some ac, Some ac' => goes G K F ac ac'
  | None, None => True
  | _, _ => False
  end.
Proof.
  induction 1 as [ | [d ac] [d' ac'] arms arms' [Hd Hgoes] _ IH]; simpl;
    [ exact I | ].
  simpl in Hd, Hgoes. subst d'.
  destruct (Z.eqb i d); [ exact Hgoes | exact IH ].
Qed.

(* Every expression is related to itself, where no entry says anything. *)
Fixpoint fwd_refl (e : expr) :
  ∀ (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards),
    (∀ k : continuation, forward_of F k = None) -> fwd G K F e e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros G K F Hnone.
  - apply Fwd_let; [ apply var_unfrozen_none, Hnone | apply fwd_refl, Hnone ].
  - apply Fwd_let_static, fwd_refl, Hnone.
  - apply (Fwd_let_cont G K F k params metadata metadata handler handler
      None);
      [ apply fwd_refl, Hnone | reflexivity
      | intros p _; apply var_unfrozen_none, Hnone
      | apply cont_unfrozen_none, Hnone | exact I | ].
    apply fwd_refl. intros k'.
    destruct (Pos.eq_dec k' k) as [-> | Hne];
      [ apply forward_of_head
      | rewrite forward_of_other by exact Hne; apply Hnone ].
  - assert (Hnone' :
      ∀ k : continuation, forward_of (plain_forwards handlers ++ F) k = None).
    { intros k. rewrite forward_of_plain.
      destruct (lookup k handlers); [ reflexivity | apply Hnone ]. }
    apply Fwd_let_cont_recursive; [ | reflexivity | apply fwd_refl, Hnone' ].
    revert Hnone'. generalize (plain_forwards handlers ++ F).
    intros F' Hnone'.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      [ apply Forall2_nil | ].
    apply Forall2_cons; [ | exact IH ].
    simpl. split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ]. split; [ apply fwd_refl, Hnone' | ].
    split; [ intros p _; apply var_unfrozen_none, Hnone
           | apply cont_unfrozen_none, Hnone ].
  - apply Fwd_apply, Returns_same.
  - apply Fwd_apply_cont, Goes_same.
  - apply Fwd_switch.
    induction arms as [ | arm arms IH]; [ apply Forall2_nil | ].
    apply Forall2_cons; [ split; [ reflexivity | apply Goes_same ] | exact IH ].
  - apply Fwd_invalid.
Qed.

(* ======================================================================== *)
(* Continuations and states                                                 *)
(* ======================================================================== *)

(* For each entry, the name stands in the second state for a closure
   whose handler is the [Apply_cont] of the entry; the variables agree
   with those of the closure on what that [Apply_cont] reads, the
   continuation that it applies is the same in the state as in the
   closure, and the entry is as [strict] asks. *)
Definition held (K : kinds_of_conts) (F : forwards) (vars : env)
    (conts' : cont_env) : Prop :=
  ∀ (k : continuation) (params : list variable) (target : apply_cont),
    forward_of F k = Some (params, target) ->
    NoDup params
    ∧ ∃ (metadata' : handler_metadata) (def_vars : env) (def_conts' : cont_env),
        resolve k conts'
        = Closure (Handler params metadata' (Apply_cont target)) def_vars
            def_conts' []
        ∧ (∀ x : variable,
            x ∉ params ->
            var_free_in_simples x (ac_args target) ->
            lookup x vars = lookup x def_vars)
        ∧ resolve (ac_continuation target) conts'
          = resolve (ac_continuation target) def_conts'
        ∧ strict_entry params metadata'
        ∧ (strict = true -> lookup k K = Some (hm_params_kinds metadata')).

(* Whether a name has no entry. *)
Definition no_entry (F : forwards) (k : continuation) : bool :=
  match forward_of F k with
  | Some _ => false
  | None => true
  end.

Lemma no_entry_some (F : forwards) (k : continuation)
    (params : list variable) (target : apply_cont) :
  forward_of F k = Some (params, target) -> no_entry F k = false.
Proof. intros Hentry. unfold no_entry. rewrite Hentry. reflexivity. Qed.

(* What a continuation name stands for in the two runs: the same thing;
   or closures whose handlers are related under the entries of where
   they were defined, under the same variables, and continuations that
   are in turn so related and for which those entries hold, the second
   handler giving the kinds of the first; or two that are related, with
   the same extra arguments; or what is left of the same
   over-application, to return and to raise to two that are related.

   Those are the pairs of the relation at [false]. At [true] there is
   one more: the second is what the handler of a closure passes its
   parameters on to, and the first is related at [false] to that
   closure. It is what a call of the second term that returns further
   on gives its callee to return to. A name that has an entry stands
   for a pair at [false]: the round of an application that goes on
   needs the closure of the entry itself. *)
Inductive cont_fwd : bool -> cont_value -> cont_value -> Prop :=
  | Same_fwd (b : bool) (c : cont_value) : cont_fwd b c c
  | Closure_fwd (b : bool) (G : kinds_in_scope) (K : kinds_of_conts)
      (F : forwards)
      (params : list variable) (metadata metadata' : handler_metadata)
      (handler handler' : expr) (def_vars : env)
      (def_conts def_conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    handlers_fwd G K (plain_forwards recursive ++ F) F recursive
      recursive' ->
    fwd (params_in_scope params (hm_params_kinds metadata) G) K
      (plain_forwards recursive ++ F) handler handler' ->
    hm_params_kinds metadata' = hm_params_kinds metadata ->
    (∀ p : variable, p ∈ params -> var_unfrozen F p) ->
    (∀ k : continuation,
      cont_fwd (no_entry F k) (resolve k def_conts) (resolve k def_conts')) ->
    held K F def_vars def_conts' ->
    kinds_hold G def_vars ->
    cont_fwd b
      (Closure (Handler params metadata handler) def_vars def_conts recursive)
      (Closure (Handler params metadata' handler') def_vars def_conts'
         recursive')
  | Extra_args_fwd (b : bool) (c c' : cont_value) (extra : list value) :
    cont_fwd true c c' ->
    cont_fwd b (Extra_args c extra) (Extra_args c' extra)
  | Over_fwd (b : bool) (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    cont_fwd true return_to return_to' ->
    cont_fwd true exn_to exn_to' ->
    cont_fwd b
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to')
  | Alias_fwd (c : cont_value) (params : list variable)
      (metadata' : handler_metadata) (target : apply_cont) (def_vars : env)
      (def_conts' : cont_env) :
    strict = false ->
    cont_fwd false c
      (Closure (Handler params metadata' (Apply_cont target)) def_vars
         def_conts' []) ->
    NoDup params ->
    ac_args target = map Var params ->
    cont_fwd true c (resolve (ac_continuation target) def_conts').

(* A pair at [false] is one at [true]. *)
Lemma cont_fwd_any (b : bool) (c c' : cont_value) :
  cont_fwd false c c' -> cont_fwd b c c'.
Proof.
  intros Hrel. inversion Hrel; subst.
  - apply Same_fwd.
  - apply (Closure_fwd b G K F); assumption.
  - apply Extra_args_fwd. assumption.
  - apply Over_fwd; assumption.
Qed.

Lemma conts_any (F : forwards) (conts conts' : cont_env) :
  (∀ k : continuation,
    cont_fwd (no_entry F k) (resolve k conts) (resolve k conts')) ->
  ∀ k : continuation, cont_fwd true (resolve k conts) (resolve k conts').
Proof.
  intros Hconts k. specialize (Hconts k).
  destruct (no_entry F k); [ exact Hconts | apply cont_fwd_any, Hconts ].
Qed.

Inductive state_fwd : state -> state -> Prop :=
  | State_fwd (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
      (e e' : expr)
      (vars : env) (conts conts' : cont_env) (statics : static_env)
        (hp : heap) (rd : read_record) :
    fwd G K F e e' ->
    (∀ k : continuation,
      cont_fwd (no_entry F k) (resolve k conts) (resolve k conts')) ->
    held K F vars conts' ->
    kinds_hold G vars ->
    state_fwd (State e vars conts statics hp rd)
      (State e' vars conts' statics hp rd).

Lemma held_nil (K : kinds_of_conts) (vars : env) (conts' : cont_env) :
  held K [] vars conts'.
Proof. intros k params target Hsome. discriminate Hsome. Qed.

Lemma state_fwd_refl (s : state) : state_fwd s s.
Proof.
  destruct s as [e vars conts statics].
  apply (State_fwd [] [] []);
    [ apply fwd_refl; reflexivity | intros k; apply Same_fwd | apply held_nil
    | apply kinds_hold_nil ].
Qed.

(* Binding a variable that no entry depends on. *)
Lemma held_var (K : kinds_of_conts) (F : forwards) (vars : env)
    (conts' : cont_env) (x : variable) (v : value) :
  var_unfrozen F x ->
  held K F vars conts' ->
  held K F ((x, v) :: vars) conts'.
Proof.
  intros Hx Hheld k params target Hsome.
  destruct (Hheld k params target Hsome)
    as (Hnodup & metadata' & def_vars & def_conts' & Hk & Hvars & Hcont
        & Hse).
  split; [ exact Hnodup | ]. exists metadata', def_vars, def_conts'.
  split; [ exact Hk | ]. split; [ | split; [ exact Hcont | exact Hse ] ].
  intros y Hnotin Hfree. rewrite <- (Hvars y Hnotin Hfree). simpl.
  destruct (Pos.eqb_spec y x) as [-> | _]; [ | reflexivity ].
  exfalso. apply (Hx k). exists params, target. auto.
Qed.

(* Binding parameters that no entry depends on. *)
Lemma held_bind (K : kinds_of_conts) (F : forwards) (params : list variable)
    (vs : list value)
    (vars bound : env) (conts' : cont_env) :
  (∀ p : variable, p ∈ params -> var_unfrozen F p) ->
  bind_params params vs vars = Some bound ->
  held K F vars conts' ->
  held K F bound conts'.
Proof.
  intros Hparams Hbind Hheld k params0 target Hsome.
  destruct (Hheld k params0 target Hsome)
    as (Hnodup & metadata' & def_vars & def_conts' & Hk & Hvars & Hcont
        & Hse).
  split; [ exact Hnodup | ]. exists metadata', def_vars, def_conts'.
  split; [ exact Hk | ]. split; [ | split; [ exact Hcont | exact Hse ] ].
  intros y Hnotin Hfree. rewrite <- (Hvars y Hnotin Hfree).
  apply (lookup_bind_params_other params vs vars bound y Hbind).
  intros Hin. apply (Hparams y Hin k). exists params0, target. auto.
Qed.

(* The closures of the handlers of a recursive [Let_cont], whose names no
   entry depends on. *)
Lemma held_rec (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
    (handlers handlers' : list (continuation * continuation_handler))
    (vars def_vars : env) (def_conts' : cont_env) :
  handlers_fwd G K (plain_forwards handlers ++ F) F handlers handlers' ->
  held K F vars def_conts' ->
  held K (plain_forwards handlers ++ F) vars
    (rec_closures handlers' def_vars def_conts' ++ def_conts').
Proof.
  intros Hhandlers Hheld k params target Hsome.
  rewrite forward_of_plain in Hsome.
  pose proof (handlers_fwd_lookup _ _ _ _ k _ _ Hhandlers) as Hlookup.
  destruct (lookup k handlers) eqn:Hk; [ discriminate Hsome | ].
  destruct (lookup k handlers') eqn:Hk'; [ contradiction | ].
  destruct (Hheld k params target Hsome)
    as (Hnodup & metadata' & def_vars0 & def_conts0 & Hresolve & Hvars
        & Hcont & Hse).
  split; [ exact Hnodup | ]. exists metadata', def_vars0, def_conts0.
  rewrite resolve_rec_closures, Hk'. split; [ exact Hresolve | ].
  split; [ exact Hvars | ]. split; [ | exact Hse ].
  rewrite resolve_rec_closures.
  pose proof
    (handlers_fwd_lookup _ _ _ _ (ac_continuation target) _ _ Hhandlers)
    as Htarget.
  destruct (lookup (ac_continuation target) handlers') eqn:Ht';
    [ | exact Hcont ].
  destruct (lookup (ac_continuation target) handlers) eqn:Ht;
    [ | contradiction ].
  exfalso.
  apply (handlers_fwd_names _ _ _ _ (ac_continuation target) _ _ Hhandlers
    ltac:(rewrite Ht; discriminate) k).
  exists params, target. auto.
Qed.

Lemma rec_conts_fwd (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
    (handlers handlers' : list (continuation * continuation_handler))
    (def_vars : env) (def_conts def_conts' : cont_env) :
  handlers_fwd G K (plain_forwards handlers ++ F) F handlers handlers' ->
  (∀ k : continuation,
    cont_fwd (no_entry F k) (resolve k def_conts) (resolve k def_conts')) ->
  held K F def_vars def_conts' ->
  kinds_hold G def_vars ->
  ∀ k : continuation,
    cont_fwd (no_entry (plain_forwards handlers ++ F) k)
      (resolve k (rec_closures handlers def_vars def_conts ++ def_conts))
      (resolve k (rec_closures handlers' def_vars def_conts' ++ def_conts')).
Proof.
  intros Hhandlers Hconts Hheld Htyped k. rewrite !resolve_rec_closures.
  pose proof (handlers_fwd_lookup _ _ _ _ k _ _ Hhandlers) as Hlookup.
  unfold no_entry. rewrite forward_of_plain.
  destruct (lookup k handlers) as [[params metadata handler] | ],
    (lookup k handlers') as [[params' metadata' handler'] | ];
    try contradiction; [ | exact (Hconts k) ].
  destruct Hlookup as (Hparams & Hmeta & Hhandler & Hunfrozen).
  simpl in Hparams, Hmeta, Hhandler, Hunfrozen. subst params'.
  apply (Closure_fwd true G K F); assumption.
Qed.

(* ======================================================================== *)
(* Applying related continuations                                           *)
(* ======================================================================== *)

(* What the first run comes to is answered by what one step of the
   second comes to. Where the first is undefined nothing is asked of
   the second, unless [strict]: then it is undefined too. *)
Definition answer (result result' : step_result) : Prop :=
  match result with
  | Undefined => strict = true -> result' = Undefined
  | Exit k vs hp rd => result' = Exit k vs hp rd
  | Step t => ∃ t' : state, result' = Step t' ∧ state_fwd t t'
  end.

Lemma answer_refl (result : step_result) : answer result result.
Proof.
  destruct result as [t | k vs | ]; simpl;
    [ | reflexivity | intros _; reflexivity ].
  exists t. split; [ reflexivity | apply state_fwd_refl ].
Qed.

Lemma answer_undefined : answer Undefined Undefined.
Proof. intros _. reflexivity. Qed.

(* Where [strict], the arguments of an application that goes on have
   values, of the kinds that the handler of its entry gives its
   parameters and as many as it has parameters: the closure of the
   entry runs on them. *)
Lemma strict_runs (G : kinds_in_scope) (K : kinds_of_conts) (k : continuation)
    (args : list simple) (params : list variable)
    (metadata' : handler_metadata) (vars : env) :
  kinds_hold G vars ->
  strict_entry params metadata'
  ∧ (strict = true -> lookup k K = Some (hm_params_kinds metadata')) ->
  (strict = true ->
   ∃ kinds : list flambda_kind,
     lookup k K = Some kinds ∧ args_have_kinds G args kinds) ->
  strict = true ->
  ∃ vs : list value,
    eval_simples vars args = Some vs
    ∧ map kind_of_value vs = hm_params_kinds metadata'
    ∧ length params = length vs.
Proof.
  intros Htyped [Hse Hkinds] Hhave Hstrict.
  destruct (Hhave Hstrict) as (kinds & Hlookup & Hargs).
  rewrite (Hkinds Hstrict) in Hlookup. injection Hlookup as <-.
  destruct (Hargs vars Htyped) as (vs & Hvs & Hk).
  exists vs. split; [ exact Hvs | ]. split; [ exact Hk | ].
  rewrite (Hse Hstrict), <- Hk. apply length_map.
Qed.

(* What is asked of the values that the closure of an entry is applied
   to, where [strict]. *)
Definition runs_on_values (params : list variable)
    (metadata' : handler_metadata) (vs : list value) : Prop :=
  strict = true ->
  map kind_of_value vs = hm_params_kinds metadata'
  ∧ length params = length vs.

(* What a result comes to after some more steps. *)
Definition run_on (n : nat) (result : step_result) : step_result :=
  match result with
  | Step s => run n s
  | Exit k vs hp rd => Exit k vs hp rd
  | Undefined => Undefined
  end.

Lemma run_on_zero (result : step_result) : run_on 0 result = result.
Proof. destruct result; reflexivity. Qed.

Lemma run_one (s : state) : run 1 s = step s.
Proof. simpl. destruct (step s); reflexivity. Qed.

(* What answers an undefined result is one. *)
Lemma answer_undefined_inv (result : step_result) :
  answer result Undefined -> result = Undefined.
Proof.
  destruct result as [t | k vs | ]; simpl;
    [ intros (t' & Hstep & _); discriminate Hstep
    | intros Hexit; discriminate Hexit | reflexivity ].
Qed.

(* The parameters, read once they are bound, are the values given. *)
Lemma eval_params_bound (params : list variable) :
  ∀ (vs : list value) (def_vars bound : env),
    NoDup params ->
    bind_params params vs def_vars = Some bound ->
    eval_simples bound (map Var params) = Some vs.
Proof.
  induction params as [ | p params IH]; intros vs def_vars bound Hnodup Hbind.
  - destruct vs; [ reflexivity | discriminate Hbind ].
  - destruct vs as [ | v vs]; [ discriminate Hbind | ].
    inversion Hnodup as [ | p0 rest Hnotin Hnodup']. subst p0 rest.
    simpl in Hbind. cbn [map eval_simples eval_simple].
    rewrite (lookup_bind_params_other params vs _ bound p Hbind Hnotin),
      lookup_head, (IH vs _ bound Hnodup' Hbind).
    reflexivity.
Qed.

(* No variable but a parameter is read by arguments that are the
   parameters. *)
Lemma params_only (params : list variable) (x : variable) :
  x ∉ params -> ~ var_free_in_simples x (map Var params).
Proof.
  intros Hnotin Hfree. unfold var_free_in_simples in Hfree.
  apply Exists_exists in Hfree as (s & Hin & Hs).
  apply in_map_iff in Hin as (p & <- & Hp). simpl in Hs. subst p.
  exact (Hnotin Hp).
Qed.

Lemma run_succ_on (n : nat) (s : state) : run (S n) s = run_on n (step s).
Proof. reflexivity. Qed.

(* What the relation says of a handler that has become one [Apply_cont],
   and of the two applications. *)
Lemma fwd_apply_cont_inv (G : kinds_in_scope) (K : kinds_of_conts)
    (F : forwards) (e : expr) (target : apply_cont) :
  fwd G K F e (Apply_cont target) ->
  ∃ b : apply_cont, e = Apply_cont b ∧ goes G K F b target.
Proof. intros Hfwd. inversion Hfwd; subst. eauto. Qed.

Lemma goes_inv (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
    (b target : apply_cont) :
  goes G K F b target ->
  target = b
  ∨ ∃ (params_b : list variable) (target_b : apply_cont)
      (action : option trap_action),
      forward_of F (ac_continuation b) = Some (params_b, target_b)
      ∧ length (ac_args b) = length params_b
      ∧ (strict = true ->
         ∃ kinds : list flambda_kind,
           lookup (ac_continuation b) K = Some kinds
           ∧ args_have_kinds G (ac_args b) kinds)
      ∧ target = forwarded_keeping action params_b (ac_args b) target_b.
Proof.
  intros [ ac
         | ac params_b target_b Hentry Hlength Hkinds
         | ac params_b target_b Hentry Hlength Hkinds ];
    [ left; reflexivity | right | right ].
  - exists params_b, target_b, (ac_trap_action target_b). auto.
  - exists params_b, target_b, (ac_trap_action ac). auto.
Qed.

(* The machine does not execute a trap action: the step of an
   [Apply_cont] that goes on is the same with any. *)
Lemma step_forwarded_keeping (action : option trap_action)
    (params : list variable) (args : list simple) (target : apply_cont)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  step
    (State (Apply_cont (forwarded_keeping action params args target)) vars
       conts statics hp rd)
  = step (State (Apply_cont (forwarded params args target)) vars conts statics
      hp rd).
Proof. rewrite !step_apply_cont. reflexivity. Qed.

(* The step of an [Apply_cont] that goes on is that of the [Apply_cont]
   of the handler under the variables of the closure with the parameters
   bound, and the continuations of the closure. *)
Lemma step_forwarded (params : list variable) (args : list simple)
    (target : apply_cont) (vs : list value) (vars def_vars bound : env)
    (conts' def_conts' : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  NoDup params ->
  eval_simples vars args = Some vs ->
  bind_params params vs def_vars = Some bound ->
  (∀ x : variable,
    x ∉ params ->
    var_free_in_simples x (ac_args target) ->
    lookup x vars = lookup x def_vars) ->
  resolve (ac_continuation target) conts'
  = resolve (ac_continuation target) def_conts' ->
  step
    (State (Apply_cont (forwarded params args target)) vars conts' statics hp
       rd)
  = step (State (Apply_cont target) bound def_conts' statics hp rd).
Proof.
  intros Hnodup Hargs Hbind Hvars Hcont. rewrite !step_apply_cont.
  unfold forwarded. cbn [ac_args ac_continuation].
  rewrite (eval_subst_simples params args vs vars def_vars bound
    (ac_args target) Hnodup Hargs Hbind Hvars).
  unfold apply_continuation. rewrite Hcont. reflexivity.
Qed.

(* The same full call, returning and raising to related continuations. *)
Lemma call_code_fwd (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_fwd true return_to return_to' ->
  cont_fwd true exn_to exn_to' ->
  answer
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c vs args_arity return_arity return_to' exn_to'
       statics hp rd).
Proof.
  intros Hreturn Hraise.
  destruct (call_code_cases closure c vs args_arity return_arity statics hp rd)
    as [Hundefined | (callee_vars & Hcall)].
  - rewrite !Hundefined. apply answer_undefined.
  - rewrite !Hcall. cbn [answer]. eexists. split; [ reflexivity | ].
    apply (State_fwd [] [] []);
      [ apply fwd_refl; reflexivity | | apply held_nil | apply kinds_hold_nil ].
    intros k'. change (no_entry [] k') with true. unfold resolve. simpl.
    destruct (Pos.eqb k' (code_return_continuation c)); [ exact Hreturn | ].
    destruct (Pos.eqb k' (code_exn_continuation c));
      [ exact Hraise | apply Same_fwd ].
Qed.

(* The same call of any shape, where the two return continuations take
   a partial application to results of which, after some steps of the
   first, one answers the other. *)
Lemma run_call_fwd (shape : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_fwd true return_to return_to' ->
  cont_fwd true exn_to exn_to' ->
  (∀ partial : value,
    ∃ n : nat,
      answer (run_on n (apply_cont_value return_to [partial] statics hp rd))
        (apply_cont_value return_to' [partial] statics hp rd)) ->
  ∃ n : nat,
    answer
      (run_on n
         (run_call
            (fun partial => apply_cont_value return_to [partial] statics hp rd)
            shape return_arity return_to exn_to statics hp rd))
      (run_call
         (fun partial => apply_cont_value return_to' [partial] statics hp rd)
         shape return_arity return_to' exn_to' statics hp rd).
Proof.
  intros Hreturn Hraise Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    cbn [run_call].
  - exists 0. rewrite run_on_zero. apply call_code_fwd; assumption.
  - exists 0. rewrite run_on_zero.
    apply call_code_fwd; [ apply Over_fwd; assumption | exact Hraise ].
  - apply Hpartial.
  - exists 0. rewrite run_on_zero. apply answer_undefined.
Qed.

(* Related continuations applied to the same values: what the first
   comes to, after some steps, is answered by what the second comes
   to. And where the two are related at [false] and the second is the
   closure of an entry, the first applied to the values of some
   arguments comes, after some steps, to what answers one step of the
   [Apply_cont] of the entry with those arguments put for the
   parameters. *)
Lemma cont_fwd_rounds (plain : bool) (c c' : cont_value) :
  cont_fwd plain c c' ->
  (∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record),
    ∃ n : nat,
      answer (run_on n (apply_cont_value c vs statics hp rd))
        (apply_cont_value c' vs statics hp rd))
  ∧ (plain = false ->
     ∀ (params : list variable) (target : apply_cont)
       (metadata' : handler_metadata) (def_vars0 : env)
       (def_conts0 : cont_env) (args : list simple) (vs : list value)
       (vars : env) (conts' : cont_env) (statics : static_env) (hp : heap)
         (rd : read_record),
       c'
       = Closure (Handler params metadata' (Apply_cont target)) def_vars0
           def_conts0 [] ->
       NoDup params ->
       runs_on_values params metadata' vs ->
       length args = length params ->
       eval_simples vars args = Some vs ->
       (∀ x : variable,
         x ∉ params ->
         var_free_in_simples x (ac_args target) ->
         lookup x vars = lookup x def_vars0) ->
       resolve (ac_continuation target) conts'
       = resolve (ac_continuation target) def_conts0 ->
       ∃ n : nat,
         answer (run_on n (apply_cont_value c vs statics hp rd))
           (step
              (State (Apply_cont (forwarded params args target)) vars conts'
                 statics hp rd))).
Proof.
  induction 1
    as [ plain c
       | plain G K F params metadata metadata' handler handler' def_vars
           def_conts def_conts' recursive recursive' Hrecursive Hhandler Hmeta
           Hparams Hdef_conts IHdef_conts Hheld Htyped
       | plain c c' extra Hc [IHsame _]
       | plain remaining remaining_arity return_arity return_to return_to'
           exn_to exn_to' Hreturn [IHreturn _] Hraise _
       | c params metadata' target def_vars def_conts' Hstrict Hc
           [IHclosure IHon] Hnodup Hpasses ].
  - (* the same continuation *)
    split;
      [ intros vs statics; exists 0; rewrite run_on_zero; apply answer_refl
      | ].
    intros _ params target metadata' def_vars0 def_conts0 args vs vars conts'
      statics hp rd -> Hnodup Hruns Hlen Hargs Hvars Hcont.
    simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata'))
      eqn:Hkinds;
      [ | exists 0; intros Hstrict; destruct (Hruns Hstrict) as [Hvs _];
          rewrite Hvs, kinds_eqb_refl in Hkinds; discriminate Hkinds ].
    destruct (bind_params params vs def_vars0) as [bound | ] eqn:Hbind;
      [ | exists 0; intros Hstrict; destruct (Hruns Hstrict) as [_ Hvs];
          destruct (bind_params_length params vs def_vars0 Hvs)
            as [bound Hsome];
          rewrite Hsome in Hbind; discriminate Hbind ].
    exists 1. unfold run_on.
    rewrite run_one,
      (step_forwarded params args target vs vars def_vars0 bound conts'
         def_conts0 statics _ rd_ Hnodup Hargs Hbind Hvars Hcont).
    apply answer_refl.
  - (* related closures *)
    split.
    + intros vs statics. exists 0. rewrite run_on_zero. simpl. rewrite Hmeta.
      destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
        eqn:Hkinds; [ | apply answer_undefined ].
      apply kinds_eqb_eq in Hkinds.
      destruct (bind_params params vs def_vars) as [bound | ] eqn:Hbind;
        [ | apply answer_undefined ].
      eexists. split; [ reflexivity | ].
      apply
        (State_fwd (params_in_scope params (hm_params_kinds metadata) G) K
           (plain_forwards recursive ++ F));
        [ exact Hhandler | | | ].
      * apply (rec_conts_fwd G K F); assumption.
      * apply (held_rec G); [ exact Hrecursive | ].
        exact (held_bind K F params vs def_vars bound def_conts' Hparams Hbind
          Hheld).
      * rewrite <- Hkinds.
        exact (params_in_scope_hold params vs G def_vars bound Htyped Hbind).
    + intros _ params0 target metadata0 def_vars0 def_conts0 args vs vars
        conts' statics hp rd Hc' Hnodup Hruns Hlen Hargs Hvars Hcont.
      injection Hc' as Heq_params Heq_metadata Heq_handler Heq_vars Heq_conts
        Heq_recursive.
      subst params0 metadata0 handler' def_vars0 def_conts0 recursive'.
      unfold handlers_fwd in Hrecursive.
      destruct recursive as [ | kh rest]; [ | inversion Hrecursive ].
      simpl in Hhandler.
      destruct (fwd_apply_cont_inv _ K F handler target Hhandler)
        as (b & -> & Hgoes).
      simpl.
      destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
        eqn:Hkinds;
        [ | exists 0; intros Hstrict; destruct (Hruns Hstrict) as [Hvs _];
            rewrite <- Hmeta, Hvs, kinds_eqb_refl in Hkinds;
            discriminate Hkinds ].
      apply kinds_eqb_eq in Hkinds.
      destruct (bind_params params vs def_vars) as [bound | ] eqn:Hbind;
        [ | exists 0; intros Hstrict; destruct (Hruns Hstrict) as [_ Hvs];
            destruct (bind_params_length params vs def_vars Hvs)
              as [bound Hsome];
            rewrite Hsome in Hbind; discriminate Hbind ].
      rewrite (step_forwarded params args target vs vars def_vars bound conts'
        def_conts' statics _ rd_ Hnodup Hargs Hbind Hvars Hcont).
      assert (Hheld_bound : held K F bound def_conts')
        by exact (held_bind K F params vs def_vars bound def_conts' Hparams
             Hbind Hheld).
      assert (Htyped_bound :
        kinds_hold (params_in_scope params (hm_params_kinds metadata) G) bound).
      { rewrite <- Hkinds.
        exact (params_in_scope_hold params vs G def_vars bound Htyped Hbind). }
      destruct (goes_inv _ K F b target Hgoes)
        as [-> | (params_b & target_b & action & Hentry & Hlength & Hkinds_b
                  & Htarget)].
      * (* the handler of the first closure is the same application *)
        destruct (IHdef_conts (ac_continuation b)) as [IHapply _].
        destruct (eval_simples bound (ac_args b)) as [vt | ] eqn:Hargs_b.
        -- destruct (IHapply vt statics hp rd) as [n Hanswer].
           exists (S n). unfold run_on at 1.
           rewrite run_succ_on, !step_apply_cont, Hargs_b.
           unfold apply_continuation. exact Hanswer.
        -- exists 1. unfold run_on.
           rewrite run_one, !step_apply_cont, Hargs_b.
           apply answer_undefined.
      * (* it goes on in its turn *)
        destruct (Hheld_bound (ac_continuation b) params_b target_b Hentry)
          as (Hnodup_b & metadata_b & def_vars_b & def_conts_b & Hresolve_b
              & Hvars_b & Hcont_b & Hse_b).
        destruct (IHdef_conts (ac_continuation b)) as [_ IHon].
        specialize
          (IHon (no_entry_some F (ac_continuation b) params_b target_b Hentry)).
        pose proof
          (strict_runs _ K (ac_continuation b) (ac_args b) params_b metadata_b
             bound Htyped_bound Hse_b Hkinds_b)
          as Hruns_b.
        destruct (eval_simples bound (ac_args b)) as [vb | ] eqn:Hargs_b.
        -- assert (Hruns_vb : runs_on_values params_b metadata_b vb).
           { intros Hstrict.
             destruct (Hruns_b Hstrict) as (vb' & Hvb' & Hrest).
             injection Hvb' as <-. exact Hrest. }
           destruct (IHon params_b target_b metadata_b def_vars_b def_conts_b
                       (ac_args b) vb bound def_conts' statics hp rd Hresolve_b
                       Hnodup_b Hruns_vb Hlength Hargs_b Hvars_b Hcont_b)
             as [n Hanswer].
           exists (S n). unfold run_on at 1.
           rewrite run_succ_on, step_apply_cont, Hargs_b.
           unfold apply_continuation.
           rewrite Htarget, step_forwarded_keeping. exact Hanswer.
        -- exists 1. unfold run_on.
           rewrite run_one, step_apply_cont, Hargs_b. intros Hstrict.
           destruct (Hruns_b Hstrict) as (vb' & Hvb' & _).
           discriminate Hvb'.
  - (* the same extra arguments *)
    split; [ intros vs statics; simpl; apply IHsame | ].
    intros _ params target metadata' def_vars0 def_conts0 args vs vars conts'
      statics hp rd Hc'.
    discriminate Hc'.
  - (* what is left of the same over-application *)
    split.
    + intros vs statics hp rd. rewrite !apply_cont_value_over_application.
      destruct vs as [ | v [ | v' vs]];
        try (exists 0; rewrite run_on_zero; apply answer_undefined).
      unfold call_function.
      apply run_call_fwd; [ exact Hreturn | exact Hraise | ].
      intros partial. apply IHreturn.
    + intros _ params target metadata' def_vars0 def_conts0 args vs vars conts'
        statics hp rd Hc'.
      discriminate Hc'.
  - (* what a closure passes its parameters on to *)
    split; [ | intros Hfalse; discriminate Hfalse ].
    intros vs statics hp rd.
    destruct (bind_params params vs def_vars) as [bound | ] eqn:Hbind.
    + (* the first comes to the application of the handler, which is one
         of the second to the values given *)
      pose proof (eval_params_bound params vs def_vars bound Hnodup Hbind)
        as Hvalues.
      assert (Hfrozen :
        ∀ x : variable,
          x ∉ params ->
          var_free_in_simples x (ac_args target) ->
          lookup x bound = lookup x def_vars).
      { intros x Hnotin Hfree. rewrite Hpasses in Hfree.
        destruct (params_only params x Hnotin Hfree). }
      destruct (IHon eq_refl params target metadata' def_vars def_conts'
                  (map Var params) vs bound def_conts' statics hp rd eq_refl
                  Hnodup
                  ltac:(intros Hs; rewrite Hstrict in Hs; discriminate Hs)
                  (length_map Var params) Hvalues Hfrozen eq_refl)
        as [n Hanswer].
      exists n.
      rewrite
        (step_forwarded params (map Var params) target vs bound def_vars bound
           def_conts' def_conts' statics _ rd_ Hnodup Hvalues Hbind Hfrozen
             eq_refl),
        step_apply_cont, Hpasses, Hvalues in Hanswer.
      exact Hanswer.
    + (* the closure does not run on these values, so the first run is
         undefined *)
      destruct (IHclosure vs statics hp rd) as [n Hanswer]. exists n.
      assert (Hundefined :
        apply_cont_value
          (Closure (Handler params metadata' (Apply_cont target)) def_vars
             def_conts' [])
          vs statics hp rd
        = Undefined).
      { simpl.
        destruct (kinds_eqb (map kind_of_value vs)
                    (hm_params_kinds metadata'));
          [ rewrite Hbind | ]; reflexivity. }
      rewrite Hundefined in Hanswer.
      rewrite (answer_undefined_inv _ Hanswer). intros Hs.
      rewrite Hstrict in Hs. discriminate Hs.
Qed.

(* ======================================================================== *)
(* The simulation                                                           *)
(* ======================================================================== *)

Lemma answered_step (result : step_result) (s' : state) :
  answer result (step s') -> round_answered state_fwd result s'.
Proof.
  destruct result as [t | k vs | ]; unfold answer, round_answered;
    [ | | exact (fun _ => I) ].
  - intros (t' & Hstep & Hrel). exists 0, t'.
    split; [ rewrite run_one; exact Hstep | exact Hrel ].
  - intros Hstep. exists 1. rewrite run_one. exact Hstep.
Qed.

(* Some steps of the first state are answered by one of the second. *)
Lemma fwd_round_answer (st st' : state) :
  state_fwd st st' -> ∃ n : nat, answer (run (S n) st) (step st').
Proof.
  intros [G K F e e' vars conts conts' statics hp rd Hfwd Hconts Hheld Htyped].
  destruct Hfwd
    as [ G K F x kind d body body' Hx Hbody | G K F bindings body body' Hbody
       | G K F k params metadata metadata' handler handler' entry body body'
           Hhandler Hmeta Hparams Hk Hfits Hbody
       | G K F handlers handlers' body body' Hhandlers Hbinds Hbody
       | G K F ap ap' Hreturns | G K F ac ac' Hgoes
       | G K F scrutinee arms arms' Harms | G K F ].
  - (* Let *)
    exists 0. rewrite run_one, !step_let_exec.
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ];
      [ | apply answer_undefined ].
    destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind;
      [ | apply answer_undefined ].
    apply kind_eqb_eq in Hkind. subst kind.
    eexists. split; [ reflexivity | ].
    apply (State_fwd ((x, kind_of_value v) :: G) K F);
      [ exact Hbody | exact Hconts | | apply kinds_hold_cons, Htyped ].
    apply held_var; assumption.
  - (* Let_static *)
    exists 0. rewrite run_one, !step_let_static.
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      [ | apply answer_undefined ].
    eexists. split; [ reflexivity | ].
    apply (State_fwd G K F); assumption.
  - (* Let_cont_non_recursive *)
    exists 0. rewrite run_one, !step_let_cont_non_recursive.
    eexists. split; [ reflexivity | ].
    apply (State_fwd G ((k, hm_params_kinds metadata') :: K) ((k, entry) :: F));
      [ exact Hbody | | | exact Htyped ].
    + intros k'. destruct (Pos.eq_dec k' k) as [-> | Hne].
      * rewrite !resolve_head.
        apply (Closure_fwd _ G K F params metadata metadata' handler handler'
          vars conts conts' [] []);
          [ apply Forall2_nil | exact Hhandler | exact Hmeta | exact Hparams
          | exact Hconts | exact Hheld | exact Htyped ].
      * rewrite !resolve_other by exact Hne. unfold no_entry.
        rewrite forward_of_other by exact Hne. exact (Hconts k').
    + intros k' params0 target Hsome.
      destruct (Pos.eq_dec k' k) as [-> | Hne].
      * rewrite forward_of_head in Hsome. subst entry.
        destruct Hfits as (-> & -> & Hnodup & Htarget & Hse).
        split; [ exact Hnodup | ]. exists metadata', vars, conts'.
        rewrite resolve_head. split; [ reflexivity | ].
        split; [ reflexivity | ].
        split;
          [ apply resolve_other; exact Htarget
          | split; [ exact Hse | intros _; apply lookup_head ] ].
      * rewrite forward_of_other in Hsome by exact Hne.
        destruct (Hheld k' params0 target Hsome)
          as (Hnodup & metadata0 & def_vars & def_conts' & Hresolve & Hvars
              & Hcont & Hse & Hkinds).
        split; [ exact Hnodup | ]. exists metadata0, def_vars, def_conts'.
        rewrite (resolve_other k' k) by exact Hne.
        split; [ exact Hresolve | ]. split; [ exact Hvars | ].
        split;
          [ | split;
              [ exact Hse
              | intros Hstrict; rewrite lookup_other by exact Hne;
                exact (Hkinds Hstrict) ] ].
        rewrite resolve_other; [ exact Hcont | ].
        intros Heq. apply (Hk k'). exists params0, target. auto.
  - (* Let_cont_recursive *)
    exists 0. rewrite run_one, !step_let_cont_recursive, Hbinds.
    destruct (handlers_bind_statics handlers); [ apply answer_undefined | ].
    eexists. split; [ reflexivity | ].
    apply (State_fwd G K (plain_forwards handlers ++ F));
      [ exact Hbody | | | exact Htyped ].
    + apply (rec_conts_fwd G K F); assumption.
    + apply (held_rec G); assumption.
  - (* Apply: the same call, returning to related continuations *)
    pose proof (conts_any F conts conts' Hconts) as Hany.
    assert (Hreturn :
      cont_fwd true (resolve (ap_continuation ap) conts)
        (resolve (ap_continuation ap') conts')).
    { destruct Hreturns as [ap | ap params target Hstrict Hentry Hpasses];
        [ apply Hany | ].
      cbn [ap_continuation].
      destruct (Hheld (ap_continuation ap) params target Hentry)
        as (Hnodup & metadata' & def_vars & def_conts' & Hresolve & _ & Hcont
            & _).
      rewrite Hcont.
      apply (Alias_fwd _ params metadata' target def_vars def_conts' Hstrict);
        [ | exact Hnodup | exact Hpasses ].
      rewrite <- Hresolve.
      pose proof (Hconts (ap_continuation ap)) as Hplain.
      rewrite (no_entry_some F _ params target Hentry) in Hplain.
      exact Hplain. }
    assert (Hsame :
      ap_callee ap' = ap_callee ap ∧ ap_call_kind ap' = ap_call_kind ap
      ∧ ap_args ap' = ap_args ap
      ∧ ap_exn_continuation ap' = ap_exn_continuation ap
      ∧ exn_extra_simples ap' = exn_extra_simples ap
      ∧ ap_args_arity ap' = ap_args_arity ap
      ∧ ap_return_arity ap' = ap_return_arity ap).
    { destruct Hreturns; repeat split; reflexivity. }
    destruct Hsame
      as (Hcallee & Hkind & Hargs & Hexn & Hextra & Hargs_arity
          & Hreturn_arity).
    assert (Hround :
      ∃ n : nat,
        answer (run_on n (step (State (Apply ap) vars conts statics hp rd)))
          (step (State (Apply ap') vars conts' statics hp rd))).
    { rewrite !step_apply, Hcallee, Hkind, Hargs, Hexn, Hextra, Hargs_arity,
        Hreturn_arity.
      destruct (eval_simple vars (ap_callee ap)) as [callee | ];
        [ | exists 0; rewrite run_on_zero; apply answer_undefined ].
      destruct (eval_simples vars (ap_args ap)) as [vs | ];
        [ | exists 0; rewrite run_on_zero; apply answer_undefined ].
      destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
        [ | exists 0; rewrite run_on_zero; apply answer_undefined ].
      unfold call_function.
      apply run_call_fwd.
      - exact Hreturn.
      - destruct extra as [ | v extra]; simpl;
          [ apply Hany | apply Extra_args_fwd, Hany ].
      - intros partial.
        exact (proj1 (cont_fwd_rounds _ _ _ Hreturn) [partial] statics hp rd_).
          }
    destruct Hround as [n Hround]. exists n. rewrite run_succ_on.
    exact Hround.
  - (* Apply_cont *)
    destruct (cont_fwd_rounds _ _ _ (Hconts (ac_continuation ac)))
      as [Hsame Hon].
    destruct (goes_inv G K F ac ac' Hgoes)
      as [-> | (params & target & action & Hentry & Hlength & Hkinds & ->)].
    + destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs.
      * destruct (Hsame vs statics hp rd) as [n Hanswer]. exists n.
        rewrite run_succ_on, !step_apply_cont, Hargs.
        unfold apply_continuation. exact Hanswer.
      * exists 0. rewrite run_one, !step_apply_cont, Hargs.
        apply answer_undefined.
    + specialize
        (Hon (no_entry_some F (ac_continuation ac) params target Hentry)).
      destruct (Hheld (ac_continuation ac) params target Hentry)
        as (Hnodup & metadata' & def_vars & def_conts' & Hresolve & Hvars
            & Hcont & Hse).
      pose proof
        (strict_runs G K (ac_continuation ac) (ac_args ac) params metadata'
           vars Htyped Hse Hkinds)
        as Hruns.
      destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs.
      * assert (Hruns_vs : runs_on_values params metadata' vs).
        { intros Hstrict. destruct (Hruns Hstrict) as (vs' & Hvs' & Hrest).
          injection Hvs' as <-. exact Hrest. }
        destruct (Hon params target metadata' def_vars def_conts' (ac_args ac)
                    vs vars conts' statics hp rd Hresolve Hnodup Hruns_vs
                      Hlength
                    Hargs Hvars Hcont)
          as [n Hanswer].
        exists n.
        rewrite step_forwarded_keeping, run_succ_on, step_apply_cont, Hargs.
        exact Hanswer.
      * exists 0. rewrite run_one, step_apply_cont, Hargs. intros Hstrict.
        destruct (Hruns Hstrict) as (vs' & Hvs' & _). discriminate Hvs'.
  - (* Switch *)
    exists 0. rewrite run_one, !step_switch.
    destruct (eval_simple vars scrutinee) as [v | ];
      [ | apply answer_undefined ].
    destruct (naked_immediate_of_value v) as [i | ];
      [ | apply answer_undefined ].
    pose proof (find_arm_goes G K F i arms arms' Harms) as Harm.
    destruct (find_arm i arms) as [ac | ], (find_arm i arms') as [ac' | ];
      try contradiction; [ | apply answer_undefined ].
    eexists. split; [ reflexivity | ].
    apply (State_fwd G K F);
      [ apply Fwd_apply_cont, Harm | exact Hconts | exact Hheld
      | exact Htyped ].
  - (* Invalid *)
    exists 0. rewrite run_one, !step_invalid. apply answer_undefined.
Qed.

Lemma fwd_round (st st' : state) :
  state_fwd st st' ->
  ∃ n : nat, round_answered state_fwd (run (S n) st) st'.
Proof.
  intros Hrel. destruct (fwd_round_answer st st' Hrel) as [n Hanswer].
  exists n. apply answered_step, Hanswer.
Qed.

End Strict.

(* Where the entries are strict, one step of the second state is
   answered by some steps of the first. *)
Lemma fwd_round_back (st' st : state) :
  state_fwd true st st' ->
  ∃ n : nat,
    round_answered (fun t' t => state_fwd true t t') (run (S n) st') st.
Proof.
  intros Hrel. destruct (fwd_round_answer true st st' Hrel) as [n Hanswer].
  exists 0. rewrite run_one. unfold answer in Hanswer.
  unfold round_answered.
  destruct (run (S n) st) as [t | k vs | ] eqn:Hrun.
  - destruct Hanswer as (t' & -> & Hrel'). exists n, t.
    split; [ exact Hrun | exact Hrel' ].
  - rewrite Hanswer. exists (S n). exact Hrun.
  - rewrite (Hanswer eq_refl). exact I.
Qed.

(* ======================================================================== *)
(* The pass relates a term to its result                                    *)
(* ======================================================================== *)

Fixpoint forward_applications_binds_statics (e : expr) :
  ∀ (calls : bool) (F : forwards),
    binds_statics (forward_applications_under calls F e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros calls F; simpl; try reflexivity.
  - apply forward_applications_binds_statics.
  - rewrite forward_applications_binds_statics. reflexivity.
  - rewrite !forward_applications_binds_statics. reflexivity.
  - rewrite forward_applications_binds_statics. f_equal.
    generalize (map (fun '(k, _) => (k, @None (list variable * apply_cont)))
                  handlers
                ++ F).
    intros F'.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite forward_applications_binds_statics, IH. reflexivity.
Qed.

Lemma goes_forward_apply_cont_keeping (G : kinds_in_scope)
    (K : kinds_of_conts) (F : forwards) (ac : apply_cont) :
  goes false G K F ac (forward_apply_cont_keeping F ac).
Proof.
  unfold forward_apply_cont_keeping.
  destruct (forward_of F (ac_continuation ac)) as [[params target] | ] eqn:Hk;
    [ | apply Goes_same ].
  destruct
    (Nat.eqb (length (ac_args ac)) (length params) && no_trap_action target)
    eqn:Hcheck; [ | apply Goes_same ].
  apply andb_true_iff in Hcheck as [Hlength _].
  apply Nat.eqb_eq in Hlength.
  apply Goes_keeping; [ assumption | assumption | intros Hs; discriminate Hs ].
Qed.

Lemma goes_forward_arm (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards)
    (ac : apply_cont) :
  goes false G K F ac (forward_arm F ac).
Proof.
  unfold forward_arm.
  destruct (forward_of F (ac_continuation ac)) as [[params target] | ] eqn:Hk;
    [ | apply goes_forward_apply_cont_keeping ].
  destruct params as [ | p params];
    [ | apply goes_forward_apply_cont_keeping ].
  destruct (ac_args ac) as [ | arg args] eqn:Hargs;
    [ | apply goes_forward_apply_cont_keeping ].
  destruct (no_trap_action ac); [ | apply goes_forward_apply_cont_keeping ].
  pose proof (Goes_on false G K F ac [] target Hk) as Hgoes.
  rewrite Hargs in Hgoes.
  apply Hgoes; [ reflexivity | intros Hs; discriminate Hs ].
Qed.

Lemma passes_on_params (params : list variable) :
  ∀ args : list simple,
    passes_on params args = true -> args = map Var params.
Proof.
  induction params as [ | p params IH]; intros [ | [x | sym | c] args] Hpasses;
    try discriminate Hpasses; [ reflexivity | ].
  simpl in Hpasses. apply andb_true_iff in Hpasses as [Hx Hrest].
  apply Pos.eqb_eq in Hx. subst x. simpl. f_equal. exact (IH args Hrest).
Qed.

Lemma returns_forward_apply (F : forwards) (ap : apply_expr) :
  returns false F ap (forward_apply F ap).
Proof.
  unfold forward_apply, alias_of.
  destruct (forward_of F (ap_continuation ap)) as [[params target] | ] eqn:Hk;
    [ | apply Returns_same ].
  destruct (no_trap_action target && passes_on params (ac_args target))
    eqn:Hcheck; [ | apply Returns_same ].
  apply andb_true_iff in Hcheck as [_ Hpasses].
  exact
    (Returns_on false F ap params target eq_refl Hk
       (passes_on_params params (ac_args target) Hpasses)).
Qed.

(* What the relation asks of an expression under the entries [F], as the
   pass can check it: the expression binds no name that an entry
   depends on. *)
Definition safe (F : forwards) (e : expr) : Prop :=
  ∀ k : continuation,
    (∀ x : variable, frozen_var (forward_of F k) x -> binds_var x e = false)
    ∧ (∀ k0 : continuation,
        frozen_cont (forward_of F k) k0 -> binds_cont k0 e = false).

Lemma safe_none (F : forwards) (e : expr) :
  (∀ k : continuation, forward_of F k = None) -> safe F e.
Proof.
  intros Hnone k. rewrite Hnone.
  split; [ intros x (params & target & Hsome & _)
         | intros k0 (params & target & Hsome & _) ];
    discriminate Hsome.
Qed.

Lemma safe_cons (F : forwards) (k0 : continuation) (e0 : entry) (e : expr) :
  safe F e ->
  (∀ x : variable, frozen_var e0 x -> binds_var x e = false) ->
  (∀ k1 : continuation, frozen_cont e0 k1 -> binds_cont k1 e = false) ->
  safe ((k0, e0) :: F) e.
Proof.
  intros Hsafe Hvars Hconts k.
  destruct (Pos.eq_dec k k0) as [-> | Hne];
    [ rewrite forward_of_head; auto
    | rewrite forward_of_other by exact Hne; apply Hsafe ].
Qed.

Lemma safe_plain (F : forwards)
    (handlers : list (continuation * continuation_handler)) (e : expr) :
  safe F e -> safe (plain_forwards handlers ++ F) e.
Proof.
  intros Hsafe k. rewrite forward_of_plain.
  destruct (lookup k handlers); [ | apply Hsafe ].
  split; [ intros x (params & target & Hsome & _)
         | intros k0 (params & target & Hsome & _) ];
    discriminate Hsome.
Qed.

(* What [forwarding] answers is an entry for the handler, which the body
   leaves alone. *)
Lemma forwarding_fits (strict : bool) (k : continuation)
    (params : list variable) (metadata metadata' : handler_metadata)
    (handler' body : expr) :
  strict_entry strict params metadata' ->
  entry_fits strict k params metadata' handler'
    (forwarding k params metadata handler' body)
  ∧ (∀ x : variable,
      frozen_var (forwarding k params metadata handler' body) x ->
      binds_var x body = false)
  ∧ (∀ k0 : continuation,
      frozen_cont (forwarding k params metadata handler' body) k0 ->
      binds_cont k0 body = false).
Proof.
  intros Hse. unfold forwarding.
  assert (Hnone :
    entry_fits strict k params metadata' handler' None
    ∧ (∀ x : variable, frozen_var None x -> binds_var x body = false)
    ∧ (∀ k0 : continuation, frozen_cont None k0 -> binds_cont k0 body = false)).
  { split; [ exact I | ].
    split; [ intros x (params0 & target & Hsome & _)
           | intros k0 (params0 & target & Hsome & _) ];
      discriminate Hsome. }
  destruct handler' as [ | | | | | target | | ]; try exact Hnone.
  match goal with
  | |- context [if ?check then None else _] =>
    destruct check eqn:Hcheck; [ exact Hnone | ]
  end.
  apply orb_false_iff in Hcheck as [Hcheck Hvars].
  apply orb_false_iff in Hcheck as [Hcheck Hcont].
  apply orb_false_iff in Hcheck as [Hcheck Hdistinct].
  apply orb_false_iff in Hcheck as [_ Hself].
  apply negb_false_iff in Hdistinct.
  unfold among, apply_cont_continuations in Hself. simpl in Hself.
  apply orb_false_iff in Hself as [Hself _]. apply Pos.eqb_neq in Hself.
  unfold apply_cont_continuations in Hcont. simpl in Hcont.
  apply orb_false_iff in Hcont as [Hcont _].
  split; [ | split ].
  - split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ apply distinct_NoDup, Hdistinct | ].
    split; [ exact (not_eq_sym Hself) | exact Hse ].
  - intros x (params0 & target0 & Hsome & Hnotin & Hfree).
    injection Hsome as <- <-.
    unfold var_free_in_simples in Hfree.
    apply Exists_exists in Hfree as (s & Hin & Hs).
    pose proof (existsb_false_in _ _ s Hvars Hin) as Hbound.
    destruct s as [y | sym | c]; simpl in Hs; try contradiction. subst y.
    simpl in Hbound. apply andb_false_iff in Hbound as [Hamong | Hbound];
      [ | exact Hbound ].
    apply negb_false_iff, among_In in Hamong. contradiction.
  - intros k0 (params0 & target0 & Hsome & <-). injection Hsome as <- <-.
    exact Hcont.
Qed.

Lemma safe_let (F : forwards) (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  safe F (Let x kind d body) -> safe F body ∧ var_unfrozen F x.
Proof.
  intros Hsafe. split.
  - intros k. destruct (Hsafe k) as [Hvars Hconts].
    split; [ | exact Hconts ].
    intros y Hy. specialize (Hvars y Hy). simpl in Hvars.
    apply orb_false_iff in Hvars. apply Hvars.
  - intros k Hx. destruct (Hsafe k) as [Hvars _].
    specialize (Hvars x Hx). simpl in Hvars.
    rewrite Pos.eqb_refl in Hvars. discriminate Hvars.
Qed.

Lemma safe_let_cont (F : forwards) (k0 : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) :
  safe F (Let_cont_non_recursive k0 (Handler params metadata handler) body) ->
  safe F handler ∧ safe F body
  ∧ (∀ p : variable, p ∈ params -> var_unfrozen F p)
  ∧ cont_unfrozen F k0.
Proof.
  intros Hsafe. split; [ | split; [ | split ] ].
  - intros k. destruct (Hsafe k) as [Hvars Hconts]. split.
    + intros x Hx. specialize (Hvars x Hx). simpl in Hvars.
      apply orb_false_iff in Hvars as [Hhandler _].
      apply orb_false_iff in Hhandler. apply Hhandler.
    + intros k1 Hk1. specialize (Hconts k1 Hk1). simpl in Hconts.
      apply orb_false_iff in Hconts as [Hhandler _].
      apply orb_false_iff in Hhandler. apply Hhandler.
  - intros k. destruct (Hsafe k) as [Hvars Hconts]. split.
    + intros x Hx. specialize (Hvars x Hx). simpl in Hvars.
      apply orb_false_iff in Hvars. apply Hvars.
    + intros k1 Hk1. specialize (Hconts k1 Hk1). simpl in Hconts.
      apply orb_false_iff in Hconts. apply Hconts.
  - intros p Hin k Hp. destruct (Hsafe k) as [Hvars _].
    specialize (Hvars p Hp). simpl in Hvars.
    apply orb_false_iff in Hvars as [Hhandler _].
    apply orb_false_iff in Hhandler as [Hamong _].
    apply among_In in Hin. congruence.
  - intros k Hk0. destruct (Hsafe k) as [_ Hconts].
    specialize (Hconts k0 Hk0). simpl in Hconts.
    rewrite Pos.eqb_refl in Hconts. discriminate Hconts.
Qed.

Lemma safe_let_cont_recursive (F : forwards)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  safe F (Let_cont_recursive handlers body) ->
  safe F body
  ∧ ∀ (k0 : continuation) (params : list variable)
      (metadata : handler_metadata) (handler : expr),
      (k0, Handler params metadata handler) ∈ handlers ->
      safe F handler
      ∧ (∀ p : variable, p ∈ params -> var_unfrozen F p)
      ∧ cont_unfrozen F k0.
Proof.
  intros Hsafe. split.
  - intros k. destruct (Hsafe k) as [Hvars Hconts]. split.
    + intros x Hx. specialize (Hvars x Hx). simpl in Hvars.
      apply orb_false_iff in Hvars. apply Hvars.
    + intros k1 Hk1. specialize (Hconts k1 Hk1). simpl in Hconts.
      apply orb_false_iff in Hconts. apply Hconts.
  - intros k0 params metadata handler Hin.
    assert (Hvars_handler :
      ∀ (k : continuation) (x : variable),
        frozen_var (forward_of F k) x ->
        among x params = false ∧ binds_var x handler = false).
    { intros k x Hx. destruct (Hsafe k) as [Hvars _].
      specialize (Hvars x Hx). simpl in Hvars.
      apply orb_false_iff in Hvars as [Hhandlers _].
      pose proof (existsb_false_in _ _ _ Hhandlers Hin) as Hhandler.
      simpl in Hhandler. apply orb_false_iff in Hhandler. exact Hhandler. }
    assert (Hconts_handler :
      ∀ k k1 : continuation,
        frozen_cont (forward_of F k) k1 ->
        Pos.eqb k1 k0 = false ∧ binds_cont k1 handler = false).
    { intros k k1 Hk1. destruct (Hsafe k) as [_ Hconts].
      specialize (Hconts k1 Hk1). simpl in Hconts.
      apply orb_false_iff in Hconts as [Hhandlers _].
      pose proof (existsb_false_in _ _ _ Hhandlers Hin) as Hhandler.
      simpl in Hhandler. apply orb_false_iff in Hhandler. exact Hhandler. }
    split; [ | split ].
    + intros k. split.
      * intros x Hx. exact (proj2 (Hvars_handler k x Hx)).
      * intros k1 Hk1. exact (proj2 (Hconts_handler k k1 Hk1)).
    + intros p Hp k Hfrozen.
      destruct (Hvars_handler k p Hfrozen) as [Hamong _].
      apply among_In in Hp. congruence.
    + intros k Hfrozen.
      destruct (Hconts_handler k k0 Hfrozen) as [Hself _].
      rewrite Pos.eqb_refl in Hself. discriminate Hself.
Qed.

Fixpoint forward_applications_fwd (e : expr) :
  ∀ (calls : bool) (G : kinds_in_scope) (K : kinds_of_conts) (F : forwards),
    safe F e -> fwd false G K F e (forward_applications_under calls F e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros calls G K F Hsafe; simpl.
  - destruct (safe_let F x kind d body Hsafe) as [Hbody Hx].
    apply Fwd_let; [ exact Hx | apply forward_applications_fwd, Hbody ].
  - apply Fwd_let_static, forward_applications_fwd. exact Hsafe.
  - destruct (safe_let_cont F k params metadata handler body Hsafe)
      as (Hhandler & Hbody & Hparams & Hk).
    destruct
      (forwarding_fits false k params metadata metadata
         (forward_applications_under calls F handler) body)
      as (Hfits & Hvars & Hconts);
      [ intros Hstrict; discriminate Hstrict | ].
    apply (Fwd_let_cont false G K F k params metadata metadata handler
      (forward_applications_under calls F handler)
      (forwarding k params metadata
         (forward_applications_under calls F handler) body));
      [ apply forward_applications_fwd, Hhandler | reflexivity
      | exact Hparams | exact Hk | exact Hfits | ].
    apply forward_applications_fwd, safe_cons; assumption.
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
      split; [ apply forward_applications_fwd, Hhandler | ].
      split; assumption.
    + clear. unfold handlers_bind_statics.
      generalize (plain_forwards handlers ++ F). intros F'.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        simpl; [ reflexivity | ].
      rewrite forward_applications_binds_statics, IH. reflexivity.
    + apply forward_applications_fwd, safe_plain, Hbody.
  - destruct calls;
      [ apply Fwd_apply, returns_forward_apply
      | apply Fwd_apply, Returns_same ].
  - apply Fwd_apply_cont, goes_forward_apply_cont_keeping.
  - apply Fwd_switch. clear Hsafe.
    induction arms as [ | arm arms IH]; simpl; [ apply Forall2_nil | ].
    apply Forall2_cons; [ | exact IH ].
    split; [ reflexivity | apply goes_forward_arm ].
  - apply Fwd_invalid.
Qed.

Theorem forward_applications_with_refines :
  forward_applications_with_refines_statement.
Proof.
  intros calls e vars conts statics hp rd.
  apply (rounds_refines (state_fwd false) (fwd_round false)).
  apply (State_fwd false [] [] []);
    [ apply forward_applications_fwd, safe_none; reflexivity
    | intros k; apply Same_fwd | apply held_nil | apply kinds_hold_nil ].
Qed.

Theorem forward_applications_refines : forward_applications_refines_statement.
Proof. intros e. exact (forward_applications_with_refines false e). Qed.
