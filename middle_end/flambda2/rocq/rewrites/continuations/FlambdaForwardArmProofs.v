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

(* Proofs of the statements of FlambdaForwardArmSpec.v. Nothing here
   needs review.

   The pass changes the applications of a continuation on the strength
   of its handler, and where the first term applies a continuation
   whose handler tests the number given, the second applies at once
   what the arm of that number applies. The term and its result are
   run side by side by rounds (FlambdaRoundsProofs.rounds_refines), as
   in FlambdaForwardProofs.v, whose relation this is with entries of
   another kind: a round is one step of each run, but for such an
   application, where the first run takes the step of the
   application, that of the [Switch] of the handler and those through
   the handlers that the arm goes through in its turn, and the second
   one step. The relation between the two states ([arm_state_fwd]) is:

   - the second expression is the first with some applications replaced
     by the [Apply_cont] of an arm of the handler of their
     continuation, the arguments put for the parameters ([arm_fwd],
     [arm_goes]), as the entries of some
     FlambdaForwardArmSpec.arm_forwards say; no binder in scope binds a
     name that an entry depends on. The names that an entry depends on
     are those of its arms that pass the tests of
     FlambdaForwardSpec.forwarding ([passes], [arm_frozen_var],
     [arm_frozen_cont]): the pass makes those tests of an arm when an
     application comes to it;
   - the variables of the two states are the same, and have the kinds
     of their binders (FlambdaNormaliseProofs.kinds_hold);
   - each continuation name stands, in the two states, for the same
     thing or for closures whose handlers are so related, under the
     same variables and continuations in turn so related
     ([arm_cont_fwd]);
   - for each entry, the name stands in the second state for a closure
     whose handler is the [Switch] of the entry on the parameter at the
     position of the entry; and for each arm that passes, the
     variables of the state agree with those of that closure on the
     variables that the arm reads, and the continuation that the arm
     applies is the same in the state as in the closure ([arm_held]).

   The round at an application that goes to an arm
   ([arm_cont_fwd_rounds]) is by induction on how the closures are
   related: the handler of the first closure is a [Switch] on the
   same parameter, which is bound to the number given
   ([tested_param_bound]); its arm of that number is an application
   that the arm of the second is related to, itself the same or one
   that goes to an arm in its turn.

   The relation is taken with a flag ([strict]), as in
   FlambdaForwardProofs.v. Without it, the second run answers the
   first where the first is defined, which is the refinement. With
   it, an application that goes to an arm has arguments that are
   known, by the binders around it, to have the kinds that the handler
   of its continuation gives its parameters, as many as the handler
   has parameters; the first run then takes its steps there too, the
   second run is undefined where the first is, a round is read the
   other way too ([arm_fwd_round_back]), and each term refines the
   other, under any variables that have the kinds the pass starts
   from ([forward_to_arms_kinded_under_equivalent_bound]).

   In the body of code the kinded pass starts from the kinds of what a
   call binds (FlambdaCodeLetsProofs.code_in_scope_hold), and a group
   of static bindings may have the bodies of its code replaced, one
   piece of code at a time
   (FlambdaContextProofs.let_static_code_refines_where), as in
   FlambdaForwardKindedProofs.v. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaContextSpec
  FlambdaContextProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaRoundsProofs
  FlambdaSinkContsProofs FlambdaUnusedParamsSpec FlambdaUnusedParamsProofs
  FlambdaParamAliasesSpec FlambdaParamAliasesProofs FlambdaLoopLetsSpec
  FlambdaLoopLetsProofs FlambdaCodeLetsSpec FlambdaCodeLetsProofs
  FlambdaForwardSpec FlambdaForwardProofs FlambdaForwardKindedSpec
  FlambdaForwardKindedProofs FlambdaForwardArmSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known of the continuations in scope                              *)
(* ======================================================================== *)

Definition arm_forward_of (F : arm_forwards) (k : continuation)
    : option arm_entry :=
  match lookup k F with
  | Some entry => entry
  | None => None
  end.

Lemma arm_forward_of_head (F : arm_forwards) (k : continuation)
    (e : option arm_entry) :
  arm_forward_of ((k, e) :: F) k = e.
Proof. unfold arm_forward_of. simpl. rewrite Pos.eqb_refl. reflexivity. Qed.

Lemma arm_forward_of_other (F : arm_forwards) (k k' : continuation)
    (e : option arm_entry) :
  k' <> k -> arm_forward_of ((k, e) :: F) k' = arm_forward_of F k'.
Proof.
  intros Hne. unfold arm_forward_of. simpl.
  destruct (Pos.eqb_spec k' k); [ contradiction | reflexivity ].
Qed.

(* The entries that the handlers of a recursive [Let_cont] add. *)
Definition arm_plain (handlers : list (continuation * continuation_handler))
    : arm_forwards :=
  map (fun '(k, _) => (k, None)) handlers.

Lemma arm_forward_of_plain
    (handlers : list (continuation * continuation_handler)) (F : arm_forwards)
    (k : continuation) :
  arm_forward_of (arm_plain handlers ++ F) k
  = match lookup k handlers with
    | Some _ => None
    | None => arm_forward_of F k
    end.
Proof.
  unfold arm_forward_of.
  induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

(* An arm that the applications of [k] may go to: it passes the tests of
   FlambdaForwardSpec.forwarding, for a handler that would be that arm
   alone. *)
Definition passes (k : continuation) (params : list variable)
    (metadata : handler_metadata) (body : expr) (target : apply_cont) : Prop :=
  forwarding k params metadata (Apply_cont target) body
  = Some (params, target).

(* What [forwarding] answers for a handler that is one [Apply_cont] is
   that [Apply_cont] with the parameters. *)
Lemma forwarding_some (k : continuation) (params params' : list variable)
    (metadata : handler_metadata) (body : expr) (target target' : apply_cont) :
  forwarding k params metadata (Apply_cont target) body
  = Some (params', target') ->
  passes k params metadata body target ∧ params' = params ∧ target' = target.
Proof.
  unfold passes, forwarding.
  match goal with
  | |- context [if ?check then None else _] => destruct check
  end; [ discriminate | ].
  intros [= <- <-]. auto.
Qed.

Lemma passes_inv (k : continuation) (params : list variable)
    (metadata : handler_metadata) (body : expr) (target : apply_cont) :
  passes k params metadata body target ->
  NoDup params
  ∧ ac_continuation target <> k
  ∧ (∀ x : variable,
      x ∉ params ->
      var_free_in_simples x (ac_args target) ->
      binds_var x body = false)
  ∧ binds_cont (ac_continuation target) body = false.
Proof.
  intros Hpasses.
  destruct
    (forwarding_fits false k params metadata metadata (Apply_cont target) body)
    as (Hfits & Hvars & Hconts);
    [ intros Hstrict; discriminate Hstrict | ].
  unfold passes in Hpasses. rewrite Hpasses in Hfits, Hvars, Hconts.
  destruct Hfits as (_ & _ & Hnodup & Hne & _).
  split; [ exact Hnodup | ]. split; [ exact Hne | ]. split.
  - intros x Hnotin Hfree. apply Hvars. exists params, target. auto.
  - apply Hconts. exists params, target. auto.
Qed.

(* The names that an entry of [k] depends on: the variables that an arm
   which passes reads and that are not parameters, and the continuation
   that such an arm applies. *)
Definition arm_frozen_var (k : continuation) (e : option arm_entry)
    (x : variable) : Prop :=
  ∃ (params : list variable) (metadata : handler_metadata) (i : nat)
    (arms : list (Z * apply_cont)) (body : expr) (n : Z)
    (target : apply_cont),
    e = Some (params, metadata, i, arms, body)
    ∧ find_arm n arms = Some target
    ∧ passes k params metadata body target
    ∧ x ∉ params
    ∧ var_free_in_simples x (ac_args target).

Definition arm_frozen_cont (k : continuation) (e : option arm_entry)
    (k0 : continuation) : Prop :=
  ∃ (params : list variable) (metadata : handler_metadata) (i : nat)
    (arms : list (Z * apply_cont)) (body : expr) (n : Z)
    (target : apply_cont),
    e = Some (params, metadata, i, arms, body)
    ∧ find_arm n arms = Some target
    ∧ passes k params metadata body target
    ∧ ac_continuation target = k0.

Lemma arm_frozen_var_none (k : continuation) (x : variable) :
  ~ arm_frozen_var k None x.
Proof.
  intros (params & metadata & i & arms & body & n & target & Hsome & _).
  discriminate Hsome.
Qed.

Lemma arm_frozen_cont_none (k k0 : continuation) :
  ~ arm_frozen_cont k None k0.
Proof.
  intros (params & metadata & i & arms & body & n & target & Hsome & _).
  discriminate Hsome.
Qed.

Definition arm_var_unfrozen (F : arm_forwards) (x : variable) : Prop :=
  ∀ k : continuation, ~ arm_frozen_var k (arm_forward_of F k) x.

Definition arm_cont_unfrozen (F : arm_forwards) (k : continuation) : Prop :=
  ∀ k' : continuation, ~ arm_frozen_cont k' (arm_forward_of F k') k.

Lemma arm_var_unfrozen_none (F : arm_forwards) (x : variable) :
  (∀ k : continuation, arm_forward_of F k = None) -> arm_var_unfrozen F x.
Proof.
  intros Hnone k. rewrite Hnone. apply arm_frozen_var_none.
Qed.

Lemma arm_cont_unfrozen_none (F : arm_forwards) (k : continuation) :
  (∀ k' : continuation, arm_forward_of F k' = None) -> arm_cont_unfrozen F k.
Proof.
  intros Hnone k'. rewrite Hnone. apply arm_frozen_cont_none.
Qed.

(* ======================================================================== *)
(* The parameter tested                                                     *)
(* ======================================================================== *)

Lemma position_of_nth (x : variable) (params : list variable) :
  ∀ i : nat, position_of x params = Some i -> nth_error params i = Some x.
Proof.
  induction params as [ | y params IH]; intros i; simpl; [ discriminate | ].
  destruct (Pos.eqb_spec x y) as [-> | _].
  - intros [= <-]. reflexivity.
  - destruct (position_of x params) as [j | ]; [ | discriminate ].
    intros [= <-]. simpl. exact (IH j eq_refl).
Qed.

(* What is put for the parameter at a position is the argument at that
   position. *)
Lemma lookup_combine_nth (params : list variable) :
  ∀ (args : list simple) (i : nat) (p : variable) (a : simple),
    NoDup params ->
    nth_error params i = Some p ->
    nth_error args i = Some a ->
    lookup p (combine params args) = Some a.
Proof.
  induction params as [ | q params IH]; intros args i p a Hnodup Hp Ha.
  - destruct i; discriminate Hp.
  - inversion Hnodup as [ | q0 rest Hnotin Hnodup']. subst q0 rest.
    destruct args as [ | b args]; [ destruct i; discriminate Ha | ].
    destruct i as [ | i]; simpl in Hp, Ha.
    + injection Hp as ->. injection Ha as ->. simpl.
      rewrite Pos.eqb_refl. reflexivity.
    + simpl. destruct (Pos.eqb_spec p q) as [-> | _].
      * destruct (Hnotin (nth_error_In _ _ Hp)).
      * exact (IH args i p a Hnodup' Hp Ha).
Qed.

(* Once the parameters are bound, the one at the position of a constant
   among the arguments is that constant. *)
Lemma tested_param_bound (params : list variable) (args : list simple)
    (vs : list value) (vars def_vars bound : env) (i : nat) (p : variable)
    (c : const) :
  NoDup params ->
  eval_simples vars args = Some vs ->
  bind_params params vs def_vars = Some bound ->
  nth_error params i = Some p ->
  nth_error args i = Some (Const c) ->
  lookup p bound = Some (Value_const c).
Proof.
  intros Hnodup Hargs Hbind Hp Ha.
  pose proof
    (lookup_combine_params params args vs vars def_vars bound p Hnodup Hargs
       Hbind)
    as Hlookup.
  rewrite (lookup_combine_nth params args i p (Const c) Hnodup Hp Ha)
    in Hlookup.
  symmetry. exact Hlookup.
Qed.

(* The step of a [Switch] on a variable that is a number an arm has. *)
Lemma step_tested_switch (p : variable) (arms : list (Z * apply_cont))
    (n : Z) (target : apply_cont) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  lookup p vars = Some (Value_const (Naked_immediate n)) ->
  find_arm n arms = Some target ->
  step (State (Switch (Var p) arms) vars conts statics hp rd)
  = Step (State (Apply_cont target) vars conts statics hp rd).
Proof.
  intros Hp Harm. rewrite step_switch. cbn [eval_simple]. rewrite Hp.
  cbn [naked_immediate_of_value]. rewrite Harm. reflexivity.
Qed.

Lemma run_on_step (m : nat) (s : state) :
  run_on (S m) (Step s) = run_on m (step s).
Proof. reflexivity. Qed.

(* ======================================================================== *)
(* An expression with applications that go to arms                          *)
(* ======================================================================== *)

(* The relation below is used twice, as that of FlambdaForwardProofs.v
   is. As it stands ([strict] false) the second run answers the first
   where the first is defined. Where the arguments of every application
   that goes to an arm are known to have the kinds of the parameters
   ([strict] true), the second run is undefined where the first is, and
   each run answers the other. *)
Section Strict.

Variable strict : bool.

(* The second application is the first, or the [Apply_cont] of the arm,
   in the handler of its continuation, of the number that it gives the
   parameter tested, as the entry of that continuation says. Where
   [strict], the arguments of the first are known, by the binders
   around it, to have the kinds that the handler gives its
   parameters. *)
Inductive arm_goes (G : kinds_in_scope) (F : arm_forwards)
    : apply_cont -> apply_cont -> Prop :=
  | Arm_goes_same (ac : apply_cont) : arm_goes G F ac ac
  | Arm_goes_on (ac : apply_cont) (params : list variable)
      (metadata : handler_metadata) (i : nat) (arms : list (Z * apply_cont))
      (body : expr) (n : Z) (target : apply_cont) :
    arm_forward_of F (ac_continuation ac)
    = Some (params, metadata, i, arms, body) ->
    nth_error (ac_args ac) i = Some (Const (Naked_immediate n)) ->
    find_arm n arms = Some target ->
    passes (ac_continuation ac) params metadata body target ->
    length (ac_args ac) = length params ->
    (strict = true ->
     args_have_kinds G (ac_args ac) (hm_params_kinds metadata)) ->
    arm_goes G F ac (forwarded params (ac_args ac) target).

(* What an entry says of a handler with these parameters and this
   metadata, as the second term has it. *)
Definition arm_entry_fits (params : list variable)
    (metadata' : handler_metadata) (handler' : expr) (e : option arm_entry)
    : Prop :=
  match e with
  | None => True
  | Some (params0, metadata0, i, arms, _) =>
    params0 = params
    ∧ metadata0 = metadata'
    ∧ (∃ p : variable,
        handler' = Switch (Var p) arms ∧ nth_error params i = Some p)
    ∧ strict_entry strict params metadata'
  end.

(* The second expression is the first with applications that go to
   arms. No binder binds a name that an entry in scope depends on. (The
   warning silenced is about induction schemes, which nothing uses.) *)
#[local] Set Warnings "-register-all".
Inductive arm_fwd : kinds_in_scope -> arm_forwards -> expr -> expr -> Prop :=
  | Arm_fwd_let (G : kinds_in_scope) (F : arm_forwards) (x : variable)
      (kind : flambda_kind) (d : named) (body body' : expr) :
    arm_var_unfrozen F x ->
    arm_fwd ((x, kind) :: G) F body body' ->
    arm_fwd G F (Let x kind d body) (Let x kind d body')
  | Arm_fwd_let_static (G : kinds_in_scope) (F : arm_forwards)
      (bindings : list static_binding) (body body' : expr) :
    arm_fwd G F body body' ->
    arm_fwd G F (Let_static bindings body) (Let_static bindings body')
  | Arm_fwd_let_cont (G : kinds_in_scope) (F : arm_forwards)
      (k : continuation) (params : list variable)
      (metadata metadata' : handler_metadata) (handler handler' : expr)
      (e : option arm_entry) (body body' : expr) :
    arm_fwd (params_in_scope params (hm_params_kinds metadata) G) F handler
      handler' ->
    hm_params_kinds metadata' = hm_params_kinds metadata ->
    (∀ p : variable, p ∈ params -> arm_var_unfrozen F p) ->
    arm_cont_unfrozen F k ->
    arm_entry_fits params metadata' handler' e ->
    arm_fwd G ((k, e) :: F) body body' ->
    arm_fwd G F
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata' handler') body')
  | Arm_fwd_let_cont_recursive (G : kinds_in_scope) (F : arm_forwards)
      (handlers handlers' : list (continuation * continuation_handler))
      (body body' : expr) :
    Forall2
      (fun kh kh' =>
         fst kh = fst kh'
         ∧ handler_params (snd kh) = handler_params (snd kh')
         ∧ handler_kinds (snd kh') = handler_kinds (snd kh)
         ∧ arm_fwd
             (params_in_scope (handler_params (snd kh))
                (handler_kinds (snd kh)) G)
             (arm_plain handlers ++ F) (handler_expr (snd kh))
             (handler_expr (snd kh'))
         ∧ (∀ p : variable,
             p ∈ handler_params (snd kh) -> arm_var_unfrozen F p)
         ∧ arm_cont_unfrozen F (fst kh))
      handlers handlers' ->
    handlers_bind_statics handlers' = handlers_bind_statics handlers ->
    arm_fwd G (arm_plain handlers ++ F) body body' ->
    arm_fwd G F (Let_cont_recursive handlers body)
      (Let_cont_recursive handlers' body')
  | Arm_fwd_apply (G : kinds_in_scope) (F : arm_forwards) (ap : apply_expr) :
    arm_fwd G F (Apply ap) (Apply ap)
  | Arm_fwd_apply_cont (G : kinds_in_scope) (F : arm_forwards)
      (ac ac' : apply_cont) :
    arm_goes G F ac ac' -> arm_fwd G F (Apply_cont ac) (Apply_cont ac')
  | Arm_fwd_switch (G : kinds_in_scope) (F : arm_forwards)
      (scrutinee : simple) (arms arms' : list (Z * apply_cont)) :
    Forall2
      (fun arm arm' => fst arm = fst arm' ∧ arm_goes G F (snd arm) (snd arm'))
      arms arms' ->
    arm_fwd G F (Switch scrutinee arms) (Switch scrutinee arms')
  | Arm_fwd_invalid (G : kinds_in_scope) (F : arm_forwards) :
    arm_fwd G F Invalid Invalid.

(* The handlers of a recursive [Let_cont], one with one. *)
Definition arm_handlers_fwd (G : kinds_in_scope) (inside F : arm_forwards)
    (handlers handlers' : list (continuation * continuation_handler)) : Prop :=
  Forall2
    (fun kh kh' =>
       fst kh = fst kh'
       ∧ handler_params (snd kh) = handler_params (snd kh')
       ∧ handler_kinds (snd kh') = handler_kinds (snd kh)
       ∧ arm_fwd
           (params_in_scope (handler_params (snd kh))
              (handler_kinds (snd kh)) G)
           inside (handler_expr (snd kh)) (handler_expr (snd kh'))
       ∧ (∀ p : variable, p ∈ handler_params (snd kh) -> arm_var_unfrozen F p)
       ∧ arm_cont_unfrozen F (fst kh))
    handlers handlers'.

Lemma arm_handlers_fwd_lookup (G : kinds_in_scope) (inside F : arm_forwards)
    (k : continuation)
    (handlers handlers' : list (continuation * continuation_handler)) :
  arm_handlers_fwd G inside F handlers handlers' ->
  match lookup k handlers, lookup k handlers' with
  | Some h, Some h' =>
    handler_params h = handler_params h'
    ∧ handler_kinds h' = handler_kinds h
    ∧ arm_fwd (params_in_scope (handler_params h) (handler_kinds h) G) inside
        (handler_expr h) (handler_expr h')
    ∧ (∀ p : variable, p ∈ handler_params h -> arm_var_unfrozen F p)
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
   on. *)
Lemma arm_handlers_fwd_names (G : kinds_in_scope) (inside F : arm_forwards)
    (k : continuation)
    (handlers handlers' : list (continuation * continuation_handler)) :
  arm_handlers_fwd G inside F handlers handlers' ->
  lookup k handlers <> None ->
  arm_cont_unfrozen F k.
Proof.
  induction 1 as [ | [k1 h] [k1' h'] handlers handlers' (Hk & Hh) _ IH];
    simpl; [ congruence | ].
  destruct (Pos.eqb_spec k k1) as [-> | _]; [ intros _ | exact IH ].
  simpl in Hh. apply Hh.
Qed.

Lemma arm_find_arm_goes (G : kinds_in_scope) (F : arm_forwards) (i : Z)
    (arms arms' : list (Z * apply_cont)) :
  Forall2
    (fun arm arm' => fst arm = fst arm' ∧ arm_goes G F (snd arm) (snd arm'))
    arms arms' ->
  match find_arm i arms, find_arm i arms' with
  | Some ac, Some ac' => arm_goes G F ac ac'
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
Fixpoint arm_fwd_refl (e : expr) :
  ∀ (G : kinds_in_scope) (F : arm_forwards),
    (∀ k : continuation, arm_forward_of F k = None) -> arm_fwd G F e e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros G F Hnone.
  - apply Arm_fwd_let;
      [ apply arm_var_unfrozen_none, Hnone | apply arm_fwd_refl, Hnone ].
  - apply Arm_fwd_let_static, arm_fwd_refl, Hnone.
  - apply (Arm_fwd_let_cont G F k params metadata metadata handler handler
      None);
      [ apply arm_fwd_refl, Hnone | reflexivity
      | intros p _; apply arm_var_unfrozen_none, Hnone
      | apply arm_cont_unfrozen_none, Hnone | exact I | ].
    apply arm_fwd_refl. intros k'.
    destruct (Pos.eq_dec k' k) as [-> | Hne];
      [ apply arm_forward_of_head
      | rewrite arm_forward_of_other by exact Hne; apply Hnone ].
  - assert (Hnone' :
      ∀ k : continuation, arm_forward_of (arm_plain handlers ++ F) k = None).
    { intros k. rewrite arm_forward_of_plain.
      destruct (lookup k handlers); [ reflexivity | apply Hnone ]. }
    apply Arm_fwd_let_cont_recursive;
      [ | reflexivity | apply arm_fwd_refl, Hnone' ].
    revert Hnone'. generalize (arm_plain handlers ++ F).
    intros F' Hnone'.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      [ apply Forall2_nil | ].
    apply Forall2_cons; [ | exact IH ].
    simpl. split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ reflexivity | ]. split; [ apply arm_fwd_refl, Hnone' | ].
    split; [ intros p _; apply arm_var_unfrozen_none, Hnone
           | apply arm_cont_unfrozen_none, Hnone ].
  - apply Arm_fwd_apply.
  - apply Arm_fwd_apply_cont, Arm_goes_same.
  - apply Arm_fwd_switch.
    induction arms as [ | arm arms IH]; [ apply Forall2_nil | ].
    apply Forall2_cons;
      [ split; [ reflexivity | apply Arm_goes_same ] | exact IH ].
  - apply Arm_fwd_invalid.
Qed.

(* ======================================================================== *)
(* Continuations and states                                                 *)
(* ======================================================================== *)

(* For each entry, the name stands in the second state for a closure
   whose handler is the [Switch] of the entry, on the parameter at the
   position of the entry, and the entry is as [strict] asks; for each
   arm that passes, the variables agree with those of the closure on
   what the arm reads, and the continuation that the arm applies is the
   same in the state as in the closure. *)
Definition arm_held (F : arm_forwards) (vars : env) (conts' : cont_env)
    : Prop :=
  ∀ (k : continuation) (params : list variable) (metadata : handler_metadata)
    (i : nat) (arms : list (Z * apply_cont)) (body : expr),
    arm_forward_of F k = Some (params, metadata, i, arms, body) ->
    ∃ (p : variable) (def_vars : env) (def_conts' : cont_env),
      nth_error params i = Some p
      ∧ resolve k conts'
        = Closure (Handler params metadata (Switch (Var p) arms)) def_vars
            def_conts' []
      ∧ strict_entry strict params metadata
      ∧ ∀ (n : Z) (target : apply_cont),
          find_arm n arms = Some target ->
          passes k params metadata body target ->
          (∀ x : variable,
            x ∉ params ->
            var_free_in_simples x (ac_args target) ->
            lookup x vars = lookup x def_vars)
          ∧ resolve (ac_continuation target) conts'
            = resolve (ac_continuation target) def_conts'.

(* What a continuation name stands for in the two runs: the same thing;
   or closures whose handlers are related under the entries of where
   they were defined, under the same variables, and continuations that
   are in turn so related and for which those entries hold, the second
   handler giving the kinds of the first; or two that are related, with
   the same extra arguments; or what is left of the same
   over-application, to return and to raise to two that are related. *)
Inductive arm_cont_fwd : cont_value -> cont_value -> Prop :=
  | Arm_same_fwd (c : cont_value) : arm_cont_fwd c c
  | Arm_closure_fwd (G : kinds_in_scope) (F : arm_forwards)
      (params : list variable) (metadata metadata' : handler_metadata)
      (handler handler' : expr) (def_vars : env)
      (def_conts def_conts' : cont_env)
      (recursive recursive' : list (continuation * continuation_handler)) :
    arm_handlers_fwd G (arm_plain recursive ++ F) F recursive recursive' ->
    arm_fwd (params_in_scope params (hm_params_kinds metadata) G)
      (arm_plain recursive ++ F) handler handler' ->
    hm_params_kinds metadata' = hm_params_kinds metadata ->
    (∀ p : variable, p ∈ params -> arm_var_unfrozen F p) ->
    (∀ k : continuation,
      arm_cont_fwd (resolve k def_conts) (resolve k def_conts')) ->
    arm_held F def_vars def_conts' ->
    kinds_hold G def_vars ->
    arm_cont_fwd
      (Closure (Handler params metadata handler) def_vars def_conts recursive)
      (Closure (Handler params metadata' handler') def_vars def_conts'
         recursive')
  | Arm_extra_args_fwd (c c' : cont_value) (extra : list value) :
    arm_cont_fwd c c' ->
    arm_cont_fwd (Extra_args c extra) (Extra_args c' extra)
  | Arm_over_fwd (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    arm_cont_fwd return_to return_to' ->
    arm_cont_fwd exn_to exn_to' ->
    arm_cont_fwd
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

Inductive arm_state_fwd : state -> state -> Prop :=
  | Arm_state_fwd (G : kinds_in_scope) (F : arm_forwards) (e e' : expr)
      (vars : env) (conts conts' : cont_env) (statics : static_env)
      (hp : heap) (rd : read_record) :
    arm_fwd G F e e' ->
    (∀ k : continuation, arm_cont_fwd (resolve k conts) (resolve k conts')) ->
    arm_held F vars conts' ->
    kinds_hold G vars ->
    arm_state_fwd (State e vars conts statics hp rd)
      (State e' vars conts' statics hp rd).

Lemma arm_held_nil (vars : env) (conts' : cont_env) : arm_held [] vars conts'.
Proof. intros k params metadata i arms body Hsome. discriminate Hsome. Qed.

Lemma arm_state_fwd_refl (s : state) : arm_state_fwd s s.
Proof.
  destruct s as [e vars conts statics].
  apply (Arm_state_fwd [] []);
    [ apply arm_fwd_refl; reflexivity | intros k; apply Arm_same_fwd
    | apply arm_held_nil | apply kinds_hold_nil ].
Qed.

(* Binding a variable that no entry depends on. *)
Lemma arm_held_var (F : arm_forwards) (vars : env) (conts' : cont_env)
    (x : variable) (v : value) :
  arm_var_unfrozen F x ->
  arm_held F vars conts' ->
  arm_held F ((x, v) :: vars) conts'.
Proof.
  intros Hx Hheld k params metadata i arms body Hsome.
  destruct (Hheld k params metadata i arms body Hsome)
    as (p & def_vars & def_conts' & Hp & Hk & Hse & Harms).
  exists p, def_vars, def_conts'.
  split; [ exact Hp | ]. split; [ exact Hk | ]. split; [ exact Hse | ].
  intros n target Hfind Hpasses.
  destruct (Harms n target Hfind Hpasses) as [Hvars Hcont].
  split; [ | exact Hcont ].
  intros y Hnotin Hfree. rewrite <- (Hvars y Hnotin Hfree). simpl.
  destruct (Pos.eqb_spec y x) as [-> | _]; [ | reflexivity ].
  exfalso. apply (Hx k). rewrite Hsome.
  exists params, metadata, i, arms, body, n, target. auto.
Qed.

(* Binding parameters that no entry depends on. *)
Lemma arm_held_bind (F : arm_forwards) (params : list variable)
    (vs : list value) (vars bound : env) (conts' : cont_env) :
  (∀ p : variable, p ∈ params -> arm_var_unfrozen F p) ->
  bind_params params vs vars = Some bound ->
  arm_held F vars conts' ->
  arm_held F bound conts'.
Proof.
  intros Hparams Hbind Hheld k params0 metadata i arms body Hsome.
  destruct (Hheld k params0 metadata i arms body Hsome)
    as (p & def_vars & def_conts' & Hp & Hk & Hse & Harms).
  exists p, def_vars, def_conts'.
  split; [ exact Hp | ]. split; [ exact Hk | ]. split; [ exact Hse | ].
  intros n target Hfind Hpasses.
  destruct (Harms n target Hfind Hpasses) as [Hvars Hcont].
  split; [ | exact Hcont ].
  intros y Hnotin Hfree. rewrite <- (Hvars y Hnotin Hfree).
  apply (lookup_bind_params_other params vs vars bound y Hbind).
  intros Hin. apply (Hparams y Hin k). rewrite Hsome.
  exists params0, metadata, i, arms, body, n, target. auto.
Qed.

(* The closures of the handlers of a recursive [Let_cont], whose names no
   entry depends on. *)
Lemma arm_held_rec (G : kinds_in_scope) (F : arm_forwards)
    (handlers handlers' : list (continuation * continuation_handler))
    (vars def_vars : env) (def_conts' : cont_env) :
  arm_handlers_fwd G (arm_plain handlers ++ F) F handlers handlers' ->
  arm_held F vars def_conts' ->
  arm_held (arm_plain handlers ++ F) vars
    (rec_closures handlers' def_vars def_conts' ++ def_conts').
Proof.
  intros Hhandlers Hheld k params metadata i arms body Hsome.
  rewrite arm_forward_of_plain in Hsome.
  pose proof (arm_handlers_fwd_lookup _ _ _ k _ _ Hhandlers) as Hlookup.
  destruct (lookup k handlers) eqn:Hk; [ discriminate Hsome | ].
  destruct (lookup k handlers') eqn:Hk'; [ contradiction | ].
  destruct (Hheld k params metadata i arms body Hsome)
    as (p & def_vars0 & def_conts0 & Hp & Hresolve & Hse & Harms).
  exists p, def_vars0, def_conts0.
  rewrite resolve_rec_closures, Hk'.
  split; [ exact Hp | ]. split; [ exact Hresolve | ]. split; [ exact Hse | ].
  intros n target Hfind Hpasses.
  destruct (Harms n target Hfind Hpasses) as [Hvars Hcont].
  split; [ exact Hvars | ].
  rewrite resolve_rec_closures.
  pose proof
    (arm_handlers_fwd_lookup _ _ _ (ac_continuation target) _ _ Hhandlers)
    as Htarget.
  destruct (lookup (ac_continuation target) handlers') eqn:Ht';
    [ | exact Hcont ].
  destruct (lookup (ac_continuation target) handlers) eqn:Ht;
    [ | contradiction ].
  exfalso.
  apply (arm_handlers_fwd_names _ _ _ (ac_continuation target) _ _ Hhandlers
    ltac:(rewrite Ht; discriminate) k).
  rewrite Hsome. exists params, metadata, i, arms, body, n, target. auto.
Qed.

Lemma arm_rec_conts_fwd (G : kinds_in_scope) (F : arm_forwards)
    (handlers handlers' : list (continuation * continuation_handler))
    (def_vars : env) (def_conts def_conts' : cont_env) :
  arm_handlers_fwd G (arm_plain handlers ++ F) F handlers handlers' ->
  (∀ k : continuation,
    arm_cont_fwd (resolve k def_conts) (resolve k def_conts')) ->
  arm_held F def_vars def_conts' ->
  kinds_hold G def_vars ->
  ∀ k : continuation,
    arm_cont_fwd
      (resolve k (rec_closures handlers def_vars def_conts ++ def_conts))
      (resolve k (rec_closures handlers' def_vars def_conts' ++ def_conts')).
Proof.
  intros Hhandlers Hconts Hheld Htyped k. rewrite !resolve_rec_closures.
  pose proof (arm_handlers_fwd_lookup _ _ _ k _ _ Hhandlers) as Hlookup.
  destruct (lookup k handlers) as [[params metadata handler] | ],
    (lookup k handlers') as [[params' metadata' handler'] | ];
    try contradiction; [ | exact (Hconts k) ].
  destruct Hlookup as (Hparams & Hmeta & Hhandler & Hunfrozen).
  simpl in Hparams, Hmeta, Hhandler, Hunfrozen. subst params'.
  apply (Arm_closure_fwd G F); assumption.
Qed.

(* ======================================================================== *)
(* Applying related continuations                                           *)
(* ======================================================================== *)

(* What the first run comes to is answered by what one step of the
   second comes to. Where the first is undefined nothing is asked of
   the second, unless [strict]: then it is undefined too. *)
Definition arm_answer (result result' : step_result) : Prop :=
  match result with
  | Undefined => strict = true -> result' = Undefined
  | Exit k vs hp rd => result' = Exit k vs hp rd
  | Step t => ∃ t' : state, result' = Step t' ∧ arm_state_fwd t t'
  end.

Lemma arm_answer_refl (result : step_result) : arm_answer result result.
Proof.
  destruct result as [t | k vs | ]; simpl;
    [ | reflexivity | intros _; reflexivity ].
  exists t. split; [ reflexivity | apply arm_state_fwd_refl ].
Qed.

Lemma arm_answer_undefined : arm_answer Undefined Undefined.
Proof. intros _. reflexivity. Qed.

(* Where [strict], the arguments of an application that goes to an arm
   have values, of the kinds that the handler of its entry gives its
   parameters and as many as it has parameters: the closure of the
   entry runs on them. *)
Lemma arm_strict_runs (G : kinds_in_scope) (args : list simple)
    (params : list variable) (metadata : handler_metadata) (vars : env) :
  kinds_hold G vars ->
  strict_entry strict params metadata ->
  (strict = true -> args_have_kinds G args (hm_params_kinds metadata)) ->
  strict = true ->
  ∃ vs : list value,
    eval_simples vars args = Some vs
    ∧ map kind_of_value vs = hm_params_kinds metadata
    ∧ length params = length vs.
Proof.
  intros Htyped Hse Hhave Hstrict.
  destruct (Hhave Hstrict vars Htyped) as (vs & Hvs & Hk).
  exists vs. split; [ exact Hvs | ]. split; [ exact Hk | ].
  rewrite (Hse Hstrict), <- Hk. apply length_map.
Qed.

(* What the relation says of a handler that has become a [Switch], and
   of the two applications. *)
Lemma arm_fwd_switch_inv (G : kinds_in_scope) (F : arm_forwards) (e : expr)
    (scrutinee : simple) (arms' : list (Z * apply_cont)) :
  arm_fwd G F e (Switch scrutinee arms') ->
  ∃ arms : list (Z * apply_cont),
    e = Switch scrutinee arms
    ∧ Forall2
        (fun arm arm' =>
           fst arm = fst arm' ∧ arm_goes G F (snd arm) (snd arm'))
        arms arms'.
Proof. intros Hfwd. inversion Hfwd; subst. eauto. Qed.

Lemma arm_goes_inv (G : kinds_in_scope) (F : arm_forwards)
    (b target : apply_cont) :
  arm_goes G F b target ->
  target = b
  ∨ ∃ (params_b : list variable) (metadata_b : handler_metadata) (i_b : nat)
      (arms_b : list (Z * apply_cont)) (body_b : expr) (n_b : Z)
      (target_b : apply_cont),
      arm_forward_of F (ac_continuation b)
      = Some (params_b, metadata_b, i_b, arms_b, body_b)
      ∧ nth_error (ac_args b) i_b = Some (Const (Naked_immediate n_b))
      ∧ find_arm n_b arms_b = Some target_b
      ∧ passes (ac_continuation b) params_b metadata_b body_b target_b
      ∧ length (ac_args b) = length params_b
      ∧ (strict = true ->
         args_have_kinds G (ac_args b) (hm_params_kinds metadata_b))
      ∧ target = forwarded params_b (ac_args b) target_b.
Proof.
  intros [ ac
         | ac params_b metadata_b i_b arms_b body_b n_b target_b Hentry Hnth
             Hfind Hpasses Hlength Hkinds ];
    [ left; reflexivity | right ].
  exists params_b, metadata_b, i_b, arms_b, body_b, n_b, target_b. auto 10.
Qed.

(* The same full call, returning and raising to related continuations. *)
Lemma arm_call_code_fwd (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  arm_cont_fwd return_to return_to' ->
  arm_cont_fwd exn_to exn_to' ->
  arm_answer
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c vs args_arity return_arity return_to' exn_to'
       statics hp rd).
Proof.
  intros Hreturn Hraise.
  destruct (call_code_cases closure c vs args_arity return_arity statics hp rd)
    as [Hundefined | (callee_vars & Hcall)].
  - rewrite !Hundefined. apply arm_answer_undefined.
  - rewrite !Hcall. cbn [arm_answer]. eexists. split; [ reflexivity | ].
    apply (Arm_state_fwd [] []);
      [ apply arm_fwd_refl; reflexivity | | apply arm_held_nil
      | apply kinds_hold_nil ].
    intros k'. unfold resolve. simpl.
    destruct (Pos.eqb k' (code_return_continuation c)); [ exact Hreturn | ].
    destruct (Pos.eqb k' (code_exn_continuation c));
      [ exact Hraise | apply Arm_same_fwd ].
Qed.

(* The same call of any shape, where the two return continuations take
   a partial application to results of which, after some steps of the
   first, one answers the other. *)
Lemma arm_run_call_fwd (shape : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  arm_cont_fwd return_to return_to' ->
  arm_cont_fwd exn_to exn_to' ->
  (∀ partial : value,
    ∃ n : nat,
      arm_answer (run_on n (apply_cont_value return_to [partial] statics hp rd))
        (apply_cont_value return_to' [partial] statics hp rd)) ->
  ∃ n : nat,
    arm_answer
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
  - exists 0. rewrite run_on_zero. apply arm_call_code_fwd; assumption.
  - exists 0. rewrite run_on_zero.
    apply arm_call_code_fwd; [ apply Arm_over_fwd; assumption | exact Hraise ].
  - apply Hpartial.
  - exists 0. rewrite run_on_zero. apply arm_answer_undefined.
Qed.

(* Related continuations applied to the same values: what the first
   comes to, after some steps, is answered by what the second comes
   to. And where the second is the closure of an entry, the first
   applied to the values of some arguments, of which the one at the
   position of the parameter tested is a number, comes, after some
   steps, to what answers one step of the [Apply_cont] of the arm of
   that number with those arguments put for the parameters. *)
Lemma arm_cont_fwd_rounds (c c' : cont_value) :
  arm_cont_fwd c c' ->
  (∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record),
    ∃ m : nat,
      arm_answer (run_on m (apply_cont_value c vs statics hp rd))
        (apply_cont_value c' vs statics hp rd))
  ∧ (∀ (params : list variable) (metadata' : handler_metadata) (p : variable)
       (arms' : list (Z * apply_cont)) (def_vars0 : env)
       (def_conts0 : cont_env) (i : nat) (n : Z) (target : apply_cont)
       (args : list simple) (vs : list value) (vars : env) (conts' : cont_env)
       (statics : static_env) (hp : heap) (rd : read_record),
       c'
       = Closure (Handler params metadata' (Switch (Var p) arms')) def_vars0
           def_conts0 [] ->
       nth_error params i = Some p ->
       nth_error args i = Some (Const (Naked_immediate n)) ->
       find_arm n arms' = Some target ->
       NoDup params ->
       runs_on_values strict params metadata' vs ->
       length args = length params ->
       eval_simples vars args = Some vs ->
       (∀ x : variable,
         x ∉ params ->
         var_free_in_simples x (ac_args target) ->
         lookup x vars = lookup x def_vars0) ->
       resolve (ac_continuation target) conts'
       = resolve (ac_continuation target) def_conts0 ->
       ∃ m : nat,
         arm_answer (run_on m (apply_cont_value c vs statics hp rd))
           (step
              (State (Apply_cont (forwarded params args target)) vars conts'
                 statics hp rd))).
Proof.
  induction 1
    as [ c
       | G F params metadata metadata' handler handler' def_vars def_conts
           def_conts' recursive recursive' Hrecursive Hhandler Hmeta Hparams
           Hdef_conts IHdef_conts Hheld Htyped
       | c c' extra Hc [IHsame _]
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn [IHreturn _] Hraise _ ].
  - (* the same continuation *)
    split;
      [ intros vs statics hp; exists 0; rewrite run_on_zero;
        apply arm_answer_refl
      | ].
    intros params metadata' p arms' def_vars0 def_conts0 i n target args vs
      vars conts' statics hp rd -> Hp Ha Harm Hnodup Hruns Hlen Hargs Hvars
      Hcont.
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
    exists 2.
    rewrite run_on_step,
      (step_tested_switch p arms' n target bound _ statics hp rd
         (tested_param_bound params args vs vars def_vars0 bound i p _ Hnodup
            Hargs Hbind Hp Ha)
         Harm),
      run_on_step, run_on_zero,
      (step_forwarded params args target vs vars def_vars0 bound conts'
         def_conts0 statics _ rd_ Hnodup Hargs Hbind Hvars Hcont).
    apply arm_answer_refl.
  - (* related closures *)
    split.
    + intros vs statics hp. exists 0. rewrite run_on_zero. simpl.
      rewrite Hmeta.
      destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
        eqn:Hkinds; [ | apply arm_answer_undefined ].
      apply kinds_eqb_eq in Hkinds.
      destruct (bind_params params vs def_vars) as [bound | ] eqn:Hbind;
        [ | apply arm_answer_undefined ].
      eexists. split; [ reflexivity | ].
      apply
        (Arm_state_fwd (params_in_scope params (hm_params_kinds metadata) G)
           (arm_plain recursive ++ F));
        [ exact Hhandler | | | ].
      * apply (arm_rec_conts_fwd G F); assumption.
      * apply (arm_held_rec G); [ exact Hrecursive | ].
        exact (arm_held_bind F params vs def_vars bound def_conts' Hparams
          Hbind Hheld).
      * rewrite <- Hkinds.
        exact (params_in_scope_hold params vs G def_vars bound Htyped Hbind).
    + intros params0 metadata0 p arms' def_vars0 def_conts0 i n target args vs
        vars conts' statics hp rd Hc' Hp Ha Harm Hnodup Hruns Hlen Hargs Hvars
        Hcont.
      injection Hc' as Heq_params Heq_metadata Heq_handler Heq_vars Heq_conts
        Heq_recursive.
      subst params0 metadata0 handler' def_vars0 def_conts0 recursive'.
      unfold arm_handlers_fwd in Hrecursive.
      destruct recursive as [ | kh rest]; [ | inversion Hrecursive ].
      simpl in Hhandler.
      destruct (arm_fwd_switch_inv _ F handler (Var p) arms' Hhandler)
        as (arms & -> & Harms).
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
      assert (Hheld_bound : arm_held F bound def_conts')
        by exact (arm_held_bind F params vs def_vars bound def_conts' Hparams
             Hbind Hheld).
      assert (Htyped_bound :
        kinds_hold (params_in_scope params (hm_params_kinds metadata) G) bound).
      { rewrite <- Hkinds.
        exact (params_in_scope_hold params vs G def_vars bound Htyped Hbind). }
      pose proof
        (tested_param_bound params args vs vars def_vars bound i p _ Hnodup
           Hargs Hbind Hp Ha)
        as Hp_bound.
      pose proof (arm_find_arm_goes _ F n arms arms' Harms) as Hb.
      rewrite Harm in Hb.
      destruct (find_arm n arms) as [b | ] eqn:Hfind_b; [ | contradiction ].
      destruct (arm_goes_inv _ F b target Hb)
        as [-> | (params_b & metadata_b & i_b & arms_b & body_b & n_b
                  & target_b & Hentry & Hnth_b & Hfind_tb & Hpasses_b
                  & Hlength & Hkinds_b & Htarget)].
      * (* the arm of the first closure is the same application *)
        destruct (IHdef_conts (ac_continuation b)) as [IHapply _].
        destruct (eval_simples bound (ac_args b)) as [vt | ] eqn:Hargs_b.
        -- destruct (IHapply vt statics hp rd) as [m Hanswer].
           exists (S (S m)).
           rewrite run_on_step,
             (step_tested_switch p arms n b bound _ statics hp rd Hp_bound
                Hfind_b),
             run_on_step, !step_apply_cont, Hargs_b.
           unfold apply_continuation. exact Hanswer.
        -- exists 2.
           rewrite run_on_step,
             (step_tested_switch p arms n b bound _ statics hp rd Hp_bound
                Hfind_b),
             run_on_step, run_on_zero, !step_apply_cont, Hargs_b.
           apply arm_answer_undefined.
      * (* it goes to an arm in its turn *)
        destruct
          (Hheld_bound (ac_continuation b) params_b metadata_b i_b arms_b
             body_b Hentry)
          as (p_b & def_vars_b & def_conts_b & Hp_b & Hresolve_b & Hse_b
              & Harms_b).
        destruct (Harms_b n_b target_b Hfind_tb Hpasses_b)
          as [Hvars_b Hcont_b].
        destruct (passes_inv _ _ _ _ _ Hpasses_b) as (Hnodup_b & _).
        destruct (IHdef_conts (ac_continuation b)) as [_ IHon].
        pose proof
          (arm_strict_runs _ (ac_args b) params_b metadata_b bound
             Htyped_bound Hse_b Hkinds_b)
          as Hruns_b.
        subst target.
        destruct (eval_simples bound (ac_args b)) as [vb | ] eqn:Hargs_b.
        -- assert (Hruns_vb : runs_on_values strict params_b metadata_b vb).
           { intros Hstrict.
             destruct (Hruns_b Hstrict) as (vb' & Hvb' & Hrest).
             injection Hvb' as <-. exact Hrest. }
           destruct
             (IHon params_b metadata_b p_b arms_b def_vars_b def_conts_b i_b
                n_b target_b (ac_args b) vb bound def_conts' statics hp
                rd Hresolve_b Hp_b Hnth_b Hfind_tb Hnodup_b Hruns_vb Hlength
                Hargs_b Hvars_b Hcont_b)
             as [m Hanswer].
           exists (S (S m)).
           rewrite run_on_step,
             (step_tested_switch p arms n b bound _ statics hp rd Hp_bound
                Hfind_b),
             run_on_step, (step_apply_cont b), Hargs_b.
           unfold apply_continuation. exact Hanswer.
        -- exists 2.
           rewrite run_on_step,
             (step_tested_switch p arms n b bound _ statics hp rd Hp_bound
                Hfind_b),
             run_on_step, run_on_zero, (step_apply_cont b), Hargs_b.
           intros Hstrict.
           destruct (Hruns_b Hstrict) as (vb' & Hvb' & _).
           discriminate Hvb'.
  - (* the same extra arguments *)
    split; [ intros vs statics hp; simpl; apply IHsame | ].
    intros params metadata' p arms' def_vars0 def_conts0 i n target args vs
      vars conts' statics hp rd Hc'.
    discriminate Hc'.
  - (* what is left of the same over-application *)
    split.
    + intros vs statics hp rd. rewrite !apply_cont_value_over_application.
      destruct vs as [ | v [ | v' vs]];
        try (exists 0; rewrite run_on_zero; apply arm_answer_undefined).
      unfold call_function.
      apply arm_run_call_fwd; [ exact Hreturn | exact Hraise | ].
      intros partial. apply IHreturn.
    + intros params metadata' p arms' def_vars0 def_conts0 i n target args vs
        vars conts' statics hp rd Hc'.
      discriminate Hc'.
Qed.

(* ======================================================================== *)
(* The simulation                                                           *)
(* ======================================================================== *)

Lemma arm_answered_step (result : step_result) (s' : state) :
  arm_answer result (step s') -> round_answered arm_state_fwd result s'.
Proof.
  destruct result as [t | k vs | ]; unfold arm_answer, round_answered;
    [ | | exact (fun _ => I) ].
  - intros (t' & Hstep & Hrel). exists 0, t'.
    split; [ rewrite run_one; exact Hstep | exact Hrel ].
  - intros Hstep. exists 1. rewrite run_one. exact Hstep.
Qed.

(* Some steps of the first state are answered by one of the second. *)
Lemma arm_fwd_round_answer (st st' : state) :
  arm_state_fwd st st' -> ∃ m : nat, arm_answer (run (S m) st) (step st').
Proof.
  intros [G F e e' vars conts conts' statics hp rd Hfwd Hconts Hheld Htyped].
  destruct Hfwd
    as [ G F x kind d body body' Hx Hbody | G F bindings body body' Hbody
       | G F k params metadata metadata' handler handler' entry body body'
           Hhandler Hmeta Hparams Hk Hfits Hbody
       | G F handlers handlers' body body' Hhandlers Hbinds Hbody
       | G F ap | G F ac ac' Hgoes
       | G F scrutinee arms arms' Harms | G F ].
  - (* Let *)
    exists 0. rewrite run_one, !step_let_exec.
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ];
      [ | apply arm_answer_undefined ].
    destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind;
      [ | apply arm_answer_undefined ].
    apply kind_eqb_eq in Hkind. subst kind.
    eexists. split; [ reflexivity | ].
    apply (Arm_state_fwd ((x, kind_of_value v) :: G) F);
      [ exact Hbody | exact Hconts | | apply kinds_hold_cons, Htyped ].
    apply arm_held_var; assumption.
  - (* Let_static *)
    exists 0. rewrite run_one, !step_let_static.
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      [ | apply arm_answer_undefined ].
    eexists. split; [ reflexivity | ].
    apply (Arm_state_fwd G F); assumption.
  - (* Let_cont_non_recursive *)
    exists 0. rewrite run_one, !step_let_cont_non_recursive.
    eexists. split; [ reflexivity | ].
    apply (Arm_state_fwd G ((k, entry) :: F));
      [ exact Hbody | | | exact Htyped ].
    + intros k'. destruct (Pos.eq_dec k' k) as [-> | Hne].
      * rewrite !resolve_head.
        apply (Arm_closure_fwd G F params metadata metadata' handler handler'
          vars conts conts' [] []);
          [ apply Forall2_nil | exact Hhandler | exact Hmeta | exact Hparams
          | exact Hconts | exact Hheld | exact Htyped ].
      * rewrite !resolve_other by exact Hne. exact (Hconts k').
    + intros k' params0 metadata0 i arms body0 Hsome.
      destruct (Pos.eq_dec k' k) as [-> | Hne].
      * rewrite arm_forward_of_head in Hsome. subst entry.
        destruct Hfits as (-> & -> & (p & -> & Hp) & Hse).
        exists p, vars, conts'. rewrite resolve_head.
        split; [ exact Hp | ]. split; [ reflexivity | ].
        split; [ exact Hse | ].
        intros n target Hfind Hpasses.
        destruct (passes_inv _ _ _ _ _ Hpasses) as (_ & Htarget & _).
        split; [ reflexivity | apply resolve_other; exact Htarget ].
      * rewrite arm_forward_of_other in Hsome by exact Hne.
        destruct (Hheld k' params0 metadata0 i arms body0 Hsome)
          as (p & def_vars & def_conts' & Hp & Hresolve & Hse & Harms).
        exists p, def_vars, def_conts'.
        rewrite (resolve_other k' k) by exact Hne.
        split; [ exact Hp | ]. split; [ exact Hresolve | ].
        split; [ exact Hse | ].
        intros n target Hfind Hpasses.
        destruct (Harms n target Hfind Hpasses) as [Hvars Hcont].
        split; [ exact Hvars | ].
        rewrite resolve_other; [ exact Hcont | ].
        intros Heq. apply (Hk k'). rewrite Hsome.
        exists params0, metadata0, i, arms, body0, n, target. auto.
  - (* Let_cont_recursive *)
    exists 0. rewrite run_one, !step_let_cont_recursive, Hbinds.
    destruct (handlers_bind_statics handlers);
      [ apply arm_answer_undefined | ].
    eexists. split; [ reflexivity | ].
    apply (Arm_state_fwd G (arm_plain handlers ++ F));
      [ exact Hbody | | | exact Htyped ].
    + apply (arm_rec_conts_fwd G F); assumption.
    + apply (arm_held_rec G); assumption.
  - (* Apply: the same call, returning to related continuations *)
    assert (Hround :
      ∃ m : nat,
        arm_answer (run_on m (step (State (Apply ap) vars conts statics hp rd)))
          (step (State (Apply ap) vars conts' statics hp rd))).
    { rewrite !step_apply.
      destruct (eval_simple vars (ap_callee ap)) as [callee | ];
        [ | exists 0; rewrite run_on_zero; apply arm_answer_undefined ].
      destruct (eval_simples vars (ap_args ap)) as [vs | ];
        [ | exists 0; rewrite run_on_zero; apply arm_answer_undefined ].
      destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
        [ | exists 0; rewrite run_on_zero; apply arm_answer_undefined ].
      unfold call_function.
      apply arm_run_call_fwd.
      - apply Hconts.
      - destruct extra as [ | v extra]; simpl;
          [ apply Hconts | apply Arm_extra_args_fwd, Hconts ].
      - intros partial.
        exact
          (proj1 (arm_cont_fwd_rounds _ _ (Hconts (ap_continuation ap)))
             [partial] statics hp rd_). }
    destruct Hround as [m Hround]. exists m. rewrite run_succ_on.
    exact Hround.
  - (* Apply_cont *)
    destruct (arm_cont_fwd_rounds _ _ (Hconts (ac_continuation ac)))
      as [Hsame Hon].
    destruct (arm_goes_inv G F ac ac' Hgoes)
      as [-> | (params & metadata & i & arms & body & n & target & Hentry
                & Hnth & Hfind & Hpasses & Hlength & Hkinds & ->)].
    + destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs.
      * destruct (Hsame vs statics hp rd) as [m Hanswer]. exists m.
        rewrite run_succ_on, !step_apply_cont, Hargs.
        unfold apply_continuation. exact Hanswer.
      * exists 0. rewrite run_one, !step_apply_cont, Hargs.
        apply arm_answer_undefined.
    + destruct (Hheld (ac_continuation ac) params metadata i arms body Hentry)
        as (p & def_vars & def_conts' & Hp & Hresolve & Hse & Harms).
      destruct (Harms n target Hfind Hpasses) as [Hvars Hcont].
      destruct (passes_inv _ _ _ _ _ Hpasses) as (Hnodup & _).
      pose proof
        (arm_strict_runs G (ac_args ac) params metadata vars Htyped Hse
           Hkinds)
        as Hruns.
      destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs.
      * assert (Hruns_vs : runs_on_values strict params metadata vs).
        { intros Hstrict. destruct (Hruns Hstrict) as (vs' & Hvs' & Hrest).
          injection Hvs' as <-. exact Hrest. }
        destruct
          (Hon params metadata p arms def_vars def_conts' i n target
             (ac_args ac) vs vars conts' statics hp rd Hresolve Hp Hnth Hfind
             Hnodup Hruns_vs Hlength Hargs Hvars Hcont)
          as [m Hanswer].
        exists m. rewrite run_succ_on, (step_apply_cont ac), Hargs.
        exact Hanswer.
      * exists 0. rewrite run_one, (step_apply_cont ac), Hargs.
        intros Hstrict.
        destruct (Hruns Hstrict) as (vs' & Hvs' & _). discriminate Hvs'.
  - (* Switch *)
    exists 0. rewrite run_one, !step_switch.
    destruct (eval_simple vars scrutinee) as [v | ];
      [ | apply arm_answer_undefined ].
    destruct (naked_immediate_of_value v) as [i | ];
      [ | apply arm_answer_undefined ].
    pose proof (arm_find_arm_goes G F i arms arms' Harms) as Harm.
    destruct (find_arm i arms) as [ac | ], (find_arm i arms') as [ac' | ];
      try contradiction; [ | apply arm_answer_undefined ].
    eexists. split; [ reflexivity | ].
    apply (Arm_state_fwd G F);
      [ apply Arm_fwd_apply_cont, Harm | exact Hconts | exact Hheld
      | exact Htyped ].
  - (* Invalid *)
    exists 0. rewrite run_one, !step_invalid. apply arm_answer_undefined.
Qed.

Lemma arm_fwd_round (st st' : state) :
  arm_state_fwd st st' ->
  ∃ m : nat, round_answered arm_state_fwd (run (S m) st) st'.
Proof.
  intros Hrel. destruct (arm_fwd_round_answer st st' Hrel) as [m Hanswer].
  exists m. apply arm_answered_step, Hanswer.
Qed.

End Strict.

(* Where the entries are strict, one step of the second state is
   answered by some steps of the first. *)
Lemma arm_fwd_round_back (st' st : state) :
  arm_state_fwd true st st' ->
  ∃ m : nat,
    round_answered (fun t' t => arm_state_fwd true t t') (run (S m) st') st.
Proof.
  intros Hrel.
  destruct (arm_fwd_round_answer true st st' Hrel) as [m Hanswer].
  exists 0. rewrite run_one. unfold arm_answer in Hanswer.
  unfold round_answered.
  destruct (run (S m) st) as [t | k vs | ] eqn:Hrun.
  - destruct Hanswer as (t' & -> & Hrel'). exists m, t.
    split; [ exact Hrun | exact Hrel' ].
  - rewrite Hanswer. exists (S m). exact Hrun.
  - rewrite (Hanswer eq_refl). exact I.
Qed.

(* ======================================================================== *)
(* The pass relates a term to its result                                    *)
(* ======================================================================== *)

Fixpoint forward_to_arms_binds_statics (e : expr) :
  ∀ (kinded : bool) (G : kinds_in_scope) (F : arm_forwards),
    binds_statics (forward_to_arms_under kinded G F e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros kinded G F; simpl; try reflexivity.
  - apply forward_to_arms_binds_statics.
  - rewrite forward_to_arms_binds_statics. reflexivity.
  - rewrite !forward_to_arms_binds_statics. reflexivity.
  - rewrite forward_to_arms_binds_statics. f_equal.
    generalize (map (fun '(k, _) => (k, @None arm_entry)) handlers ++ F).
    intros F'.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite forward_to_arms_binds_statics, IH. reflexivity.
Qed.

(* An application that the pass makes go to an arm is one that the
   relation lets go there. *)
Lemma arm_goes_forward (kinded : bool) (G : kinds_in_scope)
    (F : arm_forwards) (ac : apply_cont) :
  arm_goes kinded G F ac (forward_to_arm_apply_cont kinded G F ac).
Proof.
  unfold forward_to_arm_apply_cont, arm_of_apply_cont.
  destruct (lookup (ac_continuation ac) F)
    as [[[[[[params metadata] i] arms] body] | ] | ] eqn:Hk;
    try apply Arm_goes_same.
  destruct (nth_error (ac_args ac) i) as [[x | sym | c] | ] eqn:Hnth;
    try apply Arm_goes_same.
  destruct c as [n | | | | | | | | | | | | | | ]; try apply Arm_goes_same.
  destruct (find_arm n arms) as [target | ] eqn:Hfind;
    [ | apply Arm_goes_same ].
  destruct
    (forwarding (ac_continuation ac) params metadata (Apply_cont target) body)
    as [[params' target'] | ] eqn:Hforwarding; [ | apply Arm_goes_same ].
  destruct (forwarding_some _ _ _ _ _ _ _ Hforwarding)
    as (Hpasses & -> & ->).
  destruct
    (Nat.eqb (length (ac_args ac)) (length params) && no_trap_action ac
     && no_trap_action target
     && (negb kinded || have_kinds G (ac_args ac) (hm_params_kinds metadata)))
    eqn:Hcheck; [ | apply Arm_goes_same ].
  apply andb_true_iff in Hcheck as [Hcheck Hhave].
  apply andb_true_iff in Hcheck as [Hcheck _].
  apply andb_true_iff in Hcheck as [Hlength _].
  apply Nat.eqb_eq in Hlength.
  apply (Arm_goes_on kinded G F ac params metadata i arms body n target);
    [ unfold arm_forward_of; rewrite Hk; reflexivity | exact Hnth
    | exact Hfind | exact Hpasses | exact Hlength | ].
  intros ->. simpl in Hhave. exact (have_kinds_sound G _ _ Hhave).
Qed.

(* What the relation asks of an expression under the entries [F], as the
   pass can check it: the expression binds no name that an entry
   depends on. *)
Definition arm_safe (F : arm_forwards) (e : expr) : Prop :=
  ∀ k : continuation,
    (∀ x : variable,
      arm_frozen_var k (arm_forward_of F k) x -> binds_var x e = false)
    ∧ (∀ k0 : continuation,
        arm_frozen_cont k (arm_forward_of F k) k0 -> binds_cont k0 e = false).

Lemma arm_safe_none (F : arm_forwards) (e : expr) :
  (∀ k : continuation, arm_forward_of F k = None) -> arm_safe F e.
Proof.
  intros Hnone k. rewrite Hnone.
  split;
    [ intros x Hx; destruct (arm_frozen_var_none k x Hx)
    | intros k0 Hk0; destruct (arm_frozen_cont_none k k0 Hk0) ].
Qed.

Lemma arm_safe_cons (F : arm_forwards) (k0 : continuation)
    (e0 : option arm_entry) (e : expr) :
  arm_safe F e ->
  (∀ x : variable, arm_frozen_var k0 e0 x -> binds_var x e = false) ->
  (∀ k1 : continuation, arm_frozen_cont k0 e0 k1 -> binds_cont k1 e = false) ->
  arm_safe ((k0, e0) :: F) e.
Proof.
  intros Hsafe Hvars Hconts k.
  destruct (Pos.eq_dec k k0) as [-> | Hne];
    [ rewrite arm_forward_of_head; auto
    | rewrite arm_forward_of_other by exact Hne; apply Hsafe ].
Qed.

Lemma arm_safe_plain (F : arm_forwards)
    (handlers : list (continuation * continuation_handler)) (e : expr) :
  arm_safe F e -> arm_safe (arm_plain handlers ++ F) e.
Proof.
  intros Hsafe k. rewrite arm_forward_of_plain.
  destruct (lookup k handlers); [ | apply Hsafe ].
  split;
    [ intros x Hx; destruct (arm_frozen_var_none k x Hx)
    | intros k0 Hk0; destruct (arm_frozen_cont_none k k0 Hk0) ].
Qed.

(* The entry that the pass makes is one for the handler, which the body
   leaves alone. *)
Lemma arm_entry_of_fits (kinded : bool) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler' body : expr) :
  arm_entry_fits kinded params metadata handler'
    (arm_entry_of kinded params metadata handler' body)
  ∧ (∀ x : variable,
      arm_frozen_var k (arm_entry_of kinded params metadata handler' body) x ->
      binds_var x body = false)
  ∧ (∀ k0 : continuation,
      arm_frozen_cont k (arm_entry_of kinded params metadata handler' body)
        k0 ->
      binds_cont k0 body = false).
Proof.
  assert (Hnone :
    arm_entry_fits kinded params metadata handler' None
    ∧ (∀ x : variable, arm_frozen_var k None x -> binds_var x body = false)
    ∧ (∀ k0 : continuation,
        arm_frozen_cont k None k0 -> binds_cont k0 body = false)).
  { split; [ exact I | ].
    split;
      [ intros x Hx; destruct (arm_frozen_var_none k x Hx)
      | intros k0 Hk0; destruct (arm_frozen_cont_none k k0 Hk0) ]. }
  unfold arm_entry_of.
  destruct handler' as [ | | | | | | scrutinee arms | ]; try exact Hnone.
  destruct scrutinee as [p | sym | c]; try exact Hnone.
  destruct
    (kinded
     && negb (Nat.eqb (length params) (length (hm_params_kinds metadata))))
    eqn:Hcheck; [ exact Hnone | ].
  destruct (position_of p params) as [i | ] eqn:Hposition;
    [ | exact Hnone ].
  split; [ | split ].
  - split; [ reflexivity | ]. split; [ reflexivity | ].
    split;
      [ exists p; split;
          [ reflexivity | exact (position_of_nth p params i Hposition) ]
      | ].
    intros ->. simpl in Hcheck. apply negb_false_iff, Nat.eqb_eq in Hcheck.
    exact Hcheck.
  - intros x
      (params0 & metadata0 & i0 & arms0 & body0 & n & target & Hsome & _
       & Hpasses & Hnotin & Hfree).
    injection Hsome as <- <- <- <- <-.
    destruct (passes_inv _ _ _ _ _ Hpasses) as (_ & _ & Hvars & _).
    exact (Hvars x Hnotin Hfree).
  - intros k0
      (params0 & metadata0 & i0 & arms0 & body0 & n & target & Hsome & _
       & Hpasses & <-).
    injection Hsome as <- <- <- <- <-.
    destruct (passes_inv _ _ _ _ _ Hpasses) as (_ & _ & _ & Hcont).
    exact Hcont.
Qed.

Lemma arm_safe_let (F : arm_forwards) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  arm_safe F (Let x kind d body) -> arm_safe F body ∧ arm_var_unfrozen F x.
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

Lemma arm_safe_let_cont (F : arm_forwards) (k0 : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) :
  arm_safe F
    (Let_cont_non_recursive k0 (Handler params metadata handler) body) ->
  arm_safe F handler ∧ arm_safe F body
  ∧ (∀ p : variable, p ∈ params -> arm_var_unfrozen F p)
  ∧ arm_cont_unfrozen F k0.
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

Lemma arm_safe_let_cont_recursive (F : arm_forwards)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  arm_safe F (Let_cont_recursive handlers body) ->
  arm_safe F body
  ∧ ∀ (k0 : continuation) (params : list variable)
      (metadata : handler_metadata) (handler : expr),
      (k0, Handler params metadata handler) ∈ handlers ->
      arm_safe F handler
      ∧ (∀ p : variable, p ∈ params -> arm_var_unfrozen F p)
      ∧ arm_cont_unfrozen F k0.
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
        arm_frozen_var k (arm_forward_of F k) x ->
        among x params = false ∧ binds_var x handler = false).
    { intros k x Hx. destruct (Hsafe k) as [Hvars _].
      specialize (Hvars x Hx). simpl in Hvars.
      apply orb_false_iff in Hvars as [Hhandlers _].
      pose proof (existsb_false_in _ _ _ Hhandlers Hin) as Hhandler.
      simpl in Hhandler. apply orb_false_iff in Hhandler. exact Hhandler. }
    assert (Hconts_handler :
      ∀ k k1 : continuation,
        arm_frozen_cont k (arm_forward_of F k) k1 ->
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

Fixpoint forward_to_arms_fwd (e : expr) :
  ∀ (kinded : bool) (G : kinds_in_scope) (F : arm_forwards),
    arm_safe F e ->
    arm_fwd kinded G F e (forward_to_arms_under kinded G F e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros kinded G F Hsafe; simpl.
  - destruct (arm_safe_let F x kind d body Hsafe) as [Hbody Hx].
    apply Arm_fwd_let; [ exact Hx | apply forward_to_arms_fwd, Hbody ].
  - apply Arm_fwd_let_static, forward_to_arms_fwd. exact Hsafe.
  - destruct (arm_safe_let_cont F k params metadata handler body Hsafe)
      as (Hhandler & Hbody & Hparams & Hk).
    set (handler' :=
      forward_to_arms_under kinded
        (params_in_scope params (hm_params_kinds metadata) G) F handler).
    destruct (arm_entry_of_fits kinded k params metadata handler' body)
      as (Hfits & Hvars & Hconts).
    apply (Arm_fwd_let_cont kinded G F k params metadata metadata handler
      handler' (arm_entry_of kinded params metadata handler' body));
      [ apply forward_to_arms_fwd, Hhandler | reflexivity
      | exact Hparams | exact Hk | exact Hfits | ].
    apply forward_to_arms_fwd, arm_safe_cons; assumption.
  - destruct (arm_safe_let_cont_recursive F handlers body Hsafe)
      as [Hbody Hhandlers].
    change (map (fun '(k, _) => (k, @None arm_entry)) handlers)
      with (arm_plain handlers).
    assert (Hall :
      Forall
        (fun kh =>
           arm_safe (arm_plain handlers ++ F) (handler_expr (snd kh))
           ∧ (∀ p : variable,
               p ∈ handler_params (snd kh) -> arm_var_unfrozen F p)
           ∧ arm_cont_unfrozen F (fst kh))
        handlers).
    { apply Forall_forall. intros [k [params metadata handler]] Hin.
      destruct (Hhandlers k params metadata handler Hin)
        as (Hhandler & Hparams & Hk).
      split; [ apply arm_safe_plain, Hhandler | split; assumption ]. }
    apply Arm_fwd_let_cont_recursive.
    + clear Hsafe Hhandlers. revert Hall.
      generalize (arm_plain handlers ++ F). intros F' Hall.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        [ apply Forall2_nil | ].
      inversion Hall as [ | kh rest (Hhandler & Hparams & Hk) Hrest].
      subst kh rest.
      apply Forall2_cons; [ | exact (IH Hrest) ].
      simpl. split; [ reflexivity | ]. split; [ reflexivity | ].
      split; [ reflexivity | ].
      split; [ apply forward_to_arms_fwd, Hhandler | ].
      split; assumption.
    + clear. unfold handlers_bind_statics.
      generalize (arm_plain handlers ++ F). intros F'.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        simpl; [ reflexivity | ].
      rewrite forward_to_arms_binds_statics, IH. reflexivity.
    + apply forward_to_arms_fwd, arm_safe_plain, Hbody.
  - apply Arm_fwd_apply.
  - apply Arm_fwd_apply_cont, arm_goes_forward.
  - apply Arm_fwd_switch. clear Hsafe.
    induction arms as [ | arm arms IH]; simpl; [ apply Forall2_nil | ].
    apply Forall2_cons; [ | exact IH ].
    split; [ reflexivity | apply arm_goes_forward ].
  - apply Arm_fwd_invalid.
Qed.

(* A term and what the pass makes of it, under the same environments
   that give the variables the kinds the pass starts from. *)
Lemma forward_to_arms_related (kinded : bool) (G : kinds_in_scope) (e : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  kinds_hold G vars ->
  arm_state_fwd kinded (State e vars conts statics hp rd)
    (State (forward_to_arms_under kinded G [] e) vars conts statics hp rd).
Proof.
  intros Hhold.
  apply (Arm_state_fwd kinded G []);
    [ apply forward_to_arms_fwd, arm_safe_none; reflexivity
    | intros k; apply Arm_same_fwd | apply arm_held_nil | exact Hhold ].
Qed.

Theorem forward_to_arms_refines : forward_to_arms_refines_statement.
Proof.
  intros e vars conts statics hp rd.
  apply (rounds_refines (arm_state_fwd false) (arm_fwd_round false)).
  apply forward_to_arms_related, kinds_hold_nil.
Qed.

(* The kinded pass that starts from some kinds is an equivalence under
   the variables that have them. *)
Lemma forward_to_arms_kinded_under_equivalent_bound (G : kinds_in_scope)
    (e : expr) :
  expr_equivalent_bound G e (forward_to_arms_under true G [] e).
Proof.
  intros vars conts statics hp rd Hhold.
  pose proof (forward_to_arms_related true G e vars conts statics hp rd Hhold)
    as Hrel.
  apply refines_antisymmetric.
  - exact (rounds_refines (arm_state_fwd true) (arm_fwd_round true) _ _ Hrel).
  - exact
      (rounds_refines (fun t' t => arm_state_fwd true t t') arm_fwd_round_back
         _ _ Hrel).
Qed.

Theorem forward_to_arms_kinded_equivalent :
  forward_to_arms_kinded_equivalent_statement.
Proof.
  intros e vars conts statics hp rd.
  exact
    (forward_to_arms_kinded_under_equivalent_bound [] e vars conts statics hp
       rd_ (kinds_hold_nil vars)).
Qed.

(* ======================================================================== *)
(* The pass in the bodies of code                                           *)
(* ======================================================================== *)

Fixpoint forward_to_arms_in_code_binds_statics (e : expr) :
  binds_statics (forward_to_arms_in_code e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - apply forward_to_arms_in_code_binds_statics.
  - rewrite forward_to_arms_in_code_binds_statics. f_equal.
    induction bindings
      as [ | [id [params my_closure ret exn metadata code_body] | id
             | sym const]
             bindings IH];
      simpl; congruence.
  - destruct h as [params metadata handler]. simpl.
    rewrite !forward_to_arms_in_code_binds_statics. reflexivity.
  - rewrite forward_to_arms_in_code_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite forward_to_arms_in_code_binds_statics, IH. reflexivity.
Qed.

(* The body of code and the body as rewritten refine each other where
   the code runs. *)
Lemma code_body_to_arms_refines (params : list variable)
    (my_closure : variable) (metadata : code_metadata) (body : expr) :
  code_refines_where always params my_closure metadata body
    (forward_to_arms_under true (code_in_scope params my_closure metadata) []
       body)
  ∧ code_refines_where always params my_closure metadata
      (forward_to_arms_under true (code_in_scope params my_closure metadata)
         [] body)
      body.
Proof.
  split; intros vs callee callee_vars conts statics hp rd _ Hkinds Hvalue Hbind;
    apply equivalent_refines; [ | apply equivalent_sym ];
    apply forward_to_arms_kinded_under_equivalent_bound,
      (code_in_scope_hold params my_closure metadata vs callee callee_vars
         Hkinds Hvalue Hbind).
Qed.

(* A group of static bindings with the bodies of its code rewritten, one
   piece of code at a time: [done] is what has been rewritten. *)
Lemma bindings_to_arms_equivalent (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
    (todo : list static_binding) :
  ∀ done : list static_binding,
    equivalent (State (Let_static (done ++ todo) body) vars conts statics hp rd)
      (State
         (Let_static (done ++ map forward_to_arms_in_binding todo) body)
         vars conts statics hp rd).
Proof.
  induction todo as [ | binding todo IH]; intros done; simpl;
    [ apply equivalent_refl | ].
  apply (equivalent_trans _
    (State
       (Let_static (done ++ forward_to_arms_in_binding binding :: todo) body)
       vars conts statics hp rd)).
  - destruct binding
      as [id [params my_closure ret exn metadata code_body] | id
         | sym const];
      simpl; try apply equivalent_refl.
    destruct (code_body_to_arms_refines params my_closure metadata code_body)
      as [Hthere Hback].
    apply refines_antisymmetric;
      apply (let_static_code_refines_where always always_extend);
      first
        [ exact I | exact Hthere | exact Hback
        | apply forward_to_arms_binds_statics
        | symmetry; apply forward_to_arms_binds_statics ].
  - specialize (IH (done ++ [forward_to_arms_in_binding binding])).
    rewrite <- !app_assoc in IH. exact IH.
Qed.

Fixpoint forward_to_arms_in_code_equivalent_aux (e : expr) :
  expr_equivalent e (forward_to_arms_in_code e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; intros vars conts statics hp rd; try apply equivalent_refl.
  - apply let_equivalent_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
    apply forward_to_arms_in_code_equivalent_aux.
  - (* down the body, then the code of the group *)
    apply (equivalent_trans _
      (State (Let_static bindings (forward_to_arms_in_code body)) vars conts
         statics hp rd)).
    + apply let_static_equivalent_step.
      destruct (eval_static_bindings vars bindings statics) as [extended | ];
        [ | exact I ].
      apply forward_to_arms_in_code_equivalent_aux.
    + exact
      (bindings_to_arms_equivalent _ vars conts statics _ rd_ bindings []).
  - apply (equivalent_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata (forward_to_arms_in_code handler)) body)
         vars conts statics hp rd)).
    + apply
        (let_cont_handler_equivalent_bound [] k params metadata handler _
           body);
        [ | intros x kind [=] ].
      intros vars' conts' statics' hp' rd' _.
      apply forward_to_arms_in_code_equivalent_aux.
    + apply let_cont_non_recursive_equivalent_step.
      apply forward_to_arms_in_code_equivalent_aux.
  - apply (equivalent_trans _
      (State
         (Let_cont_recursive
            (map (fun '(k, h) => (k, forward_to_arms_in_code_handler h))
               handlers)
            body)
         vars conts statics hp rd)).
    + apply (let_cont_recursive_handlers_equivalent_bound [] handlers _ body);
        [ | | intros x kind [=] ].
      * induction handlers as [ | [k [params metadata handler]] handlers IH];
          simpl; [ apply Forall2_nil | ].
        apply Forall2_cons; [ | exact IH ].
        split; [ reflexivity | ]. simpl.
        exists params, metadata, handler, (forward_to_arms_in_code handler).
        split; [ reflexivity | ]. split; [ reflexivity | ].
        intros vars' conts' statics' hp' rd' _.
        apply forward_to_arms_in_code_equivalent_aux.
      * unfold handlers_bind_statics.
        induction handlers as [ | [k [params metadata handler]] handlers IH];
          simpl; [ reflexivity | ].
        rewrite forward_to_arms_in_code_binds_statics, IH. reflexivity.
    + apply let_cont_recursive_equivalent_step; [ reflexivity | intros _ ].
      apply forward_to_arms_in_code_equivalent_aux.
Qed.

Theorem forward_to_arms_in_code_equivalent :
  forward_to_arms_in_code_equivalent_statement.
Proof. exact forward_to_arms_in_code_equivalent_aux. Qed.
