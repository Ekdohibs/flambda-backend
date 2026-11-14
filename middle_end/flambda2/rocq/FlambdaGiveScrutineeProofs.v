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


(* Proofs for FlambdaGiveScrutineeSpec.v. This file needs no review: what
   is to be reviewed is the statement there.

   [give_scrutinee x k i] changes an argument only where the variable
   holds the constant that the argument is, so the two runs are in
   step and pass the same values: a bisimulation, in which the term
   and the result are run under the same variables, and a continuation
   stands in the second run for the closure of the handler rewritten.
   What is known of [x] where an expression is holds of the variables
   that it runs under ([holds]).

   Some closures are not so: that of a continuation whose handler is
   rewritten for the number of the arm that applies it (arm_fact),
   when [x] is not that number. It is in scope in the [Let_cont]s
   between its own and the [Switch], whose handlers do not mention
   it, in the [Switch], and in the arm that the [Switch] takes; the
   arm taken has another number, so does not apply it. The relation
   of states has the list of such continuations ([dead]), and asks
   that two continuations be related only where the expression may
   apply them and they are not in the list; a closure asks the same
   of the continuations that its handler, or a handler bound with
   it, mentions ([closure_mentions]), so one made while such a
   continuation is in scope is related to its counterpart. The arm
   taken has its own case ([Rel_state_arm]), which asks only of the
   continuation that it applies. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextSpec FlambdaContextProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaUnusedContsSpec
  FlambdaUnaliasParamsSpec FlambdaUnaliasParamsProofs
  FlambdaGiveScrutineeSpec.
Import ListNotations.

(* ======================================================================== *)
(* What the rewrite keeps of the syntax                                     *)
(* ======================================================================== *)

Lemma give_in_apply_cont_continuation (x : variable) (k : continuation)
    (i : nat) (known : option Z) (ac : apply_cont) :
  ac_continuation (give_in_apply_cont x k i known ac) = ac_continuation ac.
Proof.
  unfold give_in_apply_cont.
  destruct known; [ destruct (Pos.eqb _ _) | ]; reflexivity.
Qed.

Fixpoint give_scrutinee_binds_statics (x : variable) (k : continuation)
    (i : nat) (known : option Z) (e : expr) {struct e} :
  binds_statics (give_scrutinee x k i known e) = binds_statics e
with give_in_handler_binds_statics (x : variable) (k : continuation)
    (i : nat) (known : option Z) (h : continuation_handler) {struct h} :
  handler_binds_statics (give_in_handler x k i known h)
  = handler_binds_statics h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [give_scrutinee binds_statics]; try reflexivity.
    + apply give_scrutinee_binds_statics.
    + rewrite give_scrutinee_binds_statics. reflexivity.
    + rewrite give_in_handler_binds_statics, give_scrutinee_binds_statics.
      reflexivity.
    + rewrite give_scrutinee_binds_statics. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map existsb].
      rewrite give_in_handler_binds_statics, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [give_in_handler handler_binds_statics].
    apply give_scrutinee_binds_statics.
Qed.

Definition give_handlers (x : variable) (k : continuation) (i : nat)
    (known : option Z) (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(k', h) => (k', give_in_handler x k i known h)) handlers.

Lemma give_handlers_bind_statics (x : variable) (k : continuation) (i : nat)
    (known : option Z)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics (give_handlers x k i known handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics, give_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map existsb]. rewrite give_in_handler_binds_statics, IH.
  reflexivity.
Qed.

Lemma lookup_give_handlers (x : variable) (k : continuation) (i : nat)
    (known : option Z) (k1 : continuation)
    (handlers : list (continuation * continuation_handler)) :
  lookup k1 (give_handlers x k i known handlers)
  = option_map (give_in_handler x k i known) (lookup k1 handlers).
Proof.
  unfold give_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map]. rewrite !lookup_cons, IH.
  destruct (Pos.eqb k1 k'); reflexivity.
Qed.

Lemma find_arm_give (x : variable) (k : continuation) (i : nat)
    (known : option Z) (scrutinee : simple) (j : Z)
    (arms : list (Z * apply_cont)) :
  find_arm j
    (map
       (fun '(d, ac) =>
          (d, give_in_apply_cont x k i (arm_known x scrutinee d known) ac))
       arms)
  = option_map (give_in_apply_cont x k i (arm_known x scrutinee j known))
      (find_arm j arms).
Proof.
  induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
  cbn [map find_arm].
  destruct (Z.eqb_spec j d) as [-> | _]; [ reflexivity | exact IH ].
Qed.

(* What [arm_fact] finds. *)
Lemma arm_fact_switch (x : variable) (k : continuation) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (c : Z) :
  arm_fact x k (Switch scrutinee arms) = Some c ->
  scrutinee = Var x ∧ only_arm k c arms = true.
Proof.
  cbn [arm_fact].
  destruct scrutinee as [y | | ]; try discriminate.
  destruct (Pos.eqb_spec y x) as [-> | _]; [ | discriminate ].
  destruct (first_arm_of k arms) as [c' | ]; [ | discriminate ].
  destruct (only_arm k c' arms) eqn:Honly; [ | discriminate ].
  intros [= <-]. split; [ reflexivity | exact Honly ].
Qed.

Lemma arm_fact_let_cont (x : variable) (k k1 : continuation)
    (hh : continuation_handler) (body : expr) (c : Z) :
  arm_fact x k (Let_cont_non_recursive k1 hh body) = Some c ->
  k <> k1 ∧ cont_free_in_handler_b k hh = false ∧ arm_fact x k body = Some c.
Proof.
  cbn [arm_fact].
  destruct (Pos.eqb_spec k1 k) as [-> | Hne]; [ discriminate | ].
  cbn [orb].
  destruct (cont_free_in_handler_b k hh); [ discriminate | ].
  intros Hfact. split; [ congruence | split; [ reflexivity | exact Hfact ] ].
Qed.

(* An arm of another number does not apply the continuation. *)
Lemma only_arm_other (k : continuation) (c : Z)
    (arms : list (Z * apply_cont)) (j : Z) (ac : apply_cont) :
  only_arm k c arms = true ->
  (j, ac) ∈ arms ->
  j <> c ->
  ac_continuation ac <> k.
Proof.
  unfold only_arm. rewrite forallb_forall. intros Hall Hin Hne Heq.
  specialize (Hall (j, ac) Hin). cbn beta iota in Hall.
  apply orb_true_iff in Hall as [Hk | Hc].
  - apply negb_true_iff, Pos.eqb_neq in Hk. exact (Hk Heq).
  - apply Z.eqb_eq in Hc. exact (Hne Hc).
Qed.

(* The continuation that an application applies is free in it. *)
Lemma ac_continuation_free (ac : apply_cont) :
  cont_free_b (ac_continuation ac) (Apply_cont ac) = true.
Proof.
  cbn [cont_free_b]. unfold apply_cont_continuations, among.
  cbn [existsb]. rewrite Pos.eqb_refl. reflexivity.
Qed.

(* A continuation free in the arm taken is free in the [Switch]. *)
Lemma arm_free_in_switch (k1 : continuation) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (j : Z) (ac : apply_cont) :
  find_arm j arms = Some ac ->
  cont_free_b k1 (Apply_cont ac) = true ->
  cont_free_b k1 (Switch scrutinee arms) = true.
Proof.
  intros Hfind Hfree. cbn [cont_free_b]. apply existsb_exists.
  exists (j, ac). split; [ exact (find_arm_in j arms ac Hfind) | exact Hfree ].
Qed.

(* ======================================================================== *)
(* Values                                                                   *)
(* ======================================================================== *)

Lemma naked_immediate_of_value_some (v : value) (j : Z) :
  naked_immediate_of_value v = Some j -> v = Value_const (Naked_immediate j).
Proof.
  destruct v; cbn [naked_immediate_of_value]; try discriminate.
  match goal with c : const |- _ => destruct c end; try discriminate.
  intros [= ->]. reflexivity.
Qed.

(* The arguments are worth what they were where the variable holds the
   constant. *)
Lemma eval_simples_give_nth (x : variable) (c : Z) (vars : env) :
  lookup x vars = Some (Value_const (Naked_immediate c)) ->
  ∀ (i : nat) (args : list simple),
    eval_simples vars (give_nth x c i args) = eval_simples vars args.
Proof.
  intros Hx. induction i as [ | i IH]; intros [ | a args]; cbn [give_nth];
    try reflexivity.
  - destruct a as [y | s | c0]; try reflexivity.
    destruct c0; try reflexivity.
    match goal with
    | |- context [Z.eqb ?c' c] => destruct (Z.eqb_spec c' c) as [-> | _]
    end; [ | reflexivity ].
    cbn [eval_simples eval_simple]. rewrite Hx. reflexivity.
  - cbn [eval_simples]. rewrite IH. reflexivity.
Qed.

(* ======================================================================== *)
(* One variable, one continuation, one position                             *)
(* ======================================================================== *)

Section Site.

Variables (x : variable) (k : continuation) (i : nat).

(* What is known of [x] holds of the variables. *)
Definition holds (known : option Z) (vars : env) : Prop :=
  ∀ c : Z,
    known = Some c -> lookup x vars = Some (Value_const (Naked_immediate c)).

Lemma eval_args_given (known : option Z) (vars : env) (ac : apply_cont) :
  holds known vars ->
  eval_simples vars (ac_args (give_in_apply_cont x k i known ac))
  = eval_simples vars (ac_args ac).
Proof.
  intros Hholds. unfold give_in_apply_cont.
  destruct known as [c | ]; [ | reflexivity ].
  destruct (Pos.eqb _ _); [ | reflexivity ].
  cbn [ac_args]. apply eval_simples_give_nth, Hholds. reflexivity.
Qed.

(* The variable holds the number, or it does not. *)
Lemma holds_or_not (c : Z) (vars : env) :
  lookup x vars = Some (Value_const (Naked_immediate c))
  ∨ lookup x vars <> Some (Value_const (Naked_immediate c)).
Proof.
  destruct (lookup x vars) as [v | ]; [ | right; discriminate ].
  destruct (naked_immediate_of_value v) as [j | ] eqn:Hj.
  - apply naked_immediate_of_value_some in Hj. subst v.
    destruct (Z.eq_dec j c) as [-> | Hne]; [ left; reflexivity | ].
    right. intros [= Heq]. exact (Hne Heq).
  - right. intros [= ->]. discriminate.
Qed.

(* The continuations that a closure may apply of those that were in
   scope where it was made: those that its handler mentions, and those
   that the handlers bound with it mention. *)
Definition closure_mentions (hh : continuation_handler)
    (recursive : list (continuation * continuation_handler))
    (k1 : continuation) : bool :=
  cont_free_in_handler_b k1 hh
  || existsb (fun '(_, h) => cont_free_in_handler_b k1 h) recursive.

(* What a continuation name stands for in the two runs: the same thing;
   or two closures of one handler, under continuations so related where
   the closure may apply them; or the closure of a handler and that of
   the handler rewritten for what is known of [x] under the variables
   of the closure; or two so related with the same extra arguments; or
   what is left of the same over-application. *)
Inductive cont_rel : cont_value -> cont_value -> Prop :=
  | Rel_same (c : cont_value) : cont_rel c c
  | Rel_plain (hh : continuation_handler) (dv : env) (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    (∀ k1 : continuation,
       closure_mentions hh recursive k1 = true ->
       cont_rel (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel (Closure hh dv dc recursive) (Closure hh dv dc' recursive)
  | Rel_given (known : option Z) (hh : continuation_handler) (dv : env)
      (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    holds known dv ->
    (∀ k1 : continuation,
       closure_mentions hh recursive k1 = true ->
       cont_rel (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel (Closure hh dv dc recursive)
      (Closure (give_in_handler x k i known hh) dv dc'
         (give_handlers x k i known recursive))
  | Rel_extra (c c' : cont_value) (extra : list value) :
    cont_rel c c' -> cont_rel (Extra_args c extra) (Extra_args c' extra)
  | Rel_over (remaining : list value) (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    cont_rel return_to return_to' ->
    cont_rel exn_to exn_to' ->
    cont_rel
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

(* Two states. An expression and the same rewritten for what is known,
   which holds, under continuations that are related where the
   expression may apply them, but for those of [dead]: each of these
   has a handler that was rewritten for a number that [x] is not, and
   is applied only by arms of that number of the [Switch] on [x] that
   the expression comes to ([arm_fact]), so it is not applied. The
   same expression. And the arm that a [Switch] takes, of which only
   the continuation that it applies need be related. *)
Inductive state_rel : state -> state -> Prop :=
  | Rel_state (dead : list (continuation * Z)) (known : option Z) (e : expr)
      (vars : env) (conts conts' : cont_env) (statics : static_env)
      (hp : heap) (rd : read_record) :
    holds known vars ->
    (∀ (k2 : continuation) (c : Z),
       (k2, c) ∈ dead ->
       arm_fact x k2 e = Some c
       ∧ lookup x vars <> Some (Value_const (Naked_immediate c))) ->
    (∀ k1 : continuation,
       cont_free_b k1 e = true ->
       k1 ∉ map fst dead ->
       cont_rel (resolve k1 conts) (resolve k1 conts')) ->
    state_rel (State e vars conts statics hp rd)
      (State (give_scrutinee x k i known e) vars conts' statics hp rd)
  | Rel_state_same (e : expr) (vars : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) :
    (∀ k1 : continuation,
       cont_free_b k1 e = true ->
       cont_rel (resolve k1 conts) (resolve k1 conts')) ->
    state_rel (State e vars conts statics hp rd)
      (State e vars conts' statics hp rd)
  | Rel_state_arm (known : option Z) (ac : apply_cont) (vars : env)
      (conts conts' : cont_env) (statics : static_env) (hp : heap)
        (rd : read_record) :
    holds known vars ->
    cont_rel (resolve (ac_continuation ac) conts)
      (resolve (ac_continuation ac) conts') ->
    state_rel (State (Apply_cont ac) vars conts statics hp rd)
      (State (Apply_cont (give_in_apply_cont x k i known ac)) vars conts'
         statics hp rd).

Lemma state_rel_refl (s : state) : state_rel s s.
Proof.
  destruct s as [e vars conts statics hp rd].
  apply Rel_state_same. intros k1 _. apply Rel_same.
Qed.

Lemma with_extra_args_rel (c c' : cont_value) (extra : list value) :
  cont_rel c c' ->
  cont_rel (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hc. destruct extra; simpl; [ exact Hc | apply Rel_extra, Hc ].
Qed.

(* No continuation is dead where the expression comes to no [Switch]
   through [Let_cont]s. *)
Lemma no_dead (dead : list (continuation * Z)) (e : expr) (vars : env) :
  (∀ k2 : continuation, arm_fact x k2 e = None) ->
  (∀ (k2 : continuation) (c : Z),
     (k2, c) ∈ dead ->
     arm_fact x k2 e = Some c
     ∧ lookup x vars <> Some (Value_const (Naked_immediate c))) ->
  dead = [].
Proof.
  intros Hnone Hdead. destruct dead as [ | [k2 c] dead]; [ reflexivity | ].
  destruct (Hdead k2 c (or_introl eq_refl)) as [Hfact _].
  rewrite Hnone in Hfact. discriminate.
Qed.

(* The closures of the handlers of a recursive [Let_cont]. *)
Lemma rec_conts_plain
    (handlers : list (continuation * continuation_handler))
    (vars : env) (conts conts' : cont_env) (k1 : continuation) :
  (∀ k2 : continuation,
     existsb (fun '(_, h) => cont_free_in_handler_b k2 h) handlers = true ->
     cont_rel (resolve k2 conts) (resolve k2 conts')) ->
  (lookup k1 handlers = None ->
   cont_rel (resolve k1 conts) (resolve k1 conts')) ->
  cont_rel (resolve k1 (rec_closures handlers vars conts ++ conts))
    (resolve k1 (rec_closures handlers vars conts' ++ conts')).
Proof.
  intros Hgroup Hout. rewrite !resolve_rec_closures.
  destruct (lookup k1 handlers) as [hh | ] eqn:Hlookup;
    [ | exact (Hout eq_refl) ].
  apply Rel_plain. intros k2 Hk2. apply Hgroup.
  unfold closure_mentions in Hk2.
  apply orb_true_iff in Hk2 as [Hk2 | Hk2]; [ | exact Hk2 ].
  apply existsb_exists. exists (k1, hh).
  split; [ exact (lookup_in k1 hh handlers Hlookup) | exact Hk2 ].
Qed.

Lemma rec_conts_given (known : option Z)
    (handlers : list (continuation * continuation_handler))
    (vars : env) (conts conts' : cont_env) (k1 : continuation) :
  holds known vars ->
  (∀ k2 : continuation,
     existsb (fun '(_, h) => cont_free_in_handler_b k2 h) handlers = true ->
     cont_rel (resolve k2 conts) (resolve k2 conts')) ->
  (lookup k1 handlers = None ->
   cont_rel (resolve k1 conts) (resolve k1 conts')) ->
  cont_rel (resolve k1 (rec_closures handlers vars conts ++ conts))
    (resolve k1
       (rec_closures (give_handlers x k i known handlers) vars conts'
        ++ conts')).
Proof.
  intros Hholds Hgroup Hout.
  rewrite !resolve_rec_closures, lookup_give_handlers.
  destruct (lookup k1 handlers) as [hh | ] eqn:Hlookup; cbn [option_map];
    [ | exact (Hout eq_refl) ].
  apply Rel_given; [ exact Hholds | ]. intros k2 Hk2. apply Hgroup.
  unfold closure_mentions in Hk2.
  apply orb_true_iff in Hk2 as [Hk2 | Hk2]; [ | exact Hk2 ].
  apply existsb_exists. exists (k1, hh).
  split; [ exact (lookup_in k1 hh handlers Hlookup) | exact Hk2 ].
Qed.

(* The same full call, returning and raising to related continuations. *)
Lemma call_code_rel (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel return_to return_to' ->
  cont_rel exn_to exn_to' ->
  result_related state_rel
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c vs args_arity return_arity return_to' exn_to'
       statics hp rd).
Proof.
  intros Hreturn Hraise.
  destruct (call_code_cases closure c vs args_arity return_arity statics hp rd)
    as [Hundefined | (callee_vars & Hcall)].
  - rewrite !Hundefined. exact I.
  - rewrite !Hcall. simpl.
    apply Rel_state_same.
    intros k' _.
    destruct (Pos.eq_dec k' (code_return_continuation c)) as [-> | Hne];
      [ rewrite !resolve_head; exact Hreturn | ].
    rewrite !(resolve_other k' (code_return_continuation c)) by exact Hne.
    destruct (Pos.eq_dec k' (code_exn_continuation c)) as [-> | Hne'];
      [ rewrite !resolve_head; exact Hraise | ].
    rewrite !(resolve_other k' (code_exn_continuation c)) by exact Hne'.
    apply Rel_same.
Qed.

Lemma run_call_rel (shape : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel return_to return_to' ->
  cont_rel exn_to exn_to' ->
  (∀ partial : value,
    result_related state_rel
      (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value return_to' [partial] statics hp rd)) ->
  result_related state_rel
    (run_call
       (fun partial => apply_cont_value return_to [partial] statics hp rd)
       shape return_arity return_to exn_to statics hp rd)
    (run_call
       (fun partial => apply_cont_value return_to' [partial] statics hp rd)
       shape return_arity return_to' exn_to' statics hp rd).
Proof.
  intros Hreturn Hraise Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - apply call_code_rel; assumption.
  - apply call_code_rel; [ apply Rel_over; assumption | exact Hraise ].
  - apply Hpartial.
  - exact I.
Qed.

(* Related continuations, applied to the same values. *)
Lemma apply_cont_value_rel (c c' : cont_value) (statics : static_env)
    (hp : heap) (rd : read_record) :
  cont_rel c c' ->
  ∀ vs : list value,
    result_related state_rel (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' vs statics hp rd).
Proof.
  intros Hc.
  induction Hc
    as [c
       | [ps m he] dv dc dc' recursive Hdc _
       | known [ps m he] dv dc dc' recursive Hholds Hdc _
       | c c' extra Hc IHc
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise _];
    intros vs.
  - apply result_related_refl, state_rel_refl.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m));
      [ | exact I ].
    destruct (bind_params ps vs dv) as [bound | ]; [ | exact I ].
    simpl. apply Rel_state_same. intros k1 Hk1. apply rec_conts_plain.
    + intros k2 Hk2. apply Hdc. unfold closure_mentions. rewrite Hk2.
      apply orb_true_r.
    + intros _. apply Hdc. unfold closure_mentions.
      cbn [cont_free_in_handler_b]. rewrite Hk1. reflexivity.
  - cbn [give_in_handler apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m));
      [ | exact I ].
    destruct (bind_params ps vs dv) as [bound | ] eqn:Hbound; [ | exact I ].
    cbn [result_related].
    apply (Rel_state [] (known_under x ps known)).
    + unfold known_under. destruct (among x ps) eqn:Hx; intros c Hc;
        [ discriminate | ].
      rewrite (lookup_bind_params_out ps vs dv bound x Hx Hbound).
      exact (Hholds c Hc).
    + intros k2 c [].
    + intros k1 Hk1 _. apply rec_conts_given; [ exact Hholds | | ].
      * intros k2 Hk2. apply Hdc. unfold closure_mentions. rewrite Hk2.
        apply orb_true_r.
      * intros _. apply Hdc. unfold closure_mentions.
        cbn [cont_free_in_handler_b]. rewrite Hk1. reflexivity.
  - exact (IHc (vs ++ extra)).
  - rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; try exact I.
    unfold call_function.
    apply run_call_rel; [ exact Hreturn | exact Hraise | ].
    intros partial. exact (IHreturn [partial]).
Qed.

Lemma call_function_rel (callee : value) (call_kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel return_to return_to' ->
  cont_rel exn_to exn_to' ->
  result_related state_rel
    (call_function callee call_kind vs args_arity return_arity return_to
       exn_to statics hp rd)
    (call_function callee call_kind vs args_arity return_arity return_to'
       exn_to' statics hp rd).
Proof.
  intros Hreturn Hraise. unfold call_function.
  apply run_call_rel; [ exact Hreturn | exact Hraise | ].
  intros partial. apply apply_cont_value_rel, Hreturn.
Qed.

(* A closure of the handler of a non-recursive [Let_cont], rewritten
   for something that holds, is related to the closure of the handler
   where the continuations that the handler mentions are. *)
Lemma closure_given (known : option Z) (hh : continuation_handler)
    (vars : env) (conts conts' : cont_env) :
  holds known vars ->
  (∀ k1 : continuation,
     cont_free_in_handler_b k1 hh = true ->
     cont_rel (resolve k1 conts) (resolve k1 conts')) ->
  cont_rel (Closure hh vars conts [])
    (Closure (give_in_handler x k i known hh) vars conts' []).
Proof.
  intros Hholds Hconts.
  apply (Rel_given known hh vars conts conts' []); [ exact Hholds | ].
  intros k1 Hk1. apply Hconts. unfold closure_mentions in Hk1.
  cbn [existsb] in Hk1. rewrite orb_false_r in Hk1. exact Hk1.
Qed.

(* Two related states step together. *)
Lemma give_step (s s' : state) :
  state_rel s s' -> result_related state_rel (step s) (step s').
Proof.
  intros
    [dead known e vars conts conts' statics hp rd Hholds Hdead Hconts
    | e vars conts conts' statics hp rd Hconts
    | known ac vars conts conts' statics hp rd Hholds Hk].
  - (* an expression and the same rewritten *)
    destruct e as [y kind1 d body | bindings body | k1 hh body
                  | handlers body | ap | ac | scrutinee arms | ];
      cbn [give_scrutinee].
    + assert (Hnil : dead = [])
        by (apply (no_dead dead (Let y kind1 d body) vars);
            [ intros k2; reflexivity | exact Hdead ]).
      subst dead.
      rewrite !step_let_exec.
      destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | exact I ].
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      cbn [result_related]. apply (Rel_state []).
      * intros c Hc. destruct (Pos.eqb_spec y x) as [-> | Hyx];
          [ discriminate | ].
        rewrite lookup_other by congruence. exact (Hholds c Hc).
      * intros k2 c [].
      * intros k1 Hk1 _. apply Hconts; [ exact Hk1 | intros [] ].
    + assert (Hnil : dead = [])
        by (apply (no_dead dead (Let_static bindings body) vars);
            [ intros k2; reflexivity | exact Hdead ]).
      subst dead.
      rewrite !step_let_static.
      destruct (eval_static_bindings vars bindings statics); [ | exact I ].
      cbn [result_related]. apply (Rel_state []); [ exact Hholds | | ].
      * intros k2 c [].
      * intros k1 Hk1 _. apply Hconts; [ exact Hk1 | intros [] ].
    + rewrite !step_let_cont_non_recursive. cbn [result_related].
      (* what is dead stays so in the body, and the handler does not
         mention it *)
      assert (Hbody :
        ∀ (k2 : continuation) (c : Z),
          (k2, c) ∈ dead ->
          k2 <> k1 ∧ cont_free_in_handler_b k2 hh = false
          ∧ arm_fact x k2 body = Some c
          ∧ lookup x vars <> Some (Value_const (Naked_immediate c))).
      { intros k2 c Hin. destruct (Hdead k2 c Hin) as [Hfact Hnot].
        destruct (arm_fact_let_cont x k2 k1 hh body c Hfact)
          as (Hne & Hfree & Hfact').
        repeat split; assumption. }
      assert (Hbody_dead :
        ∀ (k2 : continuation) (c : Z),
          (k2, c) ∈ dead ->
          arm_fact x k2 body = Some c
          ∧ lookup x vars <> Some (Value_const (Naked_immediate c))).
      { intros k2 c Hin. destruct (Hbody k2 c Hin) as (_ & _ & Hf & Hn).
        split; assumption. }
      (* the continuations that the handler mentions are related *)
      assert (Hhandler :
        ∀ k3 : continuation,
          cont_free_in_handler_b k3 hh = true ->
          cont_rel (resolve k3 conts) (resolve k3 conts')).
      { intros k3 Hk3. apply Hconts.
        - cbn [cont_free_b]. rewrite Hk3. reflexivity.
        - intros Hin. apply in_map_iff in Hin as ([k2 c2] & Heq & Hin).
          cbn [fst] in Heq. subst k2.
          destruct (Hbody k3 c2 Hin) as (_ & Hfree & _). congruence. }
      (* those of the body, but for the one bound, are related *)
      assert (Hrest :
        ∀ k3 : continuation,
          cont_free_b k3 body = true ->
          k3 <> k1 ->
          k3 ∉ map fst dead ->
          cont_rel (resolve k3 ((k1, Closure hh vars conts []) :: conts))
            (resolve k3
               ((k1,
                 Closure
                   (give_in_handler x k i
                      match arm_fact x k1 body with
                      | Some c => Some c
                      | None => known
                      end hh) vars conts' []) :: conts'))).
      { intros k3 Hk3 Hne Hk3dead.
        rewrite !(resolve_other k3 k1) by exact Hne.
        apply Hconts; [ | exact Hk3dead ].
        cbn [cont_free_b]. rewrite Hk3.
        apply Pos.eqb_neq in Hne. rewrite Hne. apply orb_true_r. }
      destruct (arm_fact x k1 body) as [c | ] eqn:Hfact.
      * destruct (holds_or_not c vars) as [Hc | Hnot].
        -- apply (Rel_state dead known body);
             [ exact Hholds | exact Hbody_dead | ].
           intros k3 Hk3 Hk3dead.
           destruct (Pos.eq_dec k3 k1) as [-> | Hne];
             [ | exact (Hrest k3 Hk3 Hne Hk3dead) ].
           rewrite !resolve_head.
           apply closure_given; [ | exact Hhandler ].
           intros c' [= <-]. exact Hc.
        -- apply (Rel_state ((k1, c) :: dead) known body);
             [ exact Hholds | | ].
           ++ intros k2 c2 [[= <- <-] | Hin];
                [ split; assumption | exact (Hbody_dead k2 c2 Hin) ].
           ++ intros k3 Hk3 Hk3dead. cbn [map fst] in Hk3dead.
              apply Hrest; [ exact Hk3 | | ].
              ** intros ->. apply Hk3dead. left. reflexivity.
              ** intros Hin. apply Hk3dead. right. exact Hin.
      * apply (Rel_state dead known body);
          [ exact Hholds | exact Hbody_dead | ].
        intros k3 Hk3 Hk3dead.
        destruct (Pos.eq_dec k3 k1) as [-> | Hne];
          [ | exact (Hrest k3 Hk3 Hne Hk3dead) ].
        rewrite !resolve_head.
        apply closure_given; [ exact Hholds | exact Hhandler ].
    + assert (Hnil : dead = [])
        by (apply (no_dead dead (Let_cont_recursive handlers body) vars);
            [ intros k2; reflexivity | exact Hdead ]).
      subst dead.
      rewrite !step_let_cont_recursive.
      fold (give_handlers x k i known handlers).
      rewrite give_handlers_bind_statics.
      destruct (handlers_bind_statics handlers); [ exact I | ].
      cbn [result_related]. apply (Rel_state [] known body);
        [ exact Hholds | | ].
      * intros k2 c [].
      * intros k3 Hk3 _. apply rec_conts_given; [ exact Hholds | | ].
        -- intros k2 Hk2. apply Hconts; [ | intros [] ].
           cbn [cont_free_b]. rewrite Hk2. reflexivity.
        -- intros _. apply Hconts; [ | intros [] ].
           cbn [cont_free_b]. rewrite Hk3. apply orb_true_r.
    + assert (Hnil : dead = [])
        by (apply (no_dead dead (Apply ap) vars);
            [ intros k2; reflexivity | exact Hdead ]).
      subst dead.
      rewrite !step_apply.
      destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
      destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
      destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
        [ | exact I ].
      apply call_function_rel;
        [ apply Hconts | apply with_extra_args_rel, Hconts ];
        try (intros []); cbn [cont_free_b]; rewrite Pos.eqb_refl;
        [ reflexivity | apply orb_true_r ].
    + assert (Hnil : dead = [])
        by (apply (no_dead dead (Apply_cont ac) vars);
            [ intros k2; reflexivity | exact Hdead ]).
      subst dead.
      rewrite !step_apply_cont, (eval_args_given known vars ac Hholds),
        give_in_apply_cont_continuation.
      destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
      unfold apply_continuation.
      apply apply_cont_value_rel, Hconts;
        [ apply ac_continuation_free | intros [] ].
    + rewrite !step_switch.
      destruct (eval_simple vars scrutinee) as [v | ] eqn:Hv; [ | exact I ].
      destruct (naked_immediate_of_value v) as [j | ] eqn:Hj; [ | exact I ].
      rewrite find_arm_give.
      destruct (find_arm j arms) as [ac | ] eqn:Hfind; [ | exact I ].
      cbn [option_map result_related].
      apply naked_immediate_of_value_some in Hj. subst v.
      apply (Rel_state_arm (arm_known x scrutinee j known) ac).
      * unfold arm_known. destruct scrutinee as [y | sym | cst];
          try exact Hholds.
        destruct (Pos.eqb_spec y x) as [-> | _]; [ | exact Hholds ].
        intros c' [= <-]. cbn [eval_simple] in Hv. exact Hv.
      * apply Hconts.
        -- exact (arm_free_in_switch _ scrutinee arms j ac Hfind
             (ac_continuation_free ac)).
        -- intros Hin. apply in_map_iff in Hin as ([k2 c] & Heq & Hin).
           cbn [fst] in Heq. subst k2.
           destruct (Hdead _ c Hin) as [Hfact Hnot].
           destruct (arm_fact_switch x _ scrutinee arms c Hfact)
             as [-> Honly].
           cbn [eval_simple] in Hv.
           assert (Hjc : j <> c) by (intros ->; exact (Hnot Hv)).
           exact (only_arm_other _ c arms j ac Honly
             (find_arm_in j arms ac Hfind) Hjc eq_refl).
    + rewrite !step_invalid. exact I.
  - (* the same expression *)
    destruct e as [y kind1 d body | bindings body | k1 hh body
                  | handlers body | ap | ac | scrutinee arms | ].
    + rewrite !step_let_exec.
      destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | exact I ].
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      cbn [result_related]. apply Rel_state_same. exact Hconts.
    + rewrite !step_let_static.
      destruct (eval_static_bindings vars bindings statics); [ | exact I ].
      cbn [result_related]. apply Rel_state_same. exact Hconts.
    + rewrite !step_let_cont_non_recursive. cbn [result_related].
      apply Rel_state_same. intros k3 Hk3.
      destruct (Pos.eq_dec k3 k1) as [-> | Hne].
      * rewrite !resolve_head. apply Rel_plain. intros k4 Hk4.
        apply Hconts. unfold closure_mentions in Hk4.
        cbn [existsb] in Hk4. rewrite orb_false_r in Hk4.
        cbn [cont_free_b]. rewrite Hk4. reflexivity.
      * rewrite !(resolve_other k3 k1) by exact Hne. apply Hconts.
        cbn [cont_free_b]. rewrite Hk3.
        apply Pos.eqb_neq in Hne. rewrite Hne. apply orb_true_r.
    + rewrite !step_let_cont_recursive.
      destruct (handlers_bind_statics handlers); [ exact I | ].
      cbn [result_related]. apply Rel_state_same. intros k3 Hk3.
      apply rec_conts_plain.
      * intros k2 Hk2. apply Hconts. cbn [cont_free_b]. rewrite Hk2.
        reflexivity.
      * intros _. apply Hconts. cbn [cont_free_b]. rewrite Hk3.
        apply orb_true_r.
    + rewrite !step_apply.
      destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
      destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
      destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
        [ | exact I ].
      apply call_function_rel;
        [ apply Hconts | apply with_extra_args_rel, Hconts ];
        cbn [cont_free_b]; rewrite Pos.eqb_refl;
        [ reflexivity | apply orb_true_r ].
    + rewrite !step_apply_cont.
      destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
      unfold apply_continuation.
      apply apply_cont_value_rel, Hconts, ac_continuation_free.
    + rewrite !step_switch.
      destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
      destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
      destruct (find_arm j arms) as [ac | ] eqn:Hfind; [ | exact I ].
      cbn [result_related]. apply Rel_state_same. intros k1 Hk1.
      apply Hconts.
      exact (arm_free_in_switch k1 scrutinee arms j ac Hfind Hk1).
    + rewrite !step_invalid. exact I.
  - (* the arm taken *)
    rewrite !step_apply_cont, (eval_args_given known vars ac Hholds),
      give_in_apply_cont_continuation.
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
    unfold apply_continuation.
    apply apply_cont_value_rel. exact Hk.
Qed.

(* The rewrite gives an equivalent term, whatever is given. *)
Lemma give_scrutinee_equivalent (e : expr) :
  expr_equivalent e (give_scrutinee x k i None e).
Proof.
  intros vars conts statics hp rd.
  apply (bisimulation_equivalent state_rel give_step).
  apply (Rel_state [] None); [ intros c; discriminate | intros k2 c [] | ].
  intros k1 _ _. apply Rel_same.
Qed.

End Site.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

Lemma give_at_equivalent (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr) :
  expr_equivalent (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h (give_at i k h body)).
Proof.
  unfold give_at.
  destruct (first_scrutinee k i [] body) as [xs | ];
    [ | intros vars conts statics hp rd; apply equivalent_refl ].
  induction xs as [ | x xs IH]; cbn [give_first];
    [ intros vars conts statics hp rd; apply equivalent_refl | ].
  destruct (unalias_param i k h (give_scrutinee x k i None body));
    [ | exact IH ].
  exact (plug_equivalent (Let_cont_non_recursive_body k h Hole) _ _
    (not_many _) (give_scrutinee_equivalent x k i body)).
Qed.

Lemma give_from_equivalent (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    expr_equivalent (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k h (give_from i k h body)).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [give_from].
  - intros vars conts statics hp rd. apply equivalent_refl.
  - exact (expr_equivalent_trans _ _ _ (give_at_equivalent i k h body)
      (IH k h (give_at i k h body))).
Qed.

(* Each layer is a context around what the induction gives; then the
   parameters of the [Let_cont] are looked at. *)
Fixpoint give_scrutinees_equivalent_aux (e : expr) :
  expr_equivalent e (give_scrutinees e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [give_scrutinees];
    try (intros vars conts statics hp rd; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (give_scrutinees_equivalent_aux body)).
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (give_scrutinees_equivalent_aux body)).
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (give_scrutinees handler)) body)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole body) _ _
        (not_many _) (give_scrutinees_equivalent_aux handler)).
    + apply (expr_equivalent_trans _
        (Let_cont_non_recursive k
           (Handler params metadata (give_scrutinees handler))
           (give_scrutinees body))).
      * exact (plug_equivalent
          (Let_cont_non_recursive_body k
             (Handler params metadata (give_scrutinees handler)) Hole)
          _ _ (not_many _) (give_scrutinees_equivalent_aux body)).
      * apply give_from_equivalent.
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole) _ _
      (not_many _) (give_scrutinees_equivalent_aux body)).
Qed.

Theorem give_scrutinees_equivalent : give_scrutinees_equivalent_statement.
Proof. exact give_scrutinees_equivalent_aux. Qed.
