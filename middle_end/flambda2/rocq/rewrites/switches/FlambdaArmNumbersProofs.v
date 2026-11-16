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



(* Proofs for FlambdaArmNumbersSpec.v. This file needs no review: what
   is to be reviewed is the statement there.

   [number_in_arms x] changes an argument only where it is the
   variable and the variable holds the number that is put for it, so
   the two runs are in step and pass the same values: a bisimulation,
   in which the term and the result are run under the same variables,
   and a continuation stands in the second run for the closure of the
   handler rewritten. What is known of [x] where an expression is
   holds of the variables that it runs under ([holds]).

   Some closures are not so: that of a continuation whose handler is
   rewritten for the number of the arm that applies it
   (FlambdaGiveScrutineeSpec.arm_fact), when [x] is not that number.
   It is in scope in the [Let_cont]s between its own and the
   [Switch], whose handlers do not mention it, in the [Switch], and
   in the arm that the [Switch] takes; the arm taken has another
   number, so does not apply it. The relation of states has the list
   of such continuations ([dead]), and asks that two continuations be
   related only where the expression may apply them and they are not
   in the list; a closure asks the same of the continuations that its
   handler, or a handler bound with it, mentions
   ([closure_mentions]). The arm taken has its own case
   ([Rel_state_arm]), which asks only of the continuation that it
   applies.

   The proof has the shape of that of FlambdaGiveScrutineeProofs.v,
   whose rewrite goes the other way at one position of one
   continuation. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaPrimitiveModel FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaUnusedContsSpec
  FlambdaUnaliasParamsSpec FlambdaUnaliasParamsProofs
  FlambdaUntagParamsSpec FlambdaUntagParamsProofs
  FlambdaGiveScrutineeSpec FlambdaArmNumbersSpec.
From OxCamlFlambda2 Require FlambdaGiveScrutineeProofs.
Import ListNotations.

(* Lemmas of the proofs of the rewrite that goes the other way: what
   [arm_fact] finds, that an arm of another number does not apply the
   continuation, which continuations are free in an application and
   in a [Switch], and the value that a [Switch] reads. *)
Local Definition arm_fact_switch :=
  FlambdaGiveScrutineeProofs.arm_fact_switch.
Local Definition arm_fact_let_cont :=
  FlambdaGiveScrutineeProofs.arm_fact_let_cont.
Local Definition only_arm_other := FlambdaGiveScrutineeProofs.only_arm_other.
Local Definition ac_continuation_free :=
  FlambdaGiveScrutineeProofs.ac_continuation_free.
Local Definition arm_free_in_switch :=
  FlambdaGiveScrutineeProofs.arm_free_in_switch.
Local Definition naked_immediate_of_value_some :=
  FlambdaGiveScrutineeProofs.naked_immediate_of_value_some.

(* ======================================================================== *)
(* What the rewrite keeps of the syntax                                     *)
(* ======================================================================== *)

Lemma number_in_apply_cont_continuation (x : variable) (known : option Z)
    (ac : apply_cont) :
  ac_continuation (number_in_apply_cont x known ac) = ac_continuation ac.
Proof. unfold number_in_apply_cont. destruct known; reflexivity. Qed.

Fixpoint number_in_region_binds_statics (x : variable)
    (source : option variable) (known : option Z) (e : expr) {struct e} :
  binds_statics (number_in_region x source known e) = binds_statics e
with number_in_handler_binds_statics (x : variable)
    (source : option variable) (known : option Z) (h : continuation_handler)
    {struct h} :
  handler_binds_statics (number_in_handler x source known h)
  = handler_binds_statics h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [number_in_region binds_statics]; try reflexivity.
    + apply number_in_region_binds_statics.
    + rewrite number_in_region_binds_statics. reflexivity.
    + rewrite number_in_handler_binds_statics, number_in_region_binds_statics.
      reflexivity.
    + rewrite number_in_region_binds_statics. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map existsb].
      rewrite number_in_handler_binds_statics, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [number_in_handler handler_binds_statics].
    apply number_in_region_binds_statics.
Qed.

Definition number_handlers (x : variable) (source : option variable)
    (known : option Z) (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(k', h) => (k', number_in_handler x source known h)) handlers.

Lemma number_handlers_bind_statics (x : variable)
    (source : option variable) (known : option Z)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics (number_handlers x source known handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics, number_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map existsb]. rewrite number_in_handler_binds_statics, IH.
  reflexivity.
Qed.

Lemma lookup_number_handlers (x : variable) (source : option variable)
    (known : option Z) (k1 : continuation)
    (handlers : list (continuation * continuation_handler)) :
  lookup k1 (number_handlers x source known handlers)
  = option_map (number_in_handler x source known) (lookup k1 handlers).
Proof.
  unfold number_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map]. rewrite !lookup_cons, IH.
  destruct (Pos.eqb k1 k'); reflexivity.
Qed.

Lemma find_arm_number (x : variable)
    (known : option Z) (scrutinee : simple) (j : Z)
    (arms : list (Z * apply_cont)) :
  find_arm j
    (map
       (fun '(d, ac) =>
          (d, number_in_apply_cont x (arm_known x scrutinee d known) ac))
       arms)
  = option_map (number_in_apply_cont x (arm_known x scrutinee j known))
      (find_arm j arms).
Proof.
  induction arms as [ | [d ac] arms IH]; [ reflexivity | ].
  cbn [map find_arm].
  destruct (Z.eqb_spec j d) as [-> | _]; [ reflexivity | exact IH ].
Qed.

(* ======================================================================== *)
(* Values                                                                   *)
(* ======================================================================== *)

(* The arguments are worth what they were where the variable holds the
   number. *)
Lemma eval_simples_number (x : variable) (c : Z) (vars : env) :
  lookup x vars = Some (Value_const (Naked_immediate c)) ->
  ∀ args : list simple,
    eval_simples vars (map (number_simple x c) args) = eval_simples vars args.
Proof.
  intros Hx. induction args as [ | a args IH]; [ reflexivity | ].
  cbn [map eval_simples]. rewrite IH. f_equal.
  destruct a as [y | s | c0]; try reflexivity.
  cbn [number_simple]. destruct (Pos.eqb_spec y x) as [-> | _];
    [ | reflexivity ].
  cbn [eval_simple]. rewrite Hx. reflexivity.
Qed.

(* A variable read is worth what it was where the variable holds the
   number, and the one that it is the untagging of the tagged
   number. *)
Lemma eval_simple_read (x : variable) (source : option variable) (c : Z)
    (vars : env) (s : simple) :
  lookup x vars = Some (Value_const (Naked_immediate c)) ->
  (∀ t : variable,
     source = Some t ->
     lookup t vars = Some (Value_const (Tagged_immediate c))) ->
  eval_simple vars (number_read x source c s) = eval_simple vars s.
Proof.
  intros Hx Ht. destruct s as [y | sym | c0]; try reflexivity.
  cbn [number_read]. destruct (Pos.eqb_spec y x) as [-> | _].
  - cbn [eval_simple]. rewrite Hx. reflexivity.
  - destruct source as [t | ]; [ | reflexivity ].
    destruct (Pos.eqb_spec y t) as [-> | _]; [ | reflexivity ].
    cbn [eval_simple]. rewrite (Ht t eq_refl). reflexivity.
Qed.

Lemma eval_simples_read (x : variable) (source : option variable) (c : Z)
    (vars : env) :
  lookup x vars = Some (Value_const (Naked_immediate c)) ->
  (∀ t : variable,
     source = Some t ->
     lookup t vars = Some (Value_const (Tagged_immediate c))) ->
  ∀ args : list simple,
    eval_simples vars (map (number_read x source c) args)
    = eval_simples vars args.
Proof.
  intros Hx Ht. induction args as [ | a args IH]; [ reflexivity | ].
  cbn [map eval_simples].
  rewrite IH, (eval_simple_read x source c vars a Hx Ht). reflexivity.
Qed.

(* What the untagging of a variable gives, and what the variable
   holds then. *)
Lemma eval_untag_var (statics : static_env) (vars : env) (t : variable)
    (v : value) :
  eval_named statics vars (Prim (Unary Untag_immediate) [Var t]) = Some v ->
  ∃ i : Z,
    lookup t vars = Some (Value_const (Tagged_immediate i))
    ∧ v = Value_const (Naked_immediate i).
Proof.
  cbn [eval_named eval_simples eval_simple].
  destruct (lookup t vars) as [w | ]; [ | discriminate ].
  simpl. rewrite eval_prim_unary. cbn [eval_unary_primitive].
  destruct (int_of_value Standard_tagged_immediate w) as [i | ] eqn:Hint;
    [ | discriminate ].
  intros [= <-]. exists i.
  apply int_of_value_const in Hint as [-> _]. split; reflexivity.
Qed.

(* ======================================================================== *)
(* One variable                                                             *)
(* ======================================================================== *)

Section Site.

Variables (x : variable).

(* What is known of [x] holds of the variables. *)
Definition holds (known : option Z) (vars : env) : Prop :=
  ∀ c : Z,
    known = Some c -> lookup x vars = Some (Value_const (Naked_immediate c)).

(* The variable that [x] is said to be the untagging of holds the
   tagged immediate of the number that [x] holds. *)
Definition sourced (source : option variable) (vars : env) : Prop :=
  ∀ t : variable,
    source = Some t ->
    ∃ i : Z,
      lookup t vars = Some (Value_const (Tagged_immediate i))
      ∧ lookup x vars = Some (Value_const (Naked_immediate i)).

Lemma sourced_none (vars : env) : sourced None vars.
Proof. intros t [=]. Qed.

Lemma sourced_tagged (source : option variable) (c : Z) (vars : env) :
  sourced source vars ->
  lookup x vars = Some (Value_const (Naked_immediate c)) ->
  ∀ t : variable,
    source = Some t ->
    lookup t vars = Some (Value_const (Tagged_immediate c)).
Proof.
  intros Hsourced Hx t Ht. destruct (Hsourced t Ht) as (i & Hti & Hxi).
  rewrite Hx in Hxi. injection Hxi as ->. exact Hti.
Qed.

(* A defining expression and the scrutinee of a [Switch] are worth what
   they were. *)
Lemma eval_named_number (source : option variable) (known : option Z)
    (statics : static_env) (vars : env) (d : named) :
  holds known vars ->
  sourced source vars ->
  eval_named statics vars (number_in_named x source known d)
  = eval_named statics vars d.
Proof.
  intros Hholds Hsourced. unfold number_in_named.
  destruct known as [c | ]; [ | reflexivity ].
  pose proof (Hholds c eq_refl) as Hx.
  pose proof (sourced_tagged source c vars Hsourced Hx) as Ht.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  - cbn [eval_named]. apply eval_simple_read; assumption.
  - cbn [eval_named]. rewrite eval_simples_read by assumption. reflexivity.
Qed.

(* And it runs as it did: the rewrite of a primitive keeps the
   primitive and what its arguments are worth. *)
Lemma exec_named_number (source : option variable) (known : option Z)
    (statics : static_env) (hp : heap) (rd : read_record) (vars : env)
    (d : named) :
  holds known vars ->
  sourced source vars ->
  exec_named statics hp rd vars (number_in_named x source known d)
  = exec_named statics hp rd vars d.
Proof.
  intros Hholds Hsourced. apply exec_named_agrees.
  - exact (eval_named_number source known statics vars d Hholds Hsourced).
  - intros p args ->. unfold number_in_named.
    destruct known as [c | ].
    + pose proof (Hholds c eq_refl) as Hx.
      pose proof (sourced_tagged source c vars Hsourced Hx) as Ht.
      eexists. split; [ reflexivity | ].
      split; [ exact (eval_simples_read x source c vars Hx Ht args) | ].
      intros vs s _ _. reflexivity.
    + exists args. split; [ reflexivity | ].
      split; [ reflexivity | ]. intros vs s _ _. reflexivity.
  - intros p args Hd. unfold number_in_named in Hd.
    destruct known as [c | ];
      destruct d as [s | p0 args0 | | | | | ]; try discriminate Hd;
      exists p0, args0; reflexivity.
Qed.

Lemma eval_scrutinee_number (source : option variable) (known : option Z)
    (vars : env) (scrutinee : simple) :
  holds known vars ->
  sourced source vars ->
  eval_simple vars (number_in_scrutinee x source known scrutinee)
  = eval_simple vars scrutinee.
Proof.
  intros Hholds Hsourced. unfold number_in_scrutinee.
  destruct known as [c | ]; [ | reflexivity ].
  pose proof (Hholds c eq_refl) as Hx.
  exact
    (eval_simple_read x source c vars scrutinee Hx
       (sourced_tagged source c vars Hsourced Hx)).
Qed.

(* The variable that [x] is the untagging of under a [Let], and in a
   handler. *)
Lemma sourced_let (source : option variable) (statics : static_env)
    (hp hp' : heap) (rd rd' : read_record) (vars : env) (y : variable)
    (d : named) (v : value) :
  sourced source vars ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  sourced (source_under_let x y d source) ((y, v) :: vars).
Proof.
  intros Hsourced Hv t Ht. unfold source_under_let in Ht.
  destruct (Pos.eqb_spec y x) as [-> | Hyx].
  - destruct (untag_of d) as [t' | ] eqn:Huntag; [ | discriminate Ht ].
    destruct (Pos.eqb_spec t' x) as [-> | Hne]; [ discriminate Ht | ].
    injection Ht as <-. apply untag_of_some in Huntag. subst d.
    (* the untagging reads no cell *)
    apply exec_named_pure in Hv as (Hv & _ & _); [ | reflexivity ].
    destruct (eval_untag_var statics vars t' v Hv) as (i & Ht' & ->).
    exists i. rewrite !lookup_cons, Pos.eqb_refl.
    apply Pos.eqb_neq in Hne. rewrite Hne. split; [ exact Ht' | reflexivity ].
  - destruct source as [t' | ]; [ | discriminate Ht ].
    destruct (Pos.eqb_spec y t') as [-> | Hyt]; [ discriminate Ht | ].
    injection Ht as <-.
    destruct (Hsourced t' eq_refl) as (i & Ht' & Hx).
    exists i. rewrite !lookup_cons.
    assert (Hty : Pos.eqb t' y = false) by (apply Pos.eqb_neq; congruence).
    assert (Hxy : Pos.eqb x y = false) by (apply Pos.eqb_neq; congruence).
    rewrite Hty, Hxy. split; assumption.
Qed.

Lemma sourced_params (source : option variable) (params : list variable)
    (vs : list value) (vars bound : env) :
  sourced source vars ->
  bind_params params vs vars = Some bound ->
  sourced (source_under x params source) bound.
Proof.
  intros Hsourced Hbound t Ht. unfold source_under in Ht.
  destruct (among x params) eqn:Hx; [ discriminate Ht | ].
  destruct source as [t' | ]; [ | discriminate Ht ].
  destruct (among t' params) eqn:Ht'; [ discriminate Ht | ].
  injection Ht as <-.
  destruct (Hsourced t' eq_refl) as (i & Hti & Hxi).
  exists i.
  rewrite (lookup_bind_params_out params vs vars bound t' Ht' Hbound),
    (lookup_bind_params_out params vs vars bound x Hx Hbound).
  split; assumption.
Qed.

Lemma eval_args_given (known : option Z) (vars : env) (ac : apply_cont) :
  holds known vars ->
  eval_simples vars (ac_args (number_in_apply_cont x known ac))
  = eval_simples vars (ac_args ac).
Proof.
  intros Hholds. unfold number_in_apply_cont.
  destruct known as [c | ]; [ | reflexivity ].
  cbn [ac_args]. apply eval_simples_number, Hholds. reflexivity.
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
  | Rel_given (source : option variable) (known : option Z)
      (hh : continuation_handler) (dv : env) (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    holds known dv ->
    sourced source dv ->
    (∀ k1 : continuation,
       closure_mentions hh recursive k1 = true ->
       cont_rel (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel (Closure hh dv dc recursive)
      (Closure (number_in_handler x source known hh) dv dc'
         (number_handlers x source known recursive))
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
  | Rel_state (dead : list (continuation * Z)) (source : option variable)
      (known : option Z) (e : expr) (vars : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) :
    holds known vars ->
    sourced source vars ->
    (∀ (k2 : continuation) (c : Z),
       (k2, c) ∈ dead ->
       arm_fact x k2 e = Some c
       ∧ lookup x vars <> Some (Value_const (Naked_immediate c))) ->
    (∀ k1 : continuation,
       cont_free_b k1 e = true ->
       k1 ∉ map fst dead ->
       cont_rel (resolve k1 conts) (resolve k1 conts')) ->
    state_rel (State e vars conts statics hp rd)
      (State (number_in_region x source known e) vars conts' statics hp rd)
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
      (State (Apply_cont (number_in_apply_cont x known ac)) vars conts'
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

Lemma rec_conts_given (source : option variable) (known : option Z)
    (handlers : list (continuation * continuation_handler))
    (vars : env) (conts conts' : cont_env) (k1 : continuation) :
  holds known vars ->
  sourced source vars ->
  (∀ k2 : continuation,
     existsb (fun '(_, h) => cont_free_in_handler_b k2 h) handlers = true ->
     cont_rel (resolve k2 conts) (resolve k2 conts')) ->
  (lookup k1 handlers = None ->
   cont_rel (resolve k1 conts) (resolve k1 conts')) ->
  cont_rel (resolve k1 (rec_closures handlers vars conts ++ conts))
    (resolve k1
       (rec_closures (number_handlers x source known handlers) vars conts'
        ++ conts')).
Proof.
  intros Hholds Hsourced Hgroup Hout.
  rewrite !resolve_rec_closures, lookup_number_handlers.
  destruct (lookup k1 handlers) as [hh | ] eqn:Hlookup; cbn [option_map];
    [ | exact (Hout eq_refl) ].
  apply Rel_given; [ exact Hholds | exact Hsourced | ].
  intros k2 Hk2. apply Hgroup.
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
       | source known [ps m he] dv dc dc' recursive Hholds Hsourced Hdc _
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
  - cbn [number_in_handler apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m));
      [ | exact I ].
    destruct (bind_params ps vs dv) as [bound | ] eqn:Hbound; [ | exact I ].
    cbn [result_related].
    apply (Rel_state [] (source_under x ps source) (known_under x ps known)).
    + unfold known_under. destruct (among x ps) eqn:Hx; intros c Hc;
        [ discriminate | ].
      rewrite (lookup_bind_params_out ps vs dv bound x Hx Hbound).
      exact (Hholds c Hc).
    + exact (sourced_params source ps vs dv bound Hsourced Hbound).
    + intros k2 c [].
    + intros k1 Hk1 _.
      apply rec_conts_given; [ exact Hholds | exact Hsourced | | ].
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
Lemma closure_given (source : option variable) (known : option Z)
    (hh : continuation_handler) (vars : env) (conts conts' : cont_env) :
  holds known vars ->
  sourced source vars ->
  (∀ k1 : continuation,
     cont_free_in_handler_b k1 hh = true ->
     cont_rel (resolve k1 conts) (resolve k1 conts')) ->
  cont_rel (Closure hh vars conts [])
    (Closure (number_in_handler x source known hh) vars conts' []).
Proof.
  intros Hholds Hsourced Hconts.
  apply (Rel_given source known hh vars conts conts' []);
    [ exact Hholds | exact Hsourced | ].
  intros k1 Hk1. apply Hconts. unfold closure_mentions in Hk1.
  cbn [existsb] in Hk1. rewrite orb_false_r in Hk1. exact Hk1.
Qed.

(* Two related states step together. *)
Lemma number_step (s s' : state) :
  state_rel s s' -> result_related state_rel (step s) (step s').
Proof.
  intros
    [dead source known e vars conts conts' statics hp rd Hholds Hsourced
       Hdead Hconts
    | e vars conts conts' statics hp rd Hconts
    | known ac vars conts conts' statics hp rd Hholds Hk].
  - (* an expression and the same rewritten *)
    destruct e as [y kind1 d body | bindings body | k1 hh body
                  | handlers body | ap | ac | scrutinee arms | ];
      cbn [number_in_region].
    + assert (Hnil : dead = [])
        by (apply (no_dead dead (Let y kind1 d body) vars);
            [ intros k2; reflexivity | exact Hdead ]).
      subst dead.
      rewrite !step_let_exec,
        (exec_named_number source known statics hp rd vars d Hholds
           Hsourced).
      destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ]
        eqn:Hv; [ | exact I ].
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      cbn [result_related]. apply (Rel_state []).
      * intros c Hc. destruct (Pos.eqb_spec y x) as [-> | Hyx];
          [ discriminate | ].
        rewrite lookup_other by congruence. exact (Hholds c Hc).
      * exact
          (sourced_let source statics hp hp' rd rd' vars y d v Hsourced Hv).
      * intros k2 c [].
      * intros k1 Hk1 _. apply Hconts; [ exact Hk1 | intros [] ].
    + assert (Hnil : dead = [])
        by (apply (no_dead dead (Let_static bindings body) vars);
            [ intros k2; reflexivity | exact Hdead ]).
      subst dead.
      rewrite !step_let_static.
      destruct (eval_static_bindings vars bindings statics); [ | exact I ].
      cbn [result_related].
      apply (Rel_state []); [ exact Hholds | exact Hsourced | | ].
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
                   (number_in_handler x source
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
        -- apply (Rel_state dead source known body);
             [ exact Hholds | exact Hsourced | exact Hbody_dead | ].
           intros k3 Hk3 Hk3dead.
           destruct (Pos.eq_dec k3 k1) as [-> | Hne];
             [ | exact (Hrest k3 Hk3 Hne Hk3dead) ].
           rewrite !resolve_head.
           apply closure_given; [ | exact Hsourced | exact Hhandler ].
           intros c' [= <-]. exact Hc.
        -- apply (Rel_state ((k1, c) :: dead) source known body);
             [ exact Hholds | exact Hsourced | | ].
           ++ intros k2 c2 [[= <- <-] | Hin];
                [ split; assumption | exact (Hbody_dead k2 c2 Hin) ].
           ++ intros k3 Hk3 Hk3dead. cbn [map fst] in Hk3dead.
              apply Hrest; [ exact Hk3 | | ].
              ** intros ->. apply Hk3dead. left. reflexivity.
              ** intros Hin. apply Hk3dead. right. exact Hin.
      * apply (Rel_state dead source known body);
          [ exact Hholds | exact Hsourced | exact Hbody_dead | ].
        intros k3 Hk3 Hk3dead.
        destruct (Pos.eq_dec k3 k1) as [-> | Hne];
          [ | exact (Hrest k3 Hk3 Hne Hk3dead) ].
        rewrite !resolve_head.
        apply closure_given;
          [ exact Hholds | exact Hsourced | exact Hhandler ].
    + assert (Hnil : dead = [])
        by (apply (no_dead dead (Let_cont_recursive handlers body) vars);
            [ intros k2; reflexivity | exact Hdead ]).
      subst dead.
      rewrite !step_let_cont_recursive.
      fold (number_handlers x source known handlers).
      rewrite number_handlers_bind_statics.
      destruct (handlers_bind_statics handlers); [ exact I | ].
      cbn [result_related]. apply (Rel_state [] source known body);
        [ exact Hholds | exact Hsourced | | ].
      * intros k2 c [].
      * intros k3 Hk3 _.
        apply rec_conts_given; [ exact Hholds | exact Hsourced | | ].
        -- intros k2 Hk2. apply Hconts; [ | intros [] ].
           cbn [cont_free_b]. rewrite Hk2. reflexivity.
        -- intros _. apply Hconts; [ | intros [] ].
           cbn [cont_free_b]. rewrite Hk3. apply orb_true_r.
    + assert (Hnil : dead = [])
        by (apply (no_dead dead (Apply ap) vars);
            [ intros k2; reflexivity | exact Hdead ]).
      subst dead.
      rewrite !step_apply.
      destruct (eval_simple vars (ap_callee ap)) as [callee | ];
        [ | exact I ].
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
        number_in_apply_cont_continuation.
      destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
      unfold apply_continuation.
      apply apply_cont_value_rel, Hconts;
        [ apply ac_continuation_free | intros [] ].
    + rewrite !step_switch,
        (eval_scrutinee_number source known vars scrutinee Hholds Hsourced).
      destruct (eval_simple vars scrutinee) as [v | ] eqn:Hv; [ | exact I ].
      destruct (naked_immediate_of_value v) as [j | ] eqn:Hj; [ | exact I ].
      rewrite find_arm_number.
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
      destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ];
        [ | exact I ].
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
      destruct (eval_simple vars (ap_callee ap)) as [callee | ];
        [ | exact I ].
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
      number_in_apply_cont_continuation.
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
    unfold apply_continuation.
    apply apply_cont_value_rel. exact Hk.
Qed.

(* The rewrite gives an equivalent term, whatever is given. *)
Lemma number_in_arms_equivalent (e : expr) :
  expr_equivalent e (number_in_arms x None e).
Proof.
  intros vars conts statics hp rd.
  apply (bisimulation_equivalent state_rel number_step).
  apply (Rel_state [] None None);
    [ intros c; discriminate | apply sourced_none | intros k2 c [] | ].
  intros k1 _ _. apply Rel_same.
Qed.

End Site.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

Lemma numbers_fold_equivalent (xs : list variable) :
  ∀ e : expr,
    expr_equivalent e (fold_left (fun e x => number_in_arms x None e) e xs).
Proof.
  induction xs as [ | x xs IH]; intros e; cbn [fold_left].
  - intros vars conts statics hp rd. apply equivalent_refl.
  - exact (expr_equivalent_trans _ _ _ (number_in_arms_equivalent x e)
      (IH (number_in_arms x None e))).
Qed.

Theorem numbers_in_arms_equivalent : numbers_in_arms_equivalent_statement.
Proof. intros e. apply numbers_fold_equivalent. Qed.
