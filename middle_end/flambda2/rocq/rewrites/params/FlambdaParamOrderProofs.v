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

(* Proofs of the statement of FlambdaParamOrderSpec.v. Nothing here
   needs review.

   One [Let_cont] first, then the pass. For the one [Let_cont], the
   term and its result run side by side, in step, under a relation of
   states ([order_site_rel]): the same expression outside the scope of
   the continuation, under variables that agree on every lookup;
   inside it, an expression and the same with the arguments of the
   continuation in the order, under continuations that hold the
   handler and the handler with its parameters in the order. The
   closures that the two runs make are related in the same way
   ([order_cont_rel]): a handler closed outside the scope, one closed
   inside it, the same with extra arguments, what is left of an
   over-application. A step of both keeps the relation ([order_step]),
   so the two states are equivalent
   (FlambdaSemanticsProofs.bisimulation_equivalent). Where the
   continuation is applied the two closures bind the same values to
   the same names, in two orders: the variables of the two runs then
   agree and are not the same list. The pass is the walk of that
   rewrite (FlambdaSpineWalkProofs.walk_spine_equivalent).

   Nothing here looks at how the order is made: it is an order of the
   positions of the parameters ([is_order]), which the rewrite
   tests.

   The relation, the lemmas on the closures applied and the cases of
   the step that do not apply the continuation follow those of
   FlambdaUnaliasParamsProofs.v, written again for this rewrite: there
   a parameter leaves the handler and the result takes one step more
   where the continuation is applied; here no parameter leaves and the
   two runs go in step. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaUnusedParamsProofs FlambdaSpineWalkSpec FlambdaSpineWalkProofs
  FlambdaUnaliasParamsProofs FlambdaParamOrderSpec.
Import ListNotations.

(* ======================================================================== *)
(* A list in an order                                                       *)
(* ======================================================================== *)

(* What is at each position of the result. *)
Lemma in_order_spec {A : Type} (order : list nat) :
  ∀ l l' : list A,
    in_order order l = Some l'
    <-> Forall2 (fun i a => nth_error l i = Some a) order l'.
Proof.
  induction order as [ | i order IH]; intros l l'; simpl.
  - split; [ intros [= <-]; constructor | ].
    intros Hall. inversion Hall. reflexivity.
  - split.
    + destruct (nth_error l i) as [a | ] eqn:Hi; [ | discriminate ].
      destruct (in_order order l) as [rest | ] eqn:Hrest; [ | discriminate ].
      intros [= <-]. constructor; [ exact Hi | apply IH, Hrest ].
    + intros Hall. inversion Hall as [ | i0 a order0 rest Hi Hrest]; subst.
      apply IH in Hrest. rewrite Hi, Hrest. reflexivity.
Qed.

Lemma in_order_length {A : Type} (order : list nat) (l l' : list A) :
  in_order order l = Some l' -> length l' = length order.
Proof.
  intros Hin. apply in_order_spec in Hin.
  symmetry. exact (Forall2_length Hin).
Qed.

(* An order of positions that the list has gives a list. *)
Lemma in_order_some {A : Type} (order : list nat) (l : list A) :
  forallb (fun j => Nat.ltb j (length l)) order = true ->
  ∃ l' : list A, in_order order l = Some l'.
Proof.
  induction order as [ | i order IH]; simpl; intros Hbelow.
  - eexists. reflexivity.
  - apply andb_true_iff in Hbelow as [Hi Hrest]. apply Nat.ltb_lt in Hi.
    destruct (IH Hrest) as [rest ->].
    destruct (nth_error l i) as [a | ] eqn:Ha; [ eexists; reflexivity | ].
    apply nth_error_None in Ha. lia.
Qed.

Lemma in_order_map {A B : Type} (f : A -> B) (order : list nat)
    (l : list A) :
  in_order order (map f l) = option_map (map f) (in_order order l).
Proof.
  induction order as [ | i order IH]; simpl; [ reflexivity | ].
  rewrite nth_error_map, IH.
  destruct (nth_error l i), (in_order order l); reflexivity.
Qed.

(* The element at a position that the order has is in the result. *)
Lemma in_order_in {A : Type} (order : list nat) (l l' : list A) (i : nat)
    (a : A) :
  in_order order l = Some l' ->
  In i order ->
  nth_error l i = Some a ->
  In a l'.
Proof.
  intros Hin. apply in_order_spec in Hin.
  induction Hin as [ | j b order l' Hj _ IH]; simpl; [ contradiction | ].
  intros [-> | Hrest] Ha; [ left; congruence | right; exact (IH Hrest Ha) ].
Qed.

(* An element of the result is one of the list, at a position of the
   order. *)
Lemma in_order_from {A : Type} (order : list nat) (l l' : list A) (j : nat)
    (a : A) :
  in_order order l = Some l' ->
  nth_error l' j = Some a ->
  ∃ i : nat, nth_error order j = Some i ∧ nth_error l i = Some a.
Proof.
  intros Hin. apply in_order_spec in Hin. revert j.
  induction Hin as [ | i b order l' Hi _ IH]; intros [ | j]; simpl;
    try discriminate.
  - intros [= <-]. exists i. auto.
  - apply IH.
Qed.

(* Two lists in one order have, at each position, the elements that
   they had at one position. *)
Lemma in_order_same_position {A B : Type} (order : list nat) (l1 l1' : list A)
    (l2 l2' : list B) (j : nat) (a : A) :
  in_order order l1 = Some l1' ->
  in_order order l2 = Some l2' ->
  nth_error l1' j = Some a ->
  ∃ i : nat,
    nth_error order j = Some i
    ∧ nth_error l1 i = Some a
    ∧ nth_error l2' j = nth_error l2 i.
Proof.
  intros H1 H2 Ha.
  destruct (in_order_from order l1 l1' j a H1 Ha) as (i & Hj & Hi).
  exists i. split; [ exact Hj | split; [ exact Hi | ] ].
  apply in_order_spec in H2. clear H1 Ha Hi. revert j Hj.
  induction H2 as [ | i0 b order l2' Hi0 _ IH]; intros [ | j]; simpl;
    try discriminate.
  - intros [= ->]. symmetry. exact Hi0.
  - apply IH.
Qed.

(* What an order of [n] positions is. *)
Lemma is_order_facts (n : nat) (order : list nat) :
  is_order n order = true ->
  length order = n
  ∧ forallb (fun j => Nat.ltb j n) order = true
  ∧ ∀ i : nat, i < n -> In i order.
Proof.
  unfold is_order. intros Horder.
  apply andb_true_iff in Horder as [Horder Hall].
  apply andb_true_iff in Horder as [Hlength Hbelow].
  apply Nat.eqb_eq in Hlength.
  split; [ exact Hlength | split; [ exact Hbelow | ] ].
  intros i Hi. rewrite forallb_forall in Hall.
  assert (Hin : In i (seq 0 n)) by (apply in_seq; lia).
  specialize (Hall i Hin). apply existsb_exists in Hall as (j & Hj & Heq).
  apply Nat.eqb_eq in Heq. subst j. exact Hj.
Qed.

(* Two lists of [n] elements are the same if they are in an order of
   [n] positions. *)
Lemma in_order_injective {A : Type} (n : nat) (order : list nat)
    (l1 l2 l' : list A) :
  (∀ i : nat, i < n -> In i order) ->
  length l1 = n ->
  length l2 = n ->
  in_order order l1 = Some l' ->
  in_order order l2 = Some l' ->
  l1 = l2.
Proof.
  intros Hall Hl1 Hl2 H1 H2. apply nth_error_ext. intros i.
  destruct (Nat.lt_ge_cases i n) as [Hi | Hi].
  - destruct (nth_error l1 i) as [a | ] eqn:Ha;
      [ | apply nth_error_None in Ha; lia ].
    destruct (nth_error l2 i) as [b | ] eqn:Hb;
      [ | apply nth_error_None in Hb; lia ].
    apply in_order_spec in H1, H2. specialize (Hall i Hi).
    clear Hl1 Hl2 Hi. revert H2 Hall.
    induction H1 as [ | j c order l' Hj _ IH]; simpl; [ contradiction | ].
    intros H2 Hin. inversion H2 as [ | j0 c0 order0 l0 Hj' Hrest]; subst.
    destruct Hin as [-> | Hin]; [ congruence | exact (IH Hrest Hin) ].
  - rewrite (proj2 (nth_error_None l1 i)), (proj2 (nth_error_None l2 i))
      by lia.
    reflexivity.
Qed.

(* ======================================================================== *)
(* Arguments, kinds and parameters in an order                              *)
(* ======================================================================== *)

Lemma eval_simples_at (vars : env) (args : list simple) :
  ∀ (vs : list value) (i : nat) (s : simple),
    eval_simples vars args = Some vs ->
    nth_error args i = Some s ->
    ∃ v : value, eval_simple vars s = Some v ∧ nth_error vs i = Some v.
Proof.
  induction args as [ | s0 args IH]; intros vs i s; simpl;
    [ intros _; destruct i; discriminate | ].
  destruct (eval_simple vars s0) as [v0 | ] eqn:Hs0; [ | discriminate ].
  destruct (eval_simples vars args) as [vs0 | ] eqn:Hargs; [ | discriminate ].
  intros [= <-]. destruct i as [ | i]; simpl.
  - intros [= <-]. exists v0. auto.
  - intros Hs. exact (IH vs0 i s eq_refl Hs).
Qed.

Lemma eval_simples_none (vars : env) (args : list simple) :
  eval_simples vars args = None ->
  ∃ s : simple, In s args ∧ eval_simple vars s = None.
Proof.
  induction args as [ | s0 args IH]; simpl; [ discriminate | ].
  destruct (eval_simple vars s0) as [v0 | ] eqn:Hs0;
    [ | intros _; exists s0; auto ].
  destruct (eval_simples vars args) as [vs0 | ]; [ discriminate | ].
  intros _. destruct (IH eq_refl) as (s & Hin & Hs). exists s. auto.
Qed.

Lemma eval_simples_with_none (vars : env) (args : list simple) (s : simple) :
  In s args -> eval_simple vars s = None -> eval_simples vars args = None.
Proof.
  induction args as [ | s0 args IH]; simpl; [ contradiction | ].
  intros [-> | Hin] Hs; [ rewrite Hs; reflexivity | ].
  rewrite (IH Hin Hs). destruct (eval_simple vars s0); reflexivity.
Qed.

Lemma eval_simples_length (vars : env) (args : list simple) :
  ∀ vs : list value,
    eval_simples vars args = Some vs -> length vs = length args.
Proof.
  induction args as [ | s0 args IH]; intros vs; simpl.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s0) as [v0 | ]; [ | discriminate ].
    destruct (eval_simples vars args) as [vs0 | ]; [ | discriminate ].
    intros [= <-]. simpl. rewrite (IH vs0 eq_refl). reflexivity.
Qed.

(* The arguments in an order of their positions have the values of the
   arguments, in the order; and have no value if one of the arguments
   has none. *)
Lemma eval_simples_in_order (vars : env) (order : list nat)
    (args args' : list simple) :
  (∀ i : nat, i < length args -> In i order) ->
  in_order order args = Some args' ->
  eval_simples vars args'
  = match eval_simples vars args with
    | Some vs => in_order order vs
    | None => None
    end.
Proof.
  intros Hall Hin.
  destruct (eval_simples vars args) as [vs | ] eqn:Hargs.
  - clear Hall. apply in_order_spec in Hin.
    induction Hin as [ | i s order args' Hi _ IH]; simpl; [ reflexivity | ].
    destruct (eval_simples_at vars args vs i s Hargs Hi) as (v & Hs & Hv).
    rewrite Hs, IH, Hv. destruct (in_order order vs); reflexivity.
  - destruct (eval_simples_none vars args Hargs) as (s & Hs_in & Hs).
    apply In_nth_error in Hs_in as [i Hi].
    assert (Hlt : i < length args)
      by (apply nth_error_Some; rewrite Hi; discriminate).
    exact (eval_simples_with_none vars args' s
      (in_order_in order args args' i s Hin (Hall i Hlt) Hi) Hs).
Qed.

(* The kinds of the values against the kinds of the parameters, both
   in an order of their positions. *)
Lemma kinds_eqb_in_order (n : nat) (order : list nat)
    (kinds1 kinds2 kinds1' kinds2' : list flambda_kind) :
  (∀ i : nat, i < n -> In i order) ->
  length kinds1 = n ->
  length kinds2 = n ->
  in_order order kinds1 = Some kinds1' ->
  in_order order kinds2 = Some kinds2' ->
  kinds_eqb kinds1' kinds2' = kinds_eqb kinds1 kinds2.
Proof.
  intros Hall Hl1 Hl2 H1 H2. unfold kinds_eqb.
  destruct (list_eq_dec kind_eq_dec kinds1' kinds2') as [Heq' | Hne'],
    (list_eq_dec kind_eq_dec kinds1 kinds2) as [Heq | Hne];
    try reflexivity.
  - exfalso. apply Hne. subst kinds2'.
    exact (in_order_injective n order kinds1 kinds2 kinds1' Hall Hl1 Hl2 H1
      H2).
  - exfalso. apply Hne'. subst kinds2. congruence.
Qed.

(* What a variable is once parameters are bound: what was bound to it
   at one of its positions among the parameters, or what it was. *)
Lemma lookup_bind_params_cases (params : list variable) :
  ∀ (vs : list value) (vars bound : env) (y : variable),
    bind_params params vs vars = Some bound ->
    (¬ In y params ∧ lookup y bound = lookup y vars)
    ∨ ∃ (j : nat) (v : value),
        nth_error params j = Some y
        ∧ nth_error vs j = Some v
        ∧ lookup y bound = Some v.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars bound y; simpl;
    try discriminate.
  - intros [= <-]. left. auto.
  - intros Hbind.
    destruct (IH vs ((p, v) :: vars) bound y Hbind)
      as [[Hout Hlookup] | (j & w & Hj & Hw & Hlookup)].
    + rewrite lookup_cons in Hlookup.
      destruct (Pos.eqb_spec y p) as [-> | Hne].
      * right. exists 0, v. auto.
      * left. split; [ | exact Hlookup ]. intros [Heq | Hin]; auto.
    + right. exists (S j), w. auto.
Qed.

Lemma bind_params_lengths (params : list variable) :
  ∀ (vs : list value) (vars : env),
    match bind_params params vs vars with
    | Some _ => length params = length vs
    | None => length params <> length vs
    end.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] vars; simpl;
    try discriminate; [ reflexivity | ].
  specialize (IH vs ((p, v) :: vars)).
  destruct (bind_params params vs ((p, v) :: vars)); lia.
Qed.

(* Distinct parameters and their values, and the same in an order of
   their positions, bound over the same variables: the two agree. *)
Lemma bind_params_in_order (order : list nat) (params params' : list variable)
    (vs vs' : list value) (vars bound bound' : env) :
  distinct params = true ->
  (∀ i : nat, i < length params -> In i order) ->
  in_order order params = Some params' ->
  in_order order vs = Some vs' ->
  bind_params params vs vars = Some bound ->
  bind_params params' vs' vars = Some bound' ->
  agree bound bound'.
Proof.
  intros Hdistinct Hall Hparams Hvs Hbound Hbound' y.
  apply distinct_NoDup in Hdistinct.
  destruct (lookup_bind_params_cases params vs vars bound y Hbound)
    as [[Hout Hlookup] | (i & v & Hi & Hv & Hlookup)];
    destruct (lookup_bind_params_cases params' vs' vars bound' y Hbound')
      as [[Hout' Hlookup'] | (j & v' & Hj & Hv' & Hlookup')].
  - congruence.
  - exfalso. apply Hout.
    destruct (in_order_from order params params' j y Hparams Hj)
      as (i & _ & Hi).
    exact (nth_error_In params i Hi).
  - exfalso. apply Hout'.
    assert (Hlt : i < length params)
      by (apply nth_error_Some; rewrite Hi; discriminate).
    exact (in_order_in order params params' i y Hparams (Hall i Hlt) Hi).
  - rewrite Hlookup, Hlookup'.
    destruct
      (in_order_same_position order params params' vs vs' j y Hparams Hvs Hj)
      as (i' & _ & Hi' & Hsame).
    assert (Heq : i' = i).
    { assert (Hlt : i' < length params)
        by (apply nth_error_Some; rewrite Hi'; discriminate).
      apply (proj1 (NoDup_nth_error params) Hdistinct i' i Hlt).
      congruence. }
    subst i'. congruence.
Qed.

(* ======================================================================== *)
(* What the rewrite keeps of the syntax                                     *)
(* ======================================================================== *)

Fixpoint order_args_binds_statics (k : continuation) (order : list nat)
    (e : expr) {struct e} :
  binds_statics (order_args k order e) = binds_statics e
with order_args_in_handler_binds_statics (k : continuation)
    (order : list nat) (h : continuation_handler) {struct h} :
  handler_binds_statics (order_args_in_handler k order h)
  = handler_binds_statics h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [order_args binds_statics]; try reflexivity.
    + apply order_args_binds_statics.
    + rewrite order_args_binds_statics. reflexivity.
    + rewrite order_args_in_handler_binds_statics, order_args_binds_statics.
      reflexivity.
    + rewrite order_args_binds_statics. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map existsb].
      rewrite order_args_in_handler_binds_statics, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [order_args_in_handler handler_binds_statics].
    apply order_args_binds_statics.
Qed.

Definition order_handlers (k : continuation) (order : list nat)
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(k', h) => (k', order_args_in_handler k order h)) handlers.

Lemma order_handlers_bind_statics (k : continuation) (order : list nat)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics (order_handlers k order handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics, order_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map existsb]. rewrite order_args_in_handler_binds_statics, IH.
  reflexivity.
Qed.

Lemma lookup_order_handlers (k : continuation) (order : list nat)
    (k1 : continuation)
    (handlers : list (continuation * continuation_handler)) :
  lookup k1 (order_handlers k order handlers)
  = option_map (order_args_in_handler k order) (lookup k1 handlers).
Proof.
  unfold order_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map]. rewrite !lookup_cons, IH.
  destruct (Pos.eqb k1 k'); reflexivity.
Qed.

Lemma find_arm_order (k : continuation) (order : list nat) (j : Z)
    (arms : list (Z * apply_cont)) :
  find_arm j (map (fun '(n, ac) => (n, order_args_of k order ac)) arms)
  = option_map (order_args_of k order) (find_arm j arms).
Proof.
  induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
  cbn [map find_arm]. destruct (Z.eqb j n); [ reflexivity | exact IH ].
Qed.

(* ======================================================================== *)
(* One [Let_cont]: the relation of the two runs                             *)
(* ======================================================================== *)

Section Site.

(* The continuation, the order, the handler with its parameters and
   with them in the order; the variables and the continuations under
   which the [Let_cont] runs. *)
Variables (k : continuation) (order : list nat)
  (params params' : list variable) (metadata : handler_metadata)
  (kinds' : list flambda_kind) (handler : expr) (vars0 : env)
  (conts0 : cont_env).

Hypothesis Hdistinct : distinct params = true.
Hypothesis Hkinds : length (hm_params_kinds metadata) = length params.
Hypothesis Horder : is_order (length params) order = true.
Hypothesis Hparams' : in_order order params = Some params'.
Hypothesis Hkinds' : in_order order (hm_params_kinds metadata) = Some kinds'.

Definition order_site_closure : cont_value :=
  Closure (Handler params metadata handler) vars0 conts0 [].

Definition order_site_closure_ordered : cont_value :=
  Closure
    (Handler params'
       {| hm_params_kinds := kinds'; hm_is_exn_handler := false |} handler)
    vars0 conts0 [].

(* The handlers of a recursive [Let_cont] in the scope of [k]: none
   binds [k], and each mentions [k] only to apply it. *)
Definition order_handlers_apply
    (handlers : list (continuation * continuation_handler)) : bool :=
  forallb
    (fun '(k', h) =>
       negb (Pos.eqb k' k) && handler_all_apply_with k (length params) h)
    handlers.

(* What a continuation name stands for in the two runs: the same
   thing; or two closures of one handler, under variables that agree
   and continuations so related; or two closures of a handler in the
   scope of [k] and of that handler with the arguments of [k] in the
   order, under continuations that have the two closures of [k]; or
   two so related with the same extra arguments; or what is left of
   the same over-application. *)
Inductive order_cont_rel : cont_value -> cont_value -> Prop :=
  | Order_same (c : cont_value) : order_cont_rel c c
  | Order_plain (hh : continuation_handler) (dv dv' : env) (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    agree dv dv' ->
    (∀ k1 : continuation, order_cont_rel (resolve k1 dc) (resolve k1 dc')) ->
    order_cont_rel (Closure hh dv dc recursive) (Closure hh dv' dc' recursive)
  | Order_inside (hh : continuation_handler) (dv dv' : env) (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    agree dv dv' ->
    handler_all_apply_with k (length params) hh = true ->
    order_handlers_apply recursive = true ->
    resolve k dc = order_site_closure ->
    resolve k dc' = order_site_closure_ordered ->
    (∀ k1 : continuation,
      k1 <> k -> order_cont_rel (resolve k1 dc) (resolve k1 dc')) ->
    order_cont_rel (Closure hh dv dc recursive)
      (Closure (order_args_in_handler k order hh) dv' dc'
         (order_handlers k order recursive))
  | Order_extra (c c' : cont_value) (extra : list value) :
    order_cont_rel c c' ->
    order_cont_rel (Extra_args c extra) (Extra_args c' extra)
  | Order_over (remaining : list value) (remaining_arity return_arity : arity)
      (return_to return_to' exn_to exn_to' : cont_value) :
    order_cont_rel return_to return_to' ->
    order_cont_rel exn_to exn_to' ->
    order_cont_rel
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      (Over_application remaining remaining_arity return_arity return_to'
         exn_to').

Definition order_conts_plain (conts conts' : cont_env) : Prop :=
  ∀ k1 : continuation, order_cont_rel (resolve k1 conts) (resolve k1 conts').

Definition order_conts_in_site (conts conts' : cont_env) : Prop :=
  resolve k conts = order_site_closure
  ∧ resolve k conts' = order_site_closure_ordered
  ∧ ∀ k1 : continuation,
      k1 <> k -> order_cont_rel (resolve k1 conts) (resolve k1 conts').

(* Two states: the same expression, outside the scope of [k]; or, in
   it, an expression that mentions [k] only to apply it and the same
   with the arguments of [k] in the order. *)
Inductive order_site_rel : state -> state -> Prop :=
  | Order_state_plain (e : expr) (vars vars' : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) :
    agree vars vars' ->
    order_conts_plain conts conts' ->
    order_site_rel (State e vars conts statics hp rd)
      (State e vars' conts' statics hp rd)
  | Order_state_in_scope (e : expr) (vars vars' : env)
      (conts conts' : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
    agree vars vars' ->
    all_apply_with k (length params) e = true ->
    order_conts_in_site conts conts' ->
    order_site_rel (State e vars conts statics hp rd)
      (State (order_args k order e) vars' conts' statics hp rd).

Lemma order_site_rel_refl (s : state) : order_site_rel s s.
Proof.
  destruct s as [e vars conts statics hp rd].
  apply Order_state_plain; [ apply agree_refl | intros k1; apply Order_same ].
Qed.

Lemma order_with_extra_args_rel (c c' : cont_value) (extra : list value) :
  order_cont_rel c c' ->
  order_cont_rel (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hc. destruct extra; simpl; [ exact Hc | apply Order_extra, Hc ].
Qed.

(* The closures of the handlers of a recursive [Let_cont]. *)
Lemma order_rec_conts_plain
    (handlers : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  agree vars vars' ->
  order_conts_plain conts conts' ->
  order_conts_plain (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers vars' conts' ++ conts').
Proof.
  intros Hvars Hconts k1. rewrite !resolve_rec_closures.
  destruct (lookup k1 handlers) as [hh | ]; [ | apply Hconts ].
  apply Order_plain; assumption.
Qed.

Lemma order_handlers_apply_lookup
    (handlers : list (continuation * continuation_handler)) :
  order_handlers_apply handlers = true ->
  lookup k handlers = None
  ∧ ∀ (k1 : continuation) (hh : continuation_handler),
      lookup k1 handlers = Some hh ->
      handler_all_apply_with k (length params) hh = true.
Proof.
  unfold order_handlers_apply.
  induction handlers as [ | [k' hh'] handlers IH]; cbn [forallb].
  - intros _. split; [ reflexivity | discriminate ].
  - intros Happly. apply andb_true_iff in Happly as [Hhead Hrest].
    apply andb_true_iff in Hhead as [Hne Hhh].
    apply negb_true_iff in Hne. destruct (IH Hrest) as [Hk Hothers].
    split.
    + rewrite lookup_cons, Pos.eqb_sym, Hne. exact Hk.
    + intros k1 hh. rewrite lookup_cons.
      destruct (Pos.eqb k1 k'); [ intros [= <-]; exact Hhh | apply Hothers ].
Qed.

Lemma order_rec_conts_in_site
    (handlers : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  agree vars vars' ->
  order_handlers_apply handlers = true ->
  order_conts_in_site conts conts' ->
  order_conts_in_site (rec_closures handlers vars conts ++ conts)
    (rec_closures (order_handlers k order handlers) vars' conts' ++ conts').
Proof.
  intros Hvars Happly (Hk & Hk' & Hconts).
  destruct (order_handlers_apply_lookup handlers Happly) as [Hnone Hothers].
  split; [ | split ].
  - rewrite resolve_rec_closures, Hnone. exact Hk.
  - rewrite resolve_rec_closures, lookup_order_handlers, Hnone. exact Hk'.
  - intros k1 Hne. rewrite !resolve_rec_closures, lookup_order_handlers.
    destruct (lookup k1 handlers) as [hh | ] eqn:Hlookup; cbn [option_map];
      [ | exact (Hconts k1 Hne) ].
    apply Order_inside; try assumption.
    exact (Hothers k1 hh Hlookup).
Qed.

(* The same full call, returning and raising to related continuations. *)
Lemma order_call_code_rel (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  order_cont_rel return_to return_to' ->
  order_cont_rel exn_to exn_to' ->
  result_related order_site_rel
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
    apply Order_state_plain; [ apply agree_refl | ].
    intros k'.
    destruct (Pos.eq_dec k' (code_return_continuation c)) as [-> | Hne];
      [ rewrite !resolve_head; exact Hreturn | ].
    rewrite !(resolve_other k' (code_return_continuation c)) by exact Hne.
    destruct (Pos.eq_dec k' (code_exn_continuation c)) as [-> | Hne'];
      [ rewrite !resolve_head; exact Hraise | ].
    rewrite !(resolve_other k' (code_exn_continuation c)) by exact Hne'.
    apply Order_same.
Qed.

Lemma order_run_call_rel (shape : option call_shape) (return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  order_cont_rel return_to return_to' ->
  order_cont_rel exn_to exn_to' ->
  (∀ partial : value,
    result_related order_site_rel
      (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value return_to' [partial] statics hp rd)) ->
  result_related order_site_rel
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
  - apply order_call_code_rel; assumption.
  - apply order_call_code_rel; [ apply Order_over; assumption | exact Hraise ].
  - apply Hpartial.
  - exact I.
Qed.

(* Related continuations, applied to the same values. *)
Lemma order_apply_cont_value_rel (c c' : cont_value) (statics : static_env)
    (hp : heap) (rd : read_record) :
  order_cont_rel c c' ->
  ∀ vs : list value,
    result_related order_site_rel (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' vs statics hp rd).
Proof.
  intros Hc.
  induction Hc
    as [c
       | [ps m he] dv dv' dc dc' recursive Hdv Hdc _
       | [ps m he] dv dv' dc dc' recursive Hdv Hhh Hrec Hk Hk' Hdc _
       | c c' extra Hc IHc
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise _];
    intros vs.
  - apply result_related_refl, order_site_rel_refl.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m));
      [ | exact I ].
    pose proof (bind_params_agree ps vs dv dv' Hdv) as Hbind.
    destruct (bind_params ps vs dv) as [bound | ],
      (bind_params ps vs dv') as [bound' | ]; try contradiction;
      [ | exact I ].
    simpl. apply Order_state_plain; [ exact Hbind | ].
    apply order_rec_conts_plain; assumption.
  - cbn [order_args_in_handler apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m));
      [ | exact I ].
    pose proof (bind_params_agree ps vs dv dv' Hdv) as Hbind.
    destruct (bind_params ps vs dv) as [bound | ],
      (bind_params ps vs dv') as [bound' | ]; try contradiction;
      [ | exact I ].
    simpl. apply Order_state_in_scope; [ exact Hbind | exact Hhh | ].
    apply order_rec_conts_in_site; try assumption.
    split; [ exact Hk | split; [ exact Hk' | exact Hdc ] ].
  - exact (IHc (vs ++ extra)).
  - rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; try exact I.
    unfold call_function.
    apply order_run_call_rel; [ exact Hreturn | exact Hraise | ].
    intros partial. exact (IHreturn [partial]).
Qed.

Lemma order_call_function_rel (callee : value) (call_kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  order_cont_rel return_to return_to' ->
  order_cont_rel exn_to exn_to' ->
  result_related order_site_rel
    (call_function callee call_kind vs args_arity return_arity return_to
       exn_to statics hp rd)
    (call_function callee call_kind vs args_arity return_arity return_to'
       exn_to' statics hp rd).
Proof.
  intros Hreturn Hraise. unfold call_function.
  apply order_run_call_rel; [ exact Hreturn | exact Hraise | ].
  intros partial. apply order_apply_cont_value_rel, Hreturn.
Qed.

(* The two closures of [k], applied to as many values as there are
   parameters and to the same values in the order. *)
Lemma order_site_applied (vs vs' : list value) (statics : static_env)
    (hp : heap) (rd : read_record) :
  length vs = length params ->
  in_order order vs = Some vs' ->
  result_related order_site_rel
    (apply_cont_value order_site_closure vs statics hp rd)
    (apply_cont_value order_site_closure_ordered vs' statics hp rd).
Proof.
  intros Hlength Hvs'.
  destruct (is_order_facts (length params) order Horder)
    as (Horder_length & _ & Hall).
  unfold order_site_closure, order_site_closure_ordered.
  cbn [apply_cont_value hm_params_kinds].
  assert (Hmapped :
    in_order order (map kind_of_value vs) = Some (map kind_of_value vs'))
    by (rewrite in_order_map, Hvs'; reflexivity).
  rewrite
    (kinds_eqb_in_order (length params) order (map kind_of_value vs)
       (hm_params_kinds metadata) (map kind_of_value vs') kinds' Hall
       ltac:(rewrite length_map; exact Hlength) Hkinds Hmapped Hkinds').
  destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
    [ | exact I ].
  pose proof (bind_params_lengths params vs vars0) as Hbound.
  pose proof (bind_params_lengths params' vs' vars0) as Hbound'.
  rewrite (in_order_length order params params' Hparams'),
    (in_order_length order vs vs' Hvs') in Hbound'.
  destruct (bind_params params vs vars0) as [bound | ] eqn:Hbind;
    [ | congruence ].
  destruct (bind_params params' vs' vars0) as [bound' | ] eqn:Hbind';
    [ | congruence ].
  simpl. apply Order_state_plain; [ | intros k1; apply Order_same ].
  exact (bind_params_in_order order params params' vs vs' vars0 bound bound'
    Hdistinct Hall Hparams' Hvs' Hbind Hbind').
Qed.

(* A step of both keeps the relation. *)
Lemma order_step (s s' : state) :
  order_site_rel s s' -> result_related order_site_rel (step s) (step s').
Proof.
  intros
    [e vars vars' conts conts' statics hp rd Hvars Hconts
    | e vars vars' conts conts' statics hp rd Hvars Happly
        (Hk & Hk' & Hconts)].
  - (* outside the scope of [k] *)
    destruct e as [y kind1 d body | bindings body | k1 hh body
                  | handlers body | ap | ac | scrutinee arms | ].
    + rewrite !step_let_exec,
        (exec_named_agree statics hp rd vars vars' d (fun z _ => Hvars z)).
      destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ];
        [ | exact I ].
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      simpl. apply Order_state_plain;
        [ apply agree_cons, Hvars | exact Hconts ].
    + rewrite !step_let_static,
        (eval_static_bindings_agree vars vars' bindings (fun z _ => Hvars z)
           statics).
      destruct (eval_static_bindings vars bindings statics); [ | exact I ].
      simpl. apply Order_state_plain; assumption.
    + rewrite !step_let_cont_non_recursive. simpl.
      apply Order_state_plain; [ exact Hvars | ].
      intros k2. destruct (Pos.eq_dec k2 k1) as [-> | Hne].
      * rewrite !resolve_head. apply Order_plain; assumption.
      * rewrite !(resolve_other k2 k1) by exact Hne. apply Hconts.
    + rewrite !step_let_cont_recursive.
      destruct (handlers_bind_statics handlers); [ exact I | ].
      simpl. apply Order_state_plain; [ exact Hvars | ].
      apply order_rec_conts_plain; assumption.
    + rewrite !step_apply,
        (eval_simple_agree vars vars' (ap_callee ap) (fun z _ => Hvars z)),
        (eval_simples_agree vars vars' (ap_args ap) (fun z _ => Hvars z)),
        (eval_simples_agree vars vars' (exn_extra_simples ap)
           (fun z _ => Hvars z)).
      destruct (eval_simple vars (ap_callee ap)) as [callee | ];
        [ | exact I ].
      destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
      destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
        [ | exact I ].
      apply order_call_function_rel;
        [ apply Hconts | apply order_with_extra_args_rel, Hconts ].
    + rewrite !step_apply_cont,
        (eval_simples_agree vars vars' (ac_args ac) (fun z _ => Hvars z)).
      destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
      unfold apply_continuation. apply order_apply_cont_value_rel, Hconts.
    + rewrite !step_switch,
        (eval_simple_agree vars vars' scrutinee (fun z _ => Hvars z)).
      destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
      destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
      destruct (find_arm j arms) as [ac | ]; [ | exact I ].
      simpl. apply Order_state_plain; assumption.
    + rewrite !step_invalid. exact I.
  - (* in the scope of [k] *)
    assert (Hinside : order_conts_in_site conts conts')
      by (split; [ exact Hk | split; [ exact Hk' | exact Hconts ] ]).
    destruct e as [y kind1 d body | bindings body | k1 hh body
                  | handlers body | ap | ac | scrutinee arms | ];
      cbn [order_args all_apply_with] in Happly |- *.
    + rewrite !step_let_exec,
        (exec_named_agree statics hp rd vars vars' d (fun z _ => Hvars z)).
      destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ];
        [ | exact I ].
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      simpl. apply Order_state_in_scope;
        [ apply agree_cons, Hvars | exact Happly | exact Hinside ].
    + rewrite !step_let_static,
        (eval_static_bindings_agree vars vars' bindings (fun z _ => Hvars z)
           statics).
      destruct (eval_static_bindings vars bindings statics); [ | exact I ].
      simpl. apply Order_state_in_scope; assumption.
    + rewrite !step_let_cont_non_recursive. simpl.
      apply andb_true_iff in Happly as [Happly Hbody].
      apply andb_true_iff in Happly as [Hne Hhh].
      apply negb_true_iff, Pos.eqb_neq in Hne.
      apply Order_state_in_scope; [ exact Hvars | exact Hbody | ].
      split; [ | split ].
      * rewrite resolve_other by congruence. exact Hk.
      * rewrite resolve_other by congruence. exact Hk'.
      * intros k2 Hk2. destruct (Pos.eq_dec k2 k1) as [-> | Hne'].
        -- rewrite !resolve_head.
           exact (Order_inside hh vars vars' conts conts' [] Hvars Hhh eq_refl
             Hk Hk' Hconts).
        -- rewrite !(resolve_other k2 k1) by exact Hne'.
           exact (Hconts k2 Hk2).
    + rewrite !step_let_cont_recursive.
      fold (order_handlers k order handlers).
      rewrite order_handlers_bind_statics.
      destruct (handlers_bind_statics handlers); [ exact I | ].
      simpl. apply andb_true_iff in Happly as [Hhandlers Hbody].
      apply Order_state_in_scope; [ exact Hvars | exact Hbody | ].
      apply order_rec_conts_in_site; assumption.
    + apply andb_true_iff in Happly as [Hreturn Hraise].
      apply negb_true_iff, Pos.eqb_neq in Hreturn, Hraise.
      rewrite !step_apply,
        (eval_simple_agree vars vars' (ap_callee ap) (fun z _ => Hvars z)),
        (eval_simples_agree vars vars' (ap_args ap) (fun z _ => Hvars z)),
        (eval_simples_agree vars vars' (exn_extra_simples ap)
           (fun z _ => Hvars z)).
      destruct (eval_simple vars (ap_callee ap)) as [callee | ];
        [ | exact I ].
      destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
      destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
        [ | exact I ].
      apply order_call_function_rel;
        [ apply Hconts, Hreturn
        | apply order_with_extra_args_rel, Hconts, Hraise ].
    + (* an application of a continuation *)
      unfold apply_cont_with in Happly.
      apply andb_true_iff in Happly as [_ Happly].
      unfold order_args_of.
      destruct (Pos.eqb_spec (ac_continuation ac) k) as [Hac | Hac].
      * (* of [k] *)
        apply Nat.eqb_eq in Happly.
        destruct (is_order_facts (length params) order Horder)
          as (_ & Hbelow & Hall).
        destruct (in_order_some order (ac_args ac)) as [args' Hargs'];
          [ rewrite Happly; exact Hbelow | ].
        rewrite Hargs', !step_apply_cont. cbn [ac_args ac_continuation].
        rewrite
          (eval_simples_agree vars vars' args' (fun z _ => Hvars z)),
          (eval_simples_in_order vars order (ac_args ac) args'
             ltac:(rewrite Happly; exact Hall) Hargs').
        unfold apply_continuation. rewrite Hac, Hk, Hk'.
        destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
          [ | exact I ].
        pose proof (eval_simples_length vars (ac_args ac) vs Hvs) as Hlength.
        rewrite Happly in Hlength.
        destruct (in_order_some order vs) as [vs' Hvs'];
          [ rewrite Hlength; exact Hbelow | ].
        rewrite Hvs'.
        exact (order_site_applied vs vs' statics hp rd Hlength Hvs').
      * rewrite !step_apply_cont,
          (eval_simples_agree vars vars' (ac_args ac) (fun z _ => Hvars z)).
        destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
        unfold apply_continuation.
        apply order_apply_cont_value_rel, Hconts, Hac.
    + rewrite !step_switch,
        (eval_simple_agree vars vars' scrutinee (fun z _ => Hvars z)).
      destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
      destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
      rewrite find_arm_order.
      destruct (find_arm j arms) as [ac | ] eqn:Hfind; [ | exact I ].
      simpl. apply (Order_state_in_scope (Apply_cont ac));
        [ exact Hvars | | exact Hinside ].
      rewrite forallb_forall in Happly.
      exact (Happly (j, ac) (find_arm_in j arms ac Hfind)).
    + rewrite !step_invalid. exact I.
Qed.

Lemma order_site_equivalent (s s' : state) :
  order_site_rel s s' -> equivalent s s'.
Proof. exact (bisimulation_equivalent order_site_rel order_step s s'). Qed.

End Site.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The rewrite of one node. *)
Lemma order_node_equivalent (e : expr) : expr_equivalent e (order_node e).
Proof.
  destruct e as [ | | k [params metadata handler] body | | | | | ];
    try (intros vars conts statics hp rd; apply equivalent_refl).
  cbn [order_node].
  destruct
    (negb (hm_is_exn_handler metadata) && distinct params
     && Nat.eqb (length (hm_params_kinds metadata)) (length params)
     && is_order (length params) (use_order params handler)
     && all_apply_with k (length params) body)
    eqn:Hcond;
    [ | intros vars conts statics hp rd; apply equivalent_refl ].
  destruct (in_order (use_order params handler) params) as [params' | ]
    eqn:Hparams';
    [ | intros vars conts statics hp rd; apply equivalent_refl ].
  destruct (in_order (use_order params handler) (hm_params_kinds metadata))
    as [kinds' | ] eqn:Hkinds';
    [ | intros vars conts statics hp rd; apply equivalent_refl ].
  apply andb_true_iff in Hcond as [Hcond Happly].
  apply andb_true_iff in Hcond as [Hcond Horder].
  apply andb_true_iff in Hcond as [Hcond Hkinds].
  apply andb_true_iff in Hcond as [_ Hdistinct].
  apply Nat.eqb_eq in Hkinds.
  intros vars conts statics hp rd.
  apply (equivalent_trans _
    (State body vars
       ((k, order_site_closure params metadata handler vars conts) :: conts)
       statics hp rd));
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  apply (equivalent_trans _
    (State (order_args k (use_order params handler) body) vars
       ((k, order_site_closure_ordered params' kinds' handler vars conts)
        :: conts)
       statics hp rd));
    [ | apply equivalent_sym, step_equivalent, step_let_cont_non_recursive ].
  apply
    (order_site_equivalent k (use_order params handler) params params' metadata
       kinds' handler vars conts Hdistinct Hkinds Horder Hparams' Hkinds').
  apply Order_state_in_scope; [ apply agree_refl | exact Happly | ].
  split; [ apply resolve_head | split; [ apply resolve_head | ] ].
  intros k1 Hne. rewrite !(resolve_other k1 k) by exact Hne. apply Order_same.
Qed.

Theorem order_params_equivalent : order_params_equivalent_statement.
Proof. exact (walk_spine_equivalent _ order_node_equivalent). Qed.
