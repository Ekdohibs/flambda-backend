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

(* Proofs of the statement of FlambdaUnaliasParamsSpec.v. Nothing here
   needs review.

   One parameter of one continuation first, then the pass. For the one
   parameter, the term and its result run side by side under a
   relation of states ([site_rel]): the same expression outside the
   scope of the continuation; inside it, an expression and the same
   with the argument dropped, under continuations that hold the handler
   and the handler without the parameter. The two runs go in step
   except where the continuation is applied: there the result takes
   one step more, its [Let]. That the result refines the term is by
   FlambdaSimulationProofs.stuttering_refines. That the term refines
   the result is by [lagging_refines], a simulation in which the first
   run may take a step that the second answers one step later. The
   pass is then by the context lemmas of FlambdaContextProofs.v.

   The one parameter is proved for a parameter that every application
   gives one simple, a variable, a symbol or a constant
   ([all_give_simple]): the check of the pass is the case of a variable
   ([all_give_var_simple]), and FlambdaBindConstParamsProofs.v takes
   the other two from here.

   It is also proved for the one continuation of a recursive
   [Let_cont] ([loop]), for FlambdaUnaliasRecParamsProofs.v. The
   closure of the continuation then holds itself, so an application of
   it is met in its own handler too, and there the argument is the
   parameter and not what the applications outside give. The relation
   therefore carries, for an expression in the scope of the
   continuation, the simple that the applications in that expression
   give ([given]), which holds what the simple given from outside held
   where the [Let_cont] was run. Outside the handler it is the simple
   given from outside. In the handler of a loop it is the parameter:
   the application that entered the handler gave that value, to the
   parameter in the first run, and in the second the [Let] at the head
   of the handler reads it. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaSimulationProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaUnusedParamsProofs
  FlambdaUnaliasParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Simulations in which the first state may take one step more              *)
(* ======================================================================== *)

(* FlambdaSimulationProofs.stuttering_refines is for a second run that
   never takes fewer steps than the first. Here the first may take one
   step that the second does not answer, if its next step is answered:
   [P] relates the first state, one step on, to the second that has not
   moved. *)
Section Lagging.

Variable R P : state -> state -> Prop.

Hypothesis R_step :
  ∀ s s' : state,
    R s s' ->
    match step s with
    | Undefined => True
    | Exit k vs hp => step s' = Exit k vs hp
    | Step t => (∃ t' : state, step s' = Step t' ∧ R t t') ∨ P t s'
    end.

Hypothesis P_step :
  ∀ u s' : state,
    P u s' ->
    match step u with
    | Undefined => True
    | Exit k vs hp => step s' = Exit k vs hp
    | Step t => ∃ t' : state, step s' = Step t' ∧ R t t'
    end.

Lemma lagging_exit (n : nat) :
  ∀ (s s' : state) (k : continuation) (vs : list value) (hp : heap),
    R s s' ∨ P s s' ->
    run n s = Exit k vs hp ->
    ∃ m : nat, run m s' = Exit k vs hp.
Proof.
  induction n as [ | n IH]; intros s s' k vs hp Hrel; simpl;
    [ discriminate | ].
  destruct Hrel as [Hrel | Hrel].
  - pose proof (R_step s s' Hrel) as Hanswer.
    destruct (step s) as [t | k' vs' hp' | ]; [ | | discriminate ].
    + intros Hexit. destruct Hanswer as [(t' & Hstep & Hrel') | Hlag].
      * destruct (IH t t' k vs hp (or_introl Hrel') Hexit) as [m Hm].
        exists (S m). simpl. rewrite Hstep. exact Hm.
      * exact (IH t s' k vs hp (or_intror Hlag) Hexit).
    + intros [= <- <- <-]. exists 1. simpl. rewrite Hanswer. reflexivity.
  - pose proof (P_step s s' Hrel) as Hanswer.
    destruct (step s) as [t | k' vs' hp' | ]; [ | | discriminate ].
    + intros Hexit. destruct Hanswer as (t' & Hstep & Hrel').
      destruct (IH t t' k vs hp (or_introl Hrel') Hexit) as [m Hm].
      exists (S m). simpl. rewrite Hstep. exact Hm.
    + intros [= <- <- <-]. exists 1. simpl. rewrite Hanswer. reflexivity.
Qed.

(* If the second run has ended, so has the first: of two steps of the
   first, one at least uses up one of the second's. *)
Lemma lagging_ends (m : nat) :
  ∀ (s s' : state) (b : behaviour),
    R s s' ∨ P s s' ->
    ended (run m s') b ->
    ∃ n : nat,
      run n s = Undefined
      ∨ ∃ (k : continuation) (vs : list value) (hp : heap),
          run n s = Exit k vs hp.
Proof.
  induction m as [m IH] using (well_founded_induction lt_wf).
  assert (Hlag :
    ∀ (u s' : state) (b : behaviour),
      P u s' ->
      ended (run m s') b ->
      ∃ n : nat,
        run n u = Undefined
        ∨ ∃ (k : continuation) (vs : list value) (hp : heap),
            run n u = Exit k vs hp).
  { intros u s' b Hrel Hended.
    pose proof (P_step u s' Hrel) as Hanswer.
    destruct (step u) as [t | k vs hp | ] eqn:Hstep.
    - destruct Hanswer as (t' & Hstep' & Hrel').
      destruct m as [ | m]; [ destruct b; contradiction | ].
      simpl in Hended. rewrite Hstep' in Hended.
      destruct (IH m (Nat.lt_succ_diag_r m) t t' b (or_introl Hrel') Hended)
        as [n Hn].
      exists (S n). simpl. rewrite Hstep. exact Hn.
    - exists 1. right. exists k, vs, hp. simpl. rewrite Hstep. reflexivity.
    - exists 1. left. simpl. rewrite Hstep. reflexivity. }
  intros s s' b [Hrel | Hrel] Hended; [ | exact (Hlag s s' b Hrel Hended) ].
  pose proof (R_step s s' Hrel) as Hanswer.
  destruct (step s) as [t | k vs hp | ] eqn:Hstep.
  - destruct Hanswer as [(t' & Hstep' & Hrel') | Hrel'].
    + destruct m as [ | m]; [ destruct b; contradiction | ].
      simpl in Hended. rewrite Hstep' in Hended.
      destruct (IH m (Nat.lt_succ_diag_r m) t t' b (or_introl Hrel') Hended)
        as [n Hn].
      exists (S n). simpl. rewrite Hstep. exact Hn.
    + destruct (Hlag t s' b Hrel' Hended) as [n Hn].
      exists (S n). simpl. rewrite Hstep. exact Hn.
  - exists 1. right. exists k, vs, hp. simpl. rewrite Hstep. reflexivity.
  - exists 1. left. simpl. rewrite Hstep. reflexivity.
Qed.

Lemma lagging_refines (s s' : state) : R s s' -> refines s s'.
Proof.
  intros Hrel.
  assert (Hexits :
    ∀ (k : continuation) (vs : list value) (hp : heap),
      behaves s (Exits k vs hp) -> behaves s' (Exits k vs hp)).
  { intros k vs hp [n Hn].
    exact (lagging_exit n s s' k vs hp (or_introl Hrel) Hn). }
  split; [ exact Hexits | ].
  intros b Hterm Hb. destruct (behaves_ended s' b Hterm Hb) as [m Hended].
  destruct (lagging_ends m s s' b (or_introl Hrel) Hended)
    as [n [Hwrong | (k & vs & hp & Hexit)]].
  - right. exists n. exact Hwrong.
  - left. assert (Hs : behaves s (Exits k vs hp)) by (exists n; exact Hexit).
    rewrite (behaviour_unique s' b (Exits k vs hp) Hb (Hexits k vs hp Hs)).
    exact Hs.
Qed.

End Lagging.

(* ======================================================================== *)
(* Lists without an element, and parameters bound without one               *)
(* ======================================================================== *)

Lemma drop_nth_succ {A : Type} (i : nat) (a : A) (l : list A) :
  drop_nth (S i) (a :: l) = a :: drop_nth i l.
Proof. reflexivity. Qed.

Lemma map_drop_nth {A B : Type} (f : A -> B) (i : nat) (l : list A) :
  map f (drop_nth i l) = drop_nth i (map f l).
Proof.
  unfold drop_nth. rewrite map_app, firstn_map, skipn_map. reflexivity.
Qed.

(* Two lists that have an element at a position are the same exactly
   when they are without it and the two elements are the same. *)
Lemma drop_nth_eq {A : Type} (i : nat) :
  ∀ (l l' : list A) (a a' : A),
    nth_error l i = Some a ->
    nth_error l' i = Some a' ->
    l = l' <-> drop_nth i l = drop_nth i l' ∧ a = a'.
Proof.
  induction i as [ | i IH]; intros [ | b l] [ | b' l'] a a'; simpl;
    try discriminate.
  - intros [= ->] [= ->]. unfold drop_nth. simpl.
    split; [ intros [= -> ->]; auto | intros [-> ->]; reflexivity ].
  - intros Ha Ha'. rewrite !drop_nth_succ.
    split.
    + intros [= -> Hl]. destruct (proj1 (IH l l' a a' Ha Ha') Hl) as [-> ->].
      auto.
    + intros [[= -> Hl] Haa].
      rewrite (proj2 (IH l l' a a' Ha Ha') (conj Hl Haa)). reflexivity.
Qed.

Definition agree (vars vars' : env) : Prop :=
  ∀ y : variable, lookup y vars = lookup y vars'.

Lemma agree_refl (vars : env) : agree vars vars.
Proof. intros y. reflexivity. Qed.

Lemma agree_cons (vars vars' : env) (y : variable) (v : value) :
  agree vars vars' -> agree ((y, v) :: vars) ((y, v) :: vars').
Proof. intros Hagree z. rewrite !lookup_cons, Hagree. reflexivity. Qed.

(* Parameters bound over variables that agree. *)
Lemma bind_params_agree (params : list variable) :
  ∀ (vs : list value) (vars vars' : env),
    agree vars vars' ->
    match bind_params params vs vars, bind_params params vs vars' with
    | Some bound, Some bound' => agree bound bound'
    | None, None => True
    | _, _ => False
    end.
Proof.
  induction params as [ | param params IH]; intros [ | v vs] vars vars' Hagree;
    simpl; auto.
  apply IH, agree_cons, Hagree.
Qed.

(* A variable that is not a parameter is what it was. *)
Lemma lookup_bind_params_out (params : list variable) :
  ∀ (vs : list value) (vars bound : env) (y : variable),
    among y params = false ->
    bind_params params vs vars = Some bound ->
    lookup y bound = lookup y vars.
Proof.
  induction params as [ | param params IH]; intros [ | v vs] vars bound y;
    cbn [bind_params]; try discriminate.
  - intros _ [= <-]. reflexivity.
  - rewrite among_cons. intros Hy Hbind.
    apply orb_false_iff in Hy as [Hne Hy]. apply Pos.eqb_neq in Hne.
    rewrite (IH vs _ bound y Hy Hbind). apply lookup_other, Hne.
Qed.

(* A binding moved in front of the parameters, none of which it is. *)
Lemma bind_params_under (params : list variable) :
  ∀ (vs : list value) (vars : env) (p : variable) (v : value),
    among p params = false ->
    match
      bind_params params vs ((p, v) :: vars), bind_params params vs vars
    with
    | Some bound, Some bound' => agree bound ((p, v) :: bound')
    | None, None => True
    | _, _ => False
    end.
Proof.
  induction params as [ | param params IH]; intros [ | w vs] vars p v Hp;
    simpl; auto; [ apply agree_refl | ].
  rewrite among_cons in Hp. apply orb_false_iff in Hp as [Hne Hp].
  apply Pos.eqb_neq in Hne.
  assert (Hswap :
    agree ((param, w) :: (p, v) :: vars) ((p, v) :: (param, w) :: vars)).
  { intros z. rewrite !lookup_cons.
    destruct (Pos.eqb_spec z param) as [-> | _]; [ | reflexivity ].
    destruct (Pos.eqb_spec param p) as [-> | _]; [ contradiction | ].
    reflexivity. }
  pose proof (bind_params_agree params vs _ _ Hswap) as Hbind.
  pose proof (IH vs ((param, w) :: vars) p v Hp) as Hunder.
  destruct (bind_params params vs ((param, w) :: (p, v) :: vars)),
    (bind_params params vs ((p, v) :: (param, w) :: vars)),
    (bind_params params vs ((param, w) :: vars));
    try contradiction; auto.
  intros z. rewrite Hbind. apply Hunder.
Qed.

(* The parameters bound, and the parameters without one bound and then
   that one: the same bindings, where the one is not among the others. *)
Lemma bind_params_drop_nth (i : nat) :
  ∀ (params : list variable) (vs : list value) (vars : env) (p : variable)
    (v : value),
    nth_error params i = Some p ->
    nth_error vs i = Some v ->
    among p (drop_nth i params) = false ->
    match
      bind_params params vs vars,
      bind_params (drop_nth i params) (drop_nth i vs) vars
    with
    | Some bound, Some bound' => agree bound ((p, v) :: bound')
    | None, None => True
    | _, _ => False
    end.
Proof.
  induction i as [ | i IH]; intros [ | param params] [ | w vs] vars p v;
    cbn [nth_error]; try discriminate.
  - intros [= ->] [= ->] Hp. unfold drop_nth in Hp |- *. simpl in Hp |- *.
    exact (bind_params_under params vs vars p v Hp).
  - intros Hparam Hv Hp. rewrite !drop_nth_succ in Hp |- *.
    cbn [bind_params].
    rewrite among_cons in Hp. apply orb_false_iff in Hp as [_ Hp].
    exact (IH params vs ((param, w) :: vars) p v Hparam Hv Hp).
Qed.

(* ======================================================================== *)
(* What the rewrite keeps of the syntax                                     *)
(* ======================================================================== *)

Fixpoint drop_arg_binds_statics (k : continuation) (i : nat) (e : expr)
    {struct e} :
  binds_statics (drop_arg k i e) = binds_statics e
with drop_arg_in_handler_binds_statics (k : continuation) (i : nat)
    (h : continuation_handler) {struct h} :
  handler_binds_statics (drop_arg_in_handler k i h) = handler_binds_statics h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [drop_arg binds_statics]; try reflexivity.
    + apply drop_arg_binds_statics.
    + rewrite drop_arg_binds_statics. reflexivity.
    + rewrite drop_arg_in_handler_binds_statics, drop_arg_binds_statics.
      reflexivity.
    + rewrite drop_arg_binds_statics. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map existsb].
      rewrite drop_arg_in_handler_binds_statics, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [drop_arg_in_handler handler_binds_statics].
    apply drop_arg_binds_statics.
Qed.

Definition drop_handlers (k : continuation) (i : nat)
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(k', h) => (k', drop_arg_in_handler k i h)) handlers.

Lemma drop_handlers_bind_statics (k : continuation) (i : nat)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics (drop_handlers k i handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics, drop_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map existsb]. rewrite drop_arg_in_handler_binds_statics, IH.
  reflexivity.
Qed.

Lemma lookup_drop_handlers (k : continuation) (i : nat) (k1 : continuation)
    (handlers : list (continuation * continuation_handler)) :
  lookup k1 (drop_handlers k i handlers)
  = option_map (drop_arg_in_handler k i) (lookup k1 handlers).
Proof.
  unfold drop_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map]. rewrite !lookup_cons, IH.
  destruct (Pos.eqb k1 k'); reflexivity.
Qed.

Lemma find_arm_drop (k : continuation) (i : nat) (j : Z)
    (arms : list (Z * apply_cont)) :
  find_arm j (map (fun '(n, ac) => (n, drop_arg_of k i ac)) arms)
  = option_map (drop_arg_of k i) (find_arm j arms).
Proof.
  induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
  cbn [map find_arm]. destruct (Z.eqb j n); [ reflexivity | exact IH ].
Qed.

(* The arguments without one, under variables that agree. *)
Lemma eval_simples_drop_nth (i : nat) :
  ∀ (args : list simple) (vars : env) (vs : list value),
    eval_simples vars args = Some vs ->
    eval_simples vars (drop_nth i args) = Some (drop_nth i vs).
Proof.
  induction i as [ | i IH]; intros [ | s args] vars vs; cbn [eval_simples].
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ]; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ] eqn:Hrest;
      [ | discriminate ].
    intros [= <-]. unfold drop_nth. simpl. exact Hrest.
  - intros [= <-]. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ] eqn:Hrest;
      [ | discriminate ].
    intros [= <-]. rewrite !drop_nth_succ. cbn [eval_simples].
    rewrite Hs, (IH args vars rest Hrest). reflexivity.
Qed.

(* The arguments have values where the one at a position has one and
   the others have. *)
Lemma eval_simples_with_nth (i : nat) :
  ∀ (args : list simple) (vars : env) (s : simple) (v : value)
    (rest : list value),
    nth_error args i = Some s ->
    eval_simple vars s = Some v ->
    eval_simples vars (drop_nth i args) = Some rest ->
    ∃ vs : list value,
      eval_simples vars args = Some vs
      ∧ nth_error vs i = Some v
      ∧ drop_nth i vs = rest.
Proof.
  induction i as [ | i IH]; intros [ | s0 args] vars s v rest;
    cbn [nth_error]; try discriminate.
  - intros [= ->] Hs. unfold drop_nth. simpl. intros Hrest.
    rewrite Hs, Hrest. exists (v :: rest). auto.
  - intros Hnth Hs. rewrite drop_nth_succ. simpl.
    destruct (eval_simple vars s0) as [v0 | ]; [ | discriminate ].
    destruct (eval_simples vars (drop_nth i args)) as [rest0 | ] eqn:Hrest0;
      [ | discriminate ].
    intros [= <-].
    destruct (IH args vars s v rest0 Hnth Hs Hrest0) as (vs & -> & Hv & <-).
    exists (v0 :: vs). auto.
Qed.

Lemma eval_simples_nth_value (i : nat) :
  ∀ (args : list simple) (vars : env) (vs : list value) (s : simple),
    eval_simples vars args = Some vs ->
    nth_error args i = Some s ->
    ∃ v : value, eval_simple vars s = Some v ∧ nth_error vs i = Some v.
Proof.
  induction i as [ | i IH]; intros [ | s0 args] vars vs s; simpl;
    try discriminate.
  - destruct (eval_simple vars s0) as [v | ] eqn:Hs0; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ]; [ | discriminate ].
    intros [= <-] [= <-]. exists v. auto.
  - destruct (eval_simple vars s0) as [v | ]; [ | discriminate ].
    destruct (eval_simples vars args) as [rest | ] eqn:Hrest;
      [ | discriminate ].
    intros [= <-] Hnth. exact (IH args vars rest s Hrest Hnth).
Qed.

(* ======================================================================== *)
(* The check, of a variable, a symbol or a constant                         *)
(* ======================================================================== *)

(* A binder of [y] binds what a simple is, if the simple is [y]; and a
   simple is among parameters if it is one of them. A symbol and a
   constant are neither. *)
Definition binds_simple (s : simple) (y : variable) : bool :=
  match s with
  | Var x => Pos.eqb x y
  | Symbol _ | Const _ => false
  end.

Definition simple_among (s : simple) (params : list variable) : bool :=
  match s with
  | Var x => among x params
  | Symbol _ | Const _ => false
  end.

(* [apply_cont_gives_var] and [all_give_var] with a simple where they
   have a variable: every mention of [k] is an application that gives
   the parameter [s], where [s] is still what it was at the [Let_cont]
   of [k] ([ok]). *)
Definition apply_cont_gives_simple (k : continuation) (i : nat) (s : simple)
    (ok : bool) (ac : apply_cont) : bool :=
  match ac_trap_action ac with
  | Some action => negb (Pos.eqb (trap_action_handler action) k)
  | None => true
  end
  && (if Pos.eqb (ac_continuation ac) k
      then
        match nth_error (ac_args ac) i with
        | Some s' => ok && (if simple_eq_dec s' s then true else false)
        | None => false
        end
      else true).

Fixpoint all_give_simple (k : continuation) (i : nat) (s : simple)
    (ok : bool) (e : expr) {struct e} : bool :=
  match e with
  | Let y _ _ body =>
    all_give_simple k i s (ok && negb (binds_simple s y)) body
  | Let_static _ body => all_give_simple k i s ok body
  | Let_cont_non_recursive k' h body =>
    negb (Pos.eqb k' k)
    && handler_all_give_simple k i s ok h
    && all_give_simple k i s ok body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(k', h) =>
         negb (Pos.eqb k' k) && handler_all_give_simple k i s ok h)
      handlers
    && all_give_simple k i s ok body
  | Apply ap =>
    negb (Pos.eqb (ap_continuation ap) k)
    && negb (Pos.eqb (ap_exn_continuation ap) k)
  | Apply_cont ac => apply_cont_gives_simple k i s ok ac
  | Switch _ arms =>
    forallb (fun '(_, ac) => apply_cont_gives_simple k i s ok ac) arms
  | Invalid => true
  end
with handler_all_give_simple (k : continuation) (i : nat) (s : simple)
    (ok : bool) (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params _ handler =>
    all_give_simple k i s (ok && negb (simple_among s params)) handler
  end.

(* The check of the pass is the one of its variable. *)
Lemma apply_cont_gives_var_simple (k : continuation) (i : nat) (x : variable)
    (ok : bool) (ac : apply_cont) :
  apply_cont_gives_var k i x ok ac
  = apply_cont_gives_simple k i (Var x) ok ac.
Proof.
  unfold apply_cont_gives_var, apply_cont_gives_simple. f_equal.
  destruct (Pos.eqb (ac_continuation ac) k); [ | reflexivity ].
  destruct (nth_error (ac_args ac) i) as [[y | sym | c] | ];
    try reflexivity.
  - f_equal. destruct (simple_eq_dec (Var y) (Var x)) as [[= ->] | Hne].
    + apply Pos.eqb_refl.
    + apply Pos.eqb_neq. intros ->. apply Hne. reflexivity.
  - destruct (simple_eq_dec (Symbol sym) (Var x)); [ discriminate | ].
    symmetry. apply andb_false_r.
  - destruct (simple_eq_dec (Const c) (Var x)); [ discriminate | ].
    symmetry. apply andb_false_r.
Qed.

Fixpoint all_give_var_simple (k : continuation) (i : nat) (x : variable)
    (ok : bool) (e : expr) {struct e} :
  all_give_var k i x ok e = all_give_simple k i (Var x) ok e
with handler_all_give_var_simple (k : continuation) (i : nat) (x : variable)
    (ok : bool) (h : continuation_handler) {struct h} :
  handler_all_give_var k i x ok h = handler_all_give_simple k i (Var x) ok h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [all_give_var all_give_simple binds_simple]; try reflexivity.
    + apply all_give_var_simple.
    + apply all_give_var_simple.
    + rewrite handler_all_give_var_simple, all_give_var_simple. reflexivity.
    + rewrite all_give_var_simple. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [forallb]. rewrite handler_all_give_var_simple, IH. reflexivity.
    + apply apply_cont_gives_var_simple.
    + induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [forallb]. rewrite apply_cont_gives_var_simple, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [handler_all_give_var handler_all_give_simple simple_among].
    apply all_give_var_simple.
Qed.

(* What a simple is under one binding more, and under parameters. *)
Lemma eval_simple_cons_other (s : simple) (y : variable) (v : value)
    (vars : env) :
  binds_simple s y = false ->
  eval_simple ((y, v) :: vars) s = eval_simple vars s.
Proof.
  destruct s as [x | sym | c]; cbn [binds_simple eval_simple];
    [ | intros _; reflexivity | intros _; reflexivity ].
  intros Hne. apply Pos.eqb_neq in Hne. apply lookup_other, Hne.
Qed.

Lemma eval_simple_bind_params_out (params : list variable) (vs : list value)
    (vars bound : env) (s : simple) :
  simple_among s params = false ->
  bind_params params vs vars = Some bound ->
  eval_simple bound s = eval_simple vars s.
Proof.
  destruct s as [x | sym | c]; cbn [simple_among eval_simple];
    [ | intros _ _; reflexivity | intros _ _; reflexivity ].
  apply lookup_bind_params_out.
Qed.

(* ======================================================================== *)
(* One parameter of one continuation                                        *)
(* ======================================================================== *)

Section Site.

(* Whether the continuation is the one continuation of a recursive
   [Let_cont]; the continuation, the position of the parameter, the
   variable, the symbol or the constant that every application outside
   the handler gives it, and the handler; the variables and the
   continuations under which the [Let_cont] runs. In the handler of a
   loop every application gives the parameter. The simple is none of
   the parameters that are left; it may be the variable that has the
   name of the parameter itself. *)
Variables (loop : bool) (k : continuation) (i : nat) (s0 : simple)
  (p : variable) (kind : flambda_kind) (params : list variable)
  (metadata : handler_metadata) (handler : expr) (vars0 : env)
  (conts0 : cont_env).

Hypothesis Hparam : nth_error params i = Some p.
Hypothesis Hkind : nth_error (hm_params_kinds metadata) i = Some kind.
Hypothesis Hx : simple_among s0 (drop_nth i params) = false.
Hypothesis Hfresh : among p (drop_nth i params) = false.
Hypothesis Hturns :
  loop = true -> all_give_simple k i (Var p) true handler = true.

(* The handlers that a closure of [k] holds: its own, in a loop. *)
Definition site_group (h : continuation_handler)
    : list (continuation * continuation_handler) :=
  if loop then [(k, h)] else [].

(* What follows the [Let] at the head of the handler of the result:
   in a loop the handler is in the scope of [k]. *)
Definition site_body : expr :=
  if loop then drop_arg k i handler else handler.

Definition site_handler : continuation_handler :=
  Handler params metadata handler.

Definition site_handler_binding : continuation_handler :=
  Handler (drop_nth i params)
    {| hm_params_kinds := drop_nth i (hm_params_kinds metadata);
       hm_is_exn_handler := false |}
    (Let p kind (Simple s0) site_body).

Definition site_closure : cont_value :=
  Closure site_handler vars0 conts0 (site_group site_handler).

Definition site_closure_binding : cont_value :=
  Closure site_handler_binding vars0 conts0
    (site_group site_handler_binding).

(* The continuations under which the two handlers run. *)
Definition site_conts : cont_env :=
  rec_closures (site_group site_handler) vars0 conts0 ++ conts0.

Definition site_conts_binding : cont_env :=
  rec_closures (site_group site_handler_binding) vars0 conts0 ++ conts0.

Definition handlers_give (given : simple) (ok : bool)
    (handlers : list (continuation * continuation_handler)) : bool :=
  forallb
    (fun '(k', h) =>
       negb (Pos.eqb k' k) && handler_all_give_simple k i given ok h)
    handlers.

(* What a continuation name other than [k] stands for in the two runs:
   the same thing; or two closures of one handler, under variables that
   agree and continuations so related; or two closures of a handler in
   the scope of [k] and of that handler with the argument dropped,
   under continuations that have the two closures of [k]; or two so
   related with the same extra arguments; or what is left of the same
   over-application. [given] is the simple that the applications of
   [k] in the handler give, and [ok] says that it holds, where the
   handler was defined, what the simple given from outside held where
   [k] was. *)
Inductive cont_rel : cont_value -> cont_value -> Prop :=
  | Rel_same (c : cont_value) : cont_rel c c
  | Rel_plain (hh : continuation_handler) (dv dv' : env) (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    agree dv dv' ->
    (∀ k1 : continuation, cont_rel (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel (Closure hh dv dc recursive) (Closure hh dv' dc' recursive)
  | Rel_inside (given : simple) (ok : bool) (hh : continuation_handler)
      (dv dv' : env) (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    agree dv dv' ->
    (ok = true -> eval_simple dv given = eval_simple vars0 s0) ->
    handler_all_give_simple k i given ok hh = true ->
    handlers_give given ok recursive = true ->
    resolve k dc = site_closure ->
    resolve k dc' = site_closure_binding ->
    (∀ k1 : continuation,
      k1 <> k -> cont_rel (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel (Closure hh dv dc recursive)
      (Closure (drop_arg_in_handler k i hh) dv' dc'
         (drop_handlers k i recursive))
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

Definition conts_plain (conts conts' : cont_env) : Prop :=
  ∀ k1 : continuation, cont_rel (resolve k1 conts) (resolve k1 conts').

Definition conts_in_site (conts conts' : cont_env) : Prop :=
  resolve k conts = site_closure
  ∧ resolve k conts' = site_closure_binding
  ∧ ∀ k1 : continuation,
      k1 <> k -> cont_rel (resolve k1 conts) (resolve k1 conts').

(* Two states: the same expression, outside the scope of [k]; or, in
   it, an expression and the same with the argument dropped, where
   the applications of [k] in the expression give [given]. *)
Inductive site_rel : state -> state -> Prop :=
  | Rel_state_plain (e : expr) (vars vars' : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) :
    agree vars vars' ->
    conts_plain conts conts' ->
    site_rel (State e vars conts statics hp) (State e vars' conts' statics hp)
  | Rel_state_in_scope (given : simple) (ok : bool) (e : expr)
      (vars vars' : env) (conts conts' : cont_env) (statics : static_env)
      (hp : heap) :
    agree vars vars' ->
    (ok = true -> eval_simple vars given = eval_simple vars0 s0) ->
    all_give_simple k i given ok e = true ->
    conts_in_site conts conts' ->
    site_rel (State e vars conts statics hp)
      (State (drop_arg k i e) vars' conts' statics hp).

Lemma site_rel_refl (s : state) : site_rel s s.
Proof.
  destruct s as [e vars conts statics hp].
  apply Rel_state_plain; [ apply agree_refl | intros k1; apply Rel_same ].
Qed.

Lemma with_extra_args_rel (c c' : cont_value) (extra : list value) :
  cont_rel c c' ->
  cont_rel (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hc. destruct extra; simpl; [ exact Hc | apply Rel_extra, Hc ].
Qed.

(* The closures of the handlers of a recursive [Let_cont]. *)
Lemma rec_conts_plain
    (handlers : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  agree vars vars' ->
  conts_plain conts conts' ->
  conts_plain (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers vars' conts' ++ conts').
Proof.
  intros Hvars Hconts k1. rewrite !resolve_rec_closures.
  destruct (lookup k1 handlers) as [hh | ]; [ | apply Hconts ].
  apply Rel_plain; assumption.
Qed.

Lemma handlers_give_lookup (given : simple) (ok : bool)
    (handlers : list (continuation * continuation_handler)) :
  handlers_give given ok handlers = true ->
  lookup k handlers = None
  ∧ ∀ (k1 : continuation) (hh : continuation_handler),
      lookup k1 handlers = Some hh ->
      handler_all_give_simple k i given ok hh = true.
Proof.
  unfold handlers_give.
  induction handlers as [ | [k' hh'] handlers IH]; cbn [forallb].
  - intros _. split; [ reflexivity | discriminate ].
  - intros Hgive. apply andb_true_iff in Hgive as [Hhead Hrest].
    apply andb_true_iff in Hhead as [Hne Hhh].
    apply negb_true_iff in Hne. destruct (IH Hrest) as [Hk Hothers].
    split.
    + rewrite lookup_cons, Pos.eqb_sym, Hne. exact Hk.
    + intros k1 hh. rewrite lookup_cons.
      destruct (Pos.eqb k1 k'); [ intros [= <-]; exact Hhh | apply Hothers ].
Qed.

Lemma rec_conts_in_site (given : simple) (ok : bool)
    (handlers : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  agree vars vars' ->
  (ok = true -> eval_simple vars given = eval_simple vars0 s0) ->
  handlers_give given ok handlers = true ->
  conts_in_site conts conts' ->
  conts_in_site (rec_closures handlers vars conts ++ conts)
    (rec_closures (drop_handlers k i handlers) vars' conts' ++ conts').
Proof.
  intros Hvars Hok Hgive (Hk & Hk' & Hconts).
  destruct (handlers_give_lookup given ok handlers Hgive)
    as [Hnone Hothers].
  split; [ | split ].
  - rewrite resolve_rec_closures, Hnone. exact Hk.
  - rewrite resolve_rec_closures, lookup_drop_handlers, Hnone. exact Hk'.
  - intros k1 Hne. rewrite !resolve_rec_closures, lookup_drop_handlers.
    destruct (lookup k1 handlers) as [hh | ] eqn:Hlookup; cbn [option_map];
      [ | exact (Hconts k1 Hne) ].
    apply (Rel_inside given ok); try assumption.
    exact (Hothers k1 hh Hlookup).
Qed.

(* The same full call, returning and raising to related continuations. *)
Lemma call_code_rel (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) :
  cont_rel return_to return_to' ->
  cont_rel exn_to exn_to' ->
  result_related site_rel
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp)
    (call_code closure c vs args_arity return_arity return_to' exn_to'
       statics hp).
Proof.
  intros Hreturn Hraise.
  destruct (call_code_cases closure c vs args_arity return_arity statics hp)
    as [Hundefined | (callee_vars & Hcall)].
  - rewrite !Hundefined. exact I.
  - rewrite !Hcall. simpl.
    apply Rel_state_plain; [ apply agree_refl | ].
    intros k'.
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
    (statics : static_env) (hp : heap) :
  cont_rel return_to return_to' ->
  cont_rel exn_to exn_to' ->
  (∀ partial : value,
    result_related site_rel
      (apply_cont_value return_to [partial] statics hp)
      (apply_cont_value return_to' [partial] statics hp)) ->
  result_related site_rel
    (run_call (fun partial => apply_cont_value return_to [partial] statics hp)
       shape return_arity return_to exn_to statics hp)
    (run_call (fun partial => apply_cont_value return_to' [partial] statics hp)
       shape return_arity return_to' exn_to' statics hp).
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
    (hp : heap) :
  cont_rel c c' ->
  ∀ vs : list value,
    result_related site_rel (apply_cont_value c vs statics hp)
      (apply_cont_value c' vs statics hp).
Proof.
  intros Hc.
  induction Hc
    as [c
       | [ps m he] dv dv' dc dc' recursive Hdv Hdc _
       | given ok [ps m he] dv dv' dc dc' recursive Hdv Hok Hgive Hrec Hk Hk'
           Hdc _
       | c c' extra Hc IHc
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise _];
    intros vs.
  - apply result_related_refl, site_rel_refl.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m));
      [ | exact I ].
    pose proof (bind_params_agree ps vs dv dv' Hdv) as Hbind.
    destruct (bind_params ps vs dv) as [bound | ],
      (bind_params ps vs dv') as [bound' | ]; try contradiction;
      [ | exact I ].
    simpl. apply Rel_state_plain; [ exact Hbind | ].
    apply rec_conts_plain; assumption.
  - cbn [drop_arg_in_handler apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m));
      [ | exact I ].
    pose proof (bind_params_agree ps vs dv dv' Hdv) as Hbind.
    destruct (bind_params ps vs dv) as [bound | ] eqn:Hbound,
      (bind_params ps vs dv') as [bound' | ]; try contradiction;
      [ | exact I ].
    simpl.
    apply (Rel_state_in_scope given (ok && negb (simple_among given ps)));
      [ exact Hbind | | | ].
    + intros Hboth. apply andb_true_iff in Hboth as [Hok' Hnot].
      apply negb_true_iff in Hnot.
      rewrite (eval_simple_bind_params_out ps vs dv bound given Hnot Hbound).
      exact (Hok Hok').
    + exact Hgive.
    + apply (rec_conts_in_site given ok); try assumption.
      split; [ exact Hk | split; [ exact Hk' | exact Hdc ] ].
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
    (statics : static_env) (hp : heap) :
  cont_rel return_to return_to' ->
  cont_rel exn_to exn_to' ->
  result_related site_rel
    (call_function callee call_kind vs args_arity return_arity return_to
       exn_to statics hp)
    (call_function callee call_kind vs args_arity return_arity return_to'
       exn_to' statics hp).
Proof.
  intros Hreturn Hraise. unfold call_function.
  apply run_call_rel; [ exact Hreturn | exact Hraise | ].
  intros partial. apply apply_cont_value_rel, Hreturn.
Qed.

(* The two closures of [k], applied. *)
Lemma apply_the_closure (vs : list value) (statics : static_env) (hp : heap) :
  apply_cont_value site_closure vs statics hp
  = if kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata)
    then
      match bind_params params vs vars0 with
      | Some bound => Step (State handler bound site_conts statics hp)
      | None => Undefined
      end
    else Undefined.
Proof. reflexivity. Qed.

Lemma apply_the_closure_binding (vs : list value) (statics : static_env)
    (hp : heap) :
  apply_cont_value site_closure_binding vs statics hp
  = if kinds_eqb (map kind_of_value vs)
         (drop_nth i (hm_params_kinds metadata))
    then
      match bind_params (drop_nth i params) vs vars0 with
      | Some bound =>
        Step
          (State (Let p kind (Simple s0) site_body) bound site_conts_binding
             statics hp)
      | None => Undefined
      end
    else Undefined.
Proof. reflexivity. Qed.

Lemma in_drop_nth {A : Type} (j : nat) :
  ∀ (l : list A) (a : A), a ∈ drop_nth j l -> a ∈ l.
Proof.
  induction j as [ | j IH]; intros [ | b l] a.
  - intros [].
  - unfold drop_nth. simpl. intros Hin. right. exact Hin.
  - intros [].
  - rewrite drop_nth_succ. intros [-> | Hin]; [ left; reflexivity | ].
    right. exact (IH l a Hin).
Qed.

Lemma among_drop_nth_false (j : nat) (y : positive) (l : list positive) :
  among y l = false -> among y (drop_nth j l) = false.
Proof.
  intros Hy. destruct (among y (drop_nth j l)) eqn:Hdrop; [ | reflexivity ].
  apply among_In in Hdrop. apply in_drop_nth, among_In in Hdrop. congruence.
Qed.

Lemma simple_among_drop_nth_false (j : nat) (s : simple)
    (l : list variable) :
  simple_among s l = false -> simple_among s (drop_nth j l) = false.
Proof.
  destruct s as [x | sym | c]; cbn [simple_among];
    [ apply among_drop_nth_false | auto | auto ].
Qed.

(* What the first step of the handler of the second closure of [k]
   reads: the simple, which none of the parameters that are left
   is. *)
Lemma eval_given (rest : list value) (bound : env) :
  bind_params (drop_nth i params) rest vars0 = Some bound ->
  eval_simple bound s0 = eval_simple vars0 s0.
Proof.
  apply eval_simple_bind_params_out, Hx.
Qed.

(* The second state has taken the step of the application of [k], and
   the first has not: either both are then undefined, or each takes one
   step, to related states. *)
Definition pending (s u' : state) : Prop :=
  (step s = Undefined ∧ step u' = Undefined)
  ∨ ∃ t t'' : state, step s = Step t ∧ step u' = Step t'' ∧ site_rel t t''.

(* The application of [k], which gives a simple that holds what the
   simple given from outside held: the first run to all the arguments,
   the second to all but one. *)
Lemma site_applied (given : simple) (vars : env) (rest : list value)
    (statics : static_env) (hp : heap) (args : list simple) :
  nth_error args i = Some given ->
  eval_simple vars given = eval_simple vars0 s0 ->
  eval_simples vars (drop_nth i args) = Some rest ->
  let first :=
    match eval_simples vars args with
    | Some vs => apply_cont_value site_closure vs statics hp
    | None => Undefined
    end
  in
  (first = Undefined
   ∧ apply_cont_value site_closure_binding rest statics hp = Undefined)
  ∨ ∃ bound : env,
      apply_cont_value site_closure_binding rest statics hp
      = Step
          (State (Let p kind (Simple s0) site_body) bound site_conts_binding
             statics hp)
      ∧ ((first = Undefined
          ∧ step
              (State (Let p kind (Simple s0) site_body) bound
                 site_conts_binding statics hp)
            = Undefined)
         ∨ ∃ (bound0 : env) (v : value),
             first = Step (State handler bound0 site_conts statics hp)
             ∧ step
                 (State (Let p kind (Simple s0) site_body) bound
                    site_conts_binding statics hp)
               = Step
                   (State site_body ((p, v) :: bound) site_conts_binding
                      statics hp)
             ∧ agree bound0 ((p, v) :: bound)
             ∧ eval_simple vars0 s0 = Some v).
Proof.
  intros Hnth Hgiven Hrest first. rewrite apply_the_closure_binding.
  destruct (eval_simple vars given) as [v | ] eqn:Hv.
  - (* the simple has a value *)
    destruct (eval_simples_with_nth i args vars given v rest Hnth Hv Hrest)
      as (vs & Hvs & Hnth_v & Hdrop).
    subst first. rewrite Hvs, apply_the_closure. subst rest.
    pose proof
      (drop_nth_eq i (map kind_of_value vs) (hm_params_kinds metadata)
         (kind_of_value v) kind (map_nth_error kind_of_value i vs Hnth_v)
         Hkind)
      as Hkinds.
    rewrite <- map_drop_nth in Hkinds.
    pose proof (bind_params_drop_nth i params vs vars0 p v Hparam Hnth_v
      Hfresh) as Hbind.
    destruct
      (kinds_eqb (map kind_of_value (drop_nth i vs))
         (drop_nth i (hm_params_kinds metadata)))
      eqn:Hrest_kinds.
    + destruct (bind_params (drop_nth i params) (drop_nth i vs) vars0)
        as [bound | ] eqn:Hbound.
      * right. exists bound. split; [ reflexivity | ].
        rewrite step_let. cbn [eval_named].
        rewrite (eval_given _ _ Hbound), <- Hgiven.
        apply kinds_eqb_eq in Hrest_kinds.
        destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind_v.
        -- right. apply kind_eqb_eq in Hkind_v.
           rewrite (proj2 (kinds_eqb_eq _ _)
             (proj2 Hkinds (conj Hrest_kinds Hkind_v))).
           destruct (bind_params params vs vars0) as [bound0 | ];
             [ | contradiction ].
           exists bound0, v.
           split; [ reflexivity | split; [ reflexivity | ] ].
           split; [ exact Hbind | reflexivity ].
        -- left. split; [ | reflexivity ].
           destruct
             (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
             eqn:Hall; [ | reflexivity ].
           apply kinds_eqb_eq in Hall.
           destruct (proj1 Hkinds Hall) as [_ Hsame].
           rewrite Hsame, kind_eqb_refl in Hkind_v. discriminate.
      * left. split; [ | reflexivity ].
        destruct (bind_params params vs vars0); [ contradiction | ].
        destruct
          (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
          reflexivity.
    + left. split; [ | reflexivity ].
      destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
        eqn:Hall; [ | reflexivity ].
      apply kinds_eqb_eq in Hall. destruct (proj1 Hkinds Hall) as [Hsame _].
      rewrite Hsame, kinds_eqb_refl in Hrest_kinds. discriminate.
  - (* the simple has none *)
    assert (Hfirst : first = Undefined).
    { subst first.
      destruct (eval_simples vars args) as [vs | ] eqn:Hvs; [ | reflexivity ].
      destruct (eval_simples_nth_value i args vars vs given Hvs Hnth)
        as (v & Hv' & _).
      congruence. }
    destruct
      (kinds_eqb (map kind_of_value rest)
         (drop_nth i (hm_params_kinds metadata)));
      [ | left; auto ].
    destruct (bind_params (drop_nth i params) rest vars0) as [bound | ]
      eqn:Hbound; [ | left; auto ].
    right. exists bound. split; [ reflexivity | ]. left.
    split; [ exact Hfirst | ].
    rewrite step_let. cbn [eval_named].
    rewrite (eval_given _ _ Hbound), <- Hgiven. reflexivity.
Qed.

(* The continuations under which the handler of a loop runs have the
   two closures of [k], and are otherwise those of the [Let_cont]. *)
Lemma loop_conts_in_site :
  loop = true -> conts_in_site site_conts site_conts_binding.
Proof.
  intros Hloop.
  unfold conts_in_site, site_conts, site_conts_binding, site_closure,
    site_closure_binding, site_group.
  rewrite Hloop, !resolve_rec_closures, !lookup_cons, Pos.eqb_refl.
  split; [ reflexivity | split; [ reflexivity | ] ].
  intros k1 Hne. rewrite !resolve_rec_closures, !lookup_cons.
  destruct (Pos.eqb_spec k1 k) as [-> | _]; [ contradiction | ].
  apply Rel_same.
Qed.

(* The two handlers entered, the second past its [Let]. Where [k] is
   not a loop its handler is outside its scope. In a loop the
   applications of [k] in the handler give the parameter, which holds
   what was given. *)
Lemma site_entered (bound0 bound : env) (v : value) (statics : static_env)
    (hp : heap) :
  agree bound0 ((p, v) :: bound) ->
  eval_simple vars0 s0 = Some v ->
  site_rel (State handler bound0 site_conts statics hp)
    (State site_body ((p, v) :: bound) site_conts_binding statics hp).
Proof.
  intros Hb Hv. destruct loop eqn:Hloop.
  - unfold site_body. rewrite Hloop.
    apply (Rel_state_in_scope (Var p) true handler);
      [ exact Hb | | exact (Hturns eq_refl)
      | exact (loop_conts_in_site Hloop) ].
    intros _. cbn [eval_simple].
    rewrite (Hb p), lookup_cons, Pos.eqb_refl. symmetry. exact Hv.
  - unfold site_body, site_conts, site_conts_binding, site_group.
    rewrite Hloop. apply Rel_state_plain; [ exact Hb | ].
    intros k1. apply Rel_same.
Qed.

(* Two related states step together, or the first is at an application
   of [k]. *)
Lemma unalias_step (s s' : state) :
  site_rel s s' ->
  result_related site_rel (step s) (step s')
  ∨ ∃ u' : state, step s' = Step u' ∧ pending s u'.
Proof.
  intros
    [e vars vars' conts conts' statics hp Hvars Hconts
    | given ok e vars vars' conts conts' statics hp Hvars Hok Hgive
        (Hk & Hk' & Hconts)].
  - (* outside the scope of [k] *)
    left.
    destruct e as [y kind1 d body | bindings body | k1 hh body
                  | handlers body | ap | ac | scrutinee arms | ].
    + rewrite !step_let,
        (eval_named_agree statics vars vars' d (fun z _ => Hvars z)).
      destruct (eval_named statics vars d) as [v | ]; [ | exact I ].
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      simpl. apply Rel_state_plain; [ apply agree_cons, Hvars | exact Hconts ].
    + rewrite !step_let_static,
        (eval_static_bindings_agree vars vars' bindings (fun z _ => Hvars z)
           statics).
      destruct (eval_static_bindings vars bindings statics); [ | exact I ].
      simpl. apply Rel_state_plain; assumption.
    + rewrite !step_let_cont_non_recursive. simpl.
      apply Rel_state_plain; [ exact Hvars | ].
      intros k2. destruct (Pos.eq_dec k2 k1) as [-> | Hne].
      * rewrite !resolve_head. apply Rel_plain; assumption.
      * rewrite !(resolve_other k2 k1) by exact Hne. apply Hconts.
    + rewrite !step_let_cont_recursive.
      destruct (handlers_bind_statics handlers); [ exact I | ].
      simpl. apply Rel_state_plain; [ exact Hvars | ].
      apply rec_conts_plain; assumption.
    + rewrite !step_apply,
        (eval_simple_agree vars vars' (ap_callee ap) (fun z _ => Hvars z)),
        (eval_simples_agree vars vars' (ap_args ap) (fun z _ => Hvars z)),
        (eval_simples_agree vars vars' (exn_extra_simples ap)
           (fun z _ => Hvars z)).
      destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
      destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
      destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
        [ | exact I ].
      apply call_function_rel;
        [ apply Hconts | apply with_extra_args_rel, Hconts ].
    + rewrite !step_apply_cont,
        (eval_simples_agree vars vars' (ac_args ac) (fun z _ => Hvars z)).
      destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
      unfold apply_continuation. apply apply_cont_value_rel, Hconts.
    + rewrite !step_switch,
        (eval_simple_agree vars vars' scrutinee (fun z _ => Hvars z)).
      destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
      destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
      destruct (find_arm j arms) as [ac | ]; [ | exact I ].
      simpl. apply Rel_state_plain; assumption.
    + rewrite !step_invalid. exact I.
  - (* in the scope of [k] *)
    assert (Hinside : conts_in_site conts conts')
      by (split; [ exact Hk | split; [ exact Hk' | exact Hconts ] ]).
    destruct e as [y kind1 d body | bindings body | k1 hh body
                  | handlers body | ap | ac | scrutinee arms | ];
      cbn [drop_arg all_give_simple] in Hgive |- *.
    + left.
      rewrite !step_let,
        (eval_named_agree statics vars vars' d (fun z _ => Hvars z)).
      destruct (eval_named statics vars d) as [v | ]; [ | exact I ].
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      simpl.
      apply (Rel_state_in_scope given (ok && negb (binds_simple given y)));
        [ apply agree_cons, Hvars | | exact Hgive | exact Hinside ].
      intros Hboth. apply andb_true_iff in Hboth as [Hok' Hne].
      apply negb_true_iff in Hne.
      rewrite <- (Hok Hok'). apply eval_simple_cons_other, Hne.
    + left.
      rewrite !step_let_static,
        (eval_static_bindings_agree vars vars' bindings (fun z _ => Hvars z)
           statics).
      destruct (eval_static_bindings vars bindings statics); [ | exact I ].
      simpl. apply (Rel_state_in_scope given ok); assumption.
    + left. rewrite !step_let_cont_non_recursive. simpl.
      apply andb_true_iff in Hgive as [Hgive Hbody].
      apply andb_true_iff in Hgive as [Hne Hhh].
      apply negb_true_iff, Pos.eqb_neq in Hne.
      apply (Rel_state_in_scope given ok);
        [ exact Hvars | exact Hok | exact Hbody | ].
      split; [ | split ].
      * rewrite resolve_other by congruence. exact Hk.
      * rewrite resolve_other by congruence. exact Hk'.
      * intros k2 Hk2. destruct (Pos.eq_dec k2 k1) as [-> | Hne'].
        -- rewrite !resolve_head.
           exact (Rel_inside given ok hh vars vars' conts conts' [] Hvars
             Hok Hhh eq_refl Hk Hk' Hconts).
        -- rewrite !(resolve_other k2 k1) by exact Hne'.
           exact (Hconts k2 Hk2).
    + left. rewrite !step_let_cont_recursive.
      fold (drop_handlers k i handlers).
      rewrite drop_handlers_bind_statics.
      destruct (handlers_bind_statics handlers); [ exact I | ].
      simpl. apply andb_true_iff in Hgive as [Hhandlers Hbody].
      apply (Rel_state_in_scope given ok);
        [ exact Hvars | exact Hok | exact Hbody | ].
      apply (rec_conts_in_site given ok); assumption.
    + left. apply andb_true_iff in Hgive as [Hreturn Hraise].
      apply negb_true_iff, Pos.eqb_neq in Hreturn, Hraise.
      rewrite !step_apply,
        (eval_simple_agree vars vars' (ap_callee ap) (fun z _ => Hvars z)),
        (eval_simples_agree vars vars' (ap_args ap) (fun z _ => Hvars z)),
        (eval_simples_agree vars vars' (exn_extra_simples ap)
           (fun z _ => Hvars z)).
      destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
      destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
      destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
        [ | exact I ].
      apply call_function_rel;
        [ apply Hconts, Hreturn | apply with_extra_args_rel, Hconts, Hraise ].
    + (* an application of a continuation *)
      unfold apply_cont_gives_simple in Hgive.
      apply andb_true_iff in Hgive as [_ Hgive].
      unfold drop_arg_of.
      destruct (Pos.eqb_spec (ac_continuation ac) k) as [Hac | Hac].
      * (* of [k] *)
        destruct (nth_error (ac_args ac) i) as [arg | ] eqn:Hnth;
          [ | discriminate ].
        apply andb_true_iff in Hgive as [Hok' Hy].
        destruct (simple_eq_dec arg given) as [-> | ]; [ | discriminate ].
        specialize (Hok Hok').
        rewrite !step_apply_cont. cbn [ac_args ac_continuation].
        rewrite
          (eval_simples_agree vars vars' (drop_nth i (ac_args ac))
             (fun z _ => Hvars z)).
        unfold apply_continuation. rewrite Hac, Hk, Hk'.
        destruct (eval_simples vars (drop_nth i (ac_args ac))) as [rest | ]
          eqn:Hrest.
        -- destruct
             (site_applied given vars rest statics hp (ac_args ac) Hnth Hok
                Hrest)
             as [[Hfirst Hsecond]
                | (bound & Hsecond
                   & [[Hfirst Hnext]
                     | (bound0 & v & Hfirst & Hnext & Hb & Hv)])].
           ++ left. rewrite Hfirst, Hsecond. exact I.
           ++ right. eexists. split; [ exact Hsecond | ].
              left. rewrite step_apply_cont. unfold apply_continuation.
              rewrite Hac, Hk. auto.
           ++ right. eexists. split; [ exact Hsecond | ].
              right. eexists. eexists. rewrite step_apply_cont.
              unfold apply_continuation. rewrite Hac, Hk.
              split; [ exact Hfirst | split; [ exact Hnext | ] ].
              exact (site_entered bound0 bound v statics hp Hb Hv).
        -- left.
           destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
             [ | exact I ].
           rewrite (eval_simples_drop_nth i _ _ _ Hvs) in Hrest.
           discriminate.
      * left.
        rewrite !step_apply_cont,
          (eval_simples_agree vars vars' (ac_args ac) (fun z _ => Hvars z)).
        destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
        unfold apply_continuation. apply apply_cont_value_rel, Hconts, Hac.
    + left.
      rewrite !step_switch,
        (eval_simple_agree vars vars' scrutinee (fun z _ => Hvars z)).
      destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
      destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
      rewrite find_arm_drop.
      destruct (find_arm j arms) as [ac | ] eqn:Hfind; [ | exact I ].
      simpl.
      apply (Rel_state_in_scope given ok (Apply_cont ac));
        [ exact Hvars | exact Hok | | exact Hinside ].
      rewrite forallb_forall in Hgive.
      exact (Hgive (j, ac) (find_arm_in j arms ac Hfind)).
    + left. rewrite !step_invalid. exact I.
Qed.

(* The second run answers the first with one step or two. *)
Lemma site_refines (s s' : state) : site_rel s s' -> refines s s'.
Proof.
  apply (stuttering_refines site_rel (fun _ vs vs' => vs' = vs));
    [ | intros last vs vs' Hsame; exact Hsame ].
  clear s s'. intros s s' Hrel.
  destruct (unalias_step s s' Hrel)
    as [Hlock
       | (u' & Hstep' & [[Hs Hu] | (t & t'' & Hs & Hu & Hrel')])].
  - destruct (step s) as [t | k1 vs hp1 | ],
      (step s') as [t' | k1' vs' hp1' | ] eqn:Hs';
      simpl in Hlock; try contradiction; unfold answered.
    + exists 0, t'. split; [ simpl; rewrite Hs'; reflexivity | exact Hlock ].
    + destruct Hlock as (<- & <- & <-). exists vs. auto.
    + exact I.
  - rewrite Hs. exact I.
  - rewrite Hs. exists 1, t''. split; [ | exact Hrel' ].
    simpl. rewrite Hstep', Hu. reflexivity.
Qed.

(* The first run answers the second, which takes one step more at an
   application of [k]. *)
Lemma site_refined (s s' : state) : site_rel s s' -> refines s' s.
Proof.
  intros Hrel.
  apply
    (lagging_refines (fun a b => site_rel b a) (fun u a => pending a u));
    [ | | exact Hrel ]; clear s s' Hrel.
  - intros s' s Hrel.
    destruct (unalias_step s s' Hrel) as [Hlock | (u' & Hstep' & Hpending)].
    + destruct (step s) as [t | k1 vs hp1 | ],
        (step s') as [t' | k1' vs' hp1' | ];
        simpl in Hlock; try contradiction.
      * left. exists t. auto.
      * destruct Hlock as (<- & <- & <-). reflexivity.
      * exact I.
    + rewrite Hstep'. right. exact Hpending.
  - intros u' s [[Hs Hu] | (t & t'' & Hs & Hu & Hrel')].
    + rewrite Hu. exact I.
    + rewrite Hu. exists t. auto.
Qed.

Lemma simple_site_equivalent (s s' : state) : site_rel s s' -> equivalent s s'.
Proof.
  intros Hrel.
  exact (refines_antisymmetric s s' (site_refines s s' Hrel)
    (site_refined s s' Hrel)).
Qed.

End Site.

(* The two closures of a continuation that is not a loop. *)
Definition the_closure (params : list variable) (metadata : handler_metadata)
    (handler : expr) (vars0 : env) (conts0 : cont_env) : cont_value :=
  Closure (Handler params metadata handler) vars0 conts0 [].

Definition the_closure_binding (i : nat) (s0 : simple) (p : variable)
    (kind : flambda_kind) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (vars0 : env)
    (conts0 : cont_env) : cont_value :=
  Closure
    (Handler (drop_nth i params)
       {| hm_params_kinds := drop_nth i (hm_params_kinds metadata);
          hm_is_exn_handler := false |}
       (Let p kind (Simple s0) handler))
    vars0 conts0 [].

Lemma not_a_loop (P : Prop) : false = true -> P.
Proof. discriminate. Qed.

(* The relations for a continuation that is not a loop, under the
   names and with the arguments that they had when the site was of
   such a continuation only. *)
Definition conts_inside (k : continuation) (i : nat) (s0 : simple)
    (p : variable) (kind : flambda_kind) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (vars0 : env)
    (conts0 conts conts' : cont_env) : Prop :=
  conts_in_site false k i s0 p kind params metadata handler vars0 conts0
    conts conts'.

Definition state_rel (k : continuation) (i : nat) (s0 : simple)
    (p : variable) (kind : flambda_kind) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (vars0 : env)
    (conts0 : cont_env) (s s' : state) : Prop :=
  site_rel false k i s0 p kind params metadata handler vars0 conts0 s s'.

(* ======================================================================== *)
(* The site, of the terms                                                   *)
(* ======================================================================== *)

(* A parameter that every application of a continuation gives one
   simple, which no binder on the way binds and which is not a
   parameter of the handler, is taken away and bound at the head of
   the handler. *)
Lemma given_param_equivalent (i : nat) (k : continuation) (s : simple)
    (p : variable) (kind : flambda_kind) (params : list variable)
    (metadata : handler_metadata) (handler body : expr) :
  nth_error params i = Some p ->
  nth_error (hm_params_kinds metadata) i = Some kind ->
  simple_among s params = false ->
  among p (drop_nth i params) = false ->
  all_give_simple k i s true body = true ->
  expr_equivalent
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k
       (Handler (drop_nth i params)
          {| hm_params_kinds := drop_nth i (hm_params_kinds metadata);
             hm_is_exn_handler := false |}
          (Let p kind (Simple s) handler))
       (drop_arg k i body)).
Proof.
  intros Hparam Hkind Hs Hfresh Hgive vars conts statics hp.
  apply (equivalent_trans _
    (State body vars
       ((k, the_closure params metadata handler vars conts) :: conts)
       statics hp));
    [ apply step_equivalent, step_let_cont_non_recursive | ].
  apply (equivalent_trans _
    (State (drop_arg k i body) vars
       ((k, the_closure_binding i s p kind params metadata handler vars conts)
        :: conts)
       statics hp));
    [ | apply equivalent_sym, step_equivalent, step_let_cont_non_recursive ].
  apply
    (simple_site_equivalent false k i s p kind params metadata handler vars
       conts Hparam Hkind (simple_among_drop_nth_false i s params Hs) Hfresh
       (not_a_loop _)).
  apply (Rel_state_in_scope false k i s p kind params metadata handler vars
    conts s true);
    [ apply agree_refl | reflexivity | exact Hgive | ].
  split; [ apply resolve_head | split; [ apply resolve_head | ] ].
  intros k1 Hne. rewrite !(resolve_other k1 k) by exact Hne. apply Rel_same.
Qed.

(* The site of a variable, under the names that it had when the site
   was of a variable only: FlambdaLiftContsProofs.v uses them. *)
Definition the_closure' (i : nat) (x p : variable) (kind : flambda_kind)
    (params : list variable) (metadata : handler_metadata) (handler : expr)
    (vars0 : env) (conts0 : cont_env) : cont_value :=
  the_closure_binding i (Var x) p kind params metadata handler vars0 conts0.

Lemma Rel_state_inside (k : continuation) (i : nat) (x p : variable)
    (kind : flambda_kind) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (vars0 : env)
    (conts0 : cont_env) (ok : bool) (e : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics : static_env) (hp : heap) :
  agree vars vars' ->
  (ok = true -> lookup x vars = lookup x vars0) ->
  all_give_var k i x ok e = true ->
  conts_inside k i (Var x) p kind params metadata handler vars0 conts0 conts
    conts' ->
  state_rel k i (Var x) p kind params metadata handler vars0 conts0
    (State e vars conts statics hp)
    (State (drop_arg k i e) vars' conts' statics hp).
Proof.
  intros Hvars Hok Hgive Hinside. rewrite all_give_var_simple in Hgive.
  exact (Rel_state_in_scope false k i (Var x) p kind params metadata handler
    vars0 conts0 (Var x) ok e vars vars' conts conts' statics hp Hvars Hok
    Hgive Hinside).
Qed.

Lemma site_equivalent (k : continuation) (i : nat) (x p : variable)
    (kind : flambda_kind) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (vars0 : env)
    (conts0 : cont_env) :
  nth_error params i = Some p ->
  nth_error (hm_params_kinds metadata) i = Some kind ->
  among x params = false ->
  among p (drop_nth i params) = false ->
  ∀ s s' : state,
    state_rel k i (Var x) p kind params metadata handler vars0 conts0 s s' ->
    equivalent s s'.
Proof.
  intros Hparam Hkind Hx Hfresh.
  exact (simple_site_equivalent false k i (Var x) p kind params metadata
    handler vars0 conts0 Hparam Hkind
    (simple_among_drop_nth_false i (Var x) params Hx) Hfresh (not_a_loop _)).
Qed.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

Lemma unalias_param_equivalent (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr) (h' : continuation_handler)
    (body' : expr) :
  unalias_param i k h body = Some (h', body') ->
  expr_equivalent (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h' body').
Proof.
  destruct h as [params metadata handler]. unfold unalias_param.
  destruct (nth_error params i) as [p | ] eqn:Hparam; [ | discriminate ].
  destruct (nth_error (hm_params_kinds metadata) i) as [kind | ] eqn:Hkind;
    [ | discriminate ].
  destruct (first_given_var k i body) as [[x | ] | ]; try discriminate.
  destruct
    (negb (hm_is_exn_handler metadata) && all_give_var k i x true body
     && negb (among x params) && negb (among p (drop_nth i params)))
    eqn:Hcond; [ | discriminate ].
  intros [= <- <-].
  apply andb_true_iff in Hcond as [Hcond Hfresh].
  apply andb_true_iff in Hcond as [Hcond Hx].
  apply andb_true_iff in Hcond as [_ Hgive].
  apply negb_true_iff in Hfresh, Hx.
  rewrite all_give_var_simple in Hgive.
  exact (given_param_equivalent i k (Var x) p kind params metadata handler
    body Hparam Hkind Hx Hfresh Hgive).
Qed.

Lemma unalias_from_equivalent (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    expr_equivalent (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k (fst (unalias_from i k h body))
         (snd (unalias_from i k h body))).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [unalias_from].
  - intros vars conts statics hp. apply equivalent_refl.
  - destruct (unalias_param i k h body) as [[h' body'] | ] eqn:Hparam;
      [ | apply IH ].
    exact (expr_equivalent_trans _ _ _
      (unalias_param_equivalent i k h body h' body' Hparam) (IH k h' body')).
Qed.

(* Each layer is a context around what the induction gives; then the
   parameters of the [Let_cont] are looked at. *)
Fixpoint unalias_params_equivalent_aux (e : expr) :
  expr_equivalent e (unalias_params e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [unalias_params];
    try (intros vars conts statics hp; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (unalias_params_equivalent_aux body)).
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (unalias_params_equivalent_aux body)).
  - assert (Hlayers :
      expr_equivalent
        (Let_cont_non_recursive k (Handler params metadata handler) body)
        (Let_cont_non_recursive k
           (Handler params metadata (unalias_params handler))
           (unalias_params body))).
    { apply (expr_equivalent_trans _
        (Let_cont_non_recursive k
           (Handler params metadata (unalias_params handler)) body)).
      - exact (plug_equivalent
          (Let_cont_non_recursive_handler k params metadata Hole body) _ _
          (not_many _) (unalias_params_equivalent_aux handler)).
      - exact (plug_equivalent
          (Let_cont_non_recursive_body k
             (Handler params metadata (unalias_params handler)) Hole)
          _ _ (not_many _) (unalias_params_equivalent_aux body)). }
    pose proof
      (unalias_from_equivalent (length params) k
         (Handler params metadata (unalias_params handler))
         (unalias_params body))
      as Hfrom.
    destruct
      (unalias_from (length params) k
         (Handler params metadata (unalias_params handler))
         (unalias_params body))
      as [h' body'].
    exact (expr_equivalent_trans _ _ _ Hlayers Hfrom).
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole) _ _
      (not_many _) (unalias_params_equivalent_aux body)).
Qed.

Theorem unalias_params_equivalent : unalias_params_equivalent_statement.
Proof. exact unalias_params_equivalent_aux. Qed.
