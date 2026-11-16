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

(* Proofs of the statement of FlambdaWrapReturnsSpec.v. Nothing here
   needs review.

   One parameter of one continuation first, then the pass. For the one
   parameter, the term and its result run side by side under a
   relation of states ([state_rel]): the same expression outside the
   scope of the continuation; inside it, an expression and the same
   with the arguments dropped and the calls wrapped, under
   continuations that hold the handler and the handler without the
   parameter; and, once the handler has been entered, two states that
   the comparison of FlambdaCompareProofs.v matches: the handler under
   variables that differ at the parameter only, which it does not
   use. The variables are the same on the two sides until then. The
   result takes one step more where a call returns to the wrapper,
   to bind it, and one more where the wrapper is applied; so that the
   result refines the term is by
   FlambdaSimulationProofs.stuttering_refines. The pass is then by the
   context lemmas of FlambdaContextProofs.v. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaSimulationProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaNominalSpec FlambdaCompareSpec FlambdaCompareProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaUnaliasParamsSpec
  FlambdaUnaliasParamsProofs FlambdaWrapReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What the rewrite keeps of the syntax                                     *)
(* ======================================================================== *)

Fixpoint wrap_uses_binds_statics (k : continuation) (i : nat)
    (w : continuation_handler) (Hw : handler_binds_statics w = false)
    (e : expr) {struct e} :
  binds_statics (wrap_uses k i w e) = binds_statics e
with wrap_uses_in_handler_binds_statics (k : continuation) (i : nat)
    (w : continuation_handler) (Hw : handler_binds_statics w = false)
    (h : continuation_handler) {struct h} :
  handler_binds_statics (wrap_uses_in_handler k i w h)
  = handler_binds_statics h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [wrap_uses binds_statics]; try reflexivity.
    + apply wrap_uses_binds_statics, Hw.
    + rewrite (wrap_uses_binds_statics k i w Hw). reflexivity.
    + rewrite (wrap_uses_in_handler_binds_statics k i w Hw),
        (wrap_uses_binds_statics k i w Hw).
      reflexivity.
    + rewrite (wrap_uses_binds_statics k i w Hw). f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map existsb].
      rewrite (wrap_uses_in_handler_binds_statics k i w Hw), IH. reflexivity.
    + destruct (Pos.eqb (ap_continuation ap) k); [ | reflexivity ].
      cbn [binds_statics]. rewrite Hw. reflexivity.
  - destruct h as [params metadata handler].
    cbn [wrap_uses_in_handler handler_binds_statics].
    apply wrap_uses_binds_statics, Hw.
Qed.

Definition wrap_handlers (k : continuation) (i : nat)
    (w : continuation_handler)
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(k', h) => (k', wrap_uses_in_handler k i w h)) handlers.

Lemma wrap_handlers_bind_statics (k : continuation) (i : nat)
    (w : continuation_handler)
    (handlers : list (continuation * continuation_handler)) :
  handler_binds_statics w = false ->
  handlers_bind_statics (wrap_handlers k i w handlers)
  = handlers_bind_statics handlers.
Proof.
  intros Hw. unfold handlers_bind_statics, wrap_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map existsb].
  rewrite (wrap_uses_in_handler_binds_statics k i w Hw), IH. reflexivity.
Qed.

Lemma lookup_wrap_handlers (k : continuation) (i : nat)
    (w : continuation_handler) (k1 : continuation)
    (handlers : list (continuation * continuation_handler)) :
  lookup k1 (wrap_handlers k i w handlers)
  = option_map (wrap_uses_in_handler k i w) (lookup k1 handlers).
Proof.
  unfold wrap_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map]. rewrite !lookup_cons, IH.
  destruct (Pos.eqb k1 k'); reflexivity.
Qed.

(* ======================================================================== *)
(* Parameters that are distinct                                             *)
(* ======================================================================== *)

(* The parameters, read where they are bound, are the values. *)
Lemma eval_params_bound (params : list variable) :
  ∀ (vs : list value) (vars bound : env),
    distinct params = true ->
    bind_params params vs vars = Some bound ->
    eval_simples bound (map Var params) = Some vs.
Proof.
  induction params as [ | x params IH]; intros [ | v vs] vars bound;
    simpl; try discriminate.
  - intros _ _. reflexivity.
  - intros Hdistinct Hbound. apply andb_true_iff in Hdistinct as [Hx Hrest].
    apply negb_true_iff in Hx.
    rewrite (lookup_bind_params_out params vs ((x, v) :: vars) bound x Hx
      Hbound), lookup_cons, Pos.eqb_refl.
    simpl. rewrite (IH vs ((x, v) :: vars) bound Hrest Hbound). reflexivity.
Qed.

Lemma bind_params_length (params : list variable) :
  ∀ (vs : list value) (vars bound : env),
    bind_params params vs vars = Some bound -> length vs = length params.
Proof.
  induction params as [ | x params IH]; intros [ | v vs] vars bound;
    simpl; try discriminate; [ reflexivity | ].
  intros Hbound. f_equal. exact (IH vs _ bound Hbound).
Qed.

(* A parameter is not among the others. *)
Lemma distinct_drop_nth (i : nat) :
  ∀ (params : list variable) (p : variable),
    distinct params = true ->
    nth_error params i = Some p ->
    among p (drop_nth i params) = false.
Proof.
  induction i as [ | i IH]; intros [ | x params] p Hdistinct Hnth;
    try (simpl in Hnth; discriminate Hnth); cbn [distinct] in Hdistinct;
    apply andb_true_iff in Hdistinct as [Hx Hrest];
    apply negb_true_iff in Hx.
  - simpl in Hnth. injection Hnth as <-. exact Hx.
  - cbn [nth_error] in Hnth. rewrite drop_nth_succ, among_cons.
    rewrite (IH params p Hrest Hnth), orb_false_r.
    apply Pos.eqb_neq. intros ->.
    apply nth_error_In in Hnth. apply among_In in Hnth. congruence.
Qed.

(* ======================================================================== *)
(* One parameter of one continuation                                        *)
(* ======================================================================== *)

Section Site.

(* The continuation, the position of the parameter, and the handler;
   the variables and the continuations under which the [Let_cont]
   runs. *)
Variables (k : continuation) (i : nat) (p : variable)
  (params : list variable) (metadata : handler_metadata) (handler : expr)
  (vars0 : env) (conts0 : cont_env).

Hypothesis Hparam : nth_error params i = Some p.
Hypothesis Hdistinct : distinct params = true.
Hypothesis Hunused : var_free_b p handler = false.

Definition the_wrapper : continuation_handler := wrapper k i params metadata.

Definition the_closure : cont_value :=
  Closure (Handler params metadata handler) vars0 conts0 [].

Definition the_closure' : cont_value :=
  Closure
    (Handler (drop_nth i params)
       {| hm_params_kinds := drop_nth i (hm_params_kinds metadata);
          hm_is_exn_handler := false |}
       handler)
    vars0 conts0 [].

Definition handlers_ok
    (handlers : list (continuation * continuation_handler)) : bool :=
  forallb
    (fun '(k', h) =>
       negb (Pos.eqb k' k) && handler_only_applied_or_returned_to k h)
    handlers.

(* What a continuation name other than [k] stands for in the two runs:
   the same thing; or two closures of one handler, under the same
   variables and continuations so related; or two closures of a
   handler in the scope of [k] and of that handler rewritten, under
   continuations that have the two closures of [k]; or the first
   closure of [k] and a closure of the wrapper, under continuations
   that have the second closure of [k]; or two so related with the
   same extra arguments; or what is left of the same
   over-application. *)
Inductive cont_rel : cont_value -> cont_value -> Prop :=
  | Rel_same (c : cont_value) : cont_rel c c
  | Rel_plain (hh : continuation_handler) (dv : env) (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    (∀ k1 : continuation, cont_rel (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel (Closure hh dv dc recursive) (Closure hh dv dc' recursive)
  | Rel_inside (hh : continuation_handler) (dv : env) (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    handler_only_applied_or_returned_to k hh = true ->
    handlers_ok recursive = true ->
    resolve k dc = the_closure ->
    resolve k dc' = the_closure' ->
    (∀ k1 : continuation,
      k1 <> k -> cont_rel (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel (Closure hh dv dc recursive)
      (Closure (wrap_uses_in_handler k i the_wrapper hh) dv dc'
         (wrap_handlers k i the_wrapper recursive))
  | Rel_wrapper (dv : env) (dc : cont_env) :
    resolve k dc = the_closure' ->
    cont_rel the_closure (Closure the_wrapper dv dc [])
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

Definition conts_inside (conts conts' : cont_env) : Prop :=
  resolve k conts = the_closure
  ∧ resolve k conts' = the_closure'
  ∧ ∀ k1 : continuation,
      k1 <> k -> cont_rel (resolve k1 conts) (resolve k1 conts').

(* Two states: the same expression, outside the scope of [k]; or, in
   it, an expression and the same rewritten; or two that the
   comparison matches, once the handler of [k] has been entered. *)
Inductive state_rel : state -> state -> Prop :=
  | Rel_state_plain (e : expr) (vars : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) :
    conts_plain conts conts' ->
    state_rel (State e vars conts statics hp rd)
      (State e vars conts' statics hp rd)
  | Rel_state_inside (e : expr) (vars : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) :
    only_applied_or_returned_to k e = true ->
    conts_inside conts conts' ->
    state_rel (State e vars conts statics hp rd)
      (State (wrap_uses k i the_wrapper e) vars conts' statics hp rd)
  | Rel_state_matched (s s' : state) : state_matches s s' -> state_rel s s'.

Lemma state_rel_refl (s : state) : state_rel s s.
Proof.
  destruct s as [e vars conts statics hp rd].
  apply Rel_state_plain. intros k1. apply Rel_same.
Qed.

(* What a step of the first run comes to, and what the step of the
   second comes to: nothing is asked if the first is undefined; an exit
   is the same exit; a step is a step, to a related state or to a state
   that is one step from a related one. *)
Definition lagged (r r' : step_result) : Prop :=
  match r with
  | Undefined => True
  | Exit k1 vs hp rd => r' = Exit k1 vs hp rd
  | Step t =>
    ∃ u : state,
      r' = Step u
      ∧ (state_rel t u ∨ ∃ t' : state, step u = Step t' ∧ state_rel t t')
  end.

Lemma lagged_step (t u : state) : state_rel t u -> lagged (Step t) (Step u).
Proof. intros Hrel. exists u. split; [ reflexivity | left; exact Hrel ]. Qed.

Lemma lagged_of_related (r r' : step_result) :
  result_related state_rel r r' -> lagged r r'.
Proof.
  destruct r as [t | k1 vs hp rd | ], r' as [t' | k1' vs' hp' rd' | ]; simpl;
    try contradiction; try (intros _; exact I).
  - apply lagged_step.
  - intros (<- & <- & <- & <-). reflexivity.
Qed.

Lemma lagged_of_matched (r r' : step_result) :
  result_related state_matches r r' -> lagged r r'.
Proof.
  destruct r as [t | k1 vs hp rd | ], r' as [t' | k1' vs' hp' rd' | ]; simpl;
    try contradiction; try (intros _; exact I).
  - intros Hmatch. apply lagged_step, Rel_state_matched, Hmatch.
  - intros (<- & <- & <- & <-). reflexivity.
Qed.

Lemma with_extra_args_rel (c c' : cont_value) (extra : list value) :
  cont_rel c c' ->
  cont_rel (with_extra_args c extra) (with_extra_args c' extra).
Proof.
  intros Hc. destruct extra; simpl; [ exact Hc | apply Rel_extra, Hc ].
Qed.

(* The closures of the handlers of a recursive [Let_cont]. *)
Lemma rec_conts_plain
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts conts' : cont_env) :
  conts_plain conts conts' ->
  conts_plain (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers vars conts' ++ conts').
Proof.
  intros Hconts k1. rewrite !resolve_rec_closures.
  destruct (lookup k1 handlers) as [hh | ]; [ | apply Hconts ].
  apply Rel_plain. exact Hconts.
Qed.

Lemma handlers_ok_lookup
    (handlers : list (continuation * continuation_handler)) :
  handlers_ok handlers = true ->
  lookup k handlers = None
  ∧ ∀ (k1 : continuation) (hh : continuation_handler),
      lookup k1 handlers = Some hh ->
      handler_only_applied_or_returned_to k hh = true.
Proof.
  unfold handlers_ok.
  induction handlers as [ | [k' hh'] handlers IH]; cbn [forallb].
  - intros _. split; [ reflexivity | discriminate ].
  - intros Hok. apply andb_true_iff in Hok as [Hhead Hrest].
    apply andb_true_iff in Hhead as [Hne Hhh].
    apply negb_true_iff in Hne. destruct (IH Hrest) as [Hk Hothers].
    split.
    + rewrite lookup_cons, Pos.eqb_sym, Hne. exact Hk.
    + intros k1 hh. rewrite lookup_cons.
      destruct (Pos.eqb k1 k'); [ intros [= <-]; exact Hhh | apply Hothers ].
Qed.

Lemma rec_conts_inside
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts conts' : cont_env) :
  handlers_ok handlers = true ->
  conts_inside conts conts' ->
  conts_inside (rec_closures handlers vars conts ++ conts)
    (rec_closures (wrap_handlers k i the_wrapper handlers) vars conts'
     ++ conts').
Proof.
  intros Hok (Hk & Hk' & Hconts).
  destruct (handlers_ok_lookup handlers Hok) as [Hnone Hothers].
  split; [ | split ].
  - rewrite resolve_rec_closures, Hnone. exact Hk.
  - rewrite resolve_rec_closures, lookup_wrap_handlers, Hnone. exact Hk'.
  - intros k1 Hne. rewrite !resolve_rec_closures, lookup_wrap_handlers.
    destruct (lookup k1 handlers) as [hh | ] eqn:Hlookup; cbn [option_map];
      [ | exact (Hconts k1 Hne) ].
    apply Rel_inside; try assumption.
    exact (Hothers k1 hh Hlookup).
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
    apply Rel_state_plain.
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
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel return_to return_to' ->
  cont_rel exn_to exn_to' ->
  (∀ partial : value,
    lagged (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value return_to' [partial] statics hp rd)) ->
  lagged
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
  - apply lagged_of_related, call_code_rel; assumption.
  - apply lagged_of_related, call_code_rel;
      [ apply Rel_over; assumption | exact Hraise ].
  - apply Hpartial.
  - exact I.
Qed.

(* The handler of [k] entered with all its parameters and without the
   one: two states that the comparison matches, as the handler does
   not use it. *)
Lemma handler_entered (vs : list value) (bound0 : env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata) = true ->
  bind_params params vs vars0 = Some bound0 ->
  ∃ bound1 : env,
    apply_cont_value the_closure' (drop_nth i vs) statics hp rd
    = Step (State handler bound1 conts0 statics hp rd)
    ∧ state_matches (State handler bound0 conts0 statics hp rd)
        (State handler bound1 conts0 statics hp rd).
Proof.
  intros Hkinds Hbound0. unfold the_closure'. cbn [apply_cont_value].
  apply kinds_eqb_eq in Hkinds.
  rewrite map_drop_nth, Hkinds. cbn [hm_params_kinds].
  rewrite kinds_eqb_refl.
  pose proof (bind_params_length params vs vars0 bound0 Hbound0) as Hlength.
  destruct (nth_error vs i) as [v | ] eqn:Hv.
  2: { apply nth_error_None in Hv. rewrite Hlength in Hv.
       apply nth_error_None in Hv. congruence. }
  pose proof
    (bind_params_drop_nth i params vs vars0 p v Hparam Hv
       (distinct_drop_nth i params p Hdistinct Hparam))
    as Hbind.
  rewrite Hbound0 in Hbind.
  destruct (bind_params (drop_nth i params) (drop_nth i vs) vars0)
    as [bound1 | ]; [ | contradiction ].
  exists bound1. split; [ reflexivity | ].
  apply (State_matches [] []).
  - apply expr_eqb_refl; apply reflexive_map_nil.
  - intros x x' Hfree Hmatch. apply names_match_nil in Hmatch. subst x'.
    rewrite (Hbind x), lookup_cons.
    destruct (Pos.eqb_spec x p) as [-> | _]; [ | reflexivity ].
    apply var_free_b_sound in Hfree. congruence.
  - apply conts_match_same.
  - apply statics_match_same.
Qed.

(* The first closure of [k] and a closure of the wrapper, applied to
   the same values: the wrapper takes one step more, to apply the
   second closure of [k]. *)
Lemma apply_wrapper (dv : env) (dc : cont_env) (vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  resolve k dc = the_closure' ->
  lagged (apply_cont_value the_closure vs statics hp rd)
    (apply_cont_value (Closure the_wrapper dv dc []) vs statics hp rd).
Proof.
  intros Hk. unfold the_closure, the_wrapper, wrapper.
  cbn [apply_cont_value].
  destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
    eqn:Hkinds; [ | exact I ].
  destruct (bind_params params vs vars0) as [bound0 | ] eqn:Hbound0;
    [ | exact I ].
  destruct (bind_params_defined params vs vars0 dv bound0 Hbound0)
    as [boundw Hboundw].
  rewrite Hboundw. cbn [rec_closures map app].
  eexists. split; [ reflexivity | right ].
  rewrite step_apply_cont. cbn [ac_args ac_continuation].
  rewrite map_drop_nth,
    (eval_simples_drop_nth i (map Var params) boundw vs
       (eval_params_bound params vs dv boundw Hdistinct Hboundw)).
  unfold apply_continuation. rewrite Hk.
  destruct (handler_entered vs bound0 statics hp rd Hkinds Hbound0)
    as (bound1 & Hstep & Hmatch).
  exists (State handler bound1 conts0 statics hp rd).
  split; [ exact Hstep | apply Rel_state_matched, Hmatch ].
Qed.

(* Related continuations, applied to the same values. *)
Lemma apply_cont_value_rel (c c' : cont_value) (statics : static_env)
    (hp : heap) (rd : read_record) :
  cont_rel c c' ->
  ∀ vs : list value,
    lagged (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' vs statics hp rd).
Proof.
  intros Hc.
  induction Hc
    as [c
       | [ps m he] dv dc dc' recursive Hdc _
       | [ps m he] dv dc dc' recursive Hok Hrec Hk Hk' Hdc _
       | dv dc Hk
       | c c' extra Hc IHc
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise _];
    intros vs.
  - apply lagged_of_related, result_related_refl, state_rel_refl.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m));
      [ | exact I ].
    destruct (bind_params ps vs dv) as [bound | ]; [ | exact I ].
    apply lagged_step, Rel_state_plain, rec_conts_plain. exact Hdc.
  - cbn [wrap_uses_in_handler apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m));
      [ | exact I ].
    destruct (bind_params ps vs dv) as [bound | ]; [ | exact I ].
    apply lagged_step, Rel_state_inside; [ exact Hok | ].
    apply rec_conts_inside; [ exact Hrec | ].
    split; [ exact Hk | split; [ exact Hk' | exact Hdc ] ].
  - apply apply_wrapper, Hk.
  - exact (IHc (vs ++ extra)).
  - rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; try exact I.
    apply run_call_rel; [ exact Hreturn | exact Hraise | ].
    intros partial. exact (IHreturn [partial]).
Qed.

Lemma call_function_rel (callee : value) (call_kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel return_to return_to' ->
  cont_rel exn_to exn_to' ->
  lagged
    (call_function callee call_kind vs args_arity return_arity return_to
       exn_to statics hp rd)
    (call_function callee call_kind vs args_arity return_arity return_to'
       exn_to' statics hp rd).
Proof.
  intros Hreturn Hraise. unfold call_function.
  apply run_call_rel; [ exact Hreturn | exact Hraise | ].
  intros partial. apply apply_cont_value_rel, Hreturn.
Qed.

(* A call that returns to a handler does not end the run in its own
   step. *)
Lemma call_to_closure_no_exit (callee : value) (call_kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (hh : continuation_handler) (dv : env) (dc : cont_env)
    (recursive : list (continuation * continuation_handler))
    (exn_to : cont_value) (statics : static_env) (hp : heap) (rd : read_record)
    (k1 : continuation) (vs1 : list value) (hp1 : heap) (rd1 : read_record) :
  call_function callee call_kind vs args_arity return_arity
    (Closure hh dv dc recursive) exn_to statics hp rd
  <> Exit k1 vs1 hp1 rd1.
Proof.
  unfold call_function, run_call.
  destruct
    (shape_of_call statics callee call_kind vs args_arity return_arity)
    as [[closure c vs' args_arity'
        | closure c vs' remaining args_arity' remaining_arity | partial] | ].
  - destruct
      (call_code_cases closure c vs' args_arity' return_arity statics hp rd)
      as [Hundefined | (callee_vars & Hcall)];
      [ rewrite Hundefined | rewrite Hcall ]; discriminate.
  - destruct
      (call_code_cases closure c vs' args_arity' (code_result_arity c)
         statics hp rd)
      as [Hundefined | (callee_vars & Hcall)];
      [ rewrite Hundefined | rewrite Hcall ]; discriminate.
  - destruct hh as [ps m he]. cbn [apply_cont_value].
    destruct (kinds_eqb (map kind_of_value [partial]) (hm_params_kinds m));
      [ | discriminate ].
    destruct (bind_params ps [partial] dv); discriminate.
  - discriminate.
Qed.

(* What the second state answers with. *)
Definition answers : step_result -> state -> Prop :=
  answered state_rel (fun _ vs vs' => vs' = vs).

Lemma lagged_answers (r : step_result) (s' : state) :
  lagged r (step s') -> answers r s'.
Proof.
  unfold answers, answered. destruct r as [t | k1 vs hp rd | ]; simpl.
  - intros (u & Hu & [Hrel | (t' & Ht' & Hrel)]).
    + exists 0, u. split; [ simpl; rewrite Hu; reflexivity | exact Hrel ].
    + exists 1, t'. split; [ simpl; rewrite Hu, Ht'; reflexivity | exact Hrel ].
  - intros Hexit. exists vs. split; [ exact Hexit | reflexivity ].
  - intros _. exact I.
Qed.

(* The same where the second state takes a step first. *)
Lemma lagged_answers_after (r : step_result) (s' s1' : state) :
  step s' = Step s1' ->
  (∀ (k1 : continuation) (vs : list value) (hp : heap) (rd : read_record), r <>
     Exit k1 vs hp rd) ->
  lagged r (step s1') ->
  answers r s'.
Proof.
  intros Hfirst Hno_exit. unfold answers, answered.
  destruct r as [t | k1 vs hp rd | ]; simpl.
  - intros (u & Hu & [Hrel | (t' & Ht' & Hrel)]).
    + exists 1, u.
      split; [ simpl; rewrite Hfirst, Hu; reflexivity | exact Hrel ].
    + exists 2, t'.
      split; [ simpl; rewrite Hfirst, Hu, Ht'; reflexivity | exact Hrel ].
  - intros _. exfalso. exact (Hno_exit k1 vs hp rd_ eq_refl).
  - intros _. exact I.
Qed.

(* Two related states: the second answers the step of the first. *)
Lemma wrap_step (s s' : state) : state_rel s s' -> answers (step s) s'.
Proof.
  intros
    [e vars conts conts' statics hp rd Hconts
    | e vars conts conts' statics hp rd Hok (Hk & Hk' & Hconts)
    | s0 s0' Hmatch].
  - (* outside the scope of [k] *)
    apply lagged_answers.
    destruct e as [y kind1 d body | bindings body | k1 hh body
                  | handlers body | ap | ac | scrutinee arms | ].
    + rewrite !step_let_exec.
      destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | exact I ].
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      apply lagged_step, Rel_state_plain, Hconts.
    + rewrite !step_let_static.
      destruct (eval_static_bindings vars bindings statics); [ | exact I ].
      apply lagged_step, Rel_state_plain, Hconts.
    + rewrite !step_let_cont_non_recursive.
      apply lagged_step, Rel_state_plain.
      intros k2. destruct (Pos.eq_dec k2 k1) as [-> | Hne].
      * rewrite !resolve_head. apply Rel_plain. exact Hconts.
      * rewrite !(resolve_other k2 k1) by exact Hne. apply Hconts.
    + rewrite !step_let_cont_recursive.
      destruct (handlers_bind_statics handlers); [ exact I | ].
      apply lagged_step, Rel_state_plain, rec_conts_plain, Hconts.
    + rewrite !step_apply.
      destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
      destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
      destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
        [ | exact I ].
      apply call_function_rel;
        [ apply Hconts | apply with_extra_args_rel, Hconts ].
    + rewrite !step_apply_cont.
      destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
      unfold apply_continuation. apply apply_cont_value_rel, Hconts.
    + rewrite !step_switch.
      destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
      destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
      destruct (find_arm j arms) as [ac | ]; [ | exact I ].
      apply lagged_step, Rel_state_plain, Hconts.
    + rewrite !step_invalid. exact I.
  - (* in the scope of [k] *)
    assert (Hinside : conts_inside conts conts')
      by (split; [ exact Hk | split; [ exact Hk' | exact Hconts ] ]).
    destruct e as [y kind1 d body | bindings body | k1 hh body
                  | handlers body | ap | ac | scrutinee arms | ];
      cbn [wrap_uses only_applied_or_returned_to] in Hok |- *.
    + apply lagged_answers. rewrite !step_let_exec.
      destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | exact I ].
      destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
      apply lagged_step, Rel_state_inside; assumption.
    + apply lagged_answers. rewrite !step_let_static.
      destruct (eval_static_bindings vars bindings statics); [ | exact I ].
      apply lagged_step, Rel_state_inside; assumption.
    + apply lagged_answers. rewrite !step_let_cont_non_recursive.
      apply andb_true_iff in Hok as [Hok Hbody].
      apply andb_true_iff in Hok as [Hne Hhh].
      apply negb_true_iff, Pos.eqb_neq in Hne.
      apply lagged_step, Rel_state_inside; [ exact Hbody | ].
      split; [ | split ].
      * rewrite resolve_other by congruence. exact Hk.
      * rewrite resolve_other by congruence. exact Hk'.
      * intros k2 Hk2. destruct (Pos.eq_dec k2 k1) as [-> | Hne'].
        -- rewrite !resolve_head.
           exact (Rel_inside hh vars conts conts' [] Hhh eq_refl Hk Hk' Hconts).
        -- rewrite !(resolve_other k2 k1) by exact Hne'.
           exact (Hconts k2 Hk2).
    + apply lagged_answers. rewrite !step_let_cont_recursive.
      fold (wrap_handlers k i the_wrapper handlers).
      rewrite (wrap_handlers_bind_statics k i the_wrapper handlers eq_refl).
      destruct (handlers_bind_statics handlers); [ exact I | ].
      apply andb_true_iff in Hok as [Hhandlers Hbody].
      apply lagged_step, Rel_state_inside; [ exact Hbody | ].
      apply rec_conts_inside; assumption.
    + (* a call *)
      apply negb_true_iff, Pos.eqb_neq in Hok.
      destruct (Pos.eqb_spec (ap_continuation ap) k) as [Hreturn | Hreturn].
      * (* that returns to [k]: the second run binds the wrapper *)
        apply
          (lagged_answers_after _ _
             (State (Apply ap) vars
                ((k, Closure the_wrapper vars conts' []) :: conts') statics
                hp rd));
          [ apply step_let_cont_non_recursive | | ].
        -- intros k1 vs1 hp1. rewrite step_apply.
           destruct (eval_simple vars (ap_callee ap)) as [callee | ];
             [ | discriminate ].
           destruct (eval_simples vars (ap_args ap)) as [vs | ];
             [ | discriminate ].
           destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
             [ | discriminate ].
           rewrite Hreturn, Hk. apply call_to_closure_no_exit.
        -- rewrite !step_apply.
           destruct (eval_simple vars (ap_callee ap)) as [callee | ];
             [ | exact I ].
           destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
           destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
             [ | exact I ].
           rewrite Hreturn, Hk, resolve_head,
             (resolve_other (ap_exn_continuation ap) k) by exact Hok.
           apply call_function_rel;
             [ apply Rel_wrapper, Hk'
             | apply with_extra_args_rel, Hconts, Hok ].
      * apply lagged_answers. rewrite !step_apply.
        destruct (eval_simple vars (ap_callee ap)) as [callee | ];
          [ | exact I ].
        destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
        destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
          [ | exact I ].
        apply call_function_rel;
          [ apply Hconts, Hreturn | apply with_extra_args_rel, Hconts, Hok ].
    + (* an application of a continuation *)
      apply lagged_answers. unfold drop_arg_of.
      destruct (Pos.eqb_spec (ac_continuation ac) k) as [Hac | Hac].
      * (* of [k] *)
        rewrite !step_apply_cont. cbn [ac_args ac_continuation].
        unfold apply_continuation. rewrite Hac, Hk, Hk'.
        destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
          [ | exact I ].
        rewrite (eval_simples_drop_nth i _ _ _ Hvs).
        unfold the_closure at 1. cbn [apply_cont_value].
        destruct
          (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata))
          eqn:Hkinds; [ | exact I ].
        destruct (bind_params params vs vars0) as [bound0 | ] eqn:Hbound0;
          [ | exact I ].
        cbn [rec_closures map app].
        destruct (handler_entered vs bound0 statics hp rd Hkinds Hbound0)
          as (bound1 & Hstep & Hmatch).
        rewrite Hstep. apply lagged_step, Rel_state_matched, Hmatch.
      * rewrite !step_apply_cont.
        destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
        unfold apply_continuation. apply apply_cont_value_rel, Hconts, Hac.
    + apply lagged_answers. rewrite !step_switch.
      destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
      destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
      rewrite find_arm_drop.
      destruct (find_arm j arms) as [ac | ] eqn:Hfind; [ | exact I ].
      cbn [option_map].
      apply lagged_step, (Rel_state_inside (Apply_cont ac));
        [ | exact Hinside ].
      rewrite forallb_forall in Hok.
      exact (Hok (j, ac) (find_arm_in j arms ac Hfind)).
    + apply lagged_answers. rewrite !step_invalid. exact I.
  - (* once the handler of [k] has been entered *)
    apply lagged_answers, lagged_of_matched, matched_step, Hmatch.
Qed.

Lemma site_refines (s s' : state) : state_rel s s' -> refines s s'.
Proof.
  apply (stuttering_refines state_rel (fun _ vs vs' => vs' = vs));
    [ exact wrap_step | intros last vs vs' Hsame; exact Hsame ].
Qed.

End Site.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

Lemma wrap_return_refines (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr) (h' : continuation_handler)
    (body' : expr) :
  wrap_return i k h body = Some (h', body') ->
  expr_refines (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h' body').
Proof.
  destruct h as [params metadata handler]. unfold wrap_return.
  destruct (nth_error params i) as [p | ] eqn:Hparam; [ | discriminate ].
  destruct
    (negb (hm_is_exn_handler metadata) && distinct params
     && negb (var_free_b p handler) && only_applied_or_returned_to k body
     && (2 <=? uses_of k body))
    eqn:Hcond; [ | discriminate ].
  intros [= <- <-].
  apply andb_true_iff in Hcond as [Hcond _].
  apply andb_true_iff in Hcond as [Hcond Hok].
  apply andb_true_iff in Hcond as [Hcond Hunused].
  apply andb_true_iff in Hcond as [_ Hdistinct].
  apply negb_true_iff in Hunused.
  intros vars conts statics hp rd.
  apply
    (refines_step _ _ _ _
       (step_let_cont_non_recursive k (Handler params metadata handler) body
          vars conts statics hp rd)
       (step_let_cont_non_recursive k _ _ vars conts statics hp rd)).
  apply
    (site_refines k i p params metadata handler vars conts Hparam Hdistinct
       Hunused).
  apply Rel_state_inside; [ exact Hok | ].
  split; [ apply resolve_head | split; [ apply resolve_head | ] ].
  intros k1 Hne. rewrite !(resolve_other k1 k) by exact Hne. apply Rel_same.
Qed.

Lemma wrap_from_refines (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    expr_refines (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k (fst (wrap_from i k h body))
         (snd (wrap_from i k h body))).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [wrap_from].
  - apply expr_refines_refl.
  - destruct (wrap_return i k h body) as [[h' body'] | ] eqn:Hreturn;
      [ | apply IH ].
    exact (wrap_return_refines i k h body h' body' Hreturn).
Qed.

(* Each layer is a context around what the induction gives; then the
   parameters of the [Let_cont] are looked at. *)
Fixpoint wrap_returns_refines_aux (e : expr) :
  expr_refines e (wrap_returns e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [wrap_returns]; try apply expr_refines_refl.
  - exact (plug_refines (Let_body x kind d Hole) _ _ (not_many _)
      (wrap_returns_refines_aux body)).
  - exact (plug_refines (Let_static_body bindings Hole) _ _ (not_many _)
      (wrap_returns_refines_aux body)).
  - assert (Hlayers :
      expr_refines
        (Let_cont_non_recursive k (Handler params metadata handler) body)
        (Let_cont_non_recursive k
           (Handler params metadata (wrap_returns handler))
           (wrap_returns body))).
    { apply (expr_refines_trans _
        (Let_cont_non_recursive k
           (Handler params metadata (wrap_returns handler)) body)).
      - exact (plug_refines
          (Let_cont_non_recursive_handler k params metadata Hole body) _ _
          (not_many _) (wrap_returns_refines_aux handler)).
      - exact (plug_refines
          (Let_cont_non_recursive_body k
             (Handler params metadata (wrap_returns handler)) Hole)
          _ _ (not_many _) (wrap_returns_refines_aux body)). }
    pose proof
      (wrap_from_refines (length params) k
         (Handler params metadata (wrap_returns handler))
         (wrap_returns body))
      as Hfrom.
    destruct
      (wrap_from (length params) k
         (Handler params metadata (wrap_returns handler))
         (wrap_returns body))
      as [h' body'].
    exact (expr_refines_trans _ _ _ Hlayers Hfrom).
  - exact (plug_refines (Let_cont_recursive_body handlers Hole) _ _
      (not_many _) (wrap_returns_refines_aux body)).
Qed.

Theorem wrap_returns_refines : wrap_returns_refines_statement.
Proof. exact wrap_returns_refines_aux. Qed.
