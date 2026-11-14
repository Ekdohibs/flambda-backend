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


(* Proofs of the statement of FlambdaShadowParamsSpec.v. Nothing here
   needs review.

   The term and its result run side by side under a relation of
   states ([state_rel]): the same expression, where neither run is in
   a term that the pass rewrote; or an expression and the same
   rewritten, under variables that agree and continuations that are
   closures of a handler and of that handler rewritten. The two runs
   go in step except where a handler is entered: there the result
   takes one step more for each [Let] that the pass put at its head,
   and each such [Let] has a value of its kind, the handler having
   been given values of the kinds of its parameters ([kinded]). The
   relation has a number, the [Let]s that the result may still take
   alone. That the result refines the term is by
   FlambdaSimulationProofs.stuttering_refines, the result run on to
   where it has taken them ([catch_up]); that the term refines the
   result is by FlambdaLaggingProofs.counted_lagging_refines. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaSimulationProofs FlambdaLaggingProofs FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaUnusedParamsProofs FlambdaUnaliasParamsProofs
  FlambdaUntagParamsSpec FlambdaUntagParamsProofs FlambdaShadowParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What the rewrite keeps of the syntax                                     *)
(* ======================================================================== *)

Lemma shadow_lets_binds_statics (us : list nat) (e : expr) :
  ∀ (params : list variable) (kinds : list flambda_kind) (i : nat),
    binds_statics (shadow_lets us i params kinds e) = binds_statics e.
Proof.
  induction params as [ | p params IH]; intros [ | kind kinds] i;
    cbn [shadow_lets]; try reflexivity.
  destruct (at_position us i); cbn [binds_statics]; apply IH.
Qed.

Fixpoint shadow_expr_binds_statics (sites : positions) (e : expr)
    {struct e} :
  binds_statics (shadow_expr sites e) = binds_statics e
with shadow_handler_binds_statics (sites : positions) (us : list nat)
    (h : continuation_handler) {struct h} :
  handler_binds_statics (shadow_handler sites us h) = handler_binds_statics h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [shadow_expr binds_statics]; try reflexivity.
    + apply shadow_expr_binds_statics.
    + rewrite shadow_expr_binds_statics. reflexivity.
    + rewrite shadow_handler_binds_statics, shadow_expr_binds_statics.
      reflexivity.
    + rewrite shadow_expr_binds_statics. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map existsb]. rewrite shadow_handler_binds_statics, IH.
      reflexivity.
  - destruct h as [params metadata handler].
    cbn [shadow_handler handler_binds_statics].
    destruct (distinct params);
      [ rewrite shadow_lets_binds_statics | ];
      apply shadow_expr_binds_statics.
Qed.

Definition shadow_handlers (sites : positions)
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(k, h) => (k, shadow_handler sites (positions_of sites k) h))
    handlers.

Lemma shadow_handlers_bind_statics (sites : positions)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics (shadow_handlers sites handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics, shadow_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map existsb]. rewrite shadow_handler_binds_statics, IH. reflexivity.
Qed.

Lemma lookup_shadow_handlers (sites : positions) (k1 : continuation)
    (handlers : list (continuation * continuation_handler)) :
  lookup k1 (shadow_handlers sites handlers)
  = option_map (shadow_handler sites (positions_of sites k1))
      (lookup k1 handlers).
Proof.
  unfold shadow_handlers.
  induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
  cbn [map]. rewrite !lookup_cons, IH.
  destruct (Pos.eqb_spec k1 k') as [-> | Hne]; reflexivity.
Qed.

(* ======================================================================== *)
(* The [Let]s at the head of a handler                                      *)
(* ======================================================================== *)

(* The variable holds a value of the kind. *)
Definition kinded (vars : env) (p : variable) (kind : flambda_kind) : Prop :=
  ∃ v : value,
    lookup p vars = Some v ∧ kind_eqb (kind_of_value v) kind = true.

(* A variable bound again to what it holds: the variables hold what
   they held. *)
Lemma agree_rebound (vars vars' : env) (p : variable) (v : value) :
  agree vars vars' ->
  lookup p vars' = Some v ->
  agree vars ((p, v) :: vars').
Proof.
  intros Hagree Hp z. rewrite lookup_cons.
  destruct (Pos.eqb_spec z p) as [-> | _]; [ | apply Hagree ].
  rewrite Hagree. exact Hp.
Qed.

Lemma kinded_rebound (vars : env) (p : variable) (v : value) (q : variable)
    (kind : flambda_kind) :
  lookup p vars = Some v ->
  kinded vars q kind ->
  kinded ((p, v) :: vars) q kind.
Proof.
  intros Hp (w & Hq & Hkind). exists w. rewrite lookup_cons.
  destruct (Pos.eqb_spec q p) as [-> | _]; [ | split; assumption ].
  rewrite Hp in Hq. split; assumption.
Qed.

(* The [Let]s for parameters that hold values of their kinds: there is
   none, or a first one, of a parameter that does, followed by those of
   fewer parameters, which do. *)
Lemma shadow_lets_cases (us : list nat) (e : expr) (vars : env) :
  ∀ (params : list variable) (kinds : list flambda_kind) (i : nat),
    Forall2 (kinded vars) params kinds ->
    shadow_lets us i params kinds e = e
    ∨ ∃ (p : variable) (kind : flambda_kind) (params' : list variable)
        (kinds' : list flambda_kind) (i' : nat),
        shadow_lets us i params kinds e
        = Let p kind (Simple (Var p)) (shadow_lets us i' params' kinds' e)
        ∧ kinded vars p kind
        ∧ Forall2 (kinded vars) params' kinds'
        ∧ length params' < length params.
Proof.
  induction params as [ | p params IH]; intros [ | kind kinds] i Hkinded;
    try (left; reflexivity).
  inversion Hkinded as [ | p0 kind0 params0 kinds0 Hp Hrest]; subst.
  cbn [shadow_lets]. destruct (at_position us i).
  - right. exists p, kind, params, kinds, (S i).
    repeat split; try assumption. cbn [length]. lia.
  - destruct (IH kinds (S i) Hrest)
      as [Heq | (p' & kind' & params' & kinds' & i' & Heq & Hp' & Hrest'
                 & Hless)].
    + left. exact Heq.
    + right. exists p', kind', params', kinds', i'.
      repeat split; try assumption. cbn [length]. lia.
Qed.

(* The parameters of a handler that was given values of its kinds hold
   values of their kinds. *)
Lemma combine_lookup_all (params : list variable) :
  ∀ vs : list value,
    distinct params = true ->
    length params = length vs ->
    Forall2 (fun p v => lookup p (combine params vs) = Some v) params vs.
Proof.
  induction params as [ | p params IH]; intros [ | v vs] Hdistinct Hlength;
    try discriminate Hlength; [ constructor | ].
  cbn [distinct] in Hdistinct.
  apply andb_true_iff in Hdistinct as [Hp Hdistinct].
  apply negb_true_iff in Hp. injection Hlength as Hlength.
  cbn [combine]. constructor.
  - rewrite lookup_cons, Pos.eqb_refl. reflexivity.
  - generalize (IH vs Hdistinct Hlength). apply Forall2_impl.
    intros q w Hq. rewrite lookup_cons.
    destruct (Pos.eqb_spec q p) as [-> | _]; [ | exact Hq ].
    rewrite (lookup_combine_none p params vs Hp) in Hq. discriminate Hq.
Qed.

Lemma params_kinded (params : list variable) (vs : list value)
    (vars bound : env) (kinds : list flambda_kind) :
  distinct params = true ->
  bind_params params vs vars = Some bound ->
  kinds_eqb (map kind_of_value vs) kinds = true ->
  Forall2 (kinded bound) params kinds.
Proof.
  intros Hdistinct Hbound Hkinds. apply kinds_eqb_eq in Hkinds. subst kinds.
  pose proof (bind_params_same_length params vs vars bound Hbound) as Hlength.
  pose proof (combine_lookup_all params vs Hdistinct Hlength) as Hall.
  clear Hlength. induction Hall as [ | p v params' vs' Hp _ IH];
    [ constructor | ].
  cbn [map]. constructor; [ | exact IH ].
  exists v.
  rewrite (lookup_bind_params params vs vars bound p Hdistinct Hbound), Hp.
  split; [ reflexivity | apply kind_eqb_refl ].
Qed.

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

Section Runs.

Variable sites : positions.

(* What a continuation name stands for in the two runs: the same
   thing; or two closures of one handler under the same variables and
   continuations so related; or a closure of a handler and a closure
   of that handler rewritten, under variables that agree and
   continuations that are related; or two so related with the same
   extra arguments; or what is left of the same over-application. *)
Inductive cont_rel : cont_value -> cont_value -> Prop :=
  | Rel_same (c : cont_value) : cont_rel c c
  | Rel_plain (hh : continuation_handler) (dv : env) (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    (∀ k1 : continuation, cont_rel (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel (Closure hh dv dc recursive) (Closure hh dv dc' recursive)
  | Rel_shadow (us : list nat) (hh : continuation_handler) (dv dv' : env)
      (dc dc' : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    agree dv dv' ->
    (∀ k1 : continuation, cont_rel (resolve k1 dc) (resolve k1 dc')) ->
    cont_rel (Closure hh dv dc recursive)
      (Closure (shadow_handler sites us hh) dv' dc'
         (shadow_handlers sites recursive))
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

Definition conts_rel (conts conts' : cont_env) : Prop :=
  ∀ k1 : continuation, cont_rel (resolve k1 conts) (resolve k1 conts').

(* Two states, and the [Let]s that the second may still take alone:
   the same expression; an expression and the same rewritten; or an
   expression and the same rewritten under [Let]s of parameters that
   hold values of their kinds. *)
Inductive state_rel : nat -> state -> state -> Prop :=
  | Rel_state_plain (e : expr) (vars : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) :
    conts_rel conts conts' ->
    state_rel 0 (State e vars conts statics hp rd)
      (State e vars conts' statics hp rd)
  | Rel_state_shadow (e : expr) (vars vars' : env) (conts conts' : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record) :
    agree vars vars' ->
    conts_rel conts conts' ->
    state_rel 0 (State e vars conts statics hp rd)
      (State (shadow_expr sites e) vars' conts' statics hp rd)
  | Rel_state_lets (us : list nat) (i : nat) (p : variable)
      (kind : flambda_kind) (params : list variable)
      (kinds : list flambda_kind) (e : expr) (vars vars' : env)
      (conts conts' : cont_env) (statics : static_env) (hp : heap)
        (rd : read_record) :
    agree vars vars' ->
    kinded vars' p kind ->
    Forall2 (kinded vars') params kinds ->
    conts_rel conts conts' ->
    state_rel (S (length params)) (State e vars conts statics hp rd)
      (State
         (Let p kind (Simple (Var p))
            (shadow_lets us i params kinds (shadow_expr sites e)))
         vars' conts' statics hp rd).

Definition related (s s' : state) : Prop := ∃ n : nat, state_rel n s s'.

Lemma related_refl (s : state) : related s s.
Proof.
  destruct s as [e vars conts statics hp rd]. exists 0.
  apply Rel_state_plain. intros k1. apply Rel_same.
Qed.

(* An expression and the same rewritten under the [Let]s of parameters
   that hold values of their kinds. *)
Lemma state_rel_lets (us : list nat) (i : nat) (params : list variable)
    (kinds : list flambda_kind) (e : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  agree vars vars' ->
  Forall2 (kinded vars') params kinds ->
  conts_rel conts conts' ->
  ∃ n : nat,
    n <= length params
    ∧ state_rel n (State e vars conts statics hp rd)
        (State (shadow_lets us i params kinds (shadow_expr sites e)) vars'
           conts' statics hp rd).
Proof.
  intros Hagree Hkinded Hconts.
  destruct (shadow_lets_cases us (shadow_expr sites e) vars' params kinds i
              Hkinded)
    as [-> | (p & kind & params' & kinds' & i' & -> & Hp & Hrest & Hless)].
  - exists 0. split; [ lia | apply Rel_state_shadow; assumption ].
  - exists (S (length params')).
    split; [ lia | apply Rel_state_lets; assumption ].
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
  conts_rel conts conts' ->
  conts_rel (rec_closures handlers vars conts ++ conts)
    (rec_closures handlers vars conts' ++ conts').
Proof.
  intros Hconts k1. rewrite !resolve_rec_closures.
  destruct (lookup k1 handlers) as [hh | ]; [ | apply Hconts ].
  apply Rel_plain. exact Hconts.
Qed.

Lemma rec_conts_shadow
    (handlers : list (continuation * continuation_handler))
    (vars vars' : env) (conts conts' : cont_env) :
  agree vars vars' ->
  conts_rel conts conts' ->
  conts_rel (rec_closures handlers vars conts ++ conts)
    (rec_closures (shadow_handlers sites handlers) vars' conts' ++ conts').
Proof.
  intros Hagree Hconts k1.
  rewrite !resolve_rec_closures, lookup_shadow_handlers.
  destruct (lookup k1 handlers) as [hh | ]; cbn [option_map];
    [ | exact (Hconts k1) ].
  apply Rel_shadow; assumption.
Qed.

(* The same full call, returning and raising to related continuations. *)
Lemma call_code_rel (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to exn_to' : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  cont_rel return_to return_to' ->
  cont_rel exn_to exn_to' ->
  result_related related
    (call_code closure c vs args_arity return_arity return_to exn_to statics hp
       rd)
    (call_code closure c vs args_arity return_arity return_to' exn_to'
       statics hp rd).
Proof.
  intros Hreturn Hraise.
  destruct (call_code_cases closure c vs args_arity return_arity statics hp rd)
    as [Hundefined | (callee_vars & Hcall)].
  - rewrite !Hundefined. exact I.
  - rewrite !Hcall. simpl. exists 0.
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
    result_related related
      (apply_cont_value return_to [partial] statics hp rd)
      (apply_cont_value return_to' [partial] statics hp rd)) ->
  result_related related
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
    result_related related (apply_cont_value c vs statics hp rd)
      (apply_cont_value c' vs statics hp rd).
Proof.
  intros Hc.
  induction Hc
    as [c
       | [ps m he] dv dc dc' recursive Hdc _
       | us [ps m he] dv dv' dc dc' recursive Hagree Hdc _
       | c c' extra Hc IHc
       | remaining remaining_arity return_arity return_to return_to' exn_to
           exn_to' Hreturn IHreturn Hraise _];
    intros vs.
  - apply result_related_refl, related_refl.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m));
      [ | exact I ].
    destruct (bind_params ps vs dv) as [bound | ]; [ | exact I ].
    simpl. exists 0. apply Rel_state_plain, rec_conts_plain. exact Hdc.
  - cbn [shadow_handler apply_cont_value].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds m))
      eqn:Hkinds; [ | exact I ].
    pose proof (bind_params_agree ps vs dv dv' Hagree) as Hbind.
    destruct (bind_params ps vs dv) as [bound | ],
      (bind_params ps vs dv') as [bound' | ] eqn:Hbound';
      try contradiction; [ | exact I ].
    cbn [result_related].
    pose proof (rec_conts_shadow recursive dv dv' dc dc' Hagree Hdc)
      as Hconts.
    destruct (distinct ps) eqn:Hdistinct.
    + destruct
        (state_rel_lets us 0 ps (hm_params_kinds m) he bound bound' _ _
           statics hp rd Hbind
           (params_kinded ps vs dv' bound' _ Hdistinct Hbound' Hkinds) Hconts)
        as (n & _ & Hrel).
      exists n. exact Hrel.
    + exists 0. apply Rel_state_shadow; assumption.
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
  result_related related
    (call_function callee call_kind vs args_arity return_arity return_to
       exn_to statics hp rd)
    (call_function callee call_kind vs args_arity return_arity return_to'
       exn_to' statics hp rd).
Proof.
  intros Hreturn Hraise. unfold call_function.
  apply run_call_rel; [ exact Hreturn | exact Hraise | ].
  intros partial. apply apply_cont_value_rel, Hreturn.
Qed.

(* The same expression under related continuations: the two states
   step together. *)
Lemma plain_step (e : expr) (vars : env) (conts conts' : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  conts_rel conts conts' ->
  result_related related (step (State e vars conts statics hp rd))
    (step (State e vars conts' statics hp rd)).
Proof.
  intros Hconts.
  destruct e as [y kind1 d body | bindings body | k1 hh body
                | handlers body | ap | ac | scrutinee arms | ].
  - rewrite !step_let_exec.
    destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
    exists 0. apply Rel_state_plain, Hconts.
  - rewrite !step_let_static.
    destruct (eval_static_bindings vars bindings statics); [ | exact I ].
    exists 0. apply Rel_state_plain, Hconts.
  - rewrite !step_let_cont_non_recursive. exists 0.
    apply Rel_state_plain.
    intros k2. destruct (Pos.eq_dec k2 k1) as [-> | Hne].
    + rewrite !resolve_head. apply Rel_plain. exact Hconts.
    + rewrite !(resolve_other k2 k1) by exact Hne. apply Hconts.
  - rewrite !step_let_cont_recursive.
    destruct (handlers_bind_statics handlers); [ exact I | ].
    exists 0. apply Rel_state_plain, rec_conts_plain, Hconts.
  - rewrite !step_apply.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    apply call_function_rel;
      [ apply Hconts | apply with_extra_args_rel, Hconts ].
  - rewrite !step_apply_cont.
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
    unfold apply_continuation. apply apply_cont_value_rel, Hconts.
  - rewrite !step_switch.
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
    destruct (find_arm j arms) as [ac | ]; [ | exact I ].
    exists 0. apply Rel_state_plain, Hconts.
  - rewrite !step_invalid. exact I.
Qed.

(* An expression and the same rewritten: the two states step
   together. *)
Lemma shadow_step (e : expr) (vars vars' : env) (conts conts' : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  agree vars vars' ->
  conts_rel conts conts' ->
  result_related related (step (State e vars conts statics hp rd))
    (step (State (shadow_expr sites e) vars' conts' statics hp rd)).
Proof.
  intros Hagree Hconts.
  destruct e as [y kind1 d body | bindings body | k1 hh body
                | handlers body | ap | ac | scrutinee arms | ];
    cbn [shadow_expr].
  - rewrite !step_let_exec,
      (exec_named_agree statics hp rd vars vars' d (fun x _ => Hagree x)).
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ];
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind1); [ | exact I ].
    exists 0. apply Rel_state_shadow; [ apply agree_cons, Hagree | ].
    exact Hconts.
  - rewrite !step_let_static,
      (eval_static_bindings_agree vars vars' bindings (fun x _ => Hagree x)).
    destruct (eval_static_bindings vars bindings statics); [ | exact I ].
    exists 0. apply Rel_state_shadow; assumption.
  - rewrite !step_let_cont_non_recursive. exists 0.
    apply Rel_state_shadow; [ exact Hagree | ].
    intros k2. destruct (Pos.eq_dec k2 k1) as [-> | Hne].
    + rewrite !resolve_head.
      exact
        (Rel_shadow (positions_of sites k1) hh vars vars' conts conts' []
           Hagree Hconts).
    + rewrite !(resolve_other k2 k1) by exact Hne. apply Hconts.
  - rewrite !step_let_cont_recursive.
    fold (shadow_handlers sites handlers).
    rewrite shadow_handlers_bind_statics.
    destruct (handlers_bind_statics handlers); [ exact I | ].
    exists 0. apply Rel_state_shadow; [ exact Hagree | ].
    apply rec_conts_shadow; assumption.
  - rewrite !step_apply,
      (eval_simple_agree vars vars' _ (fun x _ => Hagree x)),
      (eval_simples_agree vars vars' (ap_args ap) (fun x _ => Hagree x)),
      (eval_simples_agree vars vars' (exn_extra_simples ap)
         (fun x _ => Hagree x)).
    destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    apply call_function_rel;
      [ apply Hconts | apply with_extra_args_rel, Hconts ].
  - rewrite !step_apply_cont,
      (eval_simples_agree vars vars' (ac_args ac) (fun x _ => Hagree x)).
    destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | exact I ].
    unfold apply_continuation. apply apply_cont_value_rel, Hconts.
  - rewrite !step_switch,
      (eval_simple_agree vars vars' scrutinee (fun x _ => Hagree x)).
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [j | ]; [ | exact I ].
    destruct (find_arm j arms) as [ac | ]; [ | exact I ].
    exists 0.
    exact
      (Rel_state_shadow (Apply_cont ac) vars vars' conts conts' statics hp
         rd Hagree Hconts).
  - rewrite !step_invalid. exact I.
Qed.

(* Two states related with no [Let] for the second to take alone step
   together. *)
Lemma together_step (s s' : state) :
  state_rel 0 s s' -> result_related related (step s) (step s').
Proof.
  intros Hrel. remember 0 as n eqn:Hn.
  destruct Hrel
    as [e vars conts conts' statics hp rd Hconts
       | e vars vars' conts conts' statics hp Hagree Hconts
       | us i p kind params kinds e vars vars' conts conts' statics hp Hagree
           Hp Hkinded Hconts];
    [ | | discriminate Hn ].
  - apply plain_step, Hconts.
  - apply shadow_step; assumption.
Qed.

(* Where the second has a [Let] to take alone, it takes it, and is
   related to the first with fewer. *)
Lemma alone_step (n : nat) (s s' : state) :
  state_rel (S n) s s' ->
  ∃ (m : nat) (t' : state), step s' = Step t' ∧ m <= n ∧ state_rel m s t'.
Proof.
  intros Hrel. remember (S n) as n' eqn:Hn.
  destruct Hrel
    as [e vars conts conts' statics hp Hconts
       | e vars vars' conts conts' statics hp Hagree Hconts
       | us i p kind params kinds e vars vars' conts conts' statics hp rd Hagree
           Hp Hkinded Hconts];
    try discriminate Hn.
  injection Hn as <-. destruct Hp as (v & Hp & Hkind).
  destruct
    (state_rel_lets us i params kinds e vars ((p, v) :: vars') conts conts'
       statics hp rd (agree_rebound vars vars' p v Hagree Hp))
    as (m & Hm & Hrel); [ | exact Hconts | ].
  { revert Hkinded. apply Forall2_impl.
    intros q kind' Hq. exact (kinded_rebound vars' p v q kind' Hp Hq). }
  exists m, (State (shadow_lets us i params kinds (shadow_expr sites e))
               ((p, v) :: vars') conts' statics hp rd).
  split; [ | split; assumption ].
  rewrite step_let_simple. cbn [eval_named eval_simple]. rewrite Hp, Hkind.
  reflexivity.
Qed.

(* The second run, on to where it has no [Let] to take alone. *)
Lemma catch_up (n : nat) :
  ∀ s s' : state,
    state_rel n s s' ->
    ∃ (j : nat) (s'' : state), run j s' = Step s'' ∧ state_rel 0 s s''.
Proof.
  induction n as [n IH] using (well_founded_induction lt_wf).
  intros s s' Hrel. destruct n as [ | n].
  - exists 0, s'. split; [ reflexivity | exact Hrel ].
  - destruct (alone_step n s s' Hrel) as (m & t' & Hstep & Hm & Hrel').
    destruct (IH m ltac:(lia) s t' Hrel')
      as (j & s'' & Hrun & Hrel'').
    exists (S j), s''. split; [ | exact Hrel'' ].
    simpl. rewrite Hstep. exact Hrun.
Qed.

(* The second run answers the first with one step or more. *)
Lemma related_refines (s s' : state) : state_rel 0 s s' -> refines s s'.
Proof.
  apply (stuttering_refines (state_rel 0) (fun _ vs vs' => vs' = vs));
    [ | intros last vs vs' Hsame; exact Hsame ].
  clear s s'. intros s s' Hrel.
  pose proof (together_step s s' Hrel) as Hstep.
  destruct (step s) as [t | k1 vs hp1 rd1 | ],
    (step s') as [t' | k1' vs' hp1' rd1' | ] eqn:Hs';
    simpl in Hstep; try contradiction; unfold answered.
  - destruct Hstep as [n Hrel'].
    destruct (catch_up n t t' Hrel') as (j & t'' & Hrun & Hrel'').
    exists j, t''. split; [ | exact Hrel'' ].
    simpl. rewrite Hs'. exact Hrun.
  - destruct Hstep as (<- & <- & <- & <-). exists vs. auto.
  - exact I.
Qed.

(* The first run answers the second, which takes the [Let]s alone. *)
Lemma related_refined (n : nat) (s s' : state) :
  state_rel n s s' -> refines s' s.
Proof.
  intros Hrel.
  refine
    (counted_lagging_refines (fun n a b => state_rel n b a) _ n
       s' s Hrel).
  clear n s s' Hrel.
  intros n s' s Hrel. destruct n as [ | n].
  - pose proof (together_step s s' Hrel) as Hstep.
    destruct (step s) as [t | k1 vs hp1 rd1 | ],
      (step s') as [t' | k1' vs' hp1' rd1' | ];
      simpl in Hstep; try contradiction.
    + destruct Hstep as [m Hrel']. left. exists m, t. auto.
    + destruct Hstep as (<- & <- & <- & <-). reflexivity.
    + exact I.
  - destruct (alone_step n s s' Hrel) as (m & t' & Hstep & Hm & Hrel').
    rewrite Hstep. right. exists m. split; [ lia | exact Hrel' ].
Qed.

End Runs.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Theorem shadow_expr_equivalent : shadow_expr_equivalent_statement.
Proof.
  intros sites e vars conts statics hp rd.
  assert (Hrel :
    state_rel sites 0 (State e vars conts statics hp rd)
      (State (shadow_expr sites e) vars conts statics hp rd)).
  { apply Rel_state_shadow; [ apply agree_refl | ].
    intros k1. apply Rel_same. }
  exact
    (refines_antisymmetric _ _ (related_refines sites _ _ Hrel)
       (related_refined sites 0 _ _ Hrel)).
Qed.

(* The pass of the validation is the rewrite at the positions that it
   computes, or the term. *)
Lemma shadow_params_equivalent (e : expr) :
  expr_equivalent e (shadow_params e).
Proof.
  unfold shadow_params.
  destruct (count_positions (shadow_positions e));
    [ intros vars conts statics hp rd; apply equivalent_refl | ].
  apply shadow_expr_equivalent.
Qed.

Lemma shadow_params_refines (e : expr) : expr_refines e (shadow_params e).
Proof. apply expr_equivalent_refines, shadow_params_equivalent. Qed.
