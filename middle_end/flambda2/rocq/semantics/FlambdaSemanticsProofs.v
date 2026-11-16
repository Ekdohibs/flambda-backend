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

(* Lemmas about the execution of FlambdaSemanticsSpec.v: what a state of
   each construct steps to (the step lemmas, through which every other
   proof reasons about [step]), bounded runs, the uniqueness of
   behaviours, and the ways of showing two states equivalent or one a
   refinement of the other. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2 Require Export FlambdaValueProofs FlambdaPrimitiveProofs.
From OxCamlFlambda2 Require FlambdaHeapProofs.
Import ListNotations.

(* ======================================================================== *)
(* States                                                                   *)
(* ======================================================================== *)

(* A state is its components. A proof about any state [s] rewrites with
   this and goes by cases on [state_expr s]; it does not destruct [s],
   which would name the components by their position. *)
Lemma state_eta (s : state) :
  s = State (state_expr s) (state_vars s) (state_conts s) (state_statics s)
    (state_heap s) (state_reads s).
Proof. destruct s. reflexivity. Qed.

(* Reduces the projections of states that are given by their components,
   and nothing else. *)
Ltac simpl_state :=
  cbn [state_expr state_vars state_conts state_statics state_heap
       state_reads] in *.

(* ======================================================================== *)
(* Steps                                                                    *)
(* ======================================================================== *)

(* What a state steps to, for each construct: one lemma for each case of
   [step], saying what the state comes to and under which conditions.

   These are how every proof reasons about a step. [step] is made opaque
   below, so [simpl] and [unfold] leave it alone: a proof rewrites with
   the lemma of the construct at hand and goes by cases on the conditions
   it names. A change to [step] is then a change to these lemmas, and to
   the proofs that use the lemma of the case that changed. *)

Lemma step_let_exec (x : variable) (kind : flambda_kind)
    (defining_expr : named) (body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  step (State (Let x kind defining_expr body) vars conts statics hp rd)
  = match exec_named statics hp rd vars defining_expr with
    | Some (v, hp', rd') =>
      if kind_eqb (kind_of_value v) kind
      then Step (State body ((x, v) :: vars) conts statics hp' rd')
      else Undefined
    | None => Undefined
    end.
Proof. reflexivity. Qed.

Lemma step_let_static (bindings : list static_binding) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  step (State (Let_static bindings body) vars conts statics hp rd)
  = match eval_static_bindings vars bindings statics with
    | Some statics' => Step (State body vars conts statics' hp rd)
    | None => Undefined
    end.
Proof. reflexivity. Qed.

Lemma step_let_cont_non_recursive (k : continuation)
    (h : continuation_handler) (body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  step (State (Let_cont_non_recursive k h body) vars conts statics hp rd)
  = Step
    (State body vars ((k, Closure h vars conts []) :: conts) statics hp rd).
Proof. reflexivity. Qed.

Lemma step_let_cont_recursive
    (handlers : list (continuation * continuation_handler)) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  step (State (Let_cont_recursive handlers body) vars conts statics hp rd)
  = if handlers_bind_statics handlers then Undefined
    else
      Step
        (State body vars (rec_closures handlers vars conts ++ conts) statics
           hp rd).
Proof. reflexivity. Qed.

Lemma step_apply (ap : apply_expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  step (State (Apply ap) vars conts statics hp rd)
  = match
      eval_simple vars (ap_callee ap), eval_simples vars (ap_args ap),
      eval_simples vars (exn_extra_simples ap)
    with
    | Some callee, Some vs, Some extra =>
      call_function callee (ap_call_kind ap) vs (ap_args_arity ap)
        (ap_return_arity ap) (resolve (ap_continuation ap) conts)
        (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
        statics hp rd
    | _, _, _ => Undefined
    end.
Proof. reflexivity. Qed.

(* A call with no extra argument for its exception continuation gives its
   callee that continuation as it is. *)
Lemma step_apply_no_extra_args (ap : apply_expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  ap_exn_extra_args ap = [] ->
  step (State (Apply ap) vars conts statics hp rd)
  = match eval_simple vars (ap_callee ap), eval_simples vars (ap_args ap) with
    | Some callee, Some vs =>
      call_function callee (ap_call_kind ap) vs (ap_args_arity ap)
        (ap_return_arity ap) (resolve (ap_continuation ap) conts)
        (resolve (ap_exn_continuation ap) conts) statics hp rd
    | _, _ => Undefined
    end.
Proof.
  intros Hextra. rewrite step_apply. unfold exn_extra_simples.
  rewrite Hextra. simpl.
  destruct (eval_simple vars (ap_callee ap)); [ | reflexivity ].
  destruct (eval_simples vars (ap_args ap)); reflexivity.
Qed.

Lemma step_apply_cont (ac : apply_cont) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  step (State (Apply_cont ac) vars conts statics hp rd)
  = match eval_simples vars (ac_args ac) with
    | Some vs => apply_continuation (ac_continuation ac) vs conts statics hp rd
    | None => Undefined
    end.
Proof. reflexivity. Qed.

Lemma step_switch (scrutinee : simple) (arms : list (Z * apply_cont))
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  step (State (Switch scrutinee arms) vars conts statics hp rd)
  = match eval_simple vars scrutinee with
    | Some v =>
      match naked_immediate_of_value v with
      | Some i =>
        match find_arm i arms with
        | Some ac => Step (State (Apply_cont ac) vars conts statics hp rd)
        | None => Undefined
        end
      | None => Undefined
      end
    | None => Undefined
    end.
Proof. reflexivity. Qed.

Lemma step_invalid (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record) :
  step (State Invalid vars conts statics hp rd) = Undefined.
Proof. reflexivity. Qed.

(* A continuation with extra arguments is, in the end, a handler, a way
   out or what is left of an over-application, applied to more
   arguments: [cont_target] is which, and [cont_args] the arguments it
   is applied to when the continuation is applied to [vs]. A proof about
   applying a continuation rewrites with [apply_cont_value_target] and
   goes by the three cases of [cont_target_cases]. *)
Fixpoint cont_target (c : cont_value) : cont_value :=
  match c with
  | Extra_args c _ => cont_target c
  | Closure _ _ _ _ | Outer _ | Over_application _ _ _ _ _ => c
  end.

Fixpoint cont_args (c : cont_value) (vs : list value) : list value :=
  match c with
  | Extra_args c extra => cont_args c (vs ++ extra)
  | Closure _ _ _ _ | Outer _ | Over_application _ _ _ _ _ => vs
  end.

Lemma apply_cont_value_target (c : cont_value) (vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  apply_cont_value c vs statics hp rd
  = apply_cont_value (cont_target c) (cont_args c vs) statics hp rd.
Proof.
  revert vs.
  induction c
    as [h def_vars def_conts recursive | k | c IH extra
       | remaining remaining_arity return_arity return_to _ exn_to _];
    intros vs; [ reflexivity | reflexivity | apply IH | reflexivity ].
Qed.

Lemma cont_target_cases (c : cont_value) :
  (∃ (h : continuation_handler) (def_vars : env) (def_conts : cont_env)
     (recursive : list (continuation * continuation_handler)),
    cont_target c = Closure h def_vars def_conts recursive)
  ∨ (∃ k : continuation, cont_target c = Outer k)
  ∨ (∃ (remaining : list value) (remaining_arity return_arity : arity)
       (return_to exn_to : cont_value),
      cont_target c
      = Over_application remaining remaining_arity return_arity return_to
          exn_to).
Proof.
  induction c
    as [h def_vars def_conts recursive | k | c IH extra
       | remaining remaining_arity return_arity return_to _ exn_to _];
    simpl.
  - left. exists h, def_vars, def_conts, recursive. reflexivity.
  - right. left. exists k. reflexivity.
  - exact IH.
  - right. right.
    exists remaining, remaining_arity, return_arity, return_to, exn_to.
    reflexivity.
Qed.

Lemma cont_target_not_extra_args (c c' : cont_value) (extra : list value) :
  cont_target c <> Extra_args c' extra.
Proof.
  destruct (cont_target_cases c)
    as [(h & def_vars & def_conts & recursive & ->)
       | [(k & ->)
         | (remaining & remaining_arity & return_arity & return_to & exn_to
            & ->)]];
    discriminate.
Qed.

Lemma cont_target_with_extra_args (c : cont_value) (extra : list value) :
  cont_target (with_extra_args c extra) = cont_target c.
Proof. destruct extra; reflexivity. Qed.

(* The two ways in which an [Apply_cont] does not go wrong: its
   continuation stands for a handler, which runs under the environments of
   its closure with its parameters bound; or for a way out of the term. *)
Lemma step_apply_cont_handler (ac : apply_cont) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
      (vs : list value)
    (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (handler_vars : env) :
  eval_simples vars (ac_args ac) = Some vs ->
  resolve (ac_continuation ac) conts
  = Closure (Handler params metadata handler) def_vars def_conts recursive ->
  kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata) = true ->
  bind_params params vs def_vars = Some handler_vars ->
  step (State (Apply_cont ac) vars conts statics hp rd)
  = Step
      (State handler handler_vars
         (rec_closures recursive def_vars def_conts ++ def_conts) statics hp
           rd).
Proof.
  intros Hargs Hresolve Hkinds Hbind. rewrite step_apply_cont, Hargs.
  unfold apply_continuation. rewrite Hresolve. simpl.
  rewrite Hkinds, Hbind. reflexivity.
Qed.

Lemma step_apply_cont_exit (ac : apply_cont) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) (vs : list value)
      (k : continuation) :
  eval_simples vars (ac_args ac) = Some vs ->
  resolve (ac_continuation ac) conts = Outer k ->
  step (State (Apply_cont ac) vars conts statics hp rd) = Exit k vs hp rd.
Proof.
  intros Hargs Hresolve. rewrite step_apply_cont, Hargs.
  unfold apply_continuation. rewrite Hresolve. reflexivity.
Qed.

(* The same two, of a continuation that may have extra arguments. *)
Lemma step_apply_cont_target_handler (ac : apply_cont) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
      (vs : list value)
    (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (handler_vars : env) :
  eval_simples vars (ac_args ac) = Some vs ->
  cont_target (resolve (ac_continuation ac) conts)
  = Closure (Handler params metadata handler) def_vars def_conts recursive ->
  kinds_eqb
    (map kind_of_value (cont_args (resolve (ac_continuation ac) conts) vs))
    (hm_params_kinds metadata)
  = true ->
  bind_params params (cont_args (resolve (ac_continuation ac) conts) vs)
    def_vars
  = Some handler_vars ->
  step (State (Apply_cont ac) vars conts statics hp rd)
  = Step
      (State handler handler_vars
         (rec_closures recursive def_vars def_conts ++ def_conts) statics hp
           rd).
Proof.
  intros Hargs Htarget Hkinds Hbind. rewrite step_apply_cont, Hargs.
  unfold apply_continuation. rewrite apply_cont_value_target, Htarget.
  simpl. rewrite Hkinds, Hbind. reflexivity.
Qed.

Lemma step_apply_cont_target_exit (ac : apply_cont) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
      (vs : list value)
    (k : continuation) :
  eval_simples vars (ac_args ac) = Some vs ->
  cont_target (resolve (ac_continuation ac) conts) = Outer k ->
  step (State (Apply_cont ac) vars conts statics hp rd)
  = Exit k (cont_args (resolve (ac_continuation ac) conts) vs) hp rd.
Proof.
  intros Hargs Htarget. rewrite step_apply_cont, Hargs.
  unfold apply_continuation. rewrite apply_cont_value_target, Htarget.
  reflexivity.
Qed.

(* A closure is not a partial application, and a call of it is what
   [shape_of_closure_call] says. *)
Lemma closure_not_partial (statics : static_env) (v : value)
    (closure : code_id * list (value_slot * value)) :
  closure_of statics v = Some closure -> partial_application_of v = None.
Proof. destruct v; simpl; try reflexivity. discriminate. Qed.

Lemma shape_of_call_closure (statics : static_env) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity) :
  partial_application_of callee = None ->
  shape_of_call statics callee kind vs args_arity return_arity
  = shape_of_closure_call statics callee kind vs args_arity return_arity.
Proof.
  intros Hpartial. unfold shape_of_call. rewrite Hpartial.
  destruct kind; reflexivity.
Qed.

(* The arguments as the code takes them are those of the call, for code
   that is not tupled. *)
Lemma arguments_for_code_not_tupled (statics : static_env) (c : code)
    (vs : list value) (args_arity : arity) :
  code_is_tupled c = false ->
  arguments_for_code statics c vs args_arity = Some (vs, args_arity).
Proof.
  intros Htupled. unfold arguments_for_code. rewrite Htupled. reflexivity.
Qed.

(* The call of a closure of code that is bound, the code named if the
   call is direct: full where the arguments are as many as the
   parameters, and otherwise what the tests of arities say. A direct
   call of a closure of the code named is what an indirect one is. *)
Lemma shape_of_closure_call_full (statics : static_env) (closure : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (called : code_id) (slots : list (value_slot * value)) (c : code) :
  closure_of statics closure = Some (called, slots) ->
  kind = Indirect_unknown_arity ∨ kind = Direct called ->
  lookup called (static_code statics) = Some c ->
  code_is_tupled c = false ->
  length args_arity = length (code_params_arity c) ->
  shape_of_closure_call statics closure kind vs args_arity return_arity
  = Some (Full_call closure c vs args_arity).
Proof.
  intros Hclosure Hkind Hcode Htupled Hfull. unfold shape_of_closure_call.
  rewrite Hclosure.
  assert (Hexpected :
    match kind with
    | Direct named => Pos.eqb called named
    | Indirect_unknown_arity => true
    end = true)
    by (destruct Hkind as [-> | ->]; [ reflexivity | apply Pos.eqb_refl ]).
  rewrite Hexpected, Hcode, (arguments_for_code_not_tupled _ _ _ _ Htupled).
  simpl. unfold shape_of_code_call. rewrite Hfull, Nat.eqb_refl. reflexivity.
Qed.

Lemma shape_of_closure_call_not_full (statics : static_env)
    (closure : value) (kind : call_kind) (vs : list value)
    (args_arity return_arity : arity) (called : code_id)
    (slots : list (value_slot * value)) (c : code) :
  closure_of statics closure = Some (called, slots) ->
  kind = Indirect_unknown_arity ∨ kind = Direct called ->
  lookup called (static_code statics) = Some c ->
  code_is_tupled c = false ->
  length args_arity <> length (code_params_arity c) ->
  shape_of_closure_call statics closure kind vs args_arity return_arity
  = if
      negb
        (kinds_eqb (map kind_of_value vs) (unarize args_arity)
         && partial_or_over_application args_arity return_arity
              (code_params_arity c) (code_result_arity c))
    then None
    else if Nat.ltb (length (code_params_arity c)) (length args_arity)
    then
      Some
        (Over_call closure c
           (firstn (cardinal_unarized (code_params_arity c)) vs)
           (skipn (cardinal_unarized (code_params_arity c)) vs)
           (firstn (length (code_params_arity c)) args_arity)
           (partially_apply args_arity (length (code_params_arity c))))
    else Some (Partial_call (Value_partial_application closure vs args_arity)).
Proof.
  intros Hclosure Hkind Hcode Htupled Hcount. unfold shape_of_closure_call.
  rewrite Hclosure.
  assert (Hexpected :
    match kind with
    | Direct named => Pos.eqb called named
    | Indirect_unknown_arity => true
    end = true)
    by (destruct Hkind as [-> | ->]; [ reflexivity | apply Pos.eqb_refl ]).
  apply Nat.eqb_neq in Hcount.
  rewrite Hexpected, Hcode, (arguments_for_code_not_tupled _ _ _ _ Htupled).
  simpl. unfold shape_of_code_call. rewrite Hcount, Htupled. reflexivity.
Qed.

Lemma shape_of_closure_call_kind (statics : static_env) (closure : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (called : code_id) (slots : list (value_slot * value)) :
  closure_of statics closure = Some (called, slots) ->
  kind = Indirect_unknown_arity ∨ kind = Direct called ->
  shape_of_closure_call statics closure kind vs args_arity return_arity
  = shape_of_closure_call statics closure Indirect_unknown_arity vs args_arity
      return_arity.
Proof.
  intros Hclosure [-> | ->]; [ reflexivity | ].
  unfold shape_of_closure_call. rewrite Hclosure, Pos.eqb_refl. reflexivity.
Qed.

Lemma eval_simples_app (vars : env) (args args' : list simple) :
  eval_simples vars (args ++ args')
  = match eval_simples vars args, eval_simples vars args' with
    | Some vs, Some vs' => Some (vs ++ vs')
    | _, _ => None
    end.
Proof.
  induction args as [ | s args IH]; simpl.
  - destruct (eval_simples vars args'); reflexivity.
  - destruct (eval_simple vars s) as [v | ]; simpl; [ | reflexivity ].
    rewrite IH.
    destruct (eval_simples vars args) as [vs | ]; simpl; [ | reflexivity ].
    destruct (eval_simples vars args'); reflexivity.
Qed.

Lemma shape_of_call_direct (statics : static_env) (callee : value)
    (named : code_id) (vs : list value) (args_arity return_arity : arity) :
  shape_of_call statics callee (Direct named) vs args_arity return_arity
  = shape_of_closure_call statics callee (Direct named) vs args_arity
      return_arity.
Proof. reflexivity. Qed.

(* The way in which a full call does not go wrong: its callee is a
   closure of code that is bound, the code named if the call is direct,
   whose body binds nothing static and has as many parameters as there
   are arguments, the extra arguments of its exception continuation
   have values, and the call agrees with the code in the three ways of
   [call_code]. The body runs with [my_closure] bound to the callee,
   which for a closure of a set is the closure at the slot of that
   code. *)
Lemma step_apply_closure (ap : apply_expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) (callee : value)
      (vs extra : list value)
    (called : code_id) (slots : list (value_slot * value)) (c : code)
    (callee_vars : env) :
  eval_simple vars (ap_callee ap) = Some callee ->
  eval_simples vars (ap_args ap) = Some vs ->
  eval_simples vars (exn_extra_simples ap) = Some extra ->
  closure_of statics callee = Some (called, slots) ->
  ap_call_kind ap = Indirect_unknown_arity ∨ ap_call_kind ap = Direct called ->
  lookup called (static_code statics) = Some c ->
  code_is_tupled c = false ->
  binds_statics (code_body c) = false ->
  kinds_eqb (map kind_of_value vs) (unarize (ap_args_arity ap)) = true ->
  kinds_per_parameter_eqb (unarize_per_parameter (ap_args_arity ap))
    (unarize_per_parameter (code_params_arity c))
  = true ->
  arity_eqb (ap_return_arity ap) (code_result_arity c) = true ->
  bind_params (code_params c) vs [(code_my_closure c, callee)]
  = Some callee_vars ->
  step (State (Apply ap) vars conts statics hp rd)
  = Step
      (State (code_body c) callee_vars
         [(code_return_continuation c, resolve (ap_continuation ap) conts);
          (code_exn_continuation c,
           with_extra_args (resolve (ap_exn_continuation ap) conts) extra)]
         statics hp rd).
Proof.
  intros Hcallee Hargs Hextra Hclosure Hkind Hcode Htupled Hbinds Hkinds
    Hparams Hresult Hbind.
  rewrite step_apply, Hcallee, Hargs, Hextra. unfold call_function.
  rewrite (shape_of_call_closure statics callee _ _ _ _
             (closure_not_partial statics callee _ Hclosure)).
  unfold shape_of_closure_call. rewrite Hclosure.
  assert (Hexpected :
    match ap_call_kind ap with
    | Direct named => Pos.eqb called named
    | Indirect_unknown_arity => true
    end = true)
    by (destruct Hkind as [-> | ->]; [ reflexivity | apply Pos.eqb_refl ]).
  rewrite Hexpected, Hcode, (arguments_for_code_not_tupled _ _ _ _ Htupled).
  simpl. unfold shape_of_code_call.
  assert (Hcount :
    Nat.eqb (length (ap_args_arity ap)) (length (code_params_arity c)) = true).
  { apply Nat.eqb_eq. pose proof Hparams as Hsame.
    unfold kinds_per_parameter_eqb in Hsame.
    destruct
      (list_eq_dec (list_eq_dec kind_eq_dec)
         (unarize_per_parameter (ap_args_arity ap))
         (unarize_per_parameter (code_params_arity c)))
      as [Hequal | ]; [ | discriminate ].
    unfold unarize_per_parameter in Hequal.
    rewrite <- (length_map unarize_component (ap_args_arity ap)), Hequal.
    apply length_map. }
  rewrite Hcount. simpl. unfold call_code.
  rewrite Hbinds, Hkinds, Hparams, Hresult, Hbind. reflexivity.
Qed.

(* A [Switch] is undefined, or steps to the arm of its scrutinee. *)
Lemma step_switch_cases (scrutinee : simple) (arms : list (Z * apply_cont))
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  step (State (Switch scrutinee arms) vars conts statics hp rd) = Undefined
  ∨ ∃ (v : value) (i : Z) (ac : apply_cont),
      eval_simple vars scrutinee = Some v
      ∧ naked_immediate_of_value v = Some i
      ∧ find_arm i arms = Some ac
      ∧ step (State (Switch scrutinee arms) vars conts statics hp rd)
        = Step (State (Apply_cont ac) vars conts statics hp rd).
Proof.
  rewrite step_switch.
  destruct (eval_simple vars scrutinee) as [v | ]; [ | left; reflexivity ].
  destruct (naked_immediate_of_value v) as [i | ] eqn:Hi;
    [ | left; reflexivity ].
  destruct (find_arm i arms) as [ac | ] eqn:Hfind; [ | left; reflexivity ].
  right. exists v, i, ac. auto.
Qed.

(* ======================================================================== *)
(* A [Let] and the heap                                                     *)
(* ======================================================================== *)

(* A call and the application of a continuation leave the heap as it
   is. *)
Lemma call_code_heap (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) (s' : state) :
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  = Step s' ->
  state_heap s' = hp.
Proof.
  unfold call_code.
  destruct (binds_statics (code_body c)); [ discriminate | ].
  destruct (negb _); [ discriminate | ].
  destruct (bind_params _ _ _); [ | discriminate ].
  intros [= <-]. reflexivity.
Qed.

Lemma run_call_heap (return_partial : value -> step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
    (s' : state) :
  (∀ (partial : value) (s' : state),
    return_partial partial = Step s' -> state_heap s' = hp) ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  = Step s' ->
  state_heap s' = hp.
Proof.
  intros Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - apply call_code_heap.
  - apply call_code_heap.
  - apply Hpartial.
  - discriminate.
Qed.

Lemma apply_cont_value_heap (c : cont_value) (vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) (s' : state) :
  apply_cont_value c vs statics hp rd = Step s' -> state_heap s' = hp.
Proof.
  revert vs s'.
  induction c
    as [handler vars conts recursive | k | c IH extra
       | remaining remaining_arity return_arity return_to IHreturn exn_to
           IHexn];
    intros vs s'; simpl.
  - destruct handler as [params metadata handler].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs vars); [ | discriminate ].
    intros [= <-]. reflexivity.
  - discriminate.
  - apply IH.
  - destruct vs as [ | v [ | v2 vs]]; try discriminate.
    apply run_call_heap. intros partial s0. apply IHreturn.
Qed.

Lemma call_function_heap (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
    (s' : state) :
  call_function callee kind vs args_arity return_arity return_to exn_to
    statics hp rd
  = Step s' ->
  state_heap s' = hp.
Proof.
  unfold call_function. apply run_call_heap. intros partial s0.
  apply apply_cont_value_heap.
Qed.

(* The same of an exit: it leaves with the heap that it was given. *)
Lemma call_code_exit_heap (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
      (k : continuation)
    (vs' : list value) :
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  = Exit k vs' hp' rd' ->
  hp' = hp.
Proof.
  unfold call_code.
  destruct (binds_statics (code_body c)); [ discriminate | ].
  destruct (negb _); [ discriminate | ].
  destruct (bind_params _ _ _); discriminate.
Qed.

Lemma run_call_exit_heap (return_partial : value -> step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp hp' : heap)
      (rd rd' : read_record)
    (k : continuation) (vs' : list value) :
  (∀ (partial : value) (k : continuation) (vs' : list value) (hp' : heap)
     (rd' : read_record),
    return_partial partial = Exit k vs' hp' rd' -> hp' = hp) ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  = Exit k vs' hp' rd' ->
  hp' = hp.
Proof.
  intros Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - apply call_code_exit_heap.
  - apply call_code_exit_heap.
  - apply Hpartial.
  - discriminate.
Qed.

Lemma apply_cont_value_exit_heap (c : cont_value) (vs : list value)
    (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
      (k : continuation)
    (vs' : list value) :
  apply_cont_value c vs statics hp rd = Exit k vs' hp' rd' -> hp' = hp.
Proof.
  revert vs k vs' hp' rd'.
  induction c
    as [handler vars conts recursive | k0 | c IH extra
       | remaining remaining_arity return_arity return_to IHreturn exn_to
           IHexn];
    intros vs k vs' hp' rd'; simpl.
  - destruct handler as [params metadata handler].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs vars); discriminate.
  - intros [= _ _ H _]. exact (eq_sym H).
  - apply IH.
  - destruct vs as [ | v [ | v2 vs]]; try discriminate.
    apply run_call_exit_heap. intros partial k1 vs1 hp1 rd1. apply IHreturn.
Qed.

Lemma call_function_exit_heap (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp hp' : heap)
      (rd rd' : read_record)
    (k : continuation) (vs' : list value) :
  call_function callee kind vs args_arity return_arity return_to exn_to
    statics hp rd
  = Exit k vs' hp' rd' ->
  hp' = hp.
Proof.
  unfold call_function. apply run_call_exit_heap.
  intros partial k1 vs1 hp1 rd1. apply apply_cont_value_exit_heap.
Qed.

(* The laws of [exec_named], from those of FlambdaHeapSpec.exec_prim. *)
Lemma exec_named_prim (statics : static_env) (hp : heap) (rd : read_record)
    (vars : env)
    (p : prim) (args : list simple) :
  exec_named statics hp rd vars (Prim p args)
  = let* vs := eval_simples vars args in
    let* '(v, hp') := exec_prim statics hp p vs in
    Some (v, hp', rd).
Proof. reflexivity. Qed.

(* A run of a primitive by [exec_named] is a run by [exec_prim] on the
   values of the arguments, and leaves the record as it was. *)
Lemma exec_named_prim_some (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (p : prim) (args : list simple)
    (v : value) :
  exec_named statics hp rd vars (Prim p args) = Some (v, hp', rd')
  <-> ∃ vs : list value,
        eval_simples vars args = Some vs
        ∧ exec_prim statics hp p vs = Some (v, hp') ∧ rd' = rd.
Proof.
  rewrite exec_named_prim.
  destruct (eval_simples vars args) as [vs | ]; simpl.
  - destruct (exec_prim statics hp p vs) as [[w h] | ] eqn:Hprim; simpl.
    + split.
      * intros [= <- <- <-]. exists vs. repeat split. exact Hprim.
      * intros (vs' & [= <-] & Hexec & ->). rewrite Hprim in Hexec.
        injection Hexec as <- <-. reflexivity.
    + split; [ discriminate | ].
      intros (vs' & [= <-] & Hexec & _). rewrite Hprim in Hexec.
      discriminate.
  - split; [ discriminate | intros (vs' & Hvs & _); discriminate ].
Qed.

Lemma eval_named_prim (statics : static_env) (vars : env) (p : prim)
    (args : list simple) :
  eval_named statics vars (Prim p args)
  = let* vs := eval_simples vars args in
    eval_prim statics p vs.
Proof. reflexivity. Qed.

Theorem exec_named_not_prim : exec_named_not_prim_statement.
Proof.
  intros statics hp rd vars d Hnot. destruct d; try reflexivity.
  exfalso. eapply Hnot. reflexivity.
Qed.

Theorem exec_named_args : exec_named_args_statement.
Proof.
  intros statics hp rd vars vars' p args args' Hargs.
  rewrite !exec_named_prim, Hargs. reflexivity.
Qed.

Theorem exec_named_no_args : exec_named_no_args_statement.
Proof.
  intros statics hp rd vars p args Hargs. rewrite exec_named_prim, Hargs.
  reflexivity.
Qed.

Theorem exec_named_static_values : exec_named_static_values_statement.
Proof.
  intros statics statics' hp rd vars p args Hsame. rewrite !exec_named_prim.
  destruct (eval_simples vars args) as [vs | ] eqn:Hvs; [ | reflexivity ].
  simpl.
  rewrite
    (FlambdaHeapProofs.exec_prim_static_values statics statics' hp p vs
       (fun s => Hsame vs s eq_refl)).
  reflexivity.
Qed.

Theorem exec_named_statics_extend : exec_named_statics_extend_statement.
Proof.
  intros statics statics' hp hp' rd rd' vars p args v Hextend Hexec.
  apply exec_named_prim_some in Hexec as (vs & Hvs & Hexec & ->).
  apply exec_named_prim_some. exists vs.
  split; [ exact Hvs | split; [ | reflexivity ] ].
  exact
    (FlambdaHeapProofs.exec_prim_statics_extend statics statics' hp hp' p vs
       v (fun s allocated => Hextend vs s allocated Hvs) Hexec).
Qed.

Theorem exec_named_heap : exec_named_heap_statement.
Proof.
  intros statics hp hp' rd rd' vars d v Hnot Hexec.
  destruct d as [s | p args | | | | | ];
    try (rewrite exec_named_not_prim in Hexec by (intros ? ?; discriminate);
         destruct (eval_named _ _ _); [ | discriminate ];
         injection Hexec as _ <- _; reflexivity).
  apply exec_named_prim_some in Hexec as (vs & _ & Hexec & _).
  refine (FlambdaHeapProofs.exec_prim_heap _ _ _ _ _ _ _ Hexec).
  intros q ->. exact (Hnot q args eq_refl).
Qed.

Theorem exec_named_eval_named : exec_named_eval_named_statement.
Proof.
  intros statics hp hp' rd rd' vars p args v Hreads Hexec.
  apply exec_named_prim_some in Hexec as (vs & Hvs & Hexec & ->).
  rewrite eval_named_prim, Hvs.
  destruct (FlambdaHeapProofs.exec_prim_eval_prim _ _ _ _ _ _ Hreads Hexec)
    as [Heval ->].
  auto.
Qed.

Theorem eval_named_exec_named : eval_named_exec_named_statement.
Proof.
  intros statics hp rd vars p args vs v Hreads Hvs Hbound Heval.
  rewrite eval_named_prim, Hvs in Heval.
  apply exec_named_prim_some. exists vs.
  split; [ exact Hvs | split; [ | reflexivity ] ].
  exact (FlambdaHeapProofs.eval_prim_exec_prim _ _ _ _ _ Hreads Hbound Heval).
Qed.

Theorem exec_named_no_location : exec_named_no_location_statement.
Proof.
  intros statics hp rd vars p args vs Hnot Hvs Hnone.
  rewrite exec_named_prim, eval_named_prim, Hvs. simpl.
  rewrite (FlambdaHeapProofs.exec_prim_no_location _ _ _ _ Hnot Hnone).
  destruct (eval_prim statics p vs); reflexivity.
Qed.

Theorem exec_named_evolves : exec_named_evolves_statement.
Proof.
  intros statics hp hp' rd rd' vars d v Hexec.
  destruct d as [s | p args | | | | | ];
    try (rewrite exec_named_not_prim in Hexec by (intros ? ?; discriminate);
         destruct (eval_named _ _ _); [ | discriminate ];
         injection Hexec as _ <- _; apply FlambdaHeapProofs.heap_evolves_refl).
  apply exec_named_prim_some in Hexec as (vs & _ & Hexec & _).
  exact (FlambdaHeapProofs.exec_prim_evolves _ _ _ _ _ _ Hexec).
Qed.

Theorem exec_named_stable : exec_named_stable_statement.
Proof.
  intros statics hp hp' later rd rd' rd_later vars d v Hstable Hevolves Hexec.
  destruct d as [s | p args | | | | | ];
    try (rewrite exec_named_not_prim in Hexec |- *
           by (intros ? ?; discriminate);
         destruct (eval_named _ _ _); [ | discriminate ];
         injection Hexec as <- _ _; reflexivity).
  apply exec_named_prim_some in Hexec as (vs & Hvs & Hexec & _).
  apply exec_named_prim_some. exists vs.
  split; [ exact Hvs | split; [ | reflexivity ] ].
  simpl in Hstable.
  destruct (effects_of_prim p) eqn:Heffects; try discriminate Hstable.
  destruct (coeffects_of_prim p) eqn:Hcoeffects; try discriminate Hstable.
  exact
    (FlambdaHeapProofs.no_coeffects _ _ _ _ _ _ _ Heffects Hcoeffects
       Hevolves Hexec).
Qed.

Theorem exec_named_preserves : exec_named_preserves_statement.
Proof.
  intros P statics hp hp' rd rd' vars p args v Hnumbers Hblocks Harrays Hstatic
    Hstatic_arrays Hlocations Hargs Hheap Hexec.
  apply exec_named_prim_some in Hexec as (vs & Hvs & Hexec & _).
  exact
    (FlambdaHeapProofs.exec_prim_preserves P statics hp hp' p vs v Hnumbers
       Hblocks Harrays Hstatic Hstatic_arrays Hlocations (Hargs vs Hvs)
       Hheap Hexec).
Qed.

(* The empty heap holds no value. *)
Lemma heap_values_all_empty (P : value -> Prop) :
  heap_values_all P empty_heap.
Proof. intros l c v [ ]. Qed.

(* What [step_values_all_statement] asks at a [Let], from what a proof has
   of [eval_named] for the named expressions that are not primitives, and
   of the values of the arguments for those that are. *)
Lemma exec_named_values_all (P : value -> Prop) (statics : static_env)
    (hp hp' : heap) (rd rd' : read_record) (vars : env) (d : named)
      (v : value) :
  holds_of_numbers P ->
  holds_of_blocks P ->
  holds_of_arrays P ->
  holds_in_static_blocks statics P ->
  holds_in_static_arrays statics P ->
  (∀ l : location, P (Value_location l)) ->
  heap_values_all P hp ->
  (∀ (p : prim) (args : list simple) (vs : list value),
    d = Prim p args ->
    eval_simples vars args = Some vs ->
    ∀ arg : value, arg ∈ vs -> P arg) ->
  ((∀ (p : prim) (args : list simple), d <> Prim p args) ->
   ∀ w : value, eval_named statics vars d = Some w -> P w) ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  P v ∧ heap_values_all P hp'.
Proof.
  intros Hnumbers Hblocks Hof_arrays Hstatic Harrays Hlocations Hheap Hargs
    Hother Hexec.
  assert (Hcases :
            (∃ (p : prim) (args : list simple), d = Prim p args)
            ∨ (∀ (p : prim) (args : list simple), d <> Prim p args)).
  { destruct d; try (right; intros ? ?; discriminate).
    left. eexists. eexists. reflexivity. }
  destruct Hcases as [(p & args & ->) | Hnot].
  - exact (exec_named_preserves P statics hp hp' rd rd' vars p args v Hnumbers
             Hblocks Hof_arrays Hstatic Harrays Hlocations
             (fun vs => Hargs p args vs eq_refl)
             Hheap Hexec).
  - rewrite (exec_named_not_prim _ _ rd_ _ _ Hnot) in Hexec.
    destruct (eval_named statics vars d) as [w | ] eqn:Heval;
      [ | discriminate ].
    injection Hexec as <- <-. split; [ | exact Hheap ].
    exact (Hother Hnot w eq_refl).
Qed.

(* Two primitives the same, on arguments that have the same values,
   under two static environments that bind the symbols among those values
   alike: they do the same. *)
Lemma exec_named_prim_agrees (statics statics' : static_env) (hp : heap)
    (rd : read_record)
    (vars vars' : env) (p : prim) (args args' : list simple) :
  eval_simples vars' args' = eval_simples vars args ->
  (∀ (vs : list value) (s : symbol),
    eval_simples vars args = Some vs ->
    Value_symbol s ∈ vs ->
    lookup s (static_values statics') = lookup s (static_values statics)) ->
  exec_named statics' hp rd vars' (Prim p args')
  = exec_named statics hp rd vars (Prim p args).
Proof.
  intros Hargs Hsame.
  rewrite (exec_named_args statics' hp rd vars vars' p args args' Hargs).
  symmetry. apply exec_named_static_values.
  intros vs s Hvs Hs. symmetry. exact (Hsame vs s Hvs Hs).
Qed.

(* Two named expressions that do the same on a heap, from what a proof
   has of [eval_named]: the two have the same value by [eval_named], or
   none; and where the first is a primitive, the second is the same
   primitive on arguments that have the same values, and the two static
   environments bind the symbols among those values alike. (For a
   primitive [eval_named] says too little: it has no value on a
   location.) *)
Lemma exec_named_agrees (statics statics' : static_env) (hp : heap)
    (rd : read_record)
    (vars vars' : env) (d d' : named) :
  eval_named statics' vars' d' = eval_named statics vars d ->
  (∀ (p : prim) (args : list simple),
    d = Prim p args ->
    ∃ args' : list simple,
      d' = Prim p args'
      ∧ eval_simples vars' args' = eval_simples vars args
      ∧ ∀ (vs : list value) (s : symbol),
          eval_simples vars args = Some vs ->
          Value_symbol s ∈ vs ->
          lookup s (static_values statics')
          = lookup s (static_values statics)) ->
  (∀ (p : prim) (args : list simple),
    d' = Prim p args -> ∃ (p0 : prim) (args0 : list simple), d = Prim p0 args0)
  ->
  exec_named statics' hp rd vars' d' = exec_named statics hp rd vars d.
Proof.
  intros Heval Hprim Hback.
  assert (Hcases :
            (∃ (p : prim) (args : list simple), d = Prim p args)
            ∨ (∀ (p : prim) (args : list simple), d <> Prim p args)).
  { destruct d; try (right; intros ? ?; discriminate).
    left. eexists. eexists. reflexivity. }
  destruct Hcases as [(p & args & ->) | Hnot].
  - destruct (Hprim p args eq_refl) as (args' & -> & Hargs & Hsame).
    exact (exec_named_prim_agrees statics statics' hp rd vars vars' p args args'
             Hargs Hsame).
  - assert (Hnot' : ∀ (p : prim) (args : list simple), d' <> Prim p args).
    { intros p args Hd'. destruct (Hback p args Hd') as (p0 & args0 & Hd).
      exact (Hnot p0 args0 Hd). }
    rewrite (exec_named_not_prim _ _ rd_ _ _ Hnot),
      (exec_named_not_prim _ _ rd_ _ _ Hnot'), Heval.
    reflexivity.
Qed.

(* The same named expression under two environments and two static
   environments. *)
Lemma exec_named_same (statics statics' : static_env) (hp : heap)
    (rd : read_record)
    (vars vars' : env) (d : named) :
  eval_named statics' vars' d = eval_named statics vars d ->
  (∀ (p : prim) (args : list simple),
    d = Prim p args ->
    eval_simples vars' args = eval_simples vars args
    ∧ ∀ (vs : list value) (s : symbol),
        eval_simples vars args = Some vs ->
        Value_symbol s ∈ vs ->
        lookup s (static_values statics')
        = lookup s (static_values statics)) ->
  exec_named statics' hp rd vars' d = exec_named statics hp rd vars d.
Proof.
  intros Heval Hprim.
  apply exec_named_agrees; [ exact Heval | | ].
  - intros p args Hd. exists args.
    exact (conj Hd (Hprim p args Hd)).
  - intros p args Hd. eauto.
Qed.

(* A named expression that has a value keeps it, and returns the same
   heap, under a static environment in which each symbol among the
   values of its arguments that is bound stays bound to the same thing,
   if [eval_named] keeps the value of what is not a primitive. *)
Lemma exec_named_extend (statics statics' : static_env) (hp hp' : heap)
    (rd rd' : read_record)
    (vars : env) (d : named) (v : value) :
  ((∀ (p : prim) (args : list simple), d <> Prim p args) ->
   eval_named statics vars d = Some v ->
   eval_named statics' vars d = Some v) ->
  (∀ (p : prim) (args : list simple) (vs : list value) (s : symbol)
     (allocated : statically_allocated_value),
    d = Prim p args ->
    eval_simples vars args = Some vs ->
    Value_symbol s ∈ vs ->
    lookup s (static_values statics) = Some allocated ->
    lookup s (static_values statics') = Some allocated) ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  exec_named statics' hp rd vars d = Some (v, hp', rd').
Proof.
  intros Heval Hprim Hexec.
  assert (Hcases :
            (∃ (p : prim) (args : list simple), d = Prim p args)
            ∨ (∀ (p : prim) (args : list simple), d <> Prim p args)).
  { destruct d; try (right; intros ? ?; discriminate).
    left. eexists. eexists. reflexivity. }
  destruct Hcases as [(p & args & ->) | Hnot].
  - exact
    (exec_named_statics_extend statics statics' hp hp' rd rd' vars p args v
             (fun vs s allocated => Hprim p args vs s allocated eq_refl)
             Hexec).
  - rewrite (exec_named_not_prim statics hp rd vars d Hnot) in Hexec.
    rewrite (exec_named_not_prim statics' hp rd vars d Hnot).
    destruct (eval_named statics vars d) as [w | ] eqn:Hw; [ | discriminate ].
    injection Hexec as <- <- <-. rewrite (Heval Hnot eq_refl). reflexivity.
Qed.

(* A named expression that does not change the heap
   (FlambdaHeapSpec.named_changes_heap) returns it as it came. *)
Lemma exec_named_same_heap (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record)
    (vars : env) (d : named) (v : value) :
  named_changes_heap d = false ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  hp' = hp.
Proof.
  intros Hchanges. apply exec_named_heap. intros q args ->.
  discriminate Hchanges.
Qed.

(* A named expression that does not read the heap
   (FlambdaHeapSpec.named_reads_heap) and that has a value has it by
   [eval_named], and the heap and the record are as they came. *)
Lemma exec_named_pure (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env)
    (d : named) (v : value) :
  named_reads_heap d = false ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  eval_named statics vars d = Some v ∧ hp' = hp ∧ rd' = rd.
Proof.
  intros Hreads Hexec.
  destruct d
    as [s | p args | code slots | fslot decls slots | slot closure
       | pfrom slot closure | move_from move_to closure];
    try (rewrite exec_named_not_prim in Hexec by (intros ? ?; discriminate);
         destruct (eval_named _ _ _) as [w | ]; [ | discriminate ];
         injection Hexec as <- <- <-; repeat split; reflexivity).
  exact
    (exec_named_eval_named statics hp hp' rd rd' vars p args v Hreads Hexec).
Qed.

(* The step of a [Let] by [eval_named], where the heap is not looked at:
   of what is not a primitive, and of a primitive that is not of the heap
   on arguments none of whose values is a location. These hold of
   [exec_named] as it is and as it is to be. *)
Lemma step_let_not_prim (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record) :
  (∀ (p : prim) (args : list simple), d <> Prim p args) ->
  step (State (Let x kind d body) vars conts statics hp rd)
  = match eval_named statics vars d with
    | Some v =>
      if kind_eqb (kind_of_value v) kind
      then Step (State body ((x, v) :: vars) conts statics hp rd)
      else Undefined
    | None => Undefined
    end.
Proof.
  intros Hnot. rewrite step_let_exec, (exec_named_not_prim _ _ rd_ _ _ Hnot).
  destruct (eval_named statics vars d); reflexivity.
Qed.

Lemma step_let_simple (x : variable) (kind : flambda_kind) (s : simple)
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record) :
  step (State (Let x kind (Simple s) body) vars conts statics hp rd)
  = match eval_simple vars s with
    | Some v =>
      if kind_eqb (kind_of_value v) kind
      then Step (State body ((x, v) :: vars) conts statics hp rd)
      else Undefined
    | None => Undefined
    end.
Proof. apply step_let_not_prim. intros p args. discriminate. Qed.

Lemma step_let_no_location (x : variable) (kind : flambda_kind) (p : prim)
    (args : list simple) (body : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) (vs : list value) :
  (∀ q : heap_primitive, p <> Heap q) ->
  eval_simples vars args = Some vs ->
  (∀ l : location, Value_location l ∉ vs) ->
  step (State (Let x kind (Prim p args) body) vars conts statics hp rd)
  = match eval_prim statics p vs with
    | Some v =>
      if kind_eqb (kind_of_value v) kind
      then Step (State body ((x, v) :: vars) conts statics hp rd)
      else Undefined
    | None => Undefined
    end.
Proof.
  intros Hnot Hvs Hnone.
  rewrite step_let_exec,
    (exec_named_no_location _ _ rd_ _ _ _ _ Hnot Hvs Hnone).
  simpl. rewrite Hvs. destruct (eval_prim statics p vs); reflexivity.
Qed.

(* The step of a [Let] by [eval_named] and with the heap as it came, as
   it was before a state had a heap: of anything but a primitive of the
   heap, where no location is among the values of the arguments of a
   primitive. A proof about states that hold no location (the values of
   the two runs are related by a relation that has no case for one)
   steps a [Let] by this. *)
Lemma step_let_eval (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record) :
  (∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args) ->
  (∀ (p : prim) (args : list simple) (vs : list value) (l : location),
    d = Prim p args ->
    eval_simples vars args = Some vs ->
    Value_location l ∉ vs) ->
  step (State (Let x kind d body) vars conts statics hp rd)
  = match eval_named statics vars d with
    | Some v =>
      if kind_eqb (kind_of_value v) kind
      then Step (State body ((x, v) :: vars) conts statics hp rd)
      else Undefined
    | None => Undefined
    end.
Proof.
  intros Hheap Hnone.
  assert (Hcases :
            (∃ (p : prim) (args : list simple), d = Prim p args)
            ∨ (∀ (p : prim) (args : list simple), d <> Prim p args)).
  { destruct d; try (right; intros ? ?; discriminate).
    left. eexists. eexists. reflexivity. }
  destruct Hcases as [(p & args & ->) | Hnot];
    [ | exact (step_let_not_prim x kind d body vars conts statics hp rd Hnot) ].
  destruct (eval_simples vars args) as [vs | ] eqn:Hvs.
  - rewrite
      (step_let_no_location x kind p args body vars conts statics hp rd vs
         (fun q Hq => Hheap q args (f_equal (fun p => Prim p args) Hq)) Hvs
         (fun l => Hnone p args vs l eq_refl Hvs)).
    simpl. rewrite Hvs. reflexivity.
  - rewrite step_let_exec, (exec_named_no_args statics hp rd vars p args Hvs).
    simpl. rewrite Hvs. reflexivity.
Qed.

(* The [Let] of a closure of a new set, and of the projection of a
   function slot. *)
Lemma step_let_closure_of_set (x : variable) (kind : flambda_kind)
    (slot : function_slot) (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  step
    (State (Let x kind (Closure_of_set slot function_decls value_slots) body)
       vars conts statics hp rd)
  = match lookup slot function_decls, eval_value_slots vars value_slots with
    | Some _, Some vs =>
      if kind_eqb Value kind
      then
        Step
          (State body
             ((x, Value_closure_of_set slot function_decls vs) :: vars) conts
             statics hp rd)
      else Undefined
    | _, _ => Undefined
    end.
Proof.
  rewrite step_let_not_prim by (intros p args; discriminate). simpl.
  destruct (lookup slot function_decls); [ | reflexivity ].
  destruct (eval_value_slots vars value_slots); reflexivity.
Qed.

Lemma step_let_project_value_slot_from (x : variable) (kind : flambda_kind)
    (project_from : function_slot) (slot : value_slot) (closure : simple)
    (body : expr) (vars : env) (conts : cont_env) (statics : static_env)
      (hp : heap) (rd : read_record) :
  step
    (State
       (Let x kind (Project_value_slot_from project_from slot closure) body)
       vars conts statics hp rd)
  = match eval_simple vars closure with
    | Some v =>
      match project_value_slot statics project_from slot v with
      | Some w =>
        if kind_eqb (kind_of_value w) kind
        then Step (State body ((x, w) :: vars) conts statics hp rd)
        else Undefined
      | None => Undefined
      end
    | None => Undefined
    end.
Proof.
  rewrite step_let_not_prim by (intros p args; discriminate). simpl.
  destruct (eval_simple vars closure); reflexivity.
Qed.

Lemma step_let_project_function_slot (x : variable) (kind : flambda_kind)
    (move_from move_to : function_slot) (closure : simple) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  step
    (State
       (Let x kind (Project_function_slot move_from move_to closure) body)
       vars conts statics hp rd)
  = match eval_simple vars closure with
    | Some v =>
      match project_function_slot statics move_from move_to v with
      | Some w =>
        if kind_eqb (kind_of_value w) kind
        then Step (State body ((x, w) :: vars) conts statics hp rd)
        else Undefined
      | None => Undefined
      end
    | None => Undefined
    end.
Proof.
  rewrite step_let_not_prim by (intros p args; discriminate). simpl.
  destruct (eval_simple vars closure); reflexivity.
Qed.

Theorem step_evolves : step_evolves_statement.
Proof.
  intros [e vars conts statics hp rd] s'.
  destruct e
    as [x kind d body | bindings body | k handler body | handlers body | ap
       | ac | scrutinee arms | ];
    [ rewrite step_let_exec | simpl .. ].
  - destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hexec;
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    intros [= <-]. simpl. exact (exec_named_evolves _ _ _ rd_ rd_ _ _ _ Hexec).
  - destruct (eval_static_bindings vars bindings statics);
      [ | discriminate ].
    intros [= <-]. apply FlambdaHeapProofs.heap_evolves_refl.
  - intros [= <-]. apply FlambdaHeapProofs.heap_evolves_refl.
  - destruct (handlers_bind_statics handlers); [ discriminate | ].
    intros [= <-]. apply FlambdaHeapProofs.heap_evolves_refl.
  - destruct (eval_simple vars (ap_callee ap)); [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)); [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)); [ | discriminate ].
    intros Hstep. rewrite (call_function_heap _ _ _ _ _ _ _ _ _ rd_ _ Hstep).
    apply FlambdaHeapProofs.heap_evolves_refl.
  - destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    intros Hstep. unfold apply_continuation in Hstep.
    rewrite (apply_cont_value_heap _ _ _ _ rd_ _ Hstep).
    apply FlambdaHeapProofs.heap_evolves_refl.
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    intros [= <-]. apply FlambdaHeapProofs.heap_evolves_refl.
  - discriminate.
Qed.

Theorem step_heap : step_heap_statement.
Proof.
  intros [e vars conts statics hp rd] s'.
  destruct e
    as [x kind d body | bindings body | k handler body | handlers body | ap
       | ac | scrutinee arms | ];
    [ rewrite step_let_exec | simpl .. ].
  - destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hexec;
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    intros [= <-]. right. exists x, kind, d, body, v.
    split; [ reflexivity | exact Hexec ].
  - destruct (eval_static_bindings vars bindings statics);
      [ | discriminate ].
    intros [= <-]. left. reflexivity.
  - intros [= <-]. left. reflexivity.
  - destruct (handlers_bind_statics handlers); [ discriminate | ].
    intros [= <-]. left. reflexivity.
  - destruct (eval_simple vars (ap_callee ap)); [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)); [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)); [ | discriminate ].
    intros Hstep. left. exact
      (call_function_heap _ _ _ _ _ _ _ _ _ rd_ _ Hstep).
  - destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    intros Hstep. unfold apply_continuation in Hstep. left.
    exact (apply_cont_value_heap _ _ _ _ rd_ _ Hstep).
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; [ | discriminate ].
    intros [= <-]. left. reflexivity.
  - discriminate.
Qed.

Theorem step_exit_heap : step_exit_heap_statement.
Proof.
  intros [e vars conts statics hp rd] k vs hp' rd'.
  destruct e
    as [x kind d body | bindings body | k0 handler body | handlers body | ap
       | ac | scrutinee arms | ];
    [ rewrite step_let_exec | simpl .. ].
  - destruct (exec_named statics hp rd vars d) as [[[v hp1] rd1] | ];
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); discriminate.
  - destruct (eval_static_bindings vars bindings statics); discriminate.
  - discriminate.
  - destruct (handlers_bind_statics handlers); discriminate.
  - destruct (eval_simple vars (ap_callee ap)); [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)); [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)); [ | discriminate ].
    apply call_function_exit_heap.
  - destruct (eval_simples vars (ac_args ac)) as [vs0 | ]; [ | discriminate ].
    unfold apply_continuation. apply apply_cont_value_exit_heap.
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ]; discriminate.
  - discriminate.
Qed.

#[global] Opaque step.

(* For a proof by cases on the construct of a state:
   [autorewrite with step] rewrites the step of each case. *)
Create Rewrite HintDb step.
#[global] Hint Rewrite step_let_exec step_let_static step_let_cont_non_recursive
  step_let_cont_recursive step_apply step_apply_cont step_switch
  step_invalid : step.

(* ======================================================================== *)
(* Closures of sets                                                         *)
(* ======================================================================== *)

Theorem closure_in_set_of : closure_in_set_of_statement.
Proof.
  intros statics v code value_slots. unfold closure_of, closure_in_set.
  destruct v as [c | code' slots | slot decls slots | s | tag fields
    | | contents | l | pa_callee pa_args pa_arity];
    [ | | | | split; [ discriminate | intros (slot & decls & [=] & _) ] .. ].
  - split; [ discriminate | intros (slot & decls & [=] & _) ].
  - split.
    + intros [= <- <-]. exists unnamed_function_slot. eexists.
      split; [ reflexivity | apply lookup_head ].
    + intros (slot & decls & [= <- <- <-] & Hcode).
      rewrite lookup_head in Hcode. injection Hcode as <-. reflexivity.
  - destruct (lookup slot decls) as [code' | ] eqn:Hslot; simpl.
    + split.
      * intros [= <- <-]. eauto.
      * intros (slot' & decls' & [= <- <- <-] & Hcode). congruence.
    + split; [ discriminate | intros (slot' & decls' & [=] & _) ].
  - destruct (lookup s (static_values statics))
      as [[code' slots | slot syms decls slots | tag fields | contents
          | bytes | elements | ] | ];
      try (split; [ discriminate | intros (slot' & decls' & [=] & _) ]).
    + split.
      * intros [= <- <-]. exists unnamed_function_slot. eexists.
        split; [ reflexivity | apply lookup_head ].
      * intros (slot & decls & [= <- <- <-] & Hcode).
        rewrite lookup_head in Hcode. injection Hcode as <-. reflexivity.
    + destruct (lookup slot decls) as [code' | ] eqn:Hslot; simpl.
      * split.
        -- intros [= <- <-]. eauto.
        -- intros (slot' & decls' & [= <- <- <-] & Hcode). congruence.
      * split; [ discriminate | intros (slot' & decls' & [=] & _) ].
Qed.

Theorem closure_in_set_slot : closure_in_set_slot_statement.
Proof.
  intros statics v slot decls slots. unfold closure_in_set, is_bound.
  destruct v as [c | code vs | fslot fdecls vs | s | tag fields | | contents
    | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - intros [= <- <- _]. reflexivity.
  - destruct (lookup fslot fdecls) as [code | ] eqn:Hslot; [ | discriminate ].
    simpl. intros [= <- <- _]. rewrite Hslot. reflexivity.
  - destruct (lookup s (static_values statics))
      as [[code vs | fslot syms fdecls vs | tag fields | contents
          | bytes | elements | ] | ];
      try discriminate.
    + intros [= <- <- _]. reflexivity.
    + destruct (lookup fslot fdecls) as [code | ] eqn:Hslot;
        [ | discriminate ].
      simpl. intros [= <- <- _]. rewrite Hslot. reflexivity.
Qed.

Theorem closure_of_set_made : closure_of_set_made_statement.
Proof.
  intros statics vars slot decls value_slots v. simpl.
  destruct (lookup slot decls) as [code | ] eqn:Hslot; [ | discriminate ].
  destruct (eval_value_slots vars value_slots) as [slot_values | ];
    [ | discriminate ].
  intros [= <-]. exists slot_values. simpl. rewrite Hslot. auto.
Qed.

Theorem static_closure_of_set_made : static_closure_of_set_made_statement.
Proof.
  intros statics vars slot symbols decls value_slots s allocated. simpl.
  destruct (lookup slot decls) as [code | ] eqn:Hslot; [ | discriminate ].
  destruct (eval_value_slots vars value_slots) as [slot_values | ];
    [ | discriminate ].
  intros [= <-] Hs. exists slot_values. simpl. rewrite Hs, Hslot. auto.
Qed.

Theorem project_value_slot_defined : project_value_slot_defined_statement.
Proof.
  intros statics project_from slot v w. unfold project_value_slot.
  destruct (closure_in_set statics v) as [[[at_slot decls] slots] | ].
  - destruct (Pos.eqb_spec at_slot project_from) as [-> | Hne].
    + split; [ eauto | intros (decls' & slots' & [= <- <-] & Hslot) ].
      exact Hslot.
    + split; [ discriminate | intros (decls' & slots' & [= Heq _ _] & _) ].
      contradiction.
  - split; [ discriminate | intros (decls' & slots' & [=] & _) ].
Qed.

(* The projection from a closure of a set of one function with no
   function slot named: it is defined at that slot only. *)
Lemma project_unnamed_function_slot (move_from move_to : function_slot)
    (v w : value) :
  (if Pos.eqb move_from unnamed_function_slot
   then if Pos.eqb move_to unnamed_function_slot then Some v else None
   else None)
  = Some w ->
  move_from = unnamed_function_slot ∧ move_to = unnamed_function_slot
  ∧ w = v.
Proof.
  destruct (Pos.eqb_spec move_from unnamed_function_slot) as [-> | _];
    [ | discriminate ].
  destruct (Pos.eqb_spec move_to unnamed_function_slot) as [-> | _];
    [ | discriminate ].
  intros [= <-]. auto.
Qed.

Theorem project_function_slot_defined :
  project_function_slot_defined_statement.
Proof.
  intros statics move_from move_to v w.
  unfold project_function_slot, closure_in_set.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | | contents | l | pa_callee pa_args pa_arity];
    [ | | | | discriminate .. ].
  - discriminate.
  - intros Hproject.
    apply project_unnamed_function_slot in Hproject as (-> & -> & _).
    exists [(unnamed_function_slot, code)], slots, code, code.
    rewrite lookup_head. auto.
  - destruct (Pos.eqb_spec slot move_from) as [-> | _]; [ | discriminate ].
    destruct (lookup move_from decls) as [code | ] eqn:Hfrom;
      [ | discriminate ].
    destruct (lookup move_to decls) as [code' | ] eqn:Hto; [ | discriminate ].
    simpl. intros _. exists decls, slots, code, code'. auto.
  - destruct (lookup s (static_values statics))
      as [[code slots | slot syms decls slots | tag fields | contents
          | bytes | elements | ] | ];
      try discriminate.
    + intros Hproject.
      apply project_unnamed_function_slot in Hproject as (-> & -> & _).
      exists [(unnamed_function_slot, code)], slots, code, code.
      rewrite lookup_head. auto.
    + destruct (Pos.eqb_spec slot move_from) as [-> | _]; [ | discriminate ].
      destruct (lookup move_from decls) as [code | ] eqn:Hfrom;
        [ | discriminate ].
      destruct (lookup move_to decls) as [code' | ] eqn:Hto;
        [ | discriminate ].
      simpl. intros _. exists decls, slots, code, code'. auto.
Qed.

Theorem project_function_slot_of_set :
  project_function_slot_of_set_statement.
Proof.
  intros statics move_from move_to decls slots code code' Hfrom Hto.
  unfold project_function_slot. rewrite Pos.eqb_refl, Hfrom, Hto.
  reflexivity.
Qed.

(* ======================================================================== *)
(* Bounded runs                                                             *)
(* ======================================================================== *)

(* A run of one step more is the first step and, if that has not ended,
   the rest of the run. *)
Lemma run_succ (n : nat) (s : state) :
  run (S n) s
  = match step s with
    | Step s' => run n s'
    | result => result
    end.
Proof. reflexivity. Qed.

(* A run of [n + m] steps is a run of [n] steps and, if that has not ended,
   one of [m] steps from the state reached. *)
Lemma run_add (n m : nat) (s : state) :
  run (n + m) s
  = match run n s with
    | Step s' => run m s'
    | result => result
    end.
Proof.
  revert s. induction n as [ | n IH]; intros s; simpl; [ reflexivity | ].
  destruct (step s); [ apply IH | reflexivity | reflexivity ].
Qed.

(* ======================================================================== *)
(* Behaviours                                                               *)
(* ======================================================================== *)

(* What a run that has ended came to, as a behaviour. *)
Definition ended (result : step_result) (b : behaviour) : Prop :=
  match result, b with
  | Exit k vs hp rd, Exits k' vs' hp' => k = k' ∧ vs = vs' ∧ hp = hp'
  | Undefined, Goes_wrong => True
  | _, _ => False
  end.

(* A terminating behaviour is a run that has ended with it. *)
Lemma behaves_ended (s : state) (b : behaviour) :
  b <> Diverges -> behaves s b -> ∃ n : nat, ended (run n s) b.
Proof.
  destruct b as [k vs hp | | ]; simpl; intros Hb; [ | | congruence ].
  - intros [n [rd Hrun]]. exists n. rewrite Hrun. simpl. auto.
  - intros [n Hrun]. exists n. rewrite Hrun. simpl. auto.
Qed.

(* More fuel does not change how a run ended. *)
Lemma ended_stable (n m : nat) (s : state) (b : behaviour) :
  ended (run n s) b -> run (n + m) s = run n s.
Proof.
  intros Hended. rewrite run_add.
  destruct (run n s); [ destruct b; contradiction | | ]; reflexivity.
Qed.

(* A run that has ended did so in one way. *)
Lemma ended_unique (result : step_result) (b b' : behaviour) :
  ended result b -> ended result b' -> b = b'.
Proof.
  destruct result, b, b'; simpl; try contradiction; try reflexivity.
  intros [<- [<- <-]] [<- [<- <-]]. reflexivity.
Qed.

(* A state that diverges has no run that ends. *)
Lemma diverges_not_ended (n : nat) (s : state) (b : behaviour) :
  behaves s Diverges -> ended (run n s) b -> False.
Proof.
  intros Hdiv Hended. destruct (Hdiv n) as [s' Hrun].
  rewrite Hrun in Hended. destruct b; contradiction.
Qed.

(* Two runs of one state that have both ended have ended alike: give both
   the fuel of the two together. *)
Theorem behaviour_unique : behaviour_unique_statement.
Proof.
  intros s b b' Hb Hb'.
  assert (Hdec : ∀ c : behaviour, c = Diverges ∨ c <> Diverges)
    by (intros [ | | ]; [ right | right | left ]; congruence).
  destruct (Hdec b) as [-> | Hterm], (Hdec b') as [-> | Hterm'].
  - reflexivity.
  - destruct (behaves_ended s b' Hterm' Hb') as [n Hended].
    destruct (diverges_not_ended n s b' Hb Hended).
  - destruct (behaves_ended s b Hterm Hb) as [n Hended].
    destruct (diverges_not_ended n s b Hb' Hended).
  - destruct (behaves_ended s b Hterm Hb) as [n Hended].
    destruct (behaves_ended s b' Hterm' Hb') as [m Hended'].
    rewrite <- (ended_stable n m s b Hended) in Hended.
    rewrite <- (ended_stable m n s b' Hended'), Nat.add_comm in Hended'.
    exact (ended_unique _ _ _ Hended Hended').
Qed.

(* The behaviours of what a step or a run comes to. *)
Definition result_behaves (result : step_result) (b : behaviour) : Prop :=
  match result with
  | Step s => behaves s b
  | Exit k vs hp rd => b = Exits k vs hp
  | Undefined => b = Goes_wrong
  end.

(* The behaviours of a state are those of what its first step comes to. *)
Lemma step_behaves (s : state) (b : behaviour) :
  behaves s b <-> result_behaves (step s) b.
Proof.
  destruct b as [k vs | | ]; simpl; split.
  - intros [[ | n] [rd Hrun]]; [ discriminate | ]. simpl in Hrun.
    destruct (step s); simpl; [ eauto | congruence | discriminate ].
  - destruct (step s) eqn:Hstep; simpl.
    + intros [n [rd Hrun]]. exists (S n), rd. simpl. rewrite Hstep.
      exact Hrun.
    + intros [= <- <- <-]. exists 1. eexists. simpl. rewrite Hstep.
      reflexivity.
    + discriminate.
  - intros [[ | n] Hrun]; [ discriminate | ]. simpl in Hrun.
    destruct (step s); simpl; [ eauto | discriminate | reflexivity ].
  - destruct (step s) eqn:Hstep; simpl.
    + intros [n Hrun]. exists (S n). simpl. rewrite Hstep. exact Hrun.
    + discriminate.
    + intros _. exists 1. simpl. rewrite Hstep. reflexivity.
  - intros Hdiv. destruct (step s) eqn:Hstep; simpl.
    + intros n. specialize (Hdiv (S n)). simpl in Hdiv.
      rewrite Hstep in Hdiv. exact Hdiv.
    + destruct (Hdiv 1) as [s' Hrun]. simpl in Hrun.
      rewrite Hstep in Hrun. discriminate.
    + destruct (Hdiv 1) as [s' Hrun]. simpl in Hrun.
      rewrite Hstep in Hrun. discriminate.
  - destruct (step s) eqn:Hstep; simpl; [ | discriminate | discriminate ].
    intros Hdiv [ | n]; simpl; [ eauto | ]. rewrite Hstep. apply Hdiv.
Qed.

(* The same for a run of any length. *)
Lemma run_behaves (n : nat) (s : state) (b : behaviour) :
  behaves s b <-> result_behaves (run n s) b.
Proof.
  revert s. induction n as [ | n IH]; intros s; [ reflexivity | ].
  rewrite step_behaves. simpl.
  destruct (step s); simpl; [ apply IH | reflexivity | reflexivity ].
Qed.

(* ======================================================================== *)
(* Equivalence and refinement                                               *)
(* ======================================================================== *)

Lemma equivalent_refl (s : state) : equivalent s s.
Proof. intros b. reflexivity. Qed.

Lemma equivalent_sym (s s' : state) : equivalent s s' -> equivalent s' s.
Proof. intros Hequiv b. symmetry. apply Hequiv. Qed.

Lemma equivalent_trans (s s' s'' : state) :
  equivalent s s' -> equivalent s' s'' -> equivalent s s''.
Proof. intros H H' b. rewrite (H b). apply H'. Qed.

(* Two states whose runs, of whatever lengths, come to the same thing are
   equivalent. *)
Lemma equivalent_of_run (n m : nat) (s s' : state) :
  run n s = run m s' -> equivalent s s'.
Proof.
  intros Hrun b. rewrite (run_behaves n s), (run_behaves m s'), Hrun.
  reflexivity.
Qed.

Theorem step_equivalent : step_equivalent_statement.
Proof.
  intros s s' Hstep. apply (equivalent_of_run 1 0). simpl.
  rewrite Hstep. reflexivity.
Qed.

(* A run that has ended with a behaviour gives that behaviour. *)
Lemma ended_behaves (n : nat) (s : state) (b : behaviour) :
  ended (run n s) b -> behaves s b.
Proof.
  destruct b as [k vs | | ], (run n s) as [s' | k' vs' | ] eqn:Hrun; simpl;
    try contradiction.
  - intros [-> [-> ->]]. exists n. eexists. exact Hrun.
  - intros _. exists n. exact Hrun.
Qed.

Lemma ended_undefined (b : behaviour) : ended Undefined b -> b = Goes_wrong.
Proof. destruct b; simpl; try contradiction. reflexivity. Qed.

Lemma ended_exit (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record)
    (b : behaviour) :
  ended (Exit k vs hp rd) b -> b = Exits k vs hp.
Proof.
  destruct b; simpl; try contradiction. intros [-> [-> ->]]. reflexivity.
Qed.

Lemma diverges_dec (b : behaviour) : b = Diverges ∨ b <> Diverges.
Proof. destruct b; [ right | right | left ]; congruence. Qed.

(* If [s] has every behaviour of [s'] that ends, and [s] diverges, then
   [s'] diverges: a run of [s'] that had ended would be one of [s]. *)
Lemma diverges_of_terminating (s s' : state) :
  (∀ b : behaviour, b <> Diverges -> behaves s' b -> behaves s b) ->
  behaves s Diverges -> behaves s' Diverges.
Proof.
  intros Hterm Hdiv. simpl. intros n.
  destruct (run n s') as [u | k vs | ] eqn:Hrun; [ exists u; reflexivity | | ];
    exfalso.
  - assert (Hexit : behaves s' (Exits k vs hp))
      by (exists n; eexists; exact Hrun).
    apply Hterm in Hexit; [ | discriminate ].
    pose proof (behaviour_unique _ _ _ Hdiv Hexit). discriminate.
  - assert (Hwrong : behaves s' Goes_wrong) by (exists n; exact Hrun).
    apply Hterm in Hwrong; [ | discriminate ].
    pose proof (behaviour_unique _ _ _ Hdiv Hwrong). discriminate.
Qed.

(* Two states with the same behaviours that end are equivalent: divergence
   need not be considered. *)
Lemma equivalent_of_terminating (s s' : state) :
  (∀ b : behaviour, b <> Diverges -> behaves s b <-> behaves s' b) ->
  equivalent s s'.
Proof.
  intros Hterm b.
  destruct (diverges_dec b) as [-> | Hb]; [ | exact (Hterm b Hb) ].
  split; apply diverges_of_terminating; intros b Hb Hbehaves;
    apply (Hterm b Hb); exact Hbehaves.
Qed.

Lemma equivalent_refines (s s' : state) : equivalent s s' -> refines s s'.
Proof.
  intros Hequiv. split.
  - intros k vs hp Hb. apply Hequiv, Hb.
  - intros b _ Hb. left. apply Hequiv, Hb.
Qed.

Lemma refines_refl (s : state) : refines s s.
Proof. apply equivalent_refines, equivalent_refl. Qed.

Lemma refines_trans (s s' s'' : state) :
  refines s s' -> refines s' s'' -> refines s s''.
Proof.
  intros [Hexits Hends] [Hexits' Hends']. split.
  - intros k vs hp Hb. apply Hexits', Hexits, Hb.
  - intros b Hterm Hb.
    destruct (Hends' b Hterm Hb) as [Hb' | Hwrong];
      [ exact (Hends b Hterm Hb') | ].
    destruct (Hends Goes_wrong ltac:(discriminate) Hwrong); right; assumption.
Qed.

(* Anything refines a state that goes wrong. *)
Lemma goes_wrong_refines (s s' : state) : behaves s Goes_wrong -> refines s s'.
Proof.
  intros Hwrong. split.
  - intros k vs hp Hb. pose proof (behaviour_unique _ _ _ Hwrong Hb).
    discriminate.
  - intros b _ _. right. exact Hwrong.
Qed.

Theorem refines_defined : refines_defined_statement.
Proof.
  intros s s' [Hexits Hends] Hdefined. apply equivalent_of_terminating.
  intros b Hterm. split; intros Hb.
  - destruct b; [ apply Hexits, Hb | contradiction | congruence ].
  - destruct (Hends b Hterm Hb); [ assumption | contradiction ].
Qed.

Theorem refines_antisymmetric : refines_antisymmetric_statement.
Proof.
  intros s s' [Hexits Hends] [Hexits' Hends']. apply equivalent_of_terminating.
  intros b Hterm. split; intros Hb.
  - destruct b; [ apply Hexits, Hb | | congruence ].
    destruct (Hends' Goes_wrong Hterm Hb); assumption.
  - destruct b; [ apply Hexits', Hb | | congruence ].
    destruct (Hends Goes_wrong Hterm Hb); assumption.
Qed.

(* Refinement is preserved backwards along steps. *)
Lemma refines_step (s s' t t' : state) :
  step s = Step s' -> step t = Step t' -> refines s' t' -> refines s t.
Proof.
  intros Hs Ht Hrefines.
  apply (refines_trans s s');
    [ apply equivalent_refines, step_equivalent, Hs | ].
  apply (refines_trans s' t'); [ exact Hrefines | ].
  apply equivalent_refines, equivalent_sym, step_equivalent, Ht.
Qed.

(* Both sides take a step, to states that are equivalent. *)
Lemma equivalent_step (s s' t t' : state) :
  step s = Step s' -> step t = Step t' -> equivalent s' t' -> equivalent s t.
Proof.
  intros Hs Ht Hequiv.
  apply (equivalent_trans s s'); [ apply step_equivalent, Hs | ].
  apply (equivalent_trans s' t'); [ exact Hequiv | ].
  apply equivalent_sym, step_equivalent, Ht.
Qed.

(* Two states whose first steps come to the same thing are equivalent. *)
Lemma step_eq_equivalent (s s' : state) : step s = step s' -> equivalent s s'.
Proof.
  intros Hstep b. rewrite (step_behaves s), (step_behaves s'), Hstep.
  reflexivity.
Qed.

(* A state whose step is undefined goes wrong, so anything refines it. *)
Lemma step_undefined_wrong (s : state) :
  step s = Undefined -> behaves s Goes_wrong.
Proof.
  intros Hstep. apply (step_behaves s Goes_wrong). rewrite Hstep.
  reflexivity.
Qed.

Lemma step_undefined_refines (s s' : state) :
  step s = Undefined -> refines s s'.
Proof. intros Hstep. apply goes_wrong_refines, step_undefined_wrong, Hstep. Qed.

Lemma expr_equivalent_refines (e e' : expr) :
  expr_equivalent e e' -> expr_refines e e'.
Proof. intros Hequiv vars conts statics hp rd. apply equivalent_refines, Hequiv.
  Qed.

(* ======================================================================== *)
(* Bisimulations                                                            *)
(* ======================================================================== *)

(* What two steps, or two runs of one length, come to, alike up to [R]:
   states related by [R], the same exit, or both undefined. *)
Definition result_related (R : state -> state -> Prop)
    (result result' : step_result) : Prop :=
  match result, result' with
  | Step s, Step s' => R s s'
  | Exit k vs hp rd, Exit k' vs' hp' rd' => k = k' ∧ vs = vs' ∧ hp = hp' ∧ rd
    = rd'
  | Undefined, Undefined => True
  | _, _ => False
  end.

Lemma result_related_refl (R : state -> state -> Prop)
    (result : step_result) :
  (∀ s : state, R s s) -> result_related R result result.
Proof. intros Hrefl. destruct result; simpl; auto. Qed.

Section Bisimulation.

Variable R : state -> state -> Prop.

Hypothesis R_step :
  ∀ s s' : state, R s s' -> result_related R (step s) (step s').

Lemma bisimulation_run (n : nat) :
  ∀ s s' : state, R s s' -> result_related R (run n s) (run n s').
Proof.
  induction n as [ | n IH]; intros s s' Hrel; simpl; [ exact Hrel | ].
  specialize (R_step s s' Hrel).
  destruct (step s), (step s'); simpl in R_step; try contradiction;
    [ apply IH; exact R_step | exact R_step | exact I ].
Qed.

(* States related by a relation that steps preserve, in step with each
   other, are equivalent. *)
Lemma bisimulation_equivalent (s s' : state) : R s s' -> equivalent s s'.
Proof.
  intros Hrel b.
  pose proof (fun n => bisimulation_run n s s' Hrel) as Hrun.
  destruct b as [k vs | | ]; simpl; split.
  - intros [n [rd Hn]]. exists n. specialize (Hrun n). rewrite Hn in Hrun.
    destruct (run n s'); simpl in Hrun; try contradiction.
    destruct Hrun as [-> [-> [-> ->]]]. eexists. reflexivity.
  - intros [n [rd Hn]]. exists n. specialize (Hrun n). rewrite Hn in Hrun.
    destruct (run n s); simpl in Hrun; try contradiction.
    destruct Hrun as [-> [-> [-> ->]]]. eexists. reflexivity.
  - intros [n Hn]. exists n. specialize (Hrun n). rewrite Hn in Hrun.
    destruct (run n s'); simpl in Hrun; try contradiction. reflexivity.
  - intros [n Hn]. exists n. specialize (Hrun n). rewrite Hn in Hrun.
    destruct (run n s); simpl in Hrun; try contradiction. reflexivity.
  - intros Hdiv n. destruct (Hdiv n) as [t Ht]. specialize (Hrun n).
    rewrite Ht in Hrun.
    destruct (run n s') as [t' | | ]; simpl in Hrun; try contradiction.
    exists t'. reflexivity.
  - intros Hdiv n. destruct (Hdiv n) as [t Ht]. specialize (Hrun n).
    rewrite Ht in Hrun.
    destruct (run n s) as [t' | | ]; simpl in Hrun; try contradiction.
    exists t'. reflexivity.
Qed.

End Bisimulation.

(* ======================================================================== *)
(* Environments                                                             *)
(* ======================================================================== *)

(* An arm found is one of the arms. *)
Lemma find_arm_in (i : Z) (arms : list (Z * apply_cont)) (ac : apply_cont) :
  find_arm i arms = Some ac -> (i, ac) ∈ arms.
Proof.
  induction arms as [ | [discriminant ac'] arms IH]; simpl; [ discriminate | ].
  destruct (Z.eqb_spec i discriminant) as [-> | _].
  - intros [= ->]. left. reflexivity.
  - intros Hfind. right. apply IH, Hfind.
Qed.

(* Looking up a handler of a recursive [Let_cont] among the closures made of
   its handlers finds the closure of that handler. [all] is the whole list
   of handlers, kept apart from the ones still to be looked through. *)
Lemma lookup_rec_closures (k : continuation) (h : continuation_handler)
    (handlers all : list (continuation * continuation_handler)) (vars : env)
    (conts rest : cont_env) :
  lookup k handlers = Some h ->
  lookup k (map (fun '(k, h) => (k, Closure h vars conts all)) handlers ++ rest)
  = Some (Closure h vars conts all).
Proof.
  induction handlers as [ | [k' h'] handlers IH]; simpl; [ discriminate | ].
  destruct (Pos.eqb k k'); [ | exact IH ].
  intros [= ->]. reflexivity.
Qed.

(* Looking up any continuation among the closures of a recursive
   [Let_cont], with other bindings after them. *)
Lemma lookup_rec_closures_aux (k : continuation)
    (handlers all : list (continuation * continuation_handler)) (vars : env)
    (conts rest : cont_env) :
  lookup k (map (fun '(k, h) => (k, Closure h vars conts all)) handlers ++ rest)
  = match lookup k handlers with
    | Some h => Some (Closure h vars conts all)
    | None => lookup k rest
    end.
Proof.
  induction handlers as [ | [k' h'] handlers IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma lookup_rec_closures_app (k : continuation)
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts rest : cont_env) :
  lookup k (rec_closures handlers vars conts ++ rest)
  = match lookup k handlers with
    | Some h => Some (Closure h vars conts handlers)
    | None => lookup k rest
    end.
Proof. apply lookup_rec_closures_aux. Qed.

(* What a continuation stands for where it has just been bound, and where
   the closures of a recursive [Let_cont] have. *)
Lemma resolve_head (k : continuation) (c : cont_value) (conts : cont_env) :
  resolve k ((k, c) :: conts) = c.
Proof. unfold resolve. simpl. rewrite Pos.eqb_refl. reflexivity. Qed.

Lemma resolve_other (k k' : continuation) (c : cont_value)
    (conts : cont_env) :
  k <> k' -> resolve k ((k', c) :: conts) = resolve k conts.
Proof.
  intros Hne. unfold resolve. simpl.
  destruct (Pos.eqb_spec k k'); [ contradiction | reflexivity ].
Qed.

Lemma resolve_rec_closures (k : continuation)
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts rest : cont_env) :
  resolve k (rec_closures handlers vars conts ++ rest)
  = match lookup k handlers with
    | Some h => Closure h vars conts handlers
    | None => resolve k rest
    end.
Proof.
  unfold resolve. rewrite lookup_rec_closures_app.
  destruct (lookup k handlers); reflexivity.
Qed.

(* The equalities of kinds, arities and metadata that FlambdaModel.v
   decides. *)

Lemma kinds_eqb_eq (kinds kinds' : list flambda_kind) :
  kinds_eqb kinds kinds' = true <-> kinds = kinds'.
Proof.
  unfold kinds_eqb.
  destruct (list_eq_dec kind_eq_dec kinds kinds'); split; congruence.
Qed.

Lemma kinds_per_parameter_eqb_eq (kinds kinds' : list (list flambda_kind)) :
  kinds_per_parameter_eqb kinds kinds' = true <-> kinds = kinds'.
Proof.
  unfold kinds_per_parameter_eqb.
  destruct (list_eq_dec (list_eq_dec kind_eq_dec) kinds kinds'); split;
    congruence.
Qed.

Lemma arity_eqb_eq (a a' : arity) : arity_eqb a a' = true <-> a = a'.
Proof.
  unfold arity_eqb.
  destruct (list_eq_dec arity_component_eq_dec a a'); split; congruence.
Qed.

Lemma handler_metadata_eqb_eq (m m' : handler_metadata) :
  handler_metadata_eqb m m' = true <-> m = m'.
Proof.
  unfold handler_metadata_eqb.
  rewrite andb_true_iff, kinds_eqb_eq, Bool.eqb_true_iff.
  destruct m, m'; simpl. split; [ intros [-> ->] | intros [= -> ->] ]; auto.
Qed.

Lemma code_metadata_eqb_eq (m m' : code_metadata) :
  code_metadata_eqb m m' = true <-> m = m'.
Proof.
  unfold code_metadata_eqb.
  rewrite !andb_true_iff, !arity_eqb_eq, !Bool.eqb_true_iff.
  destruct m, m'; simpl.
  split; [ intros [[[-> ->] ->] ->] | intros [= -> -> -> ->] ]; auto.
Qed.

Lemma kinds_eqb_refl (kinds : list flambda_kind) :
  kinds_eqb kinds kinds = true.
Proof. apply kinds_eqb_eq. reflexivity. Qed.

(* One value against one kind. *)
Lemma kinds_eqb_singleton (k k' : flambda_kind) :
  kinds_eqb [k] [k'] = kind_eqb k k'.
Proof.
  destruct (kind_eqb k k') eqn:Hkind.
  - apply kind_eqb_eq in Hkind. subst k'. apply kinds_eqb_refl.
  - destruct (kinds_eqb [k] [k']) eqn:Hkinds; [ | reflexivity ].
    apply kinds_eqb_eq in Hkinds. injection Hkinds as ->.
    rewrite kind_eqb_refl in Hkind. discriminate.
Qed.

Lemma arity_eqb_refl (a : arity) : arity_eqb a a = true.
Proof. apply arity_eqb_eq. reflexivity. Qed.

Lemma handler_metadata_eqb_refl (m : handler_metadata) :
  handler_metadata_eqb m m = true.
Proof. apply handler_metadata_eqb_eq. reflexivity. Qed.

Lemma code_metadata_eqb_refl (m : code_metadata) :
  code_metadata_eqb m m = true.
Proof. apply code_metadata_eqb_eq. reflexivity. Qed.

(* The kinds of an arity whose parameters are not unboxed products. *)
Lemma unarize_arity_of_kinds (kinds : list flambda_kind) :
  unarize (arity_of_kinds kinds) = kinds.
Proof.
  unfold unarize, arity_of_kinds.
  induction kinds as [ | k kinds IH]; simpl; [ | rewrite IH ]; reflexivity.
Qed.

(* Arities that agree parameter by parameter have the same kinds. *)
Lemma unarize_per_parameter_eq (a a' : arity) :
  unarize_per_parameter a = unarize_per_parameter a' ->
  unarize a = unarize a'.
Proof.
  unfold unarize, unarize_per_parameter. rewrite !flat_map_concat_map.
  intros ->. reflexivity.
Qed.

(* What can be called, or projected from, has kind [Value]. *)
Lemma closure_of_kind (statics : static_env) (v : value) (code : code_id)
    (vs : list (value_slot * value)) :
  closure_of statics v = Some (code, vs) -> kind_of_value v = Value.
Proof. destruct v; try discriminate; reflexivity. Qed.

(* The fields of a tuple are values of the projections, as many as the
   size asked for. *)
Lemma tuple_fields_from_forall (P : value -> Prop) (statics : static_env)
    (size : nat) (tuple : value) :
  (∀ (field : Z) (v : value),
    eval_prim statics
      (Unary (Block_load (tuple_access size) Immutable field)) [tuple]
    = Some v ->
    P v) ->
  ∀ (count field : nat) (fields : list value),
    tuple_fields_from statics size tuple field count = Some fields ->
    Forall P fields ∧ length fields = count.
Proof.
  intros Hprim. induction count as [ | count IH]; intros field fields.
  - intros [= <-]. split; [ apply Forall_nil | reflexivity ].
  - cbn [tuple_fields_from].
    destruct
      (eval_prim statics
         (Unary (Block_load (tuple_access size) Immutable (Z.of_nat field)))
         [tuple])
      as [v | ] eqn:Hv; [ | discriminate ].
    destruct (tuple_fields_from statics size tuple (S field) count)
      as [vs | ] eqn:Hvs; [ | discriminate ].
    intros [= <-]. destruct (IH _ _ Hvs) as [Hall Hlength].
    split; [ | simpl; rewrite Hlength; reflexivity ].
    apply Forall_cons; [ exact (Hprim _ _ Hv) | exact Hall ].
Qed.

Lemma tuple_fields_forall (P : value -> Prop) (statics : static_env)
    (size : nat) (tuple : value) (fields : list value) :
  (∀ (field : Z) (v : value),
    eval_prim statics
      (Unary (Block_load (tuple_access size) Immutable field)) [tuple]
    = Some v ->
    P v) ->
  tuple_fields statics size tuple = Some fields ->
  Forall P fields.
Proof.
  intros Hprim Hfields.
  exact (proj1 (tuple_fields_from_forall P statics size tuple Hprim _ _ _
                  Hfields)).
Qed.

Lemma tuple_fields_length (statics : static_env) (size : nat) (tuple : value)
    (fields : list value) :
  tuple_fields statics size tuple = Some fields -> length fields = size.
Proof.
  intros Hfields.
  exact (proj2 (tuple_fields_from_forall (fun _ => True) statics size tuple
                  (fun _ _ _ => I) _ _ _ Hfields)).
Qed.

Lemma map_all_repeat {A B : Type} (g : A -> B) (b : B) (l : list A) :
  Forall (fun a => g a = b) l -> map g l = repeat b (length l).
Proof.
  induction 1 as [ | a l Ha _ IH]; simpl; [ reflexivity | ].
  rewrite Ha, IH. reflexivity.
Qed.

(* The fields of a tuple are of kind [Value]. *)
Lemma tuple_fields_kinds (statics : static_env) (size : nat) (tuple : value)
    (fields : list value) :
  tuple_fields statics size tuple = Some fields ->
  map kind_of_value fields = repeat Value size.
Proof.
  intros Hfields.
  rewrite <- (tuple_fields_length _ _ _ _ Hfields). apply map_all_repeat.
  apply (tuple_fields_forall _ statics size tuple fields); [ | exact Hfields ].
  intros field v Hv. exact (proj2 (eval_prim_kinds _ _ _ _ Hv)).
Qed.

(* What a projection of a tuple reads: the tuple is a block of tag 0
   with as many fields as the size, and the value is the field. *)
Lemma eval_tuple_field (statics : static_env) (size : nat) (tuple : value)
    (field : nat) (v : value) :
  eval_prim statics
    (Unary (Block_load (tuple_access size) Immutable (Z.of_nat field)))
    [tuple]
  = Some v ->
  ∃ fields : list value,
    block_of statics tuple = Some (0%Z, fields)
    ∧ length fields = size
    ∧ nth_error fields field = Some v.
Proof.
  cbn [eval_prim eval_unary_primitive]. unfold eval_block_load, tuple_access.
  destruct (block_of statics tuple) as [[tag fields] | ]; [ | discriminate ].
  cbn [known_is].
  destruct (0 =? tag)%Z eqn:Htag; [ | discriminate ].
  destruct (Z.of_nat size =? Z.of_nat (length fields))%Z eqn:Hsize;
    [ | discriminate ].
  destruct (0 <=? Z.of_nat field)%Z; [ | discriminate ].
  cbn [andb]. rewrite Nat2Z.id.
  destruct (nth_error fields field) as [loaded | ] eqn:Hnth;
    [ | discriminate ].
  destruct (kind_eqb (kind_of_value loaded) Value); [ | discriminate ].
  intros [= <-]. apply Z.eqb_eq in Htag, Hsize. subst tag.
  exists fields. split; [ reflexivity | ]. split; [ lia | exact Hnth ].
Qed.

(* The fields that [tuple_fields_from] reads, one by one. *)
Lemma tuple_fields_from_nth (statics : static_env) (size : nat)
    (tuple : value) :
  ∀ (count field : nat) (fields : list value),
    tuple_fields_from statics size tuple field count = Some fields ->
    length fields = count
    ∧ ∀ i : nat,
        i < count ->
        eval_prim statics
          (Unary
             (Block_load (tuple_access size) Immutable (Z.of_nat (field + i))))
          [tuple]
        = nth_error fields i.
Proof.
  induction count as [ | count IH]; intros field fields.
  - intros [= <-]. split; [ reflexivity | intros i Hi; lia ].
  - cbn [tuple_fields_from].
    destruct
      (eval_prim statics
         (Unary (Block_load (tuple_access size) Immutable (Z.of_nat field)))
         [tuple])
      as [v | ] eqn:Hv; [ | discriminate ].
    destruct (tuple_fields_from statics size tuple (S field) count)
      as [vs | ] eqn:Hvs; [ | discriminate ].
    intros [= <-]. destruct (IH _ _ Hvs) as [Hlength Hnth].
    split; [ simpl; rewrite Hlength; reflexivity | ].
    intros [ | i] Hi; simpl.
    + rewrite Nat.add_0_r. exact Hv.
    + replace (field + S i) with (S field + i) by lia. apply Hnth. lia.
Qed.

Lemma nth_error_same {A : Type} (l : list A) :
  ∀ l' : list A, (∀ i : nat, nth_error l i = nth_error l' i) -> l = l'.
Proof.
  induction l as [ | a l IH]; intros [ | a' l'] Hnth.
  - reflexivity.
  - discriminate (Hnth 0).
  - discriminate (Hnth 0).
  - pose proof (Hnth 0) as Hhead. injection Hhead as <-. f_equal.
    apply IH. intros i. exact (Hnth (S i)).
Qed.

(* The fields of a tuple are those of a block of tag 0: the tuple is
   that block, or a symbol bound to it. A tuple of no component is not
   looked at. *)
Lemma tuple_fields_block (statics : static_env) (size : nat) (tuple : value)
    (fields : list value) :
  tuple_fields statics size tuple = Some fields ->
  fields = [] ∨ block_of statics tuple = Some (0%Z, fields).
Proof.
  unfold tuple_fields. intros Hfields.
  destruct (tuple_fields_from_nth statics size tuple size 0 fields Hfields)
    as [Hlength Hnth].
  destruct fields as [ | v fields]; [ left; reflexivity | right ].
  assert (Hpositive : 0 < size) by (simpl in Hlength; lia).
  pose proof (Hnth 0 Hpositive) as Hfirst. simpl in Hfirst.
  destruct (eval_tuple_field statics size tuple 0 v Hfirst)
    as (block & Hblock & Hblock_length & _).
  rewrite Hblock. f_equal. f_equal. apply nth_error_same. intros i.
  destruct (Nat.lt_ge_cases i size) as [Hi | Hi].
  - pose proof (Hnth i Hi) as Hfield. simpl in Hfield.
    destruct (nth_error (v :: fields) i) as [w | ] eqn:Hw.
    + destruct (eval_tuple_field statics size tuple i w Hfield)
        as (block' & Hblock' & _ & Hnth').
      rewrite Hblock in Hblock'. injection Hblock' as <-. exact Hnth'.
    + apply nth_error_None in Hw. lia.
  - rewrite (proj2 (nth_error_None block i)) by lia.
    symmetry. apply nth_error_None. lia.
Qed.

(* A value read as a block is one, or a symbol bound to one. *)
Lemma block_of_cases (statics : static_env) (v : value) (tag : Z)
    (fields : list value) :
  block_of statics v = Some (tag, fields) ->
  v = Value_block tag fields
  ∨ ∃ s : symbol,
      v = Value_symbol s
      ∧ lookup s (static_values statics) = Some (Static_block tag fields).
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag' fields'
    | | contents | l | callee args args_arity];
    simpl; try discriminate.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hs;
      [ | discriminate ].
    destruct allocated; try discriminate.
    intros [= <- <-]. right. exists s. split; [ reflexivity | exact Hs ].
  - intros [= <- <-]. left. reflexivity.
Qed.

(* A property that the fields of a block have is one that the fields of
   a tuple have. *)
Lemma tuple_fields_of_blocks (P : value -> Prop) (statics : static_env)
    (size : nat) (tuple : value) (fields : list value) :
  (∀ (tag : Z) (block : list value),
    block_of statics tuple = Some (tag, block) -> Forall P block) ->
  tuple_fields statics size tuple = Some fields ->
  Forall P fields.
Proof.
  intros Hblock Hfields.
  destruct (tuple_fields_block _ _ _ _ Hfields) as [-> | Hb];
    [ apply Forall_nil | exact (Hblock _ _ Hb) ].
Qed.

(* Two static environments under which the primitives read a value alike
   give it the same fields. *)
Lemma tuple_fields_agree (statics statics' : static_env) (size : nat)
    (tuple : value) :
  (∀ p : prim, eval_prim statics' p [tuple] = eval_prim statics p [tuple]) ->
  tuple_fields statics' size tuple = tuple_fields statics size tuple.
Proof.
  intros Hprim. unfold tuple_fields. generalize 0 as field.
  generalize size at 2 4 as count.
  induction count as [ | count IH]; intros field; [ reflexivity | ].
  cbn [tuple_fields_from]. rewrite Hprim, IH. reflexivity.
Qed.

(* What [detuple] gives, where it gives something. *)
Lemma detuple_cases (statics : static_env) (size : nat) (vs : list value)
    (args_arity : arity) (vs' : list value) (args_arity' : arity) :
  detuple statics size vs args_arity = Some (vs', args_arity') ->
  ∃ (tuple : value) (rest : list value) (first : arity_component)
    (rest_arity : arity) (fields : list value),
    vs = tuple :: rest
    ∧ args_arity = first :: rest_arity
    ∧ map kind_of_value vs = unarize args_arity
    ∧ unarize_component first = [Value]
    ∧ tuple_fields statics size tuple = Some fields
    ∧ vs' = fields ++ rest
    ∧ args_arity' = arity_of_kinds (repeat Value size) ++ rest_arity.
Proof.
  unfold detuple.
  destruct vs as [ | tuple rest]; [ discriminate | ].
  destruct args_arity as [ | first rest_arity]; [ discriminate | ].
  match goal with
  | |- context [negb ?tests] => destruct tests eqn:Htests
  end; cbn [negb]; [ | discriminate ].
  apply andb_true_iff in Htests as [Hkinds Hfirst].
  apply kinds_eqb_eq in Hkinds, Hfirst.
  destruct (tuple_fields statics size tuple) as [fields | ] eqn:Hfields;
    [ | discriminate ].
  intros [= <- <-]. exists tuple, rest, first, rest_arity, fields. auto 8.
Qed.

(* A property of the arguments of a call that the fields of a tuple
   have when the tuple has it holds of the arguments as the code takes
   them. *)
Lemma arguments_for_code_forall (P : value -> Prop) (statics : static_env)
    (c : code) (vs : list value) (args_arity : arity) (vs' : list value)
    (args_arity' : arity) :
  (∀ (size : nat) (tuple : value) (fields : list value),
    P tuple -> tuple_fields statics size tuple = Some fields ->
    Forall P fields) ->
  Forall P vs ->
  arguments_for_code statics c vs args_arity = Some (vs', args_arity') ->
  Forall P vs'.
Proof.
  intros Hfields Hvs. unfold arguments_for_code.
  destruct (code_is_tupled c); [ | intros [= <- _]; exact Hvs ].
  intros Hdetuple.
  destruct (detuple_cases _ _ _ _ _ _ Hdetuple)
    as (tuple & rest & first & rest_arity & fields & -> & _ & _ & _ & Htuple
        & -> & _).
  apply Forall_app.
  split; [ | exact (Forall_inv_tail Hvs) ].
  exact (Hfields _ tuple fields (Forall_inv Hvs) Htuple).
Qed.

(* The arguments as the code takes them, under two static environments
   under which the primitives read the first of them alike. *)
Lemma arguments_for_code_agree (statics statics' : static_env) (c : code)
    (vs : list value) (args_arity : arity) :
  (∀ (tuple : value) (rest : list value) (p : prim),
    vs = tuple :: rest ->
    eval_prim statics' p [tuple] = eval_prim statics p [tuple]) ->
  arguments_for_code statics' c vs args_arity
  = arguments_for_code statics c vs args_arity.
Proof.
  intros Hprim. unfold arguments_for_code.
  destruct (code_is_tupled c); [ | reflexivity ].
  unfold detuple. destruct vs as [ | tuple rest]; [ reflexivity | ].
  destruct args_arity as [ | first rest_arity]; [ reflexivity | ].
  rewrite (tuple_fields_agree statics statics' _ tuple
             (fun p => Hprim tuple rest p eq_refl)).
  reflexivity.
Qed.

(* The fields of a tuple are the same under a static environment under
   which the primitives that have a value keep it. *)
Lemma tuple_fields_from_mono (statics statics' : static_env) (size : nat)
    (tuple : value) :
  (∀ (p : prim) (v : value),
    eval_prim statics p [tuple] = Some v ->
    eval_prim statics' p [tuple] = Some v) ->
  ∀ (count field : nat) (fields : list value),
    tuple_fields_from statics size tuple field count = Some fields ->
    tuple_fields_from statics' size tuple field count = Some fields.
Proof.
  intros Hprim. induction count as [ | count IH]; intros field fields;
    [ exact (fun H => H) | ].
  cbn [tuple_fields_from].
  destruct
    (eval_prim statics
       (Unary (Block_load (tuple_access size) Immutable (Z.of_nat field)))
       [tuple])
    as [v | ] eqn:Hv; [ | discriminate ].
  rewrite (Hprim _ _ Hv).
  destruct (tuple_fields_from statics size tuple (S field) count)
    as [vs | ] eqn:Hvs; [ | discriminate ].
  rewrite (IH _ _ Hvs). exact (fun H => H).
Qed.

(* So are the arguments as the code takes them, for code with the same
   mark and the same arity of parameters. *)
Lemma arguments_for_code_mono (statics statics' : static_env) (c c' : code)
    (vs : list value) (args_arity : arity) (result : list value * arity) :
  code_is_tupled c' = code_is_tupled c ->
  code_params_arity c' = code_params_arity c ->
  (∀ (tuple : value) (rest : list value) (p : prim) (v : value),
    vs = tuple :: rest ->
    eval_prim statics p [tuple] = Some v ->
    eval_prim statics' p [tuple] = Some v) ->
  arguments_for_code statics c vs args_arity = Some result ->
  arguments_for_code statics' c' vs args_arity = Some result.
Proof.
  intros Htupled Hparams Hprim. unfold arguments_for_code.
  rewrite Htupled, Hparams.
  destruct (code_is_tupled c); [ | exact (fun H => H) ].
  unfold detuple. destruct vs as [ | tuple rest]; [ discriminate | ].
  destruct args_arity as [ | first rest_arity]; [ discriminate | ].
  destruct (negb _); [ discriminate | ].
  unfold tuple_fields.
  destruct
    (tuple_fields_from statics (cardinal_unarized (code_params_arity c))
       tuple 0 (cardinal_unarized (code_params_arity c)))
    as [fields | ] eqn:Hfields; [ | discriminate ].
  rewrite (tuple_fields_from_mono statics statics' _ tuple
             (fun p v => Hprim tuple rest p v eq_refl) _ _ _ Hfields).
  exact (fun H => H).
Qed.

(* Two tuples related by a relation that the observations respect have
   related fields. *)
Lemma tuple_fields_from_related (R : value -> value -> Prop)
    (statics statics' : static_env) (size : nat) (tuple tuple' : value) :
  respects_observations R statics statics' ->
  R tuple tuple' ->
  ∀ (count field : nat) (fields : list value),
    tuple_fields_from statics size tuple field count = Some fields ->
    ∃ fields' : list value,
      tuple_fields_from statics' size tuple' field count = Some fields'
      ∧ Forall2 R fields fields'.
Proof.
  intros Hrespects Htuple.
  induction count as [ | count IH]; intros field fields.
  - intros [= <-]. exists []. split; [ reflexivity | apply Forall2_nil ].
  - cbn [tuple_fields_from].
    destruct
      (eval_prim statics
         (Unary (Block_load (tuple_access size) Immutable (Z.of_nat field)))
         [tuple])
      as [v | ] eqn:Hv; [ | discriminate ].
    destruct
      (eval_prim_related R statics statics' _ [tuple] [tuple'] v Hrespects
         (Forall2_cons _ _ Htuple (Forall2_nil R)) Hv)
      as (v' & -> & Hv').
    destruct (tuple_fields_from statics size tuple (S field) count)
      as [vs | ] eqn:Hvs; [ | discriminate ].
    destruct (IH _ _ Hvs) as (vs' & -> & Hvs').
    intros [= <-]. exists (v' :: vs').
    split; [ reflexivity | apply Forall2_cons; assumption ].
Qed.

(* So the arguments as the code takes them are related, for two lists
   of related arguments of the same kinds and code with the same mark
   and the same arity of parameters. *)
Lemma arguments_for_code_related (R : value -> value -> Prop)
    (statics statics' : static_env) (c c' : code) (vs vs' : list value)
    (args_arity : arity) (ws : list value) (args_arity' : arity) :
  respects_observations R statics statics' ->
  code_is_tupled c' = code_is_tupled c ->
  code_params_arity c' = code_params_arity c ->
  Forall2 R vs vs' ->
  map kind_of_value vs' = map kind_of_value vs ->
  arguments_for_code statics c vs args_arity = Some (ws, args_arity') ->
  ∃ ws' : list value,
    arguments_for_code statics' c' vs' args_arity = Some (ws', args_arity')
    ∧ Forall2 R ws ws'.
Proof.
  intros Hrespects Htupled Hparams Hvs Hkinds. unfold arguments_for_code.
  rewrite Htupled, Hparams.
  destruct (code_is_tupled c);
    [ | intros [= <- <-]; exists vs'; split; [ reflexivity | exact Hvs ] ].
  unfold detuple.
  destruct Hvs as [ | tuple tuple' rest rest' Htuple Hrest]; [ discriminate | ].
  destruct args_arity as [ | first rest_arity]; [ discriminate | ].
  rewrite Hkinds. destruct (negb _); [ discriminate | ].
  unfold tuple_fields.
  destruct
    (tuple_fields_from statics (cardinal_unarized (code_params_arity c))
       tuple 0 (cardinal_unarized (code_params_arity c)))
    as [fields | ] eqn:Hfields; [ | discriminate ].
  destruct
    (tuple_fields_from_related R statics statics' _ tuple tuple' Hrespects
       Htuple _ _ _ Hfields)
    as (fields' & -> & Hfields').
  intros [= <- <-]. exists (fields' ++ rest').
  split; [ reflexivity | apply Forall2_app; assumption ].
Qed.

(* What is left of an over-application, applied: an indirect call of the
   one value it is given, with the arguments that remain. *)
Lemma apply_cont_value_over_application (remaining : list value)
    (remaining_arity return_arity : arity) (return_to exn_to : cont_value)
    (vs : list value) (statics : static_env) (hp : heap) (rd : read_record) :
  apply_cont_value
    (Over_application remaining remaining_arity return_arity return_to exn_to)
    vs statics hp rd
  = match vs with
    | [v] =>
      call_function v Indirect_unknown_arity remaining remaining_arity
        return_arity return_to exn_to statics hp rd
    | _ => Undefined
    end.
Proof. destruct vs as [ | v [ | v' vs]]; reflexivity. Qed.

(* Induction on what a continuation stands for, through extra arguments
   and what is left of an over-application; the continuations that a
   closure keeps are not gone into. *)
Lemma cont_value_over_ind (P : cont_value -> Prop) :
  (∀ (h : continuation_handler) (vars : env)
     (conts : list (continuation * cont_value))
     (recursive : list (continuation * continuation_handler)),
    P (Closure h vars conts recursive)) ->
  (∀ k : continuation, P (Outer k)) ->
  (∀ c : cont_value, P c -> ∀ extra : list value, P (Extra_args c extra)) ->
  (∀ (remaining : list value) (remaining_arity return_arity : arity)
     (return_to exn_to : cont_value),
    P return_to -> P exn_to ->
    P (Over_application remaining remaining_arity return_arity return_to
         exn_to)) ->
  ∀ c : cont_value, P c.
Proof.
  intros Hclosure Houter Hextra Hover.
  fix IH 1. intros [h vars conts recursive | k | c extra
    | remaining remaining_arity return_arity return_to exn_to].
  - apply Hclosure.
  - apply Houter.
  - apply Hextra, IH.
  - apply Hover; apply IH.
Qed.

(* A full call that is not undefined: the body binds nothing static,
   the arguments are of the kinds of the call's arity, the call is made
   at the arities of the code, and the body runs with the parameters
   bound. *)
Lemma call_code_step (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) (s : state) :
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  = Step s ->
  ∃ callee_vars : env,
    binds_statics (code_body c) = false
    ∧ map kind_of_value vs = unarize args_arity
    ∧ unarize_per_parameter args_arity
      = unarize_per_parameter (code_params_arity c)
    ∧ return_arity = code_result_arity c
    ∧ bind_params (code_params c) vs [(code_my_closure c, closure)]
      = Some callee_vars
    ∧ s
      = State (code_body c) callee_vars
          [(code_return_continuation c, return_to);
           (code_exn_continuation c, exn_to)]
          statics hp rd.
Proof.
  unfold call_code.
  destruct (binds_statics (code_body c)); [ discriminate | ].
  destruct
    (kinds_eqb (map kind_of_value vs) (unarize args_arity)
     && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
          (unarize_per_parameter (code_params_arity c))
     && arity_eqb return_arity (code_result_arity c))
    eqn:Hagree; simpl; [ | discriminate ].
  apply andb_true_iff in Hagree as [Hagree Hreturn].
  apply andb_true_iff in Hagree as [Hkinds Hargs].
  apply kinds_eqb_eq in Hkinds. apply kinds_per_parameter_eqb_eq in Hargs.
  apply arity_eqb_eq in Hreturn.
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [callee_vars | ] eqn:Hbind; [ | discriminate ].
  intros [= <-]. exists callee_vars. auto 7.
Qed.

(* A full call is undefined, or runs the body of the code under
   variables that depend on the closure and the arguments only, with the
   code's two continuations bound to what the caller gave. *)
Lemma call_code_cases (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (statics : static_env) (hp : heap)
      (rd : read_record) :
  (∀ return_to exn_to : cont_value,
    call_code closure c vs args_arity return_arity return_to exn_to statics hp
      rd
    = Undefined)
  ∨ (∃ callee_vars : env,
      ∀ return_to exn_to : cont_value,
        call_code closure c vs args_arity return_arity return_to exn_to
          statics hp rd
        = Step
            (State (code_body c) callee_vars
               [(code_return_continuation c, return_to);
                (code_exn_continuation c, exn_to)]
               statics hp rd)).
Proof.
  unfold call_code.
  destruct (binds_statics (code_body c)); [ left; reflexivity | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (code_params_arity c))
        && arity_eqb return_arity (code_result_arity c)));
    [ left; reflexivity | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [callee_vars | ]; [ | left; reflexivity ].
  right. eauto.
Qed.

(* What the call of a closure of code is, given the arguments as the
   code takes them, where it is something: full, or an over-application,
   or a partial application, by the number of its arguments. *)
Lemma shape_of_code_call_cases (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (shape : call_shape) :
  shape_of_code_call closure c vs args_arity return_arity = Some shape ->
  length args_arity = length (code_params_arity c)
  ∧ shape = Full_call closure c vs args_arity
  ∨ length (code_params_arity c) < length args_arity
    ∧ map kind_of_value vs = unarize args_arity
    ∧ partial_or_over_application args_arity return_arity
        (code_params_arity c) (code_result_arity c)
      = true
    ∧ shape
      = Over_call closure c
          (firstn (cardinal_unarized (code_params_arity c)) vs)
          (skipn (cardinal_unarized (code_params_arity c)) vs)
          (firstn (length (code_params_arity c)) args_arity)
          (partially_apply args_arity (length (code_params_arity c)))
  ∨ length args_arity < length (code_params_arity c)
    ∧ map kind_of_value vs = unarize args_arity
    ∧ partial_or_over_application args_arity return_arity
        (code_params_arity c) (code_result_arity c)
      = true
    ∧ code_is_tupled c = false
    ∧ shape = Partial_call (Value_partial_application closure vs args_arity).
Proof.
  unfold shape_of_code_call. cbv zeta. intros Hshape.
  destruct (Nat.eqb (length args_arity) (length (code_params_arity c)))
    eqn:Hcount.
  - left. injection Hshape as <-.
    split; [ apply Nat.eqb_eq; exact Hcount | reflexivity ].
  - apply Nat.eqb_neq in Hcount.
    destruct
      (kinds_eqb (map kind_of_value vs) (unarize args_arity)
       && partial_or_over_application args_arity return_arity
            (code_params_arity c) (code_result_arity c))
      eqn:Htests; simpl in Hshape; [ | discriminate ].
    apply andb_true_iff in Htests as [Hkinds Htest].
    apply kinds_eqb_eq in Hkinds.
    destruct (Nat.ltb (length (code_params_arity c)) (length args_arity))
      eqn:Hover.
    + injection Hshape as <-. apply Nat.ltb_lt in Hover. right. left. auto.
    + destruct (code_is_tupled c) eqn:Htupled; [ discriminate | ].
      injection Hshape as <-. apply Nat.ltb_ge in Hover. right. right.
      split; [ lia | auto ].
Qed.

(* What the call of a closure is, where it is something: the closure is
   one of code that is bound, the code named if the call is direct, and
   the call is what [shape_of_code_call] says of the arguments as the
   code takes them. *)
Lemma shape_of_closure_call_cases (statics : static_env) (closure : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (shape : call_shape) :
  shape_of_closure_call statics closure kind vs args_arity return_arity
  = Some shape ->
  ∃ (called : code_id) (slots : list (value_slot * value)) (c : code)
    (vs' : list value) (args_arity' : arity),
    closure_of statics closure = Some (called, slots)
    ∧ lookup called (static_code statics) = Some c
    ∧ (∀ named : code_id, kind = Direct named -> called = named)
    ∧ arguments_for_code statics c vs args_arity = Some (vs', args_arity')
    ∧ shape_of_code_call closure c vs' args_arity' return_arity = Some shape.
Proof.
  unfold shape_of_closure_call.
  destruct (closure_of statics closure) as [[called slots] | ] eqn:Hclosure;
    [ | discriminate ].
  destruct
    (match kind with
     | Direct named => Pos.eqb called named
     | Indirect_unknown_arity => true
     end)
    eqn:Hexpected; [ | discriminate ].
  destruct (lookup called (static_code statics)) as [c | ] eqn:Hcode;
    [ | discriminate ].
  destruct (arguments_for_code statics c vs args_arity)
    as [[vs' args_arity'] | ] eqn:Hargs; [ | discriminate ].
  intros Hshape. exists called, slots, c, vs', args_arity'.
  split; [ reflexivity | ]. split; [ exact Hcode | ].
  split; [ intros named ->; apply Pos.eqb_eq; exact Hexpected | ].
  split; [ exact Hargs | exact Hshape ].
Qed.

(* What a call is, where it is something: the call of the callee, a
   closure; or, the call being indirect and giving one argument at
   least, that of the closure of a partial application with the
   arguments so far and then those of the call. *)
Lemma shape_of_call_cases (statics : static_env) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (shape : call_shape) :
  shape_of_call statics callee kind vs args_arity return_arity = Some shape ->
  shape_of_closure_call statics callee kind vs args_arity return_arity
  = Some shape
  ∨ ∃ (closure : value) (applied : list value) (applied_arity : arity),
      kind = Indirect_unknown_arity
      ∧ callee = Value_partial_application closure applied applied_arity
      ∧ args_arity <> []
      ∧ shape_of_closure_call statics closure Indirect_unknown_arity
          (applied ++ vs) (applied_arity ++ args_arity) return_arity
        = Some shape.
Proof.
  unfold shape_of_call.
  destruct kind as [named | ]; [ intros Hshape; left; exact Hshape | ].
  destruct callee
    as [k | code slots | slot decls slots | s | tag fields | | contents
       | l | closure applied applied_arity];
    simpl; try (intros Hshape; left; exact Hshape).
  destruct args_arity as [ | component args_arity]; [ discriminate | ].
  destruct (kinds_eqb _ _); [ | discriminate ].
  intros Hshape. right. exists closure, applied, applied_arity.
  split; [ reflexivity | ]. split; [ reflexivity | ].
  split; [ discriminate | exact Hshape ].
Qed.

(* The closure and the code of a call that calls something are a closure
   of that code, which is bound, and is the code named if the call is
   direct. *)
Definition shape_calls (statics : static_env) (kind : call_kind)
    (shape : option call_shape) : Prop :=
  match shape with
  | Some (Full_call closure c _ _) | Some (Over_call closure c _ _ _ _) =>
    ∃ (called : code_id) (slots : list (value_slot * value)),
      closure_of statics closure = Some (called, slots)
      ∧ lookup called (static_code statics) = Some c
      ∧ ∀ named : code_id, kind = Direct named -> called = named
  | Some (Partial_call _) | None => True
  end.

Lemma shape_of_closure_call_calls (statics : static_env) (closure : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity) :
  shape_calls statics kind
    (shape_of_closure_call statics closure kind vs args_arity return_arity).
Proof.
  destruct
    (shape_of_closure_call statics closure kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape; [ | exact I ].
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
    as (called & slots & c & vs' & args_arity' & Hcode & Hlookup & Hnamed
        & _ & Hcall).
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
    as [[_ ->] | [(_ & _ & _ & ->) | (_ & _ & _ & _ & ->)]];
    simpl; [ | | exact I ]; exists called, slots; auto.
Qed.

Lemma shape_of_call_calls (statics : static_env) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity) :
  shape_calls statics kind
    (shape_of_call statics callee kind vs args_arity return_arity).
Proof.
  destruct (shape_of_call statics callee kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape; [ | exact I ].
  destruct (shape_of_call_cases _ _ _ _ _ _ _ Hshape)
    as [H | (closure & applied & applied_arity & -> & _ & _ & H)];
    rewrite <- H; apply shape_of_closure_call_calls.
Qed.

(* What the tests of arities give of a call that is not full: the code
   of an over-application returns one [Value], and a partial application
   is of kind [Value] and is made by a call that expects one [Value]. *)
Lemma is_one_param_of_kind_value_eq (a : arity) :
  is_one_param_of_kind_value a = true -> a = [Singleton Value].
Proof.
  destruct a as [ | [[ | n | | ] | components] [ | c a]]; simpl; intros H;
    try discriminate H.
  reflexivity.
Qed.

Definition shape_arities (return_arity : arity) (shape : option call_shape)
    : Prop :=
  match shape with
  | Some (Over_call _ c _ _ _ _) => code_result_arity c = [Singleton Value]
  | Some (Partial_call partial) =>
    return_arity = [Singleton Value] ∧ kind_of_value partial = Value
  | Some (Full_call _ _ _ _) | None => True
  end.

Lemma shape_of_closure_call_arities (statics : static_env) (closure : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity) :
  shape_arities return_arity
    (shape_of_closure_call statics closure kind vs args_arity return_arity).
Proof.
  destruct
    (shape_of_closure_call statics closure kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape; [ | exact I ].
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
    as (called & slots & c & vs' & args_arity' & _ & _ & _ & _ & Hcall).
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
    as [[_ ->] | [(Hlt & _ & Htest & ->) | (Hlt & _ & Htest & _ & ->)]];
    simpl; [ exact I | | ];
    unfold partial_or_over_application in Htest; cbv zeta in Htest;
    apply andb_true_iff in Htest as [_ Htest].
  - apply Nat.ltb_lt in Hlt. rewrite Hlt in Htest.
    exact (is_one_param_of_kind_value_eq _ Htest).
  - assert (Hge :
      Nat.ltb (length (code_params_arity c)) (length args_arity') = false)
      by (apply Nat.ltb_ge; lia).
    rewrite Hge in Htest. apply andb_true_iff in Htest as [_ Htest].
    split; [ exact (is_one_param_of_kind_value_eq _ Htest) | reflexivity ].
Qed.

Lemma shape_of_call_arities (statics : static_env) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity) :
  shape_arities return_arity
    (shape_of_call statics callee kind vs args_arity return_arity).
Proof.
  destruct (shape_of_call statics callee kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape; [ | exact I ].
  destruct (shape_of_call_cases _ _ _ _ _ _ _ Hshape)
    as [H | (closure & applied & applied_arity & -> & _ & _ & H)];
    rewrite <- H; apply shape_of_closure_call_arities.
Qed.

(* A call, and the application of a continuation, keep the static
   environment. *)
Lemma call_code_statics (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) (s' : state) :
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  = Step s' ->
  state_statics s' = statics.
Proof.
  intros Hstep.
  apply call_code_step in Hstep as (callee_vars & _ & _ & _ & _ & _ & ->).
  reflexivity.
Qed.

Lemma run_call_statics (return_partial : value -> step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
      (s' : state) :
  (∀ partial : value,
    return_partial partial = Step s' -> state_statics s' = statics) ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  = Step s' ->
  state_statics s' = statics.
Proof.
  intros Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl;
    [ apply call_code_statics | apply call_code_statics | apply Hpartial
    | discriminate ].
Qed.

Lemma apply_cont_value_statics (c : cont_value) :
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record)
    (s' : state),
    apply_cont_value c vs statics hp rd = Step s' -> state_statics s' = statics.
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs statics hp rd s' Hstep.
  - simpl in Hstep.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs def_vars); [ | discriminate ].
    injection Hstep as <-. reflexivity.
  - discriminate.
  - exact (IH _ _ _ _ _ Hstep).
  - rewrite apply_cont_value_over_application in Hstep.
    destruct vs as [ | v [ | v' vs]]; [ discriminate | | discriminate ].
    unfold call_function in Hstep.
    apply run_call_statics in Hstep; [ exact Hstep | ].
    intros partial. apply IHreturn.
Qed.

Lemma call_function_statics (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record)
      (s' : state) :
  call_function callee kind vs args_arity return_arity return_to exn_to
    statics hp rd
  = Step s' ->
  state_statics s' = statics.
Proof.
  unfold call_function. apply run_call_statics.
  intros partial. apply apply_cont_value_statics.
Qed.

(* Where a call, or the application of a continuation, lands: on the
   body of code that is bound, with the two continuations of the code
   standing for continuations of which [Kc] holds; or on the handler of
   a closure of which [Kc] holds. [Kc] of a continuation with extra
   arguments gives [Kc] of the continuation inside, and [Kc] of what is
   left of an over-application is [Kc] of its two continuations. *)
Section Lands.

Variable statics : static_env.
Variable hp : heap.
Variable rd : read_record.
Variable Kc : cont_value -> Prop.
Variable Q : state -> Prop.

Hypothesis Kc_extra :
  ∀ (c : cont_value) (extra : list value), Kc (Extra_args c extra) -> Kc c.

Hypothesis Kc_over :
  ∀ (remaining : list value) (remaining_arity return_arity : arity)
    (return_to exn_to : cont_value),
    Kc
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
    <-> Kc return_to ∧ Kc exn_to.

Hypothesis Q_code :
  ∀ (called : code_id) (c : code) (callee_vars : env)
    (return_to exn_to : cont_value),
    lookup called (static_code statics) = Some c ->
    Kc return_to ->
    Kc exn_to ->
    Q (State (code_body c) callee_vars
         [(code_return_continuation c, return_to);
          (code_exn_continuation c, exn_to)]
         statics hp rd).

Hypothesis Q_handler :
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (handler_vars : env),
    Kc
      (Closure (Handler params metadata handler) def_vars def_conts
         recursive) ->
    Q (State handler handler_vars
         (rec_closures recursive def_vars def_conts ++ def_conts) statics hp
           rd).

Lemma call_code_lands (closure : value) (called : code_id) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (s' : state) :
  lookup called (static_code statics) = Some c ->
  Kc return_to ->
  Kc exn_to ->
  call_code closure c vs args_arity return_arity return_to exn_to statics hp rd
  = Step s' ->
  Q s'.
Proof.
  intros Hcode Hreturn Hexn Hstep.
  apply call_code_step in Hstep as (callee_vars & _ & _ & _ & _ & _ & ->).
  exact (Q_code called c callee_vars return_to exn_to Hcode Hreturn Hexn).
Qed.

Lemma run_call_lands (return_partial : value -> step_result)
    (kind : call_kind) (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (s' : state) :
  shape_calls statics kind shape ->
  Kc return_to ->
  Kc exn_to ->
  (∀ partial : value, return_partial partial = Step s' -> Q s') ->
  run_call return_partial shape return_arity return_to exn_to statics hp rd
  = Step s' ->
  Q s'.
Proof.
  intros Hcalls Hreturn Hexn Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hcalls |- *.
  - destruct Hcalls as (called & slots & _ & Hcode & _).
    exact (call_code_lands closure called c vs _ _ _ _ s' Hcode Hreturn
             Hexn).
  - destruct Hcalls as (called & slots & _ & Hcode & _).
    apply (call_code_lands closure called c vs _ _ _ _ s' Hcode);
      [ apply Kc_over; split; assumption | exact Hexn ].
  - apply Hpartial.
  - discriminate.
Qed.

Lemma apply_cont_value_lands (c : cont_value) :
  ∀ (vs : list value) (s' : state),
    Kc c -> apply_cont_value c vs statics hp rd = Step s' -> Q s'.
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs s' Hc Hstep.
  - simpl in Hstep.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs def_vars) as [handler_vars | ];
      [ | discriminate ].
    injection Hstep as <-.
    exact (Q_handler params metadata handler def_vars def_conts recursive
             handler_vars Hc).
  - discriminate.
  - exact (IH _ s' (Kc_extra c extra Hc) Hstep).
  - rewrite apply_cont_value_over_application in Hstep.
    destruct vs as [ | v [ | v' vs]]; [ discriminate | | discriminate ].
    apply Kc_over in Hc as [Hreturn Hexn].
    unfold call_function in Hstep.
    apply (run_call_lands _ Indirect_unknown_arity _ _ _ _ s'
             (shape_of_call_calls _ _ _ _ _ _) Hreturn Hexn)
      in Hstep; [ exact Hstep | ].
    intros partial. exact (IHreturn [partial] s' Hreturn).
Qed.

Lemma call_function_lands (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (s' : state) :
  Kc return_to ->
  Kc exn_to ->
  call_function callee kind vs args_arity return_arity return_to exn_to
    statics hp rd
  = Step s' ->
  Q s'.
Proof.
  intros Hreturn Hexn. unfold call_function.
  apply (run_call_lands _ kind _ _ _ _ s'
           (shape_of_call_calls _ _ _ _ _ _) Hreturn Hexn).
  intros partial. exact (apply_cont_value_lands return_to _ s' Hreturn).
Qed.

End Lands.

(* Calls under two static environments. The closure that a call calls:
   the callee, or the one that a partial application keeps. *)
Definition called_closure (callee : value) : value :=
  match partial_application_of callee with
  | Some (closure, _, _) => closure
  | None => callee
  end.

(* The arguments that the closure of a call is called with: those of the
   call, after those that a partial application keeps. *)
Definition called_arguments (callee : value) (vs : list value) : list value :=
  match partial_application_of callee with
  | Some (_, applied, _) => applied ++ vs
  | None => vs
  end.

(* Two calls have the same shape, with the same values, of code related
   by [R]. *)
Inductive shapes_related (R : code -> code -> Prop)
    : option call_shape -> option call_shape -> Prop :=
  | Shapes_none : shapes_related R None None
  | Shapes_full (closure : value) (c c' : code) (vs : list value)
      (args_arity : arity) :
    R c c' ->
    shapes_related R (Some (Full_call closure c vs args_arity))
      (Some (Full_call closure c' vs args_arity))
  | Shapes_over (closure : value) (c c' : code) (vs remaining : list value)
      (args_arity remaining_arity : arity) :
    R c c' ->
    shapes_related R
      (Some (Over_call closure c vs remaining args_arity remaining_arity))
      (Some (Over_call closure c' vs remaining args_arity remaining_arity))
  | Shapes_partial (partial : value) :
    shapes_related R (Some (Partial_call partial))
      (Some (Partial_call partial)).

(* Two static environments bind a code id alike: to code related by [R]
   with the same arities and the same mark, or not at all. *)
Definition codes_agree (R : code -> code -> Prop)
    (statics statics' : static_env) (called : code_id) : Prop :=
  match
    lookup called (static_code statics), lookup called (static_code statics')
  with
  | Some c, Some c' =>
    R c c'
    ∧ code_params_arity c' = code_params_arity c
    ∧ code_result_arity c' = code_result_arity c
    ∧ code_is_tupled c' = code_is_tupled c
  | None, None => True
  | _, _ => False
  end.

Lemma shape_of_closure_call_related (R : code -> code -> Prop)
    (statics statics' : static_env) (closure : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity) :
  closure_of statics' closure = closure_of statics closure ->
  (∀ (called : code_id) (slots : list (value_slot * value)),
    closure_of statics closure = Some (called, slots) ->
    codes_agree R statics statics' called) ->
  (∀ (tuple : value) (rest : list value) (p : prim),
    vs = tuple :: rest ->
    eval_prim statics' p [tuple] = eval_prim statics p [tuple]) ->
  shapes_related R
    (shape_of_closure_call statics closure kind vs args_arity return_arity)
    (shape_of_closure_call statics' closure kind vs args_arity return_arity).
Proof.
  intros Hclosure Hcodes Hprim. unfold shape_of_closure_call.
  rewrite Hclosure.
  destruct (closure_of statics closure) as [[called slots] | ];
    [ | apply Shapes_none ].
  specialize (Hcodes called slots eq_refl). unfold codes_agree in Hcodes.
  destruct
    (match kind with
     | Direct named => Pos.eqb called named
     | Indirect_unknown_arity => true
     end);
    [ | apply Shapes_none ].
  destruct (lookup called (static_code statics)) as [c | ],
    (lookup called (static_code statics')) as [c' | ];
    try contradiction; [ | apply Shapes_none ].
  destruct Hcodes as (Hc & Hparams & Hresult & Htupled).
  assert (Hargs :
    arguments_for_code statics' c' vs args_arity
    = arguments_for_code statics c vs args_arity).
  { rewrite (arguments_for_code_agree statics statics' c' vs args_arity Hprim).
    unfold arguments_for_code. rewrite Htupled, Hparams. reflexivity. }
  rewrite Hargs.
  destruct (arguments_for_code statics c vs args_arity)
    as [[vs' args_arity'] | ]; [ | apply Shapes_none ].
  cbv beta iota. unfold shape_of_code_call.
  rewrite Hparams, Hresult, Htupled.
  destruct (Nat.eqb (length args_arity') (length (code_params_arity c)));
    [ apply Shapes_full; exact Hc | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs') (unarize args_arity')
        && partial_or_over_application args_arity' return_arity
             (code_params_arity c) (code_result_arity c)));
    [ apply Shapes_none | ].
  destruct (Nat.ltb (length (code_params_arity c)) (length args_arity'));
    [ apply Shapes_over; exact Hc | ].
  destruct (code_is_tupled c); [ apply Shapes_none | apply Shapes_partial ].
Qed.

Lemma shape_of_call_related (R : code -> code -> Prop)
    (statics statics' : static_env) (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity) :
  closure_of statics' (called_closure callee)
  = closure_of statics (called_closure callee) ->
  (∀ (called : code_id) (slots : list (value_slot * value)),
    closure_of statics (called_closure callee) = Some (called, slots) ->
    codes_agree R statics statics' called) ->
  (∀ (tuple : value) (rest : list value) (p : prim),
    called_arguments callee vs = tuple :: rest ->
    eval_prim statics' p [tuple] = eval_prim statics p [tuple]) ->
  shapes_related R
    (shape_of_call statics callee kind vs args_arity return_arity)
    (shape_of_call statics' callee kind vs args_arity return_arity).
Proof.
  intros Hclosure Hcodes Hprim. unfold shape_of_call.
  destruct kind as [named | ];
    destruct callee
      as [k | code slots | slot decls slots | s | tag fields | | contents
         | l | closure applied applied_arity];
    simpl in Hclosure, Hcodes, Hprim |- *;
    try (apply shape_of_closure_call_related; assumption).
  - apply Shapes_none.
  - destruct args_arity as [ | component args_arity];
      [ apply Shapes_none | ].
    destruct (kinds_eqb _ _); [ | apply Shapes_none ].
    apply shape_of_closure_call_related; assumption.
Qed.

(* Full calls of two codes that differ in their bodies only: undefined
   both, or both run their bodies under the same variables. *)
Lemma call_code_alike (closure : value) (c c' : code) (vs : list value)
    (args_arity return_arity : arity) (statics statics' : static_env)
      (hp : heap) (rd : read_record) :
  code_params c' = code_params c ->
  code_my_closure c' = code_my_closure c ->
  code_return_continuation c' = code_return_continuation c ->
  code_exn_continuation c' = code_exn_continuation c ->
  code_params_arity c' = code_params_arity c ->
  code_result_arity c' = code_result_arity c ->
  binds_statics (code_body c') = binds_statics (code_body c) ->
  (∀ return_to exn_to return_to' exn_to' : cont_value,
    call_code closure c vs args_arity return_arity return_to exn_to statics hp
      rd
    = Undefined
    ∧ call_code closure c' vs args_arity return_arity return_to' exn_to'
        statics' hp rd
      = Undefined)
  ∨ (∃ callee_vars : env,
      ∀ return_to exn_to return_to' exn_to' : cont_value,
        call_code closure c vs args_arity return_arity return_to exn_to
          statics hp rd
        = Step
            (State (code_body c) callee_vars
               [(code_return_continuation c, return_to);
                (code_exn_continuation c, exn_to)]
               statics hp rd)
        ∧ call_code closure c' vs args_arity return_arity return_to' exn_to'
            statics' hp rd
          = Step
              (State (code_body c') callee_vars
                 [(code_return_continuation c, return_to');
                  (code_exn_continuation c, exn_to')]
                 statics' hp rd)).
Proof.
  intros Hparams Hmy_closure Hret Hexn Hparams_arity Hresult_arity Hbinds.
  unfold call_code.
  rewrite Hparams, Hmy_closure, Hret, Hexn, Hparams_arity, Hresult_arity,
    Hbinds.
  destruct (binds_statics (code_body c)); [ left; split; reflexivity | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (code_params_arity c))
        && arity_eqb return_arity (code_result_arity c)));
    [ left; split; reflexivity | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [callee_vars | ]; [ | left; split; reflexivity ].
  right. exists callee_vars. split; reflexivity.
Qed.

(* Every parameter bound, and every variable bound before, is bound. *)
Lemma bind_params_bound (params : list variable) :
  ∀ (vs : list value) (vars handler_vars : env) (x : variable),
    bind_params params vs vars = Some handler_vars ->
    x ∈ params ∨ (∃ v : value, lookup x vars = Some v) ->
    ∃ v : value, lookup x handler_vars = Some v.
Proof.
  induction params as [ | param params IH];
    intros [ | v vs] vars handler_vars x; simpl; try discriminate.
  - intros [= <-] [[] | Hbound]. exact Hbound.
  - intros Hbind Hx. apply (IH vs _ _ x Hbind). simpl.
    destruct (Pos.eqb_spec x param) as [-> | Hne]; [ right; eauto | ].
    destruct Hx as [[Heq | Hin] | Hbound];
      [ congruence | left; exact Hin | right; exact Hbound ].
Qed.

(* Binding parameters under more variables binds them the same way. *)
Lemma bind_params_app (params : list variable) :
  ∀ (vs : list value) (vars bound rest : env),
    bind_params params vs vars = Some bound ->
    bind_params params vs (vars ++ rest) = Some (bound ++ rest).
Proof.
  induction params as [ | param params IH]; intros [ | v vs] vars bound rest;
    simpl; try discriminate.
  - intros [= <-]. reflexivity.
  - intros Hbind. exact (IH vs ((param, v) :: vars) bound rest Hbind).
Qed.

(* ======================================================================== *)
(* Simulations                                                              *)
(* ======================================================================== *)

(* What two steps, or two runs of one length, come to, where the second
   may do anything if the first is undefined: otherwise states related by
   [R], or the same exit. *)
Definition result_simulated (R : state -> state -> Prop)
    (result result' : step_result) : Prop :=
  match result, result' with
  | Undefined, _ => True
  | Step s, Step s' => R s s'
  | Exit k vs hp rd, Exit k' vs' hp' rd' => k = k' ∧ vs = vs' ∧ hp = hp' ∧ rd
    = rd'
  | _, _ => False
  end.

Section Simulation.

Variable R : state -> state -> Prop.

Hypothesis R_step :
  ∀ s s' : state, R s s' -> result_simulated R (step s) (step s').

Lemma simulation_run (n : nat) :
  ∀ s s' : state, R s s' -> result_simulated R (run n s) (run n s').
Proof.
  induction n as [ | n IH]; intros s s' Hrel; simpl; [ exact Hrel | ].
  specialize (R_step s s' Hrel).
  destruct (step s), (step s'); simpl in R_step; try contradiction;
    try exact I; [ apply IH; exact R_step | exact R_step ].
Qed.

(* A state refines one related to it by a relation that steps preserve, in
   step with each other, as long as the first is defined. *)
Lemma simulation_refines (s s' : state) : R s s' -> refines s s'.
Proof.
  intros Hrel.
  pose proof (fun n => simulation_run n s s' Hrel) as Hrun.
  split.
  - intros k vs hp [n [rd Hn]]. exists n. specialize (Hrun n).
    rewrite Hn in Hrun.
    destruct (run n s'); simpl in Hrun; try contradiction.
    destruct Hrun as [-> [-> [-> ->]]]. eexists. reflexivity.
  - intros b Hterm Hb. destruct (behaves_ended s' b Hterm Hb) as [n Hended].
    specialize (Hrun n).
    destruct (run n s) as [t | k vs hp rd | ] eqn:Hs.
    + destruct (run n s'); simpl in Hrun; contradiction.
    + destruct (run n s') as [t' | k' vs' hp' rd' | ]; simpl in Hrun;
        try contradiction.
      destruct Hrun as [-> [-> [-> ->]]]. apply ended_exit in Hended.
      subst b. left. exists n. eexists. exact Hs.
    + right. exists n. exact Hs.
Qed.

End Simulation.

(* ======================================================================== *)
(* Two states at the same construct                                         *)
(* ======================================================================== *)

(* For each construct that binds, how to relate two states that are both
   at it and take the same step: both are undefined, or both step to their
   bodies, which are then to be related. A proof that goes under a binder
   uses these, and so does not depend on what the step of the binder is. *)

(* Two [Let]s whose defining expressions do the same on the heap: the
   same value and the same heap after, or no value. *)
Lemma let_equivalent_step_exec (x x' : variable) (kind : flambda_kind)
    (d d' : named) (body body' : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
      (rd : read_record) :
  exec_named statics' hp rd vars' d' = exec_named statics hp rd vars d ->
  (∀ (v : value) (hp' : heap) (rd' : read_record),
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    kind_of_value v = kind ->
    equivalent (State body ((x, v) :: vars) conts statics hp' rd')
      (State body' ((x', v) :: vars') conts' statics' hp' rd')) ->
  equivalent (State (Let x kind d body) vars conts statics hp rd)
    (State (Let x' kind d' body') vars' conts' statics' hp rd).
Proof.
  intros Hd' Hbodies.
  destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hd.
  - destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind.
    + eapply equivalent_step;
        [ rewrite step_let_exec, Hd, Hkind; reflexivity
        | rewrite step_let_exec, Hd', Hkind; reflexivity
        | exact (Hbodies v hp' rd_ eq_refl (proj1 (kind_eqb_eq _ _) Hkind)) ].
    + apply step_eq_equivalent. rewrite !step_let_exec, Hd', Hd, Hkind.
      reflexivity.
  - apply step_eq_equivalent. rewrite !step_let_exec, Hd', Hd. reflexivity.
Qed.

Lemma let_refines_step_exec (x x' : variable) (kind : flambda_kind)
    (d d' : named) (body body' : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
      (rd : read_record) :
  exec_named statics' hp rd vars' d' = exec_named statics hp rd vars d ->
  (∀ (v : value) (hp' : heap) (rd' : read_record),
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    kind_of_value v = kind ->
    refines (State body ((x, v) :: vars) conts statics hp' rd')
      (State body' ((x', v) :: vars') conts' statics' hp' rd')) ->
  refines (State (Let x kind d body) vars conts statics hp rd)
    (State (Let x' kind d' body') vars' conts' statics' hp rd).
Proof.
  intros Hd' Hbodies.
  destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hd.
  - destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind.
    + eapply refines_step;
        [ rewrite step_let_exec, Hd, Hkind; reflexivity
        | rewrite step_let_exec, Hd', Hkind; reflexivity
        | exact (Hbodies v hp' rd_ eq_refl (proj1 (kind_eqb_eq _ _) Hkind)) ].
    + apply step_undefined_refines. rewrite step_let_exec, Hd, Hkind.
      reflexivity.
  - apply step_undefined_refines. rewrite step_let_exec, Hd. reflexivity.
Qed.

(* Two [Let]s where the second defining expression has the value of the
   first, and returns the same heap, whenever the first has a value: a
   refinement only, as the second may have a value where the first has
   none. *)
Lemma let_refines_step_forward (x x' : variable) (kind : flambda_kind)
    (d d' : named) (body body' : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
      (rd : read_record) :
  (∀ (v : value) (hp' : heap) (rd' : read_record),
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    exec_named statics' hp rd vars' d' = Some (v, hp', rd')) ->
  (∀ (v : value) (hp' : heap) (rd' : read_record),
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    kind_of_value v = kind ->
    refines (State body ((x, v) :: vars) conts statics hp' rd')
      (State body' ((x', v) :: vars') conts' statics' hp' rd')) ->
  refines (State (Let x kind d body) vars conts statics hp rd)
    (State (Let x' kind d' body') vars' conts' statics' hp rd).
Proof.
  intros Hd' Hbodies.
  destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hd.
  - destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind.
    + eapply refines_step;
        [ rewrite step_let_exec, Hd, Hkind; reflexivity
        | rewrite step_let_exec, (Hd' v hp' rd_ eq_refl), Hkind; reflexivity
        | exact (Hbodies v hp' rd_ eq_refl (proj1 (kind_eqb_eq _ _) Hkind)) ].
    + apply step_undefined_refines. rewrite step_let_exec, Hd, Hkind.
      reflexivity.
  - apply step_undefined_refines. rewrite step_let_exec, Hd. reflexivity.
Qed.

(* Two [Let]s of what is not a primitive: by [eval_named], and the heap
   is as it came. *)
Lemma let_equivalent_step_not_prim (x x' : variable) (kind : flambda_kind)
    (d d' : named) (body body' : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
      (rd : read_record) :
  (∀ (p : prim) (args : list simple), d <> Prim p args) ->
  (∀ (p : prim) (args : list simple), d' <> Prim p args) ->
  eval_named statics' vars' d' = eval_named statics vars d ->
  (∀ v : value,
    eval_named statics vars d = Some v ->
    kind_of_value v = kind ->
    equivalent (State body ((x, v) :: vars) conts statics hp rd)
      (State body' ((x', v) :: vars') conts' statics' hp rd)) ->
  equivalent (State (Let x kind d body) vars conts statics hp rd)
    (State (Let x' kind d' body') vars' conts' statics' hp rd).
Proof.
  intros Hd Hd' Heq Hbodies. apply let_equivalent_step_exec.
  - rewrite (exec_named_not_prim _ _ rd_ _ _ Hd),
    (exec_named_not_prim _ _ rd_ _ _ Hd'),
      Heq.
    reflexivity.
  - intros v hp' rd' Hexec Hkind.
    rewrite (exec_named_not_prim _ _ rd_ _ _ Hd) in Hexec.
    destruct (eval_named statics vars d) as [w | ] eqn:Hw; [ | discriminate ].
    injection Hexec as <- <- ->. exact (Hbodies w eq_refl Hkind).
Qed.

Lemma let_refines_step_not_prim (x x' : variable) (kind : flambda_kind)
    (d d' : named) (body body' : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
      (rd : read_record) :
  (∀ (p : prim) (args : list simple), d <> Prim p args) ->
  (∀ (p : prim) (args : list simple), d' <> Prim p args) ->
  eval_named statics' vars' d' = eval_named statics vars d ->
  (∀ v : value,
    eval_named statics vars d = Some v ->
    kind_of_value v = kind ->
    refines (State body ((x, v) :: vars) conts statics hp rd)
      (State body' ((x', v) :: vars') conts' statics' hp rd)) ->
  refines (State (Let x kind d body) vars conts statics hp rd)
    (State (Let x' kind d' body') vars' conts' statics' hp rd).
Proof.
  intros Hd Hd' Heq Hbodies. apply let_refines_step_exec.
  - rewrite (exec_named_not_prim _ _ rd_ _ _ Hd),
    (exec_named_not_prim _ _ rd_ _ _ Hd'),
      Heq.
    reflexivity.
  - intros v hp' rd' Hexec Hkind.
    rewrite (exec_named_not_prim _ _ rd_ _ _ Hd) in Hexec.
    destruct (eval_named statics vars d) as [w | ] eqn:Hw; [ | discriminate ].
    injection Hexec as <- <- ->. exact (Hbodies w eq_refl Hkind).
Qed.

(* Two [Let_static]s: both groups are bound, and the bodies are related
   under the static environments that result, or neither is. For a
   refinement the second may be anything when the first is not bound. *)
Lemma let_static_equivalent_step (bindings bindings' : list static_binding)
    (body body' : expr) (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  match
    eval_static_bindings vars bindings statics,
    eval_static_bindings vars' bindings' statics'
  with
  | Some extended, Some extended' =>
    equivalent (State body vars conts extended hp rd)
      (State body' vars' conts' extended' hp rd)
  | None, None => True
  | _, _ => False
  end ->
  equivalent (State (Let_static bindings body) vars conts statics hp rd)
    (State (Let_static bindings' body') vars' conts' statics' hp rd).
Proof.
  intros Hbodies.
  destruct (eval_static_bindings vars bindings statics) as [extended | ]
    eqn:Hbindings,
    (eval_static_bindings vars' bindings' statics') as [extended' | ]
      eqn:Hbindings';
    try contradiction.
  - eapply equivalent_step;
      [ rewrite step_let_static, Hbindings; reflexivity
      | rewrite step_let_static, Hbindings'; reflexivity | exact Hbodies ].
  - apply step_eq_equivalent.
    rewrite !step_let_static, Hbindings, Hbindings'. reflexivity.
Qed.

Lemma let_static_refines_step (bindings bindings' : list static_binding)
    (body body' : expr) (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  match
    eval_static_bindings vars bindings statics,
    eval_static_bindings vars' bindings' statics'
  with
  | Some extended, Some extended' =>
    refines (State body vars conts extended hp rd)
      (State body' vars' conts' extended' hp rd)
  | Some _, None => False
  | None, _ => True
  end ->
  refines (State (Let_static bindings body) vars conts statics hp rd)
    (State (Let_static bindings' body') vars' conts' statics' hp rd).
Proof.
  intros Hbodies.
  destruct (eval_static_bindings vars bindings statics) as [extended | ]
    eqn:Hbindings;
    [ | apply step_undefined_refines; rewrite step_let_static, Hbindings;
        reflexivity ].
  destruct (eval_static_bindings vars' bindings' statics')
    as [extended' | ] eqn:Hbindings'; [ | contradiction ].
  eapply refines_step;
    [ rewrite step_let_static, Hbindings; reflexivity
    | rewrite step_let_static, Hbindings'; reflexivity | exact Hbodies ].
Qed.

(* Two [Let_cont_non_recursive]s always step to their bodies. *)
Lemma let_cont_non_recursive_equivalent_step (k k' : continuation)
    (h h' : continuation_handler) (body body' : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
      (rd : read_record) :
  equivalent
    (State body vars ((k, Closure h vars conts []) :: conts) statics hp rd)
    (State body' vars' ((k', Closure h' vars' conts' []) :: conts')
       statics' hp rd) ->
  equivalent (State (Let_cont_non_recursive k h body) vars conts statics hp rd)
    (State (Let_cont_non_recursive k' h' body') vars' conts' statics' hp rd).
Proof.
  apply equivalent_step; apply step_let_cont_non_recursive.
Qed.

Lemma let_cont_non_recursive_refines_step (k k' : continuation)
    (h h' : continuation_handler) (body body' : expr) (vars vars' : env)
    (conts conts' : cont_env) (statics statics' : static_env) (hp : heap)
      (rd : read_record) :
  refines
    (State body vars ((k, Closure h vars conts []) :: conts) statics hp rd)
    (State body' vars' ((k', Closure h' vars' conts' []) :: conts')
       statics' hp rd) ->
  refines (State (Let_cont_non_recursive k h body) vars conts statics hp rd)
    (State (Let_cont_non_recursive k' h' body') vars' conts' statics' hp rd).
Proof.
  apply refines_step; apply step_let_cont_non_recursive.
Qed.

(* Two [Let_cont_recursive]s that agree on whether their handlers bind
   something static: both are undefined, or both step to their bodies. *)
Lemma let_cont_recursive_equivalent_step
    (handlers handlers' : list (continuation * continuation_handler))
    (body body' : expr) (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  handlers_bind_statics handlers = handlers_bind_statics handlers' ->
  (handlers_bind_statics handlers = false ->
   equivalent
     (State body vars (rec_closures handlers vars conts ++ conts) statics hp rd)
     (State body' vars' (rec_closures handlers' vars' conts' ++ conts')
        statics' hp rd)) ->
  equivalent (State (Let_cont_recursive handlers body) vars conts statics hp rd)
    (State (Let_cont_recursive handlers' body') vars' conts' statics' hp rd).
Proof.
  intros Hbinds Hbodies.
  destruct (handlers_bind_statics handlers) eqn:Hsource.
  - apply step_eq_equivalent.
    rewrite !step_let_cont_recursive, <- Hbinds, Hsource. reflexivity.
  - eapply equivalent_step;
      [ rewrite step_let_cont_recursive, Hsource; reflexivity
      | rewrite step_let_cont_recursive, <- Hbinds; reflexivity
      | exact (Hbodies eq_refl) ].
Qed.

Lemma let_cont_recursive_refines_step
    (handlers handlers' : list (continuation * continuation_handler))
    (body body' : expr) (vars vars' : env) (conts conts' : cont_env)
    (statics statics' : static_env) (hp : heap) (rd : read_record) :
  handlers_bind_statics handlers = handlers_bind_statics handlers' ->
  (handlers_bind_statics handlers = false ->
   refines
     (State body vars (rec_closures handlers vars conts ++ conts) statics hp rd)
     (State body' vars' (rec_closures handlers' vars' conts' ++ conts')
        statics' hp rd)) ->
  refines (State (Let_cont_recursive handlers body) vars conts statics hp rd)
    (State (Let_cont_recursive handlers' body') vars' conts' statics' hp rd).
Proof.
  intros Hbinds Hbodies.
  destruct (handlers_bind_statics handlers) eqn:Hsource.
  - apply step_undefined_refines.
    rewrite step_let_cont_recursive, Hsource. reflexivity.
  - eapply refines_step;
      [ rewrite step_let_cont_recursive, Hsource; reflexivity
      | rewrite step_let_cont_recursive, <- Hbinds; reflexivity
      | exact (Hbodies eq_refl) ].
Qed.

(* ======================================================================== *)
(* Where a state holds values                                               *)
(* ======================================================================== *)

Lemma env_values_all_lookup (P : value -> Prop) (vars : env) (x : variable)
    (v : value) :
  env_values_all P vars -> lookup x vars = Some v -> P v.
Proof.
  intros Hvars Hlookup. apply lookup_in in Hlookup.
  unfold env_values_all in Hvars. rewrite Forall_forall in Hvars.
  exact (Hvars (x, v) Hlookup).
Qed.

Lemma forallb_Forall {A : Type} (f : A -> bool) (l : list A) :
  forallb f l = true <-> Forall (fun a => f a = true) l.
Proof. rewrite forallb_forall, Forall_forall. reflexivity. Qed.

Section Values_all.

Variable P : value -> Prop.

(* For a [P] that holds of every constant and of every symbol, the
   values of the simples read under an environment satisfy it. *)
Lemma eval_simple_values_all (vars : env) (s : simple) (v : value) :
  (∀ (not_a_variable : simple) (v' : value),
    eval_simple [] not_a_variable = Some v' -> P v') ->
  env_values_all P vars -> eval_simple vars s = Some v -> P v.
Proof.
  intros Hsimple Hvars. destruct s as [x | sym | c].
  - apply env_values_all_lookup, Hvars.
  - apply (Hsimple (Symbol sym)).
  - apply (Hsimple (Const c)).
Qed.

Lemma eval_simples_values_all (vars : env) (args : list simple) :
  (∀ (not_a_variable : simple) (v' : value),
    eval_simple [] not_a_variable = Some v' -> P v') ->
  env_values_all P vars ->
  ∀ vs : list value, eval_simples vars args = Some vs -> Forall P vs.
Proof.
  intros Hsimple Hvars. induction args as [ | s args IH]; simpl; intros vs.
  - intros [= <-]. apply Forall_nil.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; simpl; [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ]; simpl; [ | discriminate ].
    intros [= <-]. apply Forall_cons; [ | apply IH; reflexivity ].
    exact (eval_simple_values_all vars s v Hsimple Hvars Hs).
Qed.

Lemma bind_params_values_all (params : list variable) :
  ∀ (vs : list value) (vars vars' : env),
    Forall P vs ->
    env_values_all P vars ->
    bind_params params vs vars = Some vars' ->
    env_values_all P vars'.
Proof.
  induction params as [ | x params IH]; intros [ | v vs] vars vars' Hvs Hvars;
    simpl; try discriminate.
  - intros [= <-]. exact Hvars.
  - apply IH; [ exact (Forall_inv_tail Hvs) | ].
    apply Forall_cons; [ exact (Forall_inv Hvs) | exact Hvars ].
Qed.

Lemma resolve_values_all (k : continuation) (conts : cont_env) :
  conts_values_all P conts -> cont_values_all P (resolve k conts).
Proof.
  unfold resolve, conts_values_all.
  induction conts as [ | [k' c] conts IH]; simpl; [ intros _; exact I | ].
  intros [Hc Hrest]. destruct (Pos.eqb k k'); [ exact Hc | exact (IH Hrest) ].
Qed.

Lemma conts_values_all_app (conts conts' : cont_env) :
  conts_values_all P conts -> conts_values_all P conts' ->
  conts_values_all P (conts ++ conts').
Proof.
  unfold conts_values_all.
  induction conts as [ | binding conts IH]; simpl; [ auto | ].
  intros [Hc Hrest] Hconts'. split; [ exact Hc | exact (IH Hrest Hconts') ].
Qed.

Lemma rec_closures_values_all (vars : env) (conts : cont_env)
    (handlers : list (continuation * continuation_handler)) :
  env_values_all P vars ->
  conts_values_all P conts ->
  conts_values_all P (rec_closures handlers vars conts).
Proof.
  intros Hvars Hconts. unfold rec_closures, conts_values_all.
  generalize handlers at 1 as all. intros all.
  induction handlers as [ | [k h] handlers IH]; simpl; [ exact I | ].
  split; [ split; [ exact Hvars | exact Hconts ] | exact IH ].
Qed.

(* From here on [P] of a partial application is [P] of the value it
   calls and of its arguments. *)
Hypothesis P_partial :
  ∀ (callee : value) (args : list value) (args_arity : arity),
    P (Value_partial_application callee args args_arity)
    <-> P callee ∧ Forall P args.

(* [P] of the fields of a tuple of which it holds, under a static
   environment. *)
Definition P_fields (statics : static_env) : Prop :=
  ∀ (size : nat) (tuple : value) (fields : list value),
    P tuple -> tuple_fields statics size tuple = Some fields ->
    Forall P fields.

(* [P] of the values of a call of a given shape. *)
Definition shape_values_all (shape : option call_shape) : Prop :=
  match shape with
  | Some (Full_call closure _ vs _) => P closure ∧ Forall P vs
  | Some (Over_call closure _ vs remaining _ _) =>
    P closure ∧ Forall P vs ∧ Forall P remaining
  | Some (Partial_call partial) => P partial
  | None => True
  end.

Lemma shape_of_closure_call_values_all (statics : static_env)
    (closure : value) (kind : call_kind) (vs : list value)
    (args_arity return_arity : arity) :
  P_fields statics ->
  P closure -> Forall P vs ->
  shape_values_all
    (shape_of_closure_call statics closure kind vs args_arity return_arity).
Proof.
  intros Hfields Hclosure Hvs.
  destruct
    (shape_of_closure_call statics closure kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape; [ | exact I ].
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
    as (called & slots & c & vs' & args_arity' & _ & _ & _ & Hargs & Hcall).
  pose proof (arguments_for_code_forall P _ _ _ _ _ _ Hfields Hvs Hargs)
    as Hvs'.
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
    as [[_ ->] | [(_ & _ & _ & ->) | (_ & _ & _ & _ & ->)]];
    simpl.
  - split; assumption.
  - rewrite <- (firstn_skipn (cardinal_unarized (code_params_arity c)) vs')
      in Hvs'.
    apply Forall_app in Hvs'. split; [ exact Hclosure | exact Hvs' ].
  - apply P_partial. split; assumption.
Qed.

Lemma shape_of_call_values_all (statics : static_env) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity) :
  P_fields statics ->
  P callee -> Forall P vs ->
  shape_values_all
    (shape_of_call statics callee kind vs args_arity return_arity).
Proof.
  intros Hfields Hcallee Hvs.
  destruct (shape_of_call statics callee kind vs args_arity return_arity)
    as [shape | ] eqn:Hshape; [ | exact I ].
  destruct (shape_of_call_cases _ _ _ _ _ _ _ Hshape)
    as [H | (closure & applied & applied_arity & _ & -> & _ & H)];
    rewrite <- H.
  - apply shape_of_closure_call_values_all; assumption.
  - apply P_partial in Hcallee as [Hclosure Happlied].
    apply shape_of_closure_call_values_all;
      [ exact Hfields | exact Hclosure | ].
    apply Forall_app. split; assumption.
Qed.

(* A full call of a closure that satisfies [P], with arguments that do,
   returning to continuations whose values do. *)
Lemma call_code_values_all (Q : state -> Prop) (closure : value) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  P closure ->
  Forall P vs ->
  cont_values_all P return_to ->
  cont_values_all P exn_to ->
  (∀ (e : expr) (vars : env) (conts : cont_env),
    env_values_all P vars -> conts_values_all P conts ->
    Q (State e vars conts statics hp rd)) ->
  match
    call_code closure c vs args_arity return_arity return_to exn_to statics hp
      rd
  with
  | Step s' => Q s'
  | Exit _ vs' _ _ => Forall P vs'
  | Undefined => True
  end.
Proof.
  intros Hclosure Hvs Hreturn Hexn Hstate. unfold call_code.
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct
    (negb
       (kinds_eqb (map kind_of_value vs) (unarize args_arity)
        && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
             (unarize_per_parameter (code_params_arity c))
        && arity_eqb return_arity (code_result_arity c)));
    [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [callee_vars | ] eqn:Hbind; [ | exact I ].
  apply Hstate.
  - apply (bind_params_values_all (code_params c) vs
             [(code_my_closure c, closure)] callee_vars Hvs);
      [ | exact Hbind ].
    apply Forall_cons; [ exact Hclosure | apply Forall_nil ].
  - split; [ exact Hreturn | ]. split; [ exact Hexn | exact I ].
Qed.

(* A call of a shape whose values satisfy [P], given what is done with a
   partial application that does. *)
Lemma run_call_values_all (Q : state -> Prop)
    (return_partial : value -> step_result) (shape : option call_shape)
    (return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  shape_values_all shape ->
  cont_values_all P return_to ->
  cont_values_all P exn_to ->
  (∀ (e : expr) (vars : env) (conts : cont_env),
    env_values_all P vars -> conts_values_all P conts ->
    Q (State e vars conts statics hp rd)) ->
  (∀ partial : value,
    P partial ->
    match return_partial partial with
    | Step s' => Q s'
    | Exit _ vs' _ _ => Forall P vs'
    | Undefined => True
    end) ->
  match run_call return_partial shape return_arity return_to exn_to statics hp
    rd
    with
  | Step s' => Q s'
  | Exit _ vs' _ _ => Forall P vs'
  | Undefined => True
  end.
Proof.
  intros Hshape Hreturn Hexn Hstate Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl in Hshape |- *.
  - destruct Hshape as [Hclosure Hvs].
    apply call_code_values_all; assumption.
  - destruct Hshape as (Hclosure & Hvs & Hremaining).
    apply call_code_values_all; try assumption.
    simpl. auto.
  - exact (Hpartial partial Hshape).
  - exact I.
Qed.

(* Applying a continuation whose values satisfy [P] to values that do. *)
Lemma apply_cont_value_values_all (Q : state -> Prop) (c : cont_value)
    (vs : list value) (statics : static_env) (hp : heap) (rd : read_record) :
  P_fields statics ->
  cont_values_all P c ->
  Forall P vs ->
  (∀ (e : expr) (vars : env) (conts : cont_env),
    env_values_all P vars -> conts_values_all P conts ->
    Q (State e vars conts statics hp rd)) ->
  match apply_cont_value c vs statics hp rd with
  | Step s' => Q s'
  | Exit _ vs' _ _ => Forall P vs'
  | Undefined => True
  end.
Proof.
  intros Hfields Hc Hvs Hstate. revert vs Hc Hvs.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs Hc Hvs.
  - simpl. destruct Hc as [Hdef_vars Hdef_conts].
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | exact I ].
    destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
      [ | exact I ].
    apply Hstate.
    + exact (bind_params_values_all params vs def_vars handler_vars Hvs
               Hdef_vars Hbind).
    + apply conts_values_all_app; [ | exact Hdef_conts ].
      apply rec_closures_values_all; assumption.
  - exact Hvs.
  - simpl. destruct Hc as [Hextra Hc]. apply (IH (vs ++ extra) Hc).
    apply Forall_app. split; assumption.
  - rewrite apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; [ exact I | | exact I ].
    destruct Hc as (Hremaining & Hreturn & Hexn).
    unfold call_function. apply run_call_values_all; try assumption.
    + apply shape_of_call_values_all;
        [ exact Hfields | exact (Forall_inv Hvs) | exact Hremaining ].
    + intros partial Hpartial.
      apply IHreturn; [ exact Hreturn | ].
      apply Forall_cons; [ exact Hpartial | apply Forall_nil ].
Qed.

(* The exception continuation that a call gives its callee. *)
Lemma with_extra_args_values_all (c : cont_value) (extra : list value) :
  cont_values_all P c ->
  Forall P extra ->
  cont_values_all P (with_extra_args c extra).
Proof.
  intros Hc Hextra. destruct extra as [ | v extra]; simpl;
    [ exact Hc | split; [ exact Hextra | exact Hc ] ].
Qed.

(* A call of a value that satisfies [P], with arguments that do,
   returning to continuations whose values do. *)
Lemma call_function_values_all (Q : state -> Prop) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  P_fields statics ->
  P callee ->
  Forall P vs ->
  cont_values_all P return_to ->
  cont_values_all P exn_to ->
  (∀ (e : expr) (vars : env) (conts : cont_env),
    env_values_all P vars -> conts_values_all P conts ->
    Q (State e vars conts statics hp rd)) ->
  match
    call_function callee kind vs args_arity return_arity return_to exn_to
      statics hp rd
  with
  | Step s' => Q s'
  | Exit _ vs' _ _ => Forall P vs'
  | Undefined => True
  end.
Proof.
  intros Hfields Hcallee Hvs Hreturn Hexn Hstate. unfold call_function.
  apply run_call_values_all; try assumption.
  - apply shape_of_call_values_all; assumption.
  - intros partial Hpartial.
    apply apply_cont_value_values_all;
      [ exact Hfields | exact Hreturn | | exact Hstate ].
    apply Forall_cons; [ exact Hpartial | apply Forall_nil ].
Qed.

End Values_all.

(* [P] of every value a continuation holds, for a weaker [P]. *)
Fixpoint cont_values_all_impl (P P' : value -> Prop)
    (Himpl : ∀ v : value, P v -> P' v) (c : cont_value) :
  cont_values_all P c -> cont_values_all P' c.
Proof.
  destruct c
    as [h vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    simpl; [ | auto | | ].
  - intros [Hvars Hconts]. split.
    + exact (Forall_impl _ (fun binding => Himpl (snd binding)) Hvars).
    + induction conts as [ | [k c] conts IH]; simpl; [ exact I | ].
      destruct Hconts as [Hc Hrest].
      split;
        [ exact (cont_values_all_impl P P' Himpl c Hc) | exact (IH Hrest) ].
  - intros [Hextra Hc].
    split;
      [ exact (Forall_impl _ Himpl Hextra)
      | exact (cont_values_all_impl P P' Himpl c Hc) ].
  - intros (Hremaining & Hreturn & Hexn).
    exact (conj (Forall_impl _ Himpl Hremaining)
             (conj (cont_values_all_impl P P' Himpl return_to Hreturn)
                (cont_values_all_impl P P' Himpl exn_to Hexn))).
Qed.

Lemma conts_values_all_impl (P P' : value -> Prop) (conts : cont_env) :
  (∀ v : value, P v -> P' v) ->
  conts_values_all P conts -> conts_values_all P' conts.
Proof.
  intros Himpl. unfold conts_values_all.
  induction conts as [ | [k c] conts IH]; simpl; [ auto | ].
  intros [Hc Hrest].
  split; [ exact (cont_values_all_impl P P' Himpl c Hc) | exact (IH Hrest) ].
Qed.

Theorem state_values_all_empty_heap : state_values_all_empty_heap_statement.
Proof.
  intros P Q s Hempty. unfold state_values_all, state_values_all_but_cells.
  rewrite Hempty. split.
  - intros (Hvars & Hconts & Hstatics & _).
    exact (conj Hvars (conj Hconts Hstatics)).
  - intros (Hvars & Hconts & Hstatics).
    exact (conj Hvars
             (conj Hconts (conj Hstatics (heap_values_all_empty _)))).
Qed.

Theorem step_values_all : step_values_all_statement.
Proof.
  intros P Q [e vars conts statics hp rd] Hpartial
    (Hvars & Hconts & Hstatics & Hheap) Hfields Hlet Hlet_static Happly
    Happly_cont.
  simpl in *.
  assert (Hstate :
    ∀ (e' : expr) (vars' : env) (conts' : cont_env),
      env_values_all (P statics) vars' ->
      conts_values_all (P statics) conts' ->
      state_values_all P Q (State e' vars' conts' statics hp rd))
    by (intros e' vars' conts' Hvars' Hconts'; repeat split; assumption).
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    [ rewrite step_let_exec | autorewrite with step .. ].
  - (* Let *)
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hd;
      [ | exact I ].
    destruct (kind_eqb (kind_of_value v) kind); [ | exact I ].
    destruct (Hlet x kind d body v hp' rd_ eq_refl Hd) as [Hv Hheap'].
    split; [ | split; [ | split ] ]; simpl;
      [ apply Forall_cons; [ exact Hv | exact Hvars ] | exact Hconts
      | exact Hstatics | exact Hheap' ].
  - (* Let_static *)
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Hbindings; [ | exact I ].
    destruct (Hlet_static bindings body statics' eq_refl Hbindings)
      as [Hkept Hstatics'].
    split; [ | split; [ | split; [ exact Hstatics' | ] ] ]; simpl.
    + exact (Forall_impl _ (fun binding => Hkept (snd binding)) Hvars).
    + exact (conts_values_all_impl _ _ conts Hkept Hconts).
    + intros l c v Hin Hv. exact (Hkept v (Hheap l c v Hin Hv)).
  - (* Let_cont_non_recursive *)
    apply Hstate; [ exact Hvars | ].
    split; [ split; [ exact Hvars | exact Hconts ] | exact Hconts ].
  - (* Let_cont_recursive *)
    destruct (handlers_bind_statics handlers); [ exact I | ].
    apply Hstate; [ exact Hvars | ].
    apply conts_values_all_app; [ | exact Hconts ].
    apply rec_closures_values_all; assumption.
  - (* Apply *)
    destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
      [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
      eqn:Hextra; [ | exact I ].
    destruct (Happly ap callee vs extra eq_refl Hcallee Hargs Hextra)
      as (Hcallee_ok & Hvs & Hextra_ok).
    apply (call_function_values_all (P statics) (Hpartial statics)
             (state_values_all P Q));
      [ exact Hfields | exact Hcallee_ok | exact Hvs
      | apply resolve_values_all, Hconts
      | apply with_extra_args_values_all;
        [ apply resolve_values_all, Hconts | exact Hextra_ok ]
      | exact Hstate ].
  - (* Apply_cont *)
    destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hargs;
      [ | exact I ].
    unfold apply_continuation.
    apply (apply_cont_value_values_all (P statics) (Hpartial statics)
             (state_values_all P Q));
      [ exact Hfields | apply resolve_values_all, Hconts | | exact Hstate ].
    exact (Happly_cont ac vs eq_refl Hargs).
  - (* Switch *)
    destruct (eval_simple vars scrutinee) as [v | ]; [ | exact I ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | exact I ].
    destruct (find_arm i arms) as [ac | ]; [ | exact I ].
    apply Hstate; assumption.
  - (* Invalid *)
    exact I.
Qed.

(* ======================================================================== *)
(* The read record does nothing yet                                         *)
(* ======================================================================== *)

Theorem exec_named_ignores_reads : exec_named_ignores_reads_statement.
Proof.
  intros statics hp rd1 rd2 vars d.
  destruct d as [s | p args | | | | | ];
    try (rewrite !exec_named_not_prim by (intros ? ?; discriminate);
         destruct (eval_named _ _ _); reflexivity).
  rewrite !exec_named_prim.
  destruct (eval_simples vars args) as [vs | ]; [ | reflexivity ].
  simpl. destruct (exec_prim statics hp p vs) as [[v hp'] | ]; reflexivity.
Qed.

Theorem exec_named_keeps_reads : exec_named_keeps_reads_statement.
Proof.
  intros statics hp hp' rd rd' vars d v Hexec.
  destruct d as [s | p args | | | | | ];
    try (rewrite exec_named_not_prim in Hexec by (intros ? ?; discriminate);
         destruct (eval_named _ _ _); [ | discriminate ];
         injection Hexec as _ _ <-; reflexivity).
  apply exec_named_prim_some in Hexec as (vs & _ & _ & ->). reflexivity.
Qed.

Lemma call_code_with_reads (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd1 rd2 : read_record) :
  call_code closure c vs args_arity return_arity return_to exn_to statics hp
    rd2
  = result_with_reads rd2
      (call_code closure c vs args_arity return_arity return_to exn_to
         statics hp rd1).
Proof.
  unfold call_code.
  destruct (binds_statics (code_body c)); [ reflexivity | ].
  destruct (negb _); [ reflexivity | ].
  destruct (bind_params _ _ _); reflexivity.
Qed.

Lemma run_call_with_reads (return1 return2 : value -> step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
    (rd1 rd2 : read_record) :
  (∀ partial : value,
    return2 partial = result_with_reads rd2 (return1 partial)) ->
  run_call return2 shape return_arity return_to exn_to statics hp rd2
  = result_with_reads rd2
      (run_call return1 shape return_arity return_to exn_to statics hp rd1).
Proof.
  intros Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl;
    [ apply call_code_with_reads | apply call_code_with_reads
    | apply Hpartial | reflexivity ].
Qed.

Lemma apply_cont_value_with_reads (c : cont_value) :
  ∀ (vs : list value) (statics : static_env) (hp : heap)
    (rd1 rd2 : read_record),
    apply_cont_value c vs statics hp rd2
    = result_with_reads rd2 (apply_cont_value c vs statics hp rd1).
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs statics hp rd1 rd2.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | reflexivity ].
    destruct (bind_params params vs def_vars); reflexivity.
  - reflexivity.
  - simpl. apply IH.
  - rewrite !apply_cont_value_over_application.
    destruct vs as [ | v [ | v' vs]]; try reflexivity.
    unfold call_function. apply run_call_with_reads.
    intros partial. apply IHreturn.
Qed.

Theorem step_ignores_reads : step_ignores_reads_statement.
Proof.
  intros rd [e vars conts statics hp rd0]. unfold with_reads. simpl.
  destruct e
    as [x kind d body | bindings body | k handler body | handlers body | ap
       | ac | scrutinee arms | ];
    autorewrite with step.
  - rewrite (exec_named_ignores_reads statics hp rd0 rd vars d).
    destruct (exec_named statics hp rd0 vars d) as [[[v hp'] rd'] | ];
      [ | reflexivity ].
    destruct (kind_eqb (kind_of_value v) kind); reflexivity.
  - destruct (eval_static_bindings vars bindings statics); reflexivity.
  - reflexivity.
  - destruct (handlers_bind_statics handlers); reflexivity.
  - destruct (eval_simple vars (ap_callee ap)); [ | reflexivity ].
    destruct (eval_simples vars (ap_args ap)); [ | reflexivity ].
    destruct (eval_simples vars (exn_extra_simples ap)); [ | reflexivity ].
    unfold call_function. apply run_call_with_reads.
    intros partial. apply apply_cont_value_with_reads.
  - destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | reflexivity ].
    unfold apply_continuation. apply apply_cont_value_with_reads.
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | reflexivity ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | reflexivity ].
    destruct (find_arm i arms) as [ac | ]; reflexivity.
  - reflexivity.
Qed.

Theorem step_keeps_reads : step_keeps_reads_statement.
Proof.
  intros [e vars conts statics hp rd].
  exact (eq_sym (step_ignores_reads rd (State e vars conts statics hp rd))).
Qed.

(* ======================================================================== *)
(* The heap and the record evolve                                           *)
(* ======================================================================== *)

Theorem memory_evolves_refl : memory_evolves_refl_statement.
Proof.
  intros hp rd. split; [ apply FlambdaHeapProofs.heap_evolves_refl | ].
  reflexivity.
Qed.

Theorem memory_evolves_trans : memory_evolves_trans_statement.
Proof.
  intros hp1 hp2 hp3 rd1 rd2 rd3 [Hheap1 ->] [Hheap2 ->].
  split; [ | reflexivity ].
  exact (FlambdaHeapProofs.heap_evolves_trans _ _ _ Hheap1 Hheap2).
Qed.

Theorem exec_named_memory_evolves : exec_named_memory_evolves_statement.
Proof.
  intros statics hp hp' rd rd' vars d v Hexec. split.
  - exact (exec_named_evolves statics hp hp' rd rd' vars d v Hexec).
  - exact (exec_named_keeps_reads statics hp hp' rd rd' vars d v Hexec).
Qed.

Theorem step_memory_evolves : step_memory_evolves_statement.
Proof.
  intros s s' Hstep. split.
  - exact (step_evolves s s' Hstep).
  - pose proof (step_keeps_reads s) as Hkeeps. rewrite Hstep in Hkeeps.
    simpl in Hkeeps. injection Hkeeps as Hkeeps.
    rewrite <- Hkeeps at 1. reflexivity.
Qed.

(* ======================================================================== *)
(* A primitive whose value stays                                            *)
(* ======================================================================== *)

Theorem stable_prim_again : stable_prim_again_statement.
Proof.
  intros statics hp hp' rd rd' vars p args v Hstable Hexec.
  apply exec_named_prim_some in Hexec as (vs & Hvs & Hprim & ->).
  destruct
    (FlambdaHeapProofs.stable_prim_holds statics hp hp' hp p vs v Hstable
       (FlambdaHeapProofs.heap_evolves_refl hp) Hprim)
    as [-> Hagain].
  apply exec_named_prim_some. exists vs.
  split; [ exact Hvs | split; [ exact Hagain | reflexivity ] ].
Qed.

Theorem stable_prim_stays : stable_prim_stays_statement.
Proof.
  intros statics hp hp2 rd rd2 vars p args v Hstable Hexec [Hevolves ->].
  apply exec_named_prim_some in Hexec as (vs & Hvs & Hprim & _).
  destruct
    (FlambdaHeapProofs.stable_prim_holds statics hp hp hp2 p vs v Hstable
       Hevolves Hprim)
    as [_ Hlater].
  apply exec_named_prim_some. exists vs.
  split; [ exact Hvs | split; [ exact Hlater | reflexivity ] ].
Qed.
