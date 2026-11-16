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

(* Proofs of the statements of FlambdaKnownAritySpec.v. Not for review:
   the statements are in FlambdaKnownAritySpec.v. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs
  FlambdaPartialApplicationSpec FlambdaTupledSpec FlambdaHeapMachineSpec
  FlambdaTrapSpec FlambdaKnownAritySpec.
Import ListNotations.

(* ======================================================================== *)
(* The shape                                                                *)
(* ======================================================================== *)

(* A call of known arity goes by the callee as a closure, whatever the
   callee is. *)
Lemma shape_of_call_known (statics : static_env) (callee : value)
    (code_ids : option (list code_id)) (vs : list value)
    (args_arity return_arity : arity) :
  shape_of_call statics callee (Indirect_known_arity code_ids) vs args_arity
    return_arity
  = shape_of_closure_call statics callee (Indirect_known_arity code_ids) vs
      args_arity return_arity.
Proof. reflexivity. Qed.

(* The call of known arity of a closure is that of unknown arity where
   its two tests pass. *)
Lemma shape_of_closure_call_known (statics : static_env) (closure : value)
    (code_ids : option (list code_id)) (vs : list value)
    (args_arity return_arity : arity) :
  shape_of_closure_call statics closure (Indirect_known_arity code_ids) vs
    args_arity return_arity
  = match closure_of statics closure with
    | Some (called, _) =>
      if
        code_ids_allow code_ids called
        && known_arity_full statics called args_arity
      then
        shape_of_closure_call statics closure Indirect_unknown_arity vs
          args_arity return_arity
      else None
    | None => None
    end.
Proof.
  unfold shape_of_closure_call.
  destruct (closure_of statics closure) as [[called slots] | ];
    [ | reflexivity ].
  destruct
    (code_ids_allow code_ids called
     && known_arity_full statics called args_arity);
    reflexivity.
Qed.

Theorem known_arity_shape : known_arity_shape_statement.
Proof.
  intros statics callee code_ids vs args_arity return_arity shape Hshape.
  rewrite shape_of_call_known, shape_of_closure_call_known in Hshape.
  destruct (closure_of statics callee) as [[called slots] | ] eqn:Hclosure;
    [ | discriminate ].
  destruct (code_ids_allow code_ids called) eqn:Hallow; [ | discriminate ].
  destruct (known_arity_full statics called args_arity) eqn:Hfull;
    [ | discriminate ].
  simpl in Hshape.
  apply known_arity_full_true in Hfull
    as [Hgroup (c & Hcode & Htupled & Hcount)].
  assert (Hunknown :
    shape_of_closure_call statics callee Indirect_unknown_arity vs args_arity
      return_arity
    = Some (Full_call callee c vs args_arity)).
  { apply (shape_of_closure_call_full statics callee Indirect_unknown_arity
             vs args_arity return_arity called slots c);
      [ exact Hclosure | left; reflexivity | exact Hcode | exact Htupled
      | exact Hcount ]. }
  rewrite Hunknown in Hshape. injection Hshape as <-.
  split.
  - rewrite
      (shape_of_call_closure statics callee Indirect_unknown_arity vs
         args_arity return_arity
         (closure_not_partial statics callee _ Hclosure)).
    exact Hunknown.
  - exists called, slots, c.
    split; [ reflexivity | ]. split; [ exact Hcode | ].
    split; [ exact Htupled | ]. split; [ exact Hgroup | ].
    split; [ exact Hcount | ]. split; [ exact Hallow | reflexivity ].
Qed.

Theorem known_arity_defined : known_arity_defined_statement.
Proof.
  intros statics callee code_ids vs args_arity return_arity called slots c
    Hclosure Hcode Htupled Hgroup Hcount Hallow.
  rewrite shape_of_call_known, shape_of_closure_call_known, Hclosure, Hallow.
  assert (Hfull : known_arity_full statics called args_arity = true).
  { apply known_arity_full_true. split; [ exact Hgroup | ].
    exists c. split; [ exact Hcode | ].
    split; [ exact Htupled | exact Hcount ]. }
  rewrite Hfull. simpl.
  apply (shape_of_closure_call_full statics callee Indirect_unknown_arity vs
           args_arity return_arity called slots c);
    [ exact Hclosure | left; reflexivity | exact Hcode | exact Htupled
    | exact Hcount ].
Qed.

(* A full call of unknown arity of a closure is of the code of the
   closure, which has as many parameters as the call has groups of
   arguments unless it is tupled. *)
Lemma shape_of_closure_call_unknown_full (statics : static_env)
    (callee closure : value) (vs vs' : list value)
    (args_arity arity' return_arity : arity) (called : code_id)
    (slots : list (value_slot * value)) (c : code) :
  closure_of statics callee = Some (called, slots) ->
  shape_of_closure_call statics callee Indirect_unknown_arity vs args_arity
    return_arity
  = Some (Full_call closure c vs' arity') ->
  lookup called (static_code statics) = Some c
  ∧ (code_is_tupled c = false ->
     length args_arity = length (code_params_arity c)).
Proof.
  intros Hclosure Hshape.
  destruct (shape_of_closure_call_cases _ _ _ _ _ _ _ Hshape)
    as (called' & slots' & c' & vs'' & arity'' & Hclosure' & Hcode & _
        & Hargs & Hcall).
  rewrite Hclosure in Hclosure'. injection Hclosure' as <- <-.
  destruct (shape_of_code_call_cases _ _ _ _ _ _ Hcall)
    as [[Hcount Hfull] | [(_ & _ & _ & Hover) | (_ & _ & _ & _ & Hpartial)]];
    [ | discriminate | discriminate ].
  injection Hfull as -> -> -> ->.
  split; [ exact Hcode | ].
  intros Htupled.
  rewrite (arguments_for_code_not_tupled _ _ _ _ Htupled) in Hargs.
  injection Hargs as <- <-. exact Hcount.
Qed.

Theorem known_arity_of_unknown : known_arity_of_unknown_statement.
Proof.
  intros statics callee code_ids vs args_arity return_arity.
  destruct
    (shape_of_call statics callee (Indirect_known_arity code_ids) vs
       args_arity return_arity)
    as [shape | ] eqn:Hknown.
  - destruct (known_arity_shape _ _ _ _ _ _ _ Hknown)
      as [Hunknown
          (called & slots & c & Hclosure & Hcode & Htupled & Hgroup & Hcount
           & Hallow & ->)].
    destruct args_arity as [ | component args_arity];
      [ exfalso; apply Hgroup; reflexivity | ].
    rewrite (closure_not_partial statics callee _ Hclosure), Hclosure,
      Hunknown, Hallow, Htupled.
    reflexivity.
  - destruct args_arity as [ | component args_arity]; [ reflexivity | ].
    destruct (partial_application_of callee)
      as [[[closure applied] applied_arity] | ] eqn:Hpartial;
      [ reflexivity | ].
    destruct (closure_of statics callee) as [[called slots] | ] eqn:Hclosure;
      [ | reflexivity ].
    rewrite (shape_of_call_closure _ _ _ _ _ _ Hpartial).
    destruct
      (shape_of_closure_call statics callee Indirect_unknown_arity vs
         (component :: args_arity) return_arity)
      as [[closure c vs' arity' | closure c vs' remaining arity'
           remaining_arity | partial] | ]
      eqn:Hunknown;
      try reflexivity.
    destruct (code_ids_allow code_ids called) eqn:Hallow; [ | reflexivity ].
    destruct (code_is_tupled c) eqn:Htupled; [ reflexivity | ].
    exfalso.
    destruct
      (shape_of_closure_call_unknown_full _ _ _ _ _ _ _ _ _ _ _ Hclosure
         Hunknown)
      as [Hcode Hcount].
    rewrite
      (known_arity_defined statics callee code_ids vs
         (component :: args_arity) return_arity called slots c Hclosure Hcode
         Htupled ltac:(discriminate) (Hcount Htupled) Hallow)
      in Hknown.
    discriminate.
Qed.

(* ======================================================================== *)
(* A step                                                                   *)
(* ======================================================================== *)

(* The shape of a call of known arity that has one is that of the call
   of unknown arity. *)
Lemma shape_of_call_known_unknown (statics : static_env) (callee : value)
    (code_ids : option (list code_id)) (vs : list value)
    (args_arity return_arity : arity) :
  shape_of_call statics callee (Indirect_known_arity code_ids) vs args_arity
    return_arity
  <> None ->
  shape_of_call statics callee Indirect_unknown_arity vs args_arity
    return_arity
  = shape_of_call statics callee (Indirect_known_arity code_ids) vs
      args_arity return_arity.
Proof.
  intros Hdefined.
  destruct
    (shape_of_call statics callee (Indirect_known_arity code_ids) vs
       args_arity return_arity)
    as [shape | ] eqn:Hknown;
    [ | exfalso; apply Hdefined; reflexivity ].
  destruct (known_arity_shape _ _ _ _ _ _ _ Hknown) as [Hunknown _].
  exact Hunknown.
Qed.

Theorem known_arity_call : known_arity_call_statement.
Proof.
  intros callee code_ids vs args_arity return_arity return_to exn_to statics
    hp rd Hdefined.
  unfold call_function in Hdefined |- *.
  rewrite
    (shape_of_call_known_unknown statics callee code_ids vs args_arity
       return_arity);
    [ reflexivity | ].
  intros Hnone. apply Hdefined. rewrite Hnone. reflexivity.
Qed.

(* A step of a call, with a kind in place of its own, in each of the
   three machines. *)
Lemma step_apply_with_kind (kind : call_kind) (ap : apply_expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record) :
  step (State (Apply (apply_with_kind kind ap)) vars conts statics hp rd)
  = match
      eval_simple vars (ap_callee ap), eval_simples vars (ap_args ap),
      eval_simples vars (exn_extra_simples ap)
    with
    | Some callee, Some vs, Some extra =>
      call_function callee kind vs (ap_args_arity ap) (ap_return_arity ap)
        (resolve (ap_continuation ap) conts)
        (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
        statics hp rd
    | _, _, _ => Undefined
    end.
Proof. reflexivity. Qed.

Lemma heap_step_apply_with_kind (kind : call_kind) (ap : apply_expr)
    (vars : env) (conts : cont_env) (statics : static_env) (h : heap) :
  heap_step (Heap_state (Apply (apply_with_kind kind ap)) vars conts statics h)
  = match
      eval_simple vars (ap_callee ap), eval_simples vars (ap_args ap),
      eval_simples vars (exn_extra_simples ap)
    with
    | Some callee, Some vs, Some extra =>
      heap_call_function callee kind vs (ap_args_arity ap)
        (ap_return_arity ap) (resolve (ap_continuation ap) conts)
        (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
        statics h
    | _, _, _ => Heap_undefined
    end.
Proof. reflexivity. Qed.

Lemma stack_step_apply_with_kind (kind : call_kind) (ap : apply_expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record) (st : stack) :
  stack_step
    (State (Apply (apply_with_kind kind ap)) vars conts statics hp rd) st
  = match
      eval_simple vars (ap_callee ap), eval_simples vars (ap_args ap),
      eval_simples vars (exn_extra_simples ap)
    with
    | Some callee, Some vs, Some extra =>
      stack_run_call
        (fun partial =>
           stack_enter (resolve (ap_continuation ap) conts) [partial] statics
             hp rd st)
        (shape_of_call statics callee kind vs (ap_args_arity ap)
           (ap_return_arity ap))
        (ap_return_arity ap) (resolve (ap_continuation ap) conts) extra
        statics hp rd st
    | _, _, _ => Stack_undefined
    end.
Proof. reflexivity. Qed.

(* A call is itself with its own kind. *)
Lemma apply_with_own_kind (ap : apply_expr) :
  apply_with_kind (ap_call_kind ap) ap = ap.
Proof. destruct ap; reflexivity. Qed.

Theorem known_arity_step : known_arity_step_statement.
Proof.
  intros ap code_ids vars conts statics hp rd Hkind.
  rewrite <- (apply_with_own_kind ap) at 1 3. rewrite Hkind.
  rewrite !step_apply_with_kind.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ];
    [ | reflexivity ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  apply known_arity_call.
Qed.

Theorem known_arity_heap_step : known_arity_heap_step_statement.
Proof.
  intros ap code_ids vars conts statics h Hkind.
  rewrite <- (apply_with_own_kind ap) at 1 3. rewrite Hkind.
  rewrite !heap_step_apply_with_kind.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ];
    [ | reflexivity ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  unfold heap_call_function. intros Hdefined.
  rewrite
    (shape_of_call_known_unknown statics callee code_ids vs
       (ap_args_arity ap) (ap_return_arity ap));
    [ reflexivity | ].
  intros Hnone. apply Hdefined. rewrite Hnone. reflexivity.
Qed.

Theorem known_arity_stack_step : known_arity_stack_step_statement.
Proof.
  intros ap code_ids vars conts statics hp rd st Hkind.
  rewrite <- (apply_with_own_kind ap) at 1 3. rewrite Hkind.
  rewrite !stack_step_apply_with_kind.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ];
    [ | reflexivity ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
    [ | reflexivity ].
  intros Hdefined.
  rewrite
    (shape_of_call_known_unknown statics callee code_ids vs
       (ap_args_arity ap) (ap_return_arity ap));
    [ reflexivity | ].
  intros Hnone. apply Hdefined. rewrite Hnone. reflexivity.
Qed.

(* ======================================================================== *)
(* Where a call of known arity is undefined                                 *)
(* ======================================================================== *)

(* A call of known arity whose test on the arity and the code fails has
   no shape. *)
Lemma shape_of_call_known_not_full (statics : static_env) (callee : value)
    (code_ids : option (list code_id)) (vs : list value)
    (args_arity return_arity : arity) :
  (∀ (called : code_id) (slots : list (value_slot * value)),
    closure_of statics callee = Some (called, slots) ->
    code_ids_allow code_ids called
    && known_arity_full statics called args_arity
    = false) ->
  shape_of_call statics callee (Indirect_known_arity code_ids) vs args_arity
    return_arity
  = None.
Proof.
  intros Htests. rewrite shape_of_call_known, shape_of_closure_call_known.
  destruct (closure_of statics callee) as [[called slots] | ];
    [ | reflexivity ].
  rewrite (Htests called slots eq_refl). reflexivity.
Qed.

Theorem known_arity_no_group : known_arity_no_group_statement.
Proof.
  intros statics callee code_ids vs return_arity.
  apply shape_of_call_known_not_full. intros called slots _.
  apply andb_false_r.
Qed.

Theorem known_arity_partial_application :
  known_arity_partial_application_statement.
Proof.
  intros statics closure applied applied_arity code_ids vs args_arity
    return_arity.
  reflexivity.
Qed.

Theorem known_arity_tupled : known_arity_tupled_statement.
Proof.
  intros statics callee code_ids vs args_arity return_arity called slots c
    Hclosure Hcode Htupled.
  apply shape_of_call_known_not_full. intros called' slots' Hclosure'.
  rewrite Hclosure in Hclosure'. injection Hclosure' as <- <-.
  unfold known_arity_full.
  destruct args_arity as [ | component args_arity];
    [ apply andb_false_r | ].
  rewrite Hcode, Htupled. apply andb_false_r.
Qed.

Theorem known_arity_other_arity : known_arity_other_arity_statement.
Proof.
  intros statics callee code_ids vs args_arity return_arity called slots c
    Hclosure Hcode Hcount.
  apply shape_of_call_known_not_full. intros called' slots' Hclosure'.
  rewrite Hclosure in Hclosure'. injection Hclosure' as <- <-.
  unfold known_arity_full.
  destruct args_arity as [ | component args_arity];
    [ apply andb_false_r | ].
  apply Nat.eqb_neq in Hcount. rewrite Hcode, Hcount, !andb_false_r.
  reflexivity.
Qed.

(* ======================================================================== *)
(* The list of code ids                                                     *)
(* ======================================================================== *)

Theorem code_ids_allow_members : code_ids_allow_members_statement.
Proof. intros ids called. apply among_In. Qed.

Theorem code_ids_allow_none : code_ids_allow_none_statement.
Proof. intros called. split; reflexivity. Qed.

Theorem known_arity_not_listed : known_arity_not_listed_statement.
Proof.
  intros statics callee ids vs args_arity return_arity called slots Hclosure
    Hlisted.
  apply shape_of_call_known_not_full. intros called' slots' Hclosure'.
  rewrite Hclosure in Hclosure'. injection Hclosure' as <- <-.
  destruct (code_ids_allow (Some ids) called) eqn:Hallow; [ | reflexivity ].
  exfalso. apply Hlisted. apply code_ids_allow_members. exact Hallow.
Qed.

Theorem known_arity_forget : known_arity_forget_statement.
Proof.
  intros statics callee ids vs args_arity return_arity shape Hshape.
  destruct (known_arity_shape _ _ _ _ _ _ _ Hshape)
    as [_
        (called & slots & c & Hclosure & Hcode & Htupled & Hgroup & Hcount
         & _ & ->)].
  apply (known_arity_defined statics callee None vs args_arity return_arity
           called slots c Hclosure Hcode Htupled Hgroup Hcount).
  reflexivity.
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Theorem known_arity_examples_run : known_arity_examples_run_statement.
Proof.
  intros e Hin; repeat destruct Hin as [<- | Hin]; [ .. | destruct Hin ];
    exists 20; eexists; vm_compute; reflexivity.
Qed.

Theorem known_arity_examples_undefined :
  known_arity_examples_undefined_statement.
Proof.
  intros e Hin; repeat destruct Hin as [<- | Hin]; [ .. | destruct Hin ];
    exists 20; vm_compute; reflexivity.
Qed.

Theorem unknown_arity_examples_run : unknown_arity_examples_run_statement.
Proof.
  intros e Hin; repeat destruct Hin as [<- | Hin]; [ .. | destruct Hin ];
    exists 20; eexists; vm_compute; reflexivity.
Qed.

Theorem unknown_arity_examples_undefined :
  unknown_arity_examples_undefined_statement.
Proof.
  intros e Hin; repeat destruct Hin as [<- | Hin]; [ .. | destruct Hin ];
    exists 20; vm_compute; reflexivity.
Qed.
