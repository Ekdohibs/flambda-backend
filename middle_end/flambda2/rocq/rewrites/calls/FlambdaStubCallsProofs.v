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

(* Proofs of the statements of FlambdaStubCallsSpec.v. Nothing here
   needs review.

   One call: the shape of a direct call that has one is the shape of the
   indirect call of the same callee (FlambdaSemanticsSpec.shape_of_call),
   so the two calls step alike. Down a term each layer is a context
   around what the induction gives
   (FlambdaContextProofs.plug_refines); the handlers of a recursive
   [Let_cont] and the code of a group are replaced all at once
   (FlambdaContextProofs.let_cont_recursive_handlers_refine_where,
   FlambdaNormaliseProofs.let_static_bindings_refine). The rewrite
   changes no [Let_static], so what binds something static does so
   once rewritten. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaContextSpec FlambdaContextProofs FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaEquivarianceTactic FlambdaStubCallsSpec.
Import ListNotations.

(* ======================================================================== *)
(* One call                                                                 *)
(* ======================================================================== *)

Lemma shape_of_closure_call_direct (statics : static_env) (closure : value)
    (code : code_id) (vs : list value) (args_arity return_arity : arity) :
  shape_of_closure_call statics closure (Direct code) vs args_arity
    return_arity
  <> None ->
  shape_of_closure_call statics closure Indirect_unknown_arity vs args_arity
    return_arity
  = shape_of_closure_call statics closure (Direct code) vs args_arity
      return_arity.
Proof.
  unfold shape_of_closure_call.
  destruct (closure_of statics closure) as [[called slot_values] | ];
    [ | intros Hshape; exfalso; apply Hshape; reflexivity ].
  destruct (Pos.eqb called code); [ reflexivity | ].
  intros Hshape. exfalso. apply Hshape.
  destruct (lookup called (static_code statics)); reflexivity.
Qed.

Lemma shape_of_call_direct (statics : static_env) (callee : value)
    (code : code_id) (vs : list value) (args_arity return_arity : arity) :
  shape_of_call statics callee (Direct code) vs args_arity return_arity
  <> None ->
  shape_of_call statics callee Indirect_unknown_arity vs args_arity return_arity
  = shape_of_call statics callee (Direct code) vs args_arity return_arity.
Proof.
  unfold shape_of_call.
  destruct (partial_application_of callee)
    as [[[closure applied] applied_arity] | ] eqn:Hpartial.
  - (* A partial application is a closure of no code. *)
    intros Hshape. exfalso. apply Hshape.
    destruct callee; try discriminate Hpartial. reflexivity.
  - intros Hshape.
    exact (shape_of_closure_call_direct statics callee code vs args_arity
             return_arity Hshape).
Qed.

Lemma call_function_direct (callee : value) (code : code_id)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  call_function callee (Direct code) vs args_arity return_arity return_to
    exn_to statics hp rd
  <> Undefined ->
  call_function callee Indirect_unknown_arity vs args_arity return_arity
    return_to exn_to statics hp rd
  = call_function callee (Direct code) vs args_arity return_arity return_to
      exn_to statics hp rd.
Proof.
  unfold call_function. intros Hstep.
  rewrite (shape_of_call_direct statics callee code vs args_arity return_arity);
    [ reflexivity | ].
  intros Hshape. apply Hstep. rewrite Hshape. reflexivity.
Qed.

Theorem demote_call_step : demote_call_step_statement.
Proof.
  intros ids ap vars conts statics hp rd Hstep.
  unfold demote_call.
  destruct (ap_call_kind ap) as [code | ] eqn:Hkind; [ | reflexivity ].
  destruct (among code ids); [ | reflexivity ].
  rewrite step_apply in Hstep. rewrite !step_apply.
  unfold exn_extra_simples in *. cbn [ap_callee ap_args ap_exn_extra_args
    ap_call_kind ap_args_arity ap_return_arity ap_continuation
    ap_exn_continuation].
  rewrite Hkind in *.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ];
    [ | reflexivity ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (map fst (ap_exn_extra_args ap)))
    as [extra | ]; [ | reflexivity ].
  apply call_function_direct, Hstep.
Qed.

Lemma demote_call_refines (ids : list code_id) (ap : apply_expr) :
  expr_refines (Apply ap) (Apply (demote_call ids ap)).
Proof.
  intros vars conts statics hp rd.
  destruct (step (State (Apply ap) vars conts statics hp rd)) eqn:Hstep;
    try (apply equivalent_refines, step_eq_equivalent; symmetry;
         apply demote_call_step; rewrite Hstep; discriminate).
  apply step_undefined_refines, Hstep.
Qed.

(* ======================================================================== *)
(* What binds something static                                              *)
(* ======================================================================== *)

Fixpoint demote_calls_binds_statics (ids : list code_id) (e : expr) :
  binds_statics (demote_calls ids e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - apply demote_calls_binds_statics.
  - rewrite demote_calls_binds_statics. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite !demote_calls_binds_statics. reflexivity.
  - rewrite demote_calls_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite demote_calls_binds_statics, IH. reflexivity.
Qed.

Lemma demote_calls_handlers_bind_statics (ids : list code_id)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, demote_calls_in_handler ids h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite demote_calls_binds_statics, IH. reflexivity.
Qed.

Fixpoint demote_in_stubs_binds_statics (ids : list code_id) (e : expr) :
  binds_statics (demote_in_stubs ids e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - apply demote_in_stubs_binds_statics.
  - rewrite demote_in_stubs_binds_statics. f_equal.
    induction bindings
      as [ | [id [params my_closure ret exn metadata code_body] | id
             | sym const]
             bindings IH];
      simpl; try congruence.
    destruct (cm_is_stub metadata); simpl; congruence.
  - destruct h as [params metadata handler]. simpl.
    rewrite !demote_in_stubs_binds_statics. reflexivity.
  - rewrite demote_in_stubs_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite demote_in_stubs_binds_statics, IH. reflexivity.
Qed.

Lemma demote_in_stubs_handlers_bind_statics (ids : list code_id)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, demote_in_stubs_of_handler ids h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite demote_in_stubs_binds_statics, IH. reflexivity.
Qed.

(* ======================================================================== *)
(* The calls of an expression                                               *)
(* ======================================================================== *)

Fixpoint demote_calls_refines_aux (ids : list code_id) (e : expr)
    {struct e} : expr_refines e (demote_calls ids e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try apply expr_refines_refl.
  - exact (plug_refines (Let_body x kind d Hole) _ _ (not_many _)
      (demote_calls_refines_aux ids body)).
  - exact (plug_refines (Let_static_body bindings Hole) _ _ (not_many _)
      (demote_calls_refines_aux ids body)).
  - destruct h as [params metadata handler]. simpl.
    apply (expr_refines_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (demote_calls ids handler)) body)).
    + exact (plug_refines
        (Let_cont_non_recursive_handler k params metadata Hole body) _ _
        (not_many _) (demote_calls_refines_aux ids handler)).
    + exact (plug_refines
        (Let_cont_non_recursive_body k
           (Handler params metadata (demote_calls ids handler)) Hole)
        _ _ (not_many _) (demote_calls_refines_aux ids body)).
  - assert (Hhandlers :
      ∀ vars : env,
        handlers_refine always vars handlers
          (map (fun '(k, h) => (k, demote_calls_in_handler ids h)) handlers)).
    { induction handlers as [ | [k [params metadata handler]] handlers IH];
        intros vars; simpl.
      - split; apply Forall2_nil.
      - destruct (IH vars) as [Hforwards Hbackwards].
        split; (apply Forall2_cons; [ | assumption ]);
          (split; [ reflexivity | ]); apply Handler_related;
          intros vs handler_vars conts statics hp rd _ _ _;
          apply (demote_calls_refines_aux ids handler). }
    eapply expr_refines_trans;
      [ | exact (plug_refines (Let_cont_recursive_body _ Hole) _ _
            (not_many _) (demote_calls_refines_aux ids body)) ].
    intros vars conts statics hp rd.
    destruct (Hhandlers vars) as [Hforwards Hbackwards].
    apply (let_cont_recursive_handlers_refine_where always always_extend);
      [ exact I | symmetry; apply demote_calls_handlers_bind_statics
      | exact Hforwards | exact Hbackwards ].
  - apply demote_call_refines.
Qed.

Theorem demote_calls_refines : demote_calls_refines_statement.
Proof. exact demote_calls_refines_aux. Qed.

(* ======================================================================== *)
(* The code of the stubs                                                    *)
(* ======================================================================== *)

Lemma demote_bindings_refine (ids : list code_id)
    (bindings : list static_binding) :
  bindings_refine always bindings (map (demote_in_stub_binding ids) bindings).
Proof.
  induction bindings
    as [ | [id [params my_closure ret exn metadata code_body] | id
           | sym const]
           bindings [Hforwards Hbackwards]];
    simpl.
  - split; apply Forall2_nil.
  - unfold code_is_stub.
    destruct (cm_is_stub metadata);
      split; (apply Forall2_cons; [ | assumption ]);
      try apply Binding_same;
      apply Binding_code, Codes_related;
      first
        [ symmetry; apply demote_calls_binds_statics
        | apply demote_calls_binds_statics
        | intros vars conts statics hp rd _; apply demote_calls_refines_aux ].
  - split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
  - split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
Qed.

Fixpoint demote_in_stubs_refines_aux (ids : list code_id) (e : expr)
    {struct e} : expr_refines e (demote_in_stubs ids e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try apply expr_refines_refl.
  - exact (plug_refines (Let_body x kind d Hole) _ _ (not_many _)
      (demote_in_stubs_refines_aux ids body)).
  - eapply expr_refines_trans;
      [ apply let_static_bindings_refine, demote_bindings_refine
      | exact (plug_refines (Let_static_body _ Hole) _ _ (not_many _)
          (demote_in_stubs_refines_aux ids body)) ].
  - destruct h as [params metadata handler]. simpl.
    apply (expr_refines_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (demote_in_stubs ids handler)) body)).
    + exact (plug_refines
        (Let_cont_non_recursive_handler k params metadata Hole body) _ _
        (not_many _) (demote_in_stubs_refines_aux ids handler)).
    + exact (plug_refines
        (Let_cont_non_recursive_body k
           (Handler params metadata (demote_in_stubs ids handler)) Hole)
        _ _ (not_many _) (demote_in_stubs_refines_aux ids body)).
  - assert (Hhandlers :
      ∀ vars : env,
        handlers_refine always vars handlers
          (map (fun '(k, h) => (k, demote_in_stubs_of_handler ids h))
             handlers)).
    { induction handlers as [ | [k [params metadata handler]] handlers IH];
        intros vars; simpl.
      - split; apply Forall2_nil.
      - destruct (IH vars) as [Hforwards Hbackwards].
        split; (apply Forall2_cons; [ | assumption ]);
          (split; [ reflexivity | ]); apply Handler_related;
          intros vs handler_vars conts statics hp rd _ _ _;
          apply (demote_in_stubs_refines_aux ids handler). }
    eapply expr_refines_trans;
      [ | exact (plug_refines (Let_cont_recursive_body _ Hole) _ _
            (not_many _) (demote_in_stubs_refines_aux ids body)) ].
    intros vars conts statics hp rd.
    destruct (Hhandlers vars) as [Hforwards Hbackwards].
    apply (let_cont_recursive_handlers_refine_where always always_extend);
      [ exact I | symmetry; apply demote_in_stubs_handlers_bind_statics
      | exact Hforwards | exact Hbackwards ].
Qed.

Theorem demote_in_stubs_refines : demote_in_stubs_refines_statement.
Proof. exact demote_in_stubs_refines_aux. Qed.

Theorem demote_stub_calls_refines : demote_stub_calls_refines_statement.
Proof. intros e. apply demote_in_stubs_refines_aux. Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The tactic looks for the laws of its database in one pass over the
   goal, which is enough for the proofs below and takes less time. *)
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

(* The law of the rewrite is proved from the definitions by the tactic
   of FlambdaEquivarianceTactic.v, which states and proves by itself
   the laws of the functions that the rewrite is made of. *)
Theorem demote_stub_calls_equivariant :
  demote_stub_calls_equivariant_statement.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* A unit, and a call                                                       *)
(* ======================================================================== *)

Theorem demote_stub_calls_demotes : demote_stub_calls_demotes_statement.
Proof. reflexivity. Qed.

Theorem demote_stub_calls_not_a_stub :
  demote_stub_calls_not_a_stub_statement.
Proof. reflexivity. Qed.

Theorem demote_call_defines : demote_call_defines_statement.
Proof. split; [ reflexivity | cbv; discriminate ]. Qed.
