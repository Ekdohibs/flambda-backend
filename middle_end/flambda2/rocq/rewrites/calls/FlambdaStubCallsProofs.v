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
   once rewritten.

   The rewrite that makes a call of known arity. One call: where the
   static environment knows the signatures of the code known, a direct
   call that has a shape and passes the test is the full call of the
   code that the environment binds ([shape_of_call_direct_full]), and
   the call of known arity is then defined and is that call
   (FlambdaKnownArityProofs.known_arity_defined). Down a term the
   induction carries what is known and that fact of the static
   environment, as FlambdaNormaliseProofs.inline_calls_refines_knowing
   does, by the lemmas of FlambdaContextProofs.v that take a property
   of the static environment which every extension keeps. At a group
   of bindings both static environments, with the code of the group as
   it is and as rewritten, know the signatures of the group
   ([eval_static_bindings_know_signatures]): the body under the one
   refines the body under the other
   (FlambdaContextProofs.related_states_refines), and the induction
   goes on under the second. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaKnownAritySpec FlambdaKnownArityProofs
  FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaEquivarianceTactic FlambdaStubCallsSpec.
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
  destruct (ap_call_kind ap) as [code | | code_ids] eqn:Hkind;
    [ | reflexivity | reflexivity ].
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
(* The rewrite that makes a call of known arity: signatures                 *)
(* ======================================================================== *)

Lemma demote_full_calls_keeps_signature (known : list (code_id * code))
    (c : code) :
  same_signature c (demote_full_calls_in_code known c).
Proof.
  destruct c as [params my_closure ret exn metadata body].
  split; reflexivity.
Qed.

Lemma statics_know_signatures_extend (known : list (code_id * code))
    (statics statics' : static_env) :
  statics_know_signatures known statics ->
  statics_extend statics statics' ->
  statics_know_signatures known statics'.
Proof.
  intros Hknow [Hcode _] id c Hid.
  destruct (Hknow id c Hid) as (c' & Hc' & Hsame).
  exists c'. split; [ apply Hcode, Hc' | exact Hsame ].
Qed.

Theorem statics_know_signatures_step :
  statics_know_signatures_step_statement.
Proof.
  intros known s s' Hknow Hstep.
  exact (statics_know_signatures_extend known _ _ Hknow
           (step_extends_statics s s' Hstep)).
Qed.

(* Two bindings of the same form, whose code, if they bind code, has the
   same id and the same signature. *)
Definition binding_same_signature (b0 b : static_binding) : Prop :=
  match b0, b with
  | Bind_code id0 c0, Bind_code id c => id0 = id ∧ same_signature c0 c
  | Bind_deleted_code _, Bind_deleted_code _ => True
  | Bind_symbol _ _, Bind_symbol _ _ => True
  | _, _ => False
  end.

Lemma bindings_same_signature_refl (bindings : list static_binding) :
  Forall2 binding_same_signature bindings bindings.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    [ apply Forall2_nil | | | ];
    (apply Forall2_cons; [ | exact IH ]); simpl; auto.
  split; [ reflexivity | split; reflexivity ].
Qed.

Lemma bindings_same_signature_demoted (known : list (code_id * code))
    (bindings : list static_binding) :
  Forall2 binding_same_signature bindings
    (map (demote_full_in_stub_binding known) bindings).
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH]; simpl;
    [ apply Forall2_nil | | | ];
    (apply Forall2_cons; [ | exact IH ]); simpl; auto.
  destruct (code_is_stub c); simpl;
    (split; [ reflexivity | ]);
    [ apply demote_full_calls_keeps_signature | split; reflexivity ].
Qed.

(* Bindings evaluated, when those of [bindings0] have the same
   signatures: the static environment knows the signatures of
   [bindings0]. *)
Lemma eval_static_bindings_know_signatures (vars : env)
    (bindings : list static_binding) :
  ∀ (bindings0 : list static_binding) (known : list (code_id * code))
    (statics extended : static_env),
    Forall2 binding_same_signature bindings0 bindings ->
    statics_know_signatures known statics ->
    eval_static_bindings vars bindings statics = Some extended ->
    statics_know_signatures (codes_of bindings0 known) extended.
Proof.
  induction bindings as [ | binding bindings IH];
    intros bindings0 known statics extended Hsame Hknow;
    inversion Hsame as [ | binding0 ? bindings0' ? Hbinding Hrest]; subst;
    simpl.
  - intros [= <-]. exact Hknow.
  - destruct binding0 as [id0 c0 | id0 | sym0 const0],
      binding as [id c | id | sym const];
      simpl in Hbinding; try contradiction; simpl.
    + destruct Hbinding as [-> Hsig].
      destruct (lookup id (static_code statics)) eqn:Hfree;
        [ discriminate | ].
      apply (IH bindings0' ((id, c0) :: known)); [ exact Hrest | ].
      intros id' c' Hlookup. simpl in Hlookup |- *.
      destruct (Pos.eqb id' id).
      * injection Hlookup as <-. exists c.
        split; [ reflexivity | exact Hsig ].
      * exact (Hknow id' c' Hlookup).
    + apply (IH bindings0' known); assumption.
    + destruct (lookup sym (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ]; simpl;
        [ | discriminate ].
      apply (IH bindings0' known); assumption.
Qed.

Theorem group_knows_signatures : group_knows_signatures_statement.
Proof.
  intros known known' vars bindings statics extended Hknow [Heval | Heval].
  - exact (eval_static_bindings_know_signatures vars bindings bindings known
             statics extended (bindings_same_signature_refl bindings) Hknow
             Heval).
  - exact (eval_static_bindings_know_signatures vars _ bindings known statics
             extended (bindings_same_signature_demoted known' bindings)
             Hknow Heval).
Qed.

(* ======================================================================== *)
(* The rewrite that makes a call of known arity: one call                   *)
(* ======================================================================== *)

(* A direct call that has a shape, of code that is known, is not tupled
   and has as many parameters as the call has groups of arguments, where
   the static environment knows the signatures: the callee is a closure,
   and no partial application, of code that the static environment
   binds, which is not tupled, and the call is the full call of it. *)
Lemma shape_of_call_direct_full (known : list (code_id * code))
    (statics : static_env) (callee : value) (called : code_id) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (shape : call_shape) :
  statics_know_signatures known statics ->
  lookup called known = Some c ->
  code_is_tupled c = false ->
  length args_arity = length (code_params_arity c) ->
  shape_of_call statics callee (Direct called) vs args_arity return_arity
  = Some shape ->
  ∃ (slots : list (value_slot * value)) (c' : code),
    partial_application_of callee = None
    ∧ closure_of statics callee = Some (called, slots)
    ∧ lookup called (static_code statics) = Some c'
    ∧ code_is_tupled c' = false
    ∧ length args_arity = length (code_params_arity c')
    ∧ shape = Full_call callee c' vs args_arity.
Proof.
  intros Hknow Hknown Htupled Hlength Hshape.
  destruct (Hknow called c Hknown) as (c' & Hc' & Harity & Htupled').
  unfold shape_of_call in Hshape.
  destruct (partial_application_of callee)
    as [[[closure applied] applied_arity] | ] eqn:Hpartial.
  - (* A partial application is a closure of no code. *)
    exfalso. destruct callee; try discriminate Hpartial.
    discriminate Hshape.
  - unfold shape_of_closure_call in Hshape.
    destruct (closure_of statics callee) as [[called' slots] | ] eqn:Hclosure;
      [ | discriminate Hshape ].
    destruct (Pos.eqb called' called) eqn:Hcalled;
      [ | destruct (lookup called' (static_code statics));
          discriminate Hshape ].
    apply Pos.eqb_eq in Hcalled. subst called'.
    rewrite Hc' in Hshape. unfold arguments_for_code in Hshape.
    rewrite Htupled', Htupled in Hshape. cbn in Hshape.
    unfold shape_of_code_call in Hshape.
    rewrite Harity, Hlength, Nat.eqb_refl in Hshape.
    injection Hshape as <-.
    exists slots, c'. rewrite Harity.
    repeat split; try assumption; try reflexivity.
    congruence.
Qed.

(* The call of known arity of the same callee, which lists no code, is
   then defined and is that full call. *)
Lemma call_function_direct_full (known : list (code_id * code))
    (callee : value) (called : code_id) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  statics_know_signatures known statics ->
  lookup called known = Some c ->
  code_is_tupled c = false ->
  args_arity <> [] ->
  length args_arity = length (code_params_arity c) ->
  call_function callee (Direct called) vs args_arity return_arity return_to
    exn_to statics hp rd
  <> Undefined ->
  call_function callee (Indirect_known_arity None) vs args_arity
    return_arity return_to exn_to statics hp rd
  = call_function callee (Direct called) vs args_arity return_arity
      return_to exn_to statics hp rd.
Proof.
  intros Hknow Hknown Htupled Hgroups Hlength Hstep.
  unfold call_function in *.
  destruct
    (shape_of_call statics callee (Direct called) vs args_arity return_arity)
    as [shape | ] eqn:Hshape;
    [ | exfalso; apply Hstep; reflexivity ].
  destruct
    (shape_of_call_direct_full known statics callee called c vs args_arity
       return_arity shape Hknow Hknown Htupled Hlength Hshape)
    as (slots & c' & _ & Hclosure & Hc' & Htupled' & Hlength' & ->).
  rewrite
    (known_arity_defined statics callee None vs args_arity return_arity
       called slots c' Hclosure Hc' Htupled' Hgroups Hlength' eq_refl).
  reflexivity.
Qed.

Theorem demote_full_call_step : demote_full_call_step_statement.
Proof.
  intros known ap vars conts statics hp rd Hknow Hstep.
  unfold demote_full_call.
  destruct (ap_call_kind ap) as [called | | code_ids] eqn:Hkind;
    try reflexivity.
  destruct (lookup called known) as [c | ] eqn:Hknown; [ | reflexivity ].
  destruct (full_call_of_b ap c) eqn:Hfull; [ | reflexivity ].
  unfold full_call_of_b in Hfull.
  apply andb_true_iff in Hfull as [Hfull Hlength].
  apply andb_true_iff in Hfull as [Hfull Hgroups].
  apply andb_true_iff in Hfull as [_ Htupled].
  apply negb_true_iff in Htupled. apply Nat.eqb_eq in Hlength.
  assert (Hnonempty : ap_args_arity ap <> [])
    by (destruct (ap_args_arity ap); [ discriminate Hgroups | discriminate ]).
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
  exact (call_function_direct_full known callee called c vs _ _ _ _ statics
           hp rd Hknow Hknown Htupled Hnonempty Hlength Hstep).
Qed.

Lemma demote_full_call_refines (known : list (code_id * code))
    (ap : apply_expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record) :
  statics_know_signatures known statics ->
  refines (State (Apply ap) vars conts statics hp rd)
    (State (Apply (demote_full_call known ap)) vars conts statics hp rd).
Proof.
  intros Hknow.
  destruct (step (State (Apply ap) vars conts statics hp rd)) eqn:Hstep;
    try (apply equivalent_refines, step_eq_equivalent; symmetry;
         apply demote_full_call_step;
         [ exact Hknow | rewrite Hstep; discriminate ]).
  apply step_undefined_refines, Hstep.
Qed.

(* ======================================================================== *)
(* The rewrite that makes a call of known arity: what binds something       *)
(* static                                                                   *)
(* ======================================================================== *)

Fixpoint demote_full_calls_binds_statics (known : list (code_id * code))
    (e : expr) :
  binds_statics (demote_full_calls known e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - apply demote_full_calls_binds_statics.
  - rewrite demote_full_calls_binds_statics. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite !demote_full_calls_binds_statics. reflexivity.
  - rewrite demote_full_calls_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite demote_full_calls_binds_statics, IH. reflexivity.
Qed.

Lemma demote_full_calls_handlers_bind_statics
    (known : list (code_id * code))
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, demote_full_calls_in_handler known h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite demote_full_calls_binds_statics, IH. reflexivity.
Qed.

Fixpoint demote_full_in_stubs_binds_statics (e : expr) :
  ∀ known : list (code_id * code),
    binds_statics (demote_full_in_stubs known e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known; simpl; try reflexivity.
  - apply demote_full_in_stubs_binds_statics.
  - rewrite demote_full_in_stubs_binds_statics. f_equal.
    generalize (codes_of bindings known) as known'. intros known'.
    induction bindings
      as [ | [id [params my_closure ret exn metadata code_body] | id
             | sym const]
             bindings IH];
      simpl; try congruence.
    destruct (cm_is_stub metadata); simpl; congruence.
  - destruct h as [params metadata handler]. simpl.
    rewrite !demote_full_in_stubs_binds_statics. reflexivity.
  - rewrite demote_full_in_stubs_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite demote_full_in_stubs_binds_statics, IH. reflexivity.
Qed.

Lemma demote_full_in_stubs_handlers_bind_statics
    (known : list (code_id * code))
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, demote_full_in_stubs_of_handler known h))
       handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite demote_full_in_stubs_binds_statics, IH. reflexivity.
Qed.

(* ======================================================================== *)
(* The rewrite that makes a call of known arity: the calls of an            *)
(* expression                                                               *)
(* ======================================================================== *)

(* Down the term, with the static environment knowing the signatures of
   what is known: at once in a body, and from then on in a handler, as
   FlambdaNormaliseProofs.inline_calls_refines_knowing. A [Let_static]
   is left as it is, and what it adds to the static environment keeps
   what was known. *)
Fixpoint demote_full_calls_refines_aux (e : expr) :
  ∀ (known : list (code_id * code)) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    statics_know_signatures known statics ->
    refines (State e vars conts statics hp rd)
      (State (demote_full_calls known e) vars conts statics hp rd).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known vars conts statics hp rd Hknow; simpl;
    try apply refines_refl.
  - apply let_refines_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
    apply demote_full_calls_refines_aux, Hknow.
  - apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings; [ | exact I ].
    apply demote_full_calls_refines_aux.
    exact (statics_know_signatures_extend known _ _ Hknow
             (eval_static_bindings_extend vars _ _ _ Hbindings)).
  - destruct h as [params metadata handler]. simpl.
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata (demote_full_calls known handler)) body)
         vars conts statics hp rd)).
    + apply (let_cont_handler_refines_where (statics_know_signatures known)
        (statics_know_signatures_extend known)); [ exact Hknow | ].
      intros vs handler_vars conts' statics' hp' rd' Hknow' _ _.
      apply demote_full_calls_refines_aux, Hknow'.
    + apply let_cont_non_recursive_refines_step,
        demote_full_calls_refines_aux, Hknow.
  - apply (refines_trans _
      (State
         (Let_cont_recursive
            (map (fun '(k, h) => (k, demote_full_calls_in_handler known h))
               handlers)
            body)
         vars conts statics hp rd)).
    + assert (Hhandlers :
        handlers_refine (statics_know_signatures known) vars handlers
          (map (fun '(k, h) => (k, demote_full_calls_in_handler known h))
             handlers)).
      { induction handlers
          as [ | [k [params metadata handler]] handlers IH];
          simpl.
        - split; apply Forall2_nil.
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ | assumption ]);
            (split; [ reflexivity | ]); apply Handler_related;
            intros vs handler_vars conts' statics' hp' rd' Hknow' _ _;
            apply demote_full_calls_refines_aux, Hknow'. }
      destruct Hhandlers as [Hforwards Hbackwards].
      apply (let_cont_recursive_handlers_refine_where
        (statics_know_signatures known)
        (statics_know_signatures_extend known));
        try assumption.
      symmetry. apply demote_full_calls_handlers_bind_statics.
    + apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply demote_full_calls_refines_aux, Hknow.
  - apply demote_full_call_refines, Hknow.
Qed.

Theorem demote_full_calls_refines : demote_full_calls_refines_statement.
Proof.
  intros known e vars conts statics hp rd Hknow.
  exact (demote_full_calls_refines_aux e known vars conts statics hp rd Hknow).
Qed.

(* ======================================================================== *)
(* The rewrite that makes a call of known arity: the code of the stubs      *)
(* ======================================================================== *)

(* The code of a group as the rewrite leaves it, knowing [known]: the
   bodies refine their counterparts where the static environment knows
   the signatures of [known]. *)
Lemma demote_full_bindings_refine (known : list (code_id * code))
    (bindings : list static_binding) :
  bindings_refine (statics_know_signatures known) bindings
    (map (demote_full_in_stub_binding known) bindings).
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
        [ symmetry; apply demote_full_calls_binds_statics
        | apply demote_full_calls_binds_statics
        | intros vars conts statics hp rd Hknow;
          apply demote_full_calls_refines_aux, Hknow ].
  - split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
  - split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
Qed.

(* Down the term. At a group of bindings the two static environments,
   with the code as it is and as rewritten, both know the signatures of
   the group: the code of the stubs of the group, and the body of the
   [Let_static], are rewritten knowing them. *)
Fixpoint demote_full_in_stubs_refines_aux (e : expr) :
  ∀ (known : list (code_id * code)) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    statics_know_signatures known statics ->
    refines (State e vars conts statics hp rd)
      (State (demote_full_in_stubs known e) vars conts statics hp rd).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known vars conts statics hp rd Hknow; simpl;
    try apply refines_refl.
  - apply let_refines_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
    apply demote_full_in_stubs_refines_aux, Hknow.
  - pose proof (demote_full_bindings_refine (codes_of bindings known)
      bindings) as [Hforwards Hbackwards].
    pose proof
      (bindings_same_signature_demoted (codes_of bindings known) bindings)
      as Hsame.
    set (known' := codes_of bindings known) in *.
    set (bindings' := map (demote_full_in_stub_binding known') bindings)
      in *.
    apply (fun H =>
      eval_static_bindings_related _ vars _ _ H statics statics
        (statics_related_refl _ statics)) in Hforwards, Hbackwards.
    apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Hbindings,
      (eval_static_bindings vars bindings' statics) as [extended' | ]
        eqn:Hbindings';
      try contradiction; [ | exact I ].
    assert (Hknown : statics_know_signatures known' extended)
      by exact
           (eval_static_bindings_know_signatures vars bindings bindings
              known statics extended
              (bindings_same_signature_refl bindings) Hknow Hbindings).
    assert (Hknown' : statics_know_signatures known' extended')
      by exact
           (eval_static_bindings_know_signatures vars bindings' bindings
              known statics extended' Hsame Hknow Hbindings').
    apply (refines_trans _ (State body vars conts extended' hp rd)).
    + apply (related_states_refines (statics_know_signatures known')
        (statics_know_signatures_extend known'));
        [ exact Hknown | exact Hknown'
        | apply cont_env_related_refl | apply cont_env_related_refl
        | exact Hforwards | exact Hbackwards ].
    + apply demote_full_in_stubs_refines_aux. exact Hknown'.
  - destruct h as [params metadata handler]. simpl.
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata (demote_full_in_stubs known handler))
            body)
         vars conts statics hp rd)).
    + apply (let_cont_handler_refines_where (statics_know_signatures known)
        (statics_know_signatures_extend known)); [ exact Hknow | ].
      intros vs handler_vars conts' statics' hp' rd' Hknow' _ _.
      apply demote_full_in_stubs_refines_aux, Hknow'.
    + apply let_cont_non_recursive_refines_step,
        demote_full_in_stubs_refines_aux, Hknow.
  - apply (refines_trans _
      (State
         (Let_cont_recursive
            (map
               (fun '(k, h) => (k, demote_full_in_stubs_of_handler known h))
               handlers)
            body)
         vars conts statics hp rd)).
    + assert (Hhandlers :
        handlers_refine (statics_know_signatures known) vars handlers
          (map (fun '(k, h) => (k, demote_full_in_stubs_of_handler known h))
             handlers)).
      { induction handlers
          as [ | [k [params metadata handler]] handlers IH];
          simpl.
        - split; apply Forall2_nil.
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ | assumption ]);
            (split; [ reflexivity | ]); apply Handler_related;
            intros vs handler_vars conts' statics' hp' rd' Hknow' _ _;
            apply demote_full_in_stubs_refines_aux, Hknow'. }
      destruct Hhandlers as [Hforwards Hbackwards].
      apply (let_cont_recursive_handlers_refine_where
        (statics_know_signatures known)
        (statics_know_signatures_extend known));
        try assumption.
      symmetry. apply demote_full_in_stubs_handlers_bind_statics.
    + apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply demote_full_in_stubs_refines_aux, Hknow.
Qed.

Theorem demote_full_in_stubs_refines :
  demote_full_in_stubs_refines_statement.
Proof.
  intros known e vars conts statics hp rd Hknow.
  exact (demote_full_in_stubs_refines_aux e known vars conts statics hp rd
           Hknow).
Qed.

Theorem demote_full_stub_calls_refines :
  demote_full_stub_calls_refines_statement.
Proof.
  intros e vars conts statics hp rd. apply demote_full_in_stubs_refines_aux.
  intros id c Hlookup. discriminate Hlookup.
Qed.

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

Theorem demote_full_stub_calls_equivariant :
  demote_full_stub_calls_equivariant_statement.
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

Theorem demote_full_stub_calls_scope :
  demote_full_stub_calls_scope_statement.
Proof. repeat split; reflexivity. Qed.

Theorem demote_full_stub_calls_test : demote_full_stub_calls_test_statement.
Proof. repeat split; reflexivity. Qed.

Theorem demote_full_call_needs_scope :
  demote_full_call_needs_scope_statement.
Proof. split; [ cbv; discriminate | reflexivity ]. Qed.
