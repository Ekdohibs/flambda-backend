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

(* Proofs of the statements of FlambdaNeverReturnsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaSemanticsProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaStaticSpec FlambdaStaticProofs FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaUnusedContsSpec FlambdaUnusedContsProofs
  FlambdaAlphaSpec FlambdaAlphaProofs FlambdaEquivarianceSpec
  FlambdaEquivarianceProofs FlambdaValidatorEquivarianceBaseProofs
  FlambdaEquivarianceTactic FlambdaNeverReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* A call of code that does not mention its return continuation             *)
(* ======================================================================== *)

Theorem call_code_never_returns : call_code_never_returns_statement.
Proof.
  intros closure c vs args_arity return_arity return_to return_to' exn_to
    statics hp rd Hfree.
  unfold call_code.
  destruct (binds_statics (code_body c)); [ exact I | ].
  destruct (negb
              (kinds_eqb (map kind_of_value vs) (unarize args_arity)
               && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
                    (unarize_per_parameter (code_params_arity c))
               && arity_eqb return_arity (code_result_arity c)));
    [ exact I | ].
  destruct (bind_params (code_params c) vs [(code_my_closure c, closure)])
    as [callee_vars | ]; [ | exact I ].
  apply
    (equivalent_trans _ _ _
       (remove_unused_conts_equivalent (code_body c) _ _ _ _ rd_)).
  refine
    (equivalent_trans _ _ _ _
       (equivalent_sym _ _
          (remove_unused_conts_equivalent (code_body c) _ _ _ _ rd_))).
  apply coincide_equivalent; [ reflexivity | ].
  intros k Hk. unfold conts_agree_at, resolve.
  change (rename_continuation identity_renaming k) with k. simpl.
  destruct (Pos.eqb_spec k (code_return_continuation c)) as [-> | _].
  - apply cont_free_b_sound in Hk. congruence.
  - exact (conts_agree_same identity_renaming k
             [(code_exn_continuation c, exn_to)] eq_refl).
Qed.

(* A continuation that is found in a term with its unused bindings
   removed is found in the term. *)
Fixpoint cont_free_b_remove_unused_conts (k : continuation) (e : expr)
    {struct e} :
  cont_free_b k (remove_unused_conts e) = true -> cont_free_b k e = true
with cont_free_in_handler_b_remove_unused_conts (k : continuation)
    (h : continuation_handler) {struct h} :
  cont_free_in_handler_b k (remove_unused_conts_in_handler h) = true ->
  cont_free_in_handler_b k h = true.
Proof.
  - destruct e
      as [ x kind d body | bindings body | bound h body | handlers body | ap
         | ac | scrutinee arms | ];
      simpl; try (intros H; exact H).
    + apply cont_free_b_remove_unused_conts.
    + apply cont_free_b_remove_unused_conts.
    + destruct (cont_free_b bound (remove_unused_conts body)) eqn:Hbound;
        simpl; intros H; apply orb_true_iff.
      * apply orb_true_iff in H as [H | H];
          [ left; exact (cont_free_in_handler_b_remove_unused_conts k h H)
          | right ].
        apply andb_true_iff in H as [Hk H]. rewrite Hk. simpl.
        exact (cont_free_b_remove_unused_conts k body H).
      * right. destruct (Pos.eqb_spec k bound) as [-> | _];
          [ congruence | ].
        simpl. exact (cont_free_b_remove_unused_conts k body H).
    + intros H. apply orb_true_iff.
      apply orb_true_iff in H as [H | H];
        [ left; exact H
        | right; exact (cont_free_b_remove_unused_conts k body H) ].
  - destruct h as [params metadata handler]. simpl.
    apply cont_free_b_remove_unused_conts.
Qed.

(* [call_code_never_returns_statement] as it was when the test was of
   the body itself: it follows. *)
Lemma call_code_never_returns_of_body (closure : value) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to : cont_value) (statics : static_env)
    (hp : heap) (rd : read_record) :
  cont_free_b (code_return_continuation c) (code_body c) = false ->
  match
    call_code closure c vs args_arity return_arity return_to exn_to
      statics hp rd,
    call_code closure c vs args_arity return_arity return_to' exn_to
      statics hp rd
  with
  | Step s, Step s' => equivalent s s'
  | Undefined, Undefined => True
  | _, _ => False
  end.
Proof.
  intros Hfree. apply call_code_never_returns.
  destruct
    (cont_free_b (code_return_continuation c)
       (remove_unused_conts (code_body c)))
    eqn:Hclean;
    [ | reflexivity ].
  rewrite (cont_free_b_remove_unused_conts _ _ Hclean) in Hfree.
  discriminate Hfree.
Qed.

(* The test passes of every call of which it passed when it was of the
   body itself. *)
Lemma never_returns_b_of_body (ap : apply_expr) (c : code) :
  negb (cont_free_b (code_return_continuation c) (code_body c))
  && negb (code_is_tupled c)
  && Nat.leb (length (code_params_arity c)) (length (ap_args_arity ap))
  && negb (Pos.eqb (ap_continuation ap) (ap_exn_continuation ap))
  = true ->
  never_returns_b ap c = true.
Proof.
  unfold never_returns_b. intros H.
  apply andb_true_iff in H as [H Hapart].
  apply andb_true_iff in H as [H Hargs].
  apply andb_true_iff in H as [Hfree Htupled].
  rewrite Htupled, Hargs, Hapart, !andb_true_r.
  apply negb_true_iff. apply negb_true_iff in Hfree.
  destruct
    (cont_free_b (code_return_continuation c)
       (remove_unused_conts (code_body c)))
    eqn:Hclean;
    [ | reflexivity ].
  rewrite (cont_free_b_remove_unused_conts _ _ Hclean) in Hfree.
  discriminate Hfree.
Qed.

(* The same of a direct call that is not a partial application. *)
Lemma call_function_never_returns (callee : value) (called : code_id)
    (c : code) (vs : list value) (args_arity return_arity : arity)
    (return_to return_to' exn_to : cont_value) (statics : static_env)
    (hp : heap) (rd : read_record) :
  lookup called (static_code statics) = Some c ->
  cont_free_b (code_return_continuation c)
    (remove_unused_conts (code_body c))
  = false ->
  code_is_tupled c = false ->
  length (code_params_arity c) <= length args_arity ->
  match
    call_function callee (Direct called) vs args_arity return_arity
      return_to exn_to statics hp rd,
    call_function callee (Direct called) vs args_arity return_arity
      return_to' exn_to statics hp rd
  with
  | Step s, Step s' => equivalent s s'
  | Undefined, Undefined => True
  | _, _ => False
  end.
Proof.
  intros Hlookup Hfree Htupled Hargs.
  unfold call_function, shape_of_call, shape_of_closure_call.
  destruct (closure_of statics callee) as [[called' slots] | ];
    [ | exact I ].
  destruct (Pos.eqb_spec called' called) as [-> | _]; [ | exact I ].
  rewrite Hlookup. unfold arguments_for_code. rewrite Htupled. simpl.
  unfold shape_of_code_call.
  destruct (Nat.eqb (length args_arity) (length (code_params_arity c)))
    eqn:Hfull.
  - apply call_code_never_returns, Hfree.
  - destruct (negb
                (kinds_eqb (map kind_of_value vs) (unarize args_arity)
                 && partial_or_over_application args_arity return_arity
                      (code_params_arity c) (code_result_arity c)));
      [ exact I | ].
    destruct (Nat.ltb (length (code_params_arity c)) (length args_arity))
      eqn:Hover.
    + apply call_code_never_returns, Hfree.
    + apply Nat.eqb_neq in Hfull. apply Nat.ltb_ge in Hover. lia.
Qed.

(* Two states that are both undefined, or that both take a step to
   states that behave alike: the first refines the second. *)
Lemma steps_alike_refines (s t : state) :
  match step s, step t with
  | Step s', Step t' => equivalent s' t'
  | Undefined, Undefined => True
  | _, _ => False
  end ->
  refines s t.
Proof.
  destruct (step s) as [s' | k vs hp rd | ] eqn:Hs;
    destruct (step t) as [t' | k' vs' hp' rd' | ] eqn:Ht; try contradiction.
  - intros Hequiv.
    exact (refines_step s s' t t' Hs Ht (equivalent_refines _ _ Hequiv)).
  - intros _. apply step_undefined_refines, Hs.
Qed.

(* ======================================================================== *)
(* The rewrite at a call                                                    *)
(* ======================================================================== *)

Lemma cut_return_refines (known : list (code_id * code)) (ap : apply_expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) :
  statics_know known statics ->
  refines (State (Apply ap) vars conts statics hp rd)
    (State (cut_return known ap) vars conts statics hp rd).
Proof.
  intros Hknow. unfold cut_return.
  destruct (ap_call_kind ap) as [called | | code_ids] eqn:Hkind;
    [ | apply refines_refl | apply refines_refl ].
  destruct (lookup called known) as [c | ] eqn:Hlookup;
    [ | apply refines_refl ].
  destruct (invalid_params c (length (unarize (ap_return_arity ap))))
    as [params | ];
    [ | apply refines_refl ].
  destruct (never_returns_b ap c) eqn:Hnever; [ | apply refines_refl ].
  unfold never_returns_b in Hnever.
  apply andb_true_iff in Hnever as [Hnever Hapart].
  apply andb_true_iff in Hnever as [Hnever Hargs].
  apply andb_true_iff in Hnever as [Hfree Htupled].
  apply negb_true_iff in Hfree, Htupled, Hapart.
  apply Nat.leb_le in Hargs.
  set (h := Handler params
              (make_handler_metadata (unarize (ap_return_arity ap))) Invalid).
  apply (refines_trans _
    (State (Apply ap) vars
       ((ap_continuation ap, Closure h vars conts []) :: conts) statics hp rd)).
  - apply steps_alike_refines. rewrite !step_apply.
    destruct (eval_simple vars (ap_callee ap)) as [callee | ]; [ | exact I ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | exact I ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | exact I ].
    rewrite Hkind.
    replace
      (resolve (ap_exn_continuation ap)
         ((ap_continuation ap, Closure h vars conts []) :: conts))
      with (resolve (ap_exn_continuation ap) conts)
      by (unfold resolve; simpl; rewrite Pos.eqb_sym, Hapart; reflexivity).
    exact (call_function_never_returns callee called c vs _ _ _ _ _ statics
             hp rd (Hknow called c Hlookup) Hfree Htupled Hargs).
  - apply equivalent_refines, equivalent_sym, step_equivalent,
      step_let_cont_non_recursive.
Qed.

(* ======================================================================== *)
(* What binds something static                                              *)
(* ======================================================================== *)

Lemma cut_return_binds_statics (known : list (code_id * code))
    (ap : apply_expr) :
  binds_statics (cut_return known ap) = false.
Proof.
  unfold cut_return.
  destruct (ap_call_kind ap) as [called | | code_ids];
    [ | reflexivity | reflexivity ].
  destruct (lookup called known) as [c | ]; [ | reflexivity ].
  destruct (invalid_params c (length (unarize (ap_return_arity ap))))
    as [params | ];
    [ | reflexivity ].
  destruct (never_returns_b ap c); reflexivity.
Qed.

Fixpoint cut_returns_binds_statics (e : expr) :
  ∀ known : list (code_id * code),
    binds_statics (cut_returns known e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known; simpl; try reflexivity.
  - apply cut_returns_binds_statics.
  - rewrite cut_returns_binds_statics. f_equal.
    induction bindings as [ | [id c | id | sym const] bindings IH]; simpl;
      congruence.
  - destruct (returns_to_invalid k h body); [ reflexivity | ].
    destruct h as [params metadata handler]. simpl.
    rewrite !cut_returns_binds_statics. reflexivity.
  - rewrite cut_returns_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite cut_returns_binds_statics, IH. reflexivity.
  - apply cut_return_binds_statics.
Qed.

Lemma cut_returns_handlers_bind_statics (known : list (code_id * code))
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, cut_returns_in_handler known h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite cut_returns_binds_statics, IH. reflexivity.
Qed.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* Down the term, with the static environment binding what is known: at
   once in a body, and from then on in a handler. The steps are those
   of FlambdaNormaliseProofs.inline_calls_refines_knowing. *)
Fixpoint cut_returns_refines_knowing (e : expr) :
  ∀ (known : list (code_id * code)) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    statics_know known statics ->
    refines (State e vars conts statics hp rd)
      (State (cut_returns known e) vars conts statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body
         | ap | ac | scrutinee arms | ];
      intros known vars conts statics hp rd Hknow; simpl;
      try apply refines_refl.
    + apply let_refines_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
      apply cut_returns_refines_knowing, Hknow.
    + assert (Hrefine :
        bindings_refine (statics_know known) bindings
          (map (cut_returns_in_binding known) bindings)).
      { induction bindings
          as [ | [id [params my_closure ret exn metadata code_body] | id
                 | sym const]
                 bindings IH];
          simpl.
        - split; apply Forall2_nil.
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ | assumption ]);
            apply Binding_code, Codes_related;
            first
              [ apply cut_returns_binds_statics
              | symmetry; apply cut_returns_binds_statics
              | intros vars' conts' statics' hp' rd' Hknow';
                apply cut_returns_refines_knowing, Hknow' ].
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ apply Binding_same | assumption ]). }
      remember (map (cut_returns_in_binding known) bindings) as bindings'
        eqn:Hdef.
      clear Hdef. destruct Hrefine as [Hforwards Hbackwards].
      apply (fun H =>
        eval_static_bindings_related _ vars _ _ H statics statics
          (statics_related_refl _ statics)) in Hforwards, Hbackwards.
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics) as [extended | ]
        eqn:Hbindings,
        (eval_static_bindings vars bindings' statics) as [extended' | ]
          eqn:Hbindings';
        try contradiction; [ | exact I ].
      apply (refines_trans _ (State body vars conts extended' hp rd)).
      * apply (related_states_refines (statics_know known)
          (statics_know_extend known));
          [ exact (statics_know_extend known _ _ Hknow
              (eval_static_bindings_extend vars _ _ _ Hbindings))
          | exact (statics_know_extend known _ _ Hknow
              (eval_static_bindings_extend vars _ _ _ Hbindings'))
          | apply cont_env_related_refl | apply cont_env_related_refl
          | exact Hforwards | exact Hbackwards ].
      * apply cut_returns_refines_knowing.
        exact (eval_static_bindings_know vars bindings' known statics
          extended' Hknow Hbindings').
    + destruct (returns_to_invalid k h body); [ apply refines_refl | ].
      destruct h as [params metadata handler]. simpl.
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k
              (Handler params metadata (cut_returns known handler)) body)
           vars conts statics hp rd)).
      * apply (let_cont_handler_refines_where (statics_know known)
          (statics_know_extend known)); [ exact Hknow | ].
        intros vs handler_vars conts' statics' hp' rd' Hknow' _ _.
        apply cut_returns_refines_knowing, Hknow'.
      * apply let_cont_non_recursive_refines_step,
          cut_returns_refines_knowing, Hknow.
    + apply (refines_trans _
        (State
           (Let_cont_recursive
              (map (fun '(k, h) => (k, cut_returns_in_handler known h))
                 handlers)
              body)
           vars conts statics hp rd)).
      * assert (Hhandlers :
          handlers_refine (statics_know known) vars handlers
            (map (fun '(k, h) => (k, cut_returns_in_handler known h))
               handlers)).
        { induction handlers
            as [ | [k [params metadata handler]] handlers IH];
            simpl.
          - split; apply Forall2_nil.
          - destruct IH as [Hforwards Hbackwards].
            split; (apply Forall2_cons; [ | assumption ]);
              (split; [ reflexivity | ]); apply Handler_related;
              intros vs handler_vars conts' statics' hp' rd' Hknow' _ _;
              apply cut_returns_refines_knowing, Hknow'. }
        destruct Hhandlers as [Hforwards Hbackwards].
        apply (let_cont_recursive_handlers_refine_where (statics_know known)
          (statics_know_extend known));
          try assumption.
        symmetry. apply cut_returns_handlers_bind_statics.
      * apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
        apply cut_returns_refines_knowing, Hknow.
    + apply cut_return_refines, Hknow.
Qed.

Theorem cut_returns_refines : cut_returns_refines_statement.
Proof.
  intros e vars conts statics hp rd. apply cut_returns_refines_knowing.
  intros id c Hlookup. discriminate Hlookup.
Qed.

(* ======================================================================== *)
(* The rewrite commutes with renaming                                       *)
(* ======================================================================== *)

(* The law of the rewrite is proved from the definitions by the tactic
   of FlambdaEquivarianceTactic.v, which states and proves by itself
   the laws of the functions that the rewrite is made of, the removal
   of unused continuations among them. It rewrites with its database
   in one pass over the goal, which is quicker and is enough here, and
   it is given three things.

   Computation is told to unfold [never_returns_b], which makes the
   proof quicker, and not to unfold [unarize]: unfolded on one side
   only, it leaves the two sides with two forms of one test of the
   parameters of the handler, which the tactic does not take for the
   same.

   At a [Let_static] the body is rewritten with the code of the
   bindings as the rewrite leaves them ([codes_of] of a mapped list).
   The code known there commutes with renaming where the rewrite of
   each binding does, which the induction gives: the law is stated
   with that premise and proved by hand, and the tactic is told how to
   show the premise ([eqv_premise]). *)
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

#[local] Arguments never_returns_b ap c /.
#[local] Arguments unarize : simpl never.

Lemma codes_of_cut_returns_renamed (r : renaming)
    (inner known : list (code_id * code)) (bindings : list static_binding) :
  (∀ binding : static_binding,
    In binding bindings ->
    cut_returns_in_binding (map_values (rename_code r) inner)
      (rename_static_binding r binding)
    = rename_static_binding r (cut_returns_in_binding inner binding)) ->
  codes_of
    (map
       (fun binding =>
          cut_returns_in_binding (map_values (rename_code r) inner)
            (rename_static_binding r binding))
       bindings)
    (map_values (rename_code r) known)
  = map_values (rename_code r)
      (codes_of (map (cut_returns_in_binding inner) bindings) known).
Proof.
  intros Hbindings. rewrite <- codes_of_equivariant, map_map. f_equal.
  apply map_ext_in. exact Hbindings.
Qed.

#[local] Ltac eqv_premise :=
  lazymatch goal with
  | |- forall x, In x ?l -> _ => eqv_enter l; intros; solve [ eqv_core ]
  | |- _ => intros; solve [ eqv_core ]
  end.
#[local] Hint Rewrite codes_of_cut_returns_renamed using eqv_premise
  : equivariance.

Theorem cut_returns_equivariant : cut_returns_equivariant_statement.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* Two examples                                                             *)
(* ======================================================================== *)

(* With no code known, as the validation applies the rewrite. *)
Lemma cut_returns_unit_equivariant (r : renaming) (e : expr) :
  cut_returns [] (rename_expr r e) = rename_expr r (cut_returns [] e).
Proof. exact (cut_returns_equivariant r [] e). Qed.

Theorem cut_returns_example : cut_returns_example_statement.
Proof. intros callee. reflexivity. Qed.

Theorem cut_returns_example_again : cut_returns_example_again_statement.
Proof. intros callee. reflexivity. Qed.

Theorem cut_returns_example_through : cut_returns_example_through_statement.
Proof. intros callee. reflexivity. Qed.
