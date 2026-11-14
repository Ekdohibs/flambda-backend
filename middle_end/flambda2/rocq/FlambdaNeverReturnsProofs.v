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
  FlambdaNeverReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* A call of code that does not mention its return continuation             *)
(* ======================================================================== *)

Theorem call_code_never_returns : call_code_never_returns_statement.
Proof.
  intros closure c vs args_arity return_arity return_to return_to' exn_to
    statics hp Hfree.
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
       (remove_unused_conts_equivalent (code_body c) _ _ _ _)).
  refine
    (equivalent_trans _ _ _ _
       (equivalent_sym _ _
          (remove_unused_conts_equivalent (code_body c) _ _ _ _))).
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
    (hp : heap) :
  cont_free_b (code_return_continuation c) (code_body c) = false ->
  match
    call_code closure c vs args_arity return_arity return_to exn_to
      statics hp,
    call_code closure c vs args_arity return_arity return_to' exn_to
      statics hp
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
    (hp : heap) :
  lookup called (static_code statics) = Some c ->
  cont_free_b (code_return_continuation c)
    (remove_unused_conts (code_body c))
  = false ->
  code_is_tupled c = false ->
  length (code_params_arity c) <= length args_arity ->
  match
    call_function callee (Direct called) vs args_arity return_arity
      return_to exn_to statics hp,
    call_function callee (Direct called) vs args_arity return_arity
      return_to' exn_to statics hp
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
  destruct (step s) as [s' | k vs hp | ] eqn:Hs;
    destruct (step t) as [t' | k' vs' hp' | ] eqn:Ht; try contradiction.
  - intros Hequiv.
    exact (refines_step s s' t t' Hs Ht (equivalent_refines _ _ Hequiv)).
  - intros _. apply step_undefined_refines, Hs.
Qed.

(* ======================================================================== *)
(* The rewrite at a call                                                    *)
(* ======================================================================== *)

Lemma cut_return_refines (known : list (code_id * code)) (ap : apply_expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap) :
  statics_know known statics ->
  refines (State (Apply ap) vars conts statics hp)
    (State (cut_return known ap) vars conts statics hp).
Proof.
  intros Hknow. unfold cut_return.
  destruct (ap_call_kind ap) as [called | ] eqn:Hkind;
    [ | apply refines_refl ].
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
       ((ap_continuation ap, Closure h vars conts []) :: conts) statics hp)).
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
             hp (Hknow called c Hlookup) Hfree Htupled Hargs).
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
  destruct (ap_call_kind ap) as [called | ]; [ | reflexivity ].
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
    (statics : static_env) (hp : heap),
    statics_know known statics ->
    refines (State e vars conts statics hp)
      (State (cut_returns known e) vars conts statics hp).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body
         | ap | ac | scrutinee arms | ];
      intros known vars conts statics hp Hknow; simpl;
      try apply refines_refl.
    + apply let_refines_step; [ reflexivity | intros v _ _ ].
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
              | intros vars' conts' statics' hp' Hknow';
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
      apply (refines_trans _ (State body vars conts extended' hp)).
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
           vars conts statics hp)).
      * apply (let_cont_handler_refines_where (statics_know known)
          (statics_know_extend known)); [ exact Hknow | ].
        intros vs handler_vars conts' statics' hp' Hknow' _ _.
        apply cut_returns_refines_knowing, Hknow'.
      * apply let_cont_non_recursive_refines_step,
          cut_returns_refines_knowing, Hknow.
    + apply (refines_trans _
        (State
           (Let_cont_recursive
              (map (fun '(k, h) => (k, cut_returns_in_handler known h))
                 handlers)
              body)
           vars conts statics hp)).
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
              intros vs handler_vars conts' statics' hp' Hknow' _ _;
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
  intros e vars conts statics hp. apply cut_returns_refines_knowing.
  intros id c Hlookup. discriminate Hlookup.
Qed.

(* ======================================================================== *)
(* The rewrite commutes with renaming                                       *)
(* ======================================================================== *)

(* As FlambdaValidatorEquivarianceProofs.remove_unused_conts_equivariant,
   which is proved after this file. *)
Fixpoint remove_unused_conts_renamed (r : renaming) (e : expr) :
  remove_unused_conts (rename_expr r e)
  = rename_expr r (remove_unused_conts e)
with remove_unused_conts_in_handler_renamed (r : renaming)
    (h : continuation_handler) :
  remove_unused_conts_in_handler (rename_handler r h)
  = rename_handler r (remove_unused_conts_in_handler h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite remove_unused_conts_renamed. reflexivity.
    + rewrite remove_unused_conts_renamed. reflexivity.
    + rewrite remove_unused_conts_renamed, cont_free_b_equivariant_aux,
        remove_unused_conts_in_handler_renamed.
      destruct (cont_free_b k (remove_unused_conts body)); reflexivity.
    + rewrite remove_unused_conts_renamed. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite remove_unused_conts_renamed. reflexivity.
Qed.

Lemma never_returns_b_equivariant (r : renaming) (ap : apply_expr)
    (c : code) :
  never_returns_b (rename_apply r ap) (rename_code r c)
  = never_returns_b ap c.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  unfold never_returns_b. simpl.
  rewrite remove_unused_conts_renamed, cont_free_b_equivariant_aux,
    rename_continuation_eqb.
  reflexivity.
Qed.

Lemma invalid_params_equivariant (r : renaming) (c : code) (count : nat) :
  invalid_params (rename_code r c) count
  = option_map (map (rename_variable r)) (invalid_params c count).
Proof.
  destruct c as [params my_closure ret exn metadata body].
  unfold invalid_params. simpl. rewrite length_map.
  destruct (Nat.leb count (S (length params))); [ | reflexivity ].
  simpl.
  change (rename_variable r my_closure :: map (rename_variable r) params)
    with (map (rename_variable r) (my_closure :: params)).
  rewrite firstn_map. reflexivity.
Qed.

Lemma cut_return_equivariant (r : renaming) (known : list (code_id * code))
    (ap : apply_expr) :
  cut_return (map_values (rename_code r) known) (rename_apply r ap)
  = rename_expr r (cut_return known ap).
Proof.
  unfold cut_return.
  replace (ap_call_kind (rename_apply r ap)) with (ap_call_kind ap)
    by reflexivity.
  destruct (ap_call_kind ap) as [called | ]; [ | reflexivity ].
  rewrite lookup_map_values.
  destruct (lookup called known) as [c | ]; cbn [option_map];
    [ | reflexivity ].
  replace (ap_return_arity (rename_apply r ap)) with (ap_return_arity ap)
    by reflexivity.
  rewrite invalid_params_equivariant, never_returns_b_equivariant.
  destruct (invalid_params c (length (unarize (ap_return_arity ap))))
    as [params | ];
    cbn [option_map]; [ | reflexivity ].
  destruct (never_returns_b ap c); reflexivity.
Qed.

Lemma returns_to_invalid_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) (body : expr) :
  returns_to_invalid (rename_continuation r k) (rename_handler r h)
    (rename_expr r body)
  = returns_to_invalid k h body.
Proof.
  destruct h as [params metadata handler].
  destruct handler; simpl; try reflexivity.
  destruct body; simpl; try reflexivity.
  apply rename_continuation_eqb.
Qed.

Fixpoint cut_returns_equivariant_aux (r : renaming) (e : expr) :
  ∀ known : list (code_id * code),
    cut_returns (map_values (rename_code r) known) (rename_expr r e)
    = rename_expr r (cut_returns known e)
with cut_returns_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ known : list (code_id * code),
    cut_returns_in_handler (map_values (rename_code r) known)
      (rename_handler r h)
    = rename_handler r (cut_returns_in_handler known h)
with cut_returns_in_binding_equivariant (r : renaming)
    (binding : static_binding) :
  ∀ known : list (code_id * code),
    cut_returns_in_binding (map_values (rename_code r) known)
      (rename_static_binding r binding)
    = rename_static_binding r (cut_returns_in_binding known binding)
with cut_returns_in_code_equivariant (r : renaming) (c : code) :
  ∀ known : list (code_id * code),
    cut_returns_in_code (map_values (rename_code r) known) (rename_code r c)
    = rename_code r (cut_returns_in_code known c).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known;
      [ | | | | exact (cut_return_equivariant r known ap) | | | ];
      try reflexivity.
    + (* Let *)
      simpl. rewrite cut_returns_equivariant_aux. reflexivity.
    + (* Let_static: the code known in the body is that of the group,
         rewritten *)
      simpl.
      assert (Hbindings :
        map (cut_returns_in_binding (map_values (rename_code r) known))
          (map (rename_static_binding r) bindings)
        = map (rename_static_binding r)
            (map (cut_returns_in_binding known) bindings)).
      { induction bindings as [ | binding bindings IH]; simpl;
          [ reflexivity | ].
        rewrite (cut_returns_in_binding_equivariant r binding), IH.
        reflexivity. }
      rewrite Hbindings, codes_of_equivariant, cut_returns_equivariant_aux.
      reflexivity.
    + (* Let_cont_non_recursive *)
      change (rename_expr r (Let_cont_non_recursive k h body))
        with (Let_cont_non_recursive (rename_continuation r k)
                (rename_handler r h) (rename_expr r body)).
      cbn [cut_returns]. rewrite returns_to_invalid_equivariant.
      destruct (returns_to_invalid k h body); [ reflexivity | ].
      rewrite cut_returns_in_handler_equivariant,
        cut_returns_equivariant_aux.
      reflexivity.
    + (* Let_cont_recursive *)
      simpl. rewrite cut_returns_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (cut_returns_in_handler_equivariant r h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros known. simpl.
    rewrite cut_returns_equivariant_aux. reflexivity.
  - destruct binding as [id c | id | sym const]; intros known; simpl;
      try reflexivity.
    rewrite cut_returns_in_code_equivariant. reflexivity.
  - destruct c as [params my_closure ret exn metadata body]. intros known.
    simpl.
    rewrite cut_returns_equivariant_aux. reflexivity.
Qed.

Theorem cut_returns_equivariant : cut_returns_equivariant_statement.
Proof. intros r known e. apply cut_returns_equivariant_aux. Qed.

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
