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


(* Proofs of the statements of FlambdaClosureCallsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextProofs FlambdaBindersSpec
  FlambdaNominalSpec FlambdaNominalProofs FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaClosureCallsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What it means for what is known to hold                                  *)
(* ======================================================================== *)

(* A variable known to be a closure is bound to a closure at the
   function slot known, of a set with the code known. *)
Definition closures_hold (known : made_closures) (vars : env) : Prop :=
  ∀ (x : variable) (slot : function_slot)
    (function_decls : list (function_slot * code_id)),
    lookup x known = Some (slot, function_decls) ->
    ∃ value_slots : list (value_slot * value),
      lookup x vars
      = Some (Value_closure_of_set slot function_decls value_slots).

(* ======================================================================== *)
(* What is known stays so                                                   *)
(* ======================================================================== *)

(* Binding a variable, once what is known of it is forgotten. *)
Lemma closures_hold_unbind (known : made_closures) (vars : env)
    (x : variable) (v : value) :
  closures_hold known vars ->
  closures_hold (unbind x known) ((x, v) :: vars).
Proof.
  intros Hhold y slot function_decls Hy. rewrite lookup_unbind in Hy.
  cbn [lookup].
  destruct (Pos.eqb y x); [ discriminate | ].
  exact (Hhold y slot function_decls Hy).
Qed.

Lemma closures_hold_bind_params (params : list variable) :
  ∀ (known : made_closures) (vs : list value) (vars handler_vars : env),
    closures_hold known vars ->
    bind_params params vs vars = Some handler_vars ->
    closures_hold (unbind_params params known) handler_vars.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hhold; simpl; try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, closures_hold_unbind, Hhold.
Qed.

(* ======================================================================== *)
(* Under a [Let]                                                            *)
(* ======================================================================== *)

(* What a named expression is said to make, it makes. *)
Lemma closure_made_sound (known : made_closures) (statics : static_env)
    (vars : env) (d : named) (slot : function_slot)
    (function_decls : list (function_slot * code_id)) (v : value) :
  closures_hold known vars ->
  closure_made known d = Some (slot, function_decls) ->
  eval_named statics vars d = Some v ->
  ∃ value_slots : list (value_slot * value),
    v = Value_closure_of_set slot function_decls value_slots.
Proof.
  intros Hhold.
  destruct d as [s | p args | code slots | fslot decls slots | vslot closure
    | pfrom vslot closure | move_from move_to closure];
    cbn [closure_made]; try discriminate.
  - intros [= <- <-]. cbn [eval_named].
    destruct (lookup fslot decls); [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ]; [ | discriminate ].
    simpl. intros [= <-]. exists vs. reflexivity.
  - destruct closure as [y | sym | c]; try discriminate.
    destruct (lookup y known) as [[at_slot decls] | ] eqn:Hy;
      [ | discriminate ].
    destruct (Pos.eqb_spec at_slot move_from) as [-> | _]; [ | discriminate ].
    intros [= <- <-].
    destruct (Hhold y move_from decls Hy) as (vs & Hlookup).
    cbn [eval_named eval_simple]. rewrite Hlookup. simpl.
    rewrite Pos.eqb_refl.
    destruct (lookup move_from decls); [ | discriminate ].
    destruct (lookup move_to decls); [ | discriminate ].
    simpl. intros [= <-]. exists vs. reflexivity.
Qed.

Lemma closures_hold_under_let (known : made_closures) (x : variable)
    (d : named) (statics : static_env) (vars : env) (v : value) :
  closures_hold known vars ->
  eval_named statics vars d = Some v ->
  closures_hold (made_under_let known x d) ((x, v) :: vars).
Proof.
  intros Hhold Hd. unfold made_under_let.
  pose proof (closures_hold_unbind known vars x v Hhold) as Hforgot.
  destruct (closure_made known d) as [[slot function_decls] | ] eqn:Hmade;
    [ | exact Hforgot ].
  destruct (closure_made_sound known statics vars d slot function_decls v
    Hhold Hmade Hd) as (vs & ->).
  intros y slot' function_decls' Hy.
  specialize (Hforgot y slot' function_decls').
  cbn [lookup] in Hy, Hforgot |- *.
  destruct (Pos.eqb y x); [ | exact (Hforgot Hy) ].
  injection Hy as <- <-. exists vs. reflexivity.
Qed.

(* ======================================================================== *)
(* A call                                                                   *)
(* ======================================================================== *)

(* The code that the callee is known to have is the code of the closure
   that the callee is. *)
Lemma made_callee_code_sound (known : made_closures) (statics : static_env)
    (vars : env) (callee : simple) (code : code_id) :
  closures_hold known vars ->
  made_callee_code known callee = Some code ->
  ∃ (v : value) (value_slots : list (value_slot * value)),
    eval_simple vars callee = Some v
    ∧ closure_of statics v = Some (code, value_slots).
Proof.
  intros Hhold.
  destruct callee as [x | sym | c]; cbn [made_callee_code];
    try discriminate.
  destruct (lookup x known) as [[slot function_decls] | ] eqn:Hx;
    [ | discriminate ].
  intros Hcode. destruct (Hhold x slot function_decls Hx) as (vs & Hlookup).
  exists (Value_closure_of_set slot function_decls vs), vs.
  split; [ exact Hlookup | ]. cbn [closure_of]. rewrite Hcode. reflexivity.
Qed.

Lemma call_made_closure_holds (known : made_closures) (statics : static_env)
    (hp : heap) (vars : env) (conts : cont_env) (ap : apply_expr) :
  closures_hold known vars ->
  step (State (Apply (call_made_closure known ap)) vars conts statics hp)
  = step (State (Apply ap) vars conts statics hp).
Proof.
  intros Hhold. unfold call_made_closure.
  destruct (ap_call_kind ap) as [named | ] eqn:Hkind; [ reflexivity | ].
  destruct (made_callee_code known (ap_callee ap)) as [code | ] eqn:Hcode;
    [ | reflexivity ].
  destruct (made_callee_code_sound known statics vars (ap_callee ap) code
    Hhold Hcode) as (v & slots & Hcallee & Hclosure).
  rewrite !step_apply. unfold exn_extra_simples.
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity].
  rewrite Hkind, Hcallee.
  destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | reflexivity ].
  destruct (eval_simples vars (map fst (ap_exn_extra_args ap)))
    as [extra | ];
    [ | reflexivity ].
  unfold call_function.
  rewrite !(shape_of_call_closure statics v _ vs _ _
              (closure_not_partial statics v _ Hclosure)),
    (shape_of_closure_call_kind statics v (Direct code) vs _ _ code slots
       Hclosure (or_intror eq_refl)).
  reflexivity.
Qed.

Theorem call_made_closure_sound : call_made_closure_sound_statement.
Proof.
  intros known statics hp vars conts ap Hhold.
  apply call_made_closure_holds. exact Hhold.
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The pass changes no binding, so what binds something static is as it
   was. *)
Fixpoint call_closures_knowing_binds_statics (e : expr) :
  ∀ known : made_closures,
    binds_statics (call_closures_knowing known e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known; simpl; try reflexivity.
  - apply call_closures_knowing_binds_statics.
  - rewrite call_closures_knowing_binds_statics. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite !call_closures_knowing_binds_statics. reflexivity.
  - rewrite call_closures_knowing_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite call_closures_knowing_binds_statics, IH. reflexivity.
Qed.

Lemma call_closures_handlers_bind_statics (known : made_closures)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, call_closures_in_handler known h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite call_closures_knowing_binds_statics, IH. reflexivity.
Qed.

(* At a [Let], what is known under it holds once it has run; a handler
   runs under the variables of its definition with its parameters
   bound, where what is known less the parameters holds; at a call, the
   call made direct steps as the call does. *)
Fixpoint call_closures_knowing_refines (e : expr) :
  ∀ (known : made_closures) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap),
    closures_hold known vars ->
    refines (State e vars conts statics hp)
      (State (call_closures_knowing known e) vars conts statics hp).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap
       | ac | scrutinee arms | ];
    intros known vars conts statics hp Hhold; cbn [call_closures_knowing].
  - (* Let *)
    destruct (eval_named statics vars d) as [v | ] eqn:Hd;
      [ destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind | ].
    + eapply refines_step;
        [ rewrite step_let, Hd, Hkind; reflexivity
        | rewrite step_let, Hd, Hkind; reflexivity
        | ].
      apply call_closures_knowing_refines.
      exact (closures_hold_under_let known x d statics vars v Hhold Hd).
    + apply step_undefined_refines. rewrite step_let, Hd, Hkind.
      reflexivity.
    + apply step_undefined_refines. rewrite step_let, Hd. reflexivity.
  - (* Let_static *)
    apply let_static_refines_step.
    destruct (eval_static_bindings vars bindings statics) as [extended | ];
      [ | exact I ].
    apply call_closures_knowing_refines, Hhold.
  - (* Let_cont_non_recursive *)
    destruct h as [params metadata handler]. cbn [call_closures_in_handler].
    apply (refines_trans _
      (State
         (Let_cont_non_recursive k
            (Handler params metadata
               (call_closures_knowing (unbind_params params known) handler))
            body)
         vars conts statics hp)).
    + apply let_cont_handler_refines_at.
      intros vs handler_vars conts' statics' hp' _ Hbind.
      apply call_closures_knowing_refines.
      exact (closures_hold_bind_params params known vs vars handler_vars
        Hhold Hbind).
    + apply let_cont_non_recursive_refines_step,
        call_closures_knowing_refines, Hhold.
  - (* Let_cont_recursive *)
    assert (Hhandlers :
      handlers_refine always vars handlers
        (map (fun '(k, h) => (k, call_closures_in_handler known h))
           handlers)).
    { induction handlers as [ | [k [params metadata handler]] handlers IH];
        simpl.
      - split; apply Forall2_nil.
      - destruct IH as [Hforwards Hbackwards].
        split; (apply Forall2_cons; [ | assumption ]);
          (split; [ reflexivity | ]); apply Handler_related;
          intros vs handler_vars conts' statics' hp' _ _ Hbind;
          apply call_closures_knowing_refines;
          exact (closures_hold_bind_params params known vs vars
            handler_vars Hhold Hbind). }
    destruct Hhandlers as [Hforwards Hbackwards].
    apply (refines_trans _
      (State
         (Let_cont_recursive
            (map (fun '(k, h) => (k, call_closures_in_handler known h))
               handlers)
            body)
         vars conts statics hp)).
    + apply (let_cont_recursive_handlers_refine_where always always_extend);
        [ exact I
        | symmetry; apply call_closures_handlers_bind_statics
        | exact Hforwards | exact Hbackwards ].
    + apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply call_closures_knowing_refines, Hhold.
  - (* Apply *)
    apply equivalent_refines, step_eq_equivalent. symmetry.
    apply call_made_closure_holds, Hhold.
  - apply refines_refl.
  - apply refines_refl.
  - apply refines_refl.
Qed.

Theorem call_made_closures_refines : call_made_closures_refines_statement.
Proof.
  intros e vars conts statics hp. apply call_closures_knowing_refines.
  intros x slot function_decls Hx. discriminate Hx.
Qed.

(* ======================================================================== *)
(* The names of the result                                                  *)
(* ======================================================================== *)

Lemma call_made_closure_free_variables (known : made_closures)
    (ap : apply_expr) :
  free_variables_of_apply (call_made_closure known ap)
  = free_variables_of_apply ap.
Proof.
  unfold call_made_closure.
  destruct (ap_call_kind ap); [ reflexivity | ].
  destruct (made_callee_code known (ap_callee ap)); reflexivity.
Qed.

Lemma call_made_closure_continuations (known : made_closures)
    (ap : apply_expr) :
  ap_continuation (call_made_closure known ap) = ap_continuation ap
  ∧ ap_exn_continuation (call_made_closure known ap) = ap_exn_continuation ap.
Proof.
  unfold call_made_closure.
  destruct (ap_call_kind ap); [ split; reflexivity | ].
  destruct (made_callee_code known (ap_callee ap)); split; reflexivity.
Qed.

Fixpoint call_closures_knowing_free_variables (e : expr) :
  ∀ known : made_closures,
    free_variables_of_expr (call_closures_knowing known e)
    = free_variables_of_expr e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known; cbn [call_closures_knowing free_variables_of_expr];
    try reflexivity.
  - rewrite call_closures_knowing_free_variables. reflexivity.
  - rewrite call_closures_knowing_free_variables. reflexivity.
  - destruct h as [params metadata handler].
    cbn [call_closures_in_handler free_variables_of_handler].
    rewrite !call_closures_knowing_free_variables. reflexivity.
  - rewrite call_closures_knowing_free_variables. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite call_closures_knowing_free_variables, IH. reflexivity.
  - apply call_made_closure_free_variables.
Qed.

Lemma call_closures_handlers_names (known : made_closures)
    (handlers : list (continuation * continuation_handler)) :
  map fst
    (map (fun '(k, h) => (k, call_closures_in_handler known h)) handlers)
  = map fst handlers.
Proof.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Fixpoint call_closures_knowing_free_continuations (e : expr) :
  ∀ known : made_closures,
    free_continuations_of_expr (call_closures_knowing known e)
    = free_continuations_of_expr e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known; cbn [call_closures_knowing free_continuations_of_expr];
    try reflexivity.
  - apply call_closures_knowing_free_continuations.
  - rewrite call_closures_knowing_free_continuations. reflexivity.
  - destruct h as [params metadata handler].
    cbn [call_closures_in_handler free_continuations_of_handler].
    rewrite !call_closures_knowing_free_continuations. reflexivity.
  - rewrite call_closures_knowing_free_continuations,
      call_closures_handlers_names.
    f_equal. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite call_closures_knowing_free_continuations, IH. reflexivity.
  - destruct (call_made_closure_continuations known ap) as [-> ->].
    reflexivity.
Qed.

Fixpoint call_closures_knowing_code_closed (e : expr) :
  ∀ known : made_closures,
    code_closed (call_closures_knowing known e) = code_closed e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros known; cbn [call_closures_knowing code_closed];
    try reflexivity.
  - apply call_closures_knowing_code_closed.
  - rewrite call_closures_knowing_code_closed. reflexivity.
  - destruct h as [params metadata handler].
    cbn [call_closures_in_handler handler_code_closed].
    rewrite !call_closures_knowing_code_closed. reflexivity.
  - rewrite call_closures_knowing_code_closed. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite call_closures_knowing_code_closed, IH. reflexivity.
Qed.

Theorem call_closures_knowing_names : call_closures_knowing_names_statement.
Proof.
  intros known e. split; [ | split ].
  - apply call_closures_knowing_free_variables.
  - apply call_closures_knowing_free_continuations.
  - apply call_closures_knowing_code_closed.
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* A renaming is injective, so a variable is found where its image
   is. *)
Lemma lookup_made_renamed (r : renaming) (x : variable)
    (known : made_closures) :
  lookup (rename_variable r x) (rename_made_closures r known)
  = lookup x known.
Proof.
  unfold rename_made_closures.
  induction known as [ | [y closure] known IH]; simpl; [ reflexivity | ].
  rewrite rename_variable_eqb. destruct (Pos.eqb x y); [ reflexivity | ].
  exact IH.
Qed.

Lemma unbind_made_renamed (r : renaming) (x : variable)
    (known : made_closures) :
  unbind (rename_variable r x) (rename_made_closures r known)
  = rename_made_closures r (unbind x known).
Proof.
  unfold unbind, rename_made_closures.
  induction known as [ | [y closure] known IH]; simpl; [ reflexivity | ].
  rewrite rename_variable_eqb, IH. destruct (Pos.eqb x y); reflexivity.
Qed.

Lemma unbind_params_renamed (r : renaming) (params : list variable) :
  ∀ known : made_closures,
    unbind_params (map (rename_variable r) params)
      (rename_made_closures r known)
    = rename_made_closures r (unbind_params params known).
Proof.
  induction params as [ | param params IH]; intros known; simpl;
    [ reflexivity | ].
  rewrite unbind_made_renamed. apply IH.
Qed.

Lemma closure_made_renamed (r : renaming) (known : made_closures)
    (d : named) :
  closure_made (rename_made_closures r known) (rename_named r d)
  = closure_made known d.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | vslot closure
    | pfrom vslot closure | move_from move_to closure];
    try reflexivity.
  destruct closure as [y | sym | c]; try reflexivity.
  cbn [rename_named rename_simple closure_made].
  rewrite lookup_made_renamed. reflexivity.
Qed.

Lemma made_under_let_renamed (r : renaming) (known : made_closures)
    (x : variable) (d : named) :
  made_under_let (rename_made_closures r known) (rename_variable r x)
    (rename_named r d)
  = rename_made_closures r (made_under_let known x d).
Proof.
  unfold made_under_let.
  rewrite closure_made_renamed, unbind_made_renamed.
  destruct (closure_made known d); reflexivity.
Qed.

Lemma call_made_closure_renamed (r : renaming) (known : made_closures)
    (ap : apply_expr) :
  call_made_closure (rename_made_closures r known) (rename_apply r ap)
  = rename_apply r (call_made_closure known ap).
Proof.
  unfold call_made_closure.
  cbn [rename_apply ap_call_kind ap_callee].
  destruct (ap_call_kind ap); [ reflexivity | ].
  destruct (ap_callee ap) as [x | sym | c]; try reflexivity.
  cbn [rename_simple made_callee_code]. rewrite lookup_made_renamed.
  destruct (lookup x known) as [[slot function_decls] | ]; [ | reflexivity ].
  destruct (lookup slot function_decls); reflexivity.
Qed.

Fixpoint call_closures_knowing_renamed (r : renaming) (e : expr) :
  ∀ known : made_closures,
    call_closures_knowing (rename_made_closures r known) (rename_expr r e)
    = rename_expr r (call_closures_knowing known e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    intros known; try reflexivity;
    cbn [rename_expr call_closures_knowing].
  - rewrite made_under_let_renamed, call_closures_knowing_renamed.
    reflexivity.
  - rewrite call_closures_knowing_renamed. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler call_closures_in_handler].
    rewrite unbind_params_renamed, !call_closures_knowing_renamed.
    reflexivity.
  - rewrite call_closures_knowing_renamed. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite unbind_params_renamed, call_closures_knowing_renamed, IH.
    reflexivity.
  - rewrite call_made_closure_renamed. reflexivity.
Qed.

Theorem call_closures_knowing_equivariant :
  call_closures_knowing_equivariant_statement.
Proof.
  intros r known e. symmetry. apply call_closures_knowing_renamed.
Qed.

Theorem call_made_closures_equivariant :
  call_made_closures_equivariant_statement.
Proof.
  intros r e. symmetry. exact (call_closures_knowing_renamed r e []).
Qed.
