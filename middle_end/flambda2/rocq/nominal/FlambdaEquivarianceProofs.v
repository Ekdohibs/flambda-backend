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

(* Proofs of the statements of FlambdaEquivarianceSpec.v. Nothing here
   needs review.

   The laws of the action on what a state holds are proved as those of
   [rename_expr] are in FlambdaAlphaProofs.v. [step_equivariant] goes by
   cases on the construct, through one lemma for each function that
   [step] is made of; each says that the function commutes with the
   action. The two rewrites are proved in the same way: one lemma for
   each function a rewrite is made of. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveSpec FlambdaPrimitiveProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaAlphaSpec FlambdaAlphaProofs FlambdaApplySpec
  FlambdaTupledSpec FlambdaNormaliseSpec FlambdaEquivarianceSpec.
Import ListNotations.

(* ======================================================================== *)
(* Evaluation under a renamed environment                                   *)
(* ======================================================================== *)

(* What reads variables gives the same under environments that agree
   along the renaming on the variables read. *)
Lemma eval_simple_renamed (r : renaming) (s : simple) (vars vars' : env) :
  (∀ x : variable,
    var_free_in_simple x s ->
    lookup (rename_variable r x) vars' = lookup x vars) ->
  eval_simple vars' (rename_simple r s) = eval_simple vars s.
Proof.
  intros Hvars. destruct s as [x | sym | c]; simpl; try reflexivity.
  apply Hvars. reflexivity.
Qed.

Lemma eval_simples_renamed (r : renaming) (args : list simple)
    (vars vars' : env) :
  (∀ x : variable,
    var_free_in_simples x args ->
    lookup (rename_variable r x) vars' = lookup x vars) ->
  eval_simples vars' (map (rename_simple r) args) = eval_simples vars args.
Proof.
  induction args as [ | s args IH]; intros Hvars; simpl; [ reflexivity | ].
  rewrite (eval_simple_renamed r s vars vars'), IH; [ reflexivity | | ];
    intros x Hx; apply Hvars; [ apply Exists_cons_tl | apply Exists_cons_hd ];
    exact Hx.
Qed.

Lemma eval_value_slots_renamed (r : renaming)
    (slots : list (value_slot * simple)) (vars vars' : env) :
  (∀ x : variable,
    Exists (fun slot => var_free_in_simple x (snd slot)) slots ->
    lookup (rename_variable r x) vars' = lookup x vars) ->
  eval_value_slots vars'
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = eval_value_slots vars slots.
Proof.
  induction slots as [ | [slot s] slots IH]; intros Hvars; simpl;
    [ reflexivity | ].
  rewrite (eval_simple_renamed r s vars vars'), IH; [ reflexivity | | ];
    intros x Hx; apply Hvars; [ apply Exists_cons_tl | apply Exists_cons_hd ];
    exact Hx.
Qed.

Lemma eval_named_renamed (r : renaming) (d : named) (vars vars' : env)
    (statics : static_env) :
  (∀ x : variable,
    var_free_in_named x d ->
    lookup (rename_variable r x) vars' = lookup x vars) ->
  eval_named statics vars' (rename_named r d) = eval_named statics vars d.
Proof.
  intros Hvars.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl.
  - apply eval_simple_renamed, Hvars.
  - rewrite (eval_simples_renamed r args vars vars' Hvars). reflexivity.
  - rewrite (eval_value_slots_renamed r slots vars vars' Hvars). reflexivity.
  - rewrite (eval_value_slots_renamed r slots vars vars' Hvars). reflexivity.
  - rewrite (eval_simple_renamed r closure vars vars' Hvars). reflexivity.
  - rewrite (eval_simple_renamed r closure vars vars' Hvars). reflexivity.
  - rewrite (eval_simple_renamed r closure vars vars' Hvars). reflexivity.
Qed.

Lemma eval_static_const_renamed (r : renaming) (const : static_const)
    (vars vars' : env) :
  (∀ x : variable,
    var_free_in_static_const x const ->
    lookup (rename_variable r x) vars' = lookup x vars) ->
  eval_static_const vars' (rename_static_const r const)
  = eval_static_const vars const.
Proof.
  intros Hvars.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | s | bytes
        | elements]; simpl.
  - rewrite (eval_value_slots_renamed r slots vars vars' Hvars). reflexivity.
  - rewrite (eval_value_slots_renamed r slots vars vars' Hvars). reflexivity.
  - rewrite (eval_simples_renamed r fields vars vars' Hvars). reflexivity.
  - rewrite (eval_simple_renamed r s vars vars' Hvars). reflexivity.
  - reflexivity.
  - rewrite (eval_simples_renamed r elements vars vars' Hvars). reflexivity.
Qed.


Lemma find_arm_renamed (r : renaming) (i : Z) (arms : list (Z * apply_cont)) :
  find_arm i (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = option_map (rename_apply_cont r) (find_arm i arms).
Proof.
  induction arms as [ | [discriminant ac] arms IH]; simpl; [ reflexivity | ].
  destruct (Z.eqb i discriminant); [ reflexivity | exact IH ].
Qed.

(* ======================================================================== *)
(* A renaming and its inverse                                               *)
(* ======================================================================== *)

(* A renaming after its inverse, and its inverse after it, are the
   identity as functions. *)
Lemma compose_inverse_l_variable (r : renaming) (x : variable) :
  rename_variable (compose_renaming (inverse_renaming r) r) x
  = rename_variable identity_renaming x.
Proof. rewrite rename_variable_compose. apply rename_variable_inverse_l. Qed.

Lemma compose_inverse_l_continuation (r : renaming) (k : continuation) :
  rename_continuation (compose_renaming (inverse_renaming r) r) k
  = rename_continuation identity_renaming k.
Proof.
  rewrite rename_continuation_compose. apply rename_continuation_inverse_l.
Qed.

Lemma compose_inverse_r_variable (r : renaming) (x : variable) :
  rename_variable (compose_renaming r (inverse_renaming r)) x
  = rename_variable identity_renaming x.
Proof. rewrite rename_variable_compose. apply rename_variable_inverse_r. Qed.

Lemma compose_inverse_r_continuation (r : renaming) (k : continuation) :
  rename_continuation (compose_renaming r (inverse_renaming r)) k
  = rename_continuation identity_renaming k.
Proof.
  rewrite rename_continuation_compose. apply rename_continuation_inverse_r.
Qed.
Lemma rename_handlers_bind_statics (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics (rename_handlers r handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics, rename_handlers.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite rename_binds_statics, IH. reflexivity.
Qed.

(* ======================================================================== *)
(* The action is that of a group, on what a state holds                     *)
(* ======================================================================== *)

Lemma rename_env_id (vars : env) : rename_env identity_renaming vars = vars.
Proof.
  induction vars as [ | [x v] vars IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma rename_handlers_id
    (handlers : list (continuation * continuation_handler)) :
  rename_handlers identity_renaming handlers = handlers.
Proof.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite rename_handler_id, IH. reflexivity.
Qed.

Fixpoint rename_cont_value_id (c : cont_value) :
  rename_cont_value identity_renaming c = c.
Proof.
  destruct c
    as [h vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    simpl;
    [ | reflexivity | rewrite (rename_cont_value_id c); reflexivity | ].
  - rewrite rename_handler_id, rename_env_id, rename_handlers_id.
    f_equal.
    induction conts as [ | [k c] conts IH]; simpl; [ reflexivity | ].
    rewrite (rename_cont_value_id c), IH. reflexivity.
  - rewrite (rename_cont_value_id return_to), (rename_cont_value_id exn_to).
    reflexivity.
Qed.

Lemma rename_cont_env_id (conts : cont_env) :
  rename_cont_env identity_renaming conts = conts.
Proof.
  induction conts as [ | [k c] conts IH]; simpl; [ reflexivity | ].
  rewrite rename_cont_value_id, IH. reflexivity.
Qed.

Lemma rename_static_env_id (statics : static_env) :
  rename_static_env identity_renaming statics = statics.
Proof.
  destruct statics as [codes values]. unfold rename_static_env.
  simpl. f_equal.
  induction codes as [ | [id c] codes IH]; simpl; [ reflexivity | ].
  rewrite rename_code_id, IH. reflexivity.
Qed.

Section Extensional_state.

Variables r r' : renaming.

Hypothesis same_variables :
  ∀ x : variable, rename_variable r x = rename_variable r' x.

Hypothesis same_continuations :
  ∀ k : continuation, rename_continuation r k = rename_continuation r' k.

Lemma rename_env_ext (vars : env) : rename_env r vars = rename_env r' vars.
Proof.
  induction vars as [ | [x v] vars IH]; simpl; [ reflexivity | ].
  rewrite same_variables, IH. reflexivity.
Qed.

Lemma rename_handlers_ext
    (handlers : list (continuation * continuation_handler)) :
  rename_handlers r handlers = rename_handlers r' handlers.
Proof.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite same_continuations, IH,
    (rename_handler_ext r r' same_variables same_continuations h).
  reflexivity.
Qed.

Fixpoint rename_cont_value_ext (c : cont_value) :
  rename_cont_value r c = rename_cont_value r' c.
Proof.
  destruct c
    as [h vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    simpl;
    [ | rewrite same_continuations; reflexivity
      | rewrite (rename_cont_value_ext c); reflexivity | ].
  - rewrite
      (rename_handler_ext r r' same_variables same_continuations h),
      rename_env_ext, rename_handlers_ext.
    f_equal.
    induction conts as [ | [k c] conts IH]; simpl; [ reflexivity | ].
    rewrite same_continuations, (rename_cont_value_ext c), IH.
    reflexivity.
  - rewrite (rename_cont_value_ext return_to), (rename_cont_value_ext exn_to).
    reflexivity.
Qed.

Lemma rename_cont_env_ext (conts : cont_env) :
  rename_cont_env r conts = rename_cont_env r' conts.
Proof.
  induction conts as [ | [k c] conts IH]; simpl; [ reflexivity | ].
  rewrite same_continuations, rename_cont_value_ext, IH.
  reflexivity.
Qed.

Lemma rename_static_env_ext (statics : static_env) :
  rename_static_env r statics = rename_static_env r' statics.
Proof.
  unfold rename_static_env. f_equal. apply map_ext. intros [id c].
  simpl.
  rewrite (rename_code_ext r r' same_variables same_continuations c).
  reflexivity.
Qed.

Lemma rename_state_ext (s : state) :
  rename_state r s = rename_state r' s.
Proof.
  unfold rename_state.
  rewrite
    (rename_expr_ext r r' same_variables same_continuations),
    rename_env_ext, rename_cont_env_ext,
    rename_static_env_ext.
  reflexivity.
Qed.

End Extensional_state.

Section Compose_state.

Variables second first : renaming.

Lemma rename_env_compose (vars : env) :
  rename_env (compose_renaming second first) vars
  = rename_env second (rename_env first vars).
Proof.
  induction vars as [ | [x v] vars IH]; simpl; [ reflexivity | ].
  rewrite rename_variable_compose, IH. reflexivity.
Qed.

Lemma rename_handlers_compose
    (handlers : list (continuation * continuation_handler)) :
  rename_handlers (compose_renaming second first) handlers
  = rename_handlers second (rename_handlers first handlers).
Proof.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite rename_continuation_compose, rename_handler_compose, IH.
  reflexivity.
Qed.

Fixpoint rename_cont_value_compose (c : cont_value) :
  rename_cont_value (compose_renaming second first) c
  = rename_cont_value second (rename_cont_value first c).
Proof.
  destruct c
    as [h vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    simpl;
    [ | rewrite rename_continuation_compose; reflexivity
      | rewrite (rename_cont_value_compose c); reflexivity | ].
  - rewrite rename_handler_compose, rename_env_compose,
      rename_handlers_compose.
    f_equal.
    induction conts as [ | [k c] conts IH]; simpl; [ reflexivity | ].
    rewrite rename_continuation_compose,
      (rename_cont_value_compose c), IH.
    reflexivity.
  - rewrite (rename_cont_value_compose return_to),
      (rename_cont_value_compose exn_to).
    reflexivity.
Qed.

Lemma rename_cont_env_compose (conts : cont_env) :
  rename_cont_env (compose_renaming second first) conts
  = rename_cont_env second (rename_cont_env first conts).
Proof.
  induction conts as [ | [k c] conts IH]; simpl; [ reflexivity | ].
  rewrite rename_continuation_compose, rename_cont_value_compose,
    IH.
  reflexivity.
Qed.

Lemma rename_static_env_compose (statics : static_env) :
  rename_static_env (compose_renaming second first) statics
  = rename_static_env second
      (rename_static_env first statics).
Proof.
  unfold rename_static_env. simpl. f_equal.
  unfold map_values. rewrite map_map. apply map_ext. intros [id c]. simpl.
  rewrite rename_code_compose. reflexivity.
Qed.

End Compose_state.

Theorem rename_state_identity :
  rename_state_identity_statement.
Proof.
  intros s. unfold rename_state.
  rewrite rename_expr_id, rename_env_id, rename_cont_env_id,
    rename_static_env_id.
  symmetry. apply state_eta.
Qed.

Theorem rename_state_compose : rename_state_compose_statement.
Proof.
  intros second first s. unfold rename_state. simpl_state.
  rewrite rename_expr_compose_aux, rename_env_compose,
    rename_cont_env_compose, rename_static_env_compose.
  reflexivity.
Qed.

Theorem rename_state_inverse : rename_state_inverse_statement.
Proof.
  intros r s. split; rewrite <- rename_state_compose.
  - rewrite (rename_state_ext _ identity_renaming);
      [ apply rename_state_identity | | ].
    + apply compose_inverse_l_variable.
    + apply compose_inverse_l_continuation.
  - rewrite (rename_state_ext _ identity_renaming);
      [ apply rename_state_identity | | ].
    + apply compose_inverse_r_variable.
    + apply compose_inverse_r_continuation.
Qed.

Lemma rename_behaviour_inverse (r : renaming) (b : behaviour) :
  rename_behaviour r
    (rename_behaviour (inverse_renaming r) b)
  = b.
Proof.
  destruct b as [k vs | | ]; simpl; try reflexivity.
  rewrite rename_continuation_inverse_r. reflexivity.
Qed.

(* ======================================================================== *)
(* What a step is made of commutes with the action                          *)
(* ======================================================================== *)

Lemma eval_simple_equivariant (r : renaming) (vars : env) (s : simple) :
  eval_simple (rename_env r vars) (rename_simple r s) = eval_simple vars s.
Proof. apply eval_simple_renamed. intros x _. apply lookup_rename_env. Qed.

Lemma eval_simples_equivariant (r : renaming) (vars : env)
    (args : list simple) :
  eval_simples (rename_env r vars) (map (rename_simple r) args)
  = eval_simples vars args.
Proof. apply eval_simples_renamed. intros x _. apply lookup_rename_env. Qed.

(* The static environment is read for its statically allocated values
   only, which the action leaves alone. *)
Lemma eval_named_static_values (statics statics' : static_env) (vars : env)
    (d : named) :
  static_values statics = static_values statics' ->
  eval_named statics vars d = eval_named statics' vars d.
Proof.
  intros Hvalues.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; simpl;
    try reflexivity.
  - destruct (eval_simples vars args) as [vs | ]; [ | reflexivity ].
    apply eval_prim_static_values. intros s _. rewrite Hvalues.
    reflexivity.
  - destruct (eval_simple vars closure) as [v | ]; [ | reflexivity ].
    unfold closure_of. rewrite Hvalues. reflexivity.
  - destruct (eval_simple vars closure) as [v | ]; [ | reflexivity ].
    unfold project_value_slot, closure_in_set. rewrite Hvalues.
    reflexivity.
  - destruct (eval_simple vars closure) as [v | ]; [ | reflexivity ].
    unfold project_function_slot. rewrite Hvalues. reflexivity.
Qed.

Lemma eval_named_equivariant (r : renaming) (statics : static_env)
    (vars : env) (d : named) :
  eval_named (rename_static_env r statics) (rename_env r vars)
    (rename_named r d)
  = eval_named statics vars d.
Proof.
  rewrite (eval_named_renamed r d vars);
    [ | intros x _; apply lookup_rename_env ].
  apply eval_named_static_values. reflexivity.
Qed.

Lemma eval_static_const_equivariant (r : renaming) (vars : env)
    (const : static_const) :
  eval_static_const (rename_env r vars) (rename_static_const r const)
  = eval_static_const vars const.
Proof.
  apply eval_static_const_renamed. intros x _. apply lookup_rename_env.
Qed.

Lemma lookup_rename_code (r : renaming) (id : code_id)
    (statics : static_env) :
  lookup id (static_code (rename_static_env r statics))
  = option_map (rename_code r) (lookup id (static_code statics)).
Proof. apply lookup_map_values. Qed.

Lemma eval_static_bindings_equivariant (r : renaming) (vars : env)
    (bindings : list static_binding) :
  ∀ statics : static_env,
    eval_static_bindings (rename_env r vars)
      (map (rename_static_binding r) bindings)
      (rename_static_env r statics)
    = option_map (rename_static_env r)
        (eval_static_bindings vars bindings statics).
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros statics; simpl; [ reflexivity | | apply IH | ].
  - rewrite (lookup_map_values _ _ (rename_code r) id).
    destruct (lookup id (static_code statics)); [ reflexivity | ].
    exact
      (IH (Static_env ((id, c) :: static_code statics)
             (static_values statics))).
  - destruct (lookup sym (static_values statics)); [ reflexivity | ].
    rewrite eval_static_const_equivariant.
    destruct (eval_static_const vars const) as [allocated | ];
      [ | reflexivity ].
    exact
      (IH (Static_env (static_code statics)
             ((sym, allocated) :: static_values statics))).
Qed.

Lemma bind_params_equivariant (r : renaming) (params : list variable) :
  ∀ (vs : list value) (vars : env),
    bind_params (map (rename_variable r) params) vs (rename_env r vars)
    = option_map (rename_env r) (bind_params params vs vars).
Proof.
  induction params as [ | x params IH]; intros [ | v vs] vars; simpl;
    try reflexivity.
  exact (IH vs ((x, v) :: vars)).
Qed.

Lemma lookup_rename_cont_env (r : renaming) (k : continuation)
    (conts : cont_env) :
  lookup (rename_continuation r k) (rename_cont_env r conts)
  = option_map (rename_cont_value r) (lookup k conts).
Proof.
  induction conts as [ | [k' c] conts IH]; simpl; [ reflexivity | ].
  rewrite rename_continuation_eqb.
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma resolve_equivariant (r : renaming) (k : continuation)
    (conts : cont_env) :
  resolve (rename_continuation r k) (rename_cont_env r conts)
  = rename_cont_value r (resolve k conts).
Proof.
  unfold resolve. rewrite lookup_rename_cont_env.
  destruct (lookup k conts); reflexivity.
Qed.

Lemma rename_cont_env_app (r : renaming) (conts conts' : cont_env) :
  rename_cont_env r (conts ++ conts')
  = rename_cont_env r conts ++ rename_cont_env r conts'.
Proof. apply map_app. Qed.

(* The closures of a recursive group hold the whole group, so the
   statement is of any part of it. *)
Lemma rec_closures_equivariant_aux (r : renaming)
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env)
    (part : list (continuation * continuation_handler)) :
  map
    (fun '(k, h) =>
       (k,
        Closure h (rename_env r vars) (rename_cont_env r conts)
          (rename_handlers r handlers)))
    (rename_handlers r part)
  = rename_cont_env r
      (map (fun '(k, h) => (k, Closure h vars conts handlers)) part).
Proof.
  induction part as [ | [k h] part IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma rec_closures_equivariant (r : renaming)
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) :
  rec_closures (rename_handlers r handlers) (rename_env r vars)
    (rename_cont_env r conts)
  = rename_cont_env r (rec_closures handlers vars conts).
Proof. apply rec_closures_equivariant_aux. Qed.

(* A call of code. *)
Lemma call_code_equivariant (r : renaming) (closure : value) (c : code)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  call_code closure (rename_code r c) vs args_arity return_arity
    (rename_cont_value r return_to) (rename_cont_value r exn_to)
    (rename_static_env r statics) hp rd
  = rename_result r
      (call_code closure c vs args_arity return_arity return_to exn_to
         statics hp rd).
Proof.
  destruct c as [params my_closure ret exn metadata body].
  unfold call_code. simpl.
  rewrite rename_binds_statics.
  destruct (binds_statics body); [ reflexivity | ].
  destruct (negb _); [ reflexivity | ].
  change [(rename_variable r my_closure, closure)]
    with (rename_env r [(my_closure, closure)]).
  rewrite bind_params_equivariant.
  destruct (bind_params params vs [(my_closure, closure)]); reflexivity.
Qed.

(* What a call is, under a renaming: the same, with its code renamed. *)
Definition rename_shape (r : renaming) (shape : call_shape) : call_shape :=
  match shape with
  | Full_call closure c vs args_arity =>
    Full_call closure (rename_code r c) vs args_arity
  | Over_call closure c vs remaining args_arity remaining_arity =>
    Over_call closure (rename_code r c) vs remaining args_arity
      remaining_arity
  | Partial_call partial => Partial_call partial
  end.

Lemma shape_of_closure_call_equivariant (r : renaming)
    (statics : static_env) (closure : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity) :
  shape_of_closure_call (rename_static_env r statics) closure kind vs
    args_arity return_arity
  = option_map (rename_shape r)
      (shape_of_closure_call statics closure kind vs args_arity
         return_arity).
Proof.
  unfold shape_of_closure_call.
  replace (closure_of (rename_static_env r statics) closure)
    with (closure_of statics closure) by reflexivity.
  destruct (closure_of statics closure) as [[called slots] | ];
    [ | reflexivity ].
  rewrite lookup_rename_code.
  destruct
    (match kind with
     | Direct named => Pos.eqb called named
     | Indirect_unknown_arity => true
     end);
    [ | reflexivity ].
  destruct (lookup called (static_code statics)) as [c | ];
    cbn [option_map]; [ | reflexivity ].
  assert (Hargs :
    arguments_for_code (rename_static_env r statics) (rename_code r c) vs
      args_arity
    = arguments_for_code statics c vs args_arity).
  { rewrite (arguments_for_code_agree statics (rename_static_env r statics)
               (rename_code r c) vs args_arity)
      by (intros tuple rest p _; symmetry;
          apply eval_prim_same_static_values; destruct statics; reflexivity).
    unfold arguments_for_code. destruct c; reflexivity. }
  rewrite Hargs.
  destruct (arguments_for_code statics c vs args_arity)
    as [[vs' args_arity'] | ]; [ | reflexivity ].
  cbv beta iota. unfold shape_of_code_call.
  destruct c as [params my_closure ret exn metadata body]. simpl.
  destruct (Nat.eqb _ _); [ reflexivity | ].
  destruct (negb _); [ reflexivity | ].
  destruct (Nat.ltb _ _); [ reflexivity | ].
  destruct (cm_is_tupled metadata); reflexivity.
Qed.

Lemma shape_of_call_equivariant (r : renaming) (statics : static_env)
    (callee : value) (kind : call_kind) (vs : list value)
    (args_arity return_arity : arity) :
  shape_of_call (rename_static_env r statics) callee kind vs args_arity
    return_arity
  = option_map (rename_shape r)
      (shape_of_call statics callee kind vs args_arity return_arity).
Proof.
  unfold shape_of_call.
  destruct kind as [named | ]; [ apply shape_of_closure_call_equivariant | ].
  destruct (partial_application_of callee)
    as [[[closure applied] applied_arity] | ];
    [ | apply shape_of_closure_call_equivariant ].
  destruct args_arity; [ reflexivity | ].
  destruct (kinds_eqb _ _);
    [ apply shape_of_closure_call_equivariant | reflexivity ].
Qed.

Lemma run_call_equivariant (r : renaming)
    (return_partial return_partial' : value -> step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  (∀ partial : value,
    return_partial' partial = rename_result r (return_partial partial)) ->
  run_call return_partial' (option_map (rename_shape r) shape) return_arity
    (rename_cont_value r return_to) (rename_cont_value r exn_to)
    (rename_static_env r statics) hp rd
  = rename_result r
      (run_call return_partial shape return_arity return_to exn_to statics hp
         rd).
Proof.
  intros Hpartial.
  destruct shape
    as [[closure c vs args_arity
        | closure c vs remaining args_arity remaining_arity | partial] | ];
    simpl.
  - apply call_code_equivariant.
  - replace (code_result_arity (rename_code r c)) with (code_result_arity c)
      by (destruct c; reflexivity).
    exact
      (call_code_equivariant r closure c vs args_arity (code_result_arity c)
         (Over_application remaining remaining_arity return_arity return_to
            exn_to)
         exn_to statics hp rd).
  - apply Hpartial.
  - reflexivity.
Qed.

Lemma apply_cont_value_equivariant (r : renaming) (c : cont_value) :
  ∀ (vs : list value) (statics : static_env) (hp : heap) (rd : read_record),
    apply_cont_value (rename_cont_value r c) vs
      (rename_static_env r statics) hp rd
    = rename_result r (apply_cont_value c vs statics hp rd).
Proof.
  induction c
    as [[params metadata handler] def_vars def_conts recursive | k
       | c IH extra
       | remaining remaining_arity return_arity return_to exn_to IHreturn _]
    using cont_value_over_ind;
    intros vs statics hp rd.
  - simpl.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | reflexivity ].
    rewrite bind_params_equivariant.
    destruct (bind_params params vs def_vars) as [handler_vars | ]; simpl;
      [ | reflexivity ].
    unfold rename_state. simpl_state.
    rewrite rename_cont_env_app, <- rec_closures_equivariant.
    reflexivity.
  - reflexivity.
  - simpl. apply IH.
  - destruct vs as [ | v [ | v' vs]]; try reflexivity.
    cbn [rename_cont_value apply_cont_value].
    rewrite shape_of_call_equivariant.
    apply run_call_equivariant. intros partial. apply IHreturn.
Qed.

Lemma with_extra_args_equivariant (r : renaming) (c : cont_value)
    (extra : list value) :
  with_extra_args (rename_cont_value r c) extra
  = rename_cont_value r (with_extra_args c extra).
Proof. destruct extra; reflexivity. Qed.

Lemma call_function_equivariant (r : renaming) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env) (hp : heap)
      (rd : read_record) :
  call_function callee kind vs args_arity return_arity
    (rename_cont_value r return_to) (rename_cont_value r exn_to)
    (rename_static_env r statics) hp rd
  = rename_result r
      (call_function callee kind vs args_arity return_arity return_to exn_to
         statics hp rd).
Proof.
  unfold call_function. rewrite shape_of_call_equivariant.
  apply run_call_equivariant. intros partial.
  apply apply_cont_value_equivariant.
Qed.

(* The executor of a named expression: a primitive reads the static
   environment for its statically allocated values, which the action
   leaves alone, and the heap and the record are not acted on. *)
Lemma exec_named_equivariant (r : renaming) (statics : static_env)
    (hp : heap) (rd : read_record) (vars : env) (d : named) :
  exec_named (rename_static_env r statics) hp rd (rename_env r vars)
    (rename_named r d)
  = exec_named statics hp rd vars d.
Proof.
  pose proof (eval_named_equivariant r statics vars d) as Heval.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    unfold exec_named; cbn [rename_named] in *;
    try (rewrite Heval; reflexivity).
  rewrite eval_simples_equivariant.
  destruct (eval_simples vars args) as [vs | ]; [ | reflexivity ].
  destruct statics as [codes values]. reflexivity.
Qed.

(* ======================================================================== *)
(* Execution is equivariant                                                 *)
(* ======================================================================== *)

Theorem step_equivariant : step_equivariant_statement.
Proof.
  intros r s. rewrite (state_eta s). unfold rename_state.
  simpl_state.
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; autorewrite with step.
  - (* Let *)
    rewrite exec_named_equivariant.
    destruct
      (exec_named (state_statics s) (state_heap s) (state_reads s)
         (state_vars s) d)
      as [[[v hp'] rd'] | ];
      [ | reflexivity ].
    destruct (kind_eqb (kind_of_value v) kind); reflexivity.
  - (* Let_static *)
    rewrite eval_static_bindings_equivariant.
    destruct
      (eval_static_bindings (state_vars s) bindings (state_statics s));
      reflexivity.
  - (* Let_cont_non_recursive *)
    reflexivity.
  - (* Let_cont_recursive *)
    fold (rename_handlers r handlers).
    rewrite rename_handlers_bind_statics.
    destruct (handlers_bind_statics handlers); [ reflexivity | ].
    rewrite rec_closures_equivariant, <- rename_cont_env_app.
    reflexivity.
  - (* Apply *)
    simpl. rewrite eval_simple_equivariant, eval_simples_equivariant.
    destruct (eval_simple (state_vars s) (ap_callee ap)) as [callee | ];
      [ | reflexivity ].
    destruct (eval_simples (state_vars s) (ap_args ap)) as [vs | ];
      [ | reflexivity ].
    rewrite exn_extra_simples_rename, eval_simples_equivariant.
    destruct (eval_simples (state_vars s) (exn_extra_simples ap))
      as [extra | ];
      [ | reflexivity ].
    rewrite !resolve_equivariant, with_extra_args_equivariant.
    apply call_function_equivariant.
  - (* Apply_cont *)
    simpl. rewrite eval_simples_equivariant.
    destruct (eval_simples (state_vars s) (ac_args ac)) as [vs | ];
      [ | reflexivity ].
    unfold apply_continuation. rewrite resolve_equivariant.
    apply apply_cont_value_equivariant.
  - (* Switch *)
    rewrite eval_simple_equivariant.
    destruct (eval_simple (state_vars s) scrutinee) as [v | ];
      [ | reflexivity ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | reflexivity ].
    rewrite find_arm_renamed.
    destruct (find_arm i arms); reflexivity.
  - (* Invalid *)
    reflexivity.
Qed.

Theorem run_equivariant : run_equivariant_statement.
Proof.
  intros r fuel. induction fuel as [ | fuel IH]; intros s; simpl;
    [ reflexivity | ].
  rewrite step_equivariant.
  destruct (step s) as [s' | k vs | ]; simpl; [ apply IH | | ]; reflexivity.
Qed.

(* A run of a renamed state that has ended, ended as the run of the state
   did, renamed. *)
Lemma run_renamed_exit (r : renaming) (fuel : nat) (s : state)
    (k : continuation) (vs : list value) (hp : heap) (rd : read_record) :
  run fuel (rename_state r s) = Exit (rename_continuation r k) vs hp rd
  <-> run fuel s = Exit k vs hp rd.
Proof.
  rewrite run_equivariant. split.
  - destruct (run fuel s) as [s' | k' vs' | ]; simpl; try discriminate.
    intros [= Hk -> -> ->]. apply rename_continuation_injective in Hk.
    subst k'. reflexivity.
  - intros ->. reflexivity.
Qed.

Lemma run_renamed_undefined (r : renaming) (fuel : nat) (s : state) :
  run fuel (rename_state r s) = Undefined
  <-> run fuel s = Undefined.
Proof.
  rewrite run_equivariant.
  destruct (run fuel s); simpl; split; intros Hrun;
    try discriminate; reflexivity.
Qed.

Lemma run_renamed_step (r : renaming) (fuel : nat) (s : state) :
  (∃ s' : state, run fuel (rename_state r s) = Step s')
  <-> ∃ s' : state, run fuel s = Step s'.
Proof.
  rewrite run_equivariant.
  destruct (run fuel s) as [s' | k vs | ]; simpl; split;
    intros [t Ht]; try discriminate; eauto.
Qed.

Theorem behaves_equivariant : behaves_equivariant_statement.
Proof.
  intros r s [k vs | | ]; simpl.
  - split; intros [fuel [rd Hrun]]; exists fuel; eexists;
      apply (run_renamed_exit r fuel s k vs); exact Hrun.
  - split; intros [fuel Hrun]; exists fuel;
      apply (run_renamed_undefined r fuel s); exact Hrun.
  - split; intros Hdiverges fuel;
      apply (run_renamed_step r fuel s), Hdiverges.
Qed.

(* A behaviour of a renamed state is the renaming of one of the state. *)
Lemma behaves_renamed (r : renaming) (s : state) (b : behaviour) :
  behaves (rename_state r s) b
  <-> behaves s (rename_behaviour (inverse_renaming r) b).
Proof.
  rewrite <- (rename_behaviour_inverse r b) at 1.
  apply behaves_equivariant.
Qed.

Theorem equivalent_equivariant : equivalent_equivariant_statement.
Proof.
  intros r s s' Hequivalent b. rewrite !behaves_renamed.
  apply Hequivalent.
Qed.

Theorem refines_equivariant : refines_equivariant_statement.
Proof.
  intros r source target [Hexits Hends]. split.
  - intros k vs hp. rewrite !behaves_renamed. apply Hexits.
  - intros b Hb. rewrite !behaves_renamed.
    change (rename_behaviour (inverse_renaming r) Goes_wrong)
      with Goes_wrong.
    apply Hends. destruct b; [ discriminate | discriminate | exact Hb ].
Qed.

(* A state with a renamed expression is the renaming of the state whose
   environments are renamed back. *)
Lemma state_of_renamed_expr (r : renaming) (e : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record) :
  State (rename_expr r e) vars conts statics hp rd
  = rename_state r
      (State e (rename_env (inverse_renaming r) vars)
         (rename_cont_env (inverse_renaming r) conts)
         (rename_static_env (inverse_renaming r) statics) hp rd).
Proof.
  pose proof (rename_state_inverse r (State e vars conts statics hp rd))
    as [_ Hinverse].
  unfold rename_state in *. simpl_state.
  injection Hinverse as _ Hvars Hconts Hstatics.
  rewrite Hvars, Hconts, Hstatics. reflexivity.
Qed.

Theorem expr_equivalent_equivariant : expr_equivalent_equivariant_statement.
Proof.
  intros r e e' Hequivalent vars conts statics hp rd.
  rewrite !state_of_renamed_expr.
  apply equivalent_equivariant, Hequivalent.
Qed.

Theorem expr_refines_equivariant : expr_refines_equivariant_statement.
Proof.
  intros r source target Hrefines vars conts statics hp rd.
  rewrite !state_of_renamed_expr.
  apply refines_equivariant, Hrefines.
Qed.

Theorem renamed_state_equivalent : renamed_state_equivalent_statement.
Proof.
  intros r s Hfixes b. rewrite behaves_renamed.
  destruct b as [k vs | | ]; simpl; try reflexivity.
  split; intros Hexits.
  - rewrite <- (Hfixes k vs hp Hexits), rename_continuation_inverse_l.
    exact Hexits.
  - pose proof (Hfixes _ vs hp Hexits) as Hfixed.
    rewrite rename_continuation_inverse_r in Hfixed.
    rewrite Hfixed. exact Hexits.
Qed.

(* ======================================================================== *)
(* Folding constants is equivariant                                         *)
(* ======================================================================== *)

Lemma consts_of_simples_renamed (r : renaming) (args : list simple) :
  consts_of_simples (map (rename_simple r) args) = consts_of_simples args.
Proof.
  induction args as [ | [x | sym | c] args IH]; simpl;
    [ reflexivity | reflexivity | reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma fold_named_equivariant (r : renaming) (d : named) :
  fold_named (rename_named r d) = rename_named r (fold_named d).
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure]; try reflexivity.
  destruct p as [u | b | v | h | rp];
    [ destruct u | | | destruct h | ];
    cbn [fold_named rename_named]; try reflexivity;
    rewrite consts_of_simples_renamed;
    match goal with
    | |- context [prim_may_fold ?q] =>
      destruct (if prim_may_fold q then consts_of_simples args else None)
        as [cs | ]; [ | reflexivity ];
      destruct (eval_prim empty_static_env q (map Value_const cs))
        as [w | ];
        [ destruct w | ]; reflexivity
    end.
Qed.

Lemma fold_switch_equivariant (r : renaming) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  fold_switch (rename_simple r scrutinee)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = rename_expr r (fold_switch scrutinee arms).
Proof.
  destruct scrutinee as [x | sym | c]; try reflexivity.
  destruct c; try reflexivity.
  simpl. rewrite find_arm_renamed.
  destruct (find_arm i arms); reflexivity.
Qed.

Fixpoint fold_constants_equivariant_aux (r : renaming) (e : expr) :
  fold_constants (rename_expr r e)
  = rename_expr r (fold_constants e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - rewrite fold_named_equivariant, fold_constants_equivariant_aux.
    reflexivity.
  - rewrite fold_constants_equivariant_aux. reflexivity.
  - rewrite !fold_constants_equivariant_aux. reflexivity.
  - rewrite fold_constants_equivariant_aux. reflexivity.
  - apply fold_switch_equivariant.
Qed.

Theorem fold_constants_equivariant : fold_constants_equivariant_statement.
Proof. exact fold_constants_equivariant_aux. Qed.

(* ======================================================================== *)
(* Inlining is equivariant                                                  *)
(* ======================================================================== *)

(* The tests that decide whether a call may be inlined compare names for
   equality only, and a renaming is injective. *)
Lemma among_renamed (p : permutation) (x : positive) (xs : list positive) :
  among (permute p x) (map (permute p) xs) = among x xs.
Proof.
  unfold among. induction xs as [ | y xs IH]; simpl; [ reflexivity | ].
  rewrite permute_eqb, IH. reflexivity.
Qed.

Lemma among_rename_variable (r : renaming) (x : variable)
    (xs : list variable) :
  among (rename_variable r x) (map (rename_variable r) xs) = among x xs.
Proof. apply among_renamed. Qed.

Lemma among_rename_continuation (r : renaming) (k : continuation)
    (ks : list continuation) :
  among (rename_continuation r k) (map (rename_continuation r) ks)
  = among k ks.
Proof. apply among_renamed. Qed.

Lemma simple_in_scope_equivariant (r : renaming) (vs : list variable)
    (s : simple) :
  simple_in_scope (map (rename_variable r) vs) (rename_simple r s)
  = simple_in_scope vs s.
Proof.
  destruct s; simpl; [ apply among_rename_variable | | ]; reflexivity.
Qed.

Lemma simples_in_scope_equivariant (r : renaming) (vs : list variable)
    (args : list simple) :
  simples_in_scope (map (rename_variable r) vs) (map (rename_simple r) args)
  = simples_in_scope vs args.
Proof.
  unfold simples_in_scope.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite simple_in_scope_equivariant, IH. reflexivity.
Qed.

Lemma value_slots_in_scope_equivariant (r : renaming) (vs : list variable)
    (slots : list (value_slot * simple)) :
  forallb
    (fun slot => simple_in_scope (map (rename_variable r) vs) (snd slot))
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = forallb (fun slot => simple_in_scope vs (snd slot)) slots.
Proof.
  induction slots as [ | [slot s] slots IH]; simpl; [ reflexivity | ].
  rewrite simple_in_scope_equivariant, IH. reflexivity.
Qed.

Lemma named_in_scope_equivariant (r : renaming) (vs : list variable)
    (d : named) :
  named_in_scope (map (rename_variable r) vs) (rename_named r d)
  = named_in_scope vs d.
Proof.
  destruct d; simpl;
    first
      [ apply simple_in_scope_equivariant
      | apply simples_in_scope_equivariant
      | apply value_slots_in_scope_equivariant ].
Qed.

Lemma static_binding_in_scope_equivariant (r : renaming)
    (vs : list variable) (binding : static_binding) :
  static_binding_in_scope (map (rename_variable r) vs)
    (rename_static_binding r binding)
  = static_binding_in_scope vs binding.
Proof.
  destruct binding as [id c | id | sym const]; simpl; try reflexivity.
  destruct const; simpl;
    first
      [ reflexivity | apply value_slots_in_scope_equivariant
      | apply simples_in_scope_equivariant
      | apply simple_in_scope_equivariant ].
Qed.

Lemma apply_cont_in_scope_equivariant (r : renaming) (vs : list variable)
    (ks : list continuation) (ac : apply_cont) :
  apply_cont_in_scope (map (rename_variable r) vs)
    (map (rename_continuation r) ks) (rename_apply_cont r ac)
  = apply_cont_in_scope vs ks ac.
Proof.
  unfold apply_cont_in_scope.
  rewrite apply_cont_continuations_renamed.
  generalize (apply_cont_continuations ac). intros l. simpl.
  rewrite simples_in_scope_equivariant. f_equal.
  unfold FlambdaMapSpec.all_among.
  induction l as [ | k l IH]; simpl; [ reflexivity | ].
  rewrite among_rename_continuation, IH. reflexivity.
Qed.

Lemma map_fst_rename_handlers (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  map fst (rename_handlers r handlers)
  = map (rename_continuation r) (map fst handlers).
Proof.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Fixpoint expr_in_scope_equivariant (r : renaming) (e : expr) :
  ∀ (vs : list variable) (ks : list continuation),
    expr_in_scope (map (rename_variable r) vs)
      (map (rename_continuation r) ks) (rename_expr r e)
    = expr_in_scope vs ks e
with handler_in_scope_equivariant (r : renaming) (h : continuation_handler) :
  ∀ (vs : list variable) (ks : list continuation),
    handler_in_scope (map (rename_variable r) vs)
      (map (rename_continuation r) ks) (rename_handler r h)
    = handler_in_scope vs ks h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros vs ks; simpl.
    + rewrite named_in_scope_equivariant.
      change (rename_variable r x :: map (rename_variable r) vs)
        with (map (rename_variable r) (x :: vs)).
      rewrite expr_in_scope_equivariant. reflexivity.
    + rewrite (expr_in_scope_equivariant r body vs ks). f_equal.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite static_binding_in_scope_equivariant, IH. reflexivity.
    + change (rename_continuation r k :: map (rename_continuation r) ks)
        with (map (rename_continuation r) (k :: ks)).
      rewrite handler_in_scope_equivariant, expr_in_scope_equivariant.
      reflexivity.
    + fold (rename_handlers r handlers).
      rewrite map_fst_rename_handlers, <- map_app.
      generalize (map fst handlers ++ ks). intros ks'.
      rewrite (expr_in_scope_equivariant r body vs ks'). f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_in_scope_equivariant r h vs ks'), IH. reflexivity.
    + rewrite exn_extra_simples_rename, simple_in_scope_equivariant,
        !simples_in_scope_equivariant, !among_rename_continuation.
      reflexivity.
    + apply apply_cont_in_scope_equivariant.
    + rewrite simple_in_scope_equivariant. f_equal.
      induction arms as [ | [i ac] arms IH]; simpl; [ reflexivity | ].
      rewrite apply_cont_in_scope_equivariant, IH. reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler]. intros vs ks. simpl.
    rewrite <- map_app. apply expr_in_scope_equivariant.
Qed.

Lemma var_in_simples_equivariant (r : renaming) (x : variable)
    (args : list simple) :
  var_in_simples (rename_variable r x) (map (rename_simple r) args)
  = var_in_simples x args.
Proof.
  unfold var_in_simples.
  induction args as [ | [y | sym | c] args IH]; simpl; rewrite ?IH;
    try reflexivity.
  rewrite rename_variable_eqb. reflexivity.
Qed.

Lemma params_not_in_args_equivariant (r : renaming) (params : list variable)
    (args : list simple) :
  forallb
    (fun param => negb (var_in_simples param (map (rename_simple r) args)))
    (map (rename_variable r) params)
  = forallb (fun param => negb (var_in_simples param args)) params.
Proof.
  induction params as [ | param params IH]; simpl; [ reflexivity | ].
  rewrite var_in_simples_equivariant, IH. reflexivity.
Qed.

Lemma inlinable_b_equivariant (r : renaming) (ap : apply_expr) (c : code) :
  inlinable_b (rename_apply r ap) (rename_code r c)
  = inlinable_b ap c.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  unfold inlinable_b. simpl.
  pose proof
    (expr_in_scope_equivariant r body (my_closure :: params) [ret; exn])
    as Hscope.
  simpl in Hscope.
  assert (Hextra : no_exn_extra_args (rename_apply r ap)
                   = no_exn_extra_args ap).
  { unfold no_exn_extra_args. simpl.
    destruct (ap_exn_extra_args ap); reflexivity. }
  rewrite Hscope, !rename_continuation_eqb, var_in_simples_equivariant,
    params_not_in_args_equivariant, !length_map,
    rename_binds_statics, Hextra.
  reflexivity.
Qed.

Lemma tupled_inlinable_b_equivariant (r : renaming) (ap : apply_expr)
    (c : code) :
  tupled_inlinable_b (rename_apply r ap) (rename_code r c)
  = tupled_inlinable_b ap c.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  unfold tupled_inlinable_b. simpl.
  pose proof
    (expr_in_scope_equivariant r body (my_closure :: params) [ret; exn])
    as Hscope.
  simpl in Hscope.
  assert (Hextra : no_exn_extra_args (rename_apply r ap)
                   = no_exn_extra_args ap).
  { unfold no_exn_extra_args. simpl.
    destruct (ap_exn_extra_args ap); reflexivity. }
  rewrite Hscope, !rename_continuation_eqb, var_in_simples_equivariant,
    params_not_in_args_equivariant, !length_map,
    rename_binds_statics, Hextra.
  reflexivity.
Qed.

(* The test that a direct call may be split, and the call split. *)
Lemma over_applied_b_equivariant (r : renaming) (ap : apply_expr) (c : code) :
  over_applied_b (rename_apply r ap) (rename_code r c)
  = over_applied_b ap c.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  destruct ap as [f kind args k k_exn extra args_arity return_arity].
  unfold over_applied_b, exn_extra_simples. simpl.
  rewrite map_map. simpl.
  replace
    (map (fun arg : simple * flambda_kind => rename_simple r (fst arg)) extra)
    with (map (rename_simple r) (map fst extra))
    by (rewrite map_map; reflexivity).
  rewrite skipn_map, !var_in_simples_equivariant, rename_continuation_eqb,
    length_map.
  reflexivity.
Qed.

Lemma split_over_application_equivariant (r : renaming) (called : code_id)
    (c : code) (ap : apply_expr) :
  split_over_application called (rename_code r c) (rename_apply r ap)
  = rename_expr r (split_over_application called c ap).
Proof.
  destruct c as [params my_closure ret exn metadata body].
  destruct ap as [f kind args k k_exn extra args_arity return_arity].
  unfold split_over_application. simpl. unfold rename_apply. simpl.
  rewrite firstn_map, skipn_map. reflexivity.
Qed.

Lemma codes_of_equivariant (r : renaming) (bindings : list static_binding) :
  ∀ known : list (code_id * code),
    codes_of (map (rename_static_binding r) bindings)
      (map_values (rename_code r) known)
    = map_values (rename_code r) (codes_of bindings known).
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros known; simpl; [ reflexivity | | apply IH | apply IH ].
  exact (IH ((id, c) :: known)).
Qed.

Lemma let_params_equivariant (r : renaming) (params : list variable) :
  ∀ (kinds : list flambda_kind) (args : list simple) (body : expr),
    rename_expr r (let_params params kinds args body)
    = let_params (map (rename_variable r) params) kinds
        (map (rename_simple r) args) (rename_expr r body).
Proof.
  induction params as [ | param params IH]; intros kinds [ | arg args] body;
    simpl; try reflexivity.
  rewrite IH. reflexivity.
Qed.

(* The renaming that inlining builds from four continuations is built
   from them by swaps, so it commutes with a renaming of the four. *)
Lemma permute_swap (p : permutation) (a b x : positive) :
  permute p (swap a b x) = swap (permute p a) (permute p b) (permute p x).
Proof.
  unfold swap. rewrite !permute_eqb.
  destruct (Pos.eqb x a); [ reflexivity | ].
  destruct (Pos.eqb x b); reflexivity.
Qed.

Lemma inline_renaming_conjugate (r : renaming)
    (ret exn k k_exn j : continuation) :
  rename_continuation
    (inline_renaming (rename_continuation r ret) (rename_continuation r exn)
       (rename_continuation r k) (rename_continuation r k_exn))
    (rename_continuation r j)
  = rename_continuation r
      (rename_continuation (inline_renaming ret exn k k_exn) j).
Proof.
  unfold rename_continuation, inline_renaming. simpl.
  rewrite !permute_swap. reflexivity.
Qed.

Lemma inlined_body_equivariant (r : renaming) (c : code) (ap : apply_expr) :
  inlined_body (rename_code r c)
    (inline_renaming (code_return_continuation (rename_code r c))
       (code_exn_continuation (rename_code r c))
       (ap_continuation (rename_apply r ap))
       (ap_exn_continuation (rename_apply r ap)))
    (rename_apply r ap)
  = rename_expr r
      (inlined_body c
         (inline_renaming (code_return_continuation c)
            (code_exn_continuation c) (ap_continuation ap)
            (ap_exn_continuation ap))
         ap).
Proof.
  destruct c as [params my_closure ret exn metadata body]. unfold inlined_body.
  simpl. rewrite let_params_equivariant. do 2 f_equal.
  rewrite <- !rename_expr_compose. apply rename_expr_ext.
  - intros x. rewrite !rename_variable_compose. reflexivity.
  - intros j. rewrite !rename_continuation_compose.
    apply inline_renaming_conjugate.
Qed.

Lemma let_tuple_fields_equivariant (r : renaming) (params : list variable) :
  ∀ (size field : nat) (tuple : simple) (body : expr),
    rename_expr r (let_tuple_fields params size field tuple body)
    = let_tuple_fields (map (rename_variable r) params) size field
        (rename_simple r tuple) (rename_expr r body).
Proof.
  induction params as [ | param params IH]; intros size field tuple body;
    simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma inlined_tupled_call_equivariant (r : renaming) (c : code)
    (ap : apply_expr) :
  inlined_tupled_call (rename_code r c)
    (inline_renaming (code_return_continuation (rename_code r c))
       (code_exn_continuation (rename_code r c))
       (ap_continuation (rename_apply r ap))
       (ap_exn_continuation (rename_apply r ap)))
    (rename_apply r ap)
  = rename_expr r
      (inlined_tupled_call c
         (inline_renaming (code_return_continuation c)
            (code_exn_continuation c) (ap_continuation ap)
            (ap_exn_continuation ap))
         ap).
Proof.
  destruct c as [params my_closure ret exn metadata body].
  unfold inlined_tupled_call.
  replace (ap_args (rename_apply r ap))
    with (map (rename_simple r) (ap_args ap)) by reflexivity.
  destruct (ap_args ap) as [ | tuple [ | arg args]]; try reflexivity.
  cbn [map]. unfold inlined_tupled_body. simpl.
  rewrite let_tuple_fields_equivariant, length_map. do 2 f_equal.
  rewrite <- !rename_expr_compose. apply rename_expr_ext.
  - intros x. rewrite !rename_variable_compose. reflexivity.
  - intros j. rewrite !rename_continuation_compose.
    apply inline_renaming_conjugate.
Qed.

Lemma inline_apply_equivariant (r : renaming) (kept : list code_id)
    (known : list (code_id * code)) (ap : apply_expr) :
  inline_apply kept (map_values (rename_code r) known) (rename_apply r ap)
  = rename_expr r (inline_apply kept known ap).
Proof.
  unfold inline_apply.
  replace (ap_call_kind (rename_apply r ap)) with (ap_call_kind ap)
    by reflexivity.
  destruct (ap_call_kind ap) as [called | ]; [ | reflexivity ].
  rewrite lookup_map_values.
  destruct (lookup called known) as [c | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite inlinable_b_equivariant, tupled_inlinable_b_equivariant.
  destruct (negb (among called kept) && inlinable_b ap c) eqn:Hinlinable;
    [ apply inlined_body_equivariant | ].
  destruct (negb (among called kept) && tupled_inlinable_b ap c);
    [ apply inlined_tupled_call_equivariant | ].
  rewrite over_applied_b_equivariant.
  destruct (over_applied_b ap c); [ | reflexivity ].
  apply split_over_application_equivariant.
Qed.

Fixpoint inline_calls_equivariant_aux (r : renaming) (kept : list code_id)
    (e : expr) :
  ∀ known : list (code_id * code),
    inline_calls kept (map_values (rename_code r) known)
      (rename_expr r e)
    = rename_expr r (inline_calls kept known e)
with inline_calls_in_handler_equivariant (r : renaming)
    (kept : list code_id) (h : continuation_handler) :
  ∀ known : list (code_id * code),
    inline_calls_in_handler kept (map_values (rename_code r) known)
      (rename_handler r h)
    = rename_handler r (inline_calls_in_handler kept known h)
with inline_calls_in_binding_equivariant (r : renaming)
    (kept : list code_id) (binding : static_binding) :
  ∀ known : list (code_id * code),
    inline_calls_in_binding kept (map_values (rename_code r) known)
      (rename_static_binding r binding)
    = rename_static_binding r (inline_calls_in_binding kept known binding)
with inline_calls_in_code_equivariant (r : renaming) (kept : list code_id)
    (c : code) :
  ∀ known : list (code_id * code),
    inline_calls_in_code kept (map_values (rename_code r) known)
      (rename_code r c)
    = rename_code r (inline_calls_in_code kept known c).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known;
      [ | | | | exact (inline_apply_equivariant r kept known ap) | | | ];
      simpl; try reflexivity.
    + (* Let *)
      rewrite inline_calls_equivariant_aux. reflexivity.
    + (* Let_static: the code known in the body is that of the group,
         rewritten *)
      assert (Hbindings :
        map
          (inline_calls_in_binding kept (map_values (rename_code r) known))
          (map (rename_static_binding r) bindings)
        = map (rename_static_binding r)
            (map (inline_calls_in_binding kept known) bindings)).
      { induction bindings as [ | binding bindings IH]; simpl;
          [ reflexivity | ].
        rewrite (inline_calls_in_binding_equivariant r kept binding), IH.
        reflexivity. }
      rewrite Hbindings, codes_of_equivariant,
        inline_calls_equivariant_aux.
      reflexivity.
    + (* Let_cont_non_recursive *)
      rewrite inline_calls_in_handler_equivariant,
        inline_calls_equivariant_aux.
      reflexivity.
    + (* Let_cont_recursive *)
      rewrite inline_calls_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (inline_calls_in_handler_equivariant r kept h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros known. simpl.
    rewrite inline_calls_equivariant_aux. reflexivity.
  - destruct binding as [id c | id | sym const]; intros known; simpl;
      try reflexivity.
    rewrite inline_calls_in_code_equivariant. reflexivity.
  - destruct c as [params my_closure ret exn metadata body]. intros known.
    simpl.
    rewrite inline_calls_equivariant_aux. reflexivity.
Qed.

Theorem inline_calls_equivariant : inline_calls_equivariant_statement.
Proof.
  intros r kept known e. apply inline_calls_equivariant_aux.
Qed.
