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

(* Proofs of the statements of FlambdaFloatCodeSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextSpec FlambdaContextProofs FlambdaStaticSpec
  FlambdaStaticProofs FlambdaNominalSpec FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaFloatCodeSpec.
Import ListNotations.

(* ======================================================================== *)
(* Binding code                                                             *)
(* ======================================================================== *)

(* Binding code reads no variable. *)
Lemma eval_code_bindings_vars (vars vars' : env)
    (codes : list static_binding) :
  forallb binds_code codes = true ->
  ∀ statics : static_env,
    eval_static_bindings vars codes statics
    = eval_static_bindings vars' codes statics.
Proof.
  induction codes as [ | binding codes IH]; intros Hcodes statics;
    [ reflexivity | ].
  simpl in Hcodes. apply andb_true_iff in Hcodes as [Hbinding Hcodes].
  destruct binding as [id c | id | sym const]; simpl in *;
    try discriminate.
  - destruct (lookup id (static_code statics)); [ reflexivity | ].
    apply IH, Hcodes.
  - apply IH, Hcodes.
Qed.

(* It binds no symbol. *)
Lemma eval_code_bindings_values (vars : env) (codes : list static_binding) :
  forallb binds_code codes = true ->
  ∀ statics statics' : static_env,
    eval_static_bindings vars codes statics = Some statics' ->
    static_values statics' = static_values statics.
Proof.
  induction codes as [ | binding codes IH]; intros Hcodes statics statics';
    simpl; [ intros [= <-]; reflexivity | ].
  simpl in Hcodes. apply andb_true_iff in Hcodes as [Hbinding Hcodes].
  destruct binding as [id c | id | sym const]; simpl in Hbinding;
    try discriminate.
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    intros Heval. exact (IH Hcodes _ _ Heval).
  - apply IH, Hcodes.
Qed.

(* A defining expression reads the statically allocated values only. *)
Lemma eval_named_same_values (statics statics' : static_env) (vars : env)
    (d : named) :
  static_values statics = static_values statics' ->
  eval_named statics vars d = eval_named statics' vars d.
Proof.
  intros Hvalues.
  destruct d as
    [s | p args | code slots | slot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    simpl; try reflexivity.
  - destruct (eval_simples vars args) as [vs | ]; simpl; [ | reflexivity ].
    apply eval_prim_same_static_values, Hvalues.
  - destruct (eval_simple vars closure) as [v | ]; simpl; [ | reflexivity ].
    unfold closure_of. rewrite Hvalues. reflexivity.
  - destruct (eval_simple vars closure) as [v | ]; simpl; [ | reflexivity ].
    unfold project_value_slot, closure_in_set. rewrite Hvalues. reflexivity.
  - destruct (eval_simple vars closure) as [v | ]; simpl; [ | reflexivity ].
    unfold project_function_slot. rewrite Hvalues. reflexivity.
Qed.

(* And it does the same on a heap. *)
Lemma exec_named_same_values (statics statics' : static_env) (hp : heap)
    (rd : read_record)
    (vars : env) (d : named) :
  static_values statics = static_values statics' ->
  exec_named statics hp rd vars d = exec_named statics' hp rd vars d.
Proof.
  intros Hvalues. symmetry. apply exec_named_same.
  - symmetry. exact (eval_named_same_values statics statics' vars d Hvalues).
  - intros p args _. split; [ reflexivity | ].
    intros vs s _ _. rewrite Hvalues. reflexivity.
Qed.

(* ======================================================================== *)
(* Code over one binder                                                     *)
(* ======================================================================== *)

Theorem float_code_over_let : float_code_over_let_statement.
Proof.
  intros x kind d codes body Hcodes vars conts statics hp rd.
  destruct (eval_static_bindings vars codes statics) as [statics' | ]
    eqn:Hbind.
  - pose proof (eval_code_bindings_values vars codes Hcodes statics statics'
      Hbind) as Hvalues.
    destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hd;
      [ destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind | ].
    + eapply equivalent_step;
        [ rewrite step_let_exec, Hd, Hkind; reflexivity
        | rewrite step_let_static, Hbind; reflexivity | ].
      eapply equivalent_step;
        [ rewrite step_let_static,
            (eval_code_bindings_vars ((x, v) :: vars) vars codes Hcodes
               statics), Hbind;
          reflexivity
        | rewrite step_let_exec,
            (exec_named_same_values statics' statics hp rd vars d Hvalues), Hd,
            Hkind;
          reflexivity
        | apply equivalent_refl ].
    + apply equivalent_sym.
      eapply equivalent_trans;
        [ apply step_equivalent; rewrite step_let_static, Hbind;
          reflexivity | ].
      apply step_eq_equivalent.
      rewrite !step_let_exec,
        (exec_named_same_values statics' statics hp rd vars d Hvalues), Hd,
          Hkind.
      reflexivity.
    + apply equivalent_sym.
      eapply equivalent_trans;
        [ apply step_equivalent; rewrite step_let_static, Hbind;
          reflexivity | ].
      apply step_eq_equivalent.
      rewrite !step_let_exec,
        (exec_named_same_values statics' statics hp rd vars d Hvalues), Hd.
      reflexivity.
  - destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hd;
      [ destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind | ].
    + eapply equivalent_trans;
        [ apply step_equivalent; rewrite step_let_exec, Hd, Hkind; reflexivity
        | ].
      apply step_eq_equivalent.
      rewrite !step_let_static,
        (eval_code_bindings_vars ((x, v) :: vars) vars codes Hcodes statics),
        Hbind.
      reflexivity.
    + apply step_eq_equivalent.
      rewrite step_let_exec, Hd, Hkind, step_let_static, Hbind. reflexivity.
    + apply step_eq_equivalent.
      rewrite step_let_exec, Hd, step_let_static, Hbind. reflexivity.
Qed.

(* A [Let_cont] binds no variable, so any group may go over it. *)
Lemma float_over_let_cont (k : continuation) (h : continuation_handler)
    (bindings : list static_binding) (body : expr) :
  expr_equivalent (Let_cont_non_recursive k h (Let_static bindings body))
    (Let_static bindings (Let_cont_non_recursive k h body)).
Proof.
  intros vars conts statics hp rd.
  destruct (eval_static_bindings vars bindings statics) as [statics' | ]
    eqn:Hbind.
  - eapply equivalent_step;
      [ apply step_let_cont_non_recursive
      | rewrite step_let_static, Hbind; reflexivity | ].
    eapply equivalent_step;
      [ rewrite step_let_static, Hbind; reflexivity
      | apply step_let_cont_non_recursive | apply equivalent_refl ].
  - eapply equivalent_trans;
      [ apply step_equivalent, step_let_cont_non_recursive | ].
    apply step_eq_equivalent. rewrite !step_let_static, Hbind. reflexivity.
Qed.

Lemma float_over_let_cont_recursive
    (handlers : list (continuation * continuation_handler))
    (bindings : list static_binding) (body : expr) :
  expr_equivalent (Let_cont_recursive handlers (Let_static bindings body))
    (Let_static bindings (Let_cont_recursive handlers body)).
Proof.
  intros vars conts statics hp rd.
  destruct (handlers_bind_statics handlers) eqn:Hhandlers;
    destruct (eval_static_bindings vars bindings statics) as [statics' | ]
      eqn:Hbind.
  - apply equivalent_sym.
    eapply equivalent_trans;
      [ apply step_equivalent; rewrite step_let_static, Hbind; reflexivity
      | ].
    apply step_eq_equivalent.
    rewrite !step_let_cont_recursive, Hhandlers. reflexivity.
  - apply step_eq_equivalent.
    rewrite step_let_cont_recursive, Hhandlers, step_let_static, Hbind.
    reflexivity.
  - eapply equivalent_step;
      [ rewrite step_let_cont_recursive, Hhandlers; reflexivity
      | rewrite step_let_static, Hbind; reflexivity | ].
    eapply equivalent_step;
      [ rewrite step_let_static, Hbind; reflexivity
      | rewrite step_let_cont_recursive, Hhandlers; reflexivity
      | apply equivalent_refl ].
  - eapply equivalent_trans;
      [ apply step_equivalent; rewrite step_let_cont_recursive, Hhandlers;
        reflexivity | ].
    apply step_eq_equivalent. rewrite !step_let_static, Hbind. reflexivity.
Qed.

(* ======================================================================== *)
(* Groups                                                                   *)
(* ======================================================================== *)

Lemma let_static_if_any_same (bindings : list static_binding) (body : expr) :
  expr_equivalent (Let_static bindings body)
    (let_static_if_any bindings body).
Proof.
  intros vars conts statics hp rd.
  destruct bindings as [ | binding bindings]; simpl;
    [ | apply equivalent_refl ].
  apply step_equivalent. reflexivity.
Qed.

(* A group is its code, then the rest. *)
Lemma split_group (bindings : list static_binding) (body : expr) :
  expr_equivalent (Let_static bindings body)
    (Let_static (filter binds_code bindings)
       (Let_static
          (filter (fun binding => negb (binds_code binding)) bindings)
          body)).
Proof.
  apply (expr_equivalent_trans _ _ _ (code_first_equivalent bindings body)).
  intros vars conts statics. unfold code_first. apply let_static_split.
Qed.

Lemma float_over_equivalent (wrap : expr -> expr) :
  (∀ e e' : expr,
    expr_equivalent e e' -> expr_equivalent (wrap e) (wrap e')) ->
  (∀ (codes : list static_binding) (body : expr),
    forallb binds_code codes = true ->
    expr_equivalent (wrap (Let_static codes body))
      (Let_static codes (wrap body))) ->
  ∀ inner : expr, expr_equivalent (wrap inner) (float_over wrap inner).
Proof.
  intros Hcong Hfloat inner.
  destruct inner
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    unfold float_over; try
      (intros vars conts statics hp rd; apply equivalent_refl).
  set (codes := filter binds_code bindings).
  set (rest := filter (fun binding => negb (binds_code binding)) bindings).
  apply (expr_equivalent_trans _
    (wrap (Let_static codes (Let_static rest body)))).
  { apply Hcong, split_group. }
  apply (expr_equivalent_trans _
    (Let_static codes (wrap (Let_static rest body)))).
  { apply Hfloat, filter_binds_code. }
  apply (expr_equivalent_trans _
    (Let_static codes (wrap (let_static_if_any rest body)))).
  { exact (plug_equivalent (Let_static_body codes Hole) _ _ (not_many _)
      (Hcong _ _ (let_static_if_any_same rest body))). }
  apply let_static_if_any_same.
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

Fixpoint float_code_equivalent_aux (e : expr) :
  expr_equivalent e (float_code e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [float_code];
    try (intros vars conts statics hp rd; apply equivalent_refl).
  - apply (expr_equivalent_trans _ (Let x kind d (float_code body))).
    + exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
        (float_code_equivalent_aux body)).
    + apply (float_over_equivalent (Let x kind d)).
      * intros e e' H.
        exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _) H).
      * intros codes body' Hcodes. apply float_code_over_let, Hcodes.
  - pose proof (plug_equivalent (Let_static_body bindings Hole) _ _
      (not_many _) (float_code_equivalent_aux body)) as Hbody.
    simpl in Hbody.
    destruct (float_code body)
      as [ x kind' d body' | bindings' body' | k h body' | handlers body' | ap
         | ac | scrutinee arms | ];
      try (apply (expr_equivalent_trans _ _ _ Hbody), code_first_equivalent).
    apply (expr_equivalent_trans _ _ _ Hbody).
    apply (expr_equivalent_trans _
      (Let_static (bindings ++ bindings') body')).
    + intros vars conts statics hp rd.
      apply equivalent_sym, let_static_split.
    + apply code_first_equivalent.
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (float_code handler)) body)).
    { exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole body)
        _ _ (not_many _) (float_code_equivalent_aux handler)). }
    apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (float_code handler)) (float_code body))).
    { exact (plug_equivalent
        (Let_cont_non_recursive_body k
           (Handler params metadata (float_code handler)) Hole)
        _ _ (not_many _) (float_code_equivalent_aux body)). }
    apply (float_over_equivalent
      (Let_cont_non_recursive k
         (Handler params metadata (float_code handler)))).
    + intros e e' H.
      exact (plug_equivalent
        (Let_cont_non_recursive_body k
           (Handler params metadata (float_code handler)) Hole)
        _ _ (not_many _) H).
    + intros codes body' _. apply float_over_let_cont.
  - apply (expr_equivalent_trans _
      (Let_cont_recursive handlers (float_code body))).
    { exact (plug_equivalent (Let_cont_recursive_body handlers Hole)
        _ _ (not_many _) (float_code_equivalent_aux body)). }
    apply (float_over_equivalent (Let_cont_recursive handlers)).
    + intros e e' H.
      exact (plug_equivalent (Let_cont_recursive_body handlers Hole)
        _ _ (not_many _) H).
    + intros codes body' _. apply float_over_let_cont_recursive.
Qed.

Theorem float_code_equivalent : float_code_equivalent_statement.
Proof. exact float_code_equivalent_aux. Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Lemma binds_code_rename (r : renaming) (binding : static_binding) :
  binds_code (rename_static_binding r binding) = binds_code binding.
Proof. destruct binding; reflexivity. Qed.

Lemma filter_rename_bindings (r : renaming) (f : static_binding -> bool)
    (bindings : list static_binding) :
  (∀ binding : static_binding,
    f (rename_static_binding r binding) = f binding) ->
  filter f (map (rename_static_binding r) bindings)
  = map (rename_static_binding r) (filter f bindings).
Proof.
  intros Hf.
  induction bindings as [ | binding bindings IH]; [ reflexivity | ].
  cbn [map filter]. rewrite Hf, IH. destruct (f binding); reflexivity.
Qed.

Lemma filter_code_rename (r : renaming) (bindings : list static_binding) :
  filter binds_code (map (rename_static_binding r) bindings)
  = map (rename_static_binding r) (filter binds_code bindings).
Proof. apply filter_rename_bindings, binds_code_rename. Qed.

Lemma filter_rest_rename (r : renaming) (bindings : list static_binding) :
  filter (fun binding => negb (binds_code binding))
    (map (rename_static_binding r) bindings)
  = map (rename_static_binding r)
      (filter (fun binding => negb (binds_code binding)) bindings).
Proof.
  apply filter_rename_bindings. intros binding. cbv beta.
  rewrite binds_code_rename. reflexivity.
Qed.

Lemma code_first_rename (r : renaming) (bindings : list static_binding) :
  code_first (map (rename_static_binding r) bindings)
  = map (rename_static_binding r) (code_first bindings).
Proof.
  unfold code_first.
  rewrite map_app, filter_code_rename, filter_rest_rename. reflexivity.
Qed.

Lemma let_static_if_any_rename (r : renaming)
    (bindings : list static_binding) (body : expr) :
  rename_expr r (let_static_if_any bindings body)
  = let_static_if_any (map (rename_static_binding r) bindings)
      (rename_expr r body).
Proof. destruct bindings; reflexivity. Qed.

Lemma float_over_rename (r : renaming) (wrap wrap' : expr -> expr) :
  (∀ e : expr, rename_expr r (wrap e) = wrap' (rename_expr r e)) ->
  ∀ inner : expr,
    rename_expr r (float_over wrap inner)
    = float_over wrap' (rename_expr r inner).
Proof.
  intros Hwrap inner.
  destruct inner
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    try (unfold float_over; rewrite Hwrap; reflexivity).
  cbn [float_over rename_expr].
  rewrite let_static_if_any_rename, Hwrap, let_static_if_any_rename,
    filter_code_rename, filter_rest_rename.
  reflexivity.
Qed.

Fixpoint float_code_rename (r : renaming) (e : expr) {struct e} :
  rename_expr r (float_code e) = float_code (rename_expr r e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    try reflexivity.
  - cbn [float_code rename_expr]. rewrite <- float_code_rename.
    apply float_over_rename. intros e'. reflexivity.
  - cbn [float_code rename_expr]. rewrite <- float_code_rename.
    destruct (float_code body)
      as [ x kind' d body' | bindings' body' | k h body' | handlers body' | ap
         | ac | scrutinee arms | ];
      cbn [rename_expr]; rewrite <- ?map_app, code_first_rename; reflexivity.
  - cbn [float_code rename_expr rename_handler].
    rewrite <- !float_code_rename.
    apply float_over_rename. intros e'. reflexivity.
  - cbn [float_code rename_expr]. rewrite <- float_code_rename.
    apply float_over_rename. intros e'. reflexivity.
Qed.

Theorem float_code_equivariant : float_code_equivariant_statement.
Proof. exact float_code_rename. Qed.
