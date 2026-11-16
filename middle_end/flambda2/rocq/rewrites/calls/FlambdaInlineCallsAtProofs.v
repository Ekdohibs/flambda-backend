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


(* Proofs of the statements of FlambdaInlineCallsAtSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaNominalSpec FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaLiftSpec
  FlambdaInlineCallsAtSpec.
Import ListNotations.

(* ======================================================================== *)
(* What binds something static is as it was                                 *)
(* ======================================================================== *)

(* As FlambdaNormaliseProofs.inline_calls_binds_statics: what is inlined
   binds nothing. *)
Fixpoint inline_calls_at_paths_binds_statics (K : calls_in_code)
    (sites : list path) (known : list (code_id * code)) (at_path : path)
    (e : expr) {struct e} :
  binds_statics (inline_calls_at_paths K sites known at_path e)
  = binds_statics e
with handler_inline_calls_at_paths_binds_statics (K : calls_in_code)
    (sites : list path) (known : list (code_id * code)) (at_path : path)
    (h : continuation_handler) {struct h} :
  handler_binds_statics (handler_inline_calls_at_paths K sites known at_path h)
  = handler_binds_statics h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body
         | ap | ac | scrutinee arms | ];
      simpl; try reflexivity.
    + apply inline_calls_at_paths_binds_statics.
    + rewrite inline_calls_at_paths_binds_statics. f_equal.
      induction bindings
        as [ | [id [params my_closure ret exn metadata body'] | id
               | sym const]
               bindings IH];
        simpl; congruence.
    + rewrite handler_inline_calls_at_paths_binds_statics,
        inline_calls_at_paths_binds_statics.
      reflexivity.
    + rewrite inline_calls_at_paths_binds_statics. f_equal.
      generalize (xI at_path).
      induction handlers as [ | [k1 h1] handlers IH]; intros q;
        [ reflexivity | ].
      cbn [rec_handlers_map existsb].
      rewrite handler_inline_calls_at_paths_binds_statics, IH. reflexivity.
    + destruct (among at_path sites); [ | reflexivity ].
      apply inline_apply_binds_statics.
  - destruct h as [params metadata handler]. simpl.
    apply inline_calls_at_paths_binds_statics.
Qed.

Lemma handlers_inline_calls_at_paths_bind_statics (K : calls_in_code)
    (sites : list path) (known : list (code_id * code)) :
  ∀ (handlers : list (continuation * continuation_handler)) (q : path),
    handlers_bind_statics
      (rec_handlers_map (handler_inline_calls_at_paths K sites known) q
         handlers)
    = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k h] handlers IH]; intros q; [ reflexivity | ].
  cbn [rec_handlers_map existsb].
  rewrite handler_inline_calls_at_paths_binds_statics, IH. reflexivity.
Qed.

(* ======================================================================== *)
(* The result refines the term                                              *)
(* ======================================================================== *)

(* Down the term, with the static environment binding what is known: at
   once in a body, and from then on in a handler. The proof is that of
   FlambdaNormaliseProofs.inline_calls_refines_knowing, with the places
   carried down; a call is given to [inline_apply] or left. *)
Fixpoint inline_calls_at_paths_refines_knowing (e : expr) :
  ∀ (K : calls_in_code) (sites : list path) (known : list (code_id * code))
    (at_path : path) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record),
    statics_know known statics ->
    refines (State e vars conts statics hp rd)
      (State (inline_calls_at_paths K sites known at_path e) vars conts
         statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body
         | ap | ac | scrutinee arms | ];
      intros K sites known at_path vars conts statics hp rd Hknow;
      cbn [inline_calls_at_paths]; try apply refines_refl.
    + apply let_refines_step_exec; [ reflexivity | intros v hp' rd' _ _ ].
      apply inline_calls_at_paths_refines_knowing, Hknow.
    + (* the code of the group is rewritten: the body under the one
         static environment refines it under the other; then down the
         body, the static environment binding the code as rewritten *)
      assert (Hrefine :
        bindings_refine (statics_know known) bindings
          (map (binding_inline_calls_at_paths K known) bindings)).
      { induction bindings
          as [ | [id [params my_closure ret exn metadata code_body] | id
                 | sym const]
                 bindings IH];
          cbn [map binding_inline_calls_at_paths].
        - split; apply Forall2_nil.
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ | assumption ]);
            apply Binding_code, Codes_related;
            first
              [ apply inline_calls_at_paths_binds_statics
              | symmetry; apply inline_calls_at_paths_binds_statics
              | intros vars' conts' statics' hp' rd' Hknow';
                apply inline_calls_at_paths_refines_knowing, Hknow' ].
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ apply Binding_same | assumption ]).
        - destruct IH as [Hforwards Hbackwards].
          split; (apply Forall2_cons; [ apply Binding_same | assumption ]). }
      cbv zeta.
      remember (map (binding_inline_calls_at_paths K known) bindings)
        as bindings' eqn:Hdef.
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
      * apply inline_calls_at_paths_refines_knowing.
        exact (eval_static_bindings_know vars bindings' known statics
          extended' Hknow Hbindings').
    + destruct h as [params metadata handler].
      cbn [handler_inline_calls_at_paths].
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k
              (Handler params metadata
                 (inline_calls_at_paths K sites known (xI at_path) handler))
              body)
           vars conts statics hp rd)).
      * apply (let_cont_handler_refines_where (statics_know known)
          (statics_know_extend known)); [ exact Hknow | ].
        intros vs handler_vars conts' statics' hp' rd' Hknow' _ _.
        apply inline_calls_at_paths_refines_knowing, Hknow'.
      * apply let_cont_non_recursive_refines_step,
          inline_calls_at_paths_refines_knowing, Hknow.
    + apply (refines_trans _
        (State
           (Let_cont_recursive
              (rec_handlers_map (handler_inline_calls_at_paths K sites known)
                 (xI at_path) handlers)
              body)
           vars conts statics hp rd)).
      * assert (Hhandlers :
          ∀ q : path,
            handlers_refine (statics_know known) vars handlers
              (rec_handlers_map (handler_inline_calls_at_paths K sites known)
                 q handlers)).
        { induction handlers
            as [ | [k [params metadata handler]] handlers IH];
            intros q;
            cbn [rec_handlers_map handler_inline_calls_at_paths].
          - split; apply Forall2_nil.
          - destruct (IH (xI q)) as [Hforwards Hbackwards].
            split; (apply Forall2_cons; [ | assumption ]);
              (split; [ reflexivity | ]); apply Handler_related;
              intros vs handler_vars conts' statics' hp' rd' Hknow' _ _;
              apply inline_calls_at_paths_refines_knowing, Hknow'. }
        destruct (Hhandlers (xI at_path)) as [Hforwards Hbackwards].
        apply (let_cont_recursive_handlers_refine_where (statics_know known)
          (statics_know_extend known));
          try assumption.
        symmetry. apply handlers_inline_calls_at_paths_bind_statics.
      * apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
        apply inline_calls_at_paths_refines_knowing, Hknow.
    + destruct (among at_path sites); [ | apply refines_refl ].
      apply inline_apply_refines, Hknow.
Qed.

Theorem inline_calls_at_refines : inline_calls_at_refines_statement.
Proof.
  intros sites K e vars conts statics hp rd. unfold inline_calls_at.
  apply inline_calls_at_paths_refines_knowing.
  intros id c Hlookup. discriminate Hlookup.
Qed.

(* ======================================================================== *)
(* The rewrite commutes with renaming                                       *)
(* ======================================================================== *)

(* As FlambdaEquivarianceProofs.inline_calls_equivariant_aux: the code
   known is renamed with the term. *)
Fixpoint inline_calls_at_paths_equivariant (r : renaming)
    (K : calls_in_code) (e : expr) :
  ∀ (sites : list path) (known : list (code_id * code)) (at_path : path),
    inline_calls_at_paths K sites (map_values (rename_code r) known) at_path
      (rename_expr r e)
    = rename_expr r (inline_calls_at_paths K sites known at_path e)
with handler_inline_calls_at_paths_equivariant (r : renaming)
    (K : calls_in_code) (h : continuation_handler) :
  ∀ (sites : list path) (known : list (code_id * code)) (at_path : path),
    handler_inline_calls_at_paths K sites (map_values (rename_code r) known)
      at_path (rename_handler r h)
    = rename_handler r (handler_inline_calls_at_paths K sites known at_path h)
with binding_inline_calls_at_paths_equivariant (r : renaming)
    (K : calls_in_code) (binding : static_binding) :
  ∀ known : list (code_id * code),
    binding_inline_calls_at_paths K (map_values (rename_code r) known)
      (rename_static_binding r binding)
    = rename_static_binding r (binding_inline_calls_at_paths K known binding).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros sites known at_path; simpl; try reflexivity.
    + (* Let *)
      rewrite inline_calls_at_paths_equivariant. reflexivity.
    + (* Let_static: the code known in the body is that of the group,
         rewritten *)
      assert (Hbindings :
        map
          (binding_inline_calls_at_paths K
             (map_values (rename_code r) known))
          (map (rename_static_binding r) bindings)
        = map (rename_static_binding r)
            (map (binding_inline_calls_at_paths K known) bindings)).
      { induction bindings as [ | binding bindings IH]; simpl;
          [ reflexivity | ].
        rewrite (binding_inline_calls_at_paths_equivariant r K binding), IH.
        reflexivity. }
      rewrite Hbindings, codes_of_equivariant,
        inline_calls_at_paths_equivariant.
      reflexivity.
    + (* Let_cont_non_recursive *)
      rewrite handler_inline_calls_at_paths_equivariant,
        inline_calls_at_paths_equivariant.
      reflexivity.
    + (* Let_cont_recursive *)
      rewrite inline_calls_at_paths_equivariant. f_equal.
      generalize (xI at_path).
      induction handlers as [ | [k h] handlers IH]; intros q;
        [ reflexivity | ].
      cbn [rec_handlers_map map].
      rewrite (handler_inline_calls_at_paths_equivariant r K h), IH.
      reflexivity.
    + (* Apply *)
      destruct (among at_path sites); [ | reflexivity ].
      exact (inline_apply_equivariant r [] known ap).
  - destruct h as [params metadata handler]. intros sites known at_path.
    simpl.
    rewrite inline_calls_at_paths_equivariant. reflexivity.
  - destruct binding
      as [id [params my_closure ret exn metadata body] | id | sym const];
      intros known; simpl; try reflexivity.
    rewrite inline_calls_at_paths_equivariant. reflexivity.
Qed.

Theorem inline_calls_at_equivariant : inline_calls_at_equivariant_statement.
Proof.
  intros r sites K e. unfold inline_calls_at.
  exact (inline_calls_at_paths_equivariant r K e sites [] root).
Qed.
