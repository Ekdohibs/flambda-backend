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

(* Proofs of the statements of FlambdaRecHandlersSpec.v. Nothing here
   needs review.

   Each layer of the term is a context around what the induction gives
   (FlambdaContextProofs.plug_refines). At a recursive [Let_cont] the
   handlers are replaced all at once
   (FlambdaContextProofs.let_cont_recursive_handlers_refine_where),
   which asks that each refines its counterpart where it runs, in the
   two directions; that is the induction followed by what is assumed of
   the rewrite. The equivalence is the refinement both ways. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaRecHandlersSpec.
Import ListNotations.

(* What [rewrite_code_body] gives binds something static exactly if what
   it was given does. *)
Lemma rewrite_code_body_binds_statics (f : expr -> expr) (body : expr) :
  binds_statics (rewrite_code_body f body) = binds_statics body.
Proof.
  unfold rewrite_code_body.
  destruct (Bool.eqb (binds_statics body) (binds_statics (f body)))
    eqn:Hbinds; [ | reflexivity ].
  symmetry. apply eqb_prop, Hbinds.
Qed.

Fixpoint rewrite_rec_handlers_binds_statics (f : expr -> expr) (e : expr) :
  binds_statics (rewrite_rec_handlers f e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - apply rewrite_rec_handlers_binds_statics.
  - rewrite rewrite_rec_handlers_binds_statics. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite !rewrite_rec_handlers_binds_statics. reflexivity.
  - rewrite rewrite_rec_handlers_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
      [ reflexivity | ].
    rewrite rewrite_code_body_binds_statics,
      rewrite_rec_handlers_binds_statics, IH.
    reflexivity.
Qed.

Lemma rewrite_rec_handlers_bind_statics (f : expr -> expr)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, rewrite_rec_handler f h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite rewrite_code_body_binds_statics,
    rewrite_rec_handlers_binds_statics, IH.
  reflexivity.
Qed.

(* [rewrite_code_body] refines, in either direction, where the rewrite
   does. *)
Lemma rewrite_code_body_refines (f : expr -> expr) (body : expr) :
  (∀ e : expr, expr_refines e (f e)) ->
  expr_refines body (rewrite_code_body f body).
Proof.
  intros Hf. unfold rewrite_code_body.
  destruct (Bool.eqb (binds_statics body) (binds_statics (f body)));
    [ apply Hf | apply expr_refines_refl ].
Qed.

Lemma rewrite_code_body_refined (f : expr -> expr) (body : expr) :
  (∀ e : expr, expr_refines (f e) e) ->
  expr_refines (rewrite_code_body f body) body.
Proof.
  intros Hf. unfold rewrite_code_body.
  destruct (Bool.eqb (binds_statics body) (binds_statics (f body)));
    [ apply Hf | apply expr_refines_refl ].
Qed.

(* The layers that are not a recursive [Let_cont], in the two directions:
   [IH] is the induction. *)
Ltac rewrite_rec_handlers_cases IH :=
  match goal with
  | |- expr_refines (Let ?x ?kind ?d ?body) (Let _ _ _ ?body') =>
    first
      [ exact (plug_refines (Let_body x kind d Hole) _ _ (not_many _)
          (IH body))
      | exact (plug_refines (Let_body x kind d Hole) _ _ (not_many _)
          (IH body')) ]
  | |- expr_refines (Let_static ?bindings ?body) (Let_static _ ?body') =>
    first
      [ exact (plug_refines (Let_static_body bindings Hole) _ _ (not_many _)
          (IH body))
      | exact (plug_refines (Let_static_body bindings Hole) _ _ (not_many _)
          (IH body')) ]
  | |- expr_refines
         (Let_cont_non_recursive ?k (Handler ?params ?metadata ?handler) ?body)
         (Let_cont_non_recursive _ (Handler _ _ ?handler') ?body') =>
    apply (expr_refines_trans _
      (Let_cont_non_recursive k (Handler params metadata handler') body));
      [ first
          [ exact (plug_refines
              (Let_cont_non_recursive_handler k params metadata Hole body) _ _
              (not_many _) (IH handler))
          | exact (plug_refines
              (Let_cont_non_recursive_handler k params metadata Hole body) _ _
              (not_many _) (IH handler')) ]
      | first
          [ exact (plug_refines
              (Let_cont_non_recursive_body k
                 (Handler params metadata handler') Hole)
              _ _ (not_many _) (IH body))
          | exact (plug_refines
              (Let_cont_non_recursive_body k
                 (Handler params metadata handler') Hole)
              _ _ (not_many _) (IH body')) ] ]
  end.

Fixpoint rewrite_rec_handlers_refines_aux (f : expr -> expr)
    (Hf : ∀ e : expr, expr_refines e (f e)) (e : expr) {struct e} :
  expr_refines e (rewrite_rec_handlers f e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try apply expr_refines_refl;
    try (destruct h as [params metadata handler]; simpl);
    try (rewrite_rec_handlers_cases (rewrite_rec_handlers_refines_aux f Hf);
         fail).
  assert (Hhandlers :
    ∀ vars : env,
      handlers_refine always vars handlers
        (map (fun '(k, h) => (k, rewrite_rec_handler f h)) handlers)).
  { induction handlers as [ | [k [params metadata handler]] handlers IH];
      intros vars; simpl.
    - split; apply Forall2_nil.
    - destruct (IH vars) as [Hforwards Hbackwards].
      split; (apply Forall2_cons; [ | assumption ]);
        (split; [ reflexivity | ]); apply Handler_related;
        intros vs handler_vars conts statics hp rd _ _ _;
        exact (expr_refines_trans _ _ _
          (rewrite_rec_handlers_refines_aux f Hf handler)
          (rewrite_code_body_refines f _ Hf) handler_vars conts statics hp rd_).
            }
  eapply expr_refines_trans;
    [ | exact (plug_refines (Let_cont_recursive_body _ Hole) _ _ (not_many _)
          (rewrite_rec_handlers_refines_aux f Hf body)) ].
  intros vars conts statics hp rd.
  destruct (Hhandlers vars) as [Hforwards Hbackwards].
  apply (let_cont_recursive_handlers_refine_where always always_extend);
    [ exact I | symmetry; apply rewrite_rec_handlers_bind_statics
    | exact Hforwards | exact Hbackwards ].
Qed.

Fixpoint rewrite_rec_handlers_refined_aux (f : expr -> expr)
    (Hf : ∀ e : expr, expr_refines (f e) e) (e : expr) {struct e} :
  expr_refines (rewrite_rec_handlers f e) e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try apply expr_refines_refl;
    try (destruct h as [params metadata handler]; simpl);
    try (rewrite_rec_handlers_cases (rewrite_rec_handlers_refined_aux f Hf);
         fail).
  assert (Hhandlers :
    ∀ vars : env,
      handlers_refine always vars
        (map (fun '(k, h) => (k, rewrite_rec_handler f h)) handlers)
        handlers).
  { induction handlers as [ | [k [params metadata handler]] handlers IH];
      intros vars; simpl.
    - split; apply Forall2_nil.
    - destruct (IH vars) as [Hforwards Hbackwards].
      split; (apply Forall2_cons; [ | assumption ]);
        (split; [ reflexivity | ]); apply Handler_related;
        intros vs handler_vars conts statics hp rd _ _ _;
        exact (expr_refines_trans _ _ _
          (rewrite_code_body_refined f _ Hf)
          (rewrite_rec_handlers_refined_aux f Hf handler)
          handler_vars conts statics hp rd_). }
  eapply expr_refines_trans;
    [ | exact (plug_refines (Let_cont_recursive_body _ Hole) _ _ (not_many _)
          (rewrite_rec_handlers_refined_aux f Hf body)) ].
  intros vars conts statics hp rd.
  destruct (Hhandlers vars) as [Hforwards Hbackwards].
  apply (let_cont_recursive_handlers_refine_where always always_extend);
    [ exact I | apply rewrite_rec_handlers_bind_statics
    | exact Hforwards | exact Hbackwards ].
Qed.

Theorem rewrite_rec_handlers_refines : rewrite_rec_handlers_refines_statement.
Proof. exact rewrite_rec_handlers_refines_aux. Qed.

Theorem rewrite_rec_handlers_equivalent :
  rewrite_rec_handlers_equivalent_statement.
Proof.
  intros f Hf e vars conts statics hp rd. apply refines_antisymmetric.
  - apply (rewrite_rec_handlers_refines_aux f).
    intros e'. apply expr_equivalent_refines, Hf.
  - apply (rewrite_rec_handlers_refined_aux f).
    intros e' vars' conts' statics' hp' rd'.
    apply equivalent_refines, equivalent_sym, Hf.
Qed.
