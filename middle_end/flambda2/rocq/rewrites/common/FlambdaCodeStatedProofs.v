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

(* Proofs for FlambdaCodeStatedSpec.v. Nothing here needs review.

   The body of code and the body as rewritten are compared where the
   code is entered: what the code states of its parameters holds there
   (FlambdaStatedScopeProofs.code_stated_bound), so the premise on the
   rewrite applies. A group of static bindings is then replaced by the
   group with its bodies rewritten
   (FlambdaStatedContextProofs.let_static_bindings_stated_refine), and
   the other layers are contexts around what the induction gives, as
   in the proof of FlambdaNormaliseSpec.rewrite_code. The two
   directions are proved alike; the equivalence is the two together. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaContextSpec
  FlambdaContextProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaStatedScopeSpec FlambdaStatedScopeProofs FlambdaStatedContextProofs
  FlambdaRecHandlersSpec FlambdaRecHandlersProofs FlambdaCodeStatedSpec.
Import ListNotations.

(* The rewrite keeps whether a term binds something static. *)
Fixpoint rewrite_code_stated_binds_statics
    (g : stated_in_scope -> expr -> expr) (e : expr) {struct e} :
  binds_statics (rewrite_code_stated g e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - apply rewrite_code_stated_binds_statics.
  - rewrite rewrite_code_stated_binds_statics. f_equal.
    induction bindings
      as [ | [id [params my_closure ret exn metadata code_body] | id
             | sym const]
             bindings IH];
      simpl; congruence.
  - destruct h as [params metadata handler]. simpl.
    rewrite !rewrite_code_stated_binds_statics. reflexivity.
  - rewrite rewrite_code_stated_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite rewrite_code_stated_binds_statics, IH. reflexivity.
Qed.

Lemma rewrite_code_stated_handlers_bind_statics
    (g : stated_in_scope -> expr -> expr)
    (handlers : list (continuation * continuation_handler)) :
  handlers_bind_statics
    (map (fun '(k, h) => (k, rewrite_code_stated_in_handler g h)) handlers)
  = handlers_bind_statics handlers.
Proof.
  unfold handlers_bind_statics.
  induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
    [ reflexivity | ].
  rewrite rewrite_code_stated_binds_statics, IH. reflexivity.
Qed.

(* The body of code is refined by the body as rewritten where the code
   is entered: what the code states holds there. *)
Lemma code_body_stated_refines (g : stated_in_scope -> expr -> expr)
    (Hg : ∀ (S : stated_in_scope) (e : expr),
        expr_refines_stated S e (g S e))
    (params : list variable) (my_closure : variable)
    (metadata : code_metadata) (body : expr) :
  code_refines_stated_where always params my_closure metadata body
    (rewrite_code_body (g (code_stated params my_closure metadata)) body).
Proof.
  intros vs callee callee_vars conts statics hp rd _ Hkinds Hclosure Hstated
    Hbind.
  unfold rewrite_code_body.
  destruct
    (Bool.eqb (binds_statics body)
       (binds_statics (g (code_stated params my_closure metadata) body)));
    [ | apply refines_refl ].
  apply Hg.
  exact
    (code_stated_bound statics params my_closure metadata vs callee
       callee_vars Hclosure Hstated Hbind).
Qed.

(* The same the other way, for a rewrite whose result is refined by
   what it was given. *)
Lemma code_body_stated_refined (g : stated_in_scope -> expr -> expr)
    (Hg : ∀ (S : stated_in_scope) (e : expr),
        expr_refines_stated S (g S e) e)
    (params : list variable) (my_closure : variable)
    (metadata : code_metadata) (body : expr) :
  code_refines_stated_where always params my_closure metadata
    (rewrite_code_body (g (code_stated params my_closure metadata)) body)
    body.
Proof.
  intros vs callee callee_vars conts statics hp rd _ Hkinds Hclosure Hstated
    Hbind.
  unfold rewrite_code_body.
  destruct
    (Bool.eqb (binds_statics body)
       (binds_statics (g (code_stated params my_closure metadata) body)));
    [ | apply refines_refl ].
  apply Hg.
  exact
    (code_stated_bound statics params my_closure metadata vs callee
       callee_vars Hclosure Hstated Hbind).
Qed.

(* A group of static bindings and the group with the bodies of its
   code rewritten: the code of the second refines that of the first
   where it is entered. *)
Lemma bindings_rewritten_refine (g : stated_in_scope -> expr -> expr)
    (Hg : ∀ (S : stated_in_scope) (e : expr),
        expr_refines_stated S e (g S e))
    (bindings : list static_binding) :
  bindings_refine_stated always bindings
    (map (rewrite_code_stated_in_binding g) bindings).
Proof.
  induction bindings
    as [ | [id [params my_closure ret exn metadata code_body] | id
           | sym const]
           bindings [Hforwards Hbackwards]];
    simpl.
  - split; apply Forall2_nil.
  - split; (apply Forall2_cons; [ | assumption ]);
      apply Binding_code_where, Codes_related_where;
      first
        [ symmetry; apply rewrite_code_body_binds_statics
        | apply rewrite_code_body_binds_statics
        | exact
            (code_body_stated_refines g Hg params my_closure metadata
               code_body) ].
  - split; (apply Forall2_cons; [ apply Binding_same_where | assumption ]).
  - split; (apply Forall2_cons; [ apply Binding_same_where | assumption ]).
Qed.

Lemma bindings_rewritten_refined (g : stated_in_scope -> expr -> expr)
    (Hg : ∀ (S : stated_in_scope) (e : expr),
        expr_refines_stated S (g S e) e)
    (bindings : list static_binding) :
  bindings_refine_stated always
    (map (rewrite_code_stated_in_binding g) bindings) bindings.
Proof.
  induction bindings
    as [ | [id [params my_closure ret exn metadata code_body] | id
           | sym const]
           bindings [Hforwards Hbackwards]];
    simpl.
  - split; apply Forall2_nil.
  - split; (apply Forall2_cons; [ | assumption ]);
      apply Binding_code_where, Codes_related_where;
      first
        [ symmetry; apply rewrite_code_body_binds_statics
        | apply rewrite_code_body_binds_statics
        | exact
            (code_body_stated_refined g Hg params my_closure metadata
               code_body) ].
  - split; (apply Forall2_cons; [ apply Binding_same_where | assumption ]).
  - split; (apply Forall2_cons; [ apply Binding_same_where | assumption ]).
Qed.

(* The rewrite refines: the layers that are not a [Let_static] nor a
   recursive [Let_cont] by the script of FlambdaRecHandlersProofs.v. *)
Fixpoint rewrite_code_stated_refines_aux
    (g : stated_in_scope -> expr -> expr)
    (Hg : ∀ (S : stated_in_scope) (e : expr),
        expr_refines_stated S e (g S e))
    (e : expr) {struct e} :
  expr_refines e (rewrite_code_stated g e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try apply expr_refines_refl;
    try (destruct h as [params metadata handler]; simpl);
    try (rewrite_rec_handlers_cases (rewrite_code_stated_refines_aux g Hg);
         fail).
  - eapply expr_refines_trans;
      [ apply let_static_bindings_stated_refine,
          (bindings_rewritten_refine g Hg)
      | exact (plug_refines (Let_static_body _ Hole) _ _ (not_many _)
          (rewrite_code_stated_refines_aux g Hg body)) ].
  - assert (Hhandlers :
      ∀ vars : env,
        handlers_refine always vars handlers
          (map (fun '(k, h) => (k, rewrite_code_stated_in_handler g h))
             handlers)).
    { induction handlers as [ | [k [params metadata handler]] handlers IH];
        intros vars; simpl.
      - split; apply Forall2_nil.
      - destruct (IH vars) as [Hforwards Hbackwards].
        split; (apply Forall2_cons; [ | assumption ]);
          (split; [ reflexivity | ]); apply Handler_related;
          intros vs handler_vars conts statics hp rd _ _ _;
          apply (rewrite_code_stated_refines_aux g Hg handler). }
    eapply expr_refines_trans;
      [ | exact (plug_refines (Let_cont_recursive_body _ Hole) _ _
            (not_many _) (rewrite_code_stated_refines_aux g Hg body)) ].
    intros vars conts statics hp rd.
    destruct (Hhandlers vars) as [Hforwards Hbackwards].
    apply (let_cont_recursive_handlers_refine_where always always_extend);
      [ exact I | symmetry; apply rewrite_code_stated_handlers_bind_statics
      | exact Hforwards | exact Hbackwards ].
Qed.

Fixpoint rewrite_code_stated_refined_aux
    (g : stated_in_scope -> expr -> expr)
    (Hg : ∀ (S : stated_in_scope) (e : expr),
        expr_refines_stated S (g S e) e)
    (e : expr) {struct e} :
  expr_refines (rewrite_code_stated g e) e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    simpl; try apply expr_refines_refl;
    try (destruct h as [params metadata handler]; simpl);
    try (rewrite_rec_handlers_cases (rewrite_code_stated_refined_aux g Hg);
         fail).
  - eapply expr_refines_trans;
      [ apply let_static_bindings_stated_refine,
          (bindings_rewritten_refined g Hg)
      | exact (plug_refines (Let_static_body _ Hole) _ _ (not_many _)
          (rewrite_code_stated_refined_aux g Hg body)) ].
  - assert (Hhandlers :
      ∀ vars : env,
        handlers_refine always vars
          (map (fun '(k, h) => (k, rewrite_code_stated_in_handler g h))
             handlers)
          handlers).
    { induction handlers as [ | [k [params metadata handler]] handlers IH];
        intros vars; simpl.
      - split; apply Forall2_nil.
      - destruct (IH vars) as [Hforwards Hbackwards].
        split; (apply Forall2_cons; [ | assumption ]);
          (split; [ reflexivity | ]); apply Handler_related;
          intros vs handler_vars conts statics hp rd _ _ _;
          apply (rewrite_code_stated_refined_aux g Hg handler). }
    eapply expr_refines_trans;
      [ | exact (plug_refines (Let_cont_recursive_body _ Hole) _ _
            (not_many _) (rewrite_code_stated_refined_aux g Hg body)) ].
    intros vars conts statics hp rd.
    destruct (Hhandlers vars) as [Hforwards Hbackwards].
    apply (let_cont_recursive_handlers_refine_where always always_extend);
      [ exact I | apply rewrite_code_stated_handlers_bind_statics
      | exact Hforwards | exact Hbackwards ].
Qed.

Theorem rewrite_code_stated_refines : rewrite_code_stated_refines_statement.
Proof. exact rewrite_code_stated_refines_aux. Qed.

Theorem rewrite_code_stated_equivalent :
  rewrite_code_stated_equivalent_statement.
Proof.
  intros g Hg e vars conts statics hp rd. apply refines_antisymmetric.
  - apply (rewrite_code_stated_refines_aux g).
    intros S e' vars' conts' statics' hp' rd' Hhold.
    apply equivalent_refines, Hg, Hhold.
  - apply (rewrite_code_stated_refined_aux g).
    intros S e' vars' conts' statics' hp' rd' Hhold.
    apply equivalent_refines, equivalent_sym, Hg, Hhold.
Qed.
