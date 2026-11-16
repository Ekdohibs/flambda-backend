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

(* Proofs of the statements of FlambdaInlineSiblingsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaNominalSpec FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs
  FlambdaInlineSiblingsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What a group has bound                                                   *)
(* ======================================================================== *)

(* The static environment that a group of bindings gives binds the code
   of each binding of the group: a code id is bound once. *)
Lemma eval_static_bindings_binds (vars : env)
    (bindings : list static_binding) :
  ∀ statics extended : static_env,
    eval_static_bindings vars bindings statics = Some extended ->
    ∀ (id : code_id) (c : code),
      In (Bind_code id c) bindings ->
      lookup id (static_code extended) = Some c.
Proof.
  induction bindings as [ | binding bindings IH];
    intros statics extended Heval id c Hin; [ destruct Hin | ].
  simpl in Heval. destruct binding as [id0 c0 | id0 | sym const].
  - destruct (lookup id0 (static_code statics)); [ discriminate | ].
    destruct Hin as [Heq | Hin]; [ | exact (IH _ _ Heval id c Hin) ].
    injection Heq as -> ->.
    destruct (eval_static_bindings_extend vars bindings _ _ Heval)
      as [Hcode _].
    apply Hcode. simpl. rewrite Pos.eqb_refl. reflexivity.
  - destruct Hin as [Heq | Hin]; [ discriminate | ].
    exact (IH _ _ Heval id c Hin).
  - destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ];
      simpl in Heval; [ | discriminate ].
    destruct Hin as [Heq | Hin]; [ discriminate | ].
    exact (IH _ _ Heval id c Hin).
Qed.

(* Code that is known from a group is the code of one of its bindings, or
   was known before. *)
Lemma codes_of_in (bindings : list static_binding) :
  ∀ (known : list (code_id * code)) (id : code_id) (c : code),
    lookup id (codes_of bindings known) = Some c ->
    In (Bind_code id c) bindings ∨ lookup id known = Some c.
Proof.
  induction bindings as [ | [id0 c0 | id0 | sym const] bindings IH];
    intros known id c Hlookup; simpl in Hlookup.
  - right. exact Hlookup.
  - destruct (IH _ _ _ Hlookup) as [Hin | Hknown];
      [ left; right; exact Hin | ].
    simpl in Hknown. destruct (Pos.eqb id id0) eqn:Heq;
      [ | right; exact Hknown ].
    apply Pos.eqb_eq in Heq. injection Hknown as <-. subst id0.
    left. left. reflexivity.
  - destruct (IH _ _ _ Hlookup) as [Hin | Hknown];
      [ left; right; exact Hin | right; exact Hknown ].
  - destruct (IH _ _ _ Hlookup) as [Hin | Hknown];
      [ left; right; exact Hin | right; exact Hknown ].
Qed.

(* Once a group is bound, the static environment binds the code of the
   bindings of the group other than one of them. *)
Lemma statics_know_others (vars : env) (done : list static_binding)
    (binding : static_binding) (todo : list static_binding)
    (statics extended : static_env) :
  eval_static_bindings vars (done ++ binding :: todo) statics = Some extended ->
  statics_know (codes_of (done ++ todo) []) extended.
Proof.
  intros Heval id c Hlookup.
  destruct (codes_of_in _ _ _ _ Hlookup) as [Hin | Hnone];
    [ | discriminate Hnone ].
  apply (eval_static_bindings_binds vars _ _ _ Heval).
  apply in_app_iff in Hin as [Hin | Hin]; apply in_app_iff;
    [ left; exact Hin | right; right; exact Hin ].
Qed.

(* ======================================================================== *)
(* The refinement                                                           *)
(* ======================================================================== *)

(* One binding of a group, the others as they are. *)
Lemma inline_in_binding_refines (kept : list code_id)
    (done : list static_binding) (binding : static_binding)
    (todo : list static_binding) (body : expr) :
  expr_refines (Let_static (done ++ binding :: todo) body)
    (Let_static
       (done
        ++ inline_calls_in_binding kept (codes_of (done ++ todo) []) binding
           :: todo)
       body).
Proof.
  intros vars conts statics hp' rd'.
  set (others := codes_of (done ++ todo) []).
  assert (Hrefine :
    bindings_refine (statics_know others) (done ++ binding :: todo)
      (done ++ inline_calls_in_binding kept others binding :: todo)).
  { split;
      (apply Forall2_app; [ apply bindings_related_refl | ]);
      (apply Forall2_cons; [ | apply bindings_related_refl ]);
      (destruct binding
         as [id [params my_closure ret exn metadata code_body] | id
            | sym const];
       simpl; try apply Binding_same);
      apply Binding_code, Codes_related;
      first
        [ apply inline_calls_binds_statics
        | symmetry; apply inline_calls_binds_statics
        | intros vars' conts' statics' hp rd Hknow';
          apply inline_calls_refines_knowing, Hknow' ]. }
  destruct Hrefine as [Hforwards Hbackwards].
  apply (fun H =>
    eval_static_bindings_related _ vars _ _ H statics statics
      (statics_related_refl _ statics)) in Hforwards, Hbackwards.
  apply let_static_refines_step.
  destruct (eval_static_bindings vars (done ++ binding :: todo) statics)
    as [extended | ] eqn:Hbindings,
    (eval_static_bindings vars
       (done ++ inline_calls_in_binding kept others binding :: todo) statics)
    as [extended' | ] eqn:Hbindings';
    try contradiction; [ | exact I ].
  apply (related_states_refines (statics_know others)
    (statics_know_extend others));
    [ exact (statics_know_others vars done _ todo statics extended Hbindings)
    | exact
        (statics_know_others vars done _ todo statics extended' Hbindings')
    | apply cont_env_related_refl | apply cont_env_related_refl
    | exact Hforwards | exact Hbackwards ].
Qed.

(* The bindings of a group, one after the other. *)
Lemma inline_in_group_refines (kept : list code_id) (body : expr) :
  ∀ todo done : list static_binding,
    expr_refines (Let_static (done ++ todo) body)
      (Let_static (inline_in_group kept done todo) body).
Proof.
  induction todo as [ | binding todo IH]; intros done; cbn [inline_in_group].
  - rewrite app_nil_r. apply expr_refines_refl.
  - eapply expr_refines_trans;
      [ apply (inline_in_binding_refines kept done binding todo body) | ].
    specialize
      (IH
         (done
          ++ [inline_calls_in_binding kept (codes_of (done ++ todo) [])
                binding])).
    rewrite <- app_assoc in IH. exact IH.
Qed.

Fixpoint inline_sibling_calls_refines_aux (kept : list code_id) (e : expr) :
  expr_refines e (inline_sibling_calls kept e).
Proof.
  destruct e
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    cbn [inline_sibling_calls]; try apply expr_refines_refl.
  - exact
      (plug_refines (Let_body x kind d Hole) body _
         (fun H => False_ind _ (Bool.diff_false_true H))
         (inline_sibling_calls_refines_aux kept body)).
  - eapply expr_refines_trans.
    + exact
        (plug_refines (Let_static_body bindings Hole) body _
           (fun H => False_ind _ (Bool.diff_false_true H))
           (inline_sibling_calls_refines_aux kept body)).
    + exact (inline_in_group_refines kept _ bindings []).
  - destruct h as [params metadata handler].
    eapply expr_refines_trans.
    + exact
        (plug_refines
           (Let_cont_non_recursive_body k (Handler params metadata handler)
              Hole)
           body _
           (fun H => False_ind _ (Bool.diff_false_true H))
           (inline_sibling_calls_refines_aux kept body)).
    + exact
        (plug_refines
           (Let_cont_non_recursive_handler k params metadata Hole
              (inline_sibling_calls kept body))
           handler _
           (fun H => False_ind _ (Bool.diff_false_true H))
           (inline_sibling_calls_refines_aux kept handler)).
  - exact
      (plug_refines (Let_cont_recursive_body handlers Hole) body _
         (fun H => False_ind _ (Bool.diff_false_true H))
         (inline_sibling_calls_refines_aux kept body)).
Qed.

Theorem inline_sibling_calls_refines :
  inline_sibling_calls_refines_statement.
Proof. exact inline_sibling_calls_refines_aux. Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Lemma inline_in_group_equivariant (r : renaming) (kept : list code_id) :
  ∀ todo done : list static_binding,
    inline_in_group kept (map (rename_static_binding r) done)
      (map (rename_static_binding r) todo)
    = map (rename_static_binding r) (inline_in_group kept done todo).
Proof.
  induction todo as [ | binding todo IH]; intros done;
    cbn [inline_in_group map]; [ reflexivity | ].
  assert (Hcodes :
    codes_of
      (map (rename_static_binding r) done
       ++ map (rename_static_binding r) todo)
      []
    = map_values (rename_code r) (codes_of (done ++ todo) [])).
  { rewrite <- map_app. exact (codes_of_equivariant r (done ++ todo) []). }
  rewrite Hcodes, (inline_calls_in_binding_equivariant r kept binding).
  specialize
    (IH
       (done
        ++ [inline_calls_in_binding kept (codes_of (done ++ todo) [])
              binding])).
  rewrite map_app in IH. cbn [map] in IH. exact IH.
Qed.

Fixpoint inline_sibling_calls_renamed (r : renaming) (kept : list code_id)
    (e : expr) :
  inline_sibling_calls kept (rename_expr r e)
  = rename_expr r (inline_sibling_calls kept e).
Proof.
  destruct e
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    cbn [inline_sibling_calls rename_expr]; try reflexivity.
  - rewrite (inline_sibling_calls_renamed r kept body). reflexivity.
  - rewrite (inline_sibling_calls_renamed r kept body).
    pose proof (inline_in_group_equivariant r kept bindings []) as Hgroup.
    cbn [map] in Hgroup. rewrite Hgroup. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler inline_sibling_calls rename_expr].
    rewrite (inline_sibling_calls_renamed r kept body),
      (inline_sibling_calls_renamed r kept handler).
    reflexivity.
  - rewrite (inline_sibling_calls_renamed r kept body). reflexivity.
Qed.

Theorem inline_sibling_calls_equivariant :
  inline_sibling_calls_equivariant_statement.
Proof.
  intros kept r e. apply inline_sibling_calls_renamed.
Qed.
