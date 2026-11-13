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

(* Proofs of the statements of FlambdaKnownSiblingsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaNominalSpec FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs
  FlambdaInlineSiblingsSpec FlambdaInlineSiblingsProofs
  FlambdaValidatorEquivarianceBaseProofs FlambdaKnownSiblingsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What a group has bound                                                   *)
(* ======================================================================== *)

(* What two lists of bindings make known of their symbols, one after
   the other. *)
Lemma symbols_of_app (bindings bindings' : list static_binding) :
  ∀ known : list (symbol * option code_id),
    symbols_of (bindings ++ bindings') known
    = symbols_of bindings' (symbols_of bindings known).
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros known; simpl; [ reflexivity | apply IH | apply IH | ].
  destruct const; simpl; apply IH.
Qed.

(* Once a group is bound, the static environment binds its symbols as
   the bindings of the group other than one of code say. *)
Lemma statics_know_other_symbols (vars : env) (done : list static_binding)
    (id : code_id) (c : code) (todo : list static_binding)
    (statics extended : static_env) :
  eval_static_bindings vars (done ++ Bind_code id c :: todo) statics
  = Some extended ->
  statics_know_symbols (symbols_of (done ++ todo) []) extended.
Proof.
  intros Heval.
  replace (symbols_of (done ++ todo) [])
    with (symbols_of (done ++ Bind_code id c :: todo) [])
    by (rewrite !symbols_of_app; reflexivity).
  apply (eval_static_bindings_know_symbols vars _ [] statics extended);
    [ | exact Heval ].
  intros s code Hs. discriminate Hs.
Qed.

(* ======================================================================== *)
(* The equivalence                                                          *)
(* ======================================================================== *)

(* One binding of a group, the others as they are. *)
Lemma inline_known_in_binding_equivalent (kept : list code_id)
    (done : list static_binding) (binding : static_binding)
    (todo : list static_binding) (body : expr) :
  expr_equivalent (Let_static (done ++ binding :: todo) body)
    (Let_static
       (done
        ++ inline_known_calls_in_binding kept (codes_of (done ++ todo) [])
             (symbols_of (done ++ todo) []) binding
           :: todo)
       body).
Proof.
  intros vars conts statics hp.
  destruct binding
    as [id [params my_closure ret exn metadata code_body] | id | sym const];
    try apply equivalent_refl.
  set (others_code := codes_of (done ++ todo) []).
  set (others_symbols := symbols_of (done ++ todo) []).
  cbn [inline_known_calls_in_binding inline_known_calls_in_code].
  set (c := Code params my_closure ret exn metadata code_body).
  set
    (c' :=
       Code params my_closure ret exn metadata
         (inline_known_calls kept others_code others_symbols code_body)).
  assert (Hrefine :
    bindings_refine (statics_know_both others_code others_symbols)
      (done ++ Bind_code id c :: todo) (done ++ Bind_code id c' :: todo)
    ∧ bindings_refine (statics_know_both others_code others_symbols)
        (done ++ Bind_code id c' :: todo) (done ++ Bind_code id c :: todo)).
  { repeat split;
      (apply Forall2_app; [ apply bindings_related_refl | ]);
      (apply Forall2_cons; [ | apply bindings_related_refl ]);
      apply Binding_code, Codes_related;
      first
        [ apply inline_known_calls_binds_statics
        | symmetry; apply inline_known_calls_binds_statics
        | intros vars' conts' statics' hp' [Hcode' Hsymbols'];
          first
            [ apply equivalent_refines,
                inline_known_calls_equivalent_knowing; assumption
            | apply equivalent_refines, equivalent_sym,
                inline_known_calls_equivalent_knowing; assumption ] ]. }
  destruct Hrefine as [[Hf Hb] [Hf' Hb']].
  apply (fun H =>
    eval_static_bindings_related _ vars _ _ H statics statics
      (statics_related_refl _ statics)) in Hf, Hb, Hf', Hb'.
  apply let_static_equivalent_step.
  destruct (eval_static_bindings vars (done ++ Bind_code id c :: todo) statics)
    as [extended | ] eqn:Hbindings,
    (eval_static_bindings vars (done ++ Bind_code id c' :: todo) statics)
    as [extended' | ] eqn:Hbindings';
    try contradiction; [ | exact I ].
  assert (Hok : statics_know_both others_code others_symbols extended).
  { split;
      [ exact (statics_know_others vars done _ todo statics extended Hbindings)
      | exact
          (statics_know_other_symbols vars done id c todo statics extended
             Hbindings) ]. }
  assert (Hok' : statics_know_both others_code others_symbols extended').
  { split;
      [ exact
          (statics_know_others vars done _ todo statics extended' Hbindings')
      | exact
          (statics_know_other_symbols vars done id c' todo statics extended'
             Hbindings') ]. }
  apply refines_antisymmetric;
    apply (related_states_refines
      (statics_know_both others_code others_symbols)
      (statics_know_both_extend others_code others_symbols));
    try assumption; apply cont_env_related_refl.
Qed.

(* The bindings of a group, one after the other. *)
Lemma inline_known_in_group_equivalent (kept : list code_id) (body : expr) :
  ∀ todo done : list static_binding,
    expr_equivalent (Let_static (done ++ todo) body)
      (Let_static (inline_known_in_group kept done todo) body).
Proof.
  induction todo as [ | binding todo IH]; intros done;
    cbn [inline_known_in_group].
  - rewrite app_nil_r. intros vars conts statics hp. apply equivalent_refl.
  - eapply expr_equivalent_trans;
      [ apply (inline_known_in_binding_equivalent kept done binding todo body)
      | ].
    cbv zeta.
    specialize
      (IH
         (done
          ++ [inline_known_calls_in_binding kept (codes_of (done ++ todo) [])
                (symbols_of (done ++ todo) []) binding])).
    rewrite <- app_assoc in IH. exact IH.
Qed.

Fixpoint inline_known_sibling_calls_equivalent_aux (kept : list code_id)
    (e : expr) :
  expr_equivalent e (inline_known_sibling_calls kept e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [inline_known_sibling_calls];
    try (intros vars conts statics hp; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (inline_known_sibling_calls_equivalent_aux kept body)).
  - eapply expr_equivalent_trans.
    + exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
        (inline_known_sibling_calls_equivalent_aux kept body)).
    + exact (inline_known_in_group_equivalent kept _ bindings []).
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (inline_known_sibling_calls kept handler))
         body)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole body)
        _ _ (not_many _)
        (inline_known_sibling_calls_equivalent_aux kept handler)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_body k
           (Handler params metadata
              (inline_known_sibling_calls kept handler))
           Hole)
        _ _ (not_many _)
        (inline_known_sibling_calls_equivalent_aux kept body)).
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole) _ _
      (not_many _) (inline_known_sibling_calls_equivalent_aux kept body)).
Qed.

Theorem inline_known_sibling_calls_equivalent :
  inline_known_sibling_calls_equivalent_statement.
Proof. exact inline_known_sibling_calls_equivalent_aux. Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Lemma inline_known_in_group_renamed (r : renaming) (kept : list code_id) :
  ∀ todo done : list static_binding,
    inline_known_in_group kept (map (rename_static_binding r) done)
      (map (rename_static_binding r) todo)
    = map (rename_static_binding r) (inline_known_in_group kept done todo).
Proof.
  induction todo as [ | binding todo IH]; intros done;
    cbn [inline_known_in_group map]; [ reflexivity | ].
  cbv zeta.
  assert (Hcodes :
    codes_of
      (map (rename_static_binding r) done
       ++ map (rename_static_binding r) todo)
      []
    = map_values (rename_code r) (codes_of (done ++ todo) [])).
  { rewrite <- map_app. exact (codes_of_equivariant r (done ++ todo) []). }
  rewrite Hcodes, <- map_app, symbols_of_renamed,
    (inline_known_calls_in_binding_equivariant r kept
       (symbols_of (done ++ todo) []) binding).
  specialize
    (IH
       (done
        ++ [inline_known_calls_in_binding kept (codes_of (done ++ todo) [])
              (symbols_of (done ++ todo) []) binding])).
  rewrite map_app in IH. cbn [map] in IH. exact IH.
Qed.

Fixpoint inline_known_sibling_calls_renamed_aux (r : renaming)
    (kept : list code_id) (e : expr) :
  inline_known_sibling_calls kept (rename_expr r e)
  = rename_expr r (inline_known_sibling_calls kept e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [inline_known_sibling_calls rename_expr rename_handler];
    try reflexivity.
  - rewrite (inline_known_sibling_calls_renamed_aux r kept body).
    reflexivity.
  - rewrite (inline_known_sibling_calls_renamed_aux r kept body).
    pose proof (inline_known_in_group_renamed r kept bindings []) as Hgroup.
    cbn [map] in Hgroup. rewrite Hgroup. reflexivity.
  - rewrite (inline_known_sibling_calls_renamed_aux r kept body),
      (inline_known_sibling_calls_renamed_aux r kept handler).
    reflexivity.
  - rewrite (inline_known_sibling_calls_renamed_aux r kept body).
    reflexivity.
Qed.

Theorem inline_known_sibling_calls_renamed :
  inline_known_sibling_calls_renamed_statement.
Proof.
  intros kept r e. apply inline_known_sibling_calls_renamed_aux.
Qed.
