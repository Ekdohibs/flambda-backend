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

(* Proofs of the statements of FlambdaAliasLetsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveProofs
  FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaContextSpec FlambdaContextProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaValidatorEquivarianceBaseProofs FlambdaAliasLetsSpec.
Import ListNotations.

(* The two [Let]s take their steps where the one does, and then the
   body runs under environments that differ only at the variable of
   the outer [Let], which is not free in the body. *)
Lemma merged_alias_let (x y : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  ~ var_free_in_expr x body ->
  expr_equivalent (Let x kind d (Let y kind (Simple (Var x)) body))
    (Let y kind d body).
Proof.
  intros Hunused vars conts statics hp rd.
  apply let_equivalent_step_exec; [ reflexivity | ].
  intros v hp' rd' _ Hkind.
  apply (equivalent_trans _
    (State body ((y, v) :: (x, v) :: vars) conts statics hp' rd')).
  - apply step_equivalent. rewrite step_let_simple. simpl.
    rewrite Pos.eqb_refl, Hkind, kind_eqb_refl. reflexivity.
  - apply coincide_equivalent.
    + intros z Hz. simpl.
      destruct (Pos.eqb z y); [ reflexivity | ].
      destruct (Pos.eqb_spec z x) as [-> | _];
        [ contradiction | reflexivity ].
    + intros k _. apply (conts_agree_same identity_renaming). reflexivity.
Qed.

Fixpoint merge_alias_lets_equivalent_aux (e : expr) :
  expr_equivalent e (merge_alias_lets e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try (intros vars conts statics hp rd; apply equivalent_refl).
  - pose proof (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (merge_alias_lets_equivalent_aux body)) as Hbody.
    simpl in Hbody.
    destruct (merge_alias_lets body)
      as [ y kind' d' body' | | | | | | | ]; try exact Hbody.
    destruct d' as [[x' | | ] | | | | | | ]; try exact Hbody.
    destruct
      (Pos.eqb x' x && kind_eqb kind' kind && negb (var_free_b x body'))
      eqn:Hmerge; [ | exact Hbody ].
    apply andb_true_iff in Hmerge as [Hmerge Hunused].
    apply andb_true_iff in Hmerge as [Hx Hkind].
    apply Pos.eqb_eq in Hx. apply kind_eqb_eq in Hkind. subst x' kind'.
    apply negb_true_iff in Hunused.
    apply (expr_equivalent_trans _ _ _ Hbody), merged_alias_let.
    intros Hfree. apply var_free_b_sound in Hfree. congruence.
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      (merge_alias_lets_equivalent_aux body)).
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (merge_alias_lets handler)) body)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole body)
        _ _ (not_many _) (merge_alias_lets_equivalent_aux handler)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_body k
           (Handler params metadata (merge_alias_lets handler)) Hole)
        _ _ (not_many _) (merge_alias_lets_equivalent_aux body)).
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole)
        _ _ (not_many _) (merge_alias_lets_equivalent_aux body)).
Qed.

Theorem merge_alias_lets_equivalent : merge_alias_lets_equivalent_statement.
Proof. exact merge_alias_lets_equivalent_aux. Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Fixpoint merge_alias_lets_renamed_aux (r : renaming) (e : expr) :
  merge_alias_lets (rename_expr r e) = rename_expr r (merge_alias_lets e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - rewrite merge_alias_lets_renamed_aux.
    destruct (merge_alias_lets body)
      as [ y kind' d' body' | | | | | | | ]; try reflexivity.
    destruct d' as [[x' | | ] | | | | | | ]; try reflexivity.
    simpl.
    rewrite rename_variable_eqb, var_free_b_equivariant_aux.
    destruct
      (Pos.eqb x' x && kind_eqb kind' kind && negb (var_free_b x body'));
      reflexivity.
  - rewrite merge_alias_lets_renamed_aux. reflexivity.
  - rewrite !merge_alias_lets_renamed_aux. reflexivity.
  - rewrite merge_alias_lets_renamed_aux. reflexivity.
Qed.

Theorem merge_alias_lets_renamed : merge_alias_lets_renamed_statement.
Proof. exact merge_alias_lets_renamed_aux. Qed.
