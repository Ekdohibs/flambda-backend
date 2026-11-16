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

(* Proofs of the statements of FlambdaTypeNominalSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec
  FlambdaNameAbstractionSpec FlambdaNameAbstractionProofs
  FlambdaNominalInstancesSpec FlambdaFreshQuantifierSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaTypeEqualSpec
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs FlambdaTypeNominalSpec.
Import ListNotations.
Import Nominal.

Lemma rename_variable_variables (p : permutation) (x : variable) :
  rename_variable (variables_renaming p) x = permute p x.
Proof. reflexivity. Qed.

(* The free variables of a renamed thing, as the interface asks. *)
Lemma among_renamed_variables (p : permutation) (name : positive)
    (names : list variable) :
  among (permute p name) (map (rename_variable (variables_renaming p)) names)
  = among name names.
Proof. exact (among_permuted p name names). Qed.

Lemma fixes_in (p : permutation) (names : list variable) (x : variable) :
  fixes p names -> x ∈ names ->
  rename_variable (variables_renaming p) x = x.
Proof.
  intros Hfixes Hin. apply Hfixes.
  exact (proj2 (keyed_among_in positive Pos.eqb Pos.eqb_eq x names) Hin).
Qed.

Theorem type_names_nominal : type_names_nominal_statement.
Proof.
  constructor; simpl.
  - reflexivity.
  - intros t t' Hsame. symmetry. exact Hsame.
  - intros t t' t'' -> ->. reflexivity.
  - exact (proj1 rename_type_identity).
  - intros second first t.
    exact (proj1 rename_type_compose (variables_renaming second)
             (variables_renaming first) t).
  - intros p p' t Hsame. apply (proj1 rename_type_extensional).
    intros x _. exact (Hsame x).
  - intros p t t' ->. reflexivity.
  - intros p t name. rewrite (proj1 free_variables_rename_type).
    apply among_renamed_variables.
  - intros t t' -> name. reflexivity.
  - intros p t Hfixes. apply (proj1 rename_type_fixes).
    intros x Hin. exact (fixes_in p _ x Hfixes Hin).
Qed.

Theorem env_extension_names_nominal : env_extension_names_nominal_statement.
Proof.
  constructor; simpl.
  - reflexivity.
  - intros t t' Hsame. symmetry. exact Hsame.
  - intros t t' t'' -> ->. reflexivity.
  - exact (proj2 rename_type_identity).
  - intros second first extension.
    exact (proj2 rename_type_compose (variables_renaming second)
             (variables_renaming first) extension).
  - intros p p' extension Hsame. apply (proj2 rename_type_extensional).
    intros x _. exact (Hsame x).
  - intros p t t' ->. reflexivity.
  - intros p extension name. rewrite (proj2 free_variables_rename_type).
    apply among_renamed_variables.
  - intros t t' -> name. reflexivity.
  - intros p extension Hfixes. apply (proj2 rename_type_fixes).
    intros x Hin. exact (fixes_in p _ x Hfixes Hin).
Qed.

Theorem kind_equivariant : kind_equivariant_statement.
Proof. intros p t. simpl. apply kind_rename_type. Qed.

Theorem type_well_formed_equivariant : type_well_formed_equivariant_statement.
Proof. intros p t. simpl. apply (proj1 type_well_formed_rename). Qed.

Theorem type_eqb_equivariant : type_eqb_equivariant_statement.
Proof. intros p [t t']. simpl. apply (proj1 type_eqb_rename). Qed.

Theorem type_satisfied_equivariant : type_satisfied_equivariant_statement.
Proof.
  intros statics v p [vars t]. simpl.
  apply (type_satisfied_rename (variables_renaming p) statics vars v t).
Qed.

Theorem env_extension_satisfied_equivariant
  : env_extension_satisfied_equivariant_statement.
Proof.
  intros statics p [vars extension]. simpl.
  apply (env_extension_satisfied_rename (variables_renaming p) statics vars
           extension).
Qed.
