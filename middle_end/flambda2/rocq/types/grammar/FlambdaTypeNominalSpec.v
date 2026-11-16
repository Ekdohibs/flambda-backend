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

(* The types of Flambda 2 as things that contain names.

   This file is to be reviewed: the two instances, against the
   interface of FlambdaNameAbstractionSpec.v, and the statements.

   [type_names] is the instance for types of the interface for things
   that contain names ([Nominal.contains_names], the model's
   [Contains_names.S]), at the sort of variables. The action of a
   permutation of the variables is [rename_type]
   (FlambdaTypeRenameSpec.v); the free names are [free_variables];
   and the equality is that of Rocq, since a type binds no name: two
   types denote the same when they are the same type.
   [env_extension_names] is the same for environment extensions. The
   names that have an equation in an extension are free names of it
   like the others: an extension is not a binder. What binds the
   names of a level is the typing environment, and that instance is
   for the file that defines one.

   A type holds no continuation, so at the sort of continuations
   types are [Nominal.no_names]; that instance is not written. The
   symbols and the code ids of a type are not names here: no renaming
   of the model moves them.

   The statements: the two instances have the laws of the interface
   ([Nominal.nominal]); [kind], [type_well_formed] and [type_eqb]
   are equivariant functions into types with no names, in the sense
   of the interface ([Nominal.equivariant]); and the meaning of types
   is an equivariant function of the environment and the type into
   the propositions. There the environment is the instance
   [env_names] of FlambdaNominalInstancesSpec.v, whose action is
   [rename_env], and the propositions are the instance [propositions]
   of FlambdaFreshQuantifierSpec.v, which are the same when they are
   equivalent. All are the theorems of FlambdaTypeRenameProofs.v, put
   in the form of the interface.

   [variables_renaming], the renaming that permutes the variables and
   no continuation, is that of FlambdaNominalInstancesSpec.v.

   The proofs are in FlambdaTypeNominalProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaNameAbstractionSpec FlambdaNominalInstancesSpec
  FlambdaFreshQuantifierSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaTypeEqualSpec
  FlambdaTypeRenameSpec.
Import ListNotations.
Import Nominal.

Definition type_names : contains_names flambda_type :=
  Contains_names (fun p t => rename_type (variables_renaming p) t)
    free_variables eq.

Definition env_extension_names : contains_names env_extension :=
  Contains_names
    (fun p extension => rename_env_extension (variables_renaming p) extension)
    free_variables_of_env_extension eq.

Definition type_names_nominal_statement : Prop := nominal type_names.

Definition env_extension_names_nominal_statement : Prop :=
  nominal env_extension_names.

(* The functions on types that answer something with no name in it. *)
Definition kind_equivariant_statement : Prop :=
  equivariant type_names (no_names flambda_kind) kind.

Definition type_well_formed_equivariant_statement : Prop :=
  equivariant type_names (no_names bool) type_well_formed.

Definition type_eqb_equivariant_statement : Prop :=
  equivariant (pair_names type_names type_names) (no_names bool)
    (fun types => type_eqb (fst types) (snd types)).

(* The meaning: whether a value has a type under an environment, as a
   function of the environment and the type. The value holds no
   variable. The static environment is kept as it is: the meaning
   reads of it what the symbols are bound to, which is made of values,
   and not its code. *)
Definition type_satisfied_equivariant_statement : Prop :=
  ∀ (statics : static_env) (v : value),
    equivariant (pair_names env_names type_names) propositions
      (fun vars_and_type =>
         type_satisfied statics (fst vars_and_type) v (snd vars_and_type)).

Definition env_extension_satisfied_equivariant_statement : Prop :=
  ∀ statics : static_env,
    equivariant (pair_names env_names env_extension_names) propositions
      (fun vars_and_extension =>
         env_extension_satisfied statics (fst vars_and_extension)
           (snd vars_and_extension)).
