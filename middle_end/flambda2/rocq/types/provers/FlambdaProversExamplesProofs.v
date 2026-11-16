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



(* Proofs of the statements of FlambdaProversExamplesSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeMeaningBoolSpec FlambdaTypeMeaningBoolProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaExpandHead FlambdaProvers FlambdaProversSpec
  FlambdaProversExamplesSpec.
Import ListNotations.
Local Open Scope Z_scope.

Theorem example_is_int : example_is_int_statement.
Proof. repeat split; reflexivity. Qed.

Theorem example_numbers : example_numbers_statement.
Proof. repeat split; reflexivity. Qed.

Theorem example_stop : example_stop_statement.
Proof. reflexivity. Qed.

Theorem example_null_answers : example_null_answers_statement.
Proof. repeat split; vm_compute; reflexivity. Qed.

(* The two names of the environment, with their types: the list that
   the map is also has the binding of each from its definition, which
   the equation shadows. *)
Lemma example_null_env_names (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t) :
  Keyed.lookup name_eqb n (Typing_env.names_to_types example_null_env)
  = Some (ty, binding_time_and_mode) ->
  (n = Name_var example_x
   ∧ (ty = example_x_type
      ∨ ty = unknown (Naked_number Kind_naked_immediate)))
  ∨ (n = Name_var example_n ∧ (ty = null ∨ ty = unknown Value)).
Proof.
  intros Hlookup.
  apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
  vm_compute in Hlookup.
  repeat (destruct Hlookup as [Hlookup | Hlookup];
          [ injection Hlookup as <- <- <- | ]);
    [ .. | contradiction ];
    first
      [ left; split; [ reflexivity | ];
        first [ left; reflexivity | right; reflexivity ]
      | right; split; [ reflexivity | ];
        first [ left; reflexivity | right; reflexivity ] ].
Qed.

Theorem example_null_env_strict : example_null_env_strict_statement.
Proof.
  split; [ reflexivity | ]. split; [ | apply empty_respected ].
  intros n ty btm v Hlookup Hvalue.
  destruct (example_null_env_names _ _ _ Hlookup)
    as [[-> [-> | ->]] | [-> [-> | ->]]];
    cbv in Hvalue; injection Hvalue as <-; right.
  - apply (proj2 (is_null_for_scrutinee_satisfied _ _ _ _)).
    exists 1, (Value_const Null).
    split; [ reflexivity | ]. split; [ reflexivity | ].
    left. split; reflexivity.
  - apply (proj2 (unknown_satisfied _ _ _ _)). reflexivity.
  - apply (proj2 (null_satisfied _ _ _)). reflexivity.
  - apply (proj2 (unknown_satisfied _ _ _ _)). reflexivity.
Qed.

Lemma example_x_type_of_zero :
  type_satisfied empty_static_env
    [(example_x, Value_const (Naked_immediate 0));
     (example_n, Value_const (Poison Value))]
    (Value_const (Naked_immediate 0)) example_x_type.
Proof.
  apply (proj2 (is_null_for_scrutinee_satisfied _ _ _ _)).
  exists 0, (Value_const (Poison Value)).
  split; [ reflexivity | ]. split; [ reflexivity | ].
  right. split; [ reflexivity | ]. split; [ discriminate | reflexivity ].
Qed.

Theorem example_null_env_poison : example_null_env_poison_statement.
Proof.
  split; [ | split; [ | split; [ | split ] ] ].
  - (* Satisfied, with [n] exempt. *)
    split; [ reflexivity | ]. split; [ | apply empty_respected ].
    intros n ty btm v Hlookup Hvalue.
    destruct (example_null_env_names _ _ _ Hlookup)
      as [[-> [-> | ->]] | [-> [-> | ->]]];
      cbv in Hvalue; injection Hvalue as <-.
    + right. exact example_x_type_of_zero.
    + right. apply (proj2 (unknown_satisfied _ _ _ _)). reflexivity.
    + left. split; [ | reflexivity ].
      left. exists (Value_const (Poison Value)).
      split; [ reflexivity | exists Value; reflexivity ].
    + right. apply (proj2 (unknown_satisfied _ _ _ _)). reflexivity.
  - (* Not strictly: poison is not null. *)
    intros [_ [Hnames _]].
    assert (Hlookup :
      ∃ btm : Binding_time.With_name_mode.t,
        Keyed.lookup name_eqb (Name_var example_n)
          (Typing_env.names_to_types example_null_env)
        = Some (null, btm)).
    { eexists. vm_compute. reflexivity. }
    destruct Hlookup as [btm Hlookup].
    specialize
      (Hnames (Name_var example_n) null btm (Value_const (Poison Value))
         Hlookup eq_refl).
    destruct Hnames as [[[] _] | Hnull].
    apply (proj1 (null_satisfied _ _ _)) in Hnull. discriminate Hnull.
  - (* Well formed. *)
    split; [ | reflexivity ].
    intros n ty btm Hlookup.
    destruct (example_null_env_names _ _ _ Hlookup)
      as [[-> [-> | ->]] | [-> [-> | ->]]];
      reflexivity.
  - exact example_x_type_of_zero.
  - intros [i [Heq [Hin | []]]]. injection Heq as <-. discriminate Hin.
Qed.

Theorem example_value_slot_kind : example_value_slot_kind_statement.
Proof.
  split; [ | repeat split; vm_compute; reflexivity ].
  apply (proj1 (type_satisfied_b_iff _ _ _ _)). vm_compute. reflexivity.
Qed.

Theorem example_blocks : example_blocks_statement.
Proof. repeat split; vm_compute; reflexivity. Qed.

Theorem example_closures : example_closures_statement.
Proof. repeat split; vm_compute; reflexivity. Qed.

Theorem example_physical_equality : example_physical_equality_statement.
Proof. repeat split; vm_compute; reflexivity. Qed.
