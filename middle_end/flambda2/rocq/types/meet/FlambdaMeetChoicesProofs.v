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


(* Proofs of the statements of FlambdaMeetChoicesSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaClosuresSpec
  FlambdaPrimitiveProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvExamplesProofs
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs
  FlambdaTypingEnvOperationsSpec FlambdaTypingEnvOperationsProofs
  FlambdaJoinEnv FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv
  FlambdaMeet FlambdaMeetSpec FlambdaMeetProofs FlambdaMeetEnvProofs
  FlambdaMeetExamplesSpec FlambdaMeetExamplesProofs
  FlambdaSimplifySwitchProofs FlambdaJoinMeet FlambdaMeetChoicesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* 1. The type of the scrutinee of [Get_tag]                                *)
(* ======================================================================== *)

Theorem get_tag_reduction_of_block : get_tag_reduction_of_block_statement.
Proof.
  intros statics vars w n numbers alloc_mode Hrelation Hin. split.
  - intros Hsatisfied.
    apply blocks_with_these_tags_satisfied in Hsatisfied.
    destruct Hsatisfied as (tag & fields & Hblock & _).
    exists tag, fields. exact Hblock.
  - intros (tag & fields & Hblock).
    apply blocks_with_these_tags_satisfied.
    destruct Hrelation as [[i ->] | [[fields' Hblock'] Hrange]].
    + discriminate Hblock.
    + exists n, fields'. split; [ exact Hblock' | ].
      apply (tag_among_tags n numbers Hin).
      unfold FlambdaTypeMeaningSpec.is_tag in Hrange. unfold is_tag.
      apply andb_true_iff. split; apply Z.leb_le; lia.
Qed.

(* The cases of a binding of an environment that is given in full. *)
Ltac binding_cases H :=
  apply (keyed_lookup_in name_eqb name_eqb_eq) in H; vm_compute in H;
  repeat (destruct H as [ H | H ]; [ injection H as <- <- <- | ]);
  [ .. | destruct H ].

(* A [Simple] of a list that is given in full is of its kind. *)
Ltac listed_simples_kinded H :=
  vm_compute in H;
  repeat (destruct H as [ [= <- <-] | H ];
          [ first [ eexists; vm_compute; reflexivity
                  | vm_compute; reflexivity ]
          | ]);
  destruct H.

Lemma example_get_tag_env_fits : typing_env_fits example_get_tag_env.
Proof.
  split.
  { split; [ vm_compute; reflexivity | ].
    intros n ty Hlookup.
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
    vm_compute in Hlookup.
    repeat (destruct Hlookup as [ [= <- <-] | Hlookup ];
            [ eexists; vm_compute; reflexivity | ]).
    destruct Hlookup. }
  split.
  { intros n ty btm Hlookup. binding_cases Hlookup.
    all: vm_compute; reflexivity. }
  split.
  { split; [ | split ].
    - intros n ty btm Hlookup s k Hin. binding_cases Hlookup.
      all: listed_simples_kinded Hin.
    - repeat (apply replace_equation_aliases_kinded;
              [ eexists; vm_compute; reflexivity | ]).
      repeat (apply add_variable_definition_aliases_kinded;
              [ vm_compute; reflexivity | reflexivity | ]).
      apply create_aliases_kinded.
    - intros s ty btm Hlookup.
      apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
      vm_compute in Hlookup.
      repeat (destruct Hlookup as [ Hlookup | Hlookup ];
              [ discriminate Hlookup | ]).
      destruct Hlookup. }
  split.
  { intros n ty btm Hlookup m k Hin. binding_cases Hlookup.
    all: listed_simples_kinded Hin. }
  intros _ n ty btm Hlookup. binding_cases Hlookup.
  all: reflexivity.
Qed.

Theorem example_get_tag_inputs : example_get_tag_inputs_statement.
Proof.
  split; [ exact example_get_tag_env_fits | ].
  split.
  { split; [ vm_compute; reflexivity | ].
    split;
      [ intros x Hin; vm_compute in Hin;
        repeat (destruct Hin as [ <- | Hin ]; [ reflexivity | ]);
        destruct Hin
      | ].
    split; [ intros s k Hin; listed_simples_kinded Hin | ].
    intros n k Hin. listed_simples_kinded Hin. }
  split; [ | reflexivity ].
  split; [ vm_compute; reflexivity | ].
  split; [ intros x Hin; vm_compute in Hin; destruct Hin | ].
  split; [ intros s k Hin; listed_simples_kinded Hin | ].
  intros n k Hin. listed_simples_kinded Hin.
Qed.

Theorem only_get_tag_not_sound : only_get_tag_not_sound_statement.
Proof.
  intros join Hsound.
  assert (Hinputs := example_get_tag_inputs).
  assert (Hsame :
    Meet.meet join only_get_tag_of_blocks 3 example_get_tag_env
      example_tag_type example_zero_type
    = compiler_meet join 3 example_get_tag_env example_tag_type
        example_zero_type) by (vm_compute; reflexivity).
  destruct example_get_tag_env_satisfied as [Henv [Ht1 Ht2]].
  destruct (compiler_meet_get_tag join) as [t [tenv' [Hmeet Hnot]]].
  rewrite <- Hsame in Hmeet.
  destruct (Hsound empty_static_env _ _ _ _ _ _ Hinputs (Forall_nil _) Henv
              Ht1 Ht2 Hmeet)
    as [t0 [tenv0 [valuation' [Hresult [Hextended [Hsatisfied _]]]]]].
  injection Hresult as <- <-. exact (Hnot valuation' Hextended Hsatisfied).
Qed.

Theorem compiler_meet_not_sound_outright
  : compiler_meet_not_sound_outright_statement.
Proof.
  intros join.
  exact (compiler_meet_not_sound join example_get_tag_inputs).
Qed.

(* ======================================================================== *)
(* 2. Defining again                                                        *)
(* ======================================================================== *)

Theorem define_in_types_choice : define_in_types_choice_statement.
Proof.
  intros choices choices' env var kind Hagain Hnot.
  unfold define_in_types. rewrite Hagain, Hnot.
  destruct (Typing_env.mem None (Meet_env.typing_env env) (Name_var var));
    cbn; [ split | ]; reflexivity.
Qed.

(* A type whose only names are aliases of variables of kind value that
   the environment knows fits. *)
Ltac type_with_aliases_fits :=
  split; [ vm_compute; reflexivity | ];
  split;
  [ intros x Hin; vm_compute in Hin;
    repeat (destruct Hin as [ <- | Hin ]; [ reflexivity | ]); destruct Hin
  | ];
  split;
  [ intros s k Hin; vm_compute in Hin;
    repeat (destruct Hin as [ [= <- <-] | Hin ];
            [ eexists; vm_compute; reflexivity | ]);
    destruct Hin
  | intros n k Hin; vm_compute in Hin; destruct Hin ].

Lemma example_three_values_fits : typing_env_fits example_three_values.
Proof.
  apply plain_types_fits; [ | | vm_compute; reflexivity ].
  - split; [ vm_compute; reflexivity | ].
    intros n ty Hlookup. vm_compute in Hlookup. discriminate Hlookup.
  - repeat (apply add_variable_definition_aliases_kinded;
            [ vm_compute; reflexivity | reflexivity | ]).
    apply create_aliases_kinded.
Qed.

Lemma example_rebind_env_fits : typing_env_fits example_rebind_env.
Proof.
  apply plain_types_fits; [ | | vm_compute; reflexivity ].
  - split; [ vm_compute; reflexivity | ].
    intros n ty Hlookup.
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
    vm_compute in Hlookup.
    repeat (destruct Hlookup as [ [= <- <-] | Hlookup ];
            [ eexists; vm_compute; reflexivity | ]).
    destruct Hlookup.
  - repeat (apply replace_equation_aliases_kinded;
            [ eexists; vm_compute; reflexivity | ]).
    repeat (apply add_variable_definition_aliases_kinded;
            [ vm_compute; reflexivity | reflexivity | ]).
    apply create_aliases_kinded.
Qed.

Theorem example_rebind_inputs : example_rebind_inputs_statement.
Proof.
  split; [ exact example_rebind_env_fits | ].
  split; [ type_with_aliases_fits | ].
  split; [ type_with_aliases_fits | reflexivity ].
Qed.

Theorem example_rebind : example_rebind_statement.
Proof.
  eexists. eexists. split; [ vm_compute; reflexivity | ].
  vm_compute. reflexivity.
Qed.

Theorem defining_again_not_keeps : defining_again_not_keeps_statement.
Proof.
  intros Hkeeps.
  destruct example_rebind as (t & tenv' & Hmeet & Hnot).
  destruct (Hkeeps _ _ _ _ _ example_rebind_inputs Hmeet) as [Hfits _].
  rewrite (proj1 (proj1 Hfits)) in Hnot. discriminate Hnot.
Qed.

Theorem example_rebind_repaired : example_rebind_repaired_statement.
Proof.
  eexists. eexists. split; [ vm_compute; reflexivity | ].
  vm_compute. reflexivity.
Qed.

(* ======================================================================== *)
(* 3. Lifting the variables of a case that is alone                         *)
(* ======================================================================== *)

(* The variables recorded are still recorded, and there are no fewer
   of them. *)
Lemma union_extra_variables_grows
    (vars extra result : list (variable * flambda_kind)) :
  union_extra_variables extra vars = Some result ->
  (length extra <= length result)%nat.
Proof.
  revert extra. induction vars as [ | [var kind] vars IH ]; intros extra H.
  - injection H as <-. reflexivity.
  - cbn [union_extra_variables] in H.
    destruct (lookup var extra) as [ kind' | ].
    + destruct (kind_eqb kind kind'); [ exact (IH extra H) | discriminate H ].
    + apply IH in H. rewrite length_app in H. cbn in H. lia.
Qed.

Theorem union_extra_variables_same : union_extra_variables_same_statement.
Proof.
  intros extra vars. split.
  - induction vars as [ | [var kind] vars IH ]; intros H var0 kind0 Hin;
      [ destruct Hin | ].
    cbn [union_extra_variables] in H.
    destruct (lookup var extra) as [ kind' | ] eqn:Hlookup.
    + destruct (kind_eqb kind kind') eqn:Hkind; [ | discriminate H ].
      apply kind_eqb_eq in Hkind. subst kind'.
      destruct Hin as [ [= <- <-] | Hin ]; [ exact Hlookup | ].
      exact (IH H var0 kind0 Hin).
    + apply union_extra_variables_grows in H. rewrite length_app in H.
      cbn in H. lia.
  - induction vars as [ | [var kind] vars IH ]; intros H; [ reflexivity | ].
    cbn [union_extra_variables].
    rewrite (H var kind (or_introl eq_refl)).
    rewrite (proj2 (kind_eqb_eq kind kind) eq_refl).
    apply IH. intros var0 kind0 Hin. exact (H var0 kind0 (or_intror Hin)).
Qed.

Theorem example_unbound_inputs : example_unbound_inputs_statement.
Proof.
  split; [ exact example_three_values_fits | ].
  split; [ type_with_aliases_fits | ].
  split; [ type_with_aliases_fits | reflexivity ].
Qed.

Theorem example_unbound : example_unbound_statement.
Proof.
  eexists. eexists. exists 4%positive.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; tauto | vm_compute; reflexivity ].
Qed.

Theorem not_lifting_not_keeps : not_lifting_not_keeps_statement.
Proof.
  intros Hkeeps.
  destruct example_unbound as (t & tenv' & x & Hmeet & Hin & Hunknown).
  destruct (Hkeeps _ _ _ _ _ example_unbound_inputs Hmeet)
    as (_ & (_ & Hclosed & _) & _).
  rewrite (Hclosed x Hin) in Hunknown. discriminate Hunknown.
Qed.

Theorem example_unbound_repaired : example_unbound_repaired_statement.
Proof.
  eexists. eexists. split; [ vm_compute; reflexivity | ].
  intros x Hin. vm_compute in Hin.
  repeat (destruct Hin as [ <- | Hin ]; [ vm_compute; reflexivity | ]).
  destruct Hin.
Qed.
