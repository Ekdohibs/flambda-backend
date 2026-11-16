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



(* Proofs of the statements of FlambdaExpandHeadSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaPrimitiveProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvKindedSpec
  FlambdaTypingEnvKindedProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaExpandHeadSpec.
Import ListNotations.

(* ======================================================================== *)
(* A type that is not an alias, expanded                                    *)
(* ======================================================================== *)

(* The cases of [of_non_alias_type]: the type is unknown, bottom, or
   given by a head. *)
Ltac of_non_alias_cases ty H :=
  destruct ty as [[ | [? | ?] | ] | [ | [? | ?] | ] | k [ | [? | ?] | ]
                  | [ | [? | ?] | ] | [ | [? | ?] | ]];
  try discriminate H; injection H as <-;
  try (destruct k).

Lemma of_non_alias_type_sound (statics : static_env) (vars : env) (v : value)
    (ty : flambda_type) (expanded : Expanded_type.t) :
  Expanded_type.of_non_alias_type ty = Some expanded ->
  type_satisfied statics vars v ty ->
  type_satisfied statics vars v (Expanded_type.to_type expanded).
Proof.
  intros Hexpanded. of_non_alias_cases ty Hexpanded; intros H; exact H.
Qed.

Lemma of_non_alias_type_free_variables (ty : flambda_type)
    (expanded : Expanded_type.t) :
  Expanded_type.of_non_alias_type ty = Some expanded ->
  free_variables (Expanded_type.to_type expanded) = free_variables ty.
Proof.
  intros Hexpanded. of_non_alias_cases ty Hexpanded; reflexivity.
Qed.

Lemma of_non_alias_type_aliases (ty : flambda_type)
    (expanded : Expanded_type.t) :
  Expanded_type.of_non_alias_type ty = Some expanded ->
  aliases_with_kinds (Expanded_type.to_type expanded) = aliases_with_kinds ty.
Proof.
  intros Hexpanded. of_non_alias_cases ty Hexpanded; reflexivity.
Qed.

Lemma of_non_alias_type_equation_names (ty : flambda_type)
    (expanded : Expanded_type.t) :
  Expanded_type.of_non_alias_type ty = Some expanded ->
  equation_names_with_kinds (Expanded_type.to_type expanded)
  = equation_names_with_kinds ty.
Proof.
  intros Hexpanded. of_non_alias_cases ty Hexpanded; reflexivity.
Qed.

Lemma of_non_alias_type_kind (ty : flambda_type)
    (expanded : Expanded_type.t) :
  Expanded_type.of_non_alias_type ty = Some expanded ->
  Expanded_type.kind expanded = kind ty
  ∧ kind (Expanded_type.to_type expanded) = kind ty.
Proof.
  intros Hexpanded. of_non_alias_cases ty Hexpanded; split; reflexivity.
Qed.

Lemma of_non_alias_type_head (ty : flambda_type)
    (expanded : Expanded_type.t) (head : Expanded_type.head) :
  Expanded_type.of_non_alias_type ty = Some expanded ->
  Expanded_type.descr expanded = Ok head ->
  head_kind head = Expanded_type.kind expanded.
Proof.
  intros Hexpanded. of_non_alias_cases ty Hexpanded; simpl;
    intros Hhead; try discriminate Hhead; injection Hhead as <-; reflexivity.
Qed.

(* ======================================================================== *)
(* The type of a constant, expanded                                         *)
(* ======================================================================== *)

Ltac open_type := repeat (autorewrite with type_satisfied; cbn).

Lemma create_const_sound (statics : static_env) (vars : env) (c : const) :
  type_satisfied statics vars (Value_const c)
    (Expanded_type.to_type (Expanded_type.create_const c)).
Proof.
  destruct c as [i | i | f | f | i | i | i | i | i | i | i | i | i | | k].
  all: try (open_type; eexists; eexists;
            split; [ reflexivity | ];
            split; [ reflexivity | ];
            split; [ reflexivity | left; reflexivity ]).
  - open_type. exists i. split; [ reflexivity | ].
    split; [ left; reflexivity | intros relation n [] ].
  - open_type. right. split; [ discriminate | ]. split; [ | exact I ].
    left. exists i. split; [ reflexivity | ]. split; [ exact I | ].
    split; [ | exact I ].
    apply (proj2 (this_const_satisfied _ _ _ (Naked_immediate i))).
    reflexivity.
  - open_type. left. split; [ reflexivity | exact I ].
  - apply (proj2 (unknown_satisfied _ _ _ k)). reflexivity.
Qed.

Lemma create_const_free_variables (c : const) :
  free_variables (Expanded_type.to_type (Expanded_type.create_const c)) = [].
Proof.
  destruct c as [i | i | f | f | i | i | i | i | i | i | i | i | i | | k];
    try reflexivity.
  destruct k as [ | [] | | ]; reflexivity.
Qed.

Lemma create_const_equation_names (c : const) :
  equation_names_with_kinds
    (Expanded_type.to_type (Expanded_type.create_const c))
  = [].
Proof.
  destruct c as [i | i | f | f | i | i | i | i | i | i | i | i | i | | k];
    try reflexivity.
  destruct k as [ | [] | | ]; reflexivity.
Qed.

Lemma create_const_kinded (tenv : Typing_env.t) (c : const) :
  type_kinded tenv (Expanded_type.to_type (Expanded_type.create_const c)).
Proof.
  intros s kind Hin.
  destruct c as [i | i | f | f | i | i | i | i | i | i | i | i | i | | k];
    simpl in Hin; try contradiction.
  - destruct Hin as [Hin | []]. injection Hin as <- <-. reflexivity.
  - destruct k as [ | [] | | ]; destruct Hin as [].
Qed.

Lemma create_const_kind (c : const) :
  Expanded_type.kind (Expanded_type.create_const c) = kind (type_for_const c)
  ∧ kind (Expanded_type.to_type (Expanded_type.create_const c))
    = kind (type_for_const c).
Proof.
  destruct c as [i | i | f | f | i | i | i | i | i | i | i | i | i | | k];
    try (split; reflexivity).
  destruct k as [ | [] | | ]; split; reflexivity.
Qed.

Lemma create_const_head (c : const) (head : Expanded_type.head) :
  Expanded_type.descr (Expanded_type.create_const c) = Ok head ->
  head_kind head = Expanded_type.kind (Expanded_type.create_const c).
Proof.
  destruct c as [i | i | f | f | i | i | i | i | i | i | i | i | i | | k];
    simpl; intros Hhead; try discriminate Hhead; injection Hhead as <-;
    reflexivity.
Qed.

(* ======================================================================== *)
(* The type that the environment has for a name                             *)
(* ======================================================================== *)

(* What [Typing_env.find] answers, and that its variables are bound,
   are [find_cases] and [find_closed] of the typing environment's own
   files. *)

Lemma bottom_aliases (k : flambda_kind) : aliases_with_kinds (bottom k) = [].
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What [expand_head] answers: the type itself, expanded, if it is not
   an alias; if it is one, the unknown type, the type of a constant, or
   the type that the environment has for a name, each expanded. *)
Lemma expand_head_cases (tenv : Typing_env.t) (ty : flambda_type)
    (expanded : Expanded_type.t) :
  expand_head tenv ty = Some expanded ->
  (get_alias_opt ty = None
   ∧ Expanded_type.of_non_alias_type ty = Some expanded)
  ∨ ∃ s : simple,
      get_alias_opt ty = Some s
      ∧ ((Typing_env.get_canonical_simple_exn tenv (Some Name_mode.in_types)
            None s
          = Some None
          ∧ Expanded_type.of_non_alias_type (unknown (kind ty))
            = Some expanded)
         ∨ (∃ c : const,
             Typing_env.get_canonical_simple_exn tenv
               (Some Name_mode.in_types) None s
             = Some (Some (Const c))
             ∧ expanded = Expanded_type.create_const c)
         ∨ ∃ (n : name) (ty' : flambda_type),
             Typing_env.get_canonical_simple_exn tenv
               (Some Name_mode.in_types) None s
             = Some (Some (simple_of_name n))
             ∧ Typing_env.find tenv n (Some (kind ty)) = Some ty'
             ∧ Expanded_type.of_non_alias_type ty' = Some expanded).
Proof.
  unfold expand_head.
  destruct (get_alias_opt ty) as [s | ]; [ | intros H; left; split; auto ].
  intros H. right. exists s. split; [ reflexivity | ].
  destruct (Typing_env.get_canonical_simple_exn tenv
              (Some Name_mode.in_types) None s)
    as [[canonical | ] | ]; simpl in H; [ | | discriminate H ].
  - right. destruct canonical as [x | sym | c]; simpl in H.
    + right. exists (Name_var x).
      destruct (Typing_env.find tenv (Name_var x) (Some (kind ty)))
        as [ty' | ]; simpl in H; [ | discriminate H ].
      exists ty'. repeat split; assumption.
    + right. exists (Name_symbol sym).
      destruct (Typing_env.find tenv (Name_symbol sym) (Some (kind ty)))
        as [ty' | ]; simpl in H; [ | discriminate H ].
      exists ty'. repeat split; assumption.
    + left. exists c. split; [ reflexivity | ]. injection H as <-.
      reflexivity.
  - left. split; [ reflexivity | exact H ].
Qed.

Theorem expand_head_sound : expand_head_sound_statement.
Proof.
  intros statics valuation tenv ty expanded v Hsatisfied Hty Hexpand.
  destruct (expand_head_cases tenv ty expanded Hexpand)
    as [[_ Hexpanded]
       | [s [Halias [[_ Hexpanded]
                    | [[c [Hcanonical ->]]
                      | [n [ty' [Hcanonical [Hfind Hexpanded]]]]]]]]].
  - exact (of_non_alias_type_sound _ _ _ _ _ Hexpanded Hty).
  - apply (of_non_alias_type_sound _ _ _ _ _ Hexpanded).
    apply (proj2 (unknown_satisfied _ _ _ _)).
    exact (proj2 (proj1 (get_alias_satisfied _ _ _ _ _ Halias) Hty)).
  - apply (get_alias_satisfied _ _ _ _ _ Halias) in Hty.
    destruct Hty as [Hvalue _].
    rewrite <- (get_canonical_simple_satisfied _ _ _ _ _ _ _ _ Hsatisfied
                  Hcanonical) in Hvalue.
    injection Hvalue as <-. apply create_const_sound.
  - apply (of_non_alias_type_sound _ _ _ _ _ Hexpanded).
    apply (get_alias_satisfied _ _ _ _ _ Halias) in Hty.
    destruct Hty as [Hvalue _].
    rewrite <- (get_canonical_simple_satisfied _ _ _ _ _ _ _ _ Hsatisfied
                  Hcanonical) in Hvalue.
    exact (find_satisfied_strictly _ _ _ _ _ _ _ Hsatisfied Hfind Hvalue).
Qed.

Theorem expand_head_closed : expand_head_closed_statement.
Proof.
  intros tenv ty expanded Hclosed Hty Hexpand x Hin.
  destruct (expand_head_cases tenv ty expanded Hexpand)
    as [[_ Hexpanded]
       | [s [Halias [[_ Hexpanded]
                    | [[c [Hcanonical ->]]
                      | [n [ty' [Hcanonical [Hfind Hexpanded]]]]]]]]].
  - rewrite (of_non_alias_type_free_variables _ _ Hexpanded) in Hin.
    exact (Hty x Hin).
  - rewrite (of_non_alias_type_free_variables _ _ Hexpanded) in Hin.
    destruct (kind ty) as [ | [] | | ]; destruct Hin.
  - rewrite create_const_free_variables in Hin. destruct Hin.
  - rewrite (of_non_alias_type_free_variables _ _ Hexpanded) in Hin.
    exact (find_closed _ _ _ _ _ Hclosed Hfind Hin).
Qed.

Theorem expand_head_kinded : expand_head_kinded_statement.
Proof.
  intros tenv ty expanded Hkinded Hty Hexpand.
  destruct (expand_head_cases tenv ty expanded Hexpand)
    as [[_ Hexpanded]
       | [s [Halias [[_ Hexpanded]
                    | [[c [Hcanonical ->]]
                      | [n [ty' [Hcanonical [Hfind Hexpanded]]]]]]]]].
  - intros s kind Hin.
    rewrite (of_non_alias_type_aliases _ _ Hexpanded) in Hin.
    exact (Hty s kind Hin).
  - intros s' kind' Hin.
    rewrite (of_non_alias_type_aliases _ _ Hexpanded) in Hin.
    destruct (kind ty) as [ | [] | | ]; destruct Hin.
  - apply create_const_kinded.
  - intros s' kind' Hin.
    rewrite (of_non_alias_type_aliases _ _ Hexpanded) in Hin.
    exact (find_kinded _ _ _ _ Hkinded Hfind s' kind' Hin).
Qed.

Theorem expand_head_equations_kinded
  : expand_head_equations_kinded_statement.
Proof.
  intros tenv ty expanded Hkinded Hty Hexpand.
  destruct (expand_head_cases tenv ty expanded Hexpand)
    as [[_ Hexpanded]
       | [s [Halias [[_ Hexpanded]
                    | [[c [Hcanonical ->]]
                      | [n [ty' [Hcanonical [Hfind Hexpanded]]]]]]]]].
  - intros n kind Hin.
    rewrite (of_non_alias_type_equation_names _ _ Hexpanded) in Hin.
    exact (Hty n kind Hin).
  - intros n kind' Hin.
    rewrite (of_non_alias_type_equation_names _ _ Hexpanded) in Hin.
    destruct (kind ty) as [ | [] | | ]; destruct Hin.
  - intros n kind Hin. rewrite create_const_equation_names in Hin.
    destruct Hin.
  - intros m kind' Hin.
    rewrite (of_non_alias_type_equation_names _ _ Hexpanded) in Hin.
    exact (find_equations_kinded _ _ _ _ Hkinded Hfind m kind' Hin).
Qed.

(* What is used of the predicate of well-formed types, beside what
   its own file states: the type of a constant, expanded, is well
   formed, and a well-formed type that is not an alias is the type
   made of what it expands to. *)
Lemma expanded_create_const_well_formed (c : const) :
  type_well_formed (Expanded_type.to_type (Expanded_type.create_const c))
  = true.
Proof.
  destruct c as [i | i | f | f | i | i | i | i | i | i | i | i | i | | k];
    try reflexivity.
  apply unknown_well_formed.
Qed.

Lemma expanded_of_non_alias_well_formed (ty : flambda_type)
    (expanded : Expanded_type.t) :
  Expanded_type.of_non_alias_type ty = Some expanded ->
  type_well_formed ty = true ->
  type_well_formed (Expanded_type.to_type expanded) = true.
Proof.
  intros Hexpanded Hwf.
  destruct ty as [ descr | descr | k descr | descr | descr ];
    destruct descr as [ | [ head | s ] | ]; cbn in Hexpanded;
    try discriminate Hexpanded; injection Hexpanded as <-;
    try exact Hwf; try reflexivity.
  - destruct k; try reflexivity; discriminate Hwf.
  - destruct k; try reflexivity; discriminate Hwf.
Qed.

Theorem expand_head_well_formed : expand_head_well_formed_statement.
Proof.
  intros tenv ty expanded Hexpand Htypes Hty.
  destruct (expand_head_cases tenv ty expanded Hexpand)
    as [[_ Hexpanded]
       | [s [Halias [[_ Hexpanded]
                    | [[c [Hcanonical ->]]
                      | [n [ty' [Hcanonical [Hfind Hexpanded]]]]]]]]].
  - exact (expanded_of_non_alias_well_formed _ _ Hexpanded Hty).
  - apply (expanded_of_non_alias_well_formed _ _ Hexpanded).
    apply unknown_well_formed.
  - apply expanded_create_const_well_formed.
  - apply (expanded_of_non_alias_well_formed _ _ Hexpanded).
    exact (find_well_formed _ _ _ _ Htypes Hfind).
Qed.

(* The alias that a type is, is among its aliases, at its kind. *)
Lemma alias_among_aliases (ty : flambda_type) (s : simple) :
  get_alias_opt ty = Some s -> (s, kind ty) ∈ aliases_with_kinds ty.
Proof.
  destruct ty as [[ | [? | ?] | ] | [ | [? | ?] | ] | k [ | [? | ?] | ]
                  | [ | [? | ?] | ] | [ | [? | ?] | ]];
    simpl; intros H; try discriminate H; injection H as <-; left;
    reflexivity.
Qed.

Lemma kind_unknown (k : flambda_kind) : kind (unknown k) = k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Theorem expand_head_kind : expand_head_kind_statement.
Proof.
  intros tenv ty expanded Hconstants Hty Hexpand.
  destruct (expand_head_cases tenv ty expanded Hexpand)
    as [[_ Hexpanded]
       | [s [Halias [[_ Hexpanded]
                    | [[c [Hcanonical ->]]
                      | [n [ty' [Hcanonical [Hfind Hexpanded]]]]]]]]].
  - exact (of_non_alias_type_kind _ _ Hexpanded).
  - destruct (of_non_alias_type_kind _ _ Hexpanded) as [H1 H2].
    rewrite kind_unknown in H1, H2. split; assumption.
  - destruct (create_const_kind c) as [H1 H2]. rewrite H1, H2.
    assert (Hkind : kind (type_for_const c) = kind ty)
      by exact (Hconstants s (kind ty) c
                  (Hty s (kind ty) (alias_among_aliases ty s Halias))
                  Hcanonical).
    split; exact Hkind.
  - destruct (of_non_alias_type_kind _ _ Hexpanded) as [H1 H2].
    rewrite (find_kind _ _ _ _ Hfind) in H1, H2. split; assumption.
Qed.

Theorem expand_head_head_of_kind : expand_head_head_of_kind_statement.
Proof.
  intros tenv ty expanded head Hexpand Hhead.
  destruct (expand_head_cases tenv ty expanded Hexpand)
    as [[_ Hexpanded]
       | [s [Halias [[_ Hexpanded]
                    | [[c [Hcanonical ->]]
                      | [n [ty' [Hcanonical [Hfind Hexpanded]]]]]]]]].
  - exact (of_non_alias_type_head _ _ _ Hexpanded Hhead).
  - exact (of_non_alias_type_head _ _ _ Hexpanded Hhead).
  - exact (create_const_head c head Hhead).
  - exact (of_non_alias_type_head _ _ _ Hexpanded Hhead).
Qed.

Theorem expand_head_other_kind : expand_head_other_kind_statement.
Proof. eexists. split; [ vm_compute; reflexivity | reflexivity ]. Qed.
