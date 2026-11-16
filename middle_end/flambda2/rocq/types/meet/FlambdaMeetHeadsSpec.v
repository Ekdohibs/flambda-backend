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


(* What the meets of the heads of kind value keep: the variables that
   stand for the result of a primitive on the value, the heads that
   are not null, variants, and the heads that may be null.

   This file is to be reviewed: the definitions of what it is for each
   part to fit an environment and to be true, and the statements, which
   are about functions of FlambdaMeet.v. They are in the form of
   FlambdaMeetStepsSpec.v ([part_meet_ok]), and are for the proof of
   the contract of the whole meet (FlambdaMeetSpec.v). The proofs are
   in FlambdaMeetHeadsProofs.v, which needs no review.

   A part of a type, here, fits and is true as the type that it makes
   does ([fits_as_type], [satisfied_as_type]): a head that is not null
   as the type of the values that are not null and have it; the row of
   the closures of a set as the type of those closures. What it is for
   a type to fit is [type_fits] (FlambdaMeetSpec.v), so what is said of
   a part follows whatever [type_fits] comes to ask.

   What is stated so far.
   - The meet of two variables that stand for the result of one
     primitive on the value in hand ([Is_int], [Get_tag], [Is_null])
     is right ([meet_relation_ok_statement]). Such a variable, if there
     is one, fits an environment that knows it and has it at the kind
     of naked immediates; it is true where it holds the number that
     the primitive answers on the value ([variable_is],
     FlambdaTypeMeaningSpec.v). The number may depend on the static
     environment, in which it is read what a block is. The meet makes
     the two variables aliases ([Meet_env.add_alias]): both hold that
     number.
   - What [meet_disjunction] answers for each of its two parts
     ([meet_disjunction_answer_statement]): the part of an input, the
     bottom of that side, or what the meet of that side answered for
     the two inputs. It is about the answer only: no environment and
     no value is supposed. It is for the head that may be null, where
     what is true of a value that is not null says something of the
     other side too (the variable for [Is_null] holds 0): the meet of
     that side never answers a new thing, so the variable answered is
     that of an input, which the environment of the meet knew, and it
     holds what it held ([extended_from] keeps the values of the
     variables that the environment knew).
   - The meet of two heads that are not null,
     [meet_head_of_kind_value_non_null], is right
     ([meet_head_of_kind_value_non_null_ok_statement]), if the meet of
     two variants and the meet of the rows of two sets of closures
     are: those two are premises here, each in the same form, to be
     discharged by their own statements. Two heads made with different
     constructors have no value in common, but for a variant and a
     mutable block; no value has the type of a mutable block, in the
     meaning of types as it is.
   - The meet of two variants, [meet_variant], is right
     ([meet_variant_ok_statement]), if the meet of the rows of two
     sets of blocks is, which is a premise in the same form: the
     blocks of a variant fit, and are true, as the type of the
     variants that are blocks among them ([blocks_type]). It is by
     the statement of the meet of two things that are each one of two
     alternatives ([meet_disjunction_ok_statement],
     FlambdaMeetScopesSpec.v), the alternatives being a tagged
     immediate and a block. The variable for [Is_int] is met outside
     the two: it holds 1 on a tagged immediate and 0 on a block, by
     [meet_relation_ok_statement] at the number that reads which the
     value is.
   - The meet of two heads that may be null,
     [meet_head_of_kind_value], is right
     ([meet_head_of_kind_value_ok_statement]), if the meet of two
     heads that are not null is; the alternatives are a value that is
     not null and null. With the three statements before it, it is
     right if the meets of the rows of blocks and of the rows of
     closures are ([meet_value_heads_ok_statement]): those two are
     what is left to prove of the heads of kind value. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvKindedSpec FlambdaMeetGrammar
  FlambdaExpandHead FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetStepsSpec FlambdaMeetEnvSpec FlambdaMeetScopesSpec
  FlambdaMeetProductsSpec.
Import ListNotations.

(* ======================================================================== *)
(* A part of a type                                                         *)
(* ======================================================================== *)

(* A part of a type fits an environment, and is true of the value in
   hand, as the type that it makes does. *)
Definition fits_as_type {X : Type} (to_type : X -> flambda_type)
    (tenv : Typing_env.t) (x : X) : Prop :=
  type_fits tenv (to_type x).

Definition satisfied_as_type {X : Type} (to_type : X -> flambda_type)
    (v : value) (statics : static_env) (valuation : env) (x : X) : Prop :=
  type_satisfied statics valuation v (to_type x).

(* The meet of such parts is right. *)
Definition type_part_meet_ok {X : Type} (to_type : X -> flambda_type)
    (v : value) (meet : Meet_env.t -> X -> X -> option (Meet_result.t X))
    : Prop :=
  part_meet_ok (fits_as_type to_type) (satisfied_as_type to_type v) meet.

(* ======================================================================== *)
(* The variables that stand for the result of a primitive                   *)
(* ======================================================================== *)

(* The variable of a type that stands for the result of a primitive on
   the value of the type, if the type has one, fits an environment: the
   environment knows it, and has it at the kind of naked immediates.
   This is what [type_fits] says of it, where a type has it (it is one
   of the free variables of the type, and [aliases_with_kinds] lists it
   at that kind). *)
Definition relation_variable_fits (tenv : Typing_env.t)
    (x : option variable) : Prop :=
  match x with
  | Some x =>
    Typing_env.mem None tenv (Name_var x) = true
    ∧ simple_of_kind tenv (Var x) (Naked_number Kind_naked_immediate)
  | None => True
  end.

(* It is true: there is a number for the value in hand, [number], and
   the variable holds it. *)
Definition relation_variable_satisfied (number : static_env -> option Z)
    (statics : static_env) (valuation : env) (x : option variable) : Prop :=
  ∃ n : Z, number statics = Some n ∧ variable_is valuation x n.

(* [meet_relation] is right, whatever the number. *)
Definition meet_relation_ok_statement : Prop :=
  ∀ (mh : meet_expanded_head) (number : static_env -> option Z),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    part_meet_ok relation_variable_fits (relation_variable_satisfied number)
      (meet_relation mh).

(* ======================================================================== *)
(* What a disjunction answers for each of its two parts                     *)
(* ======================================================================== *)

(* What [meet_disjunction] may have answered for one of its two parts,
   of which the inputs have [c1] and [c2]: one of the two; the bottom
   of that side; or what a call of the meet of that side on the two
   answered. *)
Definition side_answer {C : Type}
    (meet_c : Meet_env.t -> C -> C -> option (Meet_result.t C))
    (bottom_c : unit -> C) (c1 c2 c : C) : Prop :=
  c = c1
  ∨ c = c2
  ∨ c = bottom_c tt
  ∨ ∃ (menv menv' : Meet_env.t) (r : meet_return_value C),
      meet_c menv c1 c2 = Some (Meet_result.Ok r menv')
      ∧ c = extract_value r c1 c2.

Definition meet_disjunction_answer_statement : Prop :=
  ∀ (A B : Type) (join : join_scoped_envs) (choices : meet_choices)
    (mh : meet_expanded_head)
    (meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (bottom_a : unit -> A) (bottom_b : unit -> B) (menv menv' : Meet_env.t)
    (a1 a2 a : A) (b1 b2 b : B) (extensions1 extensions2 extensions :
      variant_extensions)
    (r : meet_return_value (A * B * variant_extensions)),
    meet_disjunction join choices mh meet_a meet_b bottom_a bottom_b menv a1
      b1 extensions1 a2 b2 extensions2
    = Some (Meet_result.Ok r menv') ->
    meet_answer r (a1, b1, extensions1) (a2, b2, extensions2)
      (a, b, extensions) ->
    side_answer meet_a bottom_a a1 a2 a ∧ side_answer meet_b bottom_b b1 b2 b.

(* ======================================================================== *)
(* The heads that are not null                                              *)
(* ======================================================================== *)

(* The parts of a variant that are met: the variables for [Is_int] and
   [Get_tag], the blocks, the immediates and the extensions, in the
   order in which [meet_variant] answers them. *)
Definition variant_parts : Type :=
  option variable * option variable * Or_unknown.t row_like_for_blocks
  * Or_unknown.t flambda_type * variant_extensions.

Definition variant_of_parts (is_unique : bool) (parts : variant_parts)
    : head_of_kind_value_non_null :=
  let '(is_int, get_tag, blocks, immediates, extensions) := parts in
  Variant is_int immediates get_tag blocks extensions is_unique.

(* [meet_variant], on the parts of two variants. *)
Definition meet_variant_parts (join : join_scoped_envs)
    (choices : meet_choices) (mh : meet_expanded_head) (env : Meet_env.t)
    (parts1 parts2 : variant_parts) : option (Meet_result.t variant_parts) :=
  let '(is_int1, get_tag1, blocks1, immediates1, extensions1) := parts1 in
  let '(is_int2, get_tag2, blocks2, immediates2, extensions2) := parts2 in
  meet_variant join choices mh env is_int1 get_tag1 blocks1 immediates1
    is_int2 get_tag2 blocks2 immediates2 extensions1 extensions2.

(* [meet_head_of_kind_value_non_null] is right, for any value, if the
   meet of two variants and the meet of the rows of two sets of
   closures are. (Whether a variant is unique, and the allocation mode
   of closures, are no part of what it is for a type to fit or to be
   true: the two premises are stated at one of each.) *)
Definition meet_head_of_kind_value_non_null_ok_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (mh : meet_expanded_head) (v : value),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    type_part_meet_ok
      (fun parts : variant_parts =>
         non_null_value (variant_of_parts false parts))
      v (meet_variant_parts join choices mh) ->
    type_part_meet_ok (create_closures Alloc_mode.Heap) v
      (meet_row_like_for_closures join choices mh) ->
    type_part_meet_ok non_null_value v
      (meet_head_of_kind_value_non_null join choices mh).

(* ======================================================================== *)
(* Variants                                                                 *)
(* ======================================================================== *)

(* The blocks of a variant, as a type: the variants that are blocks
   among them. *)
Definition blocks_type (blocks : row_like_for_blocks) : flambda_type :=
  create_variant false
    (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
    (Or_unknown.Known blocks) No_extensions.

(* [meet_variant] is right, for any value, if the meet of the rows of
   two sets of blocks is. The join is to have its contract, and the
   meet is not to define again a variable that the environment knows,
   as [meet_disjunction_ok_statement] supposes. *)
Definition meet_variant_ok_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (mh : meet_expanded_head) (v : value),
    join_scoped_envs_sound join ->
    define_again choices = false ->
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    type_part_meet_ok blocks_type v
      (meet_row_like_for_blocks join choices mh) ->
    type_part_meet_ok
      (fun parts : variant_parts =>
         non_null_value (variant_of_parts false parts))
      v (meet_variant_parts join choices mh).

(* ======================================================================== *)
(* The heads that may be null                                               *)
(* ======================================================================== *)

(* The type that a head of kind value makes. *)
Definition value_head_type (head : head_of_kind_value) : flambda_type :=
  Type_value (Ok (No_alias head)).

(* [meet_head_of_kind_value] is right, for any value, if the meet of
   two heads that are not null is. *)
Definition meet_head_of_kind_value_ok_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (mh : meet_expanded_head) (v : value),
    join_scoped_envs_sound join ->
    define_again choices = false ->
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    type_part_meet_ok non_null_value v
      (meet_head_of_kind_value_non_null join choices mh) ->
    type_part_meet_ok value_head_type v
      (meet_head_of_kind_value join choices mh).

(* The three statements together: the meet of two heads of kind value
   is right if the meets of the rows of blocks and of the rows of
   closures are. *)
Definition meet_value_heads_ok_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (mh : meet_expanded_head) (v : value),
    join_scoped_envs_sound join ->
    define_again choices = false ->
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    type_part_meet_ok blocks_type v
      (meet_row_like_for_blocks join choices mh) ->
    type_part_meet_ok (create_closures Alloc_mode.Heap) v
      (meet_row_like_for_closures join choices mh) ->
    type_part_meet_ok value_head_type v
      (meet_head_of_kind_value join choices mh).
