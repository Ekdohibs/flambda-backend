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


(* The three places where the compiler's meet is not the meet that is
   proved: when each matters, and what fails where it does.

   This file is to be reviewed: three definitions, two examples and
   fourteen statements. The proofs are in FlambdaMeetChoicesProofs.v, which
   needs no review.

   The main theorem (FlambdaMeetClosuresSpec.v) is for the meet with
   [repaired_choices] (FlambdaMeet.v). The compiler's meet is the one
   with [compiler_choices], which differs at three places. For each
   place this file has:
   - the condition, at the function where the choice is read, under
     which the compiler's choice does what the repaired one does, and
     what the two do where it fails;
   - an example, computed, of the meet with that one choice as the
     compiler has it and the two others repaired, on which a part of
     the contract of the meet ([meet_contract], FlambdaMeetSpec.v)
     fails.

   1. The type that the step for [Get_tag] gives the scrutinee
      (discrepancy T4-6; "The step that is not sound" in
      FlambdaMeetSpec.v). The compiler's type,
      [blocks_with_these_tags], is true of a scrutinee that the
      relation holds of exactly where the scrutinee is a block
      ([get_tag_reduction_of_block_statement]); the repaired one is
      true of every such scrutinee. So the choice matters only where
      a name on which a [Get_tag] relation is recorded may hold an
      immediate; unboxing records such relations. What fails:
      [meet_sound]. The example is that of FlambdaMeetExamplesSpec.v,
      here with this choice alone
      ([only_get_tag_not_sound_statement]); its input is shown to be
      one that the contract speaks of
      ([example_get_tag_inputs_statement]), which the statement there
      supposes, so that the compiler's meet has not [meet_sound]
      outright ([compiler_meet_not_sound_outright_statement]). The
      compiler's two meets
      answer as the model does on such inputs (the differential test
      of the types library); no term that does not itself read the
      tag of an immediate was found to reach it.

   2. Whether a variable that the environment knows is defined again
      (discrepancy T4-1). The choice is read where a variable is
      defined after a join ([define_in_types], FlambdaMeet.v). The
      two choices do the same where the environment does not know
      the variable; where it does, the repaired choice does nothing
      and the compiler's defines it a second time
      ([define_in_types_choice_statement]). So the choice matters
      only where the environment that a join answered already has a
      variable that the meet then defines: where the join brought in
      a variable of an environment that it joined. What fails:
      [meet_keeps_well_formed], the environment answered knows a
      variable at two binding times and is not well formed
      ([example_rebind_statement],
      [defining_again_not_keeps_statement]). With the repaired choice
      the environment answered on the same input is well formed
      ([example_rebind_repaired_statement]). The compiler shows this
      fault with types built by hand only (a fatal error, "Cannot
      rebind", under heavy invariant checks); it was not reached from
      OCaml source.

   3. Whether the variables of a case that is alone are lifted
      (discrepancy T4-9). The choice is read in [meet_row_like],
      where one case only is kept. The lifting meet records, among
      the variables that it defines in its result, those that the
      environment of the case defines after the scope of the meet;
      the other records nothing. Recording changes nothing exactly
      where each of those variables is recorded already
      ([union_extra_variables_same_statement]); in particular where
      there is none. (To record them the lifting meet cuts the level
      of the case's environment, and stops if that fails, which is a
      fatal error of the compiler.) So the choice matters only where
      a meet inside the one case that is kept made a variable, which
      only a join does. What fails: [meet_keeps_well_formed], the
      type answered mentions a variable that the environment
      answered does not know ([example_unbound_statement],
      [not_lifting_not_keeps_statement]). With the repaired choice
      every variable of the type answered is known
      ([example_unbound_repaired_statement]). The compiler shows this
      fault with types built by hand only; it was not reached from
      OCaml source.

   Why there is not one theorem "on inputs with P, the compiler's
   meet has the contract". For the second and the third place the
   condition is about what the join makes or brings in, at each call
   inside the meet, and not about the types met; and a condition on
   the types would have to hold of every environment and type that
   the traversal reaches. The two other parts of the contract,
   [meet_kinds] and [meet_bottom_env], hold of the meet with any
   choices ([meet_of_types_kinds_statement], FlambdaMeetKeepsSpec.v).

   The examples of the second and third places are run with the join
   of the model as the compiler has it ([model_meet] with
   [Join_env.compiler_choices], FlambdaJoinMeet.v), since the
   condition is about the join. Nothing is supposed of that join: the
   statements say what is computed, and that the input is one that
   the contract speaks of.

   The state at which the model stops and the compiler does not
   (discrepancy T4-10; FlambdaMeetEnv.v) is not a choice, and no
   premise that excludes it is stated here. The contract says
   nothing of a meet that does not answer. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec FlambdaJoinEnv
  FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaMeetExamplesSpec FlambdaJoinMeet.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* One choice as the compiler has it, the two others repaired               *)
(* ======================================================================== *)

Definition only_get_tag_of_blocks : meet_choices :=
  {| type_of_get_tag_scrutinee := blocks_with_these_tags;
     define_again := false;
     lift_variables_of_single_case := true |}.

Definition only_defining_again : meet_choices :=
  {| type_of_get_tag_scrutinee := immediate_or_blocks_with_these_tags;
     define_again := true;
     lift_variables_of_single_case := true |}.

Definition only_not_lifting : meet_choices :=
  {| type_of_get_tag_scrutinee := immediate_or_blocks_with_these_tags;
     define_again := false;
     lift_variables_of_single_case := false |}.

(* ======================================================================== *)
(* 1. The type of the scrutinee of [Get_tag]                                *)
(* ======================================================================== *)

(* The compiler's type is true of a value that the relation holds of
   exactly where the value is a block. The tags are those that the
   step computes from the numbers, as in
   [get_tag_reduction_repaired_statement] (FlambdaMeetSpec.v). *)
Definition get_tag_reduction_of_block_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (w : value) (n : Z) (numbers : list Z)
    (alloc_mode : Alloc_mode.t),
    relation_holds statics Relation.Get_tag w n ->
    n ∈ numbers ->
    (type_satisfied statics vars w
       (blocks_with_these_tags (Keyed.dedup Z.eqb (filter is_tag numbers))
          alloc_mode)
     <-> is_block statics w).

(* The input of the example of FlambdaMeetExamplesSpec.v is one that
   a meet may be given. The statement there
   ([compiler_meet_not_sound_statement]) supposes it. *)
Definition example_get_tag_inputs_statement : Prop :=
  meet_inputs_well_formed example_get_tag_env example_tag_type
    example_zero_type.

(* So the meet with this choice alone has not [meet_sound], whatever
   the join; nor has the meet with the compiler's three choices. The
   compiler's meets answer the same on that input, built with its
   library of types; no term that does not itself read the tag of an
   immediate was found to reach it. *)
Definition only_get_tag_not_sound_statement : Prop :=
  ∀ join : join_scoped_envs,
    ~ meet_sound (Meet.meet join only_get_tag_of_blocks 3).

Definition compiler_meet_not_sound_outright_statement : Prop :=
  ∀ join : join_scoped_envs, ~ meet_sound (compiler_meet join 3).

(* ======================================================================== *)
(* 2. Defining again                                                        *)
(* ======================================================================== *)

(* Where the choice is read: the two do the same on a variable that
   the environment does not know; on one that it knows, the repaired
   choice leaves the environment as it is and the compiler's defines
   the variable. *)
Definition define_in_types_choice_statement : Prop :=
  ∀ (choices choices' : meet_choices) (env : Meet_env.t) (var : variable)
    (kind : flambda_kind),
    define_again choices = true ->
    define_again choices' = false ->
    if Typing_env.mem None (Meet_env.typing_env env) (Name_var var)
    then
      define_in_types choices' env var kind = Some env
      ∧ define_in_types choices env var kind
        = Meet_env.add_definition env (Name_var var) Name_mode.in_types kind
    else
      define_in_types choices env var kind
      = define_in_types choices' env var kind.

(* The variant whose immediates are [immediates] and whose blocks are
   those of the tags given, each with fields of the types given and no
   others. *)
Definition example_variant (immediates : Or_unknown.t flambda_type)
    (cases : list (Z * list flambda_type)) : flambda_type :=
  create_variant false immediates
    (Or_unknown.Known
       (Row_like_for_blocks
          (map
             (fun case : Z * list flambda_type =>
                (fst case,
                 Or_unknown.Known
                   (Row_like_case (snd case)
                      (Row_like_index
                         (Row_like_index_domain.Known (length (snd case)))
                         (Scannable Value_only))
                      empty_env_extension)))
             cases)
          Or_bottom.Bottom Alloc_mode.Heap))
    No_extensions.

(* Blocks only. *)
Definition example_blocks (cases : list (Z * list flambda_type))
    : flambda_type :=
  example_variant
    (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate))) cases.

Definition example_alias (x : variable) : flambda_type :=
  alias_type_of Value (Var x).

Definition example_v1 : variable := 1%positive.
Definition example_v2 : variable := 2%positive.
Definition example_v3 : variable := 3%positive.

(* Three variables of kind value. *)
Definition example_three_values : Typing_env.t :=
  Typing_env.add_variable_definition
    (Typing_env.add_variable_definition
       (Typing_env.add_variable_definition Typing_env.create example_v1 Value
          Name_mode.normal)
       example_v2 Value Name_mode.normal)
    example_v3 Value Name_mode.normal.

Definition example_choices_fuel : fuels :=
  {| fuel_of_env := 20; fuel_of_types := 20; fuel_of_meet := 20 |}.

(* The meet of the model with the join of the model as the compiler
   has it. *)
Definition example_choices_meet (choices : meet_choices) : meet_function :=
  model_meet Join_env.compiler_choices choices 5 example_choices_fuel.

(* The second and the third variable are tagged immediates. *)
Definition example_rebind_env : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.replace_equation example_three_values (Name_var example_v2)
       any_tagged_immediate)
    (Name_var example_v3) any_tagged_immediate.

Definition example_pair (a b : flambda_type) : flambda_type :=
  immutable_block false 0 (Scannable Value_only) Alloc_mode.Heap [a; b].

(* The two types met. Under tag 0, the first has a block of tag 0 or
   1 whose field is the first variable, and the second a block whose
   field is a pair of the second variable twice (tag 0) or of the
   third twice (tag 1). The meet under tag 0 joins its two cases, and
   the join makes a variable for the component of the pair. The outer
   meet has tag 0 alone in both types: one case, whose environment
   defines that variable; the join of that one environment brings the
   variable in, and the meet then defines the variables of the
   case. *)
Definition example_rebind_left : flambda_type :=
  example_blocks
    [(0,
      [example_blocks
         [(0, [example_alias example_v1]); (1, [example_alias example_v1])]]);
     (1, [])].

Definition example_rebind_right : flambda_type :=
  example_blocks
    [(0,
      [example_blocks
         [(0,
           [example_pair (example_alias example_v2)
              (example_alias example_v2)]);
          (1,
           [example_pair (example_alias example_v3)
              (example_alias example_v3)])]]);
     (2, [])].

(* The input is one that the contract of the meet speaks of. *)
Definition example_rebind_inputs_statement : Prop :=
  meet_inputs_well_formed example_rebind_env example_rebind_left
    example_rebind_right.

(* Defining again: the environment answered is not well formed. The
   compiler shows this fault with types built by hand only. *)
Definition example_rebind_statement : Prop :=
  ∃ (t : flambda_type) (tenv' : Typing_env.t),
    example_choices_meet only_defining_again example_rebind_env
      example_rebind_left example_rebind_right
    = Some (Or_bottom.Ok (t, tenv'))
    ∧ typing_env_well_formed tenv' = false.

Definition defining_again_not_keeps_statement : Prop :=
  ~ meet_keeps_well_formed (example_choices_meet only_defining_again).

(* Not defining again: it is. *)
Definition example_rebind_repaired_statement : Prop :=
  ∃ (t : flambda_type) (tenv' : Typing_env.t),
    example_choices_meet repaired_choices example_rebind_env
      example_rebind_left example_rebind_right
    = Some (Or_bottom.Ok (t, tenv'))
    ∧ typing_env_well_formed tenv' = true.

(* ======================================================================== *)
(* 3. Lifting the variables of a case that is alone                         *)
(* ======================================================================== *)

(* Recording the variables of a level among the extra variables
   leaves them as they were exactly where each is recorded already,
   with its kind. *)
Definition union_extra_variables_same_statement : Prop :=
  ∀ extra vars : list (variable * flambda_kind),
    union_extra_variables extra vars = Some extra
    <-> ∀ (var : variable) (kind : flambda_kind),
          (var, kind) ∈ vars -> lookup var extra = Some kind.

(* The two types met: blocks of tag 0 with one field. In the first
   the field is an immediate or a block of tag 0 or 1 whose field is
   the first variable; in the second, an immediate or a block whose
   field is a pair of the second variable twice (tag 0) or of the
   third twice (tag 1). The outer meet keeps its one case; the meet
   of the field joins and makes a variable. *)
Definition example_unbound_left : flambda_type :=
  example_blocks
    [(0,
      [example_variant Or_unknown.Unknown
         [(0, [example_alias example_v1]); (1, [example_alias example_v1])]])].

Definition example_unbound_right : flambda_type :=
  example_blocks
    [(0,
      [example_variant Or_unknown.Unknown
         [(0,
           [example_blocks
              [(0, [example_alias example_v2; example_alias example_v2])]]);
          (1,
           [example_blocks
              [(0, [example_alias example_v3; example_alias example_v3])]])]])].

Definition example_unbound_inputs_statement : Prop :=
  meet_inputs_well_formed example_three_values example_unbound_left
    example_unbound_right.

(* Not lifting: the type answered mentions a variable that the
   environment answered does not know. The compiler shows this fault
   with types built by hand only. *)
Definition example_unbound_statement : Prop :=
  ∃ (t : flambda_type) (tenv' : Typing_env.t) (x : variable),
    example_choices_meet only_not_lifting example_three_values
      example_unbound_left example_unbound_right
    = Some (Or_bottom.Ok (t, tenv'))
    ∧ x ∈ free_variables t
    ∧ Typing_env.mem None tenv' (Name_var x) = false.

Definition not_lifting_not_keeps_statement : Prop :=
  ~ meet_keeps_well_formed (example_choices_meet only_not_lifting).

(* Lifting: every variable of the type answered is known. *)
Definition example_unbound_repaired_statement : Prop :=
  ∃ (t : flambda_type) (tenv' : Typing_env.t),
    example_choices_meet repaired_choices example_three_values
      example_unbound_left example_unbound_right
    = Some (Or_bottom.Ok (t, tenv'))
    ∧ ∀ x : variable,
        x ∈ free_variables t ->
        Typing_env.mem None tenv' (Name_var x) = true.
