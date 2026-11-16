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


(* Examples of meets: the steps of a match.

   This file is to be reviewed: each statement says what the
   transcribed meet answers on inputs made by hand, after the calls
   of the types library that Simplify makes for the function

     type t = A | B | C of int
     let f x = match x with A -> 10 | B -> 20 | C n -> n

   The inputs are scenarios that the function suggests, written from
   a reading of Simplify. They are not the trace of a run of the
   compiler on it.

   The match asks whether [x] is an immediate. If it is, it untags
   [x] and switches on the number; if it is not, it loads the field
   of the block. The calls (in simplify/):
   - [Is_int] of [x], of a match on a variant
     ([simplify_relational_primitive], simplify_unary_primitive.ml):
     the result is defined and [Typing_env.add_is_int_relation] is
     called;
   - an arm of a [Switch] ([simplify_switch],
     simplify_switch_expr.ml): [Meet.meet] of the type of the
     scrutinee, which is an alias type, and of the type of the number
     of the arm. The arm is removed if the answer is bottom, and is
     simplified in the environment answered if it is not;
   - [Untag_immediate] of [x], and the load of a field of [x]
     ([simplify_projection], simplify_common.ml, called by
     [simplify_untag_immediate] and [simplify_immutable_block_load0]
     of simplify_unary_primitive.ml): the result is defined and the
     type of [x] is met with a shape that names the result
     ([Meet.meet_shape], which is [Meet.meet] without the type that
     it answers; the statements are about that meet). The primitive
     is invalid if the answer is bottom. A load comes to this meet
     only where a faster reading of the type of [x] has neither
     found the field nor shown the load invalid
     ([meet_block_field_simple]).

   The environments answered are long, so each is named here as what
   the meet that is to be proved answers with one join
   ([forgetful_join] of FlambdaMeetExamplesSpec.v), and the statements
   say two things of it. The first is that the meet answers it
   whatever the join, and that the compiler's meet ([compiler_meet],
   FlambdaMeetSpec.v) answers the same: no step here calls the join,
   and none is at one of the three places where the two meets differ.
   The second is what the environment says of the variables: the type
   that [Typing_env.find] answers and the canonical element.

   - [match_is_int_statement]: after [Is_int], the result has the
     relation to [x] and [x] is a variant that has the result for
     [Is_int].
   - [match_arm_immediate_statement]: in the arm for 1 of the switch
     on the result, the result is the number 1 and [x] has no blocks.
   - [match_arm_block_statement]: in the arm for 0, the result is the
     number 0 and [x] has no immediates.
   - [match_no_other_arm_statement]: an arm for 2 would be removed.
   - [match_untag_statement]: after [Untag_immediate] in the arm of
     the immediates, the immediates of [x] are the result.
   - [match_arm_constructor_statement]: in the arm for 0 of the
     switch on that result (the constructor [A]), the result is the
     number 0. The type of [x] is not changed: it names the result.
   - [match_inner_arm_statement]: in that arm, the arm for 1 of
     another switch on the result (a second match on [x], its arm for
     [B]) is removed.
   - [match_load_statement]: after the load in the arm of the blocks,
     [x] is a block of tag 0 that has at least one field, which is
     the result.
   - [match_load_of_immediate_statement]: the meet of the same load
     in the arm of the immediates answers bottom.

   The fuel is 6 throughout. The answers are those of the model.
   The corresponding concrete scenarios have since been run with
   both of the compiler's meets, by the workstream of the differential
   tests of the types library. The type or bottom that the meet
   returns and the selected queries of a type or of a canonical
   element agree with the answers recorded here. That does not compare
   whole resulting environments, nor test the universally quantified
   parts of these statements.
   Examples of meets that call the join of the model belong with
   that join, which is another workstream's. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaMeetExamplesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* What is observed                                                         *)
(* ======================================================================== *)

Local Abbreviation naked_immediate := (Naked_number Kind_naked_immediate).

(* The type that an environment has for a variable, and the canonical
   element of the variable. *)
Definition type_of_variable (tenv : Typing_env.t) (x : variable)
    (kind : flambda_kind) : option flambda_type :=
  Typing_env.find tenv (Name_var x) (Some kind).

Definition canonical_of_variable (tenv : Typing_env.t) (x : variable)
    : option (option simple) :=
  Typing_env.get_canonical_simple_exn tenv (Some Name_mode.in_types) None
    (Var x).

(* The environment that a function answered, and the empty one if it
   did not answer one. *)
Definition env_answered (result : option Typing_env.t) : Typing_env.t :=
  match result with
  | Some tenv => tenv
  | None => Typing_env.create
  end.

Definition env_of_meet
    (result : option (Or_bottom.t (flambda_type * Typing_env.t)))
    : Typing_env.t :=
  match result with
  | Some (Or_bottom.Ok (_, tenv)) => tenv
  | Some Or_bottom.Bottom | None => Typing_env.create
  end.

(* Both meets answer a type and an environment, whatever the join. *)
Definition meets_answer (tenv : Typing_env.t) (t1 t2 t : flambda_type)
    (tenv' : Typing_env.t) : Prop :=
  ∀ join : join_scoped_envs,
    meet_to_prove join 6 tenv t1 t2 = Some (Or_bottom.Ok (t, tenv'))
    ∧ compiler_meet join 6 tenv t1 t2 = Some (Or_bottom.Ok (t, tenv')).

(* Both meets answer bottom, whatever the join. *)
Definition meets_answer_bottom (tenv : Typing_env.t) (t1 t2 : flambda_type)
    : Prop :=
  ∀ join : join_scoped_envs,
    meet_to_prove join 6 tenv t1 t2 = Some Or_bottom.Bottom
    ∧ compiler_meet join 6 tenv t1 t2 = Some Or_bottom.Bottom.

(* ======================================================================== *)
(* The variables                                                            *)
(* ======================================================================== *)

(* The parameter, the result of [Is_int], the result of
   [Untag_immediate] and the result of the load. *)
Definition match_x : variable := 1%positive.
Definition match_is_int : variable := 2%positive.
Definition match_untagged : variable := 3%positive.
Definition match_n : variable := 4%positive.

(* The types of a variant that are answered below: with a variable for
   [Is_int], immediates and blocks. *)
Definition match_variant (immediates : Or_unknown.t flambda_type)
    (blocks : Or_unknown.t row_like_for_blocks) : flambda_type :=
  Type_value
    (Ok
       (No_alias
          (Head_of_kind_value
             (Ok
                (Variant (Some match_is_int) immediates None blocks
                   No_extensions false))
             Not_null))).

(* ======================================================================== *)
(* [Is_int]                                                                 *)
(* ======================================================================== *)

(* The parameter is defined, and then the result of [Is_int]. *)
Definition match_env_start : Typing_env.t :=
  let tenv :=
    Typing_env.add_variable_definition Typing_env.create match_x Value
      Name_mode.normal
  in
  Typing_env.add_variable_definition tenv match_is_int naked_immediate
    Name_mode.normal.

Definition match_env_is_int : Typing_env.t :=
  env_answered
    (Flambda2_types.add_is_int_relation forgetful_join repaired_choices 6
       match_env_start (Name_var match_is_int) (Var match_x)).

(* The relation is added, whatever the join and the choices; the
   result is 0 or 1 and is [Is_int] of [x], and [x] is a variant that
   has the result for [Is_int]. *)
Definition match_is_int_statement : Prop :=
  (∀ (join : join_scoped_envs) (choices : meet_choices),
    Flambda2_types.add_is_int_relation join choices 6 match_env_start
      (Name_var match_is_int) (Var match_x)
    = Some match_env_is_int)
  ∧ type_of_variable match_env_is_int match_is_int naked_immediate
    = Some (is_int_for_scrutinee (Name_var match_x))
  ∧ type_of_variable match_env_is_int match_x Value
    = Some (match_variant Or_unknown.Unknown Or_unknown.Unknown).

(* ======================================================================== *)
(* The switch on the result of [Is_int]                                     *)
(* ======================================================================== *)

Definition match_is_int_type : flambda_type :=
  alias_type_of naked_immediate (Var match_is_int).

(* The arm for 1: [x] is an immediate. *)
Definition match_env_immediate : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6 match_env_is_int match_is_int_type
       (this_naked_immediate 1)).

Definition match_arm_immediate_statement : Prop :=
  meets_answer match_env_is_int match_is_int_type (this_naked_immediate 1)
    match_is_int_type match_env_immediate
  ∧ canonical_of_variable match_env_immediate match_is_int
    = Some (Some (Const (Naked_immediate 1)))
  ∧ type_of_variable match_env_immediate match_x Value
    = Some
        (match_variant Or_unknown.Unknown
           (Or_unknown.Known row_like_for_blocks_bottom)).

(* The arm for 0: [x] is a block. *)
Definition match_env_block : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6 match_env_is_int match_is_int_type
       (this_naked_immediate 0)).

Definition match_arm_block_statement : Prop :=
  meets_answer match_env_is_int match_is_int_type (this_naked_immediate 0)
    match_is_int_type match_env_block
  ∧ canonical_of_variable match_env_block match_is_int
    = Some (Some (Const (Naked_immediate 0)))
  ∧ type_of_variable match_env_block match_x Value
    = Some
        (match_variant (Or_unknown.Known (bottom naked_immediate))
           Or_unknown.Unknown).

(* There is no other arm. *)
Definition match_no_other_arm_statement : Prop :=
  meets_answer_bottom match_env_is_int match_is_int_type
    (this_naked_immediate 2).

(* ======================================================================== *)
(* The immediates                                                           *)
(* ======================================================================== *)

(* [Untag_immediate] of [x], in the arm of the immediates: the result
   is defined, and the type of [x] is met with the type of the tagged
   immediates of which the result is the number. *)
Definition match_env_untag_start : Typing_env.t :=
  Typing_env.add_variable_definition match_env_immediate match_untagged
    naked_immediate Name_mode.normal.

Definition match_x_type : flambda_type := alias_type_of Value (Var match_x).

Definition match_env_untag : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6 match_env_untag_start match_x_type
       (tagged_immediate_alias_to match_untagged)).

Definition match_untag_statement : Prop :=
  meets_answer match_env_untag_start match_x_type
    (tagged_immediate_alias_to match_untagged) match_x_type match_env_untag
  ∧ type_of_variable match_env_untag match_x Value
    = Some
        (match_variant
           (Or_unknown.Known
              (alias_type_of naked_immediate (Var match_untagged)))
           (Or_unknown.Known row_like_for_blocks_bottom))
  ∧ type_of_variable match_env_untag match_untagged naked_immediate
    = Some (unknown naked_immediate).

(* The arm for 0 of the switch on the result, which is the constructor
   [A]. *)
Definition match_untagged_type : flambda_type :=
  alias_type_of naked_immediate (Var match_untagged).

Definition match_env_constructor : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6 match_env_untag match_untagged_type
       (this_naked_immediate 0)).

Definition match_arm_constructor_statement : Prop :=
  meets_answer match_env_untag match_untagged_type (this_naked_immediate 0)
    match_untagged_type match_env_constructor
  ∧ canonical_of_variable match_env_constructor match_untagged
    = Some (Some (Const (Naked_immediate 0)))
  ∧ type_of_variable match_env_constructor match_x Value
    = type_of_variable match_env_untag match_x Value.

(* In that arm, the result is not 1. *)
Definition match_inner_arm_statement : Prop :=
  meets_answer_bottom match_env_constructor match_untagged_type
    (this_naked_immediate 1).

(* ======================================================================== *)
(* The block                                                                *)
(* ======================================================================== *)

(* The load of the field of [x], in the arm of the blocks: the result
   is defined, and the type of [x] is met with the type of the blocks
   of tag 0 that have at least one field, which is the result. *)
Definition match_env_load_start : Typing_env.t :=
  Typing_env.add_variable_definition match_env_block match_n Value
    Name_mode.normal.

Definition match_load_shape : flambda_type :=
  match
    immutable_block_with_size_at_least (Or_unknown.Known 0) 1
      (Scannable Value_only) match_n
  with
  | Some shape => shape
  | None => bottom Value
  end.

(* The blocks of tag 0 that have at least one field, which is the
   result of the load. *)
Definition match_blocks_loaded : row_like_for_blocks :=
  Row_like_for_blocks
    [(0,
      Or_unknown.Known
        {| case_maps_to := [alias_type_of Value (Var match_n)];
           case_index :=
             {| index_domain := Row_like_index_domain.At_least 1%nat;
                index_shape := Scannable Value_only |};
           case_env_extension := empty_env_extension |})]
    Or_bottom.Bottom Alloc_mode.Heap_or_local.

Definition match_env_load : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6 match_env_load_start match_x_type
       match_load_shape).

Definition match_load_statement : Prop :=
  meets_answer match_env_load_start match_x_type match_load_shape
    match_x_type match_env_load
  ∧ type_of_variable match_env_load match_x Value
    = Some
        (match_variant (Or_unknown.Known (bottom naked_immediate))
           (Or_unknown.Known match_blocks_loaded))
  ∧ type_of_variable match_env_load match_n Value = Some (unknown Value).

(* The same load in the arm of the immediates has no value. *)
Definition match_load_of_immediate_statement : Prop :=
  meets_answer_bottom
    (Typing_env.add_variable_definition match_env_immediate match_n Value
       Name_mode.normal)
    match_x_type match_load_shape.
