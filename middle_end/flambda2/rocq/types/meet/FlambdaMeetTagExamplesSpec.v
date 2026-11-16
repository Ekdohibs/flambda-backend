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


(* Examples of meets: the tag of a block, and null.

   This file is to be reviewed: two more matches, in the manner of
   FlambdaMeetMatchExamplesSpec.v, whose definitions are used. The
   inputs are made by hand: they are scenarios that the two functions
   suggest, written from a reading of Simplify, and not the trace of
   a run of the compiler. The fuel is 6. The answers are those of
   the model.
   The corresponding concrete scenarios have since been run with
   both of the compiler's meets, by the workstream of the differential
   tests of the types library. The type or bottom that the meet
   returns and the selected queries of a type or of a canonical
   element agree with the answers recorded here. That does not compare
   whole resulting environments, nor test the universally quantified
   parts of these statements.

   - A variant that has two constructors with arguments and none
     without:

       type s = P of int | Q of int * int
       let g x = match x with P a -> a | Q (a, _) -> a

     The match reads the tag of [x] and switches on it; each arm
     loads a field. [Get_tag] records its relation
     ([simplify_get_tag], simplify/simplify_unary_primitive.ml): the
     environment is [example_get_tag_env] of
     FlambdaMeetExamplesSpec.v. The arm for 0 is the step where the
     meet that is to be proved is not the compiler's
     (FlambdaMeetSpec.v, "The step that is not sound"): the
     compiler's says that [x] is a block of tag 0, and the other that
     [x] is an immediate or a block of tag 0, which calls the join
     ([tags_arm_statement]; with the join that forgets). The load of
     the field in that arm then says that [x] is a block: in this
     scenario the later load gives the same type for [x] after either
     meet ([tags_load_statement], which is of that type and not of
     the whole environments).
   - A value that may be null:

       let get (x : int or_null) = match x with Null -> 0 | This n -> n

     [Is_null] records its relation ([simplify_is_null], the same
     file): the result is 0 or 1 and is [Is_null] of [x], and [x] may
     be null and has the result for [Is_null]
     ([null_relation_statement]). In the arm for 1 of the switch on
     the result, [x] is null ([null_arm_null_statement]); in the arm
     for 0 it is not, and its type no longer names the result
     ([null_arm_value_statement]). Both meets answer the same,
     whatever the join. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaMeetExamplesSpec FlambdaMeetMatchExamplesSpec.
Import ListNotations.

Local Open Scope Z_scope.

Local Abbreviation naked_immediate := (Naked_number Kind_naked_immediate).

(* ======================================================================== *)
(* The tag of a block                                                       *)
(* ======================================================================== *)

(* The parameter and the result of [Get_tag] are [example_p] and
   [example_tag]; the result of the load. *)
Definition tags_a : variable := 3%positive.

(* A variant that has [example_tag] for [Get_tag]. *)
Definition tags_variant (immediates : Or_unknown.t flambda_type)
    (blocks : row_like_for_blocks) : flambda_type :=
  Type_value
    (Ok
       (No_alias
          (Head_of_kind_value
             (Ok
                (Variant None immediates (Some example_tag)
                   (Or_unknown.Known blocks) No_extensions false))
             Not_null))).

(* The blocks of tag 0, of which nothing more is known. *)
Definition tags_blocks_of_tag_0 : row_like_for_blocks :=
  Row_like_for_blocks [(0, Or_unknown.Unknown)] Or_bottom.Bottom
    Alloc_mode.Heap_or_local.

(* The arm for 0 of the switch on the tag, with each of the two
   meets. *)
Definition tags_env_arm : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6 example_get_tag_env example_tag_type
       example_zero_type).

Definition tags_env_arm_compiler : Typing_env.t :=
  env_of_meet
    (compiler_meet forgetful_join 6 example_get_tag_env example_tag_type
       example_zero_type).

(* The compiler's meet says that [x] is a block of tag 0, whatever
   the join; the meet that is to be proved, that it is an immediate or
   a block of tag 0. *)
Definition tags_arm_statement : Prop :=
  meet_to_prove forgetful_join 6 example_get_tag_env example_tag_type
    example_zero_type
  = Some (Or_bottom.Ok (example_tag_type, tags_env_arm))
  ∧ (∀ join : join_scoped_envs,
      compiler_meet join 6 example_get_tag_env example_tag_type
        example_zero_type
      = Some (Or_bottom.Ok (example_tag_type, tags_env_arm_compiler)))
  ∧ type_of_variable tags_env_arm example_p Value
    = Some (tags_variant Or_unknown.Unknown tags_blocks_of_tag_0)
  ∧ type_of_variable tags_env_arm_compiler example_p Value
    = Some
        (tags_variant (Or_unknown.Known (bottom naked_immediate))
           tags_blocks_of_tag_0).

(* The load of the first field in that arm: the result is defined, and
   the type of [x] is met with the type of the blocks of tag 0 that
   have at least one field, which is the result. *)
Definition tags_x_type : flambda_type := alias_type_of Value (Var example_p).

Definition tags_load_shape : flambda_type :=
  match
    immutable_block_with_size_at_least (Or_unknown.Known 0) 1
      (Scannable Value_only) tags_a
  with
  | Some shape => shape
  | None => bottom Value
  end.

Definition tags_env_load : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6
       (Typing_env.add_variable_definition tags_env_arm tags_a Value
          Name_mode.normal)
       tags_x_type tags_load_shape).

Definition tags_env_load_compiler : Typing_env.t :=
  env_of_meet
    (compiler_meet forgetful_join 6
       (Typing_env.add_variable_definition tags_env_arm_compiler tags_a
          Value Name_mode.normal)
       tags_x_type tags_load_shape).

(* The blocks of tag 0 that have at least one field, which is the
   result of the load. *)
Definition tags_blocks_loaded : row_like_for_blocks :=
  Row_like_for_blocks
    [(0,
      Or_unknown.Known
        {| case_maps_to := [alias_type_of Value (Var tags_a)];
           case_index :=
             {| index_domain := Row_like_index_domain.At_least 1%nat;
                index_shape := Scannable Value_only |};
           case_env_extension := empty_env_extension |})]
    Or_bottom.Bottom Alloc_mode.Heap_or_local.

(* Each meet answers, after the arm that it made, and the two
   environments have one type for [x]: a block. *)
Definition tags_load_statement : Prop :=
  meet_to_prove forgetful_join 6
    (Typing_env.add_variable_definition tags_env_arm tags_a Value
       Name_mode.normal)
    tags_x_type tags_load_shape
  = Some (Or_bottom.Ok (tags_x_type, tags_env_load))
  ∧ compiler_meet forgetful_join 6
      (Typing_env.add_variable_definition tags_env_arm_compiler tags_a Value
         Name_mode.normal)
      tags_x_type tags_load_shape
    = Some (Or_bottom.Ok (tags_x_type, tags_env_load_compiler))
  ∧ type_of_variable tags_env_load example_p Value
    = Some
        (tags_variant (Or_unknown.Known (bottom naked_immediate))
           tags_blocks_loaded)
  ∧ type_of_variable tags_env_load_compiler example_p Value
    = type_of_variable tags_env_load example_p Value.

(* ======================================================================== *)
(* Null                                                                     *)
(* ======================================================================== *)

(* The parameter and the result of [Is_null]. *)
Definition null_x : variable := 1%positive.
Definition null_is_null : variable := 2%positive.

Definition null_env_start : Typing_env.t :=
  let tenv :=
    Typing_env.add_variable_definition Typing_env.create null_x Value
      Name_mode.normal
  in
  Typing_env.add_variable_definition tenv null_is_null naked_immediate
    Name_mode.normal.

Definition null_env_relation : Typing_env.t :=
  env_answered
    (Flambda2_types.add_is_null_relation forgetful_join repaired_choices 6
       null_env_start (Name_var null_is_null) (Var null_x)).

(* The relation is added, whatever the join and the choices. *)
Definition null_relation_statement : Prop :=
  (∀ (join : join_scoped_envs) (choices : meet_choices),
    Flambda2_types.add_is_null_relation join choices 6 null_env_start
      (Name_var null_is_null) (Var null_x)
    = Some null_env_relation)
  ∧ type_of_variable null_env_relation null_is_null naked_immediate
    = Some (is_null_for_scrutinee (Name_var null_x))
  ∧ type_of_variable null_env_relation null_x Value
    = Some
        (Type_value
           (Ok
              (No_alias
                 (Head_of_kind_value Unknown
                    (Maybe_null (Some null_is_null)))))).

Definition null_is_null_type : flambda_type :=
  alias_type_of naked_immediate (Var null_is_null).

(* The arm for 1: [x] is null. *)
Definition null_env_null : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6 null_env_relation null_is_null_type
       (this_naked_immediate 1)).

Definition null_arm_null_statement : Prop :=
  meets_answer null_env_relation null_is_null_type (this_naked_immediate 1)
    null_is_null_type null_env_null
  ∧ canonical_of_variable null_env_null null_is_null
    = Some (Some (Const (Naked_immediate 1)))
  ∧ type_of_variable null_env_null null_x Value
    = Some
        (Type_value
           (Ok
              (No_alias
                 (Head_of_kind_value Bottom
                    (Maybe_null (Some null_is_null)))))).

(* The arm for 0: [x] is not null. *)
Definition null_env_value : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6 null_env_relation null_is_null_type
       (this_naked_immediate 0)).

Definition null_arm_value_statement : Prop :=
  meets_answer null_env_relation null_is_null_type (this_naked_immediate 0)
    null_is_null_type null_env_value
  ∧ canonical_of_variable null_env_value null_is_null
    = Some (Some (Const (Naked_immediate 0)))
  ∧ type_of_variable null_env_value null_x Value = Some any_non_null_value.
