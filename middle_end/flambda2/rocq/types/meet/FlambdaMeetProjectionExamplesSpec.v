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


(* Examples of meets: projections.

   This file is to be reviewed: each statement says what the
   transcribed meet answers on inputs made by hand, after the calls
   of the types library that Simplify makes for a projection. The
   inputs are scenarios that three functions suggest, written from a
   reading of Simplify. They are not the trace of a run of the
   compiler on them. A projection defines its result and meets the
   type of what is projected from with a shape that names the result
   ([simplify_projection], simplify/simplify_common.ml, through
   [Meet.meet_shape], which is [Meet.meet] without the type that it
   answers; the statements are about that meet); the primitive is
   invalid if the answer is bottom. A load of a field and a
   projection of a value slot come to this meet only where a faster
   reading of the type has neither found the field or the slot nor
   shown the primitive invalid ([meet_block_field_simple],
   [meet_project_value_slot_simple]). As in
   FlambdaMeetMatchExamplesSpec.v, whose definitions are used, an
   environment answered is named as what the meet that is to be
   proved answers with [forgetful_join]; a statement says that both
   meets answer it, and what it says of the variables. Where the
   join is not called, they answer it whatever the join. The meet of
   two rows that have one case in common calls the join, with the
   one environment of that case ([meet_row_like], FlambdaMeet.v):
   the second load and the second projection below are such meets,
   and are stated with the join that forgets
   ([meets_answer_forgetting]). The fuel is 6. The answers are those
   of the model.
   The corresponding concrete scenarios have since been run with
   both of the compiler's meets, by the workstream of the differential
   tests of the types library. The type or bottom that the meet
   returns and the selected queries of a type or of a canonical
   element agree with the answers recorded here. That does not compare
   whole resulting environments, nor test the universally quantified
   parts of these statements.

   - The fields of a pair:

       let swap p = let (a, b) = p in (b, a)

     Each field is loaded ([simplify_immutable_block_load0],
     simplify/simplify_unary_primitive.ml; the shape is
     [immutable_block_with_size_at_least]). After the first load [p]
     is a block of tag 0 that has at least one field, which is [a]
     ([pair_first_load_statement]). After the second it has at least
     two, which are [a] and [b] ([pair_second_load_statement]): the
     meet of the two rows keeps the longer one and the field that
     only the first names.
   - A float that is boxed and then unboxed:

       let scale x y = (x +. y) *. 2.

     The sum is a naked float [s], which is boxed for the product,
     and the product unboxes it. The boxing gives the box the type
     of the boxes of [s] ([simplify_box_number], the same file;
     [float_boxed_statement]). The unboxing ([simplify_unbox_number])
     meets that type with the type of the boxes of its result: the
     result becomes an alias of [s] ([float_unboxed_statement]), and
     that is how the unboxing of a box is found to be what was boxed.
   - The value slots of a closure:

       let add3 a b = fun c -> a + b + c

     The body of the inner function projects [a] and [b] from its
     closure ([simplify_project_value_slot], the same file; the shape
     is [closure_with_at_least_this_value_slot]). In the scenario
     nothing is known of the closure at first. That is the case of
     code that is simplified without its set of closures, a stub
     ([dacc_inside_function], simplify/simplify_set_of_closures.ml,
     where it is given no function slot); it is not the case of the
     inner function of [add3] simplified with its set, where the
     closure is an alias of a name of the set. After the first
     projection it is a closure at the function slot whose set has
     at least the first value slot, which is [a]
     ([closure_first_slot_statement]); after the second, at least the
     two, which are [a] and [b] ([closure_second_slot_statement]). *)

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
Local Abbreviation naked_float := (Naked_number Kind_naked_float).

(* Both meets answer a type and an environment, with the join that
   forgets. *)
Definition meets_answer_forgetting (tenv : Typing_env.t)
    (t1 t2 t : flambda_type) (tenv' : Typing_env.t) : Prop :=
  meet_to_prove forgetful_join 6 tenv t1 t2 = Some (Or_bottom.Ok (t, tenv'))
  ∧ compiler_meet forgetful_join 6 tenv t1 t2
    = Some (Or_bottom.Ok (t, tenv')).

(* ======================================================================== *)
(* The fields of a pair                                                     *)
(* ======================================================================== *)

Definition pair_p : variable := 1%positive.
Definition pair_a : variable := 2%positive.
Definition pair_b : variable := 3%positive.

Definition pair_p_type : flambda_type := alias_type_of Value (Var pair_p).

(* The shape of a load of the field at [n - 1] of a block of tag 0
   into a variable. *)
Definition pair_load_shape (n : nat) (result : variable) : flambda_type :=
  match
    immutable_block_with_size_at_least (Or_unknown.Known 0) n
      (Scannable Value_only) result
  with
  | Some shape => shape
  | None => bottom Value
  end.

(* The blocks of tag 0 that have at least the fields given. *)
Definition pair_blocks (fields : list flambda_type) : flambda_type :=
  Type_value
    (Ok
       (No_alias
          (Head_of_kind_value
             (Ok
                (Variant None (Or_unknown.Known (bottom naked_immediate))
                   None
                   (Or_unknown.Known
                      (Row_like_for_blocks
                         [(0,
                           Or_unknown.Known
                             {| case_maps_to := fields;
                                case_index :=
                                  {| index_domain :=
                                       Row_like_index_domain.At_least
                                         (length fields);
                                     index_shape := Scannable Value_only |};
                                case_env_extension := empty_env_extension |})]
                         Or_bottom.Bottom Alloc_mode.Heap_or_local))
                   No_extensions false))
             Not_null))).

(* The parameter is defined, and the result of the first load. *)
Definition pair_env_start : Typing_env.t :=
  let tenv :=
    Typing_env.add_variable_definition Typing_env.create pair_p Value
      Name_mode.normal
  in
  Typing_env.add_variable_definition tenv pair_a Value Name_mode.normal.

Definition pair_env_first : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6 pair_env_start pair_p_type
       (pair_load_shape 1 pair_a)).

Definition pair_first_load_statement : Prop :=
  meets_answer pair_env_start pair_p_type (pair_load_shape 1 pair_a)
    pair_p_type pair_env_first
  ∧ type_of_variable pair_env_first pair_p Value
    = Some (pair_blocks [alias_type_of Value (Var pair_a)]).

(* The result of the second load is defined. *)
Definition pair_env_second_start : Typing_env.t :=
  Typing_env.add_variable_definition pair_env_first pair_b Value
    Name_mode.normal.

Definition pair_env_second : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6 pair_env_second_start pair_p_type
       (pair_load_shape 2 pair_b)).

Definition pair_second_load_statement : Prop :=
  meets_answer_forgetting pair_env_second_start pair_p_type
    (pair_load_shape 2 pair_b) pair_p_type pair_env_second
  ∧ type_of_variable pair_env_second pair_p Value
    = Some
        (pair_blocks
           [alias_type_of Value (Var pair_a);
            alias_type_of Value (Var pair_b)]).

(* ======================================================================== *)
(* A float that is boxed and then unboxed                                   *)
(* ======================================================================== *)

(* The sum, its box and the result of the unboxing. *)
Definition float_s : variable := 1%positive.
Definition float_box : variable := 2%positive.
Definition float_u : variable := 3%positive.

(* The sum is defined, and then the box. *)
Definition float_env_start : Typing_env.t :=
  let tenv :=
    Typing_env.add_variable_definition Typing_env.create float_s naked_float
      Name_mode.normal
  in
  Typing_env.add_variable_definition tenv float_box Value Name_mode.normal.

(* The type that the boxing gives the box: the boxes of [s], on the
   heap. *)
Definition float_box_type : flambda_type :=
  box_number Boxable_naked_float (alias_type_of naked_float (Var float_s))
    Alloc_mode.Heap.

Definition float_env_boxed : Typing_env.t :=
  env_answered
    (Flambda2_types.add_equation forgetful_join repaired_choices 6
       float_env_start (Name_var float_box) float_box_type).

(* The box gets that type, whatever the join and the choices. *)
Definition float_boxed_statement : Prop :=
  (∀ (join : join_scoped_envs) (choices : meet_choices),
    Flambda2_types.add_equation join choices 6 float_env_start
      (Name_var float_box) float_box_type
    = Some float_env_boxed)
  ∧ type_of_variable float_env_boxed float_box Value = Some float_box_type.

(* The result of the unboxing is defined, and the type of the box is
   met with the boxes of the result, of which the mode is not
   known. *)
Definition float_env_unbox_start : Typing_env.t :=
  Typing_env.add_variable_definition float_env_boxed float_u naked_float
    Name_mode.normal.

Definition float_box_alias : flambda_type :=
  alias_type_of Value (Var float_box).

Definition float_unbox_shape : flambda_type :=
  boxed_number_alias_to Boxable_naked_float float_u Alloc_mode.Heap_or_local.

Definition float_env_unboxed : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6 float_env_unbox_start float_box_alias
       float_unbox_shape).

(* The result is an alias of [s], and the type of the box is not
   changed. *)
Definition float_unboxed_statement : Prop :=
  meets_answer float_env_unbox_start float_box_alias float_unbox_shape
    float_box_alias float_env_unboxed
  ∧ canonical_of_variable float_env_unboxed float_u = Some (Some (Var float_s))
  ∧ type_of_variable float_env_unboxed float_box Value = Some float_box_type.

(* ======================================================================== *)
(* The value slots of a closure                                             *)
(* ======================================================================== *)

(* The closure of the inner function and the results of the two
   projections. *)
Definition closure_my_closure : variable := 1%positive.
Definition closure_a : variable := 2%positive.
Definition closure_b : variable := 3%positive.

(* The function slot of the inner function and the value slots of [a]
   and of [b]. *)
Definition closure_function_slot : function_slot := 1%positive.
Definition closure_slot_a : value_slot := 1%positive.
Definition closure_slot_b : value_slot := 2%positive.

Definition closure_type : flambda_type :=
  alias_type_of Value (Var closure_my_closure).

(* The closure is defined, and the result of the first projection. *)
Definition closure_env_start : Typing_env.t :=
  let tenv :=
    Typing_env.add_variable_definition Typing_env.create closure_my_closure
      Value Name_mode.normal
  in
  Typing_env.add_variable_definition tenv closure_a Value Name_mode.normal.

Definition closure_first_shape : flambda_type :=
  closure_with_at_least_this_value_slot closure_function_slot closure_slot_a
    closure_a Value.

Definition closure_env_first : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6 closure_env_start closure_type
       closure_first_shape).

(* The closure has the type of the shape. *)
Definition closure_first_slot_statement : Prop :=
  meets_answer closure_env_start closure_type closure_first_shape
    closure_type closure_env_first
  ∧ type_of_variable closure_env_first closure_my_closure Value
    = Some closure_first_shape.

(* The result of the second projection is defined. *)
Definition closure_env_second_start : Typing_env.t :=
  Typing_env.add_variable_definition closure_env_first closure_b Value
    Name_mode.normal.

Definition closure_second_shape : flambda_type :=
  closure_with_at_least_this_value_slot closure_function_slot closure_slot_b
    closure_b Value.

Definition closure_env_second : Typing_env.t :=
  env_of_meet
    (meet_to_prove forgetful_join 6 closure_env_second_start closure_type
       closure_second_shape).

(* The closure has the two value slots, each held by its variable. *)
Definition closure_second_slot_statement : Prop :=
  meets_answer_forgetting closure_env_second_start closure_type
    closure_second_shape closure_type closure_env_second
  ∧ type_of_variable closure_env_second closure_my_closure Value
    = Some
        (closure_with_at_least_these_value_slots closure_function_slot
           [(closure_slot_b, (closure_b, Value));
            (closure_slot_a, (closure_a, Value))]).
