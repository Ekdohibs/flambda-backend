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

(* The provers of provers.mli that the model has, and reification, as pairs.

   Each is the function of the compiler, on an environment of the compiler, and
   its transcription in the model, on an environment of the model, with the
   answers of both as values of one type (prover_answers.ml). What a prover is
   asked is a [question]: the type, as a type of the model and as a type of the
   compiler, and the arguments of the provers that take more than a type.

   Two tests use the table: provers_difftest.ml, on environments that it makes
   from generated equations, where the type of the compiler is made from the
   model's; and provers_of_compilations.ml, on the environments that Simplify
   computes, where the type of the compiler is Simplify's own and the model's is
   made from it. The table does not know which. *)

open Flambda2_numbers
open Flambda2_nominal
open Type_comparisons
module TG = Flambda2_types__Type_grammar
module TE = Flambda2_types__Typing_env
module P = Flambda2_types__Provers
module Reify = Flambda2_types__Reify
module M = Flambda2_rocq_semantics.FlambdaModel
module MK = Flambda2_rocq_semantics.FlambdaKindModel
module MT = Flambda2_rocq_semantics.FlambdaTypeGrammar
module MP = Flambda2_rocq_semantics.FlambdaProvers.Provers
module MR = Flambda2_rocq_semantics.FlambdaReify.Reify
module Model_env = Flambda2_rocq_semantics.FlambdaTypingEnv.Typing_env
module Model_name_mode = Flambda2_rocq_semantics.FlambdaBindingTime.Name_mode
module Converter = Flambda2_rocq_types_converter.Type_converter
module Model_names = Flambda2_rocq_types_converter.Model_names
module N = Flambda2_rocq_types_converter.Model_numbers
module A = Flambda2_rocq_types_converter.Prover_answers

(* The two name modes that a prover is given as the least mode of the [Simple]
   that it answers. *)
type mode =
  | Normal
  | In_types

let mode_of_compiler (mode : mode) : Name_mode.t =
  match mode with Normal -> Name_mode.normal | In_types -> Name_mode.in_types

let mode_of_model (mode : mode) : Model_name_mode.t =
  match mode with
  | Normal -> Model_name_mode.normal
  | In_types -> Model_name_mode.in_types

(* The environment that says which variables may be in what reification lifts:
   the environment in which the type is reified, or one that defines fewer
   variables. *)
type allowed_in =
  | The_environment
  | Fewer_variables

(* The arguments of the provers that take more than a type, and of
   reification. *)
type arguments =
  { min_name_mode : mode;
    kind : MK.flambda_kind;
    field_index : int;
    function_slot : M.function_slot;
    value_slot : M.value_slot;
    variable : M.variable;
    (* Of reification: the variables that are defined at toplevel, and those
       that are projections from symbols. *)
    allowed_in : allowed_in;
    at_toplevel : M.variable list;
    symbol_projections : M.variable list
  }

(* What a prover is asked: a type, as a type of the model and as a type of the
   compiler; a second type, of the one prover that takes two; and the
   arguments. *)
type question =
  { asked : MT.flambda_type;
    other : MT.flambda_type;
    arguments : arguments;
    asked_of_compiler : TG.t Lazy.t;
    other_of_compiler : TG.t Lazy.t
  }

(* A prover: its name in provers.mli, and its answer to a question, of the
   compiler in an environment of the compiler and of the model in an environment
   of the model. *)
type prover =
  { prover : string;
    of_compiler : TE.t -> question -> A.answer;
    of_model : Model_env.t -> question -> A.answer
  }

(* The table. The kind of a value slot is an argument of the model's
   [meet_project_value_slot_simple], which the compiler reads off the slot. *)
let provers ~(numbering : Model_names.t)
    ~(kind_of_value_slot : M.value_slot -> MK.flambda_kind) : prover list =
  let asked input = Lazy.force input.asked_of_compiler in
  let module C = A.Of_compiler in
  let module O = A.Of_model in
  (* A prover of an environment and a type. *)
  let proof prover of_compiler value_of_compiler of_model value_of_model =
    { prover;
      of_compiler =
        (fun env input ->
          C.proof value_of_compiler (of_compiler env (asked input)));
      of_model =
        (fun env input -> O.proof value_of_model (of_model env input.asked))
    }
  in
  let shortcut prover of_compiler value_of_compiler of_model value_of_model =
    { prover;
      of_compiler =
        (fun env input ->
          C.shortcut value_of_compiler (of_compiler env (asked input)));
      of_model =
        (fun env input -> O.shortcut value_of_model (of_model env input.asked))
    }
  in
  (* A prover that takes the least name mode of the [Simple] it answers. *)
  let shortcut_with_mode prover of_compiler of_model =
    { prover;
      of_compiler =
        (fun env input ->
          C.shortcut (C.simple numbering)
            (of_compiler env
               ~min_name_mode:(mode_of_compiler input.arguments.min_name_mode)
               (asked input)));
      of_model =
        (fun env input ->
          O.shortcut O.simple
            (of_model env
               (mode_of_model input.arguments.min_name_mode)
               input.asked))
    }
  in
  let simple = C.simple numbering in
  let variable_of_model = Converter.variable_of_model numbering in
  [ { prover = "prove_equals_to_simple_of_kind";
      of_compiler =
        (fun env input ->
          C.proof simple
            (P.prove_equals_to_simple_of_kind env (asked input)
               (Converter.kind_of_model input.arguments.kind)));
      of_model =
        (fun env input ->
          O.proof O.simple
            (MP.prove_equals_to_simple_of_kind env input.asked
               input.arguments.kind))
    };
    proof "prove_equals_tagged_immediates" P.prove_equals_tagged_immediates
      C.target_ints MP.prove_equals_tagged_immediates O.numbers;
    shortcut "meet_equals_tagged_immediates" P.meet_equals_tagged_immediates
      C.target_ints MP.meet_equals_tagged_immediates O.numbers;
    shortcut "meet_naked_immediates" P.meet_naked_immediates C.target_ints
      MP.meet_naked_immediates O.numbers;
    shortcut "meet_equals_single_tagged_immediate"
      P.meet_equals_single_tagged_immediate C.number
      MP.meet_equals_single_tagged_immediate O.number;
    shortcut "meet_naked_float32s" P.meet_naked_float32s C.float32s
      MP.meet_naked_float32s O.numbers;
    shortcut "meet_naked_floats" P.meet_naked_floats C.floats
      MP.meet_naked_floats O.numbers;
    shortcut "meet_naked_int8s" P.meet_naked_int8s C.int8s MP.meet_naked_int8s
      O.numbers;
    shortcut "meet_naked_int16s" P.meet_naked_int16s C.int16s
      MP.meet_naked_int16s O.numbers;
    shortcut "meet_naked_int32s" P.meet_naked_int32s C.int32s
      MP.meet_naked_int32s O.numbers;
    shortcut "meet_naked_int64s" P.meet_naked_int64s C.int64s
      MP.meet_naked_int64s O.numbers;
    shortcut "meet_naked_nativeints" P.meet_naked_nativeints C.nativeints
      MP.meet_naked_nativeints O.numbers;
    shortcut "meet_naked_vec128s" P.meet_naked_vec128s C.vec128s
      MP.meet_naked_vec128s O.numbers;
    shortcut "meet_naked_vec256s" P.meet_naked_vec256s C.vec256s
      MP.meet_naked_vec256s O.numbers;
    shortcut "meet_naked_vec512s" P.meet_naked_vec512s C.vec512s
      MP.meet_naked_vec512s O.numbers;
    shortcut "meet_naked_masks" P.meet_naked_masks C.masks MP.meet_naked_masks
      O.numbers;
    shortcut "meet_variant_like" P.meet_variant_like C.variant_like
      MP.meet_variant_like O.variant_like;
    proof "prove_variant_like" P.prove_variant_like C.variant_like
      MP.prove_variant_like O.variant_like;
    proof "prove_is_a_boxed_or_tagged_number"
      P.prove_is_a_boxed_or_tagged_number
      (C.boxed_or_tagged_number numbering)
      MP.prove_is_a_boxed_or_tagged_number O.boxed_or_tagged_number;
    proof "prove_nothing" P.prove_nothing C.unit MP.prove_nothing O.unit;
    proof "prove_is_a_tagged_immediate" P.prove_is_a_tagged_immediate C.unit
      MP.prove_is_a_tagged_immediate O.unit;
    proof "prove_is_a_boxed_float32" P.prove_is_a_boxed_float32 C.unit
      MP.prove_is_a_boxed_float32 O.unit;
    proof "prove_is_a_boxed_float" P.prove_is_a_boxed_float C.unit
      MP.prove_is_a_boxed_float O.unit;
    proof "prove_is_a_boxed_int32" P.prove_is_a_boxed_int32 C.unit
      MP.prove_is_a_boxed_int32 O.unit;
    proof "prove_is_a_boxed_int64" P.prove_is_a_boxed_int64 C.unit
      MP.prove_is_a_boxed_int64 O.unit;
    proof "prove_is_a_boxed_nativeint" P.prove_is_a_boxed_nativeint C.unit
      MP.prove_is_a_boxed_nativeint O.unit;
    proof "prove_is_a_boxed_vec128" P.prove_is_a_boxed_vec128 C.unit
      MP.prove_is_a_boxed_vec128 O.unit;
    proof "prove_is_a_boxed_vec256" P.prove_is_a_boxed_vec256 C.unit
      MP.prove_is_a_boxed_vec256 O.unit;
    proof "prove_is_a_boxed_vec512" P.prove_is_a_boxed_vec512 C.unit
      MP.prove_is_a_boxed_vec512 O.unit;
    proof "prove_is_a_boxed_mask" P.prove_is_a_boxed_mask C.unit
      MP.prove_is_a_boxed_mask O.unit;
    proof "prove_is_or_is_not_a_boxed_float" P.prove_is_or_is_not_a_boxed_float
      C.bool MP.prove_is_or_is_not_a_boxed_float O.bool;
    proof "prove_unique_tag_and_size" P.prove_unique_tag_and_size C.tag_and_size
      MP.prove_unique_tag_and_size O.tag_and_size;
    proof "prove_is_int" P.prove_is_int C.bool MP.prove_is_int O.bool;
    proof "prove_is_not_a_pointer" P.prove_is_not_a_pointer C.bool
      MP.prove_is_not_a_pointer O.bool;
    shortcut "meet_is_int_variant_only" P.meet_is_int_variant_only C.bool
      MP.meet_is_int_variant_only O.bool;
    proof "prove_get_tag" P.prove_get_tag C.tags MP.prove_get_tag O.numbers;
    shortcut "meet_is_null" P.meet_is_null C.bool MP.meet_is_null O.bool;
    proof "prove_unique_fully_constructed_immutable_heap_block"
      P.prove_unique_fully_constructed_immutable_heap_block (C.block numbering)
      MP.prove_unique_fully_constructed_immutable_heap_block O.block;
    shortcut "meet_single_closures_entry" P.meet_single_closures_entry
      (C.closures_entry numbering)
      MP.meet_single_closures_entry O.closures_entry;
    proof "prove_single_closures_entry" P.prove_single_closures_entry
      (C.closures_entry numbering)
      MP.prove_single_closures_entry O.closures_entry;
    shortcut "meet_code_ids" P.meet_code_ids (C.code_ids numbering)
      MP.meet_code_ids O.code_ids;
    proof "prove_code_ids" P.prove_code_ids (C.code_ids numbering)
      MP.prove_code_ids O.code_ids;
    shortcut "meet_strings" P.meet_strings C.strings MP.meet_strings O.strings;
    proof "prove_strings" P.prove_strings C.strings MP.prove_strings O.strings;
    { prover = "prove_tagging_of_simple";
      of_compiler =
        (fun env input ->
          C.proof simple
            (P.prove_tagging_of_simple env
               ~min_name_mode:(mode_of_compiler input.arguments.min_name_mode)
               (asked input)));
      of_model =
        (fun env input ->
          O.proof O.simple
            (MP.prove_tagging_of_simple env
               (mode_of_model input.arguments.min_name_mode)
               input.asked))
    };
    shortcut_with_mode "meet_tagging_of_simple" P.meet_tagging_of_simple
      MP.meet_tagging_of_simple ]
  @ List.map
      (fun (prover, of_compiler, of_model) ->
        shortcut_with_mode prover of_compiler of_model)
      [ ( "meet_boxed_float32_containing_simple",
          P.meet_boxed_float32_containing_simple,
          MP.meet_boxed_float32_containing_simple );
        ( "meet_boxed_float_containing_simple",
          P.meet_boxed_float_containing_simple,
          MP.meet_boxed_float_containing_simple );
        ( "meet_boxed_int32_containing_simple",
          P.meet_boxed_int32_containing_simple,
          MP.meet_boxed_int32_containing_simple );
        ( "meet_boxed_int64_containing_simple",
          P.meet_boxed_int64_containing_simple,
          MP.meet_boxed_int64_containing_simple );
        ( "meet_boxed_nativeint_containing_simple",
          P.meet_boxed_nativeint_containing_simple,
          MP.meet_boxed_nativeint_containing_simple );
        ( "meet_boxed_vec128_containing_simple",
          P.meet_boxed_vec128_containing_simple,
          MP.meet_boxed_vec128_containing_simple );
        ( "meet_boxed_vec256_containing_simple",
          P.meet_boxed_vec256_containing_simple,
          MP.meet_boxed_vec256_containing_simple );
        ( "meet_boxed_vec512_containing_simple",
          P.meet_boxed_vec512_containing_simple,
          MP.meet_boxed_vec512_containing_simple );
        ( "meet_boxed_mask_containing_simple",
          P.meet_boxed_mask_containing_simple,
          MP.meet_boxed_mask_containing_simple ) ]
  @ [ { prover = "meet_block_field_simple";
        of_compiler =
          (fun env input ->
            C.shortcut simple
              (P.meet_block_field_simple env
                 ~min_name_mode:(mode_of_compiler input.arguments.min_name_mode)
                 ~field_kind:(Converter.kind_of_model input.arguments.kind)
                 (asked input)
                 (Target_ocaml_int.of_int machine_width
                    input.arguments.field_index)));
        of_model =
          (fun env input ->
            O.shortcut O.simple
              (MP.meet_block_field_simple env
                 (mode_of_model input.arguments.min_name_mode)
                 input.arguments.kind input.asked
                 (N.nat_of_int input.arguments.field_index)))
      };
      { prover = "meet_project_value_slot_simple";
        of_compiler =
          (fun env input ->
            C.shortcut simple
              (P.meet_project_value_slot_simple env
                 ~min_name_mode:(mode_of_compiler input.arguments.min_name_mode)
                 (asked input)
                 (Converter.value_slot_of_model numbering
                    input.arguments.value_slot)));
        of_model =
          (fun env input ->
            (* The kind that the compiler reads off the value slot. *)
            O.shortcut O.simple
              (MP.meet_project_value_slot_simple env
                 (mode_of_model input.arguments.min_name_mode)
                 input.asked input.arguments.value_slot
                 (kind_of_value_slot input.arguments.value_slot)))
      };
      { prover = "meet_project_function_slot_simple";
        of_compiler =
          (fun env input ->
            C.shortcut simple
              (P.meet_project_function_slot_simple env
                 ~min_name_mode:(mode_of_compiler input.arguments.min_name_mode)
                 (asked input)
                 (Converter.function_slot_of_model numbering
                    input.arguments.function_slot)));
        of_model =
          (fun env input ->
            O.shortcut O.simple
              (MP.meet_project_function_slot_simple env
                 (mode_of_model input.arguments.min_name_mode)
                 input.asked input.arguments.function_slot))
      };
      proof "prove_alloc_mode_of_boxed_number"
        P.prove_alloc_mode_of_boxed_number C.alloc_mode
        MP.prove_alloc_mode_of_boxed_number O.alloc_mode;
      { prover = "never_holds_locally_allocated_values";
        of_compiler =
          (fun env input ->
            C.proof C.unit
              (P.never_holds_locally_allocated_values env
                 (variable_of_model input.arguments.variable)));
        of_model =
          (fun env input ->
            O.proof O.unit
              (MP.never_holds_locally_allocated_values env
                 input.arguments.variable))
      };
      { prover = "prove_physical_equality";
        of_compiler =
          (fun env input ->
            C.proof C.bool
              (P.prove_physical_equality env (asked input)
                 (Lazy.force input.other_of_compiler)));
        of_model =
          (fun env input ->
            O.proof O.bool
              (MP.prove_physical_equality env input.asked input.other))
      } ]

(* Reification ([Reify.reify]), asked as a prover is. Its first three arguments
   are of the input: the environment that says which variables are allowed (the
   environment itself, or the one given here that defines fewer), and the two
   sets of variables, as functions. *)
let reification ~(numbering : Model_names.t) ~(fewer_in_compiler : TE.t)
    ~(fewer_in_model : Model_env.t) : prover =
  { prover = "reify";
    of_compiler =
      (fun env input ->
        let among variables var =
          List.mem (Converter.variable_to_model numbering var) variables
        in
        A.Of_compiler.reification numbering
          (Reify.reify
             ~allowed_if_free_vars_defined_in:
               (match input.arguments.allowed_in with
               | The_environment -> env
               | Fewer_variables -> fewer_in_compiler)
             ~var_is_defined_at_toplevel:(among input.arguments.at_toplevel)
             ~var_is_symbol_projection:
               (among input.arguments.symbol_projections)
             env
             (Lazy.force input.asked_of_compiler)));
    of_model =
      (fun env input ->
        let among variables x = List.mem x variables in
        A.Of_model.reification
          (MR.reify
             (match input.arguments.allowed_in with
             | The_environment -> env
             | Fewer_variables -> fewer_in_model)
             (among input.arguments.at_toplevel)
             (among input.arguments.symbol_projections)
             env input.asked))
  }

(* The answer of the compiler, where it may stop: with a fatal error, or with an
   exception that a prover does not catch. The conversion of the arguments and
   of the answer runs inside the same guard, so an exception of these that the
   conversion raised would be taken for a stop of the compiler: that a caught
   stop comes from the function of the compiler is trusted. *)
let answer_of_compiler (prover : prover) (env : TE.t) (question : question) :
    A.answer =
  match
    Misc.Colours.without_colours ~f:(fun () ->
        or_fatal_error (fun () -> prover.of_compiler env question))
  with
  | Ok answer -> answer
  | Error line -> Stops ("a fatal error: " ^ general line)
  | exception ((Not_found | Invalid_argument _ | Assert_failure _) as exn) ->
    Stops ("an exception: " ^ Printexc.to_string exn)
