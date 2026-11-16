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



(* The meet of two Flambda types, as functions.

   This file is to be reviewed, function by function against
   middle_end/flambda2/types/meet_and_n_way_join.ml (from
   [map_return_value] to [meet_function_type], and [meet] with
   [meet_expanded_head] as it exposes them), types/meet.ml, and the
   functions that types/flambda2_types.ml adds to [Typing_env]. It
   holds definitions only and it extracts. What the functions
   guarantee is FlambdaMeetSpec.v; the environment of a meet is
   FlambdaMeetEnv.v; expanded types are FlambdaExpandHead.v.

   The compiler has a second meet, types/meet_and_join.ml, which it
   runs by default. The two files have the same functions but for
   [meet_disjunction] and [meet_row_like], which differ in the join
   that they call. That join is a parameter here ([join], below), and
   the functions are those of meet_and_n_way_join.ml. The other file
   is not transcribed.

   The shape of the file. In the compiler the functions from [meet] to
   [meet_function_type] are one recursive definition. They do not
   recur on the types: a meet with an alias reads the type that the
   environment has for the name, and meets that. Here every cycle
   among them goes through [meet], and [meet] is given the meet of
   expanded types as a parameter, [meet_expanded_head_rec]. So each
   function is a plain definition, in a section that has that
   parameter, and one fixpoint, on a number, ties them:
   [Meet.meet_expanded_head], which gives the functions of the section
   the meet of one less. The number is the fuel; with none the answer
   is [None]. The functions that the compiler has first
   ([meet_disjunction], [meet_row_like] and the helpers), which take
   [~meet_expanded_head] as an argument there, are in the same
   section.

   The parameters, which are those of every function that is exported:
   - [join]: [Join_env.cut_and_n_way_join ~n_way_join_type
     ~meet_expanded_head ~cut_after initial_env initial_tenv
     scoped_envs], without its argument [~n_way_join_type], which is
     the join of types and is the join's own business. It is given the
     meet of less fuel. It is another workstream's (T5), and its
     contract is in FlambdaMeetSpec.v. It is to be read as the compiler
     run with [-flambda2-join-sharing structural] (decision D30).
   - [choices], a record of the three places where the meet can do
     other than the compiler does ([meet_choices], which the compiler
     has not):
     - [type_of_get_tag_scrutinee]: the type that
       [reduce_inverse_relations] gives a name of which a number among
       some tags is [Get_tag]. The compiler's is
       [blocks_with_these_tags] (FlambdaTypeGrammar.v), with which the
       step is not sound; [immediate_or_blocks_with_these_tags],
       below, is the type with which it is (FlambdaMeetSpec.v, "The
       step that is not sound").
     - [define_again]: whether a variable that the environment already
       knows is defined again, where the compiler defines the
       variables of a level in the environment that a join has
       answered ([add_defined_vars], and the extra variables of
       [meet_row_like]). The compiler does ([true]), and then the
       environment may know one variable at two binding times; with
       [false] only the variables that the environment does not know
       are defined (FlambdaMeetSpec.v, discrepancy T4-1).
     - [lift_variables_of_single_case]: [meet_row_like], where it
       keeps one case only ([need_join] is false), makes no extension
       of that case and does not record the variables that the
       environment of the case defines; the join is then given that
       one environment, and keeps of its variables those that the
       equations it joins mention. A variable that a meet inside the
       case defined, and that only the type answered mentions, is
       then not known to the environment answered. The compiler does
       not lift ([false]); with [true] those variables are defined in
       the result, as they are where there is more than one case
       (FlambdaMeetSpec.v, discrepancy T4-9).
   - the fuel.
   The compiler is [Meet.meet join compiler_choices fuel], but for one
   test of FlambdaMeetEnv.v that no choice governs
   ([Meet_env.replace_concrete_equation], on a name that is no longer
   canonical: the model answers [None] where the compiler goes on;
   discrepancy T4-10). What is to be proved sound is
   [Meet.meet join repaired_choices fuel].

   What the model has, in the order of the file, and how each differs
   from the compiler's:
   - [map_return_value], [combine_meet_return_values], [set_meet],
     [meet_mapping], [Map_meet.meet] ([map_meet]), [meet_alloc_mode],
     [meet_unknown], [meet_or_unknown_or_bottom]: in full.
     [extract_value] is in FlambdaMeetEnv.v.
   - [combine_results]: the compiler's takes a list of meets, each of
     a type of its own, and is used with two and with three. Here
     there is a function for each, [combine_results2] and
     [combine_results3].
   - [meet_code_id]: without the code age relation, which the model's
     environment has not. Two different code ids are bottom. The
     compiler answers the newer of two that the relation relates, and
     both inputs if it cannot tell (FlambdaMeetSpec.v, discrepancy
     T4-4).
   - [reduce_head_of_kind_naked_immediate], [meet_inverse_relations],
     [reduce_inverse_relations], [meet_head_of_kind_naked_immediate]:
     in full. A set of names per relation is a list of pairs of a
     relation and a name (FlambdaTypeGrammar.v), compared as a set.
     [meet_inverse_relations] follows the path of the compiler that
     does not test physical equality.
   - [meet_head_of_kind_naked_number]: the compiler has eleven
     functions, one for each kind of naked number other than the
     immediates, each [set_meet] on the set of numbers of that kind.
     The model has one set of numbers for the eleven
     (FlambdaTypeGrammar.v) and one function.
   - [meet_head_of_kind_rec_info], [meet_head_of_kind_region]: in
     full; both answer the inputs.
   - [add_defined_vars], [meet_disjunction], [meet_row_like]: in
     full, but for the choices above and the names to avoid (below).
   - [meet_relation], [meet_variant], [meet_head_of_kind_value],
     [meet_int_indexed_product], [meet_array_of_types],
     [meet_product_function_slot_indexed],
     [meet_product_value_slot_indexed], [meet_closures_entry],
     [meet_row_like_for_blocks], [meet_row_like_for_closures]: in full.
   - [meet_head_of_kind_value_non_null]: without the case of arrays,
     which the grammar of the model has not; [meet_array_type] and
     [meet_array_contents] are not here. The cases of the boxed
     numbers are one, as the grammar has one constructor for them.
   - [meet_function_type]: without [rec_info], which the function
     types of the model have not. The compiler meets the two as types
     of kind [Rec_info], which answers the inputs and adds nothing
     unless one of them is an alias.
   - [meet_expanded_head0], [meet_expanded_head], [meet]: in full.
     [meet] is [Meet_env.meet_type] (FlambdaMeetEnv.v) with the meet
     of expanded types.
   - [Meet.meet], [Meet.meet_expanded_head], [Meet.meet_shape]
     (types/meet.ml). The compiler's [Meet.meet] catches a fatal error
     to print the two types and raises it again; that is [None] here
     as elsewhere.
   - [Flambda2_types.add_equation], [add_equations_on_params],
     [add_env_extension], [add_is_null_relation],
     [add_is_int_relation], [add_get_tag_relation]: the functions of
     the module [Typing_env] of flambda2_types.ml. The three for
     relations are one function here, [add_relation], given the two
     types. The types of the names ([Type_grammar.is_null],
     [is_int_for_scrutinee], [get_tag_for_block]) are those of
     FlambdaTypeGrammar.v where the scrutinee is a name;
     [is_null_of_simple] and the two beside it add the case of a
     constant.

   The differences that are not of one function:
   - Exceptions and references. The compiler raises [Bottom_result]
     out of a fold and catches it outside; here the state of the fold
     says so, and the rest of the fold changes nothing. [meet_row_like]
     keeps four references ([extra_variables], [result_env],
     [result_is_t1], [result_is_t2]); here they are a record that the
     functions hand on ([row_like_state]). A fatal error or an
     assertion that fails is [None].
   - The order of what is gone through. The model may not look at the
     order of names (decision D29). The cases of a row-like structure
     of blocks are gone through in the order of their tags, which are
     numbers, as the compiler does. The compiler goes through function
     slots, value slots and the names of an extension in the order of
     its maps; here they are gone through in the order in which the
     first map lists them, then those that only the second has
     ([Keyed.keys_of_both], FlambdaMapSpec.v). The extra variables of
     [meet_row_like] are defined in the order in which the cases were
     met, where the compiler defines them in the order of the
     variables. So this file is the compiler's algorithm on maps whose
     order is not the compiler's, and it is not claimed that the two
     answer the same thing. Where the order can show: which of two
     fatal errors or of two bottoms comes first; the environment that
     one meet hands the next, and so what the next one reads; the
     binding times of variables defined after a join; the order of the
     bindings of the maps and of the equations of the levels
     answered. Giving the model its inputs with the bindings in the
     compiler's order does not remove these: the maps and the levels
     that the model builds on the way (an entry replaced, a level cut,
     an extension made of a level) are lists in the order in which
     they were built, and are gone through in that order later. No
     statement here says that the answer does not depend on the order,
     up to the order of bindings and to binding times; until one does,
     a comparison with the compiler is to be made up to those, and a
     difference beyond them is to be looked at, not assumed to be the
     order. One such difference is known (T4-21). What a meet adds in
     a scope is added again when the scope is left, and so more than
     once where scopes are nested: by the join that [meet_row_like]
     calls ([Join_env.define_or_eliminate_variables_and_add_equations])
     and then by [direct_return] of each [meet_disjunction] around it.
     The compiler's n-way meet goes through a map of names each time
     (as its source reads: [Typing_env_extension.fold], a
     [Name.Map.fold]), so each time in the order of the names. In the
     model a level is a list with the equation added last in front.
     The join of one environment adds the equations that the level has
     for variables of the target from the back, in the order in which
     they were added (FlambdaJoinEnv.v, "Order"). [direct_return] adds
     the level that it cuts from the front ([extension_of_level]), so
     in the reverse of the order in which its equations were added;
     and that level holds what the join added. So, where at least two
     distinct names are involved, two addings that follow each other
     can go through the same names in opposite orders in the model,
     and no single order of names gives both; and the order can decide
     whether the answer is bottom: on the line of T7-14 in the tests
     of the types ([the_lines_of_t7_14_and_t7_15],
     tests/semantics/types/types_difftest.ml) the model answers bottom
     and the compiler's n-way meet, in each of the twenty-four orders
     of the identifiers of the four variables of that line, does not.
   - Names to avoid. The compiler's join makes variables with
     [Variable.create], which are new by construction. The model's
     join is to avoid the names of [Meet_env.names_to_avoid]
     (FlambdaMeetEnv.v) and to add those it makes. So [meet_disjunction]
     gives the second alternative the names to avoid of the first, and
     [meet_row_like] hands them from one case to the next ([avoid]);
     and the join is given those of the last. Nothing else reads them.
   - The two alternatives of [meet_disjunction]. The compiler adds the
     extensions to the environments of the two alternatives in turn
     and then meets each; here the first alternative is done, then the
     second. The two environments do not depend on each other, so the
     answers are the same.
   - Physical equality. No function here tests it (decision D30). The
     answers [Left_input], [Right_input] and [Both_inputs] are kept:
     they decide which reductions [meet_head_of_kind_naked_immediate]
     makes and whether an equation is written
     ([Meet_env.add_concrete_equation_on_canonical_name]).
   - Machine width. Numbers are integers; [Target_ocaml_int.bool] is 0
     or 1. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningBoolSpec FlambdaTypeEqualSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaTypingEnv FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv.
Import ListNotations.

(* ======================================================================== *)
(* What a meet answers with                                                 *)
(* ======================================================================== *)

(* [map_return_value]. *)
Definition map_return_value {A B : Type} (f : A -> B)
    (x : meet_return_value A) : meet_return_value B :=
  match x with
  | Left_input => Left_input
  | Right_input => Right_input
  | Both_inputs => Both_inputs
  | New_result x => New_result (f x)
  end.

(* [combine_meet_return_values]: the answer for a pair from the answers
   for its two components. *)
Definition combine_meet_return_values {A B C : Type}
    (a : meet_return_value A) (b : meet_return_value B)
    (compute_value : unit -> C) : meet_return_value C :=
  match a, b with
  | Both_inputs, Both_inputs => Both_inputs
  | Left_input, (Left_input | Both_inputs) | Both_inputs, Left_input =>
    Left_input
  | Right_input, (Right_input | Both_inputs) | Both_inputs, Right_input =>
    Right_input
  | New_result _, _
  | _, New_result _
  | Left_input, Right_input
  | Right_input, Left_input => New_result (compute_value tt)
  end.

(* An answer whose value may be a fatal error is one, or an answer. *)
Definition sequence_return_value {A : Type} (x : meet_return_value (option A))
    : option (meet_return_value A) :=
  match x with
  | Left_input => Some Left_input
  | Right_input => Some Right_input
  | Both_inputs => Some Both_inputs
  | New_result None => None
  | New_result (Some x) => Some (New_result x)
  end.

(* [set_meet]: the meet of two sets is their intersection. *)
Definition set_meet {K B : Type} (eqb : K -> K -> bool)
    (of_set : list K -> B) (env : Meet_env.t) (s1 s2 : list K)
    : Meet_result.t B :=
  match Keyed.all_among eqb s1 s2, Keyed.all_among eqb s2 s1 with
  | true, true => Meet_result.Ok Both_inputs env
  | true, false => Meet_result.Ok Left_input env
  | false, true => Meet_result.Ok Right_input env
  | false, false =>
    let s := Keyed.inter eqb s1 s2 in
    if Keyed.is_empty s
    then Meet_result.Bottom (New_result tt)
    else Meet_result.Ok (New_result (of_set s)) env
  end.

(* The state of [meet_mapping] as it goes through the keys: whether
   every component so far is that of the left input, and of the right
   one; the components so far, the last first; the environment. *)
Record mapping_acc (K A : Type) : Type := {
  all_left : bool;
  all_right : bool;
  mapping : list (K * A);
  mapping_env : Meet_env.t
}.

Arguments all_left {K A} _.
Arguments all_right {K A} _.
Arguments mapping {K A} _.
Arguments mapping_env {K A} _.

(* [None] is a fatal error and [Bottom] the exception [Bottom_result]:
   after either, the rest of the keys are passed over. *)
Definition mapping_state (K A : Type) : Type :=
  option (Or_bottom.t (mapping_acc K A)).

(* [meet_mapping]: the meet of two products, component by component. A
   component that only one has is kept; the meet is bottom as soon as
   that of one component is. [fold2] goes through the keys of both. *)
Definition meet_mapping {K A M : Type}
    (meet_data : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (fold2 :
      (K -> option A -> option A -> mapping_state K A -> mapping_state K A) ->
      M -> M -> mapping_state K A -> mapping_state K A)
    (env : Meet_env.t) (left right : M) (rebuild : list (K * A) -> M)
    : option (Meet_result.t M) :=
  let step (key : K) (data_left_opt data_right_opt : option A)
      (state : mapping_state K A) : mapping_state K A :=
    match state with
    | None => None
    | Some Or_bottom.Bottom => Some Or_bottom.Bottom
    | Some (Or_bottom.Ok acc) =>
      match data_left_opt, data_right_opt with
      | None, None => None
      | Some data_left, None =>
        Some
          (Or_bottom.Ok
             {| all_left := all_left acc;
                all_right := false;
                mapping := (key, data_left) :: mapping acc;
                mapping_env := mapping_env acc |})
      | None, Some data_right =>
        Some
          (Or_bottom.Ok
             {| all_left := false;
                all_right := all_right acc;
                mapping := (key, data_right) :: mapping acc;
                mapping_env := mapping_env acc |})
      | Some data_left, Some data_right =>
        match meet_data (mapping_env acc) data_left data_right with
        | None => None
        | Some (Meet_result.Bottom _) => Some Or_bottom.Bottom
        | Some (Meet_result.Ok res env) =>
          let result (all_left all_right : bool) (data : A)
              : mapping_state K A :=
            Some
              (Or_bottom.Ok
                 {| all_left := all_left;
                    all_right := all_right;
                    mapping := (key, data) :: mapping acc;
                    mapping_env := env |})
          in
          match res with
          | Both_inputs => result (all_left acc) (all_right acc) data_left
          | Left_input => result (all_left acc) false data_left
          | Right_input => result false (all_right acc) data_right
          | New_result data => result false false data
          end
        end
      end
    end
  in
  match
    fold2 step left right
      (Some
         (Or_bottom.Ok
            {| all_left := true;
               all_right := true;
               mapping := [];
               mapping_env := env |}))
  with
  | None => None
  | Some Or_bottom.Bottom => Some (Meet_result.Bottom (New_result tt))
  | Some (Or_bottom.Ok res) =>
    let result :=
      match all_left res, all_right res with
      | true, true => Both_inputs
      | true, false => Left_input
      | false, true => Right_input
      | false, false => New_result (rebuild (mapping res))
      end
    in
    Some (Meet_result.Ok result (mapping_env res))
  end.

(* [Map_meet.meet]: the meet of two maps of one kind of key, by
   [meet_mapping]. The keys are gone through in the order of
   [keys_of_both] (see the header). *)
Definition map_meet {K A : Type} (eqb : K -> K -> bool)
    (meet_data : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (env : Meet_env.t) (left right : Keyed.finite_map K A)
    : option (Meet_result.t (Keyed.finite_map K A)) :=
  let fold2
      (f :
        K -> option A -> option A -> mapping_state K A -> mapping_state K A)
      (m1 m2 : Keyed.finite_map K A) (init : mapping_state K A)
      : mapping_state K A :=
    Keyed.fold2_on eqb (Keyed.keys_of_both eqb m1 m2) f m1 m2 init
  in
  let rebuild (l : list (K * A)) : Keyed.finite_map K A :=
    fold_left (fun m binding => binding :: m) l []
  in
  meet_mapping meet_data fold2 env left right rebuild.

(* [combine_results], for two meets and for three: the meets are made
   one after the other, each in the environment that the one before
   answered, and the answers combined. *)
Definition combine_results2 {A B C : Type} (env : Meet_env.t)
    (meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (left_a right_a : A) (left_b right_b : B) (rebuild : A -> B -> C)
    : option (Meet_result.t C) :=
  let* result_a := meet_a env left_a right_a in
  match result_a with
  | Meet_result.Bottom r => Some (Meet_result.Bottom r)
  | Meet_result.Ok result_a env =>
    let* result_b := meet_b env left_b right_b in
    match result_b with
    | Meet_result.Bottom r => Some (Meet_result.Bottom r)
    | Meet_result.Ok result_b env =>
      let return_value :=
        combine_meet_return_values result_a result_b
          (fun _ =>
             rebuild
               (extract_value result_a left_a right_a)
               (extract_value result_b left_b right_b))
      in
      Some (Meet_result.Ok return_value env)
    end
  end.

Definition combine_results3 {A B C D : Type} (env : Meet_env.t)
    (meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (meet_c : Meet_env.t -> C -> C -> option (Meet_result.t C))
    (left_a right_a : A) (left_b right_b : B) (left_c right_c : C)
    (rebuild : A -> B -> C -> D) : option (Meet_result.t D) :=
  let* result_a := meet_a env left_a right_a in
  match result_a with
  | Meet_result.Bottom r => Some (Meet_result.Bottom r)
  | Meet_result.Ok result_a env =>
    let* result_b := meet_b env left_b right_b in
    match result_b with
    | Meet_result.Bottom r => Some (Meet_result.Bottom r)
    | Meet_result.Ok result_b env =>
      let* result_c := meet_c env left_c right_c in
      match result_c with
      | Meet_result.Bottom r => Some (Meet_result.Bottom r)
      | Meet_result.Ok result_c env =>
        let result_bc :=
          combine_meet_return_values result_b result_c
            (fun _ =>
               (extract_value result_b left_b right_b,
                extract_value result_c left_c right_c))
        in
        let return_value :=
          combine_meet_return_values result_a result_bc
            (fun _ =>
               let bc :=
                 extract_value result_bc (left_b, left_c) (right_b, right_c)
               in
               rebuild
                 (extract_value result_a left_a right_a)
                 (fst bc) (snd bc))
        in
        Some (Meet_result.Ok return_value env)
      end
    end
  end.

(* ======================================================================== *)
(* The meets that need no other                                             *)
(* ======================================================================== *)

(* [meet_code_id], without the code age relation: two different code
   ids have no code in common. *)
Definition meet_code_id (env : Meet_env.t) (code_id1 code_id2 : code_id)
    : Meet_result.t code_id :=
  if Pos.eqb code_id1 code_id2
  then Meet_result.Ok Both_inputs env
  else Meet_result.Bottom (New_result tt).

(* [meet_alloc_mode]. *)
Definition meet_alloc_mode (env : Meet_env.t)
    (alloc_mode1 alloc_mode2 : Alloc_mode.t) : Meet_result.t Alloc_mode.t :=
  match alloc_mode1, alloc_mode2 with
  | (Alloc_mode.Heap_or_local | Alloc_mode.Local),
    (Alloc_mode.Heap_or_local | Alloc_mode.Local)
  | Alloc_mode.Heap, Alloc_mode.Heap => Meet_result.Ok Both_inputs env
  | (Alloc_mode.Heap_or_local | Alloc_mode.Local), _ =>
    Meet_result.Ok Right_input env
  | _, (Alloc_mode.Heap_or_local | Alloc_mode.Local) =>
    Meet_result.Ok Left_input env
  end.

(* [meet_unknown]: the meet of two things of which either may be
   unknown; it is bottom if one is known to be. *)
Definition meet_unknown {A : Type}
    (meet_contents : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (contents_is_bottom : A -> bool) (env : Meet_env.t)
    (or_unknown1 or_unknown2 : Or_unknown.t A)
    : option (Meet_result.t (Or_unknown.t A)) :=
  match or_unknown1, or_unknown2 with
  | Or_unknown.Unknown, Or_unknown.Unknown =>
    Some (Meet_result.Ok Both_inputs env)
  | Or_unknown.Known contents1, Or_unknown.Known contents2 =>
    if contents_is_bottom contents1 && contents_is_bottom contents2
    then Some (Meet_result.Bottom Both_inputs)
    else if contents_is_bottom contents1
    then Some (Meet_result.Bottom Left_input)
    else if contents_is_bottom contents2
    then Some (Meet_result.Bottom Right_input)
    else
      option_map (map_result Or_unknown.Known)
        (meet_contents env contents1 contents2)
  | Or_unknown.Known contents, Or_unknown.Unknown =>
    if contents_is_bottom contents
    then Some (Meet_result.Bottom Left_input)
    else Some (Meet_result.Ok Left_input env)
  | Or_unknown.Unknown, Or_unknown.Known contents =>
    if contents_is_bottom contents
    then Some (Meet_result.Bottom Right_input)
    else Some (Meet_result.Ok Right_input env)
  end.

(* [meet_or_unknown_or_bottom]. *)
Definition meet_or_unknown_or_bottom {A B : Type}
    (meet_elt : Meet_env.t -> A -> A -> option (Meet_result.t B))
    (env : Meet_env.t) (input1 input2 : or_unknown_or_bottom A)
    : option (Meet_result.t B) :=
  match input1, input2 with
  | Bottom, Bottom => Some (Meet_result.Bottom Both_inputs)
  | Bottom, _ => Some (Meet_result.Bottom Left_input)
  | _, Bottom => Some (Meet_result.Bottom Right_input)
  | Unknown, Unknown => Some (Meet_result.Ok Both_inputs env)
  | _, Unknown => Some (Meet_result.Ok Left_input env)
  | Unknown, _ => Some (Meet_result.Ok Right_input env)
  | Ok elt1, Ok elt2 => meet_elt env elt1 elt2
  end.

(* ======================================================================== *)
(* Naked immediates: the relations                                          *)
(* ======================================================================== *)

Definition relation_eqb (relation relation' : Relation.t) : bool :=
  match relation, relation' with
  | Relation.Is_null, Relation.Is_null
  | Relation.Is_int, Relation.Is_int
  | Relation.Get_tag, Relation.Get_tag => true
  | _, _ => false
  end.

Definition inverse_relation_eqb (r r' : Relation.t * name) : bool :=
  relation_eqb (fst r) (fst r') && name_eqb (snd r) (snd r').

(* [Type_grammar.Head_of_kind_naked_immediate.from_descr]. *)
Definition head_of_kind_naked_immediate_from_descr
    (naked_immediates : Or_unknown.t (list Z))
    (inverse_relations : list (Relation.t * name))
    : Or_bottom.t head_of_kind_naked_immediate :=
  match naked_immediates with
  | Or_unknown.Known [] => Or_bottom.Bottom
  | Or_unknown.Known (_ :: _) | Or_unknown.Unknown =>
    Or_bottom.Ok
      (Head_of_kind_naked_immediate naked_immediates inverse_relations)
  end.

(* [Type_grammar.Head_of_kind_naked_immediate.from_descr_non_empty].
   [None] is its fatal error. *)
Definition head_of_kind_naked_immediate_from_descr_non_empty
    (naked_immediates : Or_unknown.t (list Z))
    (inverse_relations : list (Relation.t * name))
    : option head_of_kind_naked_immediate :=
  match
    head_of_kind_naked_immediate_from_descr naked_immediates
      inverse_relations
  with
  | Or_bottom.Bottom => None
  | Or_bottom.Ok head => Some head
  end.

(* The state of [reduce_head_of_kind_naked_immediate] as it goes
   through the relations: the numbers, the relations kept, the last
   first, and whether anything changed. *)
Record reduce_acc : Type := {
  reduce_imms : Or_unknown.t (list Z);
  reduce_relations : list (Relation.t * name);
  reduce_changed : bool
}.

(* [reduce_head_of_kind_naked_immediate]: a relation to a name whose
   canonical element is a constant gives the number, where the compiler
   reads one ([Relation.of_const]), and goes; a relation to a name that
   is not canonical becomes one to its canonical element. *)
Definition reduce_head_of_kind_naked_immediate (tenv : Typing_env.t)
    (head : head_of_kind_naked_immediate)
    : Or_bottom.t head_of_kind_naked_immediate :=
  let step (state : Or_bottom.t reduce_acc) (inverse : Relation.t * name)
      : Or_bottom.t reduce_acc :=
    match state with
    | Or_bottom.Bottom => Or_bottom.Bottom
    | Or_bottom.Ok acc =>
      let relation := fst inverse in
      let name := snd inverse in
      let changed (imms : Or_unknown.t (list Z)) : Or_bottom.t reduce_acc :=
        Or_bottom.Ok
          {| reduce_imms := imms;
             reduce_relations := reduce_relations acc;
             reduce_changed := true |}
      in
      let keep (name' : FlambdaTypeGrammar.name) : Or_bottom.t reduce_acc :=
        Or_bottom.Ok
          {| reduce_imms := reduce_imms acc;
             reduce_relations := (relation, name') :: reduce_relations acc;
             reduce_changed :=
               reduce_changed acc || negb (name_eqb name' name) |}
      in
      match
        Typing_env.get_canonical_simple_ignoring_name_mode tenv
          (simple_of_name name)
      with
      | Const const =>
        match Relation.of_const relation const with
        | Or_bottom.Bottom => changed (reduce_imms acc)
        | Or_bottom.Ok imm =>
          let imms' := [imm] in
          match reduce_imms acc with
          | Or_unknown.Known imms =>
            if Keyed.same_keys_b Z.eqb imms' imms
            then Or_bottom.Ok acc
            else if negb (Keyed.among Z.eqb imm imms)
            then Or_bottom.Bottom
            else changed (Or_unknown.Known imms')
          | Or_unknown.Unknown => changed (Or_unknown.Known imms')
          end
        end
      | Var x => keep (Name_var x)
      | Symbol s => keep (Name_symbol s)
      end
    end
  in
  match
    fold_left step (inverse_relations head)
      (Or_bottom.Ok
         {| reduce_imms := naked_immediates head;
            reduce_relations := [];
            reduce_changed := false |})
  with
  | Or_bottom.Bottom => Or_bottom.Bottom
  | Or_bottom.Ok acc =>
    if negb (reduce_changed acc)
    then Or_bottom.Ok head
    else
      head_of_kind_naked_immediate_from_descr (reduce_imms acc)
        (Keyed.dedup inverse_relation_eqb (rev (reduce_relations acc)))
  end.

(* [reverse_mapping_meet_return_value]. *)
Record reverse_mapping_meet_return_value (A : Type) : Type := {
  left_inputs_only : A;
  meet_return_value_of : meet_return_value A;
  right_inputs_only : A
}.

Arguments left_inputs_only {A} _.
Arguments meet_return_value_of {A} _.
Arguments right_inputs_only {A} _.

(* [meet_inverse_relations]: the relations of both, with those that
   only the left one has and those that only the right one has. *)
Definition meet_inverse_relations
    (inverse_relations1 inverse_relations2 : list (Relation.t * name))
    : reverse_mapping_meet_return_value (list (Relation.t * name)) :=
  let left_inputs_only :=
    Keyed.without inverse_relation_eqb inverse_relations2 inverse_relations1
  in
  let right_inputs_only :=
    Keyed.without inverse_relation_eqb inverse_relations1 inverse_relations2
  in
  let inverse_relations :=
    Keyed.dedup inverse_relation_eqb (inverse_relations1 ++ inverse_relations2)
  in
  let meet_return_value :=
    match Keyed.is_empty left_inputs_only, Keyed.is_empty right_inputs_only with
    | true, true => Both_inputs
    | false, true => Left_input
    | true, false => Right_input
    | false, false => New_result inverse_relations
    end
  in
  {| left_inputs_only := left_inputs_only;
     meet_return_value_of := meet_return_value;
     right_inputs_only := right_inputs_only |}.

(* [Tag.create_from_targetint]: a number is the tag of a block. *)
Definition is_tag (i : Z) : bool := (0 <=? i)%Z && (i <=? 255)%Z.

(* [Tag.is_structured_block]. *)
Definition is_structured_block (tag : Z) : bool :=
  (0 <=? tag)%Z && (tag <? no_scan_tag)%Z.

(* The type that is true of a name of which a number among [tags] is
   [Get_tag]: an immediate, or a block with one of the tags. It is
   [blocks_with_these_tags] (FlambdaTypeGrammar.v) with
   [~immediates:Unknown]. The compiler has no such function; see
   [type_of_get_tag_scrutinee] below. *)
Definition immediate_or_blocks_with_these_tags (tags : list Z)
    (alloc_mode : Alloc_mode.t) : flambda_type :=
  create_variant false Or_unknown.Unknown
    (Or_unknown.Known
       (Row_like_for_blocks
          (map (fun tag => (tag, Or_unknown.Unknown)) tags)
          Or_bottom.Bottom alloc_mode))
    No_extensions.

(* The places where the transcribed meet can do other than the
   compiler does. The compiler has no such record. *)
Record meet_choices : Type := {
  (* The type that [reduce_inverse_relations] gives a name of which a
     number among some tags is [Get_tag]. *)
  type_of_get_tag_scrutinee : list Z -> Alloc_mode.t -> flambda_type;
  (* Whether a variable that the environment already knows is defined
     again by [add_defined_vars] and by [meet_row_like]. *)
  define_again : bool;
  (* Whether [meet_row_like] defines in its result the variables that
     the environment of its one case defines, where it keeps one case
     only and makes no extension of it ([need_join] is false). *)
  lift_variables_of_single_case : bool
}.

(* What the compiler does: [blocks_with_these_tags]; it defines again;
   it does not lift. The meet with these choices is the compiler's
   but for one thing that no choice governs: the test of
   [Meet_env.replace_concrete_equation] on a name that is no longer
   canonical, where the model answers [None] and the compiler goes on
   (FlambdaMeetEnv.v; discrepancy T4-10). *)
Definition compiler_choices : meet_choices :=
  {| type_of_get_tag_scrutinee := blocks_with_these_tags;
     define_again := true;
     lift_variables_of_single_case := false |}.

(* What the contract of FlambdaMeetSpec.v is to be proved of. *)
Definition repaired_choices : meet_choices :=
  {| type_of_get_tag_scrutinee := immediate_or_blocks_with_these_tags;
     define_again := false;
     lift_variables_of_single_case := true |}.

(* ======================================================================== *)
(* The meet, given the meet of less fuel                                    *)
(* ======================================================================== *)

Section Meet.

(* The join of the environments of the alternatives of a disjunction:
   [Join_env.cut_and_n_way_join ~n_way_join_type ~meet_expanded_head
   ~cut_after initial_env initial_tenv scoped_envs]. *)
Variable join :
  meet_expanded_head -> Scope.t -> Meet_env.t -> Typing_env.t ->
  list Typing_env.t -> option Meet_env.t.

(* Where the meet can do other than the compiler: see
   [meet_choices]. *)
Variable choices : meet_choices.

(* [meet_expanded_head], of less fuel. *)
Variable meet_expanded_head_rec : meet_expanded_head.

(* The extension that has the equations of a level
   ([Typing_env_level.as_extension_without_bindings], which
   [Meet_env.cut_as_extension] and [meet_row_like] apply to the level
   that [Typing_env.cut] answers): of the equations that the level has
   for a name, the latest only. The compiler's level is a map and has
   one. *)
Definition extension_of_level (level : Typing_env_level.t) : env_extension :=
  Env_extension (Keyed.compact name_eqb (Typing_env_level.equations level)).

(* [meet]. Here the recursion goes to the meet of less fuel. *)
Definition meet (env : Meet_env.t) (t1 t2 : flambda_type)
    : option (Meet_result.t flambda_type) :=
  Meet_env.meet_type env t1 t2 meet_expanded_head_rec.

(* A variable is defined with the name mode of the variables that exist
   only in types. *)
Definition define_in_types (env : Meet_env.t) (var : variable)
    (kind : flambda_kind) : option Meet_env.t :=
  if define_again choices
     || negb (Typing_env.mem None (Meet_env.typing_env env) (Name_var var))
  then Meet_env.add_definition env (Name_var var) Name_mode.in_types kind
  else Some env.

(* [add_defined_vars]: the variables that a level defines are defined
   in an environment. *)
Definition add_defined_vars (env : Meet_env.t) (level : Typing_env_level.t)
    : option Meet_env.t :=
  let* env :=
    Typing_env_level.fold_on_defined_vars
      (fun var kind env =>
         let* env := env in
         define_in_types env var kind)
      level (Some env)
  in
  env.

(* [meet_disjunction]: the meet of two things that are each one of two
   alternatives, [a] or [b] (an immediate or a block; not null or
   null). Each alternative is met in an environment of its own, one
   scope further, to which the extensions of that alternative are
   added first. If one alternative is bottom, what the other added goes
   to the environment of the meet; if neither is, the join of the two
   environments does, and what each added is kept as its extension. *)
Definition meet_disjunction {A B : Type}
    (meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (bottom_a : unit -> A) (bottom_b : unit -> B) (initial_env : Meet_env.t)
    (val_a1 : A) (val_b1 : B) (extensions1 : variant_extensions)
    (val_a2 : A) (val_b2 : B) (extensions2 : variant_extensions)
    : option (Meet_result.t (A * B * variant_extensions)) :=
  let '(join_scope, initial_tenv, env) := Meet_env.enter_scope initial_env in
  let direct_return
      (result : meet_return_value (A * B * variant_extensions))
      (scoped_env : Meet_env.t)
      : option (Meet_result.t (A * B * variant_extensions)) :=
    let* level := Typing_env.cut (Meet_env.typing_env scoped_env) join_scope in
    let* initial_env :=
      add_defined_vars
        (Meet_env.with_names_to_avoid initial_env
           (Meet_env.names_to_avoid scoped_env))
        level
    in
    let ext := extension_of_level level in
    let* added :=
      Meet_env.add_env_extension_strict initial_env ext meet_expanded_head_rec
    in
    match added with
    | Or_bottom.Bottom =>
      Some (Meet_result.Bottom (map_return_value (fun _ => tt) result))
    | Or_bottom.Ok env => Some (Meet_result.Ok result env)
    end
  in
  let when_a (extensions : variant_extensions) : option env_extension :=
    match extensions with
    | No_extensions => None
    | Ext when_a _ => Some when_a
    end
  in
  let when_b (extensions : variant_extensions) : option env_extension :=
    match extensions with
    | No_extensions => None
    | Ext _ when_b => Some when_b
    end
  in
  let add_extension (env : Or_bottom.t Meet_env.t)
      (extension : option env_extension) : option (Or_bottom.t Meet_env.t) :=
    match extension, env with
    | None, _ => Some env
    | Some _, Or_bottom.Bottom => Some Or_bottom.Bottom
    | Some extension, Or_bottom.Ok env =>
      Meet_env.add_env_extension_strict env extension meet_expanded_head_rec
    end
  in
  let meet_side {C : Type}
      (meet_c : Meet_env.t -> C -> C -> option (Meet_result.t C))
      (env : Or_bottom.t Meet_env.t) (val1 val2 : C)
      : option (Meet_result.t C) :=
    match env with
    | Or_bottom.Bottom => Some (Meet_result.Bottom (New_result tt))
    | Or_bottom.Ok env =>
      let* result := meet_c env val1 val2 in
      match result with
      | Meet_result.Bottom r => Some (Meet_result.Bottom r)
      | Meet_result.Ok result env =>
        match Meet_env.final_typing_env_strict env with
        | Or_bottom.Bottom => Some (Meet_result.Bottom (New_result tt))
        | Or_bottom.Ok _ => Some (Meet_result.Ok result env)
        end
      end
    end
  in
  let* env_a := add_extension (Or_bottom.Ok env) (when_a extensions1) in
  let* env_a := add_extension env_a (when_a extensions2) in
  let* a_result := meet_side meet_a env_a val_a1 val_a2 in
  let env :=
    match a_result with
    | Meet_result.Bottom _ => env
    | Meet_result.Ok _ env_a =>
      Meet_env.with_names_to_avoid env (Meet_env.names_to_avoid env_a)
    end
  in
  let* env_b := add_extension (Or_bottom.Ok env) (when_b extensions1) in
  let* env_b := add_extension env_b (when_b extensions2) in
  let* b_result := meet_side meet_b env_b val_b1 val_b2 in
  match a_result, b_result with
  | Meet_result.Bottom r1, Meet_result.Bottom r2 =>
    Some
      (Meet_result.Bottom (combine_meet_return_values r1 r2 (fun _ => tt)))
  | Meet_result.Ok a_result env, Meet_result.Bottom b =>
    let result :=
      combine_meet_return_values a_result b
        (fun _ =>
           let val_a := extract_value a_result val_a1 val_a2 in
           let val_b := bottom_b tt in
           (val_a, val_b, No_extensions))
    in
    direct_return result env
  | Meet_result.Bottom a, Meet_result.Ok b_result env =>
    let result :=
      combine_meet_return_values a b_result
        (fun _ =>
           let val_b := extract_value b_result val_b1 val_b2 in
           let val_a := bottom_a tt in
           (val_a, val_b, No_extensions))
    in
    direct_return result env
  | Meet_result.Ok a_result env_a, Meet_result.Ok b_result env_b =>
    let tenv_a := Meet_env.final_typing_env env_a in
    let tenv_b := Meet_env.final_typing_env env_b in
    let* result_env :=
      join meet_expanded_head_rec join_scope
        (Meet_env.with_names_to_avoid initial_env
           (Meet_env.names_to_avoid env_b))
        initial_tenv [tenv_a; tenv_b]
    in
    let* when_a_level := Typing_env.cut tenv_a join_scope in
    let* when_b_level := Typing_env.cut tenv_b join_scope in
    let* result_env := add_defined_vars result_env when_a_level in
    let* result_env := add_defined_vars result_env when_b_level in
    let extensions :=
      if Typing_env_level.is_empty when_a_level
         && Typing_env_level.is_empty when_b_level
      then No_extensions
      else
        let when_a := extension_of_level when_a_level in
        let when_b := extension_of_level when_b_level in
        Ext when_a when_b
    in
    let env_extension_result : meet_return_value unit :=
      match extensions, extensions1, extensions2 with
      | No_extensions, No_extensions, No_extensions => Both_inputs
      | No_extensions, No_extensions, Ext _ _ => Left_input
      | No_extensions, Ext _ _, No_extensions => Right_input
      | (No_extensions | Ext _ _), _, _ => New_result tt
      end
    in
    let result :=
      combine_meet_return_values
        (combine_meet_return_values a_result b_result (fun _ => tt))
        env_extension_result
        (fun _ =>
           let val_a := extract_value a_result val_a1 val_a2 in
           let val_b := extract_value b_result val_b1 val_b2 in
           (val_a, val_b, extensions))
    in
    Some (Meet_result.Ok result result_env)
  end.

(* What [meet_row_like] keeps in references in the compiler: the
   variables that the environments of the cases define, in the order in
   which they were met; those environments, the last first, if there is
   one; whether the result is the first input, and the second. And the
   names to avoid, which the model hands from one case to the next. *)
Record row_like_state : Type := {
  extra_variables : list (variable * flambda_kind);
  result_env : option (list Typing_env.t);
  result_is_t1 : bool;
  result_is_t2 : bool;
  avoid : list variable
}.

Definition not_t1 (state : row_like_state) : row_like_state :=
  {| extra_variables := extra_variables state;
     result_env := result_env state;
     result_is_t1 := false;
     result_is_t2 := result_is_t2 state;
     avoid := avoid state |}.

Definition not_t2 (state : row_like_state) : row_like_state :=
  {| extra_variables := extra_variables state;
     result_env := result_env state;
     result_is_t1 := result_is_t1 state;
     result_is_t2 := false;
     avoid := avoid state |}.

(* [update_refs]. *)
Definition update_refs {A : Type} (r : meet_return_value A)
    (state : row_like_state) : row_like_state :=
  match r with
  | Both_inputs => state
  | Left_input => not_t2 state
  | Right_input => not_t1 state
  | New_result _ => not_t1 (not_t2 state)
  end.

(* The variables of a level are added to the extra variables. [None] is
   a fatal error of the compiler: a variable with two kinds. *)
Fixpoint union_extra_variables (extra vars : list (variable * flambda_kind))
    : option (list (variable * flambda_kind)) :=
  match vars with
  | [] => Some extra
  | (var, kind) :: vars =>
    match lookup var extra with
    | Some kind' =>
      if kind_eqb kind kind' then union_extra_variables extra vars else None
    | None => union_extra_variables (extra ++ [(var, kind)]) vars
    end
  end.

Section Row_like.

Context {lattice shape maps_to row_tag known : Type}.

Let case : Type := row_like_case lattice shape maps_to env_extension.

Variable meet_maps_to :
  Meet_env.t -> maps_to -> maps_to -> option (Meet_result.t maps_to).
Variable equal_index : lattice -> lattice -> bool.
Variable subset_index : lattice -> lattice -> bool.
Variable union_index : lattice -> lattice -> lattice.
Variable meet_shape : shape -> shape -> Or_bottom.t shape.
Variable is_empty_map_known : known -> bool.
Variable get_singleton_map_known : known -> option (row_tag * case).
(* [merge_map_known] goes through the cases of both maps, with the
   state. [None] is a fatal error. *)
Variable merge_map_known :
  (row_tag -> option (Or_unknown.t case) -> option (Or_unknown.t case) ->
   row_like_state -> option (option (Or_unknown.t case) * row_like_state)) ->
  known -> known -> row_like_state -> option (known * row_like_state).

(* [meet_row_like]: the meet of two row-like structures, case by case:
   the cases of a tag that both have, and each case that one has with
   the case of the other tags of the other. Each pair of cases is met
   in an environment of its own; a pair whose meet is bottom goes; the
   environments of those that stay are joined. *)
Definition meet_row_like (initial_env : Meet_env.t) (known1 known2 : known)
    (other1 other2 : Or_bottom.t case)
    : option (Meet_result.t (known * Or_bottom.t case)) :=
  let '(common_scope, initial_tenv, base_env) :=
    Meet_env.enter_scope initial_env
  in
  let base_tenv := Meet_env.final_typing_env base_env in
  let add_extra_variables_and_extract_extension (scoped_env : Typing_env.t)
      (state : row_like_state) : option (env_extension * row_like_state) :=
    let* level := Typing_env.cut scoped_env common_scope in
    let* extra :=
      union_extra_variables (extra_variables state)
        (rev (Typing_env_level.defined_variables_with_kinds level))
    in
    Some
      (extension_of_level level,
       {| extra_variables := extra;
          result_env := result_env state;
          result_is_t1 := result_is_t1 state;
          result_is_t2 := result_is_t2 state;
          avoid := avoid state |})
  in
  let extract_and_join_extensions (scoped_envs : list Typing_env.t)
      (state : row_like_state) : option Meet_env.t :=
    let* result_env :=
      join meet_expanded_head_rec common_scope
        (Meet_env.with_names_to_avoid initial_env (avoid state))
        initial_tenv scoped_envs
    in
    fold_left
      (fun env binding =>
         let* env := env in
         define_in_types env (fst binding) (snd binding))
      (extra_variables state) (Some result_env)
  in
  let need_join : bool :=
    match
      other1, get_singleton_map_known known1, other2,
      get_singleton_map_known known2
    with
    | Or_bottom.Bottom, Some _, _, _ | _, _, Or_bottom.Bottom, Some _ => false
    | (Or_bottom.Ok _ | Or_bottom.Bottom), _,
      (Or_bottom.Ok _ | Or_bottom.Bottom), _ =>
      if is_empty_map_known known1 && is_empty_map_known known2
      then false
      else true
    end
  in
  (* [join_result_env], with the names to avoid of the environment that
     the scoped one came from. [None] is the assertion of the compiler
     that only one environment is kept unless a join is needed. *)
  let join_result_env (scoped_env : Typing_env.t) (names : list variable)
      (state : row_like_state) : option row_like_state :=
    let* new_result_env :=
      match result_env state with
      | None => Some [scoped_env]
      | Some other_envs =>
        if need_join then Some (scoped_env :: other_envs) else None
      end
    in
    Some
      {| extra_variables := extra_variables state;
         result_env := Some new_result_env;
         result_is_t1 := result_is_t1 state;
         result_is_t2 := result_is_t2 state;
         avoid := names |}
  in
  let meet_index (env : Meet_env.t) (i1 i2 : row_like_index lattice shape)
      : Meet_result.t (row_like_index lattice shape) :=
    match meet_shape (index_shape i1) (index_shape i2) with
    | Or_bottom.Bottom => Meet_result.Bottom (New_result tt)
    | Or_bottom.Ok shape =>
      match index_domain i1, index_domain i2 with
      | Row_like_index_domain.Known i1', Row_like_index_domain.Known i2' =>
        if equal_index i1' i2'
        then Meet_result.Ok Both_inputs env
        else Meet_result.Bottom (New_result tt)
      | Row_like_index_domain.Known known,
        Row_like_index_domain.At_least at_least =>
        if subset_index at_least known
        then Meet_result.Ok Left_input env
        else Meet_result.Bottom (New_result tt)
      | Row_like_index_domain.At_least at_least,
        Row_like_index_domain.Known known =>
        if subset_index at_least known
        then Meet_result.Ok Right_input env
        else Meet_result.Bottom (New_result tt)
      | Row_like_index_domain.At_least i1',
        Row_like_index_domain.At_least i2' =>
        if subset_index i1' i2'
        then
          if subset_index i2' i1'
          then Meet_result.Ok Both_inputs env
          else Meet_result.Ok Right_input env
        else if subset_index i2' i1'
        then Meet_result.Ok Left_input env
        else
          let domain :=
            Row_like_index_domain.At_least (union_index i1' i2')
          in
          Meet_result.Ok (New_result (Row_like_index domain shape)) env
      end
    end
  in
  let bottom_case {A : Type} (r : meet_return_value A)
      (state : row_like_state)
      : option (option (Or_unknown.t case) * row_like_state) :=
    Some (None, update_refs r state)
  in
  (* The environment of a case to which an extension is added. *)
  let add_case_extension (extension : env_extension) (state : row_like_state)
      : option (Or_bottom.t Meet_env.t) :=
    let* added :=
      Meet_env.add_env_extension_strict
        (Meet_env.with_names_to_avoid base_env (avoid state))
        extension meet_expanded_head_rec
    in
    match added with
    | Or_bottom.Bottom => Some Or_bottom.Bottom
    | Or_bottom.Ok env =>
      match Meet_env.final_typing_env_strict env with
      | Or_bottom.Bottom => Some Or_bottom.Bottom
      | Or_bottom.Ok _ => Some (Or_bottom.Ok env)
      end
    end
  in
  let meet_case (env : Meet_env.t) (case1 case2 : case)
      (state : row_like_state)
      : option (option (Or_unknown.t case) * row_like_state) :=
    let env := Meet_env.with_names_to_avoid env (avoid state) in
    match meet_index env (case_index case1) (case_index case2) with
    | Meet_result.Bottom r => bottom_case r state
    | Meet_result.Ok index_result env =>
      let* maps_to :=
        meet_maps_to env (case_maps_to case1) (case_maps_to case2)
      in
      match maps_to with
      | Meet_result.Bottom r => bottom_case r state
      | Meet_result.Ok maps_to_result env =>
        let* env :=
          Meet_env.add_env_extension_strict env (case_env_extension case1)
            meet_expanded_head_rec
        in
        let* env :=
          match env with
          | Or_bottom.Bottom => Some Or_bottom.Bottom
          | Or_bottom.Ok env =>
            Meet_env.add_env_extension_strict env (case_env_extension case2)
              meet_expanded_head_rec
          end
        in
        let env :=
          match env with
          | Or_bottom.Bottom => Or_bottom.Bottom
          | Or_bottom.Ok env =>
            match Meet_env.final_typing_env_strict env with
            | Or_bottom.Bottom => Or_bottom.Bottom
            | Or_bottom.Ok _ => Or_bottom.Ok env
            end
          end
        in
        match env with
        | Or_bottom.Bottom => bottom_case (@New_result unit tt) state
        | Or_bottom.Ok env =>
          let tenv := Meet_env.final_typing_env env in
          let* state :=
            join_result_env tenv (Meet_env.names_to_avoid env) state
          in
          let state := update_refs index_result state in
          let state := update_refs maps_to_result state in
          let index :=
            extract_value index_result (case_index case1) (case_index case2)
          in
          let maps_to :=
            extract_value maps_to_result (case_maps_to case1)
              (case_maps_to case2)
          in
          (* Not in the compiler: where no extension is made of the
             case, its variables may be lifted all the same (see
             [meet_choices]). *)
          let* '(env_extension, state) :=
            if need_join
            then add_extra_variables_and_extract_extension tenv state
            else if lift_variables_of_single_case choices
            then
              let* '(_, state) :=
                add_extra_variables_and_extract_extension tenv state
              in
              Some (empty_env_extension, state)
            else Some (empty_env_extension, state)
          in
          let state :=
            match env_extension with
            | Env_extension [] => state
            | Env_extension (_ :: _) => not_t1 (not_t2 state)
            end
          in
          let case := Row_like_case maps_to index env_extension in
          Some (Some (Or_unknown.Known case), state)
        end
      end
    end
  in
  (* A case of one input that is unknown, with a case of the other: the
     case of the other, if its extension has a value. *)
  let meet_case_with_unknown (other_case : case)
      (change : row_like_state -> row_like_state) (state : row_like_state)
      : option (option (Or_unknown.t case) * row_like_state) :=
    let* env := add_case_extension (case_env_extension other_case) state in
    match env with
    | Or_bottom.Bottom => Some (None, state)
    | Or_bottom.Ok env =>
      let* state :=
        join_result_env (Meet_env.final_typing_env env)
          (Meet_env.names_to_avoid env) state
      in
      Some (Some (Or_unknown.Known other_case), change state)
    end
  in
  let meet_knowns (case1 case2 : option (Or_unknown.t case))
      (state : row_like_state)
      : option (option (Or_unknown.t case) * row_like_state) :=
    match case1, case2 with
    | None, None => Some (None, state)
    | Some case1, None =>
      match other2 with
      | Or_bottom.Bottom => Some (None, not_t1 state)
      | Or_bottom.Ok other_case =>
        match case1 with
        | Or_unknown.Unknown =>
          meet_case_with_unknown other_case
            (fun state => not_t1 (not_t2 state))
            state
        | Or_unknown.Known case1 => meet_case base_env case1 other_case state
        end
      end
    | None, Some case2 =>
      match other1 with
      | Or_bottom.Bottom => Some (None, not_t2 state)
      | Or_bottom.Ok other_case =>
        match case2 with
        | Or_unknown.Unknown =>
          meet_case_with_unknown other_case
            (fun state => not_t1 (not_t2 state))
            state
        | Or_unknown.Known case2 => meet_case base_env other_case case2 state
        end
      end
    | Some case1, Some case2 =>
      match case1, case2 with
      | Or_unknown.Unknown, Or_unknown.Unknown =>
        let* state := join_result_env base_tenv (avoid state) state in
        Some (Some Or_unknown.Unknown, state)
      | Or_unknown.Known case, Or_unknown.Unknown =>
        meet_case_with_unknown case not_t2 state
      | Or_unknown.Unknown, Or_unknown.Known case =>
        meet_case_with_unknown case not_t1 state
      | Or_unknown.Known case1, Or_unknown.Known case2 =>
        meet_case base_env case1 case2 state
      end
    end
  in
  let state :=
    {| extra_variables := [];
       result_env := None;
       result_is_t1 := true;
       result_is_t2 := true;
       avoid := Meet_env.names_to_avoid base_env |}
  in
  let* '(known, state) :=
    merge_map_known (fun _tag case1 case2 => meet_knowns case1 case2) known1
      known2 state
  in
  let* '(other, state) :=
    match other1, other2 with
    | Or_bottom.Bottom, Or_bottom.Bottom => Some (Or_bottom.Bottom, state)
    | Or_bottom.Bottom, _ => Some (Or_bottom.Bottom, not_t2 state)
    | _, Or_bottom.Bottom => Some (Or_bottom.Bottom, not_t1 state)
    | Or_bottom.Ok other1, Or_bottom.Ok other2 =>
      let* '(other, state) := meet_case base_env other1 other2 state in
      match other with
      | None => Some (Or_bottom.Bottom, state)
      | Some Or_unknown.Unknown => None
      | Some (Or_unknown.Known r) => Some (Or_bottom.Ok r, state)
      end
    end
  in
  if is_empty_map_known known
     && match other with
        | Or_bottom.Bottom => true
        | Or_bottom.Ok _ => false
        end
  then Some (Meet_result.Bottom (New_result tt))
  else
    let* env :=
      match result_env state with
      | None => Some Or_bottom.Bottom
      | Some scoped_envs =>
        let* env := extract_and_join_extensions scoped_envs state in
        Some (Or_bottom.Ok env)
      end
    in
    let match_with_input {A : Type} (v : A) : meet_return_value A :=
      match result_is_t1 state, result_is_t2 state with
      | true, true => Both_inputs
      | true, false => Left_input
      | false, true => Right_input
      | false, false => New_result v
      end
    in
    match env with
    | Or_bottom.Bottom => Some (Meet_result.Bottom (match_with_input tt))
    | Or_bottom.Ok env =>
      Some (Meet_result.Ok (match_with_input (known, other)) env)
    end.

End Row_like.

(* [meet_relation]: the meet of two variables that stand for the result
   of one primitive on one value: they are made aliases. *)
Definition meet_relation (env : Meet_env.t) (var1 var2 : option variable)
    : option (Meet_result.t (option variable)) :=
  match var1, var2 with
  | None, None => Some (Meet_result.Ok Both_inputs env)
  | Some _, None => Some (Meet_result.Ok Left_input env)
  | None, Some _ => Some (Meet_result.Ok Right_input env)
  | Some var1, Some var2 =>
    let* added :=
      Meet_env.add_alias env (Var var1) (Var var2) meet_expanded_head_rec
    in
    match added with
    | Or_bottom.Ok env => Some (Meet_result.Ok Both_inputs env)
    | Or_bottom.Bottom => Some (Meet_result.Bottom (New_result tt))
    end
  end.

(* [meet_function_type]. The model's function types have no
   [rec_info]. The compiler meets the two as types of kind [Rec_info],
   which answers the inputs and adds nothing unless one of them is an
   alias. *)
Definition meet_function_type (env : Meet_env.t)
    (func_type1 func_type2 : Or_unknown.t function_type)
    : option (Meet_result.t (Or_unknown.t function_type)) :=
  match func_type1, func_type2 with
  | Or_unknown.Unknown, Or_unknown.Unknown =>
    Some (Meet_result.Ok Both_inputs env)
  | _, Or_unknown.Unknown => Some (Meet_result.Ok Left_input env)
  | Or_unknown.Unknown, _ => Some (Meet_result.Ok Right_input env)
  | Or_unknown.Known func_type1, Or_unknown.Known func_type2 =>
    Some
      (map_result
         (fun code_id => Or_unknown.Known (Function_type code_id))
         (meet_code_id env
            (function_type_code_id func_type1)
            (function_type_code_id func_type2)))
  end.

(* [meet_array_of_types]: the meet of two lists of types, position by
   position, as far as [length]. *)
Definition meet_array_of_types (env : Meet_env.t)
    (fields1 fields2 : list flambda_type) (length : nat)
    : option (Meet_result.t (list flambda_type)) :=
  let fold2
      (f :
        nat -> option flambda_type -> option flambda_type ->
        mapping_state nat flambda_type -> mapping_state nat flambda_type)
      (left right : list flambda_type)
      (init : mapping_state nat flambda_type)
      : mapping_state nat flambda_type :=
    fold_left
      (fun acc i => f i (nth_error left i) (nth_error right i) acc)
      (seq 0 length) init
  in
  let rebuild (l : list (nat * flambda_type)) : list flambda_type :=
    match l with
    | [] => []
    | (_key, data) :: _ =>
      map
        (fun i =>
           match Keyed.lookup Nat.eqb i l with
           | Some data => data
           | None => data
           end)
        (seq 0 length)
    end
  in
  meet_mapping meet fold2 env fields1 fields2 rebuild.

(* [meet_int_indexed_product]: the fields of a block. *)
Definition meet_int_indexed_product (env : Meet_env.t)
    (fields1 fields2 : list flambda_type)
    : option (Meet_result.t (list flambda_type)) :=
  let length := Nat.max (List.length fields1) (List.length fields2) in
  meet_array_of_types env fields1 fields2 length.

(* [meet_product_function_slot_indexed]: the closures of a set. *)
Definition meet_product_function_slot_indexed (env : Meet_env.t)
    (components_by_index1 components_by_index2 : finite_map flambda_type)
    : option (Meet_result.t (finite_map flambda_type)) :=
  map_meet Pos.eqb meet env components_by_index1 components_by_index2.

(* [meet_product_value_slot_indexed]: the value slots of a set. *)
Definition meet_product_value_slot_indexed (env : Meet_env.t)
    (components_by_index1 components_by_index2 : finite_map flambda_type)
    : option (Meet_result.t (finite_map flambda_type)) :=
  map_meet Pos.eqb meet env components_by_index1 components_by_index2.

(* [meet_closures_entry]. *)
Definition meet_closures_entry (env : Meet_env.t)
    (entry1 entry2 : closures_entry) : option (Meet_result.t closures_entry) :=
  match entry1, entry2 with
  | Closures_entry function_types1 closure_types1 value_slot_types1,
    Closures_entry function_types2 closure_types2 value_slot_types2 =>
    combine_results3 env
      (map_meet Pos.eqb meet_function_type)
      meet_product_function_slot_indexed meet_product_value_slot_indexed
      function_types1 function_types2 closure_types1 closure_types2
      value_slot_types1 value_slot_types2
      (fun function_types closure_types value_slot_types =>
         Closures_entry function_types closure_types value_slot_types)
  end.

(* A merge of two maps that threads a state, and may stop with a fatal
   error. *)
Definition merge_fold_failing {K A B C S : Type} (eqb : K -> K -> bool)
    (keys : list K)
    (f : K -> option A -> option B -> S -> option (option C * S))
    (m1 : Keyed.finite_map K A) (m2 : Keyed.finite_map K B) (init : S)
    : option (Keyed.finite_map K C * S) :=
  let '(out, state) :=
    Keyed.merge_fold_on eqb keys
      (fun key a b (state : option S) =>
         match state with
         | None => (None, None)
         | Some state =>
           match f key a b state with
           | None => (None, None)
           | Some (c, state) => (c, Some state)
           end
         end)
      m1 m2 (Some init)
  in
  option_map (fun state => (out, state)) state.

(* [meet_row_like_for_blocks]. The cases are gone through in the order
   of their tags, as [Tag.Map.merge] does. *)
Definition meet_row_like_for_blocks (env : Meet_env.t)
    (blocks1 blocks2 : row_like_for_blocks)
    : option (Meet_result.t row_like_for_blocks) :=
  match blocks1, blocks2 with
  | Row_like_for_blocks known1 other1 alloc_mode1,
    Row_like_for_blocks known2 other2 alloc_mode2 =>
    let meet_shape (shape1 shape2 : block_shape) : Or_bottom.t block_shape :=
      if block_shape_eqb shape1 shape2
      then Or_bottom.Ok shape1
      else Or_bottom.Bottom
    in
    let get_singleton_map_known
        (known : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
        : option (Z * row_like_block_case) :=
      match Keyed.get_singleton_binding Z.eqb known with
      | Some (tag, Or_unknown.Known case) => Some (tag, case)
      | Some (_, Or_unknown.Unknown) | None => None
      end
    in
    let merge_map_known
        (merge_case :
          Z -> option (Or_unknown.t row_like_block_case) ->
          option (Or_unknown.t row_like_block_case) -> row_like_state ->
          option (option (Or_unknown.t row_like_block_case) * row_like_state))
        (known1 known2 :
          Keyed.finite_map Z (Or_unknown.t row_like_block_case))
        (state : row_like_state) :=
      merge_fold_failing Z.eqb
        (sort_numbers (Keyed.keys_of_both Z.eqb known1 known2))
        merge_case known1 known2 state
    in
    combine_results2 env
      (fun env
           (left right :
             Keyed.finite_map Z (Or_unknown.t row_like_block_case)
             * Or_bottom.t row_like_block_case) =>
         meet_row_like meet_int_indexed_product Nat.eqb Nat.leb Nat.max
           meet_shape Keyed.is_empty get_singleton_map_known merge_map_known
           env (fst left) (fst right) (snd left) (snd right))
      (fun env alloc_mode1 alloc_mode2 =>
         Some (meet_alloc_mode env alloc_mode1 alloc_mode2))
      (known1, other1) (known2, other2) alloc_mode1 alloc_mode2
      (fun known_and_other alloc_mode =>
         Row_like_for_blocks (fst known_and_other) (snd known_and_other)
           alloc_mode)
  end.

(* [meet_row_like_for_closures]. A row-like structure of closures has
   no case for the other slots. The cases are gone through in the order
   of [keys_of_both] (see the header). *)
Definition meet_row_like_for_closures (env : Meet_env.t)
    (closures1 closures2 : row_like_for_closures)
    : option (Meet_result.t row_like_for_closures) :=
  match closures1, closures2 with
  | Row_like_for_closures known1, Row_like_for_closures known2 =>
    let meet_shape (_ _ : unit) : Or_bottom.t unit := Or_bottom.Ok tt in
    let merge_map_known
        (merge_case :
          function_slot -> option (Or_unknown.t row_like_closures_case) ->
          option (Or_unknown.t row_like_closures_case) -> row_like_state ->
          option
            (option (Or_unknown.t row_like_closures_case) * row_like_state))
        (known1 known2 : finite_map row_like_closures_case)
        (state : row_like_state) :=
      merge_fold_failing Pos.eqb
        (Keyed.keys_of_both Pos.eqb known1 known2)
        (fun fslot case1 case2 state =>
           let case1 := option_map Or_unknown.Known case1 in
           let case2 := option_map Or_unknown.Known case2 in
           let* '(case, state) := merge_case fslot case1 case2 state in
           match case with
           | None => Some (None, state)
           | Some (Or_unknown.Known case) => Some (Some case, state)
           | Some Or_unknown.Unknown => None
           end)
        known1 known2 state
    in
    let* result :=
      meet_row_like meet_closures_entry set_of_closures_contents_equal
        set_of_closures_contents_subset set_of_closures_contents_union
        meet_shape Keyed.is_empty (Keyed.get_singleton_binding Pos.eqb)
        merge_map_known env known1 known2 Or_bottom.Bottom Or_bottom.Bottom
    in
    match result with
    | Meet_result.Bottom r => Some (Meet_result.Bottom r)
    | Meet_result.Ok (New_result (_, Or_bottom.Ok _)) _ => None
    | Meet_result.Ok r env =>
      Some
        (Meet_result.Ok
           (map_return_value
              (fun known_and_other =>
                 Row_like_for_closures (fst known_and_other))
              r)
           env)
    end
  end.

(* What [MTC.blocks_with_these_tags] answers: nothing if one of the
   tags is not that of a structured block. *)
Definition type_of_get_tag_scrutinee_or_unknown (tags : list Z)
    (alloc_mode : Alloc_mode.t) : Or_unknown.t flambda_type :=
  if forallb is_structured_block tags
  then Or_unknown.Known (type_of_get_tag_scrutinee choices tags alloc_mode)
  else Or_unknown.Unknown.

(* [reduce_inverse_relations]: what the numbers that a naked immediate
   may be say of the names of which it is the result of a relation. *)
Definition reduce_inverse_relations (env : Meet_env.t)
    (naked_immediates : Or_unknown.t (list Z))
    (inverse_relations : list (Relation.t * name))
    : option (Or_bottom.t Meet_env.t) :=
  match naked_immediates with
  | Or_unknown.Unknown => Some (Or_bottom.Ok env)
  | Or_unknown.Known imms =>
    let step (state : option (Or_bottom.t Meet_env.t))
        (inverse : Relation.t * name) : option (Or_bottom.t Meet_env.t) :=
      match state with
      | None => None
      | Some Or_bottom.Bottom => Some Or_bottom.Bottom
      | Some (Or_bottom.Ok env) =>
        let name := snd inverse in
        let add_equation (ty : flambda_type) :=
          Meet_env.add_equation_strict env name ty meet_expanded_head_rec
        in
        match fst inverse with
        | Relation.Is_null =>
          match Keyed.among Z.eqb 0%Z imms, Keyed.among Z.eqb 1%Z imms with
          | false, false => Some Or_bottom.Bottom
          | true, true => Some (Or_bottom.Ok env)
          | true, false => add_equation any_non_null_value
          | false, true => add_equation null
          end
        | Relation.Is_int =>
          match Keyed.among Z.eqb 0%Z imms, Keyed.among Z.eqb 1%Z imms with
          | false, false => Some Or_bottom.Bottom
          | true, true => Some (Or_bottom.Ok env)
          | true, false => add_equation any_block
          | false, true => add_equation any_tagged_immediate
          end
        | Relation.Get_tag =>
          let tags := Keyed.dedup Z.eqb (filter is_tag imms) in
          match
            type_of_get_tag_scrutinee_or_unknown tags Alloc_mode.Heap_or_local
          with
          | Or_unknown.Known shape => add_equation shape
          | Or_unknown.Unknown => Some (Or_bottom.Ok env)
          end
        end
      end
    in
    fold_left step inverse_relations (Some (Or_bottom.Ok env))
  end.

(* [meet_head_of_kind_naked_immediate]. *)
Definition meet_head_of_kind_naked_immediate (env : Meet_env.t)
    (t1 t2 : head_of_kind_naked_immediate)
    : option (Meet_result.t head_of_kind_naked_immediate) :=
  match
    reduce_head_of_kind_naked_immediate (Meet_env.current_typing_env env) t1,
    reduce_head_of_kind_naked_immediate (Meet_env.current_typing_env env) t2
  with
  | Or_bottom.Bottom, Or_bottom.Bottom =>
    Some (Meet_result.Bottom Both_inputs)
  | Or_bottom.Bottom, Or_bottom.Ok _ => Some (Meet_result.Bottom Left_input)
  | Or_bottom.Ok _, Or_bottom.Bottom => Some (Meet_result.Bottom Right_input)
  | Or_bottom.Ok t1, Or_bottom.Ok t2 =>
    let* naked_immediates :=
      meet_unknown
        (fun env s1 s2 => Some (set_meet Z.eqb (fun s => s) env s1 s2))
        Keyed.is_empty env (naked_immediates t1) (naked_immediates t2)
    in
    match naked_immediates with
    | Meet_result.Bottom r => Some (Meet_result.Bottom r)
    | Meet_result.Ok naked_immediates env =>
      let inverse_relations :=
        meet_inverse_relations (inverse_relations t1) (inverse_relations t2)
      in
      let* head :=
        sequence_return_value
          (combine_meet_return_values naked_immediates
             (meet_return_value_of inverse_relations)
             (fun _ =>
                let naked_immediates :=
                  extract_value naked_immediates
                    (FlambdaTypeGrammar.naked_immediates t1)
                    (FlambdaTypeGrammar.naked_immediates t2)
                in
                let inverse_relations :=
                  extract_value (meet_return_value_of inverse_relations)
                    (FlambdaTypeGrammar.inverse_relations t1)
                    (FlambdaTypeGrammar.inverse_relations t2)
                in
                head_of_kind_naked_immediate_from_descr_non_empty
                  naked_immediates inverse_relations))
      in
      let head_in_env_or_bottom (env_ob : Or_bottom.t Meet_env.t)
          : Meet_result.t head_of_kind_naked_immediate :=
        match env_ob with
        | Or_bottom.Bottom => Meet_result.Bottom (New_result tt)
        | Or_bottom.Ok env => Meet_result.Ok head env
        end
      in
      match naked_immediates with
      | Both_inputs => Some (Meet_result.Ok head env)
      | Left_input =>
        option_map head_in_env_or_bottom
          (reduce_inverse_relations env
             (FlambdaTypeGrammar.naked_immediates t1)
             (right_inputs_only inverse_relations))
      | Right_input =>
        option_map head_in_env_or_bottom
          (reduce_inverse_relations env
             (FlambdaTypeGrammar.naked_immediates t2)
             (left_inputs_only inverse_relations))
      | New_result naked_immediates =>
        option_map head_in_env_or_bottom
          (reduce_inverse_relations env naked_immediates
             (extract_value (meet_return_value_of inverse_relations)
                (FlambdaTypeGrammar.inverse_relations t1)
                (FlambdaTypeGrammar.inverse_relations t2)))
      end
    end
  end.

(* [meet_head_of_kind_naked_float32], ..., [meet_head_of_kind_naked_mask]:
   the eleven kinds of naked numbers other than the immediates. *)
Definition meet_head_of_kind_naked_number (env : Meet_env.t)
    (t1 t2 : list Z) : Meet_result.t (list Z) :=
  set_meet Z.eqb (fun s => s) env t1 t2.

(* [meet_head_of_kind_rec_info] and [meet_head_of_kind_region]. *)
Definition meet_head_of_kind_rec_info (env : Meet_env.t) (_ _ : unit)
    : Meet_result.t unit :=
  Meet_result.Ok Both_inputs env.

Definition meet_head_of_kind_region (env : Meet_env.t) (_ _ : unit)
    : Meet_result.t unit :=
  Meet_result.Ok Both_inputs env.

(* [meet_variant]. The variables for [Is_int] are met in the
   environment of the meet: every variant is an immediate or a block.
   Those for [Get_tag] are met with the blocks, in the environment of
   that alternative: only a block has a tag. *)
Definition meet_variant (env : Meet_env.t) (is_int1 get_tag1 : option variable)
    (blocks1 : Or_unknown.t row_like_for_blocks)
    (imms1 : Or_unknown.t flambda_type) (is_int2 get_tag2 : option variable)
    (blocks2 : Or_unknown.t row_like_for_blocks)
    (imms2 : Or_unknown.t flambda_type)
    (extensions1 extensions2 : variant_extensions)
    : option
        (Meet_result.t
           (option variable * option variable
            * Or_unknown.t row_like_for_blocks * Or_unknown.t flambda_type
            * variant_extensions)) :=
  let meet_a := meet_unknown meet is_obviously_bottom in
  let meet_b (env : Meet_env.t)
      (left right : option variable * Or_unknown.t row_like_for_blocks) :=
    combine_results2 env meet_relation
      (meet_unknown meet_row_like_for_blocks row_like_for_blocks_is_bottom)
      (fst left) (fst right) (snd left) (snd right)
      (fun get_tag blocks => (get_tag, blocks))
  in
  let bottom_a (_ : unit) : Or_unknown.t flambda_type :=
    Or_unknown.Known (bottom (Naked_number Kind_naked_immediate))
  in
  let bottom_b (_ : unit)
      : option variable * Or_unknown.t row_like_for_blocks :=
    (None, Or_unknown.Known row_like_for_blocks_bottom)
  in
  combine_results2 env meet_relation
    (fun env (left right :
                option variable * Or_unknown.t row_like_for_blocks
                * Or_unknown.t flambda_type * variant_extensions) =>
       let '(get_tag1, blocks1, imms1, extensions1) := left in
       let '(get_tag2, blocks2, imms2, extensions2) := right in
       option_map
         (map_result
            (fun result : Or_unknown.t flambda_type
                          * (option variable
                             * Or_unknown.t row_like_for_blocks)
                          * variant_extensions =>
               let '(imms, (get_tag, blocks), extensions) := result in
               (get_tag, blocks, imms, extensions)))
         (meet_disjunction meet_a meet_b bottom_a bottom_b env imms1
            (get_tag1, blocks1) extensions1 imms2 (get_tag2, blocks2)
            extensions2))
    is_int1 is_int2
    (get_tag1, blocks1, imms1, extensions1)
    (get_tag2, blocks2, imms2, extensions2)
    (fun is_int rest =>
       let '(get_tag, blocks, imms, extensions) := rest in
       (is_int, get_tag, blocks, imms, extensions)).

(* [meet_head_of_kind_value_non_null]. Two heads made with different
   constructors have no value in common, but for a variant and a
   mutable block. *)
Definition meet_head_of_kind_value_non_null (env : Meet_env.t)
    (head1 head2 : head_of_kind_value_non_null)
    : option (Meet_result.t head_of_kind_value_non_null) :=
  match head1, head2 with
  | Variant is_int1 imms1 get_tag1 blocks1 extensions1 is_unique1,
    Variant is_int2 imms2 get_tag2 blocks2 extensions2 is_unique2 =>
    let is_unique := is_unique1 || is_unique2 in
    option_map
      (map_result
         (fun result : option variable * option variable
                       * Or_unknown.t row_like_for_blocks
                       * Or_unknown.t flambda_type * variant_extensions =>
            let '(is_int, get_tag, blocks, immediates, extensions) := result in
            Variant is_int immediates get_tag blocks extensions is_unique))
      (meet_variant env is_int1 get_tag1 blocks1 imms1 is_int2 get_tag2
         blocks2 imms2 extensions1 extensions2)
  | Mutable_block alloc_mode1, Mutable_block alloc_mode2 =>
    Some
      (map_result Mutable_block (meet_alloc_mode env alloc_mode1 alloc_mode2))
  | Variant _ _ _ blocks _ _, Mutable_block alloc_mode_right =>
    match blocks with
    | Or_unknown.Unknown => Some (Meet_result.Ok Right_input env)
    | Or_unknown.Known (Row_like_for_blocks _ _ alloc_mode_left) =>
      match meet_alloc_mode env alloc_mode_left alloc_mode_right with
      | Meet_result.Bottom r => Some (Meet_result.Bottom r)
      | Meet_result.Ok (Both_inputs | Right_input) env =>
        Some (Meet_result.Ok Right_input env)
      | Meet_result.Ok Left_input env =>
        Some (Meet_result.Ok (New_result (Mutable_block alloc_mode_left)) env)
      | Meet_result.Ok (New_result alloc_mode) env =>
        Some (Meet_result.Ok (New_result (Mutable_block alloc_mode)) env)
      end
    end
  | Mutable_block alloc_mode_left, Variant _ _ _ blocks _ _ =>
    match blocks with
    | Or_unknown.Unknown => Some (Meet_result.Ok Left_input env)
    | Or_unknown.Known (Row_like_for_blocks _ _ alloc_mode_right) =>
      match meet_alloc_mode env alloc_mode_left alloc_mode_right with
      | Meet_result.Bottom r => Some (Meet_result.Bottom r)
      | Meet_result.Ok (Both_inputs | Left_input) env =>
        Some (Meet_result.Ok Left_input env)
      | Meet_result.Ok Right_input env =>
        Some (Meet_result.Ok (New_result (Mutable_block alloc_mode_right)) env)
      | Meet_result.Ok (New_result alloc_mode) env =>
        Some (Meet_result.Ok (New_result (Mutable_block alloc_mode)) env)
      end
    end
  | Boxed_number k1 n1 alloc_mode1, Boxed_number k2 n2 alloc_mode2 =>
    if boxable_number_eqb k1 k2
    then
      combine_results2 env meet
        (fun env alloc_mode1 alloc_mode2 =>
           Some (meet_alloc_mode env alloc_mode1 alloc_mode2))
        n1 n2 alloc_mode1 alloc_mode2 (Boxed_number k1)
    else Some (Meet_result.Bottom (New_result tt))
  | Closures by_function_slot1 alloc_mode1,
    Closures by_function_slot2 alloc_mode2 =>
    combine_results2 env meet_row_like_for_closures
      (fun env alloc_mode1 alloc_mode2 =>
         Some (meet_alloc_mode env alloc_mode1 alloc_mode2))
      by_function_slot1 by_function_slot2 alloc_mode1 alloc_mode2 Closures
  | String strs1, String strs2 =>
    Some
      (map_result String
         (set_meet (lists_eqb Z.eqb) (fun s => s) env strs1 strs2))
  | (Variant _ _ _ _ _ _ | Mutable_block _ | Boxed_number _ _ _
    | Closures _ _ | String _), _ =>
    Some (Meet_result.Bottom (New_result tt))
  end.

(* [meet_head_of_kind_value]: a value is not null, or null. *)
Definition meet_head_of_kind_value (env : Meet_env.t)
    (head1 head2 : head_of_kind_value)
    : option (Meet_result.t head_of_kind_value) :=
  match head1, head2 with
  | Head_of_kind_value non_null1 is_null1,
    Head_of_kind_value non_null2 is_null2 =>
    let meet_a :=
      let meet_elt (env : Meet_env.t)
          (elt1 elt2 : head_of_kind_value_non_null) :=
        option_map
          (map_result
             (fun x : head_of_kind_value_non_null =>
                @Ok head_of_kind_value_non_null x))
          (meet_head_of_kind_value_non_null env elt1 elt2)
      in
      meet_or_unknown_or_bottom meet_elt
    in
    let meet_b (env : Meet_env.t) (is_null1 is_null2 : is_null)
        : option (Meet_result.t is_null) :=
      match is_null1, is_null2 with
      | Not_null, Not_null => Some (Meet_result.Bottom Both_inputs)
      | Not_null, Maybe_null _ => Some (Meet_result.Bottom Left_input)
      | Maybe_null _, Not_null => Some (Meet_result.Bottom Right_input)
      | Maybe_null is_null1, Maybe_null is_null2 =>
        option_map (map_result Maybe_null)
          (meet_relation env is_null1 is_null2)
      end
    in
    let bottom_a (_ : unit)
        : or_unknown_or_bottom head_of_kind_value_non_null :=
      Bottom
    in
    let bottom_b (_ : unit) : is_null := Not_null in
    option_map
      (map_result
         (fun result : or_unknown_or_bottom head_of_kind_value_non_null
                       * is_null * variant_extensions =>
            let '(non_null, is_null, _extensions) := result in
            Head_of_kind_value non_null is_null))
      (meet_disjunction meet_a meet_b bottom_a bottom_b env non_null1
         is_null1 No_extensions non_null2 is_null2 No_extensions)
  end.

(* [meet_expanded_head0]. [None] is the assertion of the compiler that
   the two heads are of one kind. *)
Definition meet_expanded_head0 (env : Meet_env.t)
    (descr1 descr2 : Expanded_type.head)
    : option (Meet_result.t Expanded_type.t) :=
  match descr1, descr2 with
  | Expanded_type.Head_value head1, Expanded_type.Head_value head2 =>
    option_map (map_result Expanded_type.create_value)
      (meet_head_of_kind_value env head1 head2)
  | Expanded_type.Head_naked_immediate head1,
    Expanded_type.Head_naked_immediate head2 =>
    option_map (map_result Expanded_type.create_naked_immediate)
      (meet_head_of_kind_naked_immediate env head1 head2)
  | Expanded_type.Head_naked_number k1 head1,
    Expanded_type.Head_naked_number k2 head2 =>
    if naked_number_kind_eqb k1 k2
    then
      Some
        (map_result (Expanded_type.create_naked_number k1)
           (meet_head_of_kind_naked_number env head1 head2))
    else None
  | Expanded_type.Head_rec_info head1, Expanded_type.Head_rec_info head2 =>
    Some
      (map_result Expanded_type.create_rec_info
         (meet_head_of_kind_rec_info env head1 head2))
  | Expanded_type.Head_region head1, Expanded_type.Head_region head2 =>
    Some
      (map_result Expanded_type.create_region
         (meet_head_of_kind_region env head1 head2))
  | (Expanded_type.Head_value _ | Expanded_type.Head_naked_immediate _
    | Expanded_type.Head_naked_number _ _ | Expanded_type.Head_rec_info _
    | Expanded_type.Head_region _), _ => None
  end.

(* [meet_expanded_head], given that of less fuel. *)
Definition meet_expanded_head_step : meet_expanded_head :=
  fun env expanded1 expanded2 =>
    meet_or_unknown_or_bottom meet_expanded_head0 env
      (Expanded_type.descr expanded1)
      (Expanded_type.descr expanded2).

End Meet.

(* ======================================================================== *)
(* The meet                                                                 *)
(* ======================================================================== *)

(* [Meet_and_n_way_join.meet_expanded_head] and what types/meet.ml
   exposes, for a join, the choices of [meet_choices], and a number,
   the fuel. *)
Module Meet.

Section Fuel.

Variable join :
  meet_expanded_head -> Scope.t -> Meet_env.t -> Typing_env.t ->
  list Typing_env.t -> option Meet_env.t.
Variable choices : meet_choices.

(* [Meet_and_n_way_join.meet_expanded_head]: the meet of two types that
   are not aliases. [None] is a fatal error of the compiler, or the
   fuel has run out, or the one state at which the model stops and the
   compiler does not (FlambdaMeetEnv.v, "A name that is no longer
   canonical"). *)
Fixpoint meet_expanded_head (fuel : nat) : FlambdaMeetEnv.meet_expanded_head :=
  match fuel with
  | O => fun _ _ _ => None
  | S fuel =>
    meet_expanded_head_step join choices (meet_expanded_head fuel)
  end.

(* [Meet.meet], which is [Meet_and_n_way_join.meet] as it is exposed:
   bottom, or the meet of the two types and the environment with what
   the meet has added. *)
Definition meet (fuel : nat) (env : Typing_env.t) (ty1 ty2 : flambda_type)
    : option (Or_bottom.t (flambda_type * Typing_env.t)) :=
  if Typing_env.is_bottom env
  then Some Or_bottom.Bottom
  else
    let* result :=
      Meet_env.meet_type (Meet_env.create env) ty1 ty2
        (meet_expanded_head fuel)
    in
    match result with
    | Meet_result.Bottom _ => Some Or_bottom.Bottom
    | Meet_result.Ok r env =>
      let res_ty := extract_value r ty1 ty2 in
      if is_obviously_bottom res_ty
      then Some Or_bottom.Bottom
      else
        match Meet_env.final_typing_env_strict env with
        | Or_bottom.Bottom => Some Or_bottom.Bottom
        | Or_bottom.Ok env => Some (Or_bottom.Ok (res_ty, env))
        end
    end.

(* [Meet.meet_shape]: the environment in which a type also has a
   shape. *)
Definition meet_shape (fuel : nat) (env : Typing_env.t)
    (t shape : flambda_type) : option (Or_bottom.t Typing_env.t) :=
  if Typing_env.is_bottom env
  then Some Or_bottom.Bottom
  else
    let* result := meet fuel env t shape in
    match result with
    | Or_bottom.Bottom => Some Or_bottom.Bottom
    | Or_bottom.Ok (_, env) => Some (Or_bottom.Ok env)
    end.

End Fuel.

End Meet.

(* ======================================================================== *)
(* Adding to a typing environment                                           *)
(* ======================================================================== *)

(* The functions that types/flambda2_types.ml adds to [Typing_env]. *)
Module Flambda2_types.

Section Fuel.

Variable join :
  meet_expanded_head -> Scope.t -> Meet_env.t -> Typing_env.t ->
  list Typing_env.t -> option Meet_env.t.
Variable choices : meet_choices.
Variable fuel : nat.

Let meet_expanded_head : FlambdaMeetEnv.meet_expanded_head :=
  Meet.meet_expanded_head join choices fuel.

(* [Typing_env.add_equation]: the name has the type given as well as
   the type it had. *)
Definition add_equation (tenv : Typing_env.t) (name : name)
    (ty : flambda_type) : option Typing_env.t :=
  Meet_env.use_meet_env tenv
    (fun env => Meet_env.add_equation env name ty meet_expanded_head).

(* [Typing_env.add_equations_on_params]. *)
Definition add_equations_on_params (tenv : Typing_env.t)
    (params : list (variable * flambda_kind))
    (param_types : list flambda_type) : option Typing_env.t :=
  Meet_env.use_meet_env tenv
    (fun env =>
       Meet_env.add_equations_on_params env params param_types
         meet_expanded_head).

(* [Typing_env.add_env_extension]. *)
Definition add_env_extension (tenv : Typing_env.t)
    (extension : env_extension) : option Typing_env.t :=
  Meet_env.use_meet_env tenv
    (fun env => Meet_env.add_env_extension env extension meet_expanded_head).

(* A name is the result of a primitive on a scrutinee: the name gets
   the type that says so, and the scrutinee, if the name is a variable,
   a type that has the variable for that result. *)
Definition add_relation (tenv : Typing_env.t) (name : name)
    (scrutinee : simple) (name_ty : flambda_type)
    (scrutinee_head : variable -> head_of_kind_value) : option Typing_env.t :=
  Meet_env.use_meet_env tenv
    (fun env =>
       let* env := Meet_env.add_equation env name name_ty meet_expanded_head in
       match name with
       | Name_symbol _ => Some env
       | Name_var var =>
         Meet_env.add_equation_on_simple env scrutinee
           (Type_value (Ok (No_alias (scrutinee_head var))))
           meet_expanded_head
       end).

(* [Type_grammar.is_null], [is_int_for_scrutinee] and
   [get_tag_for_block], for a scrutinee that may be a constant. *)
Definition is_null_of_simple (scrutinee : simple) : flambda_type :=
  match scrutinee with
  | Const Null => this_naked_immediate 1
  | Const _ => this_naked_immediate 0
  | Var x => is_null_for_scrutinee (Name_var x)
  | Symbol s => is_null_for_scrutinee (Name_symbol s)
  end.

Definition is_int_of_simple (scrutinee : simple) : flambda_type :=
  match scrutinee with
  | Const _ => this_naked_immediate 1
  | Var x => is_int_for_scrutinee (Name_var x)
  | Symbol s => is_int_for_scrutinee (Name_symbol s)
  end.

Definition get_tag_of_simple (block : simple) : flambda_type :=
  match block with
  | Const _ => unknown (Naked_number Kind_naked_immediate)
  | Var x => get_tag_for_block (Name_var x)
  | Symbol s => get_tag_for_block (Name_symbol s)
  end.

(* [Typing_env.add_is_null_relation]. *)
Definition add_is_null_relation (tenv : Typing_env.t) (name : name)
    (scrutinee : simple) : option Typing_env.t :=
  add_relation tenv name scrutinee (is_null_of_simple scrutinee)
    (fun var => Head_of_kind_value Unknown (Maybe_null (Some var))).

(* [Typing_env.add_is_int_relation]. *)
Definition add_is_int_relation (tenv : Typing_env.t) (name : name)
    (scrutinee : simple) : option Typing_env.t :=
  add_relation tenv name scrutinee (is_int_of_simple scrutinee)
    (fun var =>
       Head_of_kind_value
         (Ok
            (Variant (Some var) Or_unknown.Unknown None Or_unknown.Unknown
               No_extensions false))
         Not_null).

(* [Typing_env.add_get_tag_relation]. *)
Definition add_get_tag_relation (tenv : Typing_env.t) (name : name)
    (scrutinee : simple) : option Typing_env.t :=
  add_relation tenv name scrutinee (get_tag_of_simple scrutinee)
    (fun var =>
       Head_of_kind_value
         (Ok
            (Variant None Or_unknown.Unknown (Some var) Or_unknown.Unknown
               No_extensions false))
         Not_null).

End Fuel.

End Flambda2_types.
