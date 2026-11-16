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


(* Examples of the n-way join of types: what the traversal of
   FlambdaJoinTypes.v answers on small types.

   To be reviewed: the environment side of the examples, which is
   defined here and is not the join of environments, and each
   statement, against what
   middle_end/flambda2/types/meet_and_n_way_join.ml answers on the
   same types.

   Why they are here. What is proved of the join of types
   (FlambdaJoinTypesSpec.v) is that its answer is an upper bound. A
   function that always answered "unknown" would be one too. These
   examples say what the transcription computes, so that it can be
   compared with the compiler: the union of sets of numbers, the cases
   of variants tag by tag, the fields of blocks of one tag joined
   position by position or kept as they are, a variable made for
   [Is_int] or for [Get_tag] where the inputs disagree, the function
   types of closures, and the answers that are [None].

   What they are not. No answer here was read off a run of the
   compiler: each is what the transcription computes, and was compared
   with the compiler's code by reading. And the environment side is
   not the compiler's, so an example says nothing of what the join of
   environments does with the variables made here.

   They have since been compared with the compiler by running, by the
   workstream of the differential tests of the types library. The
   types of an example were given as the types of an argument at the
   uses of a continuation and joined by
   the compiler's n-way join of environments (the join of types cannot
   be called alone from outside the types library). No answer of the
   compiler was unexpected in the cases compared. Where no variable
   is made, the compiler answers what is written here, sets compared
   as sets. Where a variable is made for [Get_tag] or for [Is_int],
   the compiler makes one, whose type is the two numbers with the
   relation. Where one is made for the [Simple]s of an alias that is
   mentioned once, the join of environments puts the join of their
   types in its place. The last example stops the compiler at the
   [assert false] that it names, and the example of closure types of
   another kind at the fatal error of the creation function of a
   product of function slots. The example of the two mutable blocks
   was not compared as it is written, of the modes [Heap] and
   [Local]: the compiler has no function that makes the mode [Local]
   of a type, and the modes [Heap] and [Heap_or_local] were compared
   in its place. Not run: no type, types of two kinds, the bound on
   the depth.

   The environment side of the examples is a stub: the least that
   lets the traversal run on types whose only aliases are to constants
   and to variables. It is not an instance of the contracts of the
   environment side (FlambdaJoinSpec.v), so nothing that is proved of
   the join of types is said of these examples through it. It looks
   nothing up: the environments joined are empty, so the variables
   that some examples mention are bound in none of them, no scope is
   checked, no canonical element is looked for, and the join of
   [Simple]s compares the [Simple]s as they are written. To compare
   with the compiler an example that mentions a variable, the
   environments given to the compiler must bind that variable and
   have it canonical. The stub:
   - the state is a number, the next variable to make;
   - every environment joined is the empty one;
   - the join of [Simple]s that are all the same is that [Simple]; of
     others, a variable made for them (the compiler makes an
     existential variable and gives it a type in the target, the join
     of the types that the [Simple]s have, which is the business of
     the join of environments and is not shown here);
   - importing a type and joining extensions do nothing;
   - a type that is not an alias expands to itself; an alias to a
     naked or a tagged immediate expands as [Expand_head] expands it,
     whatever the kind of the alias; another alias is not expanded;
   - a head of naked immediates reduces to itself.
   The two tests of equality are not stubs: they are those of
   FlambdaTypeEqualSpec.v.

   The sets of numbers of the model are lists. The order of the
   numbers in an answer is the one that the transcription makes, and
   has no meaning.

   The proofs are by computation, in FlambdaJoinTypesExamplesProofs.v,
   which needs no review. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypeEqualSpec FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaJoinEnv
  FlambdaJoinTypes.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The environment side of the examples                                     *)
(* ======================================================================== *)

Definition example_joined_env (_ : positive) (_ : Index.t)
    : option Typing_env.t :=
  Some Typing_env.create.

Definition example_n_way_join_simples (s : positive) (_ : flambda_kind)
    (simples : list (Index.t * simple))
    : option (Or_bottom.t simple * positive) :=
  match simples with
  | [] => Some (Or_bottom.Bottom, s)
  | (_, first) :: others =>
    if forallb (fun other => simple_equal (snd other) first) others
    then Some (Or_bottom.Ok first, s)
    else Some (Or_bottom.Ok (Var s), Pos.succ s)
  end.

Definition example_import_type (s : positive) (_ : flambda_type)
    : option positive :=
  Some s.

Definition example_n_way_join_env_extension (s : positive)
    (_ : list (Index.t * env_extension))
    : option (Or_bottom.t (env_extension * positive)) :=
  Some (Or_bottom.Ok (empty_env_extension, s)).

Definition example_expand_head (_ : Typing_env.t) (ty : flambda_type)
    : option Expanded_type.t :=
  let of_descr {head : Type} (kind : flambda_kind)
      (inject : head -> Expanded_type.t) (ty : type_descr head)
      : option Expanded_type.t :=
    match ty with
    | Unknown => Some (Expanded_type.create_unknown kind)
    | Bottom => Some (Expanded_type.create_bottom kind)
    | Ok (No_alias head) => Some (inject head)
    | Ok (Equals (Const (Naked_immediate i))) =>
      Some
        (Expanded_type.create_naked_immediate
           (Head_of_kind_naked_immediate (Or_unknown.Known [i]) []))
    | Ok (Equals (Const (Tagged_immediate i))) =>
      Some
        (Expanded_type.create_value
           (Head_of_kind_value
              (Ok
                 (Variant None (Or_unknown.Known (this_naked_immediate i))
                    None (Or_unknown.Known row_like_for_blocks_bottom)
                    No_extensions false))
              Not_null))
    | Ok (Equals _) => None
    end
  in
  match ty with
  | Type_value ty => of_descr Value Expanded_type.create_value ty
  | Type_naked_immediate ty =>
    of_descr (Naked_number Kind_naked_immediate)
      Expanded_type.create_naked_immediate ty
  | Type_naked_number k ty =>
    of_descr (Naked_number k) (Expanded_type.create_naked_number k) ty
  | Type_rec_info ty => of_descr Rec_info Expanded_type.create_rec_info ty
  | Type_region ty => of_descr Region Expanded_type.create_region ty
  end.

Definition example_reduce (_ : Typing_env.t)
    (head : head_of_kind_naked_immediate)
    : Or_bottom.t head_of_kind_naked_immediate :=
  Or_bottom.Ok head.

(* The join of types of the examples, with the bound on its depth, from
   the state 1. *)
Definition example_join (fuel : nat) (ts : list flambda_type)
    : option (Or_unknown.t flambda_type * positive) :=
  n_way_join example_joined_env example_n_way_join_simples example_import_type
    example_n_way_join_env_extension example_expand_head example_reduce
    type_eqb block_shape_eqb fuel 1%positive
    (combine (seq 0 (length ts)) ts).

(* ======================================================================== *)
(* What the examples are written with                                       *)
(* ======================================================================== *)

Definition example_join_naked_immediate : flambda_kind :=
  Naked_number Kind_naked_immediate.

(* A variant with the variables given for [Is_int] and [Get_tag], the
   immediates given, and for each tag listed the blocks on the heap of
   exactly the fields given, which are values. With no variable and no
   immediate, and one tag, it is [immutable_block]. *)
Definition example_join_variant (is_int : option variable)
    (immediates : flambda_type) (get_tag : option variable)
    (blocks : list (Z * list flambda_type)) : flambda_type :=
  non_null_value
    (Variant is_int (Or_unknown.Known immediates) get_tag
       (Or_unknown.Known
          (Row_like_for_blocks
             (map
                (fun block =>
                   (fst block,
                    Or_unknown.Known
                      (Row_like_case (snd block)
                         (Row_like_index
                            (Row_like_index_domain.Known (length (snd block)))
                            (Scannable Value_only))
                         empty_env_extension)))
                blocks)
             Or_bottom.Bottom Alloc_mode.Heap))
       No_extensions false).

(* A block on the heap with the tag and the fields given. *)
Definition example_join_block (tag : Z) (fields : list flambda_type)
    : flambda_type :=
  immutable_block false tag (Scannable Value_only) Alloc_mode.Heap fields.

(* The closure at a function slot of a set of closures of that one
   function, with or without a function type for it, and one value
   slot. *)
Definition example_join_function_slot : function_slot := 1%positive.
Definition example_join_value_slot : value_slot := 1%positive.

Definition example_join_closure (function_types : list code_id)
    (value_slot_type : flambda_type) : flambda_type :=
  exactly_this_closure example_join_function_slot
    (map
       (fun code_id =>
          (example_join_function_slot,
           Or_unknown.Known (Function_type code_id)))
       function_types)
    []
    [(example_join_value_slot, value_slot_type)]
    Alloc_mode.Heap.

(* ======================================================================== *)
(* Numbers                                                                  *)
(* ======================================================================== *)

(* The join of two sets of naked immediates is their union. One level
   of calls is enough. *)
Definition example_join_naked_immediates_statement : Prop :=
  example_join 1 [these_naked_immediates [1; 2]; these_naked_immediates [2; 3]]
  = Some (Or_unknown.Known (these_naked_immediates [1; 2; 3]), 1%positive).

(* Of tagged immediates too. They are variants with no block, whose
   immediates are a type that is joined in its turn: two levels. *)
Definition example_join_tagged_immediates_statement : Prop :=
  example_join 2
    [these_tagged_immediates [1; 2]; these_tagged_immediates [3; 4]]
  = Some (Or_unknown.Known (these_tagged_immediates [3; 4; 1; 2]), 1%positive).

(* The contents of boxed floats are joined. *)
Definition example_join_boxed_floats_statement : Prop :=
  example_join 2
    [box_number Boxable_naked_float
       (Type_naked_number Kind_naked_float (Ok (No_alias [1; 2])))
       Alloc_mode.Heap;
     box_number Boxable_naked_float
       (Type_naked_number Kind_naked_float (Ok (No_alias [3])))
       Alloc_mode.Heap]
  = Some
      (Or_unknown.Known
         (box_number Boxable_naked_float
            (Type_naked_number Kind_naked_float (Ok (No_alias [1; 2; 3])))
            Alloc_mode.Heap),
       1%positive).

(* The join of two sets of strings is their union. *)
Definition example_join_strings_statement : Prop :=
  example_join 1
    [this_immutable_string [104; 105]; this_immutable_string [111]]
  = Some
      (Or_unknown.Known (non_null_value (String [[104; 105]; [111]])),
       1%positive).

(* ======================================================================== *)
(* Constants and other aliases                                              *)
(* ======================================================================== *)

(* The type of a number is an alias to it. The join of aliases is an
   alias to the join of the [Simple]s: the constant, if they are all
   that constant. *)
Definition example_join_one_constant_statement : Prop :=
  example_join 1 [this_tagged_immediate 5; this_tagged_immediate 5]
  = Some (Or_unknown.Known (this_tagged_immediate 5), 1%positive).

(* If they are two constants, a variable is made, and the state moves.
   (In the compiler the variable is existential, and its type in the
   target is the join of the types of the constants.) *)
Definition example_join_two_constants_statement : Prop :=
  example_join 1 [this_tagged_immediate 5; this_tagged_immediate 6]
  = Some (Or_unknown.Known (alias_type_of Value (Var 1%positive)), 2%positive).

(* An alias beside a type that is not one is expanded, and the heads
   are joined. *)
Definition example_join_constant_and_set_statement : Prop :=
  example_join 2 [this_tagged_immediate 5; these_tagged_immediates [1; 2]]
  = Some (Or_unknown.Known (these_tagged_immediates [1; 2; 5]), 1%positive).

(* ======================================================================== *)
(* Unknown, bottom, and heads of different forms                            *)
(* ======================================================================== *)

(* An unknown input makes the join unknown: the answer is the unknown
   type of the kind, as a known answer. *)
Definition example_join_unknown_statement : Prop :=
  example_join 1
    [these_naked_immediates [1; 2]; unknown example_join_naked_immediate]
  = Some
      (Or_unknown.Known (unknown example_join_naked_immediate), 1%positive).

(* A bottom input is left out. *)
Definition example_join_bottom_statement : Prop :=
  example_join 1
    [these_naked_immediates [1; 2]; bottom example_join_naked_immediate]
  = Some (Or_unknown.Known (these_naked_immediates [1; 2]), 1%positive).

(* If every input is bottom, so is the join. *)
Definition example_join_all_bottom_statement : Prop :=
  example_join 1 [bottom Value; bottom Value]
  = Some (Or_unknown.Known (bottom Value), 1%positive).

(* A string and a tagged immediate: the heads are not of one form, and
   all that is kept is that the value is not null. *)
Definition example_join_string_and_immediate_statement : Prop :=
  example_join 2 [this_immutable_string [104; 105]; any_tagged_immediate]
  = Some (Or_unknown.Known any_non_null_value, 1%positive).

(* Null and a tagged immediate: a tagged immediate that may be null,
   with no variable for [Is_null]. *)
Definition example_join_null_and_immediate_statement : Prop :=
  example_join 2 [null; any_tagged_immediate]
  = Some
      (Or_unknown.Known
         (Type_value
            (Ok
               (No_alias
                  (Head_of_kind_value
                     (Ok
                        (Variant None Or_unknown.Unknown None
                           (Or_unknown.Known row_like_for_blocks_bottom)
                           No_extensions false))
                     (Maybe_null None))))),
       1%positive).

(* Two mutable blocks: the allocation modes are joined. *)
Definition example_join_mutable_blocks_statement : Prop :=
  example_join 1
    [mutable_block Alloc_mode.Heap; mutable_block Alloc_mode.Local]
  = Some
      (Or_unknown.Known (mutable_block Alloc_mode.Heap_or_local), 1%positive).

(* ======================================================================== *)
(* Blocks and variants                                                      *)
(* ======================================================================== *)

(* Blocks of one tag and one size: the fields are joined position by
   position. *)
Definition example_join_fields_statement : Prop :=
  example_join 3
    [example_join_block 0 [these_tagged_immediates [1; 2]];
     example_join_block 0 [these_tagged_immediates [3; 4]]]
  = Some
      (Or_unknown.Known
         (example_join_block 0 [these_tagged_immediates [1; 2; 3; 4]]),
       1%positive).

(* If the fields are the same in every input (here aliases to one
   variable), they are kept as they are and nothing is joined. *)
Definition example_join_same_fields_statement : Prop :=
  example_join 2
    [example_join_block 0 [alias_type_of Value (Var 7%positive)];
     example_join_block 0 [alias_type_of Value (Var 7%positive)]]
  = Some
      (Or_unknown.Known
         (example_join_block 0 [alias_type_of Value (Var 7%positive)]),
       1%positive).

(* If they are aliases to two variables, the field is an alias to a
   variable made for them. *)
Definition example_join_fields_of_two_variables_statement : Prop :=
  example_join 3
    [example_join_block 0 [alias_type_of Value (Var 7%positive)];
     example_join_block 0 [alias_type_of Value (Var 8%positive)]]
  = Some
      (Or_unknown.Known
         (example_join_block 0 [alias_type_of Value (Var 1%positive)]),
       2%positive).

(* Blocks of one tag and two sizes: blocks of that tag with at least
   as many fields as the smaller, those fields joined. *)
Definition example_join_two_sizes_statement : Prop :=
  example_join 3
    [example_join_block 0 [these_tagged_immediates [1; 2]];
     example_join_block 0
       [these_tagged_immediates [3; 4]; any_tagged_immediate]]
  = Some
      (Or_unknown.Known
         (create_variant false
            (Or_unknown.Known (bottom example_join_naked_immediate))
            (Or_unknown.Known
               (row_like_for_blocks_create_at_least 0 1
                  (Scannable Value_only)
                  [these_tagged_immediates [1; 2; 3; 4]] Alloc_mode.Heap))
            No_extensions),
       1%positive).

(* Blocks of one tag and two shapes (two fields that are values; a
   value and then a naked float): nothing is known of the blocks of
   that tag. *)
Definition example_join_two_shapes_statement : Prop :=
  example_join 2
    [example_join_block 0 [any_tagged_immediate; any_tagged_immediate];
     immutable_block false 0 (Scannable (Mixed_record 1 [Kind_naked_float]))
       Alloc_mode.Heap
       [any_tagged_immediate; unknown (Naked_number Kind_naked_float)]]
  = Some
      (Or_unknown.Known (blocks_with_these_tags [0] Alloc_mode.Heap),
       1%positive).

(* Blocks of two tags: a variant with the two cases, each as it was.
   The tag is the constant 0 in one input and 1 in the other, so a
   variable is made for [Get_tag]. (In the compiler its type in the
   target is the join of the two constants.) *)
Definition example_join_two_tags_statement : Prop :=
  example_join 2
    [example_join_block 0 [any_tagged_immediate];
     example_join_block 1
       [these_tagged_immediates [1; 2]; any_tagged_immediate]]
  = Some
      (Or_unknown.Known
         (example_join_variant None (bottom example_join_naked_immediate)
            (Some 1%positive)
            [(0, [any_tagged_immediate]);
             (1, [these_tagged_immediates [1; 2]; any_tagged_immediate])]),
       2%positive).

(* A constant constructor and a block, as [None] and [Some _] are: a
   variant with the immediate and the block. [Is_int] is the constant
   1 in one input and 0 in the other, so a variable is made for it;
   the only tag is 0, a constant, and no variable is made for
   [Get_tag]. *)
Definition example_join_immediate_and_block_statement : Prop :=
  example_join 2
    [this_tagged_immediate 0; example_join_block 0 [any_tagged_immediate]]
  = Some
      (Or_unknown.Known
         (example_join_variant (Some 1%positive) (this_naked_immediate 0) None
            [(0, [any_tagged_immediate])]),
       2%positive).

(* Two variants with immediates and blocks: the immediates are joined,
   the case of a tag that both have is joined field by field, and the
   case of a tag that one has is kept. Neither [Is_int] nor [Get_tag]
   is a constant in the first input, which has no variable for them:
   none is made. *)
Definition example_join_variants_statement : Prop :=
  example_join 4
    [example_join_variant None (this_naked_immediate 0) None
       [(0, [these_tagged_immediates [1; 2]]);
        (1, [these_tagged_immediates [3; 4]; any_tagged_immediate])];
     example_join_variant None (these_naked_immediates [0; 1]) None
       [(1, [these_tagged_immediates [5; 6]; any_tagged_immediate])]]
  = Some
      (Or_unknown.Known
         (example_join_variant None (these_naked_immediates [0; 1]) None
            [(0, [these_tagged_immediates [1; 2]]);
             (1,
              [these_tagged_immediates [3; 4; 5; 6]; any_tagged_immediate])]),
       1%positive).

(* ======================================================================== *)
(* Closures                                                                 *)
(* ======================================================================== *)

(* Two closures of one set of closures and one code: the function type
   is kept and the types of the value slot are joined. *)
Definition example_join_closures_statement : Prop :=
  example_join 4
    [example_join_closure [1%positive] (these_tagged_immediates [1; 2]);
     example_join_closure [1%positive] (these_tagged_immediates [3; 4])]
  = Some
      (Or_unknown.Known
         (example_join_closure [1%positive]
            (these_tagged_immediates [3; 4; 1; 2])),
       1%positive).

(* Of two codes: the function type of the slot is unknown. *)
Definition example_join_closures_of_two_codes_statement : Prop :=
  example_join 4
    [example_join_closure [1%positive] (these_tagged_immediates [1; 2]);
     example_join_closure [2%positive] (these_tagged_immediates [3; 4])]
  = Some
      (Or_unknown.Known
         (exactly_this_closure example_join_function_slot
            [(example_join_function_slot, Or_unknown.Unknown)] []
            [(example_join_value_slot, these_tagged_immediates [3; 4; 1; 2])]
            Alloc_mode.Heap),
       1%positive).

(* One input has a function type for the slot and the other, the
   closures of at least that value slot, has none: the join has none.
   (A function type kept from the one input that has it would not be
   an upper bound: discrepancy T1-4.) The sets of closures are of
   different contents, and the join is of the sets that have at least
   what both have. *)
Definition example_join_closures_one_without_function_type_statement : Prop :=
  example_join 4
    [example_join_closure [1%positive] (these_tagged_immediates [1; 2]);
     at_least_this_closure example_join_function_slot [] []
       [(example_join_value_slot, these_tagged_immediates [3; 4])]
       Alloc_mode.Heap]
  = Some
      (Or_unknown.Known
         (at_least_this_closure example_join_function_slot [] []
            [(example_join_value_slot, these_tagged_immediates [3; 4; 1; 2])]
            Alloc_mode.Heap),
       1%positive).

(* ======================================================================== *)
(* The answers that are [None]                                              *)
(* ======================================================================== *)

(* No type; types of two kinds: the two fatal errors of [n_way_join]. *)
Definition example_join_no_type_statement : Prop :=
  example_join 1 [] = None.

Definition example_join_two_kinds_statement : Prop :=
  example_join 1 [these_naked_immediates [1; 2]; any_tagged_immediate] = None.

(* The bound on the depth of the calls is reached: the compiler has no
   such bound. *)
Definition example_join_bound_reached_statement : Prop :=
  example_join 1
    [these_tagged_immediates [1; 2]; these_tagged_immediates [3; 4]]
  = None.

(* The closures of a set have types, slot by slot, which are of kind
   value: the compiler's creation function of the product stops on
   another kind ([Product.Function_slot_indexed.create]), where a type
   is made and where the join makes the product of the joined types.
   Here the two inputs are of kind value, and the compiler can make
   them: the type that the first has for the function slot is an alias
   of kind value to a naked immediate, and the second has the unknown
   type. The alias expands to a head of naked immediates, the join of
   it and of an unknown type is the unknown type of naked immediates,
   and the product of the joined types is refused. *)
Definition example_join_closure_types_of_another_kind_statement : Prop :=
  example_join 3
    [exactly_this_closure example_join_function_slot
       [(example_join_function_slot,
         Or_unknown.Known (Function_type 1%positive))]
       [(example_join_function_slot,
         alias_type_of Value (Const (Naked_immediate 1)))]
       [] Alloc_mode.Heap;
     exactly_this_closure example_join_function_slot
       [(example_join_function_slot,
         Or_unknown.Known (Function_type 1%positive))]
       [(example_join_function_slot, unknown Value)] [] Alloc_mode.Heap]
  = None.

(* An alias of kind value to a naked immediate, beside a type of kind
   value. The kinds of the types as they are written agree; the alias
   expands to a head of naked immediates, and the heads are then not of
   one kind: the [assert false] of [n_way_join_expanded_head]
   (discrepancy T5b-3). The statement that the join is an upper bound
   excludes this input: such an alias is not kinded. *)
Definition example_join_alias_of_another_kind_statement : Prop :=
  example_join 2
    [any_tagged_immediate; Type_value (Ok (Equals (Const (Naked_immediate 1))))]
  = None.
