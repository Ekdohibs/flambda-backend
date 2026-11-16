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


(* When a Flambda type is well formed.

   This file is to be reviewed: the table below against the compiler,
   [type_well_formed] and the tests defined with it against the table,
   and the statements. The grammar (FlambdaTypeGrammar.v) is that of the
   compiler's private types, and so admits types that the compiler never
   builds. The operations on types rely on the difference: meet, join,
   the provers and reification read a type as their creation functions
   make it. The compiler has no one definition of what they rely on;
   this is to be it, the part for types of the one definition of
   well-formedness (FlambdaWellFormedSpec.v has the part for terms).

   It is a test, [type_well_formed : flambda_type -> bool], with one
   test for each structure of the grammar, so that it can be run on the
   types that the compiler builds. It is meant to be true of every type
   that the compiler builds, whatever the program. A type of the
   compiler that fails it is a finding: the condition is too strict, or
   the compiler is wrong. The meaning of types
   (FlambdaTypeMeaningSpec.v) does not need it: a type that is not well
   formed has a meaning all the same.

   The conditions. "Where the compiler establishes it" is one of: a
   type that cannot express what is excluded; a creation function that
   checks, or that gives it by the way it builds; a pass that stops
   when it meets what is excluded; an invariant check, which runs only
   with [-flambda2-invariants]; or nothing. The compiler is taken as it
   runs with no option that adds checks.

   1. A set of numbers in a head, and the set of strings of a [String],
      is not empty.
      Numbers: type. The heads of naked numbers are private sets, made
      only by [create], [create_set], [create_non_empty_set], [union]
      and [inter] of [Make_head_of_kind_naked_number], none of which
      gives an empty one (type_grammar.mli: "Invariant: the
      float/integer sets ... are non-empty"); a head of naked
      immediates is abstract, and [Head_of_kind_naked_immediate] gives
      bottom or stops on an empty set.
      Strings: nothing ([Head_of_kind_value_non_null.create_string]).
      Relied on by the meet of two sets ([set_meet]), which answers
      with an input, and not with bottom, when it is included in the
      other.
   2. There is no type [Type_naked_number Kind_naked_immediate].
      Type: the compiler's [Naked_immediate] is [Type_naked_immediate]
      here, and [Type_naked_number] is for the eleven other kinds.
   3. The type [immediates] of a variant is of kind naked immediate.
      Creation function: [Type_grammar.create_variant] and
      [tag_immediate] check. [Head_of_kind_value_non_null.
      create_variant], which meet and join build with: nothing.
   4. The contents of a boxed number are of the kind boxed.
      Creation function: [Type_grammar.box_float] and the eight like
      it check. The [create_boxed_float] and the like of
      [Head_of_kind_value_non_null]: nothing.
   5. A closure type of an entry of closures is of kind value.
      Creation function: [Product.Function_slot_indexed.create]
      checks, and meet builds its products with it.
   6. The product of a case of blocks has as many types as the size of
      its index if the size is exact, and no more if it is from below.
      Creation function, by the way it builds:
      [Row_like_for_blocks.create] and [create_exactly_multiple] take
      the size from the number of types, and
      [create_blocks_with_these_tags] gives no type and at least no
      field. [Row_like_for_blocks.create_raw], which meet and join
      build with: nothing (it has a comment that asks whether to check
      an invariant). Relied on by the meet of products, which makes a
      product with a type of no value empty ([meet_mapping]), and by
      [Reify], which asserts the equality for an exact size.
   7. A shape is written in one way: a mixed record has a flat suffix
      that is not empty ([block_shape_canonical]).
      Creation function: [Scannable_block_shape.from_mixed_block_shape]
      gives [Value_only] for an empty suffix.
      [Mixed_block_shape.from_prefix_size_and_suffix_elements], which
      only the parser of fexpr calls: nothing. Relied on by the meet
      of two cases of blocks, which takes two shapes that are not equal
      to have no block in common ([meet_row_like_for_blocks]).
   8. In a case of closures:
      a. the function slots of which the entry has a function type or
         a closure type are among the closures of the index, and the
         value slots of which it has a type are among its value slots.
         Creation function, by the way it builds, for the function
         types and the value slots
         ([More_type_creators.exactly_this_closure],
         [at_least_this_closure], [static_closure_with_this_code],
         [closure_with_at_least_these_function_slots] and
         [closure_with_at_least_these_value_slots] make the index out
         of the slots of the entry). For the closure types of the
         first two: nothing, it is their caller's.
         [Row_like_for_closures.create_raw]: nothing. The n-way join
         did not keep it before f489f9445e (see
         FlambdaTypeMeaningSpec.v, on products);
      b. an exact index has the function slot of the case among its
         closures.
         Pass: [Row_like_for_closures.get_single_tag] stops on a
         single case that breaks it. Where the type is made: nothing;
      c. an exact index has a function type, which may be unknown, for
         each of its closures.
         Creation function, by the way it builds
         ([exactly_this_closure]). Elsewhere nothing. Relied on by
         the provers: [Closures_entry.find_function_type] reads a
         function type that is missing under an exact index as no
         closure at all ([Provers.prove_single_closures_entry]).
   9. A map binds a key once: the cases of a row-like structure by
      tag or by function slot, the three maps of an entry of closures,
      the equations of an environment extension.
      Type: the compiler's maps cannot do otherwise. The model's are
      lists of which the first binding wins (FlambdaMapSpec.v), so a
      second binding of a key would be dead text; with the condition,
      every binding of a map counts, and two maps with the same
      lookups have the same bindings.
   10. Every type inside a well-formed type is well formed: those of a
      variant, of a boxed number, of the cases of a row-like structure
      and of the equations of an environment extension.

   A row that says nothing, or nothing for some way of building, is an
   assumption of the compiler that it does not check. There the model
   is stricter than the compiler, which it may be where a condition
   ought to be an invariant (decision D32 of the project): the
   condition is asked of every type, however built. Those places are
   marked below at the definition that asks, and the record of
   discrepancies has them (T1-5, T1-7).

   What each condition gives is stated below for 6, 7 and 8, with the
   statements of FlambdaTypeMeaningSpec.v that have the condition as a
   premise. The creation functions of FlambdaTypeGrammar.v give
   well-formed types, each under what its caller owes it; the
   statements say what that is, as an equivalence where there is
   something to say.

   Not well-formedness, and kept out:
   - Agreement of kinds where the compiler does not check it. Two
     cases. The types of a product of blocks have the kinds of the
     fields of the shape: an invariant check only
     ([Row_like_for_blocks.check_field_tys]), and
     [Simplify_variadic_primitive.simplify_make_block] gives a block
     the types of its arguments as they are. An alias to a constant is
     at the kind of the constant: nothing checks it
     ([Type_grammar.alias_type_of]), and [Expand_head] makes the head
     of such an alias from the constant alone. Code whose kinds do not
     agree exists in real programs, in branches that do not run, so a
     type of the compiler may break either; a type that does has no
     value in the first case and none in the second. Nothing may take
     them as a premise about the compiler's types.
   - That the tag of a case of blocks fits its shape (below
     [no_scan_tag], or [double_array_tag] for a record of floats).
     [Row_like_for_blocks.create] checks it. A case that breaks it has
     no block, and nothing was found to rely on it.
   - That a set lists an element once: the numbers and the strings of
     a head, the relations of a naked immediate, the slots of an
     index. A set is read by membership only.
   - That a number is in the range of its kind.
   - What the model has not got: arrays, the [rec_info] of a function
     type, the kinds of value slots, the code age relation.
   - What a type has to do with an environment (that the names it
     mentions are bound there, at which binding time): that is for the
     typing environment to say.

   The proofs are in FlambdaTypeWellFormedProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec.
From OxCamlFlambda2 Require FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec.
Import ListNotations.

(* ======================================================================== *)
(* The conditions on one structure                                          *)
(* ======================================================================== *)

(* Condition 1: a set, as the list of its elements, is not empty.
   Stricter than the compiler for the strings of a [String], which
   nothing there checks. *)
Definition non_empty {A : Type} (l : list A) : bool :=
  match l with
  | [] => false
  | _ :: _ => true
  end.

(* Condition 9: the keys of a map, in the order of its bindings. A map
   binds a key once when they are distinct ([Keyed.distinct]). *)
Definition keys {K A : Type} (bindings : Keyed.finite_map K A) : list K :=
  map fst bindings.

(* Condition 2: a kind of naked numbers whose types are made with
   [Type_naked_number]. *)
Definition has_naked_number_types (k : naked_number_kind) : bool :=
  match k with
  | Kind_naked_immediate => false
  | _ => true
  end.

(* Condition 7: [block_shape_canonical], as a test. Stricter than the
   compiler, in which a mixed shape with an empty flat suffix can be
   made (by the parser of fexpr only). *)
Definition block_shape_canonical_b (shape : block_shape) : bool :=
  match shape with
  | Scannable (Mixed_record _ flat_suffix) => non_empty flat_suffix
  | Scannable Value_only | Float_record => true
  end.

Definition block_shape_canonical_b_iff_statement : Prop :=
  ∀ shape : block_shape,
    block_shape_canonical_b shape = true <-> block_shape_canonical shape.

(* A type that is unknown, bottom or an alias is well formed; one given
   by a head is if the head is. *)
Definition descr_well_formed {head : Type} (head_well_formed : head -> bool)
    (ty : type_descr head) : bool :=
  match ty with
  | Ok (No_alias head) => head_well_formed head
  | Unknown | Bottom | Ok (Equals _) => true
  end.

(* Condition 1, for naked immediates. *)
Definition head_of_kind_naked_immediate_well_formed
    (head : head_of_kind_naked_immediate) : bool :=
  match naked_immediates head with
  | Or_unknown.Known numbers => non_empty numbers
  | Or_unknown.Unknown => true
  end.

(* Condition 6: the product of a case of blocks against the size of its
   index. Stricter than the compiler, whose raw constructor checks
   nothing. *)
Definition block_product_fits (index : row_like_index nat block_shape)
    (maps_to : list flambda_type) : bool :=
  match index_domain index with
  | Row_like_index_domain.Known size => Nat.eqb (length maps_to) size
  | Row_like_index_domain.At_least size => Nat.leb (length maps_to) size
  end.

(* Conditions 6, 7 and 9 for a case of blocks. The two arguments stand
   for the functions being defined below. *)
Definition block_case_well_formed (type_well_formed : flambda_type -> bool)
    (env_extension_well_formed : env_extension -> bool)
    (case : row_like_block_case) : bool :=
  block_product_fits (case_index case) (case_maps_to case)
  && block_shape_canonical_b (index_shape (case_index case))
  && every_b type_well_formed (case_maps_to case)
  && env_extension_well_formed (case_env_extension case).

(* Condition 8b: an exact index has the function slot of its case.
   Stricter than the compiler, which checks it only when it reads a
   single case, and then stops. *)
Definition closures_index_has_slot (slot : function_slot)
    (index : row_like_index set_of_closures_contents unit) : bool :=
  match index_domain index with
  | Row_like_index_domain.Known contents =>
    among slot (contents_closures contents)
  | Row_like_index_domain.At_least _ => true
  end.

(* Conditions 8a and 8c: the slots of which an entry has function
   types, closure types and value slot types, against the index of its
   case. Stricter than the compiler, which checks neither: its creation
   functions give 8a for the function types and the value slots and 8c
   by the way they build, and its n-way join did not keep 8a before
   f489f9445e. *)
Definition closures_entry_slots_fit
    (index : row_like_index set_of_closures_contents unit)
    (function_slots closure_slots : list function_slot)
    (value_slots : list value_slot) : bool :=
  all_among function_slots
    (contents_closures (closures_index_contents index))
  && all_among closure_slots
       (contents_closures (closures_index_contents index))
  && all_among value_slots
       (contents_value_slots (closures_index_contents index))
  && match index_domain index with
     | Row_like_index_domain.Known contents =>
       all_among (contents_closures contents) function_slots
     | Row_like_index_domain.At_least _ => true
     end.

(* ======================================================================== *)
(* A well-formed type                                                       *)
(* ======================================================================== *)

(* [type_well_formed] and the six tests defined with it, one for each
   structure of the grammar. The number of a condition is beside the
   line that asks it. A test of the cases of a map asks of the bindings
   that count, which are all of them in a map that binds a key once
   (see [Keyed.for_all_bindings_b] in FlambdaMapSpec.v, and the
   statements after the definition). Conditions 3 and 4 are
   stricter than the compiler for a head that meet or join builds: the
   creation functions of types check them, those of heads do not. *)
Fixpoint type_well_formed (t : flambda_type) {struct t} : bool :=
  match t with
  | Type_value ty => descr_well_formed head_of_kind_value_well_formed ty
  | Type_naked_immediate ty =>
    descr_well_formed head_of_kind_naked_immediate_well_formed ty (* 1 *)
  | Type_naked_number k ty =>
    has_naked_number_types k (* 2 *)
    && descr_well_formed non_empty ty (* 1 *)
  | Type_rec_info _ | Type_region _ => true
  end

with head_of_kind_value_well_formed (head : head_of_kind_value)
    {struct head} : bool :=
  match head with
  | Head_of_kind_value non_null _ =>
    match non_null with
    | Ok head => head_of_kind_value_non_null_well_formed head
    | Unknown | Bottom => true
    end
  end

with head_of_kind_value_non_null_well_formed
    (head : head_of_kind_value_non_null) {struct head} : bool :=
  match head with
  | Variant _ immediates _ blocks extensions _ =>
    match immediates with
    | Or_unknown.Known ty =>
      kind_eqb (kind ty) (Naked_number Kind_naked_immediate) (* 3 *)
      && type_well_formed ty
    | Or_unknown.Unknown => true
    end
    && match blocks with
       | Or_unknown.Known blocks => row_like_for_blocks_well_formed blocks
       | Or_unknown.Unknown => true
       end
    && match extensions with
       | No_extensions => true
       | Ext when_immediate when_block =>
         env_extension_well_formed when_immediate
         && env_extension_well_formed when_block
       end
  | Mutable_block _ => true
  | Boxed_number k contents _ =>
    kind_eqb (kind contents) (unboxed_kind k) (* 4 *)
    && type_well_formed contents
  | Closures by_function_slot _ =>
    row_like_for_closures_well_formed by_function_slot
  | String strs => non_empty strs (* 1 *)
  end

with row_like_for_blocks_well_formed (blocks : row_like_for_blocks)
    {struct blocks} : bool :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    Keyed.distinct Z.eqb (keys known_tags) (* 9 *)
    && Keyed.for_all_bindings_b Z.eqb
         (fun _ case =>
            match case with
            | Or_unknown.Known case =>
              block_case_well_formed type_well_formed
                env_extension_well_formed case (* 6, 7 *)
            | Or_unknown.Unknown => true
            end)
         known_tags
    && match other_tags with
       | Or_bottom.Ok case =>
         block_case_well_formed type_well_formed env_extension_well_formed
           case (* 6, 7 *)
       | Or_bottom.Bottom => true
       end
  end

with row_like_for_closures_well_formed (closures : row_like_for_closures)
    {struct closures} : bool :=
  match closures with
  | Row_like_for_closures known_closures =>
    distinct (keys known_closures) (* 9 *)
    && for_all_bindings_b
         (fun slot case =>
            closures_index_has_slot slot (case_index case) (* 8b *)
            && closures_entry_well_formed (case_index case)
                 (case_maps_to case)
            && env_extension_well_formed (case_env_extension case))
         known_closures
  end

(* An entry of closures, under the index of its case. *)
with closures_entry_well_formed
    (index : row_like_index set_of_closures_contents unit)
    (entry : closures_entry) {struct entry} : bool :=
  match entry with
  | Closures_entry function_types closure_types value_slot_types =>
    closures_entry_slots_fit index (domain function_types)
      (domain closure_types) (domain value_slot_types) (* 8a, 8c *)
    && distinct (keys function_types) (* 9 *)
    && distinct (keys closure_types) (* 9 *)
    && distinct (keys value_slot_types) (* 9 *)
    && for_all_bindings_b
         (fun _ ty => kind_eqb (kind ty) Value (* 5 *) && type_well_formed ty)
         closure_types
    && for_all_bindings_b (fun _ ty => type_well_formed ty) value_slot_types
  end

with env_extension_well_formed (extension : env_extension)
    {struct extension} : bool :=
  match extension with
  | Env_extension equations =>
    Keyed.distinct name_eqb (keys equations) (* 9 *)
    && Keyed.for_all_bindings_b name_eqb (fun _ ty => type_well_formed ty)
         equations
  end.

(* A case of blocks, and a case of closures at a function slot. *)
Definition row_like_block_case_well_formed (case : row_like_block_case)
    : bool :=
  block_case_well_formed type_well_formed env_extension_well_formed case.

Definition row_like_closures_case_well_formed (slot : function_slot)
    (case : row_like_closures_case) : bool :=
  closures_index_has_slot slot (case_index case)
  && closures_entry_well_formed (case_index case) (case_maps_to case)
  && env_extension_well_formed (case_env_extension case).

(* ======================================================================== *)
(* The structures that hold maps, by what the maps bind                     *)
(* ======================================================================== *)

Definition row_like_block_case_well_formed_iff_statement : Prop :=
  ∀ case : row_like_block_case,
    row_like_block_case_well_formed case = true
    <-> block_product_fits (case_index case) (case_maps_to case) = true
        ∧ block_shape_canonical (index_shape (case_index case))
        ∧ (∀ ty : flambda_type,
            ty ∈ case_maps_to case -> type_well_formed ty = true)
        ∧ env_extension_well_formed (case_env_extension case) = true.

Definition row_like_for_blocks_well_formed_iff_statement : Prop :=
  ∀ (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other_tags : Or_bottom.t row_like_block_case)
    (alloc_mode : Alloc_mode.t),
    row_like_for_blocks_well_formed
      (Row_like_for_blocks known_tags other_tags alloc_mode)
    = true
    <-> Keyed.distinct Z.eqb (keys known_tags) = true
        ∧ (∀ (tag : Z) (case : row_like_block_case),
            Keyed.lookup Z.eqb tag known_tags
            = Some (Or_unknown.Known case) ->
            row_like_block_case_well_formed case = true)
        ∧ match other_tags with
          | Or_bottom.Ok case => row_like_block_case_well_formed case = true
          | Or_bottom.Bottom => True
          end.

Definition row_like_for_closures_well_formed_iff_statement : Prop :=
  ∀ known_closures : finite_map row_like_closures_case,
    row_like_for_closures_well_formed (Row_like_for_closures known_closures)
    = true
    <-> distinct (keys known_closures) = true
        ∧ (∀ (slot : function_slot) (case : row_like_closures_case),
            lookup slot known_closures = Some case ->
            row_like_closures_case_well_formed slot case = true).

Definition closures_entry_well_formed_iff_statement : Prop :=
  ∀ (index : row_like_index set_of_closures_contents unit)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type),
    closures_entry_well_formed index
      (Closures_entry function_types closure_types value_slot_types)
    = true
    <-> closures_entry_slots_fit index (domain function_types)
          (domain closure_types) (domain value_slot_types)
        = true
        ∧ (distinct (keys function_types) = true
           ∧ distinct (keys closure_types) = true
           ∧ distinct (keys value_slot_types) = true)
        ∧ (∀ (g : function_slot) (ty : flambda_type),
            lookup g closure_types = Some ty ->
            kind ty = Value ∧ type_well_formed ty = true)
        ∧ (∀ (w : value_slot) (ty : flambda_type),
            lookup w value_slot_types = Some ty ->
            type_well_formed ty = true).

Definition env_extension_well_formed_iff_statement : Prop :=
  ∀ equations : Keyed.finite_map name flambda_type,
    env_extension_well_formed (Env_extension equations) = true
    <-> Keyed.distinct name_eqb (keys equations) = true
        ∧ (∀ (n : name) (ty : flambda_type),
            Keyed.lookup name_eqb n equations = Some ty ->
            type_well_formed ty = true).

(* ======================================================================== *)
(* The creation functions give well-formed types                            *)
(* ======================================================================== *)

(* Those that take no type give a well-formed type, whatever they are
   given. *)

Definition unknown_well_formed_statement : Prop :=
  ∀ k : flambda_kind, type_well_formed (unknown k) = true.

Definition bottom_well_formed_statement : Prop :=
  ∀ k : flambda_kind, type_well_formed (bottom k) = true.

Definition alias_type_of_well_formed_statement : Prop :=
  ∀ (k : flambda_kind) (s : simple),
    type_well_formed (alias_type_of k s) = true.

Definition type_for_const_well_formed_statement : Prop :=
  ∀ c : const, type_well_formed (type_for_const c) = true.

Definition null_well_formed_statement : Prop :=
  type_well_formed null = true.

Definition any_non_null_value_well_formed_statement : Prop :=
  type_well_formed any_non_null_value = true.

Definition mutable_block_well_formed_statement : Prop :=
  ∀ alloc_mode : Alloc_mode.t,
    type_well_formed (mutable_block alloc_mode) = true.

Definition this_immutable_string_well_formed_statement : Prop :=
  ∀ bytes : list Z, type_well_formed (this_immutable_string bytes) = true.

Definition these_naked_immediates_well_formed_statement : Prop :=
  ∀ imms : list Z, type_well_formed (these_naked_immediates imms) = true.

Definition these_tagged_immediates_well_formed_statement : Prop :=
  ∀ imms : list Z, type_well_formed (these_tagged_immediates imms) = true.

Definition any_tagged_immediate_well_formed_statement : Prop :=
  type_well_formed any_tagged_immediate = true.

Definition any_block_well_formed_statement : Prop :=
  type_well_formed any_block = true.

Definition relation_types_well_formed_statement : Prop :=
  ∀ n : name,
    type_well_formed (is_int_for_scrutinee n) = true
    ∧ type_well_formed (get_tag_for_block n) = true
    ∧ type_well_formed (is_null_for_scrutinee n) = true.

(* Those that take types, or a structure, give a well-formed type
   exactly when they are given what they expect. For conditions 3 and 4
   that is what the compiler's function checks. *)

Definition create_variant_well_formed_statement : Prop :=
  ∀ (is_unique : bool) (immediates : Or_unknown.t flambda_type)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions),
    type_well_formed (create_variant is_unique immediates blocks extensions)
    = true
    <-> match immediates with
        | Or_unknown.Known ty =>
          kind ty = Naked_number Kind_naked_immediate
          ∧ type_well_formed ty = true
        | Or_unknown.Unknown => True
        end
        ∧ match blocks with
          | Or_unknown.Known blocks =>
            row_like_for_blocks_well_formed blocks = true
          | Or_unknown.Unknown => True
          end
        ∧ match extensions with
          | No_extensions => True
          | Ext when_immediate when_block =>
            env_extension_well_formed when_immediate = true
            ∧ env_extension_well_formed when_block = true
          end.

Definition tag_immediate_well_formed_statement : Prop :=
  ∀ t : flambda_type,
    type_well_formed (tag_immediate t) = true
    <-> kind t = Naked_number Kind_naked_immediate
        ∧ type_well_formed t = true.

Definition box_number_well_formed_statement : Prop :=
  ∀ (k : boxable_number) (contents : flambda_type)
    (alloc_mode : Alloc_mode.t),
    type_well_formed (box_number k contents alloc_mode) = true
    <-> kind contents = unboxed_kind k ∧ type_well_formed contents = true.

(* The tags are a set in the compiler, and the cases a map on it. *)
Definition blocks_with_these_tags_well_formed_statement : Prop :=
  ∀ (tags : list Z) (alloc_mode : Alloc_mode.t),
    type_well_formed (blocks_with_these_tags tags alloc_mode) = true
    <-> Keyed.distinct Z.eqb tags = true.

Definition create_closures_well_formed_statement : Prop :=
  ∀ (alloc_mode : Alloc_mode.t) (by_function_slot : row_like_for_closures),
    type_well_formed (create_closures alloc_mode by_function_slot) = true
    <-> row_like_for_closures_well_formed by_function_slot = true.

(* The three ways to make blocks of one case: the caller owes the size
   of the product (the compiler's [Row_like_for_blocks.create] takes
   the size from the product), a shape written in the one way, and
   well-formed types. *)
Definition row_like_for_blocks_create_well_formed_statement : Prop :=
  ∀ (tag : Z) (size : nat) (shape : block_shape)
    (maps_to : list flambda_type) (alloc_mode : Alloc_mode.t),
    let types_well_formed :=
      block_shape_canonical shape
      ∧ ∀ ty : flambda_type, ty ∈ maps_to -> type_well_formed ty = true
    in
    (row_like_for_blocks_well_formed
       (row_like_for_blocks_create_exactly tag size shape maps_to alloc_mode)
     = true
     <-> length maps_to = size ∧ types_well_formed)
    ∧ (row_like_for_blocks_well_formed
         (row_like_for_blocks_create_at_least tag size shape maps_to
            alloc_mode)
       = true
       <-> length maps_to <= size ∧ types_well_formed)
    ∧ (row_like_for_blocks_well_formed
         (row_like_for_blocks_create_at_least_unknown_tag size shape maps_to
            alloc_mode)
       = true
       <-> length maps_to <= size ∧ types_well_formed).

Definition immutable_block_well_formed_statement : Prop :=
  ∀ (is_unique : bool) (tag : Z) (shape : block_shape)
    (alloc_mode : Alloc_mode.t) (fields : list flambda_type),
    type_well_formed (immutable_block is_unique tag shape alloc_mode fields)
    = true
    <-> block_shape_canonical shape
        ∧ (∀ ty : flambda_type, ty ∈ fields -> type_well_formed ty = true).

(* The closure at a function slot of a set. The caller owes: for the
   set that is known exactly, that the function slot is one of those
   given; three maps that bind a key once; closure types of function
   slots among those given, of kind value and well formed; well-formed
   types of value slots. *)
Definition exactly_this_closure_well_formed_statement : Prop :=
  ∀ (this_function_slot : function_slot)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (alloc_mode : Alloc_mode.t),
    type_well_formed
      (exactly_this_closure this_function_slot function_types closure_types
         value_slot_types alloc_mode)
    = true
    <-> is_bound this_function_slot function_types = true
        ∧ all_among (domain closure_types) (domain function_types) = true
        ∧ (distinct (keys function_types) = true
           ∧ distinct (keys closure_types) = true
           ∧ distinct (keys value_slot_types) = true)
        ∧ (∀ (g : function_slot) (ty : flambda_type),
            lookup g closure_types = Some ty ->
            kind ty = Value ∧ type_well_formed ty = true)
        ∧ (∀ (w : value_slot) (ty : flambda_type),
            lookup w value_slot_types = Some ty ->
            type_well_formed ty = true).

Definition at_least_this_closure_well_formed_statement : Prop :=
  ∀ (this_function_slot : function_slot)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (alloc_mode : Alloc_mode.t),
    type_well_formed
      (at_least_this_closure this_function_slot function_types closure_types
         value_slot_types alloc_mode)
    = true
    <-> all_among (domain closure_types) (domain function_types) = true
        ∧ (distinct (keys function_types) = true
           ∧ distinct (keys closure_types) = true
           ∧ distinct (keys value_slot_types) = true)
        ∧ (∀ (g : function_slot) (ty : flambda_type),
            lookup g closure_types = Some ty ->
            kind ty = Value ∧ type_well_formed ty = true)
        ∧ (∀ (w : value_slot) (ty : flambda_type),
            lookup w value_slot_types = Some ty ->
            type_well_formed ty = true).

(* ======================================================================== *)
(* What the conditions give                                                 *)
(* ======================================================================== *)

(* The four statements of FlambdaTypeMeaningSpec.v on what the
   operations rely on, for a well-formed case: the premise of each on
   the case is a condition here. *)

(* Condition 6. A well-formed case of blocks admits no block if one of
   the types of its product has no value. *)
Definition well_formed_block_case_empty_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (tag : Z) (fields : list value)
    (case : row_like_block_case) (ty : flambda_type),
    row_like_block_case_well_formed case = true ->
    ty ∈ case_maps_to case ->
    (∀ v : value, ~ type_satisfied statics vars v ty) ->
    ~ row_like_block_case_satisfied statics vars tag fields case.

(* Condition 7. Two well-formed cases of blocks that admit one block
   have the same shape. *)
Definition well_formed_block_cases_shape_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (tag : Z) (fields : list value)
    (case case' : row_like_block_case),
    row_like_block_case_well_formed case = true ->
    row_like_block_case_well_formed case' = true ->
    row_like_block_case_satisfied statics vars tag fields case ->
    row_like_block_case_satisfied statics vars tag fields case' ->
    index_shape (case_index case) = index_shape (case_index case').

(* Condition 8a. In a well-formed case of closures that admits a
   closure, a function of which the entry knows the code is in the set
   with that code, and a value slot of which the entry has a type is in
   the set with contents of that type. *)
Definition well_formed_closures_case_function_code_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (index : row_like_index set_of_closures_contents unit)
    (extension : env_extension) (g : function_slot)
    (function_type : function_type),
    let case :=
      Row_like_case
        (Closures_entry function_types closure_types value_slot_types) index
        extension
    in
    row_like_closures_case_well_formed slot case = true ->
    row_like_closures_case_satisfied statics vars v slot functions
      value_slots case ->
    lookup g function_types = Some (Or_unknown.Known function_type) ->
    lookup g functions = Some (function_type_code_id function_type).

Definition well_formed_closures_case_value_slot_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (index : row_like_index set_of_closures_contents unit)
    (extension : env_extension) (w : value_slot) (ty : flambda_type),
    let case :=
      Row_like_case
        (Closures_entry function_types closure_types value_slot_types) index
        extension
    in
    row_like_closures_case_well_formed slot case = true ->
    row_like_closures_case_satisfied statics vars v slot functions
      value_slots case ->
    lookup w value_slot_types = Some ty ->
    ∃ contents : value,
      lookup w value_slots = Some contents
      ∧ type_satisfied statics vars contents ty.

(* Condition 8a, for the closure types. In a well-formed case of
   closures that admits a closure, a function slot of which the entry
   has a closure type can be projected from the closure, and gives a
   closure of that type. This one needs what the symbols are bound to
   to be valid (FlambdaClosuresSpec.statics_valid, which every state
   of the run of a well-formed unit has). The projection from a
   symbol reads the symbol that the static set names for the slot,
   and a static set that is not valid may have the function and name
   no symbol for it: the closure types of an entry say nothing of such
   a closure, and two types whose closure types at the slot have no
   value in common both hold of it. So what is stated of an operation
   that takes a closure type with no value to leave no closure (the
   meet of two entries) has this premise on the static environment.
   The function types and the value slot types above need none: they
   are read from what [closure_in_set] answers. *)
Definition well_formed_closures_case_closure_type_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (index : row_like_index set_of_closures_contents unit)
    (extension : env_extension) (g : function_slot) (ty : flambda_type),
    let case :=
      Row_like_case
        (Closures_entry function_types closure_types value_slot_types) index
        extension
    in
    FlambdaClosuresSpec.statics_valid statics ->
    closure_in_set statics v = Some (slot, functions, value_slots) ->
    row_like_closures_case_well_formed slot case = true ->
    row_like_closures_case_satisfied statics vars v slot functions
      value_slots case ->
    lookup g closure_types = Some ty ->
    ∃ closure : value,
      project_function_slot statics slot g v = Some closure
      ∧ type_satisfied statics vars closure ty.

(* Condition 8c. In a well-formed case of closures whose index is
   exact, the function at the slot of the case has a function type. *)
Definition well_formed_exact_case_function_type_statement : Prop :=
  ∀ (slot : function_slot)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (contents : set_of_closures_contents) (extension : env_extension),
    row_like_closures_case_well_formed slot
      (Row_like_case
         (Closures_entry function_types closure_types value_slot_types)
         (Row_like_index (Row_like_index_domain.Known contents) tt)
         extension)
    = true ->
    is_bound slot function_types = true.
