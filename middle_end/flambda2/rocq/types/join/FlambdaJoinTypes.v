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


(* The n-way join of types: the functions of
   middle_end/flambda2/types/meet_and_n_way_join.ml that join.

   This file is to be reviewed line by line against that file. It holds
   definitions only. What they are proved to guarantee is
   FlambdaJoinTypesSpec.v, against the interface with the join of
   environments that FlambdaJoinSpec.v fixes.

   The compiler that is transcribed is the one with the change that
   repairs [n_way_join_closures_entry]: the merge of the function types
   answers [None] unless both sides have a function type. The
   integration branch has taken that change; a tree that has not it yet
   has, in that merge, the cases [None, Some val2] and [Some _, None]
   that [gather_function_types_before_fix] has here. Lines are those of
   the file before the change, which moves nothing above line 2920.

   Every join function of that file is here, and the helpers that they
   call; what the model has of each is said below.

   How the compiler's functions are written here.
   - The state of the join of environments ([Join_env.t]) is any type:
     the functions are in a section whose variables are that type, the
     four functions of join_env.ml that the traversal calls
     ([Join_env.joined_env], [n_way_join_simples], [import_type] and
     [n_way_join_env_extension], the last under the name
     [join_env_n_way_join_env_extension], this file having a function
     of the same name as the compiler has), and four functions of
     other files: [Expand_head.expand_head],
     [reduce_head_of_kind_naked_immediate] (it is in
     meet_and_n_way_join.ml, with the meet), the test of structural
     equality of types that the compiler makes under
     -flambda2-join-sharing structural ([Type_grammar.equal]), and
     [Flambda_kind.Block_shape.equal].
   - A function that joins answers a ['a n_way_join_result], a pair of
     an ['a Or_unknown.t] and the state; here it answers an option of
     that, [None] being a fatal error of the compiler, an
     [assert false] or an exception that nothing catches. The header of
     each function says which. An exception that the compiler raises to
     leave a fold and catches a few lines below ([Unknown_result],
     [Cannot_track], [No_extensions], [Not_found]) is an inner [None]
     of a function that gathers a list, and is said so there.
   - The join of types calls itself through the functions that join
     the parts of types, and it is not a recursion on a type: an alias
     is expanded to a type of an environment. So the functions that
     call it are given it as [n_way_join_rec], and [n_way_join] is
     defined after them by recursion on a number that bounds the depth
     of its calls to itself; [None] is also that number run out. Every
     cycle among the compiler's functions goes through [n_way_join],
     so the others are plain definitions, but for
     [n_way_join_expanded_head], which is a recursion on its list.
   - A reference that the compiler threads through a [filter_map] or
     an [Array.init] is the state threaded through a fold. A map of the
     model is gone through in the order of its list, a map of the
     compiler in the order of its keys; no statement depends on the
     order.
   - [Join_env.machine_width] is not there: a number of the model is an
     integer whatever the machine (FlambdaModel.v).
   - A set of numbers or of strings is a list (FlambdaTypeGrammar.v);
     the union of two is their concatenation without repeats.

   What is here, in the order of the compiler's file.
   - [map_join_result] (line 49), [join_alloc_mode] (768),
     [n_way_join_unknown] (790),
     [n_way_join_head_of_kind_naked_number] (803), [deduce_is_int] and
     [deduce_is_int_simple] (867), [deduce_get_tag] and
     [deduce_get_tag_simple] (880), [n_way_join_relation_simples]
     (903): in full.
   - [generic_merge_map_known] (807): in full. Its first fold, which
     gathers the cases of each key, is [gather_known]; the
     [filter_map] with a reference is [Keyed.filter_map_fold]
     (FlambdaMapSpec.v).
   - [join_array_element_kinds] (936) is left out: the grammar has no
     arrays.
   - [n_way_join] (1885): [n_way_join_body], and [n_way_join] with the
     number that bounds its depth. The compiler stops on no type and
     on types of two kinds, whatever its flags.
   - [n_way_join_expanded_head] (1915): in full. The eleven cases of
     the naked numbers other than immediates are one, as in the
     grammar: two heads of naked numbers of different kinds are heads
     of different kinds.
   - [n_way_join_head_of_kind_value] (2108): in full. Its first
     binding, which joins the variables that say whether the value is
     null, is [n_way_join_is_null]. The compiler stops if the join of
     those [Simple]s answers a symbol.
   - [n_way_join_head_of_kind_value_non_null] (2175): the cases
     [Variant], [Mutable_block], [Closures] and [String] in full; the
     nine cases of the boxed numbers are one, as in the grammar; the
     case [Array] is left out with the arrays. The folds over the heads
     that follow the first are [gather_heads]. If the join of the
     blocks, immediates and extensions of variants is unknown, the
     relations [Is_int] and [Get_tag] that were joined before it are
     dropped, as in the compiler.
   - [n_way_join_array_contents] (2495) is left out with the arrays.
   - [n_way_join_variant] (2550): in full.
   - [n_way_join_head_of_kind_naked_immediate] (2588): in full. The
     inverse relations of the heads are dropped, as in the compiler;
     the join of environments rebuilds them. Each head is first reduced
     in the environment of its index. If every head reduces to bottom
     the compiler stops, in [create_naked_immediates_non_empty], and
     the answer here is [None].
   - [n_way_join_head_of_kind_naked_numbers]: the eleven functions
     [n_way_join_head_of_kind_naked_float32] to
     [n_way_join_head_of_kind_naked_mask] (2619 to 2661), which differ
     only in the kind.
   - [n_way_join_head_of_kind_rec_info] (2663): the head of the model
     has no content (FlambdaTypeGrammar.v), so the answer is always
     known. The compiler answers unknown if the expressions differ. So
     the compiler's answer may be unknown where the model's is known,
     never the reverse.
   - [n_way_join_head_of_kind_region] (2668): in full.
   - [n_way_join_row_like] (2678): in full, in a section whose
     variables are what its two callers give it; its local functions
     are [n_way_join_index], [n_way_join_case] and
     [n_way_join_knowns].
   - [n_way_join_row_like_for_blocks] (2806) and
     [n_way_join_row_like_for_closures] (2851): in full. The second
     stops, in the compiler, on a case for the other function slots and
     on an unknown join.
   - [n_way_join_closures_entry] (2895): in full, with the change (see
     above) in its function types.
   - [n_way_join_generic_product] (2955),
     [n_way_join_function_slot_indexed_product] (3010) and
     [n_way_join_value_slot_indexed_product] (3023): in full. The first
     fold of the first is [gather_components]. The second stops, in
     the compiler, on a component that is not of kind value: the
     product is made with [Product.Function_slot_indexed.create]
     (types/grammar/type_grammar.ml), which tests it whatever the
     flags.
   - [n_way_join_int_indexed_product] (3036): in full, with the test of
     -flambda2-join-sharing structural where the compiler tests whether
     two fields are one value in memory (decision D30). The compiler
     stops on no array of fields.
   - [n_way_join_env_extension] (3133): in full. The compiler's
     function gives the environment's the join of types and the meet;
     the environment's uses neither (FlambdaJoinSpec.v).
   - [n_way_join_function_type] (3098): a function type of the model
     has no [rec_info] (FlambdaTypeGrammar.v), so the join of the
     [rec_info]s is not there, nor the state, which only that join
     threads. A consequence for the differential tests: where every
     function type is known and all name one code, the model answers
     that code; the compiler answers unknown instead if the join of the
     [rec_info]s is unknown (their heads differ). So the compiler's
     answer may be unknown where the model's is known, never the
     reverse, and never a different code. With invariant checks the
     compiler stops on an empty list; without them it answers unknown,
     as here.
   - [n_way_join_function_types]: of [n_way_join_closures_entry], the
     part that joins the function types of the entries. A function slot
     is kept only if every entry has a function type for it, as
     [n_way_join_generic_product] does for the closures and the value
     slots, and as the binary join does ([join_closures_entry] in
     meet_and_join.ml).
   - [n_way_join_function_types_before_fix]: the same part as it was
     before the change. For each function slot that some entry has a
     function type for, it joins the function types of the entries that
     have one. An entry that has none for the slot is left out of that
     join: it is not taken to say that the code is unknown. It is not
     an upper bound (FlambdaJoinSpec.v), and is kept only as the record
     of that; nothing else is to use it.

   What was found of the compiler here: the join of types has no bound
   on its depth and no test for a cycle that is always reached
   (discrepancy T5b-1); the join of what heads say of null gives a
   variable to a head that is not null (T5b-2).

   The lists of contributors are in the order in which the compiler
   makes them (a fold from the left that conses reverses its list);
   no statement depends on the order.

   [Expanded_type], the type of expanded types and its creation
   functions, is that of the file of [Expand_head]
   (FlambdaExpandHead.v). It is given a name in this file too, where
   this file had a copy of it before that file could be imported, so
   that what is written against this file reads as it did. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningBoolSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypingEnv FlambdaJoinEnv.
From OxCamlFlambda2Types Require FlambdaExpandHead.
Import ListNotations.

(* ======================================================================== *)
(* Expanded types                                                           *)
(* ======================================================================== *)

(* [Expand_head.Expanded_type]: a type that is not an alias, given by
   its kind and its head. The module of that name of
   FlambdaExpandHead.v. *)
Module Expanded_type := FlambdaExpandHead.Expanded_type.

(* ======================================================================== *)
(* Function types                                                           *)
(* ======================================================================== *)

(* The function types of a list, if none is unknown. *)
Fixpoint known_function_types
    (func_types : list (Index.t * Or_unknown.t function_type))
    : option (list function_type) :=
  match func_types with
  | [] => Some []
  | (_, Or_unknown.Unknown) :: _ => None
  | (_, Or_unknown.Known func_type) :: func_types =>
    match known_function_types func_types with
    | None => None
    | Some known => Some (func_type :: known)
    end
  end.

(* [n_way_join_function_type]: the code is known if every function type
   is known and all name the same code. *)
Definition n_way_join_function_type
    (func_types : list (Index.t * Or_unknown.t function_type))
    : Or_unknown.t function_type :=
  match known_function_types func_types with
  | None | Some [] => Or_unknown.Unknown
  | Some (func_type :: func_types) =>
    if forallb
         (fun func_type' =>
            Pos.eqb (function_type_code_id func_type)
              (function_type_code_id func_type'))
         func_types
    then Or_unknown.Known func_type
    else Or_unknown.Unknown
  end.

(* The function types of the entries, slot by slot, as
   [n_way_join_closures_entry] gathers them: a slot is kept only if
   every entry has a function type for it. *)
Definition gather_function_types
    (entries : list (Index.t * finite_map (Or_unknown.t function_type)))
    : finite_map (list (Index.t * Or_unknown.t function_type)) :=
  match entries with
  | [] => []
  | (id1, function_types1) :: entries =>
    fold_left
      (fun function_types entry =>
         merge
           (fun vals_opt val2_opt =>
              match vals_opt, val2_opt with
              | None, _ | _, None => None
              | Some vals, Some val2 => Some ((fst entry, val2) :: vals)
              end)
           function_types (snd entry))
      entries
      (map_values (fun func_type => [(id1, func_type)]) function_types1)
  end.

Definition n_way_join_function_types
    (entries : list (Index.t * finite_map (Or_unknown.t function_type)))
    : finite_map (Or_unknown.t function_type) :=
  map_values n_way_join_function_type (gather_function_types entries).

(* The same before the change: the entries that have no function type
   for a slot are not among its contributors, and the slot is kept. *)
Definition gather_function_types_before_fix
    (entries : list (Index.t * finite_map (Or_unknown.t function_type)))
    : finite_map (list (Index.t * Or_unknown.t function_type)) :=
  match entries with
  | [] => []
  | (id1, function_types1) :: entries =>
    fold_left
      (fun function_types entry =>
         merge
           (fun vals_opt val2_opt =>
              match vals_opt, val2_opt with
              | None, None => None
              | None, Some val2 => Some [(fst entry, val2)]
              | Some vals, None => Some vals
              | Some vals, Some val2 => Some ((fst entry, val2) :: vals)
              end)
           function_types (snd entry))
      entries
      (map_values (fun func_type => [(id1, func_type)]) function_types1)
  end.

Definition n_way_join_function_types_before_fix
    (entries : list (Index.t * finite_map (Or_unknown.t function_type)))
    : finite_map (Or_unknown.t function_type) :=
  map_values n_way_join_function_type
    (gather_function_types_before_fix entries).

(* ======================================================================== *)
(* What the traversal is made with                                          *)
(* ======================================================================== *)

(* [Or_unknown.map]. *)
Definition or_unknown_map {A B : Type} (f : A -> B) (a : Or_unknown.t A)
    : Or_unknown.t B :=
  match a with
  | Or_unknown.Known a => Or_unknown.Known (f a)
  | Or_unknown.Unknown => Or_unknown.Unknown
  end.

(* [join_alloc_mode]. *)
Definition join_alloc_mode (alloc_mode1 alloc_mode2 : Alloc_mode.t)
    : Alloc_mode.t :=
  match alloc_mode1, alloc_mode2 with
  | (Alloc_mode.Heap_or_local | Alloc_mode.Local), _
  | _, (Alloc_mode.Heap_or_local | Alloc_mode.Local) =>
    Alloc_mode.Heap_or_local
  | Alloc_mode.Heap, Alloc_mode.Heap => Alloc_mode.Heap
  end.

(* The union of two sets of numbers. *)
Definition numbers_union (numbers numbers' : list Z) : list Z :=
  Keyed.dedup Z.eqb (numbers ++ numbers').

(* [Simple.untagged_const_bool] and [Simple.untagged_const_int] of a
   tag. *)
Definition untagged_const_bool (b : bool) : simple :=
  Const (Naked_immediate (if b then 1%Z else 0%Z)).

Definition untagged_const_int (i : Z) : simple := Const (Naked_immediate i).

(* [Type_grammar.Row_like_for_blocks.all_tags]. *)
Definition row_like_for_blocks_all_tags (blocks : row_like_for_blocks)
    : Or_unknown.t (list Z) :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    match other_tags with
    | Or_bottom.Ok _ => Or_unknown.Unknown
    | Or_bottom.Bottom => Or_unknown.Known (Keyed.domain Z.eqb known_tags)
    end
  end.

(* [deduce_is_int]: whether a variant is an immediate, where its blocks
   or its immediates show that it has none of the one sort. *)
Definition deduce_is_int (blocks : Or_unknown.t row_like_for_blocks)
    (imms : Or_unknown.t flambda_type) : Or_unknown.t bool :=
  let no_blocks :=
    match blocks with
    | Or_unknown.Known blocks => row_like_for_blocks_is_bottom blocks
    | Or_unknown.Unknown => false
    end
  in
  let no_immediates :=
    match imms with
    | Or_unknown.Known imms => is_obviously_bottom imms
    | Or_unknown.Unknown => false
    end
  in
  if no_blocks
  then Or_unknown.Known true
  else if no_immediates then Or_unknown.Known false else Or_unknown.Unknown.

(* [deduce_is_int_simple]. *)
Definition deduce_is_int_simple (blocks : Or_unknown.t row_like_for_blocks)
    (imms : Or_unknown.t flambda_type) (is_int_var : option variable)
    : option simple :=
  match deduce_is_int blocks imms, is_int_var with
  | Or_unknown.Known is_int, _ => Some (untagged_const_bool is_int)
  | Or_unknown.Unknown, Some var => Some (Var var)
  | Or_unknown.Unknown, None => None
  end.

(* [deduce_get_tag]: the tag of the blocks of a variant, if they all
   have one tag; bottom if there is no block. *)
Definition deduce_get_tag (blocks : Or_unknown.t row_like_for_blocks)
    : or_unknown_or_bottom Z :=
  match blocks with
  | Or_unknown.Unknown => Unknown
  | Or_unknown.Known blocks =>
    match row_like_for_blocks_all_tags blocks with
    | Or_unknown.Unknown => Unknown
    | Or_unknown.Known all_tags =>
      if Keyed.is_empty all_tags
      then Bottom
      else
        match Keyed.get_singleton Z.eqb all_tags with
        | Some tag => Ok tag
        | None => Unknown
        end
    end
  end.

(* [deduce_get_tag_simple]. *)
Definition deduce_get_tag_simple (blocks : Or_unknown.t row_like_for_blocks)
    (get_tag_var : option variable) : or_unknown_or_bottom simple :=
  match deduce_get_tag blocks, get_tag_var with
  | Ok get_tag, _ => Ok (untagged_const_int get_tag)
  | Bottom, _ => Bottom
  | Unknown, Some var => Ok (Var var)
  | Unknown, None => Unknown
  end.

(* The contents of a list, if none is unknown, the last first: the fold
   of [n_way_join_unknown]. [None] is its exception [Unknown_result]. *)
Fixpoint known_contents {A : Type} (acc : list (Index.t * A))
    (or_unknowns : list (Index.t * Or_unknown.t A))
    : option (list (Index.t * A)) :=
  match or_unknowns with
  | [] => Some acc
  | (id, Or_unknown.Known contents) :: or_unknowns =>
    known_contents ((id, contents) :: acc) or_unknowns
  | (_, Or_unknown.Unknown) :: _ => None
  end.

(* The [Simple]s that say whether the values of each head are null: the
   [List.map] of [n_way_join_head_of_kind_value]. [None] is its
   exception [Cannot_track]: a head that may be null has no variable
   that says whether it is. *)
Fixpoint is_null_simples (heads : list (Index.t * head_of_kind_value))
    : option (list (Index.t * simple)) :=
  match heads with
  | [] => Some []
  | (id, Head_of_kind_value _ is_null) :: heads =>
    match is_null with
    | Not_null =>
      option_map (cons (id, untagged_const_bool false)) (is_null_simples heads)
    | Maybe_null None => None
    | Maybe_null (Some is_null) =>
      option_map (cons (id, Var is_null)) (is_null_simples heads)
    end
  end.

(* The union of two sets of strings. *)
Definition strings_union (strs strs' : list (list Z)) : list (list Z) :=
  Keyed.dedup (lists_eqb Z.eqb) (strs ++ strs').

(* [More_type_creators.unknown_from_shape]. *)
Definition unknown_from_shape (shape : block_shape) (index : nat)
    : option flambda_type :=
  option_map unknown (block_shape_element_kind shape index).

(* [Type_grammar.Env_extension.is_empty]. *)
Definition env_extension_is_empty (extension : env_extension) : bool :=
  match extension with
  | Env_extension equations => Keyed.is_empty equations
  end.

(* The fields of a [closures_entry], of a [row_like_for_blocks] and of
   a [row_like_for_closures]. *)
Definition closures_entry_function_types (entry : closures_entry)
    : finite_map (Or_unknown.t function_type) :=
  match entry with
  | Closures_entry function_types _ _ => function_types
  end.

Definition closures_entry_closure_types (entry : closures_entry)
    : finite_map flambda_type :=
  match entry with
  | Closures_entry _ closure_types _ => closure_types
  end.

Definition closures_entry_value_slot_types (entry : closures_entry)
    : finite_map flambda_type :=
  match entry with
  | Closures_entry _ _ value_slot_types => value_slot_types
  end.

Definition row_like_known_tags (blocks : row_like_for_blocks)
    : Keyed.finite_map Z (Or_unknown.t row_like_block_case) :=
  match blocks with
  | Row_like_for_blocks known_tags _ _ => known_tags
  end.

Definition row_like_other_tags (blocks : row_like_for_blocks)
    : Or_bottom.t row_like_block_case :=
  match blocks with
  | Row_like_for_blocks _ other_tags _ => other_tags
  end.

Definition row_like_alloc_mode (blocks : row_like_for_blocks) : Alloc_mode.t :=
  match blocks with
  | Row_like_for_blocks _ _ alloc_mode => alloc_mode
  end.

Definition row_like_known_closures (closures : row_like_for_closures)
    : finite_map row_like_closures_case :=
  match closures with
  | Row_like_for_closures known_closures => known_closures
  end.

(* The components of the products given, index by index, as
   [n_way_join_generic_product] gathers them: an index is kept only if
   every product has a component for it. Each list has the last
   product's component first. *)
Definition gather_components {A : Type}
    (indexed_products : list (Index.t * finite_map A))
    : finite_map (list (Index.t * A)) :=
  match indexed_products with
  | [] => []
  | (id1, indexed_product1) :: indexed_products =>
    fold_left
      (fun components_by_index indexed_product2 =>
         merge
           (fun vals_opt val2_opt =>
              match vals_opt, val2_opt with
              | None, _ | _, None => None
              | Some vals, Some val2 =>
                Some ((fst indexed_product2, val2) :: vals)
              end)
           components_by_index (snd indexed_product2))
      indexed_products
      (map_values (fun val1 => [(id1, val1)]) indexed_product1)
  end.

(* The cases of the row-like structures given, key by key, as
   [generic_merge_map_known] gathers them. For a key that some
   structure has a case for, each structure gives its case for the key,
   or, if it has none, its case for the other keys, if it has one. *)
Section Merge_map_known.

Context {K value other : Type} (eqb : K -> K -> bool)
  (of_other : other -> value).

(* [add_other]. *)
Definition add_other (acc : list (Index.t * value)) (id : Index.t)
    (other_or_bottom : Or_bottom.t other) : list (Index.t * value) :=
  match other_or_bottom with
  | Or_bottom.Bottom => acc
  | Or_bottom.Ok known => (id, of_other known) :: acc
  end.

Definition gather_known
    (maps : list (Index.t * (Keyed.finite_map K value * Or_bottom.t other)))
    : Keyed.finite_map K (list (Index.t * value)) :=
  match maps with
  | [] => []
  | (id1, (map1, other1)) :: maps =>
    fst
      (fold_left
         (fun (acc :
                 Keyed.finite_map K (list (Index.t * value))
                 * list (Index.t * value)) input =>
            let id2 := fst input in
            let map2 := fst (snd input) in
            let other2 := snd (snd input) in
            (Keyed.merge eqb
               (fun vals_opt val2_opt =>
                  match vals_opt, val2_opt with
                  | None, None => None
                  | None, Some val2 => Some ((id2, val2) :: snd acc)
                  | Some vals, None => Some (add_other vals id2 other2)
                  | Some vals, Some val2 => Some ((id2, val2) :: vals)
                  end)
               (fst acc) map2,
             add_other (snd acc) id2 other2))
         maps
         (Keyed.map_values (fun val1 => [(id1, val1)]) map1,
          add_other [] id1 other1))
  end.

End Merge_map_known.

(* The things of a list, if none is unknown, in their order: the
   [List.map] of [n_way_join_row_like]. [None] is its exception
   [Unknown_result]. *)
Fixpoint all_known {A : Type} (or_unknowns : list (Index.t * Or_unknown.t A))
    : option (list (Index.t * A)) :=
  match or_unknowns with
  | [] => Some []
  | (id, Or_unknown.Known a) :: or_unknowns =>
    option_map (cons (id, a)) (all_known or_unknowns)
  | (_, Or_unknown.Unknown) :: _ => None
  end.

(* The things of a list that are not bottom, the last first: a fold of
   [n_way_join_row_like]. *)
Fixpoint non_bottom {A : Type} (acc : list (Index.t * A))
    (or_bottoms : list (Index.t * Or_bottom.t A)) : list (Index.t * A) :=
  match or_bottoms with
  | [] => acc
  | (_, Or_bottom.Bottom) :: or_bottoms => non_bottom acc or_bottoms
  | (id, Or_bottom.Ok a) :: or_bottoms => non_bottom ((id, a) :: acc) or_bottoms
  end.

(* The extensions of the variants given, the last first: the fold of
   [n_way_join_variant]. [None] is its exception [No_extensions]: one
   of the variants has none. *)
Fixpoint gather_extensions
    (when_immediates when_blocks : list (Index.t * env_extension))
    (extensions : list (Index.t * variant_extensions))
    : option (list (Index.t * env_extension) * list (Index.t * env_extension))
    :=
  match extensions with
  | [] => Some (when_immediates, when_blocks)
  | (_, No_extensions) :: _ => None
  | (id, Ext when_immediate when_block) :: extensions =>
    gather_extensions ((id, when_immediate) :: when_immediates)
      ((id, when_block) :: when_blocks) extensions
  end.

(* The heads of kind value that are not null, of the heads given, the
   last first: the second fold of [n_way_join_head_of_kind_value].
   [None] is its exception [Unknown_result]: one of them is unknown. *)
Fixpoint non_null_heads
    (acc : list (Index.t * head_of_kind_value_non_null))
    (heads : list (Index.t * head_of_kind_value))
    : option (list (Index.t * head_of_kind_value_non_null)) :=
  match heads with
  | [] => Some acc
  | (id2, Head_of_kind_value non_null _) :: heads =>
    match non_null with
    | Unknown => None
    | Bottom => non_null_heads acc heads
    | Ok head => non_null_heads ((id2, head) :: acc) heads
    end
  end.

(* The folds of [n_way_join_head_of_kind_value_non_null] over the heads
   that follow the first: [List.fold_right], with [read] for the
   pattern that each head must match. [None] is the exception
   [Unknown_result]: a head is made with another constructor than the
   first. *)
Fixpoint gather_heads {A B : Type}
    (read : head_of_kind_value_non_null -> option A)
    (combine : Index.t -> A -> B -> B)
    (other_heads : list (Index.t * head_of_kind_value_non_null)) (init : B)
    : option B :=
  match other_heads with
  | [] => Some init
  | (other_id, other_head) :: other_heads =>
    let* acc := gather_heads read combine other_heads init in
    let* a := read other_head in
    Some (combine other_id a acc)
  end.

(* What the fold over variants accumulates. *)
Record variant_parts : Type := {
  parts_blocks : list (Index.t * Or_unknown.t row_like_for_blocks);
  parts_immediates : list (Index.t * Or_unknown.t flambda_type);
  parts_extensions : list (Index.t * variant_extensions);
  parts_is_unique : bool;
  parts_is_int : option (list (Index.t * simple));
  parts_get_tag : option (list (Index.t * simple))
}.

(* The fields of a variant, as a record of the compiler has them. *)
Record variant_fields : Type := {
  variant_is_int : option variable;
  variant_immediates : Or_unknown.t flambda_type;
  variant_get_tag : option variable;
  variant_blocks : Or_unknown.t row_like_for_blocks;
  variant_extensions_of : variant_extensions;
  variant_is_unique : bool
}.

Definition read_variant (head : head_of_kind_value_non_null)
    : option variant_fields :=
  match head with
  | Variant is_int immediates get_tag blocks extensions is_unique =>
    Some
      {| variant_is_int := is_int;
         variant_immediates := immediates;
         variant_get_tag := get_tag;
         variant_blocks := blocks;
         variant_extensions_of := extensions;
         variant_is_unique := is_unique |}
  | Mutable_block _ | Boxed_number _ _ _ | Closures _ _ | String _ => None
  end.

(* What the first variant gives the fold. *)
Definition first_variant_parts (first_id : Index.t) (first : variant_fields)
    : variant_parts :=
  {| parts_blocks := [(first_id, variant_blocks first)];
     parts_immediates := [(first_id, variant_immediates first)];
     parts_extensions := [(first_id, variant_extensions_of first)];
     parts_is_unique := variant_is_unique first;
     parts_is_int :=
       match
         deduce_is_int_simple (variant_blocks first)
           (variant_immediates first) (variant_is_int first)
       with
       | Some is_int => Some [(first_id, is_int)]
       | None => None
       end;
     parts_get_tag :=
       match
         deduce_get_tag_simple (variant_blocks first) (variant_get_tag first)
       with
       | Bottom => Some []
       | Unknown => None
       | Ok get_tag => Some [(first_id, get_tag)]
       end |}.

(* What another variant adds to it. *)
Definition add_variant_parts (other_id : Index.t) (other : variant_fields)
    (acc : variant_parts) : variant_parts :=
  {| parts_blocks := (other_id, variant_blocks other) :: parts_blocks acc;
     parts_immediates :=
       (other_id, variant_immediates other) :: parts_immediates acc;
     parts_extensions :=
       (other_id, variant_extensions_of other) :: parts_extensions acc;
     parts_is_unique := parts_is_unique acc || variant_is_unique other;
     parts_is_int :=
       match parts_is_int acc with
       | None => None
       | Some is_int =>
         match
           deduce_is_int_simple (variant_blocks other)
             (variant_immediates other) (variant_is_int other)
         with
         | Some other_is_int => Some ((other_id, other_is_int) :: is_int)
         | None => None
         end
       end;
     parts_get_tag :=
       match parts_get_tag acc with
       | None => None
       | Some get_tag =>
         match
           deduce_get_tag_simple (variant_blocks other)
             (variant_get_tag other)
         with
         | Bottom => Some get_tag
         | Unknown => None
         | Ok other_get_tag => Some ((other_id, other_get_tag) :: get_tag)
         end
       end |}.

Definition read_mutable_block (head : head_of_kind_value_non_null)
    : option Alloc_mode.t :=
  match head with
  | Mutable_block alloc_mode => Some alloc_mode
  | Variant _ _ _ _ _ _ | Boxed_number _ _ _ | Closures _ _ | String _ => None
  end.

(* A boxed number of the kind [k]: the compiler has a constructor for
   each kind. *)
Definition read_boxed_number (k : boxable_number)
    (head : head_of_kind_value_non_null)
    : option (flambda_type * Alloc_mode.t) :=
  match head with
  | Boxed_number k' n alloc_mode =>
    if kind_eqb (unboxed_kind k) (unboxed_kind k')
    then Some (n, alloc_mode)
    else None
  | Variant _ _ _ _ _ _ | Mutable_block _ | Closures _ _ | String _ => None
  end.

Definition read_closures (head : head_of_kind_value_non_null)
    : option (row_like_for_closures * Alloc_mode.t) :=
  match head with
  | Closures by_function_slot alloc_mode => Some (by_function_slot, alloc_mode)
  | Variant _ _ _ _ _ _ | Mutable_block _ | Boxed_number _ _ _ | String _ =>
    None
  end.

Definition read_string (head : head_of_kind_value_non_null)
    : option (list (list Z)) :=
  match head with
  | String strs => Some strs
  | Variant _ _ _ _ _ _ | Mutable_block _ | Boxed_number _ _ _ | Closures _ _
    => None
  end.

(* The heads of the expanded types that follow the first, those that
   are bottom left out: [extract_head_exn] of
   [n_way_join_expanded_head], with [extract] for the pattern that each
   head must match. The outer [None] is the [assert false] of the
   compiler: a head of another kind. The inner one is its exception
   [Unknown_result]: an expanded type that is unknown. *)
Fixpoint extract_heads {A : Type}
    (extract : Expanded_type.head -> option A)
    (expandeds : list (Index.t * Expanded_type.t))
    : option (option (list (Index.t * A))) :=
  match expandeds with
  | [] => Some (Some [])
  | (id2, expanded2) :: expandeds =>
    match Expanded_type.descr expanded2 with
    | Bottom => extract_heads extract expandeds
    | Unknown => Some None
    | Ok head2 =>
      let* head := extract head2 in
      let* heads := extract_heads extract expandeds in
      Some (option_map (cons (id2, head)) heads)
    end
  end.

(* The [Simple]s that the types given are aliases to: the [List.map] of
   [n_way_join]. [None] is its exception [Not_found]: a type that is
   not an alias. *)
Fixpoint alias_simples (ts : list (Index.t * flambda_type))
    : option (list (Index.t * simple)) :=
  match ts with
  | [] => Some []
  | (id, ty) :: ts =>
    let* simple := get_alias_opt ty in
    let* simples := alias_simples ts in
    Some ((id, simple) :: simples)
  end.

(* The fields at an index of each of the arrays given:
   [List.map (fun (id, fields) -> id, fields.(index))]. [None] is the
   exception of the compiler for an array that has no such index. *)
Fixpoint fields_at (index : nat) (fields : list (Index.t * list flambda_type))
    : option (list (Index.t * flambda_type)) :=
  match fields with
  | [] => Some []
  | (id, fields_of_id) :: fields =>
    let* field := nth_error fields_of_id index in
    let* fields := fields_at index fields in
    Some ((id, field) :: fields)
  end.

(* ======================================================================== *)
(* The traversal                                                            *)
(* ======================================================================== *)

Section Traversal.

(* The state of the join of environments, and the functions of
   join_env.ml that the traversal calls. *)
Context {state : Type}
  (joined_env : state -> Index.t -> option Typing_env.t)
  (n_way_join_simples :
     state -> flambda_kind -> list (Index.t * simple)
     -> option (Or_bottom.t simple * state))
  (import_type : state -> flambda_type -> option state)
  (join_env_n_way_join_env_extension :
     state -> list (Index.t * env_extension)
     -> option (Or_bottom.t (env_extension * state))).

(* [Expand_head.expand_head], [reduce_head_of_kind_naked_immediate],
   [Type_grammar.equal] and [Flambda_kind.Block_shape.equal]. *)
Context
  (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
  (reduce_head_of_kind_naked_immediate :
     Typing_env.t -> head_of_kind_naked_immediate
     -> Or_bottom.t head_of_kind_naked_immediate)
  (type_eqb : flambda_type -> flambda_type -> bool)
  (block_shape_eqb : block_shape -> block_shape -> bool).

(* ['a n_way_join_result], or [None] where the compiler stops. *)
Definition n_way_join_result (A : Type) : Type :=
  option (Or_unknown.t A * state).

(* [map_join_result], which [let>>+] is. *)
Definition map_join_result {A B : Type} (f : A -> B)
    (result : n_way_join_result A) : n_way_join_result B :=
  let* '(v, env) := result in
  Some (or_unknown_map f v, env).

(* [n_way_join_unknown]: the join of things each of which may be
   unknown is unknown if one is. *)
Definition n_way_join_unknown {A B : Type}
    (n_way_join_contents : state -> list (Index.t * A) -> n_way_join_result B)
    (env : state) (or_unknowns : list (Index.t * Or_unknown.t A))
    : n_way_join_result B :=
  match known_contents [] or_unknowns with
  | Some contents => n_way_join_contents env contents
  | None => Some (Or_unknown.Unknown, env)
  end.

(* [n_way_join_head_of_kind_naked_number]: the union of the sets. *)
Definition n_way_join_head_of_kind_naked_number (env : state)
    (t1 : list Z * Index.t) (ts : list (Index.t * list Z))
    : n_way_join_result (list Z) :=
  Some
    (Or_unknown.Known
       (fold_left (fun t1 t2 => numbers_union t1 (snd t2)) ts (fst t1)),
     env).

(* [n_way_join_relation_simples]: the variable that holds, in the
   target, what the [Simple]s given hold in the environments joined, if
   the join of the [Simple]s answers a variable. *)
Definition n_way_join_relation_simples (env : state)
    (simples_opt : option (list (Index.t * simple)))
    : option (option variable * state) :=
  match simples_opt with
  | None => Some (None, env)
  | Some simples =>
    let* '(joined, env) :=
      n_way_join_simples env (Naked_number Kind_naked_immediate) simples
    in
    match joined with
    | Or_bottom.Bottom => Some (None, env)
    | Or_bottom.Ok (Const _) => Some (None, env)
    | Or_bottom.Ok (Symbol _) => Some (None, env)
    | Or_bottom.Ok (Var var) => Some (Some var, env)
    end
  end.

(* The first binding of [n_way_join_head_of_kind_value]: whether the
   joined head may be null, and the variable that says whether it is.
   [None] is the fatal error of the compiler for a symbol. *)
Definition n_way_join_is_null (env : state)
    (heads : list (Index.t * head_of_kind_value)) : option (is_null * state) :=
  match is_null_simples heads with
  | None => Some (Maybe_null None, env)
  | Some is_null_simples =>
    let* '(joined, env) :=
      n_way_join_simples env (Naked_number Kind_naked_immediate)
        is_null_simples
    in
    match joined with
    | Or_bottom.Bottom => Some (Maybe_null None, env)
    | Or_bottom.Ok (Const (Naked_immediate i)) =>
      Some (if Z.eqb i 0 then Not_null else Maybe_null None, env)
    | Or_bottom.Ok (Const _) => Some (Maybe_null None, env)
    | Or_bottom.Ok (Symbol _) => None
    | Or_bottom.Ok (Var var) => Some (Maybe_null (Some var), env)
    end
  end.

(* The fold of [n_way_join_head_of_kind_naked_immediate]: the union of
   the numbers of the heads, each reduced in the environment of its
   index; a head that reduces to bottom is left out. [None] is the
   fatal error of [Join_env.joined_env] for an index that it has
   not. *)
Fixpoint join_naked_immediates (env : state)
    (immediates : Or_unknown.t (list Z))
    (heads : list (Index.t * head_of_kind_naked_immediate))
    : option (Or_unknown.t (list Z)) :=
  match heads with
  | [] => Some immediates
  | (id2, head2) :: heads =>
    let* tenv := joined_env env id2 in
    match reduce_head_of_kind_naked_immediate tenv head2 with
    | Or_bottom.Bottom => join_naked_immediates env immediates heads
    | Or_bottom.Ok head2 =>
      let immediates :=
        match immediates, naked_immediates head2 with
        | Or_unknown.Unknown, _ | _, Or_unknown.Unknown => Or_unknown.Unknown
        | Or_unknown.Known imms, Or_unknown.Known imms2 =>
          Or_unknown.Known (numbers_union imms imms2)
        end
      in
      join_naked_immediates env immediates heads
    end
  end.

(* [n_way_join_head_of_kind_naked_immediate]. [None] is also the fatal
   error of [create_naked_immediates_non_empty] for the empty set. *)
Definition n_way_join_head_of_kind_naked_immediate (env : state)
    (heads : list (Index.t * head_of_kind_naked_immediate))
    : n_way_join_result head_of_kind_naked_immediate :=
  let* immediates :=
    join_naked_immediates env (Or_unknown.Known []) heads
  in
  match immediates with
  | Or_unknown.Unknown => Some (Or_unknown.Unknown, env)
  | Or_unknown.Known imms =>
    if Keyed.is_empty imms
    then None
    else
      Some
        (Or_unknown.Known
           (Head_of_kind_naked_immediate (Or_unknown.Known imms) []),
         env)
  end.

(* [n_way_join_head_of_kind_naked_float32], ...,
   [n_way_join_head_of_kind_naked_mask]. *)
Definition n_way_join_head_of_kind_naked_numbers (env : state)
    (t1 : list Z * Index.t) (ts : list (Index.t * list Z))
    : n_way_join_result (list Z) :=
  n_way_join_head_of_kind_naked_number env t1 ts.

(* [n_way_join_head_of_kind_rec_info]. *)
Definition n_way_join_head_of_kind_rec_info (env : state)
    (t1 : unit * Index.t) (ts : list (Index.t * unit))
    : n_way_join_result unit :=
  Some (Or_unknown.Known (fst t1), env).

(* [n_way_join_head_of_kind_region]. *)
Definition n_way_join_head_of_kind_region (env : state)
    (t1 : unit * Index.t) (ts : list (Index.t * unit))
    : n_way_join_result unit :=
  Some (Or_unknown.Known tt, env).

(* [n_way_join_env_extension]: the join of the environment side, with
   the empty extension where it answers bottom. *)
Definition n_way_join_env_extension (env : state)
    (exts : list (Index.t * env_extension)) : option (env_extension * state) :=
  let* joined := join_env_n_way_join_env_extension env exts in
  match joined with
  | Or_bottom.Bottom => Some (empty_env_extension, env)
  | Or_bottom.Ok (ext, env) => Some (ext, env)
  end.

(* [generic_merge_map_known]: the cases are gathered key by key, then
   joined, the state threaded through the keys. A key whose join is
   bottom is dropped. With no row-like structure the answer is the
   empty map. *)
Definition generic_merge_map_known {K value other : Type}
    (eqb : K -> K -> bool) (of_other : other -> value)
    (n_way_join :
       state -> list (Index.t * value) -> option (Or_bottom.t (value * state)))
    (env : state)
    (maps : list (Index.t * (Keyed.finite_map K value * Or_bottom.t other)))
    : option (Keyed.finite_map K value * state) :=
  match maps with
  | [] => Some ([], env)
  | _ :: _ =>
    Keyed.filter_map_fold eqb
      (fun _index values env =>
         let* joined := n_way_join env values in
         match joined with
         | Or_bottom.Bottom => Some (None, env)
         | Or_bottom.Ok (value, env) => Some (Some value, env)
         end)
      (gather_known eqb of_other maps) env
  end.

(* [n_way_join_row_like], with what it is given by its two callers. *)
Section Row_like.

Context {lattice shape maps_to known : Type}
  (n_way_join_maps_to :
     state -> shape -> list (Index.t * maps_to) -> option (maps_to * state))
  (equal_index : lattice -> lattice -> bool)
  (inter_index : lattice -> lattice -> lattice)
  (n_way_join_shape : list shape -> or_unknown_or_bottom shape)
  (merge_map_known :
     (state
      -> list
           (Index.t
            * Or_unknown.t (row_like_case lattice shape maps_to env_extension))
      -> option
           (Or_bottom.t
              (Or_unknown.t (row_like_case lattice shape maps_to env_extension)
               * state)))
     -> state
     -> list
          (Index.t
           * (known
              * Or_bottom.t
                  (row_like_case lattice shape maps_to env_extension)))
     -> option (known * state)).

(* [n_way_join_index]: the index of the join of cases. The size, or the
   contents, is known if all are known and the same; otherwise it is at
   least what all have. *)
Definition n_way_join_index (indexes : list (row_like_index lattice shape))
    : or_unknown_or_bottom (row_like_index lattice shape) :=
  match n_way_join_shape (map index_shape indexes) with
  | Bottom => Bottom
  | Unknown => Unknown
  | Ok shape =>
    match indexes with
    | [] => Bottom
    | i1 :: indexes =>
      let at_least (i1' : lattice) :=
        Ok
          (Row_like_index
             (Row_like_index_domain.At_least
                (fold_left
                   (fun i1' i2 =>
                      match index_domain i2 with
                      | Row_like_index_domain.At_least i2'
                      | Row_like_index_domain.Known i2' =>
                        inter_index i1' i2'
                      end)
                   indexes i1'))
             shape)
      in
      match index_domain i1 with
      | Row_like_index_domain.Known i1' =>
        if forallb
             (fun i2 =>
                match index_domain i2 with
                | Row_like_index_domain.Known i2' => equal_index i1' i2'
                | Row_like_index_domain.At_least _ => false
                end)
             indexes
        then Ok (Row_like_index (index_domain i1) shape)
        else at_least i1'
      | Row_like_index_domain.At_least i1' => at_least i1'
      end
    end
  end.

(* [n_way_join_case]. *)
Definition n_way_join_case (join_env : state)
    (cases :
       list (Index.t * row_like_case lattice shape maps_to env_extension))
    : option
        (Or_bottom.t
           (Or_unknown.t (row_like_case lattice shape maps_to env_extension)
            * state)) :=
  match n_way_join_index (map (fun case => case_index (snd case)) cases) with
  | Unknown => Some (Or_bottom.Ok (Or_unknown.Unknown, join_env))
  | Bottom => Some Or_bottom.Bottom
  | Ok index =>
    let* '(maps_to, join_env) :=
      n_way_join_maps_to join_env (index_shape index)
        (map (fun case => (fst case, case_maps_to (snd case))) cases)
    in
    let* '(env_extension, join_env) :=
      n_way_join_env_extension join_env
        (map (fun case => (fst case, case_env_extension (snd case))) cases)
    in
    Some
      (Or_bottom.Ok
         (Or_unknown.Known (Row_like_case maps_to index env_extension),
          join_env))
  end.

(* [n_way_join_knowns]. *)
Definition n_way_join_knowns (join_env : state)
    (cases :
       list
         (Index.t
          * Or_unknown.t (row_like_case lattice shape maps_to env_extension)))
    : option
        (Or_bottom.t
           (Or_unknown.t (row_like_case lattice shape maps_to env_extension)
            * state)) :=
  match all_known cases with
  | None => Some (Or_bottom.Ok (Or_unknown.Unknown, join_env))
  | Some cases => n_way_join_case join_env cases
  end.

Definition n_way_join_row_like (join_env : state)
    (known_cases :
       list
         (Index.t
          * (known
             * Or_bottom.t
                 (row_like_case lattice shape maps_to env_extension))))
    (other :
       list
         (Index.t
          * Or_bottom.t (row_like_case lattice shape maps_to env_extension)))
    : n_way_join_result
        (known
         * Or_bottom.t (row_like_case lattice shape maps_to env_extension)) :=
  let* '(known, join_env) :=
    merge_map_known n_way_join_knowns join_env known_cases
  in
  let* other := n_way_join_case join_env (non_bottom [] other) in
  match other with
  | Or_bottom.Bottom =>
    Some (Or_unknown.Known (known, Or_bottom.Bottom), join_env)
  | Or_bottom.Ok (Or_unknown.Known case, join_env) =>
    Some (Or_unknown.Known (known, Or_bottom.Ok case), join_env)
  | Or_bottom.Ok (Or_unknown.Unknown, join_env) =>
    Some (Or_unknown.Unknown, join_env)
  end.

End Row_like.

(* The shape of the join of cases of blocks: theirs, if they have one
   shape. *)
Definition n_way_join_block_shape (shapes : list block_shape)
    : or_unknown_or_bottom block_shape :=
  match shapes with
  | [] => Bottom
  | shape1 :: shapes =>
    if forallb (block_shape_eqb shape1) shapes then Ok shape1 else Unknown
  end.

(* What is said of the join of types by the functions that call it:
   [n_way_join_rec] stands for [n_way_join], which is defined after
   them, with the number that bounds its depth. *)
Section Body.

Variable n_way_join_rec :
  state -> list (Index.t * flambda_type) -> n_way_join_result flambda_type.

(* [n_way_join_generic_product]: the join, index by index, of the
   components that every product has; an index whose join is unknown
   is dropped. *)
Definition n_way_join_generic_product (env : state)
    (indexed_products : list (Index.t * finite_map flambda_type))
    : option (finite_map flambda_type * state) :=
  match indexed_products with
  | [] => Some ([], env)
  | _ :: _ =>
    Keyed.filter_map_fold Pos.eqb
      (fun _index tys env =>
         let* '(joined, env) := n_way_join_rec env tys in
         match joined with
         | Or_unknown.Known ty => Some (Some ty, env)
         | Or_unknown.Unknown => Some (None, env)
         end)
      (gather_components indexed_products) env
  end.

(* [n_way_join_function_slot_indexed_product] and
   [n_way_join_value_slot_indexed_product]: a product of the model is
   the map that it holds. [Product.Function_slot_indexed.create], with
   which the compiler makes the first, stops with a fatal error if one
   of the components is not of kind value: [None]. The creation
   function of the second tests nothing. *)
Definition n_way_join_function_slot_indexed_product (env : state)
    (function_slots_indexed : list (Index.t * finite_map flambda_type))
    : option (finite_map flambda_type * state) :=
  let* '(components, env) :=
    n_way_join_generic_product env function_slots_indexed
  in
  if for_all_bindings_b (fun _ ty => kind_eqb (kind ty) Value) components
  then Some (components, env)
  else None.

Definition n_way_join_value_slot_indexed_product (env : state)
    (value_slots_indexed : list (Index.t * finite_map flambda_type))
    : option (finite_map flambda_type * state) :=
  n_way_join_generic_product env value_slots_indexed.

(* The first [length] fields of two arrays are the same, as the test
   chosen by -flambda2-join-sharing structural has it. *)
Fixpoint fields_the_same (length : nat)
    (first_fields other_fields : list flambda_type) : bool :=
  match length, first_fields, other_fields with
  | O, _, _ => true
  | S length, first_field :: first_fields, other_field :: other_fields =>
    type_eqb other_field first_field
    && fields_the_same length first_fields other_fields
  | S _, _, _ => false
  end.

(* [Array.fold_left (fun env ty -> Join_env.import_type env ty)]. *)
Fixpoint import_types (env : state) (tys : list flambda_type) : option state :=
  match tys with
  | [] => Some env
  | ty :: tys =>
    let* env := import_type env ty in
    import_types env tys
  end.

(* The [Array.init] of the slow path of
   [n_way_join_int_indexed_product]: the joins of the fields from
   [index] on, [count] of them, the state threaded through them. A
   field whose join is unknown is the unknown type of its kind. *)
Fixpoint join_fields (env : state) (shape : block_shape)
    (fields : list (Index.t * list flambda_type)) (index count : nat)
    : option (list flambda_type * state) :=
  match count with
  | O => Some ([], env)
  | S count =>
    let* tys := fields_at index fields in
    let* '(joined, env) := n_way_join_rec env tys in
    let* ty :=
      match joined with
      | Or_unknown.Unknown => unknown_from_shape shape index
      | Or_unknown.Known ty => Some ty
      end
    in
    let* '(tys, env) := join_fields env shape fields (S index) count in
    Some (ty :: tys, env)
  end.

(* [n_way_join_int_indexed_product]: as many fields as the shortest
   array has. If every array has the same fields up to there, they are
   those of the first array of that length, as they are, their
   variables imported; otherwise each is the join of the fields at its
   index. [None] is the fatal error of the compiler for no array, and
   its [assert false]. *)
Definition n_way_join_int_indexed_product (env : state) (shape : block_shape)
    (fields : list (Index.t * list flambda_type))
    : option (list flambda_type * state) :=
  match fields with
  | [] => None
  | (_, first_fields) :: other_fields =>
    let least_length :=
      fold_left
        (fun least_length other_fields =>
           Nat.min least_length (length (snd other_fields)))
        other_fields (length first_fields)
    in
    let all_the_same :=
      forallb
        (fun other_fields =>
           fields_the_same least_length first_fields (snd other_fields))
        other_fields
    in
    if all_the_same
    then
      match
        find
          (fun fields_of_id =>
             Nat.eqb (length (snd fields_of_id)) least_length)
          fields
      with
      | None => None
      | Some (_, fields_of_id) =>
        let* env := import_types env fields_of_id in
        Some (fields_of_id, env)
      end
    else join_fields env shape fields 0 least_length
  end.

(* [n_way_join_closures_entry]. *)
Definition n_way_join_closures_entry (env : state)
    (entries : list (Index.t * closures_entry))
    : option (closures_entry * state) :=
  let function_types :=
    n_way_join_function_types
      (map
         (fun entry => (fst entry, closures_entry_function_types (snd entry)))
         entries)
  in
  let closure_types :=
    rev
      (map
         (fun entry => (fst entry, closures_entry_closure_types (snd entry)))
         entries)
  in
  let value_slot_types :=
    rev
      (map
         (fun entry =>
            (fst entry, closures_entry_value_slot_types (snd entry)))
         entries)
  in
  let* '(closure_types, env) :=
    n_way_join_function_slot_indexed_product env closure_types
  in
  let* '(value_slot_types, env) :=
    n_way_join_value_slot_indexed_product env value_slot_types
  in
  Some (Closures_entry function_types closure_types value_slot_types, env).

(* [n_way_join_row_like_for_blocks]: the row-like structures that are
   bottom are left out. *)
Definition n_way_join_row_like_for_blocks (env : state)
    (row_likes : list (Index.t * row_like_for_blocks))
    : n_way_join_result row_like_for_blocks :=
  let split_or_bottom :=
    fold_left
      (fun (acc :
              Or_bottom.t
                (list
                   (Index.t
                    * (Keyed.finite_map Z (Or_unknown.t row_like_block_case)
                       * Or_bottom.t row_like_block_case))
                 * list (Index.t * Or_bottom.t row_like_block_case)
                 * Alloc_mode.t)) row_like =>
         let id := fst row_like in
         let row_like := snd row_like in
         if row_like_for_blocks_is_bottom row_like
         then acc
         else
           match acc with
           | Or_bottom.Bottom =>
             Or_bottom.Ok
               ([(id,
                  (row_like_known_tags row_like,
                   row_like_other_tags row_like))],
                [(id, row_like_other_tags row_like)],
                row_like_alloc_mode row_like)
           | Or_bottom.Ok (known, other, alloc_mode) =>
             Or_bottom.Ok
               ((id,
                 (row_like_known_tags row_like, row_like_other_tags row_like))
                :: known,
                (id, row_like_other_tags row_like) :: other,
                join_alloc_mode alloc_mode (row_like_alloc_mode row_like))
           end)
      row_likes Or_bottom.Bottom
  in
  match split_or_bottom with
  | Or_bottom.Bottom =>
    Some (Or_unknown.Known row_like_for_blocks_bottom, env)
  | Or_bottom.Ok (known, other, alloc_mode) =>
    map_join_result
      (fun joined => Row_like_for_blocks (fst joined) (snd joined) alloc_mode)
      (n_way_join_row_like n_way_join_int_indexed_product Nat.eqb Nat.min
         n_way_join_block_shape
         (generic_merge_map_known Z.eqb Or_unknown.Known) env known other)
  end.

(* [n_way_join_row_like_for_closures]. [None] is one of the two fatal
   errors of the compiler: a case for the other function slots, or an
   unknown join. *)
Definition n_way_join_row_like_for_closures (env : state)
    (closures : list (Index.t * row_like_for_closures))
    : option (row_like_for_closures * state) :=
  let known :=
    rev
      (map
         (fun closures =>
            (fst closures,
             (row_like_known_closures (snd closures),
              @Or_bottom.Bottom row_like_closures_case)))
         closures)
  in
  let merge_map_known join_case env knowns :=
    generic_merge_map_known Pos.eqb (fun case : row_like_closures_case => case)
      (fun env cases =>
         let cases :=
           map (fun case => (fst case, Or_unknown.Known (snd case))) cases
         in
         let* joined := join_case env cases in
         match joined with
         | Or_bottom.Bottom => Some Or_bottom.Bottom
         | Or_bottom.Ok (Or_unknown.Known case, env) =>
           Some (Or_bottom.Ok (case, env))
         | Or_bottom.Ok (Or_unknown.Unknown, _) => None
         end)
      env knowns
  in
  let* '(joined, env) :=
    n_way_join_row_like
      (fun env (_ : unit) entries => n_way_join_closures_entry env entries)
      set_of_closures_contents_equal set_of_closures_contents_inter
      (fun _ => Ok tt) merge_map_known env known []
  in
  match joined with
  | Or_unknown.Known (known_closures, Or_bottom.Bottom) =>
    Some (Row_like_for_closures known_closures, env)
  | Or_unknown.Known (_, Or_bottom.Ok _) => None
  | Or_unknown.Unknown => None
  end.

(* [n_way_join_variant]. If the two joined extensions are empty the
   state is the one before they were joined. *)
Definition n_way_join_variant (env : state)
    (blocks : list (Index.t * Or_unknown.t row_like_for_blocks))
    (imms : list (Index.t * Or_unknown.t flambda_type))
    (extensions : list (Index.t * variant_extensions))
    : n_way_join_result
        (Or_unknown.t row_like_for_blocks * Or_unknown.t flambda_type
         * variant_extensions) :=
  let* '(blocks, env) :=
    n_way_join_unknown n_way_join_row_like_for_blocks env blocks
  in
  let* '(imms, env) := n_way_join_unknown n_way_join_rec env imms in
  let* '(env, extensions) :=
    match gather_extensions [] [] extensions with
    | None => Some (env, No_extensions)
    | Some (when_immediates, when_blocks) =>
      let env0 := env in
      let* '(when_immediate, env) :=
        n_way_join_env_extension env when_immediates
      in
      let* '(when_block, env) := n_way_join_env_extension env when_blocks in
      if env_extension_is_empty when_immediate
         && env_extension_is_empty when_block
      then Some (env0, No_extensions)
      else Some (env, Ext when_immediate when_block)
    end
  in
  match blocks, imms, extensions with
  | Or_unknown.Unknown, Or_unknown.Unknown, No_extensions =>
    Some (Or_unknown.Unknown, env)
  | _, _, _ => Some (Or_unknown.Known (blocks, imms, extensions), env)
  end.

(* [n_way_join_head_of_kind_value_non_null]: heads made with one
   constructor are joined part by part; with two, the join is unknown.
   [None] is the fatal error of the compiler for no head. If the join
   of the parts of variants is unknown, so is the head, and the
   relations joined before it are dropped. *)
Definition n_way_join_head_of_kind_value_non_null (env : state)
    (heads : list (Index.t * head_of_kind_value_non_null))
    : n_way_join_result head_of_kind_value_non_null :=
  match heads with
  | [] => None
  | (first_id, first_head) :: other_heads =>
    match first_head with
    | Variant is_int immediates get_tag blocks extensions is_unique =>
      let first :=
        {| variant_is_int := is_int;
           variant_immediates := immediates;
           variant_get_tag := get_tag;
           variant_blocks := blocks;
           variant_extensions_of := extensions;
           variant_is_unique := is_unique |}
      in
      match
        gather_heads read_variant add_variant_parts other_heads
          (first_variant_parts first_id first)
      with
      | None => Some (Or_unknown.Unknown, env)
      | Some parts =>
        let* '(is_int, env) :=
          n_way_join_relation_simples env (parts_is_int parts)
        in
        let* '(get_tag, env) :=
          n_way_join_relation_simples env (parts_get_tag parts)
        in
        map_join_result
          (fun joined =>
             Variant is_int (snd (fst joined)) get_tag (fst (fst joined))
               (snd joined) (parts_is_unique parts))
          (n_way_join_variant env (parts_blocks parts)
             (parts_immediates parts) (parts_extensions parts))
      end
    | Mutable_block alloc_mode =>
      match
        gather_heads read_mutable_block
          (fun _ other_alloc_mode alloc_mode =>
             join_alloc_mode alloc_mode other_alloc_mode)
          other_heads alloc_mode
      with
      | None => Some (Or_unknown.Unknown, env)
      | Some alloc_mode =>
        Some (Or_unknown.Known (Mutable_block alloc_mode), env)
      end
    | Boxed_number k n alloc_mode =>
      match
        gather_heads (read_boxed_number k)
          (fun other_id other acc =>
             ((other_id, fst other) :: fst acc,
              join_alloc_mode (snd acc) (snd other)))
          other_heads
          ([(first_id, n)], alloc_mode)
      with
      | None => Some (Or_unknown.Unknown, env)
      | Some (ns, alloc_mode) =>
        map_join_result (fun n => Boxed_number k n alloc_mode)
          (n_way_join_rec env ns)
      end
    | Closures by_function_slot alloc_mode =>
      match
        gather_heads read_closures
          (fun other_id other acc =>
             ((other_id, fst other) :: fst acc,
              join_alloc_mode (snd acc) (snd other)))
          other_heads
          ([(first_id, by_function_slot)], alloc_mode)
      with
      | None => Some (Or_unknown.Unknown, env)
      | Some (function_slots, alloc_mode) =>
        let* '(by_function_slot, env) :=
          n_way_join_row_like_for_closures env function_slots
        in
        Some (Or_unknown.Known (Closures by_function_slot alloc_mode), env)
      end
    | String strs =>
      match
        gather_heads read_string
          (fun _ other_strs all_strs => strings_union other_strs all_strs)
          other_heads strs
      with
      | None => Some (Or_unknown.Unknown, env)
      | Some all_strs => Some (Or_unknown.Known (String all_strs), env)
      end
    end
  end.

(* [n_way_join_head_of_kind_value]. *)
Definition n_way_join_head_of_kind_value (env : state)
    (heads : list (Index.t * head_of_kind_value))
    : n_way_join_result head_of_kind_value :=
  let* '(is_null, env) := n_way_join_is_null env heads in
  let* '(non_null, env) :=
    match non_null_heads [] heads with
    | None => Some (@Unknown head_of_kind_value_non_null, env)
    | Some [] => Some (Bottom, env)
    | Some ((_ :: _) as non_nulls) =>
      let* '(joined, env) :=
        n_way_join_head_of_kind_value_non_null env non_nulls
      in
      match joined with
      | Or_unknown.Unknown => Some (Unknown, env)
      | Or_unknown.Known head => Some (Ok head, env)
      end
    end
  in
  match non_null, is_null with
  | Unknown, Maybe_null None => Some (Or_unknown.Unknown, env)
  | _, _ => Some (Or_unknown.Known (Head_of_kind_value non_null is_null), env)
  end.

(* [n_way_join_expanded_head]. The expanded types that are bottom are
   left out; if one is unknown, or if the join of the heads is, the
   answer is the unknown type of the kind. [None] is the [assert false]
   of the compiler for two heads of different kinds. *)
Fixpoint n_way_join_expanded_head (env : state) (kind : flambda_kind)
    (expandeds : list (Index.t * Expanded_type.t))
    : n_way_join_result Expanded_type.t :=
  match expandeds with
  | [] => Some (Or_unknown.Known (Expanded_type.create_bottom kind), env)
  | (id1, expanded1) :: expandeds =>
    match Expanded_type.descr expanded1 with
    | Bottom => n_way_join_expanded_head env kind expandeds
    | Unknown =>
      Some (Or_unknown.Known (Expanded_type.create_unknown kind), env)
    | Ok descr1 =>
      let* '(expanded_or_unknown, env) :=
        match descr1 with
        | Expanded_type.Head_value head1 =>
          let* heads :=
            extract_heads
              (fun head =>
                 match head with
                 | Expanded_type.Head_value head => Some head
                 | _ => None
                 end)
              expandeds
          in
          match heads with
          | None => Some (@Or_unknown.Unknown Expanded_type.t, env)
          | Some heads =>
            map_join_result Expanded_type.create_value
              (n_way_join_head_of_kind_value env ((id1, head1) :: heads))
          end
        | Expanded_type.Head_naked_immediate head1 =>
          let* heads :=
            extract_heads
              (fun head =>
                 match head with
                 | Expanded_type.Head_naked_immediate head => Some head
                 | _ => None
                 end)
              expandeds
          in
          match heads with
          | None => Some (Or_unknown.Unknown, env)
          | Some heads =>
            map_join_result Expanded_type.create_naked_immediate
              (n_way_join_head_of_kind_naked_immediate env
                 ((id1, head1) :: heads))
          end
        | Expanded_type.Head_naked_number k head1 =>
          let* heads :=
            extract_heads
              (fun head =>
                 match head with
                 | Expanded_type.Head_naked_number k' head =>
                   if kind_eqb (Naked_number k) (Naked_number k')
                   then Some head
                   else None
                 | _ => None
                 end)
              expandeds
          in
          match heads with
          | None => Some (Or_unknown.Unknown, env)
          | Some heads =>
            map_join_result (Expanded_type.create_naked_number k)
              (n_way_join_head_of_kind_naked_numbers env (head1, id1) heads)
          end
        | Expanded_type.Head_rec_info head1 =>
          let* heads :=
            extract_heads
              (fun head =>
                 match head with
                 | Expanded_type.Head_rec_info head => Some head
                 | _ => None
                 end)
              expandeds
          in
          match heads with
          | None => Some (Or_unknown.Unknown, env)
          | Some heads =>
            map_join_result Expanded_type.create_rec_info
              (n_way_join_head_of_kind_rec_info env (head1, id1) heads)
          end
        | Expanded_type.Head_region head1 =>
          let* heads :=
            extract_heads
              (fun head =>
                 match head with
                 | Expanded_type.Head_region head => Some head
                 | _ => None
                 end)
              expandeds
          in
          match heads with
          | None => Some (Or_unknown.Unknown, env)
          | Some heads =>
            map_join_result Expanded_type.create_region
              (n_way_join_head_of_kind_region env (head1, id1) heads)
          end
        end
      in
      match expanded_or_unknown with
      | Or_unknown.Known expanded => Some (Or_unknown.Known expanded, env)
      | Or_unknown.Unknown =>
        Some (Or_unknown.Known (Expanded_type.unknown_like expanded1), env)
      end
    end
  end.

(* [List.map (fun (id, ty) -> id, Expand_head.expand_head
   (Join_env.joined_env env id) ty)]. [None] is a fatal error of the
   compiler, in [Join_env.joined_env] or in [Expand_head]. *)
Fixpoint expand_heads (env : state) (ts : list (Index.t * flambda_type))
    : option (list (Index.t * Expanded_type.t)) :=
  match ts with
  | [] => Some []
  | (id, ty) :: ts =>
    let* tenv := joined_env env id in
    let* expanded := expand_head tenv ty in
    let* expandeds := expand_heads env ts in
    Some ((id, expanded) :: expandeds)
  end.

(* The body of [n_way_join]. [None] is one of its two fatal errors: no
   type, or types of two kinds. The types that are obviously bottom are
   left out. If the others are all aliases, the join is an alias to
   the join of the [Simple]s; if not, each is expanded in the
   environment of its index and the heads are joined. *)
Definition n_way_join_body (env : state) (ts : list (Index.t * flambda_type))
    : n_way_join_result flambda_type :=
  match ts with
  | [] => None
  | (_, t1) :: others =>
    let kind := kind t1 in
    if negb
         (forallb
            (fun t => kind_eqb kind (FlambdaTypeGrammar.kind (snd t)))
            others)
    then None
    else
      let ts := filter (fun t => negb (is_obviously_bottom (snd t))) ts in
      match alias_simples ts with
      | Some canonical_simples =>
        let* '(joined, env) := n_way_join_simples env kind canonical_simples in
        match joined with
        | Or_bottom.Bottom => Some (Or_unknown.Known (bottom kind), env)
        | Or_bottom.Ok simple =>
          Some (Or_unknown.Known (alias_type_of kind simple), env)
        end
      | None =>
        let* expandeds := expand_heads env ts in
        map_join_result Expanded_type.to_type
          (n_way_join_expanded_head env kind expandeds)
      end
  end.

End Body.

(* [n_way_join]. The first argument bounds the depth of its calls to
   itself: [None] is also that bound reached. *)
Fixpoint n_way_join (fuel : nat)
    : state -> list (Index.t * flambda_type) -> n_way_join_result flambda_type
    :=
  match fuel with
  | O => fun _ _ => None
  | S fuel => n_way_join_body (n_way_join fuel)
  end.

(* ------------------------------------------------------------------------ *)
(* The join of types with a bound on its expansions                         *)
(* ------------------------------------------------------------------------ *)

(* A variant of [n_way_join] that is not the compiler's: the
   transcribed function with a bound on the recursion that goes
   through the expansion of an alias (discrepancy T5b-1: the
   compiler's function has none, and does not end on types that go
   round a cycle out of step). *)
(* CR mshinwell: nothing else in the model uses this variant. It is
   here so that the model says of a join with such a bound what it
   says of the join without one: to answer the unknown type at the
   bound keeps the contract ([n_way_join_bounded_sound_statement],
   FlambdaJoinTypesSpec.v). It is transcribed from no compiler: that
   a change of the compiler computes what this variant computes
   would have to be checked against that change. Whether the
   compiler is to have the bound, and which one, is yours to
   decide; if it has it, its function is to be transcribed and
   [n_way_join] goes. *)

(* Whether the join of these types expands an alias: the types that
   are not obviously bottom are not all aliases, and one of them is.
   (If all are aliases the join of their [Simple]s is taken, and no
   type is expanded. If none is, the heads joined are parts of the
   types given.) *)
Definition expands_an_alias (ts : list (Index.t * flambda_type)) : bool :=
  let ts := filter (fun t => negb (is_obviously_bottom (snd t))) ts in
  match alias_simples ts with
  | Some _ => false
  | None =>
    existsb
      (fun t =>
         match get_alias_opt (snd t) with
         | Some _ => true
         | None => false
         end)
      ts
  end.

(* What is answered at the bound: the unknown type of the kind of the
   types given, in the state given, after the two tests with which
   [n_way_join_body] begins. *)
Definition unknown_at_bound (env : state)
    (ts : list (Index.t * flambda_type)) : n_way_join_result flambda_type :=
  match ts with
  | [] => None
  | (_, t1) :: others =>
    let kind := kind t1 in
    if negb
         (forallb
            (fun t => kind_eqb kind (FlambdaTypeGrammar.kind (snd t)))
            others)
    then None
    else Some (Or_unknown.Known (unknown kind), env)
  end.

(* [n_way_join], with [depth] the number of expansions of an alias
   that the calls of the function to itself may still go through, one
   inside another: at 0, a join that would expand an alias answers
   the unknown type. The first argument bounds the depth of all the
   calls, as it does for [n_way_join]; it is there for the definition
   to be accepted, and [None] is also that bound reached. *)
Fixpoint n_way_join_bounded (fuel depth : nat)
    : state -> list (Index.t * flambda_type) -> n_way_join_result flambda_type
    :=
  match fuel with
  | O => fun _ _ => None
  | S fuel =>
    fun env ts =>
      if expands_an_alias ts
      then
        match depth with
        | O => unknown_at_bound env ts
        | S depth => n_way_join_body (n_way_join_bounded fuel depth) env ts
        end
      else n_way_join_body (n_way_join_bounded fuel depth) env ts
  end.

End Traversal.
