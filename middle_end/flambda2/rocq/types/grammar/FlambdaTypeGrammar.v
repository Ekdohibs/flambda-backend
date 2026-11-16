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


(* The grammar of Flambda types, as the model sees it.

   This file is to be reviewed, line by line against
   middle_end/flambda2/types/grammar/type_grammar.ml (the types at its
   head, and the creation functions) and type_descr.ml, as FlambdaModel.v
   is against flambda.mli. It holds definitions only, it is an ordinary
   inductive type, and it extracts: the converter from the compiler's
   types (workstream T7) is to be a case analysis onto it, and the
   operations on types (meet, join, the provers) are functions on it.
   What a type means is FlambdaTypeMeaningSpec.v.

   What the model has of [Type_grammar.t], constructor by constructor.

   - [t]. The fifteen constructors, one for each kind, are five here.
     [Value] is [Type_value] and [Naked_immediate] is
     [Type_naked_immediate]. The eleven of the other naked numbers
     ([Naked_float32] to [Naked_mask]), whose heads are all sets of
     numbers, are the one [Type_naked_number], which carries the kind:
     [Naked_float ty] is [Type_naked_number Kind_naked_float ty].
     [Rec_info] and [Region] are [Type_rec_info] and [Type_region]. The
     prefix is there because [Value], [Naked_number], [Naked_immediate]
     and the others are the kinds and the constants of FlambdaModel.v.
     [Type_naked_number Kind_naked_immediate] corresponds to no type of
     the compiler, and the converter never gives it.
   - ['head Type_descr.t] is [type_descr head], as [Type_descr.descr]
     shows it: [Unknown], [Bottom], or [Ok] of [No_alias head] or of
     [Equals simple]. The cache of free names
     ([With_cached_free_names]) is not modelled. A [Simple] has no
     coercion (FlambdaModel.v).
   - [head_of_kind_value], with [non_null] and [is_null], and [is_null]
     itself: in full.
   - [head_of_kind_value_non_null]:
     - [Variant]: in full, with its six fields.
     - [Mutable_block]: in full.
     - [Boxed_float32], [Boxed_float], [Boxed_int32], [Boxed_int64],
       [Boxed_nativeint], [Boxed_vec128], [Boxed_vec256], [Boxed_vec512]
       and [Boxed_mask] are the one [Boxed_number], which says which
       number is boxed, a [Flambda_kind.Boxable_number.t]
       ([boxable_number], in FlambdaKindModel.v, which has these nine):
       [Boxed_float (ty, alloc_mode)] is
       [Boxed_number Boxable_naked_float ty alloc_mode].
     - [Closures]: in full.
     - [String]: in full; a string is the list of its bytes.
     - [Array] is left out. Its [element_kind] is a
       [Flambda_kind.With_subkind.t], and the model has no subkinds; the
       machine has no arrays either. It is to be added as a constructor
       when the terms have arrays; until then the converter must refuse
       a type that has one.
   - [head_of_kind_naked_immediate] is the record that
     [Head_of_kind_naked_immediate.descr] gives, [naked_immediates] and
     [inverse_relations], and not the three constructors that represent
     it. [inverse_relations], a [Name.Set.t Relation.Map.t], is the list
     of the pairs of a relation and a name in it.
   - The heads of the other naked numbers are sets of numbers: lists of
     integers, the numbers of the floats and vectors being bit patterns
     as in FlambdaModel.v. Nothing here restricts a number to its range.
   - [head_of_kind_rec_info], a [Rec_info_expr.t], is not modelled: the
     head of a [Type_rec_info] is [unit], as that of a [Type_region] is.
     The terms of the model have no [Rec_info] (FlambdaModel.v).
   - [row_like_index_domain], [row_like_index] and [row_like_case]: in
     full, with the three fields of a case. The index of a case of blocks
     is its size, a [Block_size.t], here a [nat], with its shape; that of
     a case of closures is a [Set_of_closures_contents.t].
   - [row_like_for_blocks]: in full. [Flambda_kind.Block_shape.t] is
     [block_shape]; of a [Mixed_block_shape.t] it keeps the size of the
     prefix of values and the kinds of the flat suffix.
   - [row_like_for_closures], [closures_entry],
     [function_slot_indexed_product] and [value_slot_indexed_product]:
     in full; a product is the map it holds.
   - [function_type]: its [code_id]. Its [rec_info] is left out, as
     [Rec_info] is.
   - [array_contents] is left out with [Array].
   - [env_extension] and [variant_extensions]: in full.
   - [Alloc_mode.For_types.t]: in full, wherever the compiler has it.
     [Alloc_mode.For_types.unknown ()], which depends on whether stack
     allocation is enabled, is [Heap_or_local] in the creation functions
     here.

   Differences of presentation:
   - A record of the compiler that is defined with the types is a
     constructor with named arguments here ([Head_of_kind_value],
     [Row_like_for_blocks], [Closures_entry], ...), Rocq having no
     records in a mutual definition. The others are records, their
     fields prefixed where the name is taken.
   - [Or_unknown.t], [Or_bottom.t], [Row_like_index_domain.t] and
     [Alloc_mode.For_types.t] are in modules of those names, so that
     [Or_unknown.Unknown], [Or_bottom.Bottom] and
     [Row_like_index_domain.Known] are told from the [Unknown], [Ok] and
     [Bottom] of [Or_unknown_or_bottom.t], which are not in a module, and
     from [Or_unknown.Known]. [Relation.descr] is [Relation.t].
   - Maps and sets are those of FlambdaMapSpec.v. Those keyed by
     function slots and by value slots are its maps on names. A map
     keyed by tags ([Tag.Map.t]) is a [Keyed.finite_map Z], read with
     [Keyed.lookup Z.eqb]; a map keyed by the names of the types, each
     a variable or a symbol ([Name.Map.t]), is a
     [Keyed.finite_map name], read with [Keyed.lookup name_eqb]. A set
     of numbers is a list of them.
   - Function slots are the [function_slot] of FlambdaModel.v, which
     are names.

   The grammar does not enforce what the creation functions of the
   compiler check: that [immediates] is of kind naked immediate, the
   contents of a boxed number of the kind boxed, the closures of a
   product of kind value, the numbers of a head not empty. Nor does it
   enforce what the raw constructors of the compiler leave to their
   callers: that a product is no wider than the size of its index,
   that the slots an entry speaks of are among those of its index,
   that a shape is written in one way. A type that breaks one of these
   has a meaning all the same (often that of bottom). The header of
   FlambdaTypeMeaningSpec.v says which operations rely on which, and
   what each gives; [type_well_formed], in FlambdaTypeWellFormedSpec.v,
   is the test that says them all of a type, with where the compiler
   establishes each. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
Import ListNotations.

(* ======================================================================== *)
(* Names                                                                    *)
(* ======================================================================== *)

(* [Name.t]: a variable or a symbol. *)
Inductive name : Type :=
  | Name_var (x : variable)
  | Name_symbol (s : symbol).

(* [Simple.name]. *)
Definition simple_of_name (n : name) : simple :=
  match n with
  | Name_var x => Var x
  | Name_symbol s => Symbol s
  end.

(* [Name.equal]: the equality test of names, and the one to use wherever
   names are keys ([Keyed], in FlambdaMapSpec.v): a variable and a
   symbol are different names, though they may be the same number. It
   decides equality ([name_eqb_eq], in FlambdaTypeGrammarProofs.v). *)
Definition name_eqb (n n' : name) : bool :=
  match n, n' with
  | Name_var x, Name_var x' => Pos.eqb x x'
  | Name_symbol s, Name_symbol s' => Pos.eqb s s'
  | Name_var _, Name_symbol _ | Name_symbol _, Name_var _ => false
  end.

(* ======================================================================== *)
(* What the grammar is built with                                           *)
(* ======================================================================== *)

(* [Or_unknown.t]. *)
Module Or_unknown.
  Inductive t (A : Type) : Type :=
    | Known (a : A)
    | Unknown.
  Arguments Known {A} a.
  Arguments Unknown {A}.
End Or_unknown.

(* [Or_bottom.t]. *)
Module Or_bottom.
  Inductive t (A : Type) : Type :=
    | Ok (a : A)
    | Bottom.
  Arguments Ok {A} a.
  Arguments Bottom {A}.
End Or_bottom.

(* [Or_unknown_or_bottom.t]. *)
Inductive or_unknown_or_bottom (A : Type) : Type :=
  | Unknown
  | Ok (a : A)
  | Bottom.

Arguments Unknown {A}.
Arguments Ok {A} a.
Arguments Bottom {A}.

(* [Type_descr.Descr.t]: a type that is not unknown or bottom is given by
   its head or is an alias, equal to a [Simple]. *)
Inductive descr (head : Type) : Type :=
  | No_alias (h : head)
  | Equals (s : simple).

Arguments No_alias {head} h.
Arguments Equals {head} s.

(* [Type_descr.t], as [Type_descr.descr] shows it. *)
Definition type_descr (head : Type) : Type :=
  or_unknown_or_bottom (descr head).

(* [Alloc_mode.For_types.t]. *)
Module Alloc_mode.
  Inductive t : Type :=
    | Heap
    | Local
    | Heap_or_local.
End Alloc_mode.

(* [Type_grammar.is_null]. The variable, if there is one, holds whether
   the value is null. *)
Inductive is_null : Type :=
  | Not_null
  | Maybe_null (is_null : option variable).

(* [Type_grammar.Relation.descr]: the relations, named after primitives,
   that a type of naked immediates can have to the value of a name. *)
Module Relation.
  Inductive t : Type :=
    | Is_null
    | Is_int
    | Get_tag.

  (* [Type_grammar.Relation.of_const]: the number that the compiler
     takes a relation to relate a constant to, where it takes it to
     relate it to one. [Reg_width_const.is_null] is the test for
     [Null]. *)
  Definition of_const (relation : t) (c : const) : Or_bottom.t Z :=
    match relation, c with
    | Is_null, Null => Or_bottom.Ok 1%Z
    | Is_null, _ => Or_bottom.Ok 0%Z
    | Is_int, Null => Or_bottom.Ok 0%Z
    | Is_int, _ => Or_bottom.Ok 1%Z
    | Get_tag, _ => Or_bottom.Bottom
    end.
End Relation.

(* [head_of_kind_naked_immediate], as [Head_of_kind_naked_immediate.descr]
   shows it: the numbers it may be, and the names of which it is the
   result of a relation. *)
Record head_of_kind_naked_immediate : Type := Head_of_kind_naked_immediate {
  naked_immediates : Or_unknown.t (list Z);
  inverse_relations : list (Relation.t * name)
}.

(* [Flambda_kind.Scannable_block_shape.t], and of a
   [Flambda_kind.Mixed_block_shape.t] what [value_prefix_size] and
   [flat_suffix] give. *)
Inductive scannable_block_shape : Type :=
  | Value_only
  | Mixed_record (value_prefix_size : nat)
      (flat_suffix : list naked_number_kind).

(* [Flambda_kind.Block_shape.t]. *)
Inductive block_shape : Type :=
  | Scannable (shape : scannable_block_shape)
  | Float_record.

(* [Obj.no_scan_tag] and [Tag.double_array_tag]. *)
Definition no_scan_tag : Z := 251.
Definition double_array_tag : Z := 254.

(* [Type_grammar.row_like_index_domain]: [Known x] is [x] alone, and
   [At_least x] everything that includes [x]. *)
Module Row_like_index_domain.
  Inductive t (lattice : Type) : Type :=
    | Known (index : lattice)
    | At_least (index : lattice).
  Arguments Known {lattice} index.
  Arguments At_least {lattice} index.
End Row_like_index_domain.

(* [Type_grammar.row_like_index]. *)
Record row_like_index (lattice shape : Type) : Type := Row_like_index {
  index_domain : Row_like_index_domain.t lattice;
  index_shape : shape
}.

Arguments Row_like_index {lattice shape} index_domain index_shape.
Arguments index_domain {lattice shape} _.
Arguments index_shape {lattice shape} _.

(* [Type_grammar.row_like_case]. The type of environment extensions is a
   parameter because it is defined with the types, below. *)
Record row_like_case (lattice shape maps_to extension : Type) : Type :=
  Row_like_case {
    case_maps_to : maps_to;
    case_index : row_like_index lattice shape;
    case_env_extension : extension
  }.

Arguments Row_like_case {lattice shape maps_to extension} case_maps_to
  case_index case_env_extension.
Arguments case_maps_to {lattice shape maps_to extension} _.
Arguments case_index {lattice shape maps_to extension} _.
Arguments case_env_extension {lattice shape maps_to extension} _.

(* [Set_of_closures_contents.t]: function slots and value slots of a set
   of closures. *)
Record set_of_closures_contents : Type := Set_of_closures_contents {
  contents_closures : list function_slot;
  contents_value_slots : list value_slot
}.

(* [Type_grammar.function_type], without its [rec_info]. *)
Record function_type : Type := Function_type {
  function_type_code_id : code_id
}.

(* ======================================================================== *)
(* The grammar                                                              *)
(* ======================================================================== *)

(* [Type_grammar.t] and the types defined with it. The cases of a
   row-like structure of blocks are [row_like_block_case], and those of
   one of closures have a [Set_of_closures_contents.t] for index and a
   [closures_entry] for what they map to; both are written out, the
   definition being recursive. *)
Inductive flambda_type : Type :=
  | Type_value (ty : type_descr head_of_kind_value)
  | Type_naked_immediate (ty : type_descr head_of_kind_naked_immediate)
  | Type_naked_number (k : naked_number_kind) (ty : type_descr (list Z))
  | Type_rec_info (ty : type_descr unit)
  | Type_region (ty : type_descr unit)

with head_of_kind_value : Type :=
  | Head_of_kind_value
      (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
      (is_null : is_null)

with head_of_kind_value_non_null : Type :=
  | Variant (is_int : option variable)
      (immediates : Or_unknown.t flambda_type) (get_tag : option variable)
      (blocks : Or_unknown.t row_like_for_blocks)
      (extensions : variant_extensions) (is_unique : bool)
  | Mutable_block (alloc_mode : Alloc_mode.t)
  | Boxed_number (k : boxable_number) (contents : flambda_type)
      (alloc_mode : Alloc_mode.t)
  | Closures (by_function_slot : row_like_for_closures)
      (alloc_mode : Alloc_mode.t)
  | String (strs : list (list Z))

with row_like_for_blocks : Type :=
  | Row_like_for_blocks
      (known_tags :
        Keyed.finite_map Z
          (Or_unknown.t
             (row_like_case nat block_shape (list flambda_type)
                env_extension)))
      (other_tags :
        Or_bottom.t
          (row_like_case nat block_shape (list flambda_type) env_extension))
      (alloc_mode : Alloc_mode.t)

with row_like_for_closures : Type :=
  | Row_like_for_closures
      (known_closures :
        finite_map
          (row_like_case set_of_closures_contents unit closures_entry
             env_extension))

with closures_entry : Type :=
  | Closures_entry (function_types : finite_map (Or_unknown.t function_type))
      (closure_types : finite_map flambda_type)
      (value_slot_types : finite_map flambda_type)

with env_extension : Type :=
  | Env_extension (equations : Keyed.finite_map name flambda_type)

with variant_extensions : Type :=
  | No_extensions
  | Ext (when_immediate when_block : env_extension).

(* [Type_grammar.row_like_block_case]. *)
Definition row_like_block_case : Type :=
  row_like_case nat block_shape (list flambda_type) env_extension.

(* The cases of [row_like_for_closures]. *)
Definition row_like_closures_case : Type :=
  row_like_case set_of_closures_contents unit closures_entry env_extension.

(* ======================================================================== *)
(* Kinds                                                                    *)
(* ======================================================================== *)

(* [Type_grammar.kind]. *)
Definition kind (t : flambda_type) : flambda_kind :=
  match t with
  | Type_value _ => Value
  | Type_naked_immediate _ => Naked_number Kind_naked_immediate
  | Type_naked_number k _ => Naked_number k
  | Type_rec_info _ => Rec_info
  | Type_region _ => Region
  end.

(* The kinds of the fields of a block of [size] fields of a shape
   ([Flambda_kind.Block_shape.element_kind], for each field). A mixed
   block has the size that its shape gives it. *)
Definition field_kinds (shape : block_shape) (size : nat)
    : list flambda_kind :=
  match shape with
  | Scannable Value_only => repeat Value size
  | Scannable (Mixed_record value_prefix_size flat_suffix) =>
    repeat Value value_prefix_size ++ map Naked_number flat_suffix
  | Float_record => repeat (Naked_number Kind_naked_float) size
  end.

(* ======================================================================== *)
(* Creation of types                                                        *)
(* ======================================================================== *)

(* [More_type_creators.unknown]: [any_value], [any_naked_immediate], ... *)
Definition unknown (k : flambda_kind) : flambda_type :=
  match k with
  | Value => Type_value Unknown
  | Naked_number Kind_naked_immediate => Type_naked_immediate Unknown
  | Naked_number k => Type_naked_number k Unknown
  | Rec_info => Type_rec_info Unknown
  | Region => Type_region Unknown
  end.

(* [More_type_creators.bottom]: [bottom_value], ... *)
Definition bottom (k : flambda_kind) : flambda_type :=
  match k with
  | Value => Type_value Bottom
  | Naked_number Kind_naked_immediate => Type_naked_immediate Bottom
  | Naked_number k => Type_naked_number k Bottom
  | Rec_info => Type_rec_info Bottom
  | Region => Type_region Bottom
  end.

(* [Type_grammar.alias_type_of]. *)
Definition alias_type_of (k : flambda_kind) (s : simple) : flambda_type :=
  match k with
  | Value => Type_value (Ok (Equals s))
  | Naked_number Kind_naked_immediate => Type_naked_immediate (Ok (Equals s))
  | Naked_number k => Type_naked_number k (Ok (Equals s))
  | Rec_info => Type_rec_info (Ok (Equals s))
  | Region => Type_region (Ok (Equals s))
  end.

(* [Type_grammar.get_alias_opt]. *)
Definition get_alias_opt (t : flambda_type) : option simple :=
  let of_descr {head : Type} (ty : type_descr head) : option simple :=
    match ty with
    | Ok (Equals s) => Some s
    | Unknown | Bottom | Ok (No_alias _) => None
    end
  in
  match t with
  | Type_value ty => of_descr ty
  | Type_naked_immediate ty => of_descr ty
  | Type_naked_number _ ty => of_descr ty
  | Type_rec_info ty => of_descr ty
  | Type_region ty => of_descr ty
  end.

(* [Type_grammar.is_obviously_bottom]. *)
Definition is_obviously_bottom (t : flambda_type) : bool :=
  let of_descr {head : Type} (ty : type_descr head) : bool :=
    match ty with
    | Bottom => true
    | Unknown | Ok _ => false
    end
  in
  match t with
  | Type_value ty => of_descr ty
  | Type_naked_immediate ty => of_descr ty
  | Type_naked_number _ ty => of_descr ty
  | Type_rec_info ty => of_descr ty
  | Type_region ty => of_descr ty
  end.

(* [Type_grammar.is_obviously_unknown]. *)
Definition is_obviously_unknown (t : flambda_type) : bool :=
  let of_descr {head : Type} (ty : type_descr head) : bool :=
    match ty with
    | Unknown => true
    | Bottom | Ok _ => false
    end
  in
  match t with
  | Type_value ty => of_descr ty
  | Type_naked_immediate ty => of_descr ty
  | Type_naked_number _ ty => of_descr ty
  | Type_rec_info ty => of_descr ty
  | Type_region ty => of_descr ty
  end.

(* [More_type_creators.type_for_const], but for [Null] and [Poison]: the
   type of a number is an alias to it ([this_tagged_immediate],
   [this_naked_immediate], [this_naked_float], ...). *)
Definition this_const (c : const) : flambda_type :=
  alias_type_of (kind_of_const c) (Const c).

Definition this_tagged_immediate (i : Z) : flambda_type :=
  this_const (Tagged_immediate i).

Definition this_naked_immediate (i : Z) : flambda_type :=
  this_const (Naked_immediate i).

(* [Type_grammar.null]. *)
Definition null : flambda_type :=
  Type_value (Ok (No_alias (Head_of_kind_value Bottom (Maybe_null None)))).

(* [Type_grammar.any_non_null_value]. *)
Definition any_non_null_value : flambda_type :=
  Type_value (Ok (No_alias (Head_of_kind_value Unknown Not_null))).

(* [More_type_creators.type_for_const]. *)
Definition type_for_const (c : const) : flambda_type :=
  match c with
  | Null => null
  | Poison k => unknown k
  | _ => this_const c
  end.

(* [Type_grammar.non_null_value]. *)
Definition non_null_value (head : head_of_kind_value_non_null)
    : flambda_type :=
  Type_value (Ok (No_alias (Head_of_kind_value (Ok head) Not_null))).

(* [Type_grammar.create_variant]. *)
Definition create_variant (is_unique : bool)
    (immediates : Or_unknown.t flambda_type)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) : flambda_type :=
  non_null_value (Variant None immediates None blocks extensions is_unique).

(* [Type_grammar.mutable_block]. *)
Definition mutable_block (alloc_mode : Alloc_mode.t) : flambda_type :=
  non_null_value (Mutable_block alloc_mode).

(* [Type_grammar.create_closures]. *)
Definition create_closures (alloc_mode : Alloc_mode.t)
    (by_function_slot : row_like_for_closures) : flambda_type :=
  non_null_value (Closures by_function_slot alloc_mode).

(* [Type_grammar.box_float32], [box_float], [box_int32], ...: the type
   of the contents is to be of the kind boxed, [unboxed_kind k], which
   the compiler checks. *)
Definition box_number (k : boxable_number) (contents : flambda_type)
    (alloc_mode : Alloc_mode.t) : flambda_type :=
  non_null_value (Boxed_number k contents alloc_mode).

(* [Type_grammar.this_immutable_string]. *)
Definition this_immutable_string (bytes : list Z) : flambda_type :=
  non_null_value (String [bytes]).

(* [Type_grammar.these_naked_immediates]. *)
Definition these_naked_immediates (imms : list Z) : flambda_type :=
  match imms with
  | [] => bottom (Naked_number Kind_naked_immediate)
  | [i] => this_naked_immediate i
  | _ =>
    Type_naked_immediate
      (Ok
         (No_alias (Head_of_kind_naked_immediate (Or_unknown.Known imms) [])))
  end.

(* [Type_grammar.Env_extension.empty]. *)
Definition empty_env_extension : env_extension :=
  Env_extension [].

(* [Type_grammar.Row_like_for_blocks.bottom]. *)
Definition row_like_for_blocks_bottom : row_like_for_blocks :=
  Row_like_for_blocks [] Or_bottom.Bottom Alloc_mode.Heap_or_local.

(* [Type_grammar.Row_like_for_blocks.is_bottom]. *)
Definition row_like_for_blocks_is_bottom (blocks : row_like_for_blocks)
    : bool :=
  match blocks with
  | Row_like_for_blocks [] Or_bottom.Bottom _ => true
  | Row_like_for_blocks _ _ _ => false
  end.

(* [Type_grammar.Row_like_for_blocks.create_exactly],
   [create_at_least] and [create_at_least_unknown_tag]: blocks of one tag
   and of a size that is known, or known from below; and blocks of any
   tag. *)
Definition row_like_for_blocks_create_exactly (tag : Z) (size : nat)
    (shape : block_shape) (maps_to : list flambda_type)
    (alloc_mode : Alloc_mode.t) : row_like_for_blocks :=
  Row_like_for_blocks
    [(tag,
      Or_unknown.Known
        (Row_like_case maps_to
           (Row_like_index (Row_like_index_domain.Known size) shape)
           empty_env_extension))]
    Or_bottom.Bottom alloc_mode.

Definition row_like_for_blocks_create_at_least (tag : Z) (size : nat)
    (shape : block_shape) (maps_to : list flambda_type)
    (alloc_mode : Alloc_mode.t) : row_like_for_blocks :=
  Row_like_for_blocks
    [(tag,
      Or_unknown.Known
        (Row_like_case maps_to
           (Row_like_index (Row_like_index_domain.At_least size) shape)
           empty_env_extension))]
    Or_bottom.Bottom alloc_mode.

Definition row_like_for_blocks_create_at_least_unknown_tag (size : nat)
    (shape : block_shape) (maps_to : list flambda_type)
    (alloc_mode : Alloc_mode.t) : row_like_for_blocks :=
  Row_like_for_blocks []
    (Or_bottom.Ok
       (Row_like_case maps_to
          (Row_like_index (Row_like_index_domain.At_least size) shape)
          empty_env_extension))
    alloc_mode.

(* [Type_grammar.tag_immediate]: the tagged immediates whose numbers have
   the type given, which is of kind naked immediate. *)
Definition tag_immediate (t : flambda_type) : flambda_type :=
  create_variant false (Or_unknown.Known t)
    (Or_unknown.Known row_like_for_blocks_bottom) No_extensions.

(* [More_type_creators.any_tagged_immediate]. *)
Definition any_tagged_immediate : flambda_type :=
  create_variant false Or_unknown.Unknown
    (Or_unknown.Known row_like_for_blocks_bottom) No_extensions.

(* [More_type_creators.these_tagged_immediates]. *)
Definition these_tagged_immediates (imms : list Z) : flambda_type :=
  match imms with
  | [] => bottom Value
  | [i] => this_tagged_immediate i
  | _ => tag_immediate (these_naked_immediates imms)
  end.

(* [More_type_creators.any_block]. *)
Definition any_block : flambda_type :=
  create_variant false
    (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
    Or_unknown.Unknown No_extensions.

(* [More_type_creators.blocks_with_these_tags], where it has an answer:
   the blocks of one of the tags, of which nothing more is known. *)
Definition blocks_with_these_tags (tags : list Z)
    (alloc_mode : Alloc_mode.t) : flambda_type :=
  create_variant false
    (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
    (Or_unknown.Known
       (Row_like_for_blocks
          (map (fun tag => (tag, Or_unknown.Unknown)) tags)
          Or_bottom.Bottom alloc_mode))
    No_extensions.

(* [More_type_creators.immutable_block]: the blocks of a tag with fields
   of the types given, and no others. *)
Definition immutable_block (is_unique : bool) (tag : Z) (shape : block_shape)
    (alloc_mode : Alloc_mode.t) (fields : list flambda_type) : flambda_type :=
  create_variant is_unique
    (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
    (Or_unknown.Known
       (row_like_for_blocks_create_exactly tag (length fields) shape fields
          alloc_mode))
    No_extensions.

(* [Type_grammar.is_int_for_scrutinee], [get_tag_for_block] and
   [is_null], for a scrutinee that is a name: a number that the relation
   relates to the value of the name. *)
Definition is_int_for_scrutinee (scrutinee : name) : flambda_type :=
  Type_naked_immediate
    (Ok
       (No_alias
          (Head_of_kind_naked_immediate (Or_unknown.Known [0%Z; 1%Z])
             [(Relation.Is_int, scrutinee)]))).

Definition get_tag_for_block (block : name) : flambda_type :=
  Type_naked_immediate
    (Ok
       (No_alias
          (Head_of_kind_naked_immediate Or_unknown.Unknown
             [(Relation.Get_tag, block)]))).

Definition is_null_for_scrutinee (scrutinee : name) : flambda_type :=
  Type_naked_immediate
    (Ok
       (No_alias
          (Head_of_kind_naked_immediate (Or_unknown.Known [0%Z; 1%Z])
             [(Relation.Is_null, scrutinee)]))).

(* [More_type_creators.exactly_this_closure]: the closure at a function
   slot of a set of closures with exactly the functions, the closures
   and the value slots given. *)
Definition exactly_this_closure (this_function_slot : function_slot)
    (all_function_slots_in_set : finite_map (Or_unknown.t function_type))
    (all_closure_types_in_set : finite_map flambda_type)
    (all_value_slots_in_set : finite_map flambda_type)
    (alloc_mode : Alloc_mode.t) : flambda_type :=
  create_closures alloc_mode
    (Row_like_for_closures
       [(this_function_slot,
         Row_like_case
           (Closures_entry all_function_slots_in_set all_closure_types_in_set
              all_value_slots_in_set)
           (Row_like_index
              (Row_like_index_domain.Known
                 (Set_of_closures_contents (domain all_function_slots_in_set)
                    (domain all_value_slots_in_set)))
              tt)
           empty_env_extension)]).

(* [More_type_creators.at_least_this_closure]: the closure at a function
   slot of a set of closures with at least the functions, the closures
   and the value slots given. *)
Definition at_least_this_closure (this_function_slot : function_slot)
    (at_least_these_function_slots : finite_map (Or_unknown.t function_type))
    (at_least_these_closure_types : finite_map flambda_type)
    (at_least_these_value_slots : finite_map flambda_type)
    (alloc_mode : Alloc_mode.t) : flambda_type :=
  create_closures alloc_mode
    (Row_like_for_closures
       [(this_function_slot,
         Row_like_case
           (Closures_entry at_least_these_function_slots
              at_least_these_closure_types at_least_these_value_slots)
           (Row_like_index
              (Row_like_index_domain.At_least
                 (Set_of_closures_contents
                    (domain at_least_these_function_slots)
                    (domain at_least_these_value_slots)))
              tt)
           empty_env_extension)]).

(* ======================================================================== *)
(* Free variables                                                           *)
(* ======================================================================== *)

(* [Type_grammar.free_names], of which only the variables are kept: the
   variables that a type mentions, on whose values its meaning depends.
   They are the variable of an alias, the variables that hold the results
   of [Is_int], [Get_tag] and [Is_null], the variables of which a naked
   immediate is the result of a relation, and the variables that an
   environment extension has an equation on. A variable may be listed
   more than once, and the variables of a binding of a map that is
   shadowed are listed too. *)

Definition free_variables_of_simple (s : simple) : list variable :=
  match s with
  | Var x => [x]
  | Symbol _ | Const _ => []
  end.

Definition free_variables_of_name (n : name) : list variable :=
  free_variables_of_simple (simple_of_name n).

Definition free_variables_of_option (x : option variable) : list variable :=
  match x with
  | Some x => [x]
  | None => []
  end.

Definition free_variables_of_descr {head : Type}
    (free_variables_of_head : head -> list variable) (ty : type_descr head)
    : list variable :=
  match ty with
  | Ok (No_alias head) => free_variables_of_head head
  | Ok (Equals s) => free_variables_of_simple s
  | Unknown | Bottom => []
  end.

Definition free_variables_of_naked_immediate
    (head : head_of_kind_naked_immediate) : list variable :=
  flat_map (fun relation => free_variables_of_name (snd relation))
    (inverse_relations head).

(* The two arguments stand for the functions being defined below. *)
Definition free_variables_of_block_case
    (free_variables : flambda_type -> list variable)
    (free_variables_of_env_extension : env_extension -> list variable)
    (case : row_like_block_case) : list variable :=
  flat_map free_variables (case_maps_to case)
  ++ free_variables_of_env_extension (case_env_extension case).

Fixpoint free_variables (t : flambda_type) : list variable :=
  match t with
  | Type_value ty =>
    free_variables_of_descr free_variables_of_head_of_kind_value ty
  | Type_naked_immediate ty =>
    free_variables_of_descr free_variables_of_naked_immediate ty
  | Type_naked_number _ ty => free_variables_of_descr (fun _ => []) ty
  | Type_rec_info ty => free_variables_of_descr (fun _ => []) ty
  | Type_region ty => free_variables_of_descr (fun _ => []) ty
  end

with free_variables_of_head_of_kind_value (head : head_of_kind_value)
    : list variable :=
  match head with
  | Head_of_kind_value non_null is_null =>
    match non_null with
    | Ok head => free_variables_of_head_of_kind_value_non_null head
    | Unknown | Bottom => []
    end
    ++ match is_null with
       | Not_null => []
       | Maybe_null is_null => free_variables_of_option is_null
       end
  end

with free_variables_of_head_of_kind_value_non_null
    (head : head_of_kind_value_non_null) : list variable :=
  match head with
  | Variant is_int immediates get_tag blocks extensions _ =>
    free_variables_of_option is_int
    ++ match immediates with
       | Or_unknown.Known ty => free_variables ty
       | Or_unknown.Unknown => []
       end
    ++ free_variables_of_option get_tag
    ++ match blocks with
       | Or_unknown.Known blocks =>
         free_variables_of_row_like_for_blocks blocks
       | Or_unknown.Unknown => []
       end
    ++ match extensions with
       | No_extensions => []
       | Ext when_immediate when_block =>
         free_variables_of_env_extension when_immediate
         ++ free_variables_of_env_extension when_block
       end
  | Mutable_block _ | String _ => []
  | Boxed_number _ contents _ => free_variables contents
  | Closures by_function_slot _ =>
    free_variables_of_row_like_for_closures by_function_slot
  end

with free_variables_of_row_like_for_blocks (blocks : row_like_for_blocks)
    : list variable :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    flat_map
      (fun case =>
         match snd case with
         | Or_unknown.Known case =>
           free_variables_of_block_case free_variables
             free_variables_of_env_extension case
         | Or_unknown.Unknown => []
         end)
      known_tags
    ++ match other_tags with
       | Or_bottom.Ok case =>
         free_variables_of_block_case free_variables
           free_variables_of_env_extension case
       | Or_bottom.Bottom => []
       end
  end

with free_variables_of_row_like_for_closures (closures : row_like_for_closures)
    : list variable :=
  match closures with
  | Row_like_for_closures known_closures =>
    flat_map
      (fun case =>
         free_variables_of_closures_entry (case_maps_to (snd case))
         ++ free_variables_of_env_extension (case_env_extension (snd case)))
      known_closures
  end

with free_variables_of_closures_entry (entry : closures_entry)
    : list variable :=
  match entry with
  | Closures_entry _ closure_types value_slot_types =>
    flat_map (fun binding => free_variables (snd binding)) closure_types
    ++ flat_map (fun binding => free_variables (snd binding)) value_slot_types
  end

with free_variables_of_env_extension (extension : env_extension)
    : list variable :=
  match extension with
  | Env_extension equations =>
    flat_map
      (fun binding =>
         free_variables_of_name (fst binding) ++ free_variables (snd binding))
      equations
  end.

(* ======================================================================== *)
(* Aliases, with their kinds                                                *)
(* ======================================================================== *)

(* The aliases of a type, each with the kind at which it stands: the
   simple of every alias in the type, with the kind of the type that the
   alias is, and the variables that hold the results of [Is_int],
   [Get_tag] and [Is_null], which are naked immediates. It is what the
   join of types gives the join of [Simple]s, with the kind at which it
   gives it. The recursion is that of [free_variables]. A pair may be
   listed more than once, and the pairs of a binding of a map that is
   shadowed are listed too.

   Not listed, though each is a name at a kind: the names of which a
   naked immediate is the result of a relation, which are of kind value,
   and the names that an environment extension has an equation on, each
   of the kind of its type ([equation_names_with_kinds], at the end of
   this file, lists those).

   A type says at which kind each of these stands and not that it is of
   that kind: nothing here, nor in the compiler, checks one against the
   other, and that they agree is no part of a type being well formed
   (FlambdaTypeWellFormedSpec.v). *)

Definition aliases_of_descr {head : Type} (k : flambda_kind)
    (aliases_of_head : head -> list (simple * flambda_kind))
    (ty : type_descr head) : list (simple * flambda_kind) :=
  match ty with
  | Ok (No_alias head) => aliases_of_head head
  | Ok (Equals s) => [(s, k)]
  | Unknown | Bottom => []
  end.

Definition aliases_of_relation (x : option variable)
    : list (simple * flambda_kind) :=
  match x with
  | Some x => [(Var x, Naked_number Kind_naked_immediate)]
  | None => []
  end.

(* The two arguments stand for the functions being defined below. *)
Definition aliases_of_block_case
    (aliases_with_kinds : flambda_type -> list (simple * flambda_kind))
    (aliases_of_env_extension : env_extension -> list (simple * flambda_kind))
    (case : row_like_block_case) : list (simple * flambda_kind) :=
  flat_map aliases_with_kinds (case_maps_to case)
  ++ aliases_of_env_extension (case_env_extension case).

Fixpoint aliases_with_kinds (t : flambda_type)
    : list (simple * flambda_kind) :=
  match t with
  | Type_value ty =>
    aliases_of_descr Value aliases_of_head_of_kind_value ty
  | Type_naked_immediate ty =>
    aliases_of_descr (Naked_number Kind_naked_immediate) (fun _ => []) ty
  | Type_naked_number k ty =>
    aliases_of_descr (Naked_number k) (fun _ => []) ty
  | Type_rec_info ty => aliases_of_descr Rec_info (fun _ => []) ty
  | Type_region ty => aliases_of_descr Region (fun _ => []) ty
  end

with aliases_of_head_of_kind_value (head : head_of_kind_value)
    : list (simple * flambda_kind) :=
  match head with
  | Head_of_kind_value non_null is_null =>
    match non_null with
    | Ok head => aliases_of_head_of_kind_value_non_null head
    | Unknown | Bottom => []
    end
    ++ match is_null with
       | Not_null => []
       | Maybe_null is_null => aliases_of_relation is_null
       end
  end

with aliases_of_head_of_kind_value_non_null
    (head : head_of_kind_value_non_null) : list (simple * flambda_kind) :=
  match head with
  | Variant is_int immediates get_tag blocks extensions _ =>
    aliases_of_relation is_int
    ++ match immediates with
       | Or_unknown.Known ty => aliases_with_kinds ty
       | Or_unknown.Unknown => []
       end
    ++ aliases_of_relation get_tag
    ++ match blocks with
       | Or_unknown.Known blocks => aliases_of_row_like_for_blocks blocks
       | Or_unknown.Unknown => []
       end
    ++ match extensions with
       | No_extensions => []
       | Ext when_immediate when_block =>
         aliases_of_env_extension when_immediate
         ++ aliases_of_env_extension when_block
       end
  | Mutable_block _ | String _ => []
  | Boxed_number _ contents _ => aliases_with_kinds contents
  | Closures by_function_slot _ =>
    aliases_of_row_like_for_closures by_function_slot
  end

with aliases_of_row_like_for_blocks (blocks : row_like_for_blocks)
    : list (simple * flambda_kind) :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    flat_map
      (fun case =>
         match snd case with
         | Or_unknown.Known case =>
           aliases_of_block_case aliases_with_kinds aliases_of_env_extension
             case
         | Or_unknown.Unknown => []
         end)
      known_tags
    ++ match other_tags with
       | Or_bottom.Ok case =>
         aliases_of_block_case aliases_with_kinds aliases_of_env_extension
           case
       | Or_bottom.Bottom => []
       end
  end

with aliases_of_row_like_for_closures (closures : row_like_for_closures)
    : list (simple * flambda_kind) :=
  match closures with
  | Row_like_for_closures known_closures =>
    flat_map
      (fun case =>
         aliases_of_closures_entry (case_maps_to (snd case))
         ++ aliases_of_env_extension (case_env_extension (snd case)))
      known_closures
  end

with aliases_of_closures_entry (entry : closures_entry)
    : list (simple * flambda_kind) :=
  match entry with
  | Closures_entry _ closure_types value_slot_types =>
    flat_map (fun binding => aliases_with_kinds (snd binding)) closure_types
    ++ flat_map (fun binding => aliases_with_kinds (snd binding))
         value_slot_types
  end

with aliases_of_env_extension (extension : env_extension)
    : list (simple * flambda_kind) :=
  match extension with
  | Env_extension equations =>
    flat_map (fun binding => aliases_with_kinds (snd binding)) equations
  end.

(* ======================================================================== *)
(* What meet and the joins use of the grammar                               *)
(* ======================================================================== *)

(* Definitions that the workstreams of meet and of the joins wrote
   against the compiler and that belong here; they are under the names
   those workstreams gave them. *)

(* [Flambda_kind.Block_shape.element_kind]: the kind of the field at an
   index of a block of a shape. [None] is the exception of the
   compiler for an index outside the fields of a mixed shape. *)
Definition block_shape_element_kind (shape : block_shape) (index : nat)
    : option flambda_kind :=
  match shape with
  | Scannable Value_only => Some Value
  | Scannable (Mixed_record value_prefix_size flat_suffix) =>
    if Nat.ltb index value_prefix_size
    then Some Value
    else
      option_map Naked_number
        (nth_error flat_suffix (index - value_prefix_size))
  | Float_record => Some (Naked_number Kind_naked_float)
  end.

(* [Reg_width_const.naked_immediate], [naked_float32], ...: the constant
   that a number of a kind is. *)
Definition const_of_naked_number (k : naked_number_kind) (i : Z) : const :=
  match k with
  | Kind_naked_immediate => Naked_immediate i
  | Kind_naked_float32 => Naked_float32 i
  | Kind_naked_float => Naked_float i
  | Kind_naked_int8 => Naked_int8 i
  | Kind_naked_int16 => Naked_int16 i
  | Kind_naked_int32 => Naked_int32 i
  | Kind_naked_int64 => Naked_int64 i
  | Kind_naked_nativeint => Naked_nativeint i
  | Kind_naked_vec128 => Naked_vec128 i
  | Kind_naked_vec256 => Naked_vec256 i
  | Kind_naked_vec512 => Naked_vec512 i
  | Kind_naked_mask => Naked_mask i
  end.

(* [Simple.must_be_const]. *)
Definition simple_must_be_const (s : simple) : option const :=
  match s with
  | Const c => Some c
  | Var _ | Symbol _ => None
  end.

(* [Type_grammar.must_be_singleton]: the constant that a type has for
   only value, if the type shows it. The outer [None] is a fatal error
   of the compiler ("Immediates case returned wrong kind of constant").
   A set of numbers is a list here, and has one number if all its
   numbers are the same one ([Keyed.get_singleton]). *)
Fixpoint must_be_singleton (t : flambda_type) : option (option const) :=
  match t with
  | Type_value ty =>
    match ty with
    | Unknown | Bottom => Some None
    | Ok (Equals simple) => Some (simple_must_be_const simple)
    | Ok
        (No_alias
           (Head_of_kind_value (Ok (Variant _ immediates _ blocks _ _))
              Not_null)) =>
      match blocks with
      | Or_unknown.Unknown => Some None
      | Or_unknown.Known blocks =>
        if negb (row_like_for_blocks_is_bottom blocks)
        then Some None
        else
          match immediates with
          | Or_unknown.Unknown => Some None
          | Or_unknown.Known immediates =>
            match must_be_singleton immediates with
            | None => None
            | Some None => Some None
            | Some (Some (Naked_immediate i)) =>
              Some (Some (Tagged_immediate i))
            | Some (Some (Poison (Naked_number Kind_naked_immediate))) =>
              Some (Some (Poison Value))
            | Some (Some _) => None
            end
          end
      end
    | Ok (No_alias _) => Some None
    end
  | Type_naked_immediate ty =>
    match ty with
    | Unknown | Bottom => Some None
    | Ok (Equals simple) => Some (simple_must_be_const simple)
    | Ok (No_alias head) =>
      match naked_immediates head with
      | Or_unknown.Unknown => Some None
      | Or_unknown.Known numbers =>
        Some (option_map Naked_immediate (Keyed.get_singleton Z.eqb numbers))
      end
    end
  | Type_naked_number k ty =>
    match ty with
    | Unknown | Bottom => Some None
    | Ok (Equals simple) => Some (simple_must_be_const simple)
    | Ok (No_alias numbers) =>
      Some
        (option_map (const_of_naked_number k)
           (Keyed.get_singleton Z.eqb numbers))
    end
  | Type_rec_info _ | Type_region _ => Some None
  end.

(* [Type_grammar.Head_of_kind_value.null]: the head of [null]. *)
Definition head_of_kind_value_null : head_of_kind_value :=
  Head_of_kind_value Bottom (Maybe_null None).

(* [Type_grammar.Head_of_kind_value.create_tagged_immediate]: the head
   of [tag_immediate (this_naked_immediate imm)]. *)
Definition head_of_kind_value_tagged_immediate (imm : Z)
    : head_of_kind_value :=
  Head_of_kind_value
    (Ok
       (Variant None (Or_unknown.Known (this_naked_immediate imm)) None
          (Or_unknown.Known row_like_for_blocks_bottom) No_extensions false))
    Not_null.

(* [Set_of_closures_contents.equal], [subset], [union] and [inter]. A
   set of slots is a list, compared as a set. *)
Definition set_of_closures_contents_equal
    (contents contents' : set_of_closures_contents) : bool :=
  Keyed.same_keys_b Pos.eqb (contents_closures contents)
    (contents_closures contents')
  && Keyed.same_keys_b Pos.eqb (contents_value_slots contents)
       (contents_value_slots contents').

Definition set_of_closures_contents_subset
    (contents contents' : set_of_closures_contents) : bool :=
  all_among (contents_closures contents) (contents_closures contents')
  && all_among (contents_value_slots contents)
       (contents_value_slots contents').

Definition set_of_closures_contents_union
    (contents contents' : set_of_closures_contents)
    : set_of_closures_contents :=
  Set_of_closures_contents
    (dedup (contents_closures contents ++ contents_closures contents'))
    (dedup (contents_value_slots contents ++ contents_value_slots contents')).

Definition set_of_closures_contents_inter
    (contents contents' : set_of_closures_contents)
    : set_of_closures_contents :=
  Set_of_closures_contents
    (Keyed.inter Pos.eqb (contents_closures contents)
       (contents_closures contents'))
    (Keyed.inter Pos.eqb (contents_value_slots contents)
       (contents_value_slots contents')).

(* ======================================================================== *)
(* The names of equations, with their kinds                                 *)
(* ======================================================================== *)

(* The names that have an equation in an environment extension inside a
   type, at any depth, each with the kind of the type of its equation.
   They are what [aliases_with_kinds] leaves out of an extension: with
   them, a type says at which kind each name of its equations stands.
   The recursion is that of [free_variables]. A pair may be listed more
   than once, and the pairs of a binding of a map that is shadowed are
   listed too. The compiler has no such function; meet needs it to say
   that adding the extension of a case to an environment keeps the
   kinds of the environment. *)

(* The two arguments stand for the functions being defined below. *)
Definition equation_names_of_block_case
    (equation_names_with_kinds : flambda_type -> list (name * flambda_kind))
    (equation_names_of_env_extension
      : env_extension -> list (name * flambda_kind))
    (case : row_like_block_case) : list (name * flambda_kind) :=
  flat_map equation_names_with_kinds (case_maps_to case)
  ++ equation_names_of_env_extension (case_env_extension case).

Fixpoint equation_names_with_kinds (t : flambda_type)
    : list (name * flambda_kind) :=
  match t with
  | Type_value (Ok (No_alias head)) =>
    equation_names_of_head_of_kind_value head
  | Type_value _ | Type_naked_immediate _ | Type_naked_number _ _
  | Type_rec_info _ | Type_region _ => []
  end

with equation_names_of_head_of_kind_value (head : head_of_kind_value)
    : list (name * flambda_kind) :=
  match head with
  | Head_of_kind_value (Ok head) _ =>
    equation_names_of_head_of_kind_value_non_null head
  | Head_of_kind_value _ _ => []
  end

with equation_names_of_head_of_kind_value_non_null
    (head : head_of_kind_value_non_null) : list (name * flambda_kind) :=
  match head with
  | Variant _ immediates _ blocks extensions _ =>
    match immediates with
    | Or_unknown.Known ty => equation_names_with_kinds ty
    | Or_unknown.Unknown => []
    end
    ++ match blocks with
       | Or_unknown.Known blocks =>
         equation_names_of_row_like_for_blocks blocks
       | Or_unknown.Unknown => []
       end
    ++ match extensions with
       | No_extensions => []
       | Ext when_immediate when_block =>
         equation_names_of_env_extension when_immediate
         ++ equation_names_of_env_extension when_block
       end
  | Mutable_block _ | String _ => []
  | Boxed_number _ contents _ => equation_names_with_kinds contents
  | Closures by_function_slot _ =>
    equation_names_of_row_like_for_closures by_function_slot
  end

with equation_names_of_row_like_for_blocks (blocks : row_like_for_blocks)
    : list (name * flambda_kind) :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    flat_map
      (fun case =>
         match snd case with
         | Or_unknown.Known case =>
           equation_names_of_block_case equation_names_with_kinds
             equation_names_of_env_extension case
         | Or_unknown.Unknown => []
         end)
      known_tags
    ++ match other_tags with
       | Or_bottom.Ok case =>
         equation_names_of_block_case equation_names_with_kinds
           equation_names_of_env_extension case
       | Or_bottom.Bottom => []
       end
  end

with equation_names_of_row_like_for_closures
    (closures : row_like_for_closures) : list (name * flambda_kind) :=
  match closures with
  | Row_like_for_closures known_closures =>
    flat_map
      (fun case =>
         equation_names_of_closures_entry (case_maps_to (snd case))
         ++ equation_names_of_env_extension (case_env_extension (snd case)))
      known_closures
  end

with equation_names_of_closures_entry (entry : closures_entry)
    : list (name * flambda_kind) :=
  match entry with
  | Closures_entry _ closure_types value_slot_types =>
    flat_map (fun binding => equation_names_with_kinds (snd binding))
      closure_types
    ++ flat_map (fun binding => equation_names_with_kinds (snd binding))
         value_slot_types
  end

with equation_names_of_env_extension (extension : env_extension)
    : list (name * flambda_kind) :=
  match extension with
  | Env_extension equations =>
    flat_map
      (fun binding =>
         (fst binding, kind (snd binding))
         :: equation_names_with_kinds (snd binding))
      equations
  end.

(* ======================================================================== *)
(* More creation functions                                                  *)
(* ======================================================================== *)

(* The other creation functions of more_type_creators.ml and of
   type_grammar.ml: those that Simplify gives to the results of
   primitives, and to what a primitive takes apart. Each is made of the
   functions above. Where the compiler has one function for each kind
   of number ([these_naked_floats], [this_boxed_int64],
   [boxed_nativeint_alias_to], ...) there is one function here, which
   takes the kind.

   Not here: [tag_int8], [tag_int16] and their [alias_to] (the machine
   has no primitive that makes a tagged immediate of a small integer);
   [variant], [Row_like_for_blocks.create_exactly_multiple] and
   [unknown_with_subkind] at a subkind that says something (the model
   has no subkinds); the types of arrays; [check_equation], which
   answers nothing. *)

(* [More_type_creators.unknown_like] and [bottom_like]. *)
Definition unknown_like (t : flambda_type) : flambda_type :=
  unknown (kind t).

Definition bottom_like (t : flambda_type) : flambda_type :=
  bottom (kind t).

(* [More_type_creators.unknown_from_shape]. [None] is the exception of
   [Flambda_kind.Block_shape.element_kind]. *)
Definition unknown_from_shape (shape : block_shape) (index : nat)
    : option flambda_type :=
  option_map unknown (block_shape_element_kind shape index).

(* [More_type_creators.unknown_with_subkind], at the subkind of a
   kind that says nothing ([Flambda_kind.With_subkind.anything]),
   which is all that the model has. At the kind of values that is not
   [unknown Value] but a head that says nothing: a value that is not
   null, of which nothing is known, or null. The two have the same
   values. The compiler gives a smaller type for a subkind that says
   more (a value that is not null, a boxed number, a variant with
   some constructors, ...). *)
Definition unknown_with_subkind (k : flambda_kind) : flambda_type :=
  match k with
  | Value =>
    Type_value (Ok (No_alias (Head_of_kind_value Unknown (Maybe_null None))))
  | Naked_number _ | Region | Rec_info => unknown k
  end.

(* [More_type_creators.unknown_types_from_arity]: such a type for each
   kind of an arity. *)
Definition unknown_types_from_arity (a : arity) : list flambda_type :=
  map unknown_with_subkind (unarize a).

(* [More_type_creators.kind_for_const]. *)
Definition kind_for_const (c : const) : flambda_kind :=
  kind (type_for_const c).

(* [More_type_creators.is_alias_of_name]. *)
Definition is_alias_of_name (t : flambda_type) (n : name) : bool :=
  match get_alias_opt t with
  | Some (Var x) => name_eqb (Name_var x) n
  | Some (Symbol s) => name_eqb (Name_symbol s) n
  | Some (Const _) | None => false
  end.

(* [Type_grammar.these_naked_float32s], [these_naked_floats], ...,
   [these_naked_masks], and [these_naked_immediates] at the kind of the
   immediates: the numbers of a kind that are in a set. As
   [these_naked_immediates] above, a set that is written with one
   number is an alias to the constant. *)
Definition these_naked_numbers (k : naked_number_kind) (numbers : list Z)
    : flambda_type :=
  match k with
  | Kind_naked_immediate => these_naked_immediates numbers
  | _ =>
    match numbers with
    | [] => bottom (Naked_number k)
    | [i] => this_const (const_of_naked_number k i)
    | _ => Type_naked_number k (Ok (No_alias numbers))
    end
  end.

(* [Target_ocaml_int.all_bools]. *)
Definition all_bools : list Z := [0%Z; 1%Z].

(* [More_type_creators.any_tagged_bool] and [any_naked_bool]. *)
Definition any_tagged_bool : flambda_type :=
  these_tagged_immediates all_bools.

Definition any_naked_bool : flambda_type :=
  these_naked_immediates all_bools.

(* [More_type_creators.any_tagged_immediate_or_null]. *)
Definition any_tagged_immediate_or_null : flambda_type :=
  Type_value
    (Ok
       (No_alias
          (Head_of_kind_value
             (Ok
                (Variant None Or_unknown.Unknown None
                   (Or_unknown.Known row_like_for_blocks_bottom)
                   No_extensions false))
             (Maybe_null None)))).

(* [Type_grammar.tagged_immediate_alias_to]: the tagged immediate
   whose number a variable holds. *)
Definition tagged_immediate_alias_to (naked_immediate : variable)
    : flambda_type :=
  tag_immediate
    (alias_type_of (Naked_number Kind_naked_immediate) (Var naked_immediate)).

(* The kind of the numbers that a boxable number boxes:
   [unboxed_kind k] is [Naked_number (unboxed_number_kind k)]. *)
Definition unboxed_number_kind (k : boxable_number) : naked_number_kind :=
  match k with
  | Boxable_naked_float32 => Kind_naked_float32
  | Boxable_naked_float => Kind_naked_float
  | Boxable_naked_int32 => Kind_naked_int32
  | Boxable_naked_int64 => Kind_naked_int64
  | Boxable_naked_nativeint => Kind_naked_nativeint
  | Boxable_naked_vec128 => Kind_naked_vec128
  | Boxable_naked_vec256 => Kind_naked_vec256
  | Boxable_naked_vec512 => Kind_naked_vec512
  | Boxable_naked_mask => Kind_naked_mask
  end.

(* [Type_grammar.boxed_float32_alias_to], [boxed_float_alias_to], ...,
   [boxed_mask_alias_to]: the boxed number whose contents a variable
   holds. *)
Definition boxed_number_alias_to (k : boxable_number)
    (naked_number : variable) (alloc_mode : Alloc_mode.t) : flambda_type :=
  box_number k (alias_type_of (unboxed_kind k) (Var naked_number)) alloc_mode.

(* [More_type_creators.this_boxed_float32], ..., [this_boxed_mask]. *)
Definition this_boxed_number (k : boxable_number) (i : Z)
    (alloc_mode : Alloc_mode.t) : flambda_type :=
  box_number k (this_const (const_of_naked_number (unboxed_number_kind k) i))
    alloc_mode.

(* [More_type_creators.these_boxed_float32s], [these_boxed_floats],
   [these_boxed_int32s], [these_boxed_int64s] and
   [these_boxed_nativeints]; the compiler has none for the vectors and
   the masks. *)
Definition these_boxed_numbers (k : boxable_number) (numbers : list Z)
    (alloc_mode : Alloc_mode.t) : flambda_type :=
  box_number k (these_naked_numbers (unboxed_number_kind k) numbers)
    alloc_mode.

(* [More_type_creators.any_boxed_float32], [any_boxed_float],
   [any_boxed_int32], [any_boxed_int64] and [any_boxed_nativeint]. *)
Definition any_boxed_number (k : boxable_number) : flambda_type :=
  box_number k (unknown (unboxed_kind k)) Alloc_mode.Heap_or_local.

(* [Type_grammar.Row_like_for_blocks.open_or_closed]. *)
Module Open_or_closed.
  Inductive t : Type :=
    | Open (tag : Or_unknown.t Z)
    | Closed (tag : Z).
End Open_or_closed.

(* [Type_grammar.Row_like_for_blocks.create]: blocks of one case, of
   the size of the product, exactly or at least. A record of floats
   has the tag of one, whatever is known of the tag. [None] is a fatal
   error of the compiler: a tag that the shape cannot have. The
   compiler also checks, when its invariants are checked, that the
   types are of the kinds that the shape gives ([check_field_tys]).
   That check is left out here, and it is no part of well-formedness
   either (FlambdaTypeWellFormedSpec.v says why). A field of a block
   of the shape has no type of another kind, so a product that breaks
   it admits no block. *)
Definition row_like_for_blocks_create (shape : block_shape)
    (field_tys : list flambda_type) (open_or_closed : Open_or_closed.t)
    (alloc_mode : Alloc_mode.t) : option row_like_for_blocks :=
  let tag : Or_unknown.t Z :=
    match open_or_closed with
    | Open_or_closed.Open tag => tag
    | Open_or_closed.Closed tag => Or_unknown.Known tag
    end
  in
  let tag : option (Or_unknown.t Z) :=
    match tag with
    | Or_unknown.Unknown =>
      match shape with
      | Scannable _ => Some Or_unknown.Unknown
      | Float_record => Some (Or_unknown.Known double_array_tag)
      end
    | Or_unknown.Known tag =>
      match shape with
      | Scannable _ =>
        if Z.ltb tag no_scan_tag then Some (Or_unknown.Known tag) else None
      | Float_record =>
        if Z.eqb tag double_array_tag
        then Some (Or_unknown.Known tag)
        else None
      end
    end
  in
  let size := length field_tys in
  match tag, open_or_closed with
  | None, _ => None
  | Some (Or_unknown.Known tag), Open_or_closed.Open _ =>
    Some
      (row_like_for_blocks_create_at_least tag size shape field_tys alloc_mode)
  | Some Or_unknown.Unknown, Open_or_closed.Open _ =>
    Some
      (row_like_for_blocks_create_at_least_unknown_tag size shape field_tys
         alloc_mode)
  | Some (Or_unknown.Known tag), Open_or_closed.Closed _ =>
    Some
      (row_like_for_blocks_create_exactly tag size shape field_tys alloc_mode)
  | Some Or_unknown.Unknown, Open_or_closed.Closed _ => None
  end.

(* An unknown type for each of the first [n] fields of a block of a
   shape. [None] where the shape does not give the kind of one. *)
Fixpoint unknown_types_from_shape (shape : block_shape) (n : nat)
    : option (list flambda_type) :=
  match n with
  | O => Some []
  | S index =>
    match
      unknown_types_from_shape shape index, unknown_from_shape shape index
    with
    | Some before, Some ty => Some (before ++ [ty])
    | _, _ => None
    end
  end.

(* The types of the first [n] fields of a block of which a variable
   holds the field before the [n]th: unknown at the kind that the
   shape gives each, but the last, which is an alias. *)
Definition size_at_least_field_tys (shape : block_shape)
    (field_n_minus_one : variable) (n : nat) : option (list flambda_type) :=
  match n with
  | O => Some []
  | S index =>
    match
      unknown_types_from_shape shape index,
      block_shape_element_kind shape index
    with
    | Some before, Some field_kind =>
      Some (before ++ [alias_type_of field_kind (Var field_n_minus_one)])
    | _, _ => None
    end
  end.

(* [More_type_creators.immutable_block_with_size_at_least]: the blocks
   of at least [n] fields, of a tag if it is known and of a shape, of
   which a variable holds the field at [n - 1]. This is what a load of
   that field says of the block. [None] where the compiler stops: a
   field outside the shape, or a tag that the shape cannot have. *)
Definition immutable_block_with_size_at_least (tag : Or_unknown.t Z)
    (n : nat) (shape : block_shape) (field_n_minus_one : variable)
    : option flambda_type :=
  match size_at_least_field_tys shape field_n_minus_one n with
  | None => None
  | Some field_tys =>
    match
      row_like_for_blocks_create shape field_tys (Open_or_closed.Open tag)
        Alloc_mode.Heap_or_local
    with
    | None => None
    | Some blocks =>
      Some
        (create_variant false
           (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
           (Or_unknown.Known blocks) No_extensions)
    end
  end.

(* [More_type_creators.closure_with_at_least_these_function_slots]:
   the closure at a function slot of a set of which each of the
   function slots given is held by the simple given with it. This is
   what a projection of a function slot says of the closure projected
   from. *)
Definition closure_with_at_least_these_function_slots
    (this_function_slot : function_slot)
    (function_slots_and_bindings : finite_map simple) : flambda_type :=
  at_least_this_closure this_function_slot
    (map_values (fun _ => Or_unknown.Unknown) function_slots_and_bindings)
    (map_values (alias_type_of Value) function_slots_and_bindings) []
    Alloc_mode.Heap_or_local.

(* [More_type_creators.static_closure_with_this_code]: a closure whose
   own function has the code given and, if a symbol is given, from
   which the projection of its own function slot, where it has a
   value, is the symbol. It has a value where what the symbols are
   bound to is valid ([well_formed_closures_case_closure_type_statement],
   in FlambdaTypeWellFormedSpec.v). *)
Definition static_closure_with_this_code (this_function_slot : function_slot)
    (closure_symbol : option symbol) (code : code_id) : flambda_type :=
  at_least_this_closure this_function_slot
    [(this_function_slot, Or_unknown.Known (Function_type code))]
    [(this_function_slot,
      match closure_symbol with
      | Some closure_symbol => alias_type_of Value (Symbol closure_symbol)
      | None => unknown Value
      end)]
    [] Alloc_mode.Heap_or_local.

(* [More_type_creators.closure_with_at_least_these_value_slots]: the
   closure at a function slot of a set of which each of the value
   slots given is held by the variable given with it, at the kind
   given. *)
Definition closure_with_at_least_these_value_slots
    (this_function_slot : function_slot)
    (value_slots : finite_map (variable * flambda_kind)) : flambda_type :=
  at_least_this_closure this_function_slot [] []
    (map_values
       (fun value_slot => alias_type_of (snd value_slot) (Var (fst value_slot)))
       value_slots)
    Alloc_mode.Heap_or_local.

(* [More_type_creators.closure_with_at_least_this_value_slot]: what a
   projection of a value slot says of the closure projected from. *)
Definition closure_with_at_least_this_value_slot
    (this_function_slot : function_slot) (value_slot : value_slot)
    (value_slot_var : variable) (value_slot_kind : flambda_kind)
    : flambda_type :=
  closure_with_at_least_these_value_slots this_function_slot
    [(value_slot, (value_slot_var, value_slot_kind))].
