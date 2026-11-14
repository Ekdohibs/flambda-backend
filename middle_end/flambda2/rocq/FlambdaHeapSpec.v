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

(* The primitives of the heap: allocation, what each primitive does to a
   heap, and the compiler's classification of primitives by effects and
   coeffects. It comes before FlambdaSemanticsSpec.v, whose machine runs
   these primitives (FlambdaSemanticsSpec.exec_named, by [exec_prim]).
   FlambdaHeapMachineSpec.v has the prototype that ran them before that
   machine did.

   This file is to be read with the request (requests/C4-heap.md). It has
   - allocation, and what the primitives of the heap compute
     ([exec_prim], over FlambdaPrimitiveSpec.eval_prim, which is
     unchanged). The mutable arrays of values are cells too
     ([Cell_array]): what their four primitives do to a cell is here,
     over the functions of the section "Arrays" of
     FlambdaPrimitiveSpec.v, which say what they compute on the
     elements;
   - the compiler's classification of primitives by effects and
     coeffects ([effects_of_prim], [coeffects_of_prim]), against
     [Flambda_primitive.effects_and_coeffects], and the laws that tie it
     to [exec_prim];
   - what a run may do to a heap ([heap_evolves]);
   - when a memory is closed: no location dangles in it
     ([memory_closed]);
   - when two heaps are alike up to a relation between values
     ([heap_alike]), and that what a primitive does on one it does on
     the other ([exec_prim_related_statement]): what a pass that
     changes values is proved with.

   What is proved is in FlambdaHeapProofs.v: every [_statement] here. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* Allocation                                                               *)
(* ======================================================================== *)

(* A location that the heap does not bind: one greater than all those
   it binds. It is FlambdaNameAbstractionSpec.fresh_name of the domain of
   the heap ([FlambdaHeapMachineSpec.fresh_location_fresh_name_statement]),
   written out because that file comes after the machine, which runs the
   primitives of this one. Nothing is to look at the number a location
   is: only [fresh_location_statement] and
   [fresh_location_keys_statement] are to be used of this function. *)
Definition fresh_location (h : heap) : location :=
  Pos.succ (fold_right Pos.max xH (domain h)).

Definition fresh_location_statement : Prop :=
  ∀ h : heap, lookup (fresh_location h) h = None.

(* ======================================================================== *)
(* The primitives of the heap                                               *)
(* ======================================================================== *)

(* The field [field] of a block of values with this tag and these fields,
   under the checks of FlambdaPrimitiveSpec.eval_block_load. *)
Definition load_field (kind : block_access_kind) (field : Z) (block_tag : Z)
    (fields : list value) : option value :=
  match kind with
  | Block_access_values tag size field_kind =>
    if known_is tag block_tag
       && known_is size (Z.of_nat (length fields))
       && (0 <=? field)
    then
      let* loaded := nth_error fields (Z.to_nat field) in
      if
        match field_kind with
        | Any_value => kind_eqb (kind_of_value loaded) Value
        | Immediate => is_tagged_immediate loaded
        end
      then Some loaded
      else None
    else None
  end.

(* The fields with [field] replaced by [new], under the same checks: the
   tag and the size are those the access kind gives, the field is one the
   block has, and the new value is of kind [Value], and a tagged
   immediate where the access kind says so. *)
Definition store_field (kind : block_access_kind) (field : Z) (new : value)
    (block_tag : Z) (fields : list value) : option (list value) :=
  match kind with
  | Block_access_values tag size field_kind =>
    if known_is tag block_tag
       && known_is size (Z.of_nat (length fields))
       && (0 <=? field)
       && match field_kind with
          | Any_value => kind_eqb (kind_of_value new) Value
          | Immediate => is_tagged_immediate new
          end
    then replace_nth (Z.to_nat field) new fields
    else None
  end.

Definition is_location (v : value) : bool :=
  match v with
  | Value_location _ => true
  | _ => false
  end.

(* [Phys_equal] where one of the two values is a location: option (b) of
   the request. Two locations are equal when they are the same location.
   A location differs from a tagged immediate, from an immutable
   allocated value and from an imported symbol of the class of
   identities or of data, which is an object that was there before the
   unit ran. Against an import of the class [Import_unknown] there is
   no answer, as there is none on such an import against anything
   (FlambdaPrimitiveSpec.eval_phys_equal): the model knows nothing of
   it, and says less than the program does (decision D41). Against
   anything else there is no answer: null, poison, a number that is
   not a value, a symbol that is not bound.
   Where neither value is a location the comparison is
   FlambdaPrimitiveSpec.eval_phys_equal, which answers of two tagged
   immediates and of one against an allocated value, and has no value on
   two immutable allocated values. That is a limit of the model and not
   what the compiler does; the CR is there (decision D19; decision 3 of
   requests/C4-heap.md; discrepancies C4-3 and R2-21). *)
Definition eval_phys_equal_location (statics : static_env) (v1 v2 : value)
    : option bool :=
  match v1, v2 with
  | Value_location l1, Value_location l2 => Some (Pos.eqb l1 l2)
  | Value_location _, v | v, Value_location _ =>
    if is_some (int_of_value Standard_tagged_immediate v)
       || is_allocated statics v || is_imported_pointer statics v
    then Some false
    else None
  | _, _ => None
  end.

Definition unit_value : value := Value_const (Tagged_immediate 0).

(* What a primitive of the heap does: its value and the heap after.
   - [Make_block_with_identity] binds a fresh location to a cell holding
     the arguments, which must be of kind [Value].
   - [Block_set] replaces a field of a mutable cell, and its value is
     unit. It has no value on anything else.
   - [Make_mutable_array] binds a fresh location to an array of the
     kind it carries, holding the arguments, of which there is at least
     one and each of which is a value that the kind holds
     (FlambdaPrimitiveSpec.array_made).
   - [Array_set] replaces an element of an array, and its value is
     unit (FlambdaPrimitiveSpec.array_stored: the index is in the
     bounds of the array, and the new value is one that the set kind
     takes and that the kind of the array holds). It has no value on
     anything else. *)
(* CR mshinwell: [Block_set] has no value on a block that was not made
   [Mutable]: an immutable block, a symbol, an [Immutable_unique] cell.
   The machine would write; Simplify keeps the facts it has about the
   block. The front end makes no such store: every store of lambda/ is
   into a record with a mutable field, a [ref], an object, a copy made
   by [Pduprecord] or a placeholder of tail-modulo-cons. [init] does not
   change what the store does; the only stores marked [Initialization]
   are those of tail-modulo-cons, which the conversion refuses
   (decision 4). Decision 5 of requests/C4-heap.md. *)
(* CR mshinwell: [Make_mutable_array] of no argument has no value. The
   compiler does not allocate the empty array: [Closure_conversion]
   makes a [Pmakearray] of no argument the one static empty array of
   its kind ([Static_const.empty_array]), whatever its mutability, so
   two empty arrays of a program may be physically equal, and a
   location for each would tell them apart. No term that the front end
   makes has a [Make_array] of no argument. It would be wrong if a
   pass comes to make one and the term is executed. Decided while the
   owner of the branch was away
   (middle_end/flambda2/rocq/status/P1.md). *)
Definition exec_heap_primitive (statics : static_env) (h : heap)
    (p : heap_primitive) (vs : list value) : option (value * heap) :=
  match p, vs with
  | Make_block_with_identity mut tag, vs =>
    if forallb (fun v => kind_eqb (kind_of_value v) Value) vs
    then
      let l := fresh_location h in
      Some (Value_location l, (l, Cell_block mut tag vs) :: h)
    else None
  | Block_set kind _ field, [v; new] =>
    match v with
    | Value_location l =>
      match lookup l h with
      | Some (Cell_block Cell_mutable tag fields) =>
        let* fields' := store_field kind field new tag fields in
        Some (unit_value, (l, Cell_block Cell_mutable tag fields') :: h)
      | _ => None
      end
    | _ => None
    end
  | Make_mutable_array kind, _ :: _ =>
    let* elements := array_made kind vs in
    let l := fresh_location h in
    Some (Value_location l, (l, Cell_array kind elements) :: h)
  | Array_set _ set_kind, [v; index; new] =>
    match v with
    | Value_location l =>
      match lookup l h with
      | Some (Cell_array kind elements) =>
        let* elements' := array_stored set_kind kind elements index new in
        Some (unit_value, (l, Cell_array kind elements') :: h)
      | _ => None
      end
    | _ => None
    end
  | _, _ => None
  end.

(* Whether a [Block_load] with this mark reads a cell of this
   mutability: one marked [Mutable] reads any cell, and one marked
   [Immutable] or [Immutable_unique] a cell made [Immutable_unique]. *)
Definition load_allowed (load_mut : mutability) (mut : cell_mutability)
    : bool :=
  match load_mut, mut with
  | Mutable, _ | (Immutable | Immutable_unique), Cell_immutable_unique => true
  | (Immutable | Immutable_unique), Cell_mutable => false
  end.

(* A unary primitive of FlambdaPrimitiveModel.v applied to a location: it
   reads the cell, and has no value on a location that is not bound.
   A [Block_load] reads the field as it is now, where its mark allows
   ([load_allowed]). [Get_tag] is the tag of the cell; [Is_int] and
   [Is_null] answer no; [Opaque_identity] is the location. The remaining
   unary primitives are on numbers or on strings, and have no value: a
   cell is neither. A primitive that is added and that reads a cell has
   its case here.
   Of an array, [Array_length] is the number of elements
   (FlambdaPrimitiveSpec.array_length_of), whatever the kind it
   carries; [Is_null] answers no, and so does [Is_int] without
   [variant_only]; [Opaque_identity] is the location. With
   [variant_only], [Is_int] has no value on an array, which is no
   variant: Simplify makes it [Invalid] there
   ([Provers.prove_is_int_generic_value]), as on a closure, a boxed
   number or a string (FlambdaPrimitiveSpec.eval_is_int). [Get_tag] and
   [Block_load] have no value on an array, nor [Array_length] on a
   block. *)
(* CR mshinwell: a [Block_load] marked [Immutable] has no value on a
   block made [Mutable]. Simplify takes a field so read never to change
   afterwards (it records the field in the type of the block and shares
   later loads), and nothing checks that (C4-2, R2-20). This first
   stage excludes more than that assumption does: every record with
   both mutable and immutable fields, from the first read of an
   immutable field.
   Two things that the front end makes are outside the model by ruling,
   at every stage, and the conversion refuses them:
   - the blocks that tail-modulo-cons builds (lambda/tmc.ml:
     placeholders made [Mutable], filled with [Heap_initialization],
     then read with immutable loads; R2-18): "we can ignore TMC for now
     b/c ... the pass produces broken Lambda code, that needs fixing in
     the prod compiler";
   - [Duplicate_block] followed by stores, which is the copy of a large
     record for [{ r with ... }] (lambda/translcore.ml,
     [transl_record]; miscompiled today, C4-4): "I think we can not
     model Pduprecord for the moment, it is broken as you say and is
     being fixed in prod".
   Decision 4 of requests/C4-heap.md. *)
Definition exec_unary_on_cell (p : unary_primitive) (l : location) (c : cell)
    : option value :=
  match c with
  | Cell_block mut tag fields =>
    match p with
    | Block_load kind load_mut field =>
      if load_allowed load_mut mut
      then load_field kind field tag fields
      else None
    | Get_tag => Some (Value_const (Naked_immediate tag))
    | Is_int _ | Is_null => Some (naked_bool false)
    | Opaque_identity kind =>
      if kind_eqb kind Value then Some (Value_location l) else None
    | _ => None
    end
  | Cell_array _ elements =>
    match p with
    | Array_length _ => array_length_of elements
    | Is_int false | Is_null => Some (naked_bool false)
    | Opaque_identity kind =>
      if kind_eqb kind Value then Some (Value_location l) else None
    | _ => None
    end
  end.

(* A binary primitive of FlambdaPrimitiveModel.v whose first argument is
   a location: it reads the cell. The only one is [Array_load], of an
   array: the element at the index, as it is now
   (FlambdaPrimitiveSpec.array_loaded), where the mark of the load
   allows. An array that has an identity is mutable, so that is where
   the load is marked [Mutable] ([load_allowed]). The remaining binary
   primitives have no value: [Phys_equal] with a location is not read
   here ([exec_bound_prim]), and the others are on numbers or on
   strings. *)
(* CR mshinwell: An [Array_load] marked [Immutable] has no value on an
   array that has an identity, as a [Block_load] so marked has none on
   a block made [Mutable] (the CR before [exec_unary_on_cell]). The
   front end makes one for the reads of an immutable array ([iarray]),
   and such an array may be a mutable array seen as immutable
   ([%array_to_iarray], which is the identity): the model refuses
   that program. Whether the reads of an array so frozen are stable is
   the question that decision 4 of requests/C4-heap.md asks of the
   fields of blocks. Decision D49. *)
Definition exec_binary_on_cell (p : binary_primitive) (l : location)
    (c : cell) (v2 : value) : option value :=
  match c, p with
  | Cell_array _ elements, Array_load _ load_kind load_mut =>
    if load_allowed load_mut Cell_mutable
    then array_loaded load_kind elements v2
    else None
  | _, _ => None
  end.

(* An argument that a primitive may be given: anything but a symbol or a
   location that is bound to nothing. *)
Definition argument_bound (statics : static_env) (h : heap) (v : value)
    : bool :=
  match v with
  | Value_symbol s => is_bound s (static_values statics)
  | Value_location l => is_bound l h
  | _ => true
  end.

(* What a primitive does to arguments that are bound. One that does not
   touch the heap is FlambdaPrimitiveSpec.eval_prim, and the heap is
   returned as it came. A unary primitive of a location reads its cell,
   [Phys_equal] with a location among its two arguments is the
   comparison above, and another binary primitive whose first argument
   is a location reads its cell. *)
Definition exec_bound_prim (statics : static_env) (h : heap) (p : prim)
    (vs : list value) : option (value * heap) :=
  match p, vs with
  | Heap p, vs => exec_heap_primitive statics h p vs
  | Unary p, [Value_location l] =>
    let* c := lookup l h in
    let* v := exec_unary_on_cell p l c in
    Some (v, h)
  | Binary (Phys_equal op), [v1; v2] =>
    if is_location v1 || is_location v2
    then
      let* same := eval_phys_equal_location statics v1 v2 in
      Some
        (naked_bool
           match op with
           | Phys_eq => same
           | Phys_neq => negb same
           end,
         h)
    else
      let* v := eval_prim statics p vs in
      Some (v, h)
  | Binary p, [Value_location l; v2] =>
    let* c := lookup l h in
    let* v := exec_binary_on_cell p l c v2 in
    Some (v, h)
  | _, _ =>
    let* v := eval_prim statics p vs in
    Some (v, h)
  end.

(* What a primitive does. It has no value when one of its arguments is a
   symbol or a location that is bound to nothing: any operation on such
   a name is undefined (the ruling on symbols, taken for locations too),
   whether or not the primitive would read what the name is bound to.
   So an allocation and a write take no such name, and [Phys_equal] does
   not compare a location that dangles. No state reached from a closed
   state has a location bound to nothing ([heap_step_closed_statement]).
   For symbols the same is intended and is not so yet:
   FlambdaWellFormedSpec.well_formed_unit does not ask that a unit bind
   the symbols it names (section 6 of the request). The machine relies
   on neither. *)
Definition exec_prim (statics : static_env) (h : heap) (p : prim)
    (vs : list value) : option (value * heap) :=
  if forallb (argument_bound statics h) vs
  then exec_bound_prim statics h p vs
  else None.

(* FlambdaPrimitiveSpec.eval_prim_symbols_bound_statement, of
   [exec_prim], and the same of locations. *)
Definition exec_prim_symbols_bound_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (p : prim) (vs : list value)
    (v : value) (s : symbol),
    exec_prim statics h p vs = Some (v, h') ->
    Value_symbol s ∈ vs ->
    lookup s (static_values statics) <> None.

Definition exec_prim_locations_bound_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (p : prim) (vs : list value)
    (v : value) (l : location),
    exec_prim statics h p vs = Some (v, h') ->
    Value_location l ∈ vs ->
    lookup l h <> None.

(* ======================================================================== *)
(* The primitives on a heap and [eval_prim]                                 *)
(* ======================================================================== *)

(* The primitives that may have, on a heap, a value that
   FlambdaPrimitiveSpec.eval_prim does not give them: those of the heap;
   the unary primitives that read the cell of a location
   ([exec_unary_on_cell]); [Array_load]; and [Phys_equal], which compares
   locations. Every other primitive has on a heap the value that
   [eval_prim] gives it, or none. A primitive that is added and that
   reads a cell has its case here. *)
Definition prim_reads_heap (p : prim) : bool :=
  match p with
  | Heap _ => true
  | Unary
      (Block_load _ _ _ | Get_tag | Is_int _ | Is_null | Opaque_identity _
      | Array_length _) =>
    true
  | Binary (Phys_equal _ | Array_load _ _ _) => true
  | Unary _ | Binary _ | Variadic _ | Region_primitive _ => false
  end.

(* A named expression that may read the heap: a primitive that may. *)
Definition named_reads_heap (d : named) : bool :=
  match d with
  | Prim p _ => prim_reads_heap p
  | Simple _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ =>
    false
  end.

(* A named expression that may change the heap: a primitive of the
   heap. Nothing else does ([exec_prim_heap_statement]). A [Let] of one
   cannot in general be removed, added or moved past another for as
   long as two exits are compared with their heaps. The test is
   conservative: it is not said that no such [Let] ever could be. *)
Definition named_changes_heap (d : named) : bool :=
  match d with
  | Prim (Heap _) _ => true
  | Prim (Unary _ | Binary _ | Variadic _ | Region_primitive _) _ | Simple _
  | Set_of_closures _ _ | Closure_of_set _ _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
    false
  end.

(* Only a primitive of the heap changes the heap. *)
Definition exec_prim_heap_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (p : prim) (vs : list value)
    (v : value),
    (∀ q : heap_primitive, p <> Heap q) ->
    exec_prim statics h p vs = Some (v, h') ->
    h' = h.

(* A primitive that does not read the heap and that has a value on a heap
   has that value by [eval_prim]. *)
Definition exec_prim_eval_prim_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (p : prim) (vs : list value)
    (v : value),
    prim_reads_heap p = false ->
    exec_prim statics h p vs = Some (v, h') ->
    eval_prim statics p vs = Some v ∧ h' = h.

(* And it has on a heap the value that [eval_prim] gives it, when the
   locations among its arguments are bound ([exec_prim] has no value on a
   location that dangles, whatever the primitive). *)
Definition eval_prim_exec_prim_statement : Prop :=
  ∀ (statics : static_env) (h : heap) (p : prim) (vs : list value)
    (v : value),
    prim_reads_heap p = false ->
    (∀ l : location, Value_location l ∈ vs -> is_bound l h = true) ->
    eval_prim statics p vs = Some v ->
    exec_prim statics h p vs = Some (v, h).

(* A primitive that is not of the heap, applied to arguments none of
   which is a location, is [eval_prim], and the heap is returned as it
   came: so are the reads of a block that is a value or that a symbol is
   bound to. *)
Definition exec_prim_no_location_statement : Prop :=
  ∀ (statics : static_env) (h : heap) (p : prim) (vs : list value),
    (∀ q : heap_primitive, p <> Heap q) ->
    (∀ l : location, Value_location l ∉ vs) ->
    exec_prim statics h p vs
    = let* v := eval_prim statics p vs in
      Some (v, h).

(* FlambdaPrimitiveSpec.eval_prim_static_values_statement, of
   [exec_prim]: what a primitive does depends on the static environment
   only through what the symbols among its arguments are bound to. *)
Definition exec_prim_static_values_statement : Prop :=
  ∀ (statics statics' : static_env) (h : heap) (p : prim) (vs : list value),
    (∀ s : symbol,
      Value_symbol s ∈ vs ->
      lookup s (static_values statics) = lookup s (static_values statics')) ->
    exec_prim statics h p vs = exec_prim statics' h p vs.

(* FlambdaPrimitiveSpec.eval_prim_statics_extend_statement, of
   [exec_prim]: a primitive that has a value keeps it, and returns the
   same heap, when the static environment grows. *)
Definition exec_prim_statics_extend_statement : Prop :=
  ∀ (statics statics' : static_env) (h h' : heap) (p : prim)
    (vs : list value) (v : value),
    (∀ (s : symbol) (allocated : statically_allocated_value),
      Value_symbol s ∈ vs ->
      lookup s (static_values statics) = Some allocated ->
      lookup s (static_values statics') = Some allocated) ->
    exec_prim statics h p vs = Some (v, h') ->
    exec_prim statics' h p vs = Some (v, h').

(* ======================================================================== *)
(* Effects and coeffects                                                    *)
(* ======================================================================== *)

(* [Effects.t]. *)
Inductive effects : Type :=
  | No_effects
  | Only_generative_effects (mut : mutability)
  | Arbitrary_effects.

(* [Coeffects.t]. *)
Inductive coeffects : Type :=
  | No_coeffects
  | Has_coeffects.

(* [Flambda_primitive.effects_and_coeffects], its first component.
   [Local_allocation] is no primitive of the compiler
   (FlambdaPrimitiveModel.region_primitive) and is given the strongest:
   arbitrary effects, and a coeffect below. *)
Definition effects_of_prim (p : prim) : effects :=
  match p with
  | Unary (Opaque_identity _) => Arbitrary_effects
  | Unary (Box_number _) => Only_generative_effects Immutable
  | Unary _ | Binary _ => No_effects
  | Variadic (Make_block _) | Variadic (Make_immutable_array _) =>
    Only_generative_effects Immutable
  | Heap (Make_block_with_identity Cell_mutable _) =>
    Only_generative_effects Mutable
  | Heap (Make_block_with_identity Cell_immutable_unique _) =>
    Only_generative_effects Immutable_unique
  | Heap (Block_set _ _ _) => Arbitrary_effects
  | Heap (Make_mutable_array _) => Only_generative_effects Mutable
  | Heap (Array_set _ _) => Arbitrary_effects
  | Region_primitive (Begin_region | Begin_try_region) =>
    Only_generative_effects Mutable
  | Region_primitive (End_region | End_try_region | Local_allocation) =>
    Arbitrary_effects
  end.

(* Its second component. *)
Definition coeffects_of_prim (p : prim) : coeffects :=
  match p with
  | Unary (Opaque_identity _) => Has_coeffects
  | Unary (Block_load _ Mutable _) => Has_coeffects
  | Binary (Array_load _ _ Mutable) => Has_coeffects
  | Region_primitive _ => Has_coeffects
  | _ => No_coeffects
  end.

(* [Flambda_primitive.at_most_generative_effects]: what Simplify asks
   before it deletes a [Let] whose variable is not used. *)
Definition at_most_generative_effects (p : prim) : bool :=
  match effects_of_prim p with
  | No_effects | Only_generative_effects _ => true
  | Arbitrary_effects => false
  end.

(* The same of a named expression, as [Simplify_let_expr] asks it
   ([Named.at_most_generative_effects]): everything that is not a
   primitive qualifies. It is the compiler's test, and it is not the
   premise of the statements of the model that erase a binding. For as
   long as an exit is compared with its heap, those instead exclude
   every primitive of the heap ([named_changes_heap], above:
   FlambdaNominalSpec.unused_let_statement; and
   FlambdaNormaliseSpec.named_may_go leaves those primitives out): an
   allocation of a cell, which this test lets through, is seen in the
   heap of an exit. Neither test implies the other. *)
Definition named_at_most_generative_effects (d : named) : bool :=
  match d with
  | Prim p _ => at_most_generative_effects p
  | Simple _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ =>
    true
  end.

(* The stricter test, for as long as exits are compared exactly: an
   allocation that is deleted is then seen. *)
Definition named_has_no_effects (d : named) : bool :=
  match d with
  | Prim p _ =>
    match effects_of_prim p with
    | No_effects => true
    | Only_generative_effects _ | Arbitrary_effects => false
    end
  | Simple _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ =>
    true
  end.

(* A named expression whose value, once it has one, is the same in every
   heap that the heap evolves to: one that is not a primitive, or a
   primitive with no effects and no coeffects. This is what a fact about
   the value of a variable needs in order to be kept across a store. *)
Definition named_stable (d : named) : bool :=
  match d with
  | Prim p _ =>
    match effects_of_prim p, coeffects_of_prim p with
    | No_effects, No_coeffects => true
    | _, _ => false
    end
  | Simple _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ =>
    true
  end.

(* What a run may do to a heap: a cell that is bound stays bound. A
   block keeps its mutability, its tag and its number of fields, and
   its fields if it is not mutable. An array keeps its kind and its
   number of elements. *)
Definition heap_evolves (h h' : heap) : Prop :=
  (∀ (l : location) (mut : cell_mutability) (tag : Z) (fields : list value),
    lookup l h = Some (Cell_block mut tag fields) ->
    ∃ fields' : list value,
      lookup l h' = Some (Cell_block mut tag fields')
      ∧ length fields' = length fields
      ∧ (mut = Cell_immutable_unique -> fields' = fields))
  ∧ (∀ (l : location) (kind : array_kind) (elements : list value),
      lookup l h = Some (Cell_array kind elements) ->
      ∃ elements' : list value,
        lookup l h' = Some (Cell_array kind elements')
        ∧ length elements' = length elements).

(* Every primitive leaves a heap that the one it was given evolves to. *)
Definition exec_prim_evolves_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (p : prim) (vs : list value)
    (v : value),
    exec_prim statics h p vs = Some (v, h') -> heap_evolves h h'.

(* The elements of each array of a heap are values that the kind of the
   array holds. It is what the kind of an array is recorded for: the
   type that Simplify gives a mutable array says it of the array for as
   long as the array lives. Every primitive keeps it. *)
Definition arrays_hold_their_kinds (h : heap) : Prop :=
  ∀ (l : location) (kind : array_kind) (elements : list value) (v : value),
    lookup l h = Some (Cell_array kind elements) ->
    v ∈ elements ->
    array_element_ok kind v = true.

Definition exec_prim_arrays_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (p : prim) (vs : list value)
    (v : value),
    arrays_hold_their_kinds h ->
    exec_prim statics h p vs = Some (v, h') ->
    arrays_hold_their_kinds h'.

(* What Simplify knows of a mutable array, from its [Make_array]
   ([Simplify_variadic_primitive.simplify_make_array]): its length. The
   length of an array is the number of arguments it was made of, in
   every heap that the one after the allocation evolves to: an
   [Array_length] of an array whose [Make_array] is in sight is that
   number ([Simplify_unary_primitive.simplify_array_length]). There is
   no length where the number is not a tagged immediate. *)
Definition array_length_known_statement : Prop :=
  ∀ (statics : static_env) (h h1 h2 : heap) (kind : array_kind)
    (length_kind : array_kind_for_length) (vs : list value) (v : value),
    exec_prim statics h (Heap (Make_mutable_array kind)) vs = Some (v, h1) ->
    heap_evolves h1 h2 ->
    exec_prim statics h2 (Unary (Array_length length_kind)) [v]
    = let* n := array_length_of vs in
      Some (n, h2).

(* The array kind that a length, a load or a store carries changes
   nothing. [Simplify_common.specialise_array_kind] replaces [Values]
   and [Gc_ignorable_values] by [Immediates] in a load and in a store,
   where the type of the array says that its elements are immediates,
   and keeps the load kind and the set kind. *)
Definition array_kind_of_access_statement : Prop :=
  ∀ (statics : static_env) (h : heap) (vs : list value)
    (kind kind' : array_kind)
    (length_kind length_kind' : array_kind_for_length)
    (load_kind : array_load_kind) (mut : mutability)
    (set_kind : array_set_kind),
    exec_prim statics h (Unary (Array_length length_kind)) vs
    = exec_prim statics h (Unary (Array_length length_kind')) vs
    ∧ exec_prim statics h (Binary (Array_load kind load_kind mut)) vs
      = exec_prim statics h (Binary (Array_load kind' load_kind mut)) vs
    ∧ exec_prim statics h (Heap (Array_set kind set_kind)) vs
      = exec_prim statics h (Heap (Array_set kind' set_kind)) vs.

(* A primitive with no effects returns the heap as it came. *)
Definition no_effects_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (p : prim) (vs : list value)
    (v : value),
    effects_of_prim p = No_effects ->
    exec_prim statics h p vs = Some (v, h') ->
    h' = h.

(* A primitive with only generative effects returns the heap as it came,
   or with one new cell at a fresh location, which is its value. One
   whose result is immutable allocates nothing. *)
Definition only_generative_effects_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (p : prim) (vs : list value)
    (v : value) (mut : mutability),
    effects_of_prim p = Only_generative_effects mut ->
    exec_prim statics h p vs = Some (v, h') ->
    (mut = Immutable ∧ h' = h)
    ∨ (mut <> Immutable
       ∧ v = Value_location (fresh_location h)
       ∧ ∃ c : cell, h' = (fresh_location h, c) :: h).

(* A primitive with neither effects nor coeffects has the value it had
   in every heap that the one it was run in evolves to: it may be run
   later, or once in place of twice. *)
Definition no_coeffects_statement : Prop :=
  ∀ (statics : static_env) (h1 h1' h2 : heap) (p : prim) (vs : list value)
    (v : value),
    effects_of_prim p = No_effects ->
    coeffects_of_prim p = No_coeffects ->
    heap_evolves h1 h2 ->
    exec_prim statics h1 p vs = Some (v, h1') ->
    exec_prim statics h2 p vs = Some (v, h2).

(* The load marked [Mutable] is not such a primitive: its value changes
   with a write, so it may not be shared with an earlier one. *)
Definition mutable_load_not_stable_statement : Prop :=
  ∃ (statics : static_env) (h1 h2 : heap) (p : prim) (vs : list value)
    (v1 v2 : value),
    effects_of_prim p = No_effects
    ∧ heap_evolves h1 h2
    ∧ exec_prim statics h1 p vs = Some (v1, h1)
    ∧ exec_prim statics h2 p vs = Some (v2, h2)
    ∧ v1 <> v2.

(* A primitive whose value, once it has one, is the same in every heap
   that the heap evolves to, and that returns the heap as it came: it
   reads no cell, or it has neither effects nor coeffects. A fact about
   the value of a variable may be about such a primitive: the primitive
   can be run again on the same values of its arguments under the same
   static environment, now and after any step, and gives that value. *)
Definition stable_prim (p : prim) : bool :=
  negb (prim_reads_heap p)
  || match effects_of_prim p, coeffects_of_prim p with
     | No_effects, No_coeffects => true
     | _, _ => false
     end.

Definition stable_prim_holds_statement : Prop :=
  ∀ (statics : static_env) (h1 h1' h2 : heap) (p : prim) (vs : list value)
    (v : value),
    stable_prim p = true ->
    heap_evolves h1 h2 ->
    exec_prim statics h1 p vs = Some (v, h1') ->
    h1' = h1 ∧ exec_prim statics h2 p vs = Some (v, h2).

(* ======================================================================== *)
(* Closed memories                                                          *)
(* ======================================================================== *)

(* The locations of a value. *)
Fixpoint locations_of_value (v : value) : list location :=
  match v with
  | Value_const _ | Value_symbol _ | Value_boxed_number _ => []
  | Value_closure _ slots | Value_closure_of_set _ _ slots =>
    flat_map (fun slot => locations_of_value (snd slot)) slots
  | Value_block _ fields | Value_immutable_array fields =>
    flat_map locations_of_value fields
  | Value_location l => [l]
  | Value_partial_application callee args _ =>
    locations_of_value callee ++ flat_map locations_of_value args
  end.

(* No location of the value dangles. *)
Definition value_closed (h : heap) (v : value) : Prop :=
  ∀ l : location, l ∈ locations_of_value v -> is_bound l h = true.

Definition values_of_cell (c : cell) : list value :=
  match c with
  | Cell_block _ _ fields => fields
  | Cell_array _ elements => elements
  end.

Definition values_of_static (a : statically_allocated_value) : list value :=
  match a with
  | Static_closure _ slots | Static_closure_of_set _ _ _ slots =>
    map snd slots
  | Static_block _ fields => fields
  | Static_immutable_array elements => elements
  | Static_boxed_number v => [v]
  | Static_string _ | Static_imported _ => []
  end.

(* No location dangles in the heap or in the static environment, and
   every field of a cell is of kind [Value]. *)
Definition memory_closed (statics : static_env) (h : heap) : Prop :=
  (∀ (l : location) (c : cell) (v : value),
    lookup l h = Some c ->
    v ∈ values_of_cell c ->
    value_closed h v ∧ kind_of_value v = Value)
  ∧ (∀ (s : symbol) (a : statically_allocated_value) (v : value),
      lookup s (static_values statics) = Some a ->
      v ∈ values_of_static a ->
      value_closed h v).

(* A primitive keeps a closed memory closed, and its value is closed,
   when its arguments are. *)
Definition exec_prim_closed_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (p : prim) (vs : list value)
    (v : value),
    memory_closed statics h ->
    (∀ arg : value, arg ∈ vs -> value_closed h arg) ->
    exec_prim statics h p vs = Some (v, h') ->
    memory_closed statics h' ∧ value_closed h' v.

(* ======================================================================== *)
(* The values that a heap holds                                             *)
(* ======================================================================== *)

(* [P] holds of every value that a heap holds: of the values of every
   entry, those of an entry that a later write hides included. *)
Definition heap_values_all (P : value -> Prop) (h : heap) : Prop :=
  ∀ (l : location) (c : cell) (v : value),
    (l, c) ∈ h -> v ∈ values_of_cell c -> P v.

(* A primitive makes its value, and the cells it makes or changes, out of
   its arguments and of the cells it reads:
   FlambdaPrimitiveSpec.eval_prim_preserves_statement, on a heap. The
   property is asked of every location, since an allocation returns
   one. *)
Definition exec_prim_preserves_statement : Prop :=
  ∀ (P : value -> Prop) (statics : static_env) (h h' : heap) (p : prim)
    (vs : list value) (v : value),
    holds_of_numbers P ->
    holds_of_blocks P ->
    holds_of_arrays P ->
    holds_in_static_blocks statics P ->
    holds_in_static_arrays statics P ->
    (∀ l : location, P (Value_location l)) ->
    (∀ arg : value, arg ∈ vs -> P arg) ->
    heap_values_all P h ->
    exec_prim statics h p vs = Some (v, h') ->
    P v ∧ heap_values_all P h'.

(* ======================================================================== *)
(* Two heaps alike up to a relation between values                          *)
(* ======================================================================== *)

(* Two cells of the same shape, holding related values: blocks of one
   mutability and tag, or arrays of one kind. *)
Definition cell_alike (R : value -> value -> Prop) (c c' : cell) : Prop :=
  match c, c' with
  | Cell_block mut tag fields, Cell_block mut' tag' fields' =>
    mut = mut' ∧ tag = tag' ∧ Forall2 R fields fields'
  | Cell_array kind elements, Cell_array kind' elements' =>
    kind = kind' ∧ Forall2 R elements elements'
  | _, _ => False
  end.

(* Two heaps alike: the same locations, entry for entry and so in the
   same order, bound to cells alike. The entries that a later write
   hides are compared too. At equality it is equality of heaps
   ([heap_alike_eq_statement]), which is how [refines] compares the
   heaps of two exits. No location is paired with another: this is
   not a correspondence between the locations of two runs. *)
Definition heap_alike (R : value -> value -> Prop) (h h' : heap) : Prop :=
  Forall2 (fun b b' => fst b = fst b' ∧ cell_alike R (snd b) (snd b')) h h'.

Definition heap_alike_eq_statement : Prop :=
  ∀ h h' : heap, heap_alike eq h h' <-> h = h'.

(* A location is related to itself, and to nothing else. *)
Definition relates_locations (R : value -> value -> Prop) : Prop :=
  (∀ l : location, R (Value_location l) (Value_location l))
  ∧ (∀ (l : location) (v' : value),
      R (Value_location l) v' -> v' = Value_location l)
  ∧ (∀ (v : value) (l : location),
      R v (Value_location l) -> v = Value_location l).

(* The location that an allocation takes depends on the heap only
   through the locations that it binds: two heaps alike allocate the
   same location. *)
Definition fresh_location_keys_statement : Prop :=
  ∀ h h' : heap,
    map fst h = map fst h' -> fresh_location h = fresh_location h'.

(* FlambdaPrimitiveSpec.eval_prim_related_statement, on a heap: what a
   primitive does on some arguments and a heap it does on related
   arguments and a heap alike, with a related value and heaps alike
   after. The premise on the relation is that of the law of
   [eval_prim], with the part that it says a heap needs: a location
   is related to itself only. As that law, it goes one way. *)
Definition exec_prim_related_statement : Prop :=
  ∀ (R : value -> value -> Prop) (statics statics' : static_env)
    (h h' h1 : heap) (p : prim) (vs vs' : list value) (v : value),
    respects_observations R statics statics' ->
    relates_locations R ->
    heap_alike R h h' ->
    Forall2 R vs vs' ->
    exec_prim statics h p vs = Some (v, h1) ->
    ∃ (v' : value) (h1' : heap),
      exec_prim statics' h' p vs' = Some (v', h1')
      ∧ R v v'
      ∧ heap_alike R h1 h1'.

(* ======================================================================== *)
(* The kinds of what a primitive run on a heap takes and gives              *)
(* ======================================================================== *)

(* The value that a primitive run on a heap gives is of the kind of the
   result of the primitive: FlambdaPrimitiveSpec.eval_prim_kinds_statement
   for its second part, on a heap, of every primitive, those of the heap
   included. *)
Definition exec_prim_result_kind_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (p : prim) (vs : list value)
    (v : value),
    exec_prim statics h p vs = Some (v, h') ->
    kind_of_value v = prim_result_kind p.

(* And its arguments are as many as it takes and of the kinds it takes,
   where the primitive is not of the heap ([prim_args_well_kinded] is
   false of every primitive of the heap). *)
Definition exec_prim_kinds_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (p : prim) (vs : list value)
    (v : value),
    (∀ q : heap_primitive, p <> Heap q) ->
    exec_prim statics h p vs = Some (v, h') ->
    prim_args_well_kinded p (map kind_of_value vs) = true
    ∧ kind_of_value v = prim_result_kind p.
