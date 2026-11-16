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

(* The values of the machine of FlambdaSemanticsSpec.v, what symbols are
   bound to, and how a value is read.

   This file is to be reviewed: [value], [kind_of_value],
   [statically_allocated_value], the observations [closure_of],
   [block_of], [boxed_number_of], [string_of] and [array_of], and the
   statements that they are exclusive, which FlambdaValueProofs.v
   proves; [import_class],
   [imported_of], [is_imported_pointer] and [identity_of]; and the
   three readings of a closure as one of a set,
   [closure_in_set], [project_value_slot] and [project_function_slot],
   whose statements are in FlambdaSemanticsSpec.v. The rest is
   definitions. FlambdaSemanticsSpec.v exports this file, so a file that
   imports the semantics has what is here.

   A value is a constant (a tagged immediate, a naked number of any kind,
   null or poison), a pointer to what a symbol is bound to, something
   allocated that is immutable (a closure, a block of values, an immutable
   array of values, a boxed number, a partial application), or a location.
   An allocated immutable value is its contents: two of them are the same
   value when they have the same contents, and nothing tells one
   allocation from another. Mutable memory is the heap, which this file
   defines after the values ([cell], [heap]) and which the state of the
   machine carries: a location ([Value_location]) names a cell of it where
   the heap binds it, a block of values that has an identity or a mutable
   array of values. None of the five observations reads a location, and a
   location is not allocated in the sense of [is_allocated]: it is read
   with a heap ([cell_of]), and what a primitive does with one is
   FlambdaHeapSpec.exec_prim's to say.

   A symbol is a value by itself, whether or not it is bound, and what it
   points to is in the static environment. So a block made by a primitive
   and a symbol bound to a block of the same tag and fields are different
   values, although no primitive tells them apart: every reading of an
   allocated value goes through an observation, which takes the static
   environment and answers the same for both. The observations are
   pairwise exclusive: a value is at most one of a closure, a block, a
   boxed number, a string and an immutable array
   ([observations_disjoint_statement], [array_of_disjoint_statement]).

   An immutable array of values is a value that a primitive makes
   ([Value_immutable_array], made by [Make_immutable_array]:
   FlambdaPrimitiveSpec.eval_variadic_primitive), or what a symbol is
   bound to ([Static_immutable_array]): the table that Simplify makes
   of a [Switch] whose arms give one continuation constants
   (simplify/simplify_switch_expr.ml), or an immutable array that it
   lifts. Either is immutable data, as a block is, and not a location
   of the heap: nothing writes it. [array_of] reads both, by the
   elements. The value keeps no array kind: an array of the model
   holds values of kind [Value], and a load says at which kind it
   reads (FlambdaPrimitiveSpec.v).

   A symbol may also be bound to something that the unit imports
   ([Static_imported]): an object or a value of another unit (a
   predefined exception is of a unit of its own), of which the model
   knows the symbol and its class ([import_class]) and nothing else.
   An import of the class [Import_identity] is an object with an
   identity, which is its symbol ([identity_of]): an exception. One of
   the class [Import_data] is an object that another unit allocated
   and that is not an identity, and is to the model what an immutable
   allocated value of the unit is, without its contents. One of the
   class [Import_unknown] is a value of which nothing is known: it may
   be a tagged immediate, an identity, or an object that another unit
   allocated, mutable or not.
   None of the five observations reads an import of any class, so
   it is not a closure, a block, a boxed number, a string or an
   immutable array, and is not allocated in the sense of
   [is_allocated]. The identity returns an import of any class, and it
   can be a field of a block that is made. Physical equality has a
   value on an import of the first two classes, of which the model
   knows that they are pointers ([is_imported_pointer]): it tells an
   identity from everything else (FlambdaPrimitiveSpec.v). On an
   import of the third class it has no value against anything, itself
   included; no other unary or binary primitive has a value on an
   import. No static binding of a term makes an import: the static
   environment in which a unit is run has them
   (FlambdaImportsSpec.v).

   A closure is one of a set of closures. The closure of a set of one
   function whose function slot has no name is its code and the values
   of its value slots ([Value_closure], [Static_closure]). The closure
   of a set of functions at named function slots is the slot it is at,
   the code of every function slot of the set, and the values of the
   value slots ([Value_closure_of_set]); the closures of one set differ
   in that slot only, and a closure does not hold the others. A symbol
   bound to a closure of a static set also has the symbols of the
   closures of its set ([Static_closure_of_set]). [closure_of] reads
   any of them as the code of its own function and the value slots;
   [closure_in_set] as the slot, the set and the value slots, the
   unnamed forms being at [unnamed_function_slot] of a set of that one
   function. A value of the shape of a closure at a function slot that
   its set has not got is not a closure: neither reads it. The header
   of FlambdaSemanticsSpec.v says more of sets of closures.

   A boxed number is recognised by the kind of its contents, which is
   one of the naked number kinds that have a boxed form. So the contents
   may be poison of such a kind, and a symbol bound to a boxed poison is
   a boxed number, and allocated. What a primitive does with it is the
   primitive's to say (FlambdaPrimitiveSpec.v). One that asks what a
   value is sees the box: [Is_null] answers no of it, and so does
   [Is_int] without [variant_only], and the identity returns it. Boxing
   a poison and unboxing a boxed poison have no value.

   The kind of a value is that of its constant, and [Value] for
   everything else. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
Import ListNotations.

(* ======================================================================== *)
(* Values                                                                   *)
(* ======================================================================== *)

(* A location: the name of a cell of the heap. It is compared by equality
   only; nothing may depend on which number it is. *)
Definition location : Type := positive.

(* The values of variables: constants, closures made by a [Let], pointers
   to what symbols are bound to, and the immutable blocks, the immutable
   arrays and the boxed numbers that primitives make. A closure of a set
   of one function is the code of its function and the values of its
   value slots ([Value_closure]). A closure of a set of functions at
   named slots is the slot it is at, the code of each function slot of
   the set, and the values of the value slots ([Value_closure_of_set]).
   The contents of a boxed number are a constant, whose constructor says
   which number it is. A partial application
   ([Value_partial_application]) is what a call that gives a function
   fewer arguments than it has parameters comes to: the value called,
   the arguments given so far, and the arity they were given at. (The
   warning silenced is about induction schemes through the lists, which
   nothing uses yet.) *)
#[local] Set Warnings "-register-all".
Inductive value : Type :=
  | Value_const (c : const)
  | Value_closure (code : code_id) (value_slots : list (value_slot * value))
  | Value_closure_of_set (slot : function_slot)
      (function_decls : list (function_slot * code_id))
      (value_slots : list (value_slot * value))
  | Value_symbol (s : symbol)
  | Value_block (tag : Z) (fields : list value)
  | Value_immutable_array (elements : list value)
  | Value_boxed_number (contents : const)
  | Value_location (l : location)
  | Value_partial_application (callee : value) (args : list value)
      (args_arity : arity).

(* The kind of a value. *)
Definition kind_of_value (v : value) : flambda_kind :=
  match v with
  | Value_const c => kind_of_const c
  | _ => Value
  end.

(* What the model knows of an object that a unit imports. An identity is
   an object that is made once and never shared or copied: a predefined
   exception, or an exception that another unit declares. Nothing that a
   unit allocates is it, and two of them are one object exactly when
   their symbols are one symbol. An allocated object of another unit
   that is not an identity is data: its contents are not known here,
   and the compiler may know them, so it is not told apart from what
   the unit allocates. Anything else is unknown: a value that another
   unit computed, of which the model knows nothing, not even that it
   is a pointer (a field of a module block that holds a number, a
   mutable object or an identity under another name). *)

(* CR mshinwell: an import of the class [Import_unknown] has no value
   under [Phys_equal], even against itself, where the program answers
   true; and nothing that inspects it has a value: no other primitive,
   no call of it, no [Switch] on it and no load from it. The model says
   less than the program does, never more: of a run that inspects such
   an import nothing is proved. Decision D41. *)
Inductive import_class : Type :=
  | Import_identity
  | Import_data
  | Import_unknown.

(* What a symbol points to. [Static_closure_of_set] is the closure at
   [slot] of a static set; [closure_symbols] are the symbols of the
   closures of the set. [Static_immutable_array] is an immutable array
   of values, by its elements. [Static_imported] is what the unit
   imports: an object of the class given, whose contents are not
   known. *)
Inductive statically_allocated_value : Type :=
  | Static_closure (code : code_id) (value_slots : list (value_slot * value))
  | Static_closure_of_set (slot : function_slot)
      (closure_symbols : list (function_slot * symbol))
      (function_decls : list (function_slot * code_id))
      (value_slots : list (value_slot * value))
  | Static_block (tag : Z) (fields : list value)
  | Static_boxed_number (contents : value)
  | Static_string (bytes : list Z)
  | Static_immutable_array (elements : list value)
  | Static_imported (c : import_class).

(* The static environment: the code and the statically allocated values
   bound so far. *)
Record static_env : Type := Static_env {
  static_code : list (code_id * code);
  static_values : list (symbol * statically_allocated_value)
}.

Definition empty_static_env : static_env := Static_env [] [].

(* ======================================================================== *)
(* The heap                                                                 *)
(* ======================================================================== *)

(* What a location is bound to: a block of values with an identity,
   mutable or not ([cell_mutability], FlambdaPrimitiveModel.v), or a
   mutable array of values, with the kind it was made at. An immutable
   block is a value and no cell. An array is always mutable: the
   compiler has none that is [Immutable_unique], and one made
   [Immutable] has no identity and is no cell (the values have not got
   one yet). *)
Inductive cell : Type :=
  | Cell_block (mut : cell_mutability) (tag : Z) (fields : list value)
  | Cell_array (kind : array_kind) (elements : list value).

(* The heap: a finite map of FlambdaMapSpec.v from locations to cells. A
   write is a new binding in front. *)
Definition heap : Type := list (location * cell).

Definition empty_heap : heap := [].

(* What the loads that read memory as immutable have read: a location,
   after a symbol is resolved, with one field of a block or with the
   whole of the bytes or of the elements. A write to what the record
   covers is undefined ([Read_whole] covers every field of its
   location); order and repetition mean nothing. Nothing adds to the
   record and nothing reads it yet: FlambdaSemanticsSpec.step carries
   it. Decisions D49 and D52. *)
Inductive read_extent : Type :=
  | Read_field (i : nat)
  | Read_whole.

Definition read_record : Type := list (location * read_extent).

Definition no_reads : read_record := [].

(* A record that the context determines. *)
Abbreviation rd_ := _ (only parsing).

(* A value as a cell: a location, and what the heap binds it to. *)
Definition cell_of (h : heap) (v : value) : option cell :=
  match v with
  | Value_location l => lookup l h
  | _ => None
  end.

(* ======================================================================== *)
(* Reading an allocated value                                               *)
(* ======================================================================== *)

(* A value as a closure: one made by a [Let], or a symbol bound to one.
   The code is that of the function the closure is of: for a closure of a
   set, the code of the slot it is at. *)
Definition closure_of (statics : static_env) (v : value)
    : option (code_id * list (value_slot * value)) :=
  match v with
  | Value_closure code value_slots => Some (code, value_slots)
  | Value_closure_of_set slot function_decls value_slots =>
    let* code := lookup slot function_decls in
    Some (code, value_slots)
  | Value_symbol s =>
    match lookup s (static_values statics) with
    | Some (Static_closure code value_slots) => Some (code, value_slots)
    | Some (Static_closure_of_set slot _ function_decls value_slots) =>
      let* code := lookup slot function_decls in
      Some (code, value_slots)
    | _ => None
    end
  | _ => None
  end.

(* The function slot that a closure of a set of one function, with no
   function slot named, is taken to be at. *)
Definition unnamed_function_slot : function_slot := 1%positive.

(* A value as a closure of a set: the function slot it is at, the code of
   each function slot of the set, and the values of the value slots. It
   is defined where [closure_of] is, and the code [closure_of] gives is
   that of the slot (FlambdaSemanticsSpec.closure_in_set_of_statement). *)

(* CR mshinwell: A value of the shape of a closure at a function slot
   that its set has not got is not a closure for [closure_in_set], nor
   for [closure_of], so calling it or projecting from it is undefined.
   No term makes one (FlambdaSemanticsSpec.eval_named), but a statement
   about every environment speaks of them. The other choice is to read
   it as a closure with no code. Decision D20. *)
Definition closure_in_set (statics : static_env) (v : value)
    : option
        (function_slot * list (function_slot * code_id)
         * list (value_slot * value)) :=
  match v with
  | Value_closure code value_slots =>
    Some (unnamed_function_slot, [(unnamed_function_slot, code)], value_slots)
  | Value_closure_of_set slot function_decls value_slots =>
    let* _ := lookup slot function_decls in
    Some (slot, function_decls, value_slots)
  | Value_symbol s =>
    match lookup s (static_values statics) with
    | Some (Static_closure code value_slots) =>
      Some
        (unnamed_function_slot, [(unnamed_function_slot, code)], value_slots)
    | Some (Static_closure_of_set slot _ function_decls value_slots) =>
      let* _ := lookup slot function_decls in
      Some (slot, function_decls, value_slots)
    | _ => None
    end
  | _ => None
  end.

(* CR mshinwell: A projection that names another function slot than the
   one its closure is at has no value ([project_value_slot],
   [project_function_slot]), so a term that executes one is undefined;
   and so has the projection of a function slot that the set has not
   got, or for which a static set names no symbol. It is not certain
   that this is right. Simplify answers such a projection from the type
   of the closure where the type knows the result, without looking at
   the slot named, and replaces it by [Invalid] where it does not; no
   check is made where the term is built, and the native backend would
   compute an address from the offsets of the two slots. If the compiler
   could produce and execute a projection that names the wrong slot,
   undefined would be wrong here. Decision D3. *)

(* The value of the value slot [slot] of the set of which [v] is the
   closure at [project_from]. There is none if [v] is not a closure at
   [project_from], or if its set has no such value slot. *)
Definition project_value_slot (statics : static_env)
    (project_from : function_slot) (slot : value_slot) (v : value)
    : option value :=
  let* '(at_slot, _, value_slots) := closure_in_set statics v in
  if Pos.eqb at_slot project_from then lookup slot value_slots else None.

(* The closure at [move_to] of the set of which [v] is the closure at
   [move_from]: the same set at the other slot, for a closure made by a
   [Let]; the symbol of the other slot, for a symbol. There is none if
   [v] is not a closure at [move_from], or if its set has no [move_to].
   The closure of a set of one function, with no function slot named, is
   its own closure at [unnamed_function_slot]. *)

(* CR mshinwell: [project_function_slot] has no value on a closure at
   another function slot than [move_from], as [project_value_slot] has
   none on one at another than [project_from]: see the comment before
   [project_value_slot]. Decision D3. *)

(* CR mshinwell: From a symbol, [project_function_slot] gives the symbol
   that the set names for [move_to], whatever that symbol is bound to,
   and whether or not it is bound: nothing here checks that the symbols
   of a static set are bound to its closures.
   FlambdaWellFormedSpec.well_formed_unit asks it of a unit, and the run
   of a well-formed unit keeps it
   (FlambdaClosuresSpec.well_formed_sets_bound_statement). The other
   choices are to check it here, which needs an equality test on values,
   or to bind a static set with one binding. Decision D4. *)
Definition project_function_slot (statics : static_env)
    (move_from move_to : function_slot) (v : value) : option value :=
  let unnamed :=
    if Pos.eqb move_from unnamed_function_slot
    then if Pos.eqb move_to unnamed_function_slot then Some v else None
    else None
  in
  match v with
  | Value_closure _ _ => unnamed
  | Value_closure_of_set slot function_decls value_slots =>
    if Pos.eqb slot move_from
    then
      let* _ := lookup slot function_decls in
      let* _ := lookup move_to function_decls in
      Some (Value_closure_of_set move_to function_decls value_slots)
    else None
  | Value_symbol s =>
    match lookup s (static_values statics) with
    | Some (Static_closure _ _) => unnamed
    | Some (Static_closure_of_set slot closure_symbols function_decls _) =>
      if Pos.eqb slot move_from
      then
        let* _ := lookup slot function_decls in
        let* _ := lookup move_to function_decls in
        let* s' := lookup move_to closure_symbols in
        Some (Value_symbol s')
      else None
    | _ => None
    end
  | _ => None
  end.

(* A value as an immutable block of values, with its tag: one made by a
   primitive, or a symbol bound to one. *)
Definition block_of (statics : static_env) (v : value)
    : option (Z * list value) :=
  match v with
  | Value_block tag fields => Some (tag, fields)
  | Value_symbol s =>
    match lookup s (static_values statics) with
    | Some (Static_block tag fields) => Some (tag, fields)
    | _ => None
    end
  | _ => None
  end.

(* The kind of number a constant is, if its kind is a naked number kind
   that has a boxed form. It goes by the kind of the constant and not by
   its constructor, so a poison of such a kind is one. *)
Definition boxable_number_of_const (c : const) : option boxable_number :=
  boxable_number_of_kind (kind_of_const c).

(* The contents, if they are of a kind that has a boxed form. *)
Definition boxable_contents (c : const) : option const :=
  match boxable_number_of_const c with
  | Some _ => Some c
  | None => None
  end.

(* A value as a boxed number, by the naked number inside: one made by a
   primitive, or a symbol bound to one. The kind of the constant says
   which boxed number it is. *)
Definition boxed_number_of (statics : static_env) (v : value)
    : option const :=
  match v with
  | Value_boxed_number c => boxable_contents c
  | Value_symbol s =>
    match lookup s (static_values statics) with
    | Some (Static_boxed_number (Value_const c)) => boxable_contents c
    | _ => None
    end
  | _ => None
  end.

(* A value as an immutable string, by its bytes: a symbol bound to one. *)
Definition string_of (statics : static_env) (v : value) : option (list Z) :=
  match v with
  | Value_symbol s =>
    match lookup s (static_values statics) with
    | Some (Static_string bytes) => Some bytes
    | _ => None
    end
  | _ => None
  end.

(* A value as an immutable array of values, by its elements: one that a
   primitive made, or a symbol bound to one. A location of the heap
   that holds an array is not one: that array may be written, and is
   read through the heap. *)
Definition array_of (statics : static_env) (v : value)
    : option (list value) :=
  match v with
  | Value_immutable_array elements => Some elements
  | Value_symbol s =>
    match lookup s (static_values statics) with
    | Some (Static_immutable_array elements) => Some elements
    | _ => None
    end
  | _ => None
  end.

(* A value as a partial application: the callee, the arguments it has
   been applied to so far, and the arity they were given at. *)
Definition partial_application_of (v : value)
    : option (value * list value * arity) :=
  match v with
  | Value_partial_application callee args args_arity =>
    Some (callee, args, args_arity)
  | _ => None
  end.

(* Whether a value is allocated: a closure, a block, a boxed number, a
   string, an immutable array or a partial application. A symbol bound
   to nothing is not, and nor is an imported symbol, of which no
   observation here says anything. *)
Definition is_allocated (statics : static_env) (v : value) : bool :=
  is_some (closure_of statics v)
  || is_some (block_of statics v)
  || is_some (boxed_number_of statics v)
  || is_some (string_of statics v)
  || is_some (array_of statics v)
  || is_some (partial_application_of v).

(* A value as an imported symbol: a symbol that the static environment
   binds to [Static_imported]. The answer is the symbol and the class of
   what it points to. *)
Definition imported_of (statics : static_env) (v : value)
    : option (symbol * import_class) :=
  match v with
  | Value_symbol s =>
    match lookup s (static_values statics) with
    | Some (Static_imported c) => Some (s, c)
    | _ => None
    end
  | _ => None
  end.

Definition is_imported (statics : static_env) (v : value) : bool :=
  is_some (imported_of statics v).

(* An imported symbol of which the model knows that it is a pointer to
   an object of another unit: one of the class of identities or of
   data. Of an import of the class [Import_unknown] it does not know
   that. *)
Definition is_imported_pointer (statics : static_env) (v : value) : bool :=
  match imported_of statics v with
  | Some (_, (Import_identity | Import_data)) => true
  | Some (_, Import_unknown) | None => false
  end.

(* A value as an identity: an imported symbol of the class
   [Import_identity]. The answer is the symbol, which is the identity:
   two such symbols point to one object exactly when they are one
   symbol. *)
Definition identity_of (statics : static_env) (v : value) : option symbol :=
  match imported_of statics v with
  | Some (s, Import_identity) => Some s
  | _ => None
  end.

(* ======================================================================== *)
(* The observations are exclusive                                           *)
(* ======================================================================== *)

(* The four ways in which a value is allocated are exclusive. *)
Definition observations_disjoint_statement : Prop :=
  ∀ (statics : static_env) (v : value),
    (block_of statics v <> None
     -> closure_of statics v = None
        ∧ boxed_number_of statics v = None
        ∧ string_of statics v = None)
    ∧ (closure_of statics v <> None
       -> boxed_number_of statics v = None ∧ string_of statics v = None)
    ∧ (boxed_number_of statics v <> None -> string_of statics v = None).

(* And an immutable array is none of the four. *)
Definition array_of_disjoint_statement : Prop :=
  ∀ (statics : static_env) (v : value),
    array_of statics v <> None ->
    closure_of statics v = None
    ∧ block_of statics v = None
    ∧ boxed_number_of statics v = None
    ∧ string_of statics v = None.
