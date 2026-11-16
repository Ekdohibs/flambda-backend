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

(* Symbols that a unit imports, and telling exceptions apart by name.

   This is a prototype, to be agreed with the owners of the files it
   touches before it is a request for a change:
   requests/C5-imported-symbols.md.

   This file is to be reviewed: [imports_statics] and [unit_refines];
   [prefix_refines_statement], of two terms that follow one group of
   static bindings; the seven statements of what a primitive does with
   an imported symbol, against FlambdaValueSpec.imported_of,
   is_imported_pointer and identity_of and
   FlambdaPrimitiveSpec.eval_phys_equal; the two statements of a unit
   whose imports are of the classes of identities and of data, with
   the two functions that the first is stated with, against
   FlambdaPrimitiveSpec.eval_phys_equal and
   FlambdaHeapSpec.eval_phys_equal_location; the two
   examples, against what the front end makes of a handler that matches
   an exception (lambda/matching.ml, [combine_extension_constructor]);
   and the example of an import of which nothing is known.

   A unit mentions symbols that it does not bind. The first that the
   front end makes are the predefined exceptions: a division tests its
   divisor and raises [Division_by_zero], a checked access raises
   [Invalid_argument], a match that is not exhaustive raises
   [Match_failure], and a handler that catches an exception by name
   compares what it caught with the symbol of the exception
   ([Closure_conversion], at [Pgetpredef], makes the symbol; it is of a
   compilation unit of its own, [Compilation_unit.predef_exn]). The
   symbols of other units are the same case.

   In the model such a symbol is bound, in the static environment that
   the unit starts with, to [Static_imported] of a class. An import of
   the class [Import_identity] is an object with an identity, which is
   its symbol: an exception. One of the class [Import_data] is an
   object that another unit allocated and that is not an identity. One
   of the class [Import_unknown] is a value of which nothing is known:
   it may be a tagged immediate, an identity or an object, mutable or
   not. The model knows the contents of none. So:
   - [Phys_equal] tells an identity from everything else: two
     identities are the same when they are one symbol, and an identity
     is not a tagged immediate, an allocated value of the unit or an
     import that is data ([phys_equal_identities_statement],
     [phys_equal_imported_immediate_statement],
     [phys_equal_identity_other_statement]);
   - an import that is data is as an immutable allocated value is: it
     is not a tagged immediate, and [Phys_equal] has no value on it
     against an allocated value, another import that is data, or itself
     ([phys_equal_data_import_statement]);
   - an import of which nothing is known is compared with nothing:
     [Phys_equal] has no value on it against any value, itself
     included ([phys_equal_unknown_import_statement]). What tells the
     two other classes from it is [is_imported_pointer], which is
     [is_imported] of a value that is not an import of this class
     ([imported_pointer_statement]). Where the static environment has
     no import of this class the comparison, and that of the machine
     with a heap against a location, are the functions written with
     [is_imported] ([phys_equal_two_classes_statement]); and a unit
     whose imports are of the two other classes is run in such
     environments only ([unit_two_classes_statement]);
   - no other primitive reads an import of any class: of the unary
     and binary primitives only [Opaque_identity], which returns it,
     and [Phys_equal] have a value on it
     ([imported_not_read_statement]). It may be an argument of a
     continuation or of a call, a field of a block, and what a unit
     returns or raises ([unknown_import_statement] is an example of
     the third class);
   - a static binding of the unit cannot bind it again: the machine is
     undefined on a symbol bound twice
     (FlambdaSemanticsSpec.eval_static_bindings).

   That is enough for a handler that catches a constant exception by
   name, whether what was raised is a constant exception or a block,
   and one that catches an exception with arguments where what was
   raised is a block ([catches_by_name_statement],
   [catches_with_argument_statement]).

   What it leaves undefined, and a program does. An exception with
   arguments is a block whose first field is the exception; a constant
   exception is the exception itself. A handler tests its constant
   cases first, by comparing what it caught with each; if none is it
   and the handler has a case for an exception with arguments, or one
   for any exception that comes after such a case, it loads the first
   field of what it caught. So when it has caught a constant exception
   that is none of its constant cases, it loads a field of an imported
   symbol (the name of the exception: the object is a block of tag 248
   with the name and a number, [Cmm_helpers.predef_exception]). That
   load has no value here: the last part of
   [catches_with_argument_statement].

   CR mshinwell: a handler whose cases are for exceptions with
   arguments, such as [Failure _] alone, is undefined in the model when
   what it catches is a constant predefined exception, such as
   [Not_found]: the handler loads field 0 of the symbol, and the model
   knows no contents of an imported symbol. The compiled code reads the
   name of the exception there and goes on to the next case. To give it
   a meaning an identity must have contents (for a predefined
   exception, a string and a number) that a load reads and that no
   other observation does; a unit that declares an exception makes the
   same object, a block of tag 248 marked [Immutable_unique], which
   needs the heap (requests/C4-heap.md, "Later: symbols as addresses").
   It is wrong only in that nothing is proved of such a program. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec.
Import ListNotations.

(* ======================================================================== *)
(* The static environment that a unit starts with                           *)
(* ======================================================================== *)

(* The static environment in which a unit that imports [imports] is run:
   each symbol is bound to something imported, of the class given with
   it, and there is no code. *)
Definition imports_statics (imports : list (symbol * import_class))
    : static_env :=
  Static_env [] (map (fun '(s, c) => (s, Static_imported c)) imports).

(* Refinement of two units that import the same symbols, each run on its
   own. With nothing imported it is refinement from the empty static
   environment (FlambdaUnusedStaticsSpec.program_refines). Refinement
   under all environments ([expr_refines]) implies it. *)
Definition unit_refines (imports : list (symbol * import_class))
    (source target : expr) : Prop :=
  refines (State source [] [] (imports_statics imports) empty_heap no_reads)
    (State target [] [] (imports_statics imports) empty_heap no_reads).

Definition expr_refines_unit_refines_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (source target : expr),
    expr_refines source target -> unit_refines imports source target.

(* The static environment binds no symbol and no code id that the term
   binds: the names of its static bindings, wherever they are but in
   code (FlambdaWellFormedSpec.static_names, which does not look into
   code; the code of a well-formed unit binds nothing static). It is
   for a law of a unit run from a static environment that binds
   something, where the law moves a static binding or reads what one
   binds: a binding of a name that is bound already is undefined, and
   a unit may hold one that its run never reaches. *)
Definition static_names_free_in (statics : static_env) (e : expr) : bool :=
  forallb (fun s => negb (is_bound s (static_values statics)))
    (static_names symbols_of_binding e)
  && forallb (fun id => negb (is_bound id (static_code statics)))
       (static_names code_ids_of_binding e).

(* ======================================================================== *)
(* A prefix of static bindings                                              *)
(* ======================================================================== *)

(* Two terms that follow one group of static bindings. Where the group
   evaluates, with no variable in scope, from a static environment to
   another (FlambdaSemanticsSpec.eval_static_bindings: it binds no name
   twice and none that the first environment binds, and each of its
   constants has its value), the two terms with the group in front
   refine each other from the first environment exactly when the two
   terms without it do from the second. The heap and the rest of the
   state are the same on both sides. So what is validated of two terms
   that follow a prefix is said of the two terms run from the
   environment that the prefix makes. Where the group does not
   evaluate, the two terms with it in front are undefined, and their
   refinement says nothing. *)
Definition prefix_refines_statement : Prop :=
  ∀ (prefix : list static_binding) (statics statics' : static_env)
    (hp : heap) (rd : read_record) (source target : expr),
    eval_static_bindings [] prefix statics = Some statics' ->
    refines (State (Let_static prefix source) [] [] statics hp rd)
      (State (Let_static prefix target) [] [] statics hp rd)
    <-> refines (State source [] [] statics' hp rd)
          (State target [] [] statics' hp rd).

(* ======================================================================== *)
(* What a primitive does with an imported symbol                            *)
(* ======================================================================== *)

(* Two identities are the same when they are one symbol. *)
Definition phys_equal_identities_statement : Prop :=
  ∀ (statics : static_env) (s1 s2 : symbol),
    identity_of statics (Value_symbol s1) = Some s1 ->
    identity_of statics (Value_symbol s2) = Some s2 ->
    eval_phys_equal statics (Value_symbol s1) (Value_symbol s2)
    = Some (Pos.eqb s1 s2).

(* An imported symbol of the class of identities or of data is not a
   tagged immediate. *)
Definition phys_equal_imported_immediate_statement : Prop :=
  ∀ (statics : static_env) (s : symbol) (v : value),
    is_imported_pointer statics (Value_symbol s) = true ->
    is_tagged_immediate v = true ->
    eval_phys_equal statics (Value_symbol s) v = Some false
    ∧ eval_phys_equal statics v (Value_symbol s) = Some false.

(* An identity is not an allocated value of the unit, and is not an
   import that is data. *)
Definition phys_equal_identity_other_statement : Prop :=
  ∀ (statics : static_env) (s : symbol) (v : value),
    identity_of statics (Value_symbol s) = Some s ->
    is_allocated statics v = true
    ∨ is_imported_pointer statics v = true ∧ identity_of statics v = None ->
    eval_phys_equal statics (Value_symbol s) v = Some false
    ∧ eval_phys_equal statics v (Value_symbol s) = Some false.

(* An import that is data is compared with nothing but a tagged
   immediate and an identity: against an allocated value, another
   import that is data and itself there is no answer. *)
Definition phys_equal_data_import_statement : Prop :=
  ∀ (statics : static_env) (s : symbol) (v : value),
    imported_of statics (Value_symbol s) = Some (s, Import_data) ->
    is_tagged_immediate v = false ->
    identity_of statics v = None ->
    eval_phys_equal statics (Value_symbol s) v = None
    ∧ eval_phys_equal statics v (Value_symbol s) = None.

(* An import of which nothing is known is compared with nothing: there
   is no answer against any value, itself included. *)
Definition phys_equal_unknown_import_statement : Prop :=
  ∀ (statics : static_env) (s : symbol) (v : value),
    imported_of statics (Value_symbol s) = Some (s, Import_unknown) ->
    eval_phys_equal statics (Value_symbol s) v = None
    ∧ eval_phys_equal statics v (Value_symbol s) = None.

(* Of which imports the model knows that they are pointers: of every
   one that is not of the class [Import_unknown], and of none that is.
   Where the value is not an import of that class, [is_imported_pointer]
   is [is_imported]. *)
Definition imported_pointer_statement : Prop :=
  (∀ (statics : static_env) (v : value),
    (∀ s : symbol, imported_of statics v <> Some (s, Import_unknown)) ->
    is_imported_pointer statics v = is_imported statics v)
  ∧ (∀ (statics : static_env) (v : value) (s : symbol),
      imported_of statics v = Some (s, Import_unknown) ->
      is_imported statics v = true
      ∧ is_imported_pointer statics v = false).

(* The comparison of two values as it is of the classes of identities
   and of data alone: the body of FlambdaPrimitiveSpec.eval_phys_equal
   with [is_imported] where that has [is_imported_pointer]. It is here
   for the statement below; nothing else uses it. *)
Definition eval_phys_equal_two_classes (statics : static_env)
    (v1 v2 : value) : option bool :=
  match
    int_of_value Standard_tagged_immediate v1,
    int_of_value Standard_tagged_immediate v2
  with
  | Some a, Some b => Some (Z.eqb a b)
  | Some _, None =>
    if is_allocated statics v2 || is_imported statics v2 then Some false
    else None
  | None, Some _ =>
    if is_allocated statics v1 || is_imported statics v1 then Some false
    else None
  | None, None =>
    match identity_of statics v1, identity_of statics v2 with
    | Some s1, Some s2 => Some (Pos.eqb s1 s2)
    | Some _, None =>
      if is_allocated statics v2 || is_imported statics v2 then Some false
      else None
    | None, Some _ =>
      if is_allocated statics v1 || is_imported statics v1 then Some false
      else None
    | None, None => None
    end
  end.

(* The same of FlambdaHeapSpec.eval_phys_equal_location. *)
Definition eval_phys_equal_location_two_classes (statics : static_env)
    (v1 v2 : value) : option bool :=
  match v1, v2 with
  | Value_location l1, Value_location l2 => Some (Pos.eqb l1 l2)
  | Value_location _, v | v, Value_location _ =>
    if is_some (int_of_value Standard_tagged_immediate v)
       || is_allocated statics v || is_imported statics v
    then Some false
    else None
  | _, _ => None
  end.

(* A static environment that binds no symbol to an import of the class
   [Import_unknown]. *)
Definition no_import_unknown (statics : static_env) : Prop :=
  ∀ s : symbol,
    lookup s (static_values statics) <> Some (Static_imported Import_unknown).

(* In such an environment the two comparisons are those two functions,
   of every two values. *)
Definition phys_equal_two_classes_statement : Prop :=
  ∀ (statics : static_env) (v1 v2 : value),
    no_import_unknown statics ->
    eval_phys_equal statics v1 v2
    = eval_phys_equal_two_classes statics v1 v2
    ∧ eval_phys_equal_location statics v1 v2
      = eval_phys_equal_location_two_classes statics v1 v2.

(* A run that starts in such an environment stays in such environments:
   a static binding binds no import. So a unit whose imports are of the
   classes of identities and of data is run in such environments only:
   the environment of its imports is one, and so is that of every state
   that its run reaches. *)
Definition unit_two_classes_statement : Prop :=
  (∀ (fuel : nat) (s reached : state),
    run fuel s = Step reached ->
    no_import_unknown (state_statics s) ->
    no_import_unknown (state_statics reached))
  ∧ (∀ (imports : list (symbol * import_class)) (e : expr) (fuel : nat)
       (reached : state),
      (∀ (s : symbol) (c : import_class),
        (s, c) ∈ imports -> c <> Import_unknown) ->
      run fuel (State e [] [] (imports_statics imports) empty_heap no_reads)
      = Step reached ->
      no_import_unknown (state_statics reached)).

(* Of the unary primitives only [Opaque_identity] has a value on an
   imported symbol of any class, which is the symbol; of the binary
   primitives only [Phys_equal] has a value where an argument is one. *)
Definition imported_not_read_statement : Prop :=
  (∀ (statics : static_env) (p : unary_primitive) (v result : value),
    is_imported statics v = true ->
    eval_unary_primitive statics p v = Some result ->
    p = Opaque_identity Value ∧ result = v)
  ∧ (∀ (statics : static_env) (p : binary_primitive) (v1 v2 result : value),
      is_imported statics v1 = true ∨ is_imported statics v2 = true ->
      eval_binary_primitive statics p v1 v2 = Some result ->
      ∃ op : equality_comparison, p = Phys_equal op).

(* ======================================================================== *)
(* Two handlers, as the front end makes them                                *)
(* ======================================================================== *)

(* The names of the examples: two continuations that the unit does not
   bind, three predefined exceptions, and a symbol of the unit. *)
Definition example_leave : continuation := 1%positive.
Definition example_error : continuation := 2%positive.
Definition example_not_found : symbol := 1%positive.
Definition example_division_by_zero : symbol := 2%positive.
Definition example_failure : symbol := 3%positive.
Definition example_bucket : symbol := 4%positive.

Definition example_imports : list (symbol * import_class) :=
  [(example_not_found, Import_identity);
   (example_division_by_zero, Import_identity);
   (example_failure, Import_identity)].

(* [Failure 7], bound statically: a block whose first field is the
   exception. *)
Definition bind_example_bucket (body : expr) : expr :=
  Let_static
    [Bind_symbol example_bucket
       (Static_const_block 0
          [Symbol example_failure; Const (Tagged_immediate 7)])]
    body.

(* [try raise] of what [in_body] says [with] the cases of [handler_body],
   which has the exception as the variable [exn]: the handler is
   installed on the way into the body, and the body raises to it. *)
Definition example_try (exn : variable) (raised : simple)
    (handler_body : expr) : expr :=
  let handler : continuation := 3%positive in
  let body : continuation := 4%positive in
  Let_cont_non_recursive handler
    (Handler [exn] (make_exn_handler_metadata [Value]) handler_body)
    (Let_cont_non_recursive body
       (Handler [] (make_handler_metadata [])
          (Apply_cont
             (Apply_cont_expr handler [raised]
                (Some (Pop handler (Some Regular))))))
       (Apply_cont (Apply_cont_expr body [] (Some (Push handler))))).

(* The exception is raised again, to the exception continuation of the
   unit. *)
Definition example_reraise (exn : variable) : apply_cont :=
  Apply_cont_expr example_error [Var exn]
    (Some (Pop example_error (Some Reraise))).

(* [try raise raised with Not_found -> 2]: the handler compares what it
   caught with the symbol of the exception, and raises it again if it is
   another (ocaml/try_not_found.ml of the examples made from OCaml
   sources, without its regions). *)
Definition catch_by_name (raised : symbol) : expr :=
  let exn : variable := 1%positive in
  let same : variable := 2%positive in
  example_try exn (Symbol raised)
    (Let same (Naked_number Kind_naked_immediate)
       (Prim (Binary (Phys_equal Phys_eq))
          [Var exn; Symbol example_not_found])
       (Switch (Var same)
          [(0%Z, example_reraise exn);
           (1%Z,
            Apply_cont_expr example_leave [Const (Tagged_immediate 2)]
              None)])).

(* The handler catches [Not_found], and lets [Division_by_zero] and
   [Failure 7] go by: the first is another identity, the second a block
   of the unit. *)
Definition catches_by_name_statement : Prop :=
  let statics := imports_statics example_imports in
  behaves
    (State (catch_by_name example_not_found) [] [] statics empty_heap no_reads)
    (Exits example_leave [Value_const (Tagged_immediate 2)] empty_heap)
  ∧ behaves
      (State (catch_by_name example_division_by_zero) [] [] statics
         empty_heap no_reads)
      (Exits example_error [Value_symbol example_division_by_zero]
         empty_heap)
  ∧ behaves
      (State (bind_example_bucket (catch_by_name example_bucket)) [] []
         statics empty_heap no_reads)
      (Exits example_error [Value_symbol example_bucket] empty_heap).

(* [try raise raised with Failure n -> n], where the unit has bound
   [Failure 7] statically: the handler loads the first field of what it
   caught, compares that with the symbol of the exception, and on a
   match returns the second field (ocaml/exn_with_argument.ml). *)
Definition catch_with_argument (raised : symbol) : expr :=
  let exn : variable := 1%positive in
  let same : variable := 2%positive in
  let first : variable := 3%positive in
  let argument : variable := 4%positive in
  let matched : continuation := 5%positive in
  let load (field : Z) : named :=
    Prim
      (Unary (Block_load (Block_access_values None None Any_value) Immutable
                field))
      [Var exn]
  in
  bind_example_bucket
    (example_try exn (Symbol raised)
       (Let first Value (load 0%Z)
          (Let same (Naked_number Kind_naked_immediate)
             (Prim (Binary (Phys_equal Phys_eq))
                [Var first; Symbol example_failure])
             (Let_cont_non_recursive matched
                (Handler [] (make_handler_metadata [])
                   (Let argument Value (load 1%Z)
                      (Apply_cont
                         (Apply_cont_expr example_leave [Var argument]
                            None))))
                (Switch (Var same)
                   [(0%Z, example_reraise exn);
                    (1%Z, Apply_cont_expr matched [] None)]))))).

(* The handler catches the [Failure] that the unit made, and reads its
   argument. It is undefined on [Not_found], whose contents the model
   does not know. *)
Definition catches_with_argument_statement : Prop :=
  let statics := imports_statics example_imports in
  behaves
    (State (catch_with_argument example_bucket) [] [] statics empty_heap
       no_reads)
    (Exits example_leave [Value_const (Tagged_immediate 7)] empty_heap)
  ∧ behaves
      (State (catch_with_argument example_not_found) [] [] statics
         empty_heap no_reads)
      Goes_wrong.

(* ======================================================================== *)
(* An import of which nothing is known                                      *)
(* ======================================================================== *)

(* A symbol for a value of another unit of which nothing is known, with
   the three predefined exceptions. *)
Definition example_hidden : symbol := 5%positive.

Definition example_imports_hidden : list (symbol * import_class) :=
  (example_hidden, Import_unknown) :: example_imports.

(* The import is passed through the identity and returned. *)
Definition carry_hidden : expr :=
  let kept : variable := 1%positive in
  Let kept Value
    (Prim (Unary (Opaque_identity Value)) [Symbol example_hidden])
    (Apply_cont (Apply_cont_expr example_leave [Var kept] None)).

(* The import is compared with itself. *)
Definition compare_hidden : expr :=
  let same : variable := 1%positive in
  Let same (Naked_number Kind_naked_immediate)
    (Prim (Binary (Phys_equal Phys_eq))
       [Symbol example_hidden; Symbol example_hidden])
    (Apply_cont (Apply_cont_expr example_leave [Var same] None)).

(* An import of which nothing is known is carried: the unit that passes
   it through the identity returns it. It is not inspected: the run of
   the unit that compares it with itself goes wrong, and so does that
   of the handler of [catch_by_name] that has caught it, which compares
   it with the symbol of an exception. *)
Definition unknown_import_statement : Prop :=
  let statics := imports_statics example_imports_hidden in
  behaves (State carry_hidden [] [] statics empty_heap no_reads)
    (Exits example_leave [Value_symbol example_hidden] empty_heap)
  ∧ behaves (State compare_hidden [] [] statics empty_heap no_reads) Goes_wrong
  ∧ behaves
      (State (catch_by_name example_hidden) [] [] statics empty_heap no_reads)
      Goes_wrong.
