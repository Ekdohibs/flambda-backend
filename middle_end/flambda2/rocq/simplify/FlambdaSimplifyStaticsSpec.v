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

(* The types that Simplify gives the symbols of a [Let_static], and the
   body of the [Let_static] rewritten with them.

   This file is to be reviewed: the definitions and the statements,
   against middle_end/flambda2/simplify/simplify_static_const.ml
   ([simplify_static_consts], [simplify_static_const_of_kind_value],
   [simplify_field_of_block], [simplify_or_variable],
   [simplify_static_const_block_type]) and, for the closures,
   middle_end/flambda2/simplify/simplify_set_of_closures.ml
   ([simplify_lifted_sets_of_closures]). It follows
   FlambdaSimplifyProjectionsSpec.v and FlambdaSimplifyBoxingSpec.v,
   which have the same for a block and a boxed number that a primitive
   makes, FlambdaSimplifyApplySpec.v, which has the type of a closure
   that a [Let] makes, and FlambdaSimplifyPrimitivesSpec.v, which has
   the type of an argument ([argument_type]), the function that
   computes it ([simplify_simples]) and what is assumed: strict
   satisfaction of the environment.

   At a group of static bindings Simplify defines the symbols that the
   group binds, and gives each one a type made from the types of the
   [Simple]s of its constant ([bind_result_sym]: [DE.define_symbol],
   then [DE.add_equation_on_symbol]; for a closure, when the set is
   simplified). The body is simplified with those types: of a load
   from a symbol bound to a block, of the unboxing of a symbol bound
   to a boxed number, or of the callee of a call that is a symbol
   bound to a closure, the provers answer as they do of a variable.
   The function that puts the rewrites together
   (FlambdaSimplifyPassSpec.v) defines the symbols with no types; here
   is what it needs to give them their types.

   The order is the compiler's ([simplify_static_consts]). The symbols
   that the group binds to closures are defined first, with no type.
   Then the other symbols, in the order of the group: the type of the
   constant is taken in the environment as it is then, in which the
   symbol itself, and the symbols that the group binds later to
   constants that are not closures, are not defined unless the
   environment had them defined before the group; the symbol is
   defined and given the type. ([add_static_bindings].) A constant
   that names a symbol not defined then gets no type here; in the
   compiler that is a fatal error, the symbol not being bound in the
   typing environment (discrepancy T3-11: the model's group may be
   recursive, and such a constant has a value). A constant that names
   a symbol that the group binds to a closure, later or not, does get
   its type: that symbol is defined by then. Last, the symbols bound
   to closures are given their types
   ([simplify_lifted_sets_of_closures]). The types of the value slots
   of all of them are taken first, in the environment in which the
   other symbols of the group have their types and no closure of the
   group has ([add_closure_bindings], whose first argument is that
   environment).

   The types ([static_const_type], from the types of the [Simple]s of
   the constant, [static_const_simples]):
   - a block: the model's static block is the compiler's immutable
     block of the shape [Value_only]. The compiler's mutable and
     unique static blocks, and those of a mixed shape, whose fields
     may hold naked numbers, are not constants of the model, and this
     says nothing of them. The type is that of the blocks of the tag
     whose fields have the types of the fields, which are aliases
     ([simplify_static_const_block_type]); it is the type that
     [Make_block] gives its variable
     (FlambdaSimplifyProjectionsSpec.make_block_type). The tag is to
     be one of a block that is scanned, and the types of the fields of
     the kind value: the compiler's tag is scannable by its type, and
     on a field whose kind is not the one that the shape gives it,
     which for this shape is value, the compiler stops with a fatal
     error ([simplify_field_of_block]).
   - a boxed number: the type of the boxed numbers whose contents have
     the type of the contents; it is the type that [Box_number] gives
     its variable. Which boxed number it is, is read off the kind of
     that type: the model's constant does not say. For contents that
     are a constant this is the type that the compiler computes
     ([T.this_boxed_float] and the like). For contents that are a
     variable it is not what the compiler does: there the compiler
     asks the environment for the type of the variable at the kind
     value, where the variable holds the naked number, and stops with
     a fatal error (discrepancy T3-10). The model's constant has a
     value there, and this is the type of that value.
   - an immutable string: the type of that string
     ([T.this_immutable_string]).
   - a closure: the type of the closures at its function slot of a
     set that has exactly the functions of the constant, each of the
     code named, and its value slots, each of the type of its
     [Simple]: FlambdaSimplifyApplySpec.closure_of_set_type and
     set_of_closures_type, the types that a closure made by a [Let]
     gets there. Simplify gives more: the types of the other closures
     of the set, each an alias of its symbol, and the mode of the
     allocation, which for a static closure is the heap. A constant
     whose function slot is not among the functions of its set gets a
     type that nothing has; such a constant has no value
     (FlambdaSemanticsSpec.eval_static_const).

   The statements:
   - a symbol that the static environment binds to the value of a
     constant, as a group that binds it leaves it, has the type of the
     constant ([static_const_typed_statement]), and that type speaks
     only of variables in scope ([static_const_type_in_scope_statement]);
   - so the environment in which the symbol is defined and given that
     type is satisfied ([symbol_typed_env_satisfies_statement]), and so
     is the environment that [add_static_bindings] makes, by the static
     environment that the group makes
     ([add_static_bindings_env_satisfies_statement]);
   - and the body of a [Let_static] may be rewritten under that
     environment ([let_static_typed_statement]);
   - the symbols that the environment made has defined are those that
     the environment had and those that the group binds
     ([add_static_bindings_defined_symbols_statement]: the function
     that puts the rewrites together reads them for the environment of
     the code that the group binds);
   - the type of a constant and the environment made commute with
     renaming ([static_const_type_in_rename_statement],
     [add_static_bindings_rename_statement]).
   - An example: the first field of a static block, loaded from its
     symbol, is the [Simple] that the field was given
     ([example_load_of_static_block_statement]).

   - The environment made is well formed
     ([add_static_bindings_invariant_statement],
     [add_static_bindings_well_formed_statement]): it has the invariant
     of typing environments
     (FlambdaTypingEnvInvariantSpec.typing_env_invariant) if the
     environment had it and had none of the symbols that the group
     binds defined; so it passes the test of well-formedness that the
     function which puts the rewrites together makes before it
     rewrites, and so does it with its variables out of scope, which
     is the first test of FlambdaSimplifyCodeEnvSpec.code_env_in. The
     two premises, plainly. The one on the environment is the model's
     invariant, which is stricter than what the compiler keeps
     (decision D32: the equations of the levels agree with the types
     of the environment, and a symbol that has a type is defined;
     discrepancies T2-1 and T2-4), so that an environment that the
     compiler has need not satisfy it. The one on the group, that its
     symbols are new to the environment, holds of a unit that binds
     each symbol once, which the compiler's units do though nothing
     checks it across a unit (C0-4); a term of the model that binds a
     symbol again is undefined there. It is about the bindings of the
     term: the compiler adds the types of the constants that it lifts
     by another function, which lets a symbol be defined already
     ([add_lifted_constant_state] of simplify/env/downwards_env.ml,
     with [maybe_already_defined]), and that is not modelled here. An
     example: a group of six bindings of every sort
     ([example_group_well_formed_statement]).

   Not here: that the test of well-formedness is passed from an
   environment that passes the test and has not the invariant (the
   test does not see that the levels agree). The types of the symbols
   as they are seen in the body of code, which runs later and under
   other variables, are in FlambdaSimplifyCodeEnvSpec.v, with an
   example of a call of a symbol bound to a closure.

   The statements of refinement assume nothing of the term: where a
   symbol is bound twice, or a constant has no value, the [Let_static]
   is undefined and anything refines it. The two statements that the
   environment made is well formed have the premise on the group that
   is said above.

   The proofs are in FlambdaSimplifyStaticsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvExamplesSpec
  FlambdaTypingEnvRefinementSpec FlambdaTypeRenameSpec FlambdaEnvRenameSpec
  FlambdaSimplifyPrimitivesSpec FlambdaSimplifyProjectionsSpec
  FlambdaSimplifyApplySpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The type of a static constant                                            *)
(* ======================================================================== *)

(* The [Simple]s of a static constant whose types make the type of its
   symbol: the fields of a block, the contents of a boxed number, the
   [Simple]s of the value slots of a closure. *)
Definition static_const_simples (const : static_const) : list simple :=
  match const with
  | Static_const_block _ fields => fields
  | Static_const_immutable_value_array elements => elements
  | Static_const_boxed_number contents => [contents]
  | Static_const_immutable_string _ => []
  | Static_const_set_of_closures _ value_slots
  | Static_const_closure_of_set _ _ _ value_slots =>
    map snd value_slots
  end.

(* The type that Simplify gives the symbol of a static constant, from
   the types of those [Simple]s. [None]: a block whose tag is not one
   of a block that is scanned, or one of whose fields is not of the
   kind value; a boxed number whose contents are not of a kind of
   number that has a boxed form; an immutable array, of which nothing
   is said here yet (the compiler gives its symbol the type of an
   immutable array of the types of its elements, [T.immutable_array]
   in [simplify_static_const_of_kind_value]). *)
Definition static_const_type (const : static_const)
    (tys : list flambda_type) : option flambda_type :=
  match const with
  | Static_const_block tag _ =>
    if
      (tag <? no_scan_tag)
      && forallb (fun ty => kind_eqb (FlambdaTypeGrammar.kind ty) Value) tys
    then Some (make_block_type tag tys)
    else None
  | Static_const_boxed_number _ =>
    match tys with
    | [ty] =>
      let* k := boxable_number_of_kind (FlambdaTypeGrammar.kind ty) in
      Some (box_number k ty Alloc_mode.Heap)
    | [] | _ :: _ :: _ => None
    end
  | Static_const_immutable_string bytes => Some (this_immutable_string bytes)
  | Static_const_immutable_value_array _ => None
  | Static_const_set_of_closures code value_slots =>
    Some (set_of_closures_type code (map fst value_slots) tys)
  | Static_const_closure_of_set slot _ function_decls value_slots =>
    Some (closure_of_set_type slot function_decls (map fst value_slots) tys)
  end.

(* The same, the types read in the environment
   (FlambdaSimplifyPrimitivesSpec.simplify_simples, which is
   [Simplify_simple.simplify_simple] at the normal name mode on each
   [Simple]). [None] also where the environment does not know one of
   the [Simple]s at that name mode, which is a fatal error of the
   compiler. *)
Definition static_const_type_in (tenv : Typing_env.t) (const : static_const)
    : option flambda_type :=
  let* answers := simplify_simples tenv (static_const_simples const) in
  static_const_type const (map fst answers).

(* A symbol bound to what the constant is, in a state that satisfies
   the environment in which the type was taken, has the type. The
   static environment is the one that the group which binds the symbol
   has made: it has the symbol bound to the value that the constant
   has under the variables. *)
Definition static_const_typed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env) (s : symbol)
    (const : static_const) (ty : flambda_type)
    (allocated : statically_allocated_value),
    env_satisfies tenv statics vars ->
    static_const_type_in tenv const = Some ty ->
    eval_static_const vars const = Some allocated ->
    lookup s (static_values statics) = Some allocated ->
    type_satisfied statics vars (Value_symbol s) ty.

(* The type speaks only of variables in scope. *)
Definition static_const_type_in_scope_statement : Prop :=
  ∀ (tenv : Typing_env.t) (const : static_const) (ty : flambda_type)
    (y : variable),
    static_const_type_in tenv const = Some ty ->
    y ∈ free_variables ty ->
    in_scope_in_terms tenv y.

(* ======================================================================== *)
(* The environment follows the term                                         *)
(* ======================================================================== *)

(* [bind_result_sym]: the symbol is defined and given the type. *)
Definition define_symbol_with_type (tenv : Typing_env.t) (s : symbol)
    (ty : flambda_type) : Typing_env.t :=
  Typing_env.replace_equation (Typing_env.add_symbol_definition tenv s)
    (Name_symbol s) ty.

(* The environment so made is satisfied where the static environment
   binds the symbol to the value of the constant. *)
Definition symbol_typed_env_satisfies_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env) (s : symbol)
    (const : static_const) (ty : flambda_type)
    (allocated : statically_allocated_value),
    static_const_type_in tenv const = Some ty ->
    eval_static_const vars const = Some allocated ->
    lookup s (static_values statics) = Some allocated ->
    env_satisfies_strictly tenv statics vars ->
    env_satisfies_strictly (define_symbol_with_type tenv s ty) statics vars.

Definition is_closure_const (const : static_const) : bool :=
  match const with
  | Static_const_set_of_closures _ _ | Static_const_closure_of_set _ _ _ _ =>
    true
  | Static_const_block _ _ | Static_const_boxed_number _
  | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    false
  end.

(* The symbols that a group binds to closures. *)
Fixpoint closure_symbols_of (bindings : list static_binding) : list symbol :=
  match bindings with
  | [] => []
  | Bind_symbol s const :: bindings =>
    if is_closure_const const
    then s :: closure_symbols_of bindings
    else closure_symbols_of bindings
  | (Bind_code _ _ | Bind_deleted_code _) :: bindings =>
    closure_symbols_of bindings
  end.

(* The symbols that a group binds to other constants, in the order of
   the group: each is defined, and given the type of its constant if it
   has one, taken in the environment before it is defined. *)
Fixpoint add_block_like_bindings (tenv : Typing_env.t)
    (bindings : list static_binding) : Typing_env.t :=
  match bindings with
  | [] => tenv
  | Bind_symbol s const :: bindings =>
    let tenv :=
      if is_closure_const const
      then tenv
      else
        match static_const_type_in tenv const with
        | Some ty => define_symbol_with_type tenv s ty
        | None => Typing_env.add_symbol_definition tenv s
        end
    in
    add_block_like_bindings tenv bindings
  | (Bind_code _ _ | Bind_deleted_code _) :: bindings =>
    add_block_like_bindings tenv bindings
  end.

(* The symbols that a group binds to closures, which are defined
   already, are given the types of their constants. The types are
   taken in [prior]: the environment in which the other symbols of the
   group have their types, and no closure of the group has. *)
Fixpoint add_closure_bindings (prior tenv : Typing_env.t)
    (bindings : list static_binding) : Typing_env.t :=
  match bindings with
  | [] => tenv
  | Bind_symbol s const :: bindings =>
    let tenv :=
      if is_closure_const const
      then
        match static_const_type_in prior const with
        | Some ty => Typing_env.replace_equation tenv (Name_symbol s) ty
        | None => tenv
        end
      else tenv
    in
    add_closure_bindings prior tenv bindings
  | (Bind_code _ _ | Bind_deleted_code _) :: bindings =>
    add_closure_bindings prior tenv bindings
  end.

(* The environment in the body of a [Let_static]
   ([Simplify_static_const.simplify_static_consts]). *)
Definition add_static_bindings (tenv : Typing_env.t)
    (bindings : list static_binding) : Typing_env.t :=
  let prior :=
    add_block_like_bindings
      (Typing_env.add_symbol_definitions tenv (closure_symbols_of bindings))
      bindings
  in
  add_closure_bindings prior prior bindings.

(* It is satisfied by the static environment that the group makes. *)
Definition add_static_bindings_env_satisfies_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics extended : static_env) (vars : env)
    (bindings : list static_binding),
    eval_static_bindings vars bindings statics = Some extended ->
    env_satisfies_strictly tenv statics vars ->
    env_satisfies_strictly (add_static_bindings tenv bindings) extended vars.

(* So the body of the [Let_static] is rewritten under it. *)
Definition let_static_typed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (bindings : list static_binding)
    (body body' : expr),
    expr_refines_under (add_static_bindings tenv bindings) body body' ->
    expr_refines_under tenv (Let_static bindings body)
      (Let_static bindings body').

(* ======================================================================== *)
(* The symbols defined, and renaming                                        *)
(* ======================================================================== *)

(* The symbols that the environment made has defined: those that the
   environment had, and those that the group binds. *)
Definition add_static_bindings_defined_symbols_statement : Prop :=
  ∀ (tenv : Typing_env.t) (bindings : list static_binding) (s : symbol),
    s ∈ Typing_env.defined_symbols (add_static_bindings tenv bindings)
    <-> s ∈ Typing_env.defined_symbols tenv
        ∨ ∃ const : static_const, Bind_symbol s const ∈ bindings.

(* The type of a constant in the renamed environment is the type
   renamed, and the environment made from the renamed environment and
   the renamed group is the environment made, renamed. Symbols, tags
   and the bytes of a string contain no variable and no continuation,
   and are as they were. *)
Definition static_const_type_in_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (const : static_const),
    static_const_type_in (rename_typing_env r tenv)
      (rename_static_const r const)
    = option_map (rename_type r) (static_const_type_in tenv const).

Definition add_static_bindings_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (bindings : list static_binding),
    add_static_bindings (rename_typing_env r tenv)
      (map (rename_static_binding r) bindings)
    = rename_typing_env r (add_static_bindings tenv bindings).

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The variables and the symbol are those of
   FlambdaTypingEnvExamplesSpec.v, and the environment that of the
   example of FlambdaSimplifyProjectionsSpec.v, in which [x] and [y]
   are defined at the kind value.

   [let_static s = Block 0 (x, y) in let q = Block_load s 0 in
   apply_cont k q]: the symbol [s] is given the type of the blocks of
   tag 0 whose two fields are [x] and [y]; of that type the prover
   answers that the field 0 is [x]; [q] is made equal to [x], and
   replaced by it in the body. *)
Definition example_load_of_static_block_statement : Prop :=
  ∀ k : continuation,
    let bindings :=
      [Bind_symbol example_s
         (Static_const_block 0 [Var example_x; Var example_y])]
    in
    expr_refines_under example_pair_env
      (Let_static bindings
         (Let example_q Value
            (Prim
               (Unary
                  (Block_load (Block_access_values (Some 0) (Some 2) Any_value)
                     Immutable 0))
               [Symbol example_s])
            (Apply_cont (Apply_cont_expr k [Var example_q] None))))
      (Let_static bindings
         (Let example_q Value (Simple (Var example_x))
            (Apply_cont (Apply_cont_expr k [Var example_x] None)))).

(* ======================================================================== *)
(* The environment made is well formed                                      *)
(* ======================================================================== *)

(* The environment made has the invariant of typing environments, if
   the environment had it and had none of the symbols that the group
   binds defined. *)
Definition add_static_bindings_invariant_statement : Prop :=
  ∀ (tenv : Typing_env.t) (bindings : list static_binding),
    typing_env_invariant tenv ->
    (∀ (s : symbol) (const : static_const),
      Bind_symbol s const ∈ bindings ->
      among s (Typing_env.defined_symbols tenv) = false) ->
    typing_env_invariant (add_static_bindings tenv bindings).

(* So it passes the test of well-formedness, and so does it with its
   variables out of scope ([Typing_env.closure_env]): the first test
   of FlambdaSimplifyCodeEnvSpec.code_env_in. *)
Definition add_static_bindings_well_formed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (bindings : list static_binding),
    typing_env_invariant tenv ->
    (∀ (s : symbol) (const : static_const),
      Bind_symbol s const ∈ bindings ->
      among s (Typing_env.defined_symbols tenv) = false) ->
    typing_env_well_formed (add_static_bindings tenv bindings) = true
    ∧ typing_env_well_formed
        (Typing_env.closure_env (add_static_bindings tenv bindings))
      = true.

(* A group of six bindings of every sort, under the environment of the
   example above, in which [x] and [y] are defined: a block of the two
   variables; a boxed constant; a closure whose value slot holds [x];
   a block that names that closure, which the group binds before it,
   the first block and a constant; a string; and a closure of a set
   that names itself for its slot and whose value slot holds the
   second block. The symbols are 1 to 6. *)
Definition example_group : list static_binding :=
  [Bind_symbol 1%positive
     (Static_const_block 0 [Var example_x; Var example_y]);
   Bind_symbol 2%positive
     (Static_const_boxed_number (Const (Naked_int64 5)));
   Bind_symbol 3%positive
     (Static_const_set_of_closures example_f [(example_w, Var example_x)]);
   Bind_symbol 4%positive
     (Static_const_block 1
        [Symbol 3%positive; Symbol 1%positive; Const (Tagged_immediate 3)]);
   Bind_symbol 5%positive (Static_const_immutable_string [104; 105]);
   Bind_symbol 6%positive
     (Static_const_closure_of_set 1%positive [(1%positive, 6%positive)]
        [(1%positive, example_f)] [(example_w, Symbol 4%positive)])].

(* The environment made passes the test, with its variables in scope
   and out of scope, and every symbol of the group has a type in it. *)
Definition example_group_well_formed_statement : Prop :=
  let after := add_static_bindings example_pair_env example_group in
  typing_env_well_formed after = true
  ∧ typing_env_well_formed (Typing_env.closure_env after) = true
  ∧ forallb
      (fun s =>
         match
           Keyed.lookup name_eqb (Name_symbol s)
             (Typing_env.names_to_types after)
         with
         | Some _ => true
         | None => false
         end)
      [1%positive; 2%positive; 3%positive; 4%positive; 5%positive;
       6%positive]
    = true.
