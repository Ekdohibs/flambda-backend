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

(* Removing, from the head of a unit, static bindings that are not used:
   the unit with them refines the unit without.

   This file is to be reviewed: [head_statics_removable] and what it
   is made of ([head_bindings], [under_head], [static_const_total],
   [removed_binding_total], [kept_code_binds_only_kept]), and
   [add_head_tables], which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/; [head_free_in]; and the six
   statements.

   FlambdaUnusedStaticsSpec.drop_statics removes static bindings that
   nothing refers to, and its result refines the term, run on its own:
   that is what a rewrite of the term before Simplify owes. The
   validation also needs the removal on the term after Simplify: once it
   has inlined, in both terms, a call that Simplify left, the term after
   still binds the code and the closure that the call used, and the term
   before, in which Simplify's own removal is followed, does not. A
   rewrite of the term after owes the converse: that the term refines
   its result.

   The converse does not hold of [drop_statics] as it is, for two
   reasons, and [drop_head_statics] is [drop_statics] under the checks
   that answer them.

   - A binding removed may have had no value: the group that binds it
     is then undefined, and the term without it is not. So each constant
     removed must have a value whatever the environment
     ([static_const_total]): its fields, elements or value slots are
     constants and symbols, no variable, and a closure of a set is at a
     slot that its set has.
   - A binding of a name that is bound already is undefined
     (FlambdaSemanticsSpec.eval_static_bindings). So the statement is of
     a term run on its own, from empty environments, and the bindings
     removed are among those of the groups at the head of the term
     ([head_bindings]), which run first; the names that those groups
     bind are pairwise distinct; and nothing under them, and no code
     that they keep, binds a name removed. Under an environment that
     binds a name removed the statement is false, and it is not made:
     this is not an equivalence under all environments.

   A second statement is of a term run from a static environment that
   binds something, as a unit is that runs after the units it uses
   ([drop_head_statics_refines_from_statement]). It asks of that
   environment what the empty one gives for nothing: that it binds
   none of the names that the groups at the head of the term bind
   ([head_free_in]), and that the names removed occur in none of the
   code and the values that it holds
   (FlambdaUnusedStaticsSpec.statics_free_of). The first statement is
   the second at the empty static environment. A third is of a term
   run under continuations as well, in which the names removed do not
   occur either ([drop_head_statics_refines_under_statement]): the
   second is the third with no continuation.

   Which names to remove is for the caller to find; a wrong guess is
   refused, not believed.

   Read the other way, the statement is that the term with more
   bindings at its head refines the term, where the checks pass of the
   larger term: a binding that nothing uses, with a name that nothing
   else binds and a constant that has a value, may be added there.
   That the term refines the larger one is
   FlambdaUnusedStaticsSpec.drop_statics_program_refines_statement,
   which needs no check beyond those of drop_statics.

   [add_head_tables] is that reading as a rewrite, for one kind of
   binding: a symbol bound to an immutable array of constants and
   symbols. Simplify makes such an array where it turns a [Switch]
   whose arms give constants into a load from a table
   (simplify/simplify_switch_expr.ml, the lookup tables), and binds
   it with the static bindings of the unit. The term before has no
   such binding. The validation puts the tables that it is given at
   the head of the term before, so that a rewrite of the term before
   which makes the load finds the table bound; that the term with
   them refines the term, each run on its own, is
   [add_head_tables_program_refines_statement]. The same is stated
   of a term run from a static environment that binds something and
   under continuations ([add_head_tables_refines_under_statement]),
   where the static environment binds no name of the head of the
   result and the symbols of the tables occur in nothing that the
   environments hold; and, as a case of it, of a unit run from its
   imports ([add_head_tables_refines_imports_statement]). Which tables
   to put is for the caller to find. Where the checks fail, the term is
   left as it is. Where they pass, the tables are bound whatever
   they hold: a table that nothing will use, or that is not one that
   Simplify made, is bound all the same, and the statement holds of
   it. Whether a table serves is not checked here: that is for the
   rewrite that reads it and for the comparison of the two terms.

   The proofs are in FlambdaUnusedHeadStaticsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaNormaliseSpec
  FlambdaUnusedStaticsSpec FlambdaLiftSpec FlambdaImportsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The head of a unit                                                       *)
(* ======================================================================== *)

(* The bindings of the groups of static bindings that a term begins
   with, in their order, and the term under those groups. *)
Fixpoint head_bindings (e : expr) : list static_binding :=
  match e with
  | Let_static bindings body => bindings ++ head_bindings body
  | Let _ _ _ _ | Let_cont_non_recursive _ _ _ | Let_cont_recursive _ _
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end.

Fixpoint under_head (e : expr) : expr :=
  match e with
  | Let_static _ body => under_head body
  | Let _ _ _ _ | Let_cont_non_recursive _ _ _ | Let_cont_recursive _ _
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

(* ======================================================================== *)
(* Constants that have a value whatever the environment                     *)
(* ======================================================================== *)

(* A simple that is a symbol or a constant is
   FlambdaLiftSpec.plain_simple. *)
Definition value_slots_plain (slots : list (value_slot * simple)) : bool :=
  forallb (fun slot => plain_simple (snd slot)) slots.

Definition static_const_total (const : static_const) : bool :=
  match const with
  | Static_const_set_of_closures _ slots => value_slots_plain slots
  | Static_const_closure_of_set slot _ decls slots =>
    match lookup slot decls with
    | Some _ => value_slots_plain slots
    | None => false
    end
  | Static_const_block _ fields => forallb plain_simple fields
  | Static_const_immutable_value_array elements =>
    forallb plain_simple elements
  | Static_const_boxed_number contents => plain_simple contents
  | Static_const_immutable_string _ => true
  end.

(* A binding that is kept, or whose constant has a value. *)
Definition removed_binding_total (D : dropped) (binding : static_binding)
    : bool :=
  match binding with
  | Bind_code _ _ | Bind_deleted_code _ => true
  | Bind_symbol s const => symbol_kept D s || static_const_total const
  end.

(* Code that is removed, or whose body binds none of the names removed. *)
Definition kept_code_binds_only_kept (D : dropped) (binding : static_binding)
    : bool :=
  match binding with
  | Bind_code id c => negb (code_kept D id) || code_binds_only_kept D c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end.

(* ======================================================================== *)
(* The removal                                                              *)
(* ======================================================================== *)

(* The checks. Nothing in the term refers to a name removed, outside
   the bindings that go (FlambdaUnusedStaticsSpec.expr_avoids). Under
   the groups of the head, nothing binds a name removed, and nor does
   the body of code that those groups bind and keep: so the bindings
   removed are bindings of those groups, and no code that runs binds
   one. The symbols that the groups of the head bind are pairwise
   distinct, and so are the code ids: with the static environment
   empty at the start, each binding of those groups is of a name that
   is not bound yet. Each symbol removed is bound to a constant that
   has a value. *)
Definition head_statics_removable (D : dropped) (e : expr) : bool :=
  expr_avoids D e
  && binds_only_kept D (under_head e)
  && forallb (kept_code_binds_only_kept D) (head_bindings e)
  && distinct (flat_map symbols_of_binding (head_bindings e))
  && distinct (flat_map code_ids_of_binding (head_bindings e))
  && forallb (removed_binding_total D) (head_bindings e).

(* FlambdaUnusedStaticsSpec.drop_statics, under the checks. *)
Definition drop_head_statics (D : dropped) (e : expr) : option expr :=
  if head_statics_removable D e then drop_statics D e else None.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

(* The term refines its result, each run on its own. With
   FlambdaUnusedStaticsSpec.drop_statics_program_refines_statement, of
   which [drop_head_statics] is a case, the two behave alike from empty
   environments. *)
Definition drop_head_statics_program_refines_statement : Prop :=
  ∀ (D : dropped) (e e' : expr),
    drop_head_statics D e = Some e' -> program_refines e' e.

(* ======================================================================== *)
(* Read the other way: tables bound at the head                             *)
(* ======================================================================== *)

(* A table is a symbol with the elements of an immutable array. The
   static binding of one, and the names to remove that are the symbols
   of the tables given. *)
Definition table_binding (table : symbol * list simple) : static_binding :=
  Bind_symbol (fst table) (Static_const_immutable_value_array (snd table)).

Definition tables_dropped (tables : list (symbol * list simple)) : dropped :=
  Dropped (map fst tables) [].

(* Every element of every table is a constant or a symbol. *)
Definition tables_plain (tables : list (symbol * list simple)) : bool :=
  forallb (fun table => forallb plain_simple (snd table)) tables.

(* The term under one group of static bindings of the tables given,
   where every element is a constant or a symbol and the checks for
   the removal of exactly their symbols pass of the larger term
   ([head_statics_removable]): nothing in the term refers to one of
   those symbols or binds it, and the symbols are distinct from each
   other and from those that the groups at the head of the term bind.
   The term as it is where that fails, and where no table is given. *)
Definition add_head_tables (tables : list (symbol * list simple)) (e : expr)
    : expr :=
  match tables with
  | [] => e
  | _ :: _ =>
    let e' := Let_static (map table_binding tables) e in
    if tables_plain tables && head_statics_removable (tables_dropped tables) e'
    then e'
    else e
  end.

(* The result refines the term, each run on its own. It is not a
   refinement under all environments, nor from a static environment
   that binds one of the symbols: a group that was put binds that
   symbol again there, and a name bound twice is undefined. *)
Definition add_head_tables_program_refines_statement : Prop :=
  ∀ (tables : list (symbol * list simple)) (e : expr),
    program_refines e (add_head_tables tables e).

(* ======================================================================== *)
(* From a static environment that binds something                           *)
(* ======================================================================== *)

(* The static environment binds no symbol and no code id that the
   groups at the head of the term bind. *)
Definition head_free_in (statics : static_env) (e : expr) : bool :=
  forallb (fun s => negb (is_bound s (static_values statics)))
    (flat_map symbols_of_binding (head_bindings e))
  && forallb (fun id => negb (is_bound id (static_code statics)))
       (flat_map code_ids_of_binding (head_bindings e)).

(* The term refines its result, each run with no variable and no
   continuation in scope from one static environment and one heap,
   where the names removed occur in nothing that the static
   environment holds and it binds no name of the head of the term.
   The second is asked of every name of the head: a name removed
   that is bound already makes the term undefined where its result is
   not, and a name kept that is bound already makes both undefined. *)
(* The heap is one in which the names removed occur in no value of a
   cell (FlambdaUnusedStaticsSpec.heap_free_of), as for
   FlambdaUnusedStaticsSpec.drop_statics_refines_statement. There the
   premise is needed: the term refines its result, and a symbol that
   only the heap names would be removed from under a read. Here the
   refinement is the other way: a read through such a symbol makes the
   result undefined, which this direction allows. The premise is asked
   here because the simulation that proves the statement keeps the
   condition that no value in a cell mentions a name removed,
   alongside its conditions on the term and the environments. It is
   not known to be needed, and the same holds of the two statements
   below. *)
Definition drop_head_statics_refines_from_statement : Prop :=
  ∀ (D : dropped) (e e' : expr) (statics : static_env) (hp : heap)
    (rd : read_record),
    drop_head_statics D e = Some e' ->
    statics_free_of D statics = true ->
    heap_free_of D hp = true ->
    head_free_in statics e = true ->
    refines (State e' [] [] statics hp rd) (State e [] [] statics hp rd).

(* The same under continuations: the term refines its result, each run
   with no variable in scope from one static environment, one heap and
   one continuation environment, where the names removed occur in
   nothing that the continuations and the static environment hold
   (FlambdaUnusedStaticsSpec.envs_free_of: no handler of a continuation
   refers to a name removed or binds one) and the static environment
   binds no name of the head of the term. A continuation whose handler
   binds a name removed would make the term undefined where its result
   is not, when the handler runs; one whose handler refers to a name
   removed is refused with it, though the result would only go wrong
   there. The statement above is this one with no continuation. *)
(* The heap is as above: the names removed occur in no value of a cell. *)
Definition drop_head_statics_refines_under_statement : Prop :=
  ∀ (D : dropped) (e e' : expr) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record),
    drop_head_statics D e = Some e' ->
    envs_free_of D [] conts statics = true ->
    heap_free_of D hp = true ->
    head_free_in statics e = true ->
    refines (State e' [] conts statics hp rd) (State e [] conts statics hp rd).

(* Tables bound at the head, from a static environment that binds
   something and under continuations: the result refines the term, each
   run with no variable in scope from one static environment, one heap
   and one continuation environment, where the symbols of the tables
   occur in nothing that the continuations and the static environment
   hold, and the static environment binds no name of the head of the
   result. Where [add_head_tables] leaves the term as it is, the second
   premise is of the head of the term, and nothing is used of either.
   [add_head_tables_program_refines_statement] is this at the empty
   environments. *)
(* The heap is as above: the symbols of the tables occur in no value of
   a cell. *)
Definition add_head_tables_refines_under_statement : Prop :=
  ∀ (tables : list (symbol * list simple)) (e : expr) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    envs_free_of (tables_dropped tables) [] conts statics = true ->
    heap_free_of (tables_dropped tables) hp = true ->
    head_free_in statics (add_head_tables tables e) = true ->
    refines (State e [] conts statics hp rd)
      (State (add_head_tables tables e) [] conts statics hp rd).

(* Of a unit run from its imports (FlambdaImportsSpec.unit_refines):
   the static environment of the imports holds no code and no value in
   which a symbol could occur, so what is left to ask is that no import
   is a name that the groups at the head of the result bind.
   FlambdaImportsSpec.static_names_free_in of the result gives that
   (FlambdaValidateImportsSpec.static_names_free_in_head_statement). *)
Definition add_head_tables_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class))
    (tables : list (symbol * list simple)) (e : expr),
    head_free_in (imports_statics imports) (add_head_tables tables e) = true ->
    unit_refines imports e (add_head_tables tables e).
