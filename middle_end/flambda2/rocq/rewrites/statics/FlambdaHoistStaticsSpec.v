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

(* The static bindings of a unit that need nothing from around them
   bound first: a normalisation of a unit that is run on its own.

   This file is to be reviewed: [is_hoisted], [hoisted_bindings],
   [without_hoisted] and [hoist_statics], which are extracted and run
   by the validation of middle_end/flambda2/tests/semantics/, and the
   statements.

   The two terms of a unit do not bind their symbols in the same
   places. Simplify lifts a closure, a block or a boxed number whose
   parts are symbols and constants out of the arm of a test, out of a
   loop and out of the code of a function, and places its binding at
   the nearest place at toplevel that it comes back through
   (FlambdaLiftFromArmsSpec.v); the validation, on the term before,
   binds such a symbol at the head of the unit, or where its [Let]
   stood. Where a symbol is bound does not matter if its binding needs
   nothing that the unit computes: [hoist_statics] binds every such
   binding first.

   What is bound first. The bindings of code, and the bindings of
   symbols whose constant has a value in the empty environment
   ([is_hoisted]): it mentions no variable, and it is not a closure at
   a function slot that its set has not got. They are taken from the
   groups of static bindings of the toplevel of the unit, in handlers
   too ([hoisted_bindings], in the order in which a run comes to
   them), and bound in one group around the unit with those bindings
   removed ([without_hoisted]). Bindings of deleted code, which bind
   nothing, and the bindings of symbols that mention a variable stay
   where they are. The code of functions and the handlers of
   recursive continuations are not looked into. This moves what
   FlambdaHoistCodeSpec.hoist_code moves, and the bindings of symbols
   besides. It does not accept every unit that that pass accepts: in
   place of that pass's check of closures it asks what is said next,
   of a unit that imports nothing, and a unit that mentions a symbol
   it imports passes there and not here.

   This is no equivalence of every unit. With its static bindings
   first a unit may be defined where it was not: where a closure is
   called before its code is bound, or a symbol is read before it is
   bound. So it is asked that every static name of the unit is in
   scope where it is mentioned, row 4 of well-formedness
   (FlambdaStaticsInScopeSpec.statics_in_scope, for a unit that
   imports nothing, with what that file asks beside it of the names of
   the unit): no state of the run then holds or looks up a symbol
   that is not bound, or a closure of code that is neither bound nor
   declared deleted
   (FlambdaStaticsInScopeSpec.statics_bound_in_run_statement).
   A closure may be of code that the unit declares deleted; a call of
   it is undefined, and stays so with the bindings first, as the unit
   binds no code id that it declares deleted
   (FlambdaStaticsInScopeSpec.static_names_apart). A unit that
   passes is equivalent, run on its own, to the unit with those
   bindings first ([hoist_statics_program_equivalent_statement]): the
   two runs are in step, with the same values, and differ only in when
   a static name is bound.

   The proofs are in FlambdaHoistStaticsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaWellFormedSpec FlambdaStaticsInScopeSpec FlambdaImportsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is bound first                                                      *)
(* ======================================================================== *)

(* A binding of code, or of a symbol to a constant that has a value in
   the empty environment. *)
Definition is_hoisted (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ _ => true
  | Bind_deleted_code _ => false
  | Bind_symbol _ const =>
    match eval_static_const [] const with
    | Some _ => true
    | None => false
    end
  end.

(* Those bindings of the groups of static bindings of an expression,
   those of the handler of a [Let_cont] after those of its body, as a
   run comes to them. *)
Fixpoint hoisted_bindings (e : expr) : list static_binding :=
  match e with
  | Let _ _ _ body => hoisted_bindings body
  | Let_static bindings body =>
    filter is_hoisted bindings ++ hoisted_bindings body
  | Let_cont_non_recursive _ h body =>
    hoisted_bindings body ++ handler_hoisted_bindings h
  | Let_cont_recursive _ body => hoisted_bindings body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_hoisted_bindings (h : continuation_handler)
    : list static_binding :=
  match h with
  | Handler _ _ handler => hoisted_bindings handler
  end.

(* The expression with those bindings removed. A group left with no
   binding stays. *)
Fixpoint without_hoisted (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (without_hoisted body)
  | Let_static bindings body =>
    Let_static (filter (fun binding => negb (is_hoisted binding)) bindings)
      (without_hoisted body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (handler_without_hoisted h)
      (without_hoisted body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (without_hoisted body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with handler_without_hoisted (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (without_hoisted handler)
  end.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The unit with those bindings first, if it is a well-formed unit
   that imports nothing and whose static names are in scope where they
   are mentioned. *)
Definition hoist_statics (u : unit_interface) (e : expr) : option expr :=
  if
    well_formed_unit u e
    && statics_in_scope [] e
    && interface_conts_not_rebound u e
    && static_names_apart [] e
  then Some (Let_static (hoisted_bindings e) (without_hoisted e))
  else None.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* A unit that passes and the unit with those bindings first, each run
   on its own from empty environments, behave alike. *)
Definition hoist_statics_program_equivalent_statement : Prop :=
  ∀ (u : unit_interface) (e e' : expr),
    hoist_statics u e = Some e' ->
    equivalent (State e [] [] empty_static_env empty_heap no_reads)
      (State e' [] [] empty_static_env empty_heap no_reads).

(* The same of a unit that imports symbols: if the unit binds no name
   that the imports bind, the two, each run from the static environment
   of its imports, behave alike. A binding that is taken to the head is
   executed before the bindings it was after, where a name that is an
   import is bound already.

   This is of the pass as it is, which asks that the unit mention no
   symbol that it does not bind ([statics_in_scope] with no import): a
   unit that mentions a symbol that only its imports bind does not
   pass, and is left as it is by the functions that call the pass. The
   pass is not given the imports: a unit that binds a symbol that is
   also an import may pass, and the premise here is then false of it.
   A pass that takes the imports of the unit to be in scope is not in
   this tree. *)
Definition hoist_statics_equivalent_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (u : unit_interface)
    (e e' : expr),
    hoist_statics u e = Some e' ->
    static_names_free_in (imports_statics imports) e = true ->
    equivalent (State e [] [] (imports_statics imports) empty_heap no_reads)
      (State e' [] [] (imports_statics imports) empty_heap no_reads).

(* The parts of the pass commute with renaming, which moves no symbol
   and no code id: a constant that has a value in the empty environment
   has one when it is renamed. *)
Definition hoisted_bindings_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    hoisted_bindings (rename_expr r e)
    = map (rename_static_binding r) (hoisted_bindings e).

Definition without_hoisted_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    rename_expr r (without_hoisted e) = without_hoisted (rename_expr r e).
