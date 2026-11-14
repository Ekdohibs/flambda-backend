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


(* A rewrite that takes what is known of symbols, applied in the bodies
   of code and in the handlers of recursive continuations with what the
   static bindings in scope make known. With it the unboxing of the
   parameters of continuations knows, in a function and in a loop, the
   static blocks and boxed numbers of the unit.

   This file is to be reviewed: [facts_of_symbols], [symbols_only] and
   [symbols_bound_as], which say what is known of symbols and when it
   holds; [rewrite_code_in_scope] and [rewrite_rec_handlers_in_scope],
   which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/; and the statements. The
   validation uses the two in the first phase of its normalisation of
   the term before (FlambdaValidateSpec.normalise_params_round).

   What Simplify does. A symbol has a type from where it is bound, and
   the type of a symbol bound to a block or to a boxed number says what
   it is bound to (simplify/simplify_static_const.ml,
   [simplify_static_const_of_kind_value], which adds the equation on the
   symbol). The code of a function is simplified where the set of
   closures that the function is of is simplified, under the types of
   the symbols bound by then (simplify/simplify_set_of_closures.ml; in
   a group of static bindings the constants that are not closures come
   first, [simplify_static_consts]). The unboxing of a parameter of a
   continuation reads the type of the argument of each application in
   the environment of that application
   (simplify/unboxing/unboxing_epa.ml, [unbox_arg]). So where one path
   of a function gives a join a constant block, [Some 8] say, which
   closure conversion has made a symbol of the unit, and another gives
   [None], Simplify unboxes the parameter of the join as it does where
   the block is made by a [Let]: the application that gives the symbol
   gives the field of its block.

   FlambdaUnboxVariantsSpec.unbox_variants_knowing does that given what
   is known, and FlambdaUnboxVariantsSpec.unbox_variant_params starts
   it with nothing known. Down a term it learns the static blocks and
   boxed numbers of each [Let_static] that it goes under. It does not
   go into code, nor into the handler of a recursive continuation: the
   validation takes a rewrite there by FlambdaNormaliseSpec.rewrite_code
   and FlambdaRecHandlersSpec.rewrite_rec_handlers, which give the
   rewrite the body or the handler alone. So in a function, and in a
   loop, nothing was known of the symbols of the unit.

   What is here. [rewrite_code_in_scope f known] and
   [rewrite_rec_handlers_in_scope f known] are those two with a rewrite
   [f] that takes what is known, and they give it what the static
   bindings in scope make known of their symbols
   ([symbols_under_statics]).

   Where a fact about a symbol is used. Only where the symbol is bound,
   by the place in the term:
   - the body of code is given what is known of the symbols of the
     [Let_static]s that its binding is under, and of the symbols of its
     own group. Code runs at a call, and a call finds the code in the
     static environment, where the [Let_static] of its group put it
     when it bound the symbols of the group: a [Let_static] binds all
     of its group in one step, and a symbol or a code id that is bound
     already is not bound again (FlambdaSemanticsSpec.eval_static_bindings
     has no value). So when the code runs, the symbols of its group are
     bound to what the group binds them to; and so are those of the
     [Let_static]s above, which ran before;
   - the handler of a recursive continuation is given what is known of
     the symbols of the [Let_static]s that its [Let_cont] is under,
     which ran before the handler is first applied.
   A symbol bound further down the unit than the code or the loop is
   not known there, though it may be bound by the time the code runs:
   to know it would take a statement about a unit run on its own
   (FlambdaKnownSymbolsSpec.fold_known_unit_program_refines_statement
   is one), and none is made here. A symbol that the unit imports is
   bound by no [Let_static] of the unit: nothing is known of it.

   So this is a rewrite that refines under all environments, with no
   premise on the unit ([rewrite_code_in_scope_refines_statement],
   [rewrite_rec_handlers_in_scope_refines_statement], at
   [no_facts]). The statements are of every state in which what is
   known holds ([symbols_bound_as]), since that is the form in which
   one is used inside the other: the handlers of the loops of a
   function are rewritten with what is known where the function is
   bound.

   What is known of symbols alone ([facts_of_symbols]). In code no
   variable of the unit is in scope, and a variable of the unit may be
   bound again in a handler. So of a static block whose field is a
   variable, that field is forgotten; the tag and the fields that are
   symbols and constants are kept. A read that a [Let] has made is of
   variables, and none is kept. What is left mentions no variable
   ([symbols_only]), holds under any variables, and is not changed by a
   renaming.

   Against Simplify: this knows of the symbols of a group in the code
   of that group, which Simplify knows where it simplifies the code
   under the closure; and it does not know, in code bound at the head
   of a unit, a symbol bound further down, which Simplify knows if the
   closure is made further down still. The validation is of terms
   after closure conversion, which binds the constants of a unit at
   its head.

   [unbox_variants_knowing] given what is known of symbols alone
   refines wherever that holds
   ([unbox_variants_knowing_symbols_refines_statement]): it is the
   rewrite that the validation gives the two.

   That the two commute with renaming and keep scope is stated in
   FlambdaUnboxInScopeRenameSpec.v. The proofs are in
   FlambdaUnboxInScopeProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaNormaliseSpec
  FlambdaBlockLoadSpec FlambdaUnboxSpec FlambdaUnboxParamsSpec
  FlambdaUnboxVariantsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known of symbols                                                 *)
(* ======================================================================== *)

(* A field that is known to hold a symbol or a constant; one that holds
   a variable is not known. *)
Definition closed_field (field : option simple) : option simple :=
  match field with
  | Some (Symbol _ | Const _) => field
  | Some (Var _) | None => None
  end.

(* What is known of symbols, with no variable in it: nothing of a
   variable, no read, and of the block of a symbol its tag and the
   fields that hold symbols and constants. *)
Definition facts_of_symbols (known : facts) : facts :=
  Facts
    (Known_blocks []
       (map
          (fun entry =>
             (fst entry,
              (fst (snd entry), map closed_field (snd (snd entry)))))
          (known_symbols (facts_blocks known))))
    (Known_numbers [] (known_boxed_symbols (facts_numbers known)))
    [].

(* What is known is of symbols alone. *)
Definition symbols_only (known : facts) : Prop :=
  facts_of_symbols known = known.

(* What is known of symbols under a group of static bindings: of the
   symbols known before, and of those of the group. *)
Definition symbols_under_statics (known : facts)
    (bindings : list static_binding) : facts :=
  facts_of_symbols (facts_under_statics known bindings).

(* The value is that of the field, if the field is known. A field that
   holds a variable has no value here. *)
Definition field_as (field : option simple) (v : value) : Prop :=
  match field with
  | Some s => eval_simple [] s = Some v
  | None => True
  end.

(* What is known of symbols holds in a static environment: a symbol of
   which a block is known is bound to a block of that tag whose fields
   are as they are known to be, and a symbol known to be a boxed
   constant is bound to the box of it. *)
Definition symbols_bound_as (known : facts) (statics : static_env) : Prop :=
  (∀ (s : symbol) (tag : Z) (fields : list (option simple)),
    (s, (tag, fields)) ∈ known_symbols (facts_blocks known) ->
    ∃ vs : list value,
      Forall2 field_as fields vs
      ∧ lookup s (static_values statics) = Some (Static_block tag vs))
  ∧ (∀ (s : symbol) (c : const),
      (s, c) ∈ known_boxed_symbols (facts_numbers known) ->
      lookup s (static_values statics)
      = Some (Static_boxed_number (Value_const c))).

(* ======================================================================== *)
(* The two rewrites                                                         *)
(* ======================================================================== *)

(* [f], given what is known of the symbols in scope, applied to the body
   of every piece of code. The symbols in scope where code is bound are
   those of the groups of static bindings that the binding is under and
   those of its own group. As in FlambdaNormaliseSpec.rewrite_code, the
   body is left alone if the rewrite would change whether it binds
   something static. *)
Fixpoint rewrite_code_in_scope (f : facts -> expr -> expr) (known : facts)
    (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (rewrite_code_in_scope f known body)
  | Let_static bindings body =>
    let known := symbols_under_statics known bindings in
    Let_static (map (rewrite_code_in_binding (f known)) bindings)
      (rewrite_code_in_scope f known body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (rewrite_code_in_scope_in_handler f known h)
      (rewrite_code_in_scope f known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, rewrite_code_in_scope_in_handler f known h))
         handlers)
      (rewrite_code_in_scope f known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with rewrite_code_in_scope_in_handler (f : facts -> expr -> expr)
    (known : facts) (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (rewrite_code_in_scope f known handler)
  end.

(* [f], given what is known of the symbols in scope, applied to the
   handler of every recursive continuation; in a handler that has a
   recursive [Let_cont] of its own, the handlers of that one first, as
   in FlambdaRecHandlersSpec.rewrite_rec_handlers. The symbols in scope
   are those of the groups of static bindings that the [Let_cont] is
   under. *)
Fixpoint rewrite_rec_handlers_in_scope (f : facts -> expr -> expr)
    (known : facts) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d (rewrite_rec_handlers_in_scope f known body)
  | Let_static bindings body =>
    Let_static bindings
      (rewrite_rec_handlers_in_scope f
         (symbols_under_statics known bindings) body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (rewrite_rec_handlers_in_scope_in_handler f known h)
      (rewrite_rec_handlers_in_scope f known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, rewrite_rec_handler_in_scope f known h))
         handlers)
      (rewrite_rec_handlers_in_scope f known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with rewrite_rec_handlers_in_scope_in_handler (f : facts -> expr -> expr)
    (known : facts) (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (rewrite_rec_handlers_in_scope f known handler)
  end
with rewrite_rec_handler_in_scope (f : facts -> expr -> expr)
    (known : facts) (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (rewrite_code_body (f known)
         (rewrite_rec_handlers_in_scope f known handler))
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* A rewrite that takes what is known of symbols refines wherever that
   holds: in every state whose static environment binds the symbols as
   they are known to be. *)
Definition refines_where_known (f : facts -> expr -> expr) : Prop :=
  ∀ (known : facts) (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    symbols_only known ->
    symbols_bound_as known statics ->
    refines (State e vars conts statics hp rd)
      (State (f known e) vars conts statics hp rd).

(* Such a rewrite applied to the bodies of code, and to the handlers of
   recursive continuations, with what is in scope known, is such a
   rewrite. *)
Definition rewrite_code_in_scope_refines_statement : Prop :=
  ∀ f : facts -> expr -> expr,
    refines_where_known f -> refines_where_known (rewrite_code_in_scope f).

Definition rewrite_rec_handlers_in_scope_refines_statement : Prop :=
  ∀ f : facts -> expr -> expr,
    refines_where_known f ->
    refines_where_known (rewrite_rec_handlers_in_scope f).

(* With nothing known at the start they refine under all environments. *)
Definition rewrite_code_in_scope_expr_refines_statement : Prop :=
  ∀ (f : facts -> expr -> expr) (e : expr),
    refines_where_known f ->
    expr_refines e (rewrite_code_in_scope f no_facts e).

Definition rewrite_rec_handlers_in_scope_expr_refines_statement : Prop :=
  ∀ (f : facts -> expr -> expr) (e : expr),
    refines_where_known f ->
    expr_refines e (rewrite_rec_handlers_in_scope f no_facts e).

(* The unboxing of the parameters of continuations is such a rewrite. *)
Definition unbox_variants_knowing_symbols_refines_statement : Prop :=
  refines_where_known unbox_variants_knowing.

(* What the two give a rewrite is of symbols alone, and holds where
   they give it: under a group of static bindings that has a value, if
   what was known before held. *)
Definition symbols_under_statics_only_statement : Prop :=
  ∀ (known : facts) (bindings : list static_binding),
    symbols_only (symbols_under_statics known bindings).

Definition symbols_under_statics_bound_statement : Prop :=
  ∀ (known : facts) (bindings : list static_binding) (vars : env)
    (statics extended : static_env),
    symbols_only known ->
    symbols_bound_as known statics ->
    eval_static_bindings vars bindings statics = Some extended ->
    symbols_bound_as (symbols_under_statics known bindings) extended.
