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


(* The liftings to symbols without the test of the exits, up to what
   may follow a unit.

   This file is to be reviewed: the four functions, each against the
   function that it stands beside; [liftings_observed] and
   [with_lifting_observed], against FlambdaValidateSpec.with_lifting;
   and the six statements.

   Each of the four liftings has a function that makes it of a unit if
   its conditions hold: FlambdaLiftFromArmsSpec.lift_to_head_unit,
   FlambdaLiftFromCodeSpec.lift_from_code_unit, FlambdaLiftSpec.lift_unit
   and FlambdaLiftAroundSpec.lift_around_unit. The first condition of
   each is FlambdaLiftSpec.exits_plain: the unit exits with symbols and
   constants only. With it, the unit lifted exits with the same values
   as the unit, and the statement of each is a refinement in which the
   exits are equal. Without it, the unit lifted may exit with a symbol
   where the unit exits with the closure, the block or the boxed number
   that the symbol is bound to (FlambdaLiftSpec.value_lifted), and the
   two exits are not equal.

   The functions here are those four with that one condition left out
   and every other condition kept. What is stated of each is
   FlambdaObserversSpec.refines_under_observers: the unit and the unit
   lifted, run from one continuation environment of what may follow a
   unit (FlambdaObserversSpec.observers), from the static environment
   of the imports of the unit (FlambdaImportsSpec.imports_statics) and
   the empty heap, come to results that
   FlambdaObserversSpec.observed_refines relates. An exit of the first
   is answered by an exit of the second through the same continuation,
   with the same constants and the same locations among its values and
   a heap that binds the same locations; and the second ends only if
   the first exits or goes wrong. What may follow a unit is a way out,
   or a handler with no environment of its own that binds nothing
   static and names no symbol and no code id: it has the values that
   the unit gave it and nothing else, and may read them, project from
   them and call them.

   So, where the unit lifted gives a symbol and the unit a closure, a
   block or a boxed number, nothing that follows and reads what it was
   given tells the two apart. The statements say nothing of a term that
   follows and names a symbol of the unit, as another unit does; and
   nothing where the unit goes wrong. They are one-way, as
   FlambdaSemanticsSpec.refines is.

   Each has the premise of the statement of the lifting from the
   imports (FlambdaLiftSpec.lift_unit_refines_imports_statement and the
   three like it): the unit lifted binds no name that the imports bind
   (FlambdaImportsSpec.static_names_free_in). With no import it holds
   of every term.

   [liftings_observed] makes the four in turn, each where its
   conditions hold, and [with_lifting_observed] is
   FlambdaValidateSpec.with_lifting with the four functions here in
   place of the four with the test: it ends by binding the tables that
   the guess gives at the head of the unit
   (FlambdaUnusedHeadStaticsSpec.add_head_tables), and then by giving
   the static sets of closures the copies of code that the guess asks
   (FlambdaCopyCodeSpec.with_copies). The four in turn are
   stated as each is ([liftings_observed_refines_statement]), and so is
   [with_lifting_observed]
   ([with_lifting_observed_refines_statement]): the tables bound at
   the head refine the term under continuations in which their symbols
   do not occur
   (FlambdaUnusedHeadStaticsSpec.add_head_tables_refines_under_statement),
   and what may follow a unit names no symbol; the copies refine the
   term under what may follow a unit, which names no code id either
   (FlambdaCopyCodeSpec.with_copies_refines_observed_statement).
   That [with_lifting_observed] commutes with renaming and keeps scope
   is in FlambdaValidatorEquivarianceSpec.v and
   FlambdaStaticsScopeSpec.v, beside the same of [with_lifting].

   Whether this is what the verdict on a unit that is lifted is to
   mean has not been decided (FlambdaObserversSpec.v). No function of
   the validation calls the functions here.

   The proofs are in FlambdaLiftObservedProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaImportsSpec
  FlambdaObserversSpec FlambdaMyClosureSpec FlambdaLiftSpec
  FlambdaLiftFromCodeSpec FlambdaLiftFromArmsSpec FlambdaLiftAroundSpec
  FlambdaUnusedHeadStaticsSpec FlambdaCopyCodeSpec.
Import ListNotations.

(* ======================================================================== *)
(* The four liftings without the test of the exits                          *)
(* ======================================================================== *)

(* FlambdaLiftFromArmsSpec.lift_to_head_unit without the test of the
   exits: the result is a well-formed unit. *)
Definition lift_to_head_unit_observed (u : unit_interface)
    (lets : list (path * lift_to)) (e : expr) : option expr :=
  let lifted := lift_to_head lets e in
  if well_formed_unit u lifted then Some lifted else None.

(* FlambdaLiftFromCodeSpec.lift_from_code_unit without the test of the
   exits: the result is a well-formed unit. *)
Definition lift_from_code_unit_observed (u : unit_interface)
    (K : lets_in_code) (e : expr) : option expr :=
  let lifted := lift_from_code K [] root e in
  if well_formed_unit u lifted then Some lifted else None.

(* FlambdaLiftSpec.lift_unit without the test of the exits: the result
   is a well-formed unit, and it makes no closure of the code that the
   guess names but the symbol given. *)
Definition lift_unit_observed (L : lifting) (e : expr) : option expr :=
  let lifted :=
    bind_my_closures (lifted_my_closures L)
      (lift_expr (lifted_lets L) root e)
  in
  if
    well_formed_unit (lifting_unit L) lifted
    && only_closures_of (lifted_my_closures L) lifted
  then Some lifted
  else None.

(* FlambdaLiftAroundSpec.lift_around_unit without the test of the
   exits: the guess passes the check, and the result is a well-formed
   unit. *)
Definition lift_around_unit_observed (u : unit_interface)
    (lets : list (path * lift_to)) (places : list (path * list path))
    (e : expr) : option expr :=
  let groups := groups_of lets places e in
  let lifted := lift_around lets groups root e in
  if around_ok lets groups [] root e && well_formed_unit u lifted
  then Some lifted
  else None.

(* ======================================================================== *)
(* The four in turn                                                         *)
(* ======================================================================== *)

(* Each of the four, made of the term where its conditions hold; the
   term as it is where they do not. The guess is that of
   FlambdaValidateSpec.with_lifting. *)
Definition to_head_observed (L : lifting) (e : expr) : expr :=
  match lift_to_head_unit_observed (lifting_unit L) (lifted_to_head L) e with
  | Some e' => e'
  | None => e
  end.

Definition out_of_code_observed (L : lifting) (e : expr) : expr :=
  match
    lift_from_code_unit_observed (lifting_unit L) (lifted_out_of_code L) e
  with
  | Some e' => e'
  | None => e
  end.

Definition lets_observed (L : lifting) (e : expr) : expr :=
  match lift_unit_observed L e with
  | Some e' => e'
  | None => e
  end.

Definition around_observed (L : lifting) (e : expr) : expr :=
  match
    lift_around_unit_observed (lifting_unit L) (lifted_around L)
      (groups_put L) e
  with
  | Some e' => e'
  | None => e
  end.

(* The four in the order of FlambdaValidateSpec.with_lifting: to the
   head of the unit, out of code, where the [Let]s stand, to a group
   around. The paths of each are paths in the term that it is given. *)
Definition liftings_observed (L : lifting) (e : expr) : expr :=
  around_observed L
    (lets_observed L (out_of_code_observed L (to_head_observed L e))).

(* FlambdaValidateSpec.with_lifting without the test of the exits: the
   four in turn, then the tables of the guess bound at the head of the
   unit, and then the copies of code of the guess. *)
Definition with_lifting_observed (L : lifting) (e : expr) : expr :=
  with_copies (code_copied L)
    (add_head_tables (tables_put L) (liftings_observed L e)).

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What each gives is the unit, up to what may follow a unit: the two,
   run from one continuation environment of what may follow a unit and
   from the static environment of the imports, come to results that
   differ in nothing that an exit is compared by. *)
Definition lift_to_head_unit_observed_refines_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (u : unit_interface)
    (lets : list (path * lift_to)) (e e' : expr),
    lift_to_head_unit_observed u lets e = Some e' ->
    static_names_free_in (imports_statics imports) e' = true ->
    refines_under_observers [] (imports_statics imports) empty_heap e e'.

Definition lift_from_code_unit_observed_refines_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (u : unit_interface)
    (K : lets_in_code) (e e' : expr),
    lift_from_code_unit_observed u K e = Some e' ->
    static_names_free_in (imports_statics imports) e' = true ->
    refines_under_observers [] (imports_statics imports) empty_heap e e'.

Definition lift_unit_observed_refines_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (L : lifting) (e e' : expr),
    lift_unit_observed L e = Some e' ->
    static_names_free_in (imports_statics imports) e' = true ->
    refines_under_observers [] (imports_statics imports) empty_heap e e'.

Definition lift_around_unit_observed_refines_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (u : unit_interface)
    (lets : list (path * lift_to)) (places : list (path * list path))
    (e e' : expr),
    lift_around_unit_observed u lets places e = Some e' ->
    static_names_free_in (imports_statics imports) e' = true ->
    refines_under_observers [] (imports_statics imports) empty_heap e e'.

(* The four in turn give the unit, up to what may follow a unit, if what
   each of the four gives binds no name that the imports bind. With no
   import that is so of every term. *)
Definition liftings_observed_refines_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (L : lifting) (e : expr),
    let statics := imports_statics imports in
    let at_head := to_head_observed L e in
    let out_of_code := out_of_code_observed L at_head in
    let lifted := lets_observed L out_of_code in
    static_names_free_in statics at_head = true ->
    static_names_free_in statics out_of_code = true ->
    static_names_free_in statics lifted = true ->
    static_names_free_in statics (liftings_observed L e) = true ->
    refines_under_observers [] statics empty_heap e (liftings_observed L e).

(* The same with the tables of the guess bound at the head after, and
   then its copies of code: the lifting without the test of the exits
   gives the unit, up to what may follow a unit. The premise on the
   fourth lifting is now of the whole result, which binds what the
   fourth gives, the tables and the copies. *)
Definition with_lifting_observed_refines_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (L : lifting) (e : expr),
    let statics := imports_statics imports in
    let at_head := to_head_observed L e in
    let out_of_code := out_of_code_observed L at_head in
    let lifted := lets_observed L out_of_code in
    static_names_free_in statics at_head = true ->
    static_names_free_in statics out_of_code = true ->
    static_names_free_in statics lifted = true ->
    static_names_free_in statics (with_lifting_observed L e) = true ->
    refines_under_observers [] statics empty_heap e
      (with_lifting_observed L e).
