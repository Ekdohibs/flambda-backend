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


(* Lifting to a group of static bindings around: a [Let] of a closure,
   an immutable block, a boxed number or an immutable array, made an
   alias of a symbol that a group of static bindings around it binds to
   what the [Let] makes.

   This file is to be reviewed: [constant_known], [forget],
   [putable_group], [group_in_scope], [lift_around], [around_ok] and
   [groups_of], against what Simplify does (below);
   [lift_around_unit], which is extracted and run by the validation
   of middle_end/flambda2/tests/semantics/; and the statements.

   What Simplify does, in three cases that the other liftings of the
   validation do not cover.

   The arm of a test. A block or a closure made in the arm of a test
   that the module makes when it runs, with a field that is a variable
   bound before the test:

     let x = ... in
     let_cont k () = (let b = Make_block 0 (x) in cont k' (b)) in
     switch c | 0 -> cont k' (0) | 1 -> cont k ()

   An arm is not at toplevel (FlambdaWellFormedSpec.handler_at_toplevel),
   so no static binding may stand in it, and the block mentions a
   variable, so it cannot be bound at the head of the unit
   (FlambdaLiftFromArmsSpec.v). Simplify binds a symbol to the block
   before the test, where the variable is in scope, and the arm gives
   the symbol.

   A loop. A block or a closure made in a loop that the module runs,
   with a field or a value slot that is a variable bound before the
   loop: the handler of a recursive continuation is executed many
   times, so no static binding may stand in it, and the constant
   mentions a variable. Simplify binds a symbol to it before the loop.

   Equal constants. Where a [Let] makes a block, a boxed number, an
   immutable array or a closure equal to the constant that a symbol of
   the unit is bound to already, Simplify uses that symbol and binds no
   other: two blocks that hold the same constants are one symbol.
   Simplify does so for a constant that mentions no variable
   (reification.ml reuses a constant that [Static_const.can_share]
   accepts, which asks that it be fully static). The check here asks
   less: a [Let] may be made an alias of a symbol that is bound to the
   same constant with variables in it, if none of them is bound again in
   between ([forget]). That is more than the sharing of Simplify does,
   and a guess need not use it.

   What is here. [lift_around] makes the [Let]s at the paths given
   aliases of symbols, and puts the groups of static bindings given
   before the [Let_cont]s at the paths given, recursive or not:

     Let_cont k h body
       becomes
     Let_static bindings (Let_cont k h' body')

   where [h'] and [body'] are [h] and [body] with their [Let]s made
   aliases, and the same for the handlers of a group of recursive
   continuations (their paths are those of
   FlambdaLiftSpec.rec_handlers_map). Which [Let]s, to which symbols,
   is a guess of the caller's, and so is which groups are put where:
   the guess names, for a [Let_cont], the [Let]s whose bindings make
   the group put before it, by their paths, and the group is read off
   the unit ([groups_of]). So a guess names paths and symbols, and no
   variable: it is the same guess for the unit under any renaming.
   Nothing is believed of it: [around_ok] checks it. The check goes
   down the unit with the
   symbols that the groups around the place bind, each with its
   constant ([known_constants]): those of the groups of the unit that
   it has passed and of the groups put. Where a variable is bound, by
   a [Let] or as a parameter of a handler, it forgets the constants
   that mention the variable ([forget]): the variable may then have
   another value than when the group was bound. A [Let] may be made
   an alias of a symbol only if the bindings that
   FlambdaLiftSpec.lift_named gives for it, to that symbol, are among
   those known: the symbol is bound, by a group around the [Let], to
   the constant that the [Let] makes, and every variable of the
   constant has the value that it had at the group. A group may be put
   only if it binds symbols, to constants that have a value wherever
   their variables have ([putable_group]), and mentions only variables
   that the [Let_cont] it is put before mentions ([group_in_scope]):
   the unit lifted then has no variable free that the unit had not,
   whatever the unit. The handlers of recursive continuations are
   gone into for [Let]s to make aliases: what is known at the
   [Let_cont] is known in a handler less what mentions its
   parameters, at every round. No group may be put in such a
   handler: the machine does not run a loop whose handlers bind
   something statically, so the check asks that the handlers
   rewritten bind something statically only if the handlers did. The
   code of functions is not gone into.

   What is made an alias is what FlambdaLiftSpec.lift_named lifts and
   nothing else: an immutable block ([Make_block], which in the model is
   the immutable one), a boxed number, an immutable array
   ([Make_immutable_array]), a closure. A block with identity, a call
   and a load are never moved or shared. This is a restriction on the
   constructors, not a claim that the [Let] has a value: a [Let] that
   boxes a number at a kind that is not its own, or makes a block of a
   symbol that is bound to nothing, is undefined where the static
   binding that stands for it has a value. The unit is then undefined
   there, and the refinement, which is one way, asks nothing of the unit
   lifted.

   Why it is right. The first run makes the block each time the [Let]
   is executed. The second has the symbol, which a group around the
   [Let] has bound to the same constant; between the group and the
   [Let] no variable of the constant is bound again, so the fields
   are the same values, up to FlambdaLiftSpec.value_lifted, as for a
   [Let] lifted where it stands. No primitive and no rule of the
   machine tells the block from the symbol. They differ at an exit of
   the unit, so [lift_around_unit] makes the check of
   FlambdaLiftSpec.lift_unit (FlambdaLiftSpec.exits_plain_knowing). It
   also checks that the result is a well-formed unit: a group put is
   then at toplevel, executed once, and binds symbols that are new; and
   its variables are in scope where it stands, so they have values there
   (FlambdaScopeSpec.v). With these the unit lifted refines the unit,
   run on its own ([lift_around_unit_program_refines_statement]).

   Two things that the refinement does not say.

   A group put before a test is bound whichever arm is taken: on the
   path through another arm the second run allocates what the first
   does not. The refinement is of the results of the unit, not of what
   it allocates, as for a [Let] lifted out of code that is never
   called. Nothing but an allocation is brought forward, and it cannot
   be undefined there: the fields of a static block are not tested for
   their kind, nor for being symbols that are bound, and a closure of
   a set is put only at a function slot that the set has.

   Two [Let]s that are made aliases of one symbol give, in the second
   run, the same symbol where the first run has two blocks; and a [Let]
   that is made an alias of a symbol that the unit binds already gives
   that symbol where the first run has a block of its own. In the model
   nothing tells them apart. What is made an alias is an immutable
   block, a boxed number, an immutable array or a closure of this unit,
   which have no identity in the model. A block or an array of this unit
   that has one (mutable or unique) is made by a primitive of the heap,
   which [lift_named] does not lift (FlambdaLiftSpec.v, "Identity") and
   which the step of the core does not execute; a value with identity
   that comes from another unit is not itself made an alias by this
   rewrite. Of the immutable allocated values that this rewrite makes
   aliases, the physical comparison of two has no value
   (FlambdaPrimitiveSpec.eval_phys_equal, decision D19), so a unit that
   compares them is undefined before and after. Physical equality of
   immutable values is outside the model, and sharing is what it would
   see.

   A [Let] in a loop that is made an alias of a symbol bound before
   the loop adds nothing to this: the first run makes the block at
   each round, with the same fields, as no variable of the constant
   is bound again in the loop.

   What is not here: a [Let] in the code of a function made an alias
   of a symbol bound around the code (FlambdaLiftFromCodeSpec.v binds
   a symbol of its own for it).

   The proofs are in FlambdaLiftAroundProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaWellFormedSpec FlambdaUnusedStaticsSpec FlambdaImportsSpec
  FlambdaLiftSpec.
Import ListNotations.

(* ======================================================================== *)
(* What the groups around a place bind                                      *)
(* ======================================================================== *)

(* Symbols that the groups of static bindings around a place bind, each
   with the constant that it is bound to. *)
Definition known_constants : Type := list (symbol * static_const).

(* The symbols that a group binds, with their constants. *)
Definition constants_of_group (bindings : list static_binding)
    : known_constants :=
  flat_map
    (fun binding =>
       match binding with
       | Bind_symbol s const => [(s, const)]
       | Bind_code _ _ | Bind_deleted_code _ => []
       end)
    bindings.

(* The binding is of a symbol, to a constant, that is known. *)
Definition constant_known (known : known_constants)
    (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol s const =>
    existsb
      (fun bound =>
         Pos.eqb s (fst bound)
         && if static_const_eq_dec const (snd bound) then true else false)
      known
  | Bind_code _ _ | Bind_deleted_code _ => false
  end.

(* What is still known once the variables given are bound again: the
   constants that mention none of them. *)
Definition forget (bound : list variable) (known : known_constants)
    : known_constants :=
  filter
    (fun constant =>
       negb
         (existsb (fun x => among x bound)
            (free_variables_of_static_const (snd constant))))
    known.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* What stands for the [Let] of [d] at [at_path]: the bindings that
   FlambdaLiftSpec.lift_named gives for it and the symbol, if the guess
   has the path. The bindings are not put anywhere: [around_ok] asks
   that they are known. *)
Definition alias_of (lets : list (path * lift_to)) (at_path : path)
    (d : named) : option (list static_binding * symbol) :=
  match lookup at_path lets with
  | Some target => lift_named target d
  | None => None
  end.

(* The group to put before the [Let_cont] at [at_path]. *)
Definition group_at (groups : list (path * list static_binding))
    (at_path : path) : list static_binding :=
  match lookup at_path groups with
  | Some bindings => bindings
  | None => []
  end.

(* The expression with the [Let]s at the paths [lets] made aliases of
   the symbols that stand for them, and the groups of [groups] put
   before the [Let_cont]s at their paths. [at_path] is the path of the
   expression; the paths are those of FlambdaLiftSpec.lift_expr, in the
   expression as it is given. *)
Fixpoint lift_around (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (at_path : path) (e : expr)
    : expr :=
  match e with
  | Let x kind d body =>
    let body := lift_around lets groups (xO at_path) body in
    match alias_of lets at_path d with
    | Some (_, s) => Let x kind (Simple (Symbol s)) body
    | None => Let x kind d body
    end
  | Let_static bindings body =>
    Let_static bindings (lift_around lets groups (xO at_path) body)
  | Let_cont_non_recursive k h body =>
    let rest :=
      Let_cont_non_recursive k (lift_around_handler lets groups (xI at_path) h)
        (lift_around lets groups (xO at_path) body)
    in
    match group_at groups at_path with
    | [] => rest
    | binding :: bindings => Let_static (binding :: bindings) rest
    end
  | Let_cont_recursive handlers body =>
    let rest :=
      Let_cont_recursive
        (rec_handlers_map (lift_around_handler lets groups) (xI at_path)
           handlers)
        (lift_around lets groups (xO at_path) body)
    in
    match group_at groups at_path with
    | [] => rest
    | binding :: bindings => Let_static (binding :: bindings) rest
    end
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with lift_around_handler (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (at_path : path)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (lift_around lets groups at_path handler)
  end.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* A constant that has a value wherever its variables have: any but a
   closure at a function slot that its set has not got
   (FlambdaSemanticsSpec.eval_static_const). *)
Definition total_const (const : static_const) : bool :=
  match const with
  | Static_const_closure_of_set slot _ function_decls _ =>
    match lookup slot function_decls with
    | Some _ => true
    | None => false
    end
  | Static_const_set_of_closures _ _ | Static_const_block _ _
  | Static_const_boxed_number _ | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    true
  end.

(* A group that may be put: it binds symbols only, to such constants. *)
Definition putable_group (bindings : list static_binding) : bool :=
  forallb
    (fun binding =>
       match binding with
       | Bind_symbol _ const => total_const const
       | Bind_code _ _ | Bind_deleted_code _ => false
       end)
    bindings.

(* The variables of the group are variables of the expression that it
   is put before. *)
Definition group_in_scope (bindings : list static_binding) (e : expr) : bool :=
  all_among (flat_map free_variables_of_binding bindings)
    (free_variables_of_expr e).

(* Every [Let] that the guess makes an alias stands for bindings that
   are known where it is; every group that the guess puts may be put,
   and mentions only variables of the [Let_cont] that it is put
   before; and no group is put in the handler of a recursive
   continuation, which the machine does not let bind anything
   statically. [known] is what the groups around the expression bind,
   less what was forgotten on the way. *)
Fixpoint around_ok (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (known : known_constants)
    (at_path : path) (e : expr) : bool :=
  match e with
  | Let x _ d body =>
    match alias_of lets at_path d with
    | Some (bindings, _) => forallb (constant_known known) bindings
    | None => true
    end
    && around_ok lets groups (forget [x] known) (xO at_path) body
  | Let_static bindings body =>
    around_ok lets groups (constants_of_group bindings ++ known)
      (xO at_path) body
  | Let_cont_non_recursive _ h body =>
    let bindings := group_at groups at_path in
    let known := constants_of_group bindings ++ known in
    putable_group bindings
    && group_in_scope bindings e
    && handler_around_ok lets groups known (xI at_path) h
    && around_ok lets groups known (xO at_path) body
  | Let_cont_recursive handlers body =>
    let bindings := group_at groups at_path in
    let known := constants_of_group bindings ++ known in
    putable_group bindings
    && group_in_scope bindings e
    && Bool.eqb
         (handlers_bind_statics
            (rec_handlers_map (lift_around_handler lets groups) (xI at_path)
               handlers))
         (handlers_bind_statics handlers)
    && rec_handlers_forall (handler_around_ok lets groups known) (xI at_path)
         handlers
    && around_ok lets groups known (xO at_path) body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_around_ok (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (known : known_constants)
    (at_path : path) (h : continuation_handler) : bool :=
  match h with
  | Handler params _ handler =>
    around_ok lets groups (forget params known) at_path handler
  end.

(* ======================================================================== *)
(* The groups, read off the unit                                            *)
(* ======================================================================== *)

(* The bindings that stand for the [Let]s of the expression that are at
   the paths [wanted], in the order of the expression. The bodies of
   code are not gone into. *)
Fixpoint bindings_of_lets (lets : list (path * lift_to))
    (wanted : list path) (at_path : path) (e : expr) : list static_binding :=
  match e with
  | Let _ _ d body =>
    (if among at_path wanted
     then
       match alias_of lets at_path d with
       | Some (bindings, _) => bindings
       | None => []
       end
     else [])
    ++ bindings_of_lets lets wanted (xO at_path) body
  | Let_static _ body => bindings_of_lets lets wanted (xO at_path) body
  | Let_cont_non_recursive _ h body =>
    bindings_of_lets lets wanted (xO at_path) body
    ++ handler_bindings_of_lets lets wanted (xI at_path) h
  | Let_cont_recursive handlers body =>
    bindings_of_lets lets wanted (xO at_path) body
    ++ rec_handlers_bindings (handler_bindings_of_lets lets wanted)
         (xI at_path) handlers
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_bindings_of_lets (lets : list (path * lift_to))
    (wanted : list path) (at_path : path) (h : continuation_handler)
    : list static_binding :=
  match h with
  | Handler _ _ handler => bindings_of_lets lets wanted at_path handler
  end.

(* The groups that a guess puts: for each [Let_cont] that it names, by
   its path, the bindings that stand for the [Let]s that it names with
   it, by their paths. *)
Definition groups_of (lets : list (path * lift_to))
    (places : list (path * list path)) (e : expr)
    : list (path * list static_binding) :=
  map (fun place => (fst place, bindings_of_lets lets (snd place) root e))
    places.

(* ======================================================================== *)
(* The unit                                                                 *)
(* ======================================================================== *)

(* The unit with the [Let]s of the guess made aliases and its groups
   put: if the unit exits with symbols and constants only, the guess
   passes the check, and the result is a well-formed unit. *)
Definition lift_around_unit (u : unit_interface)
    (lets : list (path * lift_to)) (places : list (path * list path))
    (e : expr) : option expr :=
  let groups := groups_of lets places e in
  let lifted := lift_around lets groups root e in
  if
    exits_plain_knowing [] [] e
    && around_ok lets groups [] root e
    && well_formed_unit u lifted
  then Some lifted
  else None.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What [lift_around_unit] gives refines the unit, run on its own. *)
Definition lift_around_unit_program_refines_statement : Prop :=
  ∀ (u : unit_interface) (lets : list (path * lift_to))
    (places : list (path * list path)) (e e' : expr),
    lift_around_unit u lets places e = Some e' -> program_refines e e'.

(* The same of a unit that imports symbols, run from the static
   environment of its imports: if the result binds no name that the
   imports bind, it refines the unit. A symbol that a group put binds
   and that is an import would be bound twice, and the group is
   executed whether the [Let]s that it stands for are reached or not. *)
Definition lift_around_unit_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (u : unit_interface)
    (lets : list (path * lift_to)) (places : list (path * list path))
    (e e' : expr),
    lift_around_unit u lets places e = Some e' ->
    static_names_free_in (imports_statics imports) e' = true ->
    unit_refines imports e e'.

(* Groups, renamed: the constants that they bind may mention
   variables. *)
Definition rename_groups (r : renaming)
    (groups : list (path * list static_binding))
    : list (path * list static_binding) :=
  map (fun group => (fst group, map (rename_static_binding r) (snd group)))
    groups.

(* What is known, renamed. *)
Definition rename_known_constants (r : renaming) (known : known_constants)
    : known_constants :=
  map (fun constant => (fst constant, rename_static_const r (snd constant)))
    known.

(* The rewrite commutes with renaming: a [Let] is named by where it is,
   and the groups put are renamed with the term. *)
Definition lift_around_equivariant_statement : Prop :=
  ∀ (r : renaming) (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (at_path : path) (e : expr),
    rename_expr r (lift_around lets groups at_path e)
    = lift_around lets (rename_groups r groups) at_path (rename_expr r e).

(* So does the check, what is known being renamed too. *)
Definition around_ok_equivariant_statement : Prop :=
  ∀ (r : renaming) (lets : list (path * lift_to))
    (groups : list (path * list static_binding)) (known : known_constants)
    (at_path : path) (e : expr),
    around_ok lets (rename_groups r groups) (rename_known_constants r known)
      at_path (rename_expr r e)
    = around_ok lets groups known at_path e.

(* The groups that a guess puts are read off the unit, so those of the
   unit renamed are those of the unit, renamed. *)
Definition groups_of_equivariant_statement : Prop :=
  ∀ (r : renaming) (lets : list (path * lift_to))
    (places : list (path * list path)) (e : expr),
    groups_of lets places (rename_expr r e)
    = rename_groups r (groups_of lets places e).
