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

(* The closure that every set of closures which has a given code at a
   given function slot holds in a value slot: of what code it is. A
   property of a unit that its run keeps.

   This file is to be reviewed: [sight_made] and [sets_seen], which
   read the sets of closures that a term makes with what is in sight
   where each is made; [sets_seen_give], the test;
   [state_slots_give], what the test gives of a state; and the
   statements.

   FlambdaSetsOfCodeSpec.v says what is known of every set that has
   the code [c] at the function slot [f] where each of them gives a
   value slot one symbol or constant. A function that captures a
   closure which is made each time an enclosing function runs is not
   such a case: the sets with [c] at [f] give the slot a variable,
   and the variable may be bound to different closure values on
   different runs. All those closures may still be of one code. That
   is what is known
   here: every set with [c] at [f] gives the value slot [slot] a
   closure of the code [d].

   - What is in sight where a set is made ([closure_sight]): the
     variables that a [Let] above binds to a closure, which is one
     that the [Let] makes or the projection of a function slot from
     a variable in sight ([sight_made]); each with the function slot
     that the closure is at and the code of the function slots of
     its set. A variable is in sight from its [Let] down to where it
     is bound again, by a [Let] or as a parameter of a handler; the
     handler of a continuation runs under the variables of the place
     where it is defined, so what is in sight there is in sight in
     it; the body of code runs under its parameters and [my_closure]
     alone, so nothing is in sight where it starts. Nothing else is
     in sight: not a parameter, and not a variable bound to another
     variable, to a symbol, to the projection of a value slot or to
     a field. A closure in sight is a closure value and not a symbol
     ([closure_value_seen]): what code it is of does not depend on
     the static environment.
   - The test ([sets_seen_give]) is of the sets that the unit makes,
     by a [Let] or statically, in handlers and in code too
     ([sets_seen]): each of them that has [c] at [f] gives [slot] a
     variable in sight that is a closure of the code [d]. A set with
     [c] at [f] that gives [slot] anything else fails it: a symbol,
     even one that the unit binds to a closure of [d]; a constant; a
     variable that is not in sight. Nothing is guessed.
   - Where the test passes, the state that the unit starts in, run on
     its own, has it ([state_slots_give];
     [state_slots_give_initial_statement]), and a step keeps it
     ([state_slots_give_step_statement]), so every state of the run
     has it. Of the terms of a state it says that they pass the test
     under a sight that holds of the variables that they run under
     ([sight_holds]): the expression under the variables of the
     state, the handlers that a continuation keeps under the
     variables that it keeps, the code that is bound under none. Of
     the values, that every closure in them which is of a set with
     [c] at [f] holds in [slot] a closure value of the code [d]
     ([value_slots_give]).
   - In a state that has it, a closure that a variable is bound to,
     at [f] of a set that has [c] there, holds in [slot] a closure
     value of the code [d] ([closure_slot_gives_statement]), which is
     a closure of that code under every static environment
     ([closure_value_of_statement]).

   Not counted: a value slot that the sets give a symbol, even where
   the unit binds each such symbol to a static closure of the code
   [d]. To count it the fact would have to say which symbols, and
   that each is bound where the slot is read: it would depend on the
   static environment and on the scope of symbols. Where every set
   gives the slot one symbol, FlambdaSetsOfCodeSpec.Slot_holds is
   the fact. One unit of the measure of these definitions is
   validated by a prototype that counts such symbols and is not
   without them: a map and an append in continuation-passing style,
   of a set of algorithms, at -O3.

   Under other environments this fails, as in
   FlambdaSetsOfCodeSpec.v: a closure from elsewhere may have [c] at
   [f] and anything in [slot]. The state may start from a static
   environment that binds something already and with continuations,
   where the code and the values that those hold have it
   ([state_slots_give_from_statement]): the static environment of
   the imports of a unit has no code and no closure, and what may
   follow a unit names no code id, so it makes no set.

   The fact is about the code and the function slot, as those of
   FlambdaSetsOfCodeSpec.v are: two sets of a unit may have [c] at
   [f] beside different code, and each is asked. A set of several
   functions has one list of value slots for them all, and the fact
   is of the set: it is asked of a closure at any function slot of a
   set that has [c] at [f]. A set whose code at [f] is a copy of the
   code [c] has the code id of the copy there and not [c]: nothing
   is asked of it and nothing is known of it by this fact, though
   its value slots have the same names.

   Nothing here is a rewrite. The proofs are in
   FlambdaSlotsOfCodeProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaSetsOfCodeSpec
  FlambdaContextWhereSpec.
Import ListNotations.

(* ======================================================================== *)
(* The closures in sight                                                    *)
(* ======================================================================== *)

(* A closure as it is seen where it is made: the function slot that it
   is at, and the code of the function slots of its set. *)
Definition closure_seen : Type :=
  (function_slot * list (function_slot * code_id))%type.

(* Its code: that of the function slot that it is at. *)
Definition seen_code (seen : closure_seen) : option code_id :=
  lookup (fst seen) (snd seen).

(* The variables in sight that are bound to closures. *)
Definition closure_sight : Type := finite_map closure_seen.

(* The closure that a defining expression gives, where it is seen: a
   set of one function, which is the set of that function at
   FlambdaValueSpec.unnamed_function_slot; a closure of a set; and the
   projection of a function slot from a variable in sight, which is
   the closure of the same set at the function slot projected to
   (FlambdaValueSpec.project_function_slot). *)
Definition sight_made (known : closure_sight) (d : named)
    : option closure_seen :=
  match d with
  | Set_of_closures code _ =>
    Some (unnamed_function_slot, [(unnamed_function_slot, code)])
  | Closure_of_set slot function_decls _ => Some (slot, function_decls)
  | Project_function_slot _ move_to (Var closure) =>
    match lookup closure known with
    | Some (_, function_decls) => Some (move_to, function_decls)
    | None => None
    end
  | Project_function_slot _ _ (Symbol _ | Const _) | Simple _ | Prim _ _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _ =>
    None
  end.

(* What is in sight in the body of a [Let]: the variable that it
   binds, if its defining expression gives a closure that is seen, and
   what was in sight less that variable. *)
Definition sight_under_let (known : closure_sight) (x : variable)
    (d : named) : closure_sight :=
  match sight_made known d with
  | Some seen => (x, seen) :: unbind x known
  | None => unbind x known
  end.

(* What is in sight once the variables given are bound again: nothing
   of them. *)
Fixpoint sight_without (xs : list variable) (known : closure_sight)
    : closure_sight :=
  match xs with
  | [] => known
  | x :: xs => sight_without xs (unbind x known)
  end.

(* The code of the closure in sight that a simple is, if it is a
   variable in sight. *)
Definition simple_seen_code (known : closure_sight) (a : simple)
    : option code_id :=
  match a with
  | Var x =>
    match lookup x known with
    | Some seen => seen_code seen
    | None => None
    end
  | Symbol _ | Const _ => None
  end.

(* ======================================================================== *)
(* The sets of closures that a term makes, as they are seen                 *)
(* ======================================================================== *)

(* A set of closures as a term makes it and as it is seen there: the
   code of its function slots and, for each of its value slots, the
   code of the closure in sight that it gives the slot, if it gives it
   one. *)
Definition set_seen : Type :=
  (list (function_slot * code_id) * list (value_slot * option code_id))%type.

Definition slots_seen (known : closure_sight)
    (value_slots : list (value_slot * simple))
    : list (value_slot * option code_id) :=
  map (fun '(slot, a) => (slot, simple_seen_code known a)) value_slots.

(* The set that a defining expression, or a static constant, makes
   (FlambdaSetsOfCodeSpec.named_sets_made,
   FlambdaSetsOfCodeSpec.static_const_sets_made), seen with what is in
   sight there. *)
Definition named_sets_seen (known : closure_sight) (d : named)
    : list set_seen :=
  match d with
  | Set_of_closures code value_slots =>
    [([(unnamed_function_slot, code)], slots_seen known value_slots)]
  | Closure_of_set _ function_decls value_slots =>
    [(function_decls, slots_seen known value_slots)]
  | Simple _ | Prim _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
    []
  end.

Definition static_const_sets_seen (known : closure_sight)
    (const : static_const) : list set_seen :=
  match const with
  | Static_const_set_of_closures code value_slots =>
    [([(unnamed_function_slot, code)], slots_seen known value_slots)]
  | Static_const_closure_of_set _ _ function_decls value_slots =>
    [(function_decls, slots_seen known value_slots)]
  | Static_const_block _ _ | Static_const_boxed_number _
  | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    []
  end.

(* All the sets that an expression makes, in handlers and in code too
   (FlambdaSetsOfCodeSpec.sets_made), each seen with what is in sight
   where it is made: under a [Let], what [sight_under_let] gives; in a
   handler, what is in sight where the handler is defined less its
   parameters; in the body of code, nothing to begin with. *)
Fixpoint sets_seen (known : closure_sight) (e : expr) : list set_seen :=
  match e with
  | Let x _ d body =>
    named_sets_seen known d ++ sets_seen (sight_under_let known x d) body
  | Let_static bindings body =>
    flat_map (binding_sets_seen known) bindings ++ sets_seen known body
  | Let_cont_non_recursive _ h body =>
    handler_sets_seen known h ++ sets_seen known body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_sets_seen known h) handlers
    ++ sets_seen known body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_sets_seen (known : closure_sight) (h : continuation_handler)
    : list set_seen :=
  match h with
  | Handler params _ handler => sets_seen (sight_without params known) handler
  end
with binding_sets_seen (known : closure_sight) (binding : static_binding)
    : list set_seen :=
  match binding with
  | Bind_code _ c => code_sets_seen c
  | Bind_deleted_code _ => []
  | Bind_symbol _ const => static_const_sets_seen known const
  end
with code_sets_seen (c : code) : list set_seen :=
  match c with
  | Code _ _ _ _ _ body => sets_seen [] body
  end.

(* ======================================================================== *)
(* The test                                                                 *)
(* ======================================================================== *)

(* A set, as it is seen, gives the value slot a closure in sight of the
   code [d]. *)
Definition set_seen_gives (slot : value_slot) (d : code_id) (set : set_seen)
    : bool :=
  match lookup slot (snd set) with
  | Some (Some code) => Pos.eqb code d
  | Some None | None => false
  end.

(* Each of the sets that has [c] at [f] does. *)
Definition sets_seen_give (c : code_id) (f : function_slot)
    (slot : value_slot) (d : code_id) (seen : list set_seen) : bool :=
  forallb
    (fun set => negb (has_code_at c f (fst set)) || set_seen_gives slot d set)
    seen.

(* ======================================================================== *)
(* What the test gives of a state                                           *)
(* ======================================================================== *)

(* A closure value as it is seen: the function slot that it is at and
   the code of the function slots of its set, as
   FlambdaValueSpec.closure_in_set reads them. A symbol is not a
   closure value, whatever it is bound to. *)
Definition closure_value_seen (v : value) : option closure_seen :=
  match v with
  | Value_closure code _ =>
    Some (unnamed_function_slot, [(unnamed_function_slot, code)])
  | Value_closure_of_set slot function_decls _ =>
    let* _ := lookup slot function_decls in
    Some (slot, function_decls)
  | Value_const _ | Value_symbol _ | Value_block _ _
  | Value_immutable_array _ | Value_boxed_number _ | Value_location _
  | Value_partial_application _ _ _ =>
    None
  end.

(* The value is a closure value of the code [d]. *)
Definition value_is_closure_of (d : code_id) (v : value) : bool :=
  match closure_value_seen v with
  | Some seen =>
    match seen_code seen with
    | Some code => Pos.eqb code d
    | None => false
    end
  | None => false
  end.

(* What is in sight holds of the variables: each variable in sight is
   bound to a closure value that is seen so. *)
Definition sight_holds (known : closure_sight) (vars : env) : Prop :=
  ∀ (x : variable) (seen : closure_seen),
    lookup x known = Some seen ->
    ∃ v : value, lookup x vars = Some v ∧ closure_value_seen v = Some seen.

(* A set that a closure is of, with the values of its value slots, if
   it has [c] at [f], holds in the value slot a closure value of the
   code [d]; nothing is asked of another. *)
Definition set_held_gives (c : code_id) (f : function_slot)
    (slot : value_slot) (d : code_id)
    (function_decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)) : bool :=
  negb (has_code_at c f function_decls)
  || match lookup slot slot_values with
     | Some v => value_is_closure_of d v
     | None => false
     end.

(* Every closure in the value is of a set that is so, through the
   values of value slots, the fields of blocks and partial
   applications (as FlambdaSetsOfCodeSpec.value_sets_have). A symbol
   is so whatever it is bound to: what it is bound to is asked to be
   so where it is bound ([allocated_slots_give]). *)
Fixpoint value_slots_give (c : code_id) (f : function_slot)
    (slot : value_slot) (d : code_id) (v : value) : bool :=
  match v with
  | Value_const _ | Value_symbol _ | Value_boxed_number _ | Value_location _ =>
    true
  | Value_closure code slot_values =>
    set_held_gives c f slot d [(unnamed_function_slot, code)] slot_values
    && forallb (fun '(_, contents) => value_slots_give c f slot d contents)
         slot_values
  | Value_closure_of_set _ function_decls slot_values =>
    set_held_gives c f slot d function_decls slot_values
    && forallb (fun '(_, contents) => value_slots_give c f slot d contents)
         slot_values
  | Value_block _ fields | Value_immutable_array fields =>
    forallb (value_slots_give c f slot d) fields
  | Value_partial_application callee args _ =>
    value_slots_give c f slot d callee
    && forallb (value_slots_give c f slot d) args
  end.

Definition allocated_slots_give (c : code_id) (f : function_slot)
    (slot : value_slot) (d : code_id) (allocated : statically_allocated_value)
    : bool :=
  match allocated with
  | Static_closure code slot_values =>
    set_held_gives c f slot d [(unnamed_function_slot, code)] slot_values
    && forallb (fun '(_, contents) => value_slots_give c f slot d contents)
         slot_values
  | Static_closure_of_set _ _ function_decls slot_values =>
    set_held_gives c f slot d function_decls slot_values
    && forallb (fun '(_, contents) => value_slots_give c f slot d contents)
         slot_values
  | Static_block _ fields => forallb (value_slots_give c f slot d) fields
  | Static_immutable_array elements =>
    forallb (value_slots_give c f slot d) elements
  | Static_boxed_number contents => value_slots_give c f slot d contents
  | Static_string _ | Static_imported _ => true
  end.

(* An expression that runs under the variables [vars] makes sets that
   pass the test, seen under some sight that holds of [vars]. *)
Definition expr_slots_give (c : code_id) (f : function_slot)
    (slot : value_slot) (d : code_id) (vars : env) (e : expr) : Prop :=
  ∃ known : closure_sight,
    sight_holds known vars
    ∧ sets_seen_give c f slot d (sets_seen known e) = true.

(* The handlers that a continuation keeps make sets that pass the
   test, seen under some sight that holds of the variables that the
   continuation keeps, which are those that each of the handlers runs
   under with its parameters bound; and so of the continuations that
   it keeps. What is left of an over-application keeps no term: it is
   so when the two continuations that it keeps are. *)
Fixpoint cont_slots_give (c : code_id) (f : function_slot)
    (slot : value_slot) (d : code_id) (k : cont_value) : Prop :=
  match k with
  | Outer _ => True
  | Closure h vars conts recursive =>
    (∃ known : closure_sight,
       sight_holds known vars
       ∧ sets_seen_give c f slot d (handler_sets_seen known h) = true
       ∧ forallb
           (fun '(_, h) =>
              sets_seen_give c f slot d (handler_sets_seen known h))
           recursive
         = true)
    ∧ fold_right (fun kc acc => cont_slots_give c f slot d (snd kc) ∧ acc)
        True conts
  | Extra_args k' _ => cont_slots_give c f slot d k'
  | Over_application _ _ _ return_to exn_to =>
    cont_slots_give c f slot d return_to ∧ cont_slots_give c f slot d exn_to
  end.

Definition conts_slots_give (c : code_id) (f : function_slot)
    (slot : value_slot) (d : code_id) (conts : cont_env) : Prop :=
  fold_right (fun kc acc => cont_slots_give c f slot d (snd kc) ∧ acc) True
    conts.

(* The code that is bound makes sets that pass the test, seen with
   nothing in sight where its body starts. *)
Definition code_slots_give (c : code_id) (f : function_slot)
    (slot : value_slot) (d : code_id) (statics : static_env) : Prop :=
  ∀ (id : code_id) (code : code),
    lookup id (static_code statics) = Some code ->
    sets_seen_give c f slot d (code_sets_seen code) = true.

(* Every set that the state holds or may come to make, and that has
   [c] at [f], holds in the value slot a closure value of the code
   [d]: those that its terms make (its expression, the handlers that
   its continuations keep, the code that is bound), and those of the
   closures in its values (those of its variables, of the environments
   that its continuations keep and of the cells of its heap, and what
   each symbol is bound to; FlambdaSemanticsSpec.state_values_all). *)
Definition state_slots_give (c : code_id) (f : function_slot)
    (slot : value_slot) (d : code_id) (s : state) : Prop :=
  expr_slots_give c f slot d (state_vars s) (state_expr s)
  ∧ conts_slots_give c f slot d (state_conts s)
  ∧ code_slots_give c f slot d (state_statics s)
  ∧ state_values_all (fun _ v => value_slots_give c f slot d v = true)
      (fun _ allocated => allocated_slots_give c f slot d allocated = true) s.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The state that a unit starts in, run on its own, has it where the
   sets that the unit makes pass the test, seen from nothing in sight.
   Nothing else is asked of the unit; it need not be well-formed. *)
Definition state_slots_give_initial_statement : Prop :=
  ∀ (c : code_id) (f : function_slot) (slot : value_slot) (d : code_id)
    (e : expr),
    sets_seen_give c f slot d (sets_seen [] e) = true ->
    state_slots_give c f slot d
      (State e [] [] empty_static_env empty_heap no_reads).

(* The same from a static environment that binds something already and
   with continuations: the code that the static environment has and
   the handlers that the continuations keep make sets that pass the
   test, and the values that they hold have it. The heap is empty. *)
Definition state_slots_give_from_statement : Prop :=
  ∀ (c : code_id) (f : function_slot) (slot : value_slot) (d : code_id)
    (e : expr) (conts : cont_env) (statics : static_env),
    sets_seen_give c f slot d (sets_seen [] e) = true ->
    conts_slots_give c f slot d conts ->
    conts_values_all (fun v => value_slots_give c f slot d v = true) conts ->
    code_slots_give c f slot d statics ->
    Forall
      (fun binding => allocated_slots_give c f slot d (snd binding) = true)
      (static_values statics) ->
    state_slots_give c f slot d
      (State e [] conts statics empty_heap no_reads).

(* A step keeps it. *)
Definition state_slots_give_step_statement : Prop :=
  ∀ (c : code_id) (f : function_slot) (slot : value_slot) (d : code_id),
    kept_by_step (state_slots_give c f slot d).

(* In a state that has it, a closure that a variable is bound to, at
   the function slot [f] of a set that has [c] there, holds in the
   value slot a closure value of the code [d]. The closure may be a
   symbol; what it holds in the slot is not. *)
Definition closure_slot_gives_statement : Prop :=
  ∀ (c : code_id) (f : function_slot) (slot : value_slot) (d : code_id)
    (s : state) (x : variable) (v : value)
    (function_decls : list (function_slot * code_id))
    (slot_values : list (value_slot * value)),
    state_slots_give c f slot d s ->
    lookup x (state_vars s) = Some v ->
    closure_in_set (state_statics s) v
    = Some (f, function_decls, slot_values) ->
    lookup f function_decls = Some c ->
    ∃ w : value,
      lookup slot slot_values = Some w ∧ value_is_closure_of d w = true.

(* A closure value of the code [d] is a closure of that code under
   every static environment. *)
Definition closure_value_of_statement : Prop :=
  ∀ (statics : static_env) (d : code_id) (w : value),
    value_is_closure_of d w = true ->
    ∃ slot_values : list (value_slot * value),
      closure_of statics w = Some (d, slot_values).
